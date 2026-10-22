// reflex-min-arch: 75
//
// Quantized-resident prefill GEMM on tensor cores: y[rows, out] = x16[rows, in] @ W^T,
// with W the GGUF's raw quantized blocks dequantized to f16 in registers (the same
// values `dequantize_*_coalesced_f16_kernel` writes to the scratch buffer), x16 the
// activations already cast by `cast_act_f16_kernel`, and f32 accumulation. Replaces
// the dequantize-to-scratch + cublasGemmEx path for 2+ rows with `--weights f16`.
// Q4_K and Q6_K, one kernel template over a per-type traits struct.
// Design: docs/design/quantized-mma-prefill.md.
//
// `mma.sync` needs sm_75, hence the directive above (build.rs adds
// `-arch=compute_75` in portable-PTX builds). In cubin/fatbin images for older
// archs the bodies compile to nothing; `Model` checks the compute capability and
// never launches them there.
//
// Block tile: 64 output rows (weight rows) by MT*16 input rows, 4 warps. Each
// k step stages one 256-element super-block per weight row and the matching 256
// activations per input row in shared memory; the next step's bytes are loaded
// into registers while this one computes. Warp w owns weight rows 16w..16w+16 as
// two n8 tiles and every m16 tile of the block.
//
// Fragments (PTX ISA, mma m16n8k16 .f16, groupID g = lane/4, t = lane%4):
//   A (activations, row-major): a0 = (g, 2t..2t+1), a1 = (g+8, 2t..),
//       a2 = (g, 2t+8..), a3 = (g+8, 2t+8..)
//   B (weights, "col" = K-contiguous per output row, as the GGUF stores them):
//       b0 = (k 2t..2t+1, n g), b1 = (k 2t+8..2t+9, n g)
//   C: c0,c1 = (g, 2t..2t+1), c2,c3 = (g+8, 2t..2t+1)
// sm_75 has no k16 shape; it runs two m16n8k8 (a0,a1,b0 then a2,a3,b1), which
// adds the same products in a different order.
//
// Q4_K weights: k16 step s (0..15) of a super-block covers elements 16s..16s+15,
// all in one 32-element run: group s/4, high nibbles if (s/2) is odd, byte offset
// 16 + 32*(s/4) + 16*(s%2), scale pair s/2. A lane's b0 pair is two adjacent
// bytes there (offset + 2t), b1 the two 8 bytes on. Each weight is
// `d1 * (float)q - m1` rounded with `__float2half_rn`, the expression and
// rounding of `dequantize_q4k_coalesced_f16_kernel`.
//
// Q6_K weights (210-byte blocks: ql[128], qh[64], int8 scales[16], f16 d): step s
// is half n = s/8, 32-run r = (s/2)%4, positions l = 16*(s%2) .. +15 of that run,
// one scale, scales[8n + s%2 + 2r]. Element l's low nibble is ql[64n + l + 32*(r&1)]
// (high nibble if r >= 2), its top two bits qh[32n + l] >> 2r. Each weight is
// `d * (float)sc * (float)q` rounded with `__float2half_rn`, as
// `dequantize_q6k_coalesced_f16_kernel` computes it. A Q6_K row is not 16-byte
// aligned (210 bytes a block), so each row stages the 14 aligned 16-byte chunks
// covering its block (a block starts at an even address: 14 chunks always cover
// it, and each holds at least one of its bytes) and reads the block at the
// address's offset within the first one.
//
// In both cases the f16 weights equal the scratch path's bit for bit
// (`q*_mma_dequant_check_kernel` checks it).
//
// The tile shape doesn't depend on `rows` (MT only sets how many m16 tiles a
// block carries), so each output element adds its products in the same order
// whatever the batch size.

#include <cuda_fp16.h>
#include <cstring>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
#define GEMM_Q_MMA_ENABLED 1
#else
#define GEMM_Q_MMA_ENABLED 0
#endif

#define QMMA_BN 64                   // weight rows per block
#define QMMA_BK 256                  // K per stage: one super-block
#define QMMA_THREADS 128             // 4 warps
#define QMMA_A_STRIDE (QMMA_BK + 8)  // halves per staged activation row (bank-conflict pad)

__device__ __forceinline__ float qmma_le_f16(const unsigned char* b) {
    unsigned short bits = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// Two f32 weights as an f16x2 register (`lo` in the low half), each rounded
// with `__float2half_rn`.
__device__ __forceinline__ unsigned int qmma_pack(float lo, float hi) {
    const __half h0 = __float2half_rn(lo);
    const __half h1 = __float2half_rn(hi);
    unsigned short b0, b1;
    memcpy(&b0, &h0, sizeof(b0));
    memcpy(&b1, &h1, sizeof(b1));
    return (unsigned int)b0 | ((unsigned int)b1 << 16);
}

// Same as dequant.cu's `get_scale_min_k4`.
__device__ __forceinline__ void qmma_q4k_scale_min(unsigned int j, const unsigned char* q, unsigned char* d_out, unsigned char* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Per-type pieces of the GEMM. `kChunks` 16-byte chunks are staged per weight
// row, `kSmemStride` bytes apart; `chunk(blk, c)` is the global address of chunk
// c for the block at `blk` and `smem_offset(blk)` where the block starts within
// the staged row. `begin` reads a block's per-block constants, `step` (called
// when `new_scales(s)`) the scales of k16 step s, and `b_frag` builds a lane's
// B fragment for step s.
struct QmmaQ4K {
    static constexpr unsigned int kBytes = 144;
    static constexpr unsigned int kChunks = 9;
    static constexpr unsigned int kSmemStride = 144;
    struct State {
        float d, dmin, d1, m1;
    };
    __device__ static const uint4* chunk(const unsigned char* blk, unsigned int c) {
        return reinterpret_cast<const uint4*>(blk) + c;
    }
    __device__ static unsigned int smem_offset(const unsigned char*) {
        return 0;
    }
    __device__ static bool new_scales(unsigned int s) {
        return (s & 1u) == 0;
    }
    __device__ static void begin(const unsigned char* block, State* st) {
        st->d = qmma_le_f16(block);
        st->dmin = qmma_le_f16(block + 2);
    }
    // d1 and m1 of scale pair s/2, as the coalesced dequant computes them.
    __device__ static void step(const unsigned char* block, unsigned int s, State* st) {
        unsigned char sc, m;
        qmma_q4k_scale_min(s >> 1, block + 4, &sc, &m);
        st->d1 = st->d * (float)sc;
        st->m1 = st->dmin * (float)m;
    }
    __device__ static unsigned int pair(const unsigned char* p, unsigned int hi, const State& st) {
        const unsigned int q0 = hi ? (p[0] >> 4) : (p[0] & 0xF);
        const unsigned int q1 = hi ? (p[1] >> 4) : (p[1] & 0xF);
        return qmma_pack(st.d1 * (float)q0 - st.m1, st.d1 * (float)q1 - st.m1);
    }
    __device__ static void b_frag(const unsigned char* block, unsigned int s, unsigned int t, const State& st, unsigned int* b0, unsigned int* b1) {
        const unsigned int hi = (s >> 1) & 1u;
        const unsigned char* p = block + 16u + 32u * (s >> 2) + 16u * (s & 1u) + 2u * t;
        *b0 = pair(p, hi, st);
        *b1 = pair(p + 8, hi, st);
    }
};

struct QmmaQ6K {
    static constexpr unsigned int kBytes = 210;
    static constexpr unsigned int kChunks = 14;
    // 60 words: the 8 rows a warp's lanes read in one access land on 8
    // different banks (a 224-byte stride would put two rows on each).
    static constexpr unsigned int kSmemStride = 240;
    struct State {
        float d, ds;
    };
    __device__ static const uint4* chunk(const unsigned char* blk, unsigned int c) {
        return reinterpret_cast<const uint4*>((unsigned long long)blk & ~15ull) + c;
    }
    __device__ static unsigned int smem_offset(const unsigned char* blk) {
        return (unsigned int)((unsigned long long)blk & 15ull);
    }
    __device__ static bool new_scales(unsigned int) {
        return true;
    }
    __device__ static void begin(const unsigned char* block, State* st) {
        st->d = qmma_le_f16(block + 208);
    }
    // `d * (float)sc`, the first product of the coalesced dequant's
    // `d * (float)sc * (float)q`.
    __device__ static void step(const unsigned char* block, unsigned int s, State* st) {
        const unsigned int r = (s >> 1) & 3u;
        const signed char sc = ((const signed char*)(block + 192))[8u * (s >> 3) + (s & 1u) + 2u * r];
        st->ds = st->d * (float)sc;
    }
    __device__ static float weight(unsigned char ql, unsigned char qh, unsigned int r, float ds) {
        const int lo = (r & 2u) ? (ql >> 4) : (ql & 0xF);
        const int q = (int)(lo | (((qh >> (2u * r)) & 3) << 4)) - 32;
        return ds * (float)q;
    }
    __device__ static void b_frag(const unsigned char* block, unsigned int s, unsigned int t, const State& st, unsigned int* b0, unsigned int* b1) {
        const unsigned int n = s >> 3;
        const unsigned int r = (s >> 1) & 3u;
        const unsigned int l = 16u * (s & 1u) + 2u * t;
        const unsigned char* ql = block + 64u * n + l + ((r & 1u) ? 32u : 0u);
        const unsigned char* qh = block + 128u + 32u * n + l;
        *b0 = qmma_pack(weight(ql[0], qh[0], r, st.ds), weight(ql[1], qh[1], r, st.ds));
        *b1 = qmma_pack(weight(ql[8], qh[8], r, st.ds), weight(ql[9], qh[9], r, st.ds));
    }
};

__device__ __forceinline__ void qmma_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}

// w: Q blocks, out_features rows of in_features/256 blocks (Q4_K rows 16-byte
// aligned; Q6_K blocks 2-byte aligned). x16: [rows, in_features] f16.
// y: [rows, out_features] f32, every element written. in_features must be a
// multiple of 256.
// Grid: (ceil(out/64), ceil(rows/(16*MT))), block 128.
template <typename Q, int MT>
__device__ __forceinline__ void gemm_q_mma_impl(
    const unsigned char* __restrict__ w,
    const __half* __restrict__ x16,
    float* __restrict__ y,
    unsigned int rows,
    unsigned int in_features,
    unsigned int out_features) {
#if GEMM_Q_MMA_ENABLED
    constexpr int BM = 16 * MT;
    constexpr int A_CHUNKS = BM * (QMMA_BK / 8) / QMMA_THREADS;  // 16-byte chunks per thread
    constexpr int W_CHUNKS_TOTAL = QMMA_BN * Q::kChunks;
    constexpr int W_CHUNKS = (W_CHUNKS_TOTAL + QMMA_THREADS - 1) / QMMA_THREADS;

    __shared__ __align__(16) __half As[BM * QMMA_A_STRIDE];
    __shared__ __align__(16) unsigned char Ws[QMMA_BN * Q::kSmemStride];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * QMMA_BN;
    const unsigned int m0 = blockIdx.y * BM;
    const unsigned int nblocks = in_features / QMMA_BK;
    const unsigned long long row_bytes = (unsigned long long)nblocks * Q::kBytes;

    uint4 a_reg[A_CHUNKS];
    uint4 w_reg[W_CHUNKS];

    // Global -> registers for super-block column kb. Rows past the matrix load zeros.
    auto load = [&](unsigned int kb) {
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            const unsigned int r = c / (QMMA_BK / 8);
            const unsigned int col = (c % (QMMA_BK / 8)) * 8;
            const unsigned int gr = m0 + r;
            a_reg[i] = gr < rows
                ? *reinterpret_cast<const uint4*>(x16 + (unsigned long long)gr * in_features + kb * QMMA_BK + col)
                : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < W_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            if (c < W_CHUNKS_TOTAL) {
                const unsigned int r = c / Q::kChunks;
                const unsigned int gn = n0 + r;
                w_reg[i] = gn < out_features
                    ? *Q::chunk(w + gn * row_bytes + (unsigned long long)kb * Q::kBytes, c % Q::kChunks)
                    : make_uint4(0, 0, 0, 0);
            }
        }
    };
    // Registers -> shared memory.
    auto store = [&]() {
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            const unsigned int r = c / (QMMA_BK / 8);
            const unsigned int col = (c % (QMMA_BK / 8)) * 8;
            *reinterpret_cast<uint4*>(As + r * QMMA_A_STRIDE + col) = a_reg[i];
        }
        #pragma unroll
        for (int i = 0; i < W_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            if (c < W_CHUNKS_TOTAL) {
                const unsigned int r = c / Q::kChunks;
                *reinterpret_cast<uint4*>(Ws + r * Q::kSmemStride + (c % Q::kChunks) * 16) = w_reg[i];
            }
        }
    };
    // Where weight row `r` of the block (block-relative) starts within its
    // staged row, for super-block column kb. Rows past the matrix staged
    // zeros; any offset that stays inside the staged row reads them.
    auto block_at = [&](unsigned int r, unsigned int kb) -> const unsigned char* {
        const unsigned int gn = n0 + r;
        const unsigned int off = gn < out_features
            ? Q::smem_offset(w + gn * row_bytes + (unsigned long long)kb * Q::kBytes)
            : 0u;
        return Ws + r * Q::kSmemStride + off;
    };

    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[mt][j][e] = 0.0f;
            }
        }
    }

    load(0);
    store();
    __syncthreads();
    for (unsigned int kb = 0; kb < nblocks; kb++) {
        if (kb + 1 < nblocks) {
            load(kb + 1);
        }
        const unsigned char* blk0 = block_at(warp * 16u + g, kb);
        const unsigned char* blk1 = block_at(warp * 16u + g + 8u, kb);
        typename Q::State st[2];
        Q::begin(blk0, &st[0]);
        Q::begin(blk1, &st[1]);
        #pragma unroll
        for (unsigned int s = 0; s < QMMA_BK / 16; s++) {
            if (Q::new_scales(s)) {
                Q::step(blk0, s, &st[0]);
                Q::step(blk1, s, &st[1]);
            }
            unsigned int b[2][2];
            Q::b_frag(blk0, s, t, st[0], &b[0][0], &b[0][1]);
            Q::b_frag(blk1, s, t, st[1], &b[1][0], &b[1][1]);
            #pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const __half* ap = As + (mt * 16 + g) * QMMA_A_STRIDE + s * 16 + 2 * t;
                unsigned int a[4];
                a[0] = *reinterpret_cast<const unsigned int*>(ap);
                a[1] = *reinterpret_cast<const unsigned int*>(ap + 8 * QMMA_A_STRIDE);
                a[2] = *reinterpret_cast<const unsigned int*>(ap + 8);
                a[3] = *reinterpret_cast<const unsigned int*>(ap + 8 * QMMA_A_STRIDE + 8);
                qmma_mma(acc[mt][0], a, b[0][0], b[0][1]);
                qmma_mma(acc[mt][1], a, b[1][0], b[1][1]);
            }
        }
        __syncthreads();
        if (kb + 1 < nblocks) {
            store();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const unsigned int n = n0 + warp * 16u + j * 8u + 2u * t;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned int r = m0 + mt * 16u + g + 8u * h;
                if (r < rows) {
                    float* yr = y + (unsigned long long)r * out_features;
                    if (n < out_features) {
                        yr[n] = acc[mt][j][2 * h];
                    }
                    if (n + 1 < out_features) {
                        yr[n + 1] = acc[mt][j][2 * h + 1];
                    }
                }
            }
        }
    }
#endif
}

#define GEMM_Q_MMA_ENTRY(NAME, Q, MT)                                              \
    extern "C" __global__ void __launch_bounds__(QMMA_THREADS)                     \
    gemm_##NAME##_mma_m##MT##_kernel(                                              \
        const unsigned char* __restrict__ w,                                       \
        const __half* __restrict__ x16,                                            \
        float* __restrict__ y,                                                     \
        unsigned int rows,                                                         \
        unsigned int in_features,                                                  \
        unsigned int out_features) {                                               \
        gemm_q_mma_impl<Q, MT>(w, x16, y, rows, in_features, out_features);        \
    }

GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 1)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 2)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 3)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 4)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 1)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 2)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 3)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 4)

// Test hook: writes the f16 weights the GEMM's lanes build, at the positions
// their fragments say they hold, as a row-major [out_features, in_features]
// matrix. Runs the B-fragment code of `gemm_q_mma_impl` straight from global
// memory; a test compares the output with `dequantize_*_coalesced_f16_kernel`.
// Grid: (ceil(out/8), in/256), block 32: lane g of the warp reads weight row
// 8*blockIdx.x + g, super-block blockIdx.y.
template <typename Q>
__device__ __forceinline__ void q_mma_dequant_check_impl(
    const unsigned char* __restrict__ w,
    unsigned short* __restrict__ out,
    unsigned int in_features,
    unsigned int out_features) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n = blockIdx.x * 8u + g;
    const unsigned int kb = blockIdx.y;
    if (n >= out_features) {
        return;
    }
    const unsigned long long row_bytes = (unsigned long long)(in_features / QMMA_BK) * Q::kBytes;
    const unsigned char* block = w + n * row_bytes + (unsigned long long)kb * Q::kBytes;
    unsigned short* row = out + (unsigned long long)n * in_features + kb * QMMA_BK;
    typename Q::State st;
    Q::begin(block, &st);
    for (unsigned int s = 0; s < QMMA_BK / 16; s++) {
        if (Q::new_scales(s)) {
            Q::step(block, s, &st);
        }
        unsigned int b0, b1;
        Q::b_frag(block, s, t, st, &b0, &b1);
        const unsigned int k = s * 16u + 2u * t;
        row[k] = (unsigned short)(b0 & 0xFFFFu);
        row[k + 1] = (unsigned short)(b0 >> 16);
        row[k + 8] = (unsigned short)(b1 & 0xFFFFu);
        row[k + 9] = (unsigned short)(b1 >> 16);
    }
}

extern "C" __global__ void q4k_mma_dequant_check_kernel(
    const unsigned char* __restrict__ w,
    unsigned short* __restrict__ out,
    unsigned int in_features,
    unsigned int out_features) {
    q_mma_dequant_check_impl<QmmaQ4K>(w, out, in_features, out_features);
}

extern "C" __global__ void q6k_mma_dequant_check_kernel(
    const unsigned char* __restrict__ w,
    unsigned short* __restrict__ out,
    unsigned int in_features,
    unsigned int out_features) {
    q_mma_dequant_check_impl<QmmaQ6K>(w, out, in_features, out_features);
}
