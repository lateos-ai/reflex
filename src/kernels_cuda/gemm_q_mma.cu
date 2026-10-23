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
// two n8 tiles and every m16 tile of the block. A fragments come in through
// `ldmatrix`. The wide tile (`gemm_q*_mma_m8_kernel`, below) carries 128 input
// rows with the same warps and per-element math.
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
// rounding of `dequantize_q4k_coalesced_f16_kernel` ((float)q through
// `qmma_u2f`, which gives the same float without an I2F).
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
// Split-K: grid z splits the super-blocks into ranges of `kb_per_split`. With one
// split the kernel writes y; with more, split z writes its partial sums to
// y + z * rows * out_features (a workspace) and `gemm_q_mma_splitk_reduce_kernel`
// adds them in split order. No atomics, so repeated runs give the same bits.
//
// The summation order doesn't depend on `rows` or the tile (MT only sets how
// many m16 tiles a block carries), so for a given split count each output
// element adds its products in the same order whatever the batch size or
// tile. The host picks the split count from the grid size, which does depend
// on `rows`.

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
// with `__float2half_rn`. sm_80+ converts both in one instruction
// (`cvt.rn.f16x2.f32`), with the same rounding.
__device__ __forceinline__ unsigned int qmma_pack(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const __half2 h = __floats2half2_rn(lo, hi);
    unsigned int bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    const __half h0 = __float2half_rn(lo);
    const __half h1 = __float2half_rn(hi);
    unsigned short b0, b1;
    memcpy(&b0, &h0, sizeof(b0));
    memcpy(&b1, &h1, sizeof(b1));
    return (unsigned int)b0 | ((unsigned int)b1 << 16);
#endif
}

// (float)q for an integer 0 <= q < 2^23, exactly: 2^23 + q as a float's bits,
// minus 2^23. Two cheap ops instead of an I2F, which runs at a quarter of
// the FP32 rate.
__device__ __forceinline__ float qmma_u2f(unsigned int q) {
    return __int_as_float(0x4B000000u | q) - 8388608.0f;
}

__device__ __forceinline__ unsigned int qmma_u16(const unsigned char* p) {
    return *reinterpret_cast<const unsigned short*>(p);
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
    // Weights from the two bytes in `v` (the first in the low byte).
    __device__ static unsigned int pair(unsigned int v, unsigned int hi, const State& st) {
        const unsigned int q0 = hi ? ((v >> 4) & 0xF) : (v & 0xF);
        const unsigned int q1 = hi ? (v >> 12) : ((v >> 8) & 0xF);
        return qmma_pack(st.d1 * qmma_u2f(q0) - st.m1, st.d1 * qmma_u2f(q1) - st.m1);
    }
    __device__ static void b_frag(const unsigned char* block, unsigned int s, unsigned int t, const State& st, unsigned int* b0, unsigned int* b1) {
        const unsigned int hi = (s >> 1) & 1u;
        const unsigned char* p = block + 16u + 32u * (s >> 2) + 16u * (s & 1u) + 2u * t;
        *b0 = pair(qmma_u16(p), hi, st);
        *b1 = pair(qmma_u16(p + 8), hi, st);
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
    // `ds * (float)q` for q = the 6-bit value minus 32. The 6-bit value u
    // goes through qmma_u2f's trick with the 32 folded into the constant:
    // 2^23 + u - (2^23 + 32) is exact, so this is (float)q.
    __device__ static float weight(unsigned int ql, unsigned int qh, unsigned int r, float ds) {
        const unsigned int lo = (r & 2u) ? (ql >> 4) : (ql & 0xF);
        const unsigned int u = lo | (((qh >> (2u * r)) & 3u) << 4);
        return ds * (__int_as_float(0x4B000000u | u) - 8388640.0f);
    }
    // Weights for the two bytes at ql / qh (each pair read as one 16-bit word).
    __device__ static unsigned int pair(const unsigned char* ql, const unsigned char* qh, unsigned int r, float ds) {
        const unsigned int l = qmma_u16(ql);
        const unsigned int h = qmma_u16(qh);
        return qmma_pack(weight(l & 0xFF, h & 0xFF, r, ds), weight(l >> 8, h >> 8, r, ds));
    }
    __device__ static void b_frag(const unsigned char* block, unsigned int s, unsigned int t, const State& st, unsigned int* b0, unsigned int* b1) {
        const unsigned int n = s >> 3;
        const unsigned int r = (s >> 1) & 3u;
        const unsigned int l = 16u * (s & 1u) + 2u * t;
        const unsigned char* ql = block + 64u * n + l + ((r & 1u) ? 32u : 0u);
        const unsigned char* qh = block + 128u + 32u * n + l;
        *b0 = pair(ql, qh, r, st.ds);
        *b1 = pair(ql + 8, qh + 8, r, st.ds);
    }
};

// mma.sync on one 16x8 accumulator tile per call; the asm has no side effects
// (not volatile), so ptxas may interleave independent tiles.
__device__ __forceinline__ void qmma_k16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void qmma_k8(float* c, unsigned int a0, unsigned int a1, unsigned int b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
#endif
}

// One k16 step for the two n8 tiles c0 (B fragment b0) and c1 (b1) sharing the
// A fragment a. sm_75 has no k16 shape: each tile gets two m16n8k8 (k 0-7,
// then 8-15), issued for both tiles before either's second half so the two
// dependent halves of one tile don't run back to back.
__device__ __forceinline__ void qmma_mma2(float* c0, float* c1, const unsigned int* a, const unsigned int* b0, const unsigned int* b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    qmma_k16(c0, a, b0[0], b0[1]);
    qmma_k16(c1, a, b1[0], b1[1]);
#else
    qmma_k8(c0, a[0], a[1], b0[0]);
    qmma_k8(c1, a[0], a[1], b1[0]);
    qmma_k8(c0, a[2], a[3], b0[1]);
    qmma_k8(c1, a[2], a[3], b1[1]);
#endif
}

// A lane's A fragment (a0..a3, as above) for the m16 x k16 tile whose rows this
// lane addresses: lane l passes row (l%8) + 8*((l/8)%2) of the tile, at k
// offset 8*(l/16). `ldmatrix` (sm_75+) loads what four 32-bit reads per lane
// would, in one instruction.
__device__ __forceinline__ void qmma_ldsm_a(unsigned int* a, const __half* p) {
#if GEMM_Q_MMA_ENABLED && defined(__CUDA_ARCH__)
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(s));
#endif
}

// cp.async (sm_80+): copy 16 bytes from global to shared without going
// through registers, or write 16 zero bytes when `valid` is false (`src` is
// then not read).
__device__ __forceinline__ void qmma_cp16(void* dst, const void* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned int s = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 16 : 0));
#endif
}

__device__ __forceinline__ void qmma_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// w: Q blocks, out_features rows of in_features/256 blocks (Q4_K rows 16-byte
// aligned; Q6_K blocks 2-byte aligned). x16: [rows, in_features] f16.
// y: [splits, rows, out_features] f32, every element written. in_features must
// be a multiple of 256; every split's range must be non-empty.
// Grid: (ceil(out/64), ceil(rows/(16*MT)), splits), block 128.
template <typename Q, int MT>
__device__ __forceinline__ void gemm_q_mma_impl(
    const unsigned char* __restrict__ w,
    const __half* __restrict__ x16,
    float* __restrict__ y,
    unsigned int rows,
    unsigned int in_features,
    unsigned int out_features,
    unsigned int kb_per_split) {
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
    const unsigned int kb_begin = blockIdx.z * kb_per_split;
    const unsigned int kb_end = min(kb_begin + kb_per_split, nblocks);
    y += (unsigned long long)blockIdx.z * rows * out_features;

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

    load(kb_begin);
    store();
    __syncthreads();
    for (unsigned int kb = kb_begin; kb < kb_end; kb++) {
        if (kb + 1 < kb_end) {
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
                // Rows are 528 bytes apart, so the 8 rows of each 8x8 matrix
                // sit on 8 different 16-byte bank groups.
                unsigned int a[4];
                qmma_ldsm_a(a, As + (mt * 16 + (lane & 7u) + 8u * ((lane >> 3) & 1u)) * QMMA_A_STRIDE + s * 16 + 8u * (lane >> 4));
                qmma_mma2(acc[mt][0], acc[mt][1], a, b[0], b[1]);
            }
        }
        __syncthreads();
        if (kb + 1 < kb_end) {
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
        unsigned int out_features,                                                 \
        unsigned int kb_per_split) {                                               \
        gemm_q_mma_impl<Q, MT>(w, x16, y, rows, in_features, out_features,         \
                               kb_per_split);                                      \
    }

GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 1)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 2)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 3)
GEMM_Q_MMA_ENTRY(q4k, QmmaQ4K, 4)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 1)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 2)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 3)
GEMM_Q_MMA_ENTRY(q6k, QmmaQ6K, 4)

// The wide tile: 64 weight rows by 128 input rows (MT = 8) per block, so each
// weight is dequantized once per 128 rows instead of once per 64, and every
// B fragment a lane builds feeds 8 m16 tiles. Same warps, fragments, k-step
// order and per-element math as `gemm_q_mma_impl`, so each output element
// gets the same products added in the same order, bit for bit, as the 64-row
// tile with the same split count.
//
// The activations come in slices of 32 K (2 k16 steps, 8 KB for 128 rows),
// double-buffered: a whole super-block's worth (64 KB) leaves no room for a
// second buffer, and 64-K slices (51-63 KB in all) fit only one block per SM
// on an A10G, which ran 20-30% slower. Each slice row is 64 bytes with no
// padding; chunk c of row r is stored at chunk c ^ ((r / 2) % 4), so the 8
// rows of each `ldmatrix` 8x8 matrix hit 8 different bank groups. The weight
// bytes of super-block kb + 1 come in an eighth per slice of kb, into the
// second of two weight buffers. sm_80+ copies both with cp.async straight
// into the idle buffer; sm_75 loads them into registers during the slice's
// math and stores them after it. One __syncthreads per slice either way.
//
// Dynamic shared memory: 2 * 128 * 32 halves of activations, then
// 2 * 64 * Q::kSmemStride bytes of weights (Q4_K 34816 bytes, Q6_K 47104,
// both under the 48 KB default). Grid: (ceil(out/64), ceil(rows/128),
// splits), block 128. m16 tiles wholly past `rows` skip their MMAs. The host
// uses this tile from sm_80 up; on a T4 it ran no faster than the 64-row one.
#define QMMA_WIDE_MT 8
#define QMMA_WIDE_SK 32  // K per activation slice
// Launch bounds: on sm_75 (the register-staged path) ptxas needs the
// one-block hint to stay below 256 registers without spilling; from sm_80
// (cp.async, no staging registers) it does better without it.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define QMMA_WIDE_BOUNDS __launch_bounds__(QMMA_THREADS, 1)
#else
#define QMMA_WIDE_BOUNDS __launch_bounds__(QMMA_THREADS)
#endif

extern __shared__ uint4 qmma_dyn_smem[];

template <typename Q>
__device__ __forceinline__ void gemm_q_mma_wide_impl(
    const unsigned char* __restrict__ w,
    const __half* __restrict__ x16,
    float* __restrict__ y,
    unsigned int rows,
    unsigned int in_features,
    unsigned int out_features,
    unsigned int kb_per_split) {
#if GEMM_Q_MMA_ENABLED
    constexpr int MT = QMMA_WIDE_MT;
    constexpr int SK = QMMA_WIDE_SK;
    constexpr int BM = 16 * MT;
    constexpr int SLICES = QMMA_BK / SK;
    constexpr int STEPS = SK / 16;           // k16 steps per slice
    constexpr int A_SLICE = BM * SK;         // halves
    constexpr int A_ROW_CHUNKS = SK / 8;     // 16-byte chunks per row
    // Rows sharing one 128-byte line; the swizzle key of row r is
    // (r / LINE_ROWS) % A_ROW_CHUNKS, so the 8 rows of an ldmatrix 8x8
    // matrix land on 8 different 16-byte bank groups.
    constexpr int LINE_ROWS = 8 / A_ROW_CHUNKS;
    constexpr int A_CHUNKS = BM * A_ROW_CHUNKS / QMMA_THREADS;
    constexpr int W_TOTAL = QMMA_BN * Q::kChunks;
    constexpr int W_QUOTA = (W_TOTAL + SLICES - 1) / SLICES;  // chunks per slice
    constexpr int W_CHUNKS = (W_QUOTA + QMMA_THREADS - 1) / QMMA_THREADS;
    constexpr int W_BUF = QMMA_BN * Q::kSmemStride;
    static_assert(A_ROW_CHUNKS == 8 || A_ROW_CHUNKS == 4, "the swizzle assumes 4 or 8 chunks per row");
    static_assert(BM * A_ROW_CHUNKS % QMMA_THREADS == 0, "even activation chunks");

    __half* As = reinterpret_cast<__half*>(qmma_dyn_smem);
    unsigned char* Ws = reinterpret_cast<unsigned char*>(qmma_dyn_smem) + 2 * A_SLICE * sizeof(__half);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * QMMA_BN;
    const unsigned int m0 = blockIdx.y * BM;
    const unsigned int nblocks = in_features / QMMA_BK;
    const unsigned long long row_bytes = (unsigned long long)nblocks * Q::kBytes;
    const unsigned int kb_begin = blockIdx.z * kb_per_split;
    const unsigned int kb_end = min(kb_begin + kb_per_split, nblocks);
    // m16 tiles holding at least one real row.
    const unsigned int live = min((unsigned int)MT, (rows - m0 + 15u) / 16u);
    y += (unsigned long long)blockIdx.z * rows * out_features;

#if __CUDA_ARCH__ < 800
    uint4 a_reg[A_CHUNKS];
    uint4 w_reg[W_CHUNKS];
#endif

    // Activation slice `sl` of super-block column kb into buffer `buf`
    // (sm_75: into registers, then `a_put`).
    auto a_get = [&](unsigned int kb, unsigned int sl, unsigned int buf) {
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            const unsigned int r = c / A_ROW_CHUNKS;
            const unsigned int ch = c % A_ROW_CHUNKS;
            const unsigned int gr = m0 + r;
            const __half* src = x16 + (unsigned long long)min(gr, rows - 1u) * in_features + kb * QMMA_BK + sl * SK + ch * 8;
#if __CUDA_ARCH__ >= 800
            qmma_cp16(As + buf * A_SLICE + r * SK + ((ch ^ ((r / LINE_ROWS) % A_ROW_CHUNKS)) * 8), src, gr < rows);
#else
            (void)buf;
            a_reg[i] = gr < rows ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
        }
    };
    auto a_put = [&](unsigned int buf) {
#if __CUDA_ARCH__ < 800
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; i++) {
            const unsigned int c = tid + i * QMMA_THREADS;
            const unsigned int r = c / A_ROW_CHUNKS;
            const unsigned int ch = c % A_ROW_CHUNKS;
            *reinterpret_cast<uint4*>(As + buf * A_SLICE + r * SK + ((ch ^ ((r / LINE_ROWS) % A_ROW_CHUNKS)) * 8)) = a_reg[i];
        }
#else
        (void)buf;
#endif
    };
    // Quarter q of super-block column kb's weight bytes into buffer `buf`.
    // Rows past the matrix stage zeros.
    auto w_get = [&](unsigned int kb, unsigned int q, unsigned int buf) {
        #pragma unroll
        for (int i = 0; i < W_CHUNKS; i++) {
            const unsigned int j = tid + i * QMMA_THREADS;
            const unsigned int c = q * W_QUOTA + j;
            if (j < W_QUOTA && c < W_TOTAL) {
                const unsigned int r = c / Q::kChunks;
                const unsigned int gn = n0 + r;
                const uint4* src = Q::chunk(w + min(gn, out_features - 1u) * row_bytes + (unsigned long long)kb * Q::kBytes, c % Q::kChunks);
#if __CUDA_ARCH__ >= 800
                qmma_cp16(Ws + buf * W_BUF + r * Q::kSmemStride + (c % Q::kChunks) * 16, src, gn < out_features);
#else
                (void)buf;
                w_reg[i] = gn < out_features ? *src : make_uint4(0, 0, 0, 0);
#endif
            }
        }
    };
    auto w_put = [&](unsigned int q, unsigned int buf) {
#if __CUDA_ARCH__ < 800
        #pragma unroll
        for (int i = 0; i < W_CHUNKS; i++) {
            const unsigned int j = tid + i * QMMA_THREADS;
            const unsigned int c = q * W_QUOTA + j;
            if (j < W_QUOTA && c < W_TOTAL) {
                const unsigned int r = c / Q::kChunks;
                *reinterpret_cast<uint4*>(Ws + buf * W_BUF + r * Q::kSmemStride + (c % Q::kChunks) * 16) = w_reg[i];
            }
        }
#else
        (void)q;
        (void)buf;
#endif
    };
    auto block_at = [&](unsigned int buf, unsigned int r, unsigned int kb) -> const unsigned char* {
        const unsigned int gn = n0 + r;
        const unsigned int off = gn < out_features
            ? Q::smem_offset(w + gn * row_bytes + (unsigned long long)kb * Q::kBytes)
            : 0u;
        return Ws + buf * W_BUF + r * Q::kSmemStride + off;
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

    // This lane's ldmatrix row within an m16 tile, its k chunk (0 or 1)
    // within a k16 step, and its row's swizzle key (the same for every m16
    // tile: 16-row offsets don't change it).
    const unsigned int a_row = (lane & 7u) + 8u * ((lane >> 3) & 1u);
    const unsigned int a_half = lane >> 4;
    const unsigned int a_key = (a_row / LINE_ROWS) % A_ROW_CHUNKS;

    a_get(kb_begin, 0, 0);
    a_put(0);
    #pragma unroll
    for (unsigned int q = 0; q < SLICES; q++) {
        w_get(kb_begin, q, 0);
        w_put(q, 0);
    }
    qmma_cp_wait();
    __syncthreads();

    unsigned int abuf = 0;
    for (unsigned int kb = kb_begin; kb < kb_end; kb++) {
        const unsigned int wbuf = (kb - kb_begin) & 1u;
        const bool more = kb + 1 < kb_end;
        const unsigned char* blk0 = block_at(wbuf, warp * 16u + g, kb);
        const unsigned char* blk1 = block_at(wbuf, warp * 16u + g + 8u, kb);
        typename Q::State st[2];
        Q::begin(blk0, &st[0]);
        Q::begin(blk1, &st[1]);
        // Fully unrolled, so every k-step's byte offsets and nibble choice
        // are constants (a partial unroll was 15-40% slower on an A10G).
        #pragma unroll
        for (unsigned int sl = 0; sl < SLICES; sl++) {
            const bool next_a = sl + 1 < SLICES || more;
            if (next_a) {
                a_get(sl + 1 < SLICES ? kb : kb + 1, (sl + 1) % SLICES, abuf ^ 1u);
            }
            if (more) {
                w_get(kb + 1, sl, wbuf ^ 1u);
            }
            const __half* Ab = As + abuf * A_SLICE;
            #pragma unroll
            for (unsigned int j = 0; j < STEPS; j++) {
                const unsigned int s = sl * STEPS + j;
                if (Q::new_scales(s)) {
                    Q::step(blk0, s, &st[0]);
                    Q::step(blk1, s, &st[1]);
                }
                unsigned int b[2][2];
                Q::b_frag(blk0, s, t, st[0], &b[0][0], &b[0][1]);
                Q::b_frag(blk1, s, t, st[1], &b[1][0], &b[1][1]);
                const __half* ap = Ab + a_row * SK + (((2u * j + a_half) ^ a_key) * 8u);
                #pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    if ((unsigned int)mt < live) {
                        unsigned int a[4];
                        qmma_ldsm_a(a, ap + mt * 16 * SK);
                        qmma_mma2(acc[mt][0], acc[mt][1], a, b[0], b[1]);
                    }
                }
            }
            if (next_a) {
                a_put(abuf ^ 1u);
            }
            if (more) {
                w_put(sl, wbuf ^ 1u);
            }
            qmma_cp_wait();
            __syncthreads();
            abuf ^= 1u;
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

#define GEMM_Q_MMA_WIDE_ENTRY(NAME, Q)                                             \
    extern "C" __global__ void QMMA_WIDE_BOUNDS                                    \
    gemm_##NAME##_mma_m8_kernel(                                                   \
        const unsigned char* __restrict__ w,                                       \
        const __half* __restrict__ x16,                                            \
        float* __restrict__ y,                                                     \
        unsigned int rows,                                                         \
        unsigned int in_features,                                                  \
        unsigned int out_features,                                                 \
        unsigned int kb_per_split) {                                               \
        gemm_q_mma_wide_impl<Q>(w, x16, y, rows, in_features, out_features,        \
                                kb_per_split);                                     \
    }

GEMM_Q_MMA_WIDE_ENTRY(q4k, QmmaQ4K)
GEMM_Q_MMA_WIDE_ENTRY(q6k, QmmaQ6K)

// y[i] = ws[0][i] + ws[1][i] + ... + ws[splits-1][i], added in that order, for
// i < n (n = rows * out_features). Grid: ceil(n/256), block 256. Not gated on
// the arch: it is plain f32 code, but only launched where the GEMM runs.
extern "C" __global__ void gemm_q_mma_splitk_reduce_kernel(
    const float* __restrict__ ws,
    float* __restrict__ y,
    unsigned int n,
    unsigned int splits) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = ws[i];
    for (unsigned int z = 1; z < splits; z++) {
        s += ws[(unsigned long long)z * n + i];
    }
    y[i] = s;
}

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
