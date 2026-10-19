// Matrix-vector (and few-row matrix-matrix) product straight from Q4_K blocks:
// y[r] = x[r] @ W^T for r in 0..rows (rows <= GEMV_Q4K_MAX_ROWS), where W is the
// GGUF's raw Q4_K data, row-major (out_features, in_features), never expanded
// to f32 in device memory. Used when weights are kept quantized on the GPU
// (REFLEX_QUANT_RESIDENT=1, see docs/design/quantized-resident-weights.md).
//
// Same launch geometry as `gemv_kernel` (gemv.cu): one warp per output row,
// lanes reduce with a warp shuffle. A 144-byte Q4_K super-block is `d: f16`,
// `dmin: f16`, `scales[12]` (8 packed 6-bit scale/min pairs) and `qs[128]`;
// `qs[32*g .. 32*g+32]` holds elements 64*g .. 64*g+64 of the block, low
// nibbles first (scale pair 2g), then high nibbles (pair 2g+1).
//
// Memory layout per warp iteration: 8 lanes per super-block, 4 super-blocks
// at a time. Lane chunk `c` (0..8) loads 16 contiguous bytes of `qs` with one
// 16-byte load (8 lanes together read the block's 128 bytes), which hold 16
// low-nibble and 16 high-nibble weights of group `c/2`. Those 32 weights are
// decoded into registers once and reused for every input row; `x` is read
// with float4 loads.
//
// Each weight is decoded with exactly the expression `dequantize_q4k_kernel`
// (dequant.cu) uses, `d1 * (float)q - m1`, so every weight value matches the
// f32-resident path bit for bit; only the order of the dot product's additions
// differs (as it already does between gemv_kernel and cuBLAS). Activations
// stay f32 -- no Q8_1 activation quantization.

#include <cuda_fp16.h>
#include <cstring>

#define GEMV_Q4K_MAX_ROWS 8

__device__ __forceinline__ float q4k_le_f16(const unsigned char* b) {
    unsigned short bits = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// Same as dequant.cu's `get_scale_min_k4` (port of ggml-quants.c's).
__device__ __forceinline__ void q4k_scale_min(unsigned int j, const unsigned char* q, unsigned char* d_out, unsigned char* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float dot16(const float* __restrict__ x, const float* w) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float4 v = x4[k];
        s += v.x * w[4 * k] + v.y * w[4 * k + 1] + v.z * w[4 * k + 2] + v.w * w[4 * k + 3];
    }
    return s;
}

// x: [rows, in_features] f32, row-major. w: Q4_K blocks, out_features rows of
// in_features/256 blocks each (row starts 16-byte aligned: 144 is a multiple of
// 16 and the arena is 256-byte aligned). y: [rows, out_features] f32.
// in_features must be a multiple of 256 (ggml's own invariant for Q4_K rows).
extern "C" __global__ void gemv_q4k_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features,
    unsigned int rows
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= out_features) {
        return;
    }

    const unsigned int nb = in_features >> 8;
    const unsigned char* row_ptr = w + (unsigned long long)row * nb * 144ull;
    const unsigned int c = lane & 7u;     // 16-byte chunk of qs
    const unsigned int g = c >> 1;        // 64-element group
    const unsigned int half = c & 1u;     // which 16 of the group's 32 bytes

    float acc[GEMV_Q4K_MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < GEMV_Q4K_MAX_ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (unsigned int b = lane >> 3; b < nb; b += 4u) {
        const unsigned char* block = row_ptr + (unsigned long long)b * 144ull;
        const float d = q4k_le_f16(block);
        const float dmin = q4k_le_f16(block + 2);
        unsigned char sc, m;
        q4k_scale_min(2u * g, block + 4, &sc, &m);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)m;
        q4k_scale_min(2u * g + 1u, block + 4, &sc, &m);
        const float d2 = d * (float)sc;
        const float m2 = dmin * (float)m;

        const uint4 qv = *reinterpret_cast<const uint4*>(block + 16 + 16u * c);
        const unsigned char* q = reinterpret_cast<const unsigned char*>(&qv);
        float wlo[16], whi[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            wlo[i] = d1 * (float)(q[i] & 0xF) - m1;
            whi[i] = d2 * (float)(q[i] >> 4) - m2;
        }

        // Element offsets of the low and high runs within the row.
        const unsigned int e_lo = (b << 8) + (g << 6) + (half << 4);
        const unsigned int e_hi = e_lo + 32u;
        #pragma unroll
        for (unsigned int r = 0; r < GEMV_Q4K_MAX_ROWS; r++) {
            if (r < rows) {
                const float* xr = x + (unsigned long long)r * in_features;
                acc[r] += dot16(xr + e_lo, wlo) + dot16(xr + e_hi, whi);
            }
        }
    }

    #pragma unroll
    for (unsigned int r = 0; r < GEMV_Q4K_MAX_ROWS; r++) {
        if (r < rows) {
            float sum = acc[r];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0u) {
                y[(unsigned long long)r * out_features + row] = sum;
            }
        }
    }
}

// One-row (decode) Q4_K GEMV, GEMV_Q4K_R1_ROWS output rows per warp: each lane
// loads its 32 activations of a super-block once and reuses them for every row,
// so activation reads per weight drop four-fold. `gemv_q4k_kernel` reads ~4
// bytes of `x` per 0.56 bytes of weight and is bound by that rather than by
// DRAM. Per row, the lane layout, decode expression and summation order are
// `gemv_q4k_kernel`'s, so every output matches it bit for bit. A nibble becomes
// a float through the exponent trick (exact for 0..15) rather than an I2F,
// which runs at quarter rate on Turing. On a T4 this is 1.6-1.8x faster than
// `gemv_q4k_kernel` with rows == 1, on Qwen3-0.6B and Mistral 7B shapes; 8
// rows per warp or two block groups in flight per lane were slower
// (`quant_resident_tests::gemv_q4k_decode_variants_timings`).
// Launch: block = 32 * GEMV_Q4K_R1_WARPS threads,
// grid = ceil(out_features / (GEMV_Q4K_R1_WARPS * GEMV_Q4K_R1_ROWS)).
#define GEMV_Q4K_R1_ROWS 4
#define GEMV_Q4K_R1_WARPS 4

__device__ __forceinline__ float q4k_nib(unsigned int q) {
    return __int_as_float(0x4B000000 | q) - 8388608.0f;
}

// `dot16` on registers: the same expression, so the same rounding.
__device__ __forceinline__ float dot16_regs(const float* xv, const float* w) {
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        s += xv[4 * k] * w[4 * k] + xv[4 * k + 1] * w[4 * k + 1] + xv[4 * k + 2] * w[4 * k + 2] + xv[4 * k + 3] * w[4 * k + 3];
    }
    return s;
}

extern "C" __global__ void gemv_q4k_r1_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features
) {
    constexpr int R = GEMV_Q4K_R1_ROWS;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row0 = (blockIdx.x * GEMV_Q4K_R1_WARPS + warp_id) * R;
    if (row0 >= out_features) {
        return;
    }
    const unsigned int nb = in_features >> 8;
    const unsigned long long row_bytes = (unsigned long long)nb * 144ull;
    const unsigned int c = lane & 7u;
    const unsigned int g = c >> 1;
    const unsigned int half = c & 1u;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    for (unsigned int b = lane >> 3; b < nb; b += 4u) {
        // Issue every row's header (`d`, `dmin`, `scales[12]`: bytes 0..16) and
        // `qs` chunk before using any of them.
        uint4 hdr[R], qv[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r < out_features) {
                const unsigned char* block = w + (row0 + r) * row_bytes + (unsigned long long)b * 144ull;
                hdr[r] = *reinterpret_cast<const uint4*>(block);
                qv[r] = *reinterpret_cast<const uint4*>(block + 16 + 16u * c);
            }
        }
        const unsigned int e_lo = (b << 8) + (g << 6) + (half << 4);
        float xlo[16], xhi[16];
        const float4* xl4 = reinterpret_cast<const float4*>(x + e_lo);
        const float4* xh4 = reinterpret_cast<const float4*>(x + e_lo + 32u);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const float4 a = xl4[k];
            const float4 h = xh4[k];
            xlo[4 * k] = a.x;
            xlo[4 * k + 1] = a.y;
            xlo[4 * k + 2] = a.z;
            xlo[4 * k + 3] = a.w;
            xhi[4 * k] = h.x;
            xhi[4 * k + 1] = h.y;
            xhi[4 * k + 2] = h.z;
            xhi[4 * k + 3] = h.w;
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r < out_features) {
                unsigned char sb[16];
                memcpy(sb, &hdr[r], 16);
                const float d = q4k_le_f16(sb);
                const float dmin = q4k_le_f16(sb + 2);
                unsigned char sc, m;
                q4k_scale_min(2u * g, sb + 4, &sc, &m);
                const float d1 = d * (float)sc;
                const float m1 = dmin * (float)m;
                q4k_scale_min(2u * g + 1u, sb + 4, &sc, &m);
                const float d2 = d * (float)sc;
                const float m2 = dmin * (float)m;
                const unsigned char* q = reinterpret_cast<const unsigned char*>(&qv[r]);
                float wlo[16], whi[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    wlo[i] = d1 * q4k_nib(q[i] & 0xF) - m1;
                    whi[i] = d2 * q4k_nib(q[i] >> 4) - m2;
                }
                acc[r] += dot16_regs(xlo, wlo) + dot16_regs(xhi, whi);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float sum = acc[r];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0u && row0 + r < out_features) {
            y[row0 + r] = sum;
        }
    }
}

// One-row (decode) Q6_K GEMV, 4 output rows per warp: each lane loads its 32
// activations of a super-block once and reuses them for all 4 rows, so
// activation reads per weight drop four-fold. `gemv_q6k_kernel` reads ~4 bytes
// of `x` per 0.73 bytes of weight and is bound by that rather than by DRAM.
// Per row, the lane layout, decode expression and summation order are
// `gemv_q6k_kernel`'s, so every output matches it bit for bit. On a T4 this
// is about 1.5-1.7x faster than `gemv_q6k_kernel` with rows == 1.
// Launch: block = 32 * GEMV_Q6K_R1_WARPS threads,
// grid = ceil(out_features / (GEMV_Q6K_R1_WARPS * GEMV_Q6K_R1_ROWS)).
#define GEMV_Q6K_R1_ROWS 4
#define GEMV_Q6K_R1_WARPS 4

extern "C" __global__ void gemv_q6k_r1_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features
) {
    constexpr int R = GEMV_Q6K_R1_ROWS;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row0 = (blockIdx.x * GEMV_Q6K_R1_WARPS + warp_id) * R;
    if (row0 >= out_features) {
        return;
    }
    const unsigned int nb = in_features >> 8;
    const unsigned long long row_bytes = (unsigned long long)nb * 210ull;
    const unsigned int l = lane;
    const unsigned int is = l / 16;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }
    for (unsigned int b = 0; b < nb; b++) {
        // Load headers for all 4 rows: d (bytes 208..209) and scales (bytes 192..207).
        uint4 hdr[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r < out_features) {
                const unsigned char* block = w + (row0 + r) * row_bytes + (unsigned long long)b * 210ull;
                hdr[r] = *reinterpret_cast<const uint4*>(block + 200);
            }
        }
        const float* xb = x + (b << 8);
        #pragma unroll
        for (unsigned int n = 0; n < 2; n++) {
            const unsigned int ql_off = 64u * n;
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (row0 + r < out_features) {
                    const unsigned char* block = w + (row0 + r) * row_bytes + (unsigned long long)b * 210ull;
                    const unsigned char ql_l = block[ql_off + l];
                    const unsigned char ql_l32 = block[ql_off + l + 32];
                    const unsigned char qh = block[128 + 32u * n + l];
                    const signed char* sc_all = (const signed char*)(block + 192);
                    const signed char* sc = sc_all + 8u * n;
                    const int q1 = (int)((ql_l & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql_l32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql_l >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql_l32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                    const float d = q4k_le_f16(&hdr[r].x);
                    const float* xn = xb + 128u * n;
                    acc[r] += xn[l] * d * (float)sc[is] * (float)q1
                            + xn[l + 32] * d * (float)sc[is + 2] * (float)q2
                            + xn[l + 64] * d * (float)sc[is + 4] * (float)q3
                            + xn[l + 96] * d * (float)sc[is + 6] * (float)q4;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float sum = acc[r];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0u && row0 + r < out_features) {
            y[row0 + r] = sum;
        }
    }
}

// Q4_K -> f32 or f16, one CUDA block per 256-element super-block and one thread
// per element, so consecutive threads write consecutive elements (dequant.cu's
// `dequantize_q4k_kernel` gives each thread a whole block, which writes 1 KB
// apart per thread and runs far below memory bandwidth). Same decode
// expression, so the f32 output is bit-identical to `dequantize_q4k_kernel`'s,
// and the f16 output is `__float2half_rn` of it, the same rounding as
// `dequantize_q4k_f16_kernel`. Used by the quantized-resident prefill path that
// feeds cuBLAS from a scratch buffer in the `--weights` dtype (`Model::gemm`).
// Launch with grid = num_blocks, block = 256.
template <typename T>
__device__ __forceinline__ T q4k_store(float v);
template <>
__device__ __forceinline__ float q4k_store<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half q4k_store<__half>(float v) {
    return __float2half_rn(v);
}

template <typename T>
__device__ __forceinline__ void dequantize_q4k_coalesced_impl(
    const unsigned char* __restrict__ blocks,
    T* __restrict__ y,
    unsigned int num_blocks
) {
    const unsigned int bi = blockIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned int j = threadIdx.x;          // element within the block
    const unsigned char* block = blocks + (unsigned long long)bi * 144ull;
    const unsigned int g = j >> 6;               // 64-element group
    const unsigned int hi = (j >> 5) & 1u;       // high nibbles?
    const unsigned int l = j & 31u;
    const float d = q4k_le_f16(block);
    const float dmin = q4k_le_f16(block + 2);
    unsigned char sc, m;
    q4k_scale_min(2u * g + hi, block + 4, &sc, &m);
    const float d1 = d * (float)sc;
    const float m1 = dmin * (float)m;
    const unsigned char q = block[16 + 32u * g + l];
    y[(unsigned long long)bi * 256ull + j] =
        q4k_store<T>(d1 * (float)(hi ? (q >> 4) : (q & 0xF)) - m1);
}

extern "C" __global__ void dequantize_q4k_coalesced_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q4k_coalesced_impl<float>(blocks, y, num_blocks);
}

extern "C" __global__ void dequantize_q4k_coalesced_f16_kernel(
    const unsigned char* __restrict__ blocks,
    __half* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q4k_coalesced_impl<__half>(blocks, y, num_blocks);
}

// Q6_K -> f32 or f16, one CUDA block per 256-element super-block and one thread
// per element (`dequantize_q6k_kernel` gives each thread a whole block, whose
// writes land 1 KB apart). Thread `j` decodes element `j` with that kernel's
// expression `d * (float)sc * (float)q`, so the output is bit-identical; the
// f16 form rounds with `__float2half_rn`, as `dequantize_q6k_f16_kernel` does.
// Used by the quantized-resident prefill path's scratch buffer for dense Q6_K
// layer tensors (`Model::gemm_quant_scratch`). Launch with grid = num_blocks,
// block = 256.
template <typename T>
__device__ __forceinline__ void dequantize_q6k_coalesced_impl(
    const unsigned char* __restrict__ blocks,
    T* __restrict__ y,
    unsigned int num_blocks
) {
    const unsigned int bi = blockIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned int j = threadIdx.x;
    const unsigned char* block = blocks + (unsigned long long)bi * 210ull;
    const unsigned int n = j >> 7;          // 128-element half
    const unsigned int t = (j >> 5) & 3u;   // which of the four 32-runs
    const unsigned int l = j & 31u;
    const unsigned int is = l >> 4;
    const unsigned char ql = block[64u * n + l + ((t & 1u) ? 32u : 0u)];
    const unsigned char qh = block[128u + 32u * n + l];
    const int lo = (t & 2u) ? (ql >> 4) : (ql & 0xF);
    const int q = (int)(lo | (((qh >> (2u * t)) & 3) << 4)) - 32;
    const signed char sc = ((const signed char*)(block + 192))[8u * n + is + 2u * t];
    const float d = q4k_le_f16(block + 208);
    y[(unsigned long long)bi * 256ull + j] = q4k_store<T>(d * (float)sc * (float)q);
}

extern "C" __global__ void dequantize_q6k_coalesced_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q6k_coalesced_impl<float>(blocks, y, num_blocks);
}

extern "C" __global__ void dequantize_q6k_coalesced_f16_kernel(
    const unsigned char* __restrict__ blocks,
    __half* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q6k_coalesced_impl<__half>(blocks, y, num_blocks);
}

// y = x @ W^T for one input row, straight from Q6_K blocks (210 bytes:
// `ql[128]`, `qh[64]`, `scales[16]` (int8), `d: f16`). Used for a
// quantized-resident LM head, whose GGUF type is Q6_K in Q4_K_M files.
//
// One warp per output row, one 256-element block per warp iteration. Lane `l`
// handles position `l` of each 32-element run, mirroring the `l` loop of
// `dequantize_q6k_kernel`: per half `n` it reads `ql[64n + l]`,
// `ql[64n + l + 32]` and `qh[32n + l]` (consecutive lanes, consecutive bytes)
// and decodes 4 weights with exactly that kernel's expression
// `d * (float)sc * (float)q`, so weights are bit-identical to the f32 path.
// Blocks are 210 bytes (not 16-byte aligned), hence byte loads.
extern "C" __global__ void gemv_q6k_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= out_features) {
        return;
    }
    const unsigned int nb = in_features >> 8;
    const unsigned char* row_ptr = w + (unsigned long long)row * nb * 210ull;
    const unsigned int l = lane;
    const unsigned int is = l / 16;
    float sum = 0.0f;
    for (unsigned int b = 0; b < nb; b++) {
        const unsigned char* block = row_ptr + (unsigned long long)b * 210ull;
        const signed char* sc_all = (const signed char*)(block + 192);
        const float d = q4k_le_f16(block + 208);
        const float* xb = x + (b << 8);
        #pragma unroll
        for (unsigned int n = 0; n < 2; n++) {
            const unsigned char* ql = block + 64u * n;
            const unsigned char qh = block[128 + 32u * n + l];
            const signed char* sc = sc_all + 8u * n;
            const int q1 = (int)((ql[l] & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;
            const float w1 = d * (float)sc[is] * (float)q1;
            const float w2 = d * (float)sc[is + 2] * (float)q2;
            const float w3 = d * (float)sc[is + 4] * (float)q3;
            const float w4 = d * (float)sc[is + 6] * (float)q4;
            const float* xn = xb + 128u * n;
            sum += xn[l] * w1 + xn[l + 32] * w2 + xn[l + 64] * w3 + xn[l + 96] * w4;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        y[row] = sum;
    }
}
