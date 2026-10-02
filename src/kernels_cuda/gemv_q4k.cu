// Matrix-vector (and few-row matrix-matrix) product straight from Q4_K blocks:
// y[r] = x[r] @ W^T for r in 0..rows (rows <= GEMV_Q4K_MAX_ROWS), where W is the
// GGUF's raw Q4_K data, row-major (out_features, in_features), never expanded
// to f32 in device memory. Used when weights are kept quantized on the GPU
// (REFLEX_QUANT_RESIDENT=1, see docs/design/quantized-resident-weights.md).
//
// Same launch geometry as `gemv_kernel` (gemv.cu): one warp per output row,
// lanes reduce with a warp shuffle. Within a row, the unit of work is one
// 32-element Q4_K sub-block: a 144-byte super-block (`d: f16`, `dmin: f16`,
// `scales[12]`, `qs[128]`) holds 8 of them, sub-block `s` taking the low
// (s even) or high (s odd) nibbles of `qs[32*(s/2) .. 32*(s/2)+32]` with
// scale/min pair `s`. Lane `lane` handles sub-blocks lane, lane+32, ...
//
// Each weight is decoded with exactly the expression `dequantize_q4k_kernel`
// (dequant.cu) uses, `d1 * (float)q - m1`, so every weight value matches the
// f32-resident path bit for bit; only the order of the dot product's additions
// differs (as it already does between gemv_kernel and cuBLAS). Activations
// stay f32 -- no Q8_1 activation quantization.
//
// Several input rows share each decoded weight: the kernel keeps one
// accumulator per row, so prefill reads the weight bytes once per
// GEMV_Q4K_MAX_ROWS rows instead of once per row.

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

// x: [rows, in_features] f32, row-major. w: Q4_K blocks, out_features rows of
// in_features/256 blocks each. y: [rows, out_features] f32, row-major.
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

    float acc[GEMV_Q4K_MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < GEMV_Q4K_MAX_ROWS; r++) {
        acc[r] = 0.0f;
    }

    const unsigned int units = nb * 8u;
    for (unsigned int u = lane; u < units; u += 32u) {
        const unsigned int b = u >> 3;
        const unsigned int s = u & 7u;
        const unsigned char* block = row_ptr + (unsigned long long)b * 144ull;

        const float d = q4k_le_f16(block);
        const float dmin = q4k_le_f16(block + 2);
        unsigned char sc, m;
        q4k_scale_min(s, block + 4, &sc, &m);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)m;

        const unsigned char* q = block + 16 + 32u * (s >> 1);
        const unsigned int shift = (s & 1u) ? 4u : 0u;
        // Element offset of this sub-block within the row:
        // 256*b + 64*(s/2) + 32*(s&1).
        const unsigned int x_off = (b << 8) + ((s >> 1) << 6) + ((s & 1u) << 5);

        #pragma unroll 4
        for (unsigned int l = 0; l < 32u; l++) {
            const float wv = d1 * (float)((q[l] >> shift) & 0xF) - m1;
            #pragma unroll
            for (unsigned int r = 0; r < GEMV_Q4K_MAX_ROWS; r++) {
                if (r < rows) {
                    acc[r] += x[(unsigned long long)r * in_features + x_off + l] * wv;
                }
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
