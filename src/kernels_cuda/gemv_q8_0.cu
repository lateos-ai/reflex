// Matrix-vector (and few-row matrix-matrix) product straight from Q8_0 blocks:
// y[r] = x[r] @ W^T for r in 0..rows (rows <= GEMV_Q8_0_MAX_ROWS), where W is
// the GGUF's raw Q8_0 data, row-major (out_features, in_features), never
// expanded in device memory. Used by the Gemma 4 path when weights are kept
// quantized on the GPU (REFLEX_QUANT_RESIDENT=1, see
// docs/design/quantized-resident-weights.md, "Q8_0 on the Gemma 4 path").
//
// A Q8_0 block is 34 bytes: `d: f16`, then `qs[32]` (int8). Blocks are 34
// bytes apart, so only 2-byte aligned; weights are read with 2-byte loads.
//
// Geometry: one warp per output row, 8 warps per 256-thread block, warp-shuffle
// reduction (as `gemv_q4k_kernel`). Each warp iteration covers 256 elements (8
// blocks): lane `l` takes 4 elements at `4l` (block `l/8`, bytes
// `4*(l%8)..+4`) and 4 at `128 + 4l` (block `4 + l/8`). So 8 consecutive lanes
// read one block's 32 quant bytes, and the lanes' float4 reads of `x` are
// consecutive 16-byte words (no shared-memory bank conflicts).
//
// `x` is staged through shared memory one tile of Q8_0_TILE elements at a
// time, for all `rows` input rows at once; the block's 8 warps (8 output rows)
// then read it from there. Without that, every warp re-reads every input row
// from L2, which made the multi-row Q4_K kernel L2-bound (see the design doc's
// Q4_K results). Dynamic shared memory: rows * Q8_0_TILE floats.
//
// Each weight is decoded with exactly `dequantize_q8_0_kernel`'s expression
// (dequant.cu), `(float)qs[j] * d`, which is exact in f32: every weight value
// matches the f32-resident path bit for bit; only the order of the dot
// product's additions differs. Activations stay f32 -- no Q8_1 activation
// quantization.

#include <cuda_fp16.h>
#include <cstring>

#define GEMV_Q8_0_MAX_ROWS 16
#define Q8_0_TILE 512

__device__ __forceinline__ float q8_0_d(const unsigned char* block) {
    const unsigned short bits = *reinterpret_cast<const unsigned short*>(block);
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// Four weights `qs[j..j+4]` of `block` (j even), decoded as `(float)q * d`.
__device__ __forceinline__ void q8_0_decode4(const unsigned char* block, unsigned int j, float d, float* w) {
    const unsigned short* p = reinterpret_cast<const unsigned short*>(block + 2 + j);
    const unsigned short lo = p[0];
    const unsigned short hi = p[1];
    w[0] = (float)(signed char)(lo & 0xFF) * d;
    w[1] = (float)(signed char)(lo >> 8) * d;
    w[2] = (float)(signed char)(hi & 0xFF) * d;
    w[3] = (float)(signed char)(hi >> 8) * d;
}

__device__ __forceinline__ float dot4(const float* x, const float* w) {
    const float4 v = *reinterpret_cast<const float4*>(x);
    return v.x * w[0] + v.y * w[1] + v.z * w[2] + v.w * w[3];
}

// x: [rows, in_features] f32, row-major. w: Q8_0 blocks, out_features rows of
// in_features/32 blocks each; the arena is 256-byte aligned and a row is
// in_features/32 * 34 bytes, so every block starts 2-byte aligned. y: [rows,
// out_features] f32. in_features must be a multiple of 32. Launch with block =
// 256 threads and rows * Q8_0_TILE * 4 bytes of dynamic shared memory.
extern "C" __global__ void gemv_q8_0_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features,
    unsigned int rows
) {
    extern __shared__ float4 xs_f4[];  // float4: 16-byte aligned for dot4
    float* xs = reinterpret_cast<float*>(xs_f4);
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row = blockIdx.x * warps_per_block + warp_id;
    // Rows past the end still take part in the tile loads and barriers.
    const bool active = row < out_features;

    const unsigned int nb = in_features >> 5;
    const unsigned char* row_ptr = w + (unsigned long long)(active ? row : 0u) * nb * 34ull;
    const unsigned int sub = lane >> 3;        // block within each half of a 256-element group
    const unsigned int j = (lane & 7u) << 2;   // first of this lane's 4 elements in the block

    float acc[GEMV_Q8_0_MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < GEMV_Q8_0_MAX_ROWS; r++) {
        acc[r] = 0.0f;
    }

    for (unsigned int k0 = 0; k0 < in_features; k0 += Q8_0_TILE) {
        const unsigned int tile = min((unsigned int)Q8_0_TILE, in_features - k0);
        __syncthreads();  // every warp is done with the previous tile
        for (unsigned int i = threadIdx.x; i < rows * tile; i += blockDim.x) {
            const unsigned int r = i / tile;
            const unsigned int c = i - r * tile;
            xs[r * Q8_0_TILE + c] = x[(unsigned long long)r * in_features + k0 + c];
        }
        __syncthreads();
        if (!active) {
            continue;
        }
        const unsigned int tile_blocks = tile >> 5;
        for (unsigned int g = 0; g < tile_blocks; g += 8u) {
            #pragma unroll
            for (unsigned int part = 0; part < 2; part++) {
                const unsigned int tb = g + 4u * part + sub;  // block within the tile
                if (tb < tile_blocks) {
                    const unsigned char* block = row_ptr + (unsigned long long)((k0 >> 5) + tb) * 34ull;
                    const float d = q8_0_d(block);
                    float wv[4];
                    q8_0_decode4(block, j, d, wv);
                    const unsigned int e = (tb << 5) + j;  // element offset within the tile
                    #pragma unroll
                    for (unsigned int r = 0; r < GEMV_Q8_0_MAX_ROWS; r++) {
                        if (r < rows) {
                            acc[r] += dot4(xs + r * Q8_0_TILE + e, wv);
                        }
                    }
                }
            }
        }
    }

    if (!active) {
        return;
    }
    #pragma unroll
    for (unsigned int r = 0; r < GEMV_Q8_0_MAX_ROWS; r++) {
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

// Q8_0 -> f32 or f16, one thread per element (256-thread blocks cover 8 Q8_0
// blocks), so consecutive threads write consecutive elements; dequant.cu's
// `dequantize_q8_0_kernel` gives each thread a whole block, which writes
// 128 bytes apart per thread. Same decode expression, so the f32 output is
// bit-identical to `dequantize_q8_0_kernel`'s and the f16 output is
// `__float2half_rn` of it, the same rounding as `dequantize_q8_0_f16_kernel`.
// Used by the quantized-resident prefill path that feeds cuBLAS from a scratch
// buffer in the `--weights` dtype (`Model::gemm`). Launch with
// grid = ceil(num_blocks * 32 / 256), block = 256.
template <typename T>
__device__ __forceinline__ T q8_0_store(float v);
template <>
__device__ __forceinline__ float q8_0_store<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __half q8_0_store<__half>(float v) {
    return __float2half_rn(v);
}

template <typename T>
__device__ __forceinline__ void dequantize_q8_0_coalesced_impl(
    const unsigned char* __restrict__ blocks,
    T* __restrict__ y,
    unsigned int num_blocks
) {
    const unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (unsigned long long)num_blocks * 32ull) {
        return;
    }
    const unsigned char* block = blocks + (e >> 5) * 34ull;
    const float d = q8_0_d(block);
    const signed char q = (signed char)block[2 + (e & 31ull)];
    y[e] = q8_0_store<T>((float)q * d);
}

extern "C" __global__ void dequantize_q8_0_coalesced_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q8_0_coalesced_impl<float>(blocks, y, num_blocks);
}

extern "C" __global__ void dequantize_q8_0_coalesced_f16_kernel(
    const unsigned char* __restrict__ blocks,
    __half* __restrict__ y,
    unsigned int num_blocks
) {
    dequantize_q8_0_coalesced_impl<__half>(blocks, y, num_blocks);
}
