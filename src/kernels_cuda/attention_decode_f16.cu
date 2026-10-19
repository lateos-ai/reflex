// attention_decode.cu's kernels for an f16 K/V cache (REFLEX_KV_CACHE=f16): the same
// code, loading each K/V element as f16 and widening it to f32. A separate module so
// the default f32 build's decode module (and its first-decode-step load) doesn't grow.

#include "attention_decode.cuh"

// At least 4 blocks per SM (128 registers per thread), what the launcher aims for:
// without it ptxas spilled the d64/h8 kernel on sm_90.
#define DEC_KERNEL_F16(D, HB)                                                               \
    extern "C" __global__ void __launch_bounds__(DEC_WARPS * 32, 4)                         \
        attention_decode_d##D##_h##HB##_f16_kernel(const float* __restrict__ q,            \
                                                   const __half* __restrict__ k,           \
                                                   const __half* __restrict__ v,           \
                                                   float* __restrict__ dst, AttnParams p) { \
        attention_decode<__half, D / 32, HB>(q, k, v, dst, p);                              \
    }

DEC_KERNEL_F16(64, 1)
DEC_KERNEL_F16(64, 2)
DEC_KERNEL_F16(64, 4)
DEC_KERNEL_F16(64, 8)
DEC_KERNEL_F16(128, 1)
DEC_KERNEL_F16(128, 2)
DEC_KERNEL_F16(128, 4)
DEC_KERNEL_F16(128, 8)
DEC_KERNEL_F16(256, 1)
DEC_KERNEL_F16(256, 2)
DEC_KERNEL_F16(256, 4)
