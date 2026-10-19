// Tiled prefill attention for head dim 256 (Qwen3.5's gated attention) for an f16 K/V
// cache (REFLEX_KV_CACHE=f16). See attention_tiled.cuh.

#include "attention_tiled.cuh"

// Two blocks per SM caps the kernel at 128 registers, which spills 8 bytes on sm_90
// (the f16 widening needs a few more registers than the f32 kernel); one block per
// SM there uses 167 and doesn't spill. sm_75/86 fit in 128 either way.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TA_F16_D256_MIN_BLOCKS 1
#else
#define TA_F16_D256_MIN_BLOCKS 2
#endif

extern "C" __global__ void __launch_bounds__(TA_THREADS, TA_F16_D256_MIN_BLOCKS) attention_tiled_d256_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<__half, 256, 64>(q, k, v, dst, p, rows);
}
