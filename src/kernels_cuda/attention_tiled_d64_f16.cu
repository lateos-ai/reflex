// Tiled prefill attention for head dim 64 for an f16 K/V cache (REFLEX_KV_CACHE=f16). See attention_tiled.cuh.

#include "attention_tiled.cuh"

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_d64_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<__half, 64, 64>(q, k, v, dst, p, rows);
}
