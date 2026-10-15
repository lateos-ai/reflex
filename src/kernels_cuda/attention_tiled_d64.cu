// Tiled prefill attention for head dim 64. See attention_tiled.cuh.

#include "attention_tiled.cuh"

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_d64_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<64, 64>(q, k, v, dst, p, rows);
}
