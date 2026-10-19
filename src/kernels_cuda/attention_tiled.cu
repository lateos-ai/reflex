// Tiled prefill attention for head dims 256 and 512: Gemma 4's sliding and global
// layers, and Qwen3.5's gated attention (256). See attention_tiled.cuh.

#include "attention_tiled.cuh"

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_d256_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<float, 256, 64>(q, k, v, dst, p, rows);
}

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_d512_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<float, 512, 32>(q, k, v, dst, p, rows);
}
