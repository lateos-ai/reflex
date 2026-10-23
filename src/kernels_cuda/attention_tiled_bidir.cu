// Tiled prefill attention without the causal mask, for embedding models
// (`<arch>.attention.causal = false`, e.g. Nemotron-3.5-Embed): every query row sees
// all start_pos + rows positions. Head dims 64 and 128, f32 K/V. A separate module so
// causal models never load it. See attention_tiled.cuh.

#include "attention_tiled.cuh"

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_bidir_d64_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<float, 64, 64, false>(q, k, v, dst, p, rows);
}

extern "C" __global__ void __launch_bounds__(TA_THREADS, 2) attention_tiled_bidir_d128_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p,
    unsigned int rows
) {
    attention_tiled<float, 128, 64, false>(q, k, v, dst, p, rows);
}
