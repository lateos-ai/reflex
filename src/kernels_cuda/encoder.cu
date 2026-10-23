// Kernels only embedding (encoder) models use: bidirectional online attention, the
// Llama-4-style query temperature scale, and mean pooling. Loaded by the dense path
// only for a model with `<arch>.attention.causal = false` (Model::load_dense), so
// causal models never load this module.

#include "attention_online.cuh"

// attention_online_kernel without the causal mask: query row r sees all
// start_pos + rows positions (rows = gridDim.x). Same layout, launch and output as
// attention_online_kernel (attention_online.cu); num_splits must be 1.
extern "C" __global__ void attention_online_bidir_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p
) {
    attention_online<float, false>(q, k, v, dst, p);
}

// Query temperature scaling (llama.cpp's build_inp_attn_scale, HF's
// _get_llama_4_attn_scale), applied to Q after RoPE: row r (absolute position
// pos = start_pos + r) is multiplied by 1 + beta * log(floor(pos / floor_scale) + 1).
// It is exactly 1 below floor_scale, so the host launches this only when a row
// reaches it.
//   q  [rows, row_len], in place.
// Launch: any 1-D grid, grid-stride loop.
extern "C" __global__ void attn_temp_scale_kernel(
    float* __restrict__ q,
    unsigned int start_pos,
    unsigned int rows,
    unsigned int row_len,
    float beta,
    float floor_scale
) {
    const unsigned long long total = (unsigned long long)rows * row_len;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float pos = (float)(start_pos + (unsigned int)(i / row_len));
        q[i] *= 1.0f + beta * logf(floorf(pos / floor_scale) + 1.0f);
    }
}

// Mean over rows: out[c] = (sum over r of x[r, c]) / rows, the sum taken in row order
// by one thread per column, so the result doesn't depend on the launch shape.
//   x [rows, cols] -> out [cols]
// Launch: 1-D grid covering cols, any block size.
extern "C" __global__ void mean_pool_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int r = 0; r < rows; r++) {
        sum += x[(unsigned long long)r * cols + c];
    }
    out[c] = sum / (float)rows;
}
