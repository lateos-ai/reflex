// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight, row-major (rows, hidden_size).
// Also used for Qwen3's QK-Norm (per-head RMSNorm on Q/K before RoPE) by simply
// calling with rows=num_heads, hidden_size=head_dim.
extern "C" __global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int hidden_size,
    float eps
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = rows * hidden_size;

    for (unsigned int idx = global_tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        unsigned int row = idx / hidden_size;
        unsigned int col = idx % hidden_size;
        unsigned int row_base = row * hidden_size;

        float sum_sq = 0.0f;
        for (unsigned int i = 0; i < hidden_size; i++) {
            float v = x[row_base + i];
            sum_sq += v * v;
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
        output[idx] = x[idx] * rms_inv * weight[col];
    }
}
