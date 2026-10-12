// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight, row-major (rows, hidden_size).
// Also used for Qwen3's QK-Norm (per-head RMSNorm on Q/K before RoPE) by simply
// calling with rows=num_heads, hidden_size=head_dim.
//
// One block per row (grid-stride over rows). The row's sum of squares is computed
// once, by one thread, in index order -- the same additions in the same order as the
// original one-thread-per-element kernel, which recomputed it in every thread
// (rows * hidden_size^2 loads; ~20 ms per norm for a 2,763 x 3,840 prefill). So the
// output is bit-identical to that kernel's, at rows * hidden_size loads.
//
// With dynamic shared memory of hidden_size floats, the block first stages the row
// there with coalesced loads, so the single summing thread reads shared memory rather
// than walking global memory alone. Launched with no shared memory (a row wider than
// the launcher's limit), it reads x directly; the values are the same either way.
extern "C" __global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ float row_s[];
    __shared__ float rms_inv_s;
    unsigned int dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    const bool staged = dyn_smem >= hidden_size * sizeof(float);

    for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* xr = x + (unsigned long long)row * hidden_size;
        const float* src = xr;
        if (staged) {
            for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
                row_s[i] = xr[i];
            }
            __syncthreads();
            src = row_s;
        }
        if (threadIdx.x == 0) {
            float sum_sq = 0.0f;
            for (unsigned int i = 0; i < hidden_size; i++) {
                float v = src[i];
                sum_sq += v * v;
            }
            rms_inv_s = 1.0f / sqrtf(sum_sq / hidden_size + eps);
        }
        __syncthreads();
        const float rms_inv = rms_inv_s;
        float* out = output + (unsigned long long)row * hidden_size;
        for (unsigned int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
            out[col] = src[col] * rms_inv * weight[col];
        }
        // The next row overwrites row_s and rms_inv_s.
        __syncthreads();
    }
}
