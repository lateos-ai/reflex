// Plain f32 matrix-vector product: y = x @ W^T (the nn.Linear convention),
// for a single input row (this MVP always forwards one token at a time).
// `w` is row-major (out_features, in_features); each output element is one
// dot product over in_features. No cuBLAS, no dequant-resident tricks --
// correctness-first for the cold-start MVP, not a throughput kernel.
extern "C" __global__ void gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int j = global_tid; j < out_features; j += gridDim.x * blockDim.x) {
        const float* row = w + (unsigned long long)j * in_features;
        float sum = 0.0f;
        for (unsigned int i = 0; i < in_features; i++) {
            sum += x[i] * row[i];
        }
        y[j] = sum;
    }
}
