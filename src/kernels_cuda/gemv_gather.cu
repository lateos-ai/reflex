// Same per-thread dot-product body as gemv_kernel (see gemv.cu), but gathers
// a caller-chosen subset of w's output rows by index instead of every row
// 0..out_features -- System1's whole point: only the handful of candidate
// token ids' logit rows are needed, not a full [hidden_size, vocab_size]
// GEMV. `w` is still row-major (out_features, in_features); `row_indices[j]`
// is an absolute row index into that out_features dimension. `y` has length
// num_rows (compact, NOT vocab-sized) -- the other half of the win, since the
// D2H transfer back to host is num_rows floats instead of vocab_size floats.
extern "C" __global__ void gemv_gather_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const unsigned int* __restrict__ row_indices,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int num_rows
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int j = global_tid; j < num_rows; j += gridDim.x * blockDim.x) {
        const float* row = w + (unsigned long long)row_indices[j] * in_features;
        float sum = 0.0f;
        for (unsigned int i = 0; i < in_features; i++) {
            sum += x[i] * row[i];
        }
        y[j] = sum;
    }
}
