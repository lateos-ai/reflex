// Same per-warp dot-product body as gemv_kernel (see gemv.cu for the
// coalescing rationale and reduction-order caveat), but gathers a
// caller-chosen subset of w's output rows by index instead of every row
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
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int j = blockIdx.x * warps_per_block + warp_id;
    if (j >= num_rows) {
        return;
    }

    const float* row_ptr = w + (unsigned long long)row_indices[j] * in_features;
    float sum = 0.0f;

    if ((in_features & 3u) == 0u) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        unsigned int vec_count = in_features >> 2;
        for (unsigned int idx = lane; idx < vec_count; idx += 32u) {
            float4 xv = x4[idx];
            float4 wv = row4[idx];
            sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (unsigned int i = lane; i < in_features; i += 32u) {
            sum += x[i] * row_ptr[i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        y[j] = sum;
    }
}
