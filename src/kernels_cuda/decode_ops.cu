// Fused single-token (decode) ops, one launch where the plain path takes two. Each
// does the same float operations in the same order as the kernels it replaces, so
// outputs are bit-identical; only launches go away. Loaded lazily on the first
// decode step (Model::decode_ops), so cold start never pays for this module.
#include "rmsnorm.cuh"

// a += b (residual add), then out = rmsnorm(a) * weight, for one row: add_kernel
// followed by rmsnorm_kernel. One block; the launcher sizes dynamic shared memory to
// the row, which it only does when the row fits.
extern "C" __global__ void add_rmsnorm_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ float4 row_s4[];
    float* row_s = reinterpret_cast<float*>(row_s4);
    __shared__ float rms_inv_s;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = a[i] + b[i];
        a[i] = v;
        row_s[i] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rms_inv_s = rmsnorm_rms_inv(row_s, hidden_size, eps, true);
    }
    __syncthreads();
    const float rms_inv = rms_inv_s;
    for (unsigned int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        out[col] = row_s[col] * rms_inv * weight[col];
    }
}

// Two per-head RMSNorms in one launch (Qwen3's Q-Norm and K-Norm): blocks
// [0, rows1) normalize x1's rows with w1 into o1, the rest x2's with w2 into o2.
// Each block is one rmsnorm_kernel row.
extern "C" __global__ void rmsnorm_pair_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ w1,
    float* __restrict__ o1,
    unsigned int rows1,
    const float* __restrict__ x2,
    const float* __restrict__ w2,
    float* __restrict__ o2,
    unsigned int hidden_size,
    float eps
) {
    extern __shared__ float4 row_s4[];
    float* row_s = reinterpret_cast<float*>(row_s4);
    __shared__ float rms_inv_s;
    const bool first = blockIdx.x < rows1;
    const unsigned int row = first ? blockIdx.x : blockIdx.x - rows1;
    const float* xr = (first ? x1 : x2) + (unsigned long long)row * hidden_size;
    const float* weight = first ? w1 : w2;
    float* out = (first ? o1 : o2) + (unsigned long long)row * hidden_size;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_s[i] = xr[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rms_inv_s = rmsnorm_rms_inv(row_s, hidden_size, eps, true);
    }
    __syncthreads();
    const float rms_inv = rms_inv_s;
    for (unsigned int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        out[col] = row_s[col] * rms_inv * weight[col];
    }
}
