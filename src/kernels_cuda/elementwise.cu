// In-place residual add: a[i] += b[i]. Used to keep residual-stream adds
// device-resident (see model.rs's forward functions) instead of downloading
// both operands to host just to add two vectors.
extern "C" __global__ void add_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    unsigned int n
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        a[i] += b[i];
    }
}
