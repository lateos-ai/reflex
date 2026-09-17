// Smoke-test kernel for the AOT compilation pipeline (build.rs -> nvcc -> cubin/PTX,
// loaded at process start via cuModuleLoadData with zero runtime source compilation).
// Not a real inference kernel yet -- proves the pipeline end to end before any
// model-architecture work starts, per the cold-start-first MVP ordering.
extern "C" __global__ void axpy_f32(float a, const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}
