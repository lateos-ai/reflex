// SwiGLU activation over two separate row-major (batch, hidden_size) input
// buffers `gate`/`up` (previously a single concatenated (batch,
// 2*hidden_size) buffer -- split into two plain GEMV outputs so callers
// never need a device-side concatenation step, see model.rs's Phase 2 round
// 2 doc comments). Output is row-major (batch, hidden_size): SiLU(gate) *
// up, where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
extern "C" __global__ void silu_and_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int hidden_size
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = batch * hidden_size;

    for (unsigned int idx = global_tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        float g = gate[idx];
        float u = up[idx];
        float silu = g / (1.0f + expf(-g));
        output[idx] = silu * u;
    }
}
