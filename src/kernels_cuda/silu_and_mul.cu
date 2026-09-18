// SwiGLU activation. Row-major (batch, 2*hidden_size) input `gate_up`, where
// each row is `gate | up` (first hidden_size elements are gate, next
// hidden_size are up). Output is row-major (batch, hidden_size):
// SiLU(gate) * up, where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
extern "C" __global__ void silu_and_mul_kernel(
    const float* __restrict__ gate_up,
    float* __restrict__ output,
    unsigned int batch,
    unsigned int hidden_size
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_elements = batch * hidden_size;

    for (unsigned int idx = global_tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        unsigned int row = idx / hidden_size;
        unsigned int col = idx % hidden_size;
        unsigned int row_base = row * 2 * hidden_size;

        float gate = gate_up[row_base + col];
        float up = gate_up[row_base + hidden_size + col];
        float silu = gate / (1.0f + expf(-gate));

        output[idx] = silu * up;
    }
}
