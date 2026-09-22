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

// Splits Qwen3.5 hybrid Gated Attention's fused query+gate projection output
// (row-major (rows, num_heads, 2*head_dim), each head's row laid out as
// [q(head_dim), gate(head_dim)]) into separate q/gate buffers (each row-major
// (rows, num_heads, head_dim)). Used by both the m=1 decode step and the
// batched-prefill path (see model.rs's Model::forward_gated_attn_mixer/
// forward_gated_attn_mixer_batched) -- replaces a per-call host round trip
// with a single device-resident kernel launch that scales to any row count.
extern "C" __global__ void split_qg_kernel(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int rows
) {
    unsigned long long total = (unsigned long long)rows * num_heads * head_dim;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int d = idx % head_dim;
        unsigned int head = (idx / head_dim) % num_heads;
        unsigned long long row = idx / (head_dim * (unsigned long long)num_heads);

        unsigned long long qg_base = (row * num_heads + head) * (2ull * head_dim);
        q[idx] = qg[qg_base + d];
        gate[idx] = qg[qg_base + head_dim + d];
    }
}

// In-place sigmoid gating: out[i] *= sigmoid(gate[i]), flat over n elements --
// row-count-agnostic (same shape as add_kernel), so the same launch serves
// both Gated Attention's m=1 decode step and its batched-prefill counterpart.
extern "C" __global__ void sigmoid_gate_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    unsigned int n
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        out[i] *= 1.0f / (1.0f + expf(-gate[i]));
    }
}
