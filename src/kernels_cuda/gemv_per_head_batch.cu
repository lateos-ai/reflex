// Batched-prefill variant of the per-head weight application `Model::gemv_per_head`
// performs on the host as a loop over `gemv_kernel` (see gemv.cu), once per head plus
// one device-to-device copy per head: applies a per-head-stacked weight tensor `w`
// ([in_features, out_features, n_head], same layout convention as MoE's per-expert
// tensors -- see gemv.cu's module doc and model.rs's Model::gemv_expert) to every
// head of every row of a batched input in ONE launch (grid gains head and row
// dimensions) instead of `rows * n_head` separate gemv_kernel launches plus
// `rows * n_head` device-to-device copies. That launch count is not a minor
// inefficiency at real prefill lengths -- see model.rs's
// Model::forward_mla_attn_block_batched doc comment for the actual math -- so this
// kernel exists specifically to avoid reintroducing the kind of per-call-overhead
// regression CLAUDE.md documents for Phase 2 round 1. Used for DeepSeek-V2/V3 MLA's
// batched-prefill absorption (`wk_b`) and decompression (`wv_b`) steps.
//
// `x` is read with explicit row/head strides plus a per-head offset instead of
// requiring a contiguous [rows, n_head, in_features] layout -- absorption reads
// directly out of the wider per-head `wq` projection buffer (q_nope is a strided
// sub-slice of each head's row, interleaved with q_pe), so no separate gather step is
// needed before this kernel for that call site. `out` is always a freshly allocated,
// contiguous [rows, n_head, out_features] buffer.
extern "C" __global__ void gemv_per_head_batch_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int n_head,
    unsigned int in_features,
    unsigned int out_features,
    unsigned int x_row_stride,
    unsigned int x_head_stride,
    unsigned int x_head_offset
) {
    unsigned int row = blockIdx.z;
    unsigned int head = blockIdx.y;
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    const float* x_row_head =
        x + (unsigned long long)row * x_row_stride + (unsigned long long)head * x_head_stride + x_head_offset;
    const float* w_head = w + (unsigned long long)head * in_features * out_features;

    for (unsigned int j = global_tid; j < out_features; j += gridDim.x * blockDim.x) {
        const float* w_row = w_head + (unsigned long long)j * in_features;
        float sum = 0.0f;
        for (unsigned int i = 0; i < in_features; i++) {
            sum += x_row_head[i] * w_row[i];
        }
        out[((unsigned long long)row * n_head + head) * out_features + j] = sum;
    }
}
