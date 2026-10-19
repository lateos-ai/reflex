// Online-softmax ("flash-decoding" style) causal attention, one kernel for every
// attention shape the engine has: GQA decode and batched prefill (Qwen3, Llama,
// Qwen3.5's gated-attention layers) and MLA decode and batched prefill (DeepSeek's
// single shared compressed KV row). It replaces the four kernels in attention.cu,
// attention_prefill.cu, mla_attention.cu and mla_attention_prefill.cu, which stay
// selectable with REFLEX_ATTN_KERNEL=legacy for A/B comparison.
//
// Why: the old kernels kept the whole score row in shared memory (capping a sequence
// at ~11K positions), paid a block-wide __syncthreads reduction per position, and ran
// the softmax on one thread. Here:
//   - each warp scores its own positions (warp-shuffle dot product, no block syncs),
//     keeping a running max `m`, running sum `l` and an output accumulator in registers
//     (online softmax), so shared memory no longer grows with sequence length;
//   - the block's warps are merged once at the end;
//   - long decode contexts are split across blocks (grid.z), each writing a partial
//     (acc, m, l) that attention_online_combine_kernel merges.
//
// Layout (all row-major, f32):
//   q    [rows, num_q_heads, qk_dim]
//   K    position p of KV head h starts at k + p*k_pos_stride + h*k_head_stride
//   V    position p of KV head h starts at v + p*v_pos_stride + h*v_head_stride
//        (MLA passes the same compressed cache as K and V, with v_dim <= qk_dim and
//        head strides 0)
//   dst  num_splits == 1: the output, [rows, num_q_heads, v_dim]
//        num_splits  > 1: partials, [rows, num_q_heads, num_splits, v_dim + 2], each
//        holding the unnormalized acc[v_dim], then m, then l
// Query row r attends to positions [0, start_pos + r] (causal), or with a sliding
// window (window > 0) only to the last `window` of them: position j is visible from
// query position i iff j <= i and i - j < window (llama.cpp's LLAMA_SWA_TYPE_STANDARD).
// Query head qh reads KV head qh / group_size.
//
// Launch: grid (rows, num_q_heads, num_splits), block ATTN_WARPS * 32 threads, dynamic
// shared memory (qk_dim + ATTN_WARPS * (v_dim + 2)) * sizeof(float). qk_dim and v_dim
// may be any size up to 32 * ATTN_MAX_V_PER_LANE (1024) -- no power-of-two padding.

#include "attention_online.cuh"

extern "C" __global__ void attention_online_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p
) {
    attention_online<float>(q, k, v, dst, p);
}

// Merges attention_online_kernel's per-split partials into the output.
//   partial [rows, num_q_heads, num_splits, v_dim + 2]  ->  out [rows, num_q_heads, v_dim]
// Launch: grid (rows, num_q_heads), block 128 threads (any size works), no shared memory.
extern "C" __global__ void attention_online_combine_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    unsigned int num_q_heads,
    unsigned int v_dim,
    unsigned int num_splits
) {
    const unsigned long long qrow = (unsigned long long)blockIdx.x * num_q_heads + blockIdx.y;
    const float* parts = partial + qrow * num_splits * (v_dim + 2);

    float m = -INFINITY;
    for (unsigned int s = 0; s < num_splits; s++) {
        m = fmaxf(m, parts[s * (v_dim + 2) + v_dim]);
    }
    float l = 0.0f;
    for (unsigned int s = 0; s < num_splits; s++) {
        const float* part = parts + s * (v_dim + 2);
        l += part[v_dim + 1] * expf(part[v_dim] - m);
    }
    for (unsigned int d = threadIdx.x; d < v_dim; d += blockDim.x) {
        float num = 0.0f;
        for (unsigned int s = 0; s < num_splits; s++) {
            const float* part = parts + s * (v_dim + 2);
            num += part[d] * expf(part[v_dim] - m);
        }
        out[qrow * v_dim + d] = num / l;
    }
}
