// GQA-aware decode attention: one new query row against the K/V cache, for head dims
// 64, 128 and 256 (qk_dim == v_dim). Same layout, masking and partial format as
// attention_online_kernel (see attention_online.cu), which still serves every other
// shape (MLA, Gemma 4's d512, prefill).
//
// Why: attention_online_kernel gives each query head its own blocks, so a KV head's
// K/V rows are read once per query head of its group, and each warp walks its
// positions one at a time with a full memory latency exposed per position. On a T4 at
// ~1,550 cached positions that ran at ~28% of memory bandwidth, 3.8 ms of an 11.3 ms
// Qwen3-0.6B decode step. Here:
//   - a block takes HB query heads that share one KV head and reads each K and V row
//     once for all of them (scores, online softmax and accumulators per head, in
//     registers);
//   - a warp handles two positions per iteration, issuing both K and V loads before
//     using them;
//   - positions are split across blocks (grid.y) by the launcher to fill the GPU, and
//     each block writes the (acc, m, l) partials attention_online_combine_kernel merges.
//
// Lane l holds dims l, l + 32, ... of a row (DL = D / 32 of them), so every load is
// one coalesced 128-byte segment per 32 dims.
//
// Launch: grid (num_kv_heads * group_size / HB, num_splits), block DEC_WARPS * 32
// threads, static shared memory. dst: num_splits == 1 the output [num_q_heads, D];
// otherwise partials [num_q_heads, num_splits, D + 2] (row index 0 only).

#include "attention_decode.cuh"

#define DEC_KERNEL(D, HB)                                                                   \
    extern "C" __global__ void __launch_bounds__(DEC_WARPS * 32)                            \
        attention_decode_d##D##_h##HB##_kernel(const float* __restrict__ q,                \
                                               const float* __restrict__ k,                \
                                               const float* __restrict__ v,                \
                                               float* __restrict__ dst, AttnParams p) {    \
        attention_decode<float, D / 32, HB>(q, k, v, dst, p);                               \
    }

DEC_KERNEL(64, 1)
DEC_KERNEL(64, 2)
DEC_KERNEL(64, 4)
DEC_KERNEL(64, 8)
DEC_KERNEL(128, 1)
DEC_KERNEL(128, 2)
DEC_KERNEL(128, 4)
DEC_KERNEL(128, 8)
DEC_KERNEL(256, 1)
DEC_KERNEL(256, 2)
DEC_KERNEL(256, 4)
