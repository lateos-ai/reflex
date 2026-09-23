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

// Batched-prefill helper for DeepSeek-V2/V3 MLA: extracts a fixed-width, fixed-offset
// sub-slice of every (row, head) entry of a batched buffer into its own contiguous
// output, in one launch -- replaces what the single-token MLA attention block does
// with a per-head device-to-device copy loop (see model.rs's
// Model::forward_mla_attn_block). Used at three distinct call sites in the batched
// path (see Model::forward_mla_attn_block_batched): extracting q_pe out of the `wq`
// projection (num_heads = n_head), and k_pe/kv_cmpr out of the `wkv_a_mqa` projection
// (num_heads = 1, treating the whole fused buffer as a single "head").
//
// src: [rows, num_heads, src_head_width] contiguous
// dst: [rows, num_heads, dst_width] contiguous, dst[r,h,:] = src[r,h,src_head_offset..src_head_offset+dst_width]
extern "C" __global__ void mla_extract_batch_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int src_head_width,
    unsigned int dst_width,
    unsigned int src_head_offset
) {
    unsigned long long total = (unsigned long long)rows * num_heads * dst_width;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % dst_width;
        unsigned int head = (idx / dst_width) % num_heads;
        unsigned long long row = idx / (dst_width * (unsigned long long)num_heads);
        unsigned long long src_idx = (row * num_heads + head) * src_head_width + src_head_offset + col;
        dst[idx] = src[src_idx];
    }
}

// Batched-prefill helper for DeepSeek-V2/V3 MLA: merges per-head absorbed q_nope
// (kv_lora_rank-wide) and already-RoPE'd q_pe (qk_rope_head_dim-wide) into Qcur's
// per-head [kv_lora_rank+qk_rope_head_dim]-wide row, in one launch -- replaces the
// single-token attention block's two-dtod-copy-per-head loop (see
// Model::forward_mla_attn_block).
//
// absorbed: [rows, n_head, kv_lora] contiguous
// q_pe:     [rows, n_head, qk_rope] contiguous
// out:      [rows, n_head, kv_lora+qk_rope] contiguous, out[r,h,:] = absorbed[r,h,:] ++ q_pe[r,h,:]
extern "C" __global__ void mla_concat_qcur_batch_kernel(
    const float* __restrict__ absorbed,
    const float* __restrict__ q_pe,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int n_head,
    unsigned int kv_lora,
    unsigned int qk_rope
) {
    unsigned int qk_dim = kv_lora + qk_rope;
    unsigned long long total = (unsigned long long)rows * n_head * qk_dim;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % qk_dim;
        unsigned int head = (idx / qk_dim) % n_head;
        unsigned long long row = idx / (qk_dim * (unsigned long long)n_head);
        if (col < kv_lora) {
            out[idx] = absorbed[(row * n_head + head) * kv_lora + col];
        } else {
            out[idx] = q_pe[(row * n_head + head) * qk_rope + (col - kv_lora)];
        }
    }
}

// Batched-prefill helper for DeepSeek-V2/V3 MLA: writes this batch's compressed Kcur
// (kv_cmpr_normed concat k_pe, a single shared MQA "row" per position) into the
// preallocated per-layer kv_cache at rows start_pos..start_pos+rows, in one launch --
// replaces the single-token attention block's two-dtod-copy-per-row loop (see
// Model::forward_mla_attn_block).
//
// kv_cmpr:  [rows, kv_lora] contiguous
// k_pe:     [rows, qk_rope] contiguous
// kv_cache: [total_len, kv_lora+qk_rope] contiguous, preallocated; this call writes
//           only rows start_pos..start_pos+rows.
extern "C" __global__ void mla_write_kv_cache_batch_kernel(
    float* __restrict__ kv_cache,
    const float* __restrict__ kv_cmpr,
    const float* __restrict__ k_pe,
    unsigned int start_pos,
    unsigned int rows,
    unsigned int kv_lora,
    unsigned int qk_rope
) {
    unsigned int qk_dim = kv_lora + qk_rope;
    unsigned long long total = (unsigned long long)rows * qk_dim;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % qk_dim;
        unsigned long long row = idx / qk_dim;
        unsigned long long dst_idx = (start_pos + row) * (unsigned long long)qk_dim + col;
        kv_cache[dst_idx] = (col < kv_lora) ? kv_cmpr[row * kv_lora + col] : k_pe[row * qk_rope + (col - kv_lora)];
    }
}

// Batched-prefill grouped-GEMM MoE helper: gathers the rows of `src` selected for one
// expert's group into a fresh contiguous buffer, so the group can be run through that
// expert's weights as one real GEMM (model.rs's Model::gemm_view) instead of
// `group_size` separate per-row GEMV launches (model.rs's Model::gemv_expert). See
// model.rs's Model::forward_layer_moe_batched/Model::forward_mla_moe_ffn_batched.
//
// src:       [rows, hidden_size] contiguous
// perm_row:  [num_assignments], each entry < rows -- this group's selected row indices
// dst:       [num_assignments, hidden_size] contiguous, dst[a,:] = src[perm_row[a],:]
extern "C" __global__ void moe_gather_kernel(
    const float* __restrict__ src,
    const unsigned int* __restrict__ perm_row,
    float* __restrict__ dst,
    unsigned int num_assignments,
    unsigned int hidden_size
) {
    unsigned long long total = (unsigned long long)num_assignments * hidden_size;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % hidden_size;
        unsigned long long a = idx / hidden_size;
        unsigned long long src_idx = (unsigned long long)perm_row[a] * hidden_size + col;
        dst[idx] = src[src_idx];
    }
}

// Inverse of `moe_gather_kernel`'s data movement: weighted scatter-add of one expert
// group's down-projected FFN output back into each selected row's slot of the
// layer-wide output accumulator. No atomics -- callers must guarantee `dest_row` has no
// duplicate entries within a single launch (true for one expert's group: top-k routing
// never selects the same expert twice for one row, so a row appears at most once per
// expert's group). `dst` must already be allocated/seeded (zeroed, or pre-seeded with
// e.g. MLA's always-on shared-expert output) before the first expert group's launch.
//
// src:       [num_assignments, hidden_size] contiguous (one expert group's down-proj output)
// dest_row:  [num_assignments], each entry < rows, no duplicates within this launch
// weight:    [num_assignments] -- per-assignment router combination weight
// dst:       [rows, hidden_size] contiguous, dst[dest_row[a],:] += weight[a] * src[a,:]
extern "C" __global__ void moe_scatter_add_kernel(
    const float* __restrict__ src,
    const unsigned int* __restrict__ dest_row,
    const float* __restrict__ weight,
    float* __restrict__ dst,
    unsigned int num_assignments,
    unsigned int hidden_size
) {
    unsigned long long total = (unsigned long long)num_assignments * hidden_size;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % hidden_size;
        unsigned long long a = idx / hidden_size;
        unsigned long long dst_idx = (unsigned long long)dest_row[a] * hidden_size + col;
        dst[dst_idx] += weight[a] * src[idx];
    }
}
