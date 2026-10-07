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

#define ATTN_WARPS 8
#define ATTN_MAX_V_PER_LANE 32

// Must match `AttnParams` in src/model/kernels.rs field for field (repr(C)).
struct AttnParams {
    unsigned long long k_pos_stride;
    unsigned long long k_head_stride;
    unsigned long long v_pos_stride;
    unsigned long long v_head_stride;
    unsigned int num_q_heads;
    unsigned int group_size;
    unsigned int qk_dim;
    unsigned int v_dim;
    unsigned int start_pos;
    unsigned int num_splits;
    unsigned int split_len;
    unsigned int window;  // 0 = no sliding window
    float scale;
};

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    return x;
}

extern "C" __global__ void attention_online_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p
) {
    extern __shared__ float smem[];
    float* q_s = smem;                         // [qk_dim]
    float* warp_m = q_s + p.qk_dim;            // [ATTN_WARPS]
    float* warp_l = warp_m + ATTN_WARPS;       // [ATTN_WARPS]
    float* warp_acc = warp_l + ATTN_WARPS;     // [ATTN_WARPS][v_dim]

    const unsigned int row = blockIdx.x;
    const unsigned int qh = blockIdx.y;
    const unsigned int split = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int kvh = qh / p.group_size;

    const unsigned long long qrow = (unsigned long long)row * p.num_q_heads + qh;
    for (unsigned int d = tid; d < p.qk_dim; d += blockDim.x) {
        q_s[d] = q[qrow * p.qk_dim + d];
    }
    __syncthreads();

    // This block's slice of the causal window [lo, seq_len).
    const unsigned int seq_len = p.start_pos + row + 1;
    const unsigned int lo = (p.window != 0 && seq_len > p.window) ? seq_len - p.window : 0;
    const unsigned int begin = max(split * p.split_len, lo);
    const unsigned int end = min(seq_len, begin + p.split_len);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[ATTN_MAX_V_PER_LANE];
#pragma unroll
    for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
        acc[i] = 0.0f;
    }

    const float* k_head = k + (unsigned long long)kvh * p.k_head_stride;
    const float* v_head = v + (unsigned long long)kvh * p.v_head_stride;
    for (unsigned int pos = begin + warp; pos < end; pos += ATTN_WARPS) {
        const float* kp = k_head + (unsigned long long)pos * p.k_pos_stride;
        float dot = 0.0f;
        for (unsigned int d = lane; d < p.qk_dim; d += 32) {
            dot += q_s[d] * kp[d];
        }
        const float s = warp_sum(dot) * p.scale;

        const float m_new = fmaxf(m, s);
        const float correction = expf(m - m_new);  // 0 on the first position (m = -inf)
        const float weight = expf(s - m_new);
        l = l * correction + weight;
        const float* vp = v_head + (unsigned long long)pos * p.v_pos_stride;
#pragma unroll
        for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
            const unsigned int d = i * 32 + lane;
            if (d >= p.v_dim) {
                break;
            }
            acc[i] = acc[i] * correction + weight * vp[d];
        }
        m = m_new;
    }

    // Merge this block's warps. A warp that saw no positions has m = -inf, l = 0.
    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
        const unsigned int d = i * 32 + lane;
        if (d >= p.v_dim) {
            break;
        }
        warp_acc[warp * p.v_dim + d] = acc[i];
    }
    __syncthreads();

    float block_m = -INFINITY;
    for (int w = 0; w < ATTN_WARPS; w++) {
        block_m = fmaxf(block_m, warp_m[w]);
    }
    float block_l = 0.0f;
    if (block_m != -INFINITY) {
        for (int w = 0; w < ATTN_WARPS; w++) {
            block_l += warp_l[w] * expf(warp_m[w] - block_m);
        }
    }

    if (p.num_splits == 1) {
        float* out = dst + qrow * p.v_dim;
        for (unsigned int d = tid; d < p.v_dim; d += blockDim.x) {
            float num = 0.0f;
            for (int w = 0; w < ATTN_WARPS; w++) {
                num += warp_acc[w * p.v_dim + d] * expf(warp_m[w] - block_m);
            }
            out[d] = num / block_l;
        }
    } else {
        float* part = dst + (qrow * p.num_splits + split) * (p.v_dim + 2);
        for (unsigned int d = tid; d < p.v_dim; d += blockDim.x) {
            float num = 0.0f;
            if (block_m != -INFINITY) {
                for (int w = 0; w < ATTN_WARPS; w++) {
                    num += warp_acc[w * p.v_dim + d] * expf(warp_m[w] - block_m);
                }
            }
            part[d] = num;
        }
        if (tid == 0) {
            part[p.v_dim] = block_m;
            part[p.v_dim + 1] = block_l;
        }
    }
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
