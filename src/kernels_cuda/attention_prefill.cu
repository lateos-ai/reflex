// Batched-prefill causal self-attention: like attention_kernel (see
// attention.cu), but one launch scores `m` new query rows at once instead of
// one launch per row. Grid adds a second dimension over query rows
// (blockIdx.y); each row's causal mask width is computed independently
// (`start_pos + qrow + 1`), so row 0 only sees the KV positions already in
// the cache before this batch, and row `m-1` sees the whole batch. This is
// still an O(seq_len) serial softmax per (head, row) block, same as
// attention_kernel -- not a fused/tiled FlashAttention rewrite -- which is
// fine at this project's prompt-length scale (cold-start prefill, not a
// long-context serving workload).
//
// q:        [m, num_q_heads, head_dim]
// k_cache:  [start_pos+m, num_kv_heads, head_dim]
// v_cache:  [start_pos+m, num_kv_heads, head_dim]
// out:      [m, num_q_heads, head_dim]
//
// Dynamic shared memory must be >= (start_pos+m) * sizeof(float) (the
// worst-case softmax scores buffer, sized for the last query row).
extern "C" __global__ void attention_prefill_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int start_pos,
    unsigned int m,
    float scale
) {
    extern __shared__ float scores[];
    __shared__ float reduce_buf[1024];

    unsigned int qh = blockIdx.x;
    unsigned int qrow = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int group_size = num_q_heads / num_kv_heads;
    unsigned int kvh = qh / group_size;
    unsigned int seq_len = start_pos + qrow + 1;

    float q_val = q[((unsigned long long)qrow * num_q_heads + qh) * head_dim + tid];

    for (unsigned int pos = 0; pos < seq_len; pos++) {
        const float* k = k_cache + ((unsigned long long)pos * num_kv_heads + kvh) * head_dim;
        reduce_buf[tid] = q_val * k[tid];
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduce_buf[tid] += reduce_buf[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            scores[pos] = reduce_buf[0] * scale;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_v = scores[0];
        for (unsigned int p = 1; p < seq_len; p++) {
            max_v = fmaxf(max_v, scores[p]);
        }
        float sum = 0.0f;
        for (unsigned int p = 0; p < seq_len; p++) {
            float e = expf(scores[p] - max_v);
            scores[p] = e;
            sum += e;
        }
        for (unsigned int p = 0; p < seq_len; p++) {
            scores[p] /= sum;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int pos = 0; pos < seq_len; pos++) {
        const float* v = v_cache + ((unsigned long long)pos * num_kv_heads + kvh) * head_dim;
        acc += scores[pos] * v[tid];
    }
    out[((unsigned long long)qrow * num_q_heads + qh) * head_dim + tid] = acc;
}
