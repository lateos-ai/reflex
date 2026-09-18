// Causal single-new-query attention against a full K/V cache, with GQA head
// grouping. One block per query head; blockDim.x == head_dim (must be a
// power of two, true for every real model's head_dim). Not paged, not
// batched, not fused across layers -- this MVP forwards one prompt token at
// a time (see model.rs), so `seq_len` is just "however many positions have
// been cached so far," always small enough for a serial per-block softmax.
//
// q:        [num_q_heads, head_dim]
// k_cache:  [seq_len, num_kv_heads, head_dim]
// v_cache:  [seq_len, num_kv_heads, head_dim]
// out:      [num_q_heads, head_dim]
//
// Dynamic shared memory must be >= seq_len * sizeof(float) (the softmax
// scores buffer).
extern "C" __global__ void attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int seq_len,
    float scale
) {
    extern __shared__ float scores[];
    __shared__ float reduce_buf[1024];

    unsigned int qh = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int group_size = num_q_heads / num_kv_heads;
    unsigned int kvh = qh / group_size;

    float q_val = q[(unsigned long long)qh * head_dim + tid];

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
    out[(unsigned long long)qh * head_dim + tid] = acc;
}
