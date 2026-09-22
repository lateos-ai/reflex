// Batched-prefill variant of mla_attention_kernel (see mla_attention.cu): scores
// `rows` new query rows against the shared compressed MQA KV cache in one launch
// (grid gains a query-row dimension, blockIdx.y), each row causally masked to its
// own `start_pos + qrow + 1` positions instead of one shared `seq_len` -- same
// relationship attention_prefill_kernel has to attention_kernel (see
// attention_prefill.cu), applied to MLA's MQA/compressed-KV shape instead of GQA's
// separate K/V caches.
//
// q:        [rows, num_q_heads, qk_dim]
// kv_cache: [start_pos+rows, qk_dim] -- single shared MQA "row" per position; the
//           same row serves as K (all qk_dim elements) and V (its first v_dim
//           elements), same convention as mla_attention_kernel.
// out:      [rows, num_q_heads, v_dim]
//
// blockDim.x must be qk_dim rounded up to the next power of two (host-side launch
// config), same padding rationale as mla_attention_kernel: threads with tid >=
// qk_dim contribute 0 to the score dot product and are idle during the (tid <
// v_dim) value-accumulation phase. Dynamic shared memory must be >= (start_pos+rows)
// * sizeof(float) (the worst-case softmax scores buffer, sized for the last query
// row), same convention as attention_prefill_kernel.
extern "C" __global__ void mla_attention_prefill_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv_cache,
    float* __restrict__ out,
    unsigned int num_q_heads,
    unsigned int qk_dim,
    unsigned int v_dim,
    unsigned int start_pos,
    unsigned int rows,
    float scale
) {
    extern __shared__ float scores[];
    __shared__ float reduce_buf[1024];

    unsigned int qh = blockIdx.x;
    unsigned int qrow = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int seq_len = start_pos + qrow + 1;

    float q_val = (tid < qk_dim) ? q[((unsigned long long)qrow * num_q_heads + qh) * qk_dim + tid] : 0.0f;

    for (unsigned int pos = 0; pos < seq_len; pos++) {
        const float* k = kv_cache + (unsigned long long)pos * qk_dim;
        reduce_buf[tid] = (tid < qk_dim) ? q_val * k[tid] : 0.0f;
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

    if (tid < v_dim) {
        float acc = 0.0f;
        for (unsigned int pos = 0; pos < seq_len; pos++) {
            const float* v = kv_cache + (unsigned long long)pos * qk_dim;
            acc += scores[pos] * v[tid];
        }
        out[((unsigned long long)qrow * num_q_heads + qh) * v_dim + tid] = acc;
    }
}
