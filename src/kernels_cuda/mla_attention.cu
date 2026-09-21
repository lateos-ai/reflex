// DeepSeek-V2/V3 Multi-head Latent Attention (MLA), single new-query, MQA-style:
// `num_q_heads` query heads (each `qk_dim` wide) score against a *single* shared
// compressed KV "head" cached in `kv_cache` (`[seq_len, qk_dim]` row-major, one row
// per position). The same row also serves as the value vector, using only its first
// `v_dim` elements (`v_dim <= qk_dim` always -- MLA's whole point is that K and V
// share one compressed representation, unlike GQA's separate K/V caches). Output is
// `[num_q_heads, v_dim]`, still in compressed latent space -- the caller
// (`Model::gemv_per_head` with `wv_b`, see model.rs) decompresses it afterward.
//
// blockDim.x must be `qk_dim` rounded up to the next power of two (host-side
// launch config) -- unlike GQA's `head_dim` (conventionally already a power of two),
// MLA's compressed dim (`kv_lora_rank + qk_rope_head_dim`) generally isn't, so the
// tree reduction needs padding: threads with `tid >= qk_dim` contribute 0 to the
// score dot product and are simply idle for the (`tid < v_dim`) value-accumulation
// phase. Dynamic shared memory must be >= seq_len * sizeof(float) (the softmax
// scores buffer), same convention as `attention_kernel`.
extern "C" __global__ void mla_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv_cache,
    float* __restrict__ out,
    unsigned int num_q_heads,
    unsigned int qk_dim,
    unsigned int v_dim,
    unsigned int seq_len,
    float scale
) {
    extern __shared__ float scores[];
    __shared__ float reduce_buf[1024];

    unsigned int qh = blockIdx.x;
    unsigned int tid = threadIdx.x;

    float q_val = (tid < qk_dim) ? q[(unsigned long long)qh * qk_dim + tid] : 0.0f;

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
        out[(unsigned long long)qh * v_dim + tid] = acc;
    }
}
