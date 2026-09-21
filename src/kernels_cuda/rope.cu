// GPT-NeoX-style (half-rotation) rotary position embedding, applied in place
// over row-major (num_heads, head_dim) `t` for a single token's position
// (this MVP forwards exactly one token per call -- batch_size is a permanent
// project constraint, see CLAUDE.md's Non-goals -- so `position` is a plain
// scalar rather than a per-token device array). Only the first rotary_dim
// elements of each head are rotated; any remaining head_dim - rotary_dim
// elements are left untouched. For i in [0, rotary_dim/2), pairs (x1, x2) =
// (t[i], t[i + rotary_dim/2]) rotate by theta_i = position / base^(2i/rotary_dim).
extern "C" __global__ void rope_kernel(
    float* __restrict__ t,
    unsigned int position,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int rotary_dim,
    float base
) {
    unsigned int half_rotary = rotary_dim / 2;
    unsigned long long total_pairs = (unsigned long long)num_heads * half_rotary;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total_pairs; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int i = idx % half_rotary;
        unsigned int head = idx / half_rotary;

        float exponent = (2.0f * (float)i) / (float)rotary_dim;
        float theta = (float)position / powf(base, exponent);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        unsigned long long base_idx = (unsigned long long)head * head_dim;
        float x1 = t[base_idx + i];
        float x2 = t[base_idx + i + half_rotary];

        t[base_idx + i] = x1 * cos_t - x2 * sin_t;
        t[base_idx + i + half_rotary] = x1 * sin_t + x2 * cos_t;
    }
}
