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

// "Normal" RoPE (llama.cpp's LLAMA_ROPE_TYPE_NORM -- rotates pairs of *consecutive*
// elements (2i, 2i+1), unlike rope_kernel's GPT-NeoX-style half-split pairs (i, i +
// rotary_dim/2)). Used by DeepSeek-V2/V3 MLA's q_pe/k_pe (confirmed against
// llama.cpp's `llama_model_rope_type`, which maps `LLM_ARCH_DEEPSEEK2` here, not to
// the NEOX case Qwen3/Qwen3.5 use) -- a real, easy-to-miss per-architecture
// difference, not a stylistic variant. Same single-token, scalar-position,
// full-rotation convention as rope_kernel otherwise (see model.rs's MLA doc
// comments): num_heads is small (or 1, for the shared MQA k_pe), rotary_dim ==
// head_dim always for MLA's q_pe/k_pe (they're already separately-extracted
// buffers, not a slice of a wider head).
extern "C" __global__ void rope_norm_kernel(
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

        unsigned long long base_idx = (unsigned long long)head * head_dim + 2 * i;
        float x1 = t[base_idx];
        float x2 = t[base_idx + 1];

        t[base_idx] = x1 * cos_t - x2 * sin_t;
        t[base_idx + 1] = x1 * sin_t + x2 * cos_t;
    }
}

// YaRN-scaled variant of rope_norm_kernel, for architectures whose GGUF declares
// rope.scaling.type == "yarn" (real DeepSeek-V2/V3 checkpoints -- the synthetic
// MVP-step-4 MLA test fixture has no YaRN scaling and uses rope_norm_kernel
// unchanged). Ported from ggml's own CUDA rope_yarn() + rope_norm()
// (ggml/src/ggml-cuda/rope.cu, read in full while implementing this), simplified
// for this project's single-token, full-rotation, no-freq-factors,
// forward-only case. See model.rs's MlaYarnConfig for where freq_scale/
// ext_factor/attn_factor/corr_dim_start/corr_dim_end come from (all constant
// across a whole forward pass, precomputed once at load time).
extern "C" __global__ void rope_norm_yarn_kernel(
    float* __restrict__ t,
    unsigned int position,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int rotary_dim,
    float base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float corr_dim_start,
    float corr_dim_end
) {
    unsigned int half_rotary = rotary_dim / 2;
    unsigned long long total_pairs = (unsigned long long)num_heads * half_rotary;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total_pairs; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int i = idx % half_rotary;
        unsigned int head = idx / half_rotary;

        float exponent = (2.0f * (float)i) / (float)rotary_dim;
        float theta_extrap = (float)position / powf(base, exponent);
        float theta_interp = freq_scale * theta_extrap;

        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float denom = fmaxf(0.001f, corr_dim_end - corr_dim_start);
            float y = ((float)i - corr_dim_start) / denom;
            float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            float ramp_mix = ramp * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        float cos_theta = cosf(theta) * mscale;
        float sin_theta = sinf(theta) * mscale;

        unsigned long long base_idx = (unsigned long long)head * head_dim + 2 * i;
        float x1 = t[base_idx];
        float x2 = t[base_idx + 1];

        t[base_idx] = x1 * cos_theta - x2 * sin_theta;
        t[base_idx + 1] = x1 * sin_theta + x2 * cos_theta;
    }
}
