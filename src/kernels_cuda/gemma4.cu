// Kernels only the Gemma 4 (`gemma4`) path uses, in their own module so no other
// architecture loads them. Conventions: docs/design/gemma4.md.

// GPT-NeoX-style RoPE with per-pair frequency factors, in place over row-major
// (rows, num_heads, head_dim) `t`, row r at absolute position start_pos + r (one
// row for a decode step, m rows for a batched prefill). Matches ggml's
// `ggml_rope_ext` NEOX case with `freq_factors` (ggml-cuda's rope_neox): for
// i in [0, n_dims/2), the pair (t[i], t[i + n_dims/2]) rotates by
//   theta_i = position * powf(base^(-2/n_dims), i) / freq_factors[i]
// (`freq_factors` has n_dims/2 entries). Elements past n_dims are left
// untouched. Gemma 4's global layers pass rope_freqs.weight here (1.0 for the
// first 64 pairs, 1e30 for the rest, so those barely rotate); its sliding
// layers pass all ones, which is exactly plain RoPE (x / 1.0f == x).
extern "C" __global__ void rope_neox_ff_kernel(
    float* __restrict__ t,
    const float* __restrict__ freq_factors,
    unsigned int start_pos,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int n_dims,
    float base
) {
    unsigned int half = n_dims / 2;
    unsigned long long total_pairs = (unsigned long long)rows * num_heads * half;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float theta_scale = powf(base, -2.0f / (float)n_dims);

    for (unsigned long long idx = global_tid; idx < total_pairs; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int i = idx % half;
        unsigned long long head_row = idx / half;  // row * num_heads + head
        unsigned int row = head_row / num_heads;
        float position = (float)(start_pos + row);

        float theta = position * powf(theta_scale, (float)i) / freq_factors[i];
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        unsigned long long base_idx = head_row * head_dim;
        float x1 = t[base_idx + i];
        float x2 = t[base_idx + i + half];
        t[base_idx + i] = x1 * cos_t - x2 * sin_t;
        t[base_idx + i + half] = x1 * sin_t + x2 * cos_t;
    }
}

// GeGLU: output = gelu_tanh(gate) * up over `n` elements, `gate`/`up` the two
// separate FFN projections (same layout as silu_and_mul_kernel). gelu_tanh is
// ggml-cuda's gelu_f32 (llama.cpp's LLM_FFN_GELU):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
extern "C" __global__ void gelu_tanh_and_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    unsigned long long n
) {
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const float GELU_COEF_A = 0.044715f;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long idx = global_tid; idx < n; idx += (unsigned long long)gridDim.x * blockDim.x) {
        float x = gate[idx];
        float g = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
        output[idx] = g * up[idx];
    }
}

// Final logit soft-cap, in place: x = cap * tanh(x / cap). Same operation order
// as llama.cpp's graph (scale by 1/cap, tanh, scale by cap).
extern "C" __global__ void softcap_kernel(
    float* __restrict__ x,
    unsigned long long n,
    float cap
) {
    float inv_cap = 1.0f / cap;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long idx = global_tid; idx < n; idx += (unsigned long long)gridDim.x * blockDim.x) {
        x[idx] = tanhf(x[idx] * inv_cap) * cap;
    }
}

// In place: x *= s. Gemma 4's per-layer `layer_output_scale` (a single f32,
// read once at load).
extern "C" __global__ void scale_kernel(
    float* __restrict__ x,
    unsigned long long n,
    float s
) {
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long idx = global_tid; idx < n; idx += (unsigned long long)gridDim.x * blockDim.x) {
        x[idx] *= s;
    }
}
