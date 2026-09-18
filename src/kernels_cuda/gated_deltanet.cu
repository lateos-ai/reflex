// Gated DeltaNet (Qwen3.5 hybrid linear-attention) kernels, single-token
// (decode/naive-prefill) path only -- no chunked/parallel-prefill kernels,
// matching this project's naive-first MVP scope (see MoE's naive per-expert
// dispatch for the same precedent). Ported from RustFeference's
// `reference/gated_deltanet_rustfeference.rs` `GDN_KERNEL_SOURCE` (itself a
// port of real llama.cpp `qwen35.cpp`/`delta-net-base.cpp` math), dropping
// the "gdn_chunk_*" kernels that implement the chunked closed form -- a
// future Phase-2-style optimization, not this milestone's scope.
//
// float math throughout, matching the host reference. The input projections
// (`attn_qkv`/`attn_gate`/`ssm_beta`/`ssm_alpha`) are plain 2-D `[in, out]`
// GEMVs in the same layout `gemv.cu`'s kernel already handles, so they reuse
// `Model::gemv` directly (see `model.rs`'s GDN mixer) instead of a
// duplicate matvec kernel here.

// Causal depthwise conv1d over the fused qkv, SiLU, and window advance.
// One thread per channel: each channel reads and rewrites only its own
// column of `conv_state`, so no cross-thread synchronization is needed.
extern "C" __global__ void gdn_conv_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    float* __restrict__ conv_out,
    unsigned int conv_dim,
    unsigned int k
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float acc = 0.0f;
    for (unsigned int kk = 0; kk < k; kk++) {
        float xv = (kk < k - 1) ? conv_state[kk * conv_dim + c] : qkv[c];
        acc += xv * conv_w[kk + c * k];
    }
    conv_out[c] = acc / (1.0f + expf(-acc));
    if (k > 1) {
        for (unsigned int kk = 0; kk + 1 < k - 1; kk++) {
            conv_state[kk * conv_dim + c] = conv_state[(kk + 1) * conv_dim + c];
        }
        conv_state[(k - 2) * conv_dim + c] = qkv[c];
    }
}

// In-place per-head x = x * (1/sqrt(sum(x^2)+eps)) * scale. One block per
// head (grid.x = number of heads to normalize); `blockDim.x` must be a
// power of two >= 1 (see `GatedDeltaNetConfig::norm_block_dim`).
extern "C" __global__ void gdn_l2_norm_kernel(
    float* __restrict__ x,
    unsigned int offset,
    unsigned int head_dim,
    float eps,
    float scale
) {
    unsigned int h = blockIdx.x;
    unsigned int t = threadIdx.x;
    float* xh = x + offset + (unsigned long long)h * head_dim;
    float ss = 0.0f;
    for (unsigned int d = t; d < head_dim; d += blockDim.x) {
        float v = xh[d];
        ss += v * v;
    }
    __shared__ float red[1024];
    red[t] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) red[t] += red[t + stride];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] + eps);
    for (unsigned int d = t; d < head_dim; d += blockDim.x) {
        xh[d] = xh[d] * inv * scale;
    }
}

// beta = sigmoid(beta_raw); decay = exp(softplus(alpha_raw + dt) * a).
extern "C" __global__ void gdn_gates_kernel(
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt,
    const float* __restrict__ a,
    float* __restrict__ decay,
    float* __restrict__ beta,
    unsigned int H
) {
    unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float alpha = alpha_raw[h] + dt[h];
    float sp = (alpha > 20.0f) ? alpha : logf(1.0f + expf(alpha));
    decay[h] = expf(sp * a[h]);
    beta[h] = 1.0f / (1.0f + expf(-beta_raw[h]));
}

// The delta rule, one block per value head and one thread per value column.
// Each thread owns column `vi` of `S` (all `head_dim` keys), so the whole
// update is race-free without block-wide reductions. `qkv` holds the
// conv/silu/normalized fused buffer; q/k/v are read through offsets.
extern "C" __global__ void gdn_delta_kernel(
    float* __restrict__ S,
    const float* __restrict__ qkv,
    unsigned int q_offset,
    unsigned int k_offset,
    unsigned int v_offset,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ o,
    unsigned int hd,
    unsigned int H_k,
    unsigned int H_v
) {
    unsigned int h = blockIdx.x;
    unsigned int vi = threadIdx.x;
    if (h >= H_v || vi >= hd) return;
    unsigned int kh = h % H_k;
    const float* qh = qkv + q_offset + (unsigned long long)kh * hd;
    const float* khp = qkv + k_offset + (unsigned long long)kh * hd;
    const float* vh = qkv + v_offset + (unsigned long long)h * hd;
    float* Sh = S + (unsigned long long)h * hd * hd;
    float b = beta[h];
    float dc = decay[h];

    for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] *= dc;

    float kv = 0.0f;
    for (unsigned int ki = 0; ki < hd; ki++) kv += Sh[ki * hd + vi] * khp[ki];
    float delta = (vh[vi] - kv) * b;

    for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] += khp[ki] * delta;

    float acc = 0.0f;
    for (unsigned int ki = 0; ki < hd; ki++) acc += Sh[ki * hd + vi] * qh[ki];
    o[(unsigned long long)h * hd + vi] = acc;
}

// y = RMSNorm(o, norm_w) * silu(z). One block per value head.
extern "C" __global__ void gdn_gated_norm_kernel(
    const float* __restrict__ o,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    float* __restrict__ y,
    unsigned int hd,
    float eps
) {
    unsigned int h = blockIdx.x;
    unsigned int t = threadIdx.x;
    const float* oh = o + (unsigned long long)h * hd;
    const float* zh = z + (unsigned long long)h * hd;
    float* yh = y + (unsigned long long)h * hd;
    float ss = 0.0f;
    for (unsigned int d = t; d < hd; d += blockDim.x) {
        float val = oh[d];
        ss += val * val;
    }
    __shared__ float red[1024];
    red[t] = ss;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) red[t] += red[t + stride];
        __syncthreads();
    }
    float mean_sq = red[0] / (float)hd;
    float rms_inv = 1.0f / sqrtf(mean_sq + eps);
    for (unsigned int d = t; d < hd; d += blockDim.x) {
        float g = zh[d] / (1.0f + expf(-zh[d]));
        yh[d] = oh[d] * rms_inv * norm_w[d] * g;
    }
}
