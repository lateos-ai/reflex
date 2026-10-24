// Gated DeltaNet (Qwen3.5 hybrid linear-attention) kernels. Ported from
// RustFeference's `reference/gated_deltanet_rustfeference.rs`
// `GDN_KERNEL_SOURCE` (itself a port of real llama.cpp
// `qwen35.cpp`/`delta-net-base.cpp` math), dropping the "gdn_chunk_*"
// kernels that implement the chunked closed form.
//
// Two forms: the single-token kernels (decode, and the token-major
// reference prefill), and `_rows` kernels for batched prefill that take all
// `rows` prompt positions in one launch. The recurrent ones (conv, delta)
// walk the rows in order inside the kernel, carrying the state, as
// `mamba2.cu` does; the projections around them are batched GEMMs. Each
// `_rows` kernel does exactly the single-token kernel's per-row arithmetic,
// in the same order, so its output is bit-identical to looping that kernel
// (src/model/gdn_rows_tests.rs). The gated norm needs no `_rows` variant:
// `gdn_gated_norm_kernel` launched over rows * H_v blocks already is one.
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

// ---- Batched prefill (`rows` positions per launch) ----

// gdn_gates_kernel over `rows` rows: inputs and outputs are [rows, H].
extern "C" __global__ void gdn_gates_rows_kernel(
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt,
    const float* __restrict__ a,
    float* __restrict__ decay,
    float* __restrict__ beta,
    unsigned int H,
    unsigned int rows
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    unsigned int h = i % H;
    float alpha = alpha_raw[i] + dt[h];
    float sp = (alpha > 20.0f) ? alpha : logf(1.0f + expf(alpha));
    decay[i] = expf(sp * a[h]);
    beta[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
}

// gdn_conv_kernel for `rows` positions in order: `qkv` and `conv_out` are
// [rows, conv_dim]. One thread per channel walks the rows, doing the
// single-token kernel's taps and window advance for each.
extern "C" __global__ void gdn_conv_rows_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    float* __restrict__ conv_out,
    unsigned int conv_dim,
    unsigned int k,
    unsigned int rows
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (unsigned int t = 0; t < rows; t++) {
        const float cur = qkv[(unsigned long long)t * conv_dim + c];
        float acc = 0.0f;
        for (unsigned int kk = 0; kk < k; kk++) {
            float xv = (kk < k - 1) ? conv_state[kk * conv_dim + c] : cur;
            acc += xv * conv_w[kk + c * k];
        }
        conv_out[(unsigned long long)t * conv_dim + c] = acc / (1.0f + expf(-acc));
        if (k > 1) {
            for (unsigned int kk = 0; kk + 1 < k - 1; kk++) {
                conv_state[kk * conv_dim + c] = conv_state[(kk + 1) * conv_dim + c];
            }
            conv_state[(k - 2) * conv_dim + c] = cur;
        }
    }
}

// gdn_l2_norm_kernel for the q and k heads of `rows` rows in one launch.
// grid = (2 * k_heads, rows): head h < k_heads is q head h (scaled by
// `q_scale`), the rest are k heads (scaled by `k_scale`); q and k sit back to
// back at the start of each [row_stride] row.
extern "C" __global__ void gdn_l2_norm_rows_kernel(
    float* __restrict__ x,
    unsigned int row_stride,
    unsigned int k_heads,
    unsigned int head_dim,
    float eps,
    float q_scale,
    float k_scale
) {
    unsigned int h = blockIdx.x;
    unsigned int t = threadIdx.x;
    float scale = h < k_heads ? q_scale : k_scale;
    float* xh = x + (unsigned long long)blockIdx.y * row_stride + (unsigned long long)h * head_dim;
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

#define GDN_ROWS_MAX_HD 128

// gdn_delta_kernel for `rows` positions in order. One block per value head,
// one thread per value column, as in the single-token kernel. For
// hd <= GDN_ROWS_MAX_HD (every real Qwen3.5 release: 128) the thread keeps
// its S column in registers across all rows (fully unrolled with a guard, so
// it stays out of local memory) and the row's q/k for the head are staged in
// shared memory; S is read and written once per launch instead of four
// passes per row. Larger hd falls back to the single-token kernel's
// global-memory loop per row. Either way each row does the same operations
// in the same order as gdn_delta_kernel. `qkv` is [rows, conv_dim] with q,
// k and v at columns 0, key_dim and 2 * key_dim (the single-token kernel's
// offsets); `beta` and `decay` are [rows, H_v]; `o` is [rows, H_v * hd].
extern "C" __global__ void gdn_delta_rows_kernel(
    float* __restrict__ S,
    const float* __restrict__ qkv,
    unsigned int conv_dim,
    unsigned int key_dim,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ o,
    unsigned int hd,
    unsigned int H_k,
    unsigned int H_v,
    unsigned int rows
) {
    unsigned int h = blockIdx.x;
    unsigned int vi = threadIdx.x;
    if (h >= H_v) return;  // whole blocks exit together
    unsigned int kh = h % H_k;
    float* Sh = S + (unsigned long long)h * hd * hd;
    const unsigned long long o_stride = (unsigned long long)H_v * hd;
    const unsigned int q_offset = 0, k_offset = key_dim, v_offset = 2 * key_dim;

    if (hd > GDN_ROWS_MAX_HD) {
        if (vi >= hd) return;
        for (unsigned int t = 0; t < rows; t++) {
            const float* row = qkv + (unsigned long long)t * conv_dim;
            const float* qh = row + q_offset + (unsigned long long)kh * hd;
            const float* khp = row + k_offset + (unsigned long long)kh * hd;
            const float* vh = row + v_offset + (unsigned long long)h * hd;
            float b = beta[(unsigned long long)t * H_v + h];
            float dc = decay[(unsigned long long)t * H_v + h];
            for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] *= dc;
            float kv = 0.0f;
            for (unsigned int ki = 0; ki < hd; ki++) kv += Sh[ki * hd + vi] * khp[ki];
            float delta = (vh[vi] - kv) * b;
            for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] += khp[ki] * delta;
            float acc = 0.0f;
            for (unsigned int ki = 0; ki < hd; ki++) acc += Sh[ki * hd + vi] * qh[ki];
            o[t * o_stride + (unsigned long long)h * hd + vi] = acc;
        }
        return;
    }

    __shared__ float qs[GDN_ROWS_MAX_HD];
    __shared__ float ks[GDN_ROWS_MAX_HD];
    const bool active = vi < hd;
    float s[GDN_ROWS_MAX_HD];
#pragma unroll
    for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
        s[ki] = (active && ki < hd) ? Sh[ki * hd + vi] : 0.0f;
    }
    for (unsigned int t = 0; t < rows; t++) {
        const float* row = qkv + (unsigned long long)t * conv_dim;
        __syncthreads();  // the previous row's reads of qs/ks are done
        if (active) {
            qs[vi] = row[q_offset + (unsigned long long)kh * hd + vi];
            ks[vi] = row[k_offset + (unsigned long long)kh * hd + vi];
        }
        __syncthreads();
        if (!active) continue;
        float b = beta[(unsigned long long)t * H_v + h];
        float dc = decay[(unsigned long long)t * H_v + h];

#pragma unroll
        for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
            if (ki >= hd) break;
            s[ki] *= dc;
        }
        float kv = 0.0f;
#pragma unroll
        for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
            if (ki >= hd) break;
            kv += s[ki] * ks[ki];
        }
        float delta = (row[v_offset + (unsigned long long)h * hd + vi] - kv) * b;
#pragma unroll
        for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
            if (ki >= hd) break;
            s[ki] += ks[ki] * delta;
        }
        float acc = 0.0f;
#pragma unroll
        for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
            if (ki >= hd) break;
            acc += s[ki] * qs[ki];
        }
        o[t * o_stride + (unsigned long long)h * hd + vi] = acc;
    }
#pragma unroll
    for (unsigned int ki = 0; ki < GDN_ROWS_MAX_HD; ki++) {
        if (active && ki < hd) Sh[ki * hd + vi] = s[ki];
    }
}
