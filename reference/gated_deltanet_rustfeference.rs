//! Gated DeltaNet (Qwen3.5/3.6 hybrid linear-attention) — host/CPU reference.
//!
//! Phase 21.15.2. This is the correctness-first host reference for one
//! Gated DeltaNet mixer layer, matching the real Qwen3.5 recurrence exactly.
//! It has no CUDA dependency (Phase 21.15.3 adds the GPU kernel), the same
//! "get the math right on host first" sequencing every foundational kernel in
//! this repo started with.
//!
//! Port provenance (fetched 2026-09-15, upstream `ggml-org/llama.cpp`):
//! - `src/models/qwen35.cpp` `build_layer_attn_linear` (input projections,
//!   conv, L2-norm, gated output) and `build_norm_gated`.
//! - `src/models/delta-net-base.cpp` `build_delta_net_autoregressive` (the
//!   per-token state update) and `build_conv_state`.
//! - `ggml/src/ggml-cpu/ops.cpp` `ggml_compute_forward_ssm_conv_f32` (exact
//!   causal depthwise-conv indexing) and `ggml_compute_forward_l2_norm_f32`.
//! - `src/models/models.h` `build_gdn_l2_norm` (`x / sqrt(sum(x^2) + eps)`).
//!
//! The first-draft plan used third-party equations that omitted the causal
//! conv1d, the q scaling, and the `exp` on the decay gate; `PHASE21_15_PLAN.md`
//! ("item 4") records the corrected step list this module implements.
//!
//! Layout convention (matching GGUF/ggml, where `ne[0]` is contiguous): a
//! weight with `[in, out]` ggml dims is stored `w[in_idx + out_idx * in_dim]`,
//! so `y[o] = sum_i w[i + o*in_dim] * x[i]`.
//!
//! Per-layer tensors (real Qwen3.5-0.8B shapes in parentheses):
//! - `attn_qkv.weight`   `[hidden, 2*key_dim + value_dim]` (1024, 6144), fused q/k/v
//! - `attn_gate.weight`  `[hidden, value_dim]` (1024, 2048), output gate `z`
//! - `ssm_beta.weight`   `[hidden, num_v_heads]` (1024, 16)
//! - `ssm_alpha.weight`  `[hidden, num_v_heads]` (1024, 16)
//! - `ssm_dt.bias`       `[num_v_heads]` (16)
//! - `ssm_a`             `[num_v_heads]` (16), already stored as `-exp(A_log)`
//! - `ssm_conv1d.weight` `[conv_kernel, conv_dim]` (4, 6144)
//! - `ssm_norm.weight`   `[head_dim]` (128)
//! - `ssm_out.weight`    `[value_dim, hidden]` (2048, 1024)

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::jit;
use crate::kernels::gemm::GemmKernel;
use crate::kernels::resident::DeviceTensor;

/// Static shapes for one Gated DeltaNet layer.
///
/// `head_dim` is shared by keys and values: llama.cpp asserts `S_k == S_v`
/// (both 128 for every real Qwen3.5 release inspected).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    /// Epsilon shared by the L2 q/k norm and the gated output RMSNorm
    /// (`{arch}.attention.layer_norm_rms_epsilon`, 1e-6 for Qwen3.5).
    pub eps: f32,
}

impl GatedDeltaNetConfig {
    pub fn new(
        hidden_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_dim: usize,
        conv_kernel_size: usize,
    ) -> Self {
        Self {
            hidden_size,
            num_k_heads,
            num_v_heads,
            head_dim,
            conv_kernel_size,
            eps: 1e-6,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_dim
    }

    /// Fused q/k/v channel count the conv runs over: `key + key + value`.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.conv_dim()
    }

    pub fn recurrent_len(&self) -> usize {
        self.num_v_heads * self.head_dim * self.head_dim
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 || self.num_k_heads == 0 || self.num_v_heads == 0 || self.head_dim == 0 {
            return Err("GatedDeltaNetConfig has a zero dimension".to_string());
        }
        if self.num_v_heads % self.num_k_heads != 0 {
            return Err(format!(
                "num_v_heads={} must be a multiple of num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            ));
        }
        if self.conv_kernel_size == 0 {
            return Err("conv_kernel_size must be >= 1".to_string());
        }
        if self.eps < 0.0 || !self.eps.is_finite() {
            return Err(format!("eps must be finite and >= 0, got {}", self.eps));
        }
        Ok(())
    }
}

/// Borrowed per-layer weights, in GGUF/ggml layout (see module docs).
#[derive(Clone, Copy, Debug)]
pub struct GatedDeltaNetWeights<'a> {
    pub attn_qkv: &'a [f32],
    pub attn_gate: &'a [f32],
    pub ssm_beta: &'a [f32],
    pub ssm_alpha: &'a [f32],
    pub ssm_dt: &'a [f32],
    pub ssm_a: &'a [f32],
    pub ssm_conv1d: &'a [f32],
    pub ssm_norm: &'a [f32],
    pub ssm_out: &'a [f32],
}

impl GatedDeltaNetWeights<'_> {
    pub fn validate(&self, cfg: &GatedDeltaNetConfig) -> Result<(), String> {
        let checks: [(&str, usize, usize); 9] = [
            ("attn_qkv", self.attn_qkv.len(), cfg.hidden_size * cfg.conv_dim()),
            ("attn_gate", self.attn_gate.len(), cfg.hidden_size * cfg.value_dim()),
            ("ssm_beta", self.ssm_beta.len(), cfg.hidden_size * cfg.num_v_heads),
            ("ssm_alpha", self.ssm_alpha.len(), cfg.hidden_size * cfg.num_v_heads),
            ("ssm_dt", self.ssm_dt.len(), cfg.num_v_heads),
            ("ssm_a", self.ssm_a.len(), cfg.num_v_heads),
            ("ssm_conv1d", self.ssm_conv1d.len(), cfg.conv_kernel_size * cfg.conv_dim()),
            ("ssm_norm", self.ssm_norm.len(), cfg.head_dim),
            ("ssm_out", self.ssm_out.len(), cfg.value_dim() * cfg.hidden_size),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!("{name}.len()={got} != expected {want}"));
            }
        }
        Ok(())
    }
}

/// Persistent per-sequence recurrent state for one Gated DeltaNet layer.
///
/// Two independent recurrent buffers, both updated once per token:
/// - `conv_state`: the causal conv's `(conv_kernel-1) x conv_dim` window of
///   *raw* fused qkv projections (never the post-SiLU conv output), oldest
///   row first.
/// - `recurrent`: the delta-rule `S` matrix, `num_v_heads x head_dim x head_dim`
///   with the key dimension contiguous (`s[key * head_dim + value]`).
#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaNetState {
    pub conv_state: Vec<f32>,
    pub recurrent: Vec<f32>,
}

impl GatedDeltaNetState {
    pub fn zeros(cfg: &GatedDeltaNetConfig) -> Self {
        Self {
            conv_state: vec![0.0; cfg.conv_state_len()],
            recurrent: vec![0.0; cfg.recurrent_len()],
        }
    }

    pub fn reset(&mut self) {
        self.conv_state.iter_mut().for_each(|v| *v = 0.0);
        self.recurrent.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Numerically stable softplus, byte-matching `ggml_compute_softplus_f32`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place `v / sqrt(sum(v^2) + eps)` (`build_gdn_l2_norm` simplifies to this).
fn l2_normalize(v: &mut [f32], eps: f32) {
    let sum_sq: f32 = v.iter().map(|a| a * a).sum();
    let inv = 1.0 / (sum_sq + eps).sqrt();
    for e in v.iter_mut() {
        *e *= inv;
    }
}

/// `out[o] = sum_i w[i + o*in_dim] * x[i]` for ggml `[in, out]` weights.
fn matvec(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize, out: &mut [f32]) {
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        out[o] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
}

/// Exact port of `ggml_compute_forward_ssm_conv_f32` for one token.
///
/// `conv_state` is the `(K-1) x conv_dim` history (oldest row first), `qkv` is
/// the current token's raw fused projection. Returns the *pre-activation*
/// conv output; the caller applies SiLU. `conv1d` is `[K, conv_dim]`
/// (`conv1d[k + c*K]`).
fn conv1d_step(
    conv_state: &[f32],
    qkv: &[f32],
    conv1d: &[f32],
    conv_kernel_size: usize,
    conv_dim: usize,
    out: &mut [f32],
) {
    for c in 0..conv_dim {
        let mut sum = 0.0f32;
        for k in 0..conv_kernel_size {
            // Rows 0..K-1 are history; the last row is the current token.
            let xv = if k < conv_kernel_size - 1 {
                conv_state[k * conv_dim + c]
            } else {
                qkv[c]
            };
            sum += xv * conv1d[k + c * conv_kernel_size];
        }
        out[c] = sum;
    }
}

/// One head's delta-rule state update (exact `build_delta_net_autoregressive`).
///
/// `s` is `head_dim x head_dim`, key-contiguous (`s[key*head_dim + value]`).
/// `q` must already be L2-normalized *and* scaled by `1/sqrt(head_dim)`;
/// `k` L2-normalized; `decay` already exponentiated.
///
/// ```text
/// S       <- S * decay
/// kv[v]   =  sum_k S[k, v] * k[k]
/// delta[v]=  (v[v] - kv[v]) * beta
/// S[k, v] <- S[k, v] + k[k] * delta[v]
/// o[v]    =  sum_k S[k, v] * q[k]
/// ```
#[allow(clippy::too_many_arguments)]
fn delta_rule_head(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: f32,
    decay: f32,
    o: &mut [f32],
) {
    let hd = q.len();

    for e in s.iter_mut() {
        *e *= decay;
    }

    // kv[v] = sum_k S[k, v] * k[k]; delta[v] = (v[v] - kv[v]) * beta.
    let mut delta = vec![0.0f32; hd];
    for vi in 0..hd {
        let mut kv = 0.0f32;
        for ki in 0..hd {
            kv += s[ki * hd + vi] * k[ki];
        }
        delta[vi] = (v[vi] - kv) * beta;
    }

    // S[k, v] += k[k] * delta[v].
    for ki in 0..hd {
        let kk = k[ki];
        let row = &mut s[ki * hd..ki * hd + hd];
        for vi in 0..hd {
            row[vi] += kk * delta[vi];
        }
    }

    // o[v] = sum_k S[k, v] * q[k].
    for vi in 0..hd {
        let mut acc = 0.0f32;
        for ki in 0..hd {
            acc += s[ki * hd + vi] * q[ki];
        }
        o[vi] = acc;
    }
}

/// Run one token through a Gated DeltaNet mixer, mutating `state` in place.
///
/// `x` is the already-normed hidden state (`attn_norm` output, `hidden_size`).
/// Returns the mixer output (`ssm_out` projection, `hidden_size`) to be added
/// to the residual by the caller. The surrounding layer wiring (pre/post norms,
/// FFN, residual) is Phase 21.15.4, not this function.
pub fn step(
    cfg: &GatedDeltaNetConfig,
    w: &GatedDeltaNetWeights,
    x: &[f32],
    state: &mut GatedDeltaNetState,
) -> Result<Vec<f32>, String> {
    cfg.validate()?;
    if x.len() != cfg.hidden_size {
        return Err(format!("x.len()={} != hidden_size={}", x.len(), cfg.hidden_size));
    }
    w.validate(cfg)?;
    if state.conv_state.len() != cfg.conv_state_len() {
        return Err(format!(
            "state.conv_state.len()={} != expected {}",
            state.conv_state.len(),
            cfg.conv_state_len()
        ));
    }
    if state.recurrent.len() != cfg.recurrent_len() {
        return Err(format!(
            "state.recurrent.len()={} != expected {}",
            state.recurrent.len(),
            cfg.recurrent_len()
        ));
    }

    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let conv_dim = cfg.conv_dim();
    let kv = cfg.conv_kernel_size;

    // Input projections.
    let mut qkv = vec![0.0f32; conv_dim];
    matvec(w.attn_qkv, x, hidden, conv_dim, &mut qkv);

    let mut z = vec![0.0f32; value_dim];
    matvec(w.attn_gate, x, hidden, value_dim, &mut z);

    let mut beta = vec![0.0f32; cfg.num_v_heads];
    {
        let mut raw = vec![0.0f32; cfg.num_v_heads];
        matvec(w.ssm_beta, x, hidden, cfg.num_v_heads, &mut raw);
        for (b, r) in beta.iter_mut().zip(raw.iter()) {
            *b = sigmoid(*r);
        }
    }

    // decay[h] = exp(softplus(alpha_h + dt_bias_h) * ssm_a_h)
    let mut decay = vec![0.0f32; cfg.num_v_heads];
    {
        let mut alpha = vec![0.0f32; cfg.num_v_heads];
        matvec(w.ssm_alpha, x, hidden, cfg.num_v_heads, &mut alpha);
        for h in 0..cfg.num_v_heads {
            decay[h] = (softplus(alpha[h] + w.ssm_dt[h]) * w.ssm_a[h]).exp();
        }
    }

    // Causal depthwise conv1d over the fused qkv, then SiLU.
    let mut conv_raw = vec![0.0f32; conv_dim];
    conv1d_step(&state.conv_state, &qkv, w.ssm_conv1d, kv, conv_dim, &mut conv_raw);
    let mut conv_out = vec![0.0f32; conv_dim];
    for (o, r) in conv_out.iter_mut().zip(conv_raw.iter()) {
        *o = silu(*r);
    }

    // Advance the conv window with the *raw* projection (drop oldest, append current).
    if kv > 1 {
        state
            .conv_state
            .copy_within(conv_dim..(kv - 1) * conv_dim, 0);
        let last = (kv - 2) * conv_dim;
        state.conv_state[last..last + conv_dim].copy_from_slice(&qkv);
    }

    // Split convolved q/k/v and L2-normalize q/k (v is left raw).
    let mut q = conv_out[..key_dim].to_vec();
    let mut k = conv_out[key_dim..2 * key_dim].to_vec();
    let v = &conv_out[2 * key_dim..];
    let q_scale = 1.0 / (hd as f32).sqrt();
    for h in 0..cfg.num_k_heads {
        l2_normalize(&mut q[h * hd..(h + 1) * hd], cfg.eps);
        for d in 0..hd {
            q[h * hd + d] *= q_scale;
        }
        l2_normalize(&mut k[h * hd..(h + 1) * hd], cfg.eps);
    }

    // Delta rule, per value head (q/k heads are tiled up when H_k < H_v).
    let mut o = vec![0.0f32; value_dim];
    for h in 0..cfg.num_v_heads {
        let kh = h % cfg.num_k_heads;
        let qh = &q[kh * hd..kh * hd + hd];
        let kk = &k[kh * hd..kh * hd + hd];
        let vh = &v[h * hd..h * hd + hd];
        let sh = &mut state.recurrent[h * hd * hd..(h + 1) * hd * hd];
        let oh = &mut o[h * hd..h * hd + hd];
        delta_rule_head(sh, qh, kk, vh, beta[h], decay[h], oh);
    }

    // Gated RMSNorm output: y = RMSNorm(o, ssm_norm) * silu(z).
    let mut y = vec![0.0f32; value_dim];
    for h in 0..cfg.num_v_heads {
        let oh = &o[h * hd..h * hd + hd];
        let zh = &z[h * hd..h * hd + hd];
        let mean_sq: f32 = oh.iter().map(|a| a * a).sum::<f32>() / hd as f32;
        let rms_inv = 1.0 / (mean_sq + cfg.eps).sqrt();
        for d in 0..hd {
            y[h * hd + d] = oh[d] * rms_inv * w.ssm_norm[d] * silu(zh[d]);
        }
    }

    // Output projection back to hidden_size.
    let mut out = vec![0.0f32; hidden];
    matvec(w.ssm_out, &y, value_dim, hidden, &mut out);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 21.16.1: host/CPU *chunked* Gated DeltaNet forward (prefill).
//
// The sequential `step` above is the reference. This is the chunked closed
// form of the SAME recurrence, ported from llama.cpp's `build_delta_net_chunking`
// (`src/models/delta-net-base.cpp`, the non-KDA branch — Qwen3.5 uses a
// per-head scalar gate). It processes a whole chunk of `T` tokens in one
// parallel pass instead of `T` sequential ones, which is the prefill-speedup
// primitive `PHASE21_15_PLAN.md`'s design decision 1 deferred and
// `PHASE21_16_PLAN.md` scopes.
//
// Per head, the sequential step is (see `delta_rule_head`):
//     W_t = exp(g_t) * S_{t-1}                  (decay first)
//     d_t = beta_t * (v_t - W_t^T k_t)
//     S_t = W_t + k_t ⊗ d_t
//     o_t = S_t^T q_t
// Writing c_t = sum_{s<=t} g_s, the closed form over a chunk of T tokens is
//     (I + M) d^T = beta_t v_t^T - beta_t exp(c_t) k_t^T S_0      (M lower-tri, M[t,j]=beta_t exp(c_t-c_j)(k_t·k_j))
//     o_t = exp(c_t) S_0^T q_t + sum_{j<=t} coef(t,j) (k_j·q_t) d_j   (coef=exp(c_t-c_j) for j<t, 1 for j=t)
//     S_T = exp(c_T) S_0 + sum_j exp(c_T-c_j) k_j ⊗ d_j
// `k` is L2-normalized, so |k_t·k_j|<=1 and the system is well conditioned.
// The causal depthwise conv is parallel across tokens (no recurrence), so it
// is simply batched below.
// ---------------------------------------------------------------------------

/// Chunked/parallel Gated DeltaNet forward over `T = xs.len()/hidden_size`
/// tokens. `xs` is row-major `(T, hidden_size)`; the return is the same shape.
/// `state` is advanced to the end of the chunk in place. Mathematically
/// identical to calling [`step`] once per token; numerically close (different
/// summation order), verified by this module's own tests.
pub fn chunked(
    cfg: &GatedDeltaNetConfig,
    w: &GatedDeltaNetWeights,
    xs: &[f32],
    state: &mut GatedDeltaNetState,
) -> Result<Vec<f32>, String> {
    cfg.validate()?;
    w.validate(cfg)?;
    let hidden = cfg.hidden_size;
    if xs.len() % hidden != 0 {
        return Err(format!("xs.len()={} not a multiple of hidden_size={hidden}", xs.len()));
    }
    let t_count = xs.len() / hidden;
    if t_count == 0 {
        return Ok(Vec::new());
    }
    if state.conv_state.len() != cfg.conv_state_len() {
        return Err(format!(
            "state.conv_state.len()={} != expected {}",
            state.conv_state.len(),
            cfg.conv_state_len()
        ));
    }
    if state.recurrent.len() != cfg.recurrent_len() {
        return Err(format!(
            "state.recurrent.len()={} != expected {}",
            state.recurrent.len(),
            cfg.recurrent_len()
        ));
    }
    let hd = cfg.head_dim;
    let hk = cfg.num_k_heads;
    let hv = cfg.num_v_heads;
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let conv_dim = cfg.conv_dim();
    let ksize = cfg.conv_kernel_size;
    let history = ksize - 1;

    // 1. Per-token input projections.
    let mut qkv = vec![0.0f32; t_count * conv_dim];
    let mut z = vec![0.0f32; t_count * value_dim];
    let mut beta = vec![0.0f32; t_count * hv];
    let mut logdecay = vec![0.0f32; t_count * hv];
    {
        let mut raw_beta = vec![0.0f32; hv];
        let mut raw_alpha = vec![0.0f32; hv];
        for t in 0..t_count {
            let x = &xs[t * hidden..(t + 1) * hidden];
            matvec(w.attn_qkv, x, hidden, conv_dim, &mut qkv[t * conv_dim..(t + 1) * conv_dim]);
            matvec(w.attn_gate, x, hidden, value_dim, &mut z[t * value_dim..(t + 1) * value_dim]);
            matvec(w.ssm_beta, x, hidden, hv, &mut raw_beta);
            for h in 0..hv {
                beta[t * hv + h] = sigmoid(raw_beta[h]);
            }
            matvec(w.ssm_alpha, x, hidden, hv, &mut raw_alpha);
            for h in 0..hv {
                // log-decay: `g = softplus(alpha + dt_bias) * ssm_a`, i.e. ln(decay).
                logdecay[t * hv + h] = softplus(raw_alpha[h] + w.ssm_dt[h]) * w.ssm_a[h];
            }
        }
    }

    // 2. Batched causal depthwise conv1d (+ SiLU). `E` = incoming history rows
    //    followed by this chunk's raw qkv rows; token `t`'s window ends at
    //    `E[history + t]`.
    let mut conv_out = vec![0.0f32; t_count * conv_dim];
    {
        let e_row = |i: usize| -> &[f32] {
            if i < history {
                &state.conv_state[i * conv_dim..(i + 1) * conv_dim]
            } else {
                &qkv[(i - history) * conv_dim..(i - history + 1) * conv_dim]
            }
        };
        for t in 0..t_count {
            for c in 0..conv_dim {
                let mut acc = 0.0f32;
                for k in 0..ksize {
                    let xv = if k < history { e_row(t + k)[c] } else { qkv[t * conv_dim + c] };
                    acc += xv * w.ssm_conv1d[k + c * ksize];
                }
                conv_out[t * conv_dim + c] = silu(acc);
            }
        }
        if history > 0 {
            let total_rows = history + t_count;
            let mut new_state = vec![0.0f32; history * conv_dim];
            for r in 0..history {
                let er = total_rows - history + r;
                new_state[r * conv_dim..(r + 1) * conv_dim].copy_from_slice(e_row(er));
            }
            state.conv_state.copy_from_slice(&new_state);
        }
    }

    // 3. Split convolved q/k and L2-normalize (q also scaled by 1/sqrt(hd)).
    let mut qn = vec![0.0f32; t_count * key_dim];
    let mut kn = vec![0.0f32; t_count * key_dim];
    {
        let q_scale = 1.0 / (hd as f32).sqrt();
        for t in 0..t_count {
            let co = &conv_out[t * conv_dim..(t + 1) * conv_dim];
            for h in 0..hk {
                let mut qh = co[h * hd..(h + 1) * hd].to_vec();
                l2_normalize(&mut qh, cfg.eps);
                for d in 0..hd {
                    qh[d] *= q_scale;
                }
                qn[t * key_dim + h * hd..t * key_dim + (h + 1) * hd].copy_from_slice(&qh);
                let mut kh = co[key_dim + h * hd..key_dim + (h + 1) * hd].to_vec();
                l2_normalize(&mut kh, cfg.eps);
                kn[t * key_dim + h * hd..t * key_dim + (h + 1) * hd].copy_from_slice(&kh);
            }
        }
    }

    // 4. Per value head: chunked delta-rule solve + read.
    let mut o = vec![0.0f32; t_count * value_dim];
    for h in 0..hv {
        let gkh = h % hk;
        let c = {
            let mut c = vec![0.0f32; t_count];
            let mut acc = 0.0f32;
            for t in 0..t_count {
                acc += logdecay[t * hv + h];
                c[t] = acc;
            }
            c
        };
        let s0 = &state.recurrent[h * hd * hd..(h + 1) * hd * hd]; // key-major [hd, hd]

        // RHS[t] = beta_t * (v_t - exp(c_t) * (k_t^T S0))
        let mut rhs = vec![0.0f32; t_count * hd];
        for t in 0..t_count {
            let kt = &kn[t * key_dim + gkh * hd..t * key_dim + (gkh + 1) * hd];
            let vt = &conv_out[t * conv_dim + 2 * key_dim + h * hd..t * conv_dim + 2 * key_dim + (h + 1) * hd];
            let bt = beta[t * hv + h];
            let ect = c[t].exp();
            for vi in 0..hd {
                let mut ks = 0.0f32;
                for ki in 0..hd {
                    ks += kt[ki] * s0[ki * hd + vi];
                }
                rhs[t * hd + vi] = bt * (vt[vi] - ect * ks);
            }
        }

        // Forward-substitute (I + M) X = rhs, M[t,j] = beta_t exp(c_t-c_j)(k_t·k_j).
        let mut d = vec![0.0f32; t_count * hd];
        for t in 0..t_count {
            let kt = &kn[t * key_dim + gkh * hd..t * key_dim + (gkh + 1) * hd];
            let bt = beta[t * hv + h];
            let mut xt = rhs[t * hd..(t + 1) * hd].to_vec();
            for j in 0..t {
                let kj = &kn[j * key_dim + gkh * hd..j * key_dim + (gkh + 1) * hd];
                let mut gram = 0.0f32;
                for d2 in 0..hd {
                    gram += kt[d2] * kj[d2];
                }
                let m = bt * (c[t] - c[j]).exp() * gram;
                let dj = &d[j * hd..(j + 1) * hd];
                for vi in 0..hd {
                    xt[vi] -= m * dj[vi];
                }
            }
            d[t * hd..(t + 1) * hd].copy_from_slice(&xt);
        }

        // o_t = exp(c_t) S0^T q_t + sum_{j<=t} coef (k_j·q_t) d_j
        for t in 0..t_count {
            let qt = &qn[t * key_dim + gkh * hd..t * key_dim + (gkh + 1) * hd];
            let ect = c[t].exp();
            for vi in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..hd {
                    acc += s0[ki * hd + vi] * qt[ki];
                }
                o[t * value_dim + h * hd + vi] = ect * acc;
            }
            for j in 0..=t {
                let kj = &kn[j * key_dim + gkh * hd..j * key_dim + (gkh + 1) * hd];
                let mut kq = 0.0f32;
                for d2 in 0..hd {
                    kq += kj[d2] * qt[d2];
                }
                let coef = if j < t { (c[t] - c[j]).exp() } else { 1.0 };
                let dj = &d[j * hd..(j + 1) * hd];
                for vi in 0..hd {
                    o[t * value_dim + h * hd + vi] += coef * kq * dj[vi];
                }
            }
        }

        // State advance: S_T = exp(c_T) S0 + sum_j exp(c_T-c_j) k_j ⊗ d_j.
        let c_last = c[t_count - 1];
        let sm = &mut state.recurrent[h * hd * hd..(h + 1) * hd * hd];
        let ec_last = c_last.exp();
        for e in sm.iter_mut() {
            *e *= ec_last;
        }
        for j in 0..t_count {
            let kj = &kn[j * key_dim + gkh * hd..j * key_dim + (gkh + 1) * hd];
            let coef = (c_last - c[j]).exp();
            let dj = &d[j * hd..(j + 1) * hd];
            for ki in 0..hd {
                let kk = coef * kj[ki];
                let row = &mut sm[ki * hd..(ki + 1) * hd];
                for vi in 0..hd {
                    row[vi] += kk * dj[vi];
                }
            }
        }
    }

    // 5. Gated RMSNorm output + ssm_out projection, per token.
    let mut y = vec![0.0f32; t_count * value_dim];
    for t in 0..t_count {
        for h in 0..hv {
            let oh = &o[t * value_dim + h * hd..t * value_dim + (h + 1) * hd];
            let zh = &z[t * value_dim + h * hd..t * value_dim + (h + 1) * hd];
            let mean_sq: f32 = oh.iter().map(|a| a * a).sum::<f32>() / hd as f32;
            let rms_inv = 1.0 / (mean_sq + cfg.eps).sqrt();
            for dd in 0..hd {
                y[t * value_dim + h * hd + dd] = oh[dd] * rms_inv * w.ssm_norm[dd] * silu(zh[dd]);
            }
        }
    }
    let mut out = vec![0.0f32; t_count * hidden];
    for t in 0..t_count {
        matvec(
            w.ssm_out,
            &y[t * value_dim..(t + 1) * value_dim],
            value_dim,
            hidden,
            &mut out[t * hidden..(t + 1) * hidden],
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 21.15.3: GPU-resident Gated DeltaNet kernel + state residency.
//
// Same math as the host `step` above, split into six NVRTC kernels so the
// per-token recurrence runs entirely on device:
//   `gdn_matvec_kernel`       GGUF-layout GEMV (`y[o] = sum_i w[o*in+i] * x[i]`)
//   `gdn_conv_kernel`         causal depthwise conv1d + SiLU + window advance
//   `gdn_l2_norm_kernel`      per-head L2 norm (q scaled, k not), in place
//   `gdn_gates_kernel`        `beta = sigmoid(...)`, `decay = exp(softplus(...)*a)`
//   `gdn_delta_kernel`        one thread per value column, the delta-rule update
//   `gdn_gated_norm_kernel`   `y = RMSNorm(o, ssm_norm) * silu(z)`, in place
//
// Design decision 2 (`PHASE21_15_PLAN.md`) puts the two recurrent buffers in
// `GatedDeltaNetDeviceState` (a `conv_state` window and a per-head `S`),
// allocated once and reused across every decode token — never a per-call
// upload. Weights are uploaded once into `GatedDeltaNetDeviceWeights`; all
// intermediates live in `GatedDeltaNetScratch`. No kernel synchronizes (the
// caller syncs once per chain), matching `RmsNormKernel::forward_resident`.
// ---------------------------------------------------------------------------

/// CUDA C source for every Gated DeltaNet kernel. `float` math throughout
/// (matching the host reference), so it needs no CUDA headers — `expf`/
/// `logf`/`sqrtf` are NVRTC device builtins.
const GDN_KERNEL_SOURCE: &str = r#"
// y[o] = sum_i w[i + o*in_dim] * x[i]  (ggml [in, out] weight layout).
extern "C" __global__ void gdn_matvec_kernel(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned int in_dim,
    unsigned int out_dim
) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    const float* row = w + (unsigned long long)o * in_dim;
    float acc = 0.0f;
    for (unsigned int i = 0; i < in_dim; i++) {
        acc += row[i] * x[i];
    }
    out[o] = acc;
}

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
// head; `blockDim.x` must be a power of two >= 1 (the caller picks it).
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
// conv/silu/normalized fused buffer; q and v are read through offsets.
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

// ---------------------------------------------------------------------------
// Phase 21.16.2: chunked (multi-token) prefill kernels.
//
// These implement the closed form of the same recurrence the six kernels
// above compute one token at a time. The projections are *not* here — they
// are batched through one cuBLAS GEMM per layer (the dominant win; see
// `GatedDeltaNetKernel::chunk_resident`). What remains genuinely chunked is
// the causal conv, the cumulative log-decay, and the lower-triangular delta
// solve + read + chunk-end state update.
// ---------------------------------------------------------------------------

// Pre-fusion originals, restored for an A/B perf comparison only.
extern "C" __global__ void gdn_chunk_conv_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_state,
    float* __restrict__ conv_out,
    unsigned int T,
    unsigned int conv_dim,
    unsigned int k
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((unsigned long long)idx >= (unsigned long long)T * conv_dim) return;
    unsigned int t = idx / conv_dim;
    unsigned int c = idx % conv_dim;
    unsigned int history = k - 1;
    float acc = 0.0f;
    for (unsigned int kk = 0; kk < k; kk++) {
        unsigned int i = t + kk;
        float xv = (i < history) ? conv_state[i * conv_dim + c]
                                 : qkv[(i - history) * conv_dim + c];
        acc += xv * conv_w[kk + c * k];
    }
    conv_out[idx] = acc / (1.0f + expf(-acc));
}

extern "C" __global__ void gdn_chunk_conv_state_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ conv_state_in,
    float* __restrict__ conv_state_out,
    unsigned int T,
    unsigned int conv_dim,
    unsigned int k
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    unsigned int history = k - 1;
    for (unsigned int r = 0; r < history; r++) {
        unsigned int i = T + r;
        float v = (i < history) ? conv_state_in[i * conv_dim + c]
                                : qkv[(i - history) * conv_dim + c];
        conv_state_out[r * conv_dim + c] = v;
    }
}

extern "C" __global__ void gdn_chunk_cumsum_kernel(
    const float* __restrict__ logdecay,
    float* __restrict__ c,
    unsigned int T,
    unsigned int H
) {
    unsigned int h = blockIdx.x;
    float acc = 0.0f;
    for (unsigned int t = 0; t < T; t++) {
        acc += logdecay[t * H + h];
        c[t * H + h] = acc;
    }
}

// Batched per-(row, head) L2 norm, in place. Grid covers `rows * heads`; the
// base of row `r`, head `h` is `x + r*row_stride + head_offset + h*head_dim`.
extern "C" __global__ void gdn_chunk_l2_kernel(
    float* __restrict__ x,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int head_offset,
    unsigned int row_stride,
    float eps,
    float scale
) {
    unsigned int row = blockIdx.x / heads;
    unsigned int h = blockIdx.x % heads;
    unsigned int t = threadIdx.x;
    float* xh = x + (unsigned long long)row * row_stride + head_offset + (unsigned long long)h * head_dim;
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

// Batched gates over a chunk: `logdecay[t,h] = softplus(alpha+dt)*a`,
// `beta[t,h] = sigmoid(beta_raw)`. `idx` is the flattened `(t, h)` index.
extern "C" __global__ void gdn_chunk_gates_kernel(
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt,
    const float* __restrict__ a,
    float* __restrict__ logdecay,
    float* __restrict__ beta,
    unsigned int T,
    unsigned int H
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * H) return;
    unsigned int h = idx % H;
    float alpha = alpha_raw[idx] + dt[h];
    float sp = (alpha > 20.0f) ? alpha : logf(1.0f + expf(alpha));
    logdecay[idx] = sp * a[h];
    beta[idx] = 1.0f / (1.0f + expf(-beta_raw[idx]));
}

// The chunked delta-rule solve, output read, and chunk-end state update for
// one value head. One block per value head, one thread per value column `vi`;
// each thread owns column `vi` of `S` (all keys) and its own deltas, so the
// whole update is race-free without block reductions.
//
// `qk` holds the *normalized* fused conv buffer, row-major `(T, row_stride)`;
// the q segment is at offset 0, k at `k_offset` (`= key_dim`), v at
// `2*H_k*hd`, with the q/k head tiled by `kh = h % H_k`. `d` is
// `(T, H_v, hd)` scratch for the per-token deltas.
extern "C" __global__ void gdn_chunk_delta_kernel(
    float* __restrict__ S,
    const float* __restrict__ qk,
    unsigned int k_offset,
    unsigned int row_stride,
    const float* __restrict__ beta,
    const float* __restrict__ c,
    float* __restrict__ d,
    float* __restrict__ o,
    unsigned int T,
    unsigned int hd,
    unsigned int H_k,
    unsigned int H_v
) {
    unsigned int h = blockIdx.x;
    unsigned int vi = threadIdx.x;
    if (h >= H_v || vi >= hd) return;
    unsigned int kh = h % H_k;
    float* Sh = S + (unsigned long long)h * hd * hd;
    const unsigned long long koff = (unsigned long long)kh * hd;
    const unsigned long long voff = 2ull * H_k * hd;
    const unsigned long long d_stride = (unsigned long long)H_v * hd;
    const unsigned long long dcol = (unsigned long long)h * hd + vi;

    // 1. RHS: beta_t * (v_t - exp(c_t) * (k_t^T S0)).
    for (unsigned int t = 0; t < T; t++) {
        float ct = c[t * H_v + h];
        float bt = beta[t * H_v + h];
        float ect = expf(ct);
        const float* kt = qk + (unsigned long long)t * row_stride + k_offset + koff;
        const float* vt = qk + (unsigned long long)t * row_stride + voff + (unsigned long long)h * hd;
        float ks = 0.0f;
        for (unsigned int ki = 0; ki < hd; ki++) ks += kt[ki] * Sh[ki * hd + vi];
        d[(unsigned long long)t * d_stride + dcol] = bt * (vt[vi] - ect * ks);
    }

    // 2. Forward substitution: (I + M) d = rhs, with
    //    M[t,j] = beta_t * exp(c_t - c_j) * (k_t . k_j).
    for (unsigned int t = 0; t < T; t++) {
        float ct = c[t * H_v + h];
        float bt = beta[t * H_v + h];
        const float* kt = qk + (unsigned long long)t * row_stride + k_offset + koff;
        float x = d[(unsigned long long)t * d_stride + dcol];
        for (unsigned int j = 0; j < t; j++) {
            const float* kj = qk + (unsigned long long)j * row_stride + k_offset + koff;
            float gram = 0.0f;
            for (unsigned int d2 = 0; d2 < hd; d2++) gram += kt[d2] * kj[d2];
            float m = bt * expf(ct - c[j * H_v + h]) * gram;
            x -= m * d[(unsigned long long)j * d_stride + dcol];
        }
        d[(unsigned long long)t * d_stride + dcol] = x;
    }

    // 3. Output read: o_t = exp(c_t) S0^T q_t + sum_{j<=t} coef (k_j . q_t) d_j.
    for (unsigned int t = 0; t < T; t++) {
        float ct = c[t * H_v + h];
        float ect = expf(ct);
        const float* qt = qk + (unsigned long long)t * row_stride + koff;
        float acc = 0.0f;
        for (unsigned int ki = 0; ki < hd; ki++) acc += Sh[ki * hd + vi] * qt[ki];
        acc *= ect;
        for (unsigned int j = 0; j <= t; j++) {
            const float* kj = qk + (unsigned long long)j * row_stride + k_offset + koff;
            float kq = 0.0f;
            for (unsigned int d2 = 0; d2 < hd; d2++) kq += kj[d2] * qt[d2];
            float coef = (j < t) ? expf(ct - c[j * H_v + h]) : 1.0f;
            acc += coef * kq * d[(unsigned long long)j * d_stride + dcol];
        }
        o[(unsigned long long)t * (unsigned long long)H_v * hd + (unsigned long long)h * hd + vi] = acc;
    }

    // 4. State advance: S_T = exp(c_T) S0 + sum_j exp(c_T - c_j) k_j (x) d_j.
    float clast = c[(T - 1) * H_v + h];
    float ecl = expf(clast);
    for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] *= ecl;
    for (unsigned int j = 0; j < T; j++) {
        const float* kj = qk + (unsigned long long)j * row_stride + k_offset + koff;
        float coef = expf(clast - c[j * H_v + h]);
        float dj = d[(unsigned long long)j * d_stride + dcol];
        for (unsigned int ki = 0; ki < hd; ki++) Sh[ki * hd + vi] += coef * kj[ki] * dj;
    }
}


// Batched gated RMSNorm output: y = RMSNorm(o, norm_w) * silu(z). One block
// per `(t, value head)`, `grid.x = T * H_v`.
extern "C" __global__ void gdn_chunk_gated_norm_kernel(
    const float* __restrict__ o,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    float* __restrict__ y,
    unsigned int H_v,
    unsigned int hd,
    float eps
) {
    unsigned int row = blockIdx.x / H_v;
    unsigned int h = blockIdx.x % H_v;
    unsigned int t = threadIdx.x;
    unsigned long long base = (unsigned long long)row * H_v * hd + (unsigned long long)h * hd;
    const float* oh = o + base;
    const float* zh = z + base;
    float* yh = y + base;
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
"#;

/// Smallest power-of-two block size >= `head_dim` that still runs the shared
/// reduction correctly, capped at the kernels' `red[1024]` shared array
/// (validated in [`GatedDeltaNetKernel::step_resident`]).
fn gdn_head_block(head_dim: usize) -> u32 {
    head_dim.next_power_of_two().clamp(32, 1024) as u32
}

fn gdn_launch_config(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Phase 21.21: shared row-range validation for the split recurrent stages.
fn check_row_range(
    scratch: &GatedDeltaNetChunkScratch,
    row_start: usize,
    t_count: usize,
    what: &str,
) -> Result<(), String> {
    if t_count == 0 {
        return Err(format!("GatedDeltaNet {what} requires t_count >= 1"));
    }
    if row_start + t_count > scratch.max_chunk {
        return Err(format!(
            "row_start={row_start} + t_count={t_count} exceeds GatedDeltaNetChunkScratch max_chunk={}",
            scratch.max_chunk
        ));
    }
    Ok(())
}

/// Phase 21.21: shared per-sequence state-size validation.
fn check_state(cfg: &GatedDeltaNetConfig, state: &GatedDeltaNetDeviceState) -> Result<(), String> {
    if state.conv_state.len() != cfg.conv_state_len() {
        return Err(format!(
            "state.conv_state.len()={} != expected {}",
            state.conv_state.len(),
            cfg.conv_state_len()
        ));
    }
    if state.recurrent.len() != cfg.recurrent_len() {
        return Err(format!(
            "state.recurrent.len()={} != expected {}",
            state.recurrent.len(),
            cfg.recurrent_len()
        ));
    }
    Ok(())
}

/// Per-layer weights uploaded once and kept resident (Phase 21.15.3).
pub struct GatedDeltaNetDeviceWeights {
    pub attn_qkv: DeviceTensor,
    pub attn_gate: DeviceTensor,
    pub ssm_beta: DeviceTensor,
    pub ssm_alpha: DeviceTensor,
    pub ssm_dt: DeviceTensor,
    pub ssm_a: DeviceTensor,
    pub ssm_conv1d: DeviceTensor,
    pub ssm_norm: DeviceTensor,
    pub ssm_out: DeviceTensor,
}

impl GatedDeltaNetDeviceWeights {
    pub fn from_host(
        device: &Arc<CudaDevice>,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetWeights<'_>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        w.validate(cfg)?;
        Ok(Self {
            attn_qkv: DeviceTensor::from_host(device, w.attn_qkv, vec![cfg.hidden_size * cfg.conv_dim()])?,
            attn_gate: DeviceTensor::from_host(device, w.attn_gate, vec![cfg.hidden_size * cfg.value_dim()])?,
            ssm_beta: DeviceTensor::from_host(device, w.ssm_beta, vec![cfg.hidden_size * cfg.num_v_heads])?,
            ssm_alpha: DeviceTensor::from_host(device, w.ssm_alpha, vec![cfg.hidden_size * cfg.num_v_heads])?,
            ssm_dt: DeviceTensor::from_host(device, w.ssm_dt, vec![cfg.num_v_heads])?,
            ssm_a: DeviceTensor::from_host(device, w.ssm_a, vec![cfg.num_v_heads])?,
            ssm_conv1d: DeviceTensor::from_host(device, w.ssm_conv1d, vec![cfg.conv_kernel_size * cfg.conv_dim()])?,
            ssm_norm: DeviceTensor::from_host(device, w.ssm_norm, vec![cfg.head_dim])?,
            ssm_out: DeviceTensor::from_host(device, w.ssm_out, vec![cfg.value_dim() * cfg.hidden_size])?,
        })
    }
}

/// Persistent GPU-resident recurrent state for one Gated DeltaNet layer.
///
/// Mirrors [`GatedDeltaNetState`]: `conv_state` is the `(K-1) x conv_dim`
/// raw-projection window (oldest row first) and `recurrent` is the delta-rule
/// `S` (`num_v_heads x head_dim x head_dim`, key-contiguous). Both are
/// allocated once and survive every decode token; [`reset`] zeroes them for a
/// fresh sequence without reallocating.
///
/// [`reset`]: GatedDeltaNetDeviceState::reset
pub struct GatedDeltaNetDeviceState {
    pub conv_state: DeviceTensor,
    pub recurrent: DeviceTensor,
}

impl GatedDeltaNetDeviceState {
    pub fn zeros(device: &Arc<CudaDevice>, cfg: &GatedDeltaNetConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            conv_state: DeviceTensor::zeros(device, vec![cfg.conv_state_len()])?,
            recurrent: DeviceTensor::zeros(device, vec![cfg.recurrent_len()])?,
        })
    }

    pub fn from_host(
        device: &Arc<CudaDevice>,
        cfg: &GatedDeltaNetConfig,
        state: &GatedDeltaNetState,
    ) -> Result<Self, String> {
        cfg.validate()?;
        if state.conv_state.len() != cfg.conv_state_len() {
            return Err(format!(
                "state.conv_state.len()={} != expected {}",
                state.conv_state.len(),
                cfg.conv_state_len()
            ));
        }
        if state.recurrent.len() != cfg.recurrent_len() {
            return Err(format!(
                "state.recurrent.len()={} != expected {}",
                state.recurrent.len(),
                cfg.recurrent_len()
            ));
        }
        Ok(Self {
            conv_state: DeviceTensor::from_host(device, &state.conv_state, vec![cfg.conv_state_len()])?,
            recurrent: DeviceTensor::from_host(device, &state.recurrent, vec![cfg.recurrent_len()])?,
        })
    }

    pub fn to_host(&self, device: &Arc<CudaDevice>) -> Result<GatedDeltaNetState, String> {
        Ok(GatedDeltaNetState {
            conv_state: self.conv_state.to_host(device)?,
            recurrent: self.recurrent.to_host(device)?,
        })
    }

    pub fn reset(&mut self, device: &Arc<CudaDevice>) -> Result<(), String> {
        device
            .memset_zeros(self.conv_state.buf_mut())
            .map_err(|e| format!("GDN conv_state reset: {e}"))?;
        device
            .memset_zeros(self.recurrent.buf_mut())
            .map_err(|e| format!("GDN recurrent reset: {e}"))?;
        Ok(())
    }
}

/// Per-call device scratch for [`GatedDeltaNetKernel::step_resident`] —
/// allocated once and reused every token, same pattern as `LayerScratch`.
pub struct GatedDeltaNetScratch {
    qkv: DeviceTensor,
    conv: DeviceTensor,
    z: DeviceTensor,
    beta_raw: DeviceTensor,
    alpha_raw: DeviceTensor,
    beta: DeviceTensor,
    decay: DeviceTensor,
    o: DeviceTensor,
    y: DeviceTensor,
}

impl GatedDeltaNetScratch {
    pub fn new(device: &Arc<CudaDevice>, cfg: &GatedDeltaNetConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            qkv: DeviceTensor::zeros(device, vec![cfg.conv_dim()])?,
            conv: DeviceTensor::zeros(device, vec![cfg.conv_dim()])?,
            z: DeviceTensor::zeros(device, vec![cfg.value_dim()])?,
            beta_raw: DeviceTensor::zeros(device, vec![cfg.num_v_heads])?,
            alpha_raw: DeviceTensor::zeros(device, vec![cfg.num_v_heads])?,
            beta: DeviceTensor::zeros(device, vec![cfg.num_v_heads])?,
            decay: DeviceTensor::zeros(device, vec![cfg.num_v_heads])?,
            o: DeviceTensor::zeros(device, vec![cfg.value_dim()])?,
            y: DeviceTensor::zeros(device, vec![cfg.value_dim()])?,
        })
    }
}

/// Per-call device scratch for [`GatedDeltaNetKernel::chunk_resident`] (Phase
/// 21.16.2) — the multi-token sibling of [`GatedDeltaNetScratch`], sized for
/// `max_chunk` tokens and allocated once (shared across every DeltaNet layer,
/// since layers run one at a time). All buffers are row-major `(T, dim)`.
pub struct GatedDeltaNetChunkScratch {
    max_chunk: usize,
    qkv: DeviceTensor,
    conv: DeviceTensor,
    z: DeviceTensor,
    beta_raw: DeviceTensor,
    alpha_raw: DeviceTensor,
    beta: DeviceTensor,
    /// `logdecay` is the per-token `g_t` (the log-decay); `c` is its
    /// per-head inclusive prefix sum.
    logdecay: DeviceTensor,
    c: DeviceTensor,
    /// Per-token deltas `d`, shape `(T, H_v, hd)`.
    d: DeviceTensor,
    o: DeviceTensor,
    y: DeviceTensor,
    /// Staging buffer for the conv-window advance, so the in-place update
    /// reads the old window and writes a distinct buffer (see
    /// `gdn_chunk_conv_state_kernel`).
    conv_state_tmp: DeviceTensor,
}

impl GatedDeltaNetChunkScratch {
    pub fn new(device: &Arc<CudaDevice>, cfg: &GatedDeltaNetConfig, max_chunk: usize) -> Result<Self, String> {
        cfg.validate()?;
        if max_chunk == 0 {
            return Err("GatedDeltaNetChunkScratch max_chunk must be >= 1".to_string());
        }
        Ok(Self {
            max_chunk,
            qkv: DeviceTensor::zeros(device, vec![max_chunk, cfg.conv_dim()])?,
            conv: DeviceTensor::zeros(device, vec![max_chunk, cfg.conv_dim()])?,
            z: DeviceTensor::zeros(device, vec![max_chunk, cfg.value_dim()])?,
            beta_raw: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads])?,
            alpha_raw: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads])?,
            beta: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads])?,
            logdecay: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads])?,
            c: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads])?,
            d: DeviceTensor::zeros(device, vec![max_chunk, cfg.num_v_heads, cfg.head_dim])?,
            o: DeviceTensor::zeros(device, vec![max_chunk, cfg.value_dim()])?,
            y: DeviceTensor::zeros(device, vec![max_chunk, cfg.value_dim()])?,
            conv_state_tmp: DeviceTensor::zeros(device, vec![cfg.conv_state_len()])?,
        })
    }

    /// The largest chunk this scratch can process.
    pub fn max_chunk(&self) -> usize {
        self.max_chunk
    }
}

struct GatedDeltaNetKernels {
    matvec: CudaFunction,
    conv: CudaFunction,
    l2_norm: CudaFunction,
    gates: CudaFunction,
    delta: CudaFunction,
    gated_norm: CudaFunction,
    // Phase 21.16.2 chunked kernels.
    chunk_conv: CudaFunction,
    chunk_conv_state: CudaFunction,
    chunk_l2: CudaFunction,
    chunk_gates: CudaFunction,
    chunk_cumsum: CudaFunction,
    chunk_delta: CudaFunction,
    chunk_gated_norm: CudaFunction,
}

/// GPU-resident Gated DeltaNet mixer (Phase 21.15.3). Keeps an optional
/// compiled NVRTC path, exactly like [`RmsNormKernel`](super::rmsnorm::RmsNormKernel):
/// construction never fails when NVRTC is unavailable, [`has_gpu`] reports
/// whether it compiled, and [`step`](Self::step) falls back to the host
/// reference rather than panicking.
///
/// [`has_gpu`]: GatedDeltaNetKernel::has_gpu
pub struct GatedDeltaNetKernel {
    device: Arc<CudaDevice>,
    kernels: Option<GatedDeltaNetKernels>,
    /// Phase 21.16.2: cuBLAS handle used only by the chunked prefill path to
    /// batch the per-token projection GEMVs into one GEMM per layer. The
    /// decode (`step_resident`) path never touches it.
    gemm: Option<GemmKernel>,
}

impl GatedDeltaNetKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
        let kernels = match Self::try_compile(&device) {
            Ok(k) => {
                eprintln!("INFO: Gated DeltaNet NVRTC kernels compiled and loaded on GPU");
                Some(k)
            }
            Err(e) => {
                eprintln!("WARNING: Gated DeltaNet NVRTC compile/load failed, GPU path unavailable: {e}");
                None
            }
        };
        let gemm = match GemmKernel::new(device.clone()) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("WARNING: Gated DeltaNet chunked path cuBLAS init failed, chunk path unavailable: {e}");
                None
            }
        };
        Ok(Self { device, kernels, gemm })
    }

    fn try_compile(device: &Arc<CudaDevice>) -> Result<GatedDeltaNetKernels, String> {
        let arch = jit::device_arch(device)?;
        let load = |name: &'static str| jit::compile_and_load(GDN_KERNEL_SOURCE, name, &arch, device);
        Ok(GatedDeltaNetKernels {
            matvec: load("gdn_matvec_kernel")?,
            conv: load("gdn_conv_kernel")?,
            l2_norm: load("gdn_l2_norm_kernel")?,
            gates: load("gdn_gates_kernel")?,
            delta: load("gdn_delta_kernel")?,
            gated_norm: load("gdn_gated_norm_kernel")?,
            chunk_conv: load("gdn_chunk_conv_kernel")?,
            chunk_conv_state: load("gdn_chunk_conv_state_kernel")?,
            chunk_l2: load("gdn_chunk_l2_kernel")?,
            chunk_gates: load("gdn_chunk_gates_kernel")?,
            chunk_cumsum: load("gdn_chunk_cumsum_kernel")?,
            chunk_delta: load("gdn_chunk_delta_kernel")?,
            chunk_gated_norm: load("gdn_chunk_gated_norm_kernel")?,
        })
    }

    /// Whether NVRTC compiled and loaded every kernel. Callers that need to
    /// know whether the GPU path is real (rather than silently falling back)
    /// should check this.
    pub fn has_gpu(&self) -> bool {
        self.kernels.is_some()
    }

    /// Whether the chunked prefill path is available — both the NVRTC kernels
    /// and the cuBLAS handle the batched projections need.
    pub fn has_chunk_gpu(&self) -> bool {
        self.kernels.is_some() && self.gemm.is_some()
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn upload_weights(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetWeights<'_>,
    ) -> Result<GatedDeltaNetDeviceWeights, String> {
        GatedDeltaNetDeviceWeights::from_host(&self.device, cfg, w)
    }

    pub fn state_zeros(&self, cfg: &GatedDeltaNetConfig) -> Result<GatedDeltaNetDeviceState, String> {
        GatedDeltaNetDeviceState::zeros(&self.device, cfg)
    }

    pub fn scratch(&self, cfg: &GatedDeltaNetConfig) -> Result<GatedDeltaNetScratch, String> {
        GatedDeltaNetScratch::new(&self.device, cfg)
    }

    /// Allocate the multi-token scratch for [`chunk_resident`](Self::chunk_resident).
    pub fn chunk_scratch(
        &self,
        cfg: &GatedDeltaNetConfig,
        max_chunk: usize,
    ) -> Result<GatedDeltaNetChunkScratch, String> {
        GatedDeltaNetChunkScratch::new(&self.device, cfg, max_chunk)
    }

    /// Device-resident one-token step (Phase 21.15.3's parity-gated kernel).
    ///
    /// `x` is the already-normed hidden state, `out` is the mixer output
    /// (`ssm_out` projection, `hidden_size`) to be added to the residual by
    /// the caller, and `state`/`scratch`/`out` are all pre-allocated resident
    /// buffers. No allocation, no host round trip, no synchronize — the
    /// caller syncs once per chain, matching
    /// [`RmsNormKernel::forward_resident`](super::rmsnorm::RmsNormKernel::forward_resident).
    ///
    /// The surrounding layer wiring (pre/post norms, residual, FFN) is Phase
    /// 21.15.4, not this method.
    #[allow(clippy::too_many_arguments)]
    pub fn step_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        x: &CudaSlice<f32>,
        state: &mut GatedDeltaNetDeviceState,
        scratch: &mut GatedDeltaNetScratch,
        out: &mut DeviceTensor,
    ) -> Result<(), String> {
        cfg.validate()?;
        if cfg.head_dim > 1024 {
            return Err(format!(
                "head_dim={} exceeds the 1024 supported by the GDN reduction kernels",
                cfg.head_dim
            ));
        }
        let k = self
            .kernels
            .as_ref()
            .ok_or("Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)")?;

        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let h_v = cfg.num_v_heads;

        if x.len() != hidden {
            return Err(format!("x.len()={} != hidden_size={hidden}", x.len()));
        }
        if out.len() != hidden {
            return Err(format!("out.len()={} != hidden_size={hidden}", out.len()));
        }
        if state.conv_state.len() != cfg.conv_state_len() {
            return Err(format!(
                "state.conv_state.len()={} != expected {}",
                state.conv_state.len(),
                cfg.conv_state_len()
            ));
        }
        if state.recurrent.len() != cfg.recurrent_len() {
            return Err(format!(
                "state.recurrent.len()={} != expected {}",
                state.recurrent.len(),
                cfg.recurrent_len()
            ));
        }
        if scratch.qkv.len() != conv_dim
            || scratch.conv.len() != conv_dim
            || scratch.z.len() != value_dim
            || scratch.beta_raw.len() != h_v
            || scratch.alpha_raw.len() != h_v
            || scratch.beta.len() != h_v
            || scratch.decay.len() != h_v
            || scratch.o.len() != value_dim
            || scratch.y.len() != value_dim
        {
            return Err("GatedDeltaNetScratch was built for a different GatedDeltaNetConfig".to_string());
        }

        let threads = 256u32;
        let head_block = gdn_head_block(hd);

        // 1. Input projections (GGUF-layout GEMV each).
        let qkv_blocks = (conv_dim as u32).div_ceil(threads);
        unsafe {
            k.matvec.clone().launch(
                gdn_launch_config(qkv_blocks, threads),
                (w.attn_qkv.buf(), x, scratch.qkv.buf_mut(), hidden as u32, conv_dim as u32),
            )
        }
        .map_err(|e| format!("GDN attn_qkv GEMV launch failed: {e}"))?;

        let gate_blocks = (value_dim as u32).div_ceil(threads);
        unsafe {
            k.matvec.clone().launch(
                gdn_launch_config(gate_blocks, threads),
                (w.attn_gate.buf(), x, scratch.z.buf_mut(), hidden as u32, value_dim as u32),
            )
        }
        .map_err(|e| format!("GDN attn_gate GEMV launch failed: {e}"))?;

        let head_blocks = (h_v as u32).div_ceil(threads);
        unsafe {
            k.matvec.clone().launch(
                gdn_launch_config(head_blocks, threads),
                (w.ssm_beta.buf(), x, scratch.beta_raw.buf_mut(), hidden as u32, h_v as u32),
            )
        }
        .map_err(|e| format!("GDN ssm_beta GEMV launch failed: {e}"))?;

        unsafe {
            k.matvec.clone().launch(
                gdn_launch_config(head_blocks, threads),
                (w.ssm_alpha.buf(), x, scratch.alpha_raw.buf_mut(), hidden as u32, h_v as u32),
            )
        }
        .map_err(|e| format!("GDN ssm_alpha GEMV launch failed: {e}"))?;

        // 2. Causal conv1d + SiLU + window advance.
        let conv_blocks = (conv_dim as u32).div_ceil(threads);
        unsafe {
            k.conv.clone().launch(
                gdn_launch_config(conv_blocks, threads),
                (
                    scratch.qkv.buf(),
                    w.ssm_conv1d.buf(),
                    state.conv_state.buf_mut(),
                    scratch.conv.buf_mut(),
                    conv_dim as u32,
                    cfg.conv_kernel_size as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN conv1d launch failed: {e}"))?;

        // 3. L2-normalize q (scaled by 1/sqrt(head_dim)) then k (unscaled).
        let q_scale = 1.0f32 / (hd as f32).sqrt();
        unsafe {
            k.l2_norm.clone().launch(
                gdn_launch_config(cfg.num_k_heads as u32, head_block),
                (scratch.conv.buf_mut(), 0u32, hd as u32, cfg.eps, q_scale),
            )
        }
        .map_err(|e| format!("GDN q L2-norm launch failed: {e}"))?;
        unsafe {
            k.l2_norm.clone().launch(
                gdn_launch_config(cfg.num_k_heads as u32, head_block),
                (scratch.conv.buf_mut(), key_dim as u32, hd as u32, cfg.eps, 1.0f32),
            )
        }
        .map_err(|e| format!("GDN k L2-norm launch failed: {e}"))?;

        // 4. beta / decay gates.
        unsafe {
            k.gates.clone().launch(
                gdn_launch_config((h_v as u32).div_ceil(64), 64),
                (
                    scratch.alpha_raw.buf(),
                    scratch.beta_raw.buf(),
                    w.ssm_dt.buf(),
                    w.ssm_a.buf(),
                    scratch.decay.buf_mut(),
                    scratch.beta.buf_mut(),
                    h_v as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN gates launch failed: {e}"))?;

        // 5. Delta-rule state update.
        unsafe {
            k.delta.clone().launch(
                gdn_launch_config(h_v as u32, head_block),
                (
                    state.recurrent.buf_mut(),
                    scratch.conv.buf(),
                    0u32,
                    key_dim as u32,
                    (2 * key_dim) as u32,
                    scratch.beta.buf(),
                    scratch.decay.buf(),
                    scratch.o.buf_mut(),
                    hd as u32,
                    cfg.num_k_heads as u32,
                    h_v as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN delta-rule launch failed: {e}"))?;

        // 6. Gated RMSNorm output: y = RMSNorm(o) * silu(z).
        unsafe {
            k.gated_norm.clone().launch(
                gdn_launch_config(h_v as u32, head_block),
                (
                    scratch.o.buf(),
                    scratch.z.buf(),
                    w.ssm_norm.buf(),
                    scratch.y.buf_mut(),
                    hd as u32,
                    cfg.eps,
                ),
            )
        }
        .map_err(|e| format!("GDN gated-norm launch failed: {e}"))?;

        // 7. Output projection back to hidden_size.
        let out_blocks = (hidden as u32).div_ceil(threads);
        unsafe {
            k.matvec.clone().launch(
                gdn_launch_config(out_blocks, threads),
                (w.ssm_out.buf(), scratch.y.buf(), out.buf_mut(), value_dim as u32, hidden as u32),
            )
        }
        .map_err(|e| format!("GDN ssm_out GEMV launch failed: {e}"))?;

        Ok(())
    }

    /// Convenience full round trip over host slices: uploads weights/state,
    /// runs [`step_resident`](Self::step_resident) once, synchronizes once,
    /// downloads the output and the advanced state, and returns the output.
    ///
    /// This is the parity gate / non-resident entry point; the real per-token
    /// decode path (Phase 21.15.4) should hold `GatedDeltaNetDeviceWeights`/
    /// `GatedDeltaNetDeviceState` resident and call `step_resident` directly,
    /// not this method. Falls back to the host reference if NVRTC failed.
    pub fn step(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetWeights<'_>,
        x: &[f32],
        state: &mut GatedDeltaNetState,
    ) -> Result<Vec<f32>, String> {
        if !self.has_gpu() {
            eprintln!("WARNING: Gated DeltaNet GPU path unavailable, falling back to host reference");
            return step(cfg, w, x, state);
        }
        cfg.validate()?;
        if x.len() != cfg.hidden_size {
            return Err(format!("x.len()={} != hidden_size={}", x.len(), cfg.hidden_size));
        }

        let dev_w = GatedDeltaNetDeviceWeights::from_host(&self.device, cfg, w)?;
        let mut dev_state = GatedDeltaNetDeviceState::from_host(&self.device, cfg, state)?;
        let dev_x = self
            .device
            .htod_sync_copy(x)
            .map_err(|e| format!("GDN upload x failed: {e}"))?;
        let mut scratch = GatedDeltaNetScratch::new(&self.device, cfg)?;
        let mut out = DeviceTensor::zeros(&self.device, vec![cfg.hidden_size])?;

        self.step_resident(cfg, &dev_w, &dev_x, &mut dev_state, &mut scratch, &mut out)?;
        self.device
            .synchronize()
            .map_err(|e| format!("GDN synchronize failed: {e}"))?;

        let result = out.to_host(&self.device)?;
        *state = dev_state.to_host(&self.device)?;
        Ok(result)
    }

    /// Device-resident *chunked* prefill step (Phase 21.16.2): processes a
    /// whole chunk of `t_count` tokens through one Gated DeltaNet mixer in a
    /// single parallel pass, advancing `state` to the end of the chunk.
    ///
    /// `xs` is row-major `(t_count, hidden_size)`; `out` is written the same
    /// shape. The projections are batched into one cuBLAS GEMM per weight
    /// (qkv/gate/beta/alpha at the front, `ssm_out` at the end) — the dominant
    /// win, since each layer's ~42 MB of f32 weights are read once for the
    /// chunk instead of once per token. The genuinely chunked pieces run as
    /// the `gdn_chunk_*` kernels: batched causal conv, cumulative log-decay,
    /// and the lower-triangular delta solve + read + chunk-end state update.
    ///
    /// Mathematically identical to `t_count` sequential
    /// [`step_resident`](Self::step_resident) calls; numerically close
    /// (different summation order), verified by this module's tests. No
    /// allocation, no host round trip, no synchronize — the caller syncs once
    /// per chain, exactly like the decode path.
    /// Phase 21.20: the four batched input projections of a chunk (step 1 of
    /// [`chunk_resident`](Self::chunk_resident)), split out so a multi-slot
    /// caller can run them **once across every slot's rows** (one GEMM per
    /// weight for the whole batch) and then only the genuinely recurrent
    /// kernels per slot. Writes `scratch.qkv`/`z`/`beta_raw`/`alpha_raw` rows
    /// `0..t_count`.
    pub fn chunk_project_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        xs: &CudaSlice<f32>,
        t_count: usize,
        scratch: &mut GatedDeltaNetChunkScratch,
    ) -> Result<(), String> {
        cfg.validate()?;
        if t_count == 0 {
            return Err("GatedDeltaNet chunk_project_resident requires t_count >= 1".to_string());
        }
        if t_count > scratch.max_chunk {
            return Err(format!(
                "t_count={t_count} exceeds GatedDeltaNetChunkScratch max_chunk={}",
                scratch.max_chunk
            ));
        }
        let gemm = self.gemm.as_ref().ok_or(
            "Gated DeltaNet chunked path requires a cuBLAS handle (GemmKernel init failed at construction)",
        )?;
        let hidden = cfg.hidden_size;
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let h_v = cfg.num_v_heads;
        if xs.len() < t_count * hidden {
            return Err(format!("xs.len()={} < t_count*hidden={}", xs.len(), t_count * hidden));
        }
        gemm.forward_resident(xs, t_count, hidden, w.attn_qkv.buf(), conv_dim, hidden, scratch.qkv.buf_mut())?;
        gemm.forward_resident(xs, t_count, hidden, w.attn_gate.buf(), value_dim, hidden, scratch.z.buf_mut())?;
        gemm.forward_resident(xs, t_count, hidden, w.ssm_beta.buf(), h_v, hidden, scratch.beta_raw.buf_mut())?;
        gemm.forward_resident(xs, t_count, hidden, w.ssm_alpha.buf(), h_v, hidden, scratch.alpha_raw.buf_mut())?;
        Ok(())
    }

    /// Phase 21.20: the genuinely recurrent half of a chunk (steps 2-6 of
    /// [`chunk_resident`](Self::chunk_resident)) — gates, conv, L2, delta
    /// solve and gated-norm — operating on rows `row_start..row_start+t_count`
    /// of the already-projected `scratch` buffers (see
    /// [`chunk_project_resident`](Self::chunk_project_resident)) with one
    /// per-sequence `state`. Result lands in `scratch.y` rows
    /// `row_start..row_start+t_count`; the caller batches the output
    /// projection ([`chunk_project_output_resident`](Self::chunk_project_output_resident)).
    /// Phase 21.21: the gates stage (`beta`, `logdecay`) of a chunk — purely
    /// elementwise over `(row, num_v_heads)`, so a multi-slot caller can run it
    /// **once over every active row** instead of once per slot. Reads
    /// `scratch.alpha_raw`/`beta_raw` rows `row_start..row_start+t_count`,
    /// writes `scratch.logdecay`/`beta`.
    pub fn chunk_gates_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_gates_resident")?;
        let h_v = cfg.num_v_heads;
        let hv_a = row_start * h_v;
        let hv_b = (row_start + t_count) * h_v;
        let alpha_raw = scratch.alpha_raw.buf().slice(hv_a..hv_b);
        let beta_raw = scratch.beta_raw.buf().slice(hv_a..hv_b);
        let mut logdecay = scratch.logdecay.buf_mut().slice_mut(hv_a..hv_b);
        let mut beta = scratch.beta.buf_mut().slice_mut(hv_a..hv_b);
        unsafe {
            k.chunk_gates.clone().launch(
                gdn_launch_config((t_count * h_v) as u32, 256u32),
                (
                    &alpha_raw,
                    &beta_raw,
                    w.ssm_dt.buf(),
                    w.ssm_a.buf(),
                    &mut logdecay,
                    &mut beta,
                    t_count as u32,
                    h_v as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN chunk gates launch failed: {e}"))?;
        Ok(())
    }

    /// Phase 21.21: the chunk-scoped, state-touching stages that are **not**
    /// elementwise — per-head cumulative log-decay, causal conv and the conv
    /// window advance — still run once per sequence (each chunk's prefix sum
    /// and conv window reset at the chunk boundary), so a multi-slot caller
    /// keeps one call per slot. Batched internally into 3 launches.
    ///
    /// Phase 21.22 tried fusing these 3 launches (`chunk_cumsum` +
    /// `chunk_conv` + `chunk_conv_state`) into one kernel: a diagnostic
    /// per-op profiler (`gdn_recurrent_op_profile`, forced-sync-after-every-op
    /// mode) showed a uniform ~0.5-0.6ms per launch regardless of real
    /// compute size, suggesting launch overhead dominated. It didn't — a
    /// same-instance, same-session A/B on the real, unsynced production path
    /// (`concurrent_throughput_benchmark`) measured the fused kernel
    /// 2-9% *slower* at every concurrency level (1/2/4/8/16 slots), most
    /// likely because the larger fused kernel body increases register
    /// pressure/lowers occupancy, and because the original 3 small kernels
    /// can already pipeline back-to-back on the GPU's command queue without
    /// waiting for each other in real (non-synced) use — exactly the
    /// pipelining the diagnostic tool's forced per-op `synchronize()`
    /// destroys, which is why its numbers were misleading. Reverted; kept as
    /// a documented honest negative, matching this project's precedent
    /// (21.9.2, 21.11, 21.21's Stage B).
    pub fn chunk_conv_phase_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_conv_phase_resident")?;
        check_state(cfg, state)?;
        let conv_dim = cfg.conv_dim();
        let h_v = cfg.num_v_heads;
        let ksize = cfg.conv_kernel_size;
        let qkv_a = row_start * conv_dim;
        let qkv_b = (row_start + t_count) * conv_dim;
        let hv_a = row_start * h_v;
        let hv_b = (row_start + t_count) * h_v;
        let qkv = scratch.qkv.buf().slice(qkv_a..qkv_b);
        let mut logdecay = scratch.logdecay.buf_mut().slice_mut(hv_a..hv_b);
        let mut c = scratch.c.buf_mut().slice_mut(hv_a..hv_b);
        let mut conv = scratch.conv.buf_mut().slice_mut(qkv_a..qkv_b);

        unsafe {
            k.chunk_cumsum.clone().launch(
                LaunchConfig { grid_dim: (h_v as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                (&mut logdecay, &mut c, t_count as u32, h_v as u32),
            )
        }
        .map_err(|e| format!("GDN chunk cumsum launch failed: {e}"))?;

        unsafe {
            k.chunk_conv.clone().launch(
                gdn_launch_config((t_count * conv_dim) as u32, 256),
                (
                    &qkv,
                    w.ssm_conv1d.buf(),
                    state.conv_state.buf(),
                    &mut conv,
                    t_count as u32,
                    conv_dim as u32,
                    ksize as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN chunk conv launch failed: {e}"))?;

        unsafe {
            k.chunk_conv_state.clone().launch(
                gdn_launch_config(conv_dim as u32, 256),
                (
                    &qkv,
                    state.conv_state.buf(),
                    scratch.conv_state_tmp.buf_mut(),
                    t_count as u32,
                    conv_dim as u32,
                    ksize as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN chunk conv-state launch failed: {e}"))?;
        self.device
            .dtod_copy(scratch.conv_state_tmp.buf(), state.conv_state.buf_mut())
            .map_err(|e| format!("GDN chunk conv-state copy failed: {e}"))?;
        Ok(())
    }

    /// Timed sibling of [`Self::chunk_conv_phase_resident`] (Phase 21.22,
    /// profiling-first per `NEXT_STEPS.md`'s proposal): identical math, but
    /// each of the 4 sub-steps (`chunk_cumsum`, `chunk_conv`,
    /// `chunk_conv_state`, and the state-update `dtod_copy`) is individually
    /// timed via `on_op`, following the exact `step!`-macro pattern
    /// `dispatch.rs`'s `forward_layer_resident_paged_timed` already
    /// established for the dense layer path. `sync_each_op` mirrors that
    /// tool's tradeoff: `true` gives a per-op breakdown (but — see
    /// `chunk_conv_phase_resident`'s doc comment — the forced sync inflates
    /// these numbers and makes launch-count reduction look like a bigger win
    /// than it is on the real, unsynced production path); `false` times the
    /// whole call with one synchronize instead.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_conv_phase_resident_timed(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
        sync_each_op: bool,
        mut on_op: impl FnMut(&'static str, std::time::Duration),
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_conv_phase_resident_timed")?;
        check_state(cfg, state)?;
        let conv_dim = cfg.conv_dim();
        let h_v = cfg.num_v_heads;
        let ksize = cfg.conv_kernel_size;
        let qkv_a = row_start * conv_dim;
        let qkv_b = (row_start + t_count) * conv_dim;
        let hv_a = row_start * h_v;
        let hv_b = (row_start + t_count) * h_v;
        let qkv = scratch.qkv.buf().slice(qkv_a..qkv_b);
        let mut logdecay = scratch.logdecay.buf_mut().slice_mut(hv_a..hv_b);
        let mut c = scratch.c.buf_mut().slice_mut(hv_a..hv_b);
        let mut conv = scratch.conv.buf_mut().slice_mut(qkv_a..qkv_b);
        let device = self.device.clone();

        macro_rules! step {
            ($label:literal, $body:expr) => {{
                let t0 = std::time::Instant::now();
                $body;
                if sync_each_op {
                    device.synchronize().map_err(|e| format!("synchronize after {}: {e}", $label))?;
                }
                on_op($label, t0.elapsed());
            }};
        }

        step!(
            "gdn_chunk_cumsum",
            unsafe {
                k.chunk_cumsum.clone().launch(
                    LaunchConfig { grid_dim: (h_v as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                    (&mut logdecay, &mut c, t_count as u32, h_v as u32),
                )
            }
            .map_err(|e| format!("GDN chunk cumsum launch failed: {e}"))?
        );

        step!(
            "gdn_chunk_conv",
            unsafe {
                k.chunk_conv.clone().launch(
                    gdn_launch_config((t_count * conv_dim) as u32, 256),
                    (
                        &qkv,
                        w.ssm_conv1d.buf(),
                        state.conv_state.buf(),
                        &mut conv,
                        t_count as u32,
                        conv_dim as u32,
                        ksize as u32,
                    ),
                )
            }
            .map_err(|e| format!("GDN chunk conv launch failed: {e}"))?
        );

        step!(
            "gdn_chunk_conv_state",
            unsafe {
                k.chunk_conv_state.clone().launch(
                    gdn_launch_config(conv_dim as u32, 256),
                    (
                        &qkv,
                        state.conv_state.buf(),
                        scratch.conv_state_tmp.buf_mut(),
                        t_count as u32,
                        conv_dim as u32,
                        ksize as u32,
                    ),
                )
            }
            .map_err(|e| format!("GDN chunk conv-state launch failed: {e}"))?
        );

        step!(
            "gdn_conv_state_dtod_copy",
            self.device
                .dtod_copy(scratch.conv_state_tmp.buf(), state.conv_state.buf_mut())
                .map_err(|e| format!("GDN chunk conv-state copy failed: {e}"))?
        );

        Ok(())
    }

    /// Phase 21.21: elementwise L2-normalize q (scaled) then k (unscaled) in
    /// place over `scratch.conv` rows `row_start..row_start+t_count` — batched
    /// across slots by the caller.
    pub fn chunk_l2_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_l2_resident")?;
        let hd = cfg.head_dim;
        let conv_dim = cfg.conv_dim();
        let h_k = cfg.num_k_heads;
        let key_dim = cfg.key_dim();
        let conv_a = row_start * conv_dim;
        let conv_b = (row_start + t_count) * conv_dim;
        let mut conv = scratch.conv.buf_mut().slice_mut(conv_a..conv_b);
        let head_block = gdn_head_block(hd);
        let q_scale = 1.0f32 / (hd as f32).sqrt();
        let l2_grid = (t_count * h_k) as u32;
        unsafe {
            k.chunk_l2.clone().launch(
                gdn_launch_config(l2_grid, head_block),
                (&mut conv, h_k as u32, hd as u32, 0u32, conv_dim as u32, cfg.eps, q_scale),
            )
        }
        .map_err(|e| format!("GDN chunk q L2 launch failed: {e}"))?;
        unsafe {
            k.chunk_l2.clone().launch(
                gdn_launch_config(l2_grid, head_block),
                (&mut conv, h_k as u32, hd as u32, key_dim as u32, conv_dim as u32, cfg.eps, 1.0f32),
            )
        }
        .map_err(|e| format!("GDN chunk k L2 launch failed: {e}"))?;
        Ok(())
    }

    /// Phase 21.21: one sequence's chunked delta solve + output read +
    /// chunk-end state update. Reads `scratch.conv` (L2-normalized),
    /// `beta`/`c` for rows `row_start..row_start+t_count`; writes
    /// `scratch.d`/`o` and advances `state.recurrent`. Still per sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_delta_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_delta_resident")?;
        if state.recurrent.len() != cfg.recurrent_len() {
            return Err(format!(
                "state.recurrent.len()={} != expected {}",
                state.recurrent.len(),
                cfg.recurrent_len()
            ));
        }
        let hd = cfg.head_dim;
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();
        let value_dim = cfg.value_dim();
        let h_v = cfg.num_v_heads;
        let conv_a = row_start * conv_dim;
        let conv_b = (row_start + t_count) * conv_dim;
        let hv_a = row_start * h_v;
        let hv_b = (row_start + t_count) * h_v;
        let v_a = row_start * value_dim;
        let v_b = (row_start + t_count) * value_dim;
        let d_a = row_start * h_v * hd;
        let d_b = (row_start + t_count) * h_v * hd;
        let mut conv = scratch.conv.buf_mut().slice_mut(conv_a..conv_b);
        let mut beta = scratch.beta.buf_mut().slice_mut(hv_a..hv_b);
        let mut c = scratch.c.buf_mut().slice_mut(hv_a..hv_b);
        let mut d = scratch.d.buf_mut().slice_mut(d_a..d_b);
        let mut o = scratch.o.buf_mut().slice_mut(v_a..v_b);
        unsafe {
            k.chunk_delta.clone().launch(
                gdn_launch_config(h_v as u32, gdn_head_block(hd)),
                (
                    state.recurrent.buf_mut(),
                    &mut conv,
                    key_dim as u32,
                    conv_dim as u32,
                    &mut beta,
                    &mut c,
                    &mut d,
                    &mut o,
                    t_count as u32,
                    hd as u32,
                    cfg.num_k_heads as u32,
                    h_v as u32,
                ),
            )
        }
        .map_err(|e| format!("GDN chunk delta launch failed: {e}"))?;
        Ok(())
    }

    /// Timed sibling of [`Self::chunk_delta_resident`] (Phase 21.22) — a
    /// single kernel, so this just wraps it with the same `on_op`/
    /// `sync_each_op` convention as [`Self::chunk_conv_phase_resident_timed`]
    /// for a uniform profiling harness across the whole per-slot residual
    /// chain.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_delta_resident_timed(
        &self,
        cfg: &GatedDeltaNetConfig,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
        sync_each_op: bool,
        mut on_op: impl FnMut(&'static str, std::time::Duration),
    ) -> Result<(), String> {
        let device = self.device.clone();
        let t0 = std::time::Instant::now();
        self.chunk_delta_resident(cfg, scratch, row_start, t_count, state)?;
        if sync_each_op {
            device.synchronize().map_err(|e| format!("synchronize after gdn_chunk_delta: {e}"))?;
        }
        on_op("gdn_chunk_delta", t0.elapsed());
        Ok(())
    }

    /// Phase 21.21: elementwise gated RMSNorm over `scratch.o`/`z` rows
    /// `row_start..row_start+t_count` into `scratch.y` — batched across slots
    /// by the caller.
    pub fn chunk_gated_norm_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
    ) -> Result<(), String> {
        let k = self.kernels.as_ref().ok_or(
            "Gated DeltaNet GPU path requires a compiled CUDA kernel (NVRTC compile failed at construction)",
        )?;
        check_row_range(scratch, row_start, t_count, "chunk_gated_norm_resident")?;
        let hd = cfg.head_dim;
        let h_v = cfg.num_v_heads;
        let value_dim = cfg.value_dim();
        let v_a = row_start * value_dim;
        let v_b = (row_start + t_count) * value_dim;
        let z = scratch.z.buf().slice(v_a..v_b);
        let mut o = scratch.o.buf_mut().slice_mut(v_a..v_b);
        let mut y = scratch.y.buf_mut().slice_mut(v_a..v_b);
        unsafe {
            k.chunk_gated_norm.clone().launch(
                gdn_launch_config((t_count * h_v) as u32, gdn_head_block(hd)),
                (&mut o, &z, w.ssm_norm.buf(), &mut y, h_v as u32, hd as u32, cfg.eps),
            )
        }
        .map_err(|e| format!("GDN chunk gated-norm launch failed: {e}"))?;
        Ok(())
    }

    /// The genuinely recurrent half of a chunk — gates, cumsum, conv,
    /// conv-state, L2, delta solve and gated-norm — operating on rows
    /// `row_start..row_start+t_count` of the already-projected `scratch`
    /// buffers with one per-sequence `state`, all in the original order.
    /// Phase 21.21's multi-slot orchestrator interleaves the elementwise
    /// stages across every slot instead of calling this; this composition is
    /// what the single-sequence [`chunk_resident`](Self::chunk_resident) (and
    /// the parity tests) use.
    pub fn chunk_recurrent_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        row_start: usize,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
    ) -> Result<(), String> {
        self.chunk_gates_resident(cfg, w, scratch, row_start, t_count)?;
        self.chunk_conv_phase_resident(cfg, w, scratch, row_start, t_count, state)?;
        self.chunk_l2_resident(cfg, scratch, row_start, t_count)?;
        self.chunk_delta_resident(cfg, scratch, row_start, t_count, state)?;
        self.chunk_gated_norm_resident(cfg, w, scratch, row_start, t_count)
    }

    /// Phase 21.20: the batched output projection (step 7 of
    /// [`chunk_resident`](Self::chunk_resident)) — `scratch.y` rows
    /// `0..t_count` to `out`. Split out so a multi-slot caller can run it once
    /// across every slot's rows.
    pub fn chunk_project_output_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        scratch: &mut GatedDeltaNetChunkScratch,
        t_count: usize,
        out: &mut DeviceTensor,
    ) -> Result<(), String> {
        cfg.validate()?;
        if t_count > scratch.max_chunk {
            return Err(format!(
                "t_count={t_count} exceeds GatedDeltaNetChunkScratch max_chunk={}",
                scratch.max_chunk
            ));
        }
        let gemm = self.gemm.as_ref().ok_or(
            "Gated DeltaNet chunked path requires a cuBLAS handle (GemmKernel init failed at construction)",
        )?;
        if out.len() < t_count * cfg.hidden_size {
            return Err(format!(
                "out.len()={} < t_count*hidden={}",
                out.len(),
                t_count * cfg.hidden_size
            ));
        }
        gemm.forward_resident(
            scratch.y.buf(),
            t_count,
            cfg.value_dim(),
            w.ssm_out.buf(),
            cfg.hidden_size,
            cfg.value_dim(),
            out.buf_mut(),
        )?;
        Ok(())
    }

    /// Mathematically identical to `t_count` sequential
    /// [`step_resident`](Self::step_resident) calls; numerically close
    /// (different summation order), verified by this module's tests. No
    /// allocation, no host round trip, no synchronize — the caller syncs once
    /// per chain, exactly like the decode path.
    ///
    /// Phase 21.20 split this into
    /// [`chunk_project_resident`](Self::chunk_project_resident) →
    /// [`chunk_recurrent_resident`](Self::chunk_recurrent_resident) →
    /// [`chunk_project_output_resident`](Self::chunk_project_output_resident)
    /// so the multi-slot server can batch the row-wise projections/FFN across
    /// slots and keep only the recurrent half per slot; this entry point is
    /// the branch-free sequential composition (behavior unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_resident(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetDeviceWeights,
        xs: &CudaSlice<f32>,
        t_count: usize,
        state: &mut GatedDeltaNetDeviceState,
        scratch: &mut GatedDeltaNetChunkScratch,
        out: &mut DeviceTensor,
    ) -> Result<(), String> {
        self.chunk_project_resident(cfg, w, xs, t_count, scratch)?;
        self.chunk_recurrent_resident(cfg, w, scratch, 0, t_count, state)?;
        self.chunk_project_output_resident(cfg, w, scratch, t_count, out)
    }

    /// Convenience full round trip over host slices for the chunked path:
    /// uploads weights/state, runs [`chunk_resident`](Self::chunk_resident)
    /// once, synchronizes once, downloads the output and the advanced state.
    /// This is the 21.16.2 parity-gate / non-resident entry point. Falls back
    /// to the host [`chunked`] reference when the GPU path is unavailable.
    pub fn chunk(
        &self,
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetWeights<'_>,
        xs: &[f32],
        state: &mut GatedDeltaNetState,
    ) -> Result<Vec<f32>, String> {
        if !self.has_chunk_gpu() {
            eprintln!("WARNING: Gated DeltaNet chunked GPU path unavailable, falling back to host chunked reference");
            return chunked(cfg, w, xs, state);
        }
        cfg.validate()?;
        if xs.len() % cfg.hidden_size != 0 {
            return Err(format!("xs.len()={} not a multiple of hidden_size={}", xs.len(), cfg.hidden_size));
        }
        let t_count = xs.len() / cfg.hidden_size;
        if t_count == 0 {
            return Ok(Vec::new());
        }

        let dev_w = GatedDeltaNetDeviceWeights::from_host(&self.device, cfg, w)?;
        let mut dev_state = GatedDeltaNetDeviceState::from_host(&self.device, cfg, state)?;
        let dev_x = self.device.htod_sync_copy(xs).map_err(|e| format!("GDN upload xs failed: {e}"))?;
        let mut scratch = GatedDeltaNetChunkScratch::new(&self.device, cfg, t_count)?;
        let mut out = DeviceTensor::zeros(&self.device, vec![t_count, cfg.hidden_size])?;

        self.chunk_resident(cfg, &dev_w, &dev_x, t_count, &mut dev_state, &mut scratch, &mut out)?;
        self.device
            .synchronize()
            .map_err(|e| format!("GDN synchronize failed: {e}"))?;

        let result = out.to_host(&self.device)?;
        *state = dev_state.to_host(&self.device)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_get_device() -> Option<Arc<CudaDevice>> {
        std::panic::catch_unwind(|| CudaDevice::new(0).ok()).ok().flatten()
    }

    fn assert_close(got: f32, want: f32, tol: f32, what: &str) {
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn test_softplus_sigmoid_silu_hand_computed() {
        // softplus(0) = ln 2; threshold branch at >20 returns input unchanged.
        assert_close(softplus(0.0), std::f32::consts::LN_2, 1e-6, "softplus(0)");
        assert_close(softplus(25.0), 25.0, 0.0, "softplus(25)");
        assert_close(softplus(-10.0), (1.0 + (-10.0f32).exp()).ln(), 1e-7, "softplus(-10)");

        assert_close(sigmoid(0.0), 0.5, 0.0, "sigmoid(0)");
        assert_close(sigmoid(1.0), 1.0 / (1.0 + (-1.0f32).exp()), 1e-7, "sigmoid(1)");

        assert_close(silu(0.0), 0.0, 0.0, "silu(0)");
        // silu(2) = 2 * sigmoid(2) = 2 / (1 + e^-2)
        assert_close(silu(2.0), 2.0 / (1.0 + (-2.0f32).exp()), 1e-7, "silu(2)");
    }

    #[test]
    fn test_l2_normalize_hand_computed() {
        // [3, 4] has norm 5; with eps=0 -> [0.6, 0.8].
        let mut v = [3.0f32, 4.0];
        l2_normalize(&mut v, 0.0);
        assert_close(v[0], 0.6, 1e-6, "v[0]");
        assert_close(v[1], 0.8, 1e-6, "v[1]");

        // A large eps keeps the vector small rather than dividing by ~0.
        let mut tiny = [0.0f32, 0.0];
        l2_normalize(&mut tiny, 1e-6);
        assert!(tiny.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_conv1d_step_hand_computed() {
        // conv_dim = 3, K = 3 (2 history rows).
        // history rows: [1,2,3], [4,5,6]; current: [7,8,9].
        let conv_state = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qkv = [7.0f32, 8.0, 9.0];
        let conv_dim = 3;
        let k = 3;

        // All-ones kernel: out[c] = sum over window.
        let ones = [1.0f32; 9];
        let mut out = [0.0f32; 3];
        conv1d_step(&conv_state, &qkv, &ones, k, conv_dim, &mut out);
        assert_eq!(out, [12.0, 15.0, 18.0]);

        // kernel[k, c] = k+1: out[c] = 1*hist0 + 2*hist1 + 3*cur.
        let mut w = [0.0f32; 9];
        for c in 0..conv_dim {
            for kk in 0..k {
                w[kk + c * k] = (kk + 1) as f32;
            }
        }
        conv1d_step(&conv_state, &qkv, &w, k, conv_dim, &mut out);
        assert_eq!(out, [30.0, 36.0, 42.0]);
    }

    #[test]
    fn test_delta_rule_head_hand_computed() {
        // S0 = [[1,0],[0,2]] (key-major), decay 0.5, k=[1,1], v=[2,3], beta=0.5,
        // q=[1,2] (already scaled). Hand work:
        //   S*decay = [[0.5,0],[0,1]]
        //   kv      = [0.5*1+0*1, 0*1+1*1] = [0.5, 1]
        //   delta   = [(2-0.5)*0.5, (3-1)*0.5] = [0.75, 1.0]
        //   S      += k (x) delta -> [[1.25,1.0],[0.75,2.0]]
        //   o[v]    = [1.25*1+0.75*2, 1.0*1+2.0*2] = [2.75, 5.0]
        let mut s = [1.0f32, 0.0, 0.0, 2.0];
        let q = [1.0f32, 2.0];
        let k = [1.0f32, 1.0];
        let v = [2.0f32, 3.0];
        let mut o = [0.0f32; 2];
        delta_rule_head(&mut s, &q, &k, &v, 0.5, 0.5, &mut o);

        assert_eq!(s, [1.25, 1.0, 0.75, 2.0]);
        assert_eq!(o, [2.75, 5.0]);
    }

    /// Deterministic pseudo-random filler (no external rand dependency).
    fn lcg_fill(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    struct OwnedWeights {
        attn_qkv: Vec<f32>,
        attn_gate: Vec<f32>,
        ssm_beta: Vec<f32>,
        ssm_alpha: Vec<f32>,
        ssm_dt: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_conv1d: Vec<f32>,
        ssm_norm: Vec<f32>,
        ssm_out: Vec<f32>,
    }

    impl OwnedWeights {
        fn new(cfg: &GatedDeltaNetConfig, seed: u32) -> Self {
            let mut s = seed;
            let mut next = |n: usize| {
                s = s.wrapping_add(1);
                lcg_fill(n, s)
            };
            Self {
                attn_qkv: next(cfg.hidden_size * cfg.conv_dim()),
                attn_gate: next(cfg.hidden_size * cfg.value_dim()),
                ssm_beta: next(cfg.hidden_size * cfg.num_v_heads),
                ssm_alpha: next(cfg.hidden_size * cfg.num_v_heads),
                // dt bias small positive; ssm_a negative (a real decay).
                ssm_dt: next(cfg.num_v_heads).iter().map(|x| x * 0.1 + 0.1).collect(),
                ssm_a: next(cfg.num_v_heads).iter().map(|x| -x.abs() - 0.05).collect(),
                ssm_conv1d: next(cfg.conv_kernel_size * cfg.conv_dim()),
                ssm_norm: next(cfg.head_dim).iter().map(|x| x + 1.0).collect(),
                ssm_out: next(cfg.value_dim() * cfg.hidden_size),
            }
        }

        fn view(&self) -> GatedDeltaNetWeights<'_> {
            GatedDeltaNetWeights {
                attn_qkv: &self.attn_qkv,
                attn_gate: &self.attn_gate,
                ssm_beta: &self.ssm_beta,
                ssm_alpha: &self.ssm_alpha,
                ssm_dt: &self.ssm_dt,
                ssm_a: &self.ssm_a,
                ssm_conv1d: &self.ssm_conv1d,
                ssm_norm: &self.ssm_norm,
                ssm_out: &self.ssm_out,
            }
        }
    }

    #[test]
    fn test_step_deterministic_shape_and_state_evolution() {
        let cfg = GatedDeltaNetConfig::new(4, 2, 2, 2, 4);
        let w = OwnedWeights::new(&cfg, 7);
        let x = [0.3f32, -0.2, 0.5, -0.1];

        let mut state = GatedDeltaNetState::zeros(&cfg);
        let out1 = step(&cfg, &w.view(), &x, &mut state).unwrap();
        assert_eq!(out1.len(), cfg.hidden_size);
        assert!(out1.iter().all(|v| v.is_finite()), "output must be finite");

        // Same input again: a fresh state must reproduce out1 exactly.
        let mut fresh = GatedDeltaNetState::zeros(&cfg);
        let out1b = step(&cfg, &w.view(), &x, &mut fresh).unwrap();
        assert_eq!(out1, out1b, "deterministic given equal initial state");

        // Advancing the same state changes the result (conv history + S evolve).
        let out2 = step(&cfg, &w.view(), &x, &mut state).unwrap();
        assert_ne!(out1, out2, "second token must not reproduce the first");

        state.reset();
        let out1c = step(&cfg, &w.view(), &x, &mut state).unwrap();
        assert_eq!(out1, out1c, "reset must restore the initial behaviour");
    }

    #[test]
    fn test_step_with_head_repeat_k_heads_lt_v_heads() {
        // H_k = 2, H_v = 6 exercises the q/k head tiling (h % num_k_heads).
        let cfg = GatedDeltaNetConfig::new(3, 2, 6, 2, 4);
        let w = OwnedWeights::new(&cfg, 11);
        let x = [0.1f32, 0.2, -0.3];
        let mut state = GatedDeltaNetState::zeros(&cfg);
        let out = step(&cfg, &w.view(), &x, &mut state).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_config_and_weight_validation() {
        let cfg = GatedDeltaNetConfig::new(4, 2, 2, 2, 4);
        let w = OwnedWeights::new(&cfg, 3);

        // Wrong x length.
        let mut state = GatedDeltaNetState::zeros(&cfg);
        assert!(step(&cfg, &w.view(), &[0.0; 5], &mut state).is_err());

        // Wrong state length.
        let mut bad_state = GatedDeltaNetState::zeros(&cfg);
        bad_state.recurrent.pop();
        assert!(step(&cfg, &w.view(), &[0.0; 4], &mut bad_state).is_err());

        // head mismatch is rejected by config validation.
        let bad_cfg = GatedDeltaNetConfig::new(4, 3, 2, 2, 4);
        assert!(bad_cfg.validate().is_err());
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Run `t` tokens through the sequential reference, returning the stacked
    /// outputs and the final state — the oracle the chunked path must match.
    fn run_sequential(
        cfg: &GatedDeltaNetConfig,
        w: &GatedDeltaNetWeights,
        xs: &[f32],
    ) -> (Vec<f32>, GatedDeltaNetState) {
        let t = xs.len() / cfg.hidden_size;
        let mut state = GatedDeltaNetState::zeros(cfg);
        let mut out = Vec::with_capacity(xs.len());
        for i in 0..t {
            let x = &xs[i * cfg.hidden_size..(i + 1) * cfg.hidden_size];
            out.extend(step(cfg, w, x, &mut state).expect("sequential step"));
        }
        (out, state)
    }

    /// Phase 21.16.1's correctness gate: the chunked closure form must
    /// reproduce the sequential recurrence's outputs *and* end state for
    /// several chunk lengths, including `T=1` (which collapses exactly to one
    /// `step`) and a chunk longer than the conv window.
    fn assert_chunked_matches_sequential(cfg: &GatedDeltaNetConfig, hidden_tokens: usize, tol: f32) {
        let w = OwnedWeights::new(cfg, 20216);
        let wv = w.view();
        let xs = lcg_fill(hidden_tokens * cfg.hidden_size, 777);

        let (seq_out, seq_state) = run_sequential(cfg, &wv, &xs);

        let mut chunk_state = GatedDeltaNetState::zeros(cfg);
        let chunk_out = chunked(cfg, &wv, &xs, &mut chunk_state).expect("chunked");

        assert_eq!(chunk_out.len(), seq_out.len());
        let out_diff = max_abs_diff(&chunk_out, &seq_out);
        let rec_diff = max_abs_diff(&chunk_state.recurrent, &seq_state.recurrent);
        let conv_diff = max_abs_diff(&chunk_state.conv_state, &seq_state.conv_state);
        eprintln!(
            "chunked vs sequential (T={hidden_tokens}, hidden={}, H={}/{}): out={out_diff:.3e} rec={rec_diff:.3e} conv={conv_diff:.3e}",
            cfg.hidden_size, cfg.num_k_heads, cfg.num_v_heads
        );
        assert!(
            out_diff < tol,
            "chunked output max_diff={out_diff} exceeds tolerance {tol} (T={hidden_tokens})"
        );
        assert!(
            rec_diff < tol,
            "chunked recurrent-state max_diff={rec_diff} exceeds tolerance {tol} (T={hidden_tokens})"
        );
        assert!(
            conv_diff < tol,
            "chunked conv-state max_diff={conv_diff} exceeds tolerance {tol} (T={hidden_tokens})"
        );
    }

    #[test]
    fn test_chunked_matches_sequential_small_t1_t3_t5() {
        // K=4 -> history 3; T=1 collapses to one step, T=3 exercises the
        // conv-window boundary, T=5 crosses it.
        let cfg = GatedDeltaNetConfig::new(4, 2, 2, 2, 4);
        for t in [1usize, 3, 5] {
            assert_chunked_matches_sequential(&cfg, t, 1e-5);
        }
    }

    // `lcg_fill` above is seeded per call; use the existing helper by name.

    #[test]
    fn test_chunked_matches_sequential_real_shapes() {
        // Real Qwen3.5-0.8B DeltaNet shapes. Measured max_diff ~5e-5 (out) /
        // ~8e-7 (recurrent state) at T=16 — well inside this repo's 1e-2 bar.
        let cfg = GatedDeltaNetConfig::new(1024, 16, 16, 128, 4);
        for t in [1usize, 4, 16] {
            assert_chunked_matches_sequential(&cfg, t, 1e-3);
        }
    }

    #[test]
    fn test_chunked_matches_sequential_head_tiling() {
        // H_k = 2, H_v = 6 exercises q/k head tiling in the chunked path.
        let cfg = GatedDeltaNetConfig::new(32, 2, 6, 8, 4);
        for t in [1usize, 4, 6] {
            assert_chunked_matches_sequential(&cfg, t, 1e-4);
        }
    }

    #[test]
    fn test_chunked_preserves_state_across_chunks() {
        // Two chunks of 3 == one chunk of 6 == sequential 6, onto the same state.
        let cfg = GatedDeltaNetConfig::new(5, 1, 1, 3, 4);
        let w = OwnedWeights::new(&cfg, 55);
        let wv = w.view();
        let xs = lcg_fill(6 * cfg.hidden_size, 4242);

        let (seq_out, seq_state) = run_sequential(&cfg, &wv, &xs);

        let mut state = GatedDeltaNetState::zeros(&cfg);
        let mut chunked_out = Vec::new();
        chunked_out.extend(chunked(&cfg, &wv, &xs[..3 * cfg.hidden_size], &mut state).unwrap());
        chunked_out.extend(chunked(&cfg, &wv, &xs[3 * cfg.hidden_size..], &mut state).unwrap());

        assert!(max_abs_diff(&chunked_out, &seq_out) < 1e-5);
        assert!(max_abs_diff(&state.recurrent, &seq_state.recurrent) < 1e-5);
        assert!(max_abs_diff(&state.conv_state, &seq_state.conv_state) < 1e-5);
    }

    #[test]
    fn test_chunked_empty_and_deterministic() {
        let cfg = GatedDeltaNetConfig::new(4, 1, 1, 2, 4);
        let w = OwnedWeights::new(&cfg, 1);
        let wv = w.view();
        let mut state = GatedDeltaNetState::zeros(&cfg);
        assert!(chunked(&cfg, &wv, &[], &mut state).unwrap().is_empty());

        let xs = lcg_fill(4 * cfg.hidden_size, 99);
        let mut s1 = GatedDeltaNetState::zeros(&cfg);
        let mut s2 = GatedDeltaNetState::zeros(&cfg);
        let o1 = chunked(&cfg, &wv, &xs, &mut s1).unwrap();
        let o2 = chunked(&cfg, &wv, &xs, &mut s2).unwrap();
        assert_eq!(o1, o2);
        assert_eq!(s1.recurrent, s2.recurrent);
    }

    /// Phase 21.15.3's real parity gate: the GPU kernel must reproduce the
    /// 21.15.2 host reference on the real Qwen3.5-0.8B shapes (hidden 1024,
    /// H_k = H_v = 16, head_dim 128, K = 4), over several tokens so the
    /// resident conv/recurrent state is actually exercised, within this
    /// repo's 1e-2 tolerance. Skips gracefully without CUDA (same pattern as
    /// every other GPU kernel test here).
    #[test]
    fn test_gpu_step_matches_host_reference_real_shapes() {
        let Some(device) = try_get_device() else {
            eprintln!("Skipping test_gpu_step_matches_host_reference_real_shapes: CUDA unavailable");
            return;
        };
        let kernel = GatedDeltaNetKernel::new(device.clone()).expect("GDN kernel init");
        if !kernel.has_gpu() {
            eprintln!("Skipping test_gpu_step_matches_host_reference_real_shapes: NVRTC unavailable");
            return;
        }

        let cfg = GatedDeltaNetConfig::new(1024, 16, 16, 128, 4);
        let w = OwnedWeights::new(&cfg, 4242);
        let wv = w.view();

        let mut host_state = GatedDeltaNetState::zeros(&cfg);
        let dev_w = kernel.upload_weights(&cfg, &wv).expect("upload weights");
        let mut dev_state = kernel.state_zeros(&cfg).expect("device state zeros");
        let mut scratch = kernel.scratch(&cfg).expect("device scratch");
        let mut dev_out = DeviceTensor::zeros(&device, vec![cfg.hidden_size]).expect("device out");

        let mut worst_out = 0.0f32;
        for t in 0..3u32 {
            let x = lcg_fill(cfg.hidden_size, 9000 + t);
            let host_out = step(&cfg, &wv, &x, &mut host_state).expect("host step");
            let dev_x = device.htod_sync_copy(&x).expect("upload x");
            kernel
                .step_resident(&cfg, &dev_w, &dev_x, &mut dev_state, &mut scratch, &mut dev_out)
                .expect("gpu step_resident");
            device.synchronize().expect("sync");
            let gpu_out = dev_out.to_host(&device).expect("download out");
            let diff = max_abs_diff(&host_out, &gpu_out);
            worst_out = worst_out.max(diff);
            assert!(diff < 1e-2, "token {t}: GPU vs host output max_diff={diff}");
        }

        let gpu_state = dev_state.to_host(&device).expect("download state");
        let conv_diff = max_abs_diff(&host_state.conv_state, &gpu_state.conv_state);
        let rec_diff = max_abs_diff(&host_state.recurrent, &gpu_state.recurrent);
        eprintln!(
            "GDN real-shape parity: worst_out={worst_out}, conv_state={conv_diff}, recurrent={rec_diff}"
        );
        assert!(conv_diff < 1e-2, "conv_state diverged: {conv_diff}");
        assert!(rec_diff < 1e-2, "recurrent state diverged: {rec_diff}");
    }

    /// Small-shape parity that exercises the `H_k < H_v` q/k head tiling
    /// (`h % num_k_heads`) and a non-power-of-two-friendly head_dim, on top
    /// of the same resident-state round trip.
    #[test]
    fn test_gpu_step_matches_host_reference_head_tiling() {
        let Some(device) = try_get_device() else {
            eprintln!("Skipping test_gpu_step_matches_host_reference_head_tiling: CUDA unavailable");
            return;
        };
        let kernel = GatedDeltaNetKernel::new(device.clone()).expect("GDN kernel init");
        if !kernel.has_gpu() {
            eprintln!("Skipping test_gpu_step_matches_host_reference_head_tiling: NVRTC unavailable");
            return;
        }

        let cfg = GatedDeltaNetConfig::new(8, 2, 6, 4, 4);
        let w = OwnedWeights::new(&cfg, 77);
        let wv = w.view();

        let mut host_state = GatedDeltaNetState::zeros(&cfg);
        let dev_w = kernel.upload_weights(&cfg, &wv).expect("upload weights");
        let mut dev_state = kernel.state_zeros(&cfg).expect("device state zeros");
        let mut scratch = kernel.scratch(&cfg).expect("device scratch");
        let mut dev_out = DeviceTensor::zeros(&device, vec![cfg.hidden_size]).expect("device out");

        for t in 0..4u32 {
            let x = lcg_fill(cfg.hidden_size, 500 + t);
            let host_out = step(&cfg, &wv, &x, &mut host_state).expect("host step");
            let dev_x = device.htod_sync_copy(&x).expect("upload x");
            kernel
                .step_resident(&cfg, &dev_w, &dev_x, &mut dev_state, &mut scratch, &mut dev_out)
                .expect("gpu step_resident");
            device.synchronize().expect("sync");
            let gpu_out = dev_out.to_host(&device).expect("download out");
            let diff = max_abs_diff(&host_out, &gpu_out);
            assert!(diff < 1e-2, "token {t}: GPU vs host output max_diff={diff}");
        }

        // A reset must send the GPU back to the same first-token result.
        dev_state.reset(&device).expect("device state reset");
        let x = lcg_fill(cfg.hidden_size, 500);
        let mut fresh_host = GatedDeltaNetState::zeros(&cfg);
        let host_first = step(&cfg, &wv, &x, &mut fresh_host).expect("host step");
        let dev_x = device.htod_sync_copy(&x).expect("upload x");
        kernel
            .step_resident(&cfg, &dev_w, &dev_x, &mut dev_state, &mut scratch, &mut dev_out)
            .expect("gpu step_resident after reset");
        device.synchronize().expect("sync");
        let gpu_first = dev_out.to_host(&device).expect("download out");
        assert!(
            max_abs_diff(&host_first, &gpu_first) < 1e-2,
            "GPU reset did not restore the first-token result"
        );
    }

    /// Phase 21.16.2's real parity gate: the GPU chunked path must reproduce
    /// the 21.16.1 host `chunked` reference on the real Qwen3.5-0.8B shapes
    /// (hidden 1024, H_k = H_v = 16, head_dim 128, K = 4) at several chunk
    /// lengths, within this repo's 1e-2 tolerance. Skips gracefully without
    /// CUDA/NVRTC/cuBLAS.
    #[test]
    fn test_gpu_chunk_matches_host_chunked_real_shapes() {
        let Some(device) = try_get_device() else {
            eprintln!("Skipping test_gpu_chunk_matches_host_chunked_real_shapes: CUDA unavailable");
            return;
        };
        let kernel = GatedDeltaNetKernel::new(device.clone()).expect("GDN kernel init");
        if !kernel.has_chunk_gpu() {
            eprintln!("Skipping test_gpu_chunk_matches_host_chunked_real_shapes: NVRTC/cuBLAS unavailable");
            return;
        }

        let cfg = GatedDeltaNetConfig::new(1024, 16, 16, 128, 4);
        let w = OwnedWeights::new(&cfg, 20216);
        let wv = w.view();

        for t in [1usize, 4, 16] {
            let xs = lcg_fill(t * cfg.hidden_size, 777 + t as u32);
            let mut host_state = GatedDeltaNetState::zeros(&cfg);
            let host_out = chunked(&cfg, &wv, &xs, &mut host_state).expect("host chunked");

            let mut gpu_state = GatedDeltaNetState::zeros(&cfg);
            let gpu_out = kernel.chunk(&cfg, &wv, &xs, &mut gpu_state).expect("gpu chunk");

            let out_diff = max_abs_diff(&host_out, &gpu_out);
            let rec_diff = max_abs_diff(&gpu_state.recurrent, &host_state.recurrent);
            let conv_diff = max_abs_diff(&gpu_state.conv_state, &host_state.conv_state);
            eprintln!("GDN chunk parity T={t}: out={out_diff:.3e} rec={rec_diff:.3e} conv={conv_diff:.3e}");
            assert!(out_diff < 1e-2, "T={t}: GPU chunk vs host chunked out max_diff={out_diff}");
            assert!(rec_diff < 1e-2, "T={t}: recurrent-state max_diff={rec_diff}");
            assert!(conv_diff < 1e-2, "T={t}: conv-state max_diff={conv_diff}");
        }
    }

    /// The `H_k < H_v` q/k head tiling (`h % H_k`), plus a chunk that crosses
    /// the K=4 conv window and a partial one that does not.
    #[test]
    fn test_gpu_chunk_matches_host_chunked_head_tiling() {
        let Some(device) = try_get_device() else {
            eprintln!("Skipping test_gpu_chunk_matches_host_chunked_head_tiling: CUDA unavailable");
            return;
        };
        let kernel = GatedDeltaNetKernel::new(device.clone()).expect("GDN kernel init");
        if !kernel.has_chunk_gpu() {
            eprintln!("Skipping test_gpu_chunk_matches_host_chunked_head_tiling: NVRTC/cuBLAS unavailable");
            return;
        }

        let cfg = GatedDeltaNetConfig::new(32, 2, 6, 8, 4);
        let w = OwnedWeights::new(&cfg, 91);
        let wv = w.view();

        for t in [1usize, 3, 6] {
            let xs = lcg_fill(t * cfg.hidden_size, 1234 + t as u32);
            let mut host_state = GatedDeltaNetState::zeros(&cfg);
            let host_out = chunked(&cfg, &wv, &xs, &mut host_state).expect("host chunked");

            let mut gpu_state = GatedDeltaNetState::zeros(&cfg);
            let gpu_out = kernel.chunk(&cfg, &wv, &xs, &mut gpu_state).expect("gpu chunk");

            let out_diff = max_abs_diff(&host_out, &gpu_out);
            let rec_diff = max_abs_diff(&gpu_state.recurrent, &host_state.recurrent);
            assert!(out_diff < 1e-2, "T={t}: out max_diff={out_diff}");
            assert!(rec_diff < 1e-2, "T={t}: recurrent max_diff={rec_diff}");
        }
    }

    /// `T=1` on the chunked GPU path must match the *sequential* host `step`
    /// (the sharpest hand-check from 21.16.1), not just the host `chunked`
    /// reference.
    #[test]
    fn test_gpu_chunk_t1_matches_host_step_real_shapes() {
        let Some(device) = try_get_device() else {
            eprintln!("Skipping test_gpu_chunk_t1_matches_host_step_real_shapes: CUDA unavailable");
            return;
        };
        let kernel = GatedDeltaNetKernel::new(device.clone()).expect("GDN kernel init");
        if !kernel.has_chunk_gpu() {
            eprintln!("Skipping test_gpu_chunk_t1_matches_host_step_real_shapes: NVRTC/cuBLAS unavailable");
            return;
        }

        let cfg = GatedDeltaNetConfig::new(1024, 16, 16, 128, 4);
        let w = OwnedWeights::new(&cfg, 5150);
        let wv = w.view();
        let xs = lcg_fill(cfg.hidden_size, 31337);

        let mut host_state = GatedDeltaNetState::zeros(&cfg);
        let host_out = step(&cfg, &wv, &xs, &mut host_state).expect("host step");

        let mut gpu_state = GatedDeltaNetState::zeros(&cfg);
        let gpu_out = kernel.chunk(&cfg, &wv, &xs, &mut gpu_state).expect("gpu chunk");

        let out_diff = max_abs_diff(&host_out, &gpu_out);
        let rec_diff = max_abs_diff(&gpu_state.recurrent, &host_state.recurrent);
        eprintln!("GDN chunk T=1 vs step: out={out_diff:.3e} rec={rec_diff:.3e}");
        assert!(out_diff < 1e-2, "T=1: GPU chunk vs host step out max_diff={out_diff}");
        assert!(rec_diff < 1e-2, "T=1: recurrent max_diff={rec_diff}");
    }
}
