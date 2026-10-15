//! Mamba-2 SSM layer (Granite 4.0-H, `granitehybrid`): host/CPU reference for
//! one token's step through the mixer, written before the CUDA kernels
//! (`src/kernels_cuda/mamba2.cu`) and used as their oracle. Unlike
//! `gated_deltanet_reference.rs` this file is compiled, into the test build
//! only (`src/model/mod.rs` includes it with `#[path]`), so the kernel tests
//! and this file's own tests run under `cargo test`. Plain Rust, no
//! dependencies.
//!
//! Port provenance: llama.cpp `79e2e74`.
//! - `src/models/mamba-base.cpp`, `build_mamba2_layer`: the in-projection
//!   split, conv + bias + SiLU, `dt + dt_bias`, the D skip, the `z` gate and
//!   the grouped RMSNorm.
//! - `ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_ssm_conv_f32` and
//!   `ggml_compute_forward_ssm_scan_f32` (scalar-A branch);
//!   `ggml/src/ggml-cuda/ssm-conv.cu` (`ssm_conv_f32`) and `ssm-scan.cu`
//!   (`ssm_scan_f32_group`), which the CUDA reference build runs.
//! - `conversion/mamba.py`: `ssm_a` is stored as `-exp(A_log)`, `ssm_d` and
//!   `ssm_a` as `[1, n_head]`, `ssm_norm` as `[d_inner / n_group, n_group]`.
//!
//! Notation: `D = d_inner`, `N = d_state`, `G = n_group`, `H = n_head`
//! (`ssm.time_step_rank` in the GGUF), `P = D / H` (head dim),
//! `C = D + 2·G·N` (conv dim). For one token, with `u` the layer's
//! `attn_norm` output:
//!
//! 1. `zxBCdt = ssm_in · u` (no bias), split in order into `z` (`D`), `xBC`
//!    (`C`) and `dt_raw` (`H`).
//! 2. Causal depthwise conv over `xBC`, per channel `c`: the window is the
//!    last `d_conv - 1` raw inputs (oldest first) then the current one;
//!    `xBC'[c] = silu(Σ_k window[k] · w[c·d_conv + k] + b[c])`. The conv state
//!    then drops its oldest column and appends the current raw `xBC`.
//! 3. `x = xBC'[0, D)` as `[H, P]`; `B = xBC'[D, D + G·N)` and
//!    `Cm = xBC'[D + G·N, D + 2·G·N)`, each `[G, N]`.
//! 4. `dt_h = softplus(dt_raw_h + dt_bias_h)`, `softplus(v) = v <= 20 ?
//!    ln(1 + e^v) : v`; `dA_h = exp(dt_h · A_h)`.
//! 5. Head `h` reads group `g = h / (H / G)`. For each `p < P`, `n < N`:
//!    `s[h,p,n] = s[h,p,n] · dA_h + B[g,n] · (x[h,p] · dt_h)`,
//!    `y[h,p] = Σ_n s[h,p,n] · Cm[g,n]`. State layout `[H, P, N]`, `n`
//!    innermost (llama.cpp's `{d_state, head_dim, n_head}`).
//! 6. `y[h,p] += D_h · x[h,p]`.
//! 7. `y *= silu(z)`, then RMSNorm over each of the `G` contiguous groups of
//!    `D / G` (`eps` = `attention.layer_norm_rms_epsilon`), times
//!    `ssm_norm.weight` over the full `D`. The gate comes before the norm.
//! 8. `out = ssm_out · y`.
//!
//! The conv state here is time-major, `[d_conv - 1, C]` (column `k` of
//! channel `c` at `k·C + c`), the same as the Gated DeltaNet conv. llama.cpp
//! stores it channel-major; the values are the same.

/// One Mamba-2 mixer's sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Shape {
    pub d_inner: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub n_head: usize,
    pub d_conv: usize,
    pub eps: f32,
}

impl Mamba2Shape {
    pub fn head_dim(&self) -> usize {
        self.d_inner / self.n_head
    }

    /// `C`: the channels the conv runs over (`x`, `B` and `C`).
    pub fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Length of `ssm_in`'s output: `z`, `xBC`, `dt`.
    pub fn d_in_proj(&self) -> usize {
        self.d_inner + self.conv_dim() + self.n_head
    }

    /// Floats in the conv state, `(d_conv - 1) · C`.
    pub fn conv_state_len(&self) -> usize {
        (self.d_conv - 1) * self.conv_dim()
    }

    /// Floats in the SSM state, `H · P · N`.
    pub fn ssm_state_len(&self) -> usize {
        self.d_inner * self.d_state
    }
}

/// `x · sigmoid(x)`, written as ggml's `x / (1 + exp(-x))`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ggml's softplus for `dt`: identity above 20, `ln(1 + e^v)` otherwise.
pub fn softplus(v: f32) -> f32 {
    if v <= 20.0 {
        v.exp().ln_1p()
    } else {
        v
    }
}

/// Step 2: one token through the causal depthwise conv, bias and SiLU,
/// advancing `state` (`[d_conv - 1, C]`, time-major). `w` is `ssm_conv1d.weight`
/// (`ne = [d_conv, C]`: channel `c`'s taps at `c·d_conv ..`, oldest first),
/// `b` is `ssm_conv1d.bias`. Taps are summed oldest first, then the bias is
/// added, as `ssm_conv_f32` does.
pub fn conv_step(xbc: &[f32], w: &[f32], b: &[f32], state: &mut [f32], d_conv: usize) -> Vec<f32> {
    let c_dim = xbc.len();
    assert_eq!(w.len(), c_dim * d_conv);
    assert_eq!(b.len(), c_dim);
    assert_eq!(state.len(), (d_conv - 1) * c_dim);
    let mut out = vec![0.0f32; c_dim];
    for c in 0..c_dim {
        let mut acc = 0.0f32;
        for k in 0..d_conv {
            let v = if k < d_conv - 1 {
                state[k * c_dim + c]
            } else {
                xbc[c]
            };
            acc += v * w[c * d_conv + k];
        }
        acc += b[c];
        out[c] = silu(acc);
        for k in 0..d_conv.saturating_sub(2) {
            state[k * c_dim + c] = state[(k + 1) * c_dim + c];
        }
        if d_conv > 1 {
            state[(d_conv - 2) * c_dim + c] = xbc[c];
        }
    }
    out
}

/// Steps 4-5: one token's SSM state update and readout. `x` is `[H, P]`,
/// `b`/`c` are `[G, N]`, `dt_raw`/`dt_bias`/`a` are `[H]`, `state` is
/// `[H, P, N]`. Returns `y` (`[H, P]`) without the D skip. The `n` sum runs in
/// index order; the GPU sums per lane then across a warp, so the two agree
/// to rounding, not bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn scan_step(
    shape: &Mamba2Shape,
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt_raw: &[f32],
    dt_bias: &[f32],
    a: &[f32],
    state: &mut [f32],
) -> Vec<f32> {
    let (h_n, p_n, n_n, g_n) = (shape.n_head, shape.head_dim(), shape.d_state, shape.n_group);
    assert_eq!(x.len(), h_n * p_n);
    assert_eq!(b.len(), g_n * n_n);
    assert_eq!(c.len(), g_n * n_n);
    assert_eq!(state.len(), h_n * p_n * n_n);
    let heads_per_group = h_n / g_n;
    let mut y = vec![0.0f32; h_n * p_n];
    for h in 0..h_n {
        let dt = softplus(dt_raw[h] + dt_bias[h]);
        let da = (dt * a[h]).exp();
        let g = h / heads_per_group;
        for p in 0..p_n {
            let x_dt = x[h * p_n + p] * dt;
            let mut sum = 0.0f32;
            for n in 0..n_n {
                let i = (h * p_n + p) * n_n + n;
                let s = state[i] * da + b[g * n_n + n] * x_dt;
                sum += s * c[g * n_n + n];
                state[i] = s;
            }
            y[h * p_n + p] = sum;
        }
    }
    y
}

/// Steps 6-7: `y + D·x`, gated by `silu(z)`, then the grouped RMSNorm times
/// `norm_w` (`[D]`). The sum of squares runs in index order, as Reflex's
/// `rmsnorm_kernel` does.
pub fn skip_gate_norm(
    shape: &Mamba2Shape,
    y: &[f32],
    x: &[f32],
    d: &[f32],
    z: &[f32],
    norm_w: &[f32],
) -> Vec<f32> {
    let (d_inner, p_n) = (shape.d_inner, shape.head_dim());
    let group = d_inner / shape.n_group;
    let mut v: Vec<f32> = (0..d_inner)
        .map(|i| (y[i] + x[i] * d[i / p_n]) * silu(z[i]))
        .collect();
    for g in 0..shape.n_group {
        let chunk = &mut v[g * group..(g + 1) * group];
        let mut sum_sq = 0.0f32;
        for &e in chunk.iter() {
            sum_sq += e * e;
        }
        let rms_inv = 1.0 / (sum_sq / group as f32 + shape.eps).sqrt();
        for (i, e) in chunk.iter_mut().enumerate() {
            *e = *e * rms_inv * norm_w[g * group + i];
        }
    }
    v
}

/// The per-layer tensors the mixer reads besides the two projections.
pub struct Mamba2StepWeights<'a> {
    pub conv_w: &'a [f32],
    pub conv_b: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a: &'a [f32],
    pub d: &'a [f32],
    pub norm_w: &'a [f32],
}

/// Steps 2-7 for one token: from `zxbcdt` (`ssm_in · u`) to the input of
/// `ssm_out`, advancing both states.
pub fn mixer_step(
    shape: &Mamba2Shape,
    zxbcdt: &[f32],
    w: &Mamba2StepWeights,
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
) -> Vec<f32> {
    let (d_inner, c_dim, gn) = (
        shape.d_inner,
        shape.conv_dim(),
        shape.n_group * shape.d_state,
    );
    assert_eq!(zxbcdt.len(), shape.d_in_proj());
    let z = &zxbcdt[..d_inner];
    let xbc = &zxbcdt[d_inner..d_inner + c_dim];
    let dt_raw = &zxbcdt[d_inner + c_dim..];
    let conv = conv_step(xbc, w.conv_w, w.conv_b, conv_state, shape.d_conv);
    let x = &conv[..d_inner];
    let b = &conv[d_inner..d_inner + gn];
    let c = &conv[d_inner + gn..];
    let y = scan_step(shape, x, b, c, dt_raw, w.dt_bias, w.a, ssm_state);
    skip_gate_norm(shape, &y, x, w.d, z, w.norm_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn softplus_switches_to_identity_above_20() {
        assert!(close(softplus(0.0), std::f32::consts::LN_2));
        assert_eq!(softplus(20.5), 20.5);
        assert!(close(softplus(-3.0), (1.0f32 + (-3.0f32).exp()).ln()));
    }

    #[test]
    fn conv_step_sums_oldest_first_adds_bias_and_shifts_state() {
        // Two channels, d_conv 3: state holds the two previous inputs.
        let w = [1.0, 2.0, 3.0, 0.5, 0.0, -1.0];
        let b = [0.25, -0.5];
        let mut state = vec![10.0, 20.0, 11.0, 21.0]; // [k=0: c0,c1][k=1: c0,c1]
        let out = conv_step(&[100.0, 200.0], &w, &b, &mut state, 3);
        let pre0 = 10.0 * 1.0 + 11.0 * 2.0 + 100.0 * 3.0 + 0.25;
        let pre1 = 20.0 * 0.5 + 21.0 * 0.0 - 200.0 - 0.5;
        assert!(close(out[0], silu(pre0)));
        assert!(close(out[1], silu(pre1)));
        assert_eq!(state, vec![11.0, 21.0, 100.0, 200.0]);
    }

    #[test]
    fn scan_step_matches_hand_computed_two_tokens() {
        let shape = Mamba2Shape {
            d_inner: 1,
            d_state: 2,
            n_group: 1,
            n_head: 1,
            d_conv: 4,
            eps: 1e-5,
        };
        let mut state = vec![0.0f32; 2];
        let (b, c) = ([1.0, 2.0], [0.5, -1.0]);
        let a = [-0.5f32];
        let y1 = scan_step(&shape, &[2.0], &b, &c, &[0.3], &[0.2], &a, &mut state);
        let dt = softplus(0.5);
        let s1 = [b[0] * 2.0 * dt, b[1] * 2.0 * dt];
        assert!(close(y1[0], s1[0] * c[0] + s1[1] * c[1]));
        let y2 = scan_step(&shape, &[-1.0], &b, &c, &[0.3], &[0.2], &a, &mut state);
        let da = (dt * a[0]).exp();
        let s2 = [s1[0] * da - b[0] * dt, s1[1] * da - b[1] * dt];
        assert!(close(y2[0], s2[0] * c[0] + s2[1] * c[1]));
        assert!(close(state[0], s2[0]) && close(state[1], s2[1]));
    }

    #[test]
    fn scan_step_maps_heads_to_groups_in_contiguous_blocks() {
        // H = 4, G = 2: heads 0-1 read group 0, heads 2-3 read group 1.
        let shape = Mamba2Shape {
            d_inner: 4,
            d_state: 1,
            n_group: 2,
            n_head: 4,
            d_conv: 4,
            eps: 1e-5,
        };
        let mut state = vec![0.0f32; 4];
        let y = scan_step(
            &shape,
            &[1.0; 4],
            &[1.0, 1.0],
            &[3.0, 7.0],
            &[25.0; 4], // dt = 25 exactly (identity branch)
            &[0.0; 4],
            &[-1.0; 4],
            &mut state,
        );
        assert_eq!(y, vec![75.0, 75.0, 175.0, 175.0]);
    }

    #[test]
    fn skip_gate_norm_gates_before_normalizing_each_group() {
        let shape = Mamba2Shape {
            d_inner: 4,
            d_state: 1,
            n_group: 2,
            n_head: 2,
            d_conv: 4,
            eps: 0.0,
        };
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0, 1.0, 1.0];
        let d = [0.5, -1.0];
        let z = [0.0, 1.0, 2.0, -2.0];
        let w = [1.0, 2.0, 3.0, 4.0];
        let out = skip_gate_norm(&shape, &y, &x, &d, &z, &w);
        let v: Vec<f32> = (0..4)
            .map(|i| (y[i] + x[i] * d[i / 2]) * silu(z[i]))
            .collect();
        for g in 0..2 {
            let ms = (v[2 * g].powi(2) + v[2 * g + 1].powi(2)) / 2.0;
            for i in 2 * g..2 * g + 2 {
                assert!(close(out[i], v[i] / ms.sqrt() * w[i]), "element {i}");
            }
        }
        // z = 0 gates element 0 to exactly zero.
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn mixer_step_splits_zxbcdt_in_order() {
        let shape = Mamba2Shape {
            d_inner: 2,
            d_state: 1,
            n_group: 1,
            n_head: 1,
            d_conv: 2,
            eps: 1e-6,
        };
        assert_eq!(shape.d_in_proj(), 2 + 4 + 1);
        // z = [1, 1]; xBC = [x0, x1, B, C]; dt = 25 (identity).
        let zxbcdt = [1.0, 1.0, 0.5, -0.5, 2.0, 3.0, 25.0];
        let ones = [1.0f32; 8];
        let w = Mamba2StepWeights {
            conv_w: &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            conv_b: &[0.0; 4],
            dt_bias: &[0.0],
            a: &[-1.0],
            d: &[0.0],
            norm_w: &ones[..2],
        };
        let mut conv_state = vec![0.0f32; shape.conv_state_len()];
        let mut ssm_state = vec![0.0f32; shape.ssm_state_len()];
        let out = mixer_step(&shape, &zxbcdt, &w, &mut conv_state, &mut ssm_state);
        let (x0, x1, b, c) = (silu(0.5), silu(-0.5), silu(2.0), silu(3.0));
        let y = [x0 * 25.0 * b * c, x1 * 25.0 * b * c];
        let v = [y[0] * silu(1.0), y[1] * silu(1.0)];
        let rms = ((v[0] * v[0] + v[1] * v[1]) / 2.0 + 1e-6).sqrt();
        assert!(close(out[0], v[0] / rms) && close(out[1], v[1] / rms));
        assert_eq!(conv_state, vec![0.5, -0.5, 2.0, 3.0]);
    }
}
