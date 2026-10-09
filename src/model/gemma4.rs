//! Gemma 4 (`gemma4`) models: the kernels only this architecture uses, with
//! host references for their tests. Conventions and sources:
//! `docs/design/gemma4.md`. Loading and the forward pass are not wired up yet,
//! so nothing outside the tests uses these.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;
use crate::error::ReflexError;

/// Threads per block for the elementwise kernels in `kernels_cuda/gemma4.cu`;
/// they are grid-stride loops, so the grid is capped.
const THREADS: u32 = 256;
const MAX_BLOCKS: u64 = 65_535;

fn grid_stride_cfg(n: u64) -> LaunchConfig {
    let blocks = n.div_ceil(THREADS as u64).clamp(1, MAX_BLOCKS) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The `kernels_cuda/gemma4.cu` kernels, loaded from one module.
pub(super) struct Gemma4Kernels {
    rope_ff: AotKernel,
    gelu_tanh_and_mul: AotKernel,
    softcap: AotKernel,
    scale: AotKernel,
}

impl Gemma4Kernels {
    pub(super) fn load(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let mut fns = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_GEMMA4")),
            "gemma4",
            &[
                "rope_neox_ff_kernel",
                "gelu_tanh_and_mul_kernel",
                "softcap_kernel",
                "scale_kernel",
            ],
        )?
        .into_iter();
        let mut next = || {
            fns.next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module gemma4"))
        };
        Ok(Self {
            rope_ff: next()?,
            gelu_tanh_and_mul: next()?,
            softcap: next()?,
            scale: next()?,
        })
    }

    /// NEOX RoPE with per-pair frequency factors, in place over `t`
    /// (`[rows, num_heads, head_dim]`, row `r` at position `start_pos + r`).
    /// `freq_factors` holds `n_dims / 2` divisors; all ones is plain RoPE.
    /// See [`rope_neox_ff_ref`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_ff(
        &self,
        t: &mut CudaSlice<f32>,
        freq_factors: &CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        num_heads: usize,
        head_dim: usize,
        n_dims: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        if n_dims == 0
            || !n_dims.is_multiple_of(2)
            || n_dims > head_dim
            || freq_factors.len() < n_dims / 2
        {
            return Err(crate::reflex_err!(
                Other,
                "rope_ff: n_dims {n_dims} must be even, at most head_dim {head_dim}, with {} freq factors (got {})",
                n_dims / 2,
                freq_factors.len()
            ));
        }
        let pairs = (rows * num_heads * (n_dims / 2)) as u64;
        unsafe {
            self.rope_ff
                .function
                .clone()
                .launch(
                    grid_stride_cfg(pairs),
                    (
                        t,
                        freq_factors,
                        start_pos as u32,
                        rows as u32,
                        num_heads as u32,
                        head_dim as u32,
                        n_dims as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_ff launch: {e}"))?;
        }
        Ok(())
    }

    /// GeGLU: `gelu_tanh(gate) * up`, elementwise over equal-length buffers.
    /// See [`gelu_tanh_ref`].
    pub(super) fn gelu_tanh_and_mul(
        &self,
        device: &Arc<CudaDevice>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let n = gate.len();
        if up.len() != n {
            return Err(crate::reflex_err!(
                Other,
                "gelu_tanh_and_mul: gate has {n} elements, up has {}",
                up.len()
            ));
        }
        let mut out = device
            .alloc_zeros::<f32>(n)
            .map_err(|e| crate::gpu_err!(e, "geglu alloc out: {e}"))?;
        unsafe {
            self.gelu_tanh_and_mul
                .function
                .clone()
                .launch(grid_stride_cfg(n as u64), (gate, up, &mut out, n as u64))
                .map_err(|e| crate::gpu_err!(e, "geglu launch: {e}"))?;
        }
        Ok(out)
    }

    /// In place: `x = cap * tanh(x / cap)` (the final logit soft-cap).
    pub(super) fn softcap(&self, x: &mut CudaSlice<f32>, cap: f32) -> Result<(), ReflexError> {
        let n = x.len() as u64;
        unsafe {
            self.softcap
                .function
                .clone()
                .launch(grid_stride_cfg(n), (x, n, cap))
                .map_err(|e| crate::gpu_err!(e, "softcap launch: {e}"))?;
        }
        Ok(())
    }

    /// In place: `x *= s` (a layer's `layer_output_scale`).
    pub(super) fn scale(&self, x: &mut CudaSlice<f32>, s: f32) -> Result<(), ReflexError> {
        let n = x.len() as u64;
        unsafe {
            self.scale
                .function
                .clone()
                .launch(grid_stride_cfg(n), (x, n, s))
                .map_err(|e| crate::gpu_err!(e, "scale launch: {e}"))?;
        }
        Ok(())
    }
}

/// Host reference for `rope_neox_ff_kernel`, same arithmetic in f32: pair
/// `(i, i + n_dims/2)` of each head rotates by
/// `position * (base^(-2/n_dims))^i / freq_factors[i]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn rope_neox_ff_ref(
    t: &mut [f32],
    freq_factors: &[f32],
    start_pos: usize,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    n_dims: usize,
    base: f32,
) {
    let half = n_dims / 2;
    let theta_scale = base.powf(-2.0 / n_dims as f32);
    for row in 0..rows {
        let position = (start_pos + row) as f32;
        for head in 0..num_heads {
            let off = (row * num_heads + head) * head_dim;
            for i in 0..half {
                let theta = position * theta_scale.powf(i as f32) / freq_factors[i];
                let (sin_t, cos_t) = theta.sin_cos();
                let (x1, x2) = (t[off + i], t[off + i + half]);
                t[off + i] = x1 * cos_t - x2 * sin_t;
                t[off + i + half] = x1 * sin_t + x2 * cos_t;
            }
        }
    }
}

/// Host reference for the GELU in `gelu_tanh_and_mul_kernel` (tanh
/// approximation, ggml's constants).
pub(super) fn gelu_tanh_ref(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Host reference for `softcap_kernel`.
pub(super) fn softcap_ref(x: f32, cap: f32) -> f32 {
    (x * (1.0 / cap)).tanh() * cap
}
