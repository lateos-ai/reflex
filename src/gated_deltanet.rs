//! Gated DeltaNet (Qwen3.5 hybrid linear-attention) shape config.
//!
//! Port provenance: RustFeference's `reference/gated_deltanet_rustfeference.rs`
//! (itself ported from real `ggml-org/llama.cpp` `src/models/delta-net-base.cpp` /
//! `qwen35.cpp`), which is this MVP step's correctness oracle. Only the static
//! shape config lives here -- the CUDA math is `src/kernels_cuda/gated_deltanet.cu`,
//! and the forward-pass wiring (weights, per-layer state, kernel dispatch) is
//! `src/model.rs`'s hybrid path, matching how `crate::moe` only owns the
//! architecture-agnostic routing math while `model.rs` owns dispatch.
//!
//! `head_dim` is shared by keys and values: llama.cpp asserts `S_k == S_v`
//! (both 128 for every real Qwen3.5 release RustFeference inspected).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    /// Epsilon shared by the L2 q/k norm and the gated output RMSNorm
    /// (`{arch}.attention.layer_norm_rms_epsilon`).
    pub eps: f32,
}

impl GatedDeltaNetConfig {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.head_dim
    }

    /// Fused q/k/v channel count the causal conv1d runs over: `key + key + value`.
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
        if self.hidden_size == 0
            || self.num_k_heads == 0
            || self.num_v_heads == 0
            || self.head_dim == 0
        {
            return Err("GatedDeltaNetConfig has a zero dimension".to_string());
        }
        if !self.num_v_heads.is_multiple_of(self.num_k_heads) {
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

    /// Smallest power-of-two block size >= `head_dim`, clamped to
    /// `[32, 1024]` -- `gdn_l2_norm_kernel`/`gdn_gated_norm_kernel`'s shared
    /// reduction buffer is sized 1024, and needs a power of two to halve
    /// cleanly down to 0.
    pub fn norm_block_dim(&self) -> u32 {
        (self.head_dim as u32).next_power_of_two().clamp(32, 1024)
    }
}
