//! Engine-imposed sequence-length limits, checked host-side before any GPU
//! allocation so an over-long request fails with a clear error instead of an
//! opaque CUDA launch failure deep inside a forward pass. Pure, no CUDA/`Model`
//! coupling (in `crate::calibration`'s style), so it's unit-testable anywhere.

/// Maximum number of cached positions (imported + prompt + generation headroom) in
/// one sequence, with the default online-softmax attention kernels
/// (`kernels_cuda/attention_online.cu`).
///
/// Those kernels no longer keep a score per position in shared memory (that capped
/// the original kernels at [`LEGACY_ATTN_MAX_POSITIONS`]), and put query rows in
/// `grid.x`, which allows 2^31 - 1 blocks. What binds now is CUDA's 65,535-block limit
/// on `grid.y`/`grid.z`: MLA's batched prefill launches `gemv_per_head_batch` with one
/// `grid.z` block per prompt row, so a prompt can't exceed 65,535 rows there. One
/// engine-wide limit keeps the rule simple; it is above Qwen3's 40,960-token native
/// context. Past it, the KV cache itself (VRAM) is the next constraint, and an
/// allocation that doesn't fit fails with `ReflexError::OutOfMemory`.
///
/// This is a Reflex engine limit, not the model's context length.
pub const ATTN_MAX_POSITIONS: usize = 65_535;

/// Maximum positions with the original attention kernels (`REFLEX_ATTN_KERNEL=legacy`:
/// `attention.cu`, `attention_prefill.cu`, `mla_attention.cu`,
/// `mla_attention_prefill.cu`). They keep the whole softmax score row in dynamic
/// shared memory (`extern __shared__ float scores[]`, `seq_len * 4` bytes) next to a
/// static `__shared__ float reduce_buf[1024]` (4 KiB), within the default 48 KiB
/// per-block limit: `(49152 - 4096) / 4 = 11264` positions.
pub const LEGACY_ATTN_MAX_POSITIONS: usize = 11264;

/// Stable prefix of [`check_positions`]'s error message. Callers that can only
/// see error text (the IPC protocol, and through it the OpenAI sidecar, which
/// maps it to `context_length_exceeded`) match on this, so don't reword it.
pub const CONTEXT_LENGTH_EXCEEDED_PREFIX: &str = "context length exceeded";

use crate::error::ReflexError;

/// Errs if `imported + prompt + max_new` positions would exceed
/// [`ATTN_MAX_POSITIONS`]. See [`check_positions_up_to`].
pub fn check_positions(imported: usize, prompt: usize, max_new: usize) -> Result<(), ReflexError> {
    check_positions_up_to(imported, prompt, max_new, ATTN_MAX_POSITIONS)
}

/// Errs with [`ReflexError::ContextOverflow`] if `imported + prompt + max_new`
/// positions would exceed `max`. `imported` is a resumed KV cache's `seq_len`,
/// `prompt` the encoded prompt length (including any inserted BOS), and `max_new`
/// the extra cache headroom reserved past the prompt (generated tokens, or System1's
/// multi-token candidate continuation). Overflow-safe: a sum that doesn't fit in
/// `usize` is reported as exceeding the limit.
pub fn check_positions_up_to(
    imported: usize,
    prompt: usize,
    max_new: usize,
    max: usize,
) -> Result<(), ReflexError> {
    let total = imported
        .checked_add(prompt)
        .and_then(|t| t.checked_add(max_new));
    match total {
        Some(total) if total <= max => Ok(()),
        requested => Err(ReflexError::ContextOverflow {
            requested,
            imported,
            prompt,
            max_new,
            max,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_limit_passes() {
        assert!(check_positions(0, ATTN_MAX_POSITIONS, 0).is_ok());
        assert!(check_positions(1000, ATTN_MAX_POSITIONS - 1100, 100).is_ok());
    }

    #[test]
    fn one_over_limit_fails_with_stable_prefix_and_numbers() {
        let err = check_positions(1000, ATTN_MAX_POSITIONS - 1100, 101)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with(CONTEXT_LENGTH_EXCEEDED_PREFIX), "{err}");
        assert!(err.contains(&(ATTN_MAX_POSITIONS + 1).to_string()), "{err}");
        assert!(err.contains(&ATTN_MAX_POSITIONS.to_string()), "{err}");
        assert!(err.contains("not the model's context length"), "{err}");
    }

    #[test]
    fn explicit_limit_is_respected() {
        let max = LEGACY_ATTN_MAX_POSITIONS;
        assert!(check_positions_up_to(0, max, 0, max).is_ok());
        let err = check_positions_up_to(0, max, 1, max)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&format!("at most {max} positions")), "{err}");
    }

    #[test]
    fn overflowing_sum_fails_instead_of_wrapping() {
        let err = check_positions(usize::MAX, 1, 0).unwrap_err().to_string();
        assert!(err.starts_with(CONTEXT_LENGTH_EXCEEDED_PREFIX), "{err}");
        assert!(check_positions(1, usize::MAX, usize::MAX).is_err());
        assert!(check_positions(0, 1, usize::MAX).is_err());
    }

    #[test]
    fn online_limit_is_cudas_grid_yz_limit() {
        assert_eq!(ATTN_MAX_POSITIONS, u16::MAX as usize);
    }

    #[test]
    fn legacy_derivation_matches_kernel_shared_memory_budget() {
        let default_smem_bytes = 48 * 1024;
        let reduce_buf_bytes = 1024 * std::mem::size_of::<f32>();
        assert_eq!(
            LEGACY_ATTN_MAX_POSITIONS,
            (default_smem_bytes - reduce_buf_bytes) / std::mem::size_of::<f32>()
        );
    }
}
