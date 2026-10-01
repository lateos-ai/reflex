//! Engine-imposed sequence-length limits, checked host-side before any GPU
//! allocation so an over-long request fails with a clear error instead of an
//! opaque CUDA launch failure deep inside a forward pass. Pure, no CUDA/`Model`
//! coupling (in `crate::calibration`'s style), so it's unit-testable anywhere.

/// Maximum number of cached positions (imported + prompt + generation
/// headroom) the attention kernels can handle in one launch.
///
/// All four attention kernels (`kernels_cuda/attention.cu`,
/// `attention_prefill.cu`, `mla_attention.cu`, `mla_attention_prefill.cu`)
/// keep the whole softmax score row in dynamic shared memory
/// (`extern __shared__ float scores[]`, `seq_len * 4` bytes) next to a static
/// `__shared__ float reduce_buf[1024]` (4 KiB). Nothing opts into more than
/// the default 48 KiB per-block limit
/// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`), so:
/// `(49152 - 4096) / 4 = 11264` positions.
///
/// This is a Reflex engine limit, not the model's context length.
pub const ATTN_MAX_POSITIONS: usize = 11264;

/// Stable prefix of [`check_positions`]'s error message. Callers that can only
/// see error text (the IPC protocol, and through it the OpenAI sidecar, which
/// maps it to `context_length_exceeded`) match on this, so don't reword it.
pub const CONTEXT_LENGTH_EXCEEDED_PREFIX: &str = "context length exceeded";

use crate::error::ReflexError;

/// Errs if `imported + prompt + max_new` positions would exceed
/// [`ATTN_MAX_POSITIONS`]. `imported` is a resumed KV cache's `seq_len`,
/// `prompt` the encoded prompt length (including any inserted BOS), and
/// `max_new` the extra cache headroom reserved past the prompt (generated
/// tokens, or System1's multi-token candidate continuation). Overflow-safe:
/// a sum that doesn't fit in `usize` is reported as exceeding the limit.
pub fn check_positions(imported: usize, prompt: usize, max_new: usize) -> Result<(), ReflexError> {
    let total = imported
        .checked_add(prompt)
        .and_then(|t| t.checked_add(max_new));
    match total {
        Some(total) if total <= ATTN_MAX_POSITIONS => Ok(()),
        requested => Err(ReflexError::ContextOverflow {
            requested,
            imported,
            prompt,
            max_new,
            max: ATTN_MAX_POSITIONS,
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
    fn overflowing_sum_fails_instead_of_wrapping() {
        let err = check_positions(usize::MAX, 1, 0).unwrap_err().to_string();
        assert!(err.starts_with(CONTEXT_LENGTH_EXCEEDED_PREFIX), "{err}");
        assert!(check_positions(1, usize::MAX, usize::MAX).is_err());
        assert!(check_positions(0, 1, usize::MAX).is_err());
    }

    #[test]
    fn derivation_matches_kernel_shared_memory_budget() {
        let default_smem_bytes = 48 * 1024;
        let reduce_buf_bytes = 1024 * std::mem::size_of::<f32>();
        assert_eq!(
            ATTN_MAX_POSITIONS,
            (default_smem_bytes - reduce_buf_bytes) / std::mem::size_of::<f32>()
        );
    }
}
