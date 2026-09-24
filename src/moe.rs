//! Mixture-of-Experts router math (Qwen3-MoE, MVP step 2 -- see README.md's
//! MVP order). Ported from RustFeference's own verified `rft-gpu/src/moe.rs`
//! (git history around commit `6a70287`, "qwen3moe support"): `qwen3moe`'s
//! real graph (`src/models/qwen3moe.cpp` in llama.cpp) calls
//! `build_moe_ffn(..., gating_op = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
//! norm_w = true, ...)`, which softmaxes the router logits over *all*
//! experts, selects the top-`k` by probability, and renormalizes just those
//! `k` probabilities to sum to 1 -- exactly Mixtral's published convention,
//! so one function covers both. `qwen3moe` passes `w_scale =
//! hparams.expert_weights_scale` (default `1.0`, a no-op) and no
//! `exp_probs_b`/expert-grouping, so there is no additional bias/scaling
//! step to model.
//!
//! GGUF metadata/tensor-naming facts this module's caller
//! ([`crate::model::parse_model_config`]) relies on, confirmed against a
//! real `Tiny-Moe.Q4_K_M.gguf` fixture (Mixtral-style: `general.architecture
//! = "llama"`, no QK-Norm) since no small real `qwen3moe`-architecture GGUF
//! was available locally: `<arch>.expert_count` / `<arch>.expert_used_count`
//! are namespaced under the file's own architecture string, not hardcoded to
//! `llama.*`, and `<arch>.expert_count` being present and nonzero (rather
//! than the architecture string itself) is the dense-vs-MoE signal.

/// Per-token MoE router: softmax `router_logits` over *all* experts, select
/// the top-`k` by probability, then renormalize just those `k` probabilities
/// to sum to 1. Returns `k` `(expert_index, combination_weight)` pairs,
/// sorted by probability descending (ties broken by ascending expert index,
/// for deterministic output on exactly-equal logits). The combination
/// weights sum to 1 (barring float rounding).
///
/// Errs if `logits` is empty, `k` is 0 or exceeds `logits.len()`, or the
/// softmax/top-k probability mass is non-finite or non-positive (e.g. every
/// logit is `-inf`).
pub fn route_top_k(logits: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
    route_top_k_with_norm(logits, k, true)
}

/// Like [`route_top_k`], but `normalize` controls whether the selected top-`k`
/// probabilities are renormalized to sum to 1 (Qwen3-MoE's convention, `norm_w =
/// true`) or left as raw softmax probabilities (real DeepSeek-V2-Lite's
/// convention -- confirmed against a real GGUF: `expert_weights_norm` is absent,
/// and llama.cpp's own converter only ever writes that key when the source
/// model's `norm_topk_prob` is truthy, so absence there means "don't
/// renormalize"). See `crate::model::MlaMoeConfig`'s doc comment for how this
/// gets threaded from GGUF metadata.
pub fn route_top_k_with_norm(
    logits: &[f32],
    k: usize,
    normalize: bool,
) -> Result<Vec<(usize, f32)>, String> {
    if logits.is_empty() {
        return Err("route_top_k: logits must not be empty".to_string());
    }
    if k == 0 || k > logits.len() {
        return Err(format!(
            "route_top_k: k ({k}) must be in 1..={} (logits.len())",
            logits.len()
        ));
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(format!(
            "route_top_k: softmax sum is non-finite or non-positive ({sum})"
        ));
    }
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let top = &order[..k];

    if !normalize {
        return Ok(top.iter().map(|&i| (i, probs[i])).collect());
    }

    let top_sum: f32 = top.iter().map(|&i| probs[i]).sum();
    if !top_sum.is_finite() || top_sum <= 0.0 {
        return Err(format!(
            "route_top_k: top-{k} probability mass is non-finite or non-positive ({top_sum})"
        ));
    }

    Ok(top.iter().map(|&i| (i, probs[i] / top_sum)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_top_k_matches_hand_computed_softmax_and_renormalization() {
        // logits = ln([1,2,3,4]) -> softmax(logits) = [1,2,3,4]/10 exactly
        // (modulo float rounding of ln/exp, not manual transcription).
        let logits = [1.0f32.ln(), 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()];
        let routed = route_top_k(&logits, 2).expect("route_top_k should succeed");
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].0, 3);
        assert_eq!(routed[1].0, 2);
        assert!(
            (routed[0].1 - 4.0 / 7.0).abs() < 1e-5,
            "got {}",
            routed[0].1
        );
        assert!(
            (routed[1].1 - 3.0 / 7.0).abs() < 1e-5,
            "got {}",
            routed[1].1
        );
        let sum: f32 = routed.iter().map(|&(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "combination weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_route_top_k_ties_break_by_ascending_expert_index() {
        let logits = [0.0f32; 6];
        let routed = route_top_k(&logits, 3).expect("route_top_k should succeed");
        assert_eq!(
            routed.iter().map(|&(i, _)| i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        for &(_, w) in &routed {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_route_top_k_errs_on_empty_logits() {
        assert!(route_top_k(&[], 1).is_err());
    }

    #[test]
    fn test_route_top_k_errs_when_k_exceeds_logits_len() {
        assert!(route_top_k(&[1.0, 2.0], 3).is_err());
    }

    #[test]
    fn test_route_top_k_errs_when_k_is_zero() {
        assert!(route_top_k(&[1.0, 2.0], 0).is_err());
    }

    #[test]
    fn test_route_top_k_errs_on_all_negative_infinity_logits() {
        assert!(route_top_k(&[f32::NEG_INFINITY, f32::NEG_INFINITY], 1).is_err());
    }
}
