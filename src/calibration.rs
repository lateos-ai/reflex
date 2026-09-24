//! System1 candidate-score calibration (see `crate::model::Model::system1_evaluate`):
//! host-side softmax over a small vector of per-candidate scores, in
//! `crate::moe::route_top_k_with_norm`'s style -- small `Vec<f32>` in, no
//! CUDA/`Model` coupling. Produces a probability distribution *over just
//! this candidate set*, not a full-vocabulary-normalized log-probability --
//! see `Model::system1_evaluate`'s doc comment for why that distinction
//! matters.

/// Softmaxes `scores` (temperature `1.0`, a no-op). See
/// [`softmax_scores_with_temperature`].
pub fn softmax_scores(scores: &[f32]) -> Result<Vec<f32>, String> {
    softmax_scores_with_temperature(scores, 1.0)
}

/// Softmaxes `scores` after dividing by `temperature` (Platt-style scaling;
/// `1.0` is a no-op, values below `1.0` sharpen the distribution, above
/// `1.0` flatten it). Errs if `scores` is empty, `temperature` is not
/// positive/finite, or the softmax sum is non-finite/non-positive.
pub fn softmax_scores_with_temperature(
    scores: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, String> {
    if scores.is_empty() {
        return Err("softmax_scores: scores must not be empty".to_string());
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(format!(
            "softmax_scores: temperature must be positive and finite, got {temperature}"
        ));
    }
    let scaled: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();
    let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(format!(
            "softmax_scores: softmax sum is non-finite or non-positive ({sum})"
        ));
    }
    Ok(exps.iter().map(|&e| e / sum).collect())
}

/// Shannon entropy (base-2, bits) of a probability distribution -- `0.0` for a
/// one-hot distribution (fully confident), `log2(probabilities.len())` for a
/// uniform one (fully uncertain). Intended for `System1Response::entropy`
/// (`crate::model`), so local agents have a single scalar confidence/escalation
/// signal alongside the raw `probabilities` vector. `0 * log2(0)` is treated as
/// `0.0` (the standard convention), never `NaN`. Errs under the same conditions
/// [`softmax_scores`]'s output can never actually produce (empty input, a
/// non-finite entry) -- defensive, since this is a public function any caller
/// could hand a hand-built distribution to, not just `system1_evaluate`'s own
/// already-validated output.
pub fn shannon_entropy(probabilities: &[f32]) -> Result<f32, String> {
    if probabilities.is_empty() {
        return Err("shannon_entropy: probabilities must not be empty".to_string());
    }
    if probabilities.iter().any(|p| !p.is_finite()) {
        return Err("shannon_entropy: probabilities must all be finite".to_string());
    }
    Ok(-probabilities
        .iter()
        .map(|&p| if p <= 0.0 { 0.0 } else { p * p.log2() })
        .sum::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_scores_matches_hand_computed_softmax() {
        // scores = ln([1,2,3,4]) -> softmax(scores) = [1,2,3,4]/10 exactly
        // (modulo float rounding of ln/exp, not manual transcription).
        let scores = [1.0f32.ln(), 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()];
        let probs = softmax_scores(&scores).expect("softmax_scores should succeed");
        assert_eq!(probs.len(), 4);
        for (got, expected) in probs.iter().zip([0.1, 0.2, 0.3, 0.4]) {
            assert!(
                (got - expected).abs() < 1e-5,
                "got {got}, expected {expected}"
            );
        }
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_softmax_scores_single_candidate_is_probability_one() {
        let probs = softmax_scores(&[42.0]).expect("softmax_scores should succeed");
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6, "got {}", probs[0]);
    }

    #[test]
    fn test_lower_temperature_sharpens_distribution() {
        let scores = [1.0f32, 2.0f32];
        let flat = softmax_scores_with_temperature(&scores, 10.0).expect("should succeed");
        let sharp = softmax_scores_with_temperature(&scores, 0.1).expect("should succeed");
        assert!(
            sharp[1] > flat[1],
            "lower temperature should sharpen toward the higher score"
        );
    }

    #[test]
    fn test_softmax_scores_errs_on_empty_input() {
        assert!(softmax_scores(&[]).is_err());
    }

    #[test]
    fn test_softmax_scores_errs_on_non_finite_or_non_positive_temperature() {
        assert!(softmax_scores_with_temperature(&[1.0, 2.0], 0.0).is_err());
        assert!(softmax_scores_with_temperature(&[1.0, 2.0], -1.0).is_err());
        assert!(softmax_scores_with_temperature(&[1.0, 2.0], f32::NAN).is_err());
        assert!(softmax_scores_with_temperature(&[1.0, 2.0], f32::INFINITY).is_err());
    }

    #[test]
    fn test_shannon_entropy_one_hot_is_zero() {
        let entropy = shannon_entropy(&[1.0, 0.0, 0.0, 0.0]).expect("should succeed");
        assert!((entropy - 0.0).abs() < 1e-6, "got {entropy}");
    }

    #[test]
    fn test_shannon_entropy_uniform_is_log2_n() {
        let entropy = shannon_entropy(&[0.25, 0.25, 0.25, 0.25]).expect("should succeed");
        assert!(
            (entropy - 2.0).abs() < 1e-5,
            "got {entropy}, expected log2(4) = 2.0"
        );
    }

    #[test]
    fn test_shannon_entropy_matches_hand_computed_value() {
        // H([0.5, 0.5]) = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0 bit exactly.
        let entropy = shannon_entropy(&[0.5, 0.5]).expect("should succeed");
        assert!((entropy - 1.0).abs() < 1e-6, "got {entropy}");
    }

    #[test]
    fn test_shannon_entropy_errs_on_empty_input() {
        assert!(shannon_entropy(&[]).is_err());
    }

    #[test]
    fn test_shannon_entropy_errs_on_non_finite_input() {
        assert!(shannon_entropy(&[0.5, f32::NAN]).is_err());
        assert!(shannon_entropy(&[0.5, f32::INFINITY]).is_err());
    }
}
