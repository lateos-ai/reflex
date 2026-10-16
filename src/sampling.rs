//! Sampling strategies for `Model::generate`'s per-step next-token choice.
//! Greedy argmax stays the default and remains available unconditionally --
//! it's load-bearing for this project's own byte-exact-vs-llama.cpp
//! verification methodology (`reflex check`, and the repeated "matched
//! byte-exact" verification rounds behind each architecture), so a caller that never touches
//! sampling gets exactly the same output as before this module existed.
//! Temperature/top-k/top-p sampling is an explicit opt-in
//! (`SamplingParams::temperature > 0.0`) -- see README.md's Non-goals
//! section: sampling strategy isn't in that permanent-constraints list, only
//! `batch_size`/concurrency/networking are, so this was unimplemented scope,
//! not a rejected feature.
//!
//! Host-side only, same small-`Vec`-in/no-CUDA-coupling convention as
//! `crate::calibration` -- the GPU produces raw logits (already downloaded
//! to host by `Model::lm_head_logits`), sampling itself needs no kernel.
//!
//! A `json_schema` constraint (`SamplingParams::json_schema`, `crate::grammar`)
//! masks the logits before either path: tokens that can't continue a value
//! conforming to the schema get `-inf`, and nothing else changes. [`TokenPicker`]
//! is the per-generation state both share.

use crate::error::ReflexError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `temperature <= 0.0` (the `Default`) selects greedy argmax --
/// [`Model::argmax`](crate::model::Model::argmax)'s existing behavior, no RNG
/// draw at all, so a request that never sets `temperature` can't diverge from
/// this project's pre-sampling output even by a rounding hair. `top_k`/
/// `top_p` only take effect once temperature sampling is active; both may be
/// combined (`top_k` narrows the candidate set first, `top_p` further narrows
/// what's left -- the common llama.cpp/HF convention). `seed` makes a
/// sampled run reproducible (`StdRng::seed_from_u64`); omitted, each call
/// draws fresh OS entropy. `json_schema` (see `crate::grammar`) constrains every
/// generated token to continue a value that conforms to the schema; `None` (the
/// default) leaves the logits untouched.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub json_schema: Option<std::sync::Arc<crate::grammar::JsonSchema>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            json_schema: None,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature <= 0.0
    }
}

/// One RNG per `Model::generate` call -- seeded deterministically from
/// `seed` when given (reproducible sampling, e.g. for a test or a caller
/// that wants a replayable trace), otherwise from OS entropy.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// One generation's token choice: the RNG, and the `json_schema` matcher when the
/// request has one. Every architecture's decode loop calls [`Self::pick`] once per
/// position and [`Self::finish`] at the end (see `Model::token_picker`).
pub struct TokenPicker<'a> {
    params: &'a SamplingParams,
    rng: StdRng,
    constraint: Option<crate::grammar::Constrained<'a>>,
    eos: Option<u32>,
    generated: Vec<u32>,
}

impl<'a> TokenPicker<'a> {
    pub fn new(
        params: &'a SamplingParams,
        constraint: Option<crate::grammar::Constrained<'a>>,
        eos: Option<u32>,
    ) -> Self {
        TokenPicker {
            params,
            rng: make_rng(params.seed),
            constraint,
            eos,
            generated: Vec::new(),
        }
    }

    /// Chooses the next token from `logits`. With a constraint, the disallowed
    /// logits are set to `-inf` first (in place, so a caller looking at `logits`
    /// afterwards sees the masked values) and the matcher then follows the choice.
    pub fn pick(&mut self, logits: &mut [f32]) -> Result<u32, ReflexError> {
        if let Some(c) = &mut self.constraint {
            c.mask(logits, &self.generated)?;
        }
        let id = sample(logits, self.params, &mut self.rng)?;
        if let Some(c) = &mut self.constraint {
            c.accept(id)?;
        }
        self.generated.push(id);
        Ok(id)
    }

    /// The streamed text for token `id` (`Tokenizer::decode_stream`). A constrained
    /// generation's EOS streams as empty text, so the streamed pieces join up to
    /// exactly the value.
    pub fn stream_text(
        &self,
        tokenizer: &crate::tokenizer::Tokenizer,
        pending: &mut Vec<u8>,
        id: u32,
    ) -> String {
        if self.constraint.is_some() && Some(id) == self.eos {
            return String::new();
        }
        tokenizer.decode_stream(pending, id)
    }

    /// The generation's text: `tokenizer.decode(generated)`, or with a constraint the
    /// value alone (without EOS's text). A constrained generation that stopped before
    /// the value was complete returns `GrammarDeadEnd` with the partial text.
    pub fn finish(
        self,
        tokenizer: &crate::tokenizer::Tokenizer,
        generated: &[u32],
    ) -> Result<String, ReflexError> {
        let Some(c) = self.constraint else {
            return Ok(tokenizer.decode(generated));
        };
        if !c.is_complete() {
            return Err(c.dead_end(
                format!(
                    "max_tokens ran out after {} tokens, before the value was complete",
                    generated.len()
                ),
                generated,
            ));
        }
        Ok(c.text())
    }
}

/// `REFLEX_TOP2_TRACE=1`: a numerics diagnostic that prints, to stderr, the
/// two highest logits of every generated position (`REFLEX_TOP2 top1_id=..
/// top1_logit=.. top2_id=.. top2_logit=..`, one line per position, in order).
/// Comparing two runs' traces gives the margin the greedy choice had where they
/// diverge, e.g. `--weights f32` against `--weights f16`.
fn top2_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("REFLEX_TOP2_TRACE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn trace_top2(logits: &[f32]) {
    let (mut top1, mut top2) = ((0usize, f32::NEG_INFINITY), (0usize, f32::NEG_INFINITY));
    for (i, &v) in logits.iter().enumerate() {
        if v > top1.1 {
            top2 = top1;
            top1 = (i, v);
        } else if v > top2.1 {
            top2 = (i, v);
        }
    }
    eprintln!(
        "REFLEX_TOP2 top1_id={} top1_logit={} top2_id={} top2_logit={}",
        top1.0, top1.1, top2.0, top2.1
    );
}

/// Samples one token id from `logits` given `params`. Greedy
/// (`params.is_greedy()`) delegates straight to
/// [`Model::argmax`](crate::model::Model::argmax) -- no RNG draw, no
/// floating-point path divergence from this project's pre-sampling
/// behavior. Otherwise: temperature-scale, softmax over the full
/// vocabulary, optionally keep only the `top_k` highest-probability
/// entries, optionally nucleus-filter to the smallest highest-probability
/// prefix whose cumulative probability reaches `top_p`, renormalize over
/// whatever survived, then draw categorically from `rng`.
pub fn sample(
    logits: &[f32],
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32, ReflexError> {
    if logits.is_empty() {
        return Err(ReflexError::InvalidInput(
            "sample: logits must not be empty".to_string(),
        ));
    }
    if top2_trace_enabled() {
        trace_top2(logits);
    }
    if params.is_greedy() {
        return crate::model::Model::argmax(logits);
    }
    if !params.temperature.is_finite() || params.temperature <= 0.0 {
        return Err(crate::reflex_err!(
            InvalidInput,
            "sample: temperature must be positive and finite when sampling, got {}",
            params.temperature
        ));
    }
    if let Some(k) = params.top_k {
        if k == 0 {
            return Err(ReflexError::InvalidInput(
                "sample: top_k must be at least 1".to_string(),
            ));
        }
    }
    if let Some(p) = params.top_p {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(crate::reflex_err!(
                InvalidInput,
                "sample: top_p must be in [0, 1], got {p}"
            ));
        }
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - max_logit) / params.temperature).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(crate::reflex_err!(
            InvalidInput,
            "sample: softmax sum is non-finite or non-positive ({sum})"
        ));
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    if let Some(k) = params.top_k {
        if k < probs.len() {
            probs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            probs.truncate(k);
        }
    }
    if let Some(p) = params.top_p {
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff.max(1));
    }

    let renorm_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    let draw: f32 = rng.gen::<f32>() * renorm_sum;
    let mut acc = 0.0f32;
    for &(id, p) in &probs {
        acc += p;
        if draw <= acc {
            return Ok(id);
        }
    }
    // Floating-point rounding can leave `draw` a hair above the accumulated
    // sum on the last entry -- fall back to it rather than erroring.
    Ok(probs
        .last()
        .map(|&(id, _)| id)
        .expect("probs is never empty: logits was checked non-empty above"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_matches_argmax_regardless_of_other_fields() {
        let logits = [1.0f32, 5.0, 3.0, -2.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: Some(1),
            top_p: Some(0.1),
            seed: Some(42),
            ..Default::default()
        };
        let mut rng = make_rng(params.seed);
        let id = sample(&logits, &params, &mut rng).expect("sample should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_sampling_with_seed_is_reproducible() {
        let logits = [1.0f32, 2.0, 3.0, 0.5, 4.0, -1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: Some(7),
            ..Default::default()
        };
        let mut rng_a = make_rng(params.seed);
        let mut rng_b = make_rng(params.seed);
        let draws_a: Vec<u32> = (0..20)
            .map(|_| sample(&logits, &params, &mut rng_a).unwrap())
            .collect();
        let draws_b: Vec<u32> = (0..20)
            .map(|_| sample(&logits, &params, &mut rng_b).unwrap())
            .collect();
        assert_eq!(draws_a, draws_b);
    }

    #[test]
    fn test_top_k_one_always_matches_argmax() {
        let logits = [1.0f32, 5.0, 3.0, -2.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
            seed: Some(1),
            ..Default::default()
        };
        let mut rng = make_rng(params.seed);
        for _ in 0..10 {
            let id = sample(&logits, &params, &mut rng).unwrap();
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn test_top_p_near_zero_always_matches_argmax() {
        let logits = [1.0f32, 5.0, 3.0, -2.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1e-6),
            seed: Some(1),
            ..Default::default()
        };
        let mut rng = make_rng(params.seed);
        for _ in 0..10 {
            let id = sample(&logits, &params, &mut rng).unwrap();
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn test_sampling_only_ever_returns_in_range_ids() {
        let logits = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let params = SamplingParams {
            temperature: 2.0,
            top_k: Some(3),
            top_p: Some(0.9),
            seed: Some(99),
            ..Default::default()
        };
        let mut rng = make_rng(params.seed);
        for _ in 0..200 {
            let id = sample(&logits, &params, &mut rng).unwrap();
            assert!((id as usize) < logits.len());
        }
    }

    #[test]
    fn test_sample_errs_on_empty_logits() {
        let params = SamplingParams::default();
        let mut rng = make_rng(Some(1));
        assert!(sample(&[], &params, &mut rng).is_err());
    }

    #[test]
    fn test_sample_errs_on_invalid_top_p() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.5),
            seed: Some(1),
            ..Default::default()
        };
        let mut rng = make_rng(params.seed);
        assert!(sample(&[1.0, 2.0], &params, &mut rng).is_err());
    }
}
