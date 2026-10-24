//! Raw label-token logits over several prompts that share a token prefix: the
//! engine side of decision models (`docs/design/decision-models.md`).
//!
//! Each prompt names label strings, each of which must encode to exactly one
//! token. The result is that token's raw LM-head logit at the prompt's last
//! position; the caller turns logits into probabilities. The prompts' longest
//! common token prefix is prefilled once and every prompt's own tokens run from a
//! copy of that state, one prompt after another (llama.cpp's
//! `server_decision_group_tasks`, with every prompt in one group). The prefix is
//! dropped when the call returns: it never touches the session's pinned prefix.

use super::prefix::{HybridDeviceState, PrefixSource};
use super::Model;
use crate::error::ReflexError;
use std::cell::Cell;
use std::collections::HashMap;

/// One prompt to score, and the labels to read at its last position.
#[derive(Debug, Clone)]
pub struct LabelPrompt {
    pub prompt: String,
    pub labels: Vec<String>,
}

/// One prompt's result, in the order of [`LabelPrompt::labels`].
#[derive(Debug, Clone)]
pub struct LabelPromptResult {
    /// The prompt's full token count, shared prefix included.
    pub tokens: usize,
    pub label_token_ids: Vec<u32>,
    pub logits: Vec<f32>,
}

/// Result of [`Model::label_logits`].
#[derive(Debug, Clone)]
pub struct LabelLogits {
    /// Same order as the prompts.
    pub results: Vec<LabelPromptResult>,
    /// Tokens prefilled once for every prompt (0 when nothing is shared).
    pub shared_tokens: usize,
}

/// Most prompts one call may score. Each one is a prefill of its own tokens.
pub const MAX_LABEL_PROMPTS: usize = 256;

/// Most labels one prompt may read (lev's label table holds 255).
pub const MAX_LABELS_PER_PROMPT: usize = 255;

/// Length of the token prefix every prompt starts with, leaving each prompt at
/// least one token of its own (the logits come from its last position). 0 for
/// fewer than two prompts, as in llama.cpp.
pub(super) fn shared_prefix_len(prompts: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = prompts.split_first() else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let mut n = first.len().saturating_sub(1);
    for ids in rest {
        let common = first.iter().zip(ids).take_while(|(a, b)| a == b).count();
        n = n.min(common).min(ids.len().saturating_sub(1));
    }
    n
}

impl Model {
    /// Scores `prompts` (see this module's doc comment). Qwen3.5 hybrid models
    /// only for now; every other architecture returns `UnsupportedArchitecture`.
    /// A prompt is encoded the way every other prompt is (special tokens parsed,
    /// BOS added if the file asks for one); a label is encoded alone with special
    /// tokens as plain text, and must give exactly one token.
    pub fn label_logits(&self, prompts: &[LabelPrompt]) -> Result<LabelLogits, ReflexError> {
        self.check_generative()?;
        let Some(h) = &self.hybrid else {
            return Err(ReflexError::UnsupportedArchitecture(
                "label_logits: only Qwen3.5 hybrid models are supported".to_string(),
            ));
        };
        if prompts.is_empty() {
            return Err(ReflexError::InvalidInput(
                "label_logits: prompts must not be empty".to_string(),
            ));
        }
        if prompts.len() > MAX_LABEL_PROMPTS {
            return Err(crate::reflex_err!(
                InvalidInput,
                "label_logits: {} prompts, at most {MAX_LABEL_PROMPTS} are allowed",
                prompts.len()
            ));
        }

        let mut label_ids: HashMap<&str, u32> = HashMap::new();
        let mut per_prompt_labels = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            if p.labels.is_empty() || p.labels.len() > MAX_LABELS_PER_PROMPT {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "label_logits: prompt {i} has {} labels, it needs 1 to {MAX_LABELS_PER_PROMPT}",
                    p.labels.len()
                ));
            }
            let mut ids = Vec::with_capacity(p.labels.len());
            for label in &p.labels {
                let id = match label_ids.get(label.as_str()) {
                    Some(&id) => id,
                    None => {
                        let toks = self.tokenizer.encode_no_special(label)?;
                        let [id] = toks[..] else {
                            return Err(crate::reflex_err!(
                                InvalidInput,
                                "label_logits: label {label:?} is {} tokens, a label must be one token",
                                toks.len()
                            ));
                        };
                        label_ids.insert(label, id);
                        id
                    }
                };
                ids.push(id);
            }
            per_prompt_labels.push(ids);
        }

        let mut token_ids = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let mut ids = self.tokenizer.encode(&p.prompt)?;
            if let Some(bos) = self.tokenizer.prompt_bos() {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
            if ids.is_empty() {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "label_logits: prompt {i} encodes to no tokens"
                ));
            }
            token_ids.push(ids);
        }

        let shared_tokens = shared_prefix_len(&token_ids);
        let shared = if shared_tokens > 0 {
            let (_hidden, layers, seq_len) =
                self.prefill_hybrid_ids(h, &token_ids[0][..shared_tokens], None, 0)?;
            Some(HybridDeviceState { seq_len, layers })
        } else {
            None
        };

        let hidden_size = h.attn_cfg.hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let copy_ms = Cell::new(None);
        let mut results = Vec::with_capacity(prompts.len());
        for (ids, labels) in token_ids.iter().zip(per_prompt_labels) {
            let src = shared.as_ref().map(|state| PrefixSource::Device {
                state,
                copy_ms: &copy_ms,
            });
            let own = &ids[shared_tokens..];
            let (hidden_batched, _states, _position) = self.prefill_hybrid_ids(h, own, src, 0)?;
            let hidden = self.last_row(&hidden_batched, own.len(), hidden_size)?;
            let normed = self.rmsnorm(&hidden, self.output_norm.f32()?, 1, hidden_size, eps)?;
            let logits = self.gemv_gather_lm_head(&normed, &labels)?;
            results.push(LabelPromptResult {
                tokens: ids.len(),
                label_token_ids: labels,
                logits,
            });
        }
        Ok(LabelLogits {
            results,
            shared_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::shared_prefix_len;

    #[test]
    fn shared_prefix_needs_two_prompts() {
        assert_eq!(shared_prefix_len(&[]), 0);
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3]]), 0);
    }

    #[test]
    fn shared_prefix_is_the_common_start() {
        let p = [vec![1, 2, 3, 4, 5], vec![1, 2, 3, 9], vec![1, 2, 3, 4, 7]];
        assert_eq!(shared_prefix_len(&p), 3);
        assert_eq!(shared_prefix_len(&[vec![1, 2], vec![3, 2]]), 0);
    }

    #[test]
    fn shared_prefix_leaves_every_prompt_one_token() {
        // Identical prompts, and one prompt that is a prefix of another.
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3]]), 2);
        assert_eq!(shared_prefix_len(&[vec![1, 2], vec![1, 2, 3, 4]]), 1);
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3, 4], vec![1, 2]]), 1);
        assert_eq!(shared_prefix_len(&[vec![1], vec![1, 2]]), 0);
    }
}
