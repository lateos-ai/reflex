//! A chat model's end-of-generation tokens, read from its GGUF once at startup.
//!
//! The engine stops at the GGUF's `eos_token_id` only. Many chat models end a turn
//! with a different token (Qwen2.5 and Qwen3.5 `<|im_end|>`/`<|endoftext|>`, Gemma
//! 4 `<turn|>`, Llama 3 `<|eot_id|>`), so this adapter passes the whole set as the
//! IPC request's `stop_token_ids` and keeps their text out of the reply: OpenAI
//! responses never contain the stop token. The set follows llama.cpp's
//! end-of-generation rule (`llama-vocab.cpp`): the EOS, EOT and EOM ids from the
//! metadata, plus any token whose text is a known end-of-turn marker.

use crate::gguf_meta::GgufMeta;

/// Token texts llama.cpp treats as end of generation wherever they appear in a
/// vocabulary, plus Gemma 4's `<turn|>`.
const END_OF_TURN_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "<end_of_utterance>",
    "<turn|>",
    "<EOT>",
    "_<EOT>",
];

#[derive(Debug, Default, Clone)]
pub struct StopTokens {
    /// `(id, text)` for every end-of-generation token, sorted by id.
    tokens: Vec<(u32, String)>,
}

impl StopTokens {
    /// The set for the GGUF at `gguf_path`; empty (the engine's EOS-only
    /// stopping, with EOS text kept) if its metadata can't be read.
    pub fn from_gguf(gguf_path: &str) -> StopTokens {
        match GgufMeta::open(gguf_path) {
            Ok(meta) => Self::from_meta(&meta),
            Err(e) => {
                eprintln!("[adapter] stop tokens: failed to read GGUF metadata ({e}); EOS only");
                StopTokens::default()
            }
        }
    }

    fn from_meta(meta: &GgufMeta) -> StopTokens {
        let Some(vocab) = meta.get_string_array("tokenizer.ggml.tokens") else {
            return StopTokens::default();
        };
        let mut ids: Vec<u32> = [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ]
        .iter()
        .filter_map(|key| meta.get_u64(key))
        .filter(|&id| (id as usize) < vocab.len())
        .map(|id| id as u32)
        .collect();
        ids.extend(
            vocab
                .iter()
                .enumerate()
                .filter(|(_, text)| END_OF_TURN_TEXTS.contains(text))
                .map(|(id, _)| id as u32),
        );
        ids.sort_unstable();
        ids.dedup();
        let tokens: Vec<(u32, String)> = ids
            .into_iter()
            .map(|id| (id, vocab[id as usize].to_string()))
            .collect();
        let names: Vec<&str> = tokens.iter().map(|(_, t)| t.as_str()).collect();
        eprintln!("[adapter] stop tokens: {names:?}");
        StopTokens { tokens }
    }

    pub fn ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|(id, _)| *id).collect()
    }

    pub fn contains(&self, id: u32) -> bool {
        self.tokens.binary_search_by_key(&id, |(i, _)| *i).is_ok()
    }

    /// `text` without the stop token's text when the generation ended on one
    /// (`last_id`), the way the engine returns it.
    pub fn strip<'t>(&self, text: &'t str, last_id: Option<u32>) -> &'t str {
        let Some(id) = last_id else {
            return text;
        };
        match self.tokens.binary_search_by_key(&id, |(i, _)| *i) {
            Ok(i) => text.strip_suffix(self.tokens[i].1.as_str()).unwrap_or(text),
            Err(_) => text,
        }
    }

    #[cfg(test)]
    pub fn from_pairs(pairs: &[(u32, &str)]) -> StopTokens {
        let mut tokens: Vec<(u32, String)> =
            pairs.iter().map(|(i, t)| (*i, t.to_string())).collect();
        tokens.sort_unstable_by_key(|(i, _)| *i);
        StopTokens { tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_a_trailing_stop_token() {
        let stops = StopTokens::from_pairs(&[(151645, "<|im_end|>"), (2, "</s>")]);
        assert_eq!(stops.strip("Paris<|im_end|>", Some(151645)), "Paris");
        assert_eq!(
            stops.strip("The capital is Paris.</s>", Some(2)),
            "The capital is Paris."
        );
        // Stopped on max_tokens, or the text doesn't end with the token's text.
        assert_eq!(stops.strip("Paris<|im_end|>", Some(7)), "Paris<|im_end|>");
        assert_eq!(stops.strip("Paris", Some(151645)), "Paris");
        assert_eq!(stops.strip("Paris<|im_end|>", None), "Paris<|im_end|>");
        assert!(stops.contains(2) && stops.contains(151645) && !stops.contains(3));
        assert_eq!(stops.ids(), vec![2, 151645]);
    }
}
