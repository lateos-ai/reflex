//! Tokenizer extraction from GGUF metadata, plus from-scratch encode/decode
//! for three real tokenizer schemes: llama.cpp's SentencePiece-derived scheme
//! (`tokenizer.ggml.model == "llama"`, Phase 21.5.1), GPT-2-style
//! byte-level BPE (`tokenizer.ggml.model == "gpt2"`, Phase 21.14 — Qwen3
//! architecture support), and Gemma 4's BPE over raw UTF-8 with
//! SentencePiece-style spaces (`tokenizer.ggml.model == "gemma4"`, see
//! [`Tokenizer::encode_gemma4`]; it decodes like the SentencePiece scheme).
//! [`Tokenizer::encode`]/[`Tokenizer::decode`]
//! dispatch on `self.architecture`; see [`Tokenizer::encode_gpt2`]'s own doc
//! for how the two schemes actually differ (regex pre-tokenization and a
//! byte-to-unicode alphabet remap, not just a vocab swap) and what's shared
//! (the BPE merge algorithm itself, [`Tokenizer::bpe_merge`]).
//!
//! **SentencePiece path**: word-boundary marker `▁` (U+2581) replacing
//! spaces, byte-fallback `<0xXX>` tokens for anything outside the vocab,
//! greedy lowest-merge-rank adjacent-pair merging — confirmed against a real
//! TinyLlama reference (`PHASE21_real_gguf_verification`).
//!
//! **GPT-2 path**: the pre-tokenization split depends on
//! `tokenizer.ggml.pre` ([`PreType`]; unknown values are rejected). Each
//! split is hand-ported from llama.cpp `79e2e74` and checked token for token
//! against its `llama-tokenize` on a tricky string set, one tokenizer-only
//! fixture per pre type (`test_*_encode_matches_llama_tokenize_if_fixture_present`;
//! Kolibri-1 is also checked against HF `tokenizers`). See
//! docs/design/tokenizer-pre-types.md. Every split classifies codepoints
//! with llama.cpp's own Unicode tables (`crate::tokenizer_unicode`), never
//! Rust's `is_alphabetic`/`is_numeric`/`is_whitespace`: those disagree with
//! the tables on combining marks such as Arabic harakat.

use crate::error::ReflexError;
use crate::gguf::{GgufFile, GgufValue};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const WORD_BOUNDARY: char = '\u{2581}'; // '▁', SentencePiece's space marker

/// A small, non-cryptographic hasher (the classic Firefox/rustc `FxHash`
/// algorithm) used only for [`Tokenizer`]'s two build-time bulk maps
/// (`token_to_id`, `merge_rank`). `std`'s default SipHash is designed for
/// HashDoS resistance on untrusted input, which these maps never see -- keys
/// come from a GGUF file the user already chose to load, in the same trust
/// boundary as the weights themselves. That is the same "speed, no DOS
/// surface" reasoning `aot.rs`'s `fnv1a_hash` already relies on. Equality is
/// still checked on every lookup, so the hash only needs to be consistent,
/// not strong (and encode only ever does direct `get`s, never iterates, so
/// byte-exact output is unaffected).
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x517cc1b727220a95;
    const ROTATE: u32 = 5;

    #[inline]
    fn add_to_hash(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(Self::ROTATE) ^ word).wrapping_mul(Self::SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while rest.len() >= 8 {
            self.add_to_hash(u64::from_ne_bytes(rest[..8].try_into().unwrap()));
            rest = &rest[8..];
        }
        if rest.len() >= 4 {
            self.add_to_hash(u32::from_ne_bytes(rest[..4].try_into().unwrap()) as u64);
            rest = &rest[4..];
        }
        if rest.len() >= 2 {
            self.add_to_hash(u16::from_ne_bytes(rest[..2].try_into().unwrap()) as u64);
            rest = &rest[2..];
        }
        for &b in rest {
            self.add_to_hash(b as u64);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;
type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

fn byte_fallback_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

fn parse_byte_fallback(tok: &str) -> Option<u8> {
    let inner = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}

/// The split regex a `tokenizer.ggml.model == "gpt2"` vocab pre-tokenizes
/// with, read from `tokenizer.ggml.pre` and named after llama.cpp's
/// `LLAMA_VOCAB_PRE_TYPE_*` (`src/llama-vocab.cpp` at `79e2e74`). Each variant
/// is a hand-port of that pre type's regex list; see
/// docs/design/tokenizer-pre-types.md for where each came from and how to add
/// one. Any other `pre` value is rejected by [`PreType::from_gguf_value`]:
/// before this enum existed every gpt2 vocab silently got the Qwen2 split, so
/// a Llama-3 or DeepSeek GGUF loaded and tokenized wrongly with no error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `LLAMA_VOCAB_PRE_TYPE_QWEN2` (`qwen2`, `kolibri1`, or no `pre` key):
    /// one digit per piece (`\p{N}`), letter runs `\p{L}+`.
    Qwen2,
    /// `LLAMA_VOCAB_PRE_TYPE_QWEN35` (`qwen35`): Qwen2's split, but letter
    /// runs also take combining marks (`[\p{L}\p{M}]+`).
    Qwen35,
    /// `LLAMA_VOCAB_PRE_TYPE_LLAMA3` (`llama-bpe`): Qwen2's split with digit
    /// runs grouped up to three (`\p{N}{1,3}`), plus llama.cpp's
    /// `ignore_merges` (a pre-token that is already one vocab entry skips BPE).
    Llama3,
    /// `LLAMA_VOCAB_PRE_TYPE_DBRX` (`dbrx`, Granite 4.0): LLAMA3's regex
    /// string (`llama-vocab.cpp`: "same as llama3"), so the same split, but
    /// without `ignore_merges`.
    Dbrx,
    /// `LLAMA_VOCAB_PRE_TYPE_GPT2` (`gpt-2`): the original GPT-2 regex.
    Gpt2,
    /// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` (`deepseek-llm`): six regexes
    /// applied in sequence (DeepSeek LLM 7B, DeepSeek-V2/V2-Lite).
    DeepseekLlm,
    /// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` (`deepseek-v3`): three regexes
    /// applied in sequence (DeepSeek-V3/R1).
    Deepseek3,
    /// Mistral's Tekken (`tekken`, `pixtral`, `nemotron-3.5-embed`): the
    /// `tokenizer.json` regex of Ministral 3 / Pixtral, with `ignore_merges`.
    /// Letter runs split at case changes (`[Lu..]*[Ll..]+`), one digit per
    /// piece. Matched with HF `tokenizers`' semantics ([`tekken_match`]):
    /// llama.cpp `79e2e74` maps `pixtral` to LLAMA3 instead, and later
    /// llama.cpp to TEKKEN on its collapsed text, where every non-ASCII letter
    /// counts as both cases.
    Tekken,
}

impl PreType {
    /// Every accepted `tokenizer.ggml.pre` value and the variant it maps to.
    /// Only values whose split was checked against llama.cpp's
    /// `llama-tokenize` are listed (aliases llama.cpp also accepts, like
    /// `llama3` for `llama-bpe`, are left out until a real file needs them).
    pub const ACCEPTED: &'static [(&'static str, PreType)] = &[
        ("qwen2", PreType::Qwen2),
        // Not a llama.cpp value at 79e2e74: the community Kolibri-1 patch
        // maps it to LLAMA_VOCAB_PRE_TYPE_QWEN2, and Reflex matches HF
        // `tokenizers` with the Qwen2 split
        // (`test_kolibri1_encode_matches_hf_tokenizers_if_fixture_present`).
        ("kolibri1", PreType::Qwen2),
        ("qwen35", PreType::Qwen35),
        ("llama-bpe", PreType::Llama3),
        ("gpt-2", PreType::Gpt2),
        ("deepseek-llm", PreType::DeepseekLlm),
        ("deepseek-v3", PreType::Deepseek3),
        // GigaChat3: llama.cpp maps it to LLAMA_VOCAB_PRE_TYPE_GPT2 with no
        // other flags.
        ("gigachat", PreType::Gpt2),
        // GLM-4.7-Flash: llama.cpp maps it to LLAMA_VOCAB_PRE_TYPE_CHATGLM4,
        // whose regex is LLAMA3's string (so `unicode_regex_split_custom_llama3`)
        // with `ignore_merges`, plus no BOS ([`PreType::value_disables_bos`]).
        ("glm4", PreType::Llama3),
        // Granite 4.0 (`granitehybrid`): llama.cpp sets no flags for `dbrx`,
        // so no `ignore_merges` and the BOS follows `add_bos_token`.
        ("dbrx", PreType::Dbrx),
        // Mistral's Tekken: Ministral 3 / Pixtral GGUFs (`pixtral`), and
        // Nemotron-3.5-Embed (`nemotron-3.5-embed` in llama.cpp after
        // `79e2e74`; earlier conversions say `pixtral`). Checked against HF
        // `tokenizers` on the model's `tokenizer.json`.
        ("tekken", PreType::Tekken),
        ("pixtral", PreType::Tekken),
        ("nemotron-3.5-embed", PreType::Tekken),
    ];

    /// Whether llama.cpp sets `special_bos_id = LLAMA_TOKEN_NULL` for this
    /// `tokenizer.ggml.pre` value (`llama_vocab::impl::load` at `79e2e74`), so
    /// no BOS is ever prepended whatever `add_bos_token` says: `glm4` (whose
    /// GGUFs declare a BOS id but no `add_bos_token` key).
    fn value_disables_bos(pre: Option<&str>) -> bool {
        pre == Some("glm4")
    }

    /// Map a gpt2-model vocab's `tokenizer.ggml.pre` to its split. A missing
    /// key keeps the Qwen2 split every gpt2 vocab got before this was read,
    /// so files that loaded before still tokenize identically (llama.cpp
    /// itself warns and falls back to its `default` regex there; no file in
    /// test-data/ lacks the key). Unknown values are an error naming the
    /// value: tokenizing with the wrong split gives wrong ids, not a crash.
    pub fn from_gguf_value(pre: Option<&str>) -> Result<Self, ReflexError> {
        let Some(pre) = pre else {
            return Ok(PreType::Qwen2);
        };
        Self::ACCEPTED
            .iter()
            .find(|(name, _)| *name == pre)
            .map(|&(_, ty)| ty)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ACCEPTED.iter().map(|(n, _)| *n).collect();
                crate::reflex_err!(
                    Tokenizer,
                    "unsupported tokenizer.ggml.pre '{pre}' for a gpt2 tokenizer \
                     (supported: {}); tokenizing with another pre type's split \
                     would give wrong token ids. See docs/design/tokenizer-pre-types.md \
                     for how to add one",
                    names.join(", ")
                )
            })
    }

    /// llama.cpp's `ignore_merges` for this pre type (set in
    /// `llama_vocab::impl::load` at `79e2e74`): when a whole pre-token is a
    /// vocab entry, emit it directly instead of BPE-merging it. LLAMA3 and
    /// Tekken set it (Tekken's `tokenizer.json` has `"ignore_merges": true`).
    fn ignore_merges(self) -> bool {
        matches!(self, PreType::Llama3 | PreType::Tekken)
    }
}

/// One piece of [`Tokenizer::fragments`]: a literal special-token match, or the
/// raw text between two of them.
enum Fragment<'t> {
    Special(u32),
    Raw(&'t str),
}

/// A tokenizer extracted from a GGUF file's `tokenizer.ggml.*` metadata keys.
pub struct Tokenizer {
    pub architecture: String,
    /// The gpt2-model split ([`PreType`]); `None` for every other
    /// `tokenizer.ggml.model` (`llama` SentencePiece and `gemma4` don't
    /// pre-tokenize with a regex, and ignore `tokenizer.ggml.pre`).
    pub pre_type: Option<PreType>,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_type: Vec<i32>,
    pub bos_token_id: Option<u32>,
    /// `tokenizer.ggml.add_bos_token` -- whether a prompt gets `bos_token_id`
    /// prepended. llama.cpp's `llama_vocab::load` honors this key when present
    /// (e.g. real Qwen3.8-27B GGUFs define `bos_token_id = 248044` but set
    /// `add_bos_token = false`, so prepending it anyway diverges from
    /// llama.cpp). When the key is absent this falls back to "prepend if a BOS
    /// id exists", the behavior every earlier golden token was recorded under.
    /// Read via [`Self::prompt_bos`], not directly.
    pub add_bos: bool,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    token_to_id: FxHashMap<String, u32>,
    merge_rank: FxHashMap<(String, String), usize>,
    /// Vocab entries whose `tokenizer.ggml.token_type` is `CONTROL` (3) or
    /// `USER_DEFINED` (4) -- e.g. ChatML's `<|im_start|>`/`<|im_end|>`, or
    /// `<think>`/`</think>`. Sorted longest-first so [`Self::encode`]'s
    /// literal-match scan prefers the longest special token starting at a
    /// given position (matters when one special token's text is a prefix of
    /// another's, though no real vocab this project has seen actually has
    /// that overlap). Real tokenizers (llama.cpp, HF `tokenizers`) always
    /// match these literally before any regex-pretokenization/BPE merging,
    /// never through the general encode path -- confirmed necessary, not
    /// just a nice-to-have, by a real end-to-end regression: rendering a
    /// GGUF's own `tokenizer.chat_template` (see `sidecar/openai-adapter`)
    /// produces prompt text containing literal `<|im_start|>` substrings,
    /// and without this list, [`Self::encode_gpt2`]'s generic BPE shreds
    /// that string into 6 meaningless byte-fragment tokens
    /// (`<`/`|`/`im`/`_start`/`|`/`>`) instead of the model's one real
    /// special-token id -- garbage input the model never saw in training,
    /// which produced visibly incoherent generations on a real Qwen3-0.6B
    /// GGUF on real GPU hardware before this field/the split in `encode`
    /// existed.
    special_tokens: Vec<String>,
    /// Undo the `[` / `<` + U+200C markers a retrieval chat template puts in
    /// user text (Nemotron-3.5-Embed's `mark_user_text`): its HF tokenizer has
    /// a normalizer replacing `"[\u{200c}"` with `"["` and `"<\u{200c}"` with
    /// `"<"`, applied after special tokens are matched, so marked user text
    /// can't form a special token but still tokenizes as the plain brackets.
    /// Set for `pre = nemotron-3.5-embed` (llama.cpp's
    /// `NEMOTRON_3_5_EMBD`), and for a Tekken vocab in an embedding model
    /// (`<arch>.pooling_type` set): those files were converted before
    /// llama.cpp had the dedicated pre type.
    strip_zwnj_markers: bool,
}

impl Tokenizer {
    /// Extract a tokenizer from `file`'s metadata. Requires
    /// `tokenizer.ggml.model` and `tokenizer.ggml.tokens`; every other key
    /// (`scores`, `token_type`, `merges`, special-token ids) is optional and
    /// defaults to empty/`None` when absent.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ReflexError> {
        let architecture = string_meta(&file.metadata, "tokenizer.ggml.model")?.to_string();
        let pre = if architecture == "gpt2" {
            match file.metadata.get("tokenizer.ggml.pre") {
                Some(v) => Some(v.as_str().ok_or_else(|| {
                    crate::reflex_err!(
                        Tokenizer,
                        "metadata key 'tokenizer.ggml.pre' is not a string"
                    )
                })?),
                None => None,
            }
        } else {
            None
        };
        let pre_type = if architecture == "gpt2" {
            Some(PreType::from_gguf_value(pre)?)
        } else {
            None
        };
        let tokens = string_array(&file.metadata, "tokenizer.ggml.tokens")?;

        let scores = match file.metadata.get("tokenizer.ggml.scores") {
            Some(v) => f32_array(v, "tokenizer.ggml.scores")?,
            None => Vec::new(),
        };
        let token_type = match file.metadata.get("tokenizer.ggml.token_type") {
            Some(v) => i32_array(v, "tokenizer.ggml.token_type")?,
            None => Vec::new(),
        };
        let merges_raw = match file.metadata.get("tokenizer.ggml.merges") {
            Some(v) => string_array_value(v, "tokenizer.ggml.merges")?,
            None => Vec::new(),
        };

        let mut merge_rank =
            FxHashMap::with_capacity_and_hasher(merges_raw.len(), FxBuildHasher::default());
        for (rank, entry) in merges_raw.iter().enumerate() {
            let mut parts = entry.split(' ');
            let left = parts.next().filter(|s| !s.is_empty()).ok_or_else(|| {
                crate::reflex_err!(
                    Tokenizer,
                    "malformed merge entry '{entry}' (expected 'left right')"
                )
            })?;
            let right = parts.next().filter(|s| !s.is_empty()).ok_or_else(|| {
                crate::reflex_err!(
                    Tokenizer,
                    "malformed merge entry '{entry}' (expected 'left right')"
                )
            })?;
            merge_rank.insert((left.to_string(), right.to_string()), rank);
        }

        let token_to_id: FxHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let bos_token_id =
            u64_meta(&file.metadata, "tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let add_bos = match file.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => bos_token_id.is_some(),
        } && !PreType::value_disables_bos(pre);
        let eos_token_id =
            u64_meta(&file.metadata, "tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let unk_token_id =
            u64_meta(&file.metadata, "tokenizer.ggml.unknown_token_id").map(|v| v as u32);
        let pad_token_id =
            u64_meta(&file.metadata, "tokenizer.ggml.padding_token_id").map(|v| v as u32);

        // CONTROL (3) / USER_DEFINED (4), llama.cpp's `llama_token_type` values --
        // confirmed against a real Qwen3-0.6B GGUF (ChatML's `<|im_start|>`/
        // `<|im_end|>`/etc. are CONTROL, `<think>`/`<tool_call>`/etc. are
        // USER_DEFINED; ordinary vocab entries are NORMAL (1)).
        const TOKEN_TYPE_CONTROL: i32 = 3;
        const TOKEN_TYPE_USER_DEFINED: i32 = 4;
        let mut special_tokens: Vec<String> = tokens
            .iter()
            .zip(token_type.iter())
            .filter(|(_, &ty)| ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED)
            .map(|(t, _)| t.clone())
            .collect();
        special_tokens.sort_by_key(|b| std::cmp::Reverse(b.len()));

        let embedding_model = file
            .metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .and_then(|arch| u64_meta(&file.metadata, &format!("{arch}.pooling_type")))
            .is_some_and(|t| t != 0);
        let strip_zwnj_markers = pre == Some("nemotron-3.5-embed")
            || (pre_type == Some(PreType::Tekken) && embedding_model);

        Ok(Tokenizer {
            architecture,
            pre_type,
            tokens,
            scores,
            token_type,
            bos_token_id,
            add_bos,
            eos_token_id,
            unk_token_id,
            pad_token_id,
            token_to_id,
            merge_rank,
            special_tokens,
            strip_zwnj_markers,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The BOS id to prepend to a fresh prompt, or `None` if this vocab
    /// doesn't add one (no `bos_token_id`, or `add_bos_token = false`).
    pub fn prompt_bos(&self) -> Option<u32> {
        self.bos_token_id.filter(|_| self.add_bos)
    }

    /// Number of BPE merge rules loaded, for tests that need to assert
    /// metadata extraction happened without reaching into private fields.
    pub fn merge_count(&self) -> usize {
        self.merge_rank.len()
    }

    /// Encode `text` to token ids. Dispatches on `self.architecture`
    /// (`tokenizer.ggml.model`, real GGUF values confirmed: `"llama"`
    /// SentencePiece-derived, `"gpt2"` byte-level BPE, Phase 21.14 —
    /// see [`Self::encode_gpt2`] — and `"gemma4"`, see
    /// [`Self::encode_gemma4`]) — every other value falls back to the
    /// SentencePiece path unchanged, this module's original Phase 21.5.1
    /// scope.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ReflexError> {
        self.encode_continuation(text, true)
    }

    /// Encodes `text` as the continuation of earlier text whose encoding ended
    /// on a special token (`after_special`, also true at the very start) or on
    /// raw text. It matters only for the SentencePiece scheme: like llama.cpp's
    /// `llm_tokenizer_spm` (`is_prev_special`), every raw-text span that starts
    /// the text or follows a special token gets the dummy-prefix
    /// [`WORD_BOUNDARY`], and one that continues raw text doesn't. So encoding
    /// `a`, then `b` with `after_special = self.ends_after_special(a, true)`,
    /// gives the same ids as encoding `a + b` whenever the split falls on a
    /// token boundary. That's what resuming from a cached prefix
    /// (`Model::generate`'s `import_kv`/`use_pinned`) relies on. `encode(text)`
    /// is `encode_continuation(text, true)`.
    pub fn encode_continuation(
        &self,
        text: &str,
        after_special: bool,
    ) -> Result<Vec<u32>, ReflexError> {
        let mut ids = Vec::new();
        let mut prev_special = after_special;
        for fragment in self.fragments(text) {
            match fragment {
                Fragment::Special(id) => {
                    ids.push(id);
                    prev_special = true;
                }
                Fragment::Raw(span) => {
                    ids.extend(self.encode_plain(span, prev_special)?);
                    prev_special = false;
                }
            }
        }
        Ok(ids)
    }

    /// Whether `text`, encoded with [`Self::encode_continuation`] after a state
    /// of `after_special`, ends on a special token: the `after_special` to encode
    /// whatever comes next with. Empty text leaves the state unchanged.
    pub fn ends_after_special(&self, text: &str, after_special: bool) -> bool {
        match self.fragments(text).last() {
            Some(Fragment::Special(_)) => true,
            Some(Fragment::Raw(_)) => false,
            None => after_special,
        }
    }

    /// Splits `text` into literal special-token matches and the raw spans
    /// between them (see [`Self::encode`]'s doc for why special tokens are
    /// matched first). Raw spans are never empty.
    fn fragments<'t>(&self, text: &'t str) -> Vec<Fragment<'t>> {
        let mut out = Vec::new();
        if self.special_tokens.is_empty() {
            if !text.is_empty() {
                out.push(Fragment::Raw(text));
            }
            return out;
        }
        let mut plain_start = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            // Every special token is pure ASCII (`<|...|>`-style markers), so any
            // position one could start at is necessarily a char boundary already --
            // safe to slice `text[i..]` here without re-checking char boundaries.
            let matched = self
                .special_tokens
                .iter()
                .find(|s| text[i..].starts_with(s.as_str()));
            match matched {
                Some(special) => {
                    if plain_start < i {
                        out.push(Fragment::Raw(&text[plain_start..i]));
                    }
                    // Only reachable when `special_tokens` (built from `tokens`) is
                    // non-empty, so the lookup below always succeeds.
                    out.push(Fragment::Special(self.token_to_id[special.as_str()]));
                    i += special.len();
                    plain_start = i;
                }
                None => {
                    i += text[i..].chars().next().map(char::len_utf8).unwrap_or(1);
                }
            }
        }
        if plain_start < text.len() {
            out.push(Fragment::Raw(&text[plain_start..]));
        }
        out
    }

    /// The general encode path (regex-pretokenization + BPE, dispatched on
    /// `self.architecture`) -- used for every span of `text` that isn't a
    /// literal special-token match. See [`Self::encode`]'s doc for why
    /// special tokens (this struct's `special_tokens` field) are matched
    /// separately, before this ever runs on their text. `dummy_prefix` is the
    /// SentencePiece scheme's leading [`WORD_BOUNDARY`] (see
    /// [`Self::encode_continuation`]); the other schemes have none.
    fn encode_plain(&self, text: &str, dummy_prefix: bool) -> Result<Vec<u32>, ReflexError> {
        match self.architecture.as_str() {
            "gpt2" => self.encode_gpt2(text),
            "gemma4" => self.encode_gemma4(text),
            _ => self.encode_sentencepiece(text, dummy_prefix),
        }
    }

    /// Encode `text` to token ids, SentencePiece style.
    ///
    /// With `tokenizer.ggml.scores` present (every real SentencePiece GGUF),
    /// merging follows llama.cpp's `llm_tokenizer_spm`: see
    /// [`Self::spm_merge_by_score`]. llama.cpp ignores `tokenizer.ggml.merges`
    /// for this tokenizer type, and many GGUFs don't carry it at all
    /// (TheBloke's Mistral-7B-v0.1 has none); merging by merge rank alone then
    /// merges nothing and encodes every prompt one character per token. Only
    /// a vocab with no scores (the synthetic test vocabs) falls back to
    /// greedy lowest-rank merging over `merge_rank`.
    ///
    /// Preprocessing: every space becomes [`WORD_BOUNDARY`] and, with
    /// `dummy_prefix`, a leading `WORD_BOUNDARY` is prepended for non-empty
    /// text (SentencePiece's
    /// "text is preceded by an implicit space" convention — confirmed
    /// against the real TinyLlama reference tokenizer, see
    /// `PHASE21_5_PLAN.md`'s Known risks; empty text is the one case where
    /// the reference does *not* apply the dummy prefix, matched here by
    /// skipping it too, rather than emitting a stray leading-space token for
    /// an empty prompt). Each resulting character becomes one symbol if the
    /// vocab has it directly; otherwise it's decomposed into UTF-8 bytes,
    /// each represented by a `<0xXX>` byte-fallback token. Errs if a
    /// byte-fallback token or a final merged symbol isn't in the vocab.
    fn encode_sentencepiece(
        &self,
        text: &str,
        dummy_prefix: bool,
    ) -> Result<Vec<u32>, ReflexError> {
        let mut preprocessed = String::with_capacity(text.len() + 1);
        if dummy_prefix && !text.is_empty() {
            preprocessed.push(WORD_BOUNDARY);
        }
        for ch in text.chars() {
            preprocessed.push(if ch == ' ' { WORD_BOUNDARY } else { ch });
        }
        if !self.scores.is_empty() {
            return self.spm_merge_by_score(&preprocessed);
        }

        let mut symbols: Vec<String> = Vec::new();
        for ch in preprocessed.chars() {
            let s = ch.to_string();
            if self.token_to_id.contains_key(&s) {
                symbols.push(s);
                continue;
            }
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                let bt = byte_fallback_token(b);
                if !self.token_to_id.contains_key(&bt) {
                    return Err(crate::reflex_err!(
                        Tokenizer,
                        "character '{ch}' not in vocab and byte-fallback token '{bt}' also missing"
                    ));
                }
                symbols.push(bt);
            }
        }

        self.bpe_merge(&mut symbols);
        symbols
            .into_iter()
            .map(|s| {
                self.token_to_id.get(&s).copied().ok_or_else(|| {
                    crate::reflex_err!(Tokenizer, "merged token '{s}' not found in vocab")
                })
            })
            .collect()
    }

    /// SentencePiece merging as llama.cpp's `llm_tokenizer_spm` does it, on
    /// already-preprocessed text (spaces replaced by [`WORD_BOUNDARY`], dummy
    /// prefix added). Start from one symbol per character; repeatedly merge
    /// the adjacent pair whose concatenation is a vocab token with the highest
    /// score (ties: the leftmost pair), re-scoring the merged symbol's new
    /// neighbours each time. A final symbol that is a vocab token becomes its
    /// id; any other (a single character missing from the vocab) becomes one
    /// `<0xXX>` byte-fallback token per UTF-8 byte.
    fn spm_merge_by_score(&self, text: &str) -> Result<Vec<u32>, ReflexError> {
        struct Symbol {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl PartialEq for Bigram {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == std::cmp::Ordering::Equal
            }
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Bigram {
            // Max-heap: highest score first, then the leftmost pair.
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score
                    .total_cmp(&other.score)
                    .then_with(|| other.left.cmp(&self.left))
            }
        }

        let mut symbols: Vec<Symbol> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, ch))| Symbol {
                start,
                len: ch.len_utf8(),
                prev: i.checked_sub(1),
                next: None,
            })
            .collect();
        let n = symbols.len();
        for (i, sym) in symbols.iter_mut().enumerate() {
            sym.next = (i + 1 < n).then_some(i + 1);
        }

        let mut heap = std::collections::BinaryHeap::new();
        let try_add = |heap: &mut std::collections::BinaryHeap<Bigram>,
                       symbols: &[Symbol],
                       left: usize,
                       right: usize| {
            let piece = &text[symbols[left].start..symbols[right].start + symbols[right].len];
            if let Some(&id) = self.token_to_id.get(piece) {
                heap.push(Bigram {
                    score: self.scores.get(id as usize).copied().unwrap_or(0.0),
                    left,
                    right,
                    size: piece.len(),
                });
            }
        };
        for i in 1..n {
            try_add(&mut heap, &symbols, i - 1, i);
        }
        while let Some(b) = heap.pop() {
            let (l, r) = (b.left, b.right);
            // Stale entry: one side was already merged away, or grew since.
            if symbols[l].len == 0
                || symbols[r].len == 0
                || symbols[l].len + symbols[r].len != b.size
            {
                continue;
            }
            symbols[l].len += symbols[r].len;
            symbols[r].len = 0;
            symbols[l].next = symbols[r].next;
            if let Some(next) = symbols[r].next {
                symbols[next].prev = Some(l);
            }
            if let Some(prev) = symbols[l].prev {
                try_add(&mut heap, &symbols, prev, l);
            }
            if let Some(next) = symbols[l].next {
                try_add(&mut heap, &symbols, l, next);
            }
        }

        let mut ids = Vec::new();
        let mut cur = (n > 0).then_some(0);
        while let Some(k) = cur {
            let piece = &text[symbols[k].start..symbols[k].start + symbols[k].len];
            match self.token_to_id.get(piece) {
                Some(&id) => ids.push(id),
                None => {
                    for &byte in piece.as_bytes() {
                        let bt = byte_fallback_token(byte);
                        let id = self.token_to_id.get(&bt).copied().ok_or_else(|| {
                            crate::reflex_err!(
                                Tokenizer,
                                "'{piece}' not in vocab and byte-fallback token '{bt}' also missing"
                            )
                        })?;
                        ids.push(id);
                    }
                }
            }
            cur = symbols[k].next;
        }
        Ok(ids)
    }

    /// Encode `text` to token ids via GPT-2-style byte-level BPE (Phase
    /// 21.14, Qwen3 architecture support — real `tokenizer.ggml.model =
    /// "gpt2"` / `tokenizer.ggml.pre = "qwen2"`, confirmed against a real
    /// Qwen3-0.6B GGUF's own metadata). Two real differences from
    /// [`Self::encode_sentencepiece`], not a small variation of it:
    ///
    /// 1. **Regex-based pre-tokenization** ([`pretokenize`]) splits
    ///    `text` into maximal chunks *before* any BPE merging happens — a
    ///    merge can never cross a pre-token boundary. Ported by hand (no
    ///    `regex` crate dependency, matching this crate's established
    ///    from-scratch-small-helper convention, e.g. [`Xorshift64`] over the
    ///    `rand` crate) from llama.cpp's splitters, e.g. the
    ///    `LLAMA_VOCAB_PRE_TYPE_QWEN2` pattern:
    ///    `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
    /// 2. **Byte-to-unicode remapping** ([`gpt2_byte_to_unicode_table`]):
    ///    every raw UTF-8 byte of each pre-token is remapped through a fixed
    ///    256-entry alphabet (OpenAI's original `bytes_to_unicode()`
    ///    construction) *before* BPE merging — unlike SentencePiece's
    ///    `▁`-marker-plus-`<0xXX>`-fallback scheme, every byte always has a
    ///    representation this way, so there's no separate byte-fallback path.
    ///
    /// The merge step itself ([`Self::bpe_merge`]) is the same lowest-rank
    /// adjacent-pair greedy algorithm as the SentencePiece path — genuine
    /// BPE merging doesn't differ between the two schemes, only what the
    /// initial symbols are and how text gets split into independently-merged
    /// chunks first.
    ///
    /// Step 1's split depends on `tokenizer.ggml.pre` ([`PreType`], see
    /// [`pretokenize`]). Every split is checked against llama.cpp's
    /// `llama-tokenize`.
    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>, ReflexError> {
        let chars: Vec<char> = if self.strip_zwnj_markers {
            text.replace("[\u{200c}", "[")
                .replace("<\u{200c}", "<")
                .chars()
                .collect()
        } else {
            text.chars().collect()
        };
        let pre_type = self.pre_type.unwrap_or(PreType::Qwen2);
        let mut ids = Vec::new();
        for (start, end) in pretokenize(pre_type, &chars) {
            self.encode_gpt2_piece(&chars[start..end], pre_type.ignore_merges(), &mut ids)?;
        }
        Ok(ids)
    }

    /// Byte-remap one pre-token and BPE-merge it into `ids`. With
    /// `ignore_merges` (llama.cpp's flag of that name, see
    /// [`PreType::ignore_merges`]) a pre-token that is already a vocab entry
    /// is emitted as that one id without merging.
    fn encode_gpt2_piece(
        &self,
        piece: &[char],
        ignore_merges: bool,
        ids: &mut Vec<u32>,
    ) -> Result<(), ReflexError> {
        let byte_to_unicode = gpt2_byte_to_unicode_table();
        let piece: String = piece.iter().collect();
        if ignore_merges {
            let word: String = piece.bytes().map(|b| byte_to_unicode[b as usize]).collect();
            if let Some(&id) = self.token_to_id.get(&word) {
                ids.push(id);
                return Ok(());
            }
        }

        let mut symbols: Vec<String> = Vec::with_capacity(piece.len());
        for &b in piece.as_bytes() {
            symbols.push(byte_to_unicode[b as usize].to_string());
        }
        self.bpe_merge(&mut symbols);
        for s in symbols {
            let id = self.token_to_id.get(&s).copied().ok_or_else(|| {
                crate::reflex_err!(
                    Tokenizer,
                    "gpt2 BPE symbol {s:?} not found in vocab (piece {piece:?})"
                )
            })?;
            ids.push(id);
        }
        Ok(())
    }

    /// Greedy lowest-rank adjacent-pair BPE merging, shared by
    /// [`Self::encode_sentencepiece`] and [`Self::encode_gpt2`] — genuine
    /// BPE merging is identical between the two schemes; only how the
    /// initial `symbols` are produced (and, for gpt2, how `text` gets split
    /// into independently-merged pieces first) differs.
    fn bpe_merge(&self, symbols: &mut Vec<String>) {
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index of left symbol)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..=i + 1, [merged]);
        }
    }

    /// Encode `text` for `tokenizer.ggml.model == "gemma4"`: BPE over raw
    /// UTF-8 with SentencePiece-style spaces, as llama.cpp's
    /// `LLAMA_VOCAB_PRE_TYPE_GEMMA4` does it (`llama-vocab.cpp`):
    ///
    /// 1. Every space becomes [`WORD_BOUNDARY`]. No dummy prefix is added
    ///    (`add_space_prefix` is false for this vocab).
    /// 2. The text is split into runs of newlines and runs of everything else
    ///    (regex `[^\n]+|[\n]+`). Real Gemma 4 vocabs have no merge or token
    ///    that mixes a newline with any other character, so this split never
    ///    separates anything a merge could have joined.
    /// 3. A newline run that is itself a vocab token becomes that token.
    ///    Every other run starts as one symbol per character and is merged by
    ///    merge rank ([`Self::bpe_merge_ranked`]).
    /// 4. A final symbol that is not a vocab token becomes one `<0xXX>`
    ///    byte-fallback token per UTF-8 byte.
    ///
    /// Unlike the gpt2 path there is no word-level pre-split on spaces, so a
    /// whole line is one BPE word; hence the heap-based merge rather than
    /// [`Self::bpe_merge`]'s quadratic rescans. Verified id-for-id against HF
    /// `tokenizers` on Gemma 4's `tokenizer.json` (see the
    /// `test_gemma4_encode_matches_reference_if_fixture_present` test).
    fn encode_gemma4(&self, text: &str) -> Result<Vec<u32>, ReflexError> {
        let escaped: String = text
            .chars()
            .map(|ch| if ch == ' ' { WORD_BOUNDARY } else { ch })
            .collect();
        let mut ids = Vec::new();
        let mut rest = escaped.as_str();
        while let Some(first) = rest.chars().next() {
            let is_newline_run = first == '\n';
            let len = rest
                .find(|c: char| (c == '\n') != is_newline_run)
                .unwrap_or(rest.len());
            let (word, tail) = rest.split_at(len);
            rest = tail;

            if is_newline_run {
                if let Some(&id) = self.token_to_id.get(word) {
                    ids.push(id);
                    continue;
                }
            }
            for symbol in self.bpe_merge_ranked(word) {
                if let Some(&id) = self.token_to_id.get(symbol) {
                    ids.push(id);
                    continue;
                }
                for &b in symbol.as_bytes() {
                    let bt = byte_fallback_token(b);
                    let id = self.token_to_id.get(&bt).copied().ok_or_else(|| {
                        crate::reflex_err!(
                            Tokenizer,
                            "gemma4 symbol {symbol:?} not in vocab and byte-fallback token '{bt}' also missing"
                        )
                    })?;
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// Lowest-rank-first BPE over `word`, starting from one symbol per
    /// character, returned as slices of `word` in order. Same result as
    /// [`Self::bpe_merge`] (lowest rank wins, ties go to the leftmost pair),
    /// but with llama.cpp's `llm_tokenizer_bpe_session` structure: a linked
    /// list of symbols plus a min-heap of candidate pairs, where a popped pair
    /// is skipped if either side has changed since it was pushed. That keeps
    /// a long word at O(n log n) instead of a full rescan per merge.
    fn bpe_merge_ranked<'a>(&self, word: &'a str) -> Vec<&'a str> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        struct Symbol {
            start: usize,
            end: usize,
            prev: Option<usize>,
            next: Option<usize>,
            alive: bool,
        }
        let n = word.chars().count();
        let mut symbols: Vec<Symbol> = word
            .char_indices()
            .enumerate()
            .map(|(i, (start, ch))| Symbol {
                start,
                end: start + ch.len_utf8(),
                prev: i.checked_sub(1),
                next: (i + 1 < n).then_some(i + 1),
                alive: true,
            })
            .collect();

        // Heap entries: (rank, left index, left end, right end). The two ends
        // identify the exact pair the rank was looked up for.
        let mut heap: BinaryHeap<Reverse<(usize, usize, usize, usize)>> = BinaryHeap::new();
        let push_pair = |heap: &mut BinaryHeap<_>, symbols: &[Symbol], left: usize| {
            let Some(right) = symbols[left].next else {
                return;
            };
            let (l, r) = (&symbols[left], &symbols[right]);
            let key = (
                word[l.start..l.end].to_string(),
                word[r.start..r.end].to_string(),
            );
            if let Some(&rank) = self.merge_rank.get(&key) {
                heap.push(Reverse((rank, left, l.end, r.end)));
            }
        };
        for i in 0..n.saturating_sub(1) {
            push_pair(&mut heap, &symbols, i);
        }

        while let Some(Reverse((_, left, left_end, right_end))) = heap.pop() {
            let l = &symbols[left];
            if !l.alive || l.end != left_end {
                continue;
            }
            let Some(right) = l.next else { continue };
            if symbols[right].end != right_end {
                continue;
            }
            let after = symbols[right].next;
            symbols[left].end = right_end;
            symbols[left].next = after;
            symbols[right].alive = false;
            if let Some(a) = after {
                symbols[a].prev = Some(left);
            }
            if let Some(p) = symbols[left].prev {
                push_pair(&mut heap, &symbols, p);
            }
            push_pair(&mut heap, &symbols, left);
        }

        let mut out = Vec::new();
        let mut cur = (n > 0).then_some(0);
        while let Some(i) = cur {
            out.push(&word[symbols[i].start..symbols[i].end]);
            cur = symbols[i].next;
        }
        out
    }

    /// Decode token ids back to text. Dispatches on `self.architecture`,
    /// same convention as [`Self::encode`].
    pub fn decode(&self, ids: &[u32]) -> String {
        if self.architecture == "gpt2" {
            self.decode_gpt2(ids)
        } else {
            self.decode_sentencepiece(ids)
        }
    }

    /// One token id's raw decoded bytes, before any word-boundary-marker
    /// substitution or UTF-8 assembly -- the unit both [`Self::decode`]
    /// (which loops this over a whole id slice and assembles once at the
    /// end) and [`Self::decode_stream`] (which needs it one id at a time,
    /// for streaming) build on. Dispatches on `self.architecture`, same
    /// convention as [`Self::decode`]/[`Self::encode`]. Out-of-range ids
    /// produce no bytes (not an error), matching this being a best-effort
    /// decode over whatever ids a caller has.
    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.architecture == "gpt2" {
            let unicode_to_byte = gpt2_unicode_to_byte_table();
            tok.chars()
                .filter_map(|ch| unicode_to_byte.get(&ch).copied())
                .collect()
        } else if let Some(byte) = parse_byte_fallback(tok) {
            vec![byte]
        } else {
            tok.as_bytes().to_vec()
        }
    }

    /// The bytes token `id` adds to [`Self::decode`]'s output: [`Self::token_bytes`],
    /// with the SentencePiece word-boundary marker already turned into a space for
    /// every scheme but gpt2. `crate::grammar` matches these against the schema. The
    /// one difference from `decode` is a `▁` assembled from byte-fallback tokens
    /// across several ids, which `decode` also turns into a space; the grammar keeps
    /// U+2581 out of its strings for these tokenizers, so it never builds one.
    pub(crate) fn token_text_bytes(&self, id: u32) -> Vec<u8> {
        let bytes = self.token_bytes(id);
        if self.architecture == "gpt2" || parse_byte_fallback(&self.tokens[id as usize]).is_some() {
            return bytes;
        }
        match String::from_utf8(bytes) {
            Ok(s) => s.replace(WORD_BOUNDARY, " ").into_bytes(),
            Err(e) => e.into_bytes(),
        }
    }

    /// Whether token `id` stands for text a constrained value may contain: not a
    /// `CONTROL` (3), `UNKNOWN` (2) or `UNUSED` (5) token per `tokenizer.ggml.token_type`
    /// (or, when the vocab has no types, not the BOS/EOS/unknown/padding id).
    /// `USER_DEFINED` tokens count as text, as they do in llama.cpp's grammar sampler.
    pub(crate) fn is_text_token(&self, id: u32) -> bool {
        if (id as usize) >= self.tokens.len() {
            return false;
        }
        match self.token_type.get(id as usize) {
            Some(&t) => !matches!(t, 2 | 3 | 5),
            None => ![
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
                self.pad_token_id,
            ]
            .contains(&Some(id)),
        }
    }

    /// Decode token ids back to text: byte-fallback tokens (`<0xXX>`)
    /// contribute one raw byte each, every other token contributes its
    /// UTF-8 bytes verbatim, then [`WORD_BOUNDARY`] is replaced with a
    /// literal space. Out-of-range ids are skipped (not an error), matching
    /// this being a best-effort decode over whatever ids a caller has.
    fn decode_sentencepiece(&self, ids: &[u32]) -> String {
        let mut byte_buf: Vec<u8> = Vec::new();
        for &id in ids {
            byte_buf.extend(self.token_bytes(id));
        }
        String::from_utf8_lossy(&byte_buf).replace(WORD_BOUNDARY, " ")
    }

    /// Incremental, one-token-at-a-time counterpart to [`Self::decode`] --
    /// for `reflex stdio`/`reflex uds`'s per-token streaming IPC hook
    /// (`crate::ipc::handle_request_streaming`), which needs to emit each
    /// generated token's text as soon as it's produced, not buffered until
    /// the whole response is ready. Appends `id`'s raw bytes
    /// ([`Self::token_bytes`]) to `pending`, then returns the longest valid-
    /// UTF-8 prefix as text (word-boundary-replaced for the SentencePiece
    /// scheme, same as [`Self::decode`]), leaving any incomplete trailing
    /// multi-byte sequence in `pending` for the next call. This avoids
    /// emitting a U+FFFD replacement character for a multi-byte character
    /// split across two generated tokens -- a real streaming concern
    /// `decode`'s single-shot, whole-buffer `from_utf8_lossy` never has to
    /// deal with, since it only ever runs once the full id sequence is
    /// already in hand. `pending` starts empty (`Vec::new()`) at the
    /// beginning of a generation run and is threaded through every
    /// `decode_stream` call for that run.
    pub fn decode_stream(&self, pending: &mut Vec<u8>, id: u32) -> String {
        pending.extend(self.token_bytes(id));
        let valid_up_to = match std::str::from_utf8(pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_up_to == 0 {
            return String::new();
        }
        let remainder = pending.split_off(valid_up_to);
        let valid_bytes = std::mem::replace(pending, remainder);
        let text = String::from_utf8(valid_bytes)
            .expect("valid_up_to is exactly the longest valid-UTF-8 prefix");
        if self.architecture == "gpt2" {
            text
        } else {
            text.replace(WORD_BOUNDARY, " ")
        }
    }

    /// Decode token ids back to text for the gpt2/byte-level-BPE scheme:
    /// each token's own string is composed of [`gpt2_byte_to_unicode_table`]
    /// alphabet characters, not raw text — the inverse table
    /// ([`gpt2_unicode_to_byte_table`]) recovers the original byte per
    /// character, and the collected byte buffer is UTF-8-decoded once at the
    /// end (mirroring [`Self::decode_sentencepiece`]'s own byte-buffer
    /// pattern). Out-of-range ids and any character missing from the inverse
    /// table are skipped, not an error — same best-effort posture as the
    /// SentencePiece path.
    fn decode_gpt2(&self, ids: &[u32]) -> String {
        let mut byte_buf: Vec<u8> = Vec::new();
        for &id in ids {
            byte_buf.extend(self.token_bytes(id));
        }
        String::from_utf8_lossy(&byte_buf).into_owned()
    }
}

/// OpenAI's original GPT-2 `bytes_to_unicode()` construction, ported
/// byte-for-byte: bytes in the three "already printable" Latin-1 ranges
/// (`!`..`~`, `¡`..`¬`, `®`..`ÿ`) map to themselves; every other byte (control
/// characters, space, DEL, and a few Latin-1 gaps) maps to a new codepoint
/// starting at 256, assigned in ascending byte order. Cached in a
/// process-wide [`std::sync::OnceLock`] — this crate's established pattern
/// for compute-once, read-many host-side tables — rather than the `once_cell`/
/// `lazy_static` crates this project doesn't otherwise depend on.
fn gpt2_byte_to_unicode_table() -> &'static [char; 256] {
    static TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut assigned = [false; 256];
        let mut codepoint = [0u32; 256];
        for b in 0x21u32..=0x7E {
            assigned[b as usize] = true;
            codepoint[b as usize] = b;
        }
        for b in 0xA1u32..=0xAC {
            assigned[b as usize] = true;
            codepoint[b as usize] = b;
        }
        for b in 0xAEu32..=0xFF {
            assigned[b as usize] = true;
            codepoint[b as usize] = b;
        }
        let mut next_extra = 256u32;
        for b in 0..256usize {
            if !assigned[b] {
                codepoint[b] = next_extra;
                next_extra += 1;
            }
        }
        let mut table = ['\0'; 256];
        for b in 0..256usize {
            table[b] = char::from_u32(codepoint[b])
                .expect("gpt2 byte-to-unicode codepoints are all valid scalar values");
        }
        table
    })
}

/// Inverse of [`gpt2_byte_to_unicode_table`], for [`Tokenizer::decode_gpt2`].
fn gpt2_unicode_to_byte_table() -> &'static HashMap<char, u8> {
    static TABLE: std::sync::OnceLock<HashMap<char, u8>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        gpt2_byte_to_unicode_table()
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Pre-tokenizer splits, one per [`PreType`].
//
// Hand-ported (no `regex` crate) from llama.cpp `79e2e74`: the regex lists in
// `src/llama-vocab.cpp`'s `llm_tokenizer_bpe` constructor, run the way
// `src/unicode.cpp`'s `unicode_regex_split` runs them. They classify
// codepoints with llama.cpp's own Unicode tables (`crate::tokenizer_unicode`,
// generated from `src/unicode-data.cpp`) instead of Rust's `is_alphabetic`/
// `is_numeric`/`is_whitespace`, which disagree with them on combining marks,
// and std has no `\p{P}`/`\p{S}`/`\p{M}` at all. Two execution models, both
// llama.cpp's:
// - `qwen2`/`qwen35`/`llama-bpe`/`gpt-2` have a hand-written splitter in
//   llama.cpp (`unicode_regex_split_custom_{qwen2,qwen35,llama3,gpt2}`), ported line for line
//   by [`split_custom`] + a step function. An unmatched codepoint becomes a
//   one-codepoint piece.
// - `deepseek-llm`/`deepseek-v3` are lists of regexes llama.cpp runs through
//   `std::regex`, each one over every piece the previous one produced
//   ([`split_stl`], `unicode_regex_split_stl`). Text between matches stays as
//   its own piece. Regexes using `\p{..}` run on llama.cpp's "collapsed" text
//   (each non-ASCII codepoint replaced by a byte standing for its category),
//   the others on the codepoints with every non-ASCII whitespace codepoint
//   replaced by `\x0B`. Either way `\s` means llama.cpp's whitespace set and
//   `\p{X}` means its category table, which is what the matchers below test.
// ---------------------------------------------------------------------------

/// llama.cpp's `unicode_cpt_flags` for one codepoint: its category bit
/// (`tokenizer_unicode::LETTER` etc.) plus `is_whitespace`. The default
/// (all zero) is what llama.cpp's splitters read past the end of a piece.
#[derive(Clone, Copy, Default)]
struct CptFlags {
    category: u8,
    whitespace: bool,
}

impl CptFlags {
    fn of(c: char) -> Self {
        use crate::tokenizer_unicode::{CATEGORY_RANGES, WHITESPACE};
        let cp = c as u32;
        let idx = CATEGORY_RANGES.partition_point(|&(start, _)| start <= cp) - 1;
        CptFlags {
            category: CATEGORY_RANGES[idx].1,
            whitespace: WHITESPACE.binary_search(&cp).is_ok(),
        }
    }
    fn is_letter(self) -> bool {
        self.category == crate::tokenizer_unicode::LETTER
    }
    fn is_number(self) -> bool {
        self.category == crate::tokenizer_unicode::NUMBER
    }
    fn is_accent_mark(self) -> bool {
        self.category == crate::tokenizer_unicode::ACCENT_MARK
    }
    fn is_punctuation(self) -> bool {
        self.category == crate::tokenizer_unicode::PUNCTUATION
    }
    fn is_symbol(self) -> bool {
        self.category == crate::tokenizer_unicode::SYMBOL
    }
    /// llama.cpp's `flags.as_uint() != 0`: true for every codepoint inside
    /// the piece (each has a category bit, `UNDEFINED` included), false past
    /// its end.
    fn in_piece(self) -> bool {
        self.category != 0 || self.whitespace
    }
}

/// Flags of `chars[pos]`, or all-zero at/after `end` (llama.cpp's
/// `_get_flags`, which reads nothing outside the current piece).
fn flags_at(chars: &[char], end: usize, pos: usize) -> CptFlags {
    if pos < end {
        CptFlags::of(chars[pos])
    } else {
        CptFlags::default()
    }
}

/// `chars[pos]`, or `None` at/after `end` (llama.cpp's `_get_cpt`, whose
/// `OUT_OF_RANGE` never equals a real codepoint).
fn cpt_at(chars: &[char], end: usize, pos: usize) -> Option<char> {
    (pos < end).then(|| chars[pos])
}

/// Split `chars` into `(start, end)` pre-token ranges for `pre_type`.
fn pretokenize(pre_type: PreType, chars: &[char]) -> Vec<(usize, usize)> {
    if chars.is_empty() {
        return Vec::new();
    }
    let whole = vec![(0, chars.len())];
    match pre_type {
        PreType::Qwen2 => split_custom(chars, &whole, |c, e, p| {
            llama3_family_step(c, e, p, 1, false)
        }),
        PreType::Qwen35 => split_custom(chars, &whole, |c, e, p| {
            llama3_family_step(c, e, p, 1, true)
        }),
        PreType::Llama3 | PreType::Dbrx => split_custom(chars, &whole, |c, e, p| {
            llama3_family_step(c, e, p, 3, false)
        }),
        PreType::Gpt2 => split_custom(chars, &whole, gpt2_step),
        PreType::DeepseekLlm => {
            // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM, in order:
            //   "[\r\n]"
            //   "\\s?[A-Za-zµÀ-ÖØ-öø-ƺ...𞤀-𞥃]+"    (DEEPSEEK_LLM_LATIN)
            //   "\\s?[!-/:-~！-／：-～‘-‟　-。]+"   (DEEPSEEK_LLM_PUNCT)
            //   "\\s+$"
            //   "[一-龥ࠀ-一가-퟿]+"                  (DEEPSEEK_LLM_CJK)
            //   "\\p{N}+"
            let classes = deepseek_classes();
            let mut pieces = split_stl(chars, &whole, |c, _, p| {
                matches!(c[p], '\r' | '\n').then_some(p + 1)
            });
            pieces = split_stl(chars, &pieces, |c, e, p| {
                match_ws_opt_class_run(c, e, p, &classes.llm_latin)
            });
            pieces = split_stl(chars, &pieces, |c, e, p| {
                match_ws_opt_class_run(c, e, p, &classes.llm_punct)
            });
            pieces = split_stl(chars, &pieces, |c, e, p| {
                let run = ws_run_end(c, e, p);
                (run > p && run == e).then_some(e)
            });
            pieces = split_stl(chars, &pieces, |c, e, p| {
                match_class_run(c, e, p, &classes.llm_cjk)
            });
            split_stl(chars, &pieces, |c, e, p| {
                let run = run_end(c, e, p, |f| f.is_number());
                (run > p).then_some(run)
            })
        }
        PreType::Deepseek3 => {
            // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM, in order:
            //   "\\p{N}{1,3}"
            //   "[一-龥぀-ゟ゠-ヿ]+"                  (DEEPSEEK_V3_CJK)
            //   "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
            let classes = deepseek_classes();
            let mut pieces = split_stl(chars, &whole, |c, e, p| {
                let run = run_end(c, e, p, |f| f.is_number());
                (run > p).then_some(run.min(p + 3))
            });
            pieces = split_stl(chars, &pieces, |c, e, p| {
                match_class_run(c, e, p, &classes.v3_cjk)
            });
            split_stl(chars, &pieces, deepseek3_main_match)
        }
        PreType::Tekken => split_stl(chars, &whole, tekken_match),
    }
}

/// Run `step` over each piece in `pieces` the way llama.cpp's
/// `unicode_regex_split_custom_*` functions do: `step(chars, piece_end, pos)`
/// returns the end of the pre-token starting at `pos` (always `> pos`).
fn split_custom(
    chars: &[char],
    pieces: &[(usize, usize)],
    step: impl Fn(&[char], usize, usize) -> usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &(start, end) in pieces {
        let mut pos = start;
        while pos < end {
            let next = step(chars, end, pos);
            debug_assert!(next > pos && next <= end);
            out.push((pos, next));
            pos = next;
        }
    }
    out
}

/// Run one regex over each piece in `pieces` the way llama.cpp's
/// `unicode_regex_split_stl` does with `std::regex_iterator`: scan for the
/// leftmost match, emit any unmatched text before it as its own piece, then
/// the match, and continue after it. `matcher(chars, piece_end, pos)` returns
/// the end of the regex's match starting exactly at `pos` (leftmost
/// alternative, greedy with backtracking, ECMAScript semantics), or `None`.
/// The piece end acts as end of input for `$` and lookaheads, as the
/// iterator's sub-range does. None of the ported regexes can match empty.
fn split_stl(
    chars: &[char],
    pieces: &[(usize, usize)],
    matcher: impl Fn(&[char], usize, usize) -> Option<usize>,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &(start, end) in pieces {
        let mut gap = start;
        let mut pos = start;
        while pos < end {
            match matcher(chars, end, pos) {
                Some(m) => {
                    debug_assert!(m > pos && m <= end);
                    if gap < pos {
                        out.push((gap, pos));
                    }
                    out.push((pos, m));
                    pos = m;
                    gap = m;
                }
                None => pos += 1,
            }
        }
        if gap < end {
            out.push((gap, end));
        }
    }
    out
}

/// End of the run of codepoints from `pos` (before `end`) whose flags pass `f`.
fn run_end(chars: &[char], end: usize, pos: usize, f: impl Fn(CptFlags) -> bool) -> usize {
    let mut j = pos;
    while j < end && f(CptFlags::of(chars[j])) {
        j += 1;
    }
    j
}

/// End of the `\s` run from `pos`. llama.cpp maps every non-ASCII whitespace
/// codepoint to `\x0B` before `std::regex` sees it, so `\s` is exactly its
/// whitespace set.
fn ws_run_end(chars: &[char], end: usize, pos: usize) -> usize {
    run_end(chars, end, pos, |f| f.whitespace)
}

/// `unicode_regex_split_custom_llama3` (`digits = 3`, `marks = false`),
/// `unicode_regex_split_custom_qwen2` (`digits = 1`, `marks = false`) and
/// `unicode_regex_split_custom_qwen35` (`digits = 1`, `marks = true`), one
/// pre-token per call. Same structure as llama.cpp's, line for line (at
/// `79e2e74` the three functions differ only in those two places):
///
/// ```text
/// llama3: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// qwen2:  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// qwen35: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// llama.cpp groups a digit run into threes inside one call; returning one
/// group per call is equivalent, since the next call starts on a digit and
/// reaches the same branch. The contraction check lowercases with ASCII
/// rules: in llama.cpp's `unicode_map_lowercase` no non-ASCII codepoint
/// lowercases to any of `s t m d r e v l`.
fn llama3_family_step(chars: &[char], end: usize, pos: usize, digits: usize, marks: bool) -> usize {
    let cpt = chars[pos];
    let flags = CptFlags::of(cpt);
    let is_word = |f: CptFlags| f.is_letter() || (marks && f.is_accent_mark());

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if cpt == '\'' && pos + 1 < end {
        let next = chars[pos + 1].to_ascii_lowercase();
        if matches!(next, 's' | 't' | 'm' | 'd') {
            return pos + 2;
        }
        if pos + 2 < end {
            let next2 = chars[pos + 2].to_ascii_lowercase();
            if matches!((next, next2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return pos + 3;
            }
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+   (qwen35: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+)
    if !(cpt == '\r' || cpt == '\n' || flags.is_number())
        && (is_word(flags) || is_word(flags_at(chars, end, pos + 1)))
    {
        let mut j = pos + 1;
        while is_word(flags_at(chars, end, j)) {
            j += 1;
        }
        return j;
    }

    // \p{N}{1,3}   (qwen35: \p{N})
    if flags.is_number() {
        return run_end(chars, end, pos, |f| f.is_number()).min(pos + digits);
    }

    // <space>?[^\s\p{L}\p{N}]+[\r\n]*   (qwen35 also excludes \p{M})
    let not_ws_word_num = |f: CptFlags| {
        !(f.whitespace || f.is_letter() || f.is_number() || (marks && f.is_accent_mark()))
    };
    let flags2 = if cpt == ' ' {
        flags_at(chars, end, pos + 1)
    } else {
        flags
    };
    if not_ws_word_num(flags2) && flags.in_piece() {
        let mut j = pos + usize::from(cpt == ' ');
        while {
            let f = flags_at(chars, end, j);
            not_ws_word_num(f) && f.in_piece()
        } {
            j += 1;
        }
        while matches!(cpt_at(chars, end, j), Some('\r' | '\n')) {
            j += 1;
        }
        return j;
    }

    whitespace_tail_step(chars, end, pos, true)
}

/// The shared whitespace tail of llama.cpp's custom splitters:
/// `\s*[\r\n]+` (only when `blank_lines`; GPT-2's regex has no such
/// alternative), then `\s+(?!\S)`, then `\s+`, else a one-codepoint
/// "no match" piece.
fn whitespace_tail_step(chars: &[char], end: usize, pos: usize, blank_lines: bool) -> usize {
    let mut num_whitespaces = 0;
    let mut last_end_r_or_n = 0;
    while flags_at(chars, end, pos + num_whitespaces).whitespace {
        if matches!(chars[pos + num_whitespaces], '\r' | '\n') {
            last_end_r_or_n = pos + num_whitespaces + 1;
        }
        num_whitespaces += 1;
    }
    // \s*[\r\n]+
    if blank_lines && last_end_r_or_n > 0 {
        return last_end_r_or_n;
    }
    // \s+(?!\S)
    if num_whitespaces > 1 && pos + num_whitespaces < end {
        return pos + num_whitespaces - 1;
    }
    // \s+
    if num_whitespaces > 0 {
        return pos + num_whitespaces;
    }
    // no match
    pos + 1
}

/// `unicode_regex_split_custom_gpt2`, one pre-token per call:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
/// (contractions are case-sensitive here, unlike the llama3 family's).
fn gpt2_step(chars: &[char], end: usize, pos: usize) -> usize {
    let cpt = chars[pos];
    let flags = CptFlags::of(cpt);

    // 's|'t|'re|'ve|'m|'ll|'d
    if cpt == '\'' && pos + 1 < end {
        let next = chars[pos + 1];
        if matches!(next, 's' | 't' | 'm' | 'd') {
            return pos + 2;
        }
        if pos + 2 < end && matches!((next, chars[pos + 2]), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return pos + 3;
        }
    }

    let flags2 = if cpt == ' ' {
        flags_at(chars, end, pos + 1)
    } else {
        flags
    };
    let start = pos + usize::from(cpt == ' ');
    // <space>?\p{L}+
    if flags2.is_letter() {
        return run_end(chars, end, start, |f| f.is_letter());
    }
    // <space>?\p{N}+
    if flags2.is_number() {
        return run_end(chars, end, start, |f| f.is_number());
    }
    // <space>?[^\s\p{L}\p{N}]+
    let is_other = |f: CptFlags| !(f.whitespace || f.is_letter() || f.is_number()) && f.in_piece();
    if is_other(flags2) {
        return run_end(chars, end, start, is_other);
    }

    whitespace_tail_step(chars, end, pos, false)
}

/// The DeepSeek character classes, verbatim from llama.cpp `79e2e74`
/// `src/llama-vocab.cpp` (the text between `[` and `]` of each regex), so a
/// diff against upstream is a plain string comparison. Parsed into
/// codepoint ranges once ([`deepseek_classes`]).
const DEEPSEEK_LLM_LATIN: &str = "A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-\u{1F7D}ᾀ-ᾴᾶ-ᾼ\u{1FBE}ῂ-ῄῆ-ῌῐ-\u{1FD3}ῖ-\u{1FDB}ῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤ\u{2126}ℨ\u{212A}-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃";
const DEEPSEEK_LLM_PUNCT: &str = "!-/:-~！-／：-～‘-‟　-。";
const DEEPSEEK_LLM_CJK: &str = "一-龥ࠀ-一가-퟿";
const DEEPSEEK_V3_CJK: &str = "一-龥぀-ゟ゠-ヿ";

struct DeepseekClasses {
    llm_latin: Vec<(u32, u32)>,
    llm_punct: Vec<(u32, u32)>,
    llm_cjk: Vec<(u32, u32)>,
    v3_cjk: Vec<(u32, u32)>,
}

fn deepseek_classes() -> &'static DeepseekClasses {
    static CLASSES: std::sync::OnceLock<DeepseekClasses> = std::sync::OnceLock::new();
    CLASSES.get_or_init(|| DeepseekClasses {
        llm_latin: parse_char_class(DEEPSEEK_LLM_LATIN),
        llm_punct: parse_char_class(DEEPSEEK_LLM_PUNCT),
        llm_cjk: parse_char_class(DEEPSEEK_LLM_CJK),
        v3_cjk: parse_char_class(DEEPSEEK_V3_CJK),
    })
}

/// Parse a bracket-expression body of single codepoints and `a-b` ranges (no
/// escapes; none of the DeepSeek classes has any) into inclusive ranges.
fn parse_char_class(body: &str) -> Vec<(u32, u32)> {
    let cps: Vec<u32> = body.chars().map(|c| c as u32).collect();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < cps.len() {
        if i + 2 < cps.len() && cps[i + 1] == '-' as u32 {
            ranges.push((cps[i], cps[i + 2]));
            i += 3;
        } else {
            ranges.push((cps[i], cps[i]));
            i += 1;
        }
    }
    ranges
}

/// Whether `c` is in `class`, as `std::wregex` sees it: llama.cpp replaces
/// every non-ASCII whitespace codepoint with `\x0B` first, so e.g. U+3000
/// (inside `　-。`) never matches a class, only `\s`.
fn in_class(c: char, class: &[(u32, u32)]) -> bool {
    let cp = c as u32;
    let cp = if cp > 0x7F && CptFlags::of(c).whitespace {
        0x0B
    } else {
        cp
    };
    class.iter().any(|&(lo, hi)| lo <= cp && cp <= hi)
}

/// `[class]+`
fn match_class_run(chars: &[char], end: usize, pos: usize, class: &[(u32, u32)]) -> Option<usize> {
    let mut j = pos;
    while j < end && in_class(chars[j], class) {
        j += 1;
    }
    (j > pos).then_some(j)
}

/// `\s?[class]+`: the optional `\s` is taken only if the class run follows
/// (backtracking to no `\s` leaves a whitespace codepoint the class doesn't
/// contain, so that fails too).
fn match_ws_opt_class_run(
    chars: &[char],
    end: usize,
    pos: usize,
    class: &[(u32, u32)],
) -> Option<usize> {
    if CptFlags::of(chars[pos]).whitespace {
        match_class_run(chars, end, pos + 1, class)
    } else {
        match_class_run(chars, end, pos, class)
    }
}

/// DeepSeek-V3's third regex, on llama.cpp's collapsed text, alternatives
/// tried in order:
///
/// 1. `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+` (ASCII only: the
///    collapsed text keeps ASCII as is, and a non-ASCII letter's category
///    byte isn't in `A-Za-z`)
/// 2. `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
/// 3. ` ?[\p{P}\p{S}]+[\r\n]*`
/// 4. `\s*[\r\n]+` (greedy `\s*` backs off to the run's last `\r`/`\n`)
/// 5. `\s+(?!\S)`
/// 6. `\s+`
fn deepseek3_main_match(chars: &[char], end: usize, pos: usize) -> Option<usize> {
    let c = chars[pos];
    let f = CptFlags::of(c);
    let is_word = |f: CptFlags| f.is_letter() || f.is_accent_mark();
    let is_ps = |f: CptFlags| f.is_punctuation() || f.is_symbol();

    // 1.
    if c.is_ascii_punctuation() && pos + 1 < end && chars[pos + 1].is_ascii_alphabetic() {
        let mut j = pos + 1;
        while j < end && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        return Some(j);
    }
    // 2. Optional leading codepoint first (greedy), then without it.
    let lead_ok = !matches!(c, '\r' | '\n') && !f.is_letter() && !is_ps(f);
    if lead_ok && is_word(flags_at(chars, end, pos + 1)) {
        return Some(run_end(chars, end, pos + 1, is_word));
    }
    if is_word(f) {
        return Some(run_end(chars, end, pos, is_word));
    }
    // 3.
    let start = if c == ' ' && is_ps(flags_at(chars, end, pos + 1)) {
        pos + 1
    } else {
        pos
    };
    if is_ps(flags_at(chars, end, start)) {
        let mut j = run_end(chars, end, start, is_ps);
        while matches!(cpt_at(chars, end, j), Some('\r' | '\n')) {
            j += 1;
        }
        return Some(j);
    }
    // 4.-6.
    let ws_end = ws_run_end(chars, end, pos);
    if ws_end == pos {
        return None;
    }
    if let Some(k) = (pos..ws_end)
        .rev()
        .find(|&k| matches!(chars[k], '\r' | '\n'))
    {
        return Some(k + 1);
    }
    if ws_end == end {
        return Some(end);
    }
    if ws_end - pos >= 2 {
        return Some(ws_end - 1);
    }
    Some(ws_end)
}

/// Tekken's regex (Ministral 3 / Pixtral / Nemotron-3.5-Embed
/// `tokenizer.json`), alternatives tried in order, with HF `tokenizers`'
/// (leftmost-first, backtracking) semantics:
///
/// 1. `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`
/// 2. `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*`
/// 3. `\p{N}`
/// 4. ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
/// 5. `\s*[\r\n]+`
/// 6. `\s+(?!\S)`
/// 7. `\s+`
///
/// `\p{L}`/`\p{N}`/`\p{M}`/`\s` come from llama.cpp's tables (as for every
/// other pre type). Those tables have no letter case, so the two case classes
/// use Rust's Unicode `Lowercase`/`Uppercase` properties: a lowercase letter
/// is only in the second class, an uppercase one only in the first, and a
/// letter that is neither (`Lo`, most `Lm`) or a mark is in both. That
/// matches `Lu`/`Ll`/`Lo`/`M` exactly; it differs from HF only for titlecase
/// letters (`Lt`, e.g. `ǅ`) and modifier letters with the `Other_Lowercase`
/// property.
fn tekken_match(chars: &[char], end: usize, pos: usize) -> Option<usize> {
    let upper = |c: char| {
        let f = CptFlags::of(c);
        f.is_accent_mark() || (f.is_letter() && !c.is_lowercase())
    };
    let lower = |c: char| {
        let f = CptFlags::of(c);
        f.is_accent_mark() || (f.is_letter() && !c.is_uppercase())
    };
    let run = |from: usize, class: &dyn Fn(char) -> bool| {
        let mut j = from;
        while j < end && class(chars[j]) {
            j += 1;
        }
        j
    };
    // `[upper]*[lower]+` from `start`: the upper run is greedy and gives back
    // codepoints until one that can start the lower run.
    let upper_then_lower = |start: usize| -> Option<usize> {
        let upper_end = run(start, &upper);
        (start..=upper_end)
            .rev()
            .find(|&k| k < end && lower(chars[k]))
            .map(|k| run(k, &lower))
    };
    let c = chars[pos];
    let f = CptFlags::of(c);
    let lead_ok = !matches!(c, '\r' | '\n') && !f.is_letter() && !f.is_number();

    // 1. With the optional leading codepoint first (greedy), then without.
    if lead_ok && pos + 1 < end {
        if let Some(m) = upper_then_lower(pos + 1) {
            return Some(m);
        }
    }
    if let Some(m) = upper_then_lower(pos) {
        return Some(m);
    }
    // 2.
    if lead_ok && pos + 1 < end && upper(chars[pos + 1]) {
        return Some(run(run(pos + 1, &upper), &lower));
    }
    if upper(c) {
        return Some(run(run(pos, &upper), &lower));
    }
    // 3.
    if f.is_number() {
        return Some(pos + 1);
    }
    // 4.
    let other = |c: char| {
        let f = CptFlags::of(c);
        !f.whitespace && !f.is_letter() && !f.is_number()
    };
    let start = if c == ' ' && pos + 1 < end && other(chars[pos + 1]) {
        pos + 1
    } else {
        pos
    };
    if other(chars[start]) {
        let mut j = run(start, &other);
        while matches!(cpt_at(chars, end, j), Some('\r' | '\n' | '/')) {
            j += 1;
        }
        return Some(j);
    }
    // 5.-7.
    let ws_end = ws_run_end(chars, end, pos);
    if ws_end == pos {
        return None;
    }
    if let Some(k) = (pos..ws_end)
        .rev()
        .find(|&k| matches!(chars[k], '\r' | '\n'))
    {
        return Some(k + 1);
    }
    if ws_end == end {
        return Some(end);
    }
    if ws_end - pos >= 2 {
        return Some(ws_end - 1);
    }
    Some(ws_end)
}

fn u64_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(GgufValue::as_u64)
}

fn string_meta<'a>(
    metadata: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, ReflexError> {
    metadata
        .get(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| crate::reflex_err!(Tokenizer, "missing or non-string metadata key '{key}'"))
}

fn string_array_value(value: &GgufValue, key: &str) -> Result<Vec<String>, ReflexError> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    crate::reflex_err!(
                        Tokenizer,
                        "expected string array element in '{key}', got {v:?}"
                    )
                })
            })
            .collect(),
        other => Err(crate::reflex_err!(
            Tokenizer,
            "expected array for '{key}', got {other:?}"
        )),
    }
}

fn string_array(
    metadata: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<String>, ReflexError> {
    let value = metadata
        .get(key)
        .ok_or_else(|| crate::reflex_err!(Tokenizer, "missing metadata key '{key}'"))?;
    string_array_value(value, key)
}

fn f32_array(value: &GgufValue, key: &str) -> Result<Vec<f32>, ReflexError> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| {
                v.as_f32().ok_or_else(|| {
                    crate::reflex_err!(
                        Tokenizer,
                        "expected f32 array element in '{key}', got {v:?}"
                    )
                })
            })
            .collect(),
        other => Err(crate::reflex_err!(
            Tokenizer,
            "expected array for '{key}', got {other:?}"
        )),
    }
}

fn i32_array(value: &GgufValue, key: &str) -> Result<Vec<i32>, ReflexError> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| match v {
                GgufValue::I32(x) => Ok(*x),
                GgufValue::I8(x) => Ok(*x as i32),
                GgufValue::I16(x) => Ok(*x as i32),
                other => Err(crate::reflex_err!(
                    Tokenizer,
                    "expected i32 array element in '{key}', got {other:?}"
                )),
            })
            .collect(),
        other => Err(crate::reflex_err!(
            Tokenizer,
            "expected array for '{key}', got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    /// Hand-built vocab + merge list, ordered so iterative lowest-rank
    /// merging reconstructs "hello" from `h,e,l,l,o` and "world" from
    /// `w,o,r,l,d` without touching any cross-word-boundary pair (those
    /// pairs are simply never listed as merges).
    fn build_synthetic_tokenizer() -> Tokenizer {
        let tokens: Vec<String> = [
            "<unk>", "<s>", "</s>", "\u{2581}", "h", "e", "l", "o", "w", "r", "d", "ll", "he",
            "hell", "hello", "wo", "wor", "worl", "world", "<0x21>",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let token_to_id: FxHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let merges: [(&str, &str); 8] = [
            ("l", "l"),
            ("h", "e"),
            ("he", "ll"),
            ("hell", "o"),
            ("w", "o"),
            ("wo", "r"),
            ("wor", "l"),
            ("worl", "d"),
        ];
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(rank, (l, r))| ((l.to_string(), r.to_string()), rank))
            .collect();

        Tokenizer {
            architecture: "llama".to_string(),
            pre_type: None,
            tokens,
            scores: Vec::new(),
            token_type: Vec::new(),
            bos_token_id: Some(1),
            add_bos: true,
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            pad_token_id: None,
            token_to_id,
            merge_rank,
            special_tokens: Vec::new(),
            strip_zwnj_markers: false,
        }
    }

    #[test]
    fn test_encode_produces_expected_merged_ids() {
        let tok = build_synthetic_tokenizer();
        let ids = tok.encode("hello world").expect("encode should succeed");
        // Final symbols after merging: ["▁", "hello", "▁", "world"].
        assert_eq!(ids, vec![3, 14, 3, 18]);
    }

    /// Confirmed against the real reference tokenizer (`sentencepiece`
    /// against TinyLlama's real `tokenizer.model`, 2026-09-12, see
    /// `scripts/reference_tokenize.py`): unlike every non-empty input, empty
    /// text does not get the dummy-prefix `WORD_BOUNDARY` token — it encodes
    /// to zero tokens, not one.
    /// A vocab with scores and no merge list (e.g. TheBloke's Mistral-7B
    /// GGUFs) merges by score, as llama.cpp does, instead of not at all.
    fn build_scored_tokenizer() -> Tokenizer {
        let mut tok = build_synthetic_tokenizer();
        tok.merge_rank = FxHashMap::default();
        // Lone "▁" scored like Mistral's (-1e9): it must still merge into
        // "▁hello" when that token exists.
        tok.tokens.push("\u{2581}hello".to_string());
        tok.token_to_id.insert("\u{2581}hello".to_string(), 20);
        tok.scores = tok
            .tokens
            .iter()
            .map(|t| match t.as_str() {
                "\u{2581}" => -1e9,
                "\u{2581}hello" => -1.0,
                // Longer pieces score higher, like a real SentencePiece vocab.
                other => -10.0 / other.chars().count() as f32,
            })
            .collect();
        tok
    }

    #[test]
    fn test_scored_vocab_without_merges_still_merges() {
        let tok = build_scored_tokenizer();
        let ids = tok.encode("hello world").expect("encode should succeed");
        // "▁hello" (one token), then "▁", "world": "▁world" isn't in the vocab.
        assert_eq!(ids, vec![20, 3, 18]);
        assert_eq!(tok.decode(&ids), " hello world");
    }

    #[test]
    fn test_scored_merge_prefers_higher_score_then_leftmost() {
        let mut tok = build_scored_tokenizer();
        // "ab" and "bc" both exist; "bc" scores higher, so "abc" -> "a", "bc".
        for (piece, score) in [
            ("a", -5.0),
            ("b", -5.0),
            ("c", -5.0),
            ("ab", -2.0),
            ("bc", -1.0),
        ] {
            tok.token_to_id
                .insert(piece.to_string(), tok.tokens.len() as u32);
            tok.tokens.push(piece.to_string());
            tok.scores.push(score);
        }
        let id = |p: &str| tok.token_to_id[p];
        assert_eq!(
            tok.encode("abc").unwrap(),
            vec![id("\u{2581}"), id("a"), id("bc")]
        );
        // Equal scores: the leftmost pair wins.
        let bc = tok.token_to_id["bc"] as usize;
        tok.scores[bc] = -2.0;
        assert_eq!(
            tok.encode("abc").unwrap(),
            vec![id("\u{2581}"), id("ab"), id("c")]
        );
    }

    #[test]
    fn test_scored_merge_byte_falls_back_for_unknown_char() {
        let tok = build_scored_tokenizer();
        let ids = tok.encode("h!").expect("encode should succeed");
        assert_eq!(ids, vec![3, 4, 19]); // "▁", "h", "<0x21>"
    }

    #[test]
    fn test_encode_empty_string_produces_no_tokens() {
        let tok = build_synthetic_tokenizer();
        assert_eq!(
            tok.encode("").expect("encode should succeed"),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn test_encode_decode_round_trips_ascii_text() {
        let tok = build_synthetic_tokenizer();
        let ids = tok.encode("hello world").expect("encode should succeed");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded.trim_start_matches(' '), "hello world");
    }

    #[test]
    fn test_decode_stream_matches_batch_decode_for_ascii_text() {
        let tok = build_synthetic_tokenizer();
        let ids = tok.encode("hello world").expect("encode should succeed");
        let mut pending: Vec<u8> = Vec::new();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&tok.decode_stream(&mut pending, id));
        }
        assert!(
            pending.is_empty(),
            "no incomplete UTF-8 tail should remain for plain ASCII text"
        );
        assert_eq!(streamed, tok.decode(&ids));
    }

    #[test]
    fn test_decode_stream_holds_back_incomplete_multibyte_sequence() {
        let mut tok = build_synthetic_tokenizer();
        // 'é' (U+00E9) is 0xC3 0xA9 in UTF-8 -- feed its two bytes as two
        // separate byte-fallback tokens, simulating a multi-byte character
        // split across a generated-token boundary.
        tok.tokens.push("<0xC3>".to_string());
        let leading_byte_id = (tok.tokens.len() - 1) as u32;
        tok.tokens.push("<0xA9>".to_string());
        let trailing_byte_id = (tok.tokens.len() - 1) as u32;

        let mut pending: Vec<u8> = Vec::new();
        let first = tok.decode_stream(&mut pending, leading_byte_id);
        assert_eq!(
            first, "",
            "an incomplete leading byte of a multi-byte char must not be emitted yet"
        );
        assert_eq!(pending, vec![0xC3]);

        let second = tok.decode_stream(&mut pending, trailing_byte_id);
        assert_eq!(second, "é");
        assert!(pending.is_empty());
    }

    #[test]
    fn test_byte_fallback_round_trips_out_of_vocab_character() {
        let tok = build_synthetic_tokenizer();
        // '!' (0x21) is not in the synthetic vocab as a single char, only
        // its byte-fallback token is — exercises the decompose-to-bytes path.
        let ids = tok.encode("h!").expect("encode should succeed");
        assert_eq!(ids, vec![3, 4, 19]); // "▁", "h", "<0x21>"
        let decoded = tok.decode(&ids);
        assert_eq!(decoded.trim_start_matches(' '), "h!");
    }

    #[test]
    fn test_encode_rejects_character_with_no_vocab_or_byte_fallback_entry() {
        let mut tok = build_synthetic_tokenizer();
        tok.tokens.pop(); // drop "<0x21>" so '!' has no byte-fallback token
        tok.token_to_id.remove("<0x21>");
        assert!(tok.encode("!").is_err());
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // value_type = STRING
        write_string(buf, value);
    }

    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = U32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_string(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes()); // value_type = ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // elem_type = STRING
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            write_string(buf, v);
        }
    }

    fn write_temp_file(bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rft_tokenizer_test_{}_{unique}.gguf",
            std::process::id()
        ));
        let mut f = File::create(&path).expect("create temp gguf file");
        f.write_all(bytes).expect("write temp gguf file");
        path
    }

    /// Builds a minimal synthetic GGUF with only `tokenizer.ggml.*`
    /// metadata (no tensors) — same synthetic-only posture as `gguf.rs`'s
    /// and `serving_ops.rs`'s own tests (no real GGUF file available in
    /// this environment). Exercises `Tokenizer::from_gguf`'s extraction of
    /// architecture, vocab, merges, and special-token ids.
    #[test]
    fn test_tokenizer_from_gguf_extracts_metadata() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic "GGUF"
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // metadata_kv_count

        write_kv_string(&mut buf, "tokenizer.ggml.model", "llama");
        write_kv_string_array(
            &mut buf,
            "tokenizer.ggml.tokens",
            &["<unk>", "\u{2581}", "hi"],
        );
        write_kv_string_array(&mut buf, "tokenizer.ggml.merges", &["\u{2581} hi"]);
        write_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
        write_kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 2);

        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        let path = write_temp_file(&buf);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();

        let tok = Tokenizer::from_gguf(&file).expect("extract tokenizer metadata");
        assert_eq!(tok.architecture, "llama");
        assert_eq!(tok.tokens, vec!["<unk>", "\u{2581}", "hi"]);
        assert_eq!(tok.bos_token_id, Some(1));
        assert_eq!(tok.eos_token_id, Some(2));
        assert_eq!(tok.unk_token_id, None);
        assert_eq!(tok.merge_count(), 1);
        // No `add_bos_token` key: keep prepending the declared BOS.
        assert!(tok.add_bos);
        assert_eq!(tok.prompt_bos(), Some(1));
    }

    fn write_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
        write_string(buf, key);
        buf.extend_from_slice(&7u32.to_le_bytes()); // value_type = BOOL
        buf.push(value as u8);
    }

    /// `add_bos_token = false` with a `bos_token_id` present (real
    /// Qwen3.8-27B's shape) must suppress the BOS, matching llama.cpp.
    #[test]
    fn test_tokenizer_honors_add_bos_token() {
        for (add_bos, expected) in [(false, None), (true, Some(1))] {
            let mut buf = Vec::new();
            buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic "GGUF"
            buf.extend_from_slice(&3u32.to_le_bytes()); // version
            buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            buf.extend_from_slice(&5u64.to_le_bytes()); // metadata_kv_count

            write_kv_string(&mut buf, "tokenizer.ggml.model", "llama");
            write_kv_string_array(
                &mut buf,
                "tokenizer.ggml.tokens",
                &["<unk>", "\u{2581}", "hi"],
            );
            write_kv_string_array(&mut buf, "tokenizer.ggml.merges", &["\u{2581} hi"]);
            write_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
            write_kv_bool(&mut buf, "tokenizer.ggml.add_bos_token", add_bos);

            while buf.len() % 32 != 0 {
                buf.push(0);
            }

            let path = write_temp_file(&buf);
            let file = GgufFile::open(&path).expect("parse synthetic GGUF");
            std::fs::remove_file(&path).ok();

            let tok = Tokenizer::from_gguf(&file).expect("extract tokenizer metadata");
            assert_eq!(tok.bos_token_id, Some(1));
            assert_eq!(tok.add_bos, add_bos);
            assert_eq!(tok.prompt_bos(), expected);
        }
    }

    #[test]
    fn test_tokenizer_from_gguf_errs_when_tokens_key_missing() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        write_kv_string(&mut buf, "tokenizer.ggml.model", "llama");
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        let path = write_temp_file(&buf);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();

        assert!(Tokenizer::from_gguf(&file).is_err());
    }

    /// The `qwen2` split ([`pretokenize`] with [`PreType::Qwen2`]) — pre-tokenizer
    /// boundary checks against the real llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`
    /// regex's documented behavior (no GPU, no real fixture needed — pure
    /// host-side function, same "test the algorithm in isolation" precedent
    /// as `dequant.rs`'s own hand-computed-expected-value tests). Also checks
    /// the pieces tile `text` exactly.
    fn qwen2_pretokenize_all(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let ranges = pretokenize(PreType::Qwen2, &chars);
        let mut pos = 0;
        for &(start, end) in &ranges {
            assert!(start == pos && end > start, "pieces must tile the text");
            pos = end;
        }
        assert_eq!(pos, chars.len(), "pieces must cover the whole text");
        ranges
            .into_iter()
            .map(|(start, end)| chars[start..end].iter().collect())
            .collect()
    }

    #[test]
    fn test_qwen2_pretokenize_harakat_are_not_letters() {
        // كَتَبَ: Arabic letters, each followed by a fatha (U+064E, Mn). In
        // llama.cpp's table a fatha is an accent mark, not a letter, so it
        // starts the next `[^\r\n\p{L}\p{N}]?\p{L}+` piece instead of
        // extending the current letter run; the last one, with no letter
        // after it, is a `[^\s\p{L}\p{N}]+` piece. Rust's `is_alphabetic`
        // (the split this replaced) counted it as a letter, giving one piece.
        assert_eq!(
            qwen2_pretokenize_all("\u{643}\u{64e}\u{62a}\u{64e}\u{628}\u{64e}"),
            vec!["\u{643}", "\u{64e}\u{62a}", "\u{64e}\u{628}", "\u{64e}"]
        );
    }

    #[test]
    fn test_qwen2_pretokenize_single_leading_space_glues_to_word() {
        // "Hello world" -> ["Hello", " world"]: the single space between the
        // two words attaches to "world", not its own piece.
        assert_eq!(
            qwen2_pretokenize_all("Hello world"),
            vec!["Hello", " world"]
        );
    }

    #[test]
    fn test_qwen2_pretokenize_extra_leading_spaces_become_their_own_piece() {
        // "   Hello" (3 spaces) -> ["  ", " Hello"]: exactly one space glues
        // to the letter run; the other two form their own whitespace piece.
        assert_eq!(qwen2_pretokenize_all("   Hello"), vec!["  ", " Hello"]);
    }

    #[test]
    fn test_qwen2_pretokenize_digits_split_one_per_token() {
        // Qwen2's pattern uses `\p{N}` (exactly one digit), not `\p{N}+`
        // (a run) — unlike original GPT-2's own pattern.
        assert_eq!(qwen2_pretokenize_all("2026"), vec!["2", "0", "2", "6"]);
    }

    #[test]
    fn test_qwen2_pretokenize_contraction_suffix_splits_from_word() {
        assert_eq!(qwen2_pretokenize_all("don't"), vec!["don", "'t"]);
    }

    #[test]
    fn test_qwen2_pretokenize_blank_line_run_grouped_with_trailing_newline() {
        // "\n\n" (two blank lines) between "linea"/"lineb" -> the whole run
        // is one piece, ending right after the last newline.
        assert_eq!(
            qwen2_pretokenize_all("linea\n\nlineb"),
            vec!["linea", "\n\n", "lineb"]
        );
    }

    #[test]
    fn test_qwen2_pretokenize_digit_inside_a_word_still_splits_per_digit() {
        // "line1" -> ["line", "1"]: digits are never absorbed into a letter
        // run (Qwen2's `\p{N}` alternative, tried before letters can extend
        // across a digit boundary — each digit is always its own piece).
        assert_eq!(qwen2_pretokenize_all("line1"), vec!["line", "1"]);
    }

    #[test]
    fn test_qwen2_pretokenize_punctuation_run_is_its_own_piece() {
        assert_eq!(
            qwen2_pretokenize_all("Hello, world!"),
            vec!["Hello", ",", " world", "!"]
        );
    }

    #[test]
    fn test_qwen2_pretokenize_empty_text_produces_no_pieces() {
        assert_eq!(qwen2_pretokenize_all(""), Vec::<String>::new());
    }

    #[test]
    fn test_gpt2_byte_to_unicode_table_is_a_real_bijection() {
        let table = gpt2_byte_to_unicode_table();
        let inverse = gpt2_unicode_to_byte_table();
        assert_eq!(
            inverse.len(),
            256,
            "every byte must map to a distinct char (a real bijection)"
        );
        for b in 0..=255u8 {
            assert_eq!(
                inverse.get(&table[b as usize]),
                Some(&b),
                "round trip failed for byte {b}"
            );
        }
        // Spot-check real, well-known values from OpenAI's own construction:
        // '!' (0x21, first "already printable" byte) maps to itself; space
        // (0x20) maps to U+0120 'Ġ' — the widely-recognized "space" glyph
        // GPT-2-family tokenizers display (32 control bytes 0x00-0x1F sort
        // before 0x20 and are also unassigned, so space is the 33rd escaped
        // byte: codepoint 256+32=288=0x120, not the naively-expected 256).
        assert_eq!(table[0x21], '!');
        assert_eq!(table[0x20], '\u{120}');
    }

    /// Real fixture (`unsloth/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q4_K_M.gguf`,
    /// `PHASE21_14_PLAN.md`'s 21.14.1) — loads the real `tokenizer.ggml.model
    /// = "gpt2"` / `tokenizer.ggml.pre = "qwen2"` tokenizer and exercises
    /// `encode`/`decode` on real strings covering every pre-tokenizer
    /// alternative this file's own synthetic tests check in isolation
    /// (spaces, digits, contractions, blank lines, punctuation, empty
    /// text) — asserts real round-trip (`decode(encode(text)) == text`),
    /// the actual bar this repo holds every tokenizer path to. Skips
    /// gracefully (no GPU needed — pure host-side GGUF/tokenizer parsing)
    /// if the fixture isn't present, matching every other real-fixture test
    /// in this crate.
    #[test]
    fn test_gpt2_encode_decode_round_trips_real_strings_if_qwen3_fixture_present() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
        let path = manifest.join("Qwen3-0.6B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!(
                "Skipping test_gpt2_encode_decode_round_trips_real_strings_if_qwen3_fixture_present: \
                 real Qwen3-0.6B fixture not present"
            );
            return;
        }
        let file = GgufFile::open(&path).expect("open real Qwen3-0.6B GGUF");
        let tok =
            Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf on real Qwen3-0.6B file");
        assert_eq!(tok.architecture, "gpt2");

        for text in [
            "Once upon a time",
            "Hello, world!",
            "I'm testing   multiple   spaces here.",
            "line1\nline2\n\nline3",
            "The year 2026 has 365 days, or 366 in a leap year.",
            "don't can't won't I'll we've",
            "",
        ] {
            let ids = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("encode({text:?}) failed: {e}"));
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, text, "round trip failed for {text:?}: ids={ids:?}");
        }
    }

    /// Regression test for the real end-to-end bug this field/split fixed:
    /// without `special_tokens`, a literal `<|im_start|>`-style substring (what
    /// a rendered chat template produces, see `sidecar/openai-adapter`) got
    /// shredded into per-byte-fragment tokens by the generic BPE path instead
    /// of mapping to its one reserved vocab id — confirmed on a real
    /// Qwen3-0.6B GGUF on real GPU hardware before this fix.
    #[test]
    fn test_encode_matches_special_token_as_single_id_not_bpe_fragments() {
        let mut tok = build_synthetic_tokenizer();
        let special_id = tok.tokens.len() as u32;
        tok.tokens.push("<|im_start|>".to_string());
        tok.token_to_id
            .insert("<|im_start|>".to_string(), special_id);
        tok.special_tokens = vec!["<|im_start|>".to_string()];

        let ids = tok.encode("<|im_start|>hello").expect("encode");
        assert_eq!(
            ids[0], special_id,
            "expected the special token's own id first, got {ids:?}"
        );
        assert!(ids.len() > 1, "expected 'hello' to still encode after it");
    }

    /// A special token embedded mid-text (not just at position 0) is matched
    /// too, and the plain-text runs on both sides still go through normal
    /// BPE encoding untouched.
    #[test]
    fn test_encode_matches_special_token_mid_text() {
        let mut tok = build_synthetic_tokenizer();
        let special_id = tok.tokens.len() as u32;
        tok.tokens.push("<|im_start|>".to_string());
        tok.token_to_id
            .insert("<|im_start|>".to_string(), special_id);
        tok.special_tokens = vec!["<|im_start|>".to_string()];

        let ids = tok.encode("hello<|im_start|>world").expect("encode");
        let special_pos = ids
            .iter()
            .position(|&id| id == special_id)
            .expect("special token id present");
        assert!(
            special_pos > 0,
            "expected plain text before the special token"
        );
        assert!(
            special_pos < ids.len() - 1,
            "expected plain text after the special token"
        );
    }

    /// Mistral's Tekken pre type on Nemotron-3.5-Embed-1B's vocab
    /// (`test-data/nemotron-3.5-embed-tokenizer.gguf`, the head of
    /// `maternion/Nemotron-3.5-Embed-1B-GGUF`'s Q8_0 file through
    /// `scripts/gguf_tokenizer_only.py`; `pre = pixtral`): ids match HF
    /// `tokenizers` on the model's own `tokenizer.json`, including case-change
    /// splits (`HELLOWorld`, `getHTTPResponse`), non-ASCII case, one digit per
    /// piece, and the normalizer that turns the chat template's `[`/`<` +
    /// U+200C markers back into plain brackets after special tokens are
    /// matched (a raw `[INST]` is special token 3, a marked one plain text).
    #[test]
    fn test_tekken_encode_matches_hf_tokenizers_if_fixture_present() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("nemotron-3.5-embed-tokenizer.gguf");
        if !path.exists() {
            eprintln!(
                "Skipping test_tekken_encode_matches_hf_tokenizers_if_fixture_present: \
                 test-data/nemotron-3.5-embed-tokenizer.gguf not present"
            );
            return;
        }
        let file = GgufFile::open(&path).expect("open nemotron tokenizer GGUF");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf");
        assert_eq!(tok.architecture, "gpt2");
        assert_eq!(tok.pre_type, Some(PreType::Tekken));
        assert_eq!(tok.vocab_size(), 131_072);
        assert_eq!(tok.prompt_bos(), None, "add_bos_token = false");
        assert!(
            tok.strip_zwnj_markers,
            "a Tekken vocab in an embedding model strips the template's markers"
        );
        // HF `tokenizers` 0.23.2 on nvidia/Nemotron-3.5-Embed-1B-BF16's tokenizer.json,
        // `encode(text, add_special_tokens=False)`, for PRE_TYPE_GOLDEN_STRINGS in order.
        let golden: [&[u32]; PRE_TYPE_GOLDEN_STRINGS.len()] = [
            &[
                1049, 1050, 1051, 1052, 1053, 1032, 1051, 1046, 1049, 1052, 1049, 1053, 1057, 1032,
                1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1048,
            ],
            &[
                21797, 2405, 1362, 1039, 23947, 1729, 6483, 2127, 1039, 3609, 1494, 1681, 54197,
                1039, 1068,
            ],
            &[
                22177, 4304, 1033, 1032, 124108, 1625, 29659, 1320, 10008, 15199, 2439, 7282,
                14742, 13854, 1749, 10709, 6912, 10709, 13266, 1320, 2316, 6430, 2897, 10323, 5221,
                23266, 1046,
            ],
            &[
                1077, 5422, 1058, 7846, 1044, 74693, 9026, 62951, 1044, 11813, 53660, 13865, 1044,
                21362, 1044, 51388, 34946,
            ],
            &[
                2839, 1925, 1010, 2839, 2295, 1013, 4688, 1032, 2495, 38498, 1009, 14133, 8217,
                1293, 1010,
            ],
            &[
                1356, 26303, 119685, 1145, 1139, 1240, 1159, 1143, 1189, 119685, 1142, 1137, 1240,
                1159, 1142, 1137, 3221, 1157, 1164, 1239, 1184, 1143, 119685, 1145, 1168, 29568,
                1240, 1159, 1145, 1169, 29568, 1240, 1159, 1145, 1167,
            ],
            &[
                1080, 29586, 65656, 55237, 24421, 1063, 6421, 26433, 7840, 36689, 1032, 126149,
                90021, 16474, 2126, 1035, 1036, 1037, 1094, 1038, 1042,
            ],
            &[1256, 8924, 18971, 1321, 49875, 1293],
            &[
                3173, 1102, 1337, 98355, 1324, 1204, 1129, 1261, 1204, 1128, 1308, 1204, 1131, 1111,
            ],
            &[
                1120, 1376, 34377, 1040, 3947, 1044, 1032, 1052, 1050, 1468, 1391, 1319, 1120,
                9218, 1032, 1049, 1048, 1041, 1445, 1850, 1429, 1662, 16255, 1826,
            ],
            &[3997, 13151, 114840, 1675, 11768, 5702, 1287, 1131, 1356],
            &[
                1342, 1166, 6495, 1149, 6495, 1140, 6495, 1140, 1342, 1183, 6495, 1137, 6495, 1132,
                6495, 1148, 6495, 1136, 1032, 11181, 13824, 17979, 1032, 122964,
            ],
            &[
                1072, 20031, 1058, 4148, 3525, 22475, 1803, 115801, 1044, 44575, 1454, 5999, 25940,
                1058, 1477, 40040, 7213, 1708, 43117, 2944,
            ],
        ];
        let extra: &[(&str, &[u32])] = &[
            ("HELLOWorld helloWorld CamelCaseWord getHTTPResponse iPhone XMLHttpRequest", &[7626, 1076, 7618, 20249, 52528, 20249, 108915, 11139, 24469, 2012, 30499, 6566, 1623, 16742, 126379, 4967]),
            ("\u{dc}berMax \u{c4}\u{d6}\u{dc}\u{e4}\u{f6}\u{fc} \u{c9}COLE \u{c9}cole \u{3a3}\u{38a}\u{3a3}\u{3a5}\u{3a6}\u{39f}\u{3a3} Stra\u{df}e", &[79904, 11309, 19369, 30773, 39453, 1654, 1792, 1671, 4664, 9335, 3561, 87370, 9681, 1206, 1138, 27026, 118583, 117850, 25527, 27026, 27277]),
            ("query: What is NVIDIA NeMo Framework?", &[5272, 1058, 5675, 1395, 1464, 10145, 20501, 4671, 37042, 41667, 1063]),
            ("passage: see [\u{200c}INST] and <\u{200c}s> tags", &[12107, 1541, 1058, 3219, 1766, 3174, 3074, 1093, 1321, 1534, 1115, 1062, 19762]),
            ("passage: see [INST] and <s> tags", &[12107, 1541, 1058, 3219, 1032, 3, 1321, 1032, 1, 19762]),
            ("path/to/file.txt\n/next ... 3.14/2", &[7368, 35466, 55998, 13477, 1010, 1047, 6651, 6421, 1032, 1051, 1046, 1049, 1052, 1047, 1050]),
        ];
        for (&text, &expected) in PRE_TYPE_GOLDEN_STRINGS.iter().zip(&golden) {
            let ids = tok.encode(text).expect("encode");
            assert_eq!(ids, expected, "ids differ from HF tokenizers for {text:?}");
            assert_eq!(tok.decode(&ids), text, "round trip failed for {text:?}");
        }
        for &(text, expected) in extra {
            let ids = tok.encode(text).expect("encode");
            assert_eq!(ids, expected, "ids differ from HF tokenizers for {text:?}");
        }
    }

    /// The marker normalization is only for embedding models: the same Tekken
    /// vocab without `<arch>.pooling_type` (a Ministral chat model) keeps a
    /// `[` + U+200C as typed, as its HF tokenizer (no normalizer) does.
    #[test]
    fn test_tekken_strips_markers_only_for_embedding_models() {
        for (pooling, strips) in [(None, false), (Some(0u32), false), (Some(1), true)] {
            let mut file = gpt2_tokenizer_gguf(Some("pixtral"));
            file.metadata.insert(
                "general.architecture".to_string(),
                GgufValue::String("mistral3".to_string()),
            );
            if let Some(p) = pooling {
                file.metadata
                    .insert("mistral3.pooling_type".to_string(), GgufValue::U32(p));
            }
            let tok = Tokenizer::from_gguf(&file).expect("from_gguf");
            assert_eq!(tok.strip_zwnj_markers, strips, "pooling_type {pooling:?}");
        }
        let file = gpt2_tokenizer_gguf(Some("nemotron-3.5-embed"));
        assert!(
            Tokenizer::from_gguf(&file)
                .expect("from_gguf")
                .strip_zwnj_markers
        );
    }

    /// Kolibri-1's tokenizer is plain byte-level BPE with Qwen2's split regex
    /// (`tokenizer.ggml.pre = "kolibri1"`, mapped to `LLAMA_VOCAB_PRE_TYPE_QWEN2`
    /// by the community llama.cpp patch), so `encode_gpt2` should handle it
    /// unchanged. Golden ids from HF `tokenizers` 0.23.2 on
    /// `Aleph-Alpha/Kolibri-1-BF16`'s `tokenizer.json`
    /// (`add_special_tokens=False`; the model has no BOS). Covers German
    /// compounds, umlauts/ß/ẞ, digits (split one per token), code, contractions,
    /// whitespace runs, emoji and the ChatML/`<think>` markers its chat
    /// template emits.
    ///
    /// `test-data/kolibri1-tokenizer.gguf` is the metadata section of
    /// `Hob-forge/Kolibri-1-GGUF`'s `Kolibri-1-Q4_K_M.gguf` with the tensor
    /// table dropped (~4.8 MB, gitignored); see docs/DEVELOPMENT.md's
    /// test-fixture section. Skips if absent.
    #[test]
    fn test_kolibri1_encode_matches_hf_tokenizers_if_fixture_present() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("kolibri1-tokenizer.gguf");
        if !path.exists() {
            eprintln!(
                "Skipping test_kolibri1_encode_matches_hf_tokenizers_if_fixture_present: \
                 test-data/kolibri1-tokenizer.gguf not present"
            );
            return;
        }
        let file = GgufFile::open(&path).expect("open kolibri1 tokenizer GGUF");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf on kolibri1");
        assert_eq!(tok.architecture, "gpt2");
        assert_eq!(tok.vocab_size(), 128_000);
        assert_eq!(tok.prompt_bos(), None, "Kolibri-1 has no BOS token");
        assert_eq!(tok.eos_token_id, Some(127_906));

        let golden: &[(&str, &[u32])] = &[
            (
                "Donaudampfschifffahrtsgesellschaftskapitän",
                &[24587, 10669, 9355, 104246, 115, 11381, 115, 116721],
            ),
            (
                "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
                &[82, 69950, 106789, 7263, 45356, 35837, 2761, 23903, 61308, 92151],
            ),
            (
                "Größenänderung, Straße, Fußgängerübergang – Ärger über Öl und ÄÖÜ äöü ß ẞ",
                &[72236, 266, 40644, 44, 12779, 44, 33059, 90022, 1829, 46525, 1489, 21262, 420, 10739, 1674, 960, 121214, 276, 264, 123143, 32, 8578, 158],
            ),
            (
                "Die Rechnung über 1.234,56 € ist am 03.10.2026 fällig.",
                &[452, 21069, 1489, 32, 49, 46, 50, 51, 52, 44, 53, 54, 13088, 2459, 10304, 32, 48, 51, 46, 49, 48, 46, 50, 48, 50, 54, 45118, 46],
            ),
            (
                "12345678901234567890 und 3,14159",
                &[49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 420, 32, 51, 44, 49, 52, 49, 53, 57],
            ),
            (
                "fn main() {\n    println!(\"Hallo, Welt!\");\n}\n",
                &[47328, 995, 765, 982, 14141, 88473, 69175, 38241, 44, 3389, 56895, 475],
            ),
            (
                "def f(x):\n\treturn x**2  # Quadrat\n",
                &[1363, 1083, 2931, 2488, 1779, 14169, 294, 50, 32, 4396, 17403, 10],
            ),
            (
                "The quick brown fox jumps over the lazy dog's back. IT'S I'M we'll",
                &[325, 13040, 10633, 39671, 61041, 1637, 262, 25423, 5926, 589, 3785, 46, 12194, 48743, 669, 121336, 527, 3597],
            ),
            (
                "  leading spaces\n\n\ntrailing   \t",
                &[32, 4018, 14356, 120038, 755, 57641, 14141, 9],
            ),
            (
                "Grüße 👋🏽 — naïve café, 日本語",
                &[72678, 53615, 32, 36121, 139, 52651, 189, 14188, 68455, 39993, 44, 32, 80815, 71163],
            ),
            // Arabic with harakat and a superscript alef (U+0670), and
            // Devanagari with vowel signs, virama and anusvara: combining
            // marks, where llama.cpp's Unicode table and Rust's
            // `is_alphabetic` disagree. Also matched by `llama-tokenize`
            // with the Kolibri-1 patch (836d571).
            (
                "\u{628}\u{650}\u{633}\u{652}\u{645}\u{650} \u{627}\u{644}\u{644}\u{651}\u{64e}\u{647}\u{650} \u{627}\u{644}\u{631}\u{651}\u{64e}\u{62d}\u{652}\u{645}\u{64e}\u{670}\u{646}\u{650} \u{627}\u{644}\u{631}\u{651}\u{64e}\u{62d}\u{650}\u{64a}\u{645}\u{650}",
                &[10622, 26824, 12833, 29960, 6725, 26824, 25691, 5010, 61766, 11659, 26824, 25691, 7565, 61766, 20431, 29960, 6725, 12759, 217, 176, 7274, 26824, 25691, 7565, 61766, 20431, 78144, 6725, 26824],
            ),
            (
                "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947} \u{926}\u{941}\u{928}\u{93f}\u{92f}\u{93e}, \u{92f}\u{939} \u{939}\u{93f}\u{902}\u{926}\u{940} \u{92e}\u{947}\u{902} \u{932}\u{93f}\u{916}\u{93e} \u{917}\u{92f}\u{93e} \u{939}\u{948}\u{964}",
                &[30759, 41737, 42118, 65034, 18066, 74280, 39882, 30759, 78870, 11683, 44, 95958, 52334, 46870, 19463, 130, 58561, 23840, 45256, 56446, 84204, 19463, 150, 11683, 90279, 59460, 11683, 46870, 53804, 53789],
            ),
            (
                "<|im_start|>user\nWie spät ist es in München?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nEs ist 14:30 Uhr.<|im_end|>",
                &[127904, 1646, 10, 12403, 30512, 2459, 14154, 622, 8047, 63, 127906, 10, 127904, 64090, 10, 127907, 263, 127908, 263, 35212, 2459, 32, 49, 52, 58, 51, 48, 5269, 46, 127906],
            ),
        ];
        for &(text, expected) in golden {
            let ids = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("encode({text:?}) failed: {e}"));
            assert_eq!(
                ids, expected,
                "token ids differ from HF tokenizers for {text:?}"
            );
            assert_eq!(tok.decode(&ids), text, "round trip failed for {text:?}");
        }
    }

    /// [`Tokenizer::bpe_merge_ranked`] (the heap-based merge the gemma4 path
    /// uses) must give the same symbols as [`Tokenizer::bpe_merge`]'s
    /// rescanning loop: lowest rank first, ties to the leftmost pair.
    /// Includes repeated letters, where one merge can invalidate a queued
    /// pair on either side.
    #[test]
    fn test_bpe_merge_ranked_matches_rescanning_merge() {
        let tok = build_synthetic_tokenizer();
        for word in [
            "",
            "h",
            "hello",
            "helloworld",
            "lll",
            "llll",
            "lllll",
            "hellllo",
            "wwoorrlldd",
            "worldhello",
            "hehehe",
            "xhellox",
            "wor",
            "olleh",
        ] {
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            tok.bpe_merge(&mut symbols);
            let ranked: Vec<String> = tok
                .bpe_merge_ranked(word)
                .into_iter()
                .map(str::to_string)
                .collect();
            assert_eq!(ranked, symbols, "merge mismatch for {word:?}");
        }
    }

    /// Gemma 4's tokenizer (`tokenizer.ggml.model = "gemma4"`). Golden ids
    /// from HF `tokenizers` 0.23.2 on `google/gemma-4-12B-it`'s
    /// `tokenizer.json` (`add_special_tokens=False`), which also matched a
    /// Python port of llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GEMMA4` tokenize on
    /// every string. Covers spaces (leading, repeated, trailing), newline
    /// runs (including one longer than any newline token), tabs, digits,
    /// contractions, accented Latin, CJK, emoji, control characters, code,
    /// a literal `▁`, and the chat-template, thinking and tool-call markers.
    ///
    /// `test-data/gemma4-tokenizer.gguf` is the metadata section of
    /// `ggml-org/gemma-4-12B-it-GGUF`'s `gemma-4-12B-it-Q8_0.gguf` with the
    /// tensor table dropped (~16 MB, gitignored); see docs/DEVELOPMENT.md's
    /// test-fixture section. Skips if absent.
    #[test]
    fn test_gemma4_encode_matches_reference_if_fixture_present() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("gemma4-tokenizer.gguf");
        if !path.exists() {
            eprintln!(
                "Skipping test_gemma4_encode_matches_reference_if_fixture_present: \
                 test-data/gemma4-tokenizer.gguf not present"
            );
            return;
        }
        let file = GgufFile::open(&path).expect("open gemma4 tokenizer GGUF");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf on gemma4");
        assert_eq!(tok.architecture, "gemma4");
        assert_eq!(tok.vocab_size(), 262_144);
        assert_eq!(tok.merge_count(), 514_906);
        assert_eq!(tok.prompt_bos(), Some(2), "Gemma 4 always adds <bos>");
        assert_eq!(tok.eos_token_id, Some(1));

        let golden: &[(&str, &[u32])] = &[
            (
                "Once upon a time",
                &[14946, 3324, 496, 990],
            ),
            (
                "Hello, world!",
                &[9259, 236764, 1902, 236888],
            ),
            (
                "The capital of France is",
                &[818, 5279, 529, 7001, 563],
            ),
            (
                " leading space",
                &[5830, 2557],
            ),
            (
                "  two leading spaces and   three inside",
                &[138, 13498, 5830, 9952, 532, 139, 19891, 4888],
            ),
            (
                "trailing spaces   ",
                &[136697, 9952, 139],
            ),
            (
                "line1\nline2\n\nline3\n\n\n\nend\n",
                &[1257, 236770, 107, 1257, 236778, 108, 1257, 236800, 110, 643, 107],
            ),
            (
                "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
                &[137, 115],
            ),
            (
                "tabs\tand\t\tmore\t",
                &[39218, 255968, 624, 255969, 5576, 255968],
            ),
            (
                "The year 2026 has 365 days, or 366 in a leap year. 3.14159",
                &[818, 1051, 236743, 236778, 236771, 236778, 236825, 815, 236743, 236800, 236825, 236810, 2668, 236764, 653, 236743, 236800, 236825, 236825, 528, 496, 42681, 1051, 236761, 236743, 236800, 236761, 236770, 236812, 236770, 236810, 236819],
            ),
            (
                "don't can't won't I'll we've IT'S",
                &[13246, 236789, 236745, 740, 236789, 236745, 2810, 236789, 236745, 564, 236789, 859, 692, 236789, 560, 9053, 236789, 236773],
            ),
            (
                "Größenänderung, Straße, naïve café — ÄÖÜ ẞ",
                &[9322, 237001, 16207, 48494, 1228, 236764, 80176, 236764, 120362, 33443, 2192, 31083, 238255, 238194, 65894],
            ),
            (
                "日本語のテキストと中文文本，한국어 텍스트",
                &[94951, 236945, 95830, 237032, 55041, 57489, 236900, 114216, 237430, 236743, 243007, 34718],
            ),
            (
                "Emoji: 👋🏽 🇩🇪 🧑‍💻 ✨",
                &[127268, 236787, 155818, 242713, 236743, 246841, 245562, 236743, 247027, 237243, 244862, 73687],
            ),
            (
                "fn main() {\n    println!(\"Hallo, Welt!\");\n}\n",
                &[15059, 1689, 825, 642, 107, 140, 4629, 36492, 45890, 236764, 30335, 21265, 107, 236783, 107],
            ),
            (
                "def f(x):\n\treturn x**2  # square\n",
                &[2063, 517, 236769, 236781, 1473, 107, 255968, 2060, 1123, 1018, 236778, 138, 236865, 6281, 107],
            ),
            (
                "Review: The battery died after two days. Sentiment:",
                &[30852, 236787, 669, 10055, 8390, 1308, 1156, 2668, 236761, 162249, 236787],
            ),
            (
                "\u{0}\u{1} control \u{7f} chars",
                &[238, 249732, 2256, 236743, 251541, 56256],
            ),
            (
                "x",
                &[236781],
            ),
            (
                " ",
                &[236743],
            ),
            (
                "▁already▁escaped",
                &[3016, 41685],
            ),
            (
                "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
                &[2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107],
            ),
            (
                "<|turn>system\n<|think|>You are terse.<turn|>\n<|turn>user\nWie spät ist es in München?<turn|>\n<|turn>model\n<|channel>thought\nCheck the time zone.<channel|>Es ist 14:30 Uhr.<turn|>",
                &[105, 9731, 107, 98, 3048, 659, 17514, 236761, 106, 107, 105, 2364, 107, 64008, 233648, 3526, 1115, 528, 67772, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 6845, 506, 990, 10202, 236761, 101, 14796, 3526, 236743, 236770, 236812, 236787, 236800, 236771, 32880, 236761, 106],
            ),
            (
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>",
                &[48, 6639, 236787, 828, 236779, 19323, 236782, 13319, 236787, 52, 50429, 52, 236783, 49, 50],
            ),
            (
                "<eos>",
                &[1],
            ),
        ];
        for &(text, expected) in golden {
            let ids = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("encode({text:?}) failed: {e}"));
            assert_eq!(
                ids, expected,
                "token ids differ from the reference for {text:?}"
            );
            // A literal `▁` in the input decodes as a space, as in llama.cpp.
            if !text.contains('\u{2581}') {
                assert_eq!(tok.decode(&ids), text, "round trip failed for {text:?}");
            }
        }
    }

    /// The tricky string set for the `tokenizer.ggml.pre` golden tests
    /// (docs/design/tokenizer-pre-types.md): digits, contractions, CJK/kana/
    /// Hangul, mixed scripts, newlines and whitespace runs, emoji with ZWJ and
    /// variation selectors, punctuation runs, leading/trailing spaces,
    /// decomposed accents, code, NBSP/ideographic/em spaces, fullwidth forms,
    /// and Devanagari/Arabic combining marks.
    const PRE_TYPE_GOLDEN_STRINGS: [&str; 13] = [
        "12345 3.14159 1234567890",
        "don't I'LL we've they'RE it's YOU'D",
        "Hello world! 你好，世界。日本語のテキスト、カタカナ。한국어 문장입니다.",
        "Mixed: English, Русский текст, Ελληνικά, العربية, ไทย",
        "line one\nline two\r\n\n\n  indented\t\ttabs   \n",
        "emoji 👋🏽 🎉🎉 ❤\u{fe0f} 👨\u{200d}👩\u{200d}👧",
        "Punctuation!!! ??? ... --- ((()))  {[<>]} @#$%^&*",
        "   leading spaces and trailing   ",
        "café naïve e\u{301} a\u{300} n\u{303}o",
        "x = foo(bar, 42);\nif (x >= 10) { return \"ok\"; }\n",
        "\u{a0}nbsp\u{3000}ideographic space\u{2003}em",
        "ＦｕｌｌＷｉｄｔｈ １２３ ！？",
        "Hindi: नमस\u{94d}त\u{947} द\u{941}न\u{93f}य\u{93e}, Arabic with harakat: م\u{64e}ر\u{652}ح\u{64e}ب\u{64b}ا",
    ];

    fn gpt2_tokenizer_gguf(pre: Option<&str>) -> GgufFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        let kv_count: u64 = if pre.is_some() { 3 } else { 2 };
        buf.extend_from_slice(&kv_count.to_le_bytes());
        write_kv_string(&mut buf, "tokenizer.ggml.model", "gpt2");
        write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "ab"]);
        if let Some(pre) = pre {
            write_kv_string(&mut buf, "tokenizer.ggml.pre", pre);
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let path = write_temp_file(&buf);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();
        file
    }

    /// An unknown `tokenizer.ggml.pre` on a gpt2 vocab is an error naming the
    /// value, not a silent fall back to the Qwen2 split (which is what every
    /// gpt2 vocab got before pre types were read).
    #[test]
    fn test_unknown_pre_type_is_rejected_and_named() {
        let file = gpt2_tokenizer_gguf(Some("jais"));
        let err = Tokenizer::from_gguf(&file)
            .err()
            .expect("an unknown pre type must not load");
        let msg = err.to_string();
        assert!(msg.contains("'jais'"), "error should name the value: {msg}");
        assert!(
            msg.contains("llama-bpe"),
            "error should list what is supported: {msg}"
        );
    }

    /// A gpt2 vocab with no `tokenizer.ggml.pre` keeps the Qwen2 split it
    /// always had; every accepted value maps to its variant.
    #[test]
    fn test_pre_type_mapping() {
        let tok = Tokenizer::from_gguf(&gpt2_tokenizer_gguf(None)).expect("no pre key");
        assert_eq!(tok.pre_type, Some(PreType::Qwen2));
        for &(name, ty) in PreType::ACCEPTED {
            let tok = Tokenizer::from_gguf(&gpt2_tokenizer_gguf(Some(name))).expect(name);
            assert_eq!(tok.pre_type, Some(ty), "{name}");
        }
        assert_eq!(
            PreType::from_gguf_value(Some("kolibri1")).unwrap(),
            PreType::Qwen2
        );
        assert_eq!(
            PreType::from_gguf_value(Some("deepseek-v3")).unwrap(),
            PreType::Deepseek3
        );
    }

    /// SentencePiece (`llama`) and `gemma4` vocabs don't pre-tokenize with a
    /// regex: they get no pre type and a `pre` value isn't checked.
    #[test]
    fn test_pre_type_ignored_for_non_gpt2_models() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        write_kv_string(&mut buf, "tokenizer.ggml.model", "llama");
        write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &["<unk>", "a"]);
        write_kv_string(&mut buf, "tokenizer.ggml.pre", "not-a-pre-type");
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let path = write_temp_file(&buf);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();
        let tok = Tokenizer::from_gguf(&file).expect("llama vocab ignores pre");
        assert_eq!(tok.pre_type, None);
    }

    /// The DeepSeek character classes must parse to ordered ranges that leave
    /// ASCII punctuation alone. Guards the `\u{..}` escapes in
    /// `DEEPSEEK_LLM_LATIN`: an editor or tool that NFC-normalizes the source
    /// turns U+212A KELVIN SIGN into ASCII `K` (so `K-...` swallows `[`, `{`
    /// and friends) and U+1F7D into U+03CE (an inverted range), which is
    /// exactly how this split once went wrong.
    #[test]
    fn test_deepseek_char_classes_are_well_formed() {
        let c = deepseek_classes();
        for class in [&c.llm_latin, &c.llm_punct, &c.llm_cjk, &c.v3_cjk] {
            for &(lo, hi) in class.iter() {
                assert!(lo <= hi, "inverted range {lo:#x}-{hi:#x}");
            }
        }
        for ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789 ".chars() {
            assert!(
                !in_class(ch, &c.llm_latin),
                "{ch:?} must not be in the Latin class"
            );
        }
        assert!(in_class('\u{212a}', &c.llm_latin));
        assert!(in_class('\u{1f7d}', &c.llm_latin));
        // U+3000 is inside the punctuation class's range but llama.cpp maps
        // it to \x0B (whitespace) before std::wregex sees it.
        assert!(!in_class('\u{3000}', &c.llm_punct));
    }

    /// Encode every [`PRE_TYPE_GOLDEN_STRINGS`] entry with the tokenizer in
    /// `test-data/<file_name>` and compare with `golden`. Skips the whole
    /// test if the fixture is absent.
    fn assert_pre_type_golden_if_fixture_present(
        file_name: &str,
        expected_pre: PreType,
        golden: &[&[u32]; PRE_TYPE_GOLDEN_STRINGS.len()],
    ) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join(file_name);
        if !path.exists() {
            eprintln!("Skipping {expected_pre:?} golden test: test-data/{file_name} not present");
            return;
        }
        let file = GgufFile::open(&path).expect("open tokenizer GGUF");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf");
        assert_eq!(tok.architecture, "gpt2");
        assert_eq!(tok.pre_type, Some(expected_pre));
        for (&text, &expected) in PRE_TYPE_GOLDEN_STRINGS.iter().zip(golden) {
            let ids = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("encode({text:?}) failed: {e}"));
            assert_eq!(
                ids, expected,
                "{file_name}: ids differ from llama-tokenize for {text:?}"
            );
            assert_eq!(
                tok.decode(&ids),
                text,
                "{file_name}: round trip failed for {text:?}"
            );
        }
    }

    // Golden ids for the tests below: llama.cpp `79e2e74`'s `llama-tokenize`
    // (CPU build) on the same fixture, `--stdin --ids --no-bos`, one string
    // per run. Recipe: docs/design/tokenizer-pre-types.md.

    /// `llama-bpe` (LLAMA3). `test-data/llama-bpe-tokenizer.gguf`: metadata of
    /// https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    #[test]
    fn test_llama_bpe_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "llama-bpe-tokenizer.gguf",
            PreType::Llama3,
            &[
                &[
                    4513, 1774, 220, 18, 13, 9335, 2946, 220, 4513, 10961, 16474, 15,
                ],
                &[
                    15357, 956, 358, 6, 4178, 584, 3077, 814, 95253, 433, 596, 15334, 28805,
                ],
                &[
                    9906, 1917, 0, 118195, 53901, 3922, 102616, 1811, 102433, 102158, 16144, 57933,
                    62903, 71634, 5486, 71493, 47307, 71493, 96452, 1811, 112699, 32179, 54535,
                    41953, 80052, 13,
                ],
                &[
                    87533, 25, 6498, 11, 121258, 101591, 71995, 11, 121029, 102434, 11, 110356, 11,
                    124554,
                ],
                &[
                    1074, 832, 198, 1074, 1403, 201, 1432, 220, 1280, 16243, 197, 3324, 3518, 5996,
                ],
                &[
                    38623, 62904, 233, 9468, 237, 121, 11410, 236, 231, 9468, 236, 231, 71570,
                    31643, 62904, 101, 102470, 9468, 239, 102, 102470, 9468, 239, 100,
                ],
                &[
                    47, 73399, 12340, 52417, 2564, 12730, 1819, 10340, 220, 46691, 21806, 14316,
                    571, 49177, 46999, 5, 9,
                ],
                &[256, 6522, 12908, 323, 28848, 262],
                &[
                    936, 59958, 95980, 588, 384, 54939, 264, 97649, 308, 119074, 78,
                ],
                &[
                    87, 284, 15586, 55730, 11, 220, 2983, 317, 333, 320, 87, 2669, 220, 605, 8,
                    314, 471, 330, 564, 5233, 457,
                ],
                &[4194, 5792, 23249, 95107, 3634, 116503, 336],
                &[
                    1569, 99, 15755, 243, 103375, 103375, 1569, 115, 104009, 15755, 226, 15755,
                    242, 15755, 230, 220, 20713, 25963, 34617, 220, 6447, 11571,
                ],
                &[
                    39, 29785, 25, 100282, 88344, 79468, 100365, 35470, 100291, 101391, 100322,
                    101201, 35217, 449, 4960, 105029, 25, 24252, 106524, 125902, 108117, 101333,
                ],
            ],
        );
    }

    /// `deepseek-llm` (DEEPSEEK_LLM, the pre type of DeepSeek LLM 7B and
    /// DeepSeek-V2/V2-Lite). `test-data/deepseek-llm-tokenizer.gguf`: metadata of
    /// https://huggingface.co/mradermacher/deepseek-llm-7b-chat-GGUF/resolve/main/deepseek-llm-7b-chat.Q4_K_M.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    /// TheBloke's conversion of the same model has no `pre` key at all.
    #[test]
    fn test_deepseek_llm_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "deepseek-llm-tokenizer.gguf",
            PreType::DeepseekLlm,
            &[
                &[
                    16, 17, 18, 19, 20, 207, 18, 13, 16, 19, 16, 20, 24, 207, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 15,
                ],
                &[
                    10314, 6, 83, 304, 6, 2369, 395, 6, 313, 657, 6, 1374, 359, 6, 82, 19326, 6, 35,
                ],
                &[
                    17464, 1843, 0, 207, 84313, 19304, 4997, 398, 10162, 13072, 239, 28213, 8550,
                    215, 9156, 242, 50145, 52462, 537, 9156, 104, 93491, 9156, 104, 8550, 219, 398,
                    76152, 91765, 242, 94627, 112, 207, 165, 105, 116, 48848, 98, 48848, 214,
                    39537, 92968, 13,
                ],
                &[
                    69036, 25, 3517, 11, 11050, 368, 1960, 954, 76887, 11, 7558, 230, 19236, 19236,
                    93467, 53382, 30791, 11, 52160, 64040, 21437, 41737, 35047, 88708, 11, 207,
                    45735, 213, 16510, 232, 16510, 95,
                ],
                &[
                    1031, 634, 185, 1031, 984, 188, 185, 185, 185, 207, 96272, 184, 184, 20526,
                    300, 185,
                ],
                &[
                    39349, 12833, 59948, 220, 10044, 224, 121, 12356, 223, 218, 10044, 223, 218,
                    207, 75655, 46713, 59948, 101, 350, 222, 10044, 226, 102, 350, 222, 10044, 226,
                    100,
                ],
                &[
                    60516, 21716, 10259, 59946, 3564, 16966, 8306, 23434, 207, 509, 44277, 29,
                    17001, 1173, 2, 3, 4, 61, 5, 9,
                ],
                &[243, 5881, 9577, 285, 38506, 300],
                &[
                    91774, 587, 10010, 2865, 313, 301, 47904, 245, 34015, 291, 134, 212, 78,
                ],
                &[
                    87, 403, 24433, 7, 1684, 11, 207, 19, 17, 476, 185, 351, 334, 87, 10441, 207,
                    16, 15, 8, 509, 972, 440, 544, 3170, 615, 185,
                ],
                &[1202, 3765, 88946, 546, 12263, 2516, 51570, 370],
                &[
                    948, 99, 68940, 230, 68940, 221, 68940, 221, 948, 115, 68940, 218, 68940, 213,
                    68940, 229, 68940, 217, 207, 948, 226, 948, 227, 948, 228, 61198, 2224,
                ],
                &[
                    39, 71643, 25, 207, 77982, 96984, 80260, 52539, 97, 63087, 207, 10072, 99,
                    23424, 210, 77982, 70168, 10072, 107, 43909, 11, 39195, 366, 6457, 559, 253,
                    25, 55993, 88980, 21437, 147, 227, 78930, 88980, 41737, 147, 220, 12241,
                ],
            ],
        );
    }

    /// `deepseek-v3` (DEEPSEEK3_LLM). `test-data/deepseek-v3-tokenizer.gguf`:
    /// metadata of the first shard
    /// https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/resolve/main/Q2_K/DeepSeek-V3-0324-Q2_K-00001-of-00005.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`,
    /// which drops the `split.*` keys).
    #[test]
    fn test_deepseek_v3_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "deepseek-v3-tokenizer.gguf",
            PreType::Deepseek3,
            &[
                &[
                    6895, 1883, 223, 21, 16, 9926, 3318, 223, 6895, 18009, 25744, 18,
                ],
                &[
                    20385, 1664, 342, 9, 3699, 579, 5270, 851, 9, 3827, 436, 734, 28253, 9, 38,
                ],
                &[
                    19923, 2058, 3, 223, 30594, 303, 3427, 320, 88768, 1576, 17383, 20367, 24552,
                    410, 15961, 11767, 15961, 27071, 320, 5634, 17564, 9227, 22426, 14301, 34941,
                    16,
                ],
                &[
                    113685, 28, 3947, 14, 87103, 16980, 121212, 14, 19217, 43595, 20975, 14, 55595,
                    14, 223, 39311,
                ],
                &[
                    1836, 834, 201, 1836, 1234, 204, 6328, 223, 1126, 19686, 200, 17299, 10284,
                    14320,
                ],
                &[
                    18872, 7063, 52780, 236, 40965, 124, 7351, 239, 234, 53153, 234, 53341, 100,
                    10759, 52780, 104, 46088, 25730, 105, 46088, 25730, 103,
                ],
                &[
                    50, 371, 119569, 34513, 93041, 4588, 11569, 12031, 72456, 223, 680, 61, 47357,
                    48663, 2390, 125463, 7, 64, 8, 12,
                ],
                &[262, 6646, 13564, 305, 52964, 361],
                &[
                    69, 2797, 619, 112752, 312, 17793, 260, 45345, 313, 59436, 81,
                ],
                &[
                    90, 438, 52735, 10, 6515, 14, 223, 3180, 833, 394, 343, 90, 11620, 223, 553,
                    11, 680, 1354, 582, 633, 21390, 1554,
                ],
                &[2162, 28706, 18524, 681, 8566, 3987, 77469, 381],
                &[
                    286, 102, 5879, 246, 5879, 237, 5879, 237, 286, 118, 125567, 5879, 229, 127088,
                    5879, 233, 223, 127167, 25081, 223, 1175, 1148,
                ],
                &[
                    77865, 28, 23597, 12524, 70223, 6011, 26227, 101422, 85639, 14, 32121, 418,
                    7088, 30805, 28, 1123, 52748, 9821, 2212, 61079, 14778,
                ],
            ],
        );
    }

    /// `gigachat` (GPT2, the pre type of GigaChat3). `test-data/gigachat-tokenizer.gguf`: metadata of
    /// https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF/resolve/main/GigaChat3.1-10B-A1.8B-q8_0.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    #[test]
    fn test_gigachat_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "gigachat-tokenizer.gguf",
            PreType::Gpt2,
            &[
                &[
                    19, 20, 21, 22, 23, 223, 21, 16, 19, 22, 19, 23, 27, 223, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 18,
                ],
                &[
                    10960, 1791, 406, 9, 4033, 859, 11705, 1478, 9, 984, 587, 1111, 58053, 9, 38,
                ],
                &[
                    16656, 3621, 3, 223, 15632, 21406, 3227, 86791, 3392, 11918, 13442, 107011,
                    6756, 33745, 49543, 41007, 8265, 46685, 17188, 46685, 90934, 3392, 22212,
                    39438, 258, 34854, 57102, 46762, 113561, 16,
                ],
                &[
                    106322, 28, 10713, 14, 88601, 13906, 14, 7763, 246, 30934, 30934, 35707, 19219,
                    99542, 44850, 14, 15851, 14, 223, 89610, 49690, 50247,
                ],
                &[
                    1568, 1356, 201, 1568, 2189, 204, 127826, 223, 3150, 25357, 200, 200, 93770,
                    750,
                ],
                &[
                    96843, 26293, 242, 236, 82927, 124, 26293, 239, 234, 12618, 239, 234, 3609,
                    254, 100, 42697, 26293, 242, 104, 96644, 97886, 105, 96644, 97886, 103,
                ],
                &[
                    50, 100091, 30982, 10604, 33, 3805, 13269, 3874, 21434, 223, 71202, 23919,
                    25478, 678, 50159, 7, 64, 8, 12,
                ],
                &[127762, 12081, 17375, 391, 40262, 127763],
                &[
                    69, 1536, 1277, 5050, 55702, 1206, 390, 7809, 296, 139, 225, 343, 139, 228, 81,
                ],
                &[
                    90, 298, 24506, 10, 3151, 14, 223, 22, 20, 1396, 201, 389, 347, 90, 3931, 223,
                    19, 18, 11, 358, 636, 372, 735, 13539, 752, 201,
                ],
                &[7546, 28149, 28393, 917, 18269, 5086, 77732, 464],
                &[
                    1984, 102, 17657, 246, 17657, 237, 17657, 237, 1984, 118, 17657, 234, 17657,
                    229, 17657, 245, 17657, 233, 223, 107567, 1984, 243, 1984, 244, 223, 25028,
                    53004,
                ],
                &[
                    42, 16297, 28, 17835, 104, 98821, 99789, 35506, 59807, 45849, 17835, 102,
                    98125, 92636, 54367, 4998, 110, 21536, 14, 49209, 613, 15701, 28, 555, 1482,
                    404, 3711, 760, 1482, 490, 1630, 289,
                ],
            ],
        );
    }

    /// `glm4` (CHATGLM4, the pre type of GLM-4.7-Flash: the LLAMA3 split with `ignore_merges`). `test-data/glm4-tokenizer.gguf`: metadata of
    /// https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    #[test]
    fn test_glm4_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "glm4-tokenizer.gguf",
            PreType::Llama3,
            &[
                &[
                    108714, 100461, 220, 18, 13, 122804, 102573, 220, 108714, 100461, 21, 100928,
                    24, 15,
                ],
                &[
                    14995, 944, 358, 6, 4086, 582, 3003, 807, 93362, 432, 594, 14973, 27612,
                ],
                &[
                    9703, 1879, 0, 101726, 98384, 3837, 99011, 1773, 99799, 131513, 15755, 56451,
                    61369, 70008, 5373, 69868, 45928, 69868, 94551, 1773, 23508, 130419, 30953,
                    53079, 40626, 78318, 13,
                ],
                &[
                    85715, 25, 6364, 11, 133177, 127616, 70363, 11, 128228, 137790, 130624, 11,
                    137930, 11, 220, 49438, 35705, 34339,
                ],
                &[
                    1056, 825, 198, 1056, 1378, 201, 1406, 220, 1257, 15852, 197, 3244, 3435, 5872,
                ],
                &[
                    37328, 61370, 233, 151821, 11157, 236, 231, 151705, 69945, 30423, 61370, 101,
                    124564, 151929, 124564, 151927,
                ],
                &[
                    47, 71744, 12063, 50979, 2503, 12442, 1781, 10126, 220, 45322, 21075, 13980,
                    569, 47775, 45627, 5, 9,
                ],
                &[256, 6388, 12615, 323, 27654, 262],
                &[
                    924, 58451, 94083, 586, 384, 53481, 264, 95739, 308, 151459, 78,
                ],
                &[
                    87, 284, 15217, 54265, 11, 220, 101961, 317, 333, 320, 87, 2604, 220, 98668, 8,
                    314, 470, 330, 562, 5123, 456,
                ],
                &[4102, 5674, 22382, 93217, 3550, 103245, 336],
                &[
                    1540, 99, 15381, 243, 15381, 234, 15381, 234, 1540, 115, 15381, 231, 15381,
                    226, 15381, 242, 15381, 230, 220, 20069, 24842, 33355, 115352, 11314,
                ],
                &[
                    39, 28583, 25, 14913, 101, 86522, 77740, 30366, 97, 34200, 14913, 99, 72094,
                    59678, 42069, 107, 23801, 11, 33952, 448, 4855, 585, 266, 25, 23300, 27815,
                    11066, 58985, 29710, 27815, 21310, 127789, 5703,
                ],
            ],
        );
    }

    /// `dbrx` (DBRX: LLAMA3's split without `ignore_merges`, the pre type of
    /// Granite 4.0). `test-data/granite-tokenizer.gguf`: metadata of
    /// https://huggingface.co/ibm-granite/granite-4.0-h-350m-GGUF/resolve/main/granite-4.0-h-350m-Q8_0.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    /// Strings 3, 4, 6, 9, 11 and 13 differ from what `ignore_merges` would give.
    #[test]
    fn test_dbrx_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "granite-tokenizer.gguf",
            PreType::Dbrx,
            &[
                &[
                    4513, 1774, 220, 18, 13, 9335, 2946, 220, 4513, 10961, 16474, 15,
                ],
                &[
                    15357, 956, 358, 6, 4178, 584, 3077, 814, 95253, 433, 596, 15334, 28805,
                ],
                &[
                    9906, 1917, 0, 220, 57668, 53901, 3922, 3574, 244, 98220, 1811, 9080, 22656,
                    45918, 252, 16144, 57933, 62903, 71634, 5486, 71493, 47307, 71493, 96452, 1811,
                    24486, 89059, 255, 32179, 54535, 41953, 80052, 13,
                ],
                &[
                    87533, 25, 6498, 11, 49520, 44155, 80112, 12415, 71995, 11, 8008, 243, 34586,
                    34586, 42524, 34369, 30862, 68437, 75234, 11, 17607, 24102, 11318, 22071,
                    74541, 11, 220, 50856, 36984, 35609,
                ],
                &[
                    1074, 832, 198, 1074, 1403, 201, 1432, 220, 1280, 16243, 197, 3324, 3518, 5996,
                ],
                &[
                    38623, 62904, 233, 9468, 237, 121, 11410, 236, 231, 9468, 236, 231, 71570,
                    31643, 62904, 101, 378, 235, 9468, 239, 102, 378, 235, 9468, 239, 100,
                ],
                &[
                    47, 73399, 12340, 52417, 2564, 12730, 1819, 10340, 220, 46691, 21806, 14316,
                    571, 49177, 46999, 5, 9,
                ],
                &[256, 6522, 12908, 323, 28848, 262],
                &[
                    936, 59958, 95980, 588, 384, 54939, 264, 97649, 308, 136, 225, 78,
                ],
                &[
                    87, 284, 15586, 55730, 11, 220, 2983, 317, 333, 320, 87, 2669, 220, 605, 8,
                    314, 471, 330, 564, 5233, 457,
                ],
                &[4194, 5792, 23249, 95107, 3634, 378, 225, 336],
                &[
                    1569, 99, 15755, 243, 15755, 234, 15755, 234, 1569, 115, 15755, 231, 15755,
                    226, 15755, 242, 15755, 230, 220, 20713, 25963, 34617, 220, 6447, 11571,
                ],
                &[
                    39, 29785, 25, 15272, 101, 88344, 79468, 31584, 97, 35470, 15272, 99, 73753,
                    61196, 43411, 107, 24810, 11, 35217, 449, 4960, 587, 266, 25, 24252, 29010,
                    11318, 60497, 30925, 29010, 22071, 149, 233, 5821,
                ],
            ],
        );
    }

    /// Granite 4.0 files declare BOS 100257 with `add_bos_token = false`, and
    /// llama.cpp's `dbrx` sets no BOS flag of its own, so no BOS is prepended.
    #[test]
    fn test_dbrx_granite_adds_no_bos_if_fixture_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data/granite-tokenizer.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let file = GgufFile::open(&path).expect("open granite-tokenizer.gguf");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf");
        assert_eq!(tok.prompt_bos(), None);
    }

    /// llama.cpp sets `special_bos_id = LLAMA_TOKEN_NULL` for `glm4`, so a GLM
    /// prompt never starts with BOS even though the file declares one (154822)
    /// and has no `add_bos_token` key.
    #[test]
    fn test_glm4_adds_no_bos_if_fixture_present() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/glm4-tokenizer.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let file = GgufFile::open(&path).expect("open glm4-tokenizer.gguf");
        let tok = Tokenizer::from_gguf(&file).expect("load glm4 tokenizer");
        assert_eq!(tok.bos_token_id, Some(154822));
        assert!(!tok.add_bos);
        assert_eq!(tok.prompt_bos(), None);
    }

    /// `qwen35` (QWEN35; Qwen3.5 and Qwen3.8). `test-data/qwen35-tokenizer.gguf`:
    /// metadata of
    /// https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
    /// (HTTP range `bytes=0-41943039`, rewritten by `scripts/gguf_tokenizer_only.py`).
    /// Before pre types were read these files got the Qwen2 split, which
    /// differs on combining marks (the Devanagari/Arabic string).
    #[test]
    fn test_qwen35_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "qwen35-tokenizer.gguf",
            PreType::Qwen35,
            &[
                &[
                    16, 17, 18, 19, 20, 220, 18, 13, 16, 19, 16, 20, 24, 220, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 15,
                ],
                &[
                    14572, 914, 353, 6, 3950, 567, 2908, 781, 90902, 424, 579, 14551, 26807,
                ],
                &[
                    9419, 1814, 0, 220, 109266, 3709, 96748, 1710, 247359, 15303, 210342, 5205,
                    167281, 221215, 1710, 174675, 30061, 51643, 39504, 151396, 13,
                ],
                &[
                    83445, 25, 6163, 11, 209428, 185386, 11, 150098, 164594, 159778, 11, 171405,
                    11, 181204,
                ],
                &[
                    1021, 799, 198, 1021, 1330, 201, 1358, 220, 1214, 15397, 197, 3139, 3325, 5690,
                ],
                &[
                    36280, 59720, 233, 9008, 237, 121, 10838, 236, 231, 9008, 236, 231, 68089,
                    29545, 59720, 101, 373, 235, 9008, 239, 102, 373, 235, 9008, 239, 100,
                ],
                &[
                    47, 69843, 11726, 49591, 2423, 12091, 1718, 9831, 220, 44074, 20475, 13587,
                    554, 46474, 44370, 5, 9,
                ],
                &[256, 6187, 12258, 321, 26849, 262],
                &[
                    895, 56868, 91603, 571, 378, 52033, 264, 93213, 307, 136, 225, 78,
                ],
                &[
                    87, 283, 14785, 52794, 11, 220, 19, 17, 1174, 198, 331, 318, 87, 2518, 220, 16,
                    15, 8, 313, 460, 328, 547, 4961, 333, 198,
                ],
                &[3966, 5496, 21742, 90762, 3433, 373, 225, 334],
                &[
                    169327, 14945, 243, 14945, 234, 14945, 234, 169617, 169730, 14945, 226, 14945,
                    242, 14945, 230, 220, 19496, 24128, 32405, 220, 190500,
                ],
                &[
                    39, 27746, 25, 149518, 84237, 150104, 153348, 184642, 235886, 11, 32988, 440,
                    4700, 178848, 25, 22639, 164865, 28850, 150765, 150021,
                ],
            ],
        );
    }

    /// `gpt-2` (GPT2), on the synthetic `test-data/tiny-qwen3moe.gguf` (its
    /// hand-built tokenizer reports `gpt-2`; source in
    /// `test-data/tiny-qwen3moe-src.tar.gz`).
    #[test]
    fn test_gpt2_pre_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "tiny-qwen3moe.gguf",
            PreType::Gpt2,
            &[
                &[10163, 2231, 513, 13, 1415, 19707, 17031, 2231, 30924, 3829],
                &[
                    9099, 470, 314, 6, 3069, 356, 1053, 484, 6, 2200, 340, 338, 7013, 6, 35,
                ],
                &[
                    15496, 995, 0, 220, 19526, 254, 25001, 121, 171, 120, 234, 10310, 244, 45911,
                    234, 16764, 33768, 98, 17312, 105, 45739, 252, 5641, 24336, 25084, 43302,
                    23513, 21763, 23376, 21763, 26229, 16764, 47991, 250, 166, 113, 255, 168, 244,
                    112, 31619, 105, 116, 168, 252, 98, 168, 252, 227, 46695, 230, 46695, 97, 13,
                ],
                &[
                    44, 2966, 25, 3594, 11, 12466, 254, 35072, 21727, 21727, 31583, 18849, 140,
                    117, 220, 20375, 16843, 31583, 21727, 20375, 11, 7377, 243, 39377, 39377, 138,
                    115, 26180, 29945, 43000, 138, 105, 11, 28981, 44690, 26897, 39848, 22654,
                    45632, 11, 220, 31479, 226, 19567, 245, 19567, 95,
                ],
                &[
                    1370, 530, 198, 1370, 734, 201, 628, 198, 220, 773, 4714, 197, 197, 8658, 82,
                    220, 220, 220, 198,
                ],
                &[
                    368, 31370, 50169, 233, 8582, 237, 121, 12520, 236, 231, 8582, 236, 231, 43074,
                    97, 37929, 50169, 101, 447, 235, 41840, 102, 447, 235, 41840, 100,
                ],
                &[
                    47, 16260, 2288, 10185, 34913, 2644, 11420, 14808, 3419, 4008, 220, 1391, 58,
                    27, 37981, 92, 2488, 29953, 4, 61, 5, 9,
                ],
                &[220, 220, 3756, 9029, 290, 25462, 220, 220, 220],
                &[
                    66, 1878, 2634, 41492, 304, 136, 223, 257, 136, 222, 299, 136, 225, 78,
                ],
                &[
                    87, 796, 22944, 7, 5657, 11, 5433, 1776, 198, 361, 357, 87, 18189, 838, 8,
                    1391, 1441, 366, 482, 8172, 1782, 198,
                ],
                &[1849, 77, 24145, 5099, 222, 485, 6826, 2272, 447, 225, 368],
                &[
                    171, 120, 99, 171, 121, 243, 171, 121, 234, 171, 121, 234, 171, 120, 115, 171,
                    121, 231, 171, 121, 226, 171, 121, 242, 171, 121, 230, 27332, 120, 239, 171,
                    120, 240, 171, 120, 241, 27332, 120, 223, 171, 120, 253,
                ],
                &[
                    39, 521, 72, 25, 28225, 101, 11976, 106, 11976, 116, 24231, 235, 11976, 97,
                    24231, 229, 28225, 99, 24231, 223, 11976, 101, 11976, 123, 11976, 107, 48077,
                    11, 17526, 351, 3971, 461, 265, 25, 47048, 24333, 26897, 48763, 148, 255,
                    24333, 39848, 149, 233, 12919,
                ],
            ],
        );
    }

    /// `qwen2` (QWEN2), on `test-data/Qwen3-0.6B-Q4_K_M.gguf`. Index 12
    /// (Arabic with harakat, Devanagari) is what the table-based split fixed:
    /// the Rust-`is_alphabetic` port it replaced counted harakat as letters
    /// and missed it (see docs/design/tokenizer-pre-types.md, "Resolved").
    #[test]
    fn test_qwen2_encode_matches_llama_tokenize_if_fixture_present() {
        assert_pre_type_golden_if_fixture_present(
            "Qwen3-0.6B-Q4_K_M.gguf",
            PreType::Qwen2,
            &[
                &[
                    16, 17, 18, 19, 20, 220, 18, 13, 16, 19, 16, 20, 24, 220, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 15,
                ],
                &[
                    15007, 944, 358, 6, 4086, 582, 3003, 807, 94153, 432, 594, 14985, 27705,
                ],
                &[
                    9707, 1879, 0, 220, 108386, 3837, 99489, 1773, 101059, 102819, 15767, 56833,
                    61803, 70534, 5373, 70393, 46207, 70393, 95352, 1773, 23573, 124785, 31079,
                    53435, 40853, 78952, 13,
                ],
                &[
                    86433, 25, 6364, 11, 140869, 126302, 70895, 11, 7851, 243, 33486, 33486, 41424,
                    33269, 29762, 67337, 74134, 11, 129071, 11, 220, 125451,
                ],
                &[
                    1056, 825, 198, 1056, 1378, 201, 1406, 220, 1257, 15864, 197, 3244, 3435, 5872,
                ],
                &[
                    37523, 61804, 233, 145375, 11162, 236, 231, 144841, 70470, 30543, 61804, 101,
                    378, 235, 145233, 378, 235, 145665,
                ],
                &[
                    47, 72299, 12069, 51317, 2503, 12448, 1781, 10131, 220, 45591, 21122, 13989,
                    569, 48077, 45899, 5, 9,
                ],
                &[256, 6388, 12621, 323, 27748, 262],
                &[
                    924, 58858, 94880, 586, 384, 53839, 264, 96549, 308, 136, 225, 78,
                ],
                &[
                    87, 284, 15229, 54630, 11, 220, 19, 17, 317, 333, 320, 87, 2604, 220, 16, 15,
                    8, 314, 470, 330, 562, 5123, 456,
                ],
                &[4102, 5674, 22441, 94007, 3550, 378, 225, 336],
                &[
                    141161, 144552, 144433, 144433, 144084, 144155, 144514, 144297, 144455, 220,
                    20109, 24918, 33517, 220, 6313, 11319,
                ],
                &[
                    39, 28685, 25, 14925, 101, 87244, 78368, 30484, 97, 34370, 14925, 99, 72653,
                    60096, 42311, 107, 23868, 11, 34117, 448, 4855, 585, 266, 25, 23364, 27910,
                    11071, 59397, 29825, 27910, 21360, 127837,
                ],
            ],
        );
    }

    /// Qwen2.5 special tokens, matched literally before BPE: ChatML, the
    /// Qwen2.5-Math chain-of-thought prompt, Coder's FIM and repo-level FIM
    /// tokens (`<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`,
    /// `<|repo_name|>`/`<|file_sep|>`, all CONTROL), the USER_DEFINED
    /// `<tool_call>` markers, and near misses that must not match. Golden ids
    /// from llama.cpp `79e2e74`'s `llama-tokenize --stdin --ids --no-bos`
    /// (special tokens parsed, its default) on the tokenizer-only version of
    /// the same file. `test-data/qwen2.5-0.5b-instruct-header.gguf`: header of
    /// https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
    /// (HTTP range `bytes=0-41943039`, `scripts/gguf_tokenizer_only.py
    /// --keep-tensor-table`); every Qwen2.5 GGUF surveyed has the same vocab
    /// (docs/design/qwen2.md).
    #[test]
    fn test_qwen25_special_tokens_match_llama_tokenize_if_fixture_present() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("qwen2.5-0.5b-instruct-header.gguf");
        if !path.exists() {
            eprintln!("Skipping Qwen2.5 special-token test: {path:?} not present");
            return;
        }
        let file = GgufFile::open(&path).expect("open Qwen2.5 header");
        let tok = Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf");
        assert_eq!(tok.pre_type, Some(PreType::Qwen2));
        assert_eq!(tok.prompt_bos(), None, "Qwen2.5 sets add_bos_token = false");
        assert_eq!(tok.eos_token_id, Some(151_645));
        let golden: &[(&str, &[u32])] = &[
            (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello! How are you today?<|im_end|>\n<|im_start|>assistant\n",
                &[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 0, 2585, 525, 498, 3351, 30, 151645, 198, 151644, 77091, 198],
            ),
            (
                "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nSolve $3x + 7 = 22$, then compute $\\frac{1}{2} + \\sqrt{16}$ and $2^{10} = 1024$.<|im_end|>\n<|im_start|>assistant\n",
                &[151644, 8948, 198, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878, 1124, 79075, 46391, 151645, 198, 151644, 872, 198, 50, 3948, 400, 18, 87, 488, 220, 22, 284, 220, 17, 17, 54876, 1221, 12564, 57960, 37018, 90, 16, 15170, 17, 92, 488, 1124, 26888, 90, 16, 21, 31716, 323, 400, 17, 47822, 16, 15, 92, 284, 220, 16, 15, 17, 19, 12947, 151645, 198, 151644, 77091, 198],
            ),
            (
                "<|fim_prefix|>def fibonacci(n):\n    if n < 2:\n        return n\n<|fim_suffix|>\n\nprint(fibonacci(10))\n<|fim_middle|>",
                &[151659, 750, 75698, 1445, 982, 262, 421, 308, 366, 220, 17, 510, 286, 470, 308, 198, 151661, 271, 1350, 955, 579, 39345, 7, 16, 15, 1171, 151660],
            ),
            (
                "<|repo_name|>reflex\n<|file_sep|>src/main.rs\nfn main() {\n    println!(\"hi\");\n}\n<|file_sep|>src/lib.rs\n<|fim_prefix|>pub fn add(a: i32, b: i32) -> i32 {\n    <|fim_suffix|>\n}\n<|fim_middle|>",
                &[151663, 1097, 2571, 198, 151664, 3548, 15351, 25638, 198, 8822, 1887, 368, 341, 262, 13751, 17223, 6023, 797, 532, 151664, 3548, 8194, 25638, 198, 151659, 9585, 5168, 912, 2877, 25, 600, 18, 17, 11, 293, 25, 600, 18, 17, 8, 1464, 600, 18, 17, 341, 257, 151661, 198, 532, 151660],
            ),
            (
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|endoftext|>done",
                &[151657, 198, 4913, 606, 788, 330, 455, 69364, 497, 330, 16370, 788, 5212, 8926, 788, 330, 59604, 95642, 151658, 151643, 10438],
            ),
            (
                "<|im_start>user <im_end|> <|fim_prefix |> <tool_call >",
                &[27, 91, 318, 4906, 29, 872, 366, 318, 6213, 91, 29, 82639, 69, 318, 13974, 58721, 366, 14172, 13429, 861],
            ),
        ];
        for &(text, expected) in golden {
            let ids = tok
                .encode(text)
                .unwrap_or_else(|e| panic!("encode({text:?}) failed: {e}"));
            assert_eq!(ids, expected, "ids differ from llama-tokenize for {text:?}");
            assert_eq!(tok.decode(&ids), text, "round trip failed for {text:?}");
        }
    }

    fn fixture_tokenizer(file_name: &str) -> Option<Tokenizer> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join(file_name);
        if !path.exists() {
            eprintln!("Skipping: test-data/{file_name} not present");
            return None;
        }
        let file = GgufFile::open(&path).expect("open tokenizer GGUF");
        Some(Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf"))
    }

    /// Encodes `parts` the way a chain of resumed requests does: each part after
    /// the state the parts before it left.
    fn encode_chain(tok: &Tokenizer, parts: &[&str]) -> Vec<u32> {
        let mut after_special = true;
        let mut ids = Vec::new();
        for p in parts {
            ids.extend(tok.encode_continuation(p, after_special).unwrap());
            after_special = tok.ends_after_special(p, after_special);
        }
        ids
    }

    /// On a real SentencePiece vocab (TinyLlama-1.1B-Chat,
    /// `test-data/tinyllama-1.1b-chat-tokenizer.gguf`), a chain of continuations
    /// encodes like the joined text, whether a split falls after raw text (no
    /// dummy-prefix space) or after the special token `</s>` (a space). Encoding
    /// each part as a fresh prompt gets the raw-text splits wrong.
    #[test]
    fn test_spm_continuation_matches_joined_text_if_fixture_present() {
        let Some(tok) = fixture_tokenizer("tinyllama-1.1b-chat-tokenizer.gguf") else {
            return;
        };
        assert_eq!(tok.architecture, "llama");
        let user = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
        let cases: [(&[&str], bool); 5] = [
            (
                &["<|system|>\nYou are a careful assistant.</s>\n", user],
                false,
            ),
            (
                &[
                    "<|system|>\nYou are a careful assistant.</s>",
                    &format!("\n{user}"),
                ],
                true,
            ),
            (
                &[
                    "You are a careful assistant in a team of agents. Answer briefly.",
                    " The capital of France is",
                    " Paris, and the capital of Italy is",
                ],
                false,
            ),
            (&["Hello world", "</s>", "Bye", " now"], false),
            (&["</s>", "\nHi"], true),
        ];
        for (parts, fresh_encoding_works) in cases {
            let joined = tok.encode(&parts.concat()).unwrap();
            assert_eq!(encode_chain(&tok, parts), joined, "{parts:?}");
            let fresh: Vec<u32> = parts.iter().flat_map(|p| tok.encode(p).unwrap()).collect();
            assert_eq!(fresh == joined, fresh_encoding_works, "{parts:?}");
        }
        assert!(tok.ends_after_special("abc</s>", false));
        assert!(!tok.ends_after_special("</s>\n", true));
        assert!(tok.ends_after_special("", true));
        assert!(!tok.ends_after_special("", false));
        // The SentencePiece dummy prefix is the only difference.
        assert_eq!(
            tok.encode_continuation("Hi", true).unwrap()[0],
            tok.token_to_id["\u{2581}Hi"]
        );
    }

    /// gpt2 byte-level BPE has no dummy prefix: a continuation encodes the same
    /// in either state.
    #[test]
    fn test_gpt2_continuation_ignores_state_if_fixture_present() {
        let Some(tok) = fixture_tokenizer("qwen3-0.6b-tokenizer.gguf") else {
            return;
        };
        for text in [
            "Hello world",
            " leading space",
            "<|im_start|>user\nHi<|im_end|>\n",
        ] {
            assert_eq!(
                tok.encode_continuation(text, false).unwrap(),
                tok.encode(text).unwrap(),
                "{text:?}"
            );
        }
        assert!(tok.ends_after_special("hi<|im_end|>", false));
        assert!(!tok.ends_after_special("<|im_end|>\n", true));
    }
}
