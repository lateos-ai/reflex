//! Tokenizer extraction from GGUF metadata, plus from-scratch encode/decode
//! for two real tokenizer schemes: llama.cpp's SentencePiece-derived scheme
//! (`tokenizer.ggml.model == "llama"`, Phase 21.5.1) and GPT-2-style
//! byte-level BPE (`tokenizer.ggml.model == "gpt2"`, Phase 21.14 — Qwen3
//! architecture support). [`Tokenizer::encode`]/[`Tokenizer::decode`]
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
//! **GPT-2/Qwen2 path — not yet verified against a real reference
//! tokenizer** (e.g. Python `transformers`/`tokenizers` on the same
//! string): the pre-tokenization regex was hand-ported from real llama.cpp
//! source (`LLAMA_VOCAB_PRE_TYPE_QWEN2`'s exact pattern), and the algorithm
//! runs and produces well-formed, in-vocab output on a real Qwen3-0.6B GGUF
//! (21.14.1's fixture), but byte-for-byte agreement with a real reference on
//! arbitrary text is `PHASE21_14_PLAN.md`'s 21.14.2b's own remaining gate,
//! same "flag the real gap honestly" posture this module's SentencePiece
//! path already held before its own real-file verification.

use crate::gguf::{GgufFile, GgufValue};
use std::collections::HashMap;

const WORD_BOUNDARY: char = '\u{2581}'; // '▁', SentencePiece's space marker

fn byte_fallback_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

fn parse_byte_fallback(tok: &str) -> Option<u8> {
    let inner = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}

/// A tokenizer extracted from a GGUF file's `tokenizer.ggml.*` metadata keys.
pub struct Tokenizer {
    pub architecture: String,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_type: Vec<i32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
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
}

impl Tokenizer {
    /// Extract a tokenizer from `file`'s metadata. Requires
    /// `tokenizer.ggml.model` and `tokenizer.ggml.tokens`; every other key
    /// (`scores`, `token_type`, `merges`, special-token ids) is optional and
    /// defaults to empty/`None` when absent.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, String> {
        let architecture = string_meta(&file.metadata, "tokenizer.ggml.model")?.to_string();
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

        let mut merge_rank = HashMap::with_capacity(merges_raw.len());
        for (rank, entry) in merges_raw.iter().enumerate() {
            let mut parts = entry.split(' ');
            let left = parts.next().filter(|s| !s.is_empty()).ok_or_else(|| {
                format!("malformed merge entry '{entry}' (expected 'left right')")
            })?;
            let right = parts.next().filter(|s| !s.is_empty()).ok_or_else(|| {
                format!("malformed merge entry '{entry}' (expected 'left right')")
            })?;
            merge_rank.insert((left.to_string(), right.to_string()), rank);
        }

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let bos_token_id =
            u64_meta(&file.metadata, "tokenizer.ggml.bos_token_id").map(|v| v as u32);
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

        Ok(Tokenizer {
            architecture,
            tokens,
            scores,
            token_type,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id,
            token_to_id,
            merge_rank,
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Number of BPE merge rules loaded, for tests that need to assert
    /// metadata extraction happened without reaching into private fields.
    pub fn merge_count(&self) -> usize {
        self.merge_rank.len()
    }

    /// Encode `text` to token ids. Dispatches on `self.architecture`
    /// (`tokenizer.ggml.model`, real GGUF values confirmed: `"llama"`
    /// SentencePiece-derived, `"gpt2"` byte-level BPE, Phase 21.14 —
    /// see [`Self::encode_gpt2`]) — every other value falls back to the
    /// SentencePiece path unchanged, this module's original Phase 21.5.1
    /// scope.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        if self.special_tokens.is_empty() {
            return self.encode_plain(text);
        }

        let mut ids = Vec::new();
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
                        ids.extend(self.encode_plain(&text[plain_start..i])?);
                    }
                    // Only reachable when `special_tokens` (built from `tokens`) is
                    // non-empty, so the lookup below always succeeds.
                    ids.push(self.token_to_id[special.as_str()]);
                    i += special.len();
                    plain_start = i;
                }
                None => {
                    i += text[i..].chars().next().map(char::len_utf8).unwrap_or(1);
                }
            }
        }
        if plain_start < text.len() {
            ids.extend(self.encode_plain(&text[plain_start..])?);
        }
        Ok(ids)
    }

    /// The general encode path (regex-pretokenization + BPE, dispatched on
    /// `self.architecture`) -- used for every span of `text` that isn't a
    /// literal special-token match. See [`Self::encode`]'s doc for why
    /// special tokens (this struct's `special_tokens` field) are matched
    /// separately, before this ever runs on their text.
    fn encode_plain(&self, text: &str) -> Result<Vec<u32>, String> {
        if self.architecture == "gpt2" {
            self.encode_gpt2(text)
        } else {
            self.encode_sentencepiece(text)
        }
    }

    /// Encode `text` to token ids via greedy lowest-rank BPE merging.
    ///
    /// Preprocessing: every space becomes [`WORD_BOUNDARY`] and a leading
    /// `WORD_BOUNDARY` is prepended for non-empty text (SentencePiece's
    /// "text is preceded by an implicit space" convention — confirmed
    /// against the real TinyLlama reference tokenizer, see
    /// `PHASE21_5_PLAN.md`'s Known risks; empty text is the one case where
    /// the reference does *not* apply the dummy prefix, matched here by
    /// skipping it too, rather than emitting a stray leading-space token for
    /// an empty prompt). Each resulting character becomes one symbol if the
    /// vocab has it directly; otherwise it's decomposed into UTF-8 bytes,
    /// each represented by a `<0xXX>` byte-fallback token. Errs if a
    /// byte-fallback token or a final merged symbol isn't in the vocab.
    fn encode_sentencepiece(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut preprocessed = String::with_capacity(text.len() + 1);
        if !text.is_empty() {
            preprocessed.push(WORD_BOUNDARY);
        }
        for ch in text.chars() {
            preprocessed.push(if ch == ' ' { WORD_BOUNDARY } else { ch });
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
                    return Err(format!(
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
                self.token_to_id
                    .get(&s)
                    .copied()
                    .ok_or_else(|| format!("merged token '{s}' not found in vocab"))
            })
            .collect()
    }

    /// Encode `text` to token ids via GPT-2-style byte-level BPE (Phase
    /// 21.14, Qwen3 architecture support — real `tokenizer.ggml.model =
    /// "gpt2"` / `tokenizer.ggml.pre = "qwen2"`, confirmed against a real
    /// Qwen3-0.6B GGUF's own metadata). Two real differences from
    /// [`Self::encode_sentencepiece`], not a small variation of it:
    ///
    /// 1. **Regex-based pre-tokenization** ([`qwen2_pretokenize`]) splits
    ///    `text` into maximal chunks *before* any BPE merging happens — a
    ///    merge can never cross a pre-token boundary. Ported by hand (see
    ///    that function's own doc for why: no `regex` crate dependency,
    ///    matching this crate's established from-scratch-small-helper
    ///    convention, e.g. [`Xorshift64`] over the `rand` crate) from the
    ///    real llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` pattern:
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
    /// **Not yet verified against a real reference tokenizer** (e.g. Python
    /// `transformers`/`tokenizers` output on the same string) — same honest
    /// caveat this module's own SentencePiece path already carried before
    /// its own real-file verification; `PHASE21_14_PLAN.md`'s 21.14.2b is
    /// where that verification happens.
    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>, String> {
        let chars: Vec<char> = text.chars().collect();
        let byte_to_unicode = gpt2_byte_to_unicode_table();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let end = qwen2_pretokenize_one(&chars, i);
            let piece: String = chars[i..end].iter().collect();
            i = end;

            let mut symbols: Vec<String> = Vec::with_capacity(piece.len());
            for &b in piece.as_bytes() {
                symbols.push(byte_to_unicode[b as usize].to_string());
            }
            self.bpe_merge(&mut symbols);
            for s in symbols {
                let id = self.token_to_id.get(&s).copied().ok_or_else(|| {
                    format!("gpt2 BPE symbol {s:?} not found in vocab (piece {piece:?})")
                })?;
                ids.push(id);
            }
        }
        Ok(ids)
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

/// `\p{L}` proxy: Rust's `char::is_alphabetic()` is Unicode-alphabetic-aware
/// but not a byte-for-byte match to Unicode's `L*` general-category union —
/// close enough for real text in the common case, not yet verified for
/// every Unicode edge case a real reference tokenizer might diverge on
/// (`PHASE21_14_PLAN.md`'s 21.14.2b correctness gate is where that gets
/// checked against a real reference, not assumed exact here).
fn is_l(c: char) -> bool {
    c.is_alphabetic()
}

/// `\p{N}` proxy — see [`is_l`]'s same caveat.
fn is_n(c: char) -> bool {
    c.is_numeric()
}

fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_l(c) && !is_n(c)
}

/// Hand-ported equivalent of the real llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`
/// pre-tokenizer regex (see [`Tokenizer::encode_gpt2`]'s own doc for the
/// exact pattern and citation) — no `regex` crate dependency (this crate's
/// established convention, see [`Xorshift64`]). Returns the end index
/// (exclusive) of the single maximal pre-token starting at `chars[start]`,
/// by trying each of the pattern's 7 alternatives in the same left-to-right
/// priority order a real regex engine's alternation would, taking whichever
/// matches first. Always returns `> start` (falls back to consuming exactly
/// one character if — this shouldn't be reachable for any real input, since
/// alternative 4's `[^\s\p{L}\p{N}]+` alone already covers every character
/// that isn't whitespace/letter/digit — none of the 7 alternatives match,
/// which would otherwise loop forever).
fn qwen2_pretokenize_one(chars: &[char], start: usize) -> usize {
    match_contraction(chars, start)
        .or_else(|| match_word(chars, start))
        .or_else(|| match_digit(chars, start))
        .or_else(|| match_other(chars, start))
        .or_else(|| match_blank_lines(chars, start))
        .or_else(|| match_whitespace_not_followed_by_nonspace(chars, start))
        .or_else(|| match_whitespace_run(chars, start))
        .unwrap_or(start + 1)
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
fn match_contraction(chars: &[char], i: usize) -> Option<usize> {
    if chars[i] != '\'' {
        return None;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let end = i + 1 + suffix.len();
        if end <= chars.len()
            && chars[i + 1..end]
                .iter()
                .collect::<String>()
                .eq_ignore_ascii_case(suffix)
        {
            return Some(end);
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+`
fn match_word(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    if is_l(chars[j]) {
        // No optional leading char needed — the letter run starts here.
    } else if chars[j] != '\r' && chars[j] != '\n' && !is_n(chars[j]) {
        // Candidate optional leading char, only real if a letter follows.
        if j + 1 < chars.len() && is_l(chars[j + 1]) {
            j += 1;
        } else {
            return None;
        }
    } else {
        return None;
    }
    if j >= chars.len() || !is_l(chars[j]) {
        return None;
    }
    while j < chars.len() && is_l(chars[j]) {
        j += 1;
    }
    Some(j)
}

/// `\p{N}` — exactly one digit character (Qwen2's own pattern, unlike
/// original GPT-2's `\p{N}+`, deliberately does not group digit runs).
fn match_digit(chars: &[char], i: usize) -> Option<usize> {
    if is_n(chars[i]) {
        Some(i + 1)
    } else {
        None
    }
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
fn match_other(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    if chars[j] == ' ' && j + 1 < chars.len() && is_other(chars[j + 1]) {
        j += 1;
    }
    let run_start = j;
    while j < chars.len() && is_other(chars[j]) {
        j += 1;
    }
    if j == run_start {
        return None;
    }
    while j < chars.len() && (chars[j] == '\r' || chars[j] == '\n') {
        j += 1;
    }
    Some(j)
}

/// `\s*[\r\n]+` — matches through the *last* newline in the contiguous
/// whitespace run starting at `i` (see [`Tokenizer::encode_gpt2`]'s doc
/// comment history / this function's own derivation: greedy `\s*` backtracks
/// minimally, from the end, until `[\r\n]+` can match — the first such
/// position is always the run's last newline character). `None` if that run
/// contains no newline at all.
fn match_blank_lines(chars: &[char], i: usize) -> Option<usize> {
    if !chars[i].is_whitespace() {
        return None;
    }
    let mut j = i;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    (i..j)
        .rev()
        .find(|&k| chars[k] == '\r' || chars[k] == '\n')
        .map(|k| k + 1)
}

/// `\s+(?!\S)` — the whole contiguous whitespace run if it reaches
/// end-of-input; otherwise the run minus its final character (that last
/// whitespace char is left for the *next* scan position to combine with
/// whatever non-whitespace follows, via [`match_word`]/[`match_other`]'s own
/// "optional single leading char" branch — the real mechanism behind "one
/// leading space glues to the following word, extra leading spaces don't").
/// `None` for a length-1 run followed by non-whitespace (the `(?!\S)`
/// lookahead can't be satisfied by a zero-length match), which correctly
/// falls through to [`match_whitespace_run`] for that case.
fn match_whitespace_not_followed_by_nonspace(chars: &[char], i: usize) -> Option<usize> {
    if !chars[i].is_whitespace() {
        return None;
    }
    let mut j = i;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    if j == chars.len() {
        return Some(j);
    }
    if j - i >= 2 {
        return Some(j - 1);
    }
    None
}

/// `\s+` — plain greedy whitespace run, the final fallback alternative.
fn match_whitespace_run(chars: &[char], i: usize) -> Option<usize> {
    if !chars[i].is_whitespace() {
        return None;
    }
    let mut j = i;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    Some(j)
}

fn u64_meta(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(GgufValue::as_u64)
}

fn string_meta<'a>(metadata: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a str, String> {
    metadata
        .get(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| format!("missing or non-string metadata key '{key}'"))
}

fn string_array_value(value: &GgufValue, key: &str) -> Result<Vec<String>, String> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("expected string array element in '{key}', got {v:?}"))
            })
            .collect(),
        other => Err(format!("expected array for '{key}', got {other:?}")),
    }
}

fn string_array(metadata: &HashMap<String, GgufValue>, key: &str) -> Result<Vec<String>, String> {
    let value = metadata
        .get(key)
        .ok_or_else(|| format!("missing metadata key '{key}'"))?;
    string_array_value(value, key)
}

fn f32_array(value: &GgufValue, key: &str) -> Result<Vec<f32>, String> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| {
                v.as_f32()
                    .ok_or_else(|| format!("expected f32 array element in '{key}', got {v:?}"))
            })
            .collect(),
        other => Err(format!("expected array for '{key}', got {other:?}")),
    }
}

fn i32_array(value: &GgufValue, key: &str) -> Result<Vec<i32>, String> {
    match value {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| match v {
                GgufValue::I32(x) => Ok(*x),
                GgufValue::I8(x) => Ok(*x as i32),
                GgufValue::I16(x) => Ok(*x as i32),
                other => Err(format!(
                    "expected i32 array element in '{key}', got {other:?}"
                )),
            })
            .collect(),
        other => Err(format!("expected array for '{key}', got {other:?}")),
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
        let token_to_id: HashMap<String, u32> = tokens
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
            tokens,
            scores: Vec::new(),
            token_type: Vec::new(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unk_token_id: Some(0),
            pad_token_id: None,
            token_to_id,
            merge_rank,
            special_tokens: Vec::new(),
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

    /// Phase 21.14's own `qwen2_pretokenize_one` — pre-tokenizer boundary
    /// checks against the real llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` regex's
    /// documented behavior (no GPU, no real fixture needed — pure host-side
    /// function, same "test the algorithm in isolation" precedent as
    /// `dequant.rs`'s own hand-computed-expected-value tests). Splits `text`
    /// into pieces by repeatedly calling `qwen2_pretokenize_one`.
    fn qwen2_pretokenize_all(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let end = qwen2_pretokenize_one(&chars, i);
            assert!(
                end > i,
                "qwen2_pretokenize_one must always make forward progress"
            );
            pieces.push(chars[i..end].iter().collect::<String>());
            i = end;
        }
        pieces
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
    /// Qwen3-0.6B GGUF on real GPU hardware before this fix (see HISTORY.md).
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
}
