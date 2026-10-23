# Tokenizer pre-types (`tokenizer.ggml.pre`)

A `tokenizer.ggml.model == "gpt2"` GGUF vocab is byte-level BPE, and its merges are
only correct if the text is first split the way the model's tokenizer splits it. The
split regex is named by `tokenizer.ggml.pre`. Reflex used to ignore that key and give
every gpt2 vocab the Qwen2 regex, so a Llama-3 or DeepSeek GGUF loaded and tokenized
wrongly with no error. Now `Tokenizer::from_gguf` maps the value to `PreType`
(`src/tokenizer.rs`) and rejects any value it has no port for, naming it.

SentencePiece (`tokenizer.ggml.model == "llama"`) and `gemma4` vocabs don't split with a
regex. They get no pre type and their `pre` key, if any, isn't read.

## Supported pre types

Reference: llama.cpp `79e2e74`, the commit every llama.cpp reference build in this repo
uses. Regex lists are in `src/llama-vocab.cpp` (the `llm_tokenizer_bpe` constructor),
the code that runs them in `src/unicode.cpp`.

| `pre` value | `PreType` | llama.cpp type | How llama.cpp runs it | Golden fixture |
|---|---|---|---|---|
| `qwen2` | `Qwen2` | `QWEN2` | `unicode_regex_split_custom_qwen2` | `Qwen3-0.6B-Q4_K_M.gguf` |
| `kolibri1` | `Qwen2` | none at `79e2e74`; the community Kolibri-1 patch maps it to `QWEN2` | as `qwen2` | `kolibri1-tokenizer.gguf` (vs HF `tokenizers`) |
| *(key missing)* | `Qwen2` | `DEFAULT`, with a "generation quality will be degraded" warning | | |
| `qwen35` | `Qwen35` | `QWEN35` | `unicode_regex_split_custom_qwen35` | `qwen35-tokenizer.gguf` |
| `llama-bpe` | `Llama3` | `LLAMA3` | `unicode_regex_split_custom_llama3`, plus `ignore_merges` | `llama-bpe-tokenizer.gguf` |
| `gpt-2` | `Gpt2` | `GPT2` | `unicode_regex_split_custom_gpt2` | `tiny-qwen3moe.gguf` |
| `deepseek-llm` | `DeepseekLlm` | `DEEPSEEK_LLM` | 6 regexes through `std::regex` | `deepseek-llm-tokenizer.gguf` |
| `deepseek-v3` | `Deepseek3` | `DEEPSEEK3_LLM` | 3 regexes through `std::regex` | `deepseek-v3-tokenizer.gguf` |
| `gigachat` | `Gpt2` | `GPT2` | as `gpt-2` | `gigachat-tokenizer.gguf` |
| `glm4` | `Llama3` | `CHATGLM4`: the `LLAMA3` regex, `ignore_merges`, and no BOS (`special_bos_id` nulled) | as `llama-bpe`, BOS off | `glm4-tokenizer.gguf` |
| `pixtral`, `tekken`, `nemotron-3.5-embed` | `Tekken` | `pixtral` is `LLAMA3` at `79e2e74` and `TEKKEN` in later llama.cpp; `nemotron-3.5-embed` is later llama.cpp's `NEMOTRON_3_5_EMBD` | Mistral's Tekken regex with `ignore_merges`, matched with HF `tokenizers`' semantics (see the gotcha below) | `nemotron-3.5-embed-tokenizer.gguf` (vs HF `tokenizers`) |

A missing key keeps the Qwen2 split, which is what every gpt2 vocab got before. That way
a file that loaded before still tokenizes the same (apart from the 2026-10-13 `qwen2`
fix for combining marks, see [Resolved](#resolved-the-original-qwen2-port)). No GGUF checked so far lacks the key
except TheBloke's older conversions (e.g. `deepseek-llm-7B-chat-GGUF`), where llama.cpp
itself falls back to a regex Reflex doesn't port.

Aliases llama.cpp accepts for the same split (`llama3` and `llama-v3` for `llama-bpe`,
`deepseek-r1-qwen` for `qwen2`, ...) are not accepted yet. Each one is a one-line
addition to `PreType::ACCEPTED`, once a real file needs it.

What uses which pre type, from a header survey (HTTP range requests on the first 40 MB):

| File | `general.architecture` | `pre` |
|---|---|---|
| Qwen3-0.6B, `deepseek-tiny-mla` (synthetic) | `qwen3`, `deepseek2` | `qwen2` |
| Kolibri-1, `tiny-kolibri1` | `kolibri1` | `kolibri1` |
| Qwen3.5-0.8B, Qwen3.8-27B, `tiny-qwen35moe` | `qwen35`, `qwen35moe` | `qwen35` |
| `tiny-qwen3moe` (synthetic) | `qwen3moe` | `gpt-2` |
| DeepSeek-V2-Lite, DeepSeek LLM 7B | `deepseek2`, `llama` | `deepseek-llm` |
| DeepSeek-V3-0324 | `deepseek2` | `deepseek-v3` |
| Llama-3.2-1B-Instruct | `llama` | `llama-bpe` |
| GigaChat3.1-10B-A1.8B | `deepseek2` | `gigachat` |
| GLM-4.7-Flash | `deepseek2` | `glm4` |
| Moonlight-16B-A3B | `deepseek2` | `kimi-k2` (not ported) |
| Kanana-2-30B-A3B | `deepseek2` | `kanana2` (not ported; llama.cpp's `gpt-4o` split) |
| TinyLlama, Mistral-7B | `llama` | none (`llama` SentencePiece) |
| Gemma 4 | `gemma4` | none (`gemma4` model) |
| Nemotron-3.5-Embed-1B (community conversions) | `mistral3` | `pixtral` |

Before this change, `qwen35`, `gpt-2` and `deepseek-llm` files all got the Qwen2 split:
- **`qwen35`** differs only on combining marks: its letter runs are `[\p{L}\p{M}]+`.
- **`gpt-2`** groups digit runs and splits newlines differently.
- **`deepseek-llm`** differs on most non-trivial text.

The earlier byte-exact checks on those models used prompts that never hit the
difference. Real DeepSeek-V2-Lite needs re-checking for that reason.

## How llama.cpp runs a regex list, and how the port mirrors it

`unicode_regex_split` applies each regex in turn, and each one runs over every piece the
previous one produced. So a piece boundary, once made, never goes away. There are two
ways a regex runs:

1. **A hand-written splitter** (`unicode_regex_split_custom_*`), used when the regex
   string is one llama.cpp recognizes (`qwen2`, `qwen35`, `llama3`, `gpt2`). It walks the
   piece one codepoint at a time. A codepoint no alternative matches becomes a
   one-codepoint piece. Reflex ports these line for line as step functions
   (`llama3_family_step` covers llama3, qwen2 and qwen35, plus `gpt2_step`) driven by
   `split_custom`.
2. **`std::regex`**, for every other regex (both DeepSeek types). `regex_iterator` finds
   leftmost matches. Text between matches becomes its own piece. The piece's end acts as
   end of input for `$` and `(?!\S)`. Reflex emulates the iterator in `split_stl` and
   ports each regex as a matcher that returns the ECMAScript match at one position:
   alternatives tried in order, greedy with backtracking.
   - A regex containing `\p{..}` runs on "collapsed" text: ASCII stays as is, and every
     other codepoint becomes a byte standing for its category (non-ASCII whitespace
     becomes `\x0B`).
   - A regex without `\p{..}` runs through `std::wregex` on the codepoints, with every
     non-ASCII whitespace codepoint replaced by `\x0B`. For example U+3000 sits inside
     DeepSeek's `　-。` range but only ever matches `\s`.

   Either way, `\s` means llama.cpp's whitespace set and `\p{X}` means its category
   table. The matchers test exactly that.

**Unicode tables.** llama.cpp classifies codepoints with its own tables
(`src/unicode-data.cpp`), not the platform's. Rust's std has no `\p{P}`/`\p{S}`/`\p{M}`,
and its `is_alphabetic`/`is_whitespace` are different sets. `scripts/gen_unicode_tables.py`
transcribes `unicode_ranges_flags` (2,273 category ranges) and `unicode_set_whitespace`
into `src/tokenizer_unicode.rs`. That's about 18 KB of data and no new dependency.
Rerun the script against the new commit when the pinned llama.cpp moves.

**`ignore_merges`.** llama.cpp sets this for `LLAMA3` (and `CHATGLM4`, which `glm4` maps to): a pre-token that is already one
vocab entry is emitted as that entry without running BPE. Reflex does the same
(`PreType::ignore_merges`). None of the other supported types set it.

**BOS.** For `glm4` llama.cpp nulls `special_bos_id`, so a GLM prompt never starts with
BOS even though the file declares one and has no `add_bos_token` key
(`PreType::value_disables_bos`). The 2026-10-13 MLA-widening check (design/mla-widening.md)
compared full prompt ids, BOS included, against `llama-tokenize` on 7 prompts (digits,
punctuation, CJK, Cyrillic, code, mixed scripts) for the DeepSeek, GigaChat and GLM
vocabs: 21/21 identical.

**Not ported:** `clean_spaces`, which only affects llama.cpp's detokenizer (it's on for
`llama-bpe`). Reflex's decode returns the exact bytes.

## Gotchas found while porting

- **Unicode normalization ate two codepoints of an upstream regex.** DeepSeek-LLM's
  Latin class contains U+212A KELVIN SIGN (`K-ℭ`) and U+1F7D (`Ὗ-ώ`), among others. A
  tool in the copy path NFC-normalized them to ASCII `K` and U+03CE. `K-ℭ` then
  swallowed ASCII `[`, `{`, `]` and `}`, and `Ὗ-ώ` became an inverted range. The only
  symptom was one wrong merge on `" {[<>]}"`. The six codepoints NFC would change are
  now `\u{..}` escapes in `DEEPSEEK_LLM_LATIN`, and
  `test_deepseek_char_classes_are_well_formed` guards them. Copy class strings from
  upstream with a script that writes bytes, never through an editor.
- **Multi-file GGUF heads.** The DeepSeek-V3 source is shard 1 of 5. Its `split.*` keys
  make `llama-tokenize` look for the other shards. `scripts/gguf_tokenizer_only.py` drops
  them.

- **Tekken: no llama.cpp version tokenizes like HF.** `79e2e74` maps `pixtral` to the
  LLAMA3 split, which merges digit runs and doesn't split letter runs at case changes.
  Later llama.cpp has a `TEKKEN` split, but it runs on its collapsed text, where every
  non-ASCII letter counts as both upper and lower case, so `ÉCOLE` or `ΣΊΣΥΦΟΣ` split
  differently. The goldens for this type therefore come from HF `tokenizers` 0.23.2 on
  the model's own `tokenizer.json`, and `tekken_match` implements the regex with Rust's
  Unicode case properties (exact for `Lu`/`Ll`/`Lo`/`M`; titlecase `Lt` letters and
  `Other_Lowercase` modifier letters may differ from HF).
- **Nemotron-3.5-Embed's bracket markers.** Its chat template puts U+200C after every `[`
  and `<` of user text, so user text can't form a special token, and its HF tokenizer
  has a normalizer that turns `[`+U+200C and `<`+U+200C back into the plain bracket
  after special tokens are matched. `Tokenizer::strip_zwnj_markers` does the same, for
  `pre = nemotron-3.5-embed` and for a Tekken vocab in an embedding model
  (`<arch>.pooling_type` set), since the published GGUFs predate the dedicated pre type.
  A Ministral chat model's Tekken vocab keeps the characters, as its HF tokenizer does.

## Resolved: the original Qwen2 port

**Before (until 2026-10-13).** `qwen2_pretokenize_one` predated the tables and
classified with Rust's `is_alphabetic`/`is_numeric`/`is_whitespace`. On the golden set
it matched `llama-tokenize` on 12 of 13 strings. The miss was Arabic with harakat
(U+064E etc.): those are Mn with Other_Alphabetic, so Rust counted them as letters, and
llama.cpp (and HF `tokenizers`) don't. It was left unchanged in the first round, so
that reading pre types didn't change any `qwen2` output.

**After.** `qwen2` now runs llama.cpp's `unicode_regex_split_custom_qwen2` through
`split_custom`, like the other custom splitters. At `79e2e74` that function is
`unicode_regex_split_custom_llama3` with `\p{N}` (one digit) instead of `\p{N}{1,3}`
(checked by diffing the two functions), so it is `llama3_family_step` with `digits = 1,
marks = false`; no new step function. The old port, its `match_*` helpers and
`is_l`/`is_n`/`is_other` are gone. Its unit tests now run through `pretokenize`, with
the same expectations, plus one for harakat.

Results, master vs this change, each against llama.cpp `79e2e74`'s `llama-tokenize`
(and, for Kolibri-1, HF `tokenizers` 0.23.2 and `llama-tokenize` built with the
Kolibri-1 patch at `836d571`, since stock llama.cpp has no `kolibri1` architecture):

| Vocab | Strings | Before | After |
|---|---|---|---|
| Qwen3-0.6B (`qwen2`) | 13 golden + 2 new (Arabic harakat, Devanagari) | 13/15: misses golden #12 and the new Arabic string | 15/15 |
| Kolibri-1 (`kolibri1`) | same 15 | 15/15 | 15/15, ids unchanged |
| Qwen2.5-0.5B-Instruct (`qwen2`) | chat, math, FIM, repo FIM, tool call, near misses | | 6/6 |

On the Kolibri-1 vocab the split now differs from before on these strings, but the
merges don't cross the moved boundaries, so the ids are the same. The two new strings
were added to its HF golden test anyway (both engines agree on them). The Qwen2 golden
test no longer skips anything. The `reflex check` regression prompts are ASCII, where
Rust's classes and the tables agree, so every fixture tokenizes them identically before
and after (host diff, all 24 rows). The GPU check confirms it: on an RTX 3090, master vs this
change built on the same pod, all 30 existing `reflex check` rows (every fixture plus
Qwen3-0.6B, three prompts each) are identical, token ids and logit checksums included.

## Verification

Golden ids come from llama.cpp `79e2e74`'s `llama-tokenize`, CPU-only, built in a Docker
container (`ubuntu:24.04`, `cmake -DGGML_CUDA=OFF`). It runs over 13 tricky strings:
- digits
- contractions in both cases
- CJK, kana and Hangul
- Cyrillic, Greek, Arabic and Thai
- `\r\n`, blank lines, tabs and trailing whitespace
- emoji with ZWJ and variation selectors
- punctuation runs
- leading spaces
- decomposed accents
- code
- NBSP, ideographic and em spaces
- fullwidth forms
- Devanagari and Arabic combining marks

Each string is a separate `--stdin --ids --no-bos` run. Results on that set:
- Every supported pre type matches 13/13. `qwen2` was 12/13 until its 2026-10-13 switch to
  the table-based split (see above).
- Before the fixes above, `deepseek-llm` was 12/13 (the normalization bug).
- `qwen2` and Kolibri-1 outputs were unchanged by the first round.

Tests: `test_*_encode_matches_llama_tokenize_if_fixture_present` in `src/tokenizer.rs`,
skip-if-absent, with golden ids inline. The fixtures are tokenizer-only GGUFs: the real
file's metadata section, with an empty tensor table, from
`scripts/gguf_tokenizer_only.py` (see [DEVELOPMENT.md](../DEVELOPMENT.md)'s test-fixture
section). Sources and byte ranges are in each test's doc comment.

On every model fixture in `test-data/`, the three `reflex check` regression prompts
tokenize identically before and after this change, except one: `tiny-qwen3moe` on the
1,352-byte prompt, which now follows the `gpt-2` split. Its prompt ids equal
`llama-tokenize`'s; before, they didn't. The GPU check confirms it. On an RTX 3090,
before and after built on the same machine, 29 of 30 `reflex check` rows (every fixture
plus Qwen3-0.6B, three prompts each) are identical, token ids and logit checksums
included. The 30th is that `tiny-qwen3moe` row.

## Adding a pre type

1. Find the value's `LLAMA_VOCAB_PRE_TYPE_*` in `src/llama-vocab.cpp` at the pinned
   commit, and its regex list. If it maps to a type already ported, add the value to
   `PreType::ACCEPTED` and skip to step 4.
2. Check how `unicode_regex_split` runs each regex: through a `unicode_regex_split_custom_*`
   function (port that function as a `split_custom` step) or through `std::regex` (write
   a `split_stl` matcher, using the collapsed-text or wregex rules above). Use
   `CptFlags`, never Rust's character predicates. Check whether the type sets
   `ignore_merges`.
3. Add the variant and its `pretokenize` arm. Cite the llama.cpp type and the regex
   strings in a comment, the way the existing arms do. Copy any non-ASCII class string
   with a script, and escape what NFC would change.
4. Make a fixture: fetch the first ~40 MB of a real GGUF using that pre type with an
   HTTP range request, then run `scripts/gguf_tokenizer_only.py`.
5. Get golden ids for the 13-string set (`PRE_TYPE_GOLDEN_STRINGS`) from `llama-tokenize`
   at the pinned commit, and add a `test_<type>_encode_matches_llama_tokenize_if_fixture_present`
   test.
