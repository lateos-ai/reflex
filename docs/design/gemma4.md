# Design: Gemma 4 support (`gemma4`)

**Status (2026-10-11): Phases 1-5 done; the 12B and the 31B are verified.**
On an RTX A40, Gemma 4 12B-it `Q8_0` generates the same greedy tokens as llama.cpp running
with f32 activations, on 4/4 prompts of 20 tokens, including one past the 1,024-token
sliding window. `system1` scores match llama.cpp's within 0.06. The first token comes
1.45x sooner than with `llama-simple`. As a classifier, Gemma 4 suits warm mode
(~74 ms per decision), not one process per decision (~2.1 s). See
[Phase 4 results](#phase-4-results-2026-10-09). On an A100 80 GB, the 31B-it `Q8_0`
(f16 weights) matches llama.cpp running fully in f32 on 5/5 prompts, up to 1,262
tokens, and `system1` within 0.03. See [31B results](#31b-results-2026-10-11).

Goal: run Google's Gemma 4 text models on Reflex. They are released under Apache 2.0.
The target is `gemma-4-31B-it`, and development uses `gemma-4-12B-it`, which has the
same text architecture at a smaller size. Correctness means byte-exact greedy agreement
with llama.cpp (`reflex check`), as for every other architecture. `system1`
classification, with multi-token labels, is part of the scope.

## Scope

In: text-only generation and `system1` for the dense `gemma4` models (12B and 31B),
loaded from GGUF with `--weights f16|f32`.

Out, each rejected at load with a clear error:

- Vision and audio inputs (`mmproj-*.gguf`). The text model loads without them.
- MTP and DFlash draft models (`mtp-*.gguf`, `dflash-*.gguf`).
- E2B/E4B. They add per-layer token embeddings
  (`embedding_length_per_layer_input > 0`) and KV cache sharing across layers
  (`attention.shared_kv_layers > 0`). These change the forward pass and teach nothing
  about the 31B.
- 26B-A4B (MoE, `expert_count > 0`). It has its own router convention.
- `--export-kv`/`--import-kv` and `--lora` on this path, at first.

## Reference implementations

- llama.cpp `79e2e74` (master, 2026-10-09): `src/models/gemma4.cpp` (graph and
  tensors), `src/llama-vocab.cpp` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`),
  `llama_model_rope_type` in `src/llama-model.cpp`.
- GGUF headers of `ggml-org/gemma-4-12B-it-GGUF` (revision `e3e6817`) and
  `ggml-org/gemma-4-31B-it-GGUF`, `*-Q8_0.gguf`. These were read from the first 40 MB of
  each file with an HTTP range request, without downloading the rest.
- Hugging Face `config.json` of `google/gemma-4-12B-it` and `google/gemma-4-31B-it`, and
  the 12B's `tokenizer.json`.

## Confirmed conventions

Both GGUFs report `general.architecture = gemma4` and use the same metadata keys and
tensor names. Only the sizes differ, so code verified on the 12B needs no change for
the 31B.

| | 12B | 31B |
|---|---|---|
| Layers × hidden | 48 × 3840 | 60 × 5376 |
| FFN width | 15360 | 21504 |
| Q heads | 16 | 32 |
| KV heads, sliding / global | 8 / 1 | 16 / 4 |
| Vocab (tied LM head) | 262,144 | 262,144 |
| Q8_0 file | 12.7 GB | 32.6 GB |

**Layer pattern.** `attention.sliding_window_pattern` is a per-layer bool array: five
sliding-window layers, then one global layer, repeated (`true` = sliding). The window is
1024 (`attention.sliding_window`), with llama.cpp's `LLAMA_SWA_TYPE_STANDARD` mask, the
same one Kolibri-1 uses. `attention.head_count_kv` is a per-layer array.

**Sliding vs global layers.** The two layer types differ in every attention dimension:

| | Sliding | Global |
|---|---|---|
| Head dim (`key_length[_swa]`) | 256 | 512 |
| RoPE base (`rope.freq_base[_swa]`) | 10,000 | 1,000,000 |
| RoPE dims (`rope.dimension_count[_swa]`) | 256 | 512 |
| RoPE frequency factors | none | `rope_freqs.weight` |
| `attn_v` tensor | present | **absent**: V = K's projection |

**Global-layer RoPE ("proportional").** RoPE is NEOX style (`llama_model_rope_type`
returns `LLAMA_ROPE_TYPE_NEOX` for `gemma4`) over all 512 dims, with `rope_freqs.weight`
(256 floats, one per pair) dividing each pair's frequency. In the real file the first 64
factors are 1.0 and the other 192 are 1e30. So pairs `(i, i+256)` for `i < 64` rotate by
`pos · 1e6^(−2i/512)`, and the rest effectively don't rotate. This is not the
`rotary_dim` partial RoPE used for Qwen3.5, which pairs `i` with `i + rotary_dim/2` and
spreads the exponent over `rotary_dim`. It needs a RoPE kernel that takes per-pair
frequency factors, like `ggml_rope_ext`'s `freq_factors`.

**Per-layer forward pass** (`llama_model_gemma4::graph`), with `h` the residual stream:

1. `x = rmsnorm(h, attn_norm)`
2. `q = rope(rmsnorm_per_head(Wq·x, attn_q_norm))`
3. `k = rope(rmsnorm_per_head(Wk·x, attn_k_norm))`
4. `v = rmsnorm_per_head(Wv·x)` with **no weight** and no RoPE. On global layers
   `Wv·x` is replaced by `Wk·x`, taken before `k`'s norm and RoPE.
5. Causal attention over `q, k, v`, with a sliding window on sliding layers and
   **scale 1.0** (`f_attention_scale = 1.0`, not `1/√head_dim`).
6. `a = rmsnorm(Wo·attn, post_attention_norm) + h`
7. `f = W_down · (gelu_tanh(W_gate·y) ⊙ (W_up·y))` with `y = rmsnorm(a, ffn_norm)`
   (GeGLU; ggml's `LLM_FFN_GELU`, tanh approximation).
8. `h' = (rmsnorm(f, post_ffw_norm) + a) · layer_output_scale`, a per-layer scalar
   tensor.

**Model level.** Token embeddings are multiplied by `√hidden_size` after the lookup.
After the last layer comes `rmsnorm(output_norm)`, then the tied LM head (`token_embd`,
since there is no `output.weight`), then a **final logit soft-cap**:
`30 · tanh(logits / 30)` (`final_logit_softcapping = 30.0`). RMSNorm epsilon is 1e-6.

**Text-only attention is causal.** `use_bidirectional_attention = "vision"` (llama.cpp's
`LLAMA_NON_CAUSAL_TYPE_SWA_ONLY`) only applies to image tokens.

**Tokenizer.** `tokenizer.ggml.model = "gemma4"`: BPE over raw UTF-8 bytes (no GPT-2
byte remap), with spaces replaced by `▁`, no dummy prefix, a split into newline runs and
non-newline runs, merges applied by rank, and `<0xXX>` byte fallback. There are 262,144
tokens and 514,906 merges. `add_bos_token = true` (`<bos>` = 2), and `<eos>` = 1. The
chat format uses `<|turn>`/`<turn|>`, not Gemma 3's `<start_of_turn>`. There are 24
CONTROL/USER_DEFINED tokens, which are matched literally before BPE. No merge or token
mixes a newline with another character, so splitting on newlines changes nothing
compared with HF's whole-text BPE.

## Approach

A separate model path, `src/model/gemma4.rs`, dispatched from `Model::load_with_options`
on `general.architecture == "gemma4"`, the same way `qwen35` goes to `load_hybrid` and
`deepseek2` to `load_mla`. The dense path shares one `LayerConfig` across all layers, but
Gemma 4 changes head dim, KV heads, RoPE and the V source from layer to layer. A separate
path keeps every existing model's code and kernels unchanged: they never run Gemma code
and never load its kernels.

It reuses the existing load pipeline (`WeightLoadPipeline`, the background
tokenizer/cuBLAS init, the lazily-resident tied LM head), `gemv`/cuBLAS GEMM, `rmsnorm`,
and the online attention kernel. That kernel already takes a sliding `window`, a `scale`
and head dims up to 1024. New kernels:

- NEOX RoPE with per-pair frequency factors (single-token and batched).
- `gelu_tanh_and_mul` (GeGLU), next to `silu_and_mul`.
- Logit soft-cap and a scalar multiply, in `elementwise.cu`.

## Phases

1. **Tokenizer** (CPU only). Done; see below.
2. **Kernels**: the three above, each tested against a host reference
   (`src/model/gemma4_kernel_tests.rs`); both pass on the A40.
3. **Model path** (written, see [Phase 3 notes](#phase-3-notes-2026-10-09)):
   `Gemma4Config` and its parser (per-layer arrays, rejects the
   out-of-scope variants), `load_gemma4`, single-token forward and batched prefill, and
   `generate`/`system1`/`check`. `system1` scores multi-token labels the way the dense
   path does, on soft-capped logits. Synthetic fixture: a tiny
   `Gemma4ForConditionalGeneration` checkpoint with random weights, run through
   llama.cpp's unmodified `convert_hf_to_gguf.py`, including global layers without
   `attn_v` and a real `rope_freqs`, and checked with `llama-simple` first.
4. **12B on a 48 GB GPU** (done, see [Phase 4 results](#phase-4-results-2026-10-09)): `reflex check` against llama.cpp greedy on several prompts,
   `system1` scores against llama.cpp's soft-capped logits, and a regression rerun of
   the existing models. Then cold-start and classifier benchmarks: cold `system1`, and
   warm `/v1/classify` p50/p95 for 1-token and multi-token labels.
5. **31B** (done, see [31B results](#31b-results-2026-10-11)): in f16 on an 80 GB GPU. To fit a 24–48 GB card, `REFLEX_QUANT_RESIDENT=1`
   would have to be extended to this path, with a Q4_K_M file. ggml-org ships only
   Q4_0/Q8_0, which quantized-resident mode does not keep resident.

## Phase 1 results (2026-10-09)

- `Tokenizer::encode_gemma4` (`src/tokenizer.rs`) implements the scheme above. Decoding
  reuses the SentencePiece path (`▁` to space, `<0xXX>` to a byte), which matches
  llama.cpp's BPE detokenize when `escape_whitespaces` is set.
- BPE over a whole line (there is no space pre-split) uses a heap of candidate pairs,
  like llama.cpp's `llm_tokenizer_bpe_session`, not the rescan-per-merge loop the gpt2
  path uses. `test_bpe_merge_ranked_matches_rescanning_merge` checks that the two agree.
  Encoding a 9,000-character single line takes 3.6 ms (release build, Windows dev box).
- Golden test `test_gemma4_encode_matches_reference_if_fixture_present`: 25 strings
  covering spaces (leading, repeated, trailing), newline runs (including one longer than
  any newline token), tabs, digits, contractions, accented Latin, CJK, emoji, control
  characters, code, a literal `▁`, and the chat, thinking and tool-call markers. Ids are
  identical to HF `tokenizers` 0.23.2 on `tokenizer.json` and to a Python port of
  llama.cpp's gemma4 tokenize. Checking against llama.cpp's own `llama-tokenize` is still
  to do (Phase 4, on the GPU host).
- Building the tokenizer takes 236 ms vs. 78 ms for Qwen3-0.6B (release build, Windows
  dev box), because of 515k merges and 262k tokens. `load_*` already builds the
  tokenizer on its background init thread, overlapping the weight load, so this stays
  off the serial cold-start path as long as `load_gemma4` keeps that pattern.
- Fixture: `test-data/gemma4-tokenizer.gguf` is the 12B Q8_0 file's metadata section,
  rewritten with an empty tensor table (15.8 MB, gitignored). sha256
  `2a18b8fb866c81222a327686b473df5a3906f271c045982e6f4ad1c09be88e14`.

## Phase 3 notes (2026-10-09)

- `Model::load_with_options` sends `gemma4` to `Model::load_gemma4`. `Model.gemma4:
  Option<Gemma4Model>` holds the per-layer weights and kernels, and
  `ArchitectureKind::Gemma4` lets callers tell the path apart. `--export-kv`,
  `--import-kv` and `--lora` return clear errors. `REFLEX_QUANT_RESIDENT=1` printed a
  notice and loaded normal weights; it now keeps Q8_0 weights quantized (see
  [quantized-resident weights](quantized-resident-weights.md#q8_0-on-the-gemma-4-path-design-2026-10-12)). `REFLEX_ATTN_KERNEL=legacy` is rejected, because the
  legacy kernels hardcode a `1/sqrt(head_dim)` scale.
- One layer function, `Model::forward_gemma4_layer`, serves both prefill (`rows` = prompt
  length, cuBLAS GEMMs, sliding window masked per row in the online attention kernel)
  and decode (`rows == 1`, GEMVs, window applied by narrowing the K/V views, as the
  dense path does). Each layer's K/V cache is sized from that layer's own shape.
- The tied LM head stays lazy (`LmHead::TiedLazy`). `system1` gathers only the candidate
  rows and applies the soft-cap to them on the host. Generation forces the full head
  resident on its first token and applies the soft-cap on the GPU. Multi-token labels
  are teacher-forced through the shared post-prompt KV headroom, the same way as on the
  dense path.
- `layer_output_scale` is read to the host at load and applied with `scale_kernel`.
  When it's absent the default is 1.0, as in llama.cpp.
- Tests:
  - `gemma4_config_tests`: host-only, real 12B and 31B header values, and each rejected
    variant.
  - `gemma4_model_tests`: `#[ignore]`d GPU tests, run against `REFLEX_GEMMA4_GGUF` or
    `test-data/tiny-gemma4.gguf`. They check that batched prefill past the sliding window
    matches sequential decode, that `system1` scores match the full-vocab soft-capped
    logits for 1-token and multi-token labels, and that generation runs end to end.
    `scripts/gpu_nightly_tests.sh` reports them as SKIP when the fixture is absent.

## Phase 4 results (2026-10-09)

Runpod secure-cloud RTX A40 (48 GB, sm_86, EU-SE-1, 9 vCPUs, CUDA 12.8 host). Reflex
was built with `REFLEX_CUDA_ARCH=sm_86`. llama.cpp `79e2e74` was built with CUDA, and
the reference program `ref_g4.cpp` prints greedy ids, first-step logits and
`system1`-style candidate scores through `llama.h`.

**Kernels.** Both `gemma4_kernel_tests` pass. The RoPE test's tolerance scales with
position: theta is an f32 the size of the position, so at position 60,000 one ulp of it
is ~0.007 rad, and device and host `powf` may differ by an ulp. llama.cpp computes theta
the same way.

**Synthetic fixture.** `test-data/tiny-gemma4.gguf` is written directly with llama.cpp's
`gguf-py`, not converted from a fake HF checkpoint. It copies every metadata key and the
real 262,144-token tokenizer from the 12B header and uses the real tensor names. Both
engines read the same GGUF, so the comparison doesn't depend on the converter. llama.cpp
loads it and generates. With f32 weights, Reflex matches llama.cpp **20/20 on all 4
prompts**: two plain, one chat-format, and one of 982 tokens against an 8-token window.
This is now the GPU test `gemma4_model_fixture_matches_llamacpp`. Batched vs sequential
prefill on the fixture, f32: rel L2 1.9e-6.

**Real 12B (`gemma-4-12B-it-Q8_0.gguf`, f16 weights).**

| Prompt | Stock llama.cpp | llama.cpp, f32 activations |
|---|---|---|
| "The capital of France is" | 20/20 | 20/20 |
| "Once upon a time" | differs at token 1 (llama.cpp's own top-2 gap 0.12) | 20/20 |
| Chat-format question | first 15 agree | 20/20 |
| 1,122 tokens (past the 1,024 window) | differs at token 1 | 20/20 |

Stock llama.cpp quantizes activations to 8 bits (Q8_1) for its `Q8_0` matmuls, and
Reflex doesn't. That moved the long prompt's logits:

| ~1,566-token natural prompt, first-token logits | 1174 | 109615 | 669 | 1030 | 799 |
|---|---|---|---|---|---|
| Reflex, f16 weights | 25.658 | 22.935 | 22.602 | 22.029 | 21.615 |
| llama.cpp, f32 activations (`LLAMA_NO_Q8=1`) | 25.644 | 22.904 | 22.579 | 22.023 | 21.617 |
| stock llama.cpp, flash attention | 25.762 | 22.142 | 22.610 | 22.046 | 21.578 |
| stock llama.cpp, no flash attention | 25.600 | 22.128 | 22.423 | 21.870 | 21.548 |

How the cause was found, ruling out alternatives one at a time:
- **Not the sliding window.** Moving Reflex's window by +-1 changed token 109615 by only
  0.05-0.3, and removing the window flipped the top-1.
- **Not Reflex's f16 activation cast.** Prefilling one token at a time with f32
  activations gave the same 22.93.
- **llama.cpp disagrees with itself.** Turning flash attention on or off moves its
  logits by up to 1.6 on a degenerate repeated-sentence prompt.
- **f32 activations close the gap.** llama.cpp with `llama-no-q8.patch` (from the
  Kolibri work; applies to `79e2e74` unchanged) agrees with Reflex within 0.03 on every
  token. So, as for Kolibri-1, **the llama.cpp reference for quantized Gemma 4 files is
  the f32-activation build**. The patch is archived in `test-data/tiny-gemma4-src.tar.gz`.

Other checks on the 12B:
- **`system1`:** label token ids identical to llama.cpp's. Scores within 0.003
  (1-token) and 0.06 (multi-token) of the f32-activation build, e.g. " the city of
  Paris" 69.012 vs 68.959. Reflex's own gather-plus-host-soft-cap path equals its
  full-vocab soft-capped logits to 1e-6.
- **Tokenizer:** all 25 golden strings identical to llama.cpp's own `llama-tokenize`.
- **f16 activation range:** largest prefill activation 2,063, 32x below f16's limit, none
  clamped. The "f16 activation range" risk below didn't materialize.
- **Batched vs sequential prefill, f16, 1,038 tokens:** rel L2 7.5e-3, same greedy
  token. The test's f16 bound for Gemma 4 is 2e-2; the shared 5e-3 bound was tuned on
  smaller Qwen/MLA models.

**No regressions.** `reflex check` on Qwen3-0.6B, TinyLlama-1.1B, Qwen3.5-0.8B and the
MLA fixture, f32 and f16, 2 prompts each: token ids and logit checksums identical between
`master` and this branch (16/16). `scripts/gpu_nightly_tests.sh` plus the Kolibri tests:
every GPU test with its prerequisites passes (32), 7 skip for absent files.

**Benchmarks** (median of n=5; wall time measured from outside the process, launch to
exit; first token for "The capital of France is"):

| | Reflex | llama.cpp `llama-simple` |
|---|---|---|
| First token, model file in page cache | **2.41 s** (engine 2.01 s, load 1.41 s) | 3.49 s |
| First token, after evicting the file's pages | **2.47 s** | 5.27 s |

The container can't drop the host page cache, and evicting the file's pages with
`posix_fadvise` only reaches this container's view. Reads after eviction ran at 1.7 vs
2.1 GB/s cached, so the second row is not a true cold-disk number.

Classifier, prompt "Review: The battery died after two days. Sentiment:":

| | 1-token labels (2) | 2-token labels (3) |
|---|---|---|
| Cold: one `reflex system1` process per decision | 2.13 s wall (1.72 s engine) | 2.38 s |
| **Warm: `reflex stdio`, per request** (30 requests, p50 / p95) | **73.6 / 74.2 ms** | 233.6 / 233.8 ms |
| Warm, Qwen3-0.6B on the same A40 | 6.5 / 6.6 ms | 20.7 / 21.5 ms |

A Gemma 4 12B classifier belongs in warm mode: about 74 ms per decision once loaded.
Multi-token labels cost one extra decode step per label beyond the first token. Model
quality on a bare prompt is its own question: the instruction-tuned 12B scores
" positive" (16.77) above " negative" (12.43) for that review, and llama.cpp gives the
same scores.

**Not done:** a true cold-disk measurement. (The 31B was run later; see
[31B results](#31b-results-2026-10-11).)

## 31B results (2026-10-11)

ThunderCompute A100-SXM4-80GB (sm_80). Reflex was built from `46fcb29` with
`REFLEX_CUDA_ARCH=sm_80`, and llama.cpp at `79e2e74`. Model: `gemma-4-31B-it-Q8_0.gguf`
with `--weights f16` (about 61 GB on the GPU). The harness, including the llama.cpp
reference program and the `LLAMA_NO_Q8` switch, is archived as
`test-data/large-model-verify-src.tar.gz`. No code changed for the 31B.

**The reference has to be fully f32.** llama.cpp with f32 activations alone (the 12B's
reference) left a 0.54 gap on the top logit of the 982-token repeated-sentence prompt
(Reflex 17.174, llama.cpp 17.718), though the greedy token was the same. The cause is on
llama.cpp's side:

| 982-token prompt, first-token logits | 236770 | 236832 |
|---|---|---|
| Reflex, f16 weights | **17.174** (top 1) | |
| llama.cpp, f32 activations, flash attention (f16) | 17.718 | 17.589 |
| llama.cpp, f32 activations, no flash attention, f16 KV | 17.322 | **17.391** (top 1) |
| llama.cpp, f32 activations, no flash attention, f32 KV | **17.162** | 16.865 |

With flash attention on, llama.cpp ignores an f32 KV cache type, so the KV setting
changes nothing until flash attention is off. Reflex keeps its KV cache in f32. Reflex's
own f16 activation cast is not the cause: batched prefill (f16 activations into cuBLAS)
and sequential prefill (f32 activations) agree on a 1,038-token prompt, top logit 16.344
vs 16.333, rel L2 3.3e-3. So the reference for long Gemma 4 prompts is f32 activations
**plus** an f32 KV cache with flash attention off.

**Greedy tokens against that reference** (20 tokens each; both sides stop only at
`<eos>`, see below):

| Prompt | Fully-f32 llama.cpp | Stock llama.cpp |
|---|---|---|
| "The capital of France is" | 20/20 | 20/20 |
| "Once upon a time" | 20/20 | 20/20 |
| Chat-format question | 17/17 (through `<eos>`) | 17/17 |
| 982 tokens | 20/20 | differs at token 1 |
| 1,262 tokens (past the 1,024 window) | 20/20 | 20/20 |
| 2,641-token natural prompt (the start of llama.cpp's README) | 10/10, top logit 20.054 vs 20.020 | differs at token 1 (a 0.05 near-tie) |

On the repeated-sentence prompts the remaining top-logit gaps are up to 0.34 (1,262
tokens); on the natural prompt it is 0.03. These degenerate prompts are unusually
sensitive: llama.cpp moves by 0.5 between its own attention paths on them.

**Stop tokens.** `reflex check`/`generate` stop only at `eos_token_id` (`<eos>` = 1).
llama.cpp stops at any end-of-generation token, including `<turn|>` (106). Compare with
EOS-only stopping on llama.cpp's side, or the chat prompt looks like a 6-token answer
there and a 17-token one in Reflex.

**`system1`**, against llama.cpp's logit sums on the same tokens (all candidate token ids
identical):

| Prompt | Candidate | Reflex | Fully-f32 llama.cpp |
|---|---|---|---|
| "The capital of France is" | " Paris" (1 token) | 27.254 | 27.257 |
| | " London" (1) | 7.566 | 7.568 |
| | " the city of Paris" (4) | 101.715 | 101.718 |
| | " Rome, Italy" (3) | 43.314 | 43.343 |
| "Review: The battery died after two days. Sentiment:" | " positive" (1) | 4.284 | 4.281 |
| | " negative" (1) | 18.771 | 18.760 |
| | " very positive" (2) | 25.288 | 25.277 |
| | " mixed feelings" (2) | 16.615 | 16.618 |

**GPU tests** with `REFLEX_GEMMA4_GGUF` pointing at the 31B:
`gemma4_model_batched_prefill_matches_sequential`, `gemma4_model_system1_scores_match_full_logits`
and `gemma4_model_generates_without_error` pass. Run them with `--test-threads=1`: in
parallel, two 61 GB loads exhaust the 80 GB card and one test fails.

**Timings** (same host, file in page cache, median of 3, wall time launch to exit):
Reflex 15.0 s to the first token vs `llama-simple` 19.0 s. ThunderCompute attaches the
GPU over the network, and loading 61 GB dominates both, so only the ratio means anything;
these numbers don't compare with the A40's.

## Risks

- **f16 activation range.** Resolved on the 12B: the largest prefill activation was
  2,063, and none were clamped (Phase 4). On the 31B, batched prefill (f16 activations)
  agrees with sequential prefill (f32) to rel L2 3.3e-3, and tokens match the fully-f32
  llama.cpp, so the cast does no visible harm there either. The fallback is still
  `--weights f32`, but the 12B in f32 (about 48 GB) doesn't fit a 48 GB card.
- **VRAM.** 12B in f16 is about 24 GB, so it needs a 48 GB card. 31B in f16 is about
  61 GB, so it needs 80 GB. On top of that comes the KV cache: f32, sized to
  prompt + max tokens, at about 1.6 MB per token across the 31B's sliding layers.
- **Cold start.** Measured in Phase 4: 2.4 s to first token with the file cached
  (load 1.4 s), against llama.cpp's 3.5 s. As expected, Gemma 4 is a warm-mode
  (`reflex stdio`/sidecar) classifier at ~74 ms per decision, not a per-decision cold
  process.
