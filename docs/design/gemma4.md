# Design: Gemma 4 support (`gemma4`)

**Status (2026-10-09): Phase 1 (tokenizer) done; Phase 2 (kernels) written, not yet
run on a GPU.** Reflex encodes text exactly like Gemma 4's own tokenizer on 25 test
strings (see [Phase 1 results](#phase-1-results-2026-10-09)). The new kernels
(`kernels_cuda/gemma4.cu`) have host references checked on the CPU, and `#[ignore]`d GPU
tests (`cargo test --release -- --ignored gemma4_`) that still need a GPU host. The model
itself still fails at load with `UnsupportedArchitecture`.

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
2. **Kernels**: the three above, each tested against a host reference. Written
   (`src/model/gemma4.rs`, `src/model/gemma4_kernel_tests.rs`); GPU run pending.
3. **Model path**: `Gemma4Config` and its parser (per-layer arrays, rejects the
   out-of-scope variants), `load_gemma4`, single-token forward and batched prefill, and
   `generate`/`system1`/`check`. `system1` scores multi-token labels the way the dense
   path does, on soft-capped logits. Synthetic fixture: a tiny
   `Gemma4ForConditionalGeneration` checkpoint with random weights, run through
   llama.cpp's unmodified `convert_hf_to_gguf.py`, including global layers without
   `attn_v` and a real `rope_freqs`, and checked with `llama-simple` first.
4. **12B on a 48 GB GPU**: `reflex check` against llama.cpp greedy on several prompts,
   `system1` scores against llama.cpp's soft-capped logits, and a regression rerun of
   the existing models. Then cold-start and classifier benchmarks: cold `system1`, and
   warm `/v1/classify` p50/p95 for 1-token and multi-token labels.
5. **31B**: in f16 on an 80 GB GPU. To fit a 24–48 GB card, `REFLEX_QUANT_RESIDENT=1`
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

## Risks

- **f16 activation range.** Prefill casts activations to f16 for GEMMs, which saturates
  at ±65504, and Gemma models are known for large activations. `f16_activation_stats`
  will show it. The fallback is `--weights f32`: 12B in f32 is about 48 GB, so it needs
  an 80 GB card.
- **VRAM.** 12B in f16 is about 24 GB, so it needs a 48 GB card. 31B in f16 is about
  61 GB, so it needs 80 GB. On top of that comes the KV cache: f32, sized to
  prompt + max tokens, at about 1.6 MB per token across the 31B's sliding layers.
- **Cold start.** The weight load dominates: about 24 GB to read and upload for the 12B.
  Expect seconds, not the ~0.5 s of a 0.6B model. Gemma 4 is likely a warm-mode
  (`reflex stdio`/sidecar) classifier rather than a per-decision cold process. Phase 4
  measures this.
