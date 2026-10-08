# Design: Kolibri-1 MoE support

**Status (2026-10-08): Phases 1-3 and Phase 4 steps 1-2 done.** The real Kolibri-1 Q4_K_M
runs with `REFLEX_QUANT_RESIDENT=1` (45.4 GiB peak VRAM, so it fits a 48 GB card) and
matches llama.cpp token for token on 5 German/English prompts up to 627 tokens, once
llama.cpp computes with f32 activations like Reflex (see
[Phase 3 step 2 results](#phase-3-step-2-results-2026-10-07)). After the parallel load
pipeline and prefetch readers, a cold first token on an RTX A6000 takes 8.5 s against
llama.cpp's 17.2 s on the same host, and 3.8-4.0 s vs 7.6 s with the file in the page
cache ([readers and System1](#prefetch-readers-and-system1-on-an-rtx-a6000-2026-10-07);
an earlier, slower-disk host measured 19.7 vs 41.3 s cold). A warm System1
classification takes ~0.2 s. With opt-in lazy expert upload (`REFLEX_LAZY_EXPERTS=1`,
Phase 4 step 2) a short prompt's cold first token drops to 2.3 s, 7.2x faster than
llama.cpp on the same host, and a cold System1 decision to ~2.9 s
([lazy upload results](#lazy-expert-upload-on-an-rtx-a6000-2026-10-08)). Next: overlap
the expert uploads with compute so long prompts gain too.
Every convention
below was read from Aleph Alpha's own checkpoint and inference code and from a real
GGUF header, and cross-checked between two independent implementations. See
[Confirmed conventions](#confirmed-conventions).

Goal: run Aleph Alpha's Kolibri-1 (released 2026-10-03, Apache 2.0) on Reflex,
measured on cold start (process launch to first token). It is a 78B-parameter MoE
(3.46B active per token) with 384 routed experts plus 1 shared expert and top-6
routing in every layer. Attention alternates four sliding-window layers with RoPE and
one full-attention layer without positional encoding (NoPE).

## Constraints that shape this plan

- **Weights don't fit unless they stay quantized.** Matrix weights are dequantized to
  `f16` at load by default, which is ~156 GB for this model. `REFLEX_QUANT_RESIDENT=1`
  keeps Q4_K/Q6_K weights quantized on the GPU, but only on the dense path (see
  [quantized-resident-weights.md](quantized-resident-weights.md)). The Q4_K_M GGUF is
  47.5 GB, so it fits on an 80 GB GPU only if the expert tensors stay quantized.
- **Single GPU only.** Every entry point uses `CudaDevice::new(0)`. Multi-GPU sharding
  is out of scope.
- **No FP8 path.** Aleph Alpha's own checkpoint is FP8 (128x128 block scales); we use
  community GGUFs converted from it instead.
- **Correctness means byte-exact greedy agreement with llama.cpp** (`reflex check`).
  Mainline llama.cpp does not support `kolibri1` yet (upstream issue
  [#29922](https://github.com/ggml-org/llama.cpp/issues/29922)), so the reference is
  llama.cpp at commit `836d571` with the community patch the published GGUFs were made
  with (see [Reference implementations](#reference-implementations)).
- **Cold start is the metric.** Sustained tokens/s is reported, not optimized (see the
  Non-goals in [DEVELOPMENT.md](../DEVELOPMENT.md)).

## Reference implementations

| Source | What it is | Role |
|---|---|---|
| [`aleph_alpha_inference/kolibri1.py`](https://github.com/Aleph-Alpha/aleph-alpha-inference) (commit `049a6a7`) | Aleph Alpha's official vLLM plugin | Ground truth for the math |
| [`kolibri1-llama.cpp.patch`](https://huggingface.co/Hob-forge/Kolibri-1-GGUF) on llama.cpp `836d571` | Community patch (converter + model + `SIGMOID_LOGIT_ADD` gating) | Byte-exact reference; produced the GGUFs we load |
| [CWBudde/llama.cpp](https://github.com/CWBudde/llama.cpp) (PRs #1, #2, #7, #9) | A second, independent community port | Third opinion when the first two disagree with Reflex |

The official plugin and the Hob-forge patch were read side by side and agree on every
convention listed below.

## Confirmed conventions

Sources: HF `config.json` and `tokenizer.json` of
[Aleph-Alpha/Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16), the
official plugin, and the header of `Kolibri-1-Q4_K_M.gguf` (read with an HTTP range
request).

**Shape.** 50 layers, `n_embd` 2560, 48 query heads, 4 KV heads, `head_dim` 128,
vocab 128,000, untied `output.weight`. Expert FFN width 512, shared-expert FFN width
512, RMSNorm eps 1e-6.

**GGUF metadata** (`general.architecture = "kolibri1"`):

| Key | Value |
|---|---|
| `kolibri1.expert_count` / `expert_used_count` | 384 / 6 |
| `kolibri1.expert_shared_count` | 1 |
| `kolibri1.expert_feed_forward_length` / `expert_shared_feed_forward_length` | 512 / 512 |
| `kolibri1.expert_gating_func` | 5 (`SIGMOID_LOGIT_ADD`, new in the patch) |
| `kolibri1.expert_weights_norm` | false |
| `kolibri1.attention.sliding_window` | 513 |
| `kolibri1.attention.sliding_window_pattern` | bool[50]: `true` = sliding layer, every 5th layer (index 4, 9, ..., 49) `false` |
| `kolibri1.rope.freq_base` | 10000, no scaling keys |
| `tokenizer.ggml.model` / `.pre` | `gpt2` / `kolibri1` |
| `tokenizer.ggml.add_bos_token` | false (no BOS token at all) |
| `tokenizer.ggml.eos_token_id` | 127906 (`<\|im_end\|>`); 127901 is padding and also a stop token in `generation_config.json` |

**Tokenizer.** Plain byte-level BPE (127,644 merges), not a new format: the
"German-tailored tokenizer" is a new vocabulary, not a new algorithm. Its split regex
is identical to Qwen2's (`\p{N}{1}` is `\p{N}`), and the patch maps `kolibri1` to
`LLAMA_VOCAB_PRE_TYPE_QWEN2`. Reflex's existing `gpt2` path should handle it as is.

**Attention.**
- Q/K per-head RMSNorm before RoPE (as in Qwen3).
- Sliding layers: RoPE **NEOX** (half-split), theta 10000. The patch adds `kolibri1`
  to llama.cpp's NEOX list, and vLLM's `get_rope` defaults to NEOX. Reflex's
  `rope_type_for` already falls through to NEOX for unknown architectures, but should
  get an explicit entry and test anyway.
- Full-attention layers: **no RoPE at all** (NoPE) and no extra scaling. The scale is
  `1/sqrt(128)` on every layer.
- Window of 513 means a query attends to itself plus the 512 previous positions: key
  `j` is visible from query `i` when `i - j < 513`. That is the semantics of vLLM's
  `per_layer_sliding_window` and llama.cpp's `LLAMA_SWA_TYPE_STANDARD`. **For prompts
  of 513 tokens or fewer the window has no effect**, so it doesn't touch most
  cold-start runs.

**Layer (sandwich norms).** The GGUF names are confusing, so here is the data flow:

```
h   = x + post_attention_norm( attn( attn_norm(x) ) )
out = h + post_ffw_norm( moe( ffn_norm(h) ) + shared_expert( ffn_norm(h) ) )
```

HF's `post_attention_layernorm` is the **pre-FFN** norm (`ffn_norm` in the GGUF), HF's
`post_attn_norm` is `post_attention_norm`, and HF's `post_ffn_norm` is
`post_ffw_norm`. The shared expert is an ordinary ungated SwiGLU added to the routed
output before the post-FFN norm. Every layer is MoE; there are no dense-lead layers.

**Router (`SIGMOID_LOGIT_ADD`).** This is new; none of Reflex's existing routers match
it.

```
logits  = ffn_gate_inp · x                          # f32, [384]
chosen  = top6( logits + exp_probs_b )              # select on biased raw logits
weights = sigmoid( logits[chosen] )                 # weight by UNbiased sigmoid
                                                    # no renormalization
```

DeepSeek-V3's convention (selecting on `sigmoid(logits) + bias`) picks different
experts whenever the bias is nonzero; using it would be a silent bug.

**Tensor types in the Q4_K_M GGUF** (903 tensors, 47.5 GB):

| Tensor | Type | Shape |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K (all 50 layers) | [2560, 512, 384] |
| `ffn_down_exps` | **Q4_K in 25 layers, Q6_K in 25** | [512, 2560, 384] |
| `ffn_gate_shexp`, `ffn_up_shexp` | Q4_K | [2560, 512] |
| `ffn_down_shexp` | Q4_K / Q6_K (25 / 25) | [512, 2560] |
| `attn_q`, `attn_k`, `attn_output` | Q4_K | |
| `attn_v` | Q4_K / Q6_K (25 / 25) | [2560, 512] |
| `token_embd` | Q4_K | [2560, 128000] |
| `output` | Q6_K | [2560, 128000] |
| norms, `ffn_gate_inp`, `exp_probs_b.bias` | F32 | |

So keeping experts quantized needs **both a Q4_K and a Q6_K expert path**, not just
Q4_K.

**Other GGUFs.** [Hob-forge/Kolibri-1-GGUF](https://huggingface.co/Hob-forge/Kolibri-1-GGUF)
also ships Q2_K, Q3_K_M, Q5_K_M, Q6_K and Q8_0;
[webmp3/Sakura-MicroQuality-Kolibri-1-GGUF](https://huggingface.co/webmp3/Sakura-MicroQuality-Kolibri-1-GGUF)
ships IQ2_XS/IQ3_XXS/IQ4_XS mixes (20.9 to 38 GiB). Q4_K_M is the target: the smallest
file whose types the quantized-resident path already has kernels for. No small real
`kolibri1` model exists.

## Phase 1: config, tokenizer and fixture

- `src/model/config.rs`: accept `kolibri1`, read the keys above (including the
  sliding-window pattern array and the gating function, rejecting any other value of
  `expert_gating_func`), add an explicit `rope_type_for` entry with a test in
  `src/model/rope_type_tests.rs`.
- Tokenizer: no new algorithm. Add a golden test comparing Reflex's token IDs with
  the patched llama.cpp's `llama-tokenize` (or HF `tokenizers`) on German compound
  nouns, umlauts/ß, digits, code and the chat-template special tokens.
- Synthetic fixture: write a small HF checkpoint (random weights, real tensor names,
  ~4 layers with at least one full-attention layer, ~16 experts, top-k below the
  expert count, nonzero `expert_bias`), convert it with the **patched**
  `convert_hf_to_gguf.py`, and quantize it to Q4_K_M so it has the same type mix.
  Same recipe as the MLA and qwen3moe fixtures (see
  [DEVELOPMENT.md](../DEVELOPMENT.md)'s test-fixture section). A nonzero bias is what
  separates `SIGMOID_LOGIT_ADD` from the DeepSeek-V3 convention, so a zero-bias
  fixture would not catch that bug.

### Phase 1 results (2026-10-06)

- **Config**: `parse_kolibri_config` / `KolibriConfig` in `src/model/config.rs`.
  `parse_model_config` now refuses `kolibri1` (it reports a nonzero `expert_count`,
  so it used to be accepted as a generic MoE model). One convention found in the
  patch's `load_arch_hparams` and not listed above: sliding layers take their RoPE
  base from `rope.freq_base_swa` when present (real Kolibri-1 doesn't set it). The
  parser is stricter than the patch in one place: a missing sliding-window pattern
  is an error, not a period-5 default. Tests: `src/model/kolibri_config_tests.rs`.
- **Tokenizer**: no code change needed. Reflex, HF `tokenizers` 0.23.2 and the
  patched `llama-tokenize` agree on all 11 golden strings (German compounds,
  umlauts/ß/ẞ, digits, code, whitespace, emoji, ChatML and `<think>` markers):
  `test_kolibri1_encode_matches_hf_tokenizers_if_fixture_present`, against
  `test-data/kolibri1-tokenizer.gguf` (the real Q4_K_M file's metadata section).
- **Fixture**: `test-data/tiny-kolibri1.gguf` (Q4_K_M, 63 MB) and
  `tiny-kolibri1-f32.gguf`: 6 layers (layer 4 full/NoPE), hidden 256, 4 Q / 2 KV
  heads of 64, 16 experts top-4, expert and shared width 256, nonzero router bias,
  **sliding window 16** (not 513, so short prompts exercise the mask), real
  vocab. Its Q4_K_M type mix matches the real file: Q6_K `attn_v`/`ffn_down_exps`/
  `ffn_down_shexp` in layers 2 and 5, Q6_K `output`, Q4_K `token_embd`. The patched
  `llama-simple` loads it and generates. Source, Dockerfile and check scripts:
  `test-data/tiny-kolibri1-src.tar.gz`.
- **Pinned reference**: llama.cpp `836d571` + `kolibri1-llama.cpp.patch` sha256
  `e0d17c26a03784a8267cb16a7287e8b4e7d799979b584334770bf7eb620c66aa`. Fixture sha256
  (the generator is seeded, so a rebuild should reproduce them):
  - `tiny-kolibri1.gguf`: `19861ca01f481f358fe5af96bf1a2a77b84c77d7c0a9ba2fd72078c4749ed78e`
  - `tiny-kolibri1-f32.gguf`: `1c419c0ed871909c6133e513161a3f3ff8d4845fea0d4bd5b734645096daf239`
  - `kolibri1-tokenizer.gguf`: `3bac2514011717e84822aa233b7abeb903dda16f033f8452e617f89686a5c06d`

## Phase 2: layer math

- NoPE: skip the RoPE launch on layers where the pattern is `false`.
- Sandwich norms: two extra RMSNorms per layer (`post_attention_norm`,
  `post_ffw_norm`), as in the data flow above.
- Router: add `route_sigmoid_logit_add` to `src/moe.rs` (host side, like the existing
  routers), with unit tests that include a case where it and the DeepSeek-V3
  convention pick different experts.
- Shared expert: ungated, added before `post_ffw_norm` (same shape as MLA's
  `MlaFfn::Moe` shared expert, different place in the layer).
- Sliding window: mask keys with `i - j >= 513` in `attention.cu`,
  `attention_online.cu` and `attention_prefill.cu`, keeping the full KV cache. Only
  matters for contexts above 513 tokens. A ring-buffer KV cache stays deferred (not
  on the cold-start path; it would also need a new cache-shape version in
  `src/kv_io.rs`).

### Phase 2 results (2026-10-06)

- **Where it lives**: Kolibri is a third layer variant on the dense/MoE path
  (`LayerWeights::Kolibri`, `src/model/dense.rs`), not a separate model like
  hybrid/MLA. Its KV cache has the dense shape (the window is a mask over the full
  cache), so generate, `--export-kv`/`--import-kv`, System1 and batched prefill work
  unchanged. `forward_attn_block{,_batched}` take an `AttnMode` (RoPE on/off, window)
  and an optional post-attention norm; every other architecture passes
  `AttnMode::STANDARD` and `None`.
- **Router**: `crate::moe::route_sigmoid_logit_add`, with a unit test where it and
  the DeepSeek-V3 rule pick different experts. `moe_ffn_grouped` now takes the router
  as a closure; the decode loop is `moe_ffn_routed`.
- **Sliding window**: the decode path narrows the K/V views to the last `window`
  positions (no kernel change); `attention_online.cu` and `attention_prefill.cu` mask
  per row (`window` argument, `0` = none).
- **`expert_weights_scale`**: the patch hardcodes `w_scale = 0` (ignored), so a file
  that sets a scale other than 0/1 is now rejected instead of applied.
- **Tests** (`src/model/kolibri_forward_tests.rs`, GPU, `#[ignore]`): byte-exact
  greedy ids vs the patched llama.cpp on `tiny-kolibri1-f32.gguf` (4 prompts up to 39
  tokens, 20 generated tokens each, so prefill and decode both cross the 16-token
  window); sequential vs batched prefill on both fixtures and both attention kernels;
  both kernels' window masking vs a host reference. The golden ids come from
  `ref_ids.cpp` (a 20-token greedy loop over `llama.h`, f32 KV cache) in
  `tiny-kolibri1-src.tar.gz`. The Q4_K_M fixture is only run end to end: llama.cpp's
  CPU path quantizes activations to Q8_K, and that random-weight fixture's top-1/top-2
  margins (down to 0.008) are too narrow for an exact comparison to mean much.
- **Verified on a T4 (2026-10-06)**: all 4 golden prompts 20/20 tokens identical to
  llama.cpp; `reflex check` passes on the f32 fixture; sequential vs batched prefill
  max abs diff 6.8e-6 (f32) and rel L2 7.3e-4 (f16); window masking vs the host
  reference 2.1e-7 worst case, both kernels. Regressions unchanged: Qwen3-0.6B
  online-vs-legacy attention (GQA and MLA) and end to end, batched prefill on
  Qwen3-0.6B, the qwen3moe, qwen35moe and MLA fixtures. Cold start on the Q4_K_M
  fixture: 412 ms to first token (`model_load_ms` 218).

## Phase 3: experts kept quantized on the GPU (core work)

- **Extend `REFLEX_QUANT_RESIDENT` to MoE expert tensors**: keep each stacked
  `[in, out, 384]` tensor as raw blocks in the device arena and read one expert's
  slice with `gemv_q4k` or `gemv_q6k`, a sliced variant of `gemv_expert` for each
  type. Expert slices are whole rows of blocks (512 and 2560 are both multiples of
  256), so an expert's slice is a contiguous byte range.
- Attention weights (`attn_v` Q6_K in half the layers), the shared expert and the LM
  head go through the same quantized path.
- Batched prefill: the batched MoE prefill path has to handle 384 experts and
  quantized experts; `prefill_dense_batched_matches_sequential_prefill` stays the
  guard test.
- Memory budget: ~47.5 GB of weights + KV cache (50 layers x 4 KV heads x 128 x 2 x
  f32 = 200 KB per token) + scratch. Fits on 80 GB, not on 48 GB.

### Phase 3 step 1 results (2026-10-06)

- **Loading**: with the flag on, Kolibri-1's ten per-layer matmul tensors
  (`KOLIBRI_MATMUL_TENSORS`, including the 3-D `ffn_*_exps` stacks) and a Q6_K
  `output.weight` go into the quantized arena if Q4_K or Q6_K. Norms, the F32 router
  and `exp_probs_b` stay f32. Other MoE models still ignore the flag, with a notice.
- **Experts**: `Model::quant_expert_weight` turns expert `e` of a quantized stack into
  its own 2-D `Weight` sharing the arena (offset `e * len / expert_count`), so the
  existing GEMV/GEMM dispatch serves it. `expert_gemv` (decode) and `expert_gemm` (an
  expert group of batched prefill) pick that or the old f32/f16 view; for f32/f16
  stacks they behave exactly as before.
- **Q6_K in batched GEMM**: one row uses `gemv_q6k_kernel`; more rows dequantize into
  the shared scratch buffer with the load-time Q6_K kernel (same values the load would
  write), then cuBLAS. There is no multi-row Q6_K kernel yet. `apply_lora`'s
  materialize step handles Q6_K and 3-D tensors too.
- **Verified on a T4**: every matmul tensor of layers 0 (Q4_K) and 2 (Q6_K
  `attn_v`/`ffn_down_*`) against its dequantized copy, first/middle/last expert, 1 to
  23 rows, both `--weights` modes, relative max diff < 1e-4
  (`kolibri1_quant_resident_matmuls_match_dequantized`); greedy ids identical to the
  flag-off path in both modes, including a prompt past the 16-token window, and
  batched vs sequential prefill agree with the flag on
  (`kolibri1_quant_resident_generate_matches_dequantized`). Regressions unchanged:
  Phase 2's tests, dense quantized-resident on Qwen3-0.6B, grouped MoE prefill on the
  qwen3moe/qwen35moe/MLA fixtures.
- **Fixture numbers** (Q4_K_M, 7-token prompt, 16 tokens, n=3, flag off vs on): same
  16 tokens; peak VRAM 279 vs 183 MiB (process total, CUDA context included);
  `prompt_eval_ms` ~31 vs ~13; first token ~427 vs ~409 ms, `model_load_ms` ~220 for
  both (this fixture's load is dominated by fixed costs, not weight bytes).

### Phase 3 step 2 results (2026-10-07)

Real `Hob-forge/Kolibri-1-GGUF` `Kolibri-1-Q4_K_M.gguf` (47,454,113,472 bytes) on a
ThunderCompute A100-SXM4-80GB, Reflex built with `REFLEX_CUDA_ARCH=sm_80`, the
reference llama.cpp `836d571` + the kolibri1 patch built with CUDA for sm_80. Prompts:
the four fixture prompts plus a 627-token German ChatML reading-comprehension prompt
(`ref_prompts/p5.txt` in `tiny-kolibri1-src.tar.gz`), 20 greedy tokens each.

- **It runs and fits**: all 501 matmul tensors quantized-resident (47.07 GB arena),
  peak VRAM 45.4 GiB. Output is coherent (p5 opens a `<think>` block and answers in
  German).
- **Against stock llama.cpp** (CUDA, all layers on the GPU): p1, p4 and p5 match 20/20
  (`--weights f32` and f16 alike). p3 differs at step 19, where llama.cpp CUDA and
  llama.cpp CPU disagree with each other and Reflex sides with the CPU (CPU top-1/top-2
  gap there 0.14). p2 differs from step 1, with llama.cpp's gap at 1.1.
- **Cause of the p2 difference: llama.cpp's 8-bit activation quantization, not
  Reflex.** For llama.cpp, Q4_K/Q6_K matmuls quantize the activations to Q8_1 (CUDA
  MMVQ/MMQ) or Q8_K (CPU); Reflex reads f32 activations. With 384 experts the router's
  top-6 boundary is very tight (6th-vs-7th biased-logit margins of 0.001 to 0.05 are
  common), so that rounding flips expert picks, and the flips compound over 50 layers.
  Evidence, all on the 6-token context `Die Hauptstadt von Deutschland ist Deutschland`:
  - Expert sets identical in all 6 tokens through layer 7; the first difference is at
    layer 8 at a Reflex margin of 0.0011. llama.cpp CPU vs llama.cpp CUDA start
    differing even earlier (layer 5) and differ as much in later layers.
  - llama.cpp's own top logits move with kernel choice alone: " Deutschland" 11.97 /
    " ist" 8.30 (CUDA), 11.70 / 10.53 (CPU), and "," on top with fusion and CUDA graphs
    disabled. Reflex f16 vs f32 differ by < 0.004.
  - llama.cpp patched to dequantize to f32 and run an f32 GEMM instead
    (`llama-no-q8.patch` in the src tarball, `LLAMA_NO_Q8=1`, with
    `GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_DISABLE_FUSION=1`) gives " ist" 12.76 /
    "," 11.96 / " Deutschland" 9.59, next to Reflex's 13.50 / 12.06 / 9.26, and
    **matches Reflex 20/20 on all five prompts**.
  So exact-token agreement with *stock* llama.cpp is not a meaningful target for this
  model; the f32-activation build is the reference to compare against.
- **Experts touched** (`REFLEX_EXPERT_TRACE=1`, distinct experts per layer, mean over
  layers):

  | Prompt | Before first token | After 19 more tokens |
  |---|---|---|
  | 4–5 tokens | 4–5% | 7% |
  | 36–39 tokens | 17–18% | 20–21% |
  | 627 tokens | 52% | 53% |

  Expert bytes are ~97% of the file, so lazy upload (Phase 4) would skip most of the
  load for short prompts and about half for long ones.
- **Cold start: not measurable on this instance.** Host-to-device copies ran at
  ~0.86 GB/s (47 GB in 54 s of `h2d_gpu_ms`; disk reads at 2.7 GB/s), so both
  engines took ~60 s to first token (Reflex `model_load_ms` ~58.5 s, llama.cpp
  first token ~59 s). This GPU appears to be attached over the network; cold-start
  numbers need a machine with a local PCIe GPU.
- **Harness notes**: a Reflex process exits without tearing down its CUDA context,
  and on this instance the driver took several seconds to release 47 GB, so
  back-to-back runs hit `CUDA_ERROR_OUT_OF_MEMORY` until each run waited for
  `nvidia-smi` to show the memory free. Pass prompts byte-exactly (`$(cat file)`
  drops the trailing newline the ChatML prompts end with).

### Cold start on an RTX A6000 (2026-10-07)

Runpod Secure Cloud RTX A6000 48 GB (PCIe Gen4 x16, measured pinned H2D 19.7 GB/s),
local NVMe container disk (O_DIRECT read 11.3 GB/s), container memory limit 49 GB.
Reflex `REFLEX_QUANT_RESIDENT=1` (sm_86 cubin) vs stock patched llama.cpp (`REF_NGL=99`,
all layers on the GPU). Short prompt = p2 (5 tokens), first token only, n=3 each.
"Cold" = the GGUF evicted from the page cache (`posix_fadvise(DONTNEED)`, verified with
`mincore`: 0.0 GB cached); "warm" = 47.3 to 47.5 GB cached. Time to first token is each
engine's own figure (Reflex from process start, llama.cpp from `main()`); wall is
launch to exit, same harness for both.

| | Reflex TTFT | Reflex wall | llama.cpp TTFT | llama.cpp wall |
|---|---|---|---|---|
| short prompt, cold | 13.6 s (13.4–13.7) | 14.6 s | 14.9 s (14.7–15.3) | 15.3 s |
| short prompt, warm | 7.5 s (7.3–7.6) | 8.6 s | 7.0 s (6.8–7.3) | 7.4 s |
| 627-token prompt, cold (n=1) | 16.8 s | 17.9 s | 14.9 s | 15.2 s |

- Both engines produce the same first token in every run; the f32-activation spot check
  on this machine matched 20/20 again. Peak VRAM with the 627-token prompt and 20
  generated tokens: 45,600 of 49,140 MiB, so the model fits a 48 GB card as is.
- **Reflex wins cold by ~1.3 s, loses warm by ~0.5 s, and loses the long prompt by
  ~2 s.** At this size, cold start is all loading: `model_load_ms` is 13.1 s cold and
  7.1 s warm of Reflex's 13.6 / 7.5 s.
- **The load is bound by one host thread copying the mmap into the pinned staging
  buffers**, not by the disk or PCIe: `pinned_fill_ms` 11.9 s cold (47 GB at ~4 GB/s,
  page-faulting from a disk that reads at 11.3 GB/s) and 6.1 s warm (~7.7 GB/s, one
  memcpy thread), against `h2d_gpu_ms` 2.4 s (~19.4 GB/s). A load that kept the disk
  and the PCIe link busy at the same time would take ~4.2 s cold and ~2.4 s warm.
- **Long prompts**: Reflex's 627-token prefill took 2.75 s, against ~0.16 s for the
  5-token prompt. Each of the ~200 experts a layer touches runs as its own small group,
  and every Q6_K group above one row is dequantized into scratch first (no multi-row
  Q6_K kernel yet).

## Phase 4: cold-start work specific to a 47.5 GB model

At this size, reading the file dominates cold start: ~16 s at 3 GB/s NVMe, against
~2 s of PCIe 4 transfer. Expert weights are ~97% of the file.

- **Measured** (Phase 3 step 2): 4–5% of experts per layer before the first token for
  a 5-token prompt, ~18% at 36 tokens, 52% at 627 tokens.
- **Before lazy upload, fix the load pipeline itself** (see the A6000 numbers above):
  the pinned staging copy runs on one thread at 4 GB/s cold / 7.7 GB/s warm while the
  disk does 11.3 GB/s and PCIe 19.7 GB/s. Options: several fill threads, reading with
  O_DIRECT straight into the pinned buffers, or registering the mmap with
  `cuMemHostRegister` and skipping the copy. This helps every model, not just Kolibri.
- If the fraction is small, **upload experts lazily**: map the file, upload
  attention/shared/router weights eagerly, and upload an expert's slice the first time
  the router selects it. This trades a small per-token stall for not reading most of
  the file before the first token.

### Phase 4 step 1: faster load pipeline (2026-10-07, T4 and A6000 measured)

Two changes to `WeightLoadPipeline` (src/model/loading.rs):

- **Parallel chunked fill.** Tensors are staged in chunks of at most 64 MB through the
  two reused pinned buffers, so a big tensor's next chunk fills while the previous one
  is on the wire. A chunk of 8 MB or more is copied by several threads
  (`REFLEX_LOAD_THREADS`, default min(cores, 8)), and a `MADV_WILLNEED` window of
  256 MB runs ahead of the fill inside a large tensor (`REFLEX_LOAD_READAHEAD=0`
  disabled it; both were later replaced by `REFLEX_LOAD_READERS`, see below). Pinned memory is now 2 x 64 MB instead of 2 x the largest tensor.
  Rejected: `O_DIRECT` (warm loads would run at disk speed) and `cuMemHostRegister`
  on the mmap (registration faults and pins on one thread, so cold gains nothing;
  memlock and container limits at 47 GB).
- **No host sync per norm tensor.** Tensors with no dequant kernel (`F32` norms) used
  `htod_sync_copy`, which synchronizes the compute stream: every norm drained the
  whole pipeline. They now take the same staging slots plus an async device copy.

T4 (g4dn.xlarge, 4 vCPUs, PCIe 3), warm page cache, `model_load_ms` medians, two
interleaved rounds:

| model | before | parallel fill | + no norm sync |
|---|---|---|---|
| Mistral 7B Q4_K_M, quant-resident | 2306 ms | 1837 ms | **1738 ms (-25%)** |
| Mistral 7B Q4_K_M, f16 | 3427 ms | 3237 ms | **3066 ms (-11%)** |
| Qwen3-4B Q4_K_M, quant-resident | 1098 ms | 895 ms | **805 ms (-27%)** |
| Qwen3-0.6B Q4_K_M (both modes) | 223-233 ms | 223-233 ms | 223-229 ms (unchanged) |

Same tokens as before on Qwen3-0.6B, Qwen3-4B, Mistral 7B, Qwen3.5-0.8B and the MLA
fixture in every mode that fits the T4. On Mistral the fill now runs at 6 GB/s on 4
threads, as fast as the T4's PCIe (`h2d_gpu_ms` 687 ms), and the old ~0.9 s stall shows
up as `pinned_wait_ms` (the host waiting for the GPU). What remains is GPU-side: H2D
plus the Q6_K dequant kernels (772 ms). Cold numbers on the T4 only measure its EBS
volume (~134 MB/s, 32.5 s for Mistral in both builds); the cold/warm Kolibri table
needs the A6000.

#### Phase 4 step 1 on an RTX A6000 (2026-10-07)

Runpod Secure Cloud RTX A6000 in CA-MTL-3, **a different host from the table above**: the
container disk reads only 4.3 GB/s with `O_DIRECT` (11.3 GB/s before), pinned H2D
25.5 GB/s, 31 vCPUs, so cold numbers compare only within this table. Same harness,
p2 short prompt, first token, n=3 interleaved, medians; Reflex with
`REFLEX_QUANT_RESIDENT=1`, sm_86, default 8 fill threads after the change.

| | Reflex before (cb590cd) | Reflex after (db9328a) | llama.cpp |
|---|---|---|---|
| short prompt, cold: TTFT | 71.6 s | **19.7 s** | 41.3 s |
| short prompt, cold: wall | 72.9 s | 21.4 s | 41.7 s |
| short prompt, warm: TTFT | 7.6 s | **3.7 s** | 7.1 s |
| short prompt, warm: wall | 8.5 s | 4.5 s | 7.5 s |
| 627-token prompt, cold (n=1): TTFT | 73.7 s | 22.3 s | 42.3 s |

- Same 20 tokens before and after; first token 1678 (p2) and 127907 (p5) match
  llama.cpp in every run.
- `model_load_ms` cold 71.1 s -> 19.2 s (fill 0.68 -> 2.56 GB/s from a 4.3 GB/s disk);
  warm 7.07 s -> 3.27 s (fill 26 GB/s; `pinned_wait_ms` ~1.2 s means the PCIe copy,
  `h2d_gpu_ms` ~2.6 s, is now the limit, as intended).
- **Fill threads (cold load, one run each): 2: 39.5 s, 4: 28.7 s, 8: 19.5 s, 16: 13.6 /
  14.1 s, 24: 14.7 s, 32: 12.6 s.** Cold keeps improving well past the default of 8,
  because each thread holds one page-fault read in flight. Warm goes the other way:
  8 threads 3.09-3.45 s (n=6), 16 threads 3.38-3.91 s (n=4): more memcpy threads
  compete with the H2D DMA for host memory bandwidth (`h2d_gpu_ms` 2.4 -> 2.8 s).
- `MADV_WILLNEED` made no measurable difference, cold or warm (18.6 s vs 18.8-20.0 s
  cold at 8 threads).
- Since done (measured below): prefetch reader threads
  (`REFLEX_LOAD_READERS`, default 16) touch pages up to 1 GB ahead of the fill and copy
  nothing, replacing `MADV_WILLNEED`; T4 tests and tokens unchanged. The original idea:
  decouple I/O depth from copy threads (e.g. prefetch threads that
  only touch pages ahead of the fill, or `O_DIRECT` reads when `mincore` says the file
  isn't cached), so cold gets 16-32 reads in flight while warm keeps ~8 copy threads.

#### Prefetch readers and System1 on an RTX A6000 (2026-10-07)

Runpod Secure Cloud RTX A6000 in US-TX-1, **a third host** (cold numbers compare only
within this table): container disk 6.1 GB/s with `O_DIRECT`, pinned H2D 26.8 GB/s, a
15.3-CPU cgroup quota, 71 GB memory limit. Reflex built from public `master` at
`9ea5240` (sm_86, `REFLEX_QUANT_RESIDENT=1`). "Readers off" is the same binary with
`REFLEX_LOAD_READERS=0`, which is the `db9328a` load minus the `MADV_WILLNEED` hint that
measured no effect above. Same harness as the tables above: p2 short prompt, first
token, n=3 interleaved, medians; llama.cpp is stock `836d571` + the kolibri1 patch, all
layers on the GPU.

| | Reflex, readers off | Reflex, readers on (default 16) | llama.cpp |
|---|---|---|---|
| short prompt, cold: TTFT | 9.25 s (9.15–9.41) | **8.52 s** (8.43–8.55) | 17.19 s (16.99–17.22) |
| short prompt, cold: wall | 11.4 s | 10.7 s | 17.6 s |
| short prompt, warm: TTFT | **3.79 s** (3.58–3.82) | 4.02 s (3.24–4.16) | 7.63 s (7.49–8.01) |
| short prompt, warm: wall | 5.6 s | 5.8 s | 8.0 s |
| 627-token prompt, cold (n=1): TTFT | | 11.16 s | 18.40 s |

- Same 20 tokens with readers off and on; first token 1678 (p2) and 127907 (p5) match
  llama.cpp in every run.
- **Readers cut the cold load by ~0.8 s (~9%)**: `model_load_ms` 8.67 -> 7.85 s, fill
  5.75 -> 7.31 GB/s, faster than one `O_DIRECT` stream reads this disk (6.1 GB/s). Warm,
  they don't help: the file is already cached, yet the readers still walk all 47 GB
  (`prefetched_mb` 46,987) and the warm median is ~0.2 s worse, within this host's warm
  spread. Skipping the readers when `mincore` shows the file cached is the obvious fix;
  not done.
- **Cold sweep (one run each, TTFT)**: readers 0 / 8 / 16 / 32 at 8 fill threads: 9.04 /
  8.19 / 8.52 / 8.73 s; 16 readers at 4 / 16 threads: 8.75 / 8.34 s; 32 readers at 16
  threads: 8.57 s. On this disk everything from 8 readers up sits within ~0.5 s, the
  run-to-run spread, so the default of 16 stays.
- **Reflex is ~2.0x faster than llama.cpp to a cold first token here** (8.5 vs 17.2 s)
  and on a warm one (3.8-4.0 vs 7.6 s), about the same ratio as the CA-MTL-3 table's
  19.7 vs 41.3 s (2.1x), on a disk that reads 1.4x faster.
- **Wall minus TTFT** is ~2 s for Reflex against ~0.4 s for llama.cpp: Reflex's process
  takes longer to exit after the first token. Not broken down yet; it matters only
  where a caller waits for the process to exit rather than for the token.

**System1 (candidate scoring, the path behind the sidecar's `POST /v1/classify`)** on
the same pod, prompt `Review: The battery died after two days. Sentiment:`, candidates
`" positive"` / `" negative"`:

| | process start to result |
|---|---|
| cold (one process per decision), n=2 | 8.60 s, 8.77 s |
| warm page cache, new process, n=2 | 4.16 s, 4.19 s |

Warm process (`reflex stdio`, the engine path `/v1/classify` uses; one process, 20
requests per case, ready 4.25 s after launch with the file cached):

| case | labels | p50 | min–max |
|---|---|---|---|
| English sentiment (above) | 2 | 190 ms | 187–311 ms |
| German sentiment (`Bewertung: Der Akku war nach zwei Tagen leer. Stimmung:`) | 2 | 203 ms | 200–205 ms |
| 4-way ticket routing (`I was charged twice for my subscription this month. Department:`) | 4 | 229 ms | 227–230 ms |

- A warm decision costs ~0.2 s, about 20x Qwen3-0.6B's ~10 ms on a T4: one prefill of a
  ~15-token prompt through 384-expert layers (`prompt_eval_ms` ~165 ms for the 5-token
  p2 prompt above). Every label here is one token.
- **Zero-shot answers with bare prompts are unreliable on Kolibri-1.** The English
  review came out `positive` at 0.74, and the double charge went to `legal` (0.45) and
  `sales` (0.41), with `billing` at 0.06. The German review was right (`negativ` 0.98).
  Three prompts are not a quality evaluation, but they are enough to say that a
  classifier on this model needs prompt work (or few-shot examples) checked on real
  data before it is trusted. The scores themselves are deterministic: identical across
  all cold, warm and stdio runs.

### Phase 4 step 2: lazy expert upload (design, 2026-10-07)

**Why.** With the load pipeline fixed, a cold first token on an A6000 is 8.5 s, nearly all
of it reading 47.5 GB. A 5-token prompt routes to 4–5% of each layer's experts before the
first token, so ~95% of the expert bytes, ~92% of the file, are read for nothing.

**What.** Opt-in `REFLEX_LAZY_EXPERTS=1`, effective only with `REFLEX_QUANT_RESIDENT=1` on
Kolibri-1 (any other model prints a notice and loads eagerly).

- **Load.** Every non-expert tensor loads as today. The three stacked expert tensors per
  layer (`ffn_{gate,up,down}_exps`) get their arena space reserved, so offsets and the
  one-allocation arena are unchanged, but nothing is copied. Each layer keeps the
  tensors' mmap byte ranges (`gguf::SharedBytes`) and a per-expert resident flag. The
  prefetch readers are off: they walk the data section in file order and would read the
  skipped expert bytes.
- **Forward.** Routing is already host-side (the router logits come back with
  `dtoh_sync_copy` and the top-k runs on the CPU), so each layer knows its experts before
  any expert matmul. The decode layer passes its 6 routed experts, the batched layer the
  union over its rows, to `Model::ensure_experts`, which uploads the ones not yet
  resident and marks them. All of an expert's three slices go together.
- **Upload.** Each expert slice is a contiguous byte range in the file and in the arena
  (`expert_idx * len / expert_count`, the same arithmetic as `quant_expert_weight`). The
  missing slices of one layer are packed into the existing pinned slots, filled in
  parallel by the existing fill workers (one slice per job, so cold page faults run
  several reads at once), and copied with `memcpy_htod_async` on the copy stream; the
  compute stream waits on the last copy's event, exactly like `stage_h2d`. The kernels
  read byte-identical blocks, so the output is identical to the eager load.
- **Fill workers outlive the load** in this mode. They are still copy helpers for one
  job at a time, not a request pool (Non-goals unchanged).

**Expected.** A 5-token prompt needs the ~1.4 GB of non-expert weights plus ~2.3 GB of
experts (≈19 experts × ~2.4 MB × 50 layers) instead of 47.5 GB. Each layer stalls the
GPU while its missing experts arrive, so a long prompt (52% of experts at 627 tokens)
may end up slower than the eager pipeline, which overlaps copies with dequant work;
that gets measured, not assumed.

**Not in this step.** Filling the remaining experts in the background (what a warm
`stdio` process would want), predicting the next layer's experts, and any model other
than Kolibri-1. A LoRA-targeted expert stack loads eagerly in f32 as today (it is
merged at load), and only the other stacks are lazy.

**Verification.** `kolibri1_lazy_experts_match_eager` on the tiny Kolibri fixture: no
expert is resident after the load, a 1–2 token prompt uploads some but not all, and
lazy and eager give identical greedy ids (batched prefill, then decode), an identical
sequential-prefill hidden state, and identical System1 scores, in f32 and f16.
`REFLEX_LAZY_EXPERTS resident=<n> total=<m>` on stderr after `generate`/`system1`
reports how many experts a run uploaded.

**T4 results (2026-10-07).** The test passes, as do the existing Kolibri,
quantized-resident, pipeline, prefill-batching and f16 GPU tests. On the Q4_K_M
fixture (6 layers × 16 experts, top-4), `generate` of 20 tokens and `system1` give the
same token ids and scores lazy and eager, with 72/96 and 45/96 experts uploaded. The
A6000 timing on the real model is below. On an A6000: cold and warm first token, a
627-token prompt, and System1 cold, lazy against eager on the same host.

#### Lazy expert upload on an RTX A6000 (2026-10-08)

The same US-TX-1 host as the readers table above (same IP; 6.1 GB/s disk, 15.3-CPU
quota). Public `master` at `0ca45e7`, sm_86, `REFLEX_QUANT_RESIDENT=1`; "eager" is the
default load (prefetch readers on), "lazy" adds `REFLEX_LAZY_EXPERTS=1`. Same harness:
first token, n=3 interleaved medians unless noted; llama.cpp is stock `836d571` + the
kolibri1 patch.

| | Reflex eager | Reflex lazy | llama.cpp | experts uploaded (lazy) |
|---|---|---|---|---|
| p2 (5 tokens), cold: TTFT | 8.55 s | **2.33 s** (2.31–2.41) | 16.80 s | 942 / 19,200 (4.9%) |
| p2, cold: wall | 10.4 s | 2.8 s | 17.1 s | |
| p2, warm: TTFT | 3.51 s | **1.21 s** (1.13–1.21) | 8.55 s | |
| p2, warm: wall | 5.4 s | 1.7 s | 8.9 s | |
| p3 (~37 tokens), cold, n=2 | 8.48, 8.85 s | **3.97, 4.09 s** | | 3,333 (17.4%) |
| p5 (627 tokens), cold, n=2 | 11.16, 11.11 s | **9.96, 10.10 s** | 19.07 s (n=1) | 9,934 (51.7%) |
| System1, cold, n=2 (process start to result) | 8.46, 8.34 s | **2.76, 2.97 s** | | 1,654 (8.6%) |
| System1, warm cache, n=2 | 3.73, 4.08 s | **1.47, 1.42 s** | | |

- **Same output.** 20 greedy tokens are identical lazy and eager on p2 and p5, and the
  System1 scores are identical to every printed digit.
- **A short prompt now reaches its first token 3.7x faster cold** (8.55 -> 2.33 s) and
  **7.2x faster than llama.cpp**. `model_load_ms` drops from ~8.0 s to ~0.9 s (the non-expert
  weights only), and the expert uploads move into `prompt_eval_ms` (0.16 ->
  ~0.95 s cold).
- **Warm (file in page cache)**: 3.51 -> 1.21 s, against llama.cpp's 8.55 s.
- **Long prompts still gain, barely.** At 627 tokens lazy uploads half the experts and
  wins by ~1.1 s, but its `prompt_eval_ms` is 8.7 s against eager's 2.8 s: each layer
  waits for its own ~200 experts to arrive (~24 GB at ~3 GB/s) with nothing overlapping
  the copies. Prefetching the next layer's experts during the current layer's compute,
  or switching to the eager pipeline above some prompt length, would recover that.
- **System1 classification**: 2.8–3.0 s from a cold process to a decision (was 8.3–8.5
  s), 1.4–1.5 s with the file cached.
- **Warm process (`reflex stdio`, lazy, page cache dropped first)**: ready 1.37 s after
  launch. The first request of each kind pays for its experts (1.57 s English sentiment,
  0.87 s German, 1.02 s 4-way routing); after that the p50 is 202 / 216 / 245 ms, in line
  with the eager process's 190–229 ms in the session above.

## Phase 5: verification and benchmarks

- Correctness: byte-exact greedy agreement with the patched llama.cpp via
  `reflex check`, first on the synthetic fixture, then on the real Q4_K_M GGUF. Use
  German and English prompts, including some longer than 513 tokens so the window is
  exercised. If Reflex and the patch disagree, check against the official vLLM plugin
  and the CWBudde port before deciding which side is wrong.
- Hardware: 1x A100 80 GB or H100 80 GB, Q4_K_M, `REFLEX_QUANT_RESIDENT=1`.
- Metrics: `process_start_to_first_token_ms`, the `model_load_ms` breakdown
  (`REFLEX_LOAD_PROFILE=1`), cold vs warm page cache, energy, and the patched
  llama.cpp's cold start on the same machine for comparison. Report tokens/s without
  optimizing for it.
- Docs: add the architecture to README's supported models and to
  [DEVELOPMENT.md](../DEVELOPMENT.md).

## Out of scope (separate proposals if wanted)

- Multi-GPU sharding (e.g. 2x RTX 4090), which is also what Aleph Alpha's own BF16/FP8
  checkpoints need.
- FP8 weights; Q5_K, Q3_K, Q2_K and IQ quantized-resident kernels.
- Ring-buffer KV cache.

## Risks

- **The reference is unmerged community code.** The math matches the official plugin
  as read, but the patch could still change, or upstream could land a different
  conversion (different tensor names or metadata keys) that breaks these GGUFs. Pin
  the patch file's hash and llama.cpp `836d571` in the test notes.
- **Load time.** 47.5 GB may make cold start look poor next to small models; lazy
  expert upload (Phase 4) is the main lever and its value is unmeasured.
- **Silent convention mismatches.** The router selection rule, the sandwich-norm order
  and NoPE layers are all ways to get wrong output without a crash. Each is pinned
  above and must be covered by the byte-exact comparison.
