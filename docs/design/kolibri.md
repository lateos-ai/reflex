# Design: Kolibri-1 MoE support

**Status: Phase 3 done (2026-10-07): the real Kolibri-1 Q4_K_M runs on one A100
80 GB with `REFLEX_QUANT_RESIDENT=1` (45.4 GB peak VRAM) and matches llama.cpp token
for token on 5 German/English prompts up to 627 tokens, once llama.cpp computes with
f32 activations like Reflex (see [Phase 3 step 2 results](#phase-3-step-2-results-2026-10-07)).
Cold start on real hardware is still unmeasured. Next: Phase 4 (lazy expert upload),
which the expert-usage numbers below support for short prompts.**
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

## Phase 4: cold-start work specific to a 47.5 GB model

At this size, reading the file dominates cold start: ~16 s at 3 GB/s NVMe, against
~2 s of PCIe 4 transfer. Expert weights are ~97% of the file.

- **Measure first**: for a set of real prompts, record how many distinct experts per
  layer prefill and the first token actually touch.
- If the fraction is small, **upload experts lazily**: map the file, upload
  attention/shared/router weights eagerly, and upload an expert's slice the first time
  the router selects it. This trades a small per-token stall for not reading most of
  the file before the first token.

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
