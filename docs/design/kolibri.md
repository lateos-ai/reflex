# Design: Kolibri-1 MoE support

**Status: Phase 1 done (2026-10-06): config, tokenizer and fixture. The forward pass
(Phase 2) is not implemented; `Model::load` rejects `kolibri1` with a clear error.**
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
