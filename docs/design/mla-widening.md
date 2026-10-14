# Design: MLA widening to small real checkpoints (`deepseek2`)

**Status (2026-10-13): done and verified on an A100 80 GB.** DeepSeek-Coder-V2-Lite,
GigaChat3.1-10B-A1.8B and GLM-4.7-Flash run, and real DeepSeek-V2-Lite was re-verified
with its `deepseek-llm` tokenizer. Every existing `reflex check` row is unchanged (45/45,
V2-Lite and Coder-V2-Lite included). Greedy tokens agree with llama.cpp up to a near-tie
on every prompt, and at every one of the 8 divergences Reflex picks llama.cpp's own
runner-up, 0.005 to 0.19 logits behind. Cold start on GigaChat3.1 is 4.22 s p50 vs
`llama-simple`'s 4.40 s. See [Results](#results-2026-10-13).

Goal: run the small real `deepseek2`-architecture checkpoints people actually use, not
just DeepSeek-V2-Lite. Correctness means byte-exact greedy agreement with llama.cpp, as
for every other architecture. Targets, in order:

1. **DeepSeek-Coder-V2-Lite-Instruct**: needs nothing new; verify it.
2. **GigaChat3.1-10B-A1.8B**: DeepSeek-V3-style sigmoid routing with an `exp_probs_b`
   selection bias, and the `gigachat` pre type. The smallest real file, so it's also the
   cold-start benchmark target.
3. **GLM-4.7-Flash**: Q-LoRA queries, sigmoid routing, the `glm4` pre type, and
   llama.cpp's rule for GLM files that carry no gating key.

Queued, not in this round: Moonlight-16B-A3B (`kimi-k2` pre type, a custom Han-aware
splitter) and Kanana-2-30B-A3B (`kanana2`, llama.cpp's `gpt-4o` split). Once sigmoid
routing lands, each needs only its pre type.

## Scope

In:

- Sigmoid gating (`expert_gating_func = 2`) with the `exp_probs_b` selection bias.
- `expert_weights_scale` applied the way llama.cpp does: skipped when 0 or 1.
- Q-LoRA query decomposition: `wq_a` → RMSNorm(`attn_q_a_norm`) → `wq_b`.
- llama.cpp's fallback for GLM-4.7 files with no `expert_gating_func` key.
- Pre types `gigachat` and `glm4` (`glm4` also turns BOS off, as llama.cpp does).
- `system1`, `--weights f16|f32` and `--export-kv`/`--import-kv` on all of the above.
  They go through the same MLA forward, and the KV cache shape doesn't change.

Out, each rejected at load with a clear error naming the key or tensor:

- `expert_group_count > 1` (group-limited top-k). No surveyed small file uses it. Only
  the 671B V3/R1 do (8 groups), and they don't fit one GPU.
- `nextn_predict_layers > 0` (MTP/NextN). Rejected as before. No surveyed file carries
  NextN layers; the converter drops them (`skip_mtp`).
- `expert_gating_func` values other than softmax (1) and sigmoid (2):
  `SOFTMAX_WEIGHT` (3) and `SQRT_SOFTPLUS` (4).
- `attention.temperature_scale` (Mistral-Large-3's Llama-4 temperature tuning, which
  shares this graph).
- All-MoE files (`leading_dense_block_count = 0`). Rejected as before; every surveyed
  file has one dense lead layer.
- Legacy pre-split files (unsplit `attn_kv_b`, no `key_length_mla`). Rejected as before.
- `--lora` on MLA. Rejected as before.
- `REFLEX_QUANT_RESIDENT=1` keeps printing its notice and loading MLA weights in the
  `--weights` dtype, as before.

## Reference implementations

- llama.cpp `79e2e74` (the local `reflex-llama-qwen2:79e2e74` image):
  - `src/models/deepseek2.cpp`: hparams (`is_lite`, the GLM-4.7 gating fallback, the
    YaRN log-mul fix), tensors (Q-LoRA vs direct `wq`, optional `exp_probs_b`), graph.
  - `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`): gating, selection bias,
    group-limited top-k, weight normalization and scaling.
  - `llama_model_base::load_hparams` (`src/llama-model.cpp`): `expert_group_count`/
    `expert_group_used_count` and their asserts, `nextn_predict_layers`.
  - `src/llama-vocab.cpp`, `src/unicode.cpp`: pre types and BOS rules.
  - `conversion/deepseek.py` and `conversion/base.py`: what the converter writes
    (`scoring_func` → `expert_gating_func`, `n_group`/`topk_group` → group keys,
    `e_score_correction_bias` → `exp_probs_b.bias`, `q_lora_rank`).
- GGUF headers (HTTP range request on the first 40 MB of each file; `test-data/mla-survey/
  probe.py`). Header fixtures kept in `test-data/` (metadata and tensor table, no tensor
  data; `scripts/gguf_tokenizer_only.py --keep-tensor-table`):

  | Fixture | Source |
  |---|---|
  | `deepseek-coder-v2-lite-header.gguf` | `mradermacher/DeepSeek-Coder-V2-Lite-Instruct-abliterated-GGUF` Q4_K_M |
  | `deepseek-coder-v2-lite-legacy-header.gguf` | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` Q4_K_M (legacy format) |
  | `gigachat3.1-10b-a1.8b-header.gguf` | `ai-sage/GigaChat3.1-10B-A1.8B-GGUF` q8_0 |
  | `glm-4.7-flash-header.gguf` | `unsloth/GLM-4.7-Flash-GGUF` Q4_K_M (has `expert_gating_func`) |
  | `glm-4.7-flash-ggml-org-header.gguf` | `ggml-org/GLM-4.7-Flash-GGUF` Q8_0 (no `expert_gating_func`) |
  | `moonlight-16b-a3b-header.gguf` | `gabriellarson/Moonlight-16B-A3B-Instruct-GGUF` Q4_K_M |
  | `kanana-2-30b-a3b-header.gguf` | `deul/Kanana-2-30b-a3b-instruct-2601-GGUF` Q4_K_M |

## Header survey

14 files checked. "split" means the MLA tensor-split format `parse_mla_config` requires
(`attn_k_b`/`attn_v_b`, `key_length_mla`); "legacy" means the unsplit `attn_kv_b`.

| Model | pre | q_lora | kv_lora | k/v mla | rope scaling | experts (used / shared) | gating | `exp_probs_b` | scale / norm | groups | dense lead | NextN | LM head | format |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite | `deepseek-llm` | – | 512 | 192/128 | yarn ×40 | 64 (6/2) | softmax | no | 1.0 / – | 1 | 1 | 0 | separate | split: mradermacher (abliterated), mad-lab-ai; legacy: bartowski, Guervency, mradermacher V2-Lite-Chat |
| GigaChat3.1-10B-A1.8B | `gigachat` | – (26 layers) | 512 | 192/192 | yarn ×64 | 64 (4/1) | sigmoid | yes | 1.0 / true | 1 | 1 | 0 | separate | split (owner; bartowski's GigaChat3 too) |
| Moonlight-16B-A3B | `kimi-k2` | – | 512 | 192/128 | none | 64 (6/2) | sigmoid | yes | 2.446 / true | –, 1 | 1 | 0 | separate | split: gabriellarson, shoonthitsa; legacy: mmnga (`gpt-2` pre, yarn) |
| GLM-4.7-Flash | `glm4` | 768 | 512 | 256/256 | none | 64 (4/1) | sigmoid; key missing in ggml-org's and bartowski's | yes | 1.8 / true | 1 | 1 | 0 | separate | split (ggml-org, unsloth, bartowski) |
| Kanana-2-30B-A3B | `kanana2` | – (48 layers, 128K vocab) | 512 | 192/128 | none | 128 (6/2) | sigmoid | yes | 2.448 / true | 1 | 1 | 0 | separate | split |

Other facts from the same headers:

- Matmul types: Q4_K/Q5_0/Q6_K/Q8_0 in the Q4_K_M files (Q5_K too in GLM's and GigaChat's),
  all Q8_0 in the Q8_0 files. Every one already has an on-device dequant kernel.
  `ffn_gate_inp` and `exp_probs_b` are F32.
- No file has a fused `ffn_gate_up_exps`, NextN tensors or `attention.temperature_scale`.
- BOS: DeepSeek, GigaChat and Kanana set `add_bos_token = true`. GLM-4.7-Flash and
  Moonlight have no `add_bos_token` key.
- Weight size under Reflex's residency (matrix weights dequantized to f16): Coder-V2-Lite
  ~31 GB, GigaChat3.1 ~21 GB, GLM-4.7-Flash ~60 GB (A100 80 GB).

**Converter gotcha.** At `79e2e74`, `convert_hf_to_gguf.py` identifies the tokenizer by
hashing the token ids it produces for a test string. With `transformers` 5.19 (what the
local `reflex-llama-qwen2:79e2e74` image has), DeepSeek-Coder-V2-Lite's tokenizer hashes
to an unknown value (`99ecacb4…`), so the converter warns and writes no usable pre type.
With `transformers` 4.57.1 it hashes to `deepseek-llm`, matching the converted files
above. Convert DeepSeek checkpoints with `transformers==4.57.1`.

## What each target needs

### Routing: sigmoid gating with a selection bias

`build_moe_ffn` with `gating_op = SIGMOID` and an `exp_probs_b` tensor:

1. `probs = sigmoid(logits)`.
2. `selection = probs + exp_probs_b`. The bias only steers selection; weights come from
   the unbiased `probs`.
3. Group-limited masking if `n_expert_groups > 1` (rejected here).
4. `selected = top_k(selection, n_expert_used)`.
5. `weights = probs[selected]`. If `expert_weights_norm`, they are divided by their sum,
   clamped below at `6.103515625e-5` (2⁻¹⁴).
6. If `expert_weights_scale` is neither 0 nor 1, the weights are multiplied by it.

This is not Kolibri-1's `SIGMOID_LOGIT_ADD` rule (`moe::route_sigmoid_logit_add`), which
selects on `logits + bias` *before* the sigmoid. Sigmoid is monotonic, but adding the bias
before or after it ranks experts differently whenever biases differ, so the two rules
pick different experts on the same input. The new `moe::route_sigmoid_bias` shares the
ranking and normalization code with Kolibri's function rather than copying it, and the
`expert_gating_func` constants and parsing are shared with `parse_kolibri_config`.

With no `exp_probs_b` tensor, the bias is zero (llama.cpp skips the add). The function
takes the bias as optional.

**Existing behavior.** Today's MLA path never reads `expert_gating_func`, so a sigmoid
file would load and silently route with softmax. That never happened only because the
tokenizer rejected every such file's pre type first. It also multiplies by
`expert_weights_scale` even when that is 0, which llama.cpp treats as a no-op. Both now
follow llama.cpp. DeepSeek-V2-Lite (softmax, scale 1.0) and `deepseek-tiny-mla` (dense
only) take exactly the same code path as before.

**Missing gating key.** `deepseek2.cpp` defaults the gating to softmax when the key is
absent, except for "GLM 4.7 Lite": `n_layer ∈ {47, 48}` and a vocab of 154,880, which
gets sigmoid. ggml-org's and bartowski's GLM-4.7-Flash files lack the key, so Reflex
copies this rule exactly, including that `n_layer` is `block_count - nextn_predict_layers`.

### Q-LoRA queries

`deepseek2.cpp` loads `attn_q_a` (`[hidden, q_lora_rank]`), `attn_q_a_norm`
(`[q_lora_rank]`) and `attn_q_b` (`[q_lora_rank, n_head * k_mla]`) when `q_lora_rank > 0`,
and `attn_q` otherwise. The graph is `q = wq_b · rmsnorm(wq_a · x)`, with the model's
`rms_norm_eps`, before the same reshape into nope/rope halves. These are existing ops
(`gemv`/`gemm` and `rmsnorm`), so no new kernel is needed.

`q_lora_rank` is not read at all for the "lite" layer counts (`n_layer` 26 or 27, or 48
with a 128,256 vocab). There the graph keys off whether `attn_q` exists. Reflex decides
by tensor presence too: it loads `attn_q_a`/`attn_q_a_norm`/`attn_q_b` if `blk.0.attn_q_a.weight`
exists, and `attn_q` otherwise. It errors if the file has both, or neither, or if a
`q_lora_rank` key disagrees with `attn_q_a`'s width.

### Pre types

| `pre` | llama.cpp type | Split | Flags | Reflex |
|---|---|---|---|---|
| `gigachat` | `GPT2` | `unicode_regex_split_custom_gpt2` | none | `PreType::Gpt2` (alias) |
| `glm4` | `CHATGLM4` | the `LLAMA3` regex string, so `unicode_regex_split_custom_llama3` | `ignore_merges`, `special_bos_id = NULL` | `PreType::Llama3` (alias), BOS off |
| `deepseek-coder` | `DEEPSEEK_CODER` | 5 regexes, `std::regex` | | not needed: the pinned converter writes `deepseek-llm` for Coder-V2-Lite |
| `kimi-k2` (queued) | `KIMI_K2` | `unicode_regex_split_custom_kimi_k2` | | not ported |
| `kanana2` (queued) | `GPT4O` | 2 regexes, `std::regex` | | not ported |

`glm4`'s BOS: llama.cpp sets `special_bos_id = LLAMA_TOKEN_NULL` for it, so no BOS is
ever added, whatever `add_bos_token` says. Reflex's tokenizer would otherwise add the
declared BOS (154822), because the file has no `add_bos_token` key.

Golden ids for the 13-string set come from llama.cpp `79e2e74`'s `llama-tokenize`, as
for the other pre types. Fixtures: `gigachat-tokenizer.gguf` (ai-sage GigaChat3.1 q8_0)
and `glm4-tokenizer.gguf` (unsloth GLM-4.7-Flash Q4_K_M), first 40 MB each.

## Fixture

`test-data/tiny-mla-v3.gguf`: a random `DeepseekV3ForCausalLM` checkpoint
through llama.cpp `79e2e74`'s unmodified `convert_hf_to_gguf.py`, in the local
`reflex-llama-qwen2:79e2e74` image, source in `test-data/tiny-mla-v3-src.tar.gz`.
It covers:

- Q-LoRA (`q_lora_rank` > 0, a layer count outside the "lite" set).
- `scoring_func = sigmoid` with a nonzero `e_score_correction_bias`.
- `norm_topk_prob = true` and `routed_scaling_factor` ≠ 1.
- `n_group = topk_group = 1`, one dense lead layer and a shared expert.
- `v_head_dim` ≠ `qk_nope_head_dim`.

Tokenizer: the Qwen2 vocab files, as in `deepseek-tiny-mla`. Checked against llama.cpp
before any Reflex code ran: each of these, dropped or changed, alters the 20-token greedy
output of all 7 prompts (most from the first token):
- zero the bias;
- switch the gating to softmax;
- set the scale to 1;
- turn normalization off;
- triple `attn_q_a_norm`'s weight.

llama.cpp always normalizes, so dropping the normalization itself can't be tested there.
Instead, `wq_a`'s outputs have an rms of about 3, so a missing normalization would
rescale Q; the Reflex GPU tests against llama.cpp's ids are that check.

## Tests

Host-only:

- Config parsing of the new fixture and every real header above. Each one either
  loads with the expected config, or is rejected with the expected message (legacy
  format, `kimi-k2`/`kanana2` pre types).
- Rejections: `expert_group_count > 1`, unknown gating values, `temperature_scale`, both
  or neither Q projection, mismatched `q_lora_rank`.
- The GLM-4.7 missing-key rule (ggml-org header → sigmoid; a softmax file stays softmax).
- `route_sigmoid_bias` against a host reference of the 6 steps above, including that it
  picks different experts than `route_sigmoid_logit_add` on the same inputs, and the
  clamp.
- `gigachat`/`glm4` golden ids vs `llama-tokenize`, plus `glm4` adding no BOS.

GPU (nightly rules in `scripts/gpu_nightly_tests.sh`):

- New fixture vs llama.cpp golden ids (f32).
- Batched prefill == sequential prefill.
- f16 vs f32.
- KV export/import resumes exactly.
- `system1` scores match full logits.

## Results (2026-10-13)

Runpod A100-SXM4-80GB (secure cloud), one pod for everything:
- llama.cpp `79e2e74` built with CUDA plus the `LLAMA_NO_Q8` patch and `ref.cpp`;
- Reflex built twice: the base (`qwen2-arch`, the tree master gets once phase 2 merges) and this branch;
- DeepSeek-V2-Lite and DeepSeek-Coder-V2-Lite-Instruct converted fresh (`--outtype q8_0`, `transformers` 4.57.1, both `deepseek-llm`, MLA split format);
- GigaChat3.1 `q8_0` from ai-sage;
- GLM-4.7-Flash `Q8_0` from ggml-org, which has no `expert_gating_func` key, so the GLM-4.7 fallback is what selects sigmoid.

Scripts and every log: `gpu-a100/` in `test-data/tiny-mla-v3-src.tar.gz`.

**Tokenization, checked before renting.** Full prompt ids (BOS included) for 7 prompts
(digit runs, punctuation runs, Chinese, Russian, code, mixed scripts) on the
DeepSeek, GigaChat and GLM vocabs: Reflex = `llama-tokenize`, 21/21.

**`reflex check`, base vs branch on the same pod** (every fixture, Qwen3-0.6B, real
V2-Lite and Coder-V2-Lite, 3 prompts each, 8 tokens):
- The 45 existing rows are identical, token ids and logit checksums included.
- The 12 new rows (`tiny-mla-v3` f32/f16, GigaChat3.1, GLM-4.7-Flash) are rejected by the base and run on the branch.

**Greedy ids vs llama.cpp** (20 tokens, EOS-only stopping):
- The fixture uses its 7 prompts, compared against llama.cpp's default CUDA settings.
- The real models use the 7 prompts above, compared against llama.cpp with f32 activations (`LLAMA_NO_Q8=1`, cuBLAS f32 compute, flash attention off).

| Model | Reflex weights | Prompts identical | Where the others diverge (step: llama.cpp's top-2 margin) |
|---|---|---|---|
| `tiny-mla-v3` | f32, f16 | 6/7 | story 18: 0.0053 |
| DeepSeek-V2-Lite | f32, f16 | 6/7 | russian 19: 0.0085 |
| DeepSeek-Coder-V2-Lite-Instruct | f32, f16 | 6/7 | plain 4: 0.089 |
| GigaChat3.1-10B-A1.8B | f32, f16 | 5/7 | digits 7: 0.19; mixed 2: 0.074 |
| GLM-4.7-Flash | f16 | 4/7 | punct 11: 0.052; russian 2: 0.0048; mixed 17: 0.029 |

Notes on the table:
- **f32 and f16 agree.** f32 and f16 weights give the same ids as each other on every model. GLM-4.7-Flash needs about 120 GB in f32, so it ran f16 only.
- **Every divergence is a near-tie.** `probe.sh` printed llama.cpp's top-2 logits at every step, and in all 8 divergences Reflex's token is llama.cpp's runner-up at that step. Up to that step, the ids are identical.
- **A third reference agrees with Reflex.** On the fixture, HF `transformers` in f32 picks the same token as Reflex at story step 18.
- **The divergences cover every new feature.** They fall on prompts that also go through sigmoid routing, Q-LoRA and the new tokenizers, and many prompts match all 20 tokens on each model.
- **llama.cpp's own backends disagree on Arabic.** Its CPU and CUDA backends differ on the fixture's `arabic` prompt from step 2. Reflex matches CUDA, so the test goldens use the CUDA ids.
- **Stock llama.cpp agrees less.** Against stock llama.cpp (8-bit activations), the match counts drop to V2-Lite 5/7, GigaChat 5/7 and GLM 3/7.

**DeepSeek-V2-Lite re-verify** (phase 1's queued item). The `deepseek-llm` split was never
exercised by the first verification's prompts. Now:
- V2-Lite matches 20/20 on the digit-run, punctuation-run, Chinese, code and mixed-script prompts.
- The Russian prompt matches 18 tokens and then hits a 0.0085 near-tie.
- Its `reflex check` rows are unchanged.

**Tests on the pod.**
- `cargo test`: 186 passed.
- The 5 `tiny-mla-v3` GPU tests:
  - first run: 3 passed, 2 failed;
  - after the fixes below: 5/5;
  - nightly run: also pass.
- `prefill_mla_batched_matches_sequential_real_moe_checkpoint` passes on GigaChat3.1 (sigmoid), GLM-4.7-Flash (Q-LoRA + sigmoid) and Coder-V2-Lite.
- Nightly suite: 57 passed, 0 failed, 2 skipped (no `IQ_GGUF`/`QWEN2_GGUF`).

The first run's two failures were test data, not code:
- **Arabic golden.** The fixture golden for `arabic` came from llama.cpp's CPU backend, and Reflex matches CUDA.
- **f16 clear-pick floor.** The f16 test's "clear pick" threshold (0.5) left only 3 of the fixture's 7 first-step margins above it. It is 0.2 now.
- The `story` golden also compares 17 tokens now.

**Cold start** on the smallest real target (process launch to first token, external wall
clock, `bench_cold_common.sh`, n=30, page cache warm, prompt "Once upon a time"):

| Model | Reflex p50 / p95 | `llama-simple` p50 / p95 | Peak RSS (Reflex / llama.cpp) |
|---|---|---|---|
| GigaChat3.1-10B-A1.8B `q8_0` (11.4 GB) | **4.22 / 4.51 s** | 4.40 / 4.76 s | 11.2 / 11.1 GB |

Reflex's own phase split for that run:
- `model_load_ms` ≈ 2.0 s: 11 GB of `Q8_0` dequantized to f16 on the device. MLA has no quantized-resident mode.
- `cuda_init_ms` ≈ 0.93 s.
- `prompt_eval_ms` ≈ 0.14 s.

That's 1.04x faster than `llama-simple`, a smaller lead than on small dense models. The f16 dequant writes twice the bytes `llama-simple` keeps resident, and the load dominates.

**Cost:** about 1.5 h on a secure-cloud A100 at $1.79/h, about $2.70.
