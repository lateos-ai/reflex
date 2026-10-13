# Design: Qwen2 / Qwen2.5 support (`qwen2`)

**Status (2026-10-13): done and verified on an RTX 3090.** Greedy tokens match llama.cpp
`79e2e74` on the synthetic fixture (7/7 prompts × 20 tokens, with and without a LoRA)
and on real Qwen2.5-0.5B-Instruct, 1.5B-Instruct, Coder-1.5B-Instruct and
Math-1.5B-Instruct (7/7 each against llama.cpp with f32 activations; `f16` and
quantized-resident weights too). Cold start on Qwen2.5-0.5B is 0.650 s p50 vs
`llama-simple`'s 0.700 s. See [Results](#results-2026-10-13).

Goal: run the dense `qwen2` architecture: Qwen2.5-Instruct (0.5B to 72B), Qwen2.5-Math
and Qwen2.5-Coder. Correctness means byte-exact greedy agreement with llama.cpp, as for
every other architecture.

## Scope

In: dense `qwen2` text generation, `system1`, `--weights f16|f32`,
`REFLEX_QUANT_RESIDENT=1`, `--lora`, `--export-kv`/`--import-kv`. All of them go
through the existing dense path (`LayerWeights::Dense`). The only new math is the Q/K/V
bias add.

Out, each rejected at load with a clear error:

- `rope.scaling.type` other than `none` (Qwen2.5's optional YaRN for long context). The
  dense path already rejects it for every architecture.
- Tensors llama.cpp's `qwen2` graph would apply but Reflex doesn't implement:
  `attn_output.bias`, `output.bias`, and the fused `attn_qkv` weight or bias. No Qwen2.5
  GGUF surveyed has any of them. The check covers the whole dense/MoE path, so a
  `qwen3` or `llama` file with one of them is rejected too, instead of silently running
  without it.
- `qwen2moe` (Qwen1.5-MoE, Qwen2-57B-A14B). It reports a nonzero `expert_count`, so it
  used to pass the generic MoE check and would have run with its shared expert ignored.
  The MoE path now rejects any file with shared-expert tensors (`ffn_*_shexp`), naming
  them. Kolibri-1 keeps its own path.
- `qwen2vl` and Qwen2.5-VL (`qwen2vl` architecture, M-RoPE): a different architecture
  string, already rejected by the whitelist.

## Reference implementations

- llama.cpp `79e2e74`:
  - `src/models/qwen2.cpp`: tensors and graph.
  - `llama_model_base::create_tensor_qkv` (`src/llama-model.cpp`) and
    `llm_graph_context::build_qkv` (`src/llama-graph.cpp`): how the optional biases
    load and where they're added.
  - `llama_model_rope_type` (`src/llama-model.cpp`): RoPE convention.
  - `unicode_regex_split_custom_qwen2` (`src/unicode.cpp`) and the EOG token detection
    in `src/llama-vocab.cpp`: tokenizer.
  - `conversion/qwen.py` (`Qwen2Model`): the converter used for the fixture.
- GGUF headers (HTTP range request on the first 40 MB of each file, without downloading
  the rest):
  - `Qwen/Qwen2.5-0.5B-Instruct-GGUF` `qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - `Qwen/Qwen2.5-1.5B-Instruct-GGUF` `qwen2.5-1.5b-instruct-q4_k_m.gguf`
  - `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF` `Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf`
    (Qwen publishes no Math GGUF)
  - `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
  - `ggml-org/Qwen2.5-Coder-1.5B-Instruct-Q8_0-GGUF` `qwen2.5-coder-1.5b-instruct-q8_0.gguf`

  Each is kept as a header fixture in `test-data/` (metadata and tensor table, no
  tensor data; `scripts/gguf_tokenizer_only.py --keep-tensor-table`).

## Header survey

All five files report `general.architecture = qwen2` and `tokenizer.ggml.pre = qwen2`.

| | 0.5B-Instruct | 1.5B-Instruct | Math-1.5B-Instruct | Coder-1.5B-Instruct |
|---|---|---|---|---|
| `block_count` | 24 | 28 | 28 | 28 |
| `embedding_length` | 896 | 1536 | 1536 | 1536 |
| `feed_forward_length` | 4864 | 8960 | 8960 | 8960 |
| `attention.head_count` / `head_count_kv` | 14 / 2 | 12 / 2 | 12 / 2 | 12 / 2 |
| head dim (`embedding_length / head_count`) | 64 | 128 | 128 | 128 |
| `rope.freq_base` | 1e6 | 1e6 | **1e4** | 1e6 |
| `context_length` | 32768 | 32768 | 4096 | 32768 |
| `attention.layer_norm_rms_epsilon` | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| LM head | `output.weight` (Q8_0) | `output.weight` (Q6_K) | tied (no `output.weight`) | `output.weight` (Q6_K); tied in ggml-org's Q8_0 |
| Matmul types | Q5_0, Q8_0, Q6_K | Q4_K, Q6_K | Q4_K, Q6_K | Q4_K, Q6_K |

Keys present in every file: `block_count`, `context_length`, `embedding_length`,
`feed_forward_length`, `attention.head_count`, `attention.head_count_kv`,
`rope.freq_base`, `attention.layer_norm_rms_epsilon`. None has
`attention.key_length`, `rope.dimension_count` or any `rope.scaling.*` key.

Per-layer tensors, identical in every file: `attn_norm`, `attn_q`/`attn_k`/`attn_v`
(weight **and** F32 `bias`), `attn_output` (weight only), `ffn_norm`,
`ffn_gate`/`ffn_up`/`ffn_down`. No `attn_q_norm`/`attn_k_norm`, no `attn_output.bias`,
no fused `attn_qkv`.

The 0.5B is all Q5_0/Q8_0 even in the "Q4_K_M" file: 896 isn't a multiple of 256, so
llama.cpp's quantizer falls back from the K-quants. `REFLEX_QUANT_RESIDENT=1` (Q4_K
only on the dense path) therefore changes nothing for it; the 1.5B files are the ones it
applies to.

Tokenizer, identical in all five: `gpt2` model, 151,936 tokens, 151,387 merges,
`bos_token_id = 151643` with `add_bos_token = false`, `eos_token_id = 151645`
(`<|im_end|>`). CONTROL tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, the
object/box/quad/vision markers, and the six FIM/repo tokens `<|fim_prefix|>`,
`<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`.
USER_DEFINED: `<tool_call>`, `</tool_call>`. 271 `[PAD…]` tokens are UNUSED.

## What differs from the dense Qwen3 path

| | Qwen3 (`qwen3`) | Qwen2/2.5 (`qwen2`) | Source |
|---|---|---|---|
| Q/K/V bias | none | `attn_{q,k,v}.bias`, added right after each projection, before RoPE | `create_tensor_qkv`, `build_qkv` |
| O bias | none | none (`wo_b` is optional in `build_attn`; no file has it) | `qwen2.cpp`, survey |
| QK-Norm | `attn_q_norm`/`attn_k_norm` | none | `qwen2.cpp`, survey |
| Head dim | `attention.key_length` (decoupled) | `embedding_length / head_count` (no key) | `llama-model.cpp` default, survey |
| RoPE | NEOX | NEOX, full head dim | `llama_model_rope_type` lists `LLM_ARCH_QWEN2` with the NEOX group; `qwen2.cpp` asserts `n_embd_head == n_rot` |
| Attention scale | 1/sqrt(head_dim) | 1/sqrt(head_dim) | `qwen2.cpp` |
| FFN | SwiGLU | SwiGLU, no biases | `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)` with NULL biases |
| LM head | tied or `output.weight` | tied or `output.weight`; optional `output.bias` | `qwen2.cpp` |
| Tokenizer `pre` | `qwen2` | `qwen2` | survey |

No hparam needs a new default. `rope.freq_base` and the RMS epsilon are always present
(the existing fallbacks, 10000 and 1e-5, stay unused). The Math model's 1e4 base is read
from its file like the others' 1e6.

So `qwen2` is Qwen3's dense path without QK-Norm (already optional: presence of the
tensor gates it) plus the Q/K/V bias.

## Approach

- `parse_model_config` whitelists `qwen2`. `rope_type_for` gets an explicit
  `"qwen2" => Neox` arm, so the mapping stays auditable in one place.
- `parse_model_config` rejects the unimplemented tensors listed under Scope, by
  checking layer 0 and the LM head, so a file fails at config time with the tensor
  named, before any weight upload.
- `DenseLayerWeights` and `MoeLayerWeights` get `attn_bias: Option<QkvBias>`, where
  `QkvBias` holds the three F32 bias vectors. It's all three or none: a layer with only
  some of them is rejected at load, naming the missing one. The lengths are checked
  against the projection widths.
- `forward_attn_block` and `forward_attn_block_batched` take `Option<&QkvBias>` and add
  it right after the Q/K/V projections, before QK-Norm and RoPE, which is where
  `build_qkv` adds it. Single-row: `add_inplace`. Batched prefill: the same `add_kernel`,
  now with a broadcast length: `a[i] += b[i % m]`, with `m == n` for every existing
  caller (a uniform branch skips the modulo there, so their results are bit-identical).
  No new kernel.
- Biases load through `load_weight_kind(name, false)`, so they stay f32 under
  `--weights f16` and `REFLEX_QUANT_RESIDENT=1`, like norms.
- Kolibri-1 passes `None`; its file has no biases.

Unchanged and reused as-is: tokenizer (`qwen2` pre type, now table-based; see
[tokenizer pre-types](tokenizer-pre-types.md#resolved-the-original-qwen2-port)),
special-token matching, `LmHead` (tied or not), LoRA targets (`attn_q` etc. map to the
same `DenseLayerWeights` fields), the version-1 dense KV-cache format, `system1`.

## Known differences, not fixed here

- **End-of-generation set.** Reflex stops only at `eos_token_id` (`<|im_end|>`).
  llama.cpp also stops at `<|endoftext|>`, `<|fim_pad|>`, `<|repo_name|>` and
  `<|file_sep|>` (its EOG detection by token text). A FIM completion on Coder usually
  ends with `<|endoftext|>`, so Reflex runs on to `--max-tokens` where llama.cpp stops.
  The greedy tokens up to that point are the same. This applies to every gpt2-vocab
  architecture (Qwen3 has the same tokens), so it's queued as its own item rather than
  changed for one architecture here. Comparisons use EOS-only stopping on both sides,
  as for Gemma 4.
- **NFC normalization.** Qwen2.5's HF `tokenizer.json` has an NFC normalizer, which
  reorders combining marks into canonical order (Arabic shadda + fatha becomes
  fatha + shadda). A GGUF doesn't carry the normalizer: llama.cpp tokenizes the raw text,
  and so does Reflex. On text that isn't already NFC, HF `transformers` therefore gives
  different ids from both (seen on the fixture's Arabic prompt: two tokens swap). Kolibri-1's
  tokenizer has no normalizer, which is why its HF golden test agrees on the same text.
- **Special-token partition order.** Reflex scans left to right and takes the longest
  special token at each position. llama.cpp partitions by each special token in turn,
  longest first. These only differ when special tokens overlap in the text, which the
  Qwen2.5 set can't do (each is a distinct `<|...|>` marker).

## Fixture

`test-data/tiny-qwen2.gguf`: a random-weight `Qwen2ForCausalLM` checkpoint (2 layers,
hidden 64, 4 query heads and 2 KV heads of dim 16, FFN 128, tied embeddings, the real
Qwen2.5-0.5B-Instruct tokenizer files) converted by llama.cpp `79e2e74`'s unmodified
`convert_hf_to_gguf.py --outtype f32`. Every bias is nonzero and large relative to the
projections, so a missing or misplaced bias add can't still pass. Source:
`test-data/tiny-qwen2-src.tar.gz`.

## Tests

Host-only:
- `qwen2` config from the fixture and the five real headers: head dim, RoPE type and
  base, biases present, tied vs separate LM head.
- Rejections: `attn_output.bias`, `output.bias`, fused `attn_qkv`, a partial bias set,
  a wrong bias length, `rope.scaling.type = yarn`, `qwen2moe`-style shared experts.
- Tokenizer golden vs `llama-tokenize` on the 0.5B header: chat, math and FIM strings.

GPU (`#[ignore]`, nightly rules in `scripts/gpu_nightly_tests.sh`):
- fixture generates; greedy tokens equal llama.cpp's
- batched prefill == sequential prefill
- f16 vs f32 weights on the fixture
- `REFLEX_QUANT_RESIDENT=1` vs dequantized on a real Q4_K `qwen2` file
- `--export-kv`/`--import-kv` resume == uninterrupted run
- LoRA on the fixture

## Results (2026-10-13)

Runpod RTX 3090 (community, driver 580.65.06), one pod for everything: llama.cpp `79e2e74`
built with CUDA plus the `LLAMA_NO_Q8` patch and `ref.cpp`, and Reflex master and this
branch built side by side. Prompts: plain, story, ChatML, Qwen2.5-Math chain-of-thought,
Coder FIM, Arabic with harakat, and a 980-token prompt (`prompts.py` in
`test-data/tiny-qwen2-src.tar.gz`). "f32 reference" means llama.cpp with
`LLAMA_NO_Q8=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_DISABLE_GRAPHS=1
GGML_CUDA_DISABLE_FUSION=1` and flash attention off. Greedy ids, 20 tokens, EOS-only
stopping on both sides:

| Model | Reflex weights | vs | Prompts identical |
|---|---|---|---|
| `tiny-qwen2` fixture | f32 | llama.cpp default and without flash attention | 7/7, 7/7 |
| `tiny-qwen2` + LoRA (`reflex generate --lora`) | f32 | llama.cpp with the same adapter (`REF_LORA`) | 7/7 |
| Qwen2.5-0.5B-Instruct Q4_K_M | f32 | f32 reference | 7/7 |
| Qwen2.5-0.5B-Instruct Q4_K_M | f16 | f32 reference | 7/7 |
| Qwen2.5-0.5B-Instruct Q4_K_M | f32 | stock llama.cpp (8-bit activations) | 4/7; the others diverge after 4, 7 and 19 tokens |
| Qwen2.5-1.5B-Instruct Q4_K_M | f32 | f32 reference | 7/7 |
| Qwen2.5-1.5B-Instruct Q4_K_M | f16, `REFLEX_QUANT_RESIDENT=1` | f32 reference | 7/7 |
| Qwen2.5-Coder-1.5B-Instruct Q4_K_M | f32 | f32 reference | 7/7 |
| Qwen2.5-Math-1.5B-Instruct Q4_K_M (bartowski) | f32 | f32 reference | 7/7 |

As with Gemma 4, stock llama.cpp's 8-bit activation quantization moves the 0.5B's logits
enough to flip some greedy picks; with f32 activations every row matches.

Tests on the same pod: the 7 `qwen2` GPU tests pass (fixture vs llama.cpp with and
without the LoRA, batched prefill == sequential in f32 and f16, f16 within 3.0e-3
relative L2 of f32 on first-step logits, KV export/import resume exact, and
quantized-resident + batched prefill on the real 1.5B). The nightly suite:
52 passed, 0 failed, 2 skipped (no `IQ_GGUF`/`MLA_REAL_GGUF`). Full `cargo test`: 167
passed.

**A regression the GPU run caught.** Giving `add_kernel` a fourth argument (the
broadcast length) missed one caller: `Model::apply_lora` launches it directly. With three
arguments the kernel read garbage for `m`, and every `--lora` run segfaulted, `qwen3moe`
included. The host build and every host test passed. Fixed (`m = n`); after the fix
`qwen3moe` + LoRA output is identical to master's and `qwen2` + LoRA matches llama.cpp.

**LoRA golden ids: llama.cpp CPU vs CUDA.** The fixture's goldens were first recorded with
llama.cpp's CPU backend. Without the LoRA, CPU and CUDA give identical ids. With it, the CPU
backend picks a different token at step 8 of the `story` prompt (the fixture's narrowest
margins). A third reference settled it: HF `transformers` in f32 with the LoRA merged by
hand (`hf_ref.py` in the archive) agrees with CUDA there and on every prompt except
`arabic`, where HF already differs without the LoRA (the NFC difference above). The tests
use the CUDA ids.

**`reflex check` before/after** (every fixture plus Qwen3-0.6B, 3 prompts, 8 tokens, master
vs branch on the same pod): the 30 existing rows are identical, token ids and logit
checksums included, as the host-side tokenization diff predicted. The 12 new rows
(`tiny-qwen2` f32/f16, Qwen2.5-0.5B, Qwen2.5-1.5B quantized-resident) are rejected by
master and run on the branch.

**Cold start** (process launch to first token, external wall clock, `bench_cold_common.sh`,
n=30, prompt "Once upon a time", same method as the 2026-09-29 T4 refresh):

| Model | Reflex p50 / p95 | `llama-simple` p50 / p95 | Peak RSS (Reflex / llama.cpp) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | **0.650 / 0.680 s** | 0.700 / 0.750 s | 993 / 731 MB |
| Qwen3-0.6B Q4_K_M (same pod) | 0.685 / 0.770 s | 0.695 / 0.750 s | 896 / 640 MB |

Reflex is 1.08x faster on Qwen2.5-0.5B and at parity on Qwen3-0.6B on this pod (the T4
refresh measured 1.13x for Qwen3-0.6B). Qwen2.5-0.5B's separate `output.weight` skips the
tied-embedding dequant that `generate` pays on Qwen3-0.6B (prompt eval 10.7 vs 177 ms).
As in the earlier refresh, `llama-simple` takes `--no-warmup` as part of its prompt.
