# Design: Kolibri-1 MoE support

**Status: proposal (2026-10-06). Nothing implemented. Phase 0 is a go/no-go gate.**

Goal: run the Kolibri-1 Mixture-of-Experts model (reported as 78B total / ~3.46B active
parameters, 384 routed experts + 1 shared expert, top-6 routing, a hybrid 4:1
sliding-window-attention (512 tokens) / NoPE layer pattern, and a "UniBPE" tokenizer) on
Reflex, measured on cold start (process launch to first token).

Every architectural fact in that sentence is unverified. Phase 0 exists to replace each
one with a value read from a real GGUF/HF checkpoint before any code is written.

## Constraints that shape this plan

- **Weights don't fit unless they stay quantized.** Matrix weights are dequantized to
  `f16` at load by default: ~156 GB for 78B parameters, more than an 80 GB A100.
  `REFLEX_QUANT_RESIDENT=1` keeps Q4_K weights quantized on the GPU, but only on the
  dense path (see [quantized-resident-weights.md](quantized-resident-weights.md)); MoE
  keeps normal storage. Extending it to stacked expert tensors is the core of this work.
- **Single GPU only.** Every entry point uses `CudaDevice::new(0)`. Multi-GPU sharding
  is out of scope here and would need its own proposal.
- **No FP8 path, and quantized-resident covers Q4_K/Q6_K only.** Targets that assume FP8
  or Q5 resident weights are out of scope.
- **Correctness means byte-exact greedy agreement with llama.cpp** (`reflex check`), not
  agreement with a vendor's own reference stack.
- **Cold start is the metric.** Sustained tokens/s is reported, not optimized (see the
  Non-goals in [DEVELOPMENT.md](../DEVELOPMENT.md)).
- Routing already runs on the host (`crate::moe::route_top_k`): a top-k over 384 logits
  costs microseconds, so no new routing kernel is needed.

## Phase 0: confirm the model and its conventions (go/no-go)

- Get the real HF `config.json`/tokenizer files and a real GGUF header. Record
  `general.architecture` and every tensor name and shape. A header can be checked
  without downloading the whole file (fetch the first few MB and parse with
  `src/gguf.rs`).
- **Check whether llama.cpp supports the architecture.** If it does, we get a GGUF
  format, `convert_hf_to_gguf.py` and a reference implementation. If it doesn't, all
  three are missing and the project becomes "build a converter and a reference first":
  stop and re-plan.
- Record from real metadata, not from model cards:
  - tokenizer type (`tokenizer.ggml.model`) and pre-tokenizer (`tokenizer.ggml.pre`);
  - sliding-window size and which layers use it (metadata key and layer pattern);
  - which layers skip RoPE (NoPE), and any per-layer attention scaling on them;
  - RoPE type, NORM vs NEOX (`rope_type_for` in `src/model/config.rs`; getting this
    wrong silently corrupts output);
  - router: softmax vs sigmoid, top-k weight renormalization or not;
  - shared expert: ungated (MLA's `MlaFfn::Moe`) or sigmoid-gated (`*_shexp` in
    `src/model/hybrid.rs`).
- Output: fill in a "Confirmed conventions" section of this document, with sources.

## Phase 1: config and tokenizer

- `src/model/config.rs`: accept the architecture string, parse the sliding-window and
  layer-pattern keys, and add an explicit `rope_type_for` entry with a test in
  `src/model/rope_type_tests.rs`.
- `src/tokenizer.rs`: if `tokenizer.ggml.model` is a new value, add it alongside
  `llama` and `gpt2`. Test against llama.cpp's own tokenizer output on the same strings,
  including German compound nouns and non-ASCII byte fallback.
- Build a small synthetic fixture with llama.cpp's unmodified `convert_hf_to_gguf.py`
  (random weights, real tensor names, a few layers, ~16 experts with top-k below the
  expert count, at least one SWA layer and one NoPE layer), the same way the MLA and
  qwen3moe fixtures were built (see [DEVELOPMENT.md](../DEVELOPMENT.md)'s
  test-fixture section). Only possible if Phase 0 found converter support.

## Phase 2: sliding-window attention and NoPE

- **Correctness first, via masking:** keep the existing full-length KV cache and mask
  keys older than `pos - window` in `attention.cu`, `attention_online.cu` and
  `attention_prefill.cu`. That is what TTFT on a prompt depends on.
- NoPE: skip the RoPE launch on the configured layers. Add per-layer attention scaling
  only if Phase 0 found one.
- **Ring-buffer KV cache: deferred.** It only saves memory on long contexts and is not
  on the cold-start path. If it lands later it needs a new cache-shape version in
  `src/kv_io.rs`, since `--export-kv`/`--import-kv` depend on the layout.

## Phase 3: 384-expert MoE with quantized-resident experts (core work)

- Routing: reuse `route_top_k` or `route_top_k_with_norm` per Phase 0; add a sigmoid
  variant only if needed.
- Shared expert: reuse whichever existing convention matches, run unconditionally next
  to the routed experts.
- **Extend `REFLEX_QUANT_RESIDENT` to MoE expert tensors.** Keep Q4_K expert slices of
  the stacked `[in, out, expert_count]` tensors as raw blocks in the device arena and
  read them per expert with `gemv_q4k`: a sliced variant of `gemv_expert`. Q4_K_M at
  78B is ~47 GB, which fits on an 80 GB A100.
- Batched prefill: make the batched MoE prefill path handle 384 experts and quantized
  experts; `prefill_dense_batched_matches_sequential_prefill` remains the guard test.
- Optional, cold-start specific: **lazy expert upload.** A short prompt touches only a
  fraction of 384 experts per layer. Measure the touched fraction before committing.

## Phase 4: verification and benchmarks

- Correctness: byte-exact greedy agreement with llama.cpp via `reflex check`, first on
  the synthetic fixture, then on the real Q4_K_M GGUF. Use German and English prompts,
  including ones longer than 512 tokens so the window is exercised.
- Hardware: 1x A100 80 GB, Q4_K_M, `REFLEX_QUANT_RESIDENT=1`.
- Metrics: `process_start_to_first_token_ms`, the `model_load_ms` breakdown
  (`REFLEX_LOAD_PROFILE=1`; at ~47 GB, load will likely dominate), energy, and
  llama.cpp's cold-start numbers on the same machine for comparison. Report tokens/s
  without optimizing for it.
- Docs: add the architecture to README's supported models and to
  [DEVELOPMENT.md](../DEVELOPMENT.md).

## Out of scope (separate proposals if wanted)

- Multi-GPU sharding (e.g. 2x RTX 4090).
- FP8 weights, Q5_K quantized-resident weights.
- Ring-buffer KV cache.

## Risks

- **No llama.cpp support** means no GGUF and no reference. This is the gate.
- **Load time.** Reading ~47 GB from disk may dominate cold start; lazy expert upload
  is the main lever.
- **Silent convention mismatches** (RoPE type, router normalization, shared-expert
  gating) produce wrong output without crashing, as happened during MLA bring-up. Each
  must be confirmed in Phase 0 and covered by byte-exact comparison.
