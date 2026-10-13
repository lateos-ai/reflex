# Changelog

What has been built, in order. Measurements live in
[docs/benchmarks.md](docs/benchmarks.md).

## 2026-10-13

- **Qwen2 / Qwen2.5 (`qwen2`), including Qwen2.5-Math and Qwen2.5-Coder.** These run on
  the dense path. The new piece is Qwen2's Q/K/V projection biases, added right after
  each projection and before RoPE, as llama.cpp's `build_qkv` does. Batched prefill
  adds them with the existing `add_kernel`, which now broadcasts one row over many.
  Residual adds don't change and give bit-identical results.
  - `--weights f16|f32`, `REFLEX_QUANT_RESIDENT=1`, `--lora`, `--export-kv`/`--import-kv`
    and `system1` all work, through the existing dense code.
  - ChatML, the Math prompt and Coder's FIM and repo tokens (`<|fim_prefix|>`,
    `<|file_sep|>`, ...) tokenize as single special tokens, matching `llama-tokenize`.
  - Rejected at load, with the tensor or key named: YaRN `rope.scaling`,
    `attn_output.bias`, `output.bias`, a fused `attn_qkv`, and a partial or misshapen
    bias set. The output-bias and fused-QKV checks cover every dense architecture.
    `qwen2moe` used to pass the generic MoE check and would have run without its shared
    expert. Any MoE file with shared-expert tensors is now rejected.
  - Tests: config parsing of five real Qwen2.5 headers (0.5B, 1.5B, Math, Coder ×2),
    rejections, and a synthetic `tiny-qwen2` fixture plus LoRA, both converted with
    llama.cpp's own converters, with greedy golden ids from llama.cpp.
  - Verified on an RTX 3090 against llama.cpp `79e2e74` with f32 activations: identical
    greedy tokens (7 prompts × 20 tokens) on the fixture with and without a LoRA, on
    Qwen2.5-0.5B-Instruct (`f32` and `f16`), Qwen2.5-1.5B-Instruct (also
    `REFLEX_QUANT_RESIDENT=1`), Coder-1.5B and Math-1.5B. Stock llama.cpp's 8-bit
    activations flip 3 of 7 on the 0.5B, as on Gemma 4.
  - Cold start on Qwen2.5-0.5B-Instruct: 0.650 s p50 vs `llama-simple`'s 0.700 s (n=30,
    same pod).
  - `reflex check` on every fixture plus Qwen3-0.6B is unchanged, 30/30 rows.

  Details: [docs/design/qwen2.md](docs/design/qwen2.md).

- **`qwen2` tokenizer split now uses llama.cpp's Unicode tables.** It was the last pre
  type still classifying characters with Rust's `is_alphabetic`/`is_numeric`, which
  count Arabic harakat as letters. It now runs llama.cpp's
  `unicode_regex_split_custom_qwen2` port, like every other pre type. Output changes only
  on text whose characters the two disagree on, such as combining marks.
  - Qwen2 golden test vs `llama-tokenize`: 12/13 -> 13/13.
  - Kolibri-1 (`kolibri1`, same split): unchanged against HF `tokenizers`, including two
    new Arabic and Devanagari strings.
  - The `reflex check` prompts tokenize identically on every fixture, and the GPU check
    shows all 30 rows unchanged.

  Details: [docs/design/tokenizer-pre-types.md](docs/design/tokenizer-pre-types.md#resolved-the-original-qwen2-port).

## 2026-10-12

- **Tokenizer pre-types: `tokenizer.ggml.pre` is now read and checked.** Before this,
  every `gpt2`-model GGUF got the Qwen2 split regex. A Llama-3 or DeepSeek file
  therefore loaded and tokenized wrongly, with no error.
  - New ports, each hand-ported from llama.cpp `79e2e74`: `llama-bpe` (Llama 3),
    `deepseek-llm` (DeepSeek LLM, DeepSeek-V2/V2-Lite), `deepseek-v3`, `qwen35`
    (Qwen3.5/3.8: letter runs also take combining marks) and `gpt-2`. They classify
    characters with llama.cpp's own Unicode tables, transcribed into
    `src/tokenizer_unicode.rs`. No new dependency.
  - Unchanged: `qwen2` and `kolibri1`.
  - Any other value fails to load, and the error names it. A missing key keeps the old
    Qwen2 behavior.
  - Each type matches `llama-tokenize` token for token on a 13-string tricky set. The
    tests use tokenizer-only fixtures, made by the new `scripts/gguf_tokenizer_only.py`.
  - `reflex check` on an RTX 3090 (every fixture plus Qwen3-0.6B) is unchanged on 29 of
    30 rows, logit checksums included. The 30th is `tiny-qwen3moe` (`gpt-2`) on the
    long prompt, whose prompt ids now match `llama-tokenize`.
  - Known gap, unchanged on purpose: the `qwen2` split still disagrees with llama.cpp on
    Arabic harakat.

  Details: [docs/design/tokenizer-pre-types.md](docs/design/tokenizer-pre-types.md).

- **Runpod Hub release `v0.2.6-runpod-hub`.** Rebuilds the Hub worker from current
  `master`, so the classifier gets the bit-identical `rmsnorm_kernel` fix below: faster
  prefill, same outputs. The image, the baked Qwen3-0.6B-Q8_0 model, `hub.json` and
  `handler.py` are unchanged. Gemma 4 support is in the engine but not reachable from
  the Hub worker, which serves only its baked model.

- **Long-prompt prefill 4-7x faster on Gemma 4, and faster prefill everywhere.** Two
  causes, measured with a new per-phase profiler (`REFLEX_PREFILL_PROFILE=1`):
  - `rmsnorm_kernel` recomputed each row's sum of squares in every thread (rows x
    hidden^2 work). It now runs one block per row with the same summation order, so
    outputs are bit-identical for every architecture (`reflex check` unchanged on 12
    configurations). This was 40-68% of Gemma 4 prefill. Qwen3-0.6B warm prefill at 449
    tokens: 67.5 -> 53.9 ms.
  - Gemma 4 prefill attention now uses a tiled f32 kernel that reuses each K/V tile
    across a block of query rows and the query heads of a GQA group. Decode and other
    architectures are unchanged.

  On an RTX A40, 12B prefill at 3,713 tokens: 13.4 s -> 1.87 s. 31B
  (`REFLEX_QUANT_RESIDENT=1`): 33.1 s -> 4.66 s. First token on a 2,641-token prompt:
  12B 10.1 s -> 3.4 s, 31B 23.0 s -> 6.1 s. That is faster than llama.cpp computing in
  f32 and 1.3-1.6x behind stock llama.cpp. Greedy tokens against the fully-f32 llama.cpp
  reference are unchanged: 12B 12/12, 31B 5/6 (the known 982-token near-tie). PTX builds
  without a warm driver cache JIT the new module once, ~340 ms; cubin and fatbin builds,
  including the Docker image, don't. Details:
  [Long-prompt prefill](docs/design/gemma4.md#long-prompt-prefill-2026-10-12).

- **Gemma 4 on 48 GB cards: quantized-resident `Q8_0`** (`REFLEX_QUANT_RESIDENT=1`,
  opt-in). Gemma 4's `Q8_0` layer weights and tied LM head stay quantized on the GPU,
  read by new `gemv_q8_0` kernels. Above 16 prompt rows they are dequantized into a
  reused f16 scratch for cuBLAS. Other architectures' outputs are unchanged (`reflex
  check` identical, checksums included). Verified on an RTX A40 against llama.cpp
  computing fully in f32:
  - **31B-it `Q8_0` now runs on a 48 GB card:** 37.9 GB peak at 2,641 tokens. Greedy
    tokens identical on 5/6 prompts up to 2,641 tokens. The sixth, a 982-token
    repeated-sentence prompt, starts with a near-tie that the llama.cpp reference resolves
    differently on an A40 and an A100; Reflex agrees with the A100 run. `system1`
    within 0.022.
  - **12B-it:** 11.1 GB resident instead of 20.9. 18/18 prompt and mode combinations
    identical to the reference, and `system1` scores closer to it than with f16 weights.
  - **Faster cold start, 12B:** model load 1.0 s instead of 1.5-2.3 s (the f16 path's 328
    per-tensor allocations cost 0.6-1.5 s). First token 1.99 s instead of 2.49 s wall
    (llama.cpp 3.85 s). Cold `system1` 1.38 s instead of 1.81 s. The 31B's first token
    is 3.6 s against llama.cpp's 8.4 s.
  - **Decode 22% faster** on the 12B (44.8 vs 57.7 ms/token).
  - **Slower:** warm prefill, by ~65 ms on long prompts, and warm 1-token `system1`
    (109 vs 73 ms). The fused kernel can't match f16 cuBLAS on tensor cores above ~4
    prompt rows.

  Details: [quantized-resident weights](docs/design/quantized-resident-weights.md#q8_0-on-gemma-4-results-2026-10-12).

## 2026-10-11

- **Gemma 4 31B and Qwen3.8-27B verified** on an A100 80 GB, both `Q8_0` with f16
  weights. No code changed.
  - **Gemma 4 31B-it:** greedy tokens identical to llama.cpp running fully in f32 on 5/5
    prompts (up to 1,262 tokens, past the 1,024-token window), and on a natural
    2,641-token prompt. `system1` scores within 0.03. For long prompts the reference
    needs an f32 KV cache with flash attention off on top of f32 activations: llama.cpp's
    f16 flash attention moves the 31B's logits by up to ~0.5 there.
  - **Qwen3.8-27B** (`qwen35`): 4/4 prompts identical to llama.cpp with f32 activations,
    including a 981-token prompt. The prompt ids match too, so the GGUF's
    `add_bos_token = false` is honored. Single-token `system1` scores within 0.007
    (the hybrid path still accepts only single-token labels).

  Gemma 4 details: [docs/design/gemma4.md](docs/design/gemma4.md#31b-results-2026-10-11).

## 2026-10-09

- **Gemma 4 support** (`gemma4`, dense, text only; verified on the 12B, the 31B is
  not run yet: its header parses, but it needs an 80 GB card in f16; verified
  2026-10-11, above). The new path is `src/model/gemma4.rs` with its own kernels (`kernels_cuda/gemma4.cu`), plus a third
  tokenizer mode (`tokenizer.ggml.model = "gemma4"`). Other architectures' code paths and
  kernels are unchanged; on an A40 their `reflex check` output is identical to before,
  checksums included. Verified on an A40:
  - **Synthetic fixture:** greedy tokens identical to llama.cpp, 20/20 on 4 prompts
    (one 982 tokens long against an 8-token window).
  - **Real Gemma 4 12B-it `Q8_0`:** 4/4 prompts identical to llama.cpp running with
    f32 activations, including a 1,122-token prompt past the 1,024-token window. Stock
    llama.cpp's 8-bit activation quantization moves some logits by up to ~0.8, so it
    disagrees on near-ties.
  - **`system1`:** multi-token labels supported, scores within 0.06 of llama.cpp's.
  - **Tokenizer:** identical ids on 25 strings to HF `tokenizers` and `llama-tokenize`.

  First token after 2.41 s vs llama.cpp's 3.49 s (`llama-simple`, file in page cache,
  n=5). As a classifier, about 74 ms per decision warm (`reflex stdio`) and about 2.1 s
  as a cold process. Details: [docs/design/gemma4.md](docs/design/gemma4.md).

- **Runpod Hub worker repackaged as a classifier listing** (`v0.2.5-runpod-hub`). The
  queue-based Hub worker now bakes **Qwen3-0.6B-Q8_0** (was Q4_K_M; `.runpod/Dockerfile`)
  and leads with `POST /v1/classify`: a job whose `input` has `labels` scores every label
  as a continuation of `prompt` in one prefill and returns per-label probabilities and
  entropy, generating nothing. Chat completions stay available on the same endpoint.
  `.runpod/README.md` gains a caller contract -- send `prompt` not `messages` for Qwen3's
  ` thinking` bias, use one-token labels with a leading space, and treat `probability` as
  relative to the label set so thresholds/escalation stay the caller's job -- and
  `tests.json` adds a 4-label routing smoke test. No engine change; `serverless/runpod/`
  (load-balancing) is untouched.

## 2026-10-07

- **Lazy expert upload for Kolibri-1** (opt-in `REFLEX_LAZY_EXPERTS=1`, with
  `REFLEX_QUANT_RESIDENT=1`): the load reserves the experts' GPU space but copies none
  of them; each layer uploads its routed experts the first time it needs them, so a
  short prompt reads a fraction of the 47.5 GB file before its first token. Output is
  identical to the eager load (tokens and System1 scores, verified on a T4 on the tiny
  Kolibri fixture, and on the real model). On an RTX A6000, a 5-token prompt's cold first
  token drops from 8.55 s to **2.33 s** (llama.cpp: 16.8 s on the same host) after
  uploading 4.9% of the experts; warm, 3.51 -> 1.21 s; a cold System1 decision 8.4 ->
  2.9 s. A 627-token prompt touches half the experts and gains only ~1 s, since each
  layer waits for its uploads. Details in
  [docs/design/kolibri.md](docs/design/kolibri.md), Phase 4 step 2.
- **`POST /v1/classify` in the OpenAI sidecar**: Reflex's System1 candidate scoring over
  HTTP. A request names a `prompt` (or chat `messages`) and up to 64 `labels`; the engine
  runs one prefill, scores each label as a continuation, and the response carries each
  label's probability within the set, the most probable label and the entropy. Nothing
  is generated. On a T4 with Qwen3-0.6B a warm request takes ~10 ms and returns exactly
  the probabilities `reflex system1` gives. A label's score sums raw token logits, so
  labels of different token lengths aren't comparable: the response carries a `warning`
  when they differ. On Kolibri-1 (RTX A6000), a warm decision takes ~0.2 s and a cold
  process reaches its result in ~8.6 s; bare zero-shot prompts got 2 of 3 test cases
  wrong there, so check a classifier on real data first
  ([docs/design/kolibri.md](docs/design/kolibri.md)). Not an OpenAI endpoint (OpenAI has no classification API). The Runpod
  Hub worker sends any job whose input has `labels` to it, and `tests.json` gains a
  classification smoke test.
- **Kolibri-1 support** (Aleph Alpha's 78B MoE, 3.46B active, 384 experts, top-6,
  released 2026-10-03 under Apache 2.0). A third layer variant on the dense/MoE path:
  sigmoid routing with a selection-only bias, sandwich norms, sliding-window RoPE layers
  alternating with full-attention NoPE layers. Its experts stay quantized on the GPU with
  `REFLEX_QUANT_RESIDENT=1` (new `Q6_K` kernels alongside `Q4_K`), so the 47.5 GB
  `Q4_K_M` runs in 45.4 GiB. On an A100 it matches llama.cpp (`836d571` plus the community
  `kolibri1` patch; mainline has no support yet) 20/20 tokens on five German and English
  prompts up to 627 tokens, once llama.cpp computes with f32 activations: its default Q8
  activations flip near-tie expert choices. Details in
  [docs/design/kolibri.md](docs/design/kolibri.md).
- **Parallel model load.** `WeightLoadPipeline` stages tensors in 64 MB chunks filled by
  several threads (`REFLEX_LOAD_THREADS`, default min(cores, 8)), with prefetch reader
  threads (`REFLEX_LOAD_READERS`, default 16) faulting pages in up to 1 GB ahead; `F32`
  norms no longer stall the pipeline with a synchronous copy. Kolibri-1 on an RTX A6000
  (4.3 GB/s disk), first token: cold 71.6 s -> 19.7 s (llama.cpp 41.3 s), warm 7.6 s ->
  3.7 s (llama.cpp 7.1 s), measured before the prefetch readers. On a second A6000 host
  (6.1 GB/s disk) with the readers: cold 8.5 s (9.25 s with readers off; llama.cpp
  17.2 s), warm 3.8-4.0 s (llama.cpp 7.6 s). T4 warm load: Mistral 7B
  quantized-resident 2306 -> 1738 ms, Qwen3-4B 1098 -> 805 ms; Qwen3-0.6B unchanged.
- **Runpod Hub worker: streaming and an `ADAPTER_ARGS` deploy field** (`v0.2.3-runpod-hub`).
  `.runpod/handler.py` is now a generator handler: a job with `"stream": true` yields each
  `chat.completion.chunk` (readable live from `/stream/{job_id}`), and any other job yields
  its one `chat.completion`. **Breaking for existing callers:** a non-streaming job's
  `output` is now a one-element list (the `worker-vllm` convention), not the bare object.
  Adapter errors fail the job with the adapter's error JSON. `hub.json` declares
  `ADAPTER_ARGS` as an advanced deploy field, allows the whole `AMPERE_16` pool (Ada cards
  included, now that the fatbin is verified on Ada), and names the baked-in model. The
  release also moves the Hub image to the fatbin kernels and slim runtime.

## 2026-10-06

- **Fixed: Mistral 7B produced garbled text.** The SentencePiece tokenizer merged
  characters only by `tokenizer.ggml.merges`, which many llama-family GGUFs (including
  TheBloke's Mistral-7B-v0.1) don't contain, so those prompts were encoded one character
  per token. It now merges by token score, as llama.cpp does. On a T4, Reflex's prompt
  token ids match llama.cpp's `llama-tokenize` on 12 of 12 test strings (spaces, digits,
  punctuation, code, accents, CJK, emoji) for both Mistral-7B and TinyLlama, and
  Mistral's generated text matches `llama-simple`. TinyLlama and Qwen3 output is
  byte-identical to before.

## 2026-10-02

- **Runpod comparison against llama.cpp, finished**: two more llama.cpp cold runs bring
  both engines to n=5 on the L4 (engine load 0.49 s vs. 0.95 s, ~1.9x; end-to-end still
  inconclusive), and the `--no-warmup` variant (n=3, 0.92 s) shows llama-server's warmup
  is not where its startup time goes. See
  [docs/runpod-llamacpp-comparison.md](docs/runpod-llamacpp-comparison.md#results).
- **Faster model load: the token embedding stays in the file mapping.** The table used to
  be copied out of the mmap at load (127.6 MB for Qwen3-0.6B Q4_K_M), which took ~103 ms of
  ~231 ms model load on a T4. Rows are now read straight from the mapping when used. On a
  T4, `system1` cold start went from ~452 to ~421 ms and `generate` from ~700 to ~674 ms
  (p50, n=10, two interleaved rounds); output is unchanged. The model keeps the GGUF file
  mapped for its lifetime.
- **f16 weight storage, the new default**: matrix weights are dequantized straight to
  `f16` on the GPU (every dequant kernel has an `_f16` twin), halving weight VRAM to about
  2 bytes per parameter. Decode uses f16-weight GEMV kernels with f32 accumulation;
  prefill uses `cublasGemmEx` (f16 inputs, f32 compute), with an activation cast that
  saturates at 65504 instead of overflowing and reports what it clamped. `--weights f32`
  / `REFLEX_WEIGHTS=f32` keeps the exact reference mode, and `reflex check` defaults to
  it. LoRA merges stay in f32. Result lines gain `weights_dtype` (`--json` schema
  `1.1.0`). With f16 weights, cuBLAS's first-call cost is paid on the load's worker
  thread. Verified on a T4 (2026-10-04): f16 and f32 give the same tokens on every tested
  model, VRAM halves, decode is 36-42% faster, and cold start is 7% (`system1`) and 8%
  (`generate`) faster than f32. Details in
  [docs/benchmarks.md](docs/benchmarks.md#f16-weight-storage).

## 2026-10-01

- **External energy measurement**: `reflex-energy` (built with `--features nvml`) measures
  a command's whole-process GPU energy from outside it, with the GPU's idle draw
  subtracted. A cold `reflex system1` on a T4 costs ~5.4 J net of ~27.7 J gross; most of
  the gross figure is the GPU idling. `scripts/bench_cold_energy.sh` runs it n times, and
  the GPU nightly workflow reports it.
- **Runpod comparison against llama.cpp, prepared**: an image for llama.cpp's official
  CUDA server on the same Runpod load-balancing setup, and a procedure in
  [docs/runpod-llamacpp-comparison.md](docs/runpod-llamacpp-comparison.md). First run on
  an L4: engine load 0.49 s (Reflex) vs. 0.95 s (llama.cpp); end-to-end wall clock is
  dominated by the platform and inconclusive (Reflex n=5, llama.cpp n=3).
- **`.gitattributes` forces LF for `*.sh`**: a Windows checkout (`core.autocrlf=true`)
  put CRLF into `serverless/runpod-llamacpp/start.sh`, and the built image failed with
  `exec /start.sh failed: No such file or directory`.

## 2026-09-30

- **Online-softmax attention kernels** (`src/kernels_cuda/attention_online.cu`) replace the
  four original attention kernels: 8x faster prefill and decode at 8K context on a T4,
  context limit raised from 11,264 to 65,535 positions. The old kernels stay selectable
  with `REFLEX_ATTN_KERNEL=legacy`.
- **Typed errors**: every engine error has a stable category (`src/error.rs`), exposed as
  IPC `error_kind`, the C FFI's `reflex_last_error_code()`, and HTTP status codes in the
  sidecar. Message text unchanged.
- **`src/model.rs` split** into `src/model/` modules (config, loading, kernels, dense,
  hybrid, mla), move-only.
- **Slim, consolidated Docker images**: one multi-target root `Dockerfile`; runtime is
  CUDA `base` + cuBLAS (images 51-66% smaller); kernels default to a multi-arch fatbin,
  which removed ~0.8 s of driver JIT from every fresh-container start.
- **Multi-arch fatbin** verified on a T4; diagnostics now fail clearly on a GPU older than
  every compiled architecture.
- **Public contributor doc** `docs/DEVELOPMENT.md`.
- **GPU nightly CI** workflow with a cold-start regression gate (off until its secrets
  are configured; see [docs/gpu-ci.md](docs/gpu-ci.md)).
- **Request limits in the OpenAI sidecar** (`max_tokens` cap, prompt size, queue depth,
  timeout), and a clean `context length exceeded` error instead of a CUDA launch failure.
- **Llama / Mistral** (dense GQA) and **`qwen35moe`** architectures.

## Model architectures

All four steps below are **done** and verified on real hardware:

1. **Dense Qwen3** — the best-understood, most well-documented architecture to build
   against first; proves the AOT-compilation + cold-start-benchmark harness works at all.
2. **Qwen3-MoE**
3. **Qwen3.5 hybrid Gated DeltaNet mixer**
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension.

### Later additions (breadth on demand, not a roadmap axis)

The four families above cover the architectural *mechanisms* — dense attention, sparse
MoE, hybrid linear attention, latent-KV attention. Further coverage is added only when a
concrete target model pulls it, never chased for its own sake: broad model support is
llama.cpp's axis, not this engine's (see [Non-goals](README.md#non-goals) — the same reason
throughput and serving features are out of scope). The two additions so far, in order:

- **Llama / Mistral (dense GQA)** — *implemented and verified byte-exact against llama.cpp
  on TinyLlama-1.1B*. The cheapest addition, because GQA, RMSNorm,
  SwiGLU, the SentencePiece tokenizer, and the device-resident dense/MoE forward path
  already exist generically; the real delta is the RoPE convention
  (`LLAMA_ROPE_TYPE_NORM`'s consecutive-pair rotation, reusing the `rope_norm_kernel` the
  MLA path already ships) plus an explicit architecture whitelist. Mistral-7B GGUFs report
  `general.architecture = "llama"` (llama.cpp has no bare `mistral` arch), so this path
  covers Mistral-7B too — though running a 7B model needs >15GB because every weight is
  held `f32`.
- **`qwen35moe`** — *implemented and verified byte-exact against llama.cpp* on a
  converted `qwen3.5-moe-tiny-random` checkpoint (128 experts / top-10, sigmoid-gated
  shared expert). The Qwen3.5 hybrid trunk is reused unchanged; every
  layer's FFN becomes routed MoE (renormalized top-k, reusing the existing per-expert and
  grouped-GEMM dispatch) plus a shared expert scaled per token by
  `sigmoid(ffn_gate_inp_shexp · x)` — the one convention that differs from MLA's
  always-on shared expert. No new kernels. Real Qwen3.5-35B-A3B needs far more VRAM than
  the verification GPU because every weight is held `f32`.

Each addition counts as done only after an independent-implementation comparison
(llama.cpp, or a host CPU reference) on real generated tokens.

## Productization phases

Once the model-architecture MVP proves the engine handles the target model
families at all, the next axis is making the *cold-start path itself* faster and
adoptable — without ever crossing into building a serving platform. The framing: let
vLLM win the warm-throughput race; Reflex wins by being the fastest way to turn
cold compute into one output token, then getting out of the way. All four phases below
are **done**:

- **Phase 1** — Single-shot CLI: process launch -> one forward pass -> exit.
- **Phase 2 — Fast IO**: weights upload to the GPU once (not re-uploaded per kernel
  call), device-resident activations through a whole layer, and on-GPU dequant kernels
  for the block types this project's fixtures use for the bulk of weight bytes. This is
  the work behind the llama.cpp benchmark result above.
- **Phase 3 — State I/O**: `--export-kv <file>`/`--import-kv <file>` for raw K/V-cache
  dump/load/resume, across all four architectures. Reflex stays ignorant of *where*
  that file lives (NVMe, an S3-backed FUSE mount, tmpfs) — that's the orchestrator's
  job, not this engine's.
- **Phase 4 — Embeddability**: `--lora <path>` (load-time adapter application, no
  runtime hot-swap multiplexer) and a Rust C-FFI surface (`src/ffi.rs`,
  `include/reflex_engine.h`) so an external orchestrator can embed Reflex directly
  instead of `exec`-ing a binary.
