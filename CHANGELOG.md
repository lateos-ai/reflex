# Changelog

What has been built, in order. Measurements live in
[docs/benchmarks.md](docs/benchmarks.md).

## 2026-10-07

- **`POST /v1/classify` in the OpenAI sidecar**: Reflex's System1 candidate scoring over
  HTTP. A request names a `prompt` (or chat `messages`) and up to 64 `labels`; the engine
  runs one prefill, scores each label as a continuation, and the response carries each
  label's probability within the set, the most probable label and the entropy. Nothing
  is generated. On a T4 with Qwen3-0.6B a warm request takes ~10 ms and returns exactly
  the probabilities `reflex system1` gives. A label's score sums raw token logits, so
  labels of different token lengths aren't comparable: the response carries a `warning`
  when they differ. Not an OpenAI endpoint (OpenAI has no classification API). The Runpod
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
  3.7 s (llama.cpp 7.1 s), measured before the prefetch readers. T4 warm load: Mistral 7B
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
