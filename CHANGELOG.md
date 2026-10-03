# Changelog

What has been built, in order. Measurements live in
[docs/benchmarks.md](docs/benchmarks.md).

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
  `1.1.0`). **Not yet verified on a GPU**: correctness, cold-start, memory and decode
  numbers are pending `scripts/verify_f16_weights.sh` (see
  [docs/benchmarks.md](docs/benchmarks.md#f16-weight-storage)).

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
