# Developing Reflex

Notes for people changing the engine: what it optimizes for, the constraints every
change has to respect, how to build and test it with or without a GPU, and a few
performance rules that were learned the hard way. The [README](../README.md) is the
user-facing overview; this page is the contributor-facing one that code comments point
to.

## What this project optimizes for

Reflex is a GGUF-native GPU inference engine optimized for **cold-start energy and
latency**: the time and energy from process launch to the first token. It is not
optimized for sustained server throughput. Competing with llama.cpp or vLLM on
steady-state throughput is a kernel-optimization race this project deliberately doesn't
enter; cold start is the underserved axis. Judge a proposed change by what it does to
that number.

## Non-goals

These are permanent constraints, not current scope. The rules, then the full
rationale [below](#the-full-rationale):

- **`batch_size` is always 1.** No internal request queue or scheduler, no continuous
  batching, no thread pool. The engine runs exactly one request at a time, which is why
  code in this repo can rely on there never being a concurrent caller (single-owner
  `RefCell`/`OnceLock` state, scalar per-token kernel arguments, and so on).
- **No multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager.** Multi-tenancy
  and persistent state belong in a host orchestrator, never inside this engine. For
  example, `kv_io.rs` reads and writes a cache file but never decides where it lives.
- **No network server in the core engine, ever.** HTTP access is provided by the
  separate [`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) crate, which
  talks to the engine over local IPC. The engine itself may only grow sequential,
  non-network interfaces (`reflex stdio`, `reflex uds`) and in-process bindings (the C
  FFI, the PyO3 module). These always target GPU 0.

### The full rationale

These are permanent constraints on this engine, not just current-MVP scope — the whole
reason Reflex exists is to win a narrower bet (cold-start energy/latency) than
sustained-server throughput. A broad serving feature set re-inherits the exact
throughput/serving race that's unwinnable against llama.cpp/vLLM/SGLang's head start.
Multi-tenancy and persistent state belong in the *host
orchestrator*, not in this engine:

> **"Optimized for serverless" describes where this engine runs well, not features it
> grows.** A "Serverless-Native Inference Engine" pitch — building platform machinery
> (internal queue, scheduler, autoscaling logic, `batch_size > 1`) *into* Reflex — was
> considered and rejected on 2026-09-17. Nothing in this page's
> serverless framing reopens that. Running the unmodified binary as a workload on
> Runpod's (or anyone's) control plane is the same arrangement as the AWS ASG pattern
> already documented here: the platform is the host orchestrator, and the engine stays
> single-shot. The violation is defined by what lives *inside* the engine, never by who
> calls it.

- **`batch_size` is always 1.** No request queue, no continuous batching, no
  PagedAttention-style dynamic allocation, no context preemption. Horizontal scaling
  (many concurrent jobs) is the orchestrator's job — spin up N `Reflex`
  processes across GPU slices/time-slices — not this engine's, ever.
- **No internal multi-tenant LoRA router/scheduler.**
- **No internal NVMe/S3 KV-cache manager or cache-hit logic.**
- **No concurrent HTTP/gRPC server**, no request auth/rate-limiting, no autoscaling
  decision-making. If a warm-context mode ever exists (see Phase 4 below), it accepts
  one job at a time, strictly sequentially — never a thread pool.

**No in-core HTTP/gRPC server, ever, not deferred.** This is not a separate exception
to the rule above — a concurrent HTTP listener is the exact same violation
("`batch_size` always 1... never a thread pool") under a different name, and an
adoption/UX ask asking for one doesn't get to reopen it. If HTTP access to this engine
is ever genuinely needed, the pattern is a **separate, optional sidecar binary** that
talks to this core engine over local IPC only — the core engine itself never grows a
network socket. That sidecar now exists:
[`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) is a standalone crate
(own `Cargo.toml`/`Cargo.lock`, not a workspace member, no dependency on
`reflex-engine`) implementing an OpenAI-compatible `POST /v1/chat/completions`
(streaming and non-streaming) in front of one managed `reflex stdio` child process —
real HTTP concurrency on the sidecar's front door, still strictly one request at a
time into the core engine underneath. Renders the loaded GGUF's own
`tokenizer.chat_template` (falling back to plain role-labeled prompt concatenation
when one isn't present or fails to render) — see its own README for usage and known
limitations.

For local, non-network ergonomics, this engine may instead expose: a **stdio JSON-line
mode** (`reflex stdio`, one JSON request per stdin line, fully processed before the next
line is read) and a **Unix Domain Socket mode** (`reflex uds <path>`, Unix-only, one
connection fully processed before the next is accepted) — both strictly sequential,
never a thread pool, mirroring the same request/response protocol. A shared-memory
ring-buffer transport was considered and deliberately deferred — crash-safety and
synchronization design is disproportionate complexity for the ergonomics it would buy
— recorded here as a future-work idea only, not designed.

## Core technical bet

Every CUDA kernel is compiled **ahead of time** by `nvcc` from `build.rs`, never at
runtime through NVRTC. Runtime compilation would add a real JIT cost to every cold
start, which is the one number this project exists to minimize. `src/aot.rs` loads the
precompiled PTX, cubin or fatbin through the CUDA driver API at process start. The build
modes (portable PTX, single-arch cubin, multi-arch fatbin) are described in the reference's
[Kernel build modes](reference.md#kernel-build-modes) section.

This is also why benchmarks against engines that JIT-compile or capture CUDA graphs at
startup measure something different from a comparison with llama.cpp, whose kernels are
also precompiled.

## Build, run and test

The crate needs the CUDA toolkit (`nvcc` on `PATH`, or `CUDA_PATH`/`CUDA_HOME` set) and
an NVIDIA GPU to build and run for real.

```
cargo build --release                           # portable PTX kernels
REFLEX_CUDA_ARCH=sm_86 cargo build --release    # one cubin for one GPU architecture
cargo run --release --bin reflex -- smoke       # first thing to run on a new GPU machine
cargo run --release --bin reflex -- generate <model.gguf> "prompt"
cargo test                                      # host-side unit tests
```

Optional features (`ipc`, `json-output`, `download`, `nvml`, `python`) and their extra
host dependencies are listed in the reference's
[Build features](reference.md#build-features) section.

`cargo test` rebuilds only the test harness binaries. It does **not** rebuild
`target/release/reflex`, so after a source change, run
`cargo build --release --bin reflex` before trusting the binary's behavior.

### Working without CUDA

```
REFLEX_SKIP_CUDA=1 cargo build
```

skips kernel compilation entirely, for editing and type-checking on a machine without
the CUDA toolkit. The resulting binary contains no real kernels, so no subcommand can
run inference. This is the mode CI uses on GitHub-hosted runners.

### No CPU fallback

There is no CPU or mock implementation behind `generate`, `system1`, `smoke` or `bench`.
Forward-pass correctness can only be verified on real GPU hardware, and that is
intentional: the point of the project is to measure real cold-start behavior. GPU-only
tests are marked `#[ignore]`; run them with `cargo test --release -- --ignored` on a GPU
machine, or see [gpu-ci.md](gpu-ci.md) for the nightly GPU workflow.

## Model loading: weights stay on the GPU

`Model::load` dequantizes every weight tensor once and uploads it once, as a
device-resident buffer. Matrix weights (the operands of the matmul kernels:
`is_matrix_weight` in `src/model/loading.rs`) are stored as `f16` by default and as
`f32` with `--weights f32` / `REFLEX_WEIGHTS=f32`. Everything else is always `f32`:
norms, biases, the Gated DeltaNet `ssm_*` tensors, the MoE routers, activations and the
KV cache. `Weight.data` is a `WeightData` enum (`F32`, `F16`, `Quant`), and every matmul
wrapper in `src/model/kernels.rs` dispatches on it.

- **The f16 path reads f16 weights and accumulates in f32.** Decode GEMVs
  (`gemv_f16_kernel` and friends) widen each weight exactly and multiply by the f32
  activation. Prefill casts activations to f16 and calls `cublasGemmEx` with
  `CUBLAS_COMPUTE_32F`. That cast saturates at f16's 65504 rather than producing inf, and
  counts what it clamped (`Model::f16_activation_stats`).
- **`--weights f32` is the reference mode.** Its dequant kernels compile to the same
  PTX instructions as before f16 existed. `reflex check` and llama.cpp comparisons
  default to it.
- **LoRA merges happen in f32.** `LoadOptions::lora_adapter` loads the adapter's target
  weights as `f32`; `apply_lora` adds the delta and then rounds to f16 once.

**Opt-in: `REFLEX_QUANT_RESIDENT=1`** (dense Qwen3/Llama/Mistral path, and Kolibri-1,
whose Q4_K and Q6_K tensors stay quantized including the stacked experts; other MoE
models, hybrid and MLA print a notice and keep their normal storage). Q4_K matmul weights are uploaded
as their raw GGUF blocks into one device arena and dequantized inside the matmul
kernels (`gemv_q4k`, the fused multi-row prefill kernel), or, for longer prompts, into
a reused device scratch buffer that cuBLAS then reads. That scratch buffer has the
`--weights` dtype: `f16` (read by `cublasGemmEx`) by default, `f32` (`Sgemm`) with
`--weights f32`. A Q6_K LM head is kept as raw blocks too. Every other matrix weight
follows `--weights`. The rules below still hold: the scratch path is device to device,
so weights are never copied host-to-device per call. See
[design/quantized-resident-weights.md](design/quantized-resident-weights.md).

Rules that follow from measured regressions:

- **Never copy weights host-to-device per call.** An early version re-uploaded weight
  buffers inside `gemv`/`rmsnorm` on every call and was about 4.3x slower than
  llama.cpp until that was removed. The same per-call-overhead reasoning is why batched
  kernels such as `gemv_per_head_batch` exist: thousands of small launches per layer
  cost as much as the copies did.
- **Dequantize on the device, not in a host loop.** The token-embedding table stays
  host-resident as raw quantized bytes, and rows are dequantized lazily as tokens need
  them. When the full vocabulary table is needed on the GPU (a tied LM head), it goes
  through the same on-device dequant path as every other tensor. A single-threaded host
  loop over the whole table once cost about 548 ms, most of model load.
- **Keep the token-embedding table in the mmap; don't copy it out.**
  `LazyTokenEmbedding` holds a `SharedBytes` handle into the GGUF mapping, and the model
  keeps the file mapped for its lifetime, as llama.cpp does. Copying the table into an
  owned `Vec` at load (127.6 MB for Qwen3-0.6B) was the largest single part of model
  load, about 103 ms of 231 ms on a T4, mostly first-touch page faults on the new
  allocation rather than file reads.
- **Keep the pipelined upload.** `WeightLoadPipeline` stages each tensor's raw bytes
  through two reused pinned host buffers, in chunks of at most 64 MB, and uploads on a
  separate stream, so one chunk's fill overlaps the previous chunk's copy and tensor
  N+1's copy overlaps tensor N's dequant kernel. Chunks of 8 MB or more are filled by
  several threads (`REFLEX_LOAD_THREADS`, default min(cores, 8)), while prefetch
  reader threads (`REFLEX_LOAD_READERS`, default 16, 0 disables them) fault the file's
  pages in up to 1 GB ahead without copying. Cold loads need many reads in flight;
  warm loads need few copy threads, since extra ones compete with the H2D copy for
  memory bandwidth. One fill thread was the bottleneck on large models, and small
  models never start either kind of thread.
  Tensors with no dequant kernel (`F32` norms) take the same slots plus an async
  device-to-device copy; a `htod_sync_copy` there synchronizes the compute stream and
  drains the pipeline at every norm. Replacing it with a blocking copy per tensor,
  or allocating the staging buffers per tensor, reintroduces a host/device race or
  serializes the pipeline. The f16 dequant kernels go through the same pipeline; only
  the output element type differs.
- **Numerics probes.** `REFLEX_F16_ROUNDTRIP=1` (with `--weights f32`) rounds every
  matrix weight to f16 and back, which isolates weight rounding from the f16 kernels.
  `REFLEX_TOP2_TRACE=1` prints each generated position's top-2 logits.
  [`scripts/verify_f16_weights.sh`](../scripts/verify_f16_weights.sh) uses both.

## GGUF metadata conventions

Model configuration is read from arch-prefixed GGUF keys: `general.architecture` names
the architecture (for example `qwen3`), and every other key is looked up as
`<architecture>.<key>`, such as `qwen3.context_length`. Whether a model uses MoE layers
is decided by `<architecture>.expert_count` being present and nonzero, not by a
hardcoded list of architecture names. Tools outside the engine (the sidecar's
`gguf_meta` reader) follow the same convention.

## Known test-fixture limitations

`.gguf` files are gitignored, so fixture-based tests are `#[ignore]`d and need the files
in `test-data/` locally:

- `test-data/tiny-qwen3moe.gguf` is a synthetic `qwen3moe` model with
  `expert_used_count` (2) smaller than `expert_count` (8), real Qwen3 QK-norm tensors and
  a `gpt2`-style tokenizer, so it exercises top-k routing that actually excludes experts.
  It was built by running a hand-written, random-weight Hugging Face checkpoint through
  llama.cpp's unmodified `convert_hf_to_gguf.py`.
- `test-data/deepseek-tiny-mla.gguf` is a synthetic `deepseek2` (MLA) model built the
  same way, because no small real `deepseek2` GGUF exists publicly. It covers the
  dense-lead layers only; the routed-MoE MLA path can only be exercised with the real
  `deepseek-ai/DeepSeek-V2-Lite` checkpoint (about 17 GB at `q8_0`), which is not kept
  in the repo. Community `deepseek2` GGUFs that predate llama.cpp's MLA tensor-split
  conversion are rejected by this engine, so convert it fresh with a current
  `convert_hf_to_gguf.py`.
- `test-data/kolibri1-tokenizer.gguf` is the metadata section of
  `Hob-forge/Kolibri-1-GGUF`'s `Kolibri-1-Q4_K_M.gguf` (the first ~4.8 MB, read with an
  HTTP range request) rewritten with an empty tensor table. It carries the real
  Kolibri-1 vocab and merges for the tokenizer golden test.
- `test-data/tiny-kolibri1.gguf` (Q4_K_M) and `tiny-kolibri1-f32.gguf` are a synthetic
  `kolibri1` model: 6 layers (layer 4 full attention), 16 experts with top-4, a nonzero
  router bias, a 16-token sliding window and the real Kolibri-1 tokenizer. Mainline
  llama.cpp has no `kolibri1` support, so it was converted and quantized with llama.cpp
  `836d571` plus the community `kolibri1-llama.cpp.patch` (see
  [design/kolibri.md](design/kolibri.md)). Source and build scripts are archived as
  `test-data/tiny-kolibri1-src.tar.gz`.
- Tests that need a real model read its path from `REFLEX_TEST_GGUF`.

[`scripts/gpu_nightly_tests.sh`](../scripts/gpu_nightly_tests.sh) maps each GPU test to
the file it needs and reports the ones it can't run as skipped.

## The `reference/` directory

`reference/gated_deltanet_reference.rs` is a host (CPU) implementation of the Qwen3.5
Gated DeltaNet recurrence. It served as the correctness oracle while the GPU kernel was
being debugged. It is not compiled as part of the crate.
