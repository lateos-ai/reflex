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

These are permanent constraints, not current scope. The full rationale is in the
README's [Non-goals](../README.md#non-goals) section; the rules themselves:

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

## Core technical bet

Every CUDA kernel is compiled **ahead of time** by `nvcc` from `build.rs`, never at
runtime through NVRTC. Runtime compilation would add a real JIT cost to every cold
start, which is the one number this project exists to minimize. `src/aot.rs` loads the
precompiled PTX, cubin or fatbin through the CUDA driver API at process start. The build
modes (portable PTX, single-arch cubin, multi-arch fatbin) are described in the README's
[Core technical bet](../README.md#core-technical-bet) section.

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
host dependencies are listed in the README's
[Build features](../README.md#build-features-core-engine-vs-optional-tooling) section.

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
device-resident `f32` buffer. Rules that follow from measured regressions:

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
- **Keep the pipelined upload.** `WeightLoadPipeline` double-buffers each tensor's raw
  bytes through pinned host memory and uploads on a separate stream, so tensor N+1's
  copy overlaps tensor N's dequant kernel. Replacing it with a blocking copy per tensor,
  or allocating the staging buffers per tensor, reintroduces a host/device race or
  serializes the pipeline.

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
- Tests that need a real model read its path from `REFLEX_TEST_GGUF`.

[`scripts/gpu_nightly_tests.sh`](../scripts/gpu_nightly_tests.sh) maps each GPU test to
the file it needs and reports the ones it can't run as skipped.

## The `reference/` directory

`reference/gated_deltanet_reference.rs` is a host (CPU) implementation of the Qwen3.5
Gated DeltaNet recurrence. It served as the correctness oracle while the GPU kernel was
being debugged. It is not compiled as part of the crate.
