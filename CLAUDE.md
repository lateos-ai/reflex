# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A GGUF-native GPU inference engine optimized for **cold-start energy and latency**
(process launch to first token) — not sustained server throughput. It's a deliberate,
narrower pivot from a prior project, [RustFeference](../RustFeference) (`rft-gpu`),
whose postmortem (`../RustFeference/LESSONS_LEARNED_RUSTFEFERENCE.md`) found that
chasing llama.cpp/vLLM on steady-state throughput is an unwinnable kernel-optimization
race. Read that postmortem before proposing throughput-oriented or serving-platform
features here — see "Non-goals" below.

**Core technical bet**: every CUDA kernel is compiled ahead-of-time by `nvcc` at *build*
time (via `build.rs`), never at runtime via NVRTC (which is what rft-gpu did, at a
measured ~4.5s JIT tax per process). `src/aot.rs` loads the precompiled PTX/cubin via
the CUDA driver API at process start.

Full narrative context, MVP milestone write-ups, and benchmark numbers live in
`README.md` — read it before starting new architecture or perf work; it's kept current
as the project's log, not just a pitch doc.

## Build / run / test commands

This crate requires the CUDA toolkit (`nvcc` on `PATH` or `CUDA_PATH`/`CUDA_HOME`) and
an NVIDIA GPU to build and run for real — `build.rs` invokes `nvcc` against every `.cu`
file in `src/kernels_cuda/` at build time and panics if `nvcc` isn't found.

- `cargo build --release` — normal build, emits portable PTX kernels (JIT'd to SASS by
  the driver at load time).
- `COLDSTART_CUDA_ARCH=sm_86 cargo build --release` — compile kernels straight to a
  `cubin` for one target architecture (zero driver-side JIT, but the binary only runs on
  that compute capability). `sm_86` is the ThunderCompute A6000 dev instance's arch.
- `COLDSTART_SKIP_CUDA=1 cargo build` — skip kernel compilation entirely, for editing/
  type-checking on a machine without CUDA. No inference binary will actually run kernels
  in this mode.
- `cargo run --release --bin smoke_coldstart` — the first thing to run on any fresh GPU
  instance; proves the AOT pipeline works end to end and prints
  `process_start_to_first_result_ms`.
- `cargo run --release --bin qwen3_coldstart <path-to-gguf> [prompt]` — runs a real
  forward pass (dense or MoE Qwen3, auto-detected from GGUF metadata) and prints
  `process_start_to_first_token_ms token_id=... token_text=...`.
- `cargo test` — runs unit tests in `dequant.rs`/`dequant_iq.rs`/`gguf.rs`/`moe.rs`/
  `tokenizer.rs`.

**Gotcha**: `cargo test` only rebuilds test-harness binaries under
`target/release/deps/` — it does **not** rebuild `target/release/<bin-name>`. After any
source change, explicitly run `cargo build --release --bin <name>` (or `cargo run
--release --bin <name>`) before trusting the standalone binary's behavior; a passing
`cargo test` is not evidence the binary itself is current.

There is no CPU/mock fallback for the model binaries — real GPU-hardware verification
(currently done on a ThunderCompute A6000 instance) is the only way to confirm forward-
pass correctness, since the whole point of the project is measuring real cold-start
behavior.

## Architecture

### Kernel compilation pipeline
`build.rs` finds `nvcc`, compiles every `src/kernels_cuda/*.cu` to PTX (or cubin if
`COLDSTART_CUDA_ARCH` is set) into `OUT_DIR`, and exposes each kernel's output path to
the binary via a `COLDSTART_KERNEL_<NAME>` env var (read with `env!(...)` at compile
time — see `smoke_coldstart.rs`). `src/aot.rs::load_kernel` loads that PTX/cubin file at
process start via `Ptx::from_file` (maps to the driver's `cuModuleLoad`, which accepts
PTX/cubin/fatbin transparently — this is why the same loader code works for both output
modes).

### Model loading and forward pass (`src/model.rs`)
`Model::load` reads a GGUF file (`src/gguf.rs`, mmap-based parsing — ported unmodified
from RustFeference) and, per layer, builds either `DenseLayerWeights` or
`MoeLayerWeights` (the `LayerWeights` enum), decided by `parse_model_config` checking
whether `<arch>.expert_count` is present and nonzero in the GGUF metadata (not a
hardcoded architecture-string check). Every weight tensor is dequantized once
(`src/dequant.rs`/`dequant_iq.rs`, byte-exact vs. `gguf-py`) and uploaded to the GPU
once as a `CudaSlice<f32>` inside `Weight` — **do not** reintroduce per-call
`htod_sync_copy` of weight buffers inside `gemv`/`gemv_expert`/`rmsnorm`; that was a real
regression (see README's "Phase 2, round 1" section) that made the engine ~4.3x slower
than llama.cpp until fixed. Only the token embedding table stays host-resident (needed
for the host-side embedding-lookup gather); its dequantized bytes are reused for
`lm_head` when the two are tied, instead of dequantizing twice.

`forward_prompt` runs: embedding lookup (host-side gather, `batch_size` always 1) → each
layer via `forward_layer` (dispatches to `forward_layer_dense` or `forward_layer_moe`,
both sharing `forward_attn_block`: RMSNorm → QKV → QK-Norm (if present) → RoPE → causal
GQA attention → O-proj residual) → FFN (dense: single SwiGLU; MoE: router GEMM via
`gemv` on `ffn_gate_inp` → `moe::route_top_k` host-side → per-selected-expert SwiGLU via
`gemv_expert`, which slices the relevant chunk out of each 3-D `[in_features,
out_features, expert_count]` per-expert-stacked tensor → weighted-summed by the router's
combination weights) → final RMSNorm → LM head → greedy argmax. No KV-cache reuse across
process runs, no batching, no sampling beyond argmax — deliberately out of scope (see
Non-goals).

### MoE routing (`src/moe.rs`)
`route_top_k`: softmax over all experts, select top-k, renormalize. Ported from
RustFeference's verified `route_top_k` (git history around commit `6a70287`). No new
CUDA kernels were needed for MoE — it reuses `gemv`/`silu_and_mul` unchanged via
`gemv_expert`'s slicing.

### Code salvaged from RustFeference, unmodified except path
`src/gguf.rs`, `src/dequant.rs`/`dequant_iq.rs`/`dequant_iq_tables.rs`,
`src/tokenizer.rs` — these don't care how kernels get compiled, so they ported as-is.
RustFeference's `jit.rs` (NVRTC compile-and-load) was deliberately **not** ported — it's
the thing this project replaces.

### `reference/`
`reference/gated_deltanet_rustfeference.rs` carries over RustFeference's host/CPU
reference recurrence math for the Qwen3.5 hybrid Gated DeltaNet mixer (MVP step 3, not
yet started) as the correctness oracle for a from-scratch AOT kernel. It is not compiled
as part of this crate yet.

## MVP order (see README.md for full detail and current status)

1. Dense Qwen3 — done.
2. Qwen3-MoE — done.
3. Qwen3.5 hybrid Gated DeltaNet mixer — not started.
4. DeepSeek-V2/V3 MLA — deliberately last (compressed latent-KV caching is a genuinely
   different mechanism from GQA, not an incremental extension). Read llama.cpp PR
   #11446 before attempting it.

## Non-goals (permanent constraints, not just current-MVP scope)

`batch_size` is always 1. No internal request queue/scheduler, no continuous batching,
no multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager, no concurrent HTTP/
gRPC server. Multi-tenancy and persistent state belong in a *host orchestrator*, never
inside this engine — see README's "Non-goals" section for the full rationale before
proposing anything in this direction. If a warm-context mode is ever added, it accepts
one job at a time, strictly sequentially, never a thread pool.

## Known test-fixture limitation

There is no small real `qwen3moe`-architecture GGUF available locally for fast
iteration (`test-data/Tiny-Moe.Q4_K_M.gguf`, used to verify the MoE path, is a
Mixtral-style synthetic fixture with `expert_used_count == expert_count`, so it cannot
prove top-k routing actually excludes any expert). Real GGUF test fixtures live outside
this repo (`.gguf` is gitignored) — check with the user for their location before
assuming a fixture path is valid.
