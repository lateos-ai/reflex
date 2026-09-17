# coldstart-infer

A GGUF-native GPU inference engine optimized for **cold-start energy and latency**
(process launch to first token) rather than sustained server throughput.

## Why this exists

This project is a direct pivot from [RustFeference](../RustFeference) (`rft-gpu`), a
from-scratch Rust/CUDA inference engine that, after ~26 phases of correctness-first
work, was measured (real hardware, same-machine A/B) behind both llama.cpp and vLLM on
every steady-state throughput metric. The full postmortem is
`../RustFeference/LESSONS_LEARNED_RUSTFEFERENCE.md` — read it first.

The short version: closing that gap is a kernel-optimization race against projects with
a multi-year head start, and Rust as a language doesn't change who wins it. What none of
llama.cpp, vLLM, or `candle` are built for or measured against is a **cold** invocation —
serverless/FaaS, single-shot CLI/dev-tool calls, batch/cron jobs, edge devices that wake
on demand. rft-gpu's own kernels paid a real, measured ~4.5s NVRTC JIT tax on first use;
vLLM took ~235s to become ready (CUDA graph capture) before serving one request;
llama.cpp avoids both because its kernels are compiled by `nvcc` at *build* time, not at
process start.

**Target metric**: energy-to-first-token from cold start (joules, process launch to
first generated token) — a real, underexplored gap. Existing energy benchmarks measure
warm/steady-state joules-per-token, not full-lifecycle cold-start cost.

**Target models**: Qwen and DeepSeek families.

## Core technical bet

Every CUDA kernel is compiled **ahead of time** (`build.rs` invokes `nvcc`, see
`build.rs` and `src/kernels_cuda/`), never at runtime via NVRTC. `src/aot.rs` loads the
precompiled PTX/cubin at process start via the CUDA driver API. Default mode emits
portable PTX (small driver-side JIT-to-SASS cost at load); set `COLDSTART_CUDA_ARCH=sm_XX`
to compile straight to a `cubin` for one target architecture (true zero-JIT, at the cost
of needing a matching cubin per deployment target). Which one actually wins on real
hardware is unverified — that's the first thing to measure, not assume.

Run `cargo run --bin smoke_coldstart` on a real GPU instance as the very first
real-hardware step: it proves the AOT pipeline works end to end and reports actual
process-start-to-first-result wall clock on the simplest possible kernel, before any
model-architecture work begins.

## MVP order

1. **Dense Qwen3** — reuses RustFeference's most mature, most-verified architecture;
   proves the AOT-compilation + cold-start-benchmark harness works at all.
2. **Qwen3-MoE**
3. **Qwen3.5 hybrid Gated DeltaNet mixer** — `reference/gated_deltanet_rustfeference.rs`
   carries over the host/CPU reference recurrence math from RustFeference as the
   correctness oracle for a from-scratch AOT kernel (not compiled as part of this crate
   yet — extract the pure-math functions when this milestone starts).
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension. Read llama.cpp's real
   implementation (PR #11446) before attempting it.

## Salvaged from RustFeference (reused as-is, unmodified except path)

- `src/gguf.rs` — GGUF metadata/tensor-directory parsing (mmap-based).
- `src/dequant.rs`, `src/dequant_iq.rs`, `src/dequant_iq_tables.rs` — standard and
  i-quant dequantization, verified byte-exact against `gguf-py` in RustFeference.
- `src/tokenizer.rs` — verified against real sentencepiece/BPE references.

None of these care how kernels get compiled, so they port unmodified. What was
deliberately **not** ported: RustFeference's `jit.rs` (NVRTC `compile_and_load`) — that's
the thing this project replaces, not reuses.

## Status

Scaffold only. `smoke_coldstart` has not yet been run on real hardware — do that before
writing any model code, and before trusting anything in this README's numbers claims
(there are none yet on purpose).
