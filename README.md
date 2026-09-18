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

`smoke_coldstart` has been run on real hardware (ThunderCompute A6000, `cuda12-9`
template, driver `nvidia-smi` 610.43.02 / CUDA 12.9, `rustc`/`cargo` 1.98.1) in **both**
of `build.rs`'s output modes, each verified with a clean rebuild (not a stale binary):

- **Default (PTX, no `COLDSTART_CUDA_ARCH`)**: `build.rs` finds `nvcc` and produces valid
  PTX; `aot::load_kernel` loads and launches it correctly via cudarc 0.11.9. Five runs of
  `./target/release/smoke_coldstart` measured `process_start_to_first_result_ms` (wall
  clock from `Instant::now()` inside `main()`, not from OS process exec) between
  **473–637ms**.
- **`COLDSTART_CUDA_ARCH=sm_86` (cubin)**: also verified end to end — five runs measured
  **480–617ms**, i.e. statistically indistinguishable from the PTX numbers above. For this
  trivial smoke kernel, CUDA context/primary-context init (`CudaDevice::new`) dominates
  the timing; the driver's PTX-JIT-vs-cubin-no-JIT difference is noise-level at this
  scale. That may not hold once real model kernels (bigger PTX, more of them) are loaded —
  worth re-measuring once dense Qwen3 exists.

Risk #1's basic pipeline question is resolved for both modes. Neither number has been
compared against llama.cpp's or rft-gpu's cold start on the *same* hardware in the same
session — don't cite either as a win until that A/B is run.

Two real bugs were found and fixed this session while getting the cubin path working for
the first time (it had never actually produced a working binary before):
- `src/bin/smoke_coldstart.rs` used `include_str!` to embed the compiled kernel, which
  fails to compile against a `.cubin` (binary, not UTF-8). Fixed by switching
  `aot::load_kernel` to take a file path and load via `Ptx::from_file`, which maps to the
  driver's `cuModuleLoad` — per the CUDA driver API docs that accepts cubin, PTX, or
  fatbin files transparently, so one code path now covers both of `build.rs`'s output
  modes. Verified on real hardware above.
- `build.rs` didn't declare `cargo:rerun-if-env-changed=COLDSTART_CUDA_ARCH` (or
  `COLDSTART_SKIP_CUDA`), so Cargo silently reused a stale build when that variable
  changed between runs instead of recompiling. Fixed and verified: switching
  `COLDSTART_CUDA_ARCH` on and off now triggers a real `nvcc` recompile each time, exactly
  the kind of stale-benchmark trap `LESSONS_LEARNED_RUSTFEFERENCE.md` warns about.

### Dense Qwen3 (MVP step 1)

`src/model.rs` + `src/bin/qwen3_coldstart.rs` + five new AOT kernels
(`src/kernels_cuda/{rmsnorm,rope,silu_and_mul,gemv,attention}.cu`) implement a real,
from-scratch dense Qwen3 forward pass: embedding lookup (host-side gather; batch is
always 1) -> every transformer layer (RMSNorm -> QKV -> QK-Norm -> RoPE -> causal GQA
attention -> O-proj residual -> RMSNorm -> SwiGLU FFN residual) -> final RMSNorm -> LM
head -> greedy argmax. No KV-cache reuse across separate process runs, no batching, no
sampling beyond argmax — deliberately out of scope (matches the project's actual target
metric: process-start-to-first-token, not sustained decode throughput). The
architecture/math was ported from RustFeference's own dense-model code as a *correctness
oracle* (git history around commits `d8ed273` and `1459330`), not copied wholesale —
RustFeference's paged-KV-cache/tensor-parallel/serving-scheduler machinery is all out of
scope; the five kernels above are fresh, simple, from-scratch AOT kernels.

Verified end to end on the same real A6000 against `test-data/Qwen3-0.6B-Q4_K_M.gguf`
(real dense Qwen3, not MoE):
- `"Once upon a time"` -> `","` (token id 11), three runs, byte-identical each time
  (greedy argmax, no randomness) — matches RustFeference's own real A100-verified
  generation of this exact prompt/model (`"Once upon a time, there was a man..."` — the
  token immediately after "time" there is also `","`).
- `"The capital of France is"` -> `" Paris"` — a real factual completion, not noise.

Both results are strong independent evidence the RMSNorm/QK-Norm/RoPE/GQA-attention/
SwiGLU math is correct, not just "doesn't crash." `process_start_to_first_token_ms` came
in around **25.6–32.0s** across these runs — far slower than `smoke_coldstart`'s
sub-second numbers, expected and not yet broken down (candidates: host-side dequant of
every weight to f32 at load time, and one host<->device round trip per kernel call per
layer — this MVP is correctness-first, none of that is optimized yet). Breaking that
down, and comparing against llama.cpp's cold start on the same model/hardware, is future
work, not this milestone's scope.

Next: Qwen3-MoE per the MVP order above.
