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

`general.architecture == "deepseek2"` dispatches to `Model::load_mla`/
`forward_prompt_mla` (MVP step 4, Multi-head Latent Attention) instead, a separate path
from the `LayerWeights`/`forward_layer` machinery above (own `MlaLayerWeights`/
`MlaModel`, same dispatch pattern as `"qwen35"` → `load_hybrid`). See README.md's MLA
sections for the math and this MVP step's scope (dense-lead layers, routed-MoE +
always-on shared-expert FFN via `MlaFfn::Dense`/`Moe`, and YaRN RoPE scaling are all
supported — real DeepSeek-V2-Lite's actual shape; only Q-LoRA query decomposition and
MTP are still rejected — see `parse_mla_config`'s doc comment for the exact rejected
cases). Reuses every device-resident op convention Phase 2 round 2 established
(`rmsnorm`/`gemv`/`silu_and_mul`/`add_inplace` unchanged) plus several new ones
specific to MLA: `gemv_view`/`gemv_per_head` (per-head GEMV via the same `gemv_kernel`,
"expert" → "head" vs. `gemv_expert`'s MoE slicing), `mla_attention` (`kernels_cuda/
mla_attention.cu`, MQA with mismatched Q/K vs. V dims — the existing `attention_kernel`
assumes a uniform head_dim, which doesn't fit MLA's compressed-KV/decompressed-output
split), and `route_top_k_with_norm` (`moe.rs` — real DeepSeek-V2-Lite doesn't
renormalize its router's top-k weights, unlike Qwen3-MoE's convention `route_top_k`
already assumed). **Important, easy-to-miss details** (each one produced
silently-wrong, non-crashing output before being caught by byte-exact llama.cpp
comparison — see DECISIONS.md's two MLA entries for the full list): MLA's `q_pe`/
`k_pe` RoPE uses llama.cpp's `LLAMA_ROPE_TYPE_NORM` convention (consecutive-pair
rotation, `rope_norm_kernel`/`rope_norm_yarn_kernel`), not the `LLAMA_ROPE_TYPE_NEOX`
(half-split) convention `rope_kernel` implements for Qwen3/Qwen3.5 — confirmed against
`llama_model_rope_type` in llama.cpp's `llama-model.cpp`; don't assume RoPE convention
is architecture-independent when adding another model family later.

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
reference recurrence math for the Qwen3.5 hybrid Gated DeltaNet mixer (MVP step 3,
done) as the correctness oracle used while debugging that kernel. It is not compiled
as part of this crate.

## MVP order (see README.md for full detail and current status)

1. Dense Qwen3 — done.
2. Qwen3-MoE — done.
3. Qwen3.5 hybrid Gated DeltaNet mixer — done.
4. DeepSeek-V2/V3 MLA — deliberately last (compressed latent-KV caching is a genuinely
   different mechanism from GQA, not an incremental extension). Read llama.cpp PR
   #11446 before attempting it. **Done**: dense-lead layers, routed-MoE +
   always-on shared-expert FFN, and YaRN RoPE scaling are all supported (real
   DeepSeek-V2-Lite's actual shape); only Q-LoRA query decomposition and MTP/NextN
   are still rejected with a clear error. Verified against both a synthetic fixture
   (no small real `deepseek2` GGUF exists publicly) and the real
   `deepseek-ai/DeepSeek-V2-Lite` checkpoint on a rented 80GB A100 (~63GB of `f32`
   device-resident weights under this project's GPU-residency model — doesn't fit
   the A6000's 48GB, needed the bigger instance). See README.md's MLA sections.

## Non-goals (permanent constraints, not just current-MVP scope)

`batch_size` is always 1. No internal request queue/scheduler, no continuous batching,
no multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager, no concurrent HTTP/
gRPC server. Multi-tenancy and persistent state belong in a *host orchestrator*, never
inside this engine — see README's "Non-goals" section for the full rationale before
proposing anything in this direction. If a warm-context mode is ever added, it accepts
one job at a time, strictly sequentially, never a thread pool.

No in-core HTTP/gRPC server, ever — this is the same rule as `batch_size`-always-1/
no-thread-pool above, not a separate exception any adoption/UX ask gets to reopen. If
HTTP access to this engine is ever needed, the pattern is a separate, optional sidecar
binary (e.g. an OpenAI-compatible adapter) that talks to this engine over local IPC —
the core engine itself never grows a network socket. The local-ergonomics surface this
engine may grow directly is limited to sequential, non-network-stack IPC (`--stdio`
JSON-line mode, `--uds` Unix Domain Socket mode — see README's "Non-goals" section for
detail) plus in-process bindings (the existing C FFI, and PyO3 Python bindings) — never
a thread pool, never a queue.

## Known test-fixture limitation

There is no small real `qwen3moe`-architecture GGUF available locally for fast
iteration (`test-data/Tiny-Moe.Q4_K_M.gguf`, used to verify the MoE path, is a
Mixtral-style synthetic fixture with `expert_used_count == expert_count`, so it cannot
prove top-k routing actually excludes any expert). Real GGUF test fixtures live outside
this repo (`.gguf` is gitignored) — check with the user for their location before
assuming a fixture path is valid.

No small real `deepseek2`-architecture GGUF exists publicly at all (not just locally
— see README.md's MLA section). `test-data/deepseek-tiny-mla.gguf` is a fully
synthetic fixture: hand-built HF-format `config.json`/`safetensors` (random weights,
authentic tensor names/shapes) run through llama.cpp's own real, unmodified
`convert_hf_to_gguf.py`, verified against a real llama.cpp build. Its source
(`config.json`, the weight-generation script, tokenizer files) is archived as
`test-data/deepseek-tiny-mla-src.tar.gz` (gitignored, like all of `test-data/`) in
case it needs regenerating or extending. MLA is also verified against the real
`deepseek-ai/DeepSeek-V2-Lite` checkpoint (converted fresh with this project's pinned
`convert_hf_to_gguf.py`, `--outtype q8_0`, ~16.7GB), but that GGUF is *not* preserved
in local `test-data/` (too large) — see STATUS.md's "Real DeepSeek-V2-Lite GGUF not
preserved locally" entry for how to regenerate it, including the gotcha that every
pre-quantized community GGUF checked predates llama.cpp's MLA tensor-split conversion
format and will be rejected by `parse_mla_config`.
