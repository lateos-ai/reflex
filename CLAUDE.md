# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Reflex is a GGUF-native GPU inference engine optimized for **cold-start energy and
latency** (process launch to first token) — not sustained server throughput. Chasing
llama.cpp/vLLM on steady-state throughput is an unwinnable kernel-optimization race;
cold start is the underserved axis. Read "Non-goals" below before proposing
throughput-oriented or serving-platform features here.

**Core technical bet**: every CUDA kernel is compiled ahead-of-time by `nvcc` at *build*
time (via `build.rs`), never at runtime via NVRTC — avoiding a real multi-second JIT tax
per process that a naive runtime-compilation design would pay on every cold start.
`src/aot.rs` loads the precompiled PTX/cubin via the CUDA driver API at process start.

Full narrative context, MVP milestone write-ups, and benchmark numbers live in
`HISTORY.md` (the full development log; `README.md` stays a short public-facing
overview) — read it before starting new architecture or perf work; it's kept current
as the project's log, not just a pitch doc.

`CLAUDE.md`, `HISTORY.md`, `STATUS.md` and `DECISIONS.md` are stripped from the public
branch, so **public files (code comments, READMEs, docs, scripts) must never point at
them**. Point at `docs/DEVELOPMENT.md` instead: the public contributor doc that mirrors
this file's Non-goals, core technical bet, build/test, model-loading and test-fixture
sections. When you change one of those sections here, update it there too.

## Build / run / test commands

This crate requires the CUDA toolkit (`nvcc` on `PATH` or `CUDA_PATH`/`CUDA_HOME`) and
an NVIDIA GPU to build and run for real — `build.rs` invokes `nvcc` against every `.cu`
file in `src/kernels_cuda/` at build time and panics if `nvcc` isn't found.

- `cargo build --release` — normal build, emits portable PTX kernels (JIT'd to SASS by
  the driver at load time). Produces a single `reflex` binary with subcommands.
- `REFLEX_CUDA_ARCH=sm_86 cargo build --release` — compile kernels straight to a
  `cubin` for one target architecture (zero driver-side JIT, but the binary only runs on
  that compute capability). `sm_86` is a common Ampere-class arch (e.g. an RTX A6000);
  match this to your actual GPU.
- `REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90 cargo build --release` — compile a
  multi-arch **fatbin** (one native cubin per listed arch + an embedded
  forward-compatible PTX for the highest listed arch so a newer GPU still loads via
  driver JIT). This is the "one image, many GPU generations, no per-arch rebuild" mode —
  the answer to a mixed-architecture GPU pool (see README's "Core technical bet").
  Mutually exclusive with `REFLEX_CUDA_ARCH` setting both is a build-time panic.
  `reflex doctor` reports whether the detected GPU gets a native or PTX-fallback image.
- `REFLEX_SKIP_CUDA=1 cargo build` — skip kernel compilation entirely, for editing/
  type-checking on a machine without CUDA. No subcommand will actually run kernels
  in this mode.
- The core build has **no default features** (explicit `default = []`): only
  `cudarc`/`half`/`memmap2`/`rand`. Optional features (`ipc`, `json-output`, `download`,
  `nvml`, `python`) are each gated behind their own Cargo feature; only `download`
  (Linux `libssl-dev`+`pkg-config`) and `python` (a Python interpreter) add extra *host*
  build deps. See README's "Build features" section for the full table.
- `cargo run --release --bin reflex -- smoke` — the first thing to run on any fresh GPU
  instance; proves the AOT pipeline works end to end and prints
  `process_start_to_first_result_ms`.
- `cargo run --release --bin reflex -- generate <path-to-gguf> [prompt]` — runs a real
  forward pass (dense, MoE, hybrid, or MLA, auto-detected from GGUF metadata) and prints
  `process_start_to_first_token_ms token_id=... token_text=...`.
- `cargo test` — runs unit tests in `dequant.rs`/`dequant_iq.rs`/`gguf.rs`/`moe.rs`/
  `tokenizer.rs`.

**Gotcha**: `cargo test` only rebuilds test-harness binaries under
`target/release/deps/` — it does **not** rebuild `target/release/reflex`. After any
source change, explicitly run `cargo build --release --bin reflex` (or `cargo run
--release --bin reflex -- <subcommand>`) before trusting the binary's behavior; a
passing `cargo test` is not evidence the binary itself is current.

There is no CPU/mock fallback for the `generate`/`system1`/`smoke`/`bench` subcommands —
real GPU-hardware verification is the only way to confirm forward-pass correctness,
since the whole point of the project is measuring real cold-start behavior.

## Architecture

### Kernel compilation pipeline
`build.rs` finds `nvcc`, compiles every `src/kernels_cuda/*.cu` into `OUT_DIR` in one of
three mutually-exclusive modes — portable PTX (default), a single-arch cubin if
`REFLEX_CUDA_ARCH` is set, or a multi-arch fatbin if `REFLEX_CUDA_ARCHS` is set — and
exposes each kernel's output path to the binary via a `REFLEX_KERNEL_<NAME>` env var
(read with `env!(...)` at compile time — see `src/bin/reflex/smoke.rs`).
`src/aot.rs::load_kernel` loads those bytes at
process start via `Ptx::from_file` (maps to the driver's `cuModuleLoad`, which accepts
PTX/cubin/fatbin transparently — this is why the same loader code works for all three
output modes).

### Model loading and forward pass (`src/model.rs`)
`Model::load` reads a GGUF file (`src/gguf.rs`, mmap-based parsing) and, per layer, builds either `DenseLayerWeights` or
`MoeLayerWeights` (the `LayerWeights` enum), decided by `parse_model_config` checking
whether `<arch>.expert_count` is present and nonzero in the GGUF metadata (not a
hardcoded architecture-string check). Every weight tensor is dequantized once
(`src/dequant.rs`/`dequant_iq.rs`, byte-exact vs. `gguf-py`) and uploaded to the GPU
once inside `Weight` (`WeightData::F16` for matrix weights by default, `WeightData::F32`
with `--weights f32`/`REFLEX_WEIGHTS=f32` and for every norm/router/`ssm_*` tensor --
`is_matrix_weight` decides; see docs/DEVELOPMENT.md's model-loading section and
`docs/reference.md`'s "Weight storage" for the f16 kernels, the 65504 activation-cast
saturation, and LoRA's f32 merge) — **do not** reintroduce per-call
`htod_sync_copy` of weight buffers inside `gemv`/`gemv_expert`/`rmsnorm`; that was a real
regression (see README's "Phase 2, round 1" section) that made the engine ~4.3x slower
than llama.cpp until fixed. Only the token embedding table's *raw quantized* bytes stay
host-resident (`LazyTokenEmbedding`, needed for the host-side embedding-lookup gather);
individual rows are dequantized lazily and cached on first use (see HISTORY.md's "Lazy
`token_embd` dequant (item 6)"). Those bytes stay **in the GGUF mmap**
(`LazyTokenEmbedding::raw` is a `gguf::SharedBytes` handle; the model keeps the file
mapped for its lifetime) — don't reintroduce a `to_vec()` copy: it was ~103 ms of ~231 ms
`model_load_ms` on a T4 (`REFLEX_LOAD_PROFILE=1`), mostly page-faulting the new `Vec`.

**Opt-in: `REFLEX_QUANT_RESIDENT=1`** (dense path, plus Kolibri-1 including its stacked
experts and Q6_K tensors, one expert at a time via `Model::quant_expert_weight`; other
MoE models, hybrid and MLA print a notice and keep their normal storage). Q4_K matmul weights stay as raw GGUF blocks in one device
arena (`WeightData::Quant`), read by `gemv_q4k`/the fused multi-row prefill kernel, or,
above a row crossover, dequantized device-to-device into a reused scratch buffer in the
`--weights` dtype (f16 scratch + `cublasGemmEx`, or f32 + `Sgemm`); a Q6_K LM head stays
quantized too (`gemv_q6k_kernel`). Every other matrix weight follows `--weights`. Design,
results and open gaps: `docs/design/quantized-resident-weights.md` — read it before
touching weight residency, the matmul kernels or `WeightLoadPipeline`. On top of it,
**`REFLEX_LAZY_EXPERTS=1`** (Kolibri-1 only) reserves the stacked-expert arena space
without copying and uploads each routed expert once, the first time a layer routes to it
(`Model::ensure_experts`, `LazyExperts`), through the same pinned pipeline; prefetch
readers are off in that mode and the fill threads outlive the load. That is one copy per
weight per process, not a per-call copy. Design and numbers: `docs/design/kolibri.md`
Phase 4 step 2. If the full vocab table is ever needed device-resident
(a tied `lm_head`'s first full-vocab-logits call, or `load_hybrid`/`load_mla`'s eager
tied case — see `Model::lm_head_resident`), it goes through the same on-device
`dequantize_tensor_to_device` path every other weight tensor uses, **not** a host-side
dequant loop — a single-threaded host loop over the whole vocab table used to sit on
this path and was a real, measured ~548ms/~63%-of-`model_load_ms` cost (still eagerly
paid by `load_hybrid`/`load_mla`'s tied case, and, after item 6 made dense/MoE's copy
lazy, silently relocated into `prompt_eval_ms` on `generate`/`check`'s first call
instead of removed — a net regression documented in item 6's own numbers); don't
reintroduce it.

That per-tensor load loop runs through `WeightLoadPipeline`, which stages each
tensor's raw quantized bytes through two reused pinned host buffers in chunks of at
most 64 MB and uploads them on a forked copy stream, so chunk N+1's fill overlaps chunk
N's H2D and tensor N+1's transfer overlaps tensor N's dequant kernel. A chunk of 8 MB or
more is filled by several threads (`REFLEX_LOAD_THREADS`, default min(cores, 8)), and
separate prefetch readers (`REFLEX_LOAD_READERS`, default 16, 0 = off) fault the
file's pages in up to 1 GB ahead of the fill without copying: cold loads want many
reads in flight, warm loads want few copy threads (more fight the H2D DMA for memory
bandwidth), so the two counts are separate knobs. On a 47.5 GB model one host memcpy
thread was the whole load's bottleneck (Kolibri Phase 4, docs/design/kolibri.md);
small models never wake either kind of thread.
Tensors with no dequant kernel (`F32` norms, the host fallback) go through the same
staging slots plus an async device-to-device copy (`upload_host_bytes`), **not**
`htod_sync_copy`: that synchronizes the compute stream, so every norm drained the whole
pipeline (~0.9 s of a 2.3 s Mistral 7B load on a T4).
**Don't "simplify" it back to a blocking `htod_sync_copy` per tensor**, and don't make
its staging buffers per-tensor allocations: both were measured, and the reasons each
alternative is wrong (a host/device race in one case, a cross-stream dependency that
serializes the pipeline in the other) are written up in HISTORY.md's "Pipelined model
load (item 5)" entry. `Model` keeps its `dequant_kernels`/`dequant_pipeline` alive past
load specifically so `lm_head_resident` can reuse this same on-device path lazily.

`forward_prompt` runs: embedding lookup (host-side gather, `batch_size` always 1) → each
layer via `forward_layer` (dispatches to `forward_layer_dense` or `forward_layer_moe`,
both sharing `forward_attn_block`: RMSNorm → QKV → QK-Norm (if present) → RoPE → causal
GQA attention → O-proj residual) → FFN (dense: single SwiGLU; MoE: router GEMM via
`gemv` on `ffn_gate_inp` → `moe::route_top_k` host-side → per-selected-expert SwiGLU via
`gemv_expert`, which slices the relevant chunk out of each 3-D `[in_features,
out_features, expert_count]` per-expert-stacked tensor → weighted-summed by the router's
combination weights) → final RMSNorm → LM head → next-token choice, `crate::sampling`'s
`Model::generate`-shared greedy argmax (still the default, and what `reflex check`'s
byte-exact-vs-llama.cpp methodology depends on) or explicit-opt-in temperature/top-k/
top-p sampling (`reflex generate --temperature/--top-k/--top-p/--seed`; IPC's
`"sampling": {...}`, see `src/ipc.rs`) — see `src/sampling.rs`'s doc comment; this was
unimplemented scope, not a Non-goals constraint (README.md's Non-goals list only
`batch_size`/concurrency/networking, never sampling strategy). No KV-cache reuse across
process runs, no batching — those stay permanent (see Non-goals).

`general.architecture == "deepseek2"` dispatches to `Model::load_mla`/
`forward_prompt_mla` (MVP step 4, Multi-head Latent Attention) instead, a separate path
from the `LayerWeights`/`forward_layer` machinery above (own `MlaLayerWeights`/
`MlaModel`, same dispatch pattern as `"qwen35"` → `load_hybrid`). See HISTORY.md's MLA
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
`route_top_k`: softmax over all experts, select top-k, renormalize. No new
CUDA kernels were needed for MoE — it reuses `gemv`/`silu_and_mul` unchanged via
`gemv_expert`'s slicing.

### `reference/`
`reference/gated_deltanet_reference.rs` is a host/CPU reference implementation of the
recurrence math for the Qwen3.5 hybrid Gated DeltaNet mixer (MVP step 3, done) — the
correctness oracle used while debugging the GPU kernel. It is not compiled as part of
this crate.

## MVP order (see HISTORY.md for full detail and current status)

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
   the A6000's 48GB, needed the bigger instance). See HISTORY.md's MLA sections.

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
engine may grow directly is limited to sequential, non-network-stack IPC (`reflex stdio`
JSON-line mode, `reflex uds` Unix Domain Socket mode — see README's "Non-goals" section
for detail) plus in-process bindings (the existing C FFI, and PyO3 Python bindings) —
never a thread pool, never a queue.

## Known test-fixture limitation

`test-data/Tiny-Moe.Q4_K_M.gguf` (used to verify the MoE path) is a Mixtral-style
synthetic fixture with `expert_used_count == expert_count`, so it cannot prove top-k
routing actually excludes any expert, and its `llama`-architecture SentencePiece
tokenizer blocks text-level byte-exact resume verification (see STATUS.md's Phase 3
round 2 entry). `test-data/tiny-qwen3moe.gguf` (post-MVP addition, hand-built the same
way as the MLA fixture below, source archived as `test-data/tiny-qwen3moe-src.tar.gz`)
closes both gaps: a real `qwen3moe`-architecture file with `expert_used_count=2 <
expert_count=8`, real Qwen3 QK-Norm tensors, and a `gpt2`-style tokenizer — verified
against this project's own parser
(`model::moe_fixture_tests::qwen3moe_fixture_has_excluding_topk_and_qk_norm`, host-only,
no GPU needed) and, since, real-hardware-verified too: `qwen3moe_fixture_generates_without_error`
passed on a real L40 GPU, and `prefill_dense_batched_matches_sequential_prefill`
(batched-vs-sequential MoE routing, byte-exact) passed against it as well. Real GGUF
test fixtures otherwise live outside this repo (`.gguf` is gitignored) — check with
the user for their location before assuming another fixture path is valid.

No small real `deepseek2`-architecture GGUF exists publicly at all (not just locally
— see HISTORY.md's MLA section). `test-data/deepseek-tiny-mla.gguf` is a fully
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
