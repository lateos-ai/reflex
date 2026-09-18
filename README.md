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
3. **Qwen3.5 hybrid Gated DeltaNet mixer** — done; see the "Qwen3.5 hybrid Gated DeltaNet
   mixer (MVP step 3)" section below for scope and real-hardware verification.
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension. Read llama.cpp's real
   implementation (PR #11446) before attempting it.

## Non-goals

These are permanent constraints on this engine, not just current-MVP scope — the whole
reason coldstart-infer exists is to win a narrower bet (cold-start energy/latency) than
sustained-server throughput, and RustFeference's own postmortem
(`../RustFeference/LESSONS_LEARNED_RUSTFEFERENCE.md`) is explicit that a broad project
re-inherits the exact throughput/serving race that's unwinnable against llama.cpp/vLLM/
SGLang's head start. Multi-tenancy and persistent state belong in the *host
orchestrator*, not in this engine:

- **`batch_size` is always 1.** No request queue, no continuous batching, no
  PagedAttention-style dynamic allocation, no context preemption. Horizontal scaling
  (many concurrent jobs) is the orchestrator's job — spin up N `coldstart-infer`
  processes across GPU slices/time-slices — not this engine's, ever.
- **No internal multi-tenant LoRA router/scheduler.**
- **No internal NVMe/S3 KV-cache manager or cache-hit logic.**
- **No concurrent HTTP/gRPC server**, no request auth/rate-limiting, no autoscaling
  decision-making. If a warm-context mode ever exists (see Phase 4 below), it accepts
  one job at a time, strictly sequentially — never a thread pool.

This mirrors RustFeference's own `serve_http.rs`, which drew this same line once before
("explicitly out of scope: gRPC, auth/rate-limiting, multi-model serving").

## Post-architecture-MVP roadmap: productization

Once the model-architecture MVP above proves the engine handles the target model
families at all, the next axis is making the *cold-start path itself* faster and
adoptable — without ever crossing into building a serving platform. The framing: let
vLLM win the warm-throughput race; coldstart-infer wins by being the fastest way to turn
cold compute into one output token, then getting out of the way.

- **Phase 1 (current)** — Single-shot CLI: process launch -> one forward pass -> exit.
  This is what MVP steps 1-2 (dense Qwen3, Qwen3-MoE) already are.
- **Phase 2 — Fast IO**: zero-copy storage-to-GPU weight loading. Investigate `mmap` +
  `io_uring` (Linux) or NVIDIA GPUDirect Storage to skip the current host-side
  dequant-then-`htod_sync_copy` round trip per weight (`model.rs`'s `load_weight`
  closure), and pre-faulted/pre-allocated CUDA memory pools instead of per-kernel-call
  `alloc_zeros` (every `gemv`/`rmsnorm`/etc. call in `model.rs` currently allocates a
  fresh device buffer). Architecture-agnostic — applies uniformly under dense, MoE, and
  future hybrid/MLA forward passes, so it doesn't block on or get blocked by remaining
  architecture-coverage work. **Now sized by real data, not assumption**: the first real
  cold-start A/B benchmark against llama.cpp (see the MoE status section above) found
  coldstart-infer ~4.3x *slower* than llama.cpp on the same hardware/model, with 4x the
  peak RSS and ~11x the system CPU time — strong evidence `load_weight`'s full-`f32`
  host-side dequant is the dominant cost. This makes Phase 2 high-priority, not
  speculative.
- **Phase 3 — State I/O**: two new CLI flags, `--export-kv <file>` and
  `--import-kv <file>`, doing raw binary dump/load of the K/V cache to/from a file
  descriptor. coldstart-infer stays ignorant of *where* that file lives or how it got
  there (NVMe, an S3-backed FUSE mount, tmpfs) — that's the orchestrator's job. No
  caching policy, no cache-hit logic, inside this engine.
- **Phase 4 — Embeddability**: a single `--lora <path>` CLI flag (load-time adapter
  application only, no runtime hot-swap multiplexer — process spin-up is already cheap
  enough that a fresh process per adapter is the scale-from-zero answer, not in-process
  swapping), plus a Rust C-FFI surface so an external orchestrator daemon can embed
  coldstart-infer directly instead of `exec`-ing a binary. If a warm-context IPC mode is
  ever built, it's stdin/stdout or a Unix domain socket, one job at a time, never a
  concurrent server (see Non-goals above).

Phase 2 can run in parallel with the remaining architecture-coverage steps (3-4) above;
Phases 3-4 are lower priority and should follow once architecture coverage and Phase 2
are solid.

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

### Qwen3-MoE (MVP step 2)

`src/moe.rs` (router: softmax over all experts, top-k select, renormalize -- ported
from RustFeference's own verified `route_top_k`, git history around commit `6a70287`
"qwen3moe support") plus `src/model.rs` changes: `parse_model_config` now returns an
`Option<MoeMetaConfig>` keyed off `<arch>.expert_count` being present and nonzero (not a
hardcoded architecture-string check -- GGUF namespaces all per-arch metadata under the
file's own `general.architecture` value, confirmed against a real fixture, not assumed),
and `LayerWeights` is now `Dense`/`Moe`, sharing one `forward_attn_block` (RMSNorm -> QKV
-> QK-Norm if present -> RoPE -> causal attention -> O-proj residual) byte-for-byte
between both. MoE's FFN replaces the dense path's single shared FFN with: a router GEMM
(`ffn_gate_inp`) reusing the existing `gemv` kernel unchanged, `route_top_k` run host-side,
then one naive per-selected-expert SwiGLU FFN (`gemv_expert` slices the relevant
contiguous chunk out of each 3-D per-expert-stacked tensor -- `[in_features,
out_features, expert_count]`, confirmed against llama.cpp's `qwen3moe.cpp` -- and reuses
the existing `gemv`/`silu_and_mul` kernels unchanged), weighted-summed by the router's
combination weights. No new CUDA kernels were needed. Naive (ungrouped) per-expert
dispatch is the deliberate MVP scope, per RustFeference's own documented finding that
it's the correct starting point.

No small real `qwen3moe`-architecture GGUF was available to test against (a real
Qwen3-30B-A3B is far too large for quick iteration), so this was verified end to end on
real hardware against RustFeference's `Tiny-Moe.Q4_K_M.gguf` fixture instead: a real,
Mixtral-style MoE GGUF (`general.architecture = "llama"`, `expert_count=2`,
`expert_used_count=2`, no QK-Norm tensors) that exercises the actual new MoE-specific
machinery (per-expert tensor slicing, router GEMM, weighted-sum dispatch) even though
it isn't Qwen3's own architecture. Three runs of `"Once upon a time"` against it produced
byte-identical output (token id 4036, `","`) -- deterministic, doesn't crash, and (since
this fixture is a randomly-initialized synthetic test model, per its own
`general.name = "Tiny_Test"`) that's the correctness bar this fixture can actually prove,
not a factual-completion check like dense Qwen3's `"Paris"` result. **Known limitation of
this fixture**: `expert_used_count` equals `expert_count` here (2 of 2), so top-k always
selects *every* expert -- this run cannot distinguish "top-k routing selects correctly"
from "all experts are always used"; it does verify per-expert weight slicing, the router
GEMM, and weighted-sum accumulation, since both experts' distinct weights are genuinely
exercised and combined. Re-verifying against a fixture with `expert_used_count <
expert_count` (ideally real `qwen3moe`-architecture metadata, to also exercise Qwen3's
QK-Norm and MoE together in one file) is future work, not this milestone's scope.

The dense Qwen3 path was re-verified byte-identical against both of its previous known
results (`"Once upon a time"` -> `","` token id 11; `"The capital of France is"` ->
`" Paris"`) after this refactor, confirming `forward_attn_block`'s extraction didn't
change dense behavior.

### First real cold-start benchmark: coldstart-infer vs. llama.cpp

The comparison flagged as outstanding since dense Qwen3 landed (see "Status" above) has
now been run, on the same A6000 instance, against the same `Qwen3-0.6B-Q4_K_M.gguf`, same
prompt (`"Once upon a time"`), greedy/`--temp 0`, full GPU offload for both. llama.cpp was
built from source this session (`ggml-org/llama.cpp` commit `972d231`, `cmake
-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`, Release) and run via `llama-cli -n 1 --temp
0 -ngl 99 --no-warmup -st --simple-io`. Both engines were measured the same way — external
wall clock via `/usr/bin/time -v` (process launch to exit, including OS exec/dynamic-
linking overhead that coldstart-infer's own internal `Instant::now()`-based metric
excludes) — three runs each:

| | run 1 | run 2 | run 3 | peak RSS | user+sys CPU time |
|---|---|---|---|---|---|
| **llama.cpp** | 6.47s | 6.59s | 6.56s | 900 MB | 1.70s + 1.27s |
| **coldstart-infer** | 29.64s | 27.93s | 28.23s | 3.68 GB | 5.13s + 14.02s |

**coldstart-infer is currently ~4.3x slower than llama.cpp on cold start, not faster —
the core thesis this project bets on is unproven and currently reversed.** This isn't a
surprise (README's own "Status"/MoE sections already flagged the load path as
correctness-first and unoptimized), but the magnitude and a concrete likely cause are new:
coldstart-infer's 14.02s of *system* time (kernel/syscall time — page faults, memory
allocation) versus llama.cpp's 1.27s, and 4x the peak resident memory, points squarely at
`model.rs`'s `load_weight` closure, which dequantizes every tensor to a fresh full-`f32`
host `Vec` before any GPU upload — exactly the "host-side dequant of every weight" and
"one host<->device round trip per kernel call per layer" candidates already named as
unoptimized. This is real, actionable evidence for sizing Phase 2 (Fast IO) below, not
just a hypothesis.

Caveats, disclosed rather than smoothed over: llama.cpp's `llama-cli` runs a
conversation-style REPL (ASCII banner, `/exit`-style commands) that coldstart-infer's
minimal binary doesn't have, and this llama-cli build gave no discovered flag to fully
confirm the raw prompt wasn't wrapped in the model's embedded chat template (`tokenizer.
chat_template` is present in this GGUF) the way coldstart-infer's raw tokenizer path
guarantees — llama.cpp's own reported prompt-processing rate (150.4 t/s over a handful of
tokens, tens of milliseconds either way) makes this negligible next to the multi-second
gap, but it means the two runs are not proven to process byte-identical token sequences.
Single-machine, single-session, `n=3` — not a rigorous statistical benchmark, but large
enough and repeatable enough (all three coldstart-infer runs within ~2s of each other) to
act on.

### Phase 2 (Fast IO), round 1: fixing the load path identified above

The `model.rs` root cause above was actually two compounding bugs, not one:
`load_weight` did dequantize every tensor to a full-`f32` host `Vec` at load time (as
suspected), but every one of `gemv`/`gemv_expert`/`rmsnorm`'s host `Vec<f32>` weight
buffers were *also* being re-uploaded to the GPU via a fresh `htod_sync_copy` on every
single call — i.e. every layer, every token position, every generated token — on top of
staying host-resident (never freed) for the model's entire lifetime. GGUF file loading
itself was already zero-copy `mmap` (`gguf.rs`), so "Fast IO" here turned out to mean
"stop re-uploading and re-retaining weights we already uploaded once", not `io_uring`.

Fix: `Weight` now holds a `CudaSlice<f32>` (uploaded once, immediately after
dequantizing, with the host scratch buffer dropped right after) instead of a host
`Vec<f32>`. `gemv`/`gemv_expert`/`rmsnorm` take that device buffer directly — MoE's
per-expert dispatch slices it with `CudaSlice::slice` (a zero-copy device-side view, no
device-to-device copy). Only the token embedding table stays host-resident (needed for
host-side embedding-lookup gather; unchanged from before) and, only when embeddings are
tied to the LM head, its already-dequantized host bytes are uploaded a second time as
`lm_head` rather than dequantized twice as the old `.or_else(|_| load_weight(...))`
fallback did.

Re-measured the same way (`/usr/bin/time -v`, same `Qwen3-0.6B-Q4_K_M.gguf`, same
prompt, three runs), after re-verifying byte-identical output on both fixtures first
(dense `"Once upon a time"` -> `","` token id 11, `"The capital of France is"` ->
`" Paris"`; MoE `Tiny-Moe.Q4_K_M.gguf` -> token id 4036 -- all unchanged):

| | run 1 | run 2 | run 3 | peak RSS | user+sys CPU time |
|---|---|---|---|---|---|
| **llama.cpp** (unchanged) | 6.47s | 6.59s | 6.56s | 900 MB | 1.70s + 1.27s |
| **coldstart-infer, before** | 29.64s | 27.93s | 28.23s | 3.68 GB | 5.13s + 14.02s |
| **coldstart-infer, after** | 11.31s | 11.70s | 10.44s | 1.33 GB | ~3.2s + ~4.0s |

Gap closed from ~4.3x to **~1.7x slower than llama.cpp** — peak RSS down ~2.75x, system
time down ~3.3x. Still not faster, and the remaining gap is most likely the CPU-bound
host-side dequantize-to-`f32` step itself (llama.cpp's CUDA backend uploads quantized
bytes as-is and dequantizes/matmuls on the GPU, never materializing a full-`f32` host
copy at all) plus this MVP's one-host-round-trip-per-op kernel structure (every `gemv`/
`rmsnorm`/`rope`/`silu`/`attention` call still does its own `htod`/`dtoh` for the
*activation* vectors, even though those are small). Candidate follow-ups, not yet
attempted: an on-GPU dequant kernel (matches llama.cpp's approach, but is a real new-
kernel-writing project, not a load-path tweak), and/or keeping activations device-
resident across a whole layer instead of round-tripping between every op.

### Qwen3.5 hybrid Gated DeltaNet mixer (MVP step 3)

`src/gated_deltanet.rs` (shape config) + `src/kernels_cuda/gated_deltanet.cu` (five new AOT
kernels: causal depthwise conv1d+SiLU+window-advance, per-head L2-norm, gate/decay
computation, the delta-rule state update, and the gated-output RMSNorm) + `src/model.rs`'s
new hybrid path implement a real Qwen3.5 (`general.architecture = "qwen35"`) forward pass.
A hybrid file interleaves two mixer kinds per transformer layer -- Gated DeltaNet
(recurrent linear attention, no KV cache) and Gated Attention (regular GQA softmax
attention with a *fused* query+gate projection and partial RoPE) -- resolved from the
file's own `qwen35.attention.recurrent_layers` array or `qwen35.full_attention_interval`
fallback (never hardcoded), matching real llama.cpp `qwen35.cpp`/`delta-net-base.cpp`
exactly (the math was ported from RustFeference's own `reference/
gated_deltanet_rustfeference.rs`, itself fetched from real llama.cpp source, as the
correctness oracle). Scope deliberately narrowed from that reference for this MVP:
single-token sequential dispatch only (no chunked/parallel-prefill kernels -- same
naive-first precedent as MoE's per-expert dispatch), the dense `qwen35` architecture only
(`qwen35moe`, which additionally replaces the FFN with routed MoE, is out of scope and
rejected with a clear error), and no MTP/NextN block support (rejected outright if
`nextn_predict_layers` is nonzero).

Verified end to end on the same real A6000 against a real `Qwen3.5-0.8B-Q4_K_M.gguf`
fixture (24 layers, Gated Attention at trunk indices `[3,7,11,15,19,23]`, GDN elsewhere --
confirmed both via metadata parsing and by scanning the file's own tensor names):
`"Once upon a time"` -> `","` (token id 11) and `"The capital of France is"` -> `" the"`
(token id 279), both **independently reproduced by a fresh `ggml-org/llama.cpp` build from
source** (`examples/simple`'s raw, non-chat-templated completion path -- `tools/llama-cli`'s
newer conversational mode always applies the model's embedded chat template even with
`-p`, the same caveat already flagged in the dense-vs-llama.cpp benchmark above, so
`examples/simple` was used instead for a true prompt-in/token-out comparison). Byte-exact
match on both prompts is strong independent evidence the implementation is correct, not
just "doesn't crash."

**One real bug found and fixed via this cross-check**: the first attempt produced
fluent-looking but semantically wrong completions (e.g. Chinese text following an English
prompt) that ran without crashing or producing NaN -- `forward_gdn_mixer` computed the
mixer's output projection but never added it back to the residual stream (`x + out_proj`),
unlike the Gated Attention mixer's forward function, which did. Since 18 of the fixture's
24 layers are Gated DeltaNet layers, this silently broke the residual stream through most
of the network. Isolated by cross-checking layer 0's mixer output against a pure-host
CPU reference (a trimmed, cudarc-free port of `reference/gated_deltanet_rustfeference.rs`'s
`step` function) given the same real dequantized weights and input -- the two matched
bit-for-bit before the fix (confirming the kernels themselves were already correct) and
the end-to-end generation matched real llama.cpp only after adding the missing residual
add. Lesson for future sessions: a plausible-looking non-crashing output is not evidence
of correctness for a new architecture path -- get an independent ground truth (here, a
fresh llama.cpp build) before trusting it, the same posture this project already takes
toward its own kernels.

Next: DeepSeek-V2/V3 MLA (MVP step 4, deliberately last per the MVP order above), or a
second Phase 2 round on the remaining cold-start gap -- open call, not yet decided.
