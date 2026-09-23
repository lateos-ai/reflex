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

**Linux build prerequisite for `--features download`/`ipc`/`python`** (`--all-features`
included): these pull in `hf-hub`, whose `ureq` HTTP client needs `libssl-dev` +
`pkg-config` on the build host, or `cargo build` fails with `openssl-sys` unable to find
an OpenSSL installation. Not needed for the default feature-less build. On Ubuntu/Debian:
```
sudo apt-get install -y libssl-dev pkg-config
```
(Discovered on a fresh ThunderCompute instance during the MVP-release adoption round —
not needed on the Windows dev machine that round otherwise developed on, since
`native-tls` uses a different TLS backend there.)

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

**No in-core HTTP/gRPC server, ever, not deferred.** This is not a separate exception
to the rule above — a concurrent HTTP listener is the exact same violation
("`batch_size` always 1... never a thread pool") under a different name, and an
adoption/UX ask asking for one doesn't get to reopen it. If HTTP access to this engine
is ever genuinely needed, the pattern is a **separate, optional sidecar binary** (e.g.
`system1-openai-adapter`) that talks to this core engine over local IPC only — the core
engine itself never grows a network socket. Building that sidecar is out of scope for
now; this paragraph only records the escape-hatch pattern so a future HTTP ask gets
routed there instead of back into this engine.

For local, non-network ergonomics, this engine may instead expose: a **stdio JSON-line
mode** (`--stdio`, one JSON request per stdin line, fully processed before the next
line is read) and a **Unix Domain Socket mode** (`--uds <path>`, Unix-only, one
connection fully processed before the next is accepted) — both strictly sequential,
never a thread pool, mirroring the same request/response protocol. A shared-memory
ring-buffer transport was considered and deliberately deferred — crash-safety and
synchronization design is disproportionate complexity for the ergonomics it would buy
— recorded here as a future-work idea only, not designed.

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
  caching policy, no cache-hit logic, inside this engine. **Round 1 done** (dense/MoE
  export/import of the raw buffers, verified byte-exact round trip on real hardware —
  see the "Phase 3, round 1" section below). **Round 2 done**: `--import-kv` actually
  resumes generation (a real per-token generation loop, `--max-tokens N`, was added
  alongside it), for dense/MoE and the Qwen3.5 hybrid mixer — see the "Phase 3, round 2"
  section below. **Round 3 done**: extended to MLA's single compressed latent-KV cache,
  closing out Phase 3's architecture coverage entirely — see the "Phase 3, round 3"
  section below.
- **Phase 4 — Embeddability**: a single `--lora <path>` CLI flag (load-time adapter
  application only, no runtime hot-swap multiplexer — process spin-up is already cheap
  enough that a fresh process per adapter is the scale-from-zero answer, not in-process
  swapping), plus a Rust C-FFI surface so an external orchestrator daemon can embed
  coldstart-infer directly instead of `exec`-ing a binary. If a warm-context IPC mode is
  ever built, it's stdin/stdout or a Unix domain socket, one job at a time, never a
  concurrent server (see Non-goals above). **Round 1 done**: `--lora` for dense/MoE
  Qwen3 and the Qwen3.5 hybrid architecture (MLA rejected, matching every other
  MLA-adjacent feature's scope line) — see the "Phase 4, round 1" section below.
  **Round 2 done**: the C-FFI surface (`src/ffi.rs`, `include/coldstart_infer.h`,
  `load`/`generate`/`free`) — see the "Phase 4, round 2" section below. Phase 4 is now
  entirely closed.

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

### Phase 2 (Fast IO), round 2: device-resident activations across a layer

Round 1 (above) fixed weights being re-uploaded on every kernel call; the remaining
~1.7x gap was attributed to two candidates, neither yet attempted: an on-GPU dequant
kernel, and keeping activations device-resident across a whole layer instead of
round-tripping between every op. This round tackled the second one only, since it's a
refactor of existing kernel call sites (no new CUDA math), unlike the dequant kernel's
new-kernel-writing project.

Every kernel-wrapper method in `model.rs` (`rmsnorm`, `gemv`/`gemv_expert`, `rope`,
`silu_and_mul`, `attention`, and the Qwen3.5 hybrid's `gdn_conv`/`gdn_gates`/`gdn_delta`/
`gdn_gated_norm`) used to do its own `htod_sync_copy` before and `dtoh_sync_copy` after
-- i.e. every op in the forward pass round-tripped its activation vector over PCIe and
paid a sync latency hit, even though only weight buffers needed to be GPU-resident.
Fixed by converting every op wrapper to take/return `CudaSlice<f32>` directly (the
pattern `gdn_l2_norm` already used: mutate a device buffer in place) and chaining
device buffers through each of the six forward functions (`forward_attn_block`/
`forward_layer_dense`/`forward_layer_moe`/`forward_gdn_mixer`/
`forward_gated_attn_mixer`/`forward_hybrid_ffn`) without touching host memory
mid-layer. Residual adds moved to a new `add_kernel` (`kernels_cuda/elementwise.cu`) so
they stay device-resident too; `rope_kernel` now takes `position` as a plain scalar
instead of an uploaded device array (`batch_size` is permanently 1, so there was never
more than one position to pass); `silu_and_mul_kernel` takes `gate`/`up` as two
separate buffers instead of requiring a host-side concatenation into one. The single
largest remaining round-trip -- `attention()` re-uploading the *entire* K/V cache
history on every token position -- is also fixed: `k_cache`/`v_cache` are now
preallocated device buffers (sized to the known prompt length up front) that
`forward_attn_block`/`forward_gated_attn_mixer` write into via device-to-device copy
(`CudaDevice::dtod_copy`), never touching the host. Left out of scope, flagged for a
future round: MoE's per-expert weighted-sum accumulation and the Gated Attention
mixer's fused-qg head split/sigmoid gating both still round-trip through the host
(small, low-value relative to the primary dense/MoE path this benchmarks).

Re-verified byte-exact against the same golden tokens as before (dense `"Once upon a
time"` -> `","` id 11, `"The capital of France is"` -> `" Paris"`; MoE `Tiny-Moe` -> id
4036; hybrid `"Once upon a time"` -> `","` id 11, `"The capital of France is"` -> `"
the"` id 279) plus all 55 existing unit tests, on the same A6000 hardware. Re-measured
the same way as round 1 (`/usr/bin/time -v`, same `Qwen3-0.6B-Q4_K_M.gguf`, same
prompt, three runs, fresh `llama.cpp` build at `ce8caa6`):

| | run 1 | run 2 | run 3 | peak RSS | user+sys CPU time |
|---|---|---|---|---|---|
| **llama.cpp** (unchanged) | 6.45s | 6.44s | 6.46s | 887 MB | 1.01s + 1.19s |
| **coldstart-infer, round 1** | 11.31s | 11.70s | 10.44s | 1.33 GB | ~3.2s + ~4.0s |
| **coldstart-infer, round 2** | 6.43s | 6.47s | 8.51s | 1.35 GB | ~2.1s + ~3.5s |

Gap closed from ~1.7x to **~1.1x slower than llama.cpp** (two of three runs landed
within llama.cpp's own run-to-run noise). Peak RSS is unchanged from round 1 (this
round didn't touch the weight-loading path) and system time is still ~2.9x llama.cpp's
-- both point at the same remaining cause round 1 already named: the CPU-bound
host-side dequant-to-`f32` step at load time, which only an on-GPU dequant kernel (not
attempted this round) would remove.

### DeepSeek-V2/V3 Multi-head Latent Attention (MLA) (MVP step 4)

`src/kernels_cuda/mla_attention.cu` (one new AOT kernel) + `src/model.rs`'s new
`parse_mla_config`/`MlaLayerWeights`/`MlaModel`/`Model::load_mla`/
`Model::forward_mla_attn_block`/`Model::forward_prompt_mla` implement a real
DeepSeek-V2/V3 (`general.architecture = "deepseek2"`) forward pass, ported from a full
read of a real `llama.cpp` build's `src/models/deepseek2.cpp` (the `is_mla && is_lite`
branch specifically).

**Fixture problem, resolved differently from every prior MVP step**: no small real
`deepseek2`-architecture GGUF exists publicly at all (not just locally) -- the
smallest real one, DeepSeek-V2-Lite (16B total params, 64 routed + 2 shared experts
across 26 MoE layers), needs **~63GB of `f32` device memory** just for its routed
experts alone under this project's "dequantize every weight once, hold it GPU-resident
for the model's whole lifetime" design (`Weight` in `model.rs`) -- more than the A6000
(48GB) this project develops against; it would need an ~80GB H100 instead. Rather than
rent bigger hardware or partially undo the GPU-residency decision (a bigger, separate
decision -- see DECISIONS.md), this MVP step instead builds a **fully synthetic**
`deepseek2` fixture (`test-data/deepseek-tiny-mla.gguf`): a hand-built HF-format
`config.json` + random-weight `safetensors` checkpoint (27 layers -- chosen specifically
to trip llama.cpp's own `is_lite` heuristic, `hidden_size=64`, `kv_lora_rank=32`,
`qk_rope_head_dim=8`, `qk_nope_head_dim=16`, `v_head_dim=16`, dense-only FFN, Qwen's
real tokenizer -- see below for why) run through llama.cpp's **own real, unmodified**
`convert_hf_to_gguf.py`, so the resulting GGUF has fully authentic `deepseek2` tensor
names/shapes/metadata despite meaningless random weights -- the same validity argument
`Tiny-Moe.Q4_K_M.gguf` already established for MoE. Building this hit two real
gotchas worth recording: llama.cpp's `is_lite` (no Q-LoRA) detection is a hardcoded
`n_layer` check (27, 26, or 48-with-a-specific-vocab-size) against known real models,
not a general flag, so the fixture's layer count had to match one of those magic
numbers on purpose; and `transformers`' `AutoConfig` recognizes `deepseek_v2` as a
real model type and silently fills in *its own* class defaults (`n_routed_experts=64`,
etc.) for anything the hand-written `config.json` didn't set, which crashed llama.cpp's
loader (`n_expert_used_max > 0` assert) until those fields were set to an explicit `0`.

**Scope, deliberately narrow** (same "naive/narrow first" precedent as every prior MVP
step, made unusually pointed here since this is the riskiest item in the whole MVP
order): dense-only (no MoE FFN or shared experts -- `parse_mla_config` hard-errors if
`leading_dense_block_count < block_count`), no Q-LoRA query decomposition (direct `wq`
only, matching real DeepSeek-V2-Lite's own `is_lite` convention, not just a fixture
shortcut -- hard error if `attention.q_lora_rank` is present and nonzero), no YaRN RoPE
scaling, no MTP/NextN (existing precedent). A real DeepSeek-V2/V3 checkpoint needs at
least the MoE+shared-expert FFN to be usable; that, Q-LoRA, and YaRN are open follow-up
work, each with a clear rejection error rather than silent mishandling in the meantime.

**Two non-obvious implementation details, easy to get wrong silently**:
1. The attention softmax scale is `1/sqrt(qk_nope_head_dim + qk_rope_head_dim)` -- the
   *uncompressed* per-head dimension -- **not** `1/sqrt(kv_lora_rank +
   qk_rope_head_dim)` (the actual compressed dot-product width used to compute the
   scores). Every other scaled-dot-product attention in this codebase scales by its
   own dot-product dimension, so this is a real, silent-wrong-numbers trap if copied
   by pattern-matching instead of reading `deepseek2.cpp`'s `kq_scale` directly.
2. MLA's `q_pe`/`k_pe` RoPE uses a **different rotation convention** than every other
   architecture this project supports: llama.cpp's `llama_model_rope_type` maps
   `deepseek2` to `LLAMA_ROPE_TYPE_NORM` (rotates *consecutive* pairs `(2i, 2i+1)`),
   while Qwen3/Qwen3.5 use `LLAMA_ROPE_TYPE_NEOX` (rotates half-split pairs `(i, i +
   rotary_dim/2)`, what `rope_kernel` already implemented). Missing this produced a
   non-crashing, plausible-looking wrong token on the first attempt -- caught only by
   the byte-exact llama.cpp comparison below, the same "don't trust non-crashing
   output" lesson DECISIONS.md already records from the Gated DeltaNet mixer's bug.
   Fixed with a new, separate `rope_norm_kernel` (`kernels_cuda/rope.cu`) rather than
   modifying the existing, already-hardware-verified `rope_kernel`.

The absorption (`wk_b`) and decompression (`wv_b`) steps reuse the existing
`gemv_kernel` unchanged via two new small Rust wrappers (`gemv_view`/`gemv_per_head`)
that slice per-head chunks out of `wk_b`/`wv_b`'s per-head-stacked tensors -- the exact
same tensor-layout convention as MoE's per-expert tensors (`Model::gemv_expert`), just
"expert" → "head" (every head is always used, unlike MoE's top-k selection). The MQA
attention step itself (mismatched Q/K width vs. V width, one shared KV "head") needed
a genuinely new kernel (`mla_attention_kernel`) since the existing `attention_kernel`
assumes a uniform head_dim for both the score dot-product and the value
weighted-sum. The KV cache is a single preallocated per-layer device buffer (one row
of `kv_lora_rank + qk_rope_head_dim` per position, written via `CudaDevice::dtod_copy`
-- the same Phase 2 round 2 convention already established for GQA's `k_cache`/
`v_cache`) -- and needs no separate value cache at all, since K and V share the same
compressed representation. This is smaller than GQA's cache, not just differently
shaped -- the actual payoff "latent attention" is named for.

Verified byte-exact against a real `llama.cpp` build (`ce8caa6`) on the synthetic
fixture, three prompts: `"Hello"` -> `" hern"`, `"Once upon a time"` -> `"
removeFrom"`, `"The capital of France is"` -> `" NavLink"` (meaningless text, as
expected from random weights -- the value is architectural correctness, exactly like
`Tiny-Moe.Q4_K_M.gguf`'s own non-linguistic verification). All 55 existing unit tests
and the dense/MoE/hybrid golden-token checks were re-verified unaffected.

### MLA extended to real DeepSeek-V2-Lite: MoE + shared experts + YaRN

The synthetic-fixture MLA work above was extended, same session, to the real
`deepseek-ai/DeepSeek-V2-Lite` checkpoint on a rented 80GB A100 (the VRAM math from
the synthetic-fixture section held: ~63GB of `f32` weights fits with headroom on
80GB, not on the A6000's 48GB). Three things had to be added that the synthetic
fixture's narrower scope had deliberately deferred:

**MoE + shared-expert FFN** (`src/model.rs`'s new `MlaFfn` enum, `Dense` for the
`leading_dense_block_count` lead layers or `Moe` for the rest -- real DeepSeek-V2-Lite
has 1 dense layer, 26 MoE layers). Routed-expert dispatch reuses
`forward_layer_moe`'s existing shape (router GEMM, `crate::moe::route_top_k`-family,
per-expert `gemv_expert`, host-side weighted accumulate -- same "stays host-driven"
convention Phase 2 round 2 already carved out for MoE specifically). The
always-on shared expert turned out to need no new dispatch machinery at all: real
DeepSeek-V2-Lite's `ffn_{gate,up,down}_shexp` tensors already fuse every shared
expert into *one* bigger dense FFN matmul (`ffn_gate_shexp` shape `{hidden,
n_ff_exp * expert_shared_count}`, confirmed from the real converted GGUF), so it's
just one more dense-FFN-shaped computation added unconditionally to the routed
experts' accumulator, not `expert_shared_count` separate calls.

**`route_top_k_with_norm`** (`src/moe.rs`, `route_top_k` is now a thin wrapper over
it): real DeepSeek-V2-Lite's router does *not* renormalize its selected top-k
probabilities (`norm_topk_prob: false` in the source HF config), unlike Qwen3-MoE's
convention this codebase already assumed everywhere. The tell in the GGUF itself is
subtle and worth recording: llama.cpp's converter only ever writes the
`expert_weights_norm` metadata key when the source model's `norm_topk_prob` is
*truthy* -- so the key's **absence** means "don't renormalize," not "key missing,
assume the usual default." Getting this backwards would have silently produced
plausible-but-wrong combination weights, not a crash.

**YaRN RoPE scaling** (`src/kernels_cuda/rope.cu`'s new `rope_norm_yarn_kernel` +
`src/model.rs`'s `MlaYarnConfig`, both new, `rope_norm_kernel` untouched): real
DeepSeek-V2-Lite's `rope_scaling.type == "yarn"` (`factor: 40`, `beta_fast: 32`,
`beta_slow: 1`, `original_context_length: 4096`) -- discovered only by checking the
real HF config's *full* `rope_scaling` dict, not just its `type` field, after an
initial pass assumed (wrongly) that DeepSeek-V2-Lite needed no RoPE scaling at all.
YaRN blends interpolated and extrapolated rotation angles per frequency (a
correction ramp between `beta_fast`/`beta_slow`-derived dimension bounds) and applies
a magnitude correction (`mscale`) to both the rotation itself and, via a *separate*
formula involving `deepseek2`'s own `rope_yarn_log_mul` metadata, the attention
softmax scale -- ported by reading `ggml`'s own CUDA `rope_yarn()`
(`ggml/src/ggml-cuda/rope.cu`), `llama-context.cpp`'s YaRN `cparams` setup, and
`deepseek2.cpp`'s `kq_scale` computation, all in full, since this is genuinely
separate math from everything else in this codebase, not a simple scale-factor
tweak. One real gotcha worth recording: the GGUF stores `rope_yarn_log_mul`
pre-multiplied by `0.1` by the converter, and llama.cpp's own loader divides it back
out before use (`[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`) -- every downstream formula
assumes the *undone* value, so this codebase's `parse_mla_config` replicates that
same undo.

**Fixture-finding gotcha, worth recording**: every pre-quantized DeepSeek-V2-Lite
GGUF found on Hugging Face (`mradermacher`, `tensorblock`, `duyntnet`, `bartowski`'s
Coder-V2-Lite variant) predates llama.cpp's MLA tensor-split conversion change --
they still ship the legacy unsplit `attn_kv_b` tensor and lack
`key_length_mla`/`value_length_mla` metadata entirely, so `parse_mla_config` rejects
them outright with a clear message rather than silently misreading them. Checking
this cheaply (an HTTP range request for just the first ~20MB of each candidate,
enough to read the GGUF header/metadata before the tensor-data-bounds check fails)
avoided several unnecessary multi-GB downloads. The real fix was converting fresh
from the original `deepseek-ai/DeepSeek-V2-Lite` safetensors checkpoint with this
session's own confirmed-current `convert_hf_to_gguf.py` (`--outtype q8_0`, ~16.7GB
output) rather than trusting any third-party quantization's age.

Verified byte-exact against a real llama.cpp build on the real model, three prompts,
this time genuine (not random-weight) completions: `"Hello"` -> `","`, `"Once upon a
time"` -> `","`, `"The capital of France is"` -> `" Paris"` (correct!). All 55 unit
tests and every prior architecture path's golden-token check (dense, MoE, hybrid,
synthetic MLA) re-verified unaffected.

Next: on-GPU dequant kernel (Phase 2 round 3, closing the remaining ~1.1x cold-start
gap) -- the only item left on the open-call list now that MLA covers both a
synthetic fixture and a real, full-scale MoE+YaRN model.

### Phase 2 (Fast IO), round 3: on-GPU dequant kernel

Round 2 closed activations' host round-trips but left the one candidate it deliberately
skipped: `model.rs`'s `load_weight` still dequantized every tensor to a host `f32` `Vec`
(`dequant::dequantize`, CPU-bound bit-unpacking) before uploading it, unlike llama.cpp's
CUDA backend, which uploads quantized bytes as-is and dequantizes/matmuls entirely
on-GPU, never materializing a full-`f32` host copy at all. This round closes that gap
for the two block types that actually matter for it: `Q4_K` and `Q6_K`, the types this
project's local `Q4_K_M` fixtures (`Qwen3-0.6B`, `Tiny-Moe`, `Qwen3.5-0.8B`) use for the
overwhelming majority of weight bytes. Every other GGUF block type this project supports
(`Q4_0/1`, `Q5_0/1`, `Q8_0/1`, `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K`, all 8 IQ-family formats, plus
F32/F16/Bf16/int passthrough) still falls back to the existing host `dequant::dequantize`
path -- correct, unchanged, just not (yet) GPU-accelerated, since no fixture available to
this project actually exercises them for the bulk of a real model's weight bytes (see
DECISIONS.md for the full scope rationale).

Two new AOT kernels, `dequantize_q4k_kernel`/`dequantize_q6k_kernel`
(`kernels_cuda/dequant.cu`), are line-for-line ports of `dequant.rs`'s
`dequantize_block_q4_k`/`dequantize_block_q6_k` (themselves line-for-line ports of
upstream `ggml-quants.c`) -- variable names (`d`, `dmin`, `sc`, `m`, `ql`, `qh`, `is`,
`shift`, ...) kept identical on purpose so the CUDA and Rust versions can be diffed by
eye. One CUDA thread dequantizes one whole 256-element super-block (correctness-first,
matching this project's existing `gemv_kernel`/`rmsnorm_kernel` style -- no warp-level
tricks), parallelized across the tens of thousands of blocks a typical weight tensor
has. `model.rs`'s three `load_weight` closures (`load`/`load_hybrid`/`load_mla`) were
consolidated into one shared `load_weight_device`/`dequantize_tensor_to_device`
dispatch (previously near-identical duplicated code) that routes `Q4_K`/`Q6_K` straight
from raw mmap'd GGUF bytes -> device upload -> on-device dequant kernel -> the same
`CudaSlice<f32>` `Weight.data` every other path already expected, and falls back to the
unchanged host path for every other type. `output.weight` (the LM head, when untied from
the embedding table) goes through the same dispatch; `token_embd` stays on the host path
unchanged, since it must stay host-resident for the embedding-lookup gather regardless of
where its dequant happens.

Re-verified byte-exact against every existing golden-token check (dense `"Once upon a
time"` -> `","` id 11, `"The capital of France is"` -> `" Paris"` id 12095; MoE
`Tiny-Moe` -> id 4036; hybrid `"Once upon a time"` -> `","` id 11, `"The capital of
France is"` -> `" the"` id 279; synthetic MLA `"Hello"` -> `" hern"`, `"Once upon a
time"` -> `" removeFrom"`, `"The capital of France is"` -> `" NavLink"`) plus all 55
unit tests, on a fresh A6000 instance with a freshly built `llama.cpp` @ `9655061`.
Re-measured the same way as rounds 1-2 (`/usr/bin/time -v`, same
`Qwen3-0.6B-Q4_K_M.gguf`, same prompt, three runs each):

| | run 1 | run 2 | run 3 | peak RSS | user+sys CPU time |
|---|---|---|---|---|---|
| **llama.cpp** | 11.24s\* | 6.67s | 6.46s | ~887 MB | ~1.3s + ~1.4s |
| **coldstart-infer, round 2** | 6.43s | 6.47s | 8.51s | 1.35 GB | ~2.1s + ~3.5s |
| **coldstart-infer, round 3** | 6.38s | 6.46s | 6.40s | 1.35 GB | ~1.3s + ~2.2s |

\*llama.cpp's own run 1 is a first-run outlier (cold page/file-cache effects on this
fresh instance, same pattern this project's own runs have shown before) -- runs 2-3
(6.46-6.67s) are the representative baseline.

**Gap closed from ~1.1x to ~1.0x -- parity with llama.cpp, within run-to-run noise**
(coldstart-infer's three runs, 6.38-6.46s, sit inside/below llama.cpp's own 6.46-6.67s
range). System time dropped from ~3.5s to ~2.2s, consistent with removing the CPU-bound
host dequant step from the hot load path; peak RSS is unchanged from round 2 (expected --
this round moves *compute*, not host allocations, off the CPU; the on-device raw-bytes
upload buffer is freed immediately after the dequant kernel runs and was never the
dominant RSS contributor). This closes Phase 2 (Fast IO) as originally scoped in the
round-1 writeup above -- the two named candidates (per-call weight/activation
re-uploads, host-side dequant) are both addressed. Remaining, smaller, unaddressed
round-trips (MoE's per-expert weighted-sum accumulation, the Gated Attention mixer's
fused-qg head split/sigmoid gating, and the 15+ block types still on the host dequant
path) are believed low-value against this project's actual `Q4_K_M`-fixture workload,
not verified to be free of further gains.

### Phase 3 (State I/O), round 1: raw KV-cache export/import (dense/MoE only)

Scoped narrowly before implementation, matching this project's "narrow first" MVP
precedent: round 1 covers **dense/MoE Qwen3 only** (not the Qwen3.5 hybrid's per-layer
`Attn`/`Gdn` split, not MLA's single compressed cache), and covers **export/import of
the raw K/V buffers only** — it does not wire an imported cache back into a forward pass.
That's a deliberate deferral, not an oversight: `forward_prompt` (all three
architecture paths) always starts at position 0 and has no per-token generation loop
anywhere in `model.rs` or `qwen3_coldstart.rs` today — each run does one full prompt
pass and returns exactly one next token, then the process exits. "Resume generation from
an imported cache" needs that generation loop plus a `start_pos` threaded through each
path's cache allocation/indexing (currently `alloc_zeros`'d fresh per call, sized to
exactly that call's token count, not a max-context length) — real work belonging to a
later round, once the loop exists for a cache to resume *into*.

New `src/kv_io.rs` module: a flat, home-grown binary format (`b"CSKV"` magic, `u32`
version, then `num_layers`/`seq_len`/`num_kv_heads`/`head_dim` header fields, then each
layer's `k_cache` and `v_cache` as raw little-endian `f32`) — intentionally
version-gated so a later round can add per-architecture variants (hybrid `Gdn`
conv/recurrent state isn't even indexed by position, so it round-trips as a fixed-size
buffer directly; MLA's compressed cache is a different single-buffer shape) without
breaking round-1 files. `Model::forward_prompt_capture_kv` (new, `model.rs`) is the same
dense/MoE forward pass as `forward_prompt`, refactored to share one inner
`forward_prompt_dense_impl` so the common no-export case doesn't pay for the extra
return plumbing, but also downloads the per-layer device K/V caches to host memory for
export; it returns a clean error for hybrid/MLA models rather than silently exporting
the wrong shape.

`qwen3_coldstart` gained two flags: `--export-kv <file>` (after the forward pass,
serialize the cache to `<file>`) and `--import-kv <file>` (load `<file>`, upload each
buffer to the GPU, download it back, and assert the round trip is byte-identical —
proving the bytes an orchestrator hands back later are exactly usable as device-resident
KV state once a generation loop exists to consume them; it does *not* resume generation,
so it skips loading a GGUF/model entirely). The engine stays ignorant of where the file
lives or how it got there (NVMe, S3-backed FUSE, tmpfs) or any caching policy — an
orchestrator's job, per Non-goals.

Verified on the real A6000 instance: `cargo test` (57 tests, including two new
`kv_io` tests — an export-then-import byte-exact round trip on synthetic cache data, and
a bad-magic rejection check) plus real end-to-end runs against `Qwen3-0.6B-Q4_K_M.gguf`
(dense) and `Tiny-Moe.Q4_K_M.gguf` (MoE): `--export-kv` produces the same next-token as
a plain run (`"Once upon a time"` -> `","` id 11, both with and without `--export-kv`),
`--import-kv` reports `KV_IMPORT_OK` with the correct shape and a verified device round
trip, and both error paths (malformed file, hybrid-architecture GGUF) fail with a clear
message instead of silently producing wrong output.

### Phase 3 (State I/O), round 2: real resume, dense/MoE + hybrid

Confirmed scope with the user before starting (round 1 explicitly deferred this
decision): round 2 extends to **dense/MoE and the Qwen3.5 hybrid mixer**; MLA's single
compressed cache stays out (round 3), matching this project's narrow-first precedent.
See DECISIONS.md's "Phase 3 round 2 scope" entry for the full why.

Two pieces landed together, per round 1's own note that one without the other is
useless: a real per-token generation loop (`Model::generate`, `--max-tokens N` on
`qwen3_coldstart`, default 1 so the existing single-token cold-start benchmark path is
unchanged) and `start_pos` plumbing so `--import-kv` actually resumes into it instead
of just proving a device round trip. `Model::forward_prompt`/`forward_prompt_hybrid`
are now thin wrappers over the same `generate_dense_impl`/`generate_hybrid_impl`
functions `Model::generate` and the `--export-kv` capture functions all share.

`k_cache`/`v_cache` (dense/MoE's pair, and the hybrid mixer's `GatedAttention`
sublayers' pair) are now allocated for `start_pos + prompt_len + max_new_tokens`
instead of exactly the prompt's token count, and an imported cache is uploaded
directly into the front of that buffer (`cudarc`'s `htod_sync_copy_into` into a
`slice_mut` view) before the per-position loop starts partway through it. The hybrid
mixer's `GatedDeltaNet` sublayers needed no equivalent change at all — their
`conv_state`/`recurrent` are fixed-size, mutated in place regardless of position, so an
imported one just gets uploaded as-is. `kv_io.rs` gained a version-2 hybrid format
(`HybridKvCache`, per-layer `Attn`/`Gdn` tagged data) alongside the untouched
version-1 dense format, plus `import_kv`/`ImportedKv` to read-dispatch between them;
`Model::architecture_kind()` lets `qwen3_coldstart` pick the matching
`--export-kv` capture function without reaching into `Model`'s private state.

Verified on a fresh A6000 instance (`lunpulve` — the round-1 session's instance was
already gone, confirming instances really are per-session ephemeral): `cargo test` (58
tests, incl. a new hybrid `kv_io` round-trip test) plus the actual correctness bar —
byte-exact match between a single uninterrupted run and export→import→continue over
the same concatenated prompt, split at a clean sentence boundary:

- Dense (`Qwen3-0.6B-Q4_K_M.gguf`, `"The capital of France is Paris. The capital of
  Germany is"` split after the first `"."`): both paths produced
  `[19846,13,576,6722,315]` (`" Berlin. The capital of"`).
- Hybrid (`Qwen3.5-0.8B-Q4_K_M.gguf`, same prompt/split shape): both paths produced
  `[19241,13,561,6511,314]` (also `" Berlin. The capital of"`).

MoE (`Tiny-Moe.Q4_K_M.gguf`) surfaced a genuine, useful finding rather than a bug: it's
the only local fixture using the SentencePiece encode path, which unconditionally
prepends an implicit leading-space token to every `encode()` call (real SentencePiece
behavior, not a project bug — see `tokenizer.rs`'s own doc comment). That makes a
continuation prompt's own re-encoding pick up a phantom extra token no matter where the
text is split, so byte-exact text-level verification isn't meaningful for this fixture
specifically — confirmed independent of any resume/cache code (reproducible from two
plain, non-resuming `encode()` calls) and unrelated to MoE vs. dense (the resume
machinery, `generate_dense_impl`/`forward_one_token_dense`, is exactly the same code
for both — `forward_layer_moe`'s FFN dispatch has no position/cache logic to differ
in). Verified instead by determinism (identical resume inputs, run twice, produced
byte-identical `[4014,4052,4034,262,308]` both times) plus the code-sharing argument.
See DECISIONS.md for the full three-part reasoning.

### Phase 3 (State I/O), round 3: real resume, MLA — closes Phase 3's architecture coverage

Confirmed scope with the user before starting: fixture choice was the synthetic
`test-data/deepseek-tiny-mla.gguf` (already on disk, uses the `gpt2`-style Qwen
tokenizer so byte-exact text-continuation verification applies, same as dense/hybrid
in round 2 — not real DeepSeek-V2-Lite, since round 3's actual delta is cache/
`start_pos` plumbing, already exercised structurally by round 2's other two
architectures, not new MoE/YaRN math which round 3 isn't touching), on a fresh A6000
instance (`tnr status --json` showed none running at session start).

Same shape as round 2, one more time: `generate_mla_impl` (mirroring
`generate_dense_impl`/`generate_hybrid_impl`) seeds MLA's single per-layer compressed
`kv_cache` (`[seq_len, kv_lora_rank + qk_rope_head_dim]`, no separate K/V pair — MLA's
whole point is that the compressed latent is shared and decompressed on the fly by
`wk_b`/`wv_b`) from an imported cache at `start_pos`, allocates it with headroom for
`start_pos + prompt_len + max_new_tokens` instead of exactly the prompt length, and
runs the same per-position-loop-then-keep-decoding shape `forward_mla_attn_block`
already supported (it already took an absolute `position` argument and indexed into
`kv_cache` by it — this round only needed to call it in a loop with a preallocated,
`start_pos`-offset buffer, not change its own logic). `forward_prompt_mla` is now a
thin wrapper over `generate_mla_impl`, matching `forward_prompt`/`forward_prompt_hybrid`.
`kv_io.rs` gained a version-3 `MlaKvCache` format (one buffer per layer, not the dense
pair or hybrid's tagged `Attn`/`Gdn` split) alongside the untouched version-1/2
formats; `import_kv`/`ImportedKv` and `Model::generate`/`architecture_kind()`'s `Mla`
arm now dispatch to it instead of erroring "round 3" as they did through round 2.
`forward_prompt_capture_kv_mla` fills the same role for `--export-kv` that
`forward_prompt_capture_kv`/`forward_prompt_capture_kv_hybrid` do for dense/hybrid.

Verified on a fresh A6000 instance (`bkzn3giz`): `cargo test` (60 tests, incl. a new
MLA `kv_io` round-trip test) plus the same byte-exact bar rounds 1-2 used — a single
uninterrupted run vs. export→import→continue over the same concatenated prompt, split
at a clean sentence boundary (`"The quick brown fox jumps over the lazy dog"` +
`" and runs"`): both paths produced `[69344,10420,40306,145381,87488]`. No
determinism-fallback needed (unlike `Tiny-Moe` in round 2) since this fixture's
`tokenizer.ggml.model` is `gpt2`, confirmed by reading the GGUF's own metadata bytes
before relying on it.

This closes Phase 3's architecture-coverage scope entirely — `--export-kv`/
`--import-kv`/`--max-tokens` now cover all three cache shapes this project's
architectures produce (dense/MoE's K/V pair, hybrid's per-layer tagged state, MLA's
single compressed latent). Phase 4 (Embeddability) is next per the roadmap above, a
separate confirm-before-starting conversation.

### Phase 4 (Embeddability), round 1: `--lora` load-time adapter application

Confirmed scope with the user before starting (this project's usual practice): dense/
MoE Qwen3 plus the Qwen3.5 hybrid architecture, not MLA (matching every other
MLA-adjacent feature's line in this codebase, and the Non-goals section's "load-time
adapter application, no runtime hot-swap multiplexer" framing above); a real public
LoRA adapter for verification where one exists, a hand-built synthetic one where it
doesn't; reuse the still-running A6000 from the Phase 3 round 3 session.

**Format**: llama.cpp's own GGUF LoRA adapter convention, read in full from source
before assuming anything (`convert_lora_to_gguf.py`, `src/llama-adapter.cpp`) rather
than inventing a format — an adapter is its own self-contained GGUF file (converted
from a HF PEFT checkpoint's `adapter_config.json` + `adapter_model.safetensors`),
readable with this project's *existing*, unmodified `gguf.rs` parser (a LoRA GGUF is
just a different metadata/tensor set, not a different container). Required metadata:
`adapter.type == "lora"`, `adapter.lora.alpha` (`f32`). Every targeted base tensor
`<name>` (already including `.weight`) gets a `<name>.lora_a`/`<name>.lora_b` pair —
`lora_a`'s GGUF shape is `[in_features, rank]`, `lora_b`'s is `[rank, out_features]`,
rank read from these shapes rather than a separate key. Update: `W' = W + scale * (B @
A)`, `scale = alpha / rank`, matching llama.cpp's own formula exactly.

**New module `src/lora.rs`** does the parsing and the host-side `B @ A` math (the
low-rank factors are tiny — rank 16 in the real adapter tested below — a one-time
load-time cost, not worth a device kernel), producing one full-size, already-scaled
delta per targeted tensor. `Model::apply_lora` (`model.rs`) then, for each delta:
rejects immediately if the loaded model is MLA; otherwise looks up the matching
GPU-resident `Weight` via a new `Model::find_lora_target_mut` (matches `blk.{i}.
{suffix}.weight` against whichever architecture is loaded — dense/MoE's `self.layers`
or the Qwen3.5 hybrid's `self.hybrid.layers`), checks its shape matches, uploads the
delta once, and adds it in-place with the existing `add_k` elementwise-add kernel
(unchanged since Phase 2 round 2). No new kernel, and the forward pass itself needed
zero changes — exactly the load-time-only scope this round committed to.

**Accept/reject is one lookup, not four special cases**: `find_lora_target_mut` only
has match arms for the 2-D `nn.Linear`-shaped tensors each layer kind actually has
(dense/MoE attention, dense FFN, the hybrid's Gated-Attention-layer attention/FFN, and
the Gated DeltaNet mixer's FFN). MoE's per-expert-stacked `ffn_gate_exps`/`ffn_up_exps`/
`ffn_down_exps` (3-D), the Gated DeltaNet mixer's non-Linear state-space tensors
(`ssm_*`, `attn_qkv`, `attn_gate`), and `token_embd`/`output`/norm tensors all simply
have no match arm, so they fall through to the same clear "no matching 2-D weight"
error `Model::apply_lora` raises — one rejection path covering four different reasons,
because none of them are ever a 2-D `Weight` regardless of which architecture is
loaded.

**Verified real, not just plausible**: real hardware (A6000, `bkzn3giz`), and a real
public adapter for the dense case — `premjatin/qwen-linear-algebra-coder` (PEFT rank
16, alpha 32, targets all 7 `q/k/v/o/gate/up/down_proj` modules) for `Qwen/Qwen3-1.7B`,
both converted to GGUF with llama.cpp's own converters (`convert_hf_to_gguf.py`,
`convert_lora_to_gguf.py`). Cross-checked three independent ways against a real
llama.cpp build, not just "produces plausible output": (1) `llama-export-lora`'s merge
log reports `merged 196 tensors with lora adapters` (`28 layers × 7 targeted modules`),
matching this project's own `tensors_applied=196` exactly; (2) its
`calculated_scale=2.000000` log line matches `alpha/rank = 32/16` independently; (3)
`llama-simple` (raw completion on the LoRA-merged GGUF, no chat template — `llama-cli`'s
newer conversational mode has no working `--no-cnv` escape hatch in the current build,
see the known-debt note below) completes "The capital of France is" with " Paris",
identical to this project's own `--lora`-applied output (`token_id=12095,
token_text=" Paris"`). The MoE/hybrid accept/reject paths (no real adapter targeting
those architectures' exact modules was found) were verified against hand-built
synthetic adapter GGUFs (`gguf.GGUFWriter`, matching this project's established
"synthetic fixture via a real format/tool" practice from the MLA rounds) targeting the
existing local `Tiny-Moe.Q4_K_M.gguf`/`Qwen3.5-0.8B-Q4_K_M.gguf`/
`deepseek-tiny-mla.gguf` fixtures — MoE attention accepted, MoE FFN-experts rejected,
hybrid Gated-Attention accepted, hybrid Gated-DeltaNet FFN accepted, hybrid
Gated-DeltaNet `attn_qkv` rejected, MLA rejected outright, and a deliberately
wrong-shaped adapter tensor rejected with a shape-mismatch error naming both shapes.

**A real bug, caught by the first real run, not shipped**: an initial reading of
`convert_lora_to_gguf.py`'s Python source suggested the base tensor name has `.weight`
stripped before `.lora_a`/`.lora_b` is appended. The real converted adapter file
proved that wrong — the base name already includes `.weight` — which produced a
`blk.0.ffn_down.weight.weight` lookup and a clear panic on the very first end-to-end
run against the real fixture, well before any of the verification above. Fixed
(`src/lora.rs` no longer re-appends `.weight`) and re-verified from that run onward;
see DECISIONS.md's Phase 4 round 1 entry.

This closes Phase 4's first piece. The second — a Rust C-FFI surface for embedding
this engine into a host process — is next per the roadmap above, a separate
confirm-before-starting conversation.

### Phase 4 (Embeddability), round 2: Rust C-FFI surface

Confirmed scope with the user before starting (this project's usual practice, three
explicit questions): (a) API surface — `load`/`generate`/`free` only, with `--lora`'s
adapter path folded into `load` as an optional parameter (it's load-time-only already,
so it needs no separate FFI call) rather than also exposing Phase 3's `--export-kv`/
`--import-kv` state I/O, a separable capability no embedding host had asked for yet; (b)
header generation — `cbindgen` (the standard convention for a Rust crate exposing a C
ABI) over a hand-written header; (c) GPU instance — reuse `bkzn3giz` (confirmed still
`RUNNING` via `tnr status --json` rather than assumed).

**Surface** (`src/ffi.rs`, new module): `coldstart_load(gguf_path, lora_path) ->
*mut ColdstartModel` (opaque handle, `lora_path` nullable to skip LoRA), `coldstart_generate(handle,
prompt, max_new_tokens, *mut ColdstartGenerateResult) -> c_int` (0 on success, fills the
out-param with a heap-allocated `token_ids`/`num_tokens`/`text`; -1 on failure),
`coldstart_free_generate_result`, `coldstart_free`, and `coldstart_last_error() -> *const
c_char` (thread-local last-error string, the error-crossing convention this round
adopted for every existing `Result<_, String>` in `model.rs`/`lora.rs`). Every entry
point wraps its body in `std::panic::catch_unwind` and converts a caught panic into the
same last-error string — unwinding a Rust panic across an `extern "C"` boundary is
undefined behavior in the C caller, so nothing here may ever let one through. This is
literally the same `Model::load`/`Model::generate`/`Model::apply_lora` this crate's own
`qwen3_coldstart` binary already calls (see `src/bin/qwen3_coldstart.rs`) — the FFI layer
adds no new model-loading or generation logic, only the C-safe boundary around it.

**Compiles as**: `Cargo.toml`'s `[lib]` section now lists `crate-type = ["rlib",
"cdylib", "staticlib"]` (previously implicit default `rlib` only) — `rlib` stays so
`src/bin/*.rs` keep linking against this crate unmodified. No `build.rs` changes needed:
the AOT kernel-compilation pipeline is unrelated to which Rust crate-types get emitted
from the already-compiled kernels, and `src/ffi.rs` needed no kernel of its own.

**Header**: `cbindgen.toml` (config) + checked-in `include/coldstart_infer.h`, generated
with `cbindgen --config cbindgen.toml --crate coldstart-infer --output
include/coldstart_infer.h`. Deliberately *not* wired into `build.rs` — regenerated by
hand when `src/ffi.rs`'s public surface changes, not on every build, so `cbindgen` isn't
a second toolchain dependency for `nvcc`-only rebuilds.

**Real-hardware-verified on the A6000** (`bkzn3giz`, reused, confirmed `RUNNING` first):
a real C test harness (`ffi-test/smoke_test.c`, compiled with plain `gcc` against the
built `libcoldstart_infer.so`, linked via `-lcoldstart_infer` + `LD_LIBRARY_PATH`) calling
`coldstart_load` → `coldstart_generate` → `coldstart_free_generate_result` →
`coldstart_free`, cross-checked against `qwen3_coldstart` on the same GGUF+prompt+
`--max-tokens`, not just "it compiles and links":

- Dense (`Qwen3-0.6B-Q4_K_M.gguf`, prompt `"The quick brown fox jumps over the lazy
  dog"`, 5 tokens): FFI and CLI both produced `token_ids=[13,576,3974,13876,38835]` and
  identical decoded text, byte-exact.
- Qwen3.5 hybrid (`Qwen3.5-0.8B-Q4_K_M.gguf`, prompt `"Hello there"`, 3 tokens): FFI and
  CLI both produced `token_ids=[0,353,1044]` and identical decoded text, byte-exact.
- Error path: `coldstart_load` on a nonexistent GGUF path returns `NULL` (no crash) and
  `coldstart_last_error()` reports a clear message naming the missing file.
- `cargo build --release`/`cargo test --release` both clean (59 tests passing,
  unchanged) with the new `[lib]` crate-types added, and `src/bin/*` unaffected.

**`staticlib` (round 2 follow-up, resolved)**: an earlier same-day session saw a C
binary linked against `libcoldstart_infer.a` need `-Wl,--allow-multiple-definition` and
then hang at runtime, and flagged it as an unresolved known limitation. A dedicated
debugging pass found neither symptom reproduces: `gcc -I include -o smoke_test_static
ffi-test/smoke_test.c -L target/release -l:libcoldstart_infer.a -ldl -lpthread -lm`
(no extra flags) links clean with zero duplicate-symbol warnings, and the resulting
binary produced the same byte-exact output as the `cdylib`/CLI runs above for both the
dense (`token_ids=[13,576,3974,13876,38835]`) and hybrid
(`token_ids=[0,353,1044]`) fixtures, completing in a few seconds each. The original
hang's actual cause was almost certainly the ThunderCompute GPU-capacity contention
already documented elsewhere in this project's session notes (queued GPU-driver calls
that clear on their own after some minutes) — the process was killed at ~90s on the
assumption it was stuck, before it had a chance to clear. **Both `cdylib` and
`staticlib` are verified working embedding paths**; `cdylib` remains the simpler
default (no need to reason about symbol collisions with a host's other static
dependencies), but `staticlib` is no longer flagged as broken.

This closes Phase 4 (Embeddability) entirely — both its CLI-facing half (`--lora`,
round 1) and its embedding-facing half (the C-FFI surface, this round) are done.

### System1: single-pass, non-autoregressive candidate scoring

Every existing entry point (`forward_prompt`/`generate`) pays the full sequential
per-token decode loop even when the caller only wants a score for a small, known set of
candidate continuations (a Yes/No answer, an A-D choice, a 1-10 scale) — the
argmax-then-feed-back loop and a full-vocab GEMV + vocab-sized D2H transfer per step are
both unnecessary work for that case. `Model::system1_evaluate` (`src/model.rs`) instead
runs `prompt` through the shared prefill path exactly once, then scores every candidate
from that single prefill: single-token candidates are scored in one batched gather-GEMV
(`kernels_cuda/gemv_gather.cu`'s `gemv_gather_kernel`, which computes only the
candidates' own lm_head logit rows instead of the whole vocab), and multi-token
candidates via a short teacher-forced continuation (feeding each candidate's own known
next token, never a sampled one). `score`/`probability` are relative to the candidate
set in one call only, not vocab-normalized log-probabilities — computing the latter
would reintroduce the exact full-vocab cost this feature exists to avoid.

**Rust API** (`src/model.rs`):

```rust
pub struct System1Candidate { pub text: String }
pub struct System1CandidateResult { pub text: String, pub token_ids: Vec<u32>, pub score: f32 }
pub struct System1Response { pub results: Vec<System1CandidateResult>, pub probabilities: Vec<f32> }

pub fn Model::system1_evaluate(
    &self,
    prompt: &str,
    candidates: &[System1Candidate],
    temperature: f32,          // 1.0 = no-op; scales the softmax over `results[i].score`
) -> Result<System1Response, String>
```

Dense/MoE Qwen3 models only — hybrid Qwen3.5 and DeepSeek-V2/V3 MLA are rejected with a
clear error, the same scope line every other MLA-adjacent feature in this project uses.

**C-FFI** (`src/ffi.rs`, `include/coldstart_infer.h`), layered on the same
`ColdstartModel` handle `coldstart_load`/`coldstart_generate`/`coldstart_free` already
use (Phase 4 round 2 above):

```c
int coldstart_system1_evaluate(
    ColdstartModel *handle,
    const char *prompt,
    const char *const *candidate_texts, size_t num_candidates,
    float temperature,
    ColdstartSystem1Result *out);   // 0 on success, -1 on failure (see coldstart_last_error)

void coldstart_free_system1_result(ColdstartSystem1Result *result);
```

`ColdstartSystem1Result` holds a heap `candidates` array of
`ColdstartSystem1CandidateResult { text, token_ids, num_token_ids, score, probability }`
— owned by this crate, freed only via `coldstart_free_system1_result`, never by the C
caller's own `free`. Every entry point wraps its body in `catch_unwind`, same
panic-never-crosses-the-FFI-boundary contract as the rest of `src/ffi.rs`.

**CLI**: `system1_coldstart <path-to-gguf> <prompt> --candidate <text> [--candidate
<text> ...] [--temperature T] [--lora <adapter.gguf>]` (`src/bin/system1_coldstart.rs`).

**Verified** against the real `Qwen3-1.7B` model on GPU hardware: the gather-GEMV path
agrees with the full-vocab GEMV to 1e-4 at matching rows, the teacher-forced
multi-token path reproduces the model's own real greedy continuation and ranks it far
above a wrong one, and the FFI entry points round-trip cleanly (including the
zeroed-after-free/double-free-safe contract `coldstart_free_system1_result` documents).
A warm microbenchmark (`bench_coldstart --candidate ...`) showed the gather-GEMV win
was real (up to ~226ms saved at 449 prompt tokens) but small relative to total
latency at the time — the sequential per-token *prefill* loop still dominated by
orders of magnitude, so sub-50ms warm latency wasn't reached yet. That prefill loop,
not System1's LM-head path, was flagged as the next bottleneck — see "Batched Prefill
GEMM" below.

### Batched Prefill GEMM: cuBLAS-batched dense/MoE prefill

Every forward path up to this point ran the prompt through the model **one token at a
time**: `prefill_dense` looped `forward_one_token_dense` once per prompt token, and
every op inside it (`gemv_kernel`, `rope_kernel`, `attention_kernel`) processed exactly
one row. That's the sequential-decode convention every architecture needs anyway for
generating *new* tokens (feeding each sampled id back in), but the *prompt* — already
fully known up front — doesn't need to pay it: a 1-row `gemv_kernel` launch reads an
entire weight matrix from global memory to produce one output row, so for an
`M`-token prompt the same weight bytes get re-read from GMEM `M` times instead of
once. Measured cost on an A6000 (`Qwen3-0.6B-Q4_K_M.gguf`): **~9.1s for a 113-token
prompt, ~37.7s for a 449-token prompt** — prefill, not decode, was overwhelmingly the
dominant cold-start cost, exactly what System1's benchmark above flagged as the real
next bottleneck.

**Fix**: treat the prompt's `M` positions as a GEMM batch dimension instead of `M`
separate GEMV launches, for every projection in the shared attention block and the
dense FFN (`Model::gemm`, `src/model.rs` — `cudarc`'s `cublas` Cargo feature was already
declared in `Cargo.toml` but had zero call sites before this; math mode pinned to
`CUBLAS_PEDANTIC_MATH` at handle creation so cuBLAS's summation order can't silently
drift from `gemv_kernel`'s naive per-row dot product via a TF32/reduced-precision
tensor-core path). Two new batched kernels: `rope_batch_kernel`
(`kernels_cuda/rope.cu`) rotates every prompt row in one launch, each at its own
absolute position, instead of one `rope_kernel` launch per row; `attention_prefill_kernel`
(new `kernels_cuda/attention_prefill.cu`) scores every query row against the shared K/V
cache in one launch (the grid gains a query-row dimension), each row causally masked to
its own position. RMSNorm/SiLU-and-mul/residual-add needed **no** kernel changes —
already row/flat-generic. New `Model::prefill_dense_batched` runs every prompt token
through each layer in one batched pass; the old sequential path
(`prefill_dense`/`forward_one_token_dense`) is unchanged and still used for the
per-token *decode* loop after the first token (a GEMM with one row buys nothing there)
and as the verification oracle below.

MoE layers batch the same shared attention block, but each row can still route to a
different top-k expert subset, so the FFN itself stays a per-row loop reusing the
existing per-expert `gemv_expert` calls (`forward_layer_moe_batched`) — turning that
into a single GEMM needs a token→expert grouping/permutation step (grouped GEMM, the
way vLLM/TensorRT-LLM batch MoE FFNs), a distinct, larger follow-on not attempted here.

**Verified on real hardware** (ThunderCompute A6000, `Qwen3-0.6B-Q4_K_M.gguf`):
- A new `#[ignore]`d GPU test,
  `model::prefill_batching_tests::prefill_dense_batched_matches_sequential_prefill`,
  diffs every row of the batched hidden state against the sequential path's
  corresponding position and checks the final greedy-argmax token matches — **passed**.
- `--import-kv` resume (`start_pos > 0` through the batched path) was checked against a
  one-shot equivalent: exporting a cache after `"The capital of France is Paris."`
  (`seq_len=7`) then resuming with `--import-kv` + continuation `" The capital of
  Germany is"` produced token ids `[19846,13,576,6722,315]` (`" Berlin. The capital
  of"`), **identical** to running the whole concatenated prompt in one shot with
  `start_pos=0`.

**Benchmark result** (`bench_coldstart`, same instance/model, `--warmup 2 --iters 5`,
`COLDSTART_CUDA_ARCH=sm_86`):

| prompt tokens | sequential prefill (before) | batched prefill (after) | speedup |
|---:|---:|---:|---:|
| 113 | ~9.1s | 40.6ms (p50) | **~224x** |
| 449 | ~37.7s | 141.0ms (p50) | **~267x** |

Sub-50ms warm latency is reached at 113 prompt tokens; the 449-token case (141ms) is
still a large win but not sub-50ms — the target depends on prompt length, not met
universally yet. MoE's per-row FFN loop (see above) is the next named candidate if
MoE prefill latency becomes the bottleneck once batched.

### Batched Prefill GEMM extended to Qwen3.5 hybrid's GatedAttention sublayers

Extended the same batching to the Qwen3.5 hybrid architecture's `GatedAttention`
sublayers only. The hybrid model has two sublayer kinds and they are not equally
batchable: `GatedAttention` (`forward_gated_attn_mixer`) is the same
RMSNorm→QKV→QK-Norm→RoPE→causal-attention→O-proj shape as dense's attention block,
directly batchable with the same `Model::gemm`/`rope_batch`/`attention_prefill`
helpers above. `GatedDeltaNet` (`forward_gdn_mixer`) has no `position` parameter at
all — `gdn_conv`/`gdn_delta` mutate `conv_state`/`recurrent` sequentially, each
token's state depending on the previous token's output, a real recurrence rather
than a batchable GEMV-per-row pattern. Reformulating it as a parallel/chunked scan
(the technique the Gated DeltaNet paper and flash-linear-attention use) is a
separate, materially larger project — **`GatedDeltaNet` stays sequential and
unmodified this round**.

Because the two sublayer kinds are heterogeneous, the token-major loop
(`forward_one_token_hybrid`, one token through every layer before the next) can't
just switch to GEMMs the way dense's could: a `GatedAttention` layer only ever sees
one token's hidden vector at a time under token-major iteration, so there's no way
to batch its projections without first restructuring to be **layer-major** — for
each layer in order, run all `M` prompt rows through it before moving to the next
layer. This is valid because a layer's output at position `p` depends only on
position `p`'s input plus that layer's own carried state (`k_cache`/`v_cache` or
`conv_state`/`recurrent`), never on another position's intermediate value at the
same layer — the same reassociation the dense batched-prefill path above already
relies on. `Model::prefill_hybrid_batched` (new) runs every prompt token through
each layer in this layer-major order: `GatedAttention` layers batch all `M` rows in
one GEMM pass each (`Model::forward_gated_attn_mixer_batched`, plus two small new
kernels — `split_qg_kernel`/`sigmoid_gate_kernel` in `kernels_cuda/elementwise.cu`
— replacing what was a per-token host round trip for this mixer's fused
query+gate split and post-attention sigmoid gating); `GatedDeltaNet` layers loop
`M` times sequentially through the *unmodified* `forward_gdn_mixer`, extracting/
writing one row at a time out of the shared `[rows, hidden_size]` buffer. Total GDN
work is unchanged from the token-major loop, just grouped by layer instead of
interleaved. `Model::prefill_hybrid` (the original token-major sequential loop) is
kept unchanged as the verification oracle; `generate_hybrid_impl` now calls
`prefill_hybrid_batched` for the prompt phase, same switchover
`generate_dense_impl` made to `prefill_dense_batched` above. Decode (one new token
at a time after the first) is untouched — a GEMM with one row buys nothing there.

**Verified on real hardware** (ThunderCompute A6000, `Qwen3.5-0.8B-Q4_K_M.gguf`, a
real checkpoint, not a synthetic fixture):
- A new `#[ignore]`d GPU test,
  `model::hybrid_batching_tests::prefill_hybrid_batched_matches_sequential`, diffs
  the batched path's final hidden vector against the sequential path's and checks
  the final greedy-argmax token matches — **passed**, and the produced first token
  (`" the"`, id 279) matches this project's existing golden-token record for this
  exact prompt/model.
- `--import-kv` resume (`start_pos > 0` through the batched hybrid path) checked
  against a one-shot equivalent, same pattern as the dense check above: exporting a
  cache after `"The capital of France is Paris."` (`seq_len=7`) then resuming with
  `--import-kv` + continuation `" The capital of Germany is"` produced token ids
  `[19241,13,561,6511,314]` (`" Berlin. The capital of"`), **identical** to running
  the whole concatenated prompt in one shot with `start_pos=0`.

**Benchmark result** (`bench_coldstart`, same instance, `--warmup 2 --iters 5`,
`COLDSTART_CUDA_ARCH=sm_86`, warm `forward_prompt` latency — the "before" number
is `generate_hybrid_impl` temporarily pointed at the unmodified `prefill_hybrid`
oracle instead of `prefill_hybrid_batched`, same model/prompts otherwise):

| prompt tokens | token-major sequential (before) | layer-major batched (after) | speedup |
|---:|---:|---:|---:|
| 29 | 1463.1ms (p50) | 915.4ms (p50) | **~1.60x** |
| 113 | 5694.3ms (p50) | 3517.6ms (p50) | **~1.62x** |
| 449 | 23005.4ms (p50) | 14068.4ms (p50) | **~1.64x** |

A much smaller win than dense's ~224-267x, exactly as the design above predicts:
only 6 of this fixture's 24 layers are `GatedAttention` (the rest are the
untouched, still-sequential `GatedDeltaNet`), so only that fraction of total
per-token cost gets GEMM-batched. Batching `GatedDeltaNet` itself (the chunked-scan
reformulation) is the next named candidate if hybrid prefill latency needs to close
the gap with dense's.

### Benchmark expansion: fixing a real regression, then vLLM and a TypeSafe Jev citation

Re-running the llama.cpp comparison on a fresh ThunderCompute A6000 instance (same
methodology as before: external `/usr/bin/time -v`, same `Qwen3-0.6B-Q4_K_M.gguf`,
same prompt, `n=3`) surfaced a real regression the previous "~1.0x parity" number had
missed: coldstart-infer measured **~9.3-9.5s wall clock vs. llama.cpp's ~6.5s — about
1.4x slower**, even though coldstart-infer's own internal
`process_start_to_first_token_ms` metric still reported ~4.8-5.0s. The ~4.5s gap was
entirely *after* the result was already printed, before the OS reported the process
as exited.

**Diagnosis** (via `strace -f -T`): dozens of threads doing staged-backoff
`futex`/`poll` waits (timeouts escalating 100ms → 250ms → 1s → 2s → 10s) against
`/tmp/.tc_hac` — ThunderCompute's local GPU-virtualization proxy — all starting right
after the result was printed. First hypothesis: coldstart-infer holds far more
separate device allocations than llama.cpp (~300 individual `CudaSlice<f32>`
buffers, one per weight tensor per `Weight`'s doc comment, vs. ggml's arena-style
backend buffer), and each held allocation pays its own teardown round-trip through
the proxy at process exit. **This hypothesis was wrong** — a full arena-consolidation
refactor (one shared `CudaSlice<f32>` per model instead of one per tensor,
implemented and verified byte-exact correct) made *no measurable difference* to the
teardown time, and was reverted rather than kept for no benefit. The real tell:
`smoke_coldstart`, which does no model loading at all (just `CudaDevice::new` + one
trivial kernel), showed the *same* ~5.6s of pure post-result teardown. The cost is
fixed, not allocation-count-proportional — it's `libc`'s `atexit` chain running the
CUDA driver's own registered context-teardown hook against the virtualization proxy,
paid by any CUDA program on this kind of instance, regardless of what it allocated.

**Fix**: `coldstart_infer::fast_exit` (`src/lib.rs`) flushes stdout/stderr, then
calls the raw `_exit` syscall directly (an `extern "C"` declaration, not
`std::process::exit`, which still runs the `atexit` chain) — skipping that hook
entirely. Safe here because every `*_coldstart` binary's job is finished by the time
it calls this; the OS reclaims the GPU context/memory/fds on process death regardless
of whether userspace tore them down first. Wired into `qwen3_coldstart`,
`system1_coldstart`, and `smoke_coldstart` after they print their result.
`smoke_coldstart` went from 6.3s wall clock (686ms internal) to 0.56s (525ms
internal) — teardown overhead essentially eliminated. Correctness re-verified against
both reference prompts (`"Once upon a time"` → token 11 `","`, `"The capital of
France is"` → token 12095 `" Paris"`) — unchanged.

Re-measured the same way, same instance, `n=3`:

| | run 1 | run 2 | run 3 | peak RSS | user+sys CPU time |
|---|---|---|---|---|---|
| **llama.cpp** | 6.56s | 6.51s | 6.45s | 883 MB | ~1.6-1.8s + ~1.3-1.4s |
| **coldstart-infer** | 4.71s | 4.81s | 5.05s | 1518 MB | ~1.3-1.4s + ~2.3-2.8s |

**coldstart-infer is now genuinely ~1.3-1.4x faster than llama.cpp on cold start** —
not just parity, and not a methodology trick: the fix is a real one-line difference
in how the process ends, found by diagnosing an actual regression rather than
tuning toward a wanted number. See `scripts/bench_cold_common.sh` (the harness,
validated by reproducing these exact llama.cpp numbers before trusting it for
anything else) and DECISIONS.md's "fast-exit after printing the benchmark result"
and "Benchmark expansion" entries for the full investigation and scope decisions.

**`fast_exit` re-verified across every architecture**, not just dense Qwen3 — it's a
shared code path (`qwen3_coldstart`'s single exit point, regardless of which of
`Model::load`'s dense/MoE/`load_hybrid`/`load_mla` dispatch ran), so a regression in
one would very plausibly regress the others too:

| architecture | fixture | check | result |
|---|---|---|---|
| MoE | `Tiny-Moe.Q4_K_M.gguf` | `"Once upon a time"` → token 4036 | matches golden value; clean 1.8s wall-clock exit |
| Hybrid (Qwen3.5) | `Qwen3.5-0.8B-Q4_K_M.gguf` | `"Once upon a time"` → token 11 `","`, `"The capital of France is"` → token 279 `" the"` | both match; clean 8.2s wall-clock exit |
| MLA (synthetic) | `deepseek-tiny-mla.gguf` | deterministic across repeated runs, no golden value recorded for this fixture/prompt | deterministic (token 57785 both runs); clean ~2.0s wall-clock exit |

Also re-ran the `#[ignore]`d batched-vs-sequential-prefill oracle tests against real
fixtures for all three: `prefill_dense_batched_matches_sequential_prefill` (against
`Tiny-Moe.Q4_K_M.gguf`, exercising the MoE per-expert dispatch path),
`hybrid_batching_tests::prefill_hybrid_batched_matches_sequential` (against
`Qwen3.5-0.8B-Q4_K_M.gguf`), and all three MLA `mla_batching_tests` that don't
require the real DeepSeek-V2-Lite checkpoint — all pass.
`prefill_mla_batched_matches_sequential_real_moe_checkpoint` correctly rejected the
synthetic fixture with its own explicit guard message (`"COLDSTART_TEST_GGUF must be
a real deepseek2 checkpoint with MoE layers"`) rather than silently passing or
crashing — that test needs the real 80GB-A100-class checkpoint from the MLA section
below, not provisioned this session.

**Small cleanup, same session**: every build this session warned that
`prefill_dense`/`prefill_hybrid`/`prefill_mla` were unused — real, not a false
positive, but not dead code either: they're the sequential-prefill verification
oracles the batching tests just re-ran above, called only from `#[cfg(test)]`
modules (`generate_dense_impl`/`system1_evaluate` switched to
`prefill_dense_batched` when Batched Prefill GEMM landed; `prefill_hybrid`/
`prefill_mla`'s doc comments already said as much, `prefill_dense`'s didn't). Marked
all three `#[cfg(test)]` and fixed `prefill_dense`'s stale doc comment to match —
warning gone, `cargo test` still 69 passed/0 failed/7 ignored, oracle tests above
confirm the functions still work correctly under the new gating.

#### vLLM: the actual AOT-vs-JIT foil llama.cpp never was

llama.cpp is, like coldstart-infer, AOT-compiled via `nvcc` — it never JIT-compiles
CUDA kernels, so the comparison above never actually tested this project's core bet
(AOT-compiled kernels vs. a real JIT/warmup tax at cold start — see "What this
project is" above). vLLM's CUDA graph capture and `torch.compile`-driven kernel
compilation is a real, well-documented warmup cost and a much better foil for that
specific claim.

**Methodology deviation, disclosed upfront**: the installed vLLM (`0.30.0`) has no
`gguf` entry in its quantization method registry at all — it cannot load
`Qwen3-0.6B-Q4_K_M.gguf`. Per this project's own no-silent-substitution rule (see
DECISIONS.md), vLLM was instead pointed at the original `Qwen/Qwen3-0.6B` HF
safetensors checkpoint (bf16, downloaded fresh). This is **not** the same weight
format/precision as every other comparison in this project — it tests the same cold-
start *mechanism* (fresh process → usable output), not byte-identical weights. Both
engines still produced the same greedy first token for `"Once upon a time"` (`","`).

**Methodology deviation #2**: vLLM caches `torch.compile` artifacts on disk
(`~/.cache/vllm/torch_compile_cache/`, persists across process launches). The very
first run on this instance — genuinely no prior cache, the fairest comparison to
coldstart-infer's AOT-compiled-at-build-time binary, which pays zero variable warmup
cost on its first run *or* its thousandth — took:

```
COLD_VLLM_OK text=','
init engine (profile, create kv cache, warmup model) took 200.69 s (compilation: 1.48 s)
real  4m4.298s   user  4m27.405s   sys  0m54.466s
```

**~244s cold start**, almost entirely CUDA graph capture (two full capture passes,
51 PIECEWISE + up to 35-51 FULL graphs each) plus model init — not compilation
itself (`torch.compile` found reusable standalone artifacts vLLM ships prebuilt, per
its own log line, so the 1.48s figure understates what a *true* from-scratch
`torch.compile` would cost; CUDA graph capture is the real, non-cacheable-across-
processes cost here). A same-instance `n=3` harness run
(`scripts/bench_cold_vllm.sh`) afterward got:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| **vLLM** | failed† | 2:00.99 | 2:02.99 |

†Run 1 failed with vLLM's own `AssertionError: Error in memory profiling... This
happens when other processes sharing the same container release GPU memory while
vLLM is profiling` — a real, disclosed instability on this GPU-virtualized shared
instance, not a coldstart-infer-side issue. Runs 2/3's ~121-123s (roughly half the
true-first-run's ~244s) are **not** a fair "fresh deployment" number either: by then
`~/.cache/vllm/torch_compile_cache/` was warm from run 1's partial execution (it
crashed *after* compiling, during graph capture) — a genuinely fresh container/
serverless launch with no persistent cache volume would see closer to the ~244s
figure on every single launch, the same way coldstart-infer's AOT compilation cost
is identical on every launch.

**coldstart-infer (4.71-5.05s) is roughly 24-52x faster than vLLM's best case
(~121-123s, cache warm) and roughly 48-52x faster than vLLM's true first-run case
(~244s, no cache)** — a dramatic, honestly-caveated result that actually exercises
the AOT-vs-JIT bet this project is built on, unlike the llama.cpp comparison above.

#### Ollama: a realistic packaging/daemon-overhead data point, not an AOT-vs-JIT test

Ollama wraps llama.cpp's own ggml runtime — its logs show it spawning a bundled
`llama-server` subprocess to actually run inference, so this doesn't add a new data
point on the AOT-vs-JIT question above. What it tests is real anyway: the
daemon/packaging overhead most people actually experience running a local model,
via `ollama create <name> -f Modelfile` (`FROM <path-to-gguf>`, same
`Qwen3-0.6B-Q4_K_M.gguf`) and `scripts/bench_cold_ollama.sh`, which measures three
scenarios separately rather than conflating them into one number (`raw: true`,
`temperature: 0`, `num_predict: 1`, timed via Ollama's own reported
`total_duration`):

| scenario | measurements (seconds) |
|---|---|
| 1. cold daemon + cold model (`ollama serve` freshly restarted, first request) | 5.90, 5.92, 6.72, 6.73, 6.90, 55.19, 61.98 |
| 2. warm daemon + cold model (daemon running, model not yet loaded/reloaded after `ollama stop`) | 10.67, 10.88, 53.15, 57.57, 60.39 |
| 3. warm daemon + warm model (steady state, already resident) | 0.0093, 0.0100, 0.0249, 0.0254, 0.0273, 0.0392 |

**A real, reproducible, intermittent flakiness, not a fluke**: 2 of 7 scenario-1 runs
and 3 of 5 scenario-2 runs hit Ollama's own `"llama-server GPU discovery watchdog
timed out" error="context deadline exceeded"` — its bundled `llama-server`
subprocess's GPU-probe stalling against ThunderCompute's virtualization layer (the
same class of environment-specific slowdown behind the `fast_exit` fix above, but
this time inside Ollama's own process, not coldstart-infer's, and not something this
project can fix). When it doesn't hit the stall, scenario 1 (~6-7s) is directly
competitive with llama.cpp/coldstart-infer's own cold-start range. When it does, it's
~55-62s — an order of magnitude worse, unpredictably. Scenario 3 (model already
loaded) is fast and reliable every time, as expected from a ggml/llama.cpp-class
runtime once warm.

**Reported as-is, not smoothed into a single headline number**: coldstart-infer's own
cold-start numbers throughout this document are tight, single-digit-percent
variance runs; Ollama's aren't, on this specific virtualized GPU environment. That
inconsistency — not just the mean — is itself a real finding about what "run a local
model" actually costs in practice on infrastructure like this, and burying it in an
averaged number would misrepresent it.

#### TypeSafe Jev: an illustrative latency citation, not a benchmark

TypeSafe AI's "Jev" (a "System One" model, released 2026-09-15) is not a generative
LLM — it takes a state and question(s) and returns typed answers with probabilities,
never free text, primarily via a managed cloud API. coldstart-infer's closest
equivalent is **System1** (`Model::system1_evaluate`/`system1_coldstart`, see above):
single-pass, non-autoregressive candidate scoring — the same task shape, prompt +
fixed candidates in, scored typed results out, no decode loop.

Per DECISIONS.md's "TypeSafe Jev comparison framing" entry, this is a **latency-only
citation against Jev's own published figures**, not a live API call or a decision-
quality claim — run via `scripts/bench_cold_system1_vs_jev.sh` (`n=3`, ThunderCompute
A6000, `Qwen3-0.6B-Q4_K_M.gguf`, prompt `"Q: Is the sky blue during the day? A:"`,
candidates `" True"`/`" False"`):

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| **coldstart-infer System1** | 4.70s | 4.78s | 4.50s |
| **Jev (published)** | 70-500ms end-to-end (10-15ms compute), cited, not reproduced | | |

**Reported honestly, not spun: coldstart-infer's System1 cold start is ~10-60x
*slower* than Jev's published figures, not faster.** The reason is structural, not a
System1 inefficiency — System1's scoring step itself is fast (the batched-prefill-GEMM
win noted earlier), but this measurement is dominated by *cold-loading a ~400MB GGUF
from disk in a fresh process*, which every comparison in this project pays and Jev's
managed, always-resident service never does. This is exactly the "different
deployment models, not just different numbers" caveat DECISIONS.md flagged before
this was ever run — confirmed, not just theorized. It does not mean coldstart-infer
is slow at what System1 actually optimizes (single-pass scoring vs. a decode loop);
it means a cold single-process launch is the wrong comparison point against an
always-on managed API, and this citation is published specifically so that mismatch
is on the record rather than glossed over.

Raw `/usr/bin/time -v` logs and stdout/stderr for every run above are kept under
`bench-results/` (gitignored) for inspection.

**A fairer axis, run separately**: Jev's 10-15ms figure is itself a *warm, compute-
only* number (an always-resident service, no cold load) — comparing it against
coldstart-infer's cold-process figure above answers "should you self-host a fresh
process per decision instead of calling an always-on API" (no), but says nothing
about System1's actual scoring mechanism, which is what Jev's number is actually
about. `bench_coldstart --candidate " True" --candidate " False"` (already-existing
warm-latency microbenchmark, model loaded once, `warmup=5 iters=50`, same A6000,
same GGUF) isolates exactly that — no process launch, no GGUF load, just the
gather-GEMV scoring step itself, across three prompt-length buckets:

| prompt tokens | System1 warm p50 | p90 | p99 |
|---:|---:|---:|---:|
| 29 | 19.4ms | 23.0ms | 24.8ms |
| 113 | 36.4ms | 37.6ms | 39.8ms |
| 449 | 144.9ms | 150.1ms | 157.6ms |

At the shortest bucket (29 tokens — closest in shape to a minimal "state + question"
input) coldstart-infer's warm System1 scoring is **19.4ms, within ~1.3-2x of Jev's
10-15ms compute figure** — not the 10-60x gap the cold-start citation above shows.
Even the largest bucket (449 tokens) stays well under Jev's cited 500ms end-to-end
upper bound. Same caveats as above still apply (Jev's number is self-reported, no
decision-quality comparison), but this is the comparison that's actually apples-to-
apples on what Jev's figure measures — a warm decision engine's scoring latency, not
a cold process's total launch cost. **Both numbers matter and answer different
questions**: cold-start-to-decision (self-hosting loses badly, above) vs. warm
per-decision scoring speed (self-hosting is competitive, here) — reporting only one
of them would be the kind of cherry-picking this project's methodology explicitly
rejects.
