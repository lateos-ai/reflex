# Reflex

A GGUF-native Rust & CUDA inference engine built for **low-latency cold starts on
serverless GPU platforms** — process launch to first token, not sustained server
throughput.

Every CUDA kernel is compiled ahead-of-time by `nvcc` at *build* time and shipped
inside the binary — never compiled at runtime via NVRTC — so there's no multi-second
JIT tax on first use, the way there is with a runtime-compilation design. That's the
whole bet: be the fastest way to turn a cold process into one output token, then get
out of the way.

**Current cold start: 627.3ms p50**, process launch to result — Tesla T4,
`Qwen3-0.6B-Q4_K_M`, `reflex system1`, `n=10`. Full phase breakdown
[below](#cold-start-phase-breakdown).

### Why serverless is the fit

On a serverless GPU platform (Runpod, and others with the same shape) you are billed
**per second of wall clock, per invocation** — and on several of them the cold-start
window is billed straight through to the caller as compute. For bursty, single-shot
work — an agentic tool-call, a classification, a structured extraction — time-to-ready
is not merely a latency property. It *is* the bill.

That inverts the usual ranking. A per-token API marketplace rewards packing many
concurrent requests onto one GPU, and an engine that deliberately never batches (see
[Non-goals](#non-goals)) cannot win there on cost per token. Per-second serverless
billing rewards the opposite: start fast, do one job, exit, stop the meter. Reflex's
constraints — one request at a time, no scheduler, no warm-pool assumption — stop being
limitations in that setting and start being the point.

The engine stays a single-shot process either way. The platform supplies the queue,
the autoscaling and the multi-tenancy; Reflex supplies a process that is ready fast.
See [`serverless/runpod/`](serverless/runpod/README.md) for the packaging, and the
[Serverless GPU platforms](#serverless-gpu-platforms-runpod) section for deployment.

**Real measurement, and a recalibration.** This has been deployed and tested end-to-end
against a real Runpod account (2026-09-26, real RTX A4500) — see
[`serverless/runpod/README.md`](serverless/runpod/README.md#real-deployment-findings-2026-09-26-real-rtx-a4500-on-runpod)
for the full detail. The result changes the claim: a real cold invocation took **~42-44
seconds wall clock**, of which Reflex's own load (container start to ready) was **~1.6
seconds** — the remaining ~40 seconds is Runpod's own GPU scheduling and container
provisioning, a floor that exists regardless of which engine runs on top of it. No engine
makes a serverless cold start on this platform sub-second, because most of the time
elapses before any engine code runs at all.

The claim this data actually supports is narrower than "Reflex starts in under a second
on serverless": it's that **Reflex adds near-zero marginal time on top of the platform's
own floor**, where a JIT/graph-compiling engine adds substantially more on top of that
same floor. **That comparison has since been run for real**: Runpod's own official
`worker-vllm` (52,327 deploys — the natural choice, not a self-built image), same GPU
tier, same platform, same day, took **~150 seconds** to a servable state versus Reflex's
~42-44 — **Reflex was ~3.4x faster**. Full methodology, disclosed caveats (different
model precision, different model-acquisition path, different endpoint type — none of
which change the direction of the result), and a real Runpod platform bug found along the
way (the autoscaler exceeding a configured `workersMax`) are in
[`docs/serverless-cost-comparison.md`](docs/serverless-cost-comparison.md). Also found and
fixed during this deployment: Runpod's load-balancer health check hits a hardcoded `/ping`
path regardless of documented override variables, and the request that triggers a cold
start can get a `502` from Runpod's own gateway even though the worker becomes healthy
moments later — both are now documented in the sidecar's and `serverless/runpod/`'s READMEs
for anyone else hitting the same platform.

## Quickstart

Requires the CUDA toolkit (`nvcc` on `PATH`, or `CUDA_PATH`/`CUDA_HOME` set) and an
NVIDIA GPU. `REFLEX_SKIP_CUDA=1 cargo build` skips kernel compilation for
editing/type-checking on a machine without CUDA (no subcommand will actually run
kernels in that mode).

```
git clone https://github.com/lateos-ai/reflex.git
cd reflex
cargo build --release

# Prove the AOT pipeline works end to end on your GPU:
cargo run --release --bin reflex -- smoke

# Run a real forward pass against a GGUF file:
cargo run --release --bin reflex -- generate <path-to-gguf> "Once upon a time"
```

`reflex` is a single binary with subcommands: `generate` (load a GGUF and generate
tokens), `system1` (single-pass, non-autoregressive candidate scoring — the "System 1"
decision-loop path), `smoke` (the AOT-pipeline check above), `bench` (warm-latency
microbenchmark), `check` (byte-exact-vs-reference correctness check, CI-scriptable),
and `stdio`/`uds` (local JSON-line IPC, both need `--features ipc`). Run `reflex
<subcommand>` with no further arguments to see that subcommand's own usage. For real
HTTP clients (OpenRouter, the OpenAI SDKs, curl), see
[`sidecar/openai-adapter`](sidecar/openai-adapter/README.md) — a separate
OpenAI-compatible `/v1/chat/completions` sidecar built on top of `reflex stdio`, not
part of the `reflex` binary itself (see Non-goals below).

### Example: download a model from Hugging Face, then run a System1 test

`system1` takes a local GGUF path, so download the file first with the
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install -U
huggingface_hub`), then point `system1` at it — this scores each `--candidate`
against the prompt in a single pass, with no autoregressive decode loop:

```
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir .

cargo run --release --bin reflex -- system1 Qwen3-0.6B-Q8_0.gguf \
  "The capital of France is" \
  --candidate " Paris" --candidate " London" --candidate " Berlin"
```

Real output from this exact command (RTX A6000):

```
REFLEX_SYSTEM1_CANDIDATE_OK idx=0 text=" Paris" token_ids=[12095] score=17.407064 probability=0.997350
REFLEX_SYSTEM1_CANDIDATE_OK idx=1 text=" London" token_ids=[7148] score=11.308186 probability=0.002239
REFLEX_SYSTEM1_CANDIDATE_OK idx=2 text=" Berlin" token_ids=[19846] score=9.612655 probability=0.000411
REFLEX_SYSTEM1_OK process_start_to_result_ms=8524.253 num_candidates=3 best_idx=0 best_text=" Paris" entropy=0.028154
```

`probability` is relative to this candidate set only, not a vocab-wide probability —
see `Model::system1_evaluate`'s doc comment in `src/model.rs`. `system1` currently
supports dense/MoE Qwen3 only; the Qwen3.5 hybrid mixer and DeepSeek-V2/V3 (MLA) are
rejected with a clear error (`generate` supports all four architectures — swap in a
DeepSeek GGUF the same way for a `generate` run instead).

`generate` can also pull a GGUF straight from the Hub itself, via this project's own
Rust `hf-hub` integration — `--model <org/repo:file.gguf>` or `--quickstart`, both
requiring `cargo build --features download`:

```
cargo run --release --features download --bin reflex -- generate --quickstart "Once upon a time"
```

### Bringing your own model (non-GGUF checkpoints)

Reflex only ever loads GGUF — this is deliberate, not a missing feature (its
Hugging Face integration is a GGUF downloader/cache only, never a new
tensor-format ingestion path). If you have a safetensors/HF-format checkpoint,
convert it to GGUF first with llama.cpp's own unmodified `convert_hf_to_gguf.py`
— the same converter this project uses internally for its own test fixtures and
for real checkpoints like DeepSeek-V2-Lite:

```
python convert_hf_to_gguf.py /path/to/hf-checkpoint --outtype q8_0 --outfile model.gguf
```

LoRA adapters convert the same way, via llama.cpp's `convert_lora_to_gguf.py`.

## Why this exists

Closing the steady-state-throughput gap with llama.cpp/vLLM is a kernel-optimization
race against projects with a multi-year head start — Rust as a language doesn't change
who wins it. So this engine competes on the axis those projects are neither built for
nor measured against: a **cold** invocation.

That axis has a natural home. Serverless GPU platforms bill per-second per invocation,
so on bursty single-shot traffic the dominant cost is not tokens-per-second — it is how
long the worker takes to become useful. The same shape shows up in single-shot CLI and
dev-tool calls, batch/cron jobs, and edge devices that wake on demand; serverless is
just the case where the economics are explicit, because a slow start appears directly
on an invoice.

Everything else follows from taking that seriously. A naive runtime-JIT design pays a
real, measured multi-second tax on first kernel use. vLLM took ~235s to become ready
(CUDA graph capture) before serving one request — which is why an always-on deployment
is its realistic baseline, and why per-invocation scale-to-zero is impractical for it.
llama.cpp avoids both, because like Reflex its kernels are compiled by `nvcc` at
*build* time rather than at process start; it is the honest comparison point here, and
the benchmarks below treat it as one.

**Target metric**: energy-to-first-token from cold start (joules, process launch to
first generated token) — a real, underexplored gap. Existing energy benchmarks measure
warm/steady-state joules-per-token, not full-lifecycle cold-start cost.

**Stated plainly: energy is not yet measured.** There is no NVML/`nvidia-smi` power
sampling anywhere in this repo — every number published here is latency (and, in
`reflex bench`, VRAM residency). Joules-to-first-token remains the *aim* this project
is organized around, not a result it can currently report. Building that
instrumentation is outstanding work, and until it exists this section should be read as
a statement of intent rather than a claim.

**Target models**: Qwen and DeepSeek families.

## Benchmarks

All comparisons are cold-start (process launch to first token/result), same
ThunderCompute A6000, `n=3`, external wall-clock (`/usr/bin/time -v` — process launch
to exit, not just Reflex's own internal timer), except the two Jev *warm* rows below,
which use a different, explicitly-disclosed methodology. Full methodology, disclosed
caveats, and per-run numbers for every comparison below are in `DECISIONS.md` and
`HISTORY.md`.

| vs. | Result | Caveat |
|---|---|---|
| **llama.cpp** | **~1.3–1.4x faster** (4.71–5.05s vs. 6.45–6.56s) | Both AOT-compiled — doesn't exercise the JIT-tax claim below |
| **vLLM** | **~24–52x faster** (4.71–5.05s vs. 121–244s, depending on `torch.compile` cache state) | Installed vLLM has no GGUF support; ran against an HF safetensors checkpoint instead, disclosed |
| **Ollama** | Directly competitive when it doesn't stall (~6–7s), but its bundled `llama-server` intermittently hits an internal GPU-discovery-watchdog timeout (~55–62s) | Wraps llama.cpp's own runtime — tests packaging/daemon overhead, not the AOT-vs-JIT bet |
| **TypeSafe Jev**, cold-start-to-decision | Reflex loses, **~33–62x slower** (18.96s vs. Jev's independently measured 307.8–569.6ms) | Different deployment model: Jev is an always-warm managed API; this measures a genuine cold local process launch. Jev's side is now a real measurement (via OpenRouter), not a citation — see HISTORY.md |
| **TypeSafe Jev**, warm compute-only | **Competitive, within ~1.3–2x** (19.4ms vs. Jev's cited 10–15ms) | Jev's *compute-only* figure is self-reported/published — structurally unmeasurable from outside their infra, still a citation |
| **TypeSafe Jev**, warm, both over the network (independently measured) | Reflex 118.8–224.9ms (network floor to a live sidecar + 20.9ms compute) vs. Jev 120.6–190ms (measured round-trip) — **roughly 10% apart at p50, Jev's max is actually better** | The fairer comparison: both sides now carry real network transit. Reflex's number is a construction (measured floor + measured compute, not one live decision call); Jev's is a direct measurement. See HISTORY.md's "Making the Jev comparison genuinely apples-to-apples" entry |

The llama.cpp/Ollama/Jev "loses" results above are reported as-is, not smoothed over —
see `DECISIONS.md`'s benchmark-methodology entries for why each comparison is framed
the way it is.

### Cold-start phase breakdown

A single aggregate number hides where the time actually goes, so both `reflex generate`
(on its `REFLEX_GENERATE_OK` line) and `reflex system1` (on `REFLEX_SYSTEM1_OK`) report
per-phase timings (`gguf_open_ms`, `cuda_init_ms`, `model_load_ms`, `prompt_eval_ms`,
each a delta between `Instant::now()` checkpoints around the corresponding call), and
`scripts/bench_cold_start_phases.sh <gguf> [n_runs]` /
`scripts/bench_cold_start_phases_system1.sh` run them N times (fresh cold process each
time, external `/usr/bin/time -v` wall clock, raw logs kept, no results discarded) to
report p50/p95 per phase instead of one sample. "Process launch" (OS
`exec`/dynamic-linking/CRT init before `main()` runs) isn't something the process can
report about itself — it's derived as external wall clock minus the internal total,
the same gap this page's other benchmark numbers already rely on.

Current numbers, real `Tesla T4` (dedicated AWS EC2 `g4dn.xlarge`),
`Qwen3-0.6B-Q4_K_M.gguf`, `reflex system1`, `n=10`:

| phase | p50 |
|---|---|
| process launch (external − internal) | ~130 ms |
| CUDA init (`CudaDevice::new`) | ~142 ms |
| model load (dequantize + upload every weight tensor) | 410.3 ms |
| scoring pass (single forward pass) | ~36 ms |
| **total** (`process_start_to_result_ms`) | **627.3 ms** |

On this dedicated instance the spread was unusually tight (p95 within ~1% of p50 at the
point in the optimization sequence where the full p50/p95 table was captured), unlike
the shared-rented-instance variance discussed below.

That 627.3ms is down from **3502.8ms** in this table's previous revision — but that
older figure was a *different measurement* (`reflex generate`, `Q8_0`, rented `NVIDIA
L40`), so it is not a like-for-like 5.6x and shouldn't be quoted as one. On this same
T4/`Q4_K_M`/`system1` setup the starting point was **~1248.6ms** (mean of `n=5`, before
the warm-latency perf plan) and it is **627.3ms** (p50 of `n=10`) now — roughly a 2x
improvement, delivered by that plan's items 2–6 (warp-per-row `gemv`, lazy `lm_head`,
phase instrumentation, pipelined model load, lazy `token_embd` dequant). The
intermediate per-item before/after numbers are recorded in `STATUS.md` and `HISTORY.md`;
they come from separate measurement runs and don't form one continuous series, so
they're not chained here.

What this breakdown shows now: **CUDA init is small and stable** (~142ms) — this is
where the AOT-compiled-kernel bet pays off, since there's no NVRTC JIT tax hiding in
this phase. **Model load still dominates** (~410 of ~627ms, ~65%), but after the lazy
`token_embd` work **no single sub-phase dominates it any more**: inside that ~410ms sit
the per-tensor weights loop + AOT kernel module load (~202ms), tokenizer construction
(~107ms), and the `token_embd` raw copy (~102ms, mmap page-fault cost rather than
compute). There is currently no plan targeting tokenizer construction or AOT module
load — they only became visible as costs once the larger ones were removed.

**Important caveat if you compare against `reflex generate` instead**: the lazy
`token_embd` win does not apply to `generate` on a *tied*-embedding model (no separate
`output.weight` — which includes `Qwen3-0.6B`). There the ~548ms cost is only *moved*
from `model_load_ms` into `prompt_eval_ms`, plus a new unconditional ~102ms copy, for a
net **+110ms (+9%) regression** on that specific path (~1206ms → ~1316ms). That is an
accepted, documented trade-off, not an unnoticed one — see `STATUS.md` and `HISTORY.md`.

**The comparison table above predates perf items 2–6** and was measured on an A6000;
it has not been re-run against the current code. Its Reflex-side figures are therefore
conservative — the engine has since gotten materially faster — but they are stale
rather than current measurements either way. Re-running that table is outstanding work.

Also disclosed rather than smoothed over: in the earlier L40 measurements, two batches
in the same session showed materially different noise levels across *every* run of the
second batch, not just a single outlier, pointing at real host-level contention on a
shared rented GPU instance. **Session-to-session variance on rented cloud GPU hardware
can exceed intra-session variance**, so treat any single-session cold-start number as
illustrative, not lab-controlled.

**Not yet covered, flagged as real follow-up work, not silently skipped**: host-vs-
container (does `cgroups`/the NVIDIA Container Toolkit/device-plugin limits change
CUDA-init overhead vs. bare metal?) and persistent-vs-`exec`'d rows (does keeping one
`reflex` process warm and re-using it change anything the `smoke`/`generate`
one-shot-process model doesn't already show?) — both real, specific, harder-to-answer
questions than the phase breakdown itself, deliberately scoped out of this round.

## Core technical bet

Every CUDA kernel is compiled **ahead of time** (`build.rs` invokes `nvcc`, see
`build.rs` and `src/kernels_cuda/`), never at runtime via NVRTC. `src/aot.rs` loads the
precompiled PTX/cubin at process start via the CUDA driver API. Default mode emits
portable PTX (small driver-side JIT-to-SASS cost at load); set `REFLEX_CUDA_ARCH=sm_XX`
to compile straight to a `cubin` for one target architecture (true zero-JIT, at the cost
of needing a matching cubin per deployment target). Which one actually wins on real
hardware is unverified — that's the first thing to measure, not assume.

Run `cargo run --bin reflex -- smoke` on a real GPU instance as the very first
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

All four steps below are **done** and real-hardware-verified (see `STATUS.md` for
current state, `HISTORY.md` for the full verification write-up of each):

1. **Dense Qwen3** — the best-understood, most well-documented architecture to build
   against first; proves the AOT-compilation + cold-start-benchmark harness works at all.
2. **Qwen3-MoE**
3. **Qwen3.5 hybrid Gated DeltaNet mixer**
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension.

## Non-goals

These are permanent constraints on this engine, not just current-MVP scope — the whole
reason Reflex exists is to win a narrower bet (cold-start energy/latency) than
sustained-server throughput. A broad serving feature set re-inherits the exact
throughput/serving race that's unwinnable against llama.cpp/vLLM/SGLang's head start.
Multi-tenancy and persistent state belong in the *host
orchestrator*, not in this engine:

> **"Optimized for serverless" describes where this engine runs well, not features it
> grows.** A "Serverless-Native Inference Engine" pitch — building platform machinery
> (internal queue, scheduler, autoscaling logic, `batch_size > 1`) *into* Reflex — was
> considered and rejected on 2026-09-17; see `DECISIONS.md`. Nothing in this page's
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
[`sidecar/openai-adapter`](sidecar/openai-adapter/README.md) is a standalone crate
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

## Post-architecture-MVP roadmap: productization

Once the model-architecture MVP proves the engine handles the target model
families at all, the next axis is making the *cold-start path itself* faster and
adoptable — without ever crossing into building a serving platform. The framing: let
vLLM win the warm-throughput race; Reflex wins by being the fastest way to turn
cold compute into one output token, then getting out of the way. All four phases below
are **done** — see `HISTORY.md` for the full per-round write-up of each:

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

## Host-side building blocks

- `src/gguf.rs` — GGUF metadata/tensor-directory parsing (mmap-based).
- `src/dequant.rs`, `src/dequant_iq.rs`, `src/dequant_iq_tables.rs` — standard and
  i-quant dequantization, verified byte-exact against `gguf-py`.
- `src/tokenizer.rs` — verified against real sentencepiece/BPE references.

None of these care how kernels get compiled — they're pure host-side GGUF/tokenizer
logic. There is deliberately no NVRTC runtime-compile-and-load path anywhere in this
codebase — that's the thing this project's AOT design replaces, not reuses.

## Docker

A multi-stage `Dockerfile` is included: the builder stage has the full CUDA devel
toolkit (`nvcc`) to compile the AOT kernels; the runtime stage only needs the CUDA
*runtime* libraries, since every kernel byte is embedded directly into the compiled
binary at build time — the runtime image never runs `nvcc` and never needs the devel
toolkit.

```
# Defaults to sm_86 (RTX A6000/3090-class). Pass --build-arg REFLEX_CUDA_ARCH=sm_XX
# for a different target compute capability, or --build-arg REFLEX_CUDA_ARCH= (empty)
# for a portable PTX build that JITs to whatever GPU the container actually runs on.
docker build --build-arg REFLEX_CUDA_ARCH=sm_86 -t reflex .

# Needs nvidia-container-toolkit on the host. The default entrypoint is
# `reflex generate`, so pass just the GGUF path and prompt:
docker run --rm --gpus all -v /path/to/models:/models \
  reflex /models/Qwen3-0.6B-Q4_K_M.gguf "Once upon a time"
```

For any other subcommand (`system1`/`smoke`/`bench`/`check`/`stdio`/`uds`), override
the entrypoint:

```
docker run --rm --gpus all -v /path/to/models:/models \
  --entrypoint /usr/local/bin/reflex reflex \
  system1 /models/Qwen3-0.6B-Q4_K_M.gguf "Q: ...? A:" --candidate " Yes" --candidate " No"
```

The CUDA major/minor version in both Docker stages must stay consistent with
`Cargo.toml`'s pinned `cudarc` feature (`"cuda-12000"`, i.e. CUDA 12.x) — a mismatch is
a build-time/runtime library version mismatch this Dockerfile can't catch for you.

For a cost-optimized AWS pattern built on this same image (Spot GPU instances, an Auto
Scaling Group with minimum capacity 0, and a `reflex uds` sidecar reachable over a local
Unix Domain Socket instead of a network load balancer), see
[`docs/aws-deployment.md`](docs/aws-deployment.md). For the opposite tradeoff -- an
always-warm On-Demand instance behind a public HTTPS load balancer with autoscaling,
built on the [`sidecar/openai-adapter`](sidecar/openai-adapter/README.md) HTTP sidecar
instead of raw UDS -- see [`docs/aws-deployment-warm.md`](docs/aws-deployment-warm.md).

## Serverless GPU platforms (Runpod)

The economics behind this deployment shape are covered in
[Why serverless is the fit](#why-serverless-is-the-fit) above; this section is the
mechanics.

[`serverless/runpod/`](serverless/runpod/README.md) packages the existing
[`sidecar/openai-adapter`](sidecar/openai-adapter/README.md) — unmodified, no new
handler code — as a Runpod Serverless **load-balancing endpoint** (the endpoint type
that proxies HTTP straight to an arbitrary custom server, rather than the queue-based
type that requires a Python SDK handler). Two details that matter there:

- **`GET /healthz` is three-state**, specifically so a platform health check can
  measure readiness honestly: `204` while the managed `reflex` child is alive but still
  loading the model, `200` once it can actually serve a request, `503` if the child
  died. Without the `204` state, a platform would clock "ready" the instant the HTTP
  port binds — well before model load finishes — and report a cold-start number that
  isn't real.
- **The GPU arch pin differs from the AWS guides, and the pool name lies.** Runpod's
  cheapest serverless pool (`AMPERE_16`, $0.58/hr, verified from the live catalog) is
  *mixed-architecture*: RTX A4000/A4500 are Ampere (`sm_86`), but RTX 2000 Ada / RTX
  4000 Ada in the same pool are Ada (`sm_89`). A pinned cubin only runs on the compute
  capability it was built for, so an endpoint free to schedule anywhere in that pool
  fails nondeterministically. Pin the SKU to **RTX A4500** (`sm_86`, 20GB, the only one
  in the tier with HIGH availability). The AWS guides target the T4 (`sm_75`) instead —
  the two images are not interchangeable.

### Container image size is part of cold start here

On a genuinely cold worker, image pull precedes everything else, and the standard
CUDA `-runtime-` base image is mostly libraries this engine never loads. Measured
locally: the current image is **3.78GB**, of which the `reflex` binary is **2.42MB** —
the bulk is the `cuda-libraries-12-4` meta-package, which ships NCCL, cuFFT, cuSPARSE,
cuSOLVER, NPP and nvJPEG. Reflex links only the CUDA **driver API** plus **cuBLAS**
(`Cargo.toml`'s `cudarc` features are `driver`/`cublas`/`cuda-12000`/`f16`); none of
those other libraries are referenced anywhere in `src/`.

Building instead on the `base` image plus `libcublas-12-4` alone produced a
**verified-working 1.28GB image containing both binaries** (vs. 3.78GB for the
standard image carrying only `reflex`) — a ~66% reduction, with cuBLAS itself
accounting for 553MB of what remains. **This slimming is a measured finding, not yet
landed in the repo's Dockerfiles**; the images under `sidecar/openai-adapter/` and
`serverless/runpod/` still build on the full `-runtime-` base.

**`serverless/runpod/` has since been deployed against a real Runpod account and works
end-to-end** — see [Why serverless is the fit](#why-serverless-is-the-fit) above and
`serverless/runpod/README.md`'s "Real deployment findings" for the measured numbers, the
platform's actual (undocumented) health-check behavior, and a gateway quirk found along
the way. The one thing still unconfirmed is the exact dollar amount billed for those test
invocations — Runpod's billing API hadn't reconciled the relevant hour yet at time of
writing; the per-second rate itself is confirmed from the live catalog.

## Kubernetes

Reflex is a single-shot CLI, not a server (see Non-goals above) — the natural
Kubernetes primitive is a **Job**, one cold-start invocation per Pod, never a
`Deployment`/`Service`. A minimal example running `generate` against a GGUF baked into
a volume, requesting one GPU via the standard NVIDIA device plugin:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: reflex-generate
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: reflex
          image: reflex:latest
          args: ["/models/Qwen3-0.6B-Q4_K_M.gguf", "Once upon a time"]
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: reflex-models
```

For `system1`/`bench`/`check`/other subcommands, set `command: ["/usr/local/bin/reflex"]`
and put the subcommand as the first entry in `args`, same as the Docker override above.
This is exactly the "orchestrator's job" this engine intentionally stays out of —
Reflex itself never grows a scheduler, a request queue, or a `batch_size > 1`; Kubernetes
(or cron, or a FaaS platform) is where that concurrency/scheduling belongs.

