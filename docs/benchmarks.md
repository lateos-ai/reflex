# Benchmarks

Every measurement behind the numbers in the [README](../README.md), with its setup,
method and caveats. Where a figure has been superseded, the old one is kept with a note
rather than removed.

Contents: [Where this engine competes](#where-this-engine-competes) ·
[Cold start vs. other engines](#cold-start-vs-other-engines) ·
[TypeSafe Jev](#typesafe-jev) · [Cold-start phase breakdown](#cold-start-phase-breakdown) ·
[Energy](#energy) · [Kernel build modes](#kernel-build-modes) ·
[Long-context attention](#long-context-attention) ·
[Container images](#container-images) · [Serverless (Runpod)](#serverless-runpod)

## Where this engine competes

On a serverless GPU platform (Runpod, and others with the same shape) you are billed
**per second of wall clock, per invocation** — and on several of them the cold-start
window is billed straight through to the caller as compute. For bursty, single-shot
work — an agentic tool-call, a classification, a structured extraction — time-to-ready
is not merely a latency property. It *is* the bill.

That inverts the usual ranking. A per-token API marketplace rewards packing many
concurrent requests onto one GPU, and an engine that deliberately never batches (see
[Non-goals](../README.md#non-goals)) cannot win there on cost per token. Per-second serverless
billing rewards the opposite: start fast, do one job, exit, stop the meter. Reflex's
constraints — one request at a time, no scheduler, no warm-pool assumption — stop being
limitations in that setting and start being the point.

The engine stays a single-shot process either way. The platform supplies the queue,
the autoscaling and the multi-tenancy; Reflex supplies a process that is ready fast.
See [`serverless/runpod/`](../serverless/runpod/README.md) for the packaging, and the
[Runpod section of the reference](reference.md#runpod) for deployment.

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

**Target models**: Qwen and DeepSeek families.

## Cold start vs. other engines

All comparisons are cold-start (process launch to first token/result), external
wall-clock (`/usr/bin/time -v` — process launch to exit, not just Reflex's own internal
timer), except the TypeSafe Jev *warm* rows below, which use a different,
explicitly-disclosed methodology. The `llama-simple`/`llama-cli`/Ollama/vLLM rows were
refreshed on 2026-09-29 on a dedicated AWS EC2 `g4dn.xlarge` (**Tesla T4**, driver
595.91.07 / CUDA 13.2), `n=30` per engine (`n=3` per Ollama scenario, `n=3` for vLLM),
single session, same `Qwen3-0.6B-Q4_K_M.gguf` and same prompt (`"Once upon a time"`) on
both sides; the TypeSafe Jev rows are unchanged from their own earlier measurements.
Raw per-run logs are kept under `bench-results/` on the machine that ran them; the
scripts that produce every number are in `scripts/bench_cold_*.sh`.

**Device-comparability caveat (disclosed, not buried).** This refresh could not run on
the published table's ThunderCompute A6000 — that GPU was unavailable — so it uses the
T4 fallback the refresh plan allows. That is not just a different card: the old A6000
rows were measured on ThunderCompute's GPU-virtualized instances, where *every* CUDA
process paid a multi-second post-result context-teardown tax through its virtualization
proxy (the issue behind Reflex's `fast_exit`). On a dedicated AWS T4
that tax is absent, so **both** engines are far faster in absolute terms and the old
A6000 numbers are not directly portable. Both engines here ran on the same T4 in the
same session, so the ratio between them is sound; the absolute numbers are not
comparable to the old A6000 table. The harness was re-validated first (same
`bench_cold_common.sh`, same commands): llama.cpp no longer reads ~6.5s precisely
because the virtualized-teardown tax it used to pay is gone — a host-class difference,
diagnosed before any number was trusted, not a harness change.

These compare a *cold* local process launch (Reflex) against llama.cpp's front-ends,
Ollama, and vLLM. Note the AOT truth before reading the "avoids runtime JIT" claim into
every row: `llama-simple`/`llama-cli` and Ollama's bundled `llama-server` are, like
Reflex, compiled ahead-of-time by `nvcc` at *build* time — none pays a runtime CUDA JIT
tax, so those rows measure *other* cold-start overheads (model load, packaging, GPU
discovery), and only the vLLM row actually exercises Reflex's no-runtime-JIT bet.

Reflex's own number on this device (`reflex generate`, `n=30`): **p50 0.83s / p95 0.84s**,
peak RSS 982 MB.

| vs. (cold process launch, Tesla T4, `n=30`) | Result (p50; p95 in parens) | Caveat |
|---|---|---|
| **llama.cpp `llama-simple`** (true prompt-in/token-out) | **Reflex ~1.13x faster** (0.83s [0.84s] vs. 0.94s [0.96s]) | Both AOT-compiled. On this dedicated T4 the win is Reflex's optimized cold load, *not* the `fast_exit` teardown fix — that only mattered on ThunderCompute's virtualized A6000; doesn't exercise the JIT-tax claim |
| **llama.cpp `llama-cli`** | **Reflex ~1.9x faster** (0.83s vs. 1.58s [1.59s]) | `llama-cli` always applies the model's chat template even with `-p` (a documented gotcha), so `llama-simple` is the true prompt-in/token-out row; kept for continuity with the published table |
| **Ollama (bundled `llama-server`)** | **Reflex ~2.6x faster** (0.83s vs. ~2.1s cold daemon + cold model; warm model ~7–11ms) | Wraps llama.cpp's AOT runtime; tests packaging/daemon overhead. The published ThunderCompute watchdog stall did **not** recur here (0 of 9 runs); one 38.8s cold-first-run outlier was recorded, not averaged away |
| **vLLM** | **Reflex ~48–150x faster** (0.83s vs. 39.8–127.3s, depending on `torch.compile` cache state) | **The actual AOT-vs-JIT foil** — vLLM's CUDA graph capture + `torch.compile` warmup at cold start. Installed vLLM 0.30.0 still has no GGUF support; ran against the HF safetensors checkpoint instead, disclosed |

![Cold-start: process launch to first token on a Tesla T4 — Reflex vs llama.cpp / Ollama / vLLM, with Reflex's phase breakdown](cold-start-t4.png)

*Figure: cold start = process launch → first token, external wall clock, Tesla T4,
`Qwen3-0.6B-Q4_K_M`, `n=30` (`n=3` Ollama/vLLM). Left: Reflex vs the other engines on a
log axis, so vLLM stays visible. Right: where Reflex's 0.83s goes. Editable vector source:
[`docs/cold-start-t4.svg`](cold-start-t4.svg).*

## TypeSafe Jev

These compare Reflex's warm-compute side against Jev, an always-warm managed decision API
— the structural opposite of a cold local process. Different, explicitly-disclosed
methodology.

| vs. | Result | Caveat |
|---|---|---|
| **TypeSafe Jev**, cold-start-to-decision | Reflex loses, **~1.1–2.0x slower** (0.63s p50 vs. Jev's independently measured 307.8–569.6ms; re-measured 2026-10-01, see below) | Different deployment model: Jev is an always-warm managed API; this measures a genuine cold local process launch. Jev's side is now a real measurement (via OpenRouter), not a citation |
| **TypeSafe Jev**, warm compute-only | **Competitive, within ~1.3–2x** (19.4ms vs. Jev's cited 10–15ms) | Jev's *compute-only* figure is self-reported/published — structurally unmeasurable from outside their infra, still a citation |
| **TypeSafe Jev**, warm, both over the network (independently measured) | Reflex 118.8–224.9ms (network floor to a live sidecar + 20.9ms compute) vs. Jev 120.6–190ms (measured round-trip) — **roughly 10% apart at p50, Jev's max is actually better** | The fairer comparison: both sides now carry real network transit. Reflex's number is a construction (measured floor + measured compute, not one live decision call); Jev's is a direct measurement. |

The llama.cpp/Ollama/Jev "loses" results above are reported as-is, not smoothed over.

**Re-measured 2026-10-01.** The cold-start row above used to read 18.96 s (~33–62x
slower), from runs on a GPU-virtualized A6000 that paid a multi-second CUDA teardown tax
per process and predated the cold-load optimizations. Re-run with the same script
(`scripts/bench_cold_system1_vs_jev.sh`, `n=10`) on a dedicated AWS T4 (`g4dn.xlarge`,
driver 595.91.07), `Qwen3-0.6B-Q4_K_M`, at commit `bc98384`: wall clock per cold
`reflex system1` process **0.62–0.64 s, p50 0.63 s**, peak RSS 896 MB. Jev's side is
unchanged (its independently measured network round-trip, 307.8–569.6 ms). The warm rows
were not re-measured.

## Cold-start phase breakdown

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

Each of those phases also gets an **additive** `REFLEX_PHASE_OK phase=<name>
duration_ms=<ms> energy_joules=<j> energy_method=<m>` line (one per phase, printed
*before* the aggregate result line; the `energy_*` fields are present only when a
measurement is available), with the same fields in the `--json` form. The existing
`REFLEX_GENERATE_OK`/`REFLEX_SYSTEM1_OK` lines and their fields are unchanged. The
per-phase `energy_joules` is the delta between consecutive cumulative readings, so its
precision follows the mode caveat above (coarse in `total_energy_counter` mode for very
short phases; continuous in `polled_power` mode). The four phases span process-start
*through the first token*, so they need not sum to the aggregate `joules` — for
`generate --max-tokens N>1` the aggregate also covers post-first-token decode, and in
`total_energy_counter` mode counter quantization can make the sum differ from the
aggregate even for `system1` (both observed on a real T4: sub-50ms phases read `0.000 J`,
and a phase sum matched the aggregate in some runs but was one counter-tick short in
others).

The same additive treatment covers the other subcommands with a real phase boundary:
`reflex smoke` emits `cuda_init` / `kernel_load` / `kernel_launch`, and `reflex bench`
emits its model-load sequence (`gguf_open` / `cuda_init` / `model_load`) ahead of its
existing per-bucket `REFLEX_BENCH_ENERGY_OK` line. `reflex doctor` emits **no** phase line
and no total joules — it runs no timed execution phase, only cheap boolean checks, so it
reports the energy *method* a real run would use (`nvml_energy` check) instead of inventing
a phase to measure.

**Current numbers, real `Tesla T4` (dedicated AWS EC2 `g4dn.xlarge`), `Qwen3-0.6B-Q4_K_M.gguf`,
`reflex system1`, `n=10`** (`REFLEX_CUDA_ARCH=sm_75`, driver 595.91.07 / CUDA 13.2;
`scripts/bench_cold_start_phases_system1.sh`, which now also aggregates the
`REFLEX_PHASE_OK` `energy_joules` fields; raw per-run logs kept under `bench-results/`):

| phase | p50 ms | p95 ms | p50 joules (`total_energy_counter`) |
|---|---|---|---|
| process launch (external − internal) | 112.1 | 125.3 | n/a (pre-`main`) |
| gguf open + metadata parse | 64.2 | 67.0 | 0.000 |
| CUDA init (`CudaDevice::new`) | 142.3 | 144.0 | 3.662 |
| model load (kernel module load + weight dequant/upload; tokenizer and cuBLAS init overlapped on a worker thread) | 235.1 | 244.4 | 9.613 |
| scoring pass (single forward pass) | 39.2 | 39.5 | 0.000 |
| **total** (`process_start_to_result_ms`) | **485.4** | **494.0** | **20.324** |

This table is `f32` weights. The `f16` default was measured on 2026-10-04 in a different
build mode, so the two don't compare row by row. See [f16 weight storage](#f16-weight-storage)
(same-session `system1` total: 451.0 ms `f32` vs. 441.3 ms `f16`).

The joules column is exactly as measured, not cleaned up: the ~39 ms scoring pass and the
~64 ms GGUF parse both round to `0.000 J` at the counter's granularity, and the per-phase
joules need not add to the total (counter quantization can place a phase's joules in the
previous boundary's snapshot). That is the device-wide/counter caveat in practice.

Forcing the `polled_power` fallback (`REFLEX_NVML_FORCE_POLLED=1`, same `n=10`) makes those
short phases measurable — the reason the fallback exists:

| phase | p50 ms | p50 joules (`polled_power`, forced) |
|---|---|---|
| gguf open + metadata parse | 61.6 | 0.957 |
| CUDA init | 187.9 | 7.129 |
| model load | 267.1 | 11.888 |
| scoring pass | 40.4 | 1.450 |
| **total** | **556.5** | **21.433** |

This T4 would normally take the counter branch (`reflex doctor` reports
`method=total_energy_counter`), so the polled path is forced here only to exercise it. Two
honest costs of that path: the polling thread adds wall-clock on this small 4-vCPU instance
(CUDA init 142→188 ms, model load 235→267 ms), and the polled totals run slightly higher
than counter mode — partly that extra time, partly the counter's coarse undercount. Both are
real and disclosed rather than hidden. `scripts/bench_cold_start_phases.sh` (the `generate`
variant) reports the same columns for the first-token metric (`total` p50: 755.8 ms /
32.16 J counter; 836.3 ms / 35.28 J forced-polled). The other benchmark tables above stay
latency-only because only Reflex emits NVML energy — llama.cpp/vLLM/Jev have no comparable
joules figure to put in a column.

On this dedicated instance the spread was tight (p95 within ~1–4% of p50 per phase in the
table above), unlike the shared-rented-instance variance discussed below.

That ~485ms is down from **3502.8ms** in this table's previous revision — but that
older figure was a *different measurement* (`reflex generate`, `Q8_0`, rented `NVIDIA
L40`), so it is not a like-for-like multiple and shouldn't be quoted as one. On this same
T4/`Q4_K_M`/`system1` setup the starting point was **~1248.6ms** (mean of `n=5`, before
a round of cold-load optimizations) and it is **~485ms** (p50 of `n=10`) now — roughly a 2.6x
improvement, delivered first by those optimizations (warp-per-row `gemv`, lazy
`lm_head`, phase instrumentation, pipelined model load, lazy `token_embd` dequant), then
by a cold-load overlap round (a fast non-cryptographic tokenizer hasher, and the
tokenizer construction + cuBLAS handle init moved onto a single worker thread overlapped
with the weight load). The intermediate per-item before/after numbers came from separate
measurement runs and don't form one continuous series, so they're not chained here.

What this breakdown shows now: **CUDA init is small and stable** (~140ms — this is
`CudaDevice::new`, not kernel loading). The AOT bet shows up *inside* model load: the
*entire* dense kernel-module load (rmsnorm, rope, silu, gemv, gemv_gather, attention,
attention_prefill, elementwise, dequant) measures **~3ms** in pinned-cubin mode (and
~3.5ms in portable PTX) — no JIT tax hiding there. **Model load is now ~237ms of the
~485ms total (~49%)**, dominated by two memory-bound pieces: the per-tensor weight
dequant/upload loop (~124ms) and the `token_embd` raw-byte copy (~102ms, mmap page-fault
cost rather than compute). Tokenizer construction (~100ms) and cuBLAS handle init
(~81ms) — the two remaining host/driver setup costs the lazy-embedding profiling surfaced — are
now built concurrently on a single worker thread and no longer sit on the serial path at
all. There is no longer any measured host-setup cost left to overlap; the remaining
model-load time is bandwidth-bound.

**Note on `reflex generate` with a *tied*-embedding model** (no separate `output.weight`
— which includes `Qwen3-0.6B`): the lazy `token_embd` optimization
initially left a **+110ms (+9%) regression** on that specific path, because the
full-vocab dequant it defers for `system1` is still needed for `generate`'s greedy
argmax, so the cost was only *moved* from `model_load_ms` into `prompt_eval_ms`. That
regression is now **closed and turned into a win**: the tied full-vocab dequant moved
on-device (2026-09-28), cutting `prompt_eval_ms` by ~426ms on that path in its own A/B
(`generate` total ~1314ms → ~894ms there).

**The comparison table has since been refreshed** (2026-09-29, dedicated AWS T4): the
`llama-simple`/`llama-cli`/Ollama/vLLM rows above are current measurements of this code,
not the older A6000 figures, which predate the cold-load optimizations.

Also disclosed rather than smoothed over: in the earlier L40 measurements, two batches
in the same session showed materially different noise levels across *every* run of the
second batch, not just a single outlier, pointing at real host-level contention on a
shared rented GPU instance. **Session-to-session variance on rented cloud GPU hardware
can exceed intra-session variance**, so treat any single-session cold-start number as
illustrative, not lab-controlled.

**Host-vs-container/cgroup, and persistent-vs-`exec`'d — both closed** (2026-09-27, real
Tesla T4 `g4dn.xlarge`, `Qwen3-0.6B-Q4_K_M.gguf`, `REFLEX_CUDA_ARCH=sm_75` pinned-cubin
build on both sides so the comparison isn't confounded by JIT-vs-cubin, `n=10` each):

**Host vs. container.** Same binary, same pinned cubin, run bare (`/usr/bin/time -v
./target/release/reflex generate ...`) vs. inside `docker run --rm --gpus all` against
this repo's own root `Dockerfile` image (already built and resident locally — this
isolates container *runtime* overhead, not registry image-pull time):

| | internal total (`process_start_to_first_token_ms`, p50) | external wall clock (p50) | derived launch overhead |
|---|---|---|---|
| Host (bare process) | 1303.9 ms | ~1400 ms | ~96 ms |
| Container (`docker run --gpus all`) | 1288.3 ms | ~1800 ms | ~512 ms |

Reflex's own internally-reported total is statistically identical either side (1303.9ms
vs. 1288.3ms, well within run-to-run noise) — the ~400ms/~29% gap in external wall clock
is entirely attributable to Docker's own container-launch machinery (dispatch to
`dockerd`, OverlayFS mount, cgroup/namespace setup, the NVIDIA Container Toolkit's device
injection), not to anything about CUDA execution being slower inside a container. Both
sides' first run of `n=10` was an outlier and is excluded from the p50s above rather than
smoothed into them: the container's first run paid an extra ~1.3s in `cuda_init_ms`
(1400ms vs. ~140ms on every subsequent run — plausibly the container's first-ever access
to the GPU device nodes through a fresh network/cgroup namespace) and the host's first run
paid an extraordinary ~16.5s inside `model_load_ms` alone (`cuda_init_ms` on that same run
was a normal 141.8ms) — real, logged, and unexplained rather than hand-waved: not
reproducible on any of the other 9 host runs, and not matching any known page-cache-cold
explanation (the model file had just been written to the same filesystem, so it should
already have been page-cache-resident). Flagged honestly as an open question for whoever
re-runs this, not swept under "variance."

**Persistent vs. `exec`'d.** `reflex stdio` started once, `REFLEX_STDIO_READY` awaited,
then 10 requests sent to the *same* resident process over its existing JSON-line IPC
(`scripts/`-adjacent throwaway harness, not a checked-in script — see `src/ipc.rs`) vs.
the cold per-process total above:

| | p50 | note |
|---|---|---|
| Cold, one process per request (this project's actual design) | 1303.9 ms | CUDA init + full model load + first token, every time |
| Persistent, resident process, steady state (requests 2–10) | 16.35 ms | pure inference + IPC round-trip, no reload |
| Persistent, resident process, *first* request after ready | 708.96 ms | one-time compute-kernel warmup cost `REFLEX_STDIO_READY` doesn't capture |

A resident process's steady-state request is **~80x faster** than this project's own
cold-start design — expected and not a criticism of the architecture (a resident,
always-on server is exactly the `batch_size`-1/no-thread-pool serving-platform shape this
project's Non-goals deliberately reject; see [Non-goals](../README.md#non-goals)). What's genuinely useful
here: the gap between a resident process's *first* request (708.96ms) and its *second*
(16.35ms) shows that even a fully warm, already-loaded model still pays a real one-time
cost the first time it actually runs its compute kernels — a cost `model_load_ms` doesn't
capture because weight upload uses dequant/memcpy kernels, not the attention/GEMV kernels
a real forward pass launches for the first time. That first-real-inference tax is not
currently broken out as its own phase; doing so is a plausible future refinement of this
breakdown, not scoped here.

## f16 weight storage

`f16` became the default weight storage on 2026-10-02 (see [reference](reference.md#weight-storage)).
**Every other figure on this page was measured with `f32` weights** and stays as
measured. Each will get a note pointing at its `f16` replacement once that's measured,
rather than being overwritten.

[`scripts/verify_f16_weights.sh`](../scripts/verify_f16_weights.sh) produces every number
in this section, on one GPU in one session:

| what | method |
|---|---|
| Numerics gate | `REFLEX_F16_ROUNDTRIP=1` (f16-rounded weights, f32 kernels) vs. `--weights f32`, 32 greedy tokens, three prompts, every supported architecture: matched/total, first divergence, top-2 logit gap there |
| f32 regression | `--weights f32` vs. a `master` build: `reflex check` token ids and logit checksum, `system1` scores, must be identical |
| f16 correctness | `--weights f16` vs. `f32` (and vs. llama.cpp `llama-simple` text when given), system1 max abs score difference, max prefill activation per model |
| Cold start | `bench_cold_start_phases_system1.sh` and `bench_cold_start_phases.sh`, n=10 each, `master` vs. `f32` vs. `f16`, p50/p95 per phase. `master` vs. `f32` isolates the larger dequant kernel module (304 KB → 525 KB of PTX) in `model_load_ms` |
| Memory, decode | `reflex bench` resident VRAM and decode ms/token, plus nvidia-smi peak during a 64-token generate, Qwen3-0.6B and a larger model, `f32` vs. `f16` |

**Measured 2026-10-04 on a dedicated AWS `g4dn.xlarge` (Tesla T4, 15 GB, driver
595.91.07, CUDA 13.2).** Both binaries were built from source on the box in the default
portable-PTX mode, so these figures compare with each other, not with the `sm_75`-cubin
figures elsewhere on this page. `master` is `3e4265a`. Models: Qwen3-0.6B-Q4_K_M,
Qwen3-1.7B-Q4_K_M, TinyLlama-1.1B, Qwen3.5-0.8B, and the random-weight `tiny-qwen3moe`,
`tiny-qwen35moe` and `deepseek-tiny-mla` fixtures. DeepSeek-V2-Lite was not run: its `f32`
side needs an 80 GB GPU.

**Correctness.**

- **Numerics gate:** f16-rounded weights alone (`REFLEX_F16_ROUNDTRIP=1`) matched `f32` for
  32/32 greedy tokens on every model and prompt (18 of 18 runs).
- **`--weights f32` vs. `master`:** identical on every model. That covers the `reflex
  check` token ids and logit checksum for all three prompts, and the `system1` scores.
- **`--weights f16` vs. `f32`:** 32/32 tokens on all 18 runs, plus 3/3 more runs on
  Qwen3-1.7B. The `system1` max abs score difference was 0.0035 (Qwen3-0.6B), 0.0032
  (TinyLlama), 0.0120 (Qwen3.5-0.8B) and ≤ 0.00013 on the fixtures. The best candidate
  was the same in every case.
- **Against llama.cpp:** compared with `llama-simple -n 32` (llama.cpp `22bdcc4`, built for
  `sm_75`), `f16` and `f32` produce the same text on every prompt. Three of the nine
  differ from llama.cpp at a near-tie fork 13–30 tokens in (Qwen3-0.6B p0; Qwen3.5-0.8B
  p1, p2). Since `f32` is identical to `master`, those differences predate `f16`.

Largest |activation| cast to f16 for a prefill GEMM (`REFLEX_F16_ACT_STATS`), with no
saturation anywhere:

| model | max abs | headroom to 65504 |
|---|---|---|
| **Qwen3-1.7B** | **15,420** | **4.2x** |
| Qwen3-0.6B | 3,644 | 18x |
| TinyLlama-1.1B | 161 | 400x |
| Qwen3.5-0.8B | 34 | 1,900x |

Qwen3's activations grow with model size: from 0.6B to 1.7B the maximum went up about
4x. Qwen3-1.7B already sits within 10x of the f16 limit, and nothing here measured
Qwen3-4B or larger. The cast saturates rather than overflowing, and the run warns when it
clamps. A larger Qwen3 should still be checked with `REFLEX_F16_ACT_STATS=1` before
relying on `f16`.

**Cold start**, Qwen3-0.6B, n=10 each, p50 (p95 within 1.5% everywhere):

| | `master` | branch `f32` | branch `f16` |
|---|---|---|---|
| `system1` model load | 232.7 ms | 234.6 ms | 220.4 ms |
| `system1` scoring pass | 39.1 ms | 39.1 ms | **44.0 ms** |
| `system1` total (internal) | 448.1 ms | 451.0 ms | 441.3 ms |
| `generate` model load | 235.6 ms | 237.6 ms | 222.2 ms |
| `generate` prompt eval (first token) | 289.6 ms | 289.6 ms | 277.1 ms |
| `generate` total (internal) | 703.4 ms | 706.0 ms | 676.5 ms |

- **Larger dequant module:** compare `master` with branch `f32`, which load the same
  `f32` weights. It costs about +2 ms of model load. In PTX mode that cost is measured
  with the driver's JIT cache warm after the first run.
- **`f16` overall:** about 6% less model load, and a shorter total: −2.1% for `system1`,
  −4.2% for `generate`.
- **One regression:** `system1`'s single scoring pass is **4.9 ms slower in `f16`** (39.1
  → 44.0 ms). The same comparison on an already-loaded model goes the other way (see the
  warm prompt column below). That suggests a first-call cost in the cuBLAS `f16` path,
  but it has not been diagnosed.

**Memory and decode** (`reflex bench`, warmup 3, 20 iterations). Resident is the
model-load delta in free VRAM. Peak is the `nvidia-smi` maximum over idle during a
64-token `generate`.

| model | weights | resident MiB | peak MiB | prompt tokens | warm prompt p50 ms | decode ms/token |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | f32 | 1,708 | 2,549 | 29 / 113 / 449 | 22.8 / 59.5 / 265.4 | 12.64 / 13.49 / 15.80 |
| Qwen3-0.6B | f16 | **876** | **1,429** | 29 / 113 / 449 | 14.1 / 34.3 / 186.5 | **8.13** / 8.85 / 11.26 |
| Qwen3-1.7B | f32 | 5,452 | 6,997 | 29 / 113 / 449 | 58.4 / 162.2 / 616.2 | 30.66 / 31.32 / 33.70 |
| Qwen3-1.7B | f16 | **2,732** | **3,703** | 29 / 113 / 449 | 35.4 / 84.6 / 365.4 | **17.82** / 18.46 / 20.80 |

- **VRAM:** `f16` halves resident weights (−49% and −50%).
- **Decode:** at the 29-token bucket, decode drops 36% (0.6B) and 42% (1.7B) per token.
- **Warm prefill:** `cublasGemmEx` prefill is faster than `f32` at every prompt length,
  by 30–48%.

**Tests.** Host tests passed (106) and sidecar tests passed (13). The `#[ignore]`d GPU
tests pass in both modes: 19 pass and 2 are skipped (no IQ GGUF and no
DeepSeek-V2-Lite on the box).

- **Fixed during the run:** `online_attention_matches_legacy_end_to_end` failed in `f16`
  mode at 8.2e-3 against its 1e-4 bound. Its greedy tokens still matched.
- **Cause:** the `f16` activation rounding amplifies the two kernels' 2.1e-5 difference.
  The `f16`-vs-`f32` gap on the same prompt is 1.3e-2, so 8.2e-3 is below the noise
  floor. It is not a kernel bug.
- **Fix:** the test now always loads `f32` weights and keeps its 1e-4 bound.

Measured hidden-state error of the batched-vs-sequential prefill tests, against the
`f16` tolerance of `rel_l2 < 2e-2` (which was an estimate):

| test | f32 rel_l2 | f16 rel_l2 | f16 max abs |
|---|---|---|---|
| dense (Qwen3-0.6B) | 1.2e-6 | 5.6e-4 | 2.8e-2 |
| hybrid (Qwen3.5-0.8B) | 5.3e-7 | 2.2e-4 | 5.9e-4 |
| MLA fixture | 5.4e-8 | 1.2e-5 | 5.1e-7 |
| MLA fixture, resumed | 6.4e-8 | 7.3e-6 | 2.9e-7 |

The largest measured value is 36x below the estimated tolerance.

## Energy

**Energy instrumentation *and* measured energy numbers (real T4; see the phase table
below).** `src/energy.rs` samples GPU energy via NVML (`nvml-wrapper`, `dlopen`'d at runtime
behind the optional `nvml` Cargo feature — never linked at build time), preferring the
Volta+ monotonic `nvmlDeviceGetTotalEnergyConsumption` counter and falling back to polling
`nvmlDeviceGetPowerUsage` and integrating `power_mw * dt_s`. `generate`/`system1`/`smoke`/
`bench` emit per-phase `REFLEX_PHASE_OK` joules lines plus a total `joules` when available
(`--features nvml` on a machine that exposes NVML); `doctor` reports the method a real run
would actually use. These are now real-hardware-verified on a dedicated Tesla T4
(`g4dn.xlarge`): the cold-start phase table below carries a p50 **joules** column, and the
`polled_power` fallback — previously never exercised on real hardware, because that T4
always had the counter — is forceable with `REFLEX_NVML_FORCE_POLLED=1` and shown below
turning short-phase deltas non-zero. Two caveats are carried openly and were both observed:
NVML has no per-process energy API, so every figure is **device-wide** (accurate on a
dedicated/rented instance, an overcount on a shared GPU); and `total_energy_counter` mode
updates coarsely, so short phases (e.g. the ~39 ms scoring pass) legitimately read `0.000 J`
there — reported as-is, not smoothed over. The `polled_power` fallback integrates
continuously and does not have that granularity limit.

**Whole-process energy, net of idle.** The in-process figures above can only bracket
`main()`, and because NVML is device-wide they include whatever the GPU draws while
idling. `reflex-energy` (`src/bin/reflex-energy.rs`, built with `--features nvml`)
measures from outside instead: it reads the energy counter over 2 s of idle to get the
GPU's idle power, then over the command's whole life (exec and CUDA init before `main()`,
driver teardown after exit, plus a 100 ms settle), and reports gross energy, the idle
baseline for the same window, and the difference. `scripts/bench_cold_energy.sh` runs it
n times. Measured on a dedicated T4 (`g4dn.xlarge`, driver 595.91.07, Qwen3-0.6B-Q4_K_M,
`reflex system1`, n=10, 2026-10-01):

| metric | p50 | p95 |
|---|---|---|
| wall clock, spawn to exit | 592 ms | 2,992 ms |
| gross energy over the run window | 27.7 J | 104.5 J |
| idle baseline for the same window | 22.3 J | 99.3 J |
| **net energy (gross − idle)** | **5.35 J** | **5.66 J** |
| idle power | 32.1 W | 32.3 W |
| in-process `joules`, `main()` to result | 19.5 J | 93.3 J |

So most of a cold start's measured energy is the GPU idling at 32 W (this T4 sat in power
state P0 when idle); the work itself costs about **5.4 J**. The p95 column is one outlier:
the first run after the instance booted, with the model file not yet in the page cache
(the other nine took 0.58–0.60 s). Two checks on the method: `reflex-energy -- sleep 1`,
which never touches the GPU, nets 0.03–0.05 J of ~35 J gross, and the net figure barely
moves between runs (5.35 vs 5.66 J) even when the gross one quadruples. Caveats: the
counter is device-wide, so this needs a dedicated GPU, and the idle baseline depends on
the GPU's power state, so compare net numbers only across runs measured the same way.

## Kernel build modes

**Portable PTX is only cheap when the driver's JIT cache is warm.** Measured on a real
T4 (driver 595.91.07, CUDA 13.2, `Qwen3-0.6B-Q4_K_M`, `reflex system1`, p50 of 10
interleaved runs, 2026-09-30):

| build | kernel files | `reflex` binary | `model_load_ms` p50 | total p50 |
|---|---|---|---|---|
| portable PTX, **no** JIT cache | 0.50 MB | 3.02 MB | **1061** | 1272 |
| portable PTX, warm JIT cache | 0.50 MB | 3.02 MB | 246 | — |
| cubin `REFLEX_CUDA_ARCH=sm_75` | 0.39 MB | 2.92 MB | 250 | 473 |
| fatbin `REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90` | 2.10 MB | 4.63 MB | 253 | 475 |

The driver caches JIT output under `~/.nv/ComputeCache`. With that cache warm, PTX loads
as fast as a cubin (an earlier measurement of ~3 ms module load for either mode was taken
this way). Without it, every process pays the JIT again: about **+0.8 s per cold start**
on this kernel set. That is the normal case for a fresh container, a serverless worker, or
a process run with no `HOME` or with `CUDA_CACHE_DISABLE=1`. So the deploy images default
to a pinned cubin or a fatbin, and a fatbin costs nothing measurable over a pinned cubin
at load time (253 vs. 250 ms, within noise) for about 1.7 MB more binary.

Three more things the same run established:

- **A fatbin's PTX fallback only covers *newer* GPUs.** The embedded PTX targets the
  highest listed arch, so it JITs on a GPU at least that new, but a GPU *older* than every
  listed arch can't load the kernels at all (`REFLEX_CUDA_ARCHS=sm_80,sm_86` on a T4 fails
  with `CUDA_ERROR_NO_BINARY_FOR_GPU`). `Model::load` and `reflex doctor` now report that
  case in plain English up front.
- **An unlisted GPU with the same major version still loads natively.** CUDA runs an
  `sm_86` image on `sm_87`/`sm_89`, so `reflex doctor` reports those as native, not JIT.
- **CUDA 13's `nvcc` can't target `sm_70` or older** (`Unsupported gpu architecture
  'compute_70'`), so with a CUDA 13 toolkit `sm_75` (T4) is the oldest possible entry.

Run `cargo run --bin reflex -- smoke` on a real GPU instance as the very first
real-hardware step: it proves the AOT pipeline works end to end and reports actual
process-start-to-first-result wall clock on the simplest possible kernel, before any
model-architecture work begins.

## Long-context attention

The attention kernels behind this (`src/kernels_cuda/attention_online.cu`) use an online
softmax: each warp scores its own positions with a running max and sum, so shared memory
no longer grows with the sequence, and long decode contexts are split across blocks. The
original kernels kept every score in shared memory, which capped a sequence at 11,264
positions; they stay selectable with `REFLEX_ATTN_KERNEL=legacy` (and keep that limit)
for comparison. Measured on a T4 (Qwen3-0.6B-Q4_K_M, 65 generated tokens, same binary,
2026-09-30):

| context | prefill, legacy → online | decode per token, legacy → online |
|---|---|---|
| ~128 tokens | 354 → 344 ms | 16.6 → 13.4 ms |
| ~2K tokens | 6.52 → 2.86 s | 68.9 → 16.7 ms |
| ~8K tokens | 363.7 → 45.4 s | 235.5 → 27.8 ms |
| ~16.5K tokens | over the legacy limit | prefill 190 s (online only) |

Greedy tokens are identical between the two on every tested model (dense, hybrid, MLA and
both MoE fixtures) and first-token logits agree to within 2.2e-5. Prefill is still
quadratic in prompt length and runs in plain `f32` without tensor-core tiling, so very
long prompts remain slow; cold start, not long-context throughput, is what this engine
optimizes.

## Container images

On a genuinely cold worker, image pull precedes everything else, and the standard
CUDA `-runtime-` base image is mostly libraries this engine never loads. Reflex
links only the CUDA **driver API** plus **cuBLAS** (`Cargo.toml`'s `cudarc`
features are `driver`/`cublas`/`cuda-12000`/`f16`); none of the
`cuda-libraries-12-4` meta-package's NCCL, cuFFT, cuSPARSE, cuSOLVER, NPP or
nvJPEG are referenced anywhere in `src/`.

Every image built from this repo now uses CUDA's `base` image plus `libcublas-12-4` (for
`libcublas.so.12` and `libcublasLt.so.12`) as its runtime, and all but the Modal image
come from one multi-target root `Dockerfile` (`--target reflex`, the default, `adapter`,
`runpod-lb`; `.runpod/Dockerfile` must stay separate for Runpod's Hub pipeline and
mirrors it). Kernels default to the multi-arch fatbin (see
[Kernel build modes](#kernel-build-modes)).

Measured on a dedicated AWS `g4dn.xlarge` (Tesla T4, Docker 29.8.1, BuildKit),
2026-09-30, each image built with its own defaults before and after:

| image | before (`-runtime-` base) | after (`base` + cuBLAS) | change |
|---|---|---|---|
| `reflex` (root, default target) | 3.78 GB | 1.27 GB | −66% |
| `adapter` (was `sidecar/openai-adapter/Dockerfile`) | 3.78 GB | 1.28 GB | −66% |
| `runpod-lb` (was `serverless/runpod/Dockerfile`; model baked in) | 4.57 GB | 2.06 GB | −55% |
| `.runpod/Dockerfile` (Hub; model + Python baked in) | 4.87 GB | 2.37 GB | −51% |

The slim images contain no CUDA library besides cuBLAS and `libcudart` (no NCCL, cuFFT,
cuSPARSE, cuSOLVER or NPP). `ldd` can't confirm cuBLAS is found, because `cudarc` loads it
with `dlopen` at runtime, so the check that matters is a real run: `reflex generate`
(which uses cuBLAS for batched prefill) returns the same tokens from the fatbin,
single-arch `sm_75` and portable-PTX variants, and the `adapter`, `runpod-lb` and Hub
images each answered a real `/v1/chat/completions` request (the Hub one through
`handler.py --test_input`).

The kernel default matters as much as the size. In a fresh container, the old root
image's portable-PTX kernels pay the driver JIT on every start; the new fatbin doesn't:

| `reflex system1`, fresh container each run (3 runs) | model load | total |
|---|---|---|
| old root image (portable PTX) | 1021–1072 ms | 1218–1308 ms |
| new root image (fatbin) | 232–234 ms | 435–441 ms |

## Serverless (Runpod)

**Real measurement, and a recalibration.** This has been deployed and tested end-to-end
against a real Runpod account (2026-09-26, real RTX A4500) — see
[`serverless/runpod/README.md`](../serverless/runpod/README.md#real-deployment-findings-2026-09-26-real-rtx-a4500-on-runpod)
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
[`docs/serverless-cost-comparison.md`](serverless-cost-comparison.md). Also found and
fixed during this deployment: Runpod's load-balancer health check hits a hardcoded `/ping`
path regardless of documented override variables, and the request that triggers a cold
start can get a `502` from Runpod's own gateway even though the worker becomes healthy
moments later — both are now documented in the sidecar's and `serverless/runpod/`'s READMEs
for anyone else hitting the same platform.

**`serverless/runpod/` has since been deployed against a real Runpod account and works
end-to-end** — see [Where this engine competes](#where-this-engine-competes) above and
`serverless/runpod/README.md`'s "Real deployment findings" for the measured numbers, the
platform's actual (undocumented) health-check behavior, and a gateway quirk found along
the way. The one thing still unconfirmed is the exact dollar amount billed for those test
invocations — Runpod's billing API hadn't reconciled the relevant hour yet at time of
writing; the per-second rate itself is confirmed from the live catalog. That deployment
used the earlier image (full `-runtime-` base, `sm_86` cubin); the slim fatbin `runpod-lb`
image above has not been redeployed to Runpod yet.
