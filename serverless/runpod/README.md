# Reflex on Runpod Serverless (load-balancing endpoint)

This packages the existing [`sidecar/openai-adapter`](../../sidecar/openai-adapter/README.md)
— unmodified, no new handler/glue code — as a Runpod Serverless deployment. It exists to
produce a real, measured comparison of cost-per-cold-invocation against vLLM on the same
platform/GPU, as part of evaluating whether Reflex's cold-start advantage translates into a
real cost advantage on serverless GPU billing (which bills cold-start/init time as compute,
unlike a per-token API marketplace). See the root project's `DECISIONS.md`/`HISTORY.md` for the
full reasoning behind this angle and why OpenRouter's per-token pricing model doesn't work for
an engine that deliberately never batches concurrent requests.

**Nothing about this changes the core engine or the sidecar's concurrency model.** This is
packaging only, running the exact same `reflex stdio`-owning, single-worker-queue sidecar
described in its own README, just fed by a Runpod-managed HTTP load balancer instead of a
directly-exposed port.

## Why a load-balancing endpoint, not a queue-based endpoint

Runpod Serverless has two distinct endpoint types, and they are not interchangeable:

- **Queue-based endpoints** require a Python handler using Runpod's own SDK
  (`runpod.serverless.start({"handler": ...})`), with jobs delivered as a
  `{"id": ..., "input": {...}}` envelope your handler must parse. This would mean writing and
  maintaining a Python wrapper that shells out to a compiled Rust binary — an extra moving part
  this project has no other use for.
- **Load-balancing endpoints** instead run a plain, arbitrary HTTP server on a configured port,
  with Runpod's load balancer proxying real HTTP requests straight through — "any HTTP
  framework... in any language" per Runpod's own docs. This is exactly the existing
  `reflex-openai-adapter` binary's shape (an axum HTTP server), so it runs here with zero new
  code, just Dockerfile/environment-variable wiring.

**Use a load-balancing endpoint.** A queue-based endpoint is the wrong fit here and would add
unnecessary Python glue code for no benefit.

## Scripted deployment (`scripts/deploy_runpod.sh`)

The manual steps above are scripted in
[`scripts/deploy_runpod.sh`](../../scripts/deploy_runpod.sh), which builds
`serverless/runpod/Dockerfile`, pushes the image, creates the (load-balancing) template +
endpoint, pins the RTX A4500 SKU, and then drives the cold-start benchmark through
[`scripts/bench_cold_runpod.sh`](../../scripts/bench_cold_runpod.sh) — itself a thin wrapper
over `bench_cold_common.sh`'s `/usr/bin/time -v` loop. It encodes the verified findings in
this directory rather than re-deriving them:

- Endpoint creation goes through the GraphQL `saveEndpoint` mutation, the only Runpod API
  that exposes the load-balancing `type: "LB"` field.
- The exact SKU is pinned immediately after creation with a REST `gpuTypeIds` PATCH, because
  the GraphQL create path can only name a *pool* (`gpuIds: "AMPERE_16"`), and that pool is
  mixed-architecture (see "GPU selection" below).
- The benchmark retries Runpod's documented first-cold-request `502` and, before each timed
  run, forces a genuine scale-from-zero (`workersMax` 0 → wait out `idleTimeout` → 1), so
  runs 2..N don't silently measure a warm worker.

`--slim` additionally builds the runtime-slimmed image (`base` + `libcublas-12-4`, tagged
`:runtime`) described in the root README's "Container image size is part of cold start
here"; like every other default here, the full `-runtime-` base stays the default and the
slim variant is opt-in. `--teardown` deletes the endpoint and template after benchmarking.
Run `scripts/deploy_runpod.sh --help` for the full flag/environment reference.

## Runpod endpoint configuration

When creating the Serverless endpoint in Runpod's console/API:

- **Endpoint type**: Load Balancing.
- **Container image**: the image built from this directory's `Dockerfile`.
- **Exposed HTTP port**: matches this image's `PORT` environment variable (default `80` — set
  both consistently if you override it; Runpod requires the exposed port to be explicitly
  declared in the endpoint's container configuration, not just implied by the `Dockerfile`).
- **Health check**: set `HEALTH_CHECK_PATH=/healthz` if you like, but **do not rely on it** —
  confirmed against a real deployment that Runpod's load-balancer gateway polls the hardcoded
  path `/ping` regardless of this variable (its own console UI says so plainly; the variable
  appears to be accepted but not actually honored for routing decisions). The sidecar serves
  the identical three-state handler at both paths (`204` while the managed `reflex` process is
  alive but still loading, `200` once ready, `503` if it died) specifically so this works either
  way — see "Real deployment findings" below for how this was discovered.
- **Environment variables**: `GGUF_PATH` (path to the model file inside the container — see
  below; this deployment bakes it in rather than using a volume), `PORT` (only if not using the
  default `80`).
- **Pin the GPU SKU** — see the architecture warning below. Do not leave the endpoint free to
  pick any SKU in its pool.

## GPU selection: pin the SKU, don't trust the pool name

Verified against the live Runpod catalog (`list-gpu-types`, `product=SERVERLESS`), not
assumed: the cheapest serverless pool, **`AMPERE_16` at $0.58/hr**, is
**mixed-architecture despite its name**:

| GPU in `AMPERE_16` | VRAM | Arch | Compute capability | Serverless availability |
|---|---|---|---|---|
| RTX A4500 | 20GB | Ampere | `sm_86` | **HIGH** (EU-RO-1) |
| RTX A4000 | 16GB | Ampere | `sm_86` | LOW |
| RTX 4000 Ada | 20GB | Ada | `sm_89` | LOW |
| RTX 2000 Ada | 16GB | Ada | `sm_89` | LOW |

An `sm_86`-pinned cubin **will not run** on the two Ada cards. An endpoint left free to
schedule anywhere in this pool therefore fails nondeterministically depending on which SKU a
worker lands on — the same hazard [`docs/aws-deployment.md`](../../docs/aws-deployment.md)
warns about for mixed-family ASGs, which is easy to miss here because the pool's *name* implies
homogeneity.

**Pin the endpoint to RTX A4500** via the `set-endpoint-gpus` control-plane tool (the routing
type and `gpuPoolIds` on `create-endpoint` can't express a SKU). A4500 is the right pick: it's
`sm_86`, has the most VRAM in the tier at 20GB, and is the only card in the pool with HIGH
serverless availability — at the same $0.58/hr pool price.

If you would rather not pin, the alternative is a portable-PTX build (`REFLEX_CUDA_ARCH`
unset), which runs on any of them — but it reintroduces driver-side JIT at module load, which
is precisely the cold-start cost this engine exists to avoid, and would corrupt the benchmark
this deployment is meant to produce.

**Driver/CUDA note**: A4500 workers report CUDA 12.8/13.0/13.2 available and 12.4 *not*. The
image ships CUDA 12.4 runtime libraries, which is fine — newer drivers run older CUDA runtimes
— but be aware the host driver will be 12.8 or newer, not the image's 12.4.

### Verified serverless pricing (live catalog, this account)

| Pool | VRAM | $/hr |
|---|---|---|
| `AMPERE_16` | 16–20GB | **$0.58** |
| `AMPERE_24` (A5000, L4) | 24GB | $0.69 |
| `ADA_24` (RTX 4090) | 24GB | $1.10 |
| `AMPERE_48` (A40, A6000) | 48GB | $1.22 |

At per-second billing, $0.58/hr is **$0.000161/sec** — the figure any cost-per-invocation
arithmetic here should use.

## Model weights: baked into the image

This deployment bakes the model directly into the image (`COPY serverless/runpod/model.gguf
/models/model.gguf` in the Dockerfile, `GGUF_PATH` defaulted to that path) rather than using a
Runpod Network Volume. Deliberate, not just simpler: a runtime download would add
HuggingFace-fetch latency directly into the cold-start number this deployment exists to
measure, corrupting the benchmark. Qwen3-0.6B-Q4_K_M is ~379MB, small enough that baking it
in costs a proportionally small amount of image size (1.2GB on-disk rootfs with the slim
`base` + cuBLAS runtime variant, vs. 2.6GB on the full `-runtime-` base — see the size
discussion in the root README).

A Network Volume remains the better choice if you need to swap models without rebuilding the
image, or the model is too large to comfortably bake in — same
avoid-re-downloading-on-every-cold-start rationale as the EFS cache in
[`docs/aws-deployment.md`](../../docs/aws-deployment.md). This repository doesn't automate
that path; it's a one-time manual/scripted step against Runpod's API if you need it.

## Real deployment findings (2026-09-26, real RTX A4500 on Runpod)

This has now actually been deployed and tested end-to-end, not just written against
documentation. Two things were wrong on the first attempt, both fixed:

**1. Runpod's health check hits `/ping`, not the documented `HEALTH_CHECK_PATH` override.**
The endpoint's own console UI states plainly: workers must have "an accessible `/ping`
endpoint ... responds with a 200 status code" — no mention of the `HEALTH_CHECK_PATH`
environment variable actually being honored, despite that variable being accepted (and
seemingly documented) elsewhere. A worker that loaded correctly and reported `/healthz` as
healthy never received a single request, because Runpod's gateway was polling `/ping` (404,
unhandled) the whole time — confirmed by zero new container log lines across multiple
requests to a demonstrably ready worker. **Fix**: the sidecar now serves the identical
three-state handler at both `/healthz` and `/ping` (`sidecar/openai-adapter/src/main.rs`).
If you deploy this pattern to another platform, check what path *it* actually polls rather
than trusting a documented override — this cost real debugging time here.

**2. The GPU-arch pin worked exactly as designed.** The worker landed on the pinned RTX A4500
(`sm_86`) as intended, confirmed via `nvidia-smi`-equivalent output in the container's own
startup log (`GPU: NVIDIA RTX A4500 (sm_86), VRAM: 19852/20042 MiB free`). No issue here — this
one just worked.

### Real measured numbers

Once `/ping` was fixed, a real cold invocation (`min=0` workers, genuine scale-from-zero)
succeeded end to end. Breaking down where the time actually goes:

| Phase | Time | Source |
|---|---|---|
| Reflex's own load (container start → `REFLEX_STDIO_READY`) | **~1.6s** | Real container logs |
| Runpod's own platform provisioning (GPU scheduling + container start, before Reflex even begins) | **~40s** | Derived: total minus Reflex's own load |
| **Total cold invocation, wall clock** | **~42-44s** | Real `curl` timing, `n=2` |

**This recalibrates the headline claim.** Runpod's own scheduling/provisioning floor is
~40 seconds regardless of engine — no engine can make a serverless cold start on this platform
sub-second, because most of the time is platform overhead that happens *before* any engine
code runs. The claim this deployment can actually support is narrower than "Reflex starts in
under a second on serverless": it's that **Reflex adds near-zero marginal time on top of that
platform floor**, where a JIT/graph-compiling engine (vLLM's CUDA graph capture, documented
elsewhere in this repo as taking minutes) would add substantially more on top of the same
floor. That's a real, still-differentiated claim — just not the one the pre-deployment framing
implied. See the root README's serverless section, updated to match.

### A real platform quirk: first cold request can 502

The request that *triggers* a scale-from-zero cold start sometimes gets a `502` from Runpod's
own gateway (observed twice), even though the worker keeps initializing in the background and
serves the *next* request successfully seconds later. This looks like a gateway-side upstream
timeout shorter than actual cold-start duration, not an application bug — confirmed via worker
logs and status showing the container became healthy shortly after the 502 was returned to the
client. **Practical implication**: a real integration calling this endpoint needs client-side
retry logic for the first request after an idle period, not just a single-shot call. Also
observed: with the default `idleTimeout: 10`, the window to catch a worker actually still warm
is narrow — a few seconds of diagnostic delay between requests is enough to trigger another
full cold start.

## What's confirmed vs. not yet verified

**Deployed and verified end-to-end against a real account** (2026-09-26, see "Real deployment
findings" above): custom HTTP servers work on load-balancing endpoints; the platform's actual
health-check path is `/ping` (not the documented `HEALTH_CHECK_PATH` override — a real
discrepancy between docs and observed behavior, not an assumption); the GPU-arch pin correctly
lands workers on the intended SKU; cold-invocation wall-clock time (~42-44s, dominated by
Runpod's own provisioning, not Reflex); and the first-cold-request `502` behavior.

Also confirmed, from the live account catalog rather than marketing pages: the per-hour rate
for each serverless pool, and the mixed-architecture composition of `AMPERE_16` (both
tabulated above).

**Still not confirmed**: the exact dollar amount billed for these test invocations — Runpod's
billing API showed no records yet for the hour these tests ran in (a reconciliation lag, not a
missing charge). The per-second rate itself is confirmed from the catalog; whether
cold-start/initialization time specifically is billed as compute on load-balancing endpoints
has not been directly confirmed from a reconciled invoice. Check the billing dashboard once
these charges settle before publishing a hard cost-per-invocation figure.
