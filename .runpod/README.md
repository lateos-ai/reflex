# Reflex on the Runpod Hub (queue-based worker)

This directory exists for exactly one reason: **Runpod's Hub publishing pipeline** — the
public, browsable catalog other people one-click-deploy from (e.g.
`runpod-workers/worker-vllm`) — only supports **queue-based** workers (a Python handler using
`runpod.serverless.start()`). It does not currently support publishing a **load-balancing**
endpoint to the Hub at all.

[`serverless/runpod/`](../serverless/runpod/README.md) is the actual recommended deployment
path for Reflex on Runpod: it's a real, measured, load-balancing-endpoint deployment with
~3.4x faster cold start than Runpod's own official `worker-vllm` (see
[`docs/serverless-cost-comparison.md`](../docs/serverless-cost-comparison.md)), and it needs
zero Python glue code. **If you're deploying manually rather than discovering Reflex via the
Hub, use `serverless/runpod/` instead — it's the leaner path.**

This directory is a second, parallel, purely additive build that exists solely so Reflex can
appear in the Hub's catalog. It does not replace or modify `serverless/runpod/` in any way.

## What `handler.py` actually is

A translation shim, not a reimplementation. On container start it spawns the exact same,
unmodified `reflex-openai-adapter` binary `serverless/runpod/` uses (over an internal-only
loopback port — queue-based endpoints don't proxy external HTTP to a configured port the way
load-balancing endpoints do), waits for its existing three-state `/healthz` check to report
ready, and then forwards each Runpod job's `input` straight to that process's existing
`POST /v1/chat/completions`, unmodified. No IPC/HTTP protocol logic is reimplemented in
Python — see [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md)'s non-goals: the core engine never grows a network socket or
an internal queue/scheduler, and this handler doesn't change that. The queue here is Runpod's
own platform queue, external to the engine, exactly like `serverless/runpod/`'s load balancer
is also external to the engine.

## Job input and output

The job `input` is an OpenAI chat-completions request body (`messages`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stream`), passed to the adapter as is.

**Classification**: a job whose `input` has `labels` goes to the adapter's
`POST /v1/classify` instead (see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md#post-v1classify)):
one engine pass scores each label as a continuation of `prompt` (or of `messages`,
chat-templated) and returns their probabilities. Its output is a one-element list
holding that `classification` object, like a non-streaming chat job:

```json
{"input": {"prompt": "Review: The battery died after two days. Sentiment:",
           "labels": [" positive", " negative"]}}
```

`handler` is a Runpod **generator** handler started with `return_aggregate_stream`, the
same convention `runpod-workers/worker-vllm` uses:

- **`"stream": true`**: each Server-Sent Event the adapter sends is yielded as one
  `chat.completion.chunk` object. Read them live from `/stream/{job_id}`; `/run` +
  `/status` and `/runsync` return the whole list once the job finishes.
- **otherwise**: the single complete `chat.completion` object is yielded once, so the job's
  `output` is a **one-element list** `[{...}]`. Up to `v0.2.2-runpod-hub` this handler
  returned that object bare and forced `stream` to `false`.

An adapter error (`400`/`429`/`503`/`504` before the first token, or a mid-stream error
event such as the request timeout) fails the job, with the adapter's OpenAI-shaped error
JSON as the job's `error`.

`ADAPTER_ARGS` (an advanced field in the Hub's deploy form, empty by default) appends
`reflex-openai-adapter` flags, e.g. `--max-tokens-cap 1024 --default-max-tokens 128`; see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md)'s flag list and
request limits.

## GPU pin: portable PTX, not sm_86 (unlike `serverless/runpod/`)

The same mixed-architecture hazard documented in
[`serverless/runpod/README.md`'s GPU-selection section](../serverless/runpod/README.md)
applies to this pool (`AMPERE_16`: Ampere `sm_86` RTX A4000/A4500 alongside Ada `sm_89` RTX
2000/4000 Ada), but the *fix* is different here. `serverless/runpod/`'s endpoint is a real,
already-deployed resource whose GPU can be pinned directly via `set-endpoint-gpus`. A Hub
listing has no equivalent lever: Runpod's own Hub build/test pipeline schedules the automated
test job wherever it wants, and **real evidence (2026-09-27, below) shows it does not reliably
honor either `hub.json`'s `gpuIds` exclusion list or `tests.json`'s `gpuTypeId` pin.** Since an
`sm_86`-pinned cubin crashes outright on an `sm_89` card (not just slower — a hard panic), this
build first switched to **portable PTX**, JIT-compiled by the driver to whichever GPU is present.
That works anywhere but was later measured to cost about 0.8 s of JIT on every fresh worker (a
new container has no driver JIT cache). It now builds a **multi-arch fatbin**
(`REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90` in `.runpod/Dockerfile`): a native image for
every card in the pool, Ada included, plus PTX that newer GPUs JIT. So it runs whichever card the
Hub picks, with no JIT on the listed ones. Confirmed on real Runpod workers on 2026-10-02 (see
"Fatbin on Ada" below), so `hub.json` now allows the whole `AMPERE_16` pool, Ada cards included.

## Non-goals for this addition

- No changes to `serverless/runpod/`, `sidecar/openai-adapter/`, or any core engine file. Both
  deployment paths remain independently valid.
- No new engine capability, no batching, no protocol redesign.
- Not a promise the Hub submission will be approved on any particular timeline — the
  automated build/test gate (`tests.json`) and Runpod's manual review are their process, not
  ours to control.

## Real deployment findings (2026-09-27, real RTX A4500 on Runpod)

This build has been tested end-to-end against a real, temporary queue-based Runpod endpoint
(created directly via the API for pre-submission verification, not via the Hub — the endpoint
was deleted immediately after this test).

**GHCR image visibility.** Runpod had no container registry credential configured for this
account, so a private image couldn't be pulled. The image (`ghcr.io/lateos-ai/reflex-runpod-hub`)
had to be made public before the test endpoint could start. If a private image is ever needed
here instead, a registry credential must be created in Runpod first (a GitHub PAT with
`read:packages` scope) — there is no way around one of these two options.

**The GPU-arch pin worked exactly as designed.** Both workers that spun up during this test
landed on the pinned RTX A4500 (`sm_86`), confirmed via `list-endpoint-workers`.

**The same autoscaler-overrun bug already documented in
[`serverless/runpod/README.md`](../serverless/runpod/README.md) reproduced here too.** With
`workers.max` explicitly set to `1`, `list-endpoint-workers` showed **2** workers (one
`RUNNING`, one `INITIALIZING`) shortly after the first job completed. This is the same platform
behavior already seen on two prior test endpoints for the load-balancing deployment — a live
Runpod bug, not a configuration mistake on this project's side. Anyone deploying this worker for
real should not assume `workers.max` is a hard ceiling and should monitor actual worker count
independently if cost bounding matters.

### Real measured numbers (queue-based endpoint)

A real cold invocation (`min=0` workers, genuine scale-from-zero) via `runsync`/`get-job-status`:

| Phase | Time | Source |
|---|---|---|
| `delayTime` (Runpod's own queueing + cold-start provisioning) | **~71.3s** | Real job status |
| `executionTime` (the actual chat-completion call, warm) | **~1.2s** | Real job status |

**This is a different profile from the load-balancing deployment's ~42-44s** (see
`serverless/runpod/README.md`) — expected, since a queue-based endpoint's delay/execution split
is a different mechanism from an LB endpoint's direct HTTP proxy, and this hasn't been tuned or
re-measured across multiple runs the way the LB deployment's numbers were. Treat this as a first
real data point, not a final benchmark.

## Real Hub build/test findings (2026-09-27)

After linking the `lateos-ai/reflex` repo in Runpod's Hub console (under the `lateos-ai` org,
not a personal account — the account/org selector in the "Add Repo" dialog defaults to the
personal account, which is easy to miss if the target repo lives in an org), Runpod's own
pipeline built `.runpod/Dockerfile` against the `v0.2.0-runpod-hub` release. **The Docker build
itself succeeded** — the build-time model fetch (see `.runpod/Dockerfile`'s comment) worked
correctly against Runpod's own build infrastructure, not just locally.

**The automated test job failed**, and the failure is the reason this build no longer pins
`sm_86` (see the GPU pin section above): the test worker's own startup log reported `GPU: NVIDIA
RTX 2000 Ada Generation (sm_89)` — one of the exact two SKUs `hub.json`'s `gpuIds` and
`tests.json`'s `gpuTypeId: "NVIDIA RTX A4500"` both tried to exclude/pin away from. The
`sm_86`-compiled `reflex` binary panicked on load with a clear, correct error (`"this binary's
CUDA kernels were compiled for compute capability 8.6 (sm_86), but the detected GPU ... has
compute capability 8.9"`), and the test timed out. This is strong evidence that at least one of
Runpod's Hub-specific GPU-selection fields is not honored by the Hub's own test-scheduling
infrastructure, independent of whatever `set-endpoint-gpus`/`gpu.excludedTypes` behavior a real
deployed endpoint has (which `serverless/runpod/`'s real deployment *did* confirm works
correctly — see that README). Switching to portable PTX resolves this by making the GPU
architecture irrelevant to correctness.

Also fixed from this same first attempt: `hub.json`'s `category` field was set to
`"language-models"`, a value not in the Hub UI's actual set (`Image`/`Video`/`Audio`/
`Language`/`Embedding`, confirmed by inspecting the live "Add Repo" form) — corrected to
`"language"`.

## Fatbin on Ada (2026-10-02)

The Ada exclusions in `hub.json` were removed after these checks on real Runpod serverless
workers, all from the fatbin images:

- This directory's queue image (`ghcr.io/lateos-ai/reflex-runpod-hub:latest`, rebuilt from
  `.runpod/Dockerfile`, 2.37 GB, down from 4.86 GB before the slim runtime) on an endpoint
  allowing the whole `AMPERE_16` pool, with no exclusions: the `tests.json` smoke input completed
  on an RTX A4500 (`sm_86`), both from a cold worker and warm. The scheduler also started a
  second worker on an **RTX 2000 Ada** (`sm_89`) in EUR-IS-1, but it stayed `THROTTLED` (no
  capacity at that host) and never served a job.
- The load-balancing image (root `Dockerfile`, `--target runpod-lb`, same kernel build args)
  served `/v1/chat/completions` on an **NVIDIA L4** (Ada, `sm_89`, the same compute capability
  as the RTX 2000/4000 Ada), engine ready ~0.4 s after container start, same output as on the
  A4500. Neither small `AMPERE_16` Ada SKU had serverless capacity on CUDA 12.x/13.0 hosts
  during the run, which is why the L4 stood in.

So the `sm_89` code path that used to panic on the RTX 2000 Ada is verified on Runpod, and the
pool's Ampere cards keep working. The listing picks up the new `hub.json` with the next Hub
release.

## Status

**Listed in Runpod's Hub catalog** (checked 2026-10-07): the Hub's public catalog API
returns `lateos-ai/reflex` (listing id `cmuj7rnqr000007jwfsbb4ipc`) both for an owner filter
and for a plain `reflex` search, which it did not on 2026-09-30. The listing page is
<https://console.runpod.io/hub/lateos-ai/reflex>; `https://www.runpod.io/hub/lateos-ai/reflex`
still 404s. The listed release at that check was **`v0.2.2-runpod-hub`** (build image
`registry.runpod.net/lateos-ai-reflex-master-runpod-dockerfile:16370973d`), which predates the
fatbin kernels and the slim runtime, so every fresh worker still pays the PTX JIT.
**`v0.2.3-runpod-hub`** brings the fatbin build, the Ada-inclusive `gpuIds`, streaming, and
the `ADAPTER_ARGS` deploy field.

**Update, checked 2026-10-09:** the listing has moved to **`v0.2.3-runpod-hub`**. The Hub
console (listing page <https://console.runpod.io/hub/lateos-ai/reflex>) now shows the
v0.2.3 release promoted from `master` at tag `v0.2.3-runpod-hub`, completed 2026-10-08,
with the fatbin/slim-runtime description and its test results passing. The build-image
digest is not exposed in the listing UI; the release name is what the Hub builds from, so
the PTX JIT the v0.2.2 image paid is no longer in the catalog path. The GHCR image the Hub
publishes is `ghcr.io/lateos-ai/reflex-runpod-hub`, digest (verified 2026-10-09)
`sha256:6ecb8e1f0200c2736141014eadc9ad38bee7560fad52b48e02dbacd31d510f33`. The v0.2.2 note
above is kept for history. See `docs/ADOPTION_PLAN.md` task A01.

`iconUrl` in `hub.json` points at a real hosted asset (`.runpod/icon.jpg` on `master`), not
the old `TODO:` placeholder. The GPU-selection fields in `hub.json`/`tests.json` are kept as
advisory intent only (real evidence above shows the Hub test scheduler does not honor them).

