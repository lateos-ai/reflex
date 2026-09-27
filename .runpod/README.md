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
Python — see the root `CLAUDE.md`'s non-goals: the core engine never grows a network socket or
an internal queue/scheduler, and this handler doesn't change that. The queue here is Runpod's
own platform queue, external to the engine, exactly like `serverless/runpod/`'s load balancer
is also external to the engine.

## Known v1 limitation

**Non-streaming only.** The `stream` field is forced to `false` regardless of what a caller
sends. Streaming would require translating Server-Sent Events into a Runpod generator
handler — real new logic, out of scope for a shim whose only job is speaking Runpod's job
envelope format. This is a deliberate, documented v1 limitation, not an oversight.

## GPU pin

Same hazard and same fix as `serverless/runpod/`: the cheapest pool, `AMPERE_16`, is
mixed-architecture (Ampere `sm_86` RTX A4000/A4500 alongside Ada `sm_89` RTX 2000/4000 Ada) —
see [`serverless/runpod/README.md`'s GPU-selection section](../serverless/runpod/README.md)
for the full table and reasoning. `hub.json`'s `gpuIds` excludes both Ada SKUs from the pool
(`AMPERE_16,-NVIDIA RTX 2000 Ada Generation,-NVIDIA RTX 4000 Ada Generation`) so this sm_86
build only ever schedules onto compatible hardware.

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

## Status

Not yet submitted to the Hub (that step needs the repo owner to link the GitHub repo in
Runpod's console, plus a tagged GitHub release — see the project plan). `iconUrl` in `hub.json`
is still a placeholder pending a real hosted icon asset.
