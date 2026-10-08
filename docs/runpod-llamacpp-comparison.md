# Reflex vs. llama.cpp cold-start comparison on Runpod Serverless

**Status: done.** n=5 successful cold invocations per engine, from two sessions on the
same GPU SKU (2026-10-01 and 2026-10-02), plus the `--no-warmup` variant (n=3). See
[Results](#results) for what the numbers do and don't support.

## Why this comparison

The existing [Reflex vs. vLLM comparison](serverless-cost-comparison.md) found Reflex's
cold invocation ~3.4x faster, but it mostly measures vLLM's CUDA-graph capture and its
model download at cold start, and it had to compare a load-balancing endpoint against a
queue-based one. Locally, llama.cpp is the closer competitor: it also loads GGUF files
and also ships precompiled CUDA kernels. This page measures the two on the same
platform under the same conditions, whichever way the result goes.

This comparison is more controlled than the vLLM one:

| | Reflex | llama.cpp |
|---|---|---|
| Image | root `Dockerfile`, `--target runpod-lb` (slim runtime, multi-arch fatbin) | [`serverless/runpod-llamacpp/Dockerfile`](../serverless/runpod-llamacpp/Dockerfile): official `ghcr.io/ggml-org/llama.cpp:server-cuda-b11277` + nginx |
| Server | `reflex-openai-adapter` over `reflex stdio` | `llama-server` behind nginx |
| Model | Qwen3-0.6B Q4_K_M, the **same file** (`serverless/runpod/model.gguf`), baked into both images | same |
| Endpoint type | load-balancing (raw HTTP) | load-balancing (same) |
| Health contract | `/ping`: 204 loading, 200 ready | same (nginx maps `/ping` onto `llama-server`'s `/health`) |
| Concurrency | one request at a time | `-np 1` (one slot) |
| Context | sized per request | `-c 4096` (llama-server would otherwise allocate the KV cache for the model's full 40,960-token context at startup) |
| GPU offload | all layers | `-ngl 99` (all layers) |
| Warmup | none | llama-server's default warmup run, left on (see the `--no-warmup` variant below) |
| Deploy and measure | `scripts/deploy_runpod.sh`, `scripts/bench_cold_runpod.sh` | the same two scripts, unchanged |

## Pre-check on a T4 (2026-10-01): what the images do, and a GPU trap

Before handing this over, both images were built and run with Docker on a dedicated AWS
T4 (`g4dn.xlarge`), the image already present locally (no pull). This is **not** the
Runpod measurement; it validated the images and turned up one thing that decides which
GPU the comparison must use.

- Both images serve `/ping` correctly (204 while loading, 200 when ready) and answer
  `/v1/chat/completions`.
- Image sizes: **Reflex 2.06 GB, llama.cpp 7.79 GB** (each including the 379 MB model).
- **The official llama.cpp image has native GPU code only for `sm_86`, `sm_89` and
  `sm_120a`** (`cuobjdump --list-elf /app/libggml-cuda.so`). Every other architecture,
  including the T4 (`sm_75`), A100 (`sm_80`) and H100 (`sm_90`), gets embedded PTX that
  the driver JIT-compiles on every fresh container. On the T4 that took tens of seconds:
  docker run to first answered request was **~35-69 s for llama.cpp vs. 1.0 s for
  Reflex**, and the timestamped log shows llama-server spending 35 s between thread-pool
  init and slot setup. With the driver's JIT cache persisted across containers, the
  second llama.cpp start took **1.72 s**, which confirms the cause.
- So on the T4, the like-for-like local numbers (JIT taken out) are **1.0 s for Reflex vs.
  1.72 s for llama.cpp**, docker run to first answered request, n=3 and n=1 respectively.

**Consequence for this comparison: use a GPU the llama.cpp image has native code for.**
The default RTX A4500 is `sm_86`, so it qualifies; the measurement then compares the
engines' real load times rather than a driver JIT. On a T4, A100 or H100, llama.cpp's
official image pays that JIT on every cold start, a real cost of deploying it there but a
different finding; report it separately if you measure it.

## Procedure

Run both engines **on the same day, in the same data center, on the same GPU SKU**,
interleaving the runs so platform drift affects both equally.

### 1. Build and push the two images

From the repo root, with `serverless/runpod/model.gguf` in place (the same Qwen3-0.6B
Q4_K_M file the earlier Runpod measurements used):

```bash
REG=ghcr.io/<you>   # a registry Runpod can pull from

docker build --target runpod-lb -t $REG/reflex-runpod:cmp .
docker build -f serverless/runpod-llamacpp/Dockerfile -t $REG/llamacpp-runpod:cmp .
docker push $REG/reflex-runpod:cmp
docker push $REG/llamacpp-runpod:cmp
docker images | grep -E 'reflex-runpod|llamacpp-runpod'   # record both sizes
```

### 2. Create two endpoints with identical settings

`scripts/deploy_runpod.sh` creates a load-balancing endpoint, pins the GPU SKU (RTX A4500
by default) and turns FlashBoot off. Use it for both images, without building:

```bash
export RUNPOD_API_KEY=...
REFLEX_REGISTRY_IMAGE=$REG/reflex-runpod REFLEX_IMAGE_TAG=cmp \
  REFLEX_ENDPOINT_NAME=cmp-reflex REFLEX_LOCATIONS=RO \
  scripts/deploy_runpod.sh --no-build --no-push --no-bench
REFLEX_REGISTRY_IMAGE=$REG/llamacpp-runpod REFLEX_IMAGE_TAG=cmp \
  REFLEX_ENDPOINT_NAME=cmp-llamacpp REFLEX_LOCATIONS=RO \
  scripts/deploy_runpod.sh --no-build --no-push --no-bench
```

Both endpoints get `workersMin=0`, `workersMax=1`, a 10 s idle timeout and FlashBoot
off. Keep those defaults: they match the earlier comparison.

**Set the same CUDA floor on both.** The llama.cpp image is built on CUDA 12.8 and
needs a host driver that supports it; the Reflex image is built on CUDA 12.4 and runs
on more hosts. Without a shared floor, the two endpoints may draw workers from different
host pools. Set `gpu.minCudaVersion` to `12.8` on **both** endpoints, in the console
(endpoint → Edit → CUDA version) or through the REST v2 `PATCH /endpoints/{id}` with
`{"gpu": {"minCudaVersion": "12.8"}}`. Changing only the CUDA floor leaves the SKU pin
alone.

### 3. Measure cold invocations

`scripts/bench_cold_runpod.sh` times one cold `POST /v1/chat/completions` per run. Before
each run, outside the timed window, it pins the endpoint's maximum workers to 0, waits
out the idle timeout so the platform tears the worker down, and allows one worker again.
That forces a genuine scale from zero every time. Alternate the endpoints:

```bash
for round in 1 2 3 4 5; do
  scripts/bench_cold_runpod.sh <reflex-endpoint-id> 1
  scripts/bench_cold_runpod.sh <llamacpp-endpoint-id> 1
done
```

That is n=5 cold invocations per engine. Raw `/usr/bin/time -v` logs land under
`bench-results/`.

### 4. Record

For each run:

- **Wall clock** of the cold request (from `bench_cold_runpod.sh`'s table).
- **Engine-ready time**, from the worker's container logs: container start to the
  engine's ready line. Reflex prints `REFLEX_STDIO_READY`; `llama-server` logs
  `llama_server: listening on http://127.0.0.1:8081` once the model is loaded and warmed
  up (its log lines carry their own elapsed-time prefix, e.g. `0.38.957`). The difference between
  wall clock and engine-ready time is Runpod's own provisioning (image pull, GPU
  scheduling, container start), which the earlier comparison measured at ~40 s.
- The **GPU** each worker actually landed on (worker logs or `list-endpoint-workers`).
  Discard and redo any run that didn't land on the pinned SKU.
- Both **image sizes**, once (`docker images`).

### 5. Optional variant: llama.cpp without warmup

`llama-server` runs a warmup decode by default, which costs time at startup. To see how
much of the gap is that warmup, redeploy the llama.cpp endpoint with
`LLAMA_ARGS="-ngl 99 -c 4096 -np 1 --no-warmup"` (an endpoint environment variable) and
repeat step 3 for it. Report it as a separate row, not a replacement for the default.

### 6. Tear down

Delete both endpoints and their templates when done (`deploy_runpod.sh --teardown`, or
the console). Workers scale to zero on their own, but the endpoints remain otherwise.

## Results

Run date: 2026-10-01, 17:31–17:51 UTC. GPU: **NVIDIA L4 (`sm_89`)**, pinned on both
endpoints; every worker landed on one. Data center: unpinned on both (the scheduler
chose EU-RO-1 or EUR-IS-1; noted per run). CUDA floor 12.8 on both, FlashBoot off,
`workersMin=0`, `workersMax=1`, 10 s idle timeout. Requests sent from a client on the US
west coast.

**Why an L4 and not the default A4500.** The run started on RTX A4500s in EU-RO-1. Within
about 15 minutes, serverless A4500 stock on CUDA ≥ 12.8 hosts went to zero (workers sat
in `THROTTLED`). Both endpoints were moved to the L4. It is `sm_89`, so the official
llama.cpp image still has native code for it (see the pre-check above), and the
engine-load comparison stays free of driver JIT. Serverless L4 stock was LOW throughout,
which is why the data center was left unpinned: a single pinned data center ran out
mid-run. L4 serverless capacity ran out completely at ~17:57 UTC, which ended the run.

| image | size |
|---|---|
| Reflex (`runpod-lb`) | 2.06 GB |
| llama.cpp (`server-cuda-b11277` + nginx + model) | 7.79 GB |

Engine-ready is measured from the container's first log line to the engine's ready line
(`REFLEX_STDIO_READY` for Reflex, `llama_server: listening` for llama.cpp), using the
worker log timestamps. For llama.cpp it agrees with llama-server's own elapsed-time
prefix to within 10 ms.

| round | Reflex wall clock | Reflex engine-ready | llama.cpp wall clock | llama.cpp engine-ready |
|---|---|---|---|---|
| 1 | 28.53 s (EUR-IS-1) | 0.55 s | 138.68 s (EU-RO-1, image pulled fresh) | 0.92 s |
| 2 | 56.11 s (EU-RO-1) | 0.43 s | 67.78 s (EU-RO-1) | 0.95 s |
| 3 | 53.12 s (EU-RO-1) | 0.39 s | failed: gateway HTTP 400 after 300 s | not captured |
| 4 | 73.16 s (EU-RO-1) | 0.61 s | 41.96 s (EU-RO-1) | 1.37 s |
| 5 | 24.06 s (EUR-IS-1) | 0.49 s | failed: gateway HTTP 400 after 300 s (image pulled fresh) | not captured |
| **median** | **53.12 s** (n=5) | **0.49 s** (n=5) | **67.78 s** (n=3) | **0.95 s** (n=3) |

Both llama.cpp failures came while Runpod's own worker-log API was timing out. In round
3 all three workers the platform started (see below) had passed `/ping` and sat `IDLE`
for ~4 minutes before the gateway gave up, so the request was never routed to a healthy
worker. A Reflex request the same day (on the A4500 endpoint, before the switch) failed
the same way. These are recorded as platform failures, not engine failures. A sixth
llama.cpp run was started as a replacement but got no worker before L4 stock ran out,
and was stopped.

### Second session (2026-10-02): the last two llama.cpp runs, and `--no-warmup`

Run 03:00–03:15 UTC, same GPU SKU (L4, pinned), same images, CUDA floor 12.8, FlashBoot
off, `workersMin=0`, `workersMax=1`, data center unpinned (every worker landed in
EU-RO-1). The idle timeout was raised to 60 s so worker logs could be read after each
request; every run was still forced cold the same way `bench_cold_runpod.sh` does it
(`workersMax` to 0, 25 s wait, back to 1). Wall clock was timed around a single `curl`
from the same client rather than through `bench_cold_runpod.sh`'s `/usr/bin/time` loop.
The `--no-warmup` variant ran as a third endpoint with
`LLAMA_ARGS="-ngl 99 -c 4096 -np 1 --no-warmup"`.

| run | engine | wall clock | engine-ready |
|---|---|---|---|
| 1 | Reflex (control) | 35.92 s | 0.44 s |
| 2 | llama.cpp | 28.12 s | 0.88 s |
| 3 | llama.cpp `--no-warmup` | 45.58 s | 0.98 s |
| 4 | Reflex (control) | 28.66 s | not captured (log API stalled) |
| 5 | llama.cpp | 60.77 s | 1.05 s |
| 6 | llama.cpp `--no-warmup` | 113.63 s (image pulled fresh) | 0.92 s |
| 7 | llama.cpp `--no-warmup` | 57.57 s | 0.86 s |
| 8 | Reflex (control) | 36.21 s | 0.40 s |

Reflex run 1 ran before the CUDA floor was raised from 12.0 to 12.8 on its endpoint; the
worker still landed on an L4 in EU-RO-1. `workersMax=1` was again exceeded: `--no-warmup`
runs 3, 6 and 7 each started a second worker (the run 6 extra one sat `THROTTLED`).

### Combined (both sessions)

| | n | engine-ready median (range) | wall clock median (range) |
|---|---|---|---|
| Reflex | 5 (2026-10-01) | **0.49 s** (0.39–0.61) | 53.12 s (24.06–73.16) |
| Reflex, same-day controls | 2 (2026-10-02) | 0.40 s, 0.44 s | 28.66–36.21 s (n=3) |
| llama.cpp | 5 (3 + 2) | **0.95 s** (0.88–1.37) | 60.77 s (28.12–138.68) |
| llama.cpp `--no-warmup` | 3 | **0.92 s** (0.86–0.98) | 57.57 s (45.58–113.63) |

### What the numbers support

- **Engine load: Reflex ~0.5 s vs. llama.cpp ~0.95 s on an L4, about 1.9x** (medians,
  n=5 each; Reflex 0.39–0.61 s, llama.cpp 0.88–1.37 s). This is the like-for-like
  engine comparison. It is consistent with the local T4 numbers in the pre-check above,
  and with the second session's same-day Reflex controls (0.40 s, 0.44 s).
- **llama-server's warmup is not where its time goes.** With `--no-warmup` its engine-ready
  median was 0.92 s (n=3) against 0.95 s with warmup, a difference inside the run-to-run
  spread. Its log shows most of the time in loading the model (~0.45 s from
  `load_model` to the tokenizer warning) and thread-pool and slot setup after it.
- **End-to-end wall clock: no conclusion.** Platform overhead dominates: scheduling a
  worker, pulling or loading the image, creating the container, and the gateway noticing
  the worker is healthy. Even with a cached image, Reflex ranged 24–73 s for a ~0.5 s
  engine load. The time from engine-ready to the response arriving alone ranged from
  roughly 7 s to 58 s across runs (approximate: the request's start is inferred from
  the cold-reset step's fixed sleep). Neither the n=5 medians (53 s vs. 61 s) nor the
  ranges support an end-to-end ratio; the second session alone shows llama.cpp's fastest
  cold request (28.12 s) beating two of three Reflex controls.
- **Image size showed up directly.** Two of the five llama.cpp workers landed on hosts
  without the 7.79 GB image and pulled it (~55 s in round 1). No Reflex worker in the
  measured runs needed a full pull: either the host had the image or Runpod loaded it
  from its own image cache (7–14 s).

### Platform behaviour seen during the run

- `workersMax=1` was not enforced: Reflex rounds 2 and 3 each started a second worker,
  and llama.cpp round 3 started three. This is the same overshoot noted in
  `serverless/runpod/README.md`.
- The cold-reset step (`workersMax` to 0, wait, back to 1) works, but a stray benchmark
  loop that keeps doing it starves every other run on that endpoint. Make sure only one
  `bench_cold_runpod.sh` is running per endpoint.

## Caveats to state with the result

- **Platform provisioning dominates.** On the earlier runs, Runpod's own provisioning
  took ~40 s of a ~42 s cold invocation. If both engines load in a few seconds, the
  wall-clock ratio will be close to 1 and the engine-ready column is where any
  difference shows. Report both; don't present the engine-ready gap as the end-to-end
  one.
- **Image size is part of the cold start.** A larger image takes longer to pull on a
  fresh host. That is a real, fair cost of each engine as shipped, but say how much of a
  wall-clock difference the size difference could explain.
- **The engines do different work at load.** Reflex dequantizes every weight to `f32`
  on the GPU at load; llama.cpp runs its quantized kernels directly. That is a design
  difference being measured, not a setup unfairness, but it is why a result in either
  direction needs the engine-ready breakdown to interpret.
- **The CUDA floor narrows the host pool** for both endpoints equally, but it means the
  numbers describe CUDA 12.8+ hosts only.
- **The result holds for GPUs the llama.cpp image has native code for** (`sm_86`,
  `sm_89`, `sm_120a`). On other GPUs llama.cpp's official image JIT-compiles its kernels
  on every fresh container (see the pre-check above), so its cold start is much slower
  there; Reflex's default multi-arch build is native on `sm_75` through `sm_90`.
