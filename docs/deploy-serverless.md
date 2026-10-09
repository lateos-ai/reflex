# Deploying Reflex on serverless GPU

Start here if you want to run Reflex at `workersMin=0` (scale-to-zero) on a
serverless GPU platform. Reflex is the engine you put behind a single cold
decision — one classification or routing call that arrives with no warm worker.
Steady chat traffic wants vLLM; see the [Non-goals](../README.md#non-goals).

This page covers Runpod first (the measured path) and points at the AWS
scale-to-zero guide. It does not replace the long-form docs; it tells you which
one to open.

| I want to... | Go to |
|---|---|
| A real HTTP endpoint I control, cold from zero | [Path 1 — Runpod load-balancing](#path-1--runpod-load-balancing-recommended) |
| To find Reflex in Runpod's catalog and one-click it | [Path 2 — Runpod Hub](#path-2--runpod-hub-discovery-only) |
| Scale to zero on AWS Spot | [AWS scale-to-zero](#aws-scale-to-zero) |
| To run it on Modal | [Modal](#modal) |

The engine is `batch_size` 1 and has no scheduler or HTTP server of its own. The
load-balancing path runs the [`reflex-openai-adapter`](../sidecar/openai-adapter/README.md)
sidecar in front of the engine over its local IPC, unmodified.

## Path 1 — Runpod load-balancing (recommended)

A Runpod Serverless **Load-Balancing** endpoint runs the sidecar's HTTP server
directly, with no Python glue. The whole path is scripted in
[`scripts/deploy_runpod.sh`](../scripts/deploy_runpod.sh) and documented by hand
in [`serverless/runpod/README.md`](../serverless/runpod/README.md).

**Image:** `ghcr.io/lateos-ai/reflex-runpod`. **Pin a digest, not `:latest`.**
`:latest` is a moving alias — every rebuild from `master` repoints it. Verified
2026-10-09:

```
ghcr.io/lateos-ai/reflex-runpod@sha256:219edd36bd6b9bf8271d4ae91793eac205c625794e55ed325570318ce31af9b7
```

The image bakes in `Qwen3-0.6B` (`Q4_K_M`) at `/models/model.gguf` (`GGUF_PATH`),
so there is no download on the cold path.

**Scripted:**

```
RUNPOD_API_KEY=... scripts/deploy_runpod.sh --no-bench
```

`--no-bench` builds/pushes the image, creates the load-balancing template and
endpoint, and configures it without running the cold-start benchmark. Run it with
`--help` for the full flag and environment reference, and `--teardown` to delete
the endpoint and template when finished. This creates billable Runpod resources.

**Or configure the endpoint by hand.** Key fields (details in
[`serverless/runpod/README.md`](../serverless/runpod/README.md)):

- **Endpoint type:** Load Balancing.
- **Container image:** the pinned digest above (or `:latest` if you accept the
  move, or your own `docker build --target runpod-lb .`).
- **Min workers:** `0` — genuine scale-from-zero.
- **Exposed port:** matches the image's `PORT` (default `80`).
- **Health check:** do **not** rely on `HEALTH_CHECK_PATH`; Runpod's gateway polls
  the hardcoded `/ping` path. The sidecar answers `/ping` and `/healthz`
  identically (`204` while loading, `200` ready, `503` dead).
- **GPU:** the default multi-arch fatbin image runs on every card in `AMPERE_16`
  (Ampere `sm_86` and Ada `sm_89`). A single-arch build must be pinned to a
  matching SKU; see the README's GPU-selection section.

**Then call it.** The endpoint serves the sidecar's own routes at
`https://<endpoint-id>.api.runpod.ai`, with the Runpod API key as a bearer token.
The job this engine exists for is a single cold decision — `POST /v1/classify`
scores each label as a continuation of the prompt in one engine pass and
generates nothing:

```
curl "https://<endpoint-id>.api.runpod.ai/v1/classify" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Review: The battery died after two days. Sentiment:",
    "labels": [" positive", " negative"]
  }'
```

Real response (Tesla T4, `Qwen3-0.6B-Q4_K_M`; ~10 ms per request once warm):

```json
{
  "id": "classify-reflex-0", "object": "classification", "created": 1791412558,
  "model": "Qwen3-0.6B-Q4_K_M",
  "label": " negative", "label_index": 1,
  "labels": [
    {"label": " positive", "probability": 0.08692727, "score": 15.241831, "tokens": 1},
    {"label": " negative", "probability": 0.9130727, "score": 17.593575, "tokens": 1}
  ],
  "entropy": 0.42612946
}
```

The probabilities are identical to `reflex system1` on the same prompt and
labels. `probability` is relative to this label set only. Use labels of equal
token length (ideally one token each) and read the request fields and the
"Choosing labels and prompts" notes in the
[`sidecar/openai-adapter` README](../sidecar/openai-adapter/README.md#post-v1classify)
before you pick labels.

## Path 2 — Runpod Hub (discovery only)

The [Hub listing](https://console.runpod.io/hub/lateos-ai/reflex) is a
**queue-based worker**: Runpod's Hub publishing pipeline does not accept
load-balancing endpoints, so this second build wraps the same sidecar with a
small Python handler. It is the path for discovery and one-click deploy, not the
leanest path — if you are deploying manually, use Path 1.

The job `input` is an OpenAI chat-completions body, or — with `labels` — the same
single-pass classification as above:

```json
{"input": {"prompt": "Review: The battery died after two days. Sentiment:",
           "labels": [" positive", " negative"]}}
```

As of hub release `v0.2.3-runpod-hub`, a non-streaming job's `output` is a
**one-element list** holding that `classification` object (the `worker-vllm`
convention), not the bare object. Job input/output, streaming, and the
`ADAPTER_ARGS` deploy field are in [`.runpod/README.md`](../.runpod/README.md).

The Hub builds the queue worker from its own image
(`ghcr.io/lateos-ai/reflex-runpod-hub`). Verified 2026-10-09:

```
ghcr.io/lateos-ai/reflex-runpod-hub@sha256:6ecb8e1f0200c2736141014eadc9ad38bee7560fad52b48e02dbacd31d510f33
```

The one-click listing tracks the Hub release, not this digest; the digest is for
a manual deployment of the same image.

## AWS scale-to-zero

For Spot GPU instances behind an Auto Scaling Group with minimum capacity 0 (one
`reflex uds` process per instance, a local Unix Domain Socket instead of a network
load balancer), see [`docs/aws-deployment.md`](aws-deployment.md). This page does
not duplicate that guide. The always-warm alternative (On-Demand, public HTTPS
load balancer, the HTTP sidecar) is [`docs/aws-deployment-warm.md`](aws-deployment-warm.md).

## Modal

[`.modal/`](../.modal/README.md) deploys the same sidecar as a Modal Server
(`python -m modal run .modal/app.py`), on the slim runtime image with a pinned `sm_89`
cubin — Modal's `gpu="L4"` requests one exact SKU, so the multi-arch fatbin isn't needed.
Measured 2026-10-09 on a real Modal L4, `n=3`: **8.1s median** local submit → first token
(6.5 / 8.8 / 8.1s). Details and caveats in [`.modal/README.md`](../.modal/README.md).

## What to expect (measured numbers)

These are the only figures to quote, each from its source. Do not quote an
engine-benchmark row here — this audience is billed for platform wall clock.

- On Runpod serverless, a cold Reflex invocation was **~42–44 s** wall clock
  versus **~150 s** for Runpod's official vLLM worker, same GPU tier, same day.
  Most of both numbers is the platform provisioning a GPU. vLLM also downloaded
  bf16 weights; Reflex served a `Q4_K_M` GGUF baked into the image. Endpoint types
  differed (load-balancing vs queue).
  [source](serverless-cost-comparison.md)
- Reflex's own engine load on that platform was **~1.6 s** of the ~42–44 s. The
  rest no engine avoids.
  [source](../README.md#why-cold-start)
- Against an official llama.cpp server image on the same platform, engine load
  was **~0.5 s vs ~0.95 s** on an L4. End-to-end, platform variance (24–73 s) was
  larger than that gap, so that comparison is **inconclusive**.
  [source](runpod-llamacpp-comparison.md)
- The job this is for is a single cold decision (`POST /v1/classify`), not a warm
  chat pool. `batch_size` is always 1. Steady traffic wants vLLM.
  [source](../README.md#non-goals)

## Image releases

Both images are rebuilt from `master`; a rebuild repoints `:latest` and moves the
digest. When a load-balancing or Hub image is pushed, its release note records
three things: the image digest, the fatbin arch list it was built with
(`sm_75,sm_80,sm_86,sm_89,sm_90` by default), and whether the cold-start numbers
changed. Do not publish a numbered GHCR tag that was never pushed — the digest and
the Hub release name are the identifiers.

## Long form

- [`serverless/runpod/README.md`](../serverless/runpod/README.md) — load-balancing
  deployment, GPU selection, pricing, real deployment findings.
- [`.runpod/README.md`](../.runpod/README.md) — the Hub queue worker and its job
  input/output contract.
- [`.modal/README.md`](../.modal/README.md) — the Modal Server deployment and its
  measured cold starts.
- [`docs/aws-deployment.md`](aws-deployment.md) — scale-to-zero on AWS.
- [`docs/serverless-cost-comparison.md`](serverless-cost-comparison.md) and
  [`docs/runpod-llamacpp-comparison.md`](runpod-llamacpp-comparison.md) — the
  methodology and caveats behind the numbers above.
- [`docs/reference.md`](reference.md#docker) — the Docker build and the sidecar
  routes.