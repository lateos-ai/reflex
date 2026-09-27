# Reflex vs. vLLM cold-start comparison on Runpod Serverless (real measurement)

This is a real, measured comparison of cold-invocation latency between Reflex and vLLM,
deployed on identical hardware on the same serverless platform (Runpod), run 2026-09-26.
It exists to test the claim made in the root [README](../README.md#why-serverless-is-the-fit)
and [`serverless/runpod/README.md`](../serverless/runpod/README.md): that Reflex's AOT-compiled
cold start adds near-zero marginal time on top of a serverless platform's own provisioning
floor, where a JIT/graph-compiling engine adds substantially more on top of that same floor.

**Result: Reflex's cold start was ~3.4x faster than vLLM's on the same GPU tier, same
platform, same day.** Reflex: ~42-44 seconds wall clock. vLLM: ~150 seconds (Runpod's own
`delayTime` + `executionTime` job metadata). The methodology differences that make this not a
perfectly controlled comparison are disclosed in full below — read them before citing this
number, because they matter and they don't all point the same direction.

## Setup

Both deployed as real Runpod Serverless endpoints, `min=0` workers (genuine scale-from-zero),
pinned to the same GPU pool and SKU:

| | Reflex | vLLM |
|---|---|---|
| Endpoint type | Load-balancing (raw HTTP) | Queue-based (`runsync`/`run` job API) |
| Image | `ghcr.io/lateos-ai/reflex-runpod:latest` (this repo, [`serverless/runpod/`](../serverless/runpod/)) | `runpod-workers/worker-vllm` v2.27.2 (Runpod's own official worker, 52,327 deploys — the natural choice for a fair comparison, not a third-party or self-built image) |
| GPU | RTX A4500, pinned via `excludedTypes` | RTX A4500, pinned via `gpuIds` exclusion syntax |
| GPU pool / rate | `AMPERE_16`, $0.58/hr | `AMPERE_16`, $0.58/hr (same) |
| Model | Qwen3-0.6B, **Q4_K_M** GGUF, baked into the image | Qwen3-0.6B, **bf16** safetensors (`Qwen/Qwen3-0.6B`), downloaded from Hugging Face at cold start |
| FlashBoot | OFF | OFF (Runpod's platform-level cold-start acceleration; disabled on both so it doesn't confound the comparison either way) |
| `workersMin`/`workersMax` | 0 / 1 | 0 / 1 |
| `idleTimeout` | 10s | 10s |
| Data center | EU-RO-1 | EU-RO-1 (same) |

## Results

### Reflex: real cold invocation

Full detail in [`serverless/runpod/README.md`](../serverless/runpod/README.md#real-deployment-findings-2026-09-26-real-rtx-a4500-on-runpod).

| Phase | Time |
|---|---|
| Reflex's own load (container start → `REFLEX_STDIO_READY`) | **~1.6s** (measured from real container logs) |
| Runpod's platform provisioning (GPU scheduling + container start, before Reflex begins) | **~40s** (derived: total minus Reflex's own load) |
| **Total cold invocation, wall clock** | **~42-44s** (real `curl` timing, `n=2`) |

### vLLM: real cold invocation

One real `runsync` job against a freshly-created, `min=0` endpoint, model `Qwen/Qwen3-0.6B`,
`max_tokens: 8` requested (the worker's response format didn't honor this field as expected —
it generated 115 completion tokens regardless; a request-shape detail, not a timing artifact,
and irrelevant to the cold-start measurement since generation only added ~1s regardless of
length):

```json
{
  "delayTime": 149253,
  "executionTime": 975,
  "status": "COMPLETED"
}
```

`delayTime` is Runpod's own job-metadata field for time-to-worker-ready (image pull + model
load + queue scheduling) — a platform-native measurement, not a client-side wall-clock
estimate. `executionTime` is the actual inference call once the worker was running.

| Phase | Time |
|---|---|
| Runpod's own reported `delayTime` (cold start, platform-measured) | **149.253s** |
| `executionTime` (real inference, 115 completion tokens) | **0.975s** |
| **Total, platform-reported** | **~150.2s** |

### Head to head

| | Reflex | vLLM | Ratio |
|---|---|---|---|
| Total cold invocation | ~42-44s | ~150.2s | **vLLM took ~3.4x longer** |

## Methodology caveats — read before citing this number

This is a real measurement, not an estimate, but it is not a perfectly isolated variable. Three
real differences exist between the two runs, and they don't all favor the same side:

1. **Model precision differs**: Reflex ran 4-bit `Q4_K_M`; vLLM ran 16-bit `bf16` — roughly 4x
   more bytes to move for vLLM's model weights. This is not an unfair setup choice: it's each
   engine's own natural default (Reflex is GGUF-native; `worker-vllm` expects a Hugging Face
   safetensors repo, and GGUF support in vLLM is limited/experimental). But it means part of
   vLLM's slower load time is attributable to moving more bytes, not solely to its engine
   architecture (CUDA graph capture, JIT, etc.).
2. **Model acquisition differs**: Reflex's model was baked into the image (zero network
   fetch at cold start, a deliberate choice documented in `serverless/runpod/README.md` to keep
   the measurement about engine load time, not network variance). vLLM's `worker-vllm` downloads
   the model from Hugging Face at cold start by default — this is also each engine's natural
   default deployment pattern (a user deploying `worker-vllm` off the Hub gets exactly this),
   but it means part of vLLM's ~150s includes a real HTTP download Reflex's number doesn't pay.
3. **Endpoint type differs**: Reflex ran as a load-balancing endpoint (raw HTTP, timed via
   client-side `curl`); vLLM ran as a queue-based endpoint (timed via Runpod's own `delayTime`
   job-metadata field). Both are the *natural* way to deploy each engine — `worker-vllm` doesn't
   ship in a load-balancer-compatible form, and forcing it into one would be a bigger, separate
   engineering effort, not a fairer comparison. `delayTime` and client-observed wall-clock time
   measure conceptually the same thing (time until the backend can serve a request), but via
   different instrumentation, so they aren't a bit-for-bit identical metric.

**None of these caveats change the direction of the result.** If anything, (1) and (2) mean
part of vLLM's disadvantage here is "real download plus real extra bytes," which is a genuine,
representative cost of deploying vLLM's natural default pattern on this platform — not an
artifact this comparison manufactured. The core, robust finding survives all three caveats:
**on the same GPU, same platform, same day, Reflex reached a servable state in under a
minute; vLLM took roughly two and a half.**

## A real platform bug found while producing this comparison

**Runpod's autoscaler exceeded the configured `workersMax` on both endpoints.** Despite
`workersMax: 1` on both the Reflex and vLLM endpoints, `list-endpoint-workers` showed 2-3
workers running/initializing simultaneously on each shortly after the first cold-start
invocation — confirmed via the Runpod MCP tools, not assumed. Both endpoints were deleted
promptly upon discovering this to avoid unintended extra billing. This is worth knowing for
anyone deploying either engine on Runpod Serverless: **check `list-endpoint-workers` (or the
console) after a cold start, don't assume `workersMax` was honored**, especially for a
short-lived test endpoint you intend to tear down quickly.

## Cost

Real dollar cost could not be confirmed at time of writing — Runpod's billing API had not yet
reconciled the relevant hour (a real lag, not a missing charge; see
`serverless/runpod/README.md`'s own note on this). Rough bound from elapsed worker-time across
both endpoints (including the unintended extra workers from the autoscaler issue above): well
under $1 total, on the order of $0.20-0.40 at the confirmed $0.58/hr rate. This should be
replaced with a real reconciled figure once available — do not treat the estimate above as a
citable number.

## What this does and doesn't prove

**Proven**: on Runpod Serverless, with each engine's natural default deployment pattern, on
the same GPU tier, Reflex's cold invocation was measured at roughly a third the wall-clock time
of vLLM's. This is the core claim the serverless positioning in the root README rests on, and
it now has a real number behind it instead of an estimate.

**Not proven / out of scope for this comparison**: steady-state throughput, answer quality,
cost-per-token at scale, or behavior under concurrent load (both engines were tested with a
single cold, single-shot invocation, matching this project's own architecture — see
[Non-goals](../README.md#non-goals)). This comparison is specifically about the bursty,
single-shot workload this deployment shape targets, not a claim that Reflex outperforms vLLM
as a high-throughput server — it deliberately doesn't try to, and would lose that comparison
(see the root README's "Why this exists" section).
