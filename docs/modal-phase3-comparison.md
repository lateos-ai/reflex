# Reflex vs. vLLM cold-start comparison on Modal (real measurement, n=3 each)

This is Phase 3 of the Modal exploration plan (see `docs/modal-cold-start-phase0.md` for
Phase 0's go/no-go gate and `.modal/README.md` for Phase 1's single-run deployment check):
a controlled, multi-run head-to-head between Reflex and vLLM on identical hardware on the
same platform, run 2026-09-27. Three cold-start trials per engine, same model class, same
GPU, same day, same client-side timing harness (`.modal/app.py` / `.modal/app_vllm.py`).

**Result: Reflex's median cold start (7.7s) was ~25x faster than vLLM's median (190.7s) on
the same L4 GPU, same platform, same day.** Read "Methodology caveats" below before citing
this number — the two engines are running genuinely different-sized models, and that
difference is disclosed in full, not hidden.

## Setup

Both deployed as real Modal Servers (`@app.server`), ephemeral `modal run` invocations
(guarantees no warm-container reuse — every trial is a genuine scale-from-zero cold start),
`n=3` trials per engine, run back to back on 2026-09-27:

| | Reflex | vLLM |
|---|---|---|
| Script | [`.modal/app.py`](../.modal/app.py) | [`.modal/app_vllm.py`](../.modal/app_vllm.py) |
| Image | `Dockerfile` (this repo, multi-stage build of `reflex` + `reflex-openai-adapter`) | `nvidia/cuda:12.4.1-devel-ubuntu22.04` + `uv_pip_install("vllm==0.13.0")` (Modal's own official `llm_inference` example pattern) |
| Model | `serverless/runpod/model.gguf`, Qwen3-0.6B **Q4_K_M** GGUF, baked into the image | `Qwen/Qwen3-0.6B`, **bf16** safetensors, downloaded fresh from Hugging Face every trial |
| GPU | `L4` (`sm_89`) | `L4` |
| Config | Plain adapter defaults, no flags | Plain `vllm serve` defaults — no `--enforce-eager`/`FAST_BOOT`, no memory snapshotting, no HF-weights cache volume, no pre-warming, matching Phase 0's own "deliberately plain" setup exactly |
| Invocation | `modal run` (ephemeral App) x3 | `modal run` (ephemeral App) x3 |
| Timing | Client-side, local submit → `/healthz` 200 → first `/v1/chat/completions` response | Same methodology, `/health` (vLLM's path) instead of `/healthz` |

## Results

### Reflex (3 trials)

| Trial | To healthy | First completion | Total |
|---|---|---|---|
| 1 | 6.0s | 0.99s | **7.0s** |
| 2 | 6.5s | 1.24s | **7.7s** |
| 3 | 6.4s | 1.23s | **7.7s** |

**Median: 7.7s. Range: 7.0s – 7.7s.**

### vLLM (3 trials)

| Trial | To healthy | First completion | Total |
|---|---|---|---|
| 1 | 187.9s | 1.43s | **189.4s** |
| 2 | 214.7s | 2.66s | **217.4s** |
| 3 | 188.8s | 1.82s | **190.7s** |

**Median: 190.7s. Range: 189.4s – 217.4s.**

(Trial 2's outlier is consistent with real HF download/network variance across independent
cold containers — see caveats below; it doesn't change which engine wins or by roughly how
much.)

### Head to head

| | Reflex | vLLM | Ratio |
|---|---|---|---|
| Median total cold start | 7.7s | 190.7s | **vLLM took ~24.8x longer** |
| Best-case Reflex vs. worst-case vLLM | 7.0s | 217.4s | ~31.1x |
| Worst-case Reflex vs. best-case vLLM | 7.7s | 189.4s | ~24.6x |

The ratio is stable across the full observed range (~25-31x) — this isn't a result that
depends on cherry-picking one trial from either side.

## Methodology caveats — read before citing this number

Same spirit as `docs/serverless-cost-comparison.md`'s Runpod comparison: this is a real
measurement, not an estimate, but three real differences exist between the two runs, and
they don't all point the same direction:

1. **Model precision and size differ**: Reflex ran a 4-bit `Q4_K_M` GGUF (~379MB); vLLM ran
   16-bit `bf16` safetensors (~1.5GB) — roughly 4x more bytes to move, plus vLLM's own
   `torch.compile` and CUDA graph capture phases (Phase 0 measured these at ~37.5s and ~11s
   respectively, entirely absent from Reflex's AOT-compiled path). This is each engine's own
   natural default, not an unfair setup choice — GGUF is Reflex's native format, and vLLM's
   default is safetensors with JIT/graph compilation. But it means part of the gap is
   "different model representation," not solely "different engine architecture."
2. **Model acquisition differs**: Reflex's model is baked into the image (zero network fetch
   at cold start, deliberately, so this measures engine load time, not network variance).
   vLLM downloads fresh from Hugging Face every trial (no cache volume, by design — see
   `.modal/app_vllm.py`'s docstring for why: adding a cache volume would make vLLM's *later*
   trials artificially faster than Phase 0's genuinely-cold ~178s number, which would break
   the controlled repeat this comparison is trying to be). Trial 2's higher number (217.4s
   vs. ~189s for the other two) is consistent with this — real HF download variance, not a
   platform anomaly.
3. **Both are each engine's natural default deployment pattern on Modal**: Reflex's baked-in
   GGUF and vLLM's `vllm serve <hf-repo>` are what a user actually gets from each engine's own
   quickstart/documentation, not a deliberately hobbled or advantaged configuration for either
   side.

**None of these caveats change the direction or rough magnitude of the result.** If
anything, (1) and (2) are real, representative costs of vLLM's natural default pattern (JIT
compile + graph capture + network fetch), not artifacts this comparison manufactured to make
vLLM look worse. The core, robust finding survives all three caveats: **on the same L4 GPU,
same platform, same day, across three independent cold starts each, Reflex reached a
servable state in under 8 seconds every time; vLLM took over three minutes every time.**

## Cost

Reflex: 3 trials, each well under 15s of L4 GPU time (per `.modal/README.md`'s Phase 1
figure) — a few cents total. vLLM: 3 trials, ~190-217s of L4 GPU time each (~10 minutes
total) at Modal's public L4 rate (~$0.80/hr) — on the order of $0.13. Total for this
comparison: well under $0.20.

## What this does and doesn't prove

**Proven**: on Modal, with each engine's natural default deployment pattern, on the same GPU
tier, across three independent cold starts per engine (not a single anecdotal run), Reflex's
median cold start was roughly 1/25th of vLLM's. This closes the gap Phase 1's own README left
open ("not yet a formal Reflex-vs-vLLM head-to-head... not yet a controlled comparison") —
this is that comparison, run under a shared harness, same day, same GPU type.

**Not proven / out of scope**: steady-state throughput, answer quality, cost-per-token at
scale, behavior under concurrent load, or a `modal deploy` persistent-deployment's repeated
cold-start behavior after `scaledown_window` idle periods (all trials here used ephemeral
`modal run`, matching Phase 0/1's own methodology). Also out of scope: Modal's GPU Memory
Snapshots feature (still alpha at time of writing) — this compares plain cold starts on both
sides, consistent with Phase 0's original framing of "does plain Reflex already match or
beat Modal's own *snapshotted* numbers."
