# Reflex on Modal (Phase 1)

Phase 1 of the Modal exploration plan (see `docs/modal-cold-start-phase0.md` for Phase
0's go/no-go gate): package the existing, unmodified `reflex` + `reflex-openai-adapter`
binaries as a real Modal deployment, mirroring the `.runpod/` / `serverless/runpod/`
pattern used on that platform. **Status: built and real-hardware-verified on a Modal L4
GPU, 2026-09-27.**

## What this is

- **`Dockerfile`**: same builder recipe as the root `Dockerfile`'s `adapter` target (compile
  `reflex --features ipc` and `sidecar/openai-adapter` in one `nvidia/cuda:12.4.1-devel`
  stage, copy both binaries plus the baked-in model into a `nvidia/cuda:12.4.1-runtime`
  stage). The only real difference: `REFLEX_CUDA_ARCH` defaults to `sm_89` (Ada
  Lovelace), not `sm_86` -- see the Dockerfile's own comment for why Modal's exact,
  non-pooled GPU selection makes a pinned cubin safe here in a way it isn't for
  Runpod's mixed-architecture serverless pools.
- **`app.py`**: a `modal.App` that builds `image` from that Dockerfile
  (`Image.from_dockerfile`, `add_python="3.12"` since the runtime stage has no Python of
  its own -- Modal's container init needs an interpreter to run `@modal.enter()`/
  `@modal.exit()`), then runs the adapter unmodified as a Modal Server (`@app.server`,
  `gpu="L4"`, `unauthenticated=True` for this exploratory check) via
  `subprocess.Popen`. No new engine code, no protocol reimplementation -- same
  "packaging only" rule every other deployment directory in this project follows.
- Model: the same `serverless/runpod/model.gguf` (Qwen3-0.6B-Q4_K_M, ~379MB) baked into
  every other deployment here, for the same reason -- a runtime download would add
  HuggingFace-fetch latency directly into the cold-start number this exists to measure.

## Running it

From the repo root (build context requirement, same as every other Dockerfile in this
project):

```
python -m modal run .modal/app.py     # one-shot: ephemeral App, real cold-start measurement, then tears down
python -m modal deploy .modal/app.py  # persistent deployment
```

**Windows gotcha**: Modal's CLI output uses Unicode characters (`✓`) that crash on a
non-UTF-8 console codepage (`UnicodeEncodeError: 'charmap' codec can't encode character
'✓'`). Set `PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`) in the environment before
invoking `python -m modal` from PowerShell/cmd -- not a Modal bug, just Windows' default
console encoding.

## Real measured numbers (2026-09-27, real Modal L4 GPU, `modal run` ephemeral App)

Same methodology as Phase 0: an ephemeral App's first (and only) invocation guarantees
no warm-container reuse is possible, so this is a genuine cold start, measured
client-side from the local machine that issued `modal run` (directly comparable to
Phase 0's own client-side "end-to-end wall clock" number):

| Phase | Time |
|---|---|
| Local submit → `/healthz` returns 200 | **6.0s** |
| First `/v1/chat/completions` response (post-healthy) | **1.13s** |
| **Total: local submit → first token** | **7.1s** |

Container-internal log confirms the GPU landed as expected (`GPU: NVIDIA L4 (sm_89),
VRAM: 22369/22563 MiB free`) and the pinned `sm_89` cubin loaded and ran without the
compute-capability mismatch panic seen on Runpod's mixed pools (see
`.runpod/README.md`) -- expected, since Modal's `gpu="L4"` request has no equivalent
pooling hazard, but confirmed rather than assumed.

**What this was not yet, at the time**: a formal Reflex-vs-vLLM head-to-head — that
comparison has since run as Phase 3, see
[`docs/modal-phase3-comparison.md`](../docs/modal-phase3-comparison.md) (`n=3` cold starts
per engine under a shared harness: Reflex median 7.7s vs. vLLM median 190.7s, ~25x). This
single Phase 1 run (7.1s) is superseded as the citable number by Phase 3's `n=3` sample;
it stays here as the original packaging-verification result.

## Cost

One `modal run` invocation: ~155s of CPU-only image build (Rust/CUDA compile, cached on
Modal's build layer for subsequent runs, not billed as GPU time -- same exclusion
rationale as Phase 0's doc) plus well under 15s of actual L4 GPU container time (cold
start to teardown). At Modal's public L4 rate (~$0.80/hr per Phase 0's doc), the
GPU-billed portion of this run was on the order of a few thousandths of a dollar --
less than Phase 0's own ~$0.05 vLLM check, consistent with Reflex's cold start being
roughly two orders of magnitude shorter.

## Non-goals for this addition

- No changes to `serverless/runpod/`, `.runpod/`, `sidecar/openai-adapter/`, or any core
  engine file. This is a fourth, independent, purely additive deployment path.
- No new engine capability, no batching, no protocol redesign -- see [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md)'s
  non-goals.
- The Phase 3 head-to-head comparison lives in
  [`docs/modal-phase3-comparison.md`](../docs/modal-phase3-comparison.md), not here.

## What's confirmed vs. not yet verified

**Confirmed, real hardware**: the image builds correctly via `Image.from_dockerfile`
against this project's existing multi-stage Dockerfile pattern; `add_python` is required
because the runtime stage has no Python; the `sm_89`-pinned cubin runs correctly on
Modal's L4; the adapter's existing three-state `/healthz` handler works unmodified as
the Server's readiness signal; `n=3` real end-to-end cold starts measured (7.0s, 7.7s,
7.7s local submit → first token — see Phase 3's doc for the full comparison against vLLM).

**Not yet verified**: a persistent (`modal deploy`) deployment's behavior under repeated
cold starts after `scaledown_window` idle periods; behavior under
`unauthenticated=False` (real auth would be needed for anything beyond this exploratory
check).
