# Modal cold-start Phase 0 check (real measurement, go/no-go gate)

This is Phase 0 of the Modal exploration plan (build a `.modal/`-equivalent deployment for
Reflex, mirroring `.runpod/`): before investing in any packaging work, do a minimal real check
of Modal's *plain* (non-snapshotted) cold-start time for a small model, to confirm the
Reflex-vs-Modal comparison is actually interesting. Run 2026-09-27, real GPU, real spend
(~$0.05, one ~3-minute L4 container).

**Result: go.** Modal's plain cold start for a 0.6B model was ~178 seconds (~3 minutes) — not
sub-second, not even close. The gate this check was designed to fail on ("if Modal's plain cold
start is already sub-second or clearly beats anything Reflex could show, stop") did not fire;
the result is the opposite extreme. Phase 1 (building the actual Reflex-on-Modal deployment) is
worth pursuing.

## Why Modal specifically

Of the alternatives considered (Replicate, Baseten, fal.ai, Beam.cloud, AWS SageMaker Serverless
Inference, Google Cloud Run GPU), Modal stood out because it markets cold-start optimization
directly as its core pitch, and publishes its own before/after numbers for GPU Memory Snapshots
(alpha) — reported ~10x faster cold start via snapshotting (~118s → ~12s for a small model in
Modal's own Mistral 3 writeup). That makes "does plain Reflex on Modal already match or beat
Modal's own *snapshotted* numbers, without using any of that machinery" a sharper, more
differentiated comparison than a generic platform benchmark.

## Setup

- **Model**: `Qwen/Qwen3-0.6B` — chosen to match the size class Reflex targets (this project's
  own benchmarks and test fixtures center on Qwen3-0.6B).
- **Serving stack**: stock vLLM (`vllm==0.13.0`) OpenAI-compatible server, Modal's own official
  `llm_inference` example pattern (`Image.from_registry(...).entrypoint([]).uv_pip_install(...)`,
  `subprocess.Popen(["vllm", "serve", ...])` inside the function body).
- **GPU**: `gpu="L4"` (one GPU).
- **Deliberately plain**: no `--enforce-eager`/`FAST_BOOT` shortcut, no memory snapshotting, no
  pre-warming, no `min_containers` — vLLM's default config, which is what a user gets from
  Modal's own quickstart examples without opting into any cold-start mitigation.
- **Invocation**: `modal run` (ephemeral App), which guarantees a genuinely cold container — no
  warm-container reuse possible on a freshly created ephemeral App's first call.
- Full run: https://modal.com/apps/lowmls/main/ap-c0GAs0WnU8zADRKBS119Gt

## Results

Internal timing, measured inside the container function itself:

| Phase | Time |
|---|---|
| Model resolve + weight download + load | ~37s |
| `torch.compile` | ~37.5s |
| CUDA graph capture (mixed prefill-decode + decode-only) | ~11s |
| Remaining engine init/startup overhead | remainder |
| Server ready → first token | 1.2s |
| **Total: container function entry → first token** | **~178s** |

Client-side, from the local machine that issued `modal run`:

| Metric | Time |
|---|---|
| End-to-end wall clock (local submit → result received) | **183.6s** |

(Image build — pulling the CUDA base image and `uv pip install`-ing vLLM/PyTorch/etc, ~150s —
happened once, before this timed invocation, and is correctly excluded: it's a one-time cost,
not part of any single cold start a repeat caller would pay, same as it would be excluded from
a real user's repeated-cold-start experience.)

## What this does and doesn't prove

**Proven**: Modal's own default, most-commonly-followed deployment pattern (`vllm serve` with no
special flags) pays a JIT/graph-compilation tax of well over two minutes for a *0.6B* model on
an L4 — a GPU/model combination nowhere near Modal's stated snapshot use case of large models
with expensive compiles. Reflex's own measured cold-start numbers for comparable model sizes
(see `README.md`'s cold-start benchmarks and `HISTORY.md`) are in the low single digits of
seconds — a gap of roughly two orders of magnitude, without Reflex doing anything special to win
this comparison (it just never has a JIT/compile phase to pay for). This is a much larger gap
than the plan's original hypothesis anticipated (which was framed around matching Modal's
*snapshotted* ~12s number, not its ~178s *plain* number).

**Not proven yet**: this is one run of one model on one GPU type, using vLLM (not a hand-rolled
minimal-`transformers` init, which might shave some time but wouldn't eliminate vLLM's default
compile/graph-capture path — and vLLM is what Modal's own official examples point users toward,
so it's a representative default, not a strawman). It also doesn't yet include Reflex's own
numbers *on Modal specifically* — that's Phase 1 (build the deployment) and Phase 3 (the real
head-to-head), not this check. Per the plan: don't cite a Reflex-vs-Modal number until Phase 3
actually runs.

## Cost

One ephemeral `modal run` invocation, L4 GPU, ~3 minutes of container time (plus a one-time,
CPU-only, unbilled-as-GPU-time image build). At Modal's public L4 rate (~$0.80/hr), total spend
was on the order of $0.05.
