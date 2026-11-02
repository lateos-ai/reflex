## What this changes

<!-- One or two sentences. Link an issue if there is one. -->

## Cold start

<!--
Reflex is judged by the time and energy from process launch to the first result.
Give the effect on a representative small model (Qwen3-0.6B-Q4_K_M), or say why there
is none. See docs/DEVELOPMENT.md, "Proposing a change".
-->

- [ ] **Host-only change** (docs, tests, scripts, sidecar, CLI parsing): no effect on the engine's cold path.
- [ ] **Measured:** `system1` / `generate` p50 before → after, `n` runs, GPU (`scripts/prove_it.sh` prints `system1` p50 per phase):
- Energy (if measured: a `--features nvml` build run with `REFLEX_ENERGY=1`, or `reflex-energy`):

## Non-goals

- [ ] This adds no batching, request queue, scheduler, thread pool, network listener in the
      core engine, multi-prefix cache or multi-tenant routing (see "Non-goals" in
      `docs/DEVELOPMENT.md`).

## Correctness

- [ ] `cargo fmt --check` and `cargo clippy --all-targets --features ipc -- -D warnings`
      pass, for the root crate and `sidecar/openai-adapter` if it changed.
- [ ] `cargo test --features ipc` passes (`REFLEX_SKIP_CUDA=1` is fine without a GPU).
- [ ] If the forward pass, kernels, weights or tokenizer changed: `reflex check` token
      ids (and logit checksum, for changes meant to be exact) on a GPU, and the GPU tests
      (`cargo test --release --features ipc -- --ignored`). Paste the result or say which
      GPU run is still needed.
