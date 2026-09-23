#!/usr/bin/env bash
# scripts/bench_cold_vllm.sh — cold-start benchmark: vLLM vs. Reflex.
#
# Reuses bench_cold_common.sh's methodology (external wall clock via
# /usr/bin/time -v, greedy decode, disclosed caveats) so the result is
# directly comparable in kind to the existing llama.cpp comparison — see
# DECISIONS.md's "Cold-start-vs-llama.cpp benchmark methodology" entry.
#
# WHY vLLM (not just another llama.cpp-style engine): llama.cpp is, like
# Reflex, AOT-compiled via nvcc — it never JIT-compiles CUDA
# kernels, so it doesn't actually test this project's core AOT-vs-JIT
# cold-start bet (see CLAUDE.md). vLLM's CUDA graph capture / historically
# JIT-driven kernel compilation is a real, well-documented warmup cost and
# a much better foil for that specific claim.
#
# Prerequisites (install once per ThunderCompute session, not automated
# here — see DECISIONS.md for why this stays a manual step rather than a
# new binary/dependency of the core engine):
#   pip install vllm
#
# IMPORTANT — resolve before trusting any number this produces: vLLM's
# GGUF loader is less mature than llama.cpp's. Confirm it can actually
# load the target Qwen3 GGUF fixture first. If it can't, do NOT silently
# substitute a different weight format (e.g. an HF safetensors checkpoint)
# — document that substitution as an explicit methodology deviation in
# README.md, the same way this project discloses every other benchmark
# caveat, rather than folding it silently into a results table.
#
# Chat-template parity is also unconfirmed by default, same caveat as the
# llama.cpp comparison — verify vLLM is completing the raw prompt, not
# wrapping it in the GGUF's embedded chat template, unless that's what you
# intend to measure.
#
# A local .gguf path needs a separate HF tokenizer repo id (vLLM's own
# convention -- see vllm_single_shot.py's doc comment); pass it as a third
# argument, e.g. `Qwen/Qwen3-0.6B` for the Qwen3-0.6B-Q4_K_M.gguf fixture.
#
# Usage: scripts/bench_cold_vllm.sh <path-to-qwen3-gguf-or-hf-model> [n_runs] [tokenizer-repo-id]

set -euo pipefail

model_path="${1:?usage: $0 <path-to-qwen3-gguf-or-hf-model> [n_runs] [tokenizer-repo-id]}"
n_runs="${2:-3}"
tokenizer="${3:-}"
prompt="Once upon a time"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "$tokenizer" ]]; then
  "$script_dir/bench_cold_common.sh" vllm "$n_runs" -- \
    python3 "$script_dir/vllm_single_shot.py" "$model_path" "$prompt" --tokenizer "$tokenizer"
else
  "$script_dir/bench_cold_common.sh" vllm "$n_runs" -- \
    python3 "$script_dir/vllm_single_shot.py" "$model_path" "$prompt"
fi
