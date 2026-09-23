#!/usr/bin/env bash
# scripts/bench_cold_system1_vs_jev.sh — latency-only, disclosed-caveat
# comparison of Reflex's System1 candidate-scoring path against
# TypeSafe AI's published Jev latency figures.
#
# THIS IS NOT A HEAD-TO-HEAD BENCHMARK — it is an illustrative citation,
# and must never be presented or tabulated alongside the engine-vs-engine
# results from bench_cold_vllm.sh or the llama.cpp comparison. It compares:
#
#   - Reflex: MEASURED here, locally, on this GPU — a real cold
#     process launch through System1's single-pass candidate scoring
#     (`Model::system1_evaluate` / the `reflex system1` subcommand),
#     reported as `process_start_to_result_ms`.
#   - Jev: TypeSafe AI's own PUBLISHED latency claims for their "Jev"
#     System One model (10-15ms compute / 70-500ms end-to-end via their
#     managed cloud API), NOT independently reproduced by this script or
#     this project.
#
# Why this is illustrative rather than comparative (see DECISIONS.md's
# "TypeSafe Jev comparison framing" entry for the full reasoning):
#   - Jev's number includes a network round-trip to TypeSafe's cloud;
#     Reflex's is a pure local process launch with zero network
#     dependency at all — different deployment models, not just different
#     numbers.
#   - Jev is (per TypeSafe's own description) a purpose-built structured-
#     decision model; Reflex's System1 path is a generic
#     instruction-tuned Qwen3 GGUF with an efficient single-pass scoring
#     head bolted on. This makes NO decision-quality/calibration claim —
#     latency only.
#   - Jev's published figures are self-reported/marketing numbers, not
#     independently verified here.
#
# Usage: scripts/bench_cold_system1_vs_jev.sh <path-to-qwen3-gguf> [n_runs]
#   (expects a release build of the `reflex` binary at target/release/)

set -euo pipefail

gguf_path="${1:?usage: $0 <path-to-qwen3-gguf> [n_runs]}"
n_runs="${2:-3}"
# Candidates must be an exact continuation of the prompt at the tokenizer
# level (Model::resolve_candidate_token_ids diffs encode(prompt) against
# encode(prompt+candidate)) -- a leading space on each candidate matches
# how a GPT-style BPE tokenizer would naturally split "...A:" + " True".
prompt="Q: Is the sky blue during the day? A:"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
bin_path="$repo_root/target/release/reflex"

if [[ ! -x "$bin_path" ]]; then
  echo "error: $bin_path not found — build it first: cargo build --release --bin reflex" >&2
  exit 1
fi

"$script_dir/bench_cold_common.sh" system1 "$n_runs" -- \
  "$bin_path" system1 "$gguf_path" "$prompt" --candidate " True" --candidate " False"

cat <<'EOF'

--- Jev published figures (TypeSafe AI "Jev" System One model, released 2026-09-15) ---
compute:      10-15ms    (TypeSafe's own reported figure)
end-to-end:   70-500ms   (via TypeSafe's managed cloud API; network round-trip included)
source:       TypeSafe AI product materials / press coverage.
              NOT independently reproduced or verified by this project.
              Read this script's header comment and DECISIONS.md's
              "TypeSafe Jev comparison framing" entry before citing these
              numbers anywhere — this is an illustrative citation, not a
              benchmark result.
EOF
