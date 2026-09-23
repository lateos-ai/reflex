#!/usr/bin/env bash
# scripts/bench_cold_common.sh — reusable cold-start benchmark harness.
#
# Runs a command N times, each under `/usr/bin/time -v` (external wall
# clock, including OS exec/dynamic-linking overhead — not the engine's own
# internal timer), and prints a markdown results table (wall clock, peak
# RSS, user time, sys time) per run. This mirrors the exact methodology
# already used for the Reflex-vs-llama.cpp comparison in
# README.md and codified in DECISIONS.md's "Cold-start-vs-llama.cpp
# benchmark methodology" entry — reuse it for every new engine instead of
# inventing a new measurement approach.
#
# Before trusting this harness for a NEW engine, validate it by re-running
# it against the llama.cpp / Reflex commands from that existing
# comparison and confirming it reproduces the already-published numbers
# (README.md's "First real cold-start benchmark" section) within noise.
#
# Usage: bench_cold_common.sh <label> <n_runs> -- <command...>
# Example:
#   scripts/bench_cold_common.sh llama.cpp 3 -- \
#     ./llama-cli -m Qwen3-0.6B-Q4_K_M.gguf -p "Once upon a time" \
#       -n 1 --temp 0 -ngl 99 --no-warmup -st --simple-io
#
# Raw /usr/bin/time -v logs and stdout/stderr per run are kept under
# bench-results/<timestamp>_<label>/ for later inspection — never
# discarded, since disclosing what actually happened (including failed or
# odd runs) matters more here than a tidy summary.

set -euo pipefail

if [[ "${3:-}" != "--" ]]; then
  echo "usage: $0 <label> <n_runs> -- <command...>" >&2
  exit 1
fi

label="$1"
n_runs="$2"
shift 3
cmd=("$@")

if [[ "${#cmd[@]}" -eq 0 ]]; then
  echo "error: no command given after --" >&2
  exit 1
fi

if ! command -v /usr/bin/time >/dev/null 2>&1; then
  echo "error: /usr/bin/time (GNU time, needs -v support) not found — this harness requires it for external wall-clock measurement" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_root/bench-results/$(date +%Y%m%d_%H%M%S)_${label}"
mkdir -p "$results_dir"

echo "results dir: $results_dir" >&2
echo "command: ${cmd[*]}" >&2
echo >&2

echo "| run | wall clock | peak RSS | user time | sys time |"
echo "|---|---|---|---|---|"

for i in $(seq 1 "$n_runs"); do
  time_log="$results_dir/time_${i}.log"
  stdout_log="$results_dir/stdout_${i}.log"
  stderr_log="$results_dir/stderr_${i}.log"

  status=0
  /usr/bin/time -v -o "$time_log" -- "${cmd[@]}" >"$stdout_log" 2>"$stderr_log" || status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "warning: run $i of '$label' exited with status $status — see $stderr_log" >&2
  fi

  wall=$(grep 'Elapsed (wall clock) time' "$time_log" | awk -F': ' '{print $2}')
  rss_kb=$(grep 'Maximum resident set size' "$time_log" | awk -F': ' '{print $2}')
  user=$(grep 'User time (seconds)' "$time_log" | awk -F': ' '{print $2}')
  sys=$(grep 'System time (seconds)' "$time_log" | awk -F': ' '{print $2}')
  rss_mb=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.0f", kb / 1024 }')

  echo "| $i | ${wall} | ${rss_mb} MB | ${user}s | ${sys}s |"
done

echo
echo "label: $label" >&2
echo "raw logs kept in: $results_dir" >&2
