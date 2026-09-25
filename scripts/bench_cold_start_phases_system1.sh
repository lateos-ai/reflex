#!/usr/bin/env bash
# scripts/bench_cold_start_phases_system1.sh — p50/p95 cold-start phase
# breakdown for `reflex system1`, mirroring bench_cold_start_phases.sh
# (which covers `reflex generate`) field-for-field.
#
# Why a separate script instead of parameterizing the existing one: the two
# subcommands take different positional/flag shapes (`system1` requires
# `--candidate`, has no `--max-tokens`) and report a differently-named final
# metric (`process_start_to_result_ms`, not `process_start_to_first_token_ms`)
# -- same reasoning bench_cold_system1_vs_jev.sh already used to justify its
# own separate script rather than bolting onto bench_cold_vllm.sh.
#
# Usage: scripts/bench_cold_start_phases_system1.sh <path-to-gguf> [n_runs] [prompt]
#   (expects a release build of the `reflex` binary at target/release/;
#   candidates are fixed to " True"/" False", matching
#   bench_cold_system1_vs_jev.sh's own prompt/candidate choice so results
#   from both scripts are directly comparable)

set -euo pipefail

gguf_path="${1:?usage: $0 <path-to-gguf> [n_runs] [prompt]}"
n_runs="${2:-10}"
prompt="${3:-Q: Is the sky blue during the day? A:}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
bin_path="$repo_root/target/release/reflex"

if [[ ! -x "$bin_path" ]]; then
  echo "error: $bin_path not found — build it first: cargo build --release --bin reflex" >&2
  exit 1
fi

tmp_stderr="$(mktemp)"
trap 'rm -f "$tmp_stderr"' EXIT
"$script_dir/bench_cold_common.sh" system1-phases "$n_runs" -- \
  "$bin_path" system1 "$gguf_path" "$prompt" --candidate " True" --candidate " False" \
  2> >(tee "$tmp_stderr" >&2)

results_dir="$(grep '^results dir: ' "$tmp_stderr" | sed 's/^results dir: //')"
if [[ -z "$results_dir" || ! -d "$results_dir" ]]; then
  echo "error: could not find bench_cold_common.sh's results dir from its stderr output" >&2
  exit 1
fi

extract_field() {
  # $1 = file, $2 = field name (as it appears "field=value" in the line)
  grep -o "$2=[0-9.]*" "$1" | head -1 | cut -d= -f2
}

percentile() {
  # $1 = percentile (0-100), reads sorted numbers from stdin (one per line)
  local p="$1"
  awk -v p="$p" '{a[NR]=$1} END {
    if (NR == 0) { print "n/a"; exit }
    idx = int((p/100) * NR + 0.9999)
    if (idx < 1) idx = 1
    if (idx > NR) idx = NR
    printf "%.3f", a[idx]
  }'
}

report_phase() {
  local label="$1"
  local field="$2"
  local values=""
  for i in $(seq 1 "$n_runs"); do
    stdout_log="$results_dir/stdout_${i}.log"
    v=$(extract_field "$stdout_log" "$field")
    [[ -n "$v" ]] && values+="$v"$'\n'
  done
  local sorted
  sorted=$(echo -n "$values" | sort -n)
  local n
  n=$(echo -n "$sorted" | grep -c . || true)
  if [[ "$n" -eq 0 ]]; then
    echo "| $label | n/a (field missing) | n/a |"
    return
  fi
  local p50 p95
  p50=$(echo "$sorted" | percentile 50)
  p95=$(echo "$sorted" | percentile 95)
  echo "| $label | ${p50} ms | ${p95} ms |"
}

report_process_launch_phase() {
  local values=""
  for i in $(seq 1 "$n_runs"); do
    time_log="$results_dir/time_${i}.log"
    stdout_log="$results_dir/stdout_${i}.log"
    wall_raw=$(grep 'Elapsed (wall clock) time' "$time_log" | awk -F': ' '{print $2}')
    internal_ms=$(extract_field "$stdout_log" "process_start_to_result_ms")
    [[ -z "$internal_ms" || -z "$wall_raw" ]] && continue
    wall_s=$(awk -F: -v t="$wall_raw" 'BEGIN {
      n = split(t, parts, ":")
      s = 0
      for (i = 1; i <= n; i++) s = s * 60 + parts[i]
      print s
    }')
    launch_ms=$(awk -v w="$wall_s" -v i="$internal_ms" 'BEGIN { printf "%.3f", (w * 1000) - i }')
    values+="$launch_ms"$'\n'
  done
  local sorted
  sorted=$(echo -n "$values" | sort -n)
  local n
  n=$(echo -n "$sorted" | grep -c . || true)
  if [[ "$n" -eq 0 ]]; then
    echo "| process launch (external − internal) | n/a | n/a |"
    return
  fi
  local p50 p95
  p50=$(echo "$sorted" | percentile 50)
  p95=$(echo "$sorted" | percentile 95)
  echo "| process launch (external − internal) | ${p50} ms | ${p95} ms |"
}

echo
echo "Cold-start phase breakdown (reflex system1), n=$n_runs runs, $(basename "$gguf_path"):"
echo
echo "| phase | p50 | p95 |"
echo "|---|---|---|"
report_process_launch_phase
report_phase "cuda init" "cuda_init_ms"
report_phase "model load" "model_load_ms"
report_phase "prompt eval (scoring pass)" "prompt_eval_ms"
report_phase "total (process_start_to_result_ms)" "process_start_to_result_ms"
echo
echo "raw logs kept in: $results_dir" >&2
