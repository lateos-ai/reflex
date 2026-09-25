#!/usr/bin/env bash
# scripts/bench_cold_start_phases.sh — p50/p95 cold-start phase breakdown.
#
# A single aggregate `process_start_to_first_token_ms` number hides where the
# time actually goes -- this splits it into the phases real user feedback
# asked for (see HISTORY.md's "cold-start phase breakdown" entry): process
# launch, CUDA init, model load, and prompt eval, each reported as p50/p95
# across N cold-process runs, not a single sample.
#
# Reuses `bench_cold_common.sh` for the actual N-run/external-timing/log-
# capture work (`/usr/bin/time -v`, one fresh process per run, raw logs kept
# under bench-results/) instead of reinventing it -- this script only adds
# the per-phase parsing/percentile layer on top of `reflex generate`'s own
# `gguf_open_ms`/`cuda_init_ms`/`model_load_ms`/`prompt_eval_ms` fields.
#
# "process_launch_ms" (OS exec/dynamic-linking/CRT init before `main()` runs)
# is not something `reflex generate` can report about itself -- it's derived
# here as (external wall clock) − (internal process_start_to_first_token_ms),
# the same external-vs-internal-timer gap this project's own README benchmark
# table already relies on.
#
# Usage: scripts/bench_cold_start_phases.sh <path-to-gguf> [n_runs] [prompt]
#   (expects a release build of the `reflex` binary at target/release/)

set -euo pipefail

gguf_path="${1:?usage: $0 <path-to-gguf> [n_runs] [prompt]}"
n_runs="${2:-10}"
prompt="${3:-Once upon a time}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
bin_path="$repo_root/target/release/reflex"

if [[ ! -x "$bin_path" ]]; then
  echo "error: $bin_path not found — build it first: cargo build --release --bin reflex" >&2
  exit 1
fi

# bench_cold_common.sh prints its results-dir path to stderr as its first
# line of output ("results dir: <path>") -- capture stderr separately so we
# can find the raw per-run logs it kept, while still passing its own stdout
# (the per-run wall-clock/RSS table) and stderr through to this script's own
# output, per that script's own "never discard results" convention.
tmp_stderr="$(mktemp)"
trap 'rm -f "$tmp_stderr"' EXIT
"$script_dir/bench_cold_common.sh" reflex-phases "$n_runs" -- \
  "$bin_path" generate "$gguf_path" "$prompt" --max-tokens 1 \
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
    internal_ms=$(extract_field "$stdout_log" "process_start_to_first_token_ms")
    [[ -z "$internal_ms" || -z "$wall_raw" ]] && continue
    # GNU time formats wall clock as [h:]mm:ss[.ss] -- normalize to seconds.
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
echo "Cold-start phase breakdown, n=$n_runs runs, $(basename "$gguf_path"):"
echo
echo "| phase | p50 | p95 |"
echo "|---|---|---|"
report_process_launch_phase
report_phase "cuda init" "cuda_init_ms"
report_phase "model load" "model_load_ms"
report_phase "prompt eval (first token)" "prompt_eval_ms"
report_phase "total (process_start_to_first_token_ms)" "process_start_to_first_token_ms"
echo
echo "raw logs kept in: $results_dir" >&2
