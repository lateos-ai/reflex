#!/usr/bin/env bash
# scripts/bench_cold_energy.sh — p50/p95 whole-process GPU energy for cold
# `reflex system1` runs, measured from outside the process by `reflex-energy`
# (NVML energy counter, idle baseline subtracted; see src/bin/reflex-energy.rs).
#
# Sibling of bench_cold_start_phases_system1.sh (same prompt and candidates, so the
# two are comparable): that script reports latency per phase plus the energy the
# process measures about itself, which can only start at main(). This one covers
# the whole process, including exec/CUDA init before main() and driver teardown after
# exit, and reports both gross energy and energy net of the GPU's idle draw.
#
# Usage: scripts/bench_cold_energy.sh <path-to-gguf> [n_runs] [prompt]
#   Build both binaries first:  cargo build --release --features nvml
#   Run on a dedicated GPU: the counter is device-wide.
# Environment: IDLE_MS (default 2000), the idle window measured before each run.

set -euo pipefail

gguf_path="${1:?usage: $0 <path-to-gguf> [n_runs] [prompt]}"
n_runs="${2:-10}"
prompt="${3:-Q: Is the sky blue during the day? A:}"
idle_ms="${IDLE_MS:-2000}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reflex="$repo_root/target/release/reflex"
meter="$repo_root/target/release/reflex-energy"
for bin in "$reflex" "$meter"; do
  [[ -x "$bin" ]] || { echo "error: $bin not found -- run: cargo build --release --features nvml" >&2; exit 1; }
done

results_dir="$repo_root/bench-results/$(date +%Y%m%d_%H%M%S)_energy"
mkdir -p "$results_dir"
echo "raw logs: $results_dir" >&2

for i in $(seq 1 "$n_runs"); do
  "$meter" --idle-ms "$idle_ms" -- "$reflex" system1 "$gguf_path" "$prompt" \
    --candidate " True" --candidate " False" >"$results_dir/run_${i}.log" 2>"$results_dir/run_${i}.err" \
    || echo "warning: run $i exited non-zero, see $results_dir/run_${i}.err" >&2
done

# field <name> <line prefix>: one value per run, sorted
field() {
  for i in $(seq 1 "$n_runs"); do
    grep "^$2" "$results_dir/run_${i}.log" | grep -o " $1=[-0-9.]*" | head -1 | cut -d= -f2
  done | sort -n
}

percentile() {  # $1 = percentile; sorted numbers on stdin
  awk -v p="$1" '{a[NR]=$1} END {
    if (NR == 0) { print "n/a"; exit }
    i = int((p/100) * NR + 0.9999); if (i < 1) i = 1; if (i > NR) i = NR
    print a[i] }'
}

row() {  # label, field, prefix, divisor (1000 turns mJ into J)
  local values p50 p95
  values="$(field "$2" "$3")"
  p50=$(echo "$values" | percentile 50)
  p95=$(echo "$values" | percentile 95)
  if [[ "$4" != 1 && "$p50" != n/a ]]; then
    p50=$(awk -v v="$p50" -v d="$4" 'BEGIN { printf "%.3f", v / d }')
    p95=$(awk -v v="$p95" -v d="$4" 'BEGIN { printf "%.3f", v / d }')
  fi
  echo "| $1 | $p50 | $p95 |"
}

echo
echo "Whole-process energy (reflex system1), n=$n_runs runs, $(basename "$gguf_path"), idle window ${idle_ms} ms:"
echo
echo "| metric | p50 | p95 |"
echo "|---|---|---|"
row "wall clock, spawn to exit (ms)" wall_ms REFLEX_ENERGY_OK 1
row "gross energy over the run (J)" energy_mj REFLEX_ENERGY_OK 1000
row "idle baseline for the same window (J)" idle_baseline_mj REFLEX_ENERGY_OK 1000
row "**net energy** (gross - idle) (J)" net_energy_mj REFLEX_ENERGY_OK 1000
row "idle power (W)" idle_power_mw REFLEX_ENERGY_OK 1000
row "in-process joules, main() to result (J)" joules REFLEX_SYSTEM1_OK 1
