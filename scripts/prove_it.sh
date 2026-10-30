#!/usr/bin/env bash
# scripts/prove_it.sh — from a fresh clone to "my numbers vs the published numbers".
#
# Builds reflex (unless REFLEX_BIN is set), runs `reflex doctor` and `reflex smoke`,
# then one untimed warm-up and n cold `reflex system1` processes on Qwen3-0.6B-Q4_K_M,
# and prints p50 per phase next to the reference row in bench/baseline-*.json for the
# closest GPU, with what differs between the two setups.
#
# Usage: scripts/prove_it.sh [path-to-gguf]
#   default model: models/Qwen3-0.6B-Q4_K_M.gguf (prints a download command if missing)
# Environment:
#   REFLEX_BIN       use this binary instead of building target/release/reflex
#   PROVE_IT_RUNS    timed runs (default 5)
#   PYTHON           Python 3 interpreter for the report (default python3)
#   PROVE_IT_OUT     directory for the logs (default bench-results/<timestamp>_prove_it)
#
# Exit codes: 0 = comparison printed (slower than the reference is not a failure),
# 1 = a real failure (build, doctor check, smoke or system1 error, unparsable output),
# 2 = usage error or model file missing.
#
# Each run's stdout/stderr is kept in the log directory, printed at the end.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
model_name="Qwen3-0.6B-Q4_K_M.gguf"
gguf_path="${1:-$repo_root/models/$model_name}"
n_runs="${PROVE_IT_RUNS:-5}"
python="${PYTHON:-python3}"
prompt="Q: Is the sky blue during the day? A:"

die() {
  echo "prove_it: $*" >&2
  exit 1
}

if [[ $# -gt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  [[ $# -gt 1 ]] && exit 2
  exit 0
fi
if ! [[ "$n_runs" =~ ^[1-9][0-9]*$ ]]; then
  echo "prove_it: PROVE_IT_RUNS must be a positive integer, got '$n_runs'" >&2
  exit 2
fi
if [[ ! -f "$gguf_path" ]]; then
  cat >&2 <<EOF
prove_it: model not found: $gguf_path

Download the reference model (397 MB) and run again:

  mkdir -p models
  curl -fL -o models/$model_name \\
    https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/$model_name
  scripts/prove_it.sh models/$model_name

(or: hf download unsloth/Qwen3-0.6B-GGUF $model_name --local-dir models)
EOF
  exit 2
fi
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null \
  || die "the report needs Python 3.8+ ('$python'); set PYTHON to another interpreter"
command -v date >/dev/null && [[ "$(date +%s%N)" =~ ^[0-9]+$ ]] \
  || die "needs GNU date (date +%s%N) for the external wall clock"

if [[ -n "${REFLEX_BIN:-}" ]]; then
  bin="$REFLEX_BIN"
else
  echo "== cargo build --release --bin reflex"
  (cd "$repo_root" && cargo build --release --bin reflex) || die "build failed"
  bin="${CARGO_TARGET_DIR:-$repo_root/target}/release/reflex"
fi
[[ -x "$bin" ]] || die "no executable reflex binary at $bin"

out="${PROVE_IT_OUT:-$repo_root/bench-results/$(date +%Y%m%d_%H%M%S)_prove_it}"
mkdir -p "$out" || die "cannot create $out"

# Runs reflex with its output in $out/<name>.log / <name>.err; returns its exit code.
run() {
  local name="$1"
  shift
  "$bin" "$@" >"$out/$name.log" 2>"$out/$name.err"
}

echo "== reflex doctor"
run doctor doctor
status=$?
grep '^REFLEX_DOCTOR_CHECK' "$out/doctor.log" | grep -v ' status=pass ' | sed 's/^REFLEX_DOCTOR_CHECK /  /'
grep -E '^REFLEX_DOCTOR_(OK|FAIL)' "$out/doctor.log" | sed 's/^/  /'
if [[ "$status" -ne 0 ]]; then
  cat "$out/doctor.err" >&2
  die "reflex doctor failed (exit $status): fix the failed check above first"
fi

echo "== reflex smoke"
run smoke smoke
status=$?
grep '^REFLEX_SMOKE_OK' "$out/smoke.log" | sed 's/^/  /'
if [[ "$status" -ne 0 ]] || ! grep -q '^REFLEX_SMOKE_OK' "$out/smoke.log"; then
  cat "$out/smoke.err" >&2
  die "reflex smoke failed (exit $status)"
fi

system1=(system1 "$gguf_path" "$prompt" --candidate " True" --candidate " False")

echo "== reflex system1 warm-up (not counted: reads the model into the page cache)"
run warmup "${system1[@]}"
status=$?
if [[ "$status" -ne 0 ]]; then
  cat "$out/warmup.err" >&2
  die "reflex system1 failed (exit $status)"
fi
grep -o 'process_start_to_result_ms=[0-9.]*' "$out/warmup.log" | head -1 | sed 's/^/  warm-up: /'

echo "== reflex system1, $n_runs cold processes"
for i in $(seq 1 "$n_runs"); do
  start=$(date +%s%N)
  run "stdout_$i" "${system1[@]}"
  status=$?
  end=$(date +%s%N)
  if [[ "$status" -ne 0 ]]; then
    cat "$out/stdout_$i.err" >&2
    die "reflex system1 run $i failed (exit $status)"
  fi
  awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f\n", (e - s) / 1e6 }' >"$out/wall_ms_$i"
  grep -o 'process_start_to_result_ms=[0-9.]*' "$out/stdout_$i.log" | head -1 | sed "s/^/  run $i: /"
done

echo
"$python" "$script_dir/prove_it_report.py" "$out" || die "could not read the runs in $out"
echo "raw logs: $out"
