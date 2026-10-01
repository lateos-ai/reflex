#!/usr/bin/env bash
# scripts/gpu_nightly_tests.sh — runs every `#[ignore]`d GPU test, one at a time,
# each pointed at the model/fixture it needs, and prints a PASS/FAIL/SKIP table.
#
# The test list is discovered (`cargo test -- --ignored --list`), not hardcoded, so a
# newly added GPU test runs automatically. Tests whose prerequisite file is absent are
# reported as SKIP with the reason -- never silently dropped. A test this script has no
# rule for runs against the dense model; if it really needs something else it fails
# loudly, which is the signal to add a rule below.
#
# Usage: scripts/gpu_nightly_tests.sh           (from the repo root, on a CUDA machine)
# Environment:
#   DENSE_GGUF     required  dense Qwen3 GGUF, e.g. Qwen3-0.6B-Q4_K_M.gguf
#   HYBRID_GGUF    optional  Qwen3.5 hybrid GGUF, e.g. Qwen3.5-0.8B-Q4_K_M.gguf
#   IQ_GGUF        optional  a GGUF containing IQ-family tensors
#   MLA_REAL_GGUF  optional  the real DeepSeek-V2-Lite GGUF (~17 GB; see docs/DEVELOPMENT.md)
#   test-data/*.gguf         optional synthetic fixtures (see docs/DEVELOPMENT.md "Known
#                            test-fixture limitations"); tests needing them SKIP if absent
#   CARGO_TEST_FLAGS         default "--release --features ipc"
#   LOG_DIR                  default bench-results/gpu-tests-<timestamp>
#   GITHUB_STEP_SUMMARY      if set, the table is appended there too
# Exit status: 1 if any test failed (or ran zero tests), else 0.

set -uo pipefail

: "${DENSE_GGUF:?set DENSE_GGUF to a dense Qwen3 GGUF path}"
HYBRID_GGUF="${HYBRID_GGUF:-}"
IQ_GGUF="${IQ_GGUF:-}"
MLA_REAL_GGUF="${MLA_REAL_GGUF:-}"
read -r -a cargo_flags <<<"${CARGO_TEST_FLAGS:---release --features ipc}"
LOG_DIR="${LOG_DIR:-bench-results/gpu-tests-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"

# Prints "<env-model-path>|<required-file>|<how to provide it>" for a test: the
# REFLEX_TEST_GGUF value to pass (may be empty), the file that must exist for the test
# to run, and what to set when it is missing.
requirement() {
  case "$1" in
    *::prefill_hybrid_batched_matches_sequential) echo "$HYBRID_GGUF|$HYBRID_GGUF|set HYBRID_GGUF" ;;
    *::iq_dequant_kernel_matches_host_on_real_tensors) echo "$IQ_GGUF|$IQ_GGUF|set IQ_GGUF" ;;
    *::prefill_mla_batched_matches_sequential_real_moe_checkpoint) echo "$MLA_REAL_GGUF|$MLA_REAL_GGUF|set MLA_REAL_GGUF" ;;
    lora::moe_expert_lora_fixture_tests::*) echo "|test-data/tiny-qwen3moe-lora.gguf|fixture missing" ;;
    *::qwen3moe_fixture_*) echo "|test-data/tiny-qwen3moe.gguf|fixture missing" ;;
    *::qwen35moe_fixture_*) echo "|test-data/tiny-qwen35moe.gguf|fixture missing" ;;
    model::mla_batching_tests::*|*::online_attention_matches_legacy_kernels_mla) echo "|test-data/deepseek-tiny-mla.gguf|fixture missing" ;;
    *) echo "$DENSE_GGUF|$DENSE_GGUF|DENSE_GGUF path does not exist" ;;
  esac
}

echo "building test binary (${cargo_flags[*]})..." >&2
if ! cargo test "${cargo_flags[@]}" --lib --no-run >"$LOG_DIR/build.log" 2>&1; then
  echo "error: test build failed, see $LOG_DIR/build.log" >&2
  tail -30 "$LOG_DIR/build.log" >&2
  exit 1
fi

mapfile -t tests < <(cargo test "${cargo_flags[@]}" --lib -- --ignored --list 2>/dev/null \
  | sed -n 's/: test$//p')
if [[ ${#tests[@]} -eq 0 ]]; then
  echo "error: found no #[ignore]d tests -- the listing itself is broken" >&2
  exit 1
fi

pass=0 fail=0 skip=0
rows=""
for t in "${tests[@]}"; do
  IFS='|' read -r model need hint <<<"$(requirement "$t")"
  if [[ -z "$need" || ! -e "$need" ]]; then
    skip=$((skip + 1))
    rows+="| \`$t\` | SKIP | prerequisite not available (${hint}${need:+: $need}) |"$'\n'
    continue
  fi
  log="$LOG_DIR/${t//::/__}.log"
  start=$(date +%s)
  REFLEX_TEST_GGUF="$model" cargo test "${cargo_flags[@]}" --lib -- --ignored --exact "$t" \
    --test-threads=1 >"$log" 2>&1
  status=$?
  secs=$(( $(date +%s) - start ))
  # Exactly one test must have run: a filter that matches nothing also "passes".
  if [[ $status -eq 0 ]] && grep -q "test result: ok. 1 passed" "$log"; then
    pass=$((pass + 1))
    rows+="| \`$t\` | PASS | ${secs}s |"$'\n'
  else
    fail=$((fail + 1))
    reason=$(grep -m1 -A1 "panicked at" "$log" | tail -1 | cut -c1-160 | tr '|' '/')
    rows+="| \`$t\` | **FAIL** | ${reason:-exit $status, see log} |"$'\n'
  fi
done

report="### GPU tests: $pass passed, $fail failed, $skip skipped

| test | result | detail |
|---|---|---|
$rows
Per-test logs: \`$LOG_DIR/\`
"
echo "$report"
[[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && echo "$report" >>"$GITHUB_STEP_SUMMARY"
[[ $fail -eq 0 && $pass -gt 0 ]]
