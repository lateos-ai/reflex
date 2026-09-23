#!/usr/bin/env bash
# examples/system1_guardrail.sh — three System1-based decision-gate templates:
# a boolean ALLOW/DENY gate, a multi-action selector, and a calibrated
# confidence-threshold gate. All three drive the existing `reflex system1`
# subcommand and parse its stable `REFLEX_SYSTEM1_CANDIDATE_OK`/
# `REFLEX_SYSTEM1_OK` stdout lines (see src/bin/reflex/system1.rs) -- the same
# contract scripts/bench_cold_system1_vs_jev.sh already relies on.
#
# Requires a release build (`cargo build --release --bin reflex`) and a real
# GPU with a local dense/MoE Qwen3 GGUF -- see examples/README.md.
#
# Usage:
#   system1_guardrail.sh bool <gguf> "<yes/no prompt>"
#   system1_guardrail.sh multi <gguf> "<prompt>" -- <action1> [action2 ...]
#   system1_guardrail.sh confidence <gguf> "<prompt>" -- <candidate1> [candidate2 ...]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
bin_path="$repo_root/target/release/reflex"

if [[ ! -x "$bin_path" ]]; then
  echo "error: $bin_path not found -- build it first: cargo build --release --bin reflex" >&2
  exit 1
fi

usage() {
  cat >&2 <<'EOF'
usage:
  system1_guardrail.sh bool <gguf> "<yes/no prompt>"
  system1_guardrail.sh multi <gguf> "<prompt>" -- <action1> [action2 ...]
  system1_guardrail.sh confidence <gguf> "<prompt>" -- <candidate1> [candidate2 ...]
EOF
  exit 1
}

# Extracts a bare (unquoted) KEY=value field's value from a `reflex system1`
# stdout line -- values here never contain spaces, so `[^ ]*` stops cleanly at
# the next field or end of line.
extract() { # $1=key $2=line
  printf '%s' "$2" | sed -n "s/.*\b$1=\([^ ]*\).*/\1/p"
}

# Extracts a Debug-quoted `key="..."` field's contents (used for `text=`/
# `best_text=`, which can contain spaces -- e.g. `text=" Paris"`).
extract_quoted() { # $1=key $2=line
  printf '%s' "$2" | sed -n "s/.*$1=\"\\([^\"]*\\)\".*/\1/p"
}

# Runs `reflex system1` against $gguf/$prompt with the given candidates and
# prints its raw stdout (one call site shared by all three subcommands below).
run_system1() { # $1=gguf $2=prompt, then candidate texts...
  local gguf="$1" prompt="$2"
  shift 2
  local cand_args=()
  for c in "$@"; do
    cand_args+=(--candidate "$c")
  done
  "$bin_path" system1 "$gguf" "$prompt" "${cand_args[@]}"
}

cmd="${1:-}"
[[ -n "$cmd" ]] || usage
shift || true

case "$cmd" in
  bool)
    gguf="${1:?gguf path required}"
    prompt="${2:?prompt required}"
    output="$(run_system1 "$gguf" "$prompt" " Yes" " No")"
    result_line="$(printf '%s\n' "$output" | grep '^REFLEX_SYSTEM1_OK')"
    best_text="$(extract_quoted best_text "$result_line")"
    if [[ "$best_text" == " Yes" ]]; then
      echo "ALLOW"
    else
      echo "DENY"
    fi
    ;;

  multi)
    gguf="${1:?gguf path required}"
    prompt="${2:?prompt required}"
    shift 2 || usage
    [[ "${1:-}" == "--" ]] || usage
    shift
    (( $# >= 2 )) || { echo "error: multi needs at least 2 actions after --" >&2; exit 1; }
    # Each action becomes its own candidate, prefixed with a leading space to
    # match how a GPT-style BPE tokenizer splits "...:" + " ACTION" (same
    # convention scripts/bench_cold_system1_vs_jev.sh documents).
    candidates=()
    for action in "$@"; do
      candidates+=(" $action")
    done
    output="$(run_system1 "$gguf" "$prompt" "${candidates[@]}")"
    result_line="$(printf '%s\n' "$output" | grep '^REFLEX_SYSTEM1_OK')"
    best_text="$(extract_quoted best_text "$result_line")"
    # Strip the leading space we added above before printing the selection.
    echo "${best_text# }"
    ;;

  confidence)
    gguf="${1:?gguf path required}"
    prompt="${2:?prompt required}"
    shift 2 || usage
    [[ "${1:-}" == "--" ]] || usage
    shift
    (( $# >= 2 )) || { echo "error: confidence needs at least 2 candidates after --" >&2; exit 1; }
    candidates=("$@")
    num_candidates="$#"
    output="$(run_system1 "$gguf" "$prompt" "${candidates[@]}")"
    result_line="$(printf '%s\n' "$output" | grep '^REFLEX_SYSTEM1_OK')"
    best_text="$(extract_quoted best_text "$result_line")"
    entropy="$(extract entropy "$result_line")"
    best_idx="$(extract best_idx "$result_line")"
    cand_line="$(printf '%s\n' "$output" | grep "^REFLEX_SYSTEM1_CANDIDATE_OK idx=$best_idx ")"
    probability="$(extract probability "$cand_line")"

    # Illustrative thresholds only -- tune per use case. `max_entropy` is
    # ln(num_candidates), the entropy of a uniform (maximally unsure)
    # distribution over this candidate set; escalating past half of that,
    # or below a flat 0.7 top-candidate probability, are just reasonable
    # starting points for a calibrated confidence gate.
    max_entropy="$(awk -v n="$num_candidates" 'BEGIN { printf "%.6f", log(n) }')"
    escalate="$(awk -v p="$probability" -v e="$entropy" -v maxe="$max_entropy" \
      'BEGIN { print (p < 0.7 || e > maxe * 0.5) ? "1" : "0" }')"

    echo "candidate=${best_text} probability=${probability} entropy=${entropy} max_entropy=${max_entropy}"
    if [[ "$escalate" == "1" ]]; then
      echo "decision=ESCALATE_TO_HUMAN (low confidence)"
    else
      echo "decision=${best_text# }"
    fi
    ;;

  *)
    usage
    ;;
esac
