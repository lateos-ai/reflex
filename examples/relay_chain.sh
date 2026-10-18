#!/bin/sh
# examples/relay_chain.sh -- the cold-chain version of relay_orchestrator.py.
#
# Every hop is a fresh `reflex` process and hops hand off through files:
#   setup   `reflex generate --export-kv` writes the team prompt's cache file once;
#   hop 0   router: `reflex system1` scores the labels over the whole prompt
#           (system1 doesn't resume from a file);
#   hop 1   worker: `reflex generate --import-kv` resumes from the team prompt and
#           writes a JSON Schema-constrained handoff, saved to a file;
#   hop 2   sub-agent: `reflex generate --import-kv` resumes from the team prompt
#           again, with the handoff as its continuation.
# `reflex generate` can't import and export in one run, so a hop can't pass its own
# state on; the shared prefix's file and the carry text are what travel.
#
# Prints one relay envelope (examples/relay_envelope.schema.json) per hop as a JSON
# line. Needs a release build with JSON output (`cargo build --release --features
# ipc --bin reflex`), a GPU and a dense/MoE GGUF, e.g. Qwen3-0.6B. POSIX sh, sed
# and awk only.
#
# Usage: relay_chain.sh <reflex-binary> <gguf> [task]
# Environment: MAX_TOKENS (default 64), WEIGHTS (f16|f32, default the engine's),
#   KV_DTYPE (f32|f16, default f32), KV_DIR (default /dev/shm, else a temp dir),
#   KEEP_FILES=1 keeps the work directory.

set -eu

if [ $# -lt 2 ]; then
  echo "usage: relay_chain.sh <reflex-binary> <gguf> [task]" >&2
  exit 2
fi
reflex=$1
model=$2
task=${3:-"Add a --dry-run flag to the deploy script so it prints the commands instead of running them."}
max_tokens=${MAX_TOKENS:-64}
kv_dtype=${KV_DTYPE:-f32}
weights=${WEIGHTS:-}

die() { echo "relay_chain: $*" >&2; exit 1; }

if [ -z "${KV_DIR:-}" ] && [ -d /dev/shm ]; then KV_DIR=/dev/shm; fi
work=$(mktemp -d "${KV_DIR:-${TMPDIR:-/tmp}}/reflex-relay-chain.XXXXXX")
cleanup() {
  if [ "${KEEP_FILES:-0}" = 1 ]; then
    echo "relay_chain: kept $work" >&2
  else
    rm -rf "$work"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# The cache file's path goes into the envelopes as written.
case $work in
  *\"* | *\\*) die "the work directory's path can't hold a quote or backslash: $work" ;;
esac

nl='
'
team="You are part of a small software team: a router, a worker and a sub-agent. The router names who handles each task, the worker writes a handoff, and the sub-agent does the work.$nl"
task_line="Task: $task$nl"

# Milliseconds since the epoch, or nothing if date has no %N (then wall_ms is left out).
now_ms() {
  ns=$(date +%s%N 2>/dev/null) || ns=
  case $ns in
    ''|*[!0-9]*) ;;
    *) echo $((ns / 1000000)) ;;
  esac
}

# run <name> <cmd...>: runs one reflex process, stdout to $work/<name>.out. Sets
# wall_ms. On failure prints its stderr and exits.
run() {
  name=$1
  shift
  t0=$(now_ms)
  if ! "$@" >"$work/$name.out" 2>"$work/$name.err"; then
    tail -n 20 "$work/$name.err" >&2
    die "$name: $1 failed"
  fi
  t1=$(now_ms)
  wall_ms=
  if [ -n "$t0" ] && [ -n "$t1" ]; then wall_ms=$((t1 - t0)); fi
}

# result_line <file> <key>: the last --json line carrying "<key>":.
result_line() {
  line=$(grep "\"$2\":" "$1" | tail -n 1) || true
  [ -n "$line" ] || die "no \"$2\" line in $1 (was reflex built with --features ipc or json-output?)"
  printf '%s\n' "$line"
}

# json_str <line> <key>: the raw JSON string literal (quotes included) of a key.
# The engine's serializer escapes every quote inside a value, so the unescaped
# "key": can't come from inside one.
json_str() {
  v=$(printf '%s\n' "$1" | sed -E -n "s/.*\"$2\":(\"([^\"\\\\]|\\\\.)*\").*/\\1/p")
  [ -n "$v" ] || die "no string \"$2\" in: $1"
  printf '%s' "$v"
}

# json_num <line> <key>: a number field, as written.
json_num() {
  v=$(printf '%s\n' "$1" | sed -E -n "s/.*\"$2\":(-?[0-9][0-9.eE+-]*).*/\\1/p")
  [ -n "$v" ] || die "no number \"$2\" in: $1"
  printf '%s' "$v"
}

# json_unescape: decodes one JSON string literal on stdin to its raw text. The
# engine escapes only quotes, backslashes and control characters (\uXXXX below
# 0x20), so \u needs no UTF-8 encoding.
json_unescape() {
  awk '
    BEGIN { hex = "0123456789abcdef" }
    {
      s = (NR > 1 ? s "\n" : "") $0
    }
    END {
      s = substr(s, 2, length(s) - 2)
      out = ""
      for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c != "\\") { out = out c; continue }
        i++
        c = substr(s, i, 1)
        if (c == "n") out = out "\n"
        else if (c == "t") out = out "\t"
        else if (c == "r") out = out "\r"
        else if (c == "b") out = out "\b"
        else if (c == "f") out = out "\f"
        else if (c == "u") {
          code = 0
          for (j = 1; j <= 4; j++)
            code = code * 16 + index(hex, tolower(substr(s, i + j, 1))) - 1
          out = out sprintf("%c", code)
          i += 4
        } else out = out c
      }
      printf "%s", out
    }'
}

# timings <wall_ms> [key value]...: an envelope timings object.
timings() {
  obj=
  if [ -n "$1" ]; then obj="\"wall_ms\":$1"; fi
  shift
  while [ $# -ge 2 ]; do
    if [ -n "$2" ]; then obj="${obj:+$obj,}\"$1\":$2"; fi
    shift 2
  done
  printf '{%s}' "$obj"
}

prefix_kv="$work/team.cskv"

# Setup: the team prompt's cache file, written once. Every later hop resumes from it.
run export "$reflex" generate "$model" "$team" --export-kv "$prefix_kv" \
  --export-kv-dtype "$kv_dtype" --json ${weights:+--weights "$weights"}

# Hop 0: the router (fresh, over the whole prompt).
run router "$reflex" system1 "$model" \
  "${team}${task_line}The team member who should handle this task is the" \
  --candidate " worker" --candidate " reviewer" --candidate " planner" \
  --json ${weights:+--weights "$weights"}
res=$(result_line "$work/router.out" best_idx)
chosen=$(json_num "$res" best_idx)
label=$(json_str "$res" best_text | sed 's/^" /"/')
entropy=$(json_num "$res" entropy)
cand=$(grep "^{\"idx\":$chosen," "$work/router.out" | tail -n 1)
[ -n "$cand" ] || die "router: no candidate line for idx $chosen"
probability=$(json_num "$cand" probability)
printf '{"version":1,"agent":"router","hop":0,"decision":{"label":%s,"chosen":%s,"probability":%s,"entropy":%s},"timings":%s}\n' \
  "$label" "$chosen" "$probability" "$entropy" "$(timings "$wall_ms")"

# Hop 1: the worker resumes from the team prompt and writes a constrained handoff.
cat >"$work/handoff.schema.json" <<'EOF'
{"type": "object", "properties": {"summary": {"type": "string"}, "next_step": {"type": "string"}}, "required": ["summary", "next_step"]}
EOF
run worker "$reflex" generate "$model" \
  "${task_line}Worker: a handoff for the sub-agent, with a one-sentence summary and the single next step.${nl}Handoff JSON:${nl}" \
  --import-kv "$prefix_kv" --json-schema "$work/handoff.schema.json" \
  --max-tokens "$max_tokens" --json ${weights:+--weights "$weights"}
res=$(result_line "$work/worker.out" token_text)
carry=$(json_str "$res" token_text)
printf '%s' "$carry" | json_unescape >"$work/handoff.txt"
printf '{"version":1,"agent":"worker","hop":1,"carry":%s,"kv":{"path":"%s","dtype":"%s"},"timings":%s}\n' \
  "$carry" "$prefix_kv" "$kv_dtype" \
  "$(timings "$wall_ms" first_token_ms "$(json_num "$res" process_start_to_first_token_ms)" \
    kv_io_ms "$(json_num "$res" kv_io_ms)")"

# Hop 2: the sub-agent resumes from the team prompt, with the handoff file's text.
run subagent "$reflex" generate "$model" \
  "${task_line}Handoff: $(cat "$work/handoff.txt")${nl}Sub-agent: I will now carry out the next step. First," \
  --import-kv "$prefix_kv" --max-tokens "$max_tokens" --json ${weights:+--weights "$weights"}
res=$(result_line "$work/subagent.out" token_text)
printf '{"version":1,"agent":"subagent","hop":2,"carry":%s,"timings":%s}\n' \
  "$(json_str "$res" token_text)" \
  "$(timings "$wall_ms" first_token_ms "$(json_num "$res" process_start_to_first_token_ms)" \
    kv_io_ms "$(json_num "$res" kv_io_ms)")"
