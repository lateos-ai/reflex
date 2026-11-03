#!/usr/bin/env bash
# scripts/bench_relay_nightly.sh -- the GPU nightly's warm agent-relay micro-bench.
#
# Runs scripts/agent_relay_bench.py three times against `reflex stdio`, each with
# the shared system prompt pinned in VRAM and fixed prompts (--fixed-notes, greedy):
#   pinned       a generate hop resumed from the pinned prefix
#   candidates   the same plus a System-1 `candidates` hop before every generate hop
#   constrained  a generate hop under scripts/relay_handoff_schema.json
# and prints one config line plus the three tables, which
# `scripts/bench_compare.py --relay` compares against bench/relay-baseline-t4.json.
# The per-hop JSON (every sample) goes to <out-dir>/relay-<kind>.json.
#
# Usage: scripts/bench_relay_nightly.sh <path-to-gguf> [hops] [out-dir]
#   (expects a release build of `reflex` with the ipc feature at target/release/)

set -euo pipefail

gguf_path="${1:?usage: $0 <path-to-gguf> [hops] [out-dir]}"
hops="${2:-20}"
out_dir="${3:-bench-results}"
warmup=3
max_tokens=32
labels="planner,researcher,coder,reviewer,tester,summarizer"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
bin_path="$repo_root/target/release/reflex"

if [[ ! -x "$bin_path" ]]; then
  echo "error: $bin_path not found — build it first: cargo build --release --features ipc --bin reflex" >&2
  exit 1
fi
mkdir -p "$out_dir"

common=(stdio --reflex "$bin_path" --model "$gguf_path" --pin-prefix --fixed-notes
        --hops "$hops" --warmup "$warmup" --max-tokens "$max_tokens")

echo "Agent relay bench config: model=$(basename "$gguf_path") target=stdio hops=$hops" \
     "warmup=$warmup max_tokens=$max_tokens fixed_notes=1 sampling=greedy"
python3 "$script_dir/agent_relay_bench.py" "${common[@]}" \
  --json-out "$out_dir/relay-pinned.json"
python3 "$script_dir/agent_relay_bench.py" "${common[@]}" --system1 "$labels" \
  --json-out "$out_dir/relay-candidates.json"
python3 "$script_dir/agent_relay_bench.py" "${common[@]}" --json-prompt \
  --json-schema "$script_dir/relay_handoff_schema.json" \
  --json-out "$out_dir/relay-constrained.json"
