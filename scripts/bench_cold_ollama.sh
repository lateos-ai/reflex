#!/usr/bin/env bash
# scripts/bench_cold_ollama.sh — cold-start benchmark: Ollama vs. Reflex.
#
# Ollama wraps llama.cpp's own ggml runtime (its logs show a bundled
# `llama-server` subprocess doing the actual inference) — it does NOT add a
# new data point on the AOT-vs-JIT question `bench_cold_vllm.sh` tests. What
# it does test is real: daemon/packaging overhead, the shape most people
# actually experience running a local model, not raw engine internals.
#
# THREE distinct scenarios, deliberately not conflated into one number:
#   1. cold daemon + cold model  — `ollama serve` freshly (re)started, first
#      request. The fairest comparison to llama.cpp/Reflex/vLLM's
#      "process launch to first token" framing.
#   2. warm daemon + cold model  — daemon already running, but this model
#      not yet loaded into it (first request for it, or reloaded after
#      `ollama stop`/keep_alive eviction). Tests model-load cost in
#      isolation from daemon startup.
#   3. warm daemon + warm model  — model already resident, steady-state
#      generation latency. Comparable to `reflex bench`'s warm numbers.
#
# Usage: scripts/bench_cold_ollama.sh <ollama-model-name> [n_runs]
#   (the model must already exist: `ollama create <name> -f Modelfile`,
#   `FROM <path-to-gguf>` — no chat template wrapping needed since this
#   script passes `raw: true`)

set -euo pipefail

model_name="${1:?usage: $0 <ollama-model-name> [n_runs]}"
n_runs="${2:-3}"
prompt="Once upon a time"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
results_dir="$repo_root/bench-results/$(date +%Y%m%d_%H%M%S)_ollama"
mkdir -p "$results_dir"

parse_response() {
  python3 - "$1" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
total_ms = d["total_duration"] / 1e6
load_ms = d.get("load_duration", 0) / 1e6
print(f"response={d['response']!r} total_ms={total_ms:.2f} load_ms={load_ms:.2f}")
PYEOF
}

wait_for_daemon() {
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/api/version 2>/dev/null | grep -q 200; then
      return 0
    fi
    sleep 0.5
  done
  echo "error: ollama daemon did not become ready within 60s" >&2
  return 1
}

do_generate() {
  local out="$1"
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\": \"$model_name\", \"prompt\": \"$prompt\", \"raw\": true, \"stream\": false, \"options\": {\"temperature\": 0, \"num_predict\": 1}}" \
    > "$out"
}

echo "=== scenario 1: cold daemon + cold model ($n_runs runs) ===" >&2
for i in $(seq 1 "$n_runs"); do
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
  setsid nohup ollama serve > "$results_dir/serve_cold_${i}.log" 2>&1 < /dev/null &
  disown
  wait_for_daemon
  out="$results_dir/cold_daemon_cold_model_${i}.json"
  time_log="$results_dir/cold_daemon_cold_model_${i}.time"
  /usr/bin/time -v -o "$time_log" -- bash -c "curl -s http://127.0.0.1:11434/api/generate -d '{\"model\": \"$model_name\", \"prompt\": \"$prompt\", \"raw\": true, \"stream\": false, \"options\": {\"temperature\": 0, \"num_predict\": 1}}' > $out"
  echo "run $i: $(parse_response "$out") wall_clock=$(grep 'Elapsed' "$time_log" | awk -F': ' '{print $2}')"
done

echo "=== scenario 2: warm daemon + cold model ($n_runs runs) ===" >&2
for i in $(seq 1 "$n_runs"); do
  ollama stop "$model_name" 2>/dev/null || true
  sleep 1
  out="$results_dir/warm_daemon_cold_model_${i}.json"
  time_log="$results_dir/warm_daemon_cold_model_${i}.time"
  /usr/bin/time -v -o "$time_log" -- bash -c "curl -s http://127.0.0.1:11434/api/generate -d '{\"model\": \"$model_name\", \"prompt\": \"$prompt\", \"raw\": true, \"stream\": false, \"options\": {\"temperature\": 0, \"num_predict\": 1}}' > $out"
  echo "run $i: $(parse_response "$out") wall_clock=$(grep 'Elapsed' "$time_log" | awk -F': ' '{print $2}')"
done

echo "=== scenario 3: warm daemon + warm model ($n_runs runs) ===" >&2
for i in $(seq 1 "$n_runs"); do
  out="$results_dir/warm_daemon_warm_model_${i}.json"
  do_generate "$out"
  echo "run $i: $(parse_response "$out")"
done

echo "raw logs kept in: $results_dir" >&2
