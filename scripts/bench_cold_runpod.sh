#!/usr/bin/env bash
# scripts/bench_cold_runpod.sh — cold-invocation benchmark against a deployed
# Runpod Serverless **load-balancing** endpoint (the deployment
# serverless/runpod/README.md describes).
#
# What this measures, and why it needs its own script rather than a plain
# command through bench_cold_common.sh: on a serverless platform a cold start
# is a property of the *endpoint's worker state*, not of a command you can just
# re-run. run N times in a row and runs 2..N hit an already-warm worker unless
# the endpoint is forced back to zero between them. That reset has to happen
# *outside* the timed window, which is exactly the optional BENCH_PREPARE_CMD
# hook bench_cold_common.sh now exposes (default no-op — see its header). So this
# script:
#   1. defines the timed command (a single cold POST to the worker, retried past
#      Runpod's documented first-cold-request gateway 502 — see
#      serverless/runpod/README.md's "A real platform quirk" section), and
#   2. exports BENCH_PREPARE_CMD pointing back at itself in "--prepare" mode,
#      which pins the endpoint's workersMax to 0, waits out its idle timeout,
#      then raises it back to 1 — forcing a genuine scale-from-zero before every
#      run.
# The timing/reporting itself is bench_cold_common.sh's /usr/bin/time -v loop,
# unchanged, so the result is directly comparable in kind to the other
# bench_cold_*.sh outputs. (bench_cold_ollama.sh is the precedent for a variant
# that manages daemon/worker state itself; the difference here is that the
# shared harness gained a hook instead, so its single timing loop is still the
# one doing the measuring.)
#
# What a number from here does and does not mean (read serverless/runpod/
# README.md before citing it): the wall clock includes Runpod's own platform
# provisioning (~40s measured, 2026-09-26), which happens before any Reflex
# code runs. Reflex's own contribution is ~1.6s of it. This is a platform-level
# cold invocation, not an engine-only cold start — do not present it as the
# latter.
#
# Prerequisites: `curl`, GNU `time` (`/usr/bin/time -v`), and RUNPOD_API_KEY in
# the environment (a Runpod API key; also used as the endpoint bearer token).
#
# Usage:
#   RUNPOD_API_KEY=... scripts/bench_cold_runpod.sh <endpoint-id-or-url> [n_runs]
#   scripts/bench_cold_runpod.sh --prepare <endpoint-id> <idle-timeout-seconds>
#     (internal: the BENCH_PREPARE hook; not meant to be called directly)

set -euo pipefail

if [[ "${1:-}" == "--prepare" ]]; then
  # Cold-state reset, run by bench_cold_common.sh before each timed run, outside
  # /usr/bin/time. Kept deliberately simple: pin max workers to 0 and wait out
  # the endpoint's idle timeout so the platform actually tears the worker down,
  # then allow one worker again. A REST PATCH that fails is warned about and
  # falls back to the timeout sleep alone rather than aborting — a best-effort
  # cold reset is still better than silently timing a warm worker.
  endpoint_id="${2:?usage: $0 --prepare <endpoint-id> <idle-timeout-seconds>}"
  idle_timeout="${3:?usage: $0 --prepare <endpoint-id> <idle-timeout-seconds>}"
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"

  set_workers_max() {
    curl -fsS -X PATCH "https://rest.runpod.io/v1/endpoints/${endpoint_id}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"workersMax\":$1}" >/dev/null
  }

  if set_workers_max 0; then
    # idle_timeout is when an *idle* worker shuts down; the scale-down is not
    # instantaneous, so leave a margin.
    sleep $(( idle_timeout + 15 ))
  else
    echo "warning: could not set workersMax=0 for ${endpoint_id}; waiting ${idle_timeout}s anyway" >&2
    sleep "$idle_timeout"
  fi
  set_workers_max 1 || echo "warning: could not restore workersMax=1 for ${endpoint_id}" >&2
  sleep 3
  exit 0
fi

target="${1:?usage: RUNPOD_API_KEY=... $0 <endpoint-id-or-url> [n_runs]}"
n_runs="${2:-3}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
command -v python3 >/dev/null 2>&1 \
  || { echo "error: python3 required to build the request JSON safely" >&2; exit 1; }

idle_timeout="${RUNPOD_IDLE_TIMEOUT:-10}"
prompt="${RUNPOD_PROMPT:-Once upon a time}"

# Accept either a bare endpoint id or the full worker URL, so a caller can pass
# whatever the deploy script printed. The id is what the --prepare REST calls
# need; the URL is what the timed request needs.
if [[ "$target" == *"://"* ]]; then
  endpoint_url="${target%/}"
  endpoint_id="$(printf '%s' "$endpoint_url" | sed -E 's#^https?://([^.]+)\..*#\1#')"
else
  endpoint_id="$target"
  endpoint_url="https://${endpoint_id}.api.runpod.ai"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The cold request: a normal, non-streaming OpenAI-compatible chat completion
# with a 1-token cap. It is served only once the worker's managed `reflex`
# process is READY, so its round trip is the platform-plus-engine cold start.
# `--retry` retries the transient statuses Runpod's gateway emits while a worker
# is still coming up (502/503/504 since curl 7.71); `--retry-connrefused` covers
# a refused connection during the same window, and `--max-time` bounds a
# request stuck waiting on a half-ready worker.
payload="$(python3 -c 'import json,sys; print(json.dumps({"model":"reflex","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":1,"stream":False,"temperature":0}))' "$prompt")"

echo "endpoint: $endpoint_url  (id: $endpoint_id, n=$n_runs, idle_timeout=${idle_timeout}s)" >&2

BENCH_PREPARE_CMD="$script_dir/bench_cold_runpod.sh --prepare $endpoint_id $idle_timeout" \
  "$script_dir/bench_cold_common.sh" runpod "$n_runs" -- \
  curl -fsS -o /dev/null \
    -w 'http_status=%{http_code} time_total_s=%{time_total}\n' \
    --retry 6 --retry-delay 3 --retry-connrefused --retry-max-time 360 --max-time 360 \
    -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H 'Content-Type: application/json' \
    --data "$payload" \
    "$endpoint_url/v1/chat/completions"
