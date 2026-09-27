#!/usr/bin/env bash
# scripts/loadtest_warm_endpoint.sh — fires concurrent /v1/chat/completions requests
# at a deployed always-warm endpoint (docs/aws-deployment-warm.md) to verify the
# target-tracking Auto Scaling policy actually scales the ASG out under load and back
# in once load stops. This is a rollout-verification tool for that deployment
# pattern, not a latency/throughput benchmark like scripts/bench_cold_*.sh — it makes
# no engine-vs-engine or cold-vs-warm comparison claim.
#
# Usage:
#   scripts/loadtest_warm_endpoint.sh <base-url> <api-key-header-value> \
#     [concurrency] [duration-seconds] [asg-name]
#
# Example:
#   scripts/loadtest_warm_endpoint.sh https://reflex-demo.example.com secret-value \
#     8 120 reflex-asg
#
# <api-key-header-value> is whatever value satisfies the WAF header-match rule from
# docs/aws-deployment-warm.md's Phase 4 (sent as `X-Reflex-Demo-Key`); requests
# missing or failing this check should come back blocked, not served, per that guide.
#
# If <asg-name> is given and the AWS CLI is configured, this script polls
# `aws autoscaling describe-auto-scaling-groups` once per second for the duration and
# prints each observed instance count, so you can see whether/when a scale-out (and
# later scale-in) event actually happens. Without an ASG name it just fires load and
# reports request-level pass/fail/latency, with no AWS-side observation.

set -euo pipefail

base_url="${1:?usage: $0 <base-url> <api-key-header-value> [concurrency] [duration-seconds] [asg-name]}"
api_key="${2:?usage: $0 <base-url> <api-key-header-value> [concurrency] [duration-seconds] [asg-name]}"
concurrency="${3:-8}"
duration="${4:-120}"
asg_name="${5:-}"

results_dir="$(mktemp -d)"
trap 'rm -rf "$results_dir"' EXIT

echo "[loadtest] target: $base_url"
echo "[loadtest] concurrency=$concurrency duration=${duration}s"

fire_one_request() {
  local i="$1"
  local start end status
  start=$(date +%s%3N)
  status=$(curl -s -o "$results_dir/body_$i.json" -w '%{http_code}' \
    -X POST "$base_url/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Reflex-Demo-Key: $api_key" \
    -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16}')
  end=$(date +%s%3N)
  echo "$i $status $((end - start))" >> "$results_dir/requests.log"
}

poll_asg() {
  local name="$1"
  local t0
  t0=$(date +%s)
  while [ -e "$results_dir/load_running" ]; do
    local counts
    counts=$(aws autoscaling describe-auto-scaling-groups \
      --auto-scaling-group-names "$name" \
      --query "AutoScalingGroups[0].Instances[].LifecycleState" \
      --output text 2>/dev/null || echo "ERROR")
    echo "$(( $(date +%s) - t0 )) $counts" >> "$results_dir/asg.log"
    sleep 1
  done
}

asg_poll_pid=""
if [ -n "$asg_name" ] && command -v aws >/dev/null 2>&1; then
  touch "$results_dir/load_running"
  poll_asg "$asg_name" &
  asg_poll_pid="$!"
  echo "[loadtest] polling ASG '$asg_name' every 1s for the duration of the load"
elif [ -n "$asg_name" ]; then
  echo "[loadtest] WARNING: asg-name given but 'aws' CLI not found on PATH -- skipping ASG observation" >&2
fi

end_time=$(( $(date +%s) + duration ))
i=0
pids=()
while [ "$(date +%s)" -lt "$end_time" ]; do
  # Keep up to $concurrency requests in flight at once.
  while [ "${#pids[@]}" -ge "$concurrency" ]; do
    wait -n 2>/dev/null || true
    # Rebuild pids with only still-running jobs.
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    pids=("${new_pids[@]}")
  done
  fire_one_request "$i" &
  pids+=("$!")
  i=$((i + 1))
done
wait

echo "[loadtest] fired $i requests"

if [ -n "$asg_poll_pid" ]; then
  rm -f "$results_dir/load_running"
  wait "$asg_poll_pid" 2>/dev/null || true
  echo "[loadtest] --- ASG instance lifecycle states over time (seconds_since_start states...) ---"
  cat "$results_dir/asg.log" 2>/dev/null || echo "[loadtest] (no ASG samples recorded)"
fi

echo "[loadtest] --- results ---"
total=0
ok=0
sum_ms=0
while read -r idx status ms; do
  total=$((total + 1))
  sum_ms=$((sum_ms + ms))
  if [ "$status" = "200" ]; then
    ok=$((ok + 1))
  fi
done < "$results_dir/requests.log"

echo "[loadtest] total=$total ok=$ok failed=$((total - ok))"
if [ "$total" -gt 0 ]; then
  echo "[loadtest] mean latency: $((sum_ms / total))ms"
fi
