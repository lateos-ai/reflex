#!/usr/bin/env bash
# scripts/deploy_runpod.sh — build/push the serverless/runpod image and drive a
# real Runpod Serverless **load-balancing** endpoint through the existing
# cold-start benchmark harness (scripts/bench_cold_runpod.sh, which in turn uses
# bench_cold_common.sh's /usr/bin/time -v loop).
#
# SCOPE: operational packaging only. This touches no engine code and no model
# code — it builds the same serverless/runpod/Dockerfile the README describes,
# pushes it, configures an endpoint, then measures it. It is the scripted form
# of the manual deployment serverless/runpod/README.md documents, and it follows
# that README's verified findings rather than re-deriving them:
#
#   * Load-balancing endpoint, not queue-based. The `type: "LB"` field only
#     exists on the GraphQL `saveEndpoint` mutation, so endpoint creation goes
#     through GraphQL; template creation and post-create configuration use the
#     documented REST API (rest.runpod.io/v1).
#   * GPU SKU pinned to RTX A4500 (sm_86). The GraphQL create path can only name
#     a *pool* (`gpuIds: "AMPERE_16"`), and that pool is mixed-architecture
#     (Ampere sm_86 alongside Ada sm_89) despite its name — so the SKU is pinned
#     right after creation with a REST PATCH of `gpuTypeIds`, exactly the
#     "create, then pin the SKU with a control-plane call" sequence the README
#     documents. Skipping this makes the endpoint fail nondeterministically
#     depending on which card a worker lands on.
#   * Health check is `/ping` (the adapter serves it as an alias of /healthz),
#     because Runpod's gateway polls the hardcoded `/ping` regardless of the
#     documented HEALTH_CHECK_PATH override.
#   * The first cold request can get a gateway 502; the benchmark retries it.
#   * FlashBoot is disabled by default to match the published comparison's
#     methodology (docs/serverless-cost-comparison.md disabled it on both
#     engines so it can't confound the number).
#
# IMAGE VARIANT (Deliverable B): `--slim` builds the measured runtime-slimmed
# image (`base` + libcublas-12-4; measured `docker images` 4.57GB -> 2.06GB, and
# a 1.2GB on-disk rootfs vs 2.6GB) and tags it `:runtime`; the default variant
# and tag are unchanged, so this never silently moves an existing deployment's
# base image. See README's "Container image size is part of cold start here".
#
# Cost warning: this creates real, billable Runpod resources. Use `--teardown`
# (or delete the endpoint in the console) when finished. Per the README, do not
# assume `workersMax` is honored — Runpod's autoscaler has been observed
# exceeding it — so check `runpodctl serverless get <id>` if cost matters.
#
# Usage:
#   RUNPOD_API_KEY=... scripts/deploy_runpod.sh [flags]
#
# Flags:
#   --slim             build the slim `:runtime` image variant (default: full image, :latest)
#   --no-build         skip `docker build` (use an image already built/pushed)
#   --no-push          skip `docker push` (local image only)
#   --no-endpoint      build/push only; never touch the Runpod API
#   --no-bench         configure the endpoint but don't run the cold-start benchmark
#   --endpoint-id ID   reuse an existing endpoint instead of creating one
#   --bench-runs N     benchmark runs to time (default 3)
#   --teardown         delete the endpoint + template after benchmarking
#   --dry-run          print every action/payload without executing any of it
#   -h, --help         show this help
#
# Environment overrides (defaults in parentheses):
#   REFLEX_REGISTRY_IMAGE (ghcr.io/lateos-ai/reflex-runpod)
#   REFLEX_IMAGE_TAG      (latest, or runtime with --slim)
#   REFLEX_CUDA_ARCH      (sm_86)
#   REFLEX_ENDPOINT_NAME  (reflex-runpod)
#   REFLEX_GPU_POOL       (AMPERE_16)
#   REFLEX_GPU_TYPE_ID    (NVIDIA RTX A4500)
#   REFLEX_IDLE_TIMEOUT   (10)
#   REFLEX_WORKERS_MAX    (1)
#   REFLEX_FLASHBOOT      (false)
#   REFLEX_LOCATIONS      ("" = any Runpod region; GraphQL two-letter code, e.g. RO)
#   REFLEX_CONTAINER_DISK_GB (10)
#   RUNPOD_REGISTRY_AUTH_ID ("" = public image; required for a private registry)
#
# Prerequisites: docker (for build/push), curl + python3, and RUNPOD_API_KEY for
# any endpoint step. `docker login <registry>` first if your image is private.

set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
REFLEX_REGISTRY_IMAGE="${REFLEX_REGISTRY_IMAGE:-ghcr.io/lateos-ai/reflex-runpod}"
REFLEX_CUDA_ARCH="${REFLEX_CUDA_ARCH:-sm_86}"
REFLEX_ENDPOINT_NAME="${REFLEX_ENDPOINT_NAME:-reflex-runpod}"
REFLEX_GPU_POOL="${REFLEX_GPU_POOL:-AMPERE_16}"
REFLEX_GPU_TYPE_ID="${REFLEX_GPU_TYPE_ID:-NVIDIA RTX A4500}"
REFLEX_IDLE_TIMEOUT="${REFLEX_IDLE_TIMEOUT:-10}"
REFLEX_WORKERS_MAX="${REFLEX_WORKERS_MAX:-1}"
REFLEX_FLASHBOOT="${REFLEX_FLASHBOOT:-false}"
REFLEX_LOCATIONS="${REFLEX_LOCATIONS:-}"
REFLEX_CONTAINER_DISK_GB="${REFLEX_CONTAINER_DISK_GB:-10}"
RUNPOD_REGISTRY_AUTH_ID="${RUNPOD_REGISTRY_AUTH_ID:-}"

SLIM=0
DO_BUILD=1
DO_PUSH=1
DO_ENDPOINT=1
DO_BENCH=1
DO_TEARDOWN=0
DRY_RUN=0
ENDPOINT_ID=""
BENCH_RUNS=3

usage() {
  # Print this file's leading comment block (everything after the shebang up to
  # the first non-comment line), with the leading "# " stripped.
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slim) SLIM=1 ;;
    --no-build) DO_BUILD=0 ;;
    --no-push) DO_PUSH=0 ;;
    --no-endpoint) DO_ENDPOINT=0 ;;
    --no-bench) DO_BENCH=0 ;;
    --endpoint-id) ENDPOINT_ID="${2:?--endpoint-id needs a value}"; shift ;;
    --bench-runs) BENCH_RUNS="${2:?--bench-runs needs a value}"; shift ;;
    --teardown) DO_TEARDOWN=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

die() { echo "error: $*" >&2; exit 1; }

run() {
  echo "+ $*"
  (( DRY_RUN )) || "$@"
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "required tool '$1' not found on PATH"
}

# --- REST helper (rest.runpod.io/v1): template create, endpoint patch, delete.
# Captures the body and only returns it on a 2xx; otherwise fails loudly with
# both status and body so an operator can see exactly what Runpod rejected.
rest_request() {
  local method="$1" path="$2" data="${3:-}"
  local tmp status out
  tmp="$(mktemp)"
  local args=(-sS -o "$tmp" -w '%{http_code}' -X "$method"
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
    -H 'Content-Type: application/json'
    "https://rest.runpod.io/v1${path}")
  [[ -n "$data" ]] && args+=(--data "$data")
  status="$(curl "${args[@]}")" || true
  out="$(cat "$tmp")"; rm -f "$tmp"
  if [[ "$status" != 2* ]]; then
    echo "error: Runpod REST ${method} ${path} failed (HTTP ${status}): ${out}" >&2
    return 1
  fi
  printf '%s' "$out"
}

# --- GraphQL helper (api.runpod.io/graphql): endpoint create/delete, the only
# API that exposes the load-balancing `type: "LB"` field. The documented auth
# form is the api_key query parameter.
graphql_request() {
  local query="$1" tmp status out body
  body="$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$query")"
  tmp="$(mktemp)"
  status="$(curl -sS -o "$tmp" -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' \
    --data "$body" \
    "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}")" || true
  out="$(cat "$tmp")"; rm -f "$tmp"
  if [[ "$status" != 2* ]]; then
    echo "error: Runpod GraphQL request failed (HTTP ${status}): ${out}" >&2
    return 1
  fi
  if printf '%s' "$out" | grep -q '"errors"'; then
    echo "error: Runpod GraphQL returned errors: ${out}" >&2
    return 1
  fi
  printf '%s' "$out"
}

json_get() {
  # $1 = dotted path (e.g. data.saveEndpoint.id), reads JSON on stdin
  python3 -c '
import json, sys
node = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    node = node[part]
print(node)
' "$1"
}

# --- tool checks -----------------------------------------------------------
if (( DO_BUILD || DO_PUSH )); then
  need docker
fi
need curl
need python3
if (( DO_ENDPOINT )); then
  [[ -n "$RUNPOD_API_KEY" ]] || die "RUNPOD_API_KEY must be set for any endpoint step"
fi
if (( DO_ENDPOINT && DO_BENCH )); then
  command -v /usr/bin/time >/dev/null 2>&1 \
    || die "/usr/bin/time (GNU time) not found — bench_cold_common.sh requires it"
fi

# --- image reference / build args -----------------------------------------
if [[ -n "${REFLEX_IMAGE_TAG:-}" ]]; then
  image_tag="$REFLEX_IMAGE_TAG"
elif (( SLIM )); then
  image_tag="runtime"
else
  image_tag="latest"
fi
image_ref="${REFLEX_REGISTRY_IMAGE}:${image_tag}"

build_args=(--build-arg "REFLEX_CUDA_ARCH=${REFLEX_CUDA_ARCH}")
if (( SLIM )); then
  build_args+=(
    --build-arg "REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-base-ubuntu22.04"
    --build-arg "REFLEX_RUNTIME_SLIM=1"
  )
fi

# --- build / push ----------------------------------------------------------
if (( DO_BUILD )); then
  run docker build -f serverless/runpod/Dockerfile "${build_args[@]}" -t "$image_ref" "$repo_root"
fi
if (( DO_PUSH )); then
  run docker push "$image_ref"
fi

if (( ! DO_ENDPOINT )); then
  echo "image: $image_ref" >&2
  echo "--no-endpoint given; skipping Runpod API steps." >&2
  exit 0
fi

# --- create template + load-balancing endpoint -----------------------------
TEMPLATE_ID=""
if [[ -z "$ENDPOINT_ID" ]]; then
  template_name="${REFLEX_ENDPOINT_NAME}-tpl-$(date +%Y%m%d%H%M%S)"
  template_payload="$(python3 - "$template_name" "$image_ref" "$REFLEX_CONTAINER_DISK_GB" "$RUNPOD_REGISTRY_AUTH_ID" <<'PY'
import json, sys
name, image, disk_gb, auth = sys.argv[1:5]
# GGUF_PATH is baked into the image at /models/model.gguf by
# serverless/runpod/Dockerfile; PORT must be declared both here and as the
# exposed port the endpoint's container configuration declares (Runpod does not
# infer it from the Dockerfile). The adapter serves the three-state health
# check at both /healthz and /ping.
body = {
    "name": name,
    "imageName": image,
    "isServerless": True,
    "containerDiskInGb": int(disk_gb),
    "ports": ["80/http"],
    "env": {"GGUF_PATH": "/models/model.gguf", "PORT": "80"},
}
if auth:
    body["containerRegistryAuthId"] = auth
print(json.dumps(body))
PY
)"

  echo "creating Serverless template '${template_name}' from ${image_ref}" >&2
  if (( DRY_RUN )); then
    echo "+ POST https://rest.runpod.io/v1/templates ${template_payload}"
    echo "(dry-run: stopping before endpoint creation)" >&2
    exit 0
  fi
  template_resp="$(rest_request POST /templates "$template_payload")"
  TEMPLATE_ID="$(printf '%s' "$template_resp" | json_get id)"

  # GraphQL saveEndpoint is how a load-balancing (type "LB") endpoint is
  # created; the pool id (gpuIds) is the only GPU lever available here — the
  # exact SKU is pinned immediately after with a REST PATCH.
  gql_query="$(python3 - "$REFLEX_ENDPOINT_NAME" "$TEMPLATE_ID" "$REFLEX_GPU_POOL" \
      "$REFLEX_WORKERS_MAX" "$REFLEX_IDLE_TIMEOUT" "$REFLEX_LOCATIONS" <<'PY'
import sys
name, tpl, pool, wmax, idle, locs = sys.argv[1:7]
parts = [
    f'name: "{name}"',
    f'templateId: "{tpl}"',
    'type: "LB"',
    f'gpuIds: "{pool}"',
    'workersMin: 0',
    f'workersMax: {int(wmax)}',
    f'idleTimeout: {int(idle)}',
    'scalerType: "QUEUE_DELAY"',
    'scalerValue: 4',
]
if locs:
    parts.append(f'locations: "{locs}"')
fields = ", ".join(parts)
print(
    "mutation { saveEndpoint(input: { " + fields + " }) "
    "{ id name gpuIds idleTimeout templateId workersMin workersMax } }"
)
PY
)"
  echo "creating load-balancing endpoint (pool ${REFLEX_GPU_POOL})" >&2
  endpoint_resp="$(graphql_request "$gql_query")"
  ENDPOINT_ID="$(printf '%s' "$endpoint_resp" | json_get data.saveEndpoint.id)"
else
  echo "reusing endpoint ${ENDPOINT_ID}" >&2
fi

# Post-create configuration, via REST: pin the exact GPU SKU (the whole point
# of this step — see the README's mixed-architecture warning) and set FlashBoot
# to the requested value. Both are documented EndpointUpdateInput fields.
patch_payload="$(python3 - "$REFLEX_GPU_TYPE_ID" "$REFLEX_FLASHBOOT" <<'PY'
import json, sys
sku, flash = sys.argv[1:3]
print(json.dumps({"gpuTypeIds": [sku], "flashboot": flash.lower() == "true"}))
PY
)"
echo "pinning SKU '${REFLEX_GPU_TYPE_ID}' and flashboot=${REFLEX_FLASHBOOT}" >&2
rest_request PATCH "/endpoints/${ENDPOINT_ID}" "$patch_payload" >/dev/null

endpoint_url="https://${ENDPOINT_ID}.api.runpod.ai"
echo >&2
echo "endpoint ready: ${ENDPOINT_ID}" >&2
echo "  worker URL : ${endpoint_url}" >&2
echo "  health      : ${endpoint_url}/ping  (200 ready / 204 loading / 503 dead)" >&2
echo "  template    : ${TEMPLATE_ID:-<reused>}" >&2

# --- benchmark -------------------------------------------------------------
if (( DO_BENCH )); then
  echo >&2
  echo "running cold-start benchmark (n=${BENCH_RUNS})..." >&2
  RUNPOD_IDLE_TIMEOUT="$REFLEX_IDLE_TIMEOUT" \
    "$script_dir/bench_cold_runpod.sh" "$ENDPOINT_ID" "$BENCH_RUNS"
fi

# --- teardown --------------------------------------------------------------
if (( DO_TEARDOWN )); then
  echo >&2
  echo "tearing down endpoint ${ENDPOINT_ID}..." >&2
  # Runpod requires workersMin/workersMax both 0 before an endpoint can be
  # deleted (documented GraphQL deleteEndpoint prerequisite).
  rest_request PATCH "/endpoints/${ENDPOINT_ID}" '{"workersMin":0,"workersMax":0}' >/dev/null || true
  sleep 5
  graphql_request "mutation { deleteEndpoint(id: \"${ENDPOINT_ID}\") }" >/dev/null || true
  if [[ -n "$TEMPLATE_ID" ]]; then
    rest_request DELETE "/templates/${TEMPLATE_ID}" >/dev/null || true
  fi
  echo "teardown requested. Verify in the console — Runpod's autoscaler has been" >&2
  echo "observed exceeding workersMax, so double-check no worker lingers." >&2
else
  echo >&2
  echo "NOTE: endpoint ${ENDPOINT_ID} is still live and billable. Tear it down with:" >&2
  echo "  scripts/deploy_runpod.sh --endpoint-id ${ENDPOINT_ID} --teardown --no-build --no-push --no-bench" >&2
fi
