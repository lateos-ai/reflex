#!/usr/bin/env bash
# scripts/aws_ec2_bootstrap_warm.sh — EC2 instance-launch (user-data) bootstrap for the
# always-warm ASG/ALB deployment pattern documented in docs/aws-deployment-warm.md.
# Read that guide first. This is a sibling of scripts/aws_ec2_bootstrap.sh (the
# scale-to-zero/local-UDS-only pattern in docs/aws-deployment.md), not a replacement
# for it — the two patterns coexist. This script differs from that one in exactly the
# ways the two deployment patterns differ:
#   - runs the combined reflex+adapter image (root Dockerfile, --target adapter) with
#     `reflex-openai-adapter` as the entrypoint, bound to a real TCP port, instead of
#     bare `reflex uds` bound to a local-only Unix Domain Socket;
#   - readiness is polled over HTTP (`/healthz`) instead of waiting for a socket file
#     to appear;
#   - the instance is meant to sit behind an ALB and receive real traffic on that
#     port, so its security group must allow inbound only from the ALB's security
#     group, never 0.0.0.0/0 — this script does not and cannot configure security
#     groups itself (that's account-level setup, done once via the AWS CLI/console
#     alongside the Launch Template — see the deployment doc).
#
# Docker/NVIDIA Container Toolkit install and the EFS model-cache mount are reused
# verbatim from scripts/aws_ec2_bootstrap.sh — see that script's own comments for why
# docker-ce's own apt repo (not docker.io) and NVIDIA Container Toolkit's own apt repo
# are used, and why curl (not the Cargo `download` feature) fetches the model.
#
# Idempotent: safe to re-run (e.g. if user-data re-executes on instance stop/start) —
# every step below checks for existing state before mutating it.
#
# Required environment variables (set via the Launch Template's user-data):
#   ECR_IMAGE    - full ECR image URI for the *combined* image built from
#                  the root Dockerfile's `adapter` target, e.g.
#                  123456789012.dkr.ecr.us-east-1.amazonaws.com/reflex-openai-adapter:latest
#   EFS_ID       - EFS filesystem id, e.g. fs-0123456789abcdef0
#   MODEL_REPO   - Hugging Face repo id, e.g. Qwen/Qwen3-0.6B-GGUF
#   MODEL_FILE   - filename within that repo, e.g. Qwen3-0.6B-Q8_0.gguf
# Optional:
#   PORT         - TCP port the adapter listens on and the ALB target group forwards
#                  to (default 8000)
#   ADAPTER_ARGS - extra reflex-openai-adapter flags, space-separated, e.g.
#                  "--max-tokens-cap 1024 --request-timeout-secs 55". Unset = the
#                  adapter's own request-limit defaults (see its README's "Request
#                  limits"), which already apply without this.
#   HF_TOKEN     - bearer token for gated/private HF repos (leave unset for public
#                  repos; see docs/aws-deployment-warm.md for why this must come from
#                  SSM/Secrets Manager into the environment, never plaintext in
#                  user-data)
#
# Usage (as EC2 user-data, run as root):
#   ECR_IMAGE=... EFS_ID=fs-... MODEL_REPO=... MODEL_FILE=... scripts/aws_ec2_bootstrap_warm.sh

set -euo pipefail

ecr_image="${ECR_IMAGE:?usage: ECR_IMAGE=... EFS_ID=fs-... MODEL_REPO=... MODEL_FILE=... $0}"
efs_id="${EFS_ID:?usage: ECR_IMAGE=... EFS_ID=fs-... MODEL_REPO=... MODEL_FILE=... $0}"
model_repo="${MODEL_REPO:?usage: ECR_IMAGE=... EFS_ID=fs-... MODEL_REPO=... MODEL_FILE=... $0}"
model_file="${MODEL_FILE:?usage: ECR_IMAGE=... EFS_ID=fs-... MODEL_REPO=... MODEL_FILE=... $0}"
port="${PORT:-8000}"
# Word-split on purpose: one flag/value per word (no values containing spaces).
read -r -a adapter_args <<<"${ADAPTER_ARGS:-}"
hf_token="${HF_TOKEN:-}"

cache_dir="/mnt/reflex-cache"
models_dir="$cache_dir/models"

echo "[bootstrap] installing Docker (docker-ce apt repo)..."
if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io
  systemctl enable --now docker
else
  echo "[bootstrap] docker already installed, skipping"
fi

echo "[bootstrap] installing NVIDIA Container Toolkit (its own apt repo)..."
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo "[bootstrap] nvidia-container-toolkit already installed, skipping"
fi

echo "[bootstrap] mounting EFS cache..."
if ! command -v mount.efs >/dev/null 2>&1; then
  apt-get install -y git binutils rustc cargo pkg-config libssl-dev
  git clone --depth 1 https://github.com/aws/efs-utils /tmp/efs-utils
  (cd /tmp/efs-utils && ./build-deb.sh && apt-get install -y ./build/amazon-efs-utils*.deb)
fi
mkdir -p "$cache_dir"
if ! mountpoint -q "$cache_dir"; then
  mount -t efs -o tls "${efs_id}:/" "$cache_dir"
fi
mkdir -p "$models_dir"

echo "[bootstrap] fetching model (skip if already cached)..."
model_path="$models_dir/$model_file"
if [ ! -s "$model_path" ]; then
  curl_auth=()
  if [ -n "$hf_token" ]; then
    curl_auth=(-H "Authorization: Bearer $hf_token")
  fi
  curl -L -C - "${curl_auth[@]}" \
    -o "$model_path.part" \
    "https://huggingface.co/${model_repo}/resolve/main/${model_file}"
  mv "$model_path.part" "$model_path"
else
  echo "[bootstrap] $model_path already present, skipping download"
fi

echo "[bootstrap] pulling combined reflex+adapter image..."
docker pull "$ecr_image"

echo "[bootstrap] starting reflex-openai-adapter..."
docker rm -f reflex-engine >/dev/null 2>&1 || true
docker run -d \
  --name reflex-engine \
  --restart=unless-stopped \
  --gpus all \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  -p "${port}:${port}" \
  -v "$models_dir:/models:ro" \
  "$ecr_image" \
  "/models/$model_file" --host 0.0.0.0 --port "$port" "${adapter_args[@]}"

echo "[bootstrap] waiting for http://127.0.0.1:$port/healthz..."
# Require exactly HTTP 200, not just "no error" -- the sidecar returns 204 (not an
# error status `curl -f` would catch) while the reflex child is alive but still
# loading the model, and only 200 once it's actually ready to serve a request. A
# plain `curl -sf` would report success on that 204 and this script would exit early
# while the model is still loading.
for _ in $(seq 1 120); do
  status=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/healthz" 2>/dev/null || echo "000")
  if [ "$status" = "200" ]; then
    echo "[bootstrap] reflex-engine is up and healthy on port $port"
    exit 0
  fi
  sleep 1
done

echo "[bootstrap] ERROR: http://127.0.0.1:$port/healthz did not become healthy within 120s -- check 'docker logs reflex-engine'" >&2
exit 1
