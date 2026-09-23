#!/usr/bin/env bash
# scripts/aws_ec2_bootstrap.sh — EC2 instance-launch (user-data) bootstrap for the
# Spot + scale-to-zero + UDS-sidecar deployment pattern documented in
# docs/aws-deployment.md. Read that guide first — this script is the host-side half
# of it; the account-level setup (Launch Template, ASG, IAM policy, ECR push) is
# copy-paste `aws` CLI in the markdown, not automated here, since those are
# one-time/operator actions, not something a freshly-booted instance should redo.
#
# WHY this exists instead of baking everything into the Docker image: the model
# weights and the EFS mount are per-deployment state (which model, which cache
# volume), not build-time constants of the `reflex` image itself — see the
# Dockerfile's own doc comment on REFLEX_CUDA_ARCH/REFLEX_FEATURES for the same
# build-time-vs-runtime split applied to the image.
#
# WHY plain curl for the model download, not `reflex generate --model`/`--quickstart`
# (the in-Rust hf-hub integration, Cargo `download` feature): `reflex uds` (the
# subcommand this script actually runs) takes a local GGUF file path only and has no
# HF-hub download support of its own (see src/bin/reflex/uds.rs) — the download has
# to happen before `uds` is invoked, not via a flag on it. Doing that with curl here
# keeps the sidecar image itself free of the `download` feature and any HF-hub/Python
# runtime dependency, consistent with this project's own avoid-heavy-runtime ethos.
#
# WHY docker-ce's own apt repo and NVIDIA Container Toolkit's own apt repo, not the
# distro `docker.io` package: `docker.io` on Ubuntu 22.04 is an older Docker release
# and does not pull in `nvidia-ctk` — `nvidia-ctk` ships from the
# `nvidia-container-toolkit` package in NVIDIA's own repo, not Ubuntu's.
#
# Idempotent: safe to re-run (e.g. if user-data re-executes on instance stop/start) —
# every step below checks for existing state before mutating it.
#
# Required environment variables (set via the Launch Template's user-data, not
# hardcoded here):
#   EFS_ID       - EFS filesystem id, e.g. fs-0123456789abcdef0
#   ECR_IMAGE    - full ECR image URI to pull, e.g.
#                  123456789012.dkr.ecr.us-east-1.amazonaws.com/reflex:ipc
#   MODEL_REPO   - Hugging Face repo id, e.g. Qwen/Qwen3-0.6B-GGUF
#   MODEL_FILE   - filename within that repo, e.g. Qwen3-0.6B-Q8_0.gguf
# Optional:
#   HF_TOKEN     - bearer token for gated/private HF repos (leave unset for public
#                  repos; see docs/aws-deployment.md for why this must come from SSM/
#                  Secrets Manager into the environment, never plaintext in user-data)
#
# Usage (as EC2 user-data, run as root):
#   EFS_ID=fs-... ECR_IMAGE=... MODEL_REPO=... MODEL_FILE=... scripts/aws_ec2_bootstrap.sh

set -euo pipefail

efs_id="${EFS_ID:?usage: EFS_ID=fs-... ECR_IMAGE=... MODEL_REPO=... MODEL_FILE=... $0}"
ecr_image="${ECR_IMAGE:?usage: EFS_ID=fs-... ECR_IMAGE=... MODEL_REPO=... MODEL_FILE=... $0}"
model_repo="${MODEL_REPO:?usage: EFS_ID=fs-... ECR_IMAGE=... MODEL_REPO=... MODEL_FILE=... $0}"
model_file="${MODEL_FILE:?usage: EFS_ID=fs-... ECR_IMAGE=... MODEL_REPO=... MODEL_FILE=... $0}"
hf_token="${HF_TOKEN:-}"

cache_dir="/mnt/reflex-cache"
models_dir="$cache_dir/models"
socket_dir="/tmp/reflex-ipc"
socket_path="$socket_dir/reflex.sock"

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

echo "[bootstrap] pulling sidecar image..."
mkdir -p "$socket_dir"
docker pull "$ecr_image"

echo "[bootstrap] starting reflex uds sidecar..."
docker rm -f reflex-engine >/dev/null 2>&1 || true
docker run -d \
  --name reflex-engine \
  --restart=unless-stopped \
  --gpus all \
  --entrypoint /usr/local/bin/reflex \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  -v "$models_dir:/models:ro" \
  -v "$socket_dir:$socket_dir" \
  "$ecr_image" \
  uds "/models/$model_file" "$socket_path"

echo "[bootstrap] waiting for socket at $socket_path..."
for _ in $(seq 1 60); do
  if [ -S "$socket_path" ]; then
    echo "[bootstrap] reflex-engine is up, socket ready at $socket_path"
    exit 0
  fi
  sleep 1
done

echo "[bootstrap] ERROR: $socket_path did not appear within 60s -- check 'docker logs reflex-engine'" >&2
exit 1
