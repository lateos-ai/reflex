# Packages the same reflex + reflex-openai-adapter combination as the root
# Dockerfile's `runpod-lb` target, but as a Runpod Serverless **queue-based** worker
# (a Python handler using runpod.serverless.start()) instead of a load-balancing
# endpoint.
#
# This file must stay self-contained at this path (Runpod's Hub build pipeline builds
# `.runpod/Dockerfile` directly), so it can't reuse the root Dockerfile's stages. It
# mirrors them instead: same builder, same kernel build args and precedence, same slim
# runtime. Change them together.
#
# This is NOT a replacement for the root Dockerfile's `runpod-lb` target -- both stay
# valid, independent deployment paths. serverless/runpod/ remains the recommended way to
# deploy this manually (already proven, zero Python glue, see its own README for
# real measured cold-start numbers). This directory exists solely because Runpod's
# Hub publishing pipeline (the public, browsable catalog other people one-click-deploy
# from) only supports queue-based workers -- a load-balancing endpoint like
# serverless/runpod/ currently cannot be published to the Hub at all. See
# .runpod/README.md for the full reasoning.
#
# handler.py is a translation shim only: it spawns the same, unmodified
# reflex-openai-adapter binary as the `runpod-lb` image and forwards each
# Runpod job to that process's existing POST /v1/chat/completions over loopback.
# No engine logic is reimplemented in Python.
#
# Kernels default to a multi-arch **fatbin** (REFLEX_CUDA_ARCHS below), not a single
# pinned cubin: a Hub listing can't pin the GPU SKU. Runpod's Hub build/test
# infrastructure schedules jobs wherever it wants, and real-hardware evidence
# (2026-09-27, see .runpod/README.md) shows it does NOT reliably honor hub.json's
# gpuIds exclusions or tests.json's gpuTypeId pin: a test run landed on an RTX 2000
# Ada (sm_89) and an sm_86-pinned cubin panicked on it. This image used portable PTX
# to survive that, but PTX costs ~0.8 s of driver JIT on every fresh worker (measured
# on a T4; a new container has no JIT cache). The fatbin carries native images for
# Turing through Hopper, Ada included, and PTX that newer GPUs JIT, so it runs on any
# card the Hub picks with no JIT on the listed ones.
#
# Build context must be the repo root:
#
#   docker build -f .runpod/Dockerfile -t reflex-hub .
#
# Unlike the `runpod-lb` image, no PORT is exposed externally here: Runpod
# queue-based endpoints deliver jobs to the container's handler process directly,
# they don't proxy external HTTP to a configured port. reflex-openai-adapter binds
# 127.0.0.1:8000 (its own code default) for handler.py's exclusive, in-container use.

# Global build arg (declared before the first FROM so a FROM line can use it). The
# runtime is CUDA `base` + libcublas-12-4, not the full `-runtime-` image: see the root
# Dockerfile.
ARG REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-base-ubuntu22.04
# Compliance bundle (docs/compliance/README.md): 1 writes the SBOMs, third-party
# licenses, pip and apt package lists, base image names and the baked model's sha256
# to /usr/share/reflex/compliance/ (a few hundred KB; nothing runs at worker start).
# On by default here, unlike the root Dockerfile: the Hub's COMPLIANCE=on report job
# reads these files, and the Hub build takes no build args. 0 skips the extra stage.
ARG COMPLIANCE_BUNDLE=1

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# The toolchain is the one rust-toolchain.toml pins, installed before the rest of the
# source is copied so a source change doesn't reinstall it.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY rust-toolchain.toml ./
RUN rustup toolchain install
COPY . .

# Kernel build mode, same args and precedence as the root Dockerfile: a non-empty
# REFLEX_CUDA_ARCH (one cubin) wins; otherwise REFLEX_CUDA_ARCHS (fatbin, the
# default); both empty = portable PTX.
ARG REFLEX_CUDA_ARCH=""
ARG REFLEX_CUDA_ARCHS="sm_75,sm_80,sm_86,sm_89,sm_90"
RUN set -- --release --bin reflex --features ipc; \
    if [ -n "$REFLEX_CUDA_ARCH" ]; then \
        env -u REFLEX_CUDA_ARCHS REFLEX_CUDA_ARCH="$REFLEX_CUDA_ARCH" cargo build "$@"; \
    elif [ -n "$REFLEX_CUDA_ARCHS" ]; then \
        env -u REFLEX_CUDA_ARCH REFLEX_CUDA_ARCHS="$REFLEX_CUDA_ARCHS" cargo build "$@"; \
    else \
        env -u REFLEX_CUDA_ARCH -u REFLEX_CUDA_ARCHS cargo build "$@"; \
    fi

# Built in this same devel image, not a separate plain `rust:*` stage, to avoid a
# glibc-version skew against the ubuntu22.04 runtime.
RUN cd sidecar/openai-adapter && cargo build --release

# SBOMs and third-party licenses, read from the lockfiles (COMPLIANCE_BUNDLE=1 only;
# BuildKit skips this stage otherwise). Tool versions match compliance.yml.
FROM builder AS compliance-src
RUN cargo install --locked cargo-cyclonedx@0.5.9 \
    && cargo install --locked --features cli cargo-about@0.9.2 \
    && sh scripts/compliance-bundle.sh /compliance \
    && cp docs/compliance/data-handling.md /compliance/ \
    && sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1 > /compliance/reflex-engine-version \
    && rustc --version > /compliance/rustc-version

FROM ${REFLEX_RUNTIME_BASE} AS runtime

# The only CUDA library these binaries load besides the driver: cuBLAS
# (libcublas.so.12 + libcublasLt.so.12).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcublas-12-4 \
    && rm -rf /var/lib/apt/lists/*

# The only new runtime dependencies this whole addition introduces: a Python
# interpreter and the runpod SDK, needed solely to speak Runpod's queue job
# envelope (see handler.py), plus curl to fetch the model below. numpy is for the
# optional RAG mode's vector search (rag.py; unused unless RAG=on). Nothing else in
# this image needs Python. Versions pinned so the image (and its SBOM) can be
# rebuilt as is; numpy 2.2 is the last line that supports Ubuntu 22.04's Python 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir runpod==1.12.0 numpy==2.2.6

COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
COPY --from=builder /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter

# Fetched at *build* time (not COPYed from serverless/runpod/model.gguf like the
# `runpod-lb` target does, and not
# downloaded at container start) -- serverless/runpod/model.gguf is untracked
# (repo .gitignore excludes *.gguf with no exception for git, only for
# .dockerignore's local-build context), so it isn't present in a fresh clone of
# this repo, which is exactly what Runpod's Hub build pipeline operates on. A
# build-time fetch keeps the same "baked into the image, zero runtime/cold-start
# download" property the `runpod-lb` target's COPY achieves for a local build, without
# requiring a 639MB binary in git history (GitHub hard-rejects any pushed file over
# 100MB without Git LFS).
#
# Q8_0, not Q4_K_M: this listing leads with classification (see hub.json), and
# bit-width is the one thing a classifier sees that a chat model mostly hides. On a
# sub-3B model Q8_0 preserves the fine-grained logits `system1_evaluate` softmaxes
# over, where a Q4_K_M can move a near-tie label; classification is a single prefill
# with no generation, so the extra bandwidth costs one load, not per-token decode.
# The same repo:file pair is `src/hf.rs`'s verified `--quickstart` model
# (`QUICKSTART_REPO`/`QUICKSTART_FILE`, ~639MB, checked real by that module's own
# network test), so it is a known-resolvable URL, not a filename guess. If the model
# is ever changed, re-verify the new URL rather than trusting a filename match alone.
RUN mkdir -p /models && curl -fSL \
    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf" \
    -o /models/model.gguf
ENV GGUF_PATH=/models/model.gguf
# The baked model's id in the hub.json MODEL dropdown. Picking it (the default)
# serves the file above; any other pick, or GGUF_URL, makes handler.py download
# that GGUF once at worker start.
ENV BAKED_MODEL=Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf

COPY .runpod/handler.py .runpod/rag.py .runpod/compliance.py .runpod/models.json /

# Reported as the attestation's engine.image_digest when COMPLIANCE=on. An image
# can't know its own registry digest while it is being built, so whoever builds it
# passes what identifies the build (a release tag, a commit, a digest from an earlier
# build). Empty by default, reported as null.
ARG REFLEX_IMAGE_DIGEST=""
ENV REFLEX_IMAGE_DIGEST=${REFLEX_IMAGE_DIGEST}

ENTRYPOINT ["python3", "-u", "/handler.py"]

FROM runtime AS bundle-0

FROM runtime AS bundle-1
ARG REFLEX_RUNTIME_BASE
COPY --from=compliance-src /compliance /usr/share/reflex/compliance
# The baked model's sha256 (read by compliance.py instead of hashing 639 MB at
# start), and what the runtime image itself contains.
RUN cd /models && sha256sum model.gguf > model.gguf.sha256 \
    && cd /usr/share/reflex/compliance \
    && pip3 freeze --all > pip-freeze.txt \
    && dpkg-query -W > dpkg.txt \
    && REFLEX_RUNTIME_BASE="$REFLEX_RUNTIME_BASE" python3 -c 'import json, os; print(json.dumps({"reflex_engine_crate_version": open("reflex-engine-version").read().strip(), "rustc": open("rustc-version").read().strip(), "base_images": {"builder": "nvidia/cuda:12.4.1-devel-ubuntu22.04", "runtime": os.environ["REFLEX_RUNTIME_BASE"]}, "baked_model": os.environ["BAKED_MODEL"], "baked_model_sha256": open("/models/model.gguf.sha256").read().split()[0]}, indent=2))' > build.json \
    && rm reflex-engine-version rustc-version

FROM bundle-${COMPLIANCE_BUNDLE}
