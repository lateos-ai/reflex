# Packages the same reflex + reflex-openai-adapter combination as
# serverless/runpod/Dockerfile, but as a Runpod Serverless **queue-based** worker
# (a Python handler using runpod.serverless.start()) instead of a load-balancing
# endpoint.
#
# This is NOT a replacement for serverless/runpod/Dockerfile -- both stay valid,
# independent deployment paths. serverless/runpod/ remains the recommended way to
# deploy this manually (already proven, zero Python glue, see its own README for
# real measured cold-start numbers). This directory exists solely because Runpod's
# Hub publishing pipeline (the public, browsable catalog other people one-click-deploy
# from) only supports queue-based workers -- a load-balancing endpoint like
# serverless/runpod/ currently cannot be published to the Hub at all. See
# .runpod/README.md for the full reasoning.
#
# handler.py is a translation shim only: it spawns the same, unmodified
# reflex-openai-adapter binary as serverless/runpod/Dockerfile and forwards each
# Runpod job to that process's existing POST /v1/chat/completions over loopback.
# No engine logic is reimplemented in Python.
#
# REFLEX_CUDA_ARCH defaults to sm_86 here, matching serverless/runpod/Dockerfile --
# Runpod's cheapest serverless tier (AMPERE_16) is Ampere (compute capability 8.6).
# See serverless/runpod/README.md's GPU-selection section: AMPERE_16 is
# mixed-architecture (also contains sm_89 Ada cards an sm_86 cubin cannot run on) --
# .runpod/hub.json's gpuIds pins away from those Ada SKUs for the same reason
# serverless/runpod's endpoint is pinned to RTX A4500 via set-endpoint-gpus.
#
# Build context must be the repo root:
#
#   docker build -f .runpod/Dockerfile --build-arg REFLEX_CUDA_ARCH=sm_86 -t reflex-hub .
#
# Unlike serverless/runpod/Dockerfile, no PORT is exposed externally here: Runpod
# queue-based endpoints deliver jobs to the container's handler process directly,
# they don't proxy external HTTP to a configured port. reflex-openai-adapter binds
# 127.0.0.1:8000 (its own code default) for handler.py's exclusive, in-container use.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY . .

ARG REFLEX_CUDA_ARCH="sm_86"
RUN set -- --release --bin reflex --features ipc; \
    if [ -n "$REFLEX_CUDA_ARCH" ]; then \
        REFLEX_CUDA_ARCH=$REFLEX_CUDA_ARCH cargo build "$@"; \
    else \
        cargo build "$@"; \
    fi

# Built in this same devel image deliberately -- see
# sidecar/openai-adapter/Dockerfile's comment on avoiding a glibc-version skew
# against a separate plain-Rust builder base image.
RUN cd sidecar/openai-adapter && cargo build --release

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# The only new runtime dependencies this whole addition introduces: a Python
# interpreter and the runpod SDK, needed solely to speak Runpod's queue job
# envelope (see handler.py), plus curl to fetch the model below. Nothing else in
# this image needs Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir runpod

COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
COPY --from=builder /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter

# Fetched at *build* time (not COPYed from serverless/runpod/model.gguf, and not
# downloaded at container start) -- serverless/runpod/model.gguf is untracked
# (repo .gitignore excludes *.gguf with no exception for git, only for
# .dockerignore's local-build context), so it isn't present in a fresh clone of
# this repo, which is exactly what Runpod's Hub build pipeline operates on. A
# build-time fetch keeps the same "baked into the image, zero runtime/cold-start
# download" property serverless/runpod/'s COPY achieves for a local build, without
# requiring a 379MB binary in git history (GitHub hard-rejects any pushed file over
# 100MB without Git LFS). This exact URL was verified byte-for-byte identical to
# serverless/runpod/model.gguf (same size, same leading bytes) before being used
# here -- if the model is ever changed, re-verify a new URL the same way rather
# than trusting a filename match alone.
RUN mkdir -p /models && curl -fSL \
    "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf" \
    -o /models/model.gguf
ENV GGUF_PATH=/models/model.gguf

COPY .runpod/handler.py /handler.py

ENTRYPOINT ["python3", "-u", "/handler.py"]
