# Packages the same reflex + reflex-openai-adapter combination as
# serverless/runpod/Dockerfile and sidecar/openai-adapter/Dockerfile, this time for
# deployment as a Modal Server (see app.py, which uses `Image.from_dockerfile` against
# this file rather than modal.Image's Python builder methods -- reusing this exact,
# already-proven build recipe instead of re-deriving an equivalent one keeps all three
# deployments' build behavior in lockstep).
#
# REFLEX_CUDA_ARCH defaults to sm_89 here (Ada Lovelace) -- unlike
# serverless/runpod/Dockerfile's sm_86 default and .runpod/Dockerfile's portable-PTX
# default, both driven by *mixed-architecture GPU pools* neither deployment can fully
# pin. Modal's `gpu="L4"` parameter in app.py requests one exact SKU (NVIDIA L4, Ada,
# compute capability 8.9) with no pool-mixing hazard, so there is no reason to give up
# the zero-driver-JIT cold-start advantage a pinned cubin gets. If app.py's GPU choice
# ever changes to a different architecture, this default must change with it.
#
# Build context must be the repo root -- app.py passes context_dir accordingly, so
# `modal deploy .modal/app.py` / `modal run .modal/app.py` from the repo root is the
# only supported invocation (same build-context requirement every other Dockerfile in
# this project already has).

# Global build arg -- declared before the first FROM so it can be used in a FROM
# line. Defaults to the full CUDA `-runtime-` base (unchanged); the slim variant
# is opt-in via a base-image override plus REFLEX_RUNTIME_SLIM=1. See the
# runtime stage below and the root README's "Container image size is part of
# cold start here".
ARG REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY . .

ARG REFLEX_CUDA_ARCH="sm_89"
# Multi-arch fatbin (comma-separated sm_XX list + embedded PTX fallback), mutually
# exclusive with REFLEX_CUDA_ARCH -- see build.rs / README's fatbin section.
ARG REFLEX_CUDA_ARCHS=""
RUN set -- --release --bin reflex --features ipc; \
    if [ -n "$REFLEX_CUDA_ARCHS" ]; then \
        unset REFLEX_CUDA_ARCH; \
        REFLEX_CUDA_ARCHS=$REFLEX_CUDA_ARCHS cargo build "$@"; \
    elif [ -n "$REFLEX_CUDA_ARCH" ]; then \
        REFLEX_CUDA_ARCH=$REFLEX_CUDA_ARCH cargo build "$@"; \
    else \
        cargo build "$@"; \
    fi

# Built in this same devel image deliberately -- see
# sidecar/openai-adapter/Dockerfile's comment on avoiding a glibc-version skew
# against a separate plain-Rust builder base image.
RUN cd sidecar/openai-adapter && cargo build --release

FROM ${REFLEX_RUNTIME_BASE} AS runtime

# Opt-in runtime slimming, default no-op. Modal's own container init installs
# the Python interpreter this file's app.py needs, so the only CUDA runtime
# library these binaries actually link is cuBLAS; when REFLEX_RUNTIME_SLIM is
# set (with a base-image override to nvidia/cuda:12.4.1-base-ubuntu22.04) just
# libcublas-12-4 is installed. Measured: 3.78GB full base -> 1.28GB slim (cuBLAS
# 553MB of the remainder). See the root README's "Container image size is part
# of cold start here". Default base unchanged.
ARG REFLEX_RUNTIME_SLIM=""
RUN if [ -n "$REFLEX_RUNTIME_SLIM" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libcublas-12-4 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
COPY --from=builder /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter

# Same file, same rationale as serverless/runpod/Dockerfile: baked in at build time so
# a runtime HuggingFace download never pollutes the cold-start number this deployment
# exists to measure. Not duplicated under .modal/ -- one baked model artifact shared
# by both deployments that already bake one in.
COPY serverless/runpod/model.gguf /models/model.gguf

# No ENTRYPOINT/CMD: unlike a plain `docker run` target, a Modal Server's container
# process is started by app.py's `@modal.enter()` method via subprocess.Popen, not by
# the image's own entrypoint -- Modal's container init supersedes it regardless of what
# this Dockerfile sets, so declaring one here would be dead, misleading configuration.
