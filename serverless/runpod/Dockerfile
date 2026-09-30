# Packages the same reflex + reflex-openai-adapter combination as
# sidecar/openai-adapter/Dockerfile, for deployment as a Runpod Serverless
# **load-balancing endpoint** (not a queue-based endpoint -- see this directory's
# README for why that distinction matters and why it means zero new handler code is
# needed: Runpod's load-balancing endpoint type proxies HTTP directly to any custom
# server listening on a configured port, so the existing OpenAI-compatible sidecar
# binary runs completely unmodified here).
#
# The only real difference from sidecar/openai-adapter/Dockerfile: REFLEX_CUDA_ARCH
# defaults to sm_86 here, not sm_75 -- Runpod's cheapest serverless tier is Ampere
# (compute capability 8.6), not the T4 (7.5) the AWS deployment guides target. Do not
# reuse a sm_75 image on Runpod or vice versa -- a pinned cubin build only runs on the
# compute capability it was built for (see docs/DEVELOPMENT.md's "Core technical bet").
#
# CRITICAL, verified against the live Runpod catalog (not assumed): the cheapest
# serverless pool `AMPERE_16` is **mixed-architecture despite its name**. It contains
# RTX A4000/A4500 (Ampere, sm_86) *and* RTX 2000 Ada / RTX 4000 Ada (Ada, sm_89). An
# sm_86 cubin will NOT run on the Ada cards, so an endpoint left free to pick any SKU
# in that pool fails nondeterministically depending on where a worker lands. Pin the
# SKU (`set-endpoint-gpus`) to **RTX A4500** -- sm_86, 20GB, and the only card in that
# pool with HIGH serverless availability. See this directory's README.
#
# Build context must be the repo root:
#
#   docker build -f serverless/runpod/Dockerfile \
#     --build-arg REFLEX_CUDA_ARCH=sm_86 \
#     -t reflex-runpod .
#
# GGUF_PATH and PORT are read from environment variables at container start (not CLI
# args), since Runpod Serverless endpoints are configured via environment variables,
# not a custom command line -- see this directory's README for the exact Runpod
# endpoint configuration this expects.

# Global build arg -- declared before the first FROM so it can be used in a FROM
# line. Defaults to the full CUDA `-runtime-` base this deployment has always
# shipped with; scripts/deploy_runpod.sh can opt into the measured slim variant
# (base + libcublas-12-4) with --slim. See the runtime stage below and the root
# README's "Container image size is part of cold start here".
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

ARG REFLEX_CUDA_ARCH="sm_86"
# Multi-arch fatbin (comma-separated sm_XX list + embedded PTX fallback), mutually
# exclusive with REFLEX_CUDA_ARCH. A fatbin covering sm_80,sm_86,sm_89,sm_90 would
# retire the AMPERE_16 mixed-arch-pool SKU-pinning workaround documented above.
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

# Opt-in runtime slimming, default no-op. On a genuinely cold Runpod worker the
# image pull precedes everything, and the standard `-runtime-` base is mostly
# libraries this engine never loads (NCCL, cuFFT, cuSPARSE, cuSOLVER, NPP,
# nvJPEG from the cuda-libraries-12-4 meta-package). Reflex links only the CUDA
# driver API + cuBLAS, so with
#   --build-arg REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-base-ubuntu22.04 \
#   --build-arg REFLEX_RUNTIME_SLIM=1
# only libcublas-12-4 is installed on the smaller base. Measured: 3.78GB full
# base -> 1.28GB slim (cuBLAS 553MB of the remainder). Build/push is driven by
# scripts/deploy_runpod.sh's --slim flag; the default is unchanged so existing
# deployments do not silently switch base images.
ARG REFLEX_RUNTIME_SLIM=""
RUN if [ -n "$REFLEX_RUNTIME_SLIM" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libcublas-12-4 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
COPY --from=builder /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter

# Baked directly into the image (~379MB, Qwen3-0.6B-Q4_K_M) rather than fetched at
# cold start or mounted from a Network Volume -- a runtime download would add
# HuggingFace-fetch latency straight into the cold-start number this deployment
# exists to measure, corrupting the benchmark. See serverless/runpod/README.md and
# the repo root .dockerignore's narrow exception for this one file. Swap models by
# rebuilding the image, or override GGUF_PATH to point at a Network Volume mount
# instead if you need to change models without a rebuild.
COPY serverless/runpod/model.gguf /models/model.gguf
ENV GGUF_PATH=/models/model.gguf

# PORT defaults to 80, matching Runpod's own stated default for load-balancing
# endpoints; override by setting a PORT environment variable on the endpoint if a
# different port is exposed.
ENV PORT=80
# Extra reflex-openai-adapter flags, space-separated (e.g. "--max-tokens-cap 1024").
# Empty = the adapter's own request-limit defaults (max_tokens <= 2048, 16 queued
# jobs, 300 s timeout -- see sidecar/openai-adapter/README.md's "Request limits").
# Deliberately unquoted below so it word-splits into separate arguments.
ENV ADAPTER_ARGS=""
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/reflex-openai-adapter \"$GGUF_PATH\" --host 0.0.0.0 --port \"$PORT\" $ADAPTER_ARGS"]
