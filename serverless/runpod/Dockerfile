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
# compute capability it was built for (see root CLAUDE.md's "Core technical bet").
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
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/reflex-openai-adapter \"$GGUF_PATH\" --host 0.0.0.0 --port \"$PORT\""]
