# One multi-target Dockerfile for every image built from the repo root:
#
#   target      what it is                                          replaces
#   ----------  --------------------------------------------------  -------------------------------
#   reflex      the `reflex` CLI, entrypoint `reflex generate`       (the default; unchanged role)
#   adapter     `reflex` (ipc) + the OpenAI-compatible sidecar       sidecar/openai-adapter/Dockerfile
#   runpod-lb   `adapter` + a baked model, for a Runpod Serverless   serverless/runpod/Dockerfile
#               load-balancing endpoint
#
#   docker build -t reflex .                                   # target `reflex` (the last stage)
#   docker build --target adapter -t reflex-openai-adapter .
#   docker build --target runpod-lb -t reflex-runpod .         # needs serverless/runpod/model.gguf
#
# Build context is always the repo root. `.runpod/Dockerfile` (Runpod Hub) and
# `.modal/Dockerfile` (Modal) must stay self-contained files at those paths for their
# platforms' build pipelines; they mirror this file's builder and runtime stages and
# build args, so change them together.
#
# Needs BuildKit (Docker's default builder since 23.0), which builds only the stages a
# target needs; the legacy builder would also try `runpod-lb`'s model COPY.
#
# Run (needs nvidia-container-toolkit on the host; some rented-GPU platforms use
# `--device nvidia.com/gpu=all` instead of `--gpus all`):
#   docker run --rm --gpus all -v /path/to/models:/models reflex /models/model.gguf "prompt"
#   docker run --rm --gpus all -p 8000:8000 -v /path/to/models:/models:ro \
#     reflex-openai-adapter /models/model.gguf --host 0.0.0.0 --port 8000
#
# The CUDA major/minor version in the base images must stay consistent with
# Cargo.toml's pinned `cudarc` feature ("cuda-12000", i.e. CUDA 12.x).

# ---- Build args (global: declared before the first FROM so a FROM line can use them) --
#
# Runtime base. `base` + libcublas-12-4 (installed below) instead of the full
# `-runtime-` image: Reflex links only the CUDA driver API and cuBLAS, and the
# `-runtime-` image's cuda-libraries meta-package (NCCL, cuFFT, cuSPARSE, cuSOLVER,
# NPP, nvJPEG...) is dead weight that adds image-pull time to every cold start.
ARG REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-base-ubuntu22.04

# ---- Shared toolchain + source -------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS toolchain

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY . .

# Kernel build mode (see docs/reference.md's "Kernel build modes"; measurements in
# docs/benchmarks.md):
#   REFLEX_CUDA_ARCH=sm_XX      one cubin for exactly that GPU architecture. When set,
#                               it wins over REFLEX_CUDA_ARCHS.
#   REFLEX_CUDA_ARCHS=<list>    the default: a fatbin with a native image for each listed
#                               architecture (T4 through H100, including Ada) plus PTX
#                               that newer GPUs JIT. No runtime JIT on any listed GPU.
#   both empty                  portable PTX. Runs anywhere, but a fresh container has no
#                               driver JIT cache, so every cold start pays the JIT again
#                               (~0.8 s measured on a T4). Opt in with REFLEX_CUDA_ARCHS=.
ARG REFLEX_CUDA_ARCH=""
ARG REFLEX_CUDA_ARCHS="sm_75,sm_80,sm_86,sm_89,sm_90"

# Shared kernel-mode selection for the two engine builds below. `$@` is the cargo
# argument list; build.rs panics if both env vars are set, so only one is passed.
RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -e' \
    'if [ -n "$REFLEX_CUDA_ARCH" ]; then' \
    '    exec env -u REFLEX_CUDA_ARCHS REFLEX_CUDA_ARCH="$REFLEX_CUDA_ARCH" cargo build "$@"' \
    'elif [ -n "$REFLEX_CUDA_ARCHS" ]; then' \
    '    exec env -u REFLEX_CUDA_ARCH REFLEX_CUDA_ARCHS="$REFLEX_CUDA_ARCHS" cargo build "$@"' \
    'else' \
    '    exec env -u REFLEX_CUDA_ARCH -u REFLEX_CUDA_ARCHS cargo build "$@"' \
    'fi' > /usr/local/bin/build-reflex && chmod +x /usr/local/bin/build-reflex

# ---- `reflex` for the CLI image ------------------------------------------------------
FROM toolchain AS build-reflex
# Optional Cargo features for the CLI image, comma-separated as Cargo takes them. Empty
# by default (the core footprint). `ipc` adds `reflex stdio`/`reflex uds`, needed by
# docs/aws-deployment.md's sidecar-over-UDS pattern.
ARG REFLEX_FEATURES=""
RUN set -- --release --bin reflex; \
    if [ -n "$REFLEX_FEATURES" ]; then set -- "$@" --features "$REFLEX_FEATURES"; fi; \
    build-reflex "$@"

# ---- `reflex` (ipc) + the sidecar for the adapter images -----------------------------
FROM toolchain AS build-adapter
RUN build-reflex --release --bin reflex --features ipc
# Built in this same devel image, not a separate plain `rust:*` stage, to avoid a
# glibc-version skew against the ubuntu22.04 runtime. The sidecar needs no CUDA.
RUN cd sidecar/openai-adapter && cargo build --release

# ---- Runtime base shared by every target ---------------------------------------------
FROM ${REFLEX_RUNTIME_BASE} AS runtime
# libcublas-12-4 brings libcublas.so.12 and libcublasLt.so.12. On a `-runtime-` base
# override this is already present and the install is a no-op.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcublas-12-4 \
    && rm -rf /var/lib/apt/lists/*

# ---- target: adapter -----------------------------------------------------------------
FROM runtime AS adapter
COPY --from=build-adapter /build/target/release/reflex /usr/local/bin/reflex
COPY --from=build-adapter /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter
# The adapter spawns `reflex stdio` as its own child process, so both binaries share
# this container, and the container needs `--gpus all`. `--host 0.0.0.0` because the
# adapter's own default (127.0.0.1) is wrong behind a load balancer.
ENTRYPOINT ["/usr/local/bin/reflex-openai-adapter"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

# ---- target: runpod-lb ---------------------------------------------------------------
# Runpod Serverless *load-balancing* endpoint: Runpod proxies HTTP straight to the
# sidecar, so no handler code is needed. See serverless/runpod/README.md.
FROM adapter AS runpod-lb
# Baked into the image rather than fetched at cold start, which would put a Hugging Face
# download inside the cold-start number this deployment exists to measure. The repo's
# .dockerignore excludes *.gguf except this one file. Override GGUF_PATH to use a
# Network Volume instead.
COPY serverless/runpod/model.gguf /models/model.gguf
ENV GGUF_PATH=/models/model.gguf
# Runpod's default port for load-balancing endpoints; override with a PORT env var.
ENV PORT=80
# Extra reflex-openai-adapter flags, space-separated (e.g. "--max-tokens-cap 1024").
# Empty = the adapter's request-limit defaults (see sidecar/openai-adapter/README.md's
# "Request limits"). Unquoted below on purpose so it splits into separate arguments.
ENV ADAPTER_ARGS=""
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/reflex-openai-adapter \"$GGUF_PATH\" --host 0.0.0.0 --port \"$PORT\" $ADAPTER_ARGS"]
CMD []

# ---- target: reflex (default: keep this stage last) ----------------------------------
FROM runtime AS reflex
COPY --from=build-reflex /build/target/release/reflex /usr/local/bin/reflex
ENTRYPOINT ["/usr/local/bin/reflex", "generate"]
