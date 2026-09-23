# Multi-stage build. The builder stage has the full CUDA devel toolkit (nvcc) to
# compile this project's AOT kernels (see build.rs/CLAUDE.md's core technical bet);
# the runtime stage only needs the CUDA *runtime* libraries, since every kernel byte
# is embedded directly into the compiled binary at build time (see src/aot.rs's
# module doc comment for why -- the previous design, loading kernels from a
# build-time absolute filesystem path at runtime, would have silently broken here,
# since a naive `COPY --from=builder` of just the binary never carries that path
# along). The runtime image never runs nvcc and never needs the devel toolkit.
#
# Build (defaults to sm_86 -- pass --build-arg REFLEX_CUDA_ARCH=sm_XX for a
# different target compute capability, or --build-arg REFLEX_CUDA_ARCH= (empty)
# for the portable PTX build instead, which JITs to whatever GPU the container
# actually runs on):
#   docker build --build-arg REFLEX_CUDA_ARCH=sm_86 -t reflex .
#
# Run (needs nvidia-container-toolkit on the host; some rented-GPU platforms use
# a different flag, e.g. `--device nvidia.com/gpu=all`, instead of the standard
# `--gpus all` below -- check your platform's own GPU-passthrough convention if
# `--gpus all` doesn't work):
#   docker run --rm --gpus all reflex generate <path-to-gguf-inside-the-container> "prompt"
#
# The CUDA major/minor version in both stages' base images must stay consistent with
# Cargo.toml's pinned `cudarc` feature (`"cuda-12000"`, i.e. CUDA 12.x) -- a mismatch
# here is a build-time/runtime library version mismatch, not something this
# Dockerfile can catch for you.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY . .

# Empty by default: unset builds the portable PTX kernels (driver JITs to whatever
# GPU the container actually runs on -- the safer default for a distributed image,
# since it isn't pinned to one compute capability the way a cubin build is). Set
# --build-arg REFLEX_CUDA_ARCH=sm_86 (or your target) for the true
# zero-runtime-JIT cubin path instead.
ARG REFLEX_CUDA_ARCH=""

# Empty by default: no optional Cargo features enabled, matching this image's original
# footprint exactly. Set --build-arg REFLEX_FEATURES=ipc to additionally enable `reflex
# stdio`/`reflex uds` (see Cargo.toml's [features] table) -- needed for the AWS
# sidecar-over-UDS pattern in docs/aws-deployment.md. Comma-separate multiple features
# the same way Cargo does, e.g. REFLEX_FEATURES=ipc,download.
ARG REFLEX_FEATURES=""
RUN set -- --release --bin reflex; \
    if [ -n "$REFLEX_FEATURES" ]; then set -- "$@" --features "$REFLEX_FEATURES"; fi; \
    if [ -n "$REFLEX_CUDA_ARCH" ]; then \
        REFLEX_CUDA_ARCH=$REFLEX_CUDA_ARCH cargo build "$@"; \
    else \
        cargo build "$@"; \
    fi

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
ENTRYPOINT ["/usr/local/bin/reflex", "generate"]
