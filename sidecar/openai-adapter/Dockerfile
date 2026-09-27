# Packages the GPU-dependent `reflex` binary (with the `ipc` feature, for `reflex
# stdio`) together with this crate's CUDA-free `reflex-openai-adapter` binary into one
# image. They have to share one container: `reflex_client.rs` spawns `reflex stdio` as
# a literal OS child process of the adapter, so both binaries need to be in the same
# filesystem/process tree, and the container as a whole needs `--gpus all` for the
# `reflex` child to see the GPU.
#
# This is a separate Dockerfile from the repo root's `Dockerfile` (not a modification
# of it) deliberately: the root Dockerfile's `ENTRYPOINT ["/usr/local/bin/reflex",
# "generate"]` is depended on by the root README's Docker/Kubernetes sections and by
# docs/aws-deployment.md's `uds` invocation -- changing its entrypoint or bolting
# conditional stage logic onto it risks regressing those already-working, documented
# paths for no benefit. Build context must be the repo root (both `src/` and
# `sidecar/` are needed):
#
#   docker build -f sidecar/openai-adapter/Dockerfile \
#     --build-arg REFLEX_CUDA_ARCH=sm_75 \
#     -t reflex-openai-adapter .
#
# Run (needs nvidia-container-toolkit on the host; mount the gguf read-only and pass
# its in-container path as the adapter's positional arg -- `--host 0.0.0.0` is
# required here since the adapter's own default, 127.0.0.1, is correct for local use
# but wrong for a containerized target behind a load balancer):
#
#   docker run --rm --gpus all -p 8000:8000 \
#     -v /path/to/models:/models:ro \
#     reflex-openai-adapter /models/Qwen3-0.6B-Q8_0.gguf --host 0.0.0.0 --port 8000
#
# The CUDA major/minor version in both stages must stay consistent with Cargo.toml's
# pinned `cudarc` feature ("cuda-12000"), same rule as the root Dockerfile.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_PATH="/usr/local/cuda"

WORKDIR /build
COPY . .

# Empty by default (portable PTX); set --build-arg REFLEX_CUDA_ARCH=sm_XX for a
# zero-driver-JIT cubin build pinned to one compute capability -- see the root
# Dockerfile's own comment on this same tradeoff.
ARG REFLEX_CUDA_ARCH=""
RUN set -- --release --bin reflex --features ipc; \
    if [ -n "$REFLEX_CUDA_ARCH" ]; then \
        REFLEX_CUDA_ARCH=$REFLEX_CUDA_ARCH cargo build "$@"; \
    else \
        cargo build "$@"; \
    fi

# Built in this same devel image (not a second, plain `rust:*` builder stage)
# deliberately -- reusing the toolchain already present here avoids a glibc-version
# skew between a Debian-bookworm-built binary and this ubuntu22.04-based runtime
# stage. This crate itself needs no CUDA toolkit, only the toolchain.
RUN cd sidecar/openai-adapter && cargo build --release

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
COPY --from=builder /build/target/release/reflex /usr/local/bin/reflex
COPY --from=builder /build/sidecar/openai-adapter/target/release/reflex-openai-adapter /usr/local/bin/reflex-openai-adapter
ENTRYPOINT ["/usr/local/bin/reflex-openai-adapter"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
