# llama.cpp's official CUDA server, packaged for the same Runpod Serverless
# load-balancing endpoint as the root Dockerfile's `runpod-lb` target, so the two
# engines' cold starts can be compared like for like. See
# docs/runpod-llamacpp-comparison.md for the procedure.
#
# Kept deliberately close to the Reflex image:
#   - the identical model file, baked in the same way (serverless/runpod/model.gguf,
#     Qwen3-0.6B Q4_K_M; the repo's .dockerignore already allows exactly that file);
#   - the same environment contract (GGUF_PATH, PORT), so scripts/deploy_runpod.sh
#     and scripts/bench_cold_runpod.sh drive it unchanged;
#   - the same health contract on `/ping` (200 ready, 204 still loading), which
#     Runpod's load balancer polls regardless of its documented health-path setting.
#     llama-server has no `/ping` (only `/health`, which returns 503 while the model
#     loads), so nginx maps one onto the other and proxies everything else.
#
# Build from the repo root:
#   docker build -f serverless/runpod-llamacpp/Dockerfile -t llamacpp-runpod .

# Pinned build so the comparison is reproducible: llama.cpp b11277, published
# 2026-09-30, CUDA 12.8.1 (so workers need a driver that supports CUDA 12.8). Its CUDA
# library has native code only for sm_86, sm_89 and sm_120a; on any other GPU (T4, A100,
# H100...) the driver JIT-compiles it on every fresh container, which took tens of
# seconds on a T4. Compare on an sm_86/sm_89 card (the default RTX A4500 is sm_86).
ARG LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b11277
FROM ${LLAMACPP_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

COPY serverless/runpod/model.gguf /models/model.gguf
COPY serverless/runpod-llamacpp/start.sh /start.sh
RUN chmod +x /start.sh

ENV GGUF_PATH=/models/model.gguf
ENV PORT=80
# llama-server flags. The defaults match Reflex's serving model: all layers on the
# GPU, one sequence at a time, a 4096-token context (llama-server would otherwise
# size the KV cache for the model's full 40,960-token training context at startup).
# llama-server's own warmup run stays on, as a default deployment would have it; set
# LLAMA_ARGS to add `--no-warmup` for the variant described in the comparison doc.
ENV LLAMA_ARGS="-ngl 99 -c 4096 -np 1"

ENTRYPOINT ["/start.sh"]
