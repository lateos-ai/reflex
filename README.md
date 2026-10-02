# Reflex

A GGUF-native Rust and CUDA inference engine built for **cold starts**: the time from
launching a process to its first result, not sustained server throughput. Every CUDA
kernel is compiled ahead of time by `nvcc` and shipped inside the binary, so nothing is
JIT-compiled when the process starts.

**~483 ms cold start, p50**: `reflex system1` from process start to its scored result
(the engine's own timer), Tesla T4, `Qwen3-0.6B-Q4_K_M`, `n=10`. Measured from outside
the process, including the ~110 ms the OS spends launching it, that is ~0.6 s wall clock.
[Phase breakdown](docs/benchmarks.md#cold-start-phase-breakdown).

> **Architectural boundary.** Reflex is a local execution engine for single-tenant
> decisions: one request at a time, no scheduler, no batching, no network server inside
> the engine. HTTP is a separate sidecar ([`sidecar/openai-adapter`](sidecar/openai-adapter/README.md))
> over local IPC. See [Non-goals](#non-goals) before proposing otherwise.

## Why cold start

Serverless GPU platforms bill per second of wall clock per invocation, so for bursty,
single-shot work (a tool call, a classification, an extraction) the time a worker takes
to become useful is the bill. Engines built for steady-state throughput pay for that
with long startups: vLLM captures CUDA graphs and compiles at launch. llama.cpp, like
Reflex, ships precompiled kernels and is the honest comparison point.

On a real Runpod serverless endpoint, Reflex's own load took ~1.6 s of a ~42-44 s cold
invocation; the rest is the platform provisioning the GPU and container, which no engine
avoids. Runpod's official vLLM worker took ~150 s on the same GPU tier the same day.
Against an official llama.cpp server image on the same platform, Reflex's engine load was
~0.5 s vs. ~0.95 s on an L4, but end-to-end the platform's own variance was far larger
than that gap ([details](docs/runpod-llamacpp-comparison.md#results)). The
full argument and measurements are in [docs/benchmarks.md](docs/benchmarks.md#where-this-engine-competes).

## Quickstart

Requires the CUDA toolkit (`nvcc` on `PATH`, or `CUDA_PATH`/`CUDA_HOME` set) and an
NVIDIA GPU. `REFLEX_SKIP_CUDA=1 cargo build` type-checks without CUDA (nothing will run).

```
git clone https://github.com/lateos-ai/reflex.git
cd reflex
cargo build --release

# Prove the AOT kernel pipeline works on your GPU:
cargo run --release --bin reflex -- smoke

# Score candidate answers in one forward pass:
cargo run --release --bin reflex -- system1 model.gguf "The capital of France is" \
  --candidate " Paris" --candidate " London"

# Generate text:
cargo run --release --bin reflex -- generate model.gguf "Once upon a time" --max-tokens 32
```

`reflex <subcommand>` with no arguments prints that subcommand's usage. All subcommands,
build options and output formats: [docs/reference.md](docs/reference.md).

## Supported models and limits

| Architecture (GGUF `general.architecture`) | Tested with |
|---|---|
| Dense Qwen3 (`qwen3`) | Qwen3-0.6B |
| Qwen3-MoE (`qwen3moe`) | synthetic fixtures (real `qwen3moe` tensor layout, top-k routing) |
| Llama / Mistral (`llama`) | TinyLlama-1.1B |
| Qwen3.5 hybrid Gated DeltaNet (`qwen35`, `qwen35moe`) | Qwen3.5-0.8B; a random-weight `qwen35moe` checkpoint |
| DeepSeek-V2/V3 MLA (`deepseek2`) | DeepSeek-V2-Lite; a synthetic fixture |

Each architecture counted as supported only after its generated tokens matched an
independent implementation (llama.cpp, or a CPU reference) on real hardware.

- **GGUF only.** Convert other checkpoints with llama.cpp's `convert_hf_to_gguf.py`.
- **Weights are held as `f32` on the GPU**, dequantized once at load: VRAM is about 4 bytes
  per parameter whatever the file's quantization.
- **Context**: up to 65,535 positions per sequence (prompt + generated tokens + any
  imported KV cache); longer requests fail with a clear `context_overflow` error.
- **`system1`** works on every architecture; on the Qwen3.5 hybrid models each candidate
  must be a single token.
- DeepSeek's Q-LoRA query compression and MTP heads are not supported.

## How it compares

Cold starts measured from outside the process, Tesla T4, `Qwen3-0.6B-Q4_K_M`, unless
noted. The engine-vs-engine rows time `reflex generate` to its first token (0.83 s p50,
`n=30`), the metric every engine shares; the headline above times `system1` scoring.

| vs. | Result | Caveat |
|---|---|---|
| llama.cpp `llama-simple` | Reflex **1.13x faster** (0.83 s vs. 0.94 s) | Both precompile their kernels; this is model-load work |
| llama.cpp `llama-cli` | Reflex **1.9x faster** (0.83 s vs. 1.58 s) | `llama-cli` always applies the chat template |
| Ollama | Reflex **2.6x faster** (0.83 s vs. ~2.1 s) | Cold daemon plus cold model |
| vLLM | Reflex **48–150x faster** (0.83 s vs. 39.8–127.3 s) | CUDA-graph capture and `torch.compile` at startup; ran from safetensors |
| vLLM on Runpod serverless | Reflex **~3.4x faster** (42–44 s vs. ~150 s) | Mostly platform provisioning; vLLM also downloaded `bf16` weights |
| llama.cpp server on Runpod serverless | Engine load **~1.9x faster** (0.49 s vs. 0.95 s on an L4, n=5 each); end-to-end inconclusive | Platform overhead (24–73 s even for Reflex) swamps the difference; [details](docs/runpod-llamacpp-comparison.md#results) |
| TypeSafe Jev (managed API) | Reflex **1.1–2.0x slower** (0.63 s vs. 0.31–0.57 s) | Jev is always warm; Reflex starts a process each time |

Methodology, per-run numbers and every caveat: [docs/benchmarks.md](docs/benchmarks.md).

**Energy.** Measured from outside the process with the GPU's idle draw subtracted, a cold
`reflex system1` on a T4 costs **~5.4 J**. Most of the ~28 J the GPU draws over that
window is idle power. [Details](docs/benchmarks.md#energy).

## Deploying

| Target | Where |
|---|---|
| Docker (one multi-target `Dockerfile`: `reflex`, `adapter`, `runpod-lb`) | [docs/reference.md](docs/reference.md#docker) |
| OpenAI-compatible HTTP (`/v1/chat/completions`) | [sidecar/openai-adapter](sidecar/openai-adapter/README.md) |
| AWS, scale to zero (Spot, local socket) | [docs/aws-deployment.md](docs/aws-deployment.md) |
| AWS, always warm (load balancer, HTTPS) | [docs/aws-deployment-warm.md](docs/aws-deployment-warm.md) |
| Runpod serverless, load-balancing endpoint | [serverless/runpod](serverless/runpod/README.md) |
| Runpod Hub, queue-based worker | [.runpod](.runpod/README.md) |
| Modal | [.modal](.modal/README.md) |
| Kubernetes (one Job per invocation) | [docs/reference.md](docs/reference.md#kubernetes) |
| Embedding in another program | C FFI (`include/reflex_engine.h`), local IPC (`reflex stdio`/`uds`) |

The Python bindings (`--features python`) do not currently build.

## Non-goals

These are permanent, not a backlog:

- **`batch_size` is always 1.** No request queue, no continuous batching, no
  PagedAttention-style allocation. Run more processes to serve more requests.
- **No network server in the engine, ever.** HTTP lives in the separate sidecar; the
  engine itself offers only sequential local IPC and in-process bindings.
- **No multi-tenant LoRA router, no internal KV-cache store.** Adapters load once at
  startup; where a KV-cache file lives is the caller's business.
- **No scheduler or autoscaling.** The platform (Kubernetes, a serverless provider, an
  autoscaling group) is the orchestrator; the engine stays a single-shot process.

The reasoning, and what the engine may grow instead:
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#the-full-rationale).

## Documentation

- [docs/reference.md](docs/reference.md): subcommands, build features and kernel modes,
  `--json` output, error categories, Docker, Kubernetes
- [docs/benchmarks.md](docs/benchmarks.md): every measurement, with method and caveats
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): building, testing, and the rules for changes
- [docs/gpu-ci.md](docs/gpu-ci.md): the nightly GPU workflow
- [CHANGELOG.md](CHANGELOG.md): what has been built, in order

## License and contributing

MIT; see [LICENSE](LICENSE). Contributions need a signed
[Contributor License Agreement](CLA.md); see [CONTRIBUTING.md](CONTRIBUTING.md).
