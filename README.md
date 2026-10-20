# Reflex

A GGUF-native Rust and CUDA inference engine built for **cold starts**: the time from
launching a process to its first result, not sustained server throughput. Every CUDA
kernel is compiled ahead of time by `nvcc` and shipped inside the binary, so nothing is
JIT-compiled when the process starts.

**~448 ms cold start, p50**: `reflex system1` from process start to its scored result
(the engine's own timer), AWS EC2 `g5.xlarge` (NVIDIA A10G, 24 GB),
`Qwen3-0.6B-Q4_K_M`, `n=10`, two interleaved rounds, quantized-resident weights (the
default since 2026-10-20). The two rounds are 445 and 451 ms; from outside the process
that is ~0.58 s wall clock. [Phase breakdown](docs/benchmarks.md#current-default-qr-on-an-a10g-2026-10-20).
`reflex generate` reaches its first token in 458–461 ms, against 620 ms with
`REFLEX_QUANT_RESIDENT=0` (`f16`). The older ~483 ms figure was a T4/`f32` measurement;
see [f16 weight storage](docs/benchmarks.md#f16-weight-storage) and
[QR as the default](docs/design/quantized-resident-weights.md#qr-as-the-default-2026-10-20).

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

No GPU of your own? Reflex is listed on the
[Runpod Hub](https://console.runpod.io/hub/listing/lateos-ai/reflex): deploy it as a
serverless endpoint in one click, no build required ([worker docs](.runpod/README.md)).

## Supported models and limits

| Architecture (GGUF `general.architecture`) | Tested with |
|---|---|
| Dense Qwen3 (`qwen3`) | Qwen3-0.6B |
| Dense Qwen2 / Qwen2.5, incl. Math and Coder (`qwen2`) | Qwen2.5-0.5B-Instruct; a synthetic fixture. YaRN long-context scaling and `qwen2moe` are not supported |
| Qwen3-MoE (`qwen3moe`) | synthetic fixtures (real `qwen3moe` tensor layout, top-k routing) |
| Llama / Mistral (`llama`) | TinyLlama-1.1B |
| Qwen3.5 hybrid Gated DeltaNet (`qwen35`, `qwen35moe`) | Qwen3.5-0.8B; Qwen3.8-27B `Q8_0` (f16 weights, 80 GB GPU); a random-weight `qwen35moe` checkpoint. MTP/NextN heads are not supported, so use a `--no-mtp` conversion or a non-MTP `Q4_K_M` file (the official `Q8_0`/`BF16`/`Q4_0` are rejected) |
| DeepSeek-V2/V3 MLA (`deepseek2`) | DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite-Instruct, GigaChat3.1-10B-A1.8B, GLM-4.7-Flash (f16 weights, 80 GB GPU); two synthetic fixtures. Group-limited routing (the 671B V3/R1), MTP heads, Moonlight (`kimi-k2` tokenizer) and Kanana-2 (`kanana2`) are not supported |
| Granite 4.0-H Mamba-2 hybrid (`granitehybrid`) | Granite 4.0-H 350M and 1B `Q8_0`; a synthetic fixture. MoE variants (Tiny, Small), Falcon-H1, Nemotron-H, Jamba and pure Mamba-2 are not supported |
| Kolibri-1 MoE (`kolibri1`) | Aleph Alpha's Kolibri-1 78B, `Q4_K_M`; a synthetic fixture |
| Gemma 4 dense, text only (`gemma4`) | Gemma 4 12B-it and 31B-it `Q8_0` (31B runs on a 48 GB card with QR; the `f16` opt-out needs ~80 GB); a synthetic fixture. E2B/E4B and 26B-A4B are not supported |

Each architecture counted as supported only after its generated tokens matched an
independent implementation (llama.cpp, or a CPU reference) on real hardware.

- **GGUF only.** Convert other checkpoints with llama.cpp's `convert_hf_to_gguf.py`.
- **Weights are held quantized on the GPU by default** (Q4_K/Q6_K on the dense path, Q8_0 on Gemma 4): on an A10G that is 270 MiB resident for Qwen3-0.6B vs 878 MiB with `f16`, 4.1 GB for Mistral 7B vs 13.8 GB, and 11.1 GB for Gemma 4 12B vs 20.9 GB. `--weights f32` (or
   `REFLEX_WEIGHTS=f32`) keeps the exact `f32` reference mode for everything that is dequantized; norms, MoE routers, activations and the KV cache are
   `f32` in both modes. `reflex check` defaults to `f32`.
- **Context**: up to 65,535 positions per sequence (prompt + generated tokens + any
  imported KV cache); longer requests fail with a clear `context_overflow` error.
- **`system1`** works on every architecture; on the Qwen3.5 hybrid models each candidate
  must be a single token.
- **QR off**: `REFLEX_QUANT_RESIDENT=0` loads matrix weights as `f16` (about 2 bytes per parameter). This makes short warm prompts and `system1` faster on the smallest model (Qwen3-0.6B), at the cost of 2–3.4x the resident VRAM and slower decode (measured ~8–10% on Qwen3-0.6B, ~30–40% on Qwen3-1.7B/TinyLlama, ~2x on Mistral 7B). On a 24 GB card it does not fit Gemma 4 12B for `generate`.
- DeepSeek's MTP heads and group-limited expert routing are not supported.

## How it compares

Cold starts measured from outside the process, Tesla T4, `Qwen3-0.6B-Q4_K_M`, unless
noted. The engine-vs-engine rows time `reflex generate` to its first token (0.83 s p50,
`n=30`), the metric every engine shares; the headline above times `system1` scoring.
Every Reflex figure in this table is the historical `f32`-weights T4 measurement, taken
before `f16` and then QR became the default; the headline above is the current QR/A10G
measurement. In a same-session f16-era comparison, `f16`'s first token was 8% sooner
([f16 weight storage](docs/benchmarks.md#f16-weight-storage)).

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

For a scale-to-zero endpoint, start with the **[serverless deployment
guide](docs/deploy-serverless.md)** — one page, Runpod first. The job Reflex is built
for is a single cold decision (`POST /v1/classify`), not a warm chat pool; see the
[sidecar's classify section](sidecar/openai-adapter/README.md#post-v1classify). The
table below is the long form.

| Target | Where |
|---|---|
| **Serverless / scale-to-zero (start here)** | **[docs/deploy-serverless.md](docs/deploy-serverless.md)** |
| Docker (one multi-target `Dockerfile`: `reflex`, `adapter`, `runpod-lb`) | [docs/reference.md](docs/reference.md#docker) |
| OpenAI-compatible HTTP (`/v1/chat/completions`, `/v1/classify`) | [sidecar/openai-adapter](sidecar/openai-adapter/README.md) |
| AWS, scale to zero (Spot, local socket) | [docs/aws-deployment.md](docs/aws-deployment.md) |
| AWS, always warm (load balancer, HTTPS) | [docs/aws-deployment-warm.md](docs/aws-deployment-warm.md) |
| Runpod serverless, load-balancing endpoint | [serverless/runpod](serverless/runpod/README.md) |
| Runpod Hub, queue-based worker (one-click deploy) | [Hub listing](https://console.runpod.io/hub/listing/lateos-ai/reflex) · [.runpod](.runpod/README.md) |
| Modal (Modal Server, L4) | [.modal](.modal/README.md) |
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
