# Developing Reflex

Notes for people changing the engine: what it optimizes for, the constraints every
change has to respect, how to build and test it with or without a GPU, and a few
performance rules that were learned the hard way. The [README](../README.md) is the
user-facing overview; this page is the contributor-facing one that code comments point
to.

## What this project optimizes for

Reflex is a GGUF-native GPU inference engine optimized for **cold-start energy and
latency**: the time and energy from process launch to the first token. It is not
optimized for sustained server throughput. Competing with llama.cpp or vLLM on
steady-state throughput is a kernel-optimization race this project deliberately doesn't
enter; cold start is the underserved axis. Judge a proposed change by what it does to
that number.

## Non-goals

These are permanent constraints, not current scope. The rules, then the full
rationale [below](#the-full-rationale):

- **`batch_size` is always 1.** No internal request queue or scheduler, no continuous
  batching, no thread pool. The engine runs exactly one request at a time, which is why
  code in this repo can rely on there never being a concurrent caller (single-owner
  `RefCell`/`OnceLock` state, scalar per-token kernel arguments, and so on).
- **No multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager.** Multi-tenancy
  and persistent state belong in a host orchestrator, never inside this engine. For
  example, `kv_io.rs` reads and writes a cache file but never decides where it lives.
  The warm `reflex stdio`/`uds` loop may hold **one** pinned prefix in VRAM
  (`pin_prefix`/`use_pinned`, see [design/agent-relay.md](design/agent-relay.md)), and
  that is the boundary: no second slot, no names, no LRU or eviction policy, no
  background work. Keeping several prefixes, choosing which to keep, or sharing them
  between processes is the orchestrator's job (cache files through
  `import_kv`/`export_kv`, or one warm process per prefix).
- **No network server in the core engine, ever.** HTTP access is provided by the
  separate [`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) crate, which
  talks to the engine over local IPC. The engine itself may only grow sequential,
  non-network interfaces (`reflex stdio`, `reflex uds`) and in-process bindings (the C
  FFI, the PyO3 module). These always target GPU 0.
- **Agents are composed outside the engine.** Routing, envelopes, carry text and
  handoff files belong to the orchestrator; the engine only reports each request's
  facts (see [design/agent-relay.md](design/agent-relay.md#composing-agents-phase-h)).

### The full rationale

These are permanent constraints on this engine, not just current-MVP scope — the whole
reason Reflex exists is to win a narrower bet (cold-start energy/latency) than
sustained-server throughput. A broad serving feature set re-inherits the exact
throughput/serving race that's unwinnable against llama.cpp/vLLM/SGLang's head start.
Multi-tenancy and persistent state belong in the *host
orchestrator*, not in this engine:

> **"Optimized for serverless" describes where this engine runs well, not features it
> grows.** A "Serverless-Native Inference Engine" pitch — building platform machinery
> (internal queue, scheduler, autoscaling logic, `batch_size > 1`) *into* Reflex — was
> considered and rejected on 2026-09-17. Nothing in this page's
> serverless framing reopens that. Running the unmodified binary as a workload on
> Runpod's (or anyone's) control plane is the same arrangement as the AWS ASG pattern
> already documented here: the platform is the host orchestrator, and the engine stays
> single-shot. The violation is defined by what lives *inside* the engine, never by who
> calls it.

- **`batch_size` is always 1.** No request queue, no continuous batching, no
  PagedAttention-style dynamic allocation, no context preemption. Horizontal scaling
  (many concurrent jobs) is the orchestrator's job — spin up N `Reflex`
  processes across GPU slices/time-slices — not this engine's, ever.
- **No internal multi-tenant LoRA router/scheduler.**
- **No internal NVMe/S3 KV-cache manager or cache-hit logic.**
- **No concurrent HTTP/gRPC server**, no request auth/rate-limiting, no autoscaling
  decision-making. If a warm-context mode ever exists (see Phase 4 below), it accepts
  one job at a time, strictly sequentially — never a thread pool.

**No in-core HTTP/gRPC server, ever, not deferred.** This is not a separate exception
to the rule above — a concurrent HTTP listener is the exact same violation
("`batch_size` always 1... never a thread pool") under a different name, and an
adoption/UX ask asking for one doesn't get to reopen it. If HTTP access to this engine
is ever genuinely needed, the pattern is a **separate, optional sidecar binary** that
talks to this core engine over local IPC only — the core engine itself never grows a
network socket. That sidecar now exists:
[`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) is a standalone crate
(own `Cargo.toml`/`Cargo.lock`, not a workspace member, no dependency on
`reflex-engine`) implementing an OpenAI-compatible `POST /v1/chat/completions`
(streaming and non-streaming) in front of one managed `reflex stdio` child process —
real HTTP concurrency on the sidecar's front door, still strictly one request at a
time into the core engine underneath. Renders the loaded GGUF's own
`tokenizer.chat_template` (falling back to plain role-labeled prompt concatenation
when one isn't present or fails to render) — see its own README for usage and known
limitations.

For local, non-network ergonomics, this engine may instead expose: a **stdio JSON-line
mode** (`reflex stdio`, one JSON request per stdin line, fully processed before the next
line is read) and a **Unix Domain Socket mode** (`reflex uds <path>`, Unix-only, one
connection fully processed before the next is accepted) — both strictly sequential,
never a thread pool, mirroring the same request/response protocol. A shared-memory
ring-buffer transport was considered and deliberately deferred — crash-safety and
synchronization design is disproportionate complexity for the ergonomics it would buy
— recorded here as a future-work idea only, not designed.

## Core technical bet

Every CUDA kernel is compiled **ahead of time** by `nvcc` from `build.rs`, never at
runtime through NVRTC. Runtime compilation would add a real JIT cost to every cold
start, which is the one number this project exists to minimize. `src/aot.rs` loads the
precompiled PTX, cubin or fatbin through the CUDA driver API at process start. The build
modes (portable PTX, single-arch cubin, multi-arch fatbin) are described in the reference's
[Kernel build modes](reference.md#kernel-build-modes) section.

This is also why benchmarks against engines that JIT-compile or capture CUDA graphs at
startup measure something different from a comparison with llama.cpp, whose kernels are
also precompiled.

## Build, run and test

The crate needs the CUDA toolkit (`nvcc` on `PATH`, or `CUDA_PATH`/`CUDA_HOME` set) and
an NVIDIA GPU to build and run for real.

```
cargo build --release                           # portable PTX kernels
REFLEX_CUDA_ARCH=sm_86 cargo build --release    # one cubin for one GPU architecture
cargo run --release --bin reflex -- smoke       # first thing to run on a new GPU machine
cargo run --release --bin reflex -- generate <model.gguf> "prompt"
cargo test                                      # host-side unit tests
```

Optional features (`ipc`, `json-output`, `download`, `nvml`, `python`) and their extra
host dependencies are listed in the reference's
[Build features](reference.md#build-features) section.

The toolchain is pinned in `rust-toolchain.toml` (rustup installs it on first use). A
new dependency must pass the license policy in `deny.toml`, which CI checks for both
Cargo projects with `cargo deny check licenses bans sources` (see
[docs/compliance/](compliance/README.md)).

`cargo test` rebuilds only the test harness binaries. It does **not** rebuild
`target/release/reflex`, so after a source change, run
`cargo build --release --bin reflex` before trusting the binary's behavior.

### Working without CUDA

```
REFLEX_SKIP_CUDA=1 cargo build
```

skips kernel compilation entirely, for editing and type-checking on a machine without
the CUDA toolkit. The resulting binary contains no real kernels, so no subcommand can
run inference. This is the mode CI uses on GitHub-hosted runners.

### No CPU fallback

There is no CPU or mock implementation behind `generate`, `system1`, `smoke` or `bench`.
Forward-pass correctness can only be verified on real GPU hardware, and that is
intentional: the point of the project is to measure real cold-start behavior. GPU-only
tests are marked `#[ignore]`; run them with `cargo test --release -- --ignored` on a GPU
machine, or see [gpu-ci.md](gpu-ci.md) for the nightly GPU workflow.

## Model loading: weights stay on the GPU

`Model::load` dequantizes every weight tensor once and uploads it once, as a
device-resident buffer. Matrix weights (the operands of the matmul kernels:
`is_matrix_weight` in `src/model/loading.rs`) are stored as `f16` by default and as
`f32` with `--weights f32` / `REFLEX_WEIGHTS=f32`. Everything else is always `f32`:
norms, biases, the Gated DeltaNet `ssm_*` tensors, Mamba-2's `ssm_conv1d`, `ssm_a`,
`ssm_d` and `ssm_norm` (2-D in the GGUF, but vectors to the kernels), the MoE routers,
activations, and (unless `REFLEX_KV_CACHE=f16`, below) the KV cache. `Weight.data` is a `WeightData` enum (`F32`, `F16`, `Quant`), and every matmul
wrapper in `src/model/kernels.rs` dispatches on it.

- **The f16 path reads f16 weights and accumulates in f32.** Decode GEMVs
  (`gemv_f16_kernel` and friends) widen each weight exactly and multiply by the f32
  activation. Prefill casts activations to f16 and calls `cublasGemmEx` with
  `CUBLAS_COMPUTE_32F`. That cast saturates at f16's 65504 rather than producing inf, and
  counts what it clamped (`Model::f16_activation_stats`).
- **`--weights f32` is the reference mode.** Its dequant kernels compile to the same
  PTX instructions as before f16 existed. `reflex check` and llama.cpp comparisons
  default to it.
- **LoRA merges happen in f32.** `LoadOptions::lora_adapter` loads the adapter's target
  weights as `f32`; `apply_lora` adds the delta and then rounds to f16 once.

**Quantized-resident weights (default).** Q4_K and Q6_K matmul weights are uploaded as their raw GGUF blocks into one device arena and dequantized inside the matmul kernels (`gemv_q4k`/`gemv_q6k` for decode, the fused multi-row Q4_K prefill kernel), or, for longer prompts, into a reused device scratch buffer that cuBLAS then reads. That scratch buffer has the `--weights` dtype: `f16` (read by `cublasGemmEx`) by default, `f32` (`Sgemm`) with `--weights f32`. A Q6_K or Q4_K LM head is kept as raw blocks too. Every other matrix weight follows `--weights`. On the Gemma 4 path Q8_0 weights stay quantized instead (the layer matmuls and the tied LM head; `gemv_q8_0`, the same fused-then-scratch prefill split), which is what fits the 31B Q8_0 on a 48 GB card. The rules below still hold: the scratch path is device to device, so weights are never copied host-to-device per call. Set `REFLEX_QUANT_RESIDENT=0` to load matrix weights as `f16` instead. Other architectures (MoE models other than Kolibri-1, hybrid, MLA, Granite 4.0-H) ignore the flag and keep their normal storage. See [design/quantized-resident-weights.md](design/quantized-resident-weights.md).

**Opt-in on top of that: `REFLEX_LAZY_EXPERTS=1`** (Kolibri-1 only; anything else
prints a notice and loads eagerly). The load reserves each layer's stacked-expert space
in the arena but copies nothing; after routing, a layer uploads the routed experts it
doesn't have yet, once each, through the same pinned pipeline, and they stay resident.
That is still one host-to-device copy per weight for the process lifetime, not a copy
per call. The prefetch readers are off in this mode. See
[design/kolibri.md](design/kolibri.md) (Phase 4 step 2).

**Opt-in: `REFLEX_KV_CACHE=f16`** stores the working K/V cache in f16.
- **Coverage.** Every attention layer of dense, MoE, Kolibri-1 and Qwen3.5 models, and
  MLA's latent; Qwen3.5's GatedDeltaNet state stays f32. Gemma 4, Granite 4.0-H and
  `REFLEX_ATTN_KERNEL=legacy` refuse it at load.
- **Per process.** It is read once at load (`src/model/kv_cache.rs`: `KvBuf`/`KvView`),
  and the pinned prefix holds its state in the same type.
- **Numerics.** The attention kernels are templated on the cache's element type and
  widen each K/V element to f32 as they load it; every cache write rounds to nearest
  even, saturating at ±65504.
- **No cost when off.** The f32 instantiations compile to the same cubins as before.
  The f16 kernels are separate modules that only an f16 process loads, the first time
  it needs them.
- **Cache files.** With an f16 cache, export defaults to f16 files, and f32 files import
  with rounding.
- See [design/decode-speed.md](design/decode-speed.md), "f16 KV cache".

Rules that follow from measured regressions:

- **Never copy weights host-to-device per call.** An early version re-uploaded weight
  buffers inside `gemv`/`rmsnorm` on every call and was about 4.3x slower than
  llama.cpp until that was removed. The same per-call-overhead reasoning is why batched
  kernels such as `gemv_per_head_batch` exist: thousands of small launches per layer
  cost as much as the copies did.
- **Dequantize on the device, not in a host loop.** The token-embedding table stays
  host-resident as raw quantized bytes, and rows are dequantized lazily as tokens need
  them. When the full vocabulary table is needed on the GPU (a tied LM head), it goes
  through the same on-device dequant path as every other tensor. A single-threaded host
  loop over the whole table once cost about 548 ms, most of model load.
- **Keep the token-embedding table in the mmap; don't copy it out.**
  `LazyTokenEmbedding` holds a `SharedBytes` handle into the GGUF mapping, and the model
  keeps the file mapped for its lifetime, as llama.cpp does. Copying the table into an
  owned `Vec` at load (127.6 MB for Qwen3-0.6B) was the largest single part of model
  load, about 103 ms of 231 ms on a T4, mostly first-touch page faults on the new
  allocation rather than file reads.
- **Keep the pipelined upload.** `WeightLoadPipeline` stages each tensor's raw bytes
  through two reused pinned host buffers, in chunks of at most 64 MB, and uploads on a
  separate stream, so one chunk's fill overlaps the previous chunk's copy and tensor
  N+1's copy overlaps tensor N's dequant kernel. Chunks of 8 MB or more are filled by
  several threads (`REFLEX_LOAD_THREADS`, default min(cores, 8)), while prefetch
  reader threads (`REFLEX_LOAD_READERS`, default 16, 0 disables them) fault the file's
  pages in up to 1 GB ahead without copying. Cold loads need many reads in flight;
  warm loads need few copy threads, since extra ones compete with the H2D copy for
  memory bandwidth. One fill thread was the bottleneck on large models, and small
  models never start either kind of thread.
  Tensors with no dequant kernel (`F32` norms) take the same slots plus an async
  device-to-device copy; a `htod_sync_copy` there synchronizes the compute stream and
  drains the pipeline at every norm. Replacing it with a blocking copy per tensor,
  or allocating the staging buffers per tensor, reintroduces a host/device race or
  serializes the pipeline. The f16 dequant kernels go through the same pipeline; only
  the output element type differs.
- **Read cache files straight into the pinned slots.** `kv_io::import_kv` reads only
  the header; `WeightLoadPipeline::upload_state` reads each value range into the
  pinned slots in chunks of at most 8 MB and copies it to the cache (an f16 file is
  widened on the GPU for an f32 cache, an f32 file rounded on the GPU for an f16 one). Reading the payload into fresh host vectors first faulted in
  every page of two new buffers: a cold process read a 342 MB file in 284 ms against
  110 ms warm. The chunk stays small because pinned memory costs ~0.46 ms per MB to
  allocate in a cold process.
- **Numerics probes.** `REFLEX_F16_ROUNDTRIP=1` (with `--weights f32`) rounds every
  matrix weight to f16 and back, which isolates weight rounding from the f16 kernels.
  `REFLEX_TOP2_TRACE=1` prints each generated position's top-2 logits.
  [`scripts/verify_f16_weights.sh`](../scripts/verify_f16_weights.sh) uses both.
- **Embedding models ride the dense path.** A GGUF with `<arch>.attention.causal =
  false` loads with an `EncoderModel` (`src/model/encoder.rs`): bidirectional attention,
  YaRN RoPE and the query temperature in the batched prefill, then `Model::embed`
  (final norm, mean pool, L2 norm). Every hook is a no-op for causal models, and the
  bidirectional kernels are separate modules (`encoder.cu`,
  `attention_tiled_bidir.cu`), so causal models never load them. Keep the causal
  attention cubins byte-identical when touching the shared kernel bodies: the
  bidirectional path is a `CAUSAL` template flag, not a runtime branch. Design:
  [design/embeddings.md](design/embeddings.md).

## GGUF metadata conventions

Model configuration is read from arch-prefixed GGUF keys: `general.architecture` names
the architecture (for example `qwen3`), and every other key is looked up as
`<architecture>.<key>`, such as `qwen3.context_length`. Whether a model uses MoE layers
is decided by `<architecture>.expert_count` being present and nonzero, not by a
hardcoded list of architecture names. Tools outside the engine (the sidecar's
`gguf_meta` reader) follow the same convention.

## Known test-fixture limitations

`.gguf` files are gitignored, so fixture-based tests are `#[ignore]`d and need the files
in `test-data/` locally:

- `test-data/tiny-qwen3moe.gguf` is a synthetic `qwen3moe` model with
  `expert_used_count` (2) smaller than `expert_count` (8), real Qwen3 QK-norm tensors and
  a `gpt2`-style tokenizer, so it exercises top-k routing that actually excludes experts.
  It was built by running a hand-written, random-weight Hugging Face checkpoint through
  llama.cpp's unmodified `convert_hf_to_gguf.py`.
- `test-data/deepseek-tiny-mla.gguf` is a synthetic `deepseek2` (MLA) model built the
  same way, because no small real `deepseek2` GGUF exists publicly. It covers the
  dense-lead layers only; the routed-MoE MLA path can only be exercised with the real
  `deepseek-ai/DeepSeek-V2-Lite` checkpoint (about 17 GB at `q8_0`), which is not kept
  in the repo. Community `deepseek2` GGUFs that predate llama.cpp's MLA tensor-split
  conversion are rejected by this engine, so convert it fresh with a current
  `convert_hf_to_gguf.py`. With `79e2e74`'s converter, use `transformers==4.57.1`:
  with 5.x the DeepSeek tokenizer hash isn't recognized and no usable pre type is
  written.
- `test-data/tiny-mla-v3.gguf` is a second synthetic `deepseek2` model, built from a
  random `DeepseekV3ForCausalLM` checkpoint the same way. It covers what the first one
  doesn't: Q-LoRA queries, sigmoid routing with a nonzero `exp_probs_b` selection bias,
  `expert_weights_norm` and `expert_weights_scale`, a shared expert and
  `v_head_dim != qk_nope_head_dim`. Dropping any one of these changes every comparison
  prompt's greedy tokens in llama.cpp. Builder, references and that check:
  `test-data/tiny-mla-v3-src.tar.gz`.
- `test-data/*-header.gguf` for DeepSeek-Coder-V2-Lite (current and legacy format),
  GigaChat3.1-10B-A1.8B, GLM-4.7-Flash (unsloth's, and ggml-org's without
  `expert_gating_func`), Moonlight-16B-A3B and Kanana-2-30B-A3B are real `deepseek2`
  headers (metadata and tensor table, see the Qwen2.5 headers below) for the MLA config
  tests. Sources: [design/mla-widening.md](design/mla-widening.md).
- `test-data/kolibri1-tokenizer.gguf` is the metadata section of
  `Hob-forge/Kolibri-1-GGUF`'s `Kolibri-1-Q4_K_M.gguf` (the first ~4.8 MB, read with an
  HTTP range request) rewritten with an empty tensor table. It carries the real
  Kolibri-1 vocab and merges for the tokenizer golden test.
- `test-data/gemma4-tokenizer.gguf` is the same thing for Gemma 4: the metadata section
  of `ggml-org/gemma-4-12B-it-GGUF`'s `gemma-4-12B-it-Q8_0.gguf` (~16 MB) with an empty
  tensor table, for the `gemma4` tokenizer golden test (see
  [design/gemma4.md](design/gemma4.md)).
- `test-data/llama-bpe-tokenizer.gguf`, `deepseek-llm-tokenizer.gguf`,
  `deepseek-v3-tokenizer.gguf`, `qwen35-tokenizer.gguf`, `gigachat-tokenizer.gguf` and
  `glm4-tokenizer.gguf` are the same thing for the `tokenizer.ggml.pre` golden tests
  (Llama-3.2-1B-Instruct, DeepSeek LLM 7B chat, DeepSeek-V3-0324 shard 1, Qwen3.5-0.8B,
  GigaChat3.1-10B-A1.8B, GLM-4.7-Flash). Golden ids come from llama.cpp's
  `llama-tokenize`. Sources and byte ranges are in each test's doc comment; see
  [design/tokenizer-pre-types.md](design/tokenizer-pre-types.md).
  `scripts/gguf_tokenizer_only.py` makes any of these from the first ~40 MB of a GGUF
  (`curl -r 0-41943039`): it keeps the metadata bytes verbatim, writes an empty tensor
  table and drops a shard's `split.*` keys. Run on Kolibri-1's head, it reproduces
  `kolibri1-tokenizer.gguf` byte for byte.
- `test-data/qwen2.5-*-header.gguf` (0.5B-Instruct, 1.5B-Instruct, Math-1.5B-Instruct,
  Coder-1.5B-Instruct, and ggml-org's Coder `Q8_0`) are real Qwen2.5 GGUF headers made
  with `scripts/gguf_tokenizer_only.py --keep-tensor-table`: metadata **and** tensor
  table, no tensor data. `parse_model_config` and the tokenizer read them like the real
  file (Reflex only bounds-checks tensor data when it reads it); llama.cpp won't load
  them. They back the `qwen2` config tests and the Qwen2.5 special-token golden test.
  Sources: [design/qwen2.md](design/qwen2.md).
- `test-data/tiny-qwen2.gguf` is a synthetic `qwen2` model: 2 layers, hidden 64, 4 query
  and 2 KV heads, tied embeddings, nonzero Q/K/V biases scaled so that dropping any one
  of them changes every comparison prompt's greedy tokens, and the real
  Qwen2.5-0.5B-Instruct tokenizer. `test-data/tiny-qwen2-lora.gguf` is a random LoRA on
  its `q_proj`/`v_proj`/`down_proj`. Both were built by running hand-written, random
  Hugging Face checkpoints through llama.cpp `79e2e74`'s unmodified
  `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py`; the golden greedy ids (with and
  without the LoRA) come from llama.cpp `79e2e74`. Builders, prompts and reference tool:
  `test-data/tiny-qwen2-src.tar.gz`.
- `test-data/tiny-gemma4.gguf` is a synthetic `gemma4` model: 6 layers (2 global, without
  `attn_v`), random f32 weights, an 8-token sliding window and the real Gemma 4
  tokenizer, written directly with llama.cpp's `gguf-py` from the real 12B header's
  metadata. Its golden greedy ids come from llama.cpp `79e2e74`. Builder and reference
  tools: `test-data/tiny-gemma4-src.tar.gz`.
- `test-data/tiny-kolibri1.gguf` (Q4_K_M) and `tiny-kolibri1-f32.gguf` are a synthetic
  `kolibri1` model: 6 layers (layer 4 full attention), 16 experts with top-4, a nonzero
  router bias, a 16-token sliding window and the real Kolibri-1 tokenizer. Mainline
  llama.cpp has no `kolibri1` support, so it was converted and quantized with llama.cpp
  `836d571` plus the community `kolibri1-llama.cpp.patch` (see
  [design/kolibri.md](design/kolibri.md)). Source and build scripts are archived as
  `test-data/tiny-kolibri1-src.tar.gz`.
- `test-data/tiny-granitehybrid.gguf` is a synthetic `granitehybrid` (Granite 4.0-H)
  model: 6 layers (4 Mamba-2, 2 attention without RoPE), 8 SSM heads of 32 over 2 groups,
  `d_state` 128, non-default muP scales, tied embeddings and the real Granite 4.0-H 350M
  tokenizer. Its shapes keep llama.cpp's CUDA backend on its SSM kernels. Dropping `D`,
  `dt_bias`, the `z` gate, the conv bias, the grouped norm or any muP scale but the
  logit scale changes every comparison prompt's greedy tokens in llama.cpp; the logit
  scale is checked through the logit values. Built from a random
  `GraniteMoeHybridForCausalLM` checkpoint through llama.cpp `79e2e74`'s unmodified
  `convert_hf_to_gguf.py` (`transformers` 4.57.1 and 5.19 give byte-identical files).
  Builder, prompts, references and the sensitivity check:
  `test-data/tiny-granitehybrid-src.tar.gz`.
- `test-data/granite-4.0-h-{350m,1b,micro,tiny}-header.gguf` are real Granite 4.0-H
  headers (metadata and tensor table) for the `granitehybrid` config tests, and
  `falcon-h1-tiny-90m-header.gguf`, `nemotron-3-nano-4b-header.gguf`,
  `jamba-900m-header.gguf` and `mamba2-130m-header.gguf` back the tests that reject
  those architectures by name. `test-data/granite-tokenizer.gguf` is the tokenizer-only
  file for the `dbrx` pre-type golden test. Sources:
  [design/mamba2-hybrid.md](design/mamba2-hybrid.md).
- `test-data/tinyllama-1.1b-chat-tokenizer.gguf` and `qwen3-0.6b-tokenizer.gguf` are
  tokenizer-only files (`scripts/gguf_tokenizer_only.py`) for the continuation-encoding
  tests in `src/tokenizer.rs` and the `json_schema` mask tests in
  `src/grammar/tests.rs`, which check every mask against a walk of every vocabulary
  token. Sources: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`'s
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (SentencePiece, the first 8 MB) and
  `unsloth/Qwen3-0.6B-GGUF`'s `Qwen3-0.6B-Q4_K_M.gguf` (gpt2 byte-level BPE).
- `test-data/tiny-mistral3-embed.gguf` is a synthetic embedding model: a random
  `Ministral3Model` text encoder (`is_causal: false`, mean pooling, 2 layers, hidden 128,
  4 query / 2 KV heads of dim 64) with the real Nemotron-3.5-Embed tokenizer, YaRN
  (factor 4) and the query temperature over an original context of 64, so inputs past
  64 tokens exercise both. Converted by llama.cpp's unmodified `convert_hf_to_gguf.py`
  at `1a24df1` (`79e2e74` predates `mistral3` encoders); `--outtype f32`.
  `tiny-mistral3-embed-ref.tsv` holds HF `transformers`' fp32 embeddings of its 15
  comparison inputs. With causal attention, without YaRN or without the temperature,
  HF's own output moves to a cosine of 0.48, 0.83 and 0.99990 against them, all below
  the f32 test bar of 0.99999. `nemotron-3.5-embed-1b-header.gguf` (the real 1B's
  header) and `nemotron-3.5-embed-tokenizer.gguf` back the config and Tekken tokenizer
  tests. The nightly passes `EMBED_GGUF` (a real Nemotron-3.5-Embed-1B GGUF) and
  `EMBED_REF` (its HF references, `nemotron-3.5-embed-ref.tsv`) to
  `nemotron_embed_real_model_matches_hf_reference`. Builder, HF reference script and
  references: `test-data/tiny-mistral3-embed-src.tar.gz`; see
  [design/embeddings.md](design/embeddings.md).
- Decision models ([design/decision-models.md](design/decision-models.md)):
  `test-data/lev-q8_0-header.gguf` and `openjev-q4_k_m-header.gguf` are the real
  headers of `ggml-org/lev-GGUF`'s `lev-Q8_0.gguf` and `ggml-org/OpenJev-GGUF`'s
  `OpenJev-Q4_K_M.gguf` (the first 12 MB, `scripts/gguf_tokenizer_only.py
  --keep-tensor-table`). They back the label-table tokenizer test and the sidecar's
  `systemone` tests (metadata, templates, rendering). For the GPU tests,
  `test-data/tinyopenjev-for-testing-Q8_0.gguf` is llama.cpp's own OpenJev test model,
  downloaded unchanged from `ggml-org/tinyopenjev-for-testing-gguf`, and
  `test-data/tiny-lev.gguf` is the same weights with lev's decision metadata and
  template written in (builder: `test-data/tiny-lev-src.tar.gz`). Both are CC BY-NC
  4.0, like OpenJev, and stay local. Their outputs are not meaningful.
  `test-data/decision-models-src.tar.gz` holds the llama.cpp comparison: a harness on
  llama.cpp's own `server_decision_context` that writes prompt and token goldens, the
  request set, the goldens for all three models, and the HTTP comparison and benchmark
  scripts. The sidecar's `render_matches_llama_cpp_goldens_if_configured` and the
  engine's `test_decision_prompt_tokens_match_llama_cpp_goldens_if_configured` check a
  GGUF against those goldens when `SYSTEMONE_GOLDEN_GGUF`, `SYSTEMONE_GOLDENS` (and,
  for the sidecar, `SYSTEMONE_GOLDEN_REQUESTS`) are set.
- Tests that need a real model read its path from `REFLEX_TEST_GGUF`. The nightly
  passes `SPM_GGUF` (a SentencePiece chat model, e.g. TinyLlama-1.1B-Chat) to the
  tests that need one (`ipc_kv_relay_spm_model_*`, `ipc_json_schema_spm_model_*`).
- A real Qwen3.5 GGUF for `HYBRID_GGUF` must not carry MTP/NextN heads. Every
  `ggml-org/Qwen3.5-0.8B-GGUF` file (`Q8_0`, `BF16`, `Q4_0`) sets
  `nextn_predict_layers=1`, so the hybrid tests fail to load with
  `qwen35.nextn_predict_layers MTP/NextN blocks (nextn_predict_layers=1) are not
  supported by this MVP (reconvert with convert_hf_to_gguf.py --no-mtp to drop them)`.
  That is a fixture-selection trap, not an engine bug. Use a non-MTP community
  conversion instead (e.g. `unsloth/Qwen3.5-0.8B-GGUF`'s `Qwen3.5-0.8B-Q4_K_M.gguf`),
  or convert the HF checkpoint with `convert_hf_to_gguf.py --no-mtp`; with such a file
  the GPU suite passes in both QR-default and `REFLEX_QUANT_RESIDENT=0` modes.
  Switching to `Q8_0` is not a workaround: the official `Q8_0` file has the same MTP
  block.

[`scripts/gpu_nightly_tests.sh`](../scripts/gpu_nightly_tests.sh) maps each GPU test to
the file it needs and reports the ones it can't run as skipped.

## The `reference/` directory

`reference/gated_deltanet_reference.rs` is a host (CPU) implementation of the Qwen3.5
Gated DeltaNet recurrence. It served as the correctness oracle while the GPU kernel was
being debugged. It is not compiled as part of the crate.

`reference/mamba2_reference.rs` is the same for one Mamba-2 step (conv, scan, D skip,
gate, grouped RMSNorm), written before `kernels_cuda/mamba2.cu`. It is plain Rust with no
dependencies and is compiled into the test build only (`src/model/mod.rs` includes it with
`#[path]`): its own unit tests run under `cargo test`, and the Mamba-2 kernel tests
compare against it.
