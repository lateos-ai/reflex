# Reflex

A high-performance, GGUF-native Rust & CUDA inference engine optimized for cold-start
latency and real-time "System 1" agent decision loops — process launch to first token,
not sustained server throughput.

Every CUDA kernel is compiled ahead-of-time by `nvcc` at *build* time and shipped
inside the binary — never compiled at runtime via NVRTC — so there's no multi-second
JIT tax on first use, the way there is with a runtime-compilation design. That's the
whole bet: be the fastest way to turn a cold process into one output token, then get
out of the way.

## Quickstart

Requires the CUDA toolkit (`nvcc` on `PATH`, or `CUDA_PATH`/`CUDA_HOME` set) and an
NVIDIA GPU. `REFLEX_SKIP_CUDA=1 cargo build` skips kernel compilation for
editing/type-checking on a machine without CUDA (no subcommand will actually run
kernels in that mode).

```
git clone https://github.com/lateos-ai/reflex.git
cd reflex
cargo build --release

# Prove the AOT pipeline works end to end on your GPU:
cargo run --release --bin reflex -- smoke

# Run a real forward pass against a GGUF file:
cargo run --release --bin reflex -- generate <path-to-gguf> "Once upon a time"
```

`reflex` is a single binary with subcommands: `generate` (load a GGUF and generate
tokens), `system1` (single-pass, non-autoregressive candidate scoring — the "System 1"
decision-loop path), `smoke` (the AOT-pipeline check above), `bench` (warm-latency
microbenchmark), `check` (byte-exact-vs-reference correctness check, CI-scriptable),
and `stdio`/`uds` (local JSON-line IPC, both need `--features ipc`). Run `reflex
<subcommand>` with no further arguments to see that subcommand's own usage.

### Example: download a model from Hugging Face, then run a System1 test

`system1` takes a local GGUF path, so download the file first with the
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install -U
huggingface_hub`), then point `system1` at it — this scores each `--candidate`
against the prompt in a single pass, with no autoregressive decode loop:

```
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir .

cargo run --release --bin reflex -- system1 Qwen3-0.6B-Q8_0.gguf \
  "The capital of France is" \
  --candidate " Paris" --candidate " London" --candidate " Berlin"
```

Real output from this exact command (RTX A6000):

```
REFLEX_SYSTEM1_CANDIDATE_OK idx=0 text=" Paris" token_ids=[12095] score=17.407064 probability=0.997350
REFLEX_SYSTEM1_CANDIDATE_OK idx=1 text=" London" token_ids=[7148] score=11.308186 probability=0.002239
REFLEX_SYSTEM1_CANDIDATE_OK idx=2 text=" Berlin" token_ids=[19846] score=9.612655 probability=0.000411
REFLEX_SYSTEM1_OK process_start_to_result_ms=8524.253 num_candidates=3 best_idx=0 best_text=" Paris" entropy=0.028154
```

`probability` is relative to this candidate set only, not a vocab-wide probability —
see `Model::system1_evaluate`'s doc comment in `src/model.rs`. `system1` currently
supports dense/MoE Qwen3 only; the Qwen3.5 hybrid mixer and DeepSeek-V2/V3 (MLA) are
rejected with a clear error (`generate` supports all four architectures — swap in a
DeepSeek GGUF the same way for a `generate` run instead).

`generate` can also pull a GGUF straight from the Hub itself, via this project's own
Rust `hf-hub` integration — `--model <org/repo:file.gguf>` or `--quickstart`, both
requiring `cargo build --features download`:

```
cargo run --release --features download --bin reflex -- generate --quickstart "Once upon a time"
```

### Bringing your own model (non-GGUF checkpoints)

Reflex only ever loads GGUF — this is deliberate, not a missing feature (its
Hugging Face integration is a GGUF downloader/cache only, never a new
tensor-format ingestion path). If you have a safetensors/HF-format checkpoint,
convert it to GGUF first with llama.cpp's own unmodified `convert_hf_to_gguf.py`
— the same converter this project uses internally for its own test fixtures and
for real checkpoints like DeepSeek-V2-Lite:

```
python convert_hf_to_gguf.py /path/to/hf-checkpoint --outtype q8_0 --outfile model.gguf
```

LoRA adapters convert the same way, via llama.cpp's `convert_lora_to_gguf.py`.

## Why this exists

Closing the steady-state-throughput gap with llama.cpp/vLLM is a kernel-optimization
race against projects with a multi-year head start — Rust as a language doesn't change
who wins it. What none of llama.cpp, vLLM, or `candle` are built for or measured
against is a **cold** invocation — serverless/FaaS, single-shot CLI/dev-tool calls,
batch/cron jobs, edge devices that wake on demand. A naive runtime-JIT design pays a
real, measured multi-second tax on first kernel use; vLLM took ~235s to become ready
(CUDA graph capture) before serving one request; llama.cpp avoids both because its
kernels are compiled by `nvcc` at *build* time, not at process start.

**Target metric**: energy-to-first-token from cold start (joules, process launch to
first generated token) — a real, underexplored gap. Existing energy benchmarks measure
warm/steady-state joules-per-token, not full-lifecycle cold-start cost.

**Target models**: Qwen and DeepSeek families.

## Benchmarks

All comparisons are cold-start (process launch to first token/result), same
ThunderCompute A6000, `n=3`, external wall-clock (`/usr/bin/time -v` — process launch
to exit, not just Reflex's own internal timer). Full methodology, disclosed caveats,
and per-run numbers for every comparison below are in `DECISIONS.md` and `HISTORY.md`.

| vs. | Result | Caveat |
|---|---|---|
| **llama.cpp** | **~1.3–1.4x faster** (4.71–5.05s vs. 6.45–6.56s) | Both AOT-compiled — doesn't exercise the JIT-tax claim below |
| **vLLM** | **~24–52x faster** (4.71–5.05s vs. 121–244s, depending on `torch.compile` cache state) | Installed vLLM has no GGUF support; ran against an HF safetensors checkpoint instead, disclosed |
| **Ollama** | Directly competitive when it doesn't stall (~6–7s), but its bundled `llama-server` intermittently hits an internal GPU-discovery-watchdog timeout (~55–62s) | Wraps llama.cpp's own runtime — tests packaging/daemon overhead, not the AOT-vs-JIT bet |
| **TypeSafe Jev**, cold-start-to-decision | Reflex loses, **~10–60x slower** | Different deployment model: Jev is an always-warm managed API; this measures a genuine cold local process launch |
| **TypeSafe Jev**, warm/compute-only | **Competitive, within ~1.3–2x** (19.4ms vs. Jev's cited 10–15ms) | Jev's figures are self-reported/published, not independently reproduced here |

The llama.cpp/Ollama/Jev "loses" results above are reported as-is, not smoothed over —
see `DECISIONS.md`'s benchmark-methodology entries for why each comparison is framed
the way it is.

## Core technical bet

Every CUDA kernel is compiled **ahead of time** (`build.rs` invokes `nvcc`, see
`build.rs` and `src/kernels_cuda/`), never at runtime via NVRTC. `src/aot.rs` loads the
precompiled PTX/cubin at process start via the CUDA driver API. Default mode emits
portable PTX (small driver-side JIT-to-SASS cost at load); set `REFLEX_CUDA_ARCH=sm_XX`
to compile straight to a `cubin` for one target architecture (true zero-JIT, at the cost
of needing a matching cubin per deployment target). Which one actually wins on real
hardware is unverified — that's the first thing to measure, not assume.

Run `cargo run --bin reflex -- smoke` on a real GPU instance as the very first
real-hardware step: it proves the AOT pipeline works end to end and reports actual
process-start-to-first-result wall clock on the simplest possible kernel, before any
model-architecture work begins.

**Linux build prerequisite for `--features download`/`ipc`/`python`** (`--all-features`
included): these pull in `hf-hub`, whose `ureq` HTTP client needs `libssl-dev` +
`pkg-config` on the build host, or `cargo build` fails with `openssl-sys` unable to find
an OpenSSL installation. Not needed for the default feature-less build. On Ubuntu/Debian:
```
sudo apt-get install -y libssl-dev pkg-config
```
(Discovered on a fresh ThunderCompute instance during the MVP-release adoption round —
not needed on the Windows dev machine that round otherwise developed on, since
`native-tls` uses a different TLS backend there.)

## MVP order

All four steps below are **done** and real-hardware-verified (see `STATUS.md` for
current state, `HISTORY.md` for the full verification write-up of each):

1. **Dense Qwen3** — the best-understood, most well-documented architecture to build
   against first; proves the AOT-compilation + cold-start-benchmark harness works at all.
2. **Qwen3-MoE**
3. **Qwen3.5 hybrid Gated DeltaNet mixer**
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension.

## Non-goals

These are permanent constraints on this engine, not just current-MVP scope — the whole
reason Reflex exists is to win a narrower bet (cold-start energy/latency) than
sustained-server throughput. A broad serving feature set re-inherits the exact
throughput/serving race that's unwinnable against llama.cpp/vLLM/SGLang's head start.
Multi-tenancy and persistent state belong in the *host
orchestrator*, not in this engine:

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
is ever genuinely needed, the pattern is a **separate, optional sidecar binary** (e.g.
`system1-openai-adapter`) that talks to this core engine over local IPC only — the core
engine itself never grows a network socket. Building that sidecar is out of scope for
now; this paragraph only records the escape-hatch pattern so a future HTTP ask gets
routed there instead of back into this engine.

For local, non-network ergonomics, this engine may instead expose: a **stdio JSON-line
mode** (`reflex stdio`, one JSON request per stdin line, fully processed before the next
line is read) and a **Unix Domain Socket mode** (`reflex uds <path>`, Unix-only, one
connection fully processed before the next is accepted) — both strictly sequential,
never a thread pool, mirroring the same request/response protocol. A shared-memory
ring-buffer transport was considered and deliberately deferred — crash-safety and
synchronization design is disproportionate complexity for the ergonomics it would buy
— recorded here as a future-work idea only, not designed.

## Post-architecture-MVP roadmap: productization

Once the model-architecture MVP proves the engine handles the target model
families at all, the next axis is making the *cold-start path itself* faster and
adoptable — without ever crossing into building a serving platform. The framing: let
vLLM win the warm-throughput race; Reflex wins by being the fastest way to turn
cold compute into one output token, then getting out of the way. All four phases below
are **done** — see `HISTORY.md` for the full per-round write-up of each:

- **Phase 1** — Single-shot CLI: process launch -> one forward pass -> exit.
- **Phase 2 — Fast IO**: weights upload to the GPU once (not re-uploaded per kernel
  call), device-resident activations through a whole layer, and on-GPU dequant kernels
  for the block types this project's fixtures use for the bulk of weight bytes. This is
  the work behind the llama.cpp benchmark result above.
- **Phase 3 — State I/O**: `--export-kv <file>`/`--import-kv <file>` for raw K/V-cache
  dump/load/resume, across all four architectures. Reflex stays ignorant of *where*
  that file lives (NVMe, an S3-backed FUSE mount, tmpfs) — that's the orchestrator's
  job, not this engine's.
- **Phase 4 — Embeddability**: `--lora <path>` (load-time adapter application, no
  runtime hot-swap multiplexer) and a Rust C-FFI surface (`src/ffi.rs`,
  `include/reflex_engine.h`) so an external orchestrator can embed Reflex directly
  instead of `exec`-ing a binary.

## Host-side building blocks

- `src/gguf.rs` — GGUF metadata/tensor-directory parsing (mmap-based).
- `src/dequant.rs`, `src/dequant_iq.rs`, `src/dequant_iq_tables.rs` — standard and
  i-quant dequantization, verified byte-exact against `gguf-py`.
- `src/tokenizer.rs` — verified against real sentencepiece/BPE references.

None of these care how kernels get compiled — they're pure host-side GGUF/tokenizer
logic. There is deliberately no NVRTC runtime-compile-and-load path anywhere in this
codebase — that's the thing this project's AOT design replaces, not reuses.

## Docker

A multi-stage `Dockerfile` is included: the builder stage has the full CUDA devel
toolkit (`nvcc`) to compile the AOT kernels; the runtime stage only needs the CUDA
*runtime* libraries, since every kernel byte is embedded directly into the compiled
binary at build time — the runtime image never runs `nvcc` and never needs the devel
toolkit.

```
# Defaults to sm_86 (RTX A6000/3090-class). Pass --build-arg REFLEX_CUDA_ARCH=sm_XX
# for a different target compute capability, or --build-arg REFLEX_CUDA_ARCH= (empty)
# for a portable PTX build that JITs to whatever GPU the container actually runs on.
docker build --build-arg REFLEX_CUDA_ARCH=sm_86 -t reflex .

# Needs nvidia-container-toolkit on the host. The default entrypoint is
# `reflex generate`, so pass just the GGUF path and prompt:
docker run --rm --gpus all -v /path/to/models:/models \
  reflex /models/Qwen3-0.6B-Q4_K_M.gguf "Once upon a time"
```

For any other subcommand (`system1`/`smoke`/`bench`/`check`/`stdio`/`uds`), override
the entrypoint:

```
docker run --rm --gpus all -v /path/to/models:/models \
  --entrypoint /usr/local/bin/reflex reflex \
  system1 /models/Qwen3-0.6B-Q4_K_M.gguf "Q: ...? A:" --candidate " Yes" --candidate " No"
```

The CUDA major/minor version in both Docker stages must stay consistent with
`Cargo.toml`'s pinned `cudarc` feature (`"cuda-12000"`, i.e. CUDA 12.x) — a mismatch is
a build-time/runtime library version mismatch this Dockerfile can't catch for you.

## Kubernetes

Reflex is a single-shot CLI, not a server (see Non-goals above) — the natural
Kubernetes primitive is a **Job**, one cold-start invocation per Pod, never a
`Deployment`/`Service`. A minimal example running `generate` against a GGUF baked into
a volume, requesting one GPU via the standard NVIDIA device plugin:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: reflex-generate
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: reflex
          image: reflex:latest
          args: ["/models/Qwen3-0.6B-Q4_K_M.gguf", "Once upon a time"]
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: reflex-models
```

For `system1`/`bench`/`check`/other subcommands, set `command: ["/usr/local/bin/reflex"]`
and put the subcommand as the first entry in `args`, same as the Docker override above.
This is exactly the "orchestrator's job" this engine intentionally stays out of —
Reflex itself never grows a scheduler, a request queue, or a `batch_size > 1`; Kubernetes
(or cron, or a FaaS platform) is where that concurrency/scheduling belongs.

