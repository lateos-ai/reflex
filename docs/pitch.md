# Reflex — pitch cheat-sheet

> **Internal cheat-sheet, partly superseded.** Do not use this file in a listing, a
> post, or a Hub description. Where any number or limit here disagrees with
> [`README.md`](../README.md) (especially weight storage and supported
> architectures) or [`docs/benchmarks.md`](benchmarks.md), those win.

**Internal doc.** What to know, what to say, and what *not* to claim. Every number
here is a real measurement from this repo's own cold-start benchmark unless it is
explicitly labeled a *citation*. When in doubt, under-claim — the whole product is built
on measurement discipline, and a pitch that survives technical scrutiny is worth more
than one that doesn't.

---

## 0. The one-liner

Pick by audience:

- **Short:** "Reflex is a cold-start-first LLM inference engine — under a second from
  process launch to first token."
- **Measured:** "Reflex turns a cold GPU into a first generated token in **0.83 s**
  (p50, external wall clock, Tesla T4, n=30) — fast enough that scale-to-zero GPUs stop
  being a latency tax."
- **Positioning:** "Everyone else races on steady-state tokens/sec. Reflex competes on
  the axis serverless billing actually charges you for: how long the worker takes to
  become useful."

---

## 1. The problem

- Serverless and bursty GPU workloads **bill per second**, so on single-shot / low-QPS
  traffic the dominant cost is not tokens-per-second — it's **time-to-first-token after a
  cold launch**. A slow start shows up directly on the invoice.
- Runtime-JIT / graph-capture serving stacks pay a **multi-second warmup on every cold
  start**. Measured in this repo: vLLM 0.30.0 took **~127 s** to first token cold on a
  T4 (its `init engine ... took 84.86 s` line, including 34.7 s compilation), and
  **~244 s** on the older A6000 run.
- The whole industry optimizes steady-state throughput; **cold start is underserved.**
  That gap is the wedge.

---

## 2. What Reflex is (the core bet)

**One sentence:** a GGUF-native GPU inference engine whose every CUDA kernel is compiled
**ahead of time** by `nvcc` at *build* time and loaded through the CUDA driver API at
process start — never JIT-compiled at runtime.

The pieces that produce the cold-start number:

| Lever | What it does | Metric |
|---|---|---|
| **AOT kernels** | `build.rs` compiles every `.cu` with `nvcc`; `aot.rs` loads the PTX/cubin at process start. No NVRTC JIT tax. | whole dense kernel-module load **~3 ms** (pinned cubin; ~3.5 ms portable PTX) |
| **`fast_exit`** | Flushes output then calls the raw `_exit` syscall, skipping the CUDA-context teardown `atexit` hook that costs **seconds** on virtualized hosts. | `reflex smoke` 6.3 s → 0.56 s on ThunderCompute |
| **Cold-load overlap** | Tokenizer construction + cuBLAS handle init moved to one worker thread; per-tensor weight dequant/upload is double-buffered through pinned host memory. | model load ~237 ms (~49% of the `system1` total) |
| **Lazy embedding work** | `token_embd` stays raw-compressed on the host; tied `lm_head` dequant is deferred and done on-device. | `system1` total 627 ms → 485 ms across the perf round |
| **Warp-per-row GEMV** | Decode-path `gemv_kernel` rewritten (coalesced loads + shuffle reduction). | decode **15.4 → 74.8 tok/s (~4.9×)** |

**Target metric:** energy-to-first-token from cold start (joules + ms, process launch to
first token) — not warm tokens/sec. **Target models:** Qwen and DeepSeek families.
**Architectures supported** (matching the [README](../README.md#supported-models-and-limits)
table): dense Qwen3, Qwen3-MoE, Llama/Mistral, Qwen3.5 hybrid (Gated DeltaNet),
DeepSeek-V2/V3 (MLA), and Kolibri-1.

---

## 3. The proof (memorize these)

### Cold start, process launch → first token
Dedicated AWS EC2 `g4dn.xlarge` (**Tesla T4**), external `/usr/bin/time -v`, same GGUF
(`Qwen3-0.6B-Q4_K_M.gguf`), same prompt, same session.

| engine | p50 | p95 | vs Reflex |
|---|---|---|---|
| **Reflex** | **0.830 s** | 0.840 s | — |
| llama.cpp `llama-simple` | 0.940 s | 0.960 s | 1.13× |
| llama.cpp `llama-cli` | 1.580 s | 1.590 s | 1.90× |
| Ollama (cold daemon + model) | ~2.1 s | — | 2.6× |
| vLLM 0.30.0 (cold, `torch.compile`) | 127 s | 39.8–65.5 s warm cache | 48–150× |

`n=30` for Reflex and llama.cpp, `n=3` for Ollama/vLLM. All 30 Reflex runs emitted the
same golden token — it isn't just fast, it's correct.

### Where Reflex's 0.83 s goes
`reflex generate`, n=30, p50: process launch 127 ms · gguf parse 37 ms · CUDA init
141 ms · model load 235 ms · prompt eval (first token) 290 ms. The AOT bet shows up
inside CUDA init/model load — no JIT tax hiding there.

### Energy (Reflex emits this; competitors don't)
Real T4, NVML, **device-wide**: `system1` **20.3 J** and `generate` **32.2 J** to first
token (counter mode). Only Reflex reports joules, so there is **no competitor energy
column** yet — don't imply one exists.

### Also true, useful for credibility
- It reads GGUF directly (no conversion step), and can pull from the Hub.
- Multi-arch **fatbin** build: one image, many GPU generations, no per-arch rebuild.
- Embeddings surfaces: C FFI, line-JSON IPC (`stdio`/`uds`), Docker; the Python (PyO3)
  bindings do not currently build; an OpenAI-compatible HTTP **sidecar** exists out-of-tree.

---

## 4. What to say — talk tracks

### 30-second elevator
> "Serving stacks are optimized for throughput, but serverless bills you for startup.
> vLLM can take minutes to produce its first token cold; llama.cpp is fast but still
> pays real cold-load and teardown costs. Reflex is built entirely around that moment:
> AOT-compiled kernels loaded with no JIT, overlapped model load, and a clean process
> exit. On a T4 it's 0.83 seconds from launch to first token — measured, n=30 — and it
> reports its own energy-to-first-token."

### Technical buyer track
1. **Lead with the mechanism, not the adjective:** "kernels compiled by `nvcc` at build
   time, loaded via the driver API; module load is ~3 ms." This is checkable.
2. **Then the honest comparison:** "Against llama.cpp, which is *also* AOT, we're
   ~1.13× on its raw completion path — our edge there is cold-load engineering, not a
   magic kernel. The dramatic gap is against runtime-JIT serving stacks: ~48–150× vs
   vLLM."
3. **Then the instrumentation:** per-phase timings and NVML joules on every run. "We
   show our work."
4. **Then the fit:** "If you're scale-to-zero, single-shot, or interactive, this is the
   axis that matters. If you're saturating a warm cluster, you want vLLM."

### Investor / executive track
- **Wedge:** cold start is the underserved axis, and it's where serverless economics
  (per-second billing) make the value explicit.
- **Credibility:** every claim is traceable to a raw log; the project publishes its
  losses and caveats (including a comparison it deliberately reports as a loss).
- **Progress:** six model architectures (dense Qwen3, Qwen3-MoE, Llama/Mistral, hybrid
  Gated-DeltaNet, MLA, Kolibri-1) are implemented and real-hardware-verified;
  embeddability (C FFI/IPC/containers) is in place (Python bindings do not currently
  build); a ~2.6× cold-start improvement landed in a single concentrated perf round.
- **Defensibility, stated honestly:** the moat is execution speed + a measurement-first
  culture on an axis incumbents don't optimize for — not a patent. Say that plainly.

### Procurement / ops track
- **Deployment shape:** one binary, GGUF in, token/result out; Docker image and C FFI/IPC
  embedding surfaces exist (the Python bindings do not currently build).
- **Resource shape:** weights are held GPU-resident; see the README's weight-storage
  paragraph for the default `f16` vs. opt-in `f32` split.
- **No network surface in-core:** HTTP, if needed, is an optional sidecar; the engine
  itself never opens a socket. That's a deliberate security/simplicity property.
- **Bring your own model:** GGUF from llama.cpp's own converters; Qwen/DeepSeek family.

### Demo script (30 s, one GPU)
```
cargo build --release --bin reflex            # AOT kernels compiled at build time
./target/release/reflex smoke                  # proves the AOT pipeline + reports startup
./target/release/reflex generate model.gguf "The capital of France is" --max-tokens 1
# -> REFLEX_GENERATE_OK process_start_to_first_token_ms=... token_text=" Paris"
```

---

## 5. Objection handling

| Objection | Honest answer |
|---|---|
| "Isn't llama.cpp already fast and already AOT?" | Yes. Against its raw completion path we're ~**1.13×**, not 10×. Our larger wins are against runtime-JIT serving stacks and against *total cold lifecycle* (teardown, energy instrumentation). Don't oversell the llama.cpp row. |
| "vLLM is far faster at throughput." | Correct, and we don't compete there. vLLM wins warm/steady-state; Reflex wins cold start (~48–150×). Different axes. |
| "Why batch size 1?" | Deliberate. Reflex is built for single-shot/interactive/cold calls. Concurrency, queues, and multi-tenancy belong in a **host orchestrator**, not the engine — that's a stated non-goal, not a missing feature. |
| "Doesn't it use a ton of VRAM?" | It depends on the weight mode. The default is `f16` matrix weights (dequantized once at load, ~2 bytes per parameter); `--weights f32` keeps the exact `f32` reference (~4 bytes per parameter). Norms, routers, activations and the KV cache are `f32` in both modes. See the README's weight-storage paragraph; f16 is no longer merely planned. |
| "Cold start stops mattering once you keep servers warm." | True for steady traffic. The addressable case is bursty / scale-to-zero / single-shot, where warm capacity is waste and cold latency is the invoice. |
| "How does it compare to Jev / always-warm managed APIs?" | Different product (a managed decision API, not a generative engine). Cold-to-decision we **lose** by construction (~2.5–18× slower on T4) because they never load anything. Warm compute-only we're within ~1.3–2×; over the network ~10% apart at p50. Say it as a loss. |
| "Can I trust these numbers?" | Every figure is external `/usr/bin/time -v`, with n and p50/p95 stated, raw logs kept, and the caveats (including session-to-session variance) published alongside. |

---

## 6. Do-not-claim list (say these wrong and you lose technical audiences)

- **Don't** quote the old **1.3–1.4× vs llama.cpp** number. It was measured on
  GPU-virtualized ThunderCompute A6000, where every CUDA process paid a multi-second
  teardown tax the dedicated T4 doesn't. Use **1.13× vs `llama-simple`**.
- **Don't** imply the AOT advantage applies to llama.cpp — it is also AOT-compiled.
- **Don't** call the vLLM comparison weight-for-weight: vLLM 0.30.0 has no GGUF support,
  so it ran on the **HF safetensors** checkpoint (disclosed deviation).
- **Don't** quote single-run or best-case numbers. Use p50/p95 with n.
- **Don't** attach a joules column to competitor rows — only Reflex emits NVML, and the
  figure is **device-wide**.
- **Don't** claim throughput / batching / multi-tenant leadership. Those are non-goals.
- **Don't** present any single cold-start number as lab-controlled across sessions —
  rented-GPU session-to-session variance can exceed intra-session variance.

---

## 7. Positioning map

| Alternative | They win | Reflex wins |
|---|---|---|
| llama.cpp / Ollama | ecosystem, breadth, maturity | cold-load engineering, energy telemetry, ~1.13–2.6× cold |
| vLLM / TGI / TensorRT-LLM | steady-state throughput, batching | cold start (~48–150×), no runtime JIT |
| Always-warm managed APIs (e.g. Jev) | instant first decision (no load at all) | cost, control, self-hosting, one cold process |
| Serverless GPU platforms | provisioning, scale | Reflex is the *engine* those platforms run; complementary |

## 8. Non-goals (state these confidently — they're features, not gaps)

`batch_size` is always 1. No request queue/scheduler, no continuous batching, no
multi-tenant LoRA router, no KV-cache manager, no in-core HTTP/gRPC server. Multi-tenancy
and persistent state live in a host orchestrator. The allowed local surface is
sequential, non-network IPC (`stdio`/`uds`) plus in-process bindings (C FFI; the PyO3
bindings do not currently build).

## 9. Quick facts for Q&A

- **Name / form:** Reflex — one `reflex` binary with subcommands (`smoke`, `generate`,
  `system1`, `bench`, `check`, `doctor`, `stdio`, `uds`).
- **Input:** GGUF, read directly (any llama.cpp-produced GGUF; Qwen/DeepSeek families).
- **Kernels:** AOT `nvcc`; portable PTX by default, `REFLEX_CUDA_ARCH=sm_XX` for pinned
  cubin, `REFLEX_CUDA_ARCHS=...` for a multi-arch fatbin.
- **Verified hardware to date:** ThunderCompute A6000, AWS T4 (`g4dn.xlarge`), L40,
  A100 (80 GB, for DeepSeek-V2-Lite MLA).
- **Embedding:** C FFI (`libreflex_engine`), stdio/UDS IPC, Docker; the PyO3
  (`--features python`) bindings do not currently build; OpenAI-compatible HTTP sidecar
  out-of-tree.
- **Benchmark provenance:** `scripts/bench_cold_common.sh`; chart at
  `docs/cold-start-t4.svg`; raw logs under `bench-results/` (gitignored).

---

### Appendix — exact benchmark commands (for the "how do you know?" question)

```
# Reflex
reflex generate Qwen3-0.6B-Q4_K_M.gguf "Once upon a time" --max-tokens 1

# llama.cpp examples/simple (true prompt-in/token-out)
llama-simple -m Qwen3-0.6B-Q4_K_M.gguf -n 1 -ngl 99 --no-warmup "Once upon a time"

# llama.cpp llama-cli (applies chat template even with -p)
llama-cli -m Qwen3-0.6B-Q4_K_M.gguf -p "Once upon a time" -n 1 --temp 0 -ngl 99 \
  --no-warmup -st --simple-io

# Ollama / vLLM
scripts/bench_cold_ollama.sh <model> 3
scripts/bench_cold_vllm.sh Qwen/Qwen3-0.6B 3
```
