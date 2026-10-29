# Reference

How to build and run Reflex, and the contracts its outputs follow. The
[README](../README.md) is the overview; [benchmarks.md](benchmarks.md) has the
measurements; [DEVELOPMENT.md](DEVELOPMENT.md) is for contributors.

Contents: [Subcommands](#subcommands) · [Build features](#build-features) ·
[Kernel build modes](#kernel-build-modes) · [Weight storage](#weight-storage) · [Examples](#examples) ·
[Context-length limit](#context-length-limit) · [`--json` output contract](#--json-output-contract) ·
[JSON Schema output](#json-schema-output) · [KV cache files](#kv-cache-files) ·
[Error categories](#error-categories) · [Host-side building blocks](#host-side-building-blocks) ·
[`reflex model-info`](#reflex-model-info) · [Docker](#docker) · [Runpod](#runpod) ·
[Kubernetes](#kubernetes)

## Subcommands

`reflex` is a single binary with subcommands: `generate` (load a GGUF and generate
tokens), `system1` (single-pass, non-autoregressive candidate scoring — the "System 1"
decision-loop path), `embed` (embed one text with an embedding model, see
[Embeddings](#embeddings)), `smoke` (the AOT-pipeline check above), `bench` (warm-latency
microbenchmark), `check` (byte-exact-vs-reference correctness check, CI-scriptable),
`doctor` (GPU/CUDA health report), `model-info` (a GGUF's license, source and sha256,
no GPU; see [`reflex model-info`](#reflex-model-info)) and `stdio`/`uds` (local
JSON-line IPC, both need `--features ipc`). Run `reflex
<subcommand>` with no further arguments to see that subcommand's own usage. For real
HTTP clients (OpenRouter, the OpenAI SDKs, curl), see
[`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) — a separate
OpenAI-compatible `/v1/chat/completions` sidecar built on top of `reflex stdio`, not
part of the `reflex` binary itself (see Non-goals below).

## Build features

Reflex's **core engine** has a deliberately minimal dependency footprint: `cudarc`
(the CUDA driver + cuBLAS bindings), `half`, `memmap2`, and `rand` — no networking,
no serialization, no Python. Every optional capability is a **Cargo feature** that
stays off by default, so `cargo build --release` and `cargo build --release
--no-default-features` are identical today: neither pulls in a single optional
dependency (the empty `default = []` in `Cargo.toml` makes that a contract, not an
accident of omission). `--all-features` is a *developer convenience for testing the
feature matrix*, not a release build — see the table for what it drags in.

| feature | adds | extra host build dependency |
|---|---|---|
| *(none — core)* | `generate`/`system1`/`smoke`/`bench`/`check`, every supported model architecture | CUDA toolkit only (`nvcc`) |
| `ipc` | `reflex stdio` / `reflex uds` (also enables `json-output` via `serde`) | none |
| `json-output` | `--json` on `generate`/`system1`/`bench`/`smoke`/`check`/`doctor`/`model-info` | none |
| `download` | `--model <org/repo:file.gguf>` / `--quickstart` (HF downloader via `hf-hub`) | `libssl-dev` + `pkg-config` on Linux (**not** Windows/macOS, where `hf-hub` uses a different TLS backend) |
| `nvml` | `reflex` energy instrumentation and the `reflex-energy` whole-process meter (dlopen'd `libnvidia-ml`, never linked) | none |
| `python` | PyO3 bindings (`src/python.rs`) | a Python 3.8+ interpreter on the build host (`pyo3-build-config` probes for it) |

The host-dependency story in one line: **only `download` (Linux) and `python` ever
need anything the core build doesn't**, and both are off by default. On Ubuntu/Debian
for the `download` feature specifically:

```
sudo apt-get install -y libssl-dev pkg-config
```

Every Dockerfile in this repo builds an explicit, minimal feature set (`--features
ipc,nvml` for the sidecar/Runpod/Modal images, `nvml` for the root image's CLI target,
which also ships `reflex-energy`) precisely so the released artifacts never pay for
`download`/`python` they don't use — see each Dockerfile's `REFLEX_FEATURES`/`--features`
line rather than assuming `--all-features`. `nvml` adds no build dependency and no
runtime requirement: where `libnvidia-ml` is missing the binary runs without energy
figures (see [Docker](#docker)).

**Binary size.** The core build stays small by design: the Rust binary itself is on the
order of a few MB, and the AOT kernel bytes it embeds are tiny (the full `src/kernels_cuda/`
source is ~87KB; even a multi-arch fatbin stays far under the ~15MB ceiling this project
targets — well below the multi-GB CUDA base images the kernels are *not* re-shipped
inside). This is a stated target, not a CI-enforced number yet; a `REFLEX_SKIP_CUDA=1`
dev build (empty placeholder kernels) measures ~1.5MB, and the real CUDA build's exact
figure is re-confirmed per release rather than asserted here.

## Kernel build modes

Every CUDA kernel is compiled **ahead of time** (`build.rs` invokes `nvcc`, see
`build.rs` and `src/kernels_cuda/`), never at runtime via NVRTC. `src/aot.rs` loads the
precompiled bytes at process start via the CUDA driver API. Three output modes, chosen
by env var (`REFLEX_CUDA_ARCH` and `REFLEX_CUDA_ARCHS` are mutually exclusive):

| env var | output | load behavior | fits |
|---|---|---|---|
| *(neither)* | portable **PTX** | driver JIT-to-SASS at load, any GPU; without a warm driver JIT cache (a fresh container) every start pays ~0.8 s on a T4 | the `cargo build` default; local development |
| `REFLEX_CUDA_ARCH=sm_XX` | single-arch **cubin** | zero JIT, exactly that one GPU, hard-fails elsewhere | a known, pinned SKU |
| `REFLEX_CUDA_ARCHS=sm_XX,sm_YY,...` | **fatbin** (the Docker images' default) (one cubin per listed arch + an embedded forward-compatible PTX) | zero JIT on any listed arch, driver-JIT fallback on anything newer | a mixed-architecture GPU pool (e.g. Runpod's `AMPERE_16`, see [below](#runpod)) |

The fatbin mode is build.rs's answer to "ship one image, run natively on many GPU
generations without a per-arch rebuild": `nvcc -fatbin` with one
`-gencode arch=compute_XX,code=sm_XX` per listed arch, plus a trailing
`-gencode arch=compute_<highest>,code=compute_<highest>` that embeds PTX for the
highest listed arch so a GPU *newer* than everything listed still loads (via driver
JIT) instead of failing. Example:

```
REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90 cargo build --release
```

`cargo build` panics if *both* `REFLEX_CUDA_ARCH` and `REFLEX_CUDA_ARCHS` are set.
A fatbin embeds one SASS image per arch, so it is fatter than a single pinned cubin —
the traded-off binary size is the honest cost of not maintaining one image per GPU
generation. `reflex doctor` reports the build's `kernel_format` and, for a fatbin,
whether the detected GPU gets a native (zero-JIT) image or falls back to the embedded
PTX.

## Weight storage

Matrix weights are dequantized once at load and stored on the GPU as `f16` by default:
about 2 bytes per parameter of VRAM, and half the weight bytes read per decoded token.
`f32` keeps the exact reference values at about 4 bytes per parameter. Norms, MoE
routers, activations and the KV cache are `f32` either way.

| how | applies to |
|---|---|
| `--weights f16` / `--weights f32` | `generate`, `system1`, `bench`, `check`, `stdio`, `uds`; wins over the env var |
| `REFLEX_WEIGHTS=f16` / `f32` | the same subcommands, the C FFI's `reflex_load` and Python's `PyModel` |
| `PyModel(path, weights="f32")` | the Python bindings |
| `--weights` on `reflex-openai-adapter` | passed through to its `reflex stdio` child |

With neither set the default is `f16`, except `reflex check`, which defaults to `f32`
because its byte-exact comparison is defined on the exact dequantized weights. Every
model-loading subcommand reports the mode as an additive `weights_dtype=` field on its
`REFLEX_*_OK` lines (and in `--json`), and on `stdio`/`uds`'s `READY` line.

f16 decode reads f16 weights and accumulates in f32. Prefill casts each GEMM's activations
to f16 for `cublasGemmEx`. f16's largest finite value is 65504; the cast saturates there
instead of producing inf. `generate`/`system1`/`bench`/`check` print a warning on stderr
if anything was clamped. `REFLEX_F16_ACT_STATS=1` prints `REFLEX_F16_ACT_STATS max_abs=..
saturated=..` after every run. If you see the warning, use `--weights f32`. The
largest measured activation was 15,420, on Qwen3-1.7B, 4.2x below the limit. Qwen3-0.6B
and Qwen3-4B peaked near 4,000, so activation size doesn't follow model size. Check a
model that isn't [measured](benchmarks.md#f16-weight-storage) with `REFLEX_F16_ACT_STATS=1`.

LoRA adapters given with `--lora` are merged in f32: their target weights load as `f32`,
take the delta, and are rounded to f16 once.

Diagnostics: `REFLEX_F16_ROUNDTRIP=1` with `--weights f32` rounds matrix weights to f16
and back while keeping every f32 kernel, to separate weight rounding from the f16 kernels.
`REFLEX_TOP2_TRACE=1` prints the top two logits of every generated position.

## Examples

### Download a model from Hugging Face, then run a System1 test

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

(That last capture is from an early pre-optimization build — the `token_id`s and scores
are current, but the `process_start_to_result_ms` figure predates the cold-load
optimizations; see the [phase breakdown](benchmarks.md#cold-start-phase-breakdown) for
current numbers.)

`probability` is relative to this candidate set only, not a vocab-wide probability —
see `Model::system1_evaluate`'s doc comment in `src/model/mod.rs`. `system1` supports
every architecture; on the Qwen3.5 hybrid models each candidate must be a single token
(multi-token candidates are rejected with an `invalid_input` error).

Over IPC (`reflex stdio`/`uds`), a request with `"candidates": [...]` runs the same
scoring. Its response lists each candidate's `token_ids`, `score` and `probability`,
plus `entropy` and `chosen` (the index of the most probable candidate). It can score
after a cached prefix: with `"use_pinned": true` (the session's pinned prefix) or
`"import_kv": "<file>"` (not both), `prompt` is the continuation after the prefix and
must add at least one token. The token ids and `chosen` match one request over the
joined text, and the scores match it up to float summation order.
`pin_prefix` and `export_kv` are `generate`-only (`invalid_input` here). Gemma 4 and
Granite 4.0-H score fresh prompts only (`unsupported_architecture` with a prefix). With
`"timings": true`, every response also reports the pinned slot after the request:
`pinned_seq_len` positions and `pinned_bytes` of VRAM. See
[design/agent-relay.md](design/agent-relay.md), "System-1 over a resumed prefix".

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

## Context-length limit

A single sequence can hold at most **65,535 positions** in total: any imported KV cache
(`--import-kv`) + the encoded prompt + `--max-tokens` (or System1's longest candidate).
That covers Qwen3's 40,960-token native context. The bound is CUDA's 65,535-block limit
on a launch's `y`/`z` grid dimension, which MLA's batched prefill uses per prompt row (see
`src/limits.rs`). Past it, the KV cache's VRAM is the next constraint, and an allocation
that doesn't fit fails as `out_of_memory`. Requests over the limit are rejected before
any GPU allocation with an error starting `context length exceeded:` and category
`context_overflow` (see [Error categories](#error-categories)) — from `reflex generate`/
`system1`, the IPC `error`/`error_kind` fields, the C FFI's `reflex_last_error()`/
`reflex_last_error_code()` (the limit is `REFLEX_ATTN_MAX_POSITIONS` in the header), and
as an HTTP `400` with code `context_length_exceeded` from the OpenAI sidecar.

## `--json` output contract

With `cargo build --release --features json-output`, `generate`/`system1`/`smoke` (and
`bench`/`check`/`doctor`) print one JSON object per existing `REFLEX_*_OK` stdout line
instead of the plain `key=value` text — same call site, same order, one object per line.
Field names match the plain-text keys 1:1, with one addition: the versioned result
objects also carry a **`schema_version`** field so a consumer can detect a shape change.

```json
{"schema_version":"1.0.0","process_start_to_first_token_ms":456.4,"gguf_open_ms":37.3,...}
```

- **Current version: `1.4.0`** (`SCHEMA_VERSION` in `src/cli_output.rs`). 1.1.0 added
  `weights_dtype` to the `generate`, `system1`, `bench` and `check` result objects.
  1.2.0 added `kv_io_ms` to `generate`'s result: the time spent reading the
  `--import-kv` file or writing the `--export-kv` file, present only with one of them
  (the plain-text line gains the same `kv_io_ms=` field). The read includes uploading
  the file into the model's cache (see [KV cache files](#kv-cache-files)).
  1.3.0 added `embed`'s `REFLEX_EMBED_OK` result (with the vector in `embedding`).
  1.4.0 added `model-info`'s result (see [`reflex model-info`](#reflex-model-info)).
- The six versioned objects (with `embed`'s and `model-info`'s results) are `generate`'s `REFLEX_GENERATE_OK` result, `system1`'s
  `REFLEX_SYSTEM1_OK` result, `smoke`'s `REFLEX_SMOKE_OK` result, and each additive
  `REFLEX_PHASE_OK` phase object. Per-item lines (`REFLEX_SYSTEM1_CANDIDATE_OK`,
  `REFLEX_LORA_OK`, `REFLEX_DOCTOR_CHECK`) and the other subcommands' result structs
  (`bench`'s `REFLEX_BENCH_*`, `check`'s `REFLEX_CHECK`, `doctor`'s
  `REFLEX_DOCTOR_OK`/`_FAIL`) still do not carry it. Extending `schema_version` to those
  structs is an additive change that was deliberately **deferred** (2026-09-29): it
  touches `bench`/`check`/`doctor`'s output shapes and the `SCHEMA_VERSION` constant for no
  measurement benefit, and every one of them already ignores-unknown-fields safely. It
  remains a mechanical follow-up, not a contract gap.
- **Additive / forward-compatible**: a *minor* bump only adds fields, so a reader that
  ignores unknown fields keeps working; a *major* bump signals that an existing field
  changed meaning or was removed.
- Plain-text output (no `--json`) is byte-for-byte unchanged by this contract, and the
  existing `REFLEX_*_OK` fields are unchanged by a `schema_version` bump.

## JSON Schema output

A `generate` can be constrained to one JSON value that validates against a JSON
Schema. The schema is applied at sampling time: tokens that can't continue a
conforming value get a `-inf` logit, before greedy or temperature/top-k/top-p
sampling. It works for every architecture.

| surface | how |
|---|---|
| IPC (`reflex stdio`/`uds`) | `"json_schema": {...}` on a request, next to `"prompt"` |
| CLI | `reflex generate <gguf> <prompt> --max-tokens N --json-schema <file>` |

- **Supported keywords:** `type` (a name or an array of names), `properties`,
  `required`, `additionalProperties` (`false` or omitted), `items`, `enum`, `const`,
  `anyOf`, and a local `$ref` (`#` or `#/$defs/<name>`). The annotations `title`,
  `description`, `$comment`, `$schema`, `default` and `examples` are accepted. Any
  other keyword is rejected with `invalid_input`, and the message names it.
- **Output format:** compact JSON. The only whitespace outside strings is one
  optional space after each `:` and `,`. Properties come in schema order, which is the order they appear in the request (a client library
  that sorts JSON keys sorts them). When `properties` is given, no other properties
  are generated. The generation ends at EOS as soon as the value is complete.
- **Result:** `text` (IPC) or `token_text` (CLI) is the value alone; the token ids end
  with EOS.
- **Combinations:** works with `use_pinned` and `import_kv`; only the generated text is
  constrained. Combining it with `candidates`, `pin_prefix` or `export_kv` (CLI:
  `--export-kv`) is `invalid_input`.
- **Running out:** if `max_tokens` ends before the value is complete, the request
  fails with `grammar_dead_end`. On IPC the error line carries the partial `text` and
  `token_ids`; the CLI prints `partial_text=...` to stderr.
- **Timings:** with `"timings": true`, `mask_ms_total` is the masking time summed over
  the request.
- **Cost:** the first constrained request in a process builds a token index of the
  vocabulary and keeps it. Unconstrained requests never build it.

Design, format rationale and measurements:
[design/agent-relay.md](design/agent-relay.md), "JSON Schema constrained decoding".

## Stop tokens

A `generate` stops at the GGUF's `tokenizer.ggml.eos_token_id`. An IPC request can add
more stop tokens with `"stop_token_ids": [<id>, ...]`: the generation then also ends at
the first of those, which is generated and returned like EOS (its id ends
`token_ids`, its text ends `text`). Chat models often end a turn on a token that isn't
their EOS (Qwen2.5 and Qwen3.5 `<|im_end|>`/`<|endoftext|>`, Gemma 4 `<turn|>`), and
the OpenAI adapter sends each model's end-of-turn tokens here (see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md)). Without the
field, stopping is EOS-only, which is also what `reflex check` compares against
llama.cpp with.

## Embeddings

A GGUF with `<arch>.attention.causal = false` is an embedding model: attention is
bidirectional and the output is the final hidden state, mean-pooled and L2-normalized.
Supported: `mistral3` encoders, i.e. NVIDIA's Nemotron-3.5-Embed-1B (text only; the
`mmproj` vision file is not supported). Other pooling types are rejected.

| surface | how |
|---|---|
| CLI | `reflex embed <gguf> "query: What is NeMo?"` prints `REFLEX_EMBEDDING_OK dims=<n> values=[...]` and `REFLEX_EMBED_OK` with the phase timings (`--json`: one object with `embedding`) |
| IPC (`reflex stdio`/`uds`) | `{"prompt": "query: What is NeMo?", "embed": true}` answers with `embedding` and `prompt_tokens` |
| C FFI | `reflex_embed(handle, text, &out)`, then `reflex_free_embed_result(&out)` |
| HTTP | the sidecar's `POST /v1/embeddings` ([`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md)) |

- **Input format:** the engine embeds the text exactly as given. Nemotron-3.5-Embed
  expects its retrieval prefix: `"query: "` for a search query, `"passage: "` for a
  document. Its chat template also puts U+200C after every `[` and `<` of the text,
  and its tokenizer strips those marks again; that only keeps user text from forming a
  special token. The HTTP sidecar renders inputs through that template for you.
- **One text per call.** A list of texts is one call each, in order: the engine never
  batches (see Non-goals).
- **Truncation:** the model is trained so a prefix of the vector, re-normalized, is a
  smaller embedding. The sidecar's `dimensions` does that; the engine always returns
  the full width.
- **Weights:** `Q4_K`/`Q6_K` stay quantized on the GPU (the default); BF16/F16/Q8_0
  files are dequantized to `--weights` (f16 by default, `f32` for the closest match to
  the reference).
- **Not for generation:** `generate`, `system1`, cache files and pinned prefixes return
  `invalid_input` on an embedding model, and `embed` returns it on a language model.
  The f16 K/V cache (`REFLEX_KV_CACHE=f16`) and `REFLEX_ATTN_KERNEL=legacy` are refused
  at load.

Design, numerics and results: [design/embeddings.md](design/embeddings.md).

## Decision models

A decision model answers typed questions about a piece of text in one forward pass per
question, with no generation: each answer is a probability distribution over the
options, read from the logits of one label token per option at the end of the prompt.
Supported: the label-token types on the Qwen3.5 hybrid path, `lev` (ggml-org's
`lev-GGUF`) and `openjev` (OpenJev's text input). `kev` and `laya` models and images are
not supported yet.

| surface | how |
|---|---|
| HTTP | the sidecar's `POST /v1/systemone`, llama.cpp's TypeSafe System One API: `state` plus typed `questions` (`choice`, `score`, `noul`) in, answers with probabilities and confidences out ([`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md#post-v1systemone)) |
| IPC (`reflex stdio`/`uds`) | `label_prompts`, below: the engine half, raw logits only |

The sidecar renders the GGUF's `systemone` template for each question, sends every
prompt in one IPC request and turns the logits into answers. The engine request:

```json
{"id": "q1", "label_prompts": [
  {"prompt": "<|im_start|>user\n...<|im_start|>assistant\n<think>\n\n</think>\n\n", "labels": ["A", "B", "C"]},
  {"prompt": "<|im_start|>user\n...", "labels": ["A", "B"]}
]}
```

```json
{"id": "q1", "event": "final", "ok": true, "shared_tokens": 67,
 "labels": [{"tokens": 113, "token_ids": [32, 33, 34], "logits": [21.43, 13.54, 12.89]},
            {"tokens": 109, "token_ids": [32, 33], "logits": [9.97, 6.04]}]}
```

- **Prompts** are encoded like every other prompt (special-token text parsed). Each
  label is encoded alone, with special-token text as plain text, and must be exactly
  one token; anything else fails the request with `invalid_input`. At most 256 prompts
  and 255 labels per prompt.
- **Results** come back in prompt order: `tokens` is the prompt's full length,
  `token_ids` the labels' tokens, `logits` their raw LM-head logits at the last
  position (no temperature, no softmax).
- **Shared prefix:** the longest token prefix common to all prompts (leaving each at
  least one token) is evaluated once, and each prompt continues from a copy of that
  state, one after another. `shared_tokens` is its length (0 for one prompt). The copy
  lives only for the request: the session's [pinned prefix](#kv-cache-files) is
  untouched, and `label_prompts` doesn't combine with `pin_prefix`/`use_pinned`.
- **Combines with** `id` and `timings` only; any other field fails with
  `invalid_input`. Architectures other than Qwen3.5 hybrid fail with
  `unsupported_architecture`.
- **Speed:** on a T4, the support-ticket example on lev `Q8_0` (283 evaluated tokens)
  takes 330 ms warm (llama-server: 300 ms) and 1.7 s from process start to the first
  answer (llama-server: 3.5 s).

Prompts, tokens and probabilities match llama.cpp on lev and on two small test models;
design and results: [design/decision-models.md](design/decision-models.md).

## KV cache files

`reflex generate --export-kv <file>` (IPC `export_kv`) writes the model's state after
the prompt; `--import-kv <file>` (IPC `import_kv`) resumes from it. The format is a
small header (`CSKV` magic, a version, the shape) followed by the values, little
endian. It's home-grown and versioned, not a stable public spec. Every version below
still imports.

| version | layout | values |
|---:|---|---|
| 1 | dense/MoE (Qwen3, Qwen2, Llama, Mistral, Kolibri) | f32 |
| 2 | Qwen3.5 hybrid: attention K/V, GatedDeltaNet conv and recurrent state | f32 |
| 3 | MLA (DeepSeek-V2/V3): one latent cache per layer | f32 |
| 4 | dense/MoE, version 1 plus one byte for SentencePiece continuations | f32 |
| 5 | version 4 | f16 |
| 6 | version 2 | attention K/V f16, GatedDeltaNet state f32 |
| 7 | version 3 | f16 |

- **Exports write f32 (versions 4, 2, 3) unless asked for f16.** The CLI flag is
  `--export-kv-dtype f16`, the IPC field `"export_kv_dtype": "f16"`. An f16 file is about
  half the size (Qwen3-0.6B, 1,489 positions: 171 MB against 342 MB) and reads faster,
  but resuming from it is no longer exact: the cached values are rounded to f16. On a
  T4 that moved the next token's logits by about 7e-4 (relative L2) and left the greedy
  tokens unchanged on every test prompt. The working cache stays f32 either way.
- **The rounding happens on the GPU**, and so does widening back to f32 on import. A
  value outside f16's range (|x| > 65504) fails the export with `kv_cache` instead of
  becoming infinite.
- **The hybrid keeps its recurrent state f32** (version 6). It's a fixed size per model,
  so a short hybrid prefix's f16 file is barely smaller than its f32 one.
- **Import reads the header first** and rejects a file whose length doesn't match it
  (`kv_cache`), before any model work. The values are then read straight into the
  engine's pinned staging buffers on their way to the GPU. Don't change a file while a
  request resumes from it: a file that shrinks in between fails that request.
- **`kv_io_ms`** reports the export's write, or the import's read, which includes the
  upload into the model's cache, since the two overlap.

Gemma 4 and Granite 4.0-H have no cache format and return `unsupported_architecture`.
Design and measurements: [design/agent-relay.md](design/agent-relay.md), "Faster cache
files (phase F)".

## Error categories

Every engine error carries a stable category alongside its message, so callers can branch
on *why* something failed instead of matching message text (the text itself is unchanged
and still safe to grep). The categories are `src/error.rs`'s `ReflexError` variants:

| category (IPC `error_kind`) | C `ReflexErrorCode` | meaning |
|---|---|---|
| `gguf` | `REFLEX_ERROR_CODE_GGUF` (1) | malformed GGUF: bad header, missing metadata or tensors |
| `unsupported_architecture` | `..._UNSUPPORTED_ARCHITECTURE` (2) | valid model this engine doesn't support |
| `cuda` | `..._CUDA` (3) | a CUDA driver call failed, or this build's kernels can't run on the GPU |
| `cublas` | `..._CUBLAS` (4) | a cuBLAS call failed |
| `out_of_memory` | `..._OUT_OF_MEMORY` (5) | a GPU allocation failed |
| `tokenizer` | `..._TOKENIZER` (6) | encoding or decoding failed |
| `context_overflow` | `..._CONTEXT_OVERFLOW` (7) | over the [context-length limit](#context-length-limit) |
| `kv_cache` | `..._KV_CACHE` (8) | an imported KV cache is malformed or doesn't match the model |
| `lora` | `..._LORA` (9) | a LoRA adapter is malformed or doesn't match the model |
| `io` | `..._IO` (10) | a file couldn't be opened, created or written |
| `invalid_input` | `..._INVALID_INPUT` (11) | a bad argument (empty prompt, `max_new_tokens` 0, NULL pointer, bad sampling value) |
| `other` | `..._OTHER` (12) | anything else, mostly internal invariant violations |
| — | `..._PANIC` (13) | the engine panicked (C FFI only; it catches the panic) |
| `grammar_dead_end` | `..._GRAMMAR_DEAD_END` (14) | a [`json_schema`](#json-schema-output) generation ran out of `max_tokens` before its value was complete |

Where they surface:

- **IPC** (`reflex stdio`/`uds`): failed responses add `"error_kind"` next to `"error"`;
  successful ones omit it.
- **C FFI**: `reflex_last_error_code()` returns the code (`REFLEX_ERROR_CODE_OK`, 0, after
  a successful call) next to `reflex_last_error()`'s message. Both are reset at the start
  of every `reflex_*` call.
- **OpenAI sidecar**: `context_overflow` and `invalid_input` become `400`, `out_of_memory`
  `503`, everything else `500` with the category as the error `code`.
- **CLI and Python**: the message only, as before.

Codes and category names are stable: existing ones never change meaning, and new ones
are only appended.

## Host-side building blocks

- `src/gguf.rs` — GGUF metadata/tensor-directory parsing (mmap-based).
- `src/dequant.rs`, `src/dequant_iq.rs`, `src/dequant_iq_tables.rs` — standard and
  i-quant dequantization, verified byte-exact against `gguf-py`.
- `src/tokenizer.rs` — verified against real sentencepiece/BPE references. For
  `gpt2`-model vocabs the split regex follows `tokenizer.ggml.pre`. Accepted values:
  `qwen2`, `kolibri1`, `qwen35`, `llama-bpe`, `gpt-2`, `deepseek-llm`, `deepseek-v3`,
  `gigachat`, `glm4`.
  Any other value fails to load, with the value named in the error. See
  [design/tokenizer-pre-types.md](design/tokenizer-pre-types.md).
  `src/tokenizer_unicode.rs` holds llama.cpp's Unicode category tables, generated by
  `scripts/gen_unicode_tables.py`.
- `src/grammar.rs` — [JSON Schema output](#json-schema-output): the schema compiler,
  the byte-level pushdown matcher and the vocabulary trie that token masks walk.

None of these care how kernels get compiled — they're pure host-side GGUF/tokenizer
logic. There is deliberately no NVRTC runtime-compile-and-load path anywhere in this
codebase — that's the thing this project's AOT design replaces, not reuses.

## `reflex model-info`

```
reflex model-info <path-to-gguf> [--sha256] [--json]
```

Prints what a GGUF says about itself, for provenance records (model cards, the Hub
worker's attestation): every `general.*` metadata key the converter wrote (name,
license, source URL, organization, quantizer, base models, ...), the architecture and
`general.file_type`. It parses only the header (mmap): no model load, no GPU, so it
works in a `REFLEX_SKIP_CUDA=1` build. `--sha256` also hashes the whole file (seconds
for a multi-GB model, so it is opt-in). Values are reported exactly as the file states
them; a missing `general.license` stays missing.

Plain text is one `REFLEX_MODEL_INFO_GENERAL key=general.<key> value="<value>"` line per
key, in key order, then:

```
REFLEX_MODEL_INFO_OK path="model.gguf" size_bytes=366195616 gguf_version=3 tensor_count=402 architecture=granitehybrid file_type=7 file_type_name=Q8_0 sha256=c7d98736...
```

`--json` (needs `--features json-output`; the adapter images have it) prints one
object instead (`ModelInfoJson`, schema 1.4.0):

```json
{"schema_version":"1.4.0","path":"model.gguf","size_bytes":366195616,"gguf_version":3,
 "tensor_count":402,"architecture":"granitehybrid","file_type":7,"file_type_name":"Q8_0",
 "general":{"license":"apache-2.0","base_model.0.organization":"Ibm Granite",...},
 "sha256":"c7d98736..."}
```

`general` is keyed without the `general.` prefix; arrays of scalars are joined with
`", "`. `file_type_name` is llama.cpp's name for the `llama_ftype` value, absent for an
unknown one. `sha256` appears only with `--sha256`. See
[docs/compliance/](compliance/README.md) for how this fits the other compliance records.

## Docker

A multi-stage `Dockerfile` is included: the builder stage has the full CUDA devel
toolkit (`nvcc`) to compile the AOT kernels; the runtime stage only needs the CUDA
*runtime* libraries, since every kernel byte is embedded directly into the compiled
binary at build time — the runtime image never runs `nvcc` and never needs the devel
toolkit.

```
# With no build-arg the kernels are a multi-arch fatbin (native on T4 through H100,
# Ada included). --build-arg REFLEX_CUDA_ARCH=sm_XX builds a single cubin for one GPU
# instead (it wins over the fatbin list). --build-arg REFLEX_CUDA_ARCHS= (empty) gives
# portable PTX, which a fresh container JITs on every start (~0.8 s on a T4; see
# "Core technical bet").
docker build -t reflex .

# The same Dockerfile also builds the OpenAI-compatible sidecar image and the Runpod
# load-balancing image: --target adapter / --target runpod-lb (see the Dockerfile header).

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

**Energy.** Every image is built with the `nvml` feature, so `generate`/`system1`/
`smoke`/`bench` print `joules=` and `energy_method=` (and the `REFLEX_PHASE_OK` lines
carry `energy_joules=`/`energy_method=`). NVML is the host driver's `libnvidia-ml.so.1`,
which nvidia-container-toolkit mounts only with the `utility` driver capability; the
`nvidia/cuda` base images set `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so `--gpus
all` is enough. Without it (another base, `-e NVIDIA_DRIVER_CAPABILITIES=compute`, or a
platform that doesn't mount NVML) the binary runs normally and leaves the energy fields
out, and `reflex doctor`'s `nvml_energy` check warns instead of failing. NVML's counter is
device-wide, so the figures need a dedicated GPU. The `reflex` image also contains
`reflex-energy`, which measures a whole process from outside, net of the GPU's idle draw
(see [benchmarks](benchmarks.md#energy)):

```
docker run --rm --gpus all -v /path/to/models:/models \
  --entrypoint /usr/local/bin/reflex-energy reflex \
  -- /usr/local/bin/reflex system1 /models/Qwen3-0.6B-Q4_K_M.gguf "Q: ...? A:" \
  --candidate " Yes" --candidate " No"
```

Build with `--build-arg REFLEX_FEATURES=` (empty) for the image without energy, or
`REFLEX_FEATURES=ipc,nvml` to add `stdio`/`uds` and keep it.

`--build-arg COMPLIANCE_BUNDLE=1` (any target) adds the SBOMs, third-party licenses,
apt package list and base image names under `/usr/share/reflex/compliance/`; the
default (`0`) leaves every image's contents unchanged. See
[docs/compliance/](compliance/README.md#the-compliance-bundle). The Rust toolchain all
Dockerfiles install is the one `rust-toolchain.toml` pins.

The CUDA major/minor version in both Docker stages must stay consistent with
`Cargo.toml`'s pinned `cudarc` feature (`"cuda-12000"`, i.e. CUDA 12.x) — a mismatch is
a build-time/runtime library version mismatch this Dockerfile can't catch for you.

### Published images

Two images are published to GHCR for the serverless deployment paths. **Pin a
digest, not `:latest`** — `:latest` is a moving alias, repointed on every rebuild
from `master`. Verified 2026-10-09:

| Image | Digest |
|---|---|
| `ghcr.io/lateos-ai/reflex-runpod` (root `Dockerfile`, `--target runpod-lb`) | `sha256:219edd36bd6b9bf8271d4ae91793eac205c625794e55ed325570318ce31af9b7` |
| `ghcr.io/lateos-ai/reflex-runpod-hub` (`.runpod/Dockerfile`) | `sha256:6ecb8e1f0200c2736141014eadc9ad38bee7560fad52b48e02dbacd31d510f33` |

The full operator walkthrough is [`docs/deploy-serverless.md`](deploy-serverless.md).

For a cost-optimized AWS pattern built on this same image (Spot GPU instances, an Auto
Scaling Group with minimum capacity 0, and a `reflex uds` sidecar reachable over a local
Unix Domain Socket instead of a network load balancer), see
[`docs/aws-deployment.md`](aws-deployment.md). For the opposite tradeoff -- an
always-warm On-Demand instance behind a public HTTPS load balancer with autoscaling,
built on the [`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) HTTP sidecar
instead of raw UDS -- see [`docs/aws-deployment-warm.md`](aws-deployment-warm.md).

## Runpod

The economics behind this deployment shape are covered in
[benchmarks](benchmarks.md#where-this-engine-competes); this section is the
mechanics.

[`serverless/runpod/`](../serverless/runpod/README.md) packages the existing
[`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md) — unmodified, no new
handler code — as a Runpod Serverless **load-balancing endpoint** (the endpoint type
that proxies HTTP straight to an arbitrary custom server, rather than the queue-based
type that requires a Python SDK handler). Two details that matter there:

- **`GET /healthz` is three-state**, specifically so a platform health check can
  measure readiness honestly: `204` while the managed `reflex` child is alive but still
  loading the model, `200` once it can actually serve a request, `503` if the child
  died. Without the `204` state, a platform would clock "ready" the instant the HTTP
  port binds — well before model load finishes — and report a cold-start number that
  isn't real.
- **The GPU arch pin differs from the AWS guides, and the pool name lies.** Runpod's
  cheapest serverless pool (`AMPERE_16`, $0.58/hr, verified from the live catalog) is
  *mixed-architecture*: RTX A4000/A4500 are Ampere (`sm_86`), but RTX 2000 Ada / RTX
  4000 Ada in the same pool are Ada (`sm_89`). A pinned cubin only runs on the compute
  capability it was built for, so an endpoint free to schedule anywhere in that pool
  fails nondeterministically. That applies to a single-arch build: the images now
  default to a multi-arch fatbin with native code for both, so they run anywhere in the
  pool; that was verified on real Runpod workers on 2026-10-02 (RTX A4500 `sm_86` and L4
  `sm_89`). `scripts/deploy_runpod.sh` still pins the SKU to **RTX A4500** by default, but
  only because benchmarks need one fixed SKU. A single-arch image for the AWS guides' T4 (`sm_75`) and one for the A4500
  (`sm_86`) are not interchangeable; the fatbin is.

The Runpod Hub's queue worker ([`.runpod/`](../.runpod/README.md)) has three opt-in
compliance options, all `off` by default:

| Option | Values | What it does |
| --- | --- | --- |
| `COMPLIANCE` | `off`, `on` | Every response gains a `compliance` block (model file sha256 and license, engine version, image digest, GPU, the job's usage when the adapter reports one (not for classify; a stream's chunk count instead) and wall time); a stream ends with a `compliance.attestation` event; `{"compliance": "report"}` returns the image's compliance bundle and a model card. A listed model whose sha256 doesn't match `.runpod/models.json` stops the worker. |
| `AUDIT_LOG` | `off`, `stdout`, `volume` | One JSON line per job, to the worker's logs or to `/runpod-volume/reflex-audit/<worker-id>.jsonl`. |
| `AUDIT_LOG_CONTENT` | `off`, `hash`, `full` | Whether audit lines leave out the prompt and answer, carry their sha256, or carry them as sent. |

Model files are hashed while they download (or at image build, for the baked model)
and the digest is kept next to the file, so these options add no hashing to later
cold starts. Shapes and details: [.runpod/README.md](../.runpod/README.md#compliance-records-and-audit-log)
and [docs/compliance/](compliance/README.md).

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
