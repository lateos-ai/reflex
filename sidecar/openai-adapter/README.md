# reflex-openai-adapter

An OpenAI-compatible `POST /v1/chat/completions` HTTP sidecar in front of the Reflex
core engine. This is the escape-hatch pattern the root project's `README.md`/
`CLAUDE.md` Non-goals sections describe and defer: **the core `reflex` engine never
grows a network socket or a thread pool** — this crate is a separate process, in its
own Cargo project, that translates real HTTP traffic (OpenRouter, the OpenAI Python/
JS SDKs, curl, anything that speaks the OpenAI Chat Completions API) into calls
against one managed `reflex stdio <gguf>` child process over the JSON-line IPC
protocol `src/ipc.rs` (in the repo root) defines.

## Why a separate crate

- **No new dependencies enter the core engine's dependency graph.** This crate has
  its own `Cargo.toml`/`Cargo.lock` and is deliberately *not* a member of the root
  `Cargo.toml` workspace (the root crate has no `[workspace]` table at all) and does
  not depend on the `reflex-engine` library crate. It never invokes `nvcc` and builds
  with a plain stable Rust toolchain — no CUDA toolkit required.
- **The core engine's `batch_size == 1`/strictly-sequential contract is preserved
  underneath.** This sidecar's HTTP side happily accepts concurrent connections
  (axum + tokio), but every request is funneled through a single background worker
  task that owns the one managed `reflex stdio` process's stdin/stdout exclusively —
  see `src/reflex_client.rs`'s module doc comment. The worker never dequeues the next
  HTTP-originated request until it has read the previous one's `"event": "final"` IPC
  line, so two racing HTTP requests can never interleave writes to the same IPC
  connection, and only ever one `reflex` process is spawned, not one per request.

## Usage

Build the core engine with the `ipc` feature first (from the repo root):

```
cargo build --release --features ipc
```

Then, from this directory:

```
cargo build --release
./target/release/reflex-openai-adapter <path-to-gguf> \
  --reflex-bin ../../target/release/reflex \
  --host 127.0.0.1 --port 8000
```

`--reflex-bin` defaults to `reflex` (resolved via `PATH`) if omitted. Full flag list:

```
reflex-openai-adapter <path-to-gguf> [--reflex-bin <path>] [--host <addr>]
  [--port <port>] [--lora <adapter.gguf>] [--model-name <name>]
  [--default-max-tokens <n>]
```

- `--lora <adapter.gguf>` is forwarded straight through to `reflex stdio`'s own
  `--lora` flag (see Phase 4/Embeddability in the root `README.md`).
- `--model-name` sets the `model` field returned in responses when a request doesn't
  specify one of its own; defaults to the GGUF file's stem.
- `--default-max-tokens` (default `256`) is used when a request omits `max_tokens`.

Once running, it logs `REFLEX_ADAPTER_READY addr=<host>:<port>` to stderr (the
managed `reflex` child's own stderr, including its `REFLEX_STDIO_READY` line, is
forwarded too, prefixed `[reflex]`).

### Example

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 8
  }'
```

Or with the OpenAI Python SDK, pointed at this sidecar instead of api.openai.com:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "The capital of France is"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

`api_key` is required by the SDK's client constructor but never checked or used —
this sidecar has no auth (see Known limitations).

## Scope

Implements `POST /v1/chat/completions` only, both non-streaming (JSON) and streaming
(`"stream": true`, Server-Sent Events, `chat.completion.chunk` objects terminated by
a `data: [DONE]` line) — see `src/main.rs::build_sse_stream`/`non_streaming_response`.
Also exposes `GET /healthz` (plain `"ok"` body) for basic liveness checks.

Request fields honored: `model`, `messages` (`role`/`content`, string content only),
`temperature`, `top_p`, `max_tokens`, `stream`, `seed`. `top_k` is accepted too as a
Reflex-specific extension (not part of the OpenAI schema) since `IpcSamplingParams`
supports it. `temperature` omitted or `<= 0` selects greedy argmax, mirroring
`IpcRequest::sampling_params`'s own rule.

## Known limitations

- **No chat template.** Reflex itself has no chat-template support — `Model::
  forward_prompt` takes a plain prompt string, not a structured message list. This
  adapter flattens `messages` into a prompt by simple role-labeled concatenation
  (`src/openai.rs::build_prompt`: `"System: ...\nUser: ...\nAssistant:"`), not the
  GGUF's own `tokenizer.chat_template` metadata (if it has one). A model trained
  against a specific chat-template format (e.g. ChatML's `<|im_start|>` markers) may
  follow instructions noticeably worse under this generic framing than it would
  under its native template. Reading and applying the GGUF's own chat template is a
  natural follow-up, deliberately not attempted here to keep this first pass small.
- **No function/tool calling, no `logprobs`, no `n > 1`, no multimodal content
  parts.** A `messages[i].content` that isn't a plain string (e.g. an image/text
  content-parts array) is rejected with a `400`.
- **`usage.prompt_tokens` is an approximation** (whitespace word count over the
  flattened prompt, `src/openai.rs::estimate_prompt_tokens`) — this sidecar has no
  tokenizer of its own, and the IPC protocol doesn't report an exact prompt token
  count. `completion_tokens` (from `token_ids.len()` in the IPC response) is exact.
- **`finish_reason` is a heuristic**: `"length"` if the number of generated tokens
  reached the `max_tokens` actually requested, `"stop"` otherwise — the IPC protocol
  has no explicit stop-reason field to report an exact EOS-vs-length distinction.
- **No auth, no rate-limiting, no TLS.** This is a local translation shim, not a
  production-hardened gateway — put a real reverse proxy in front of it if either is
  needed. Consistent with this being local sidecar tooling, not a serving platform
  (see the root README's Non-goals section).
- **One managed `reflex` process, not auto-restarted.** If the child process crashes,
  every subsequent request fails with a clear `500` (`src/reflex_client.rs`'s
  stdin/stdout-closed error paths) rather than the sidecar transparently respawning
  it. Restart the sidecar process itself to recover.
