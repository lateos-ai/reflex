# reflex-openai-adapter

An OpenAI-compatible `POST /v1/chat/completions` HTTP sidecar in front of the Reflex
core engine. This is the escape-hatch pattern the root project's `README.md`'s
Non-goals section describes and defers: **the core `reflex` engine never
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
  [--default-max-tokens <n>] [--no-chat-template] [--chat-template-file <path>]
```

- `--lora <adapter.gguf>` is forwarded straight through to `reflex stdio`'s own
  `--lora` flag (see Phase 4/Embeddability in the root `README.md`).
- `--model-name` sets the `model` field returned in responses when a request doesn't
  specify one of its own; defaults to the GGUF file's stem.
- `--default-max-tokens` (default `256`) is used when a request omits `max_tokens`.
- `--no-chat-template` forces the old generic role-labeled flattening
  (`"System: ...\nUser: ...\nAssistant:"`) even when the loaded GGUF ships its own
  `tokenizer.chat_template`. Useful as an A/B-comparison switch and as an escape
  hatch if a template is misbehaving.
- `--chat-template-file <path>` supplies an explicit Jinja2 chat-template file
  (same syntax as a GGUF's `tokenizer.chat_template` string) for a GGUF that doesn't
  ship one of its own. Mutually exclusive with `--no-chat-template`.

At startup the adapter tries to load and render-test the GGUF's own
`tokenizer.chat_template` metadata once (or `--chat-template-file`'s contents, if
given) -- not per-request -- and logs exactly one line to stderr saying which source
it's using, or why it fell back to the generic flattening (no template present, the
template failed to compile, or it failed a render self-test). If the template loads
successfully at startup but a specific request still fails to render against it
(e.g. an unusual message shape a template's own Jinja logic doesn't handle), that one
request falls back to the flattening too, with its own stderr line -- a bad template
degrades gracefully, it never takes the sidecar down or silently produces a garbled
prompt.

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

- **Chat template support is real but narrow.** Reflex itself has no chat-template
  support — `Model::forward_prompt` takes a plain prompt string, not a structured
  message list — so this adapter reads and renders the GGUF's own
  `tokenizer.chat_template` metadata (a Jinja2 template string, the same mechanism
  HF's `AutoTokenizer.apply_chat_template` uses) via a minimal standalone GGUF
  metadata reader (`src/gguf_meta.rs`) and `minijinja` (`src/chat_template.rs`),
  instead of the old generic role-labeled flattening (`src/openai.rs::build_prompt`:
  `"System: ...\nUser: ...\nAssistant:"`), whenever one is present and renders
  successfully. Verified byte-exact against real Python `jinja2`/HF's own template
  settings on Qwen3-0.6B's real ChatML template (single- and multi-turn), and
  end-to-end against a real running model on a GPU instance (real Tesla T4):
  templated requests produce genuinely coherent output, including Qwen3's own
  `<think>...</think>` reasoning traces that explicitly reference the system
  prompt's persona, where the old flattening never engages the model's native
  chat behavior at all. That GPU round also surfaced and fixed a real
  prerequisite gap in the *core engine's* tokenizer (`src/tokenizer.rs`, not
  this crate): rendering a real chat template produces prompt text containing
  literal control-token substrings like `<|im_start|>`, and without special-
  token-aware tokenization, the core BPE encoder shredded them into meaningless
  byte fragments instead of their real reserved ids — silently corrupting every
  templated prompt before this fix landed alongside this feature. What's
  still out of scope:
  - **No tool-calling-style templates.** `messages[i].tool_calls`/a `tools` request
    field aren't modeled — a template branch that expects them (e.g. Qwen3's own
    template has one) either renders as if no tools were supplied, or fails to
    render and falls back to flattening, depending on the branch's exact logic.
  - **No multimodal template blocks** — consistent with this adapter's existing
    no-multimodal-content-parts limitation below; a template branch expecting an
    image/audio content part isn't exercised.
  - A template's `raise_exception(...)` calls (common in real HF templates for
    input validation, e.g. "system message must come first") are wired to a real
    minijinja error, so a template correctly rejecting a malformed message list
    surfaces as a render failure (logged, then falls back to flattening for that
    request) rather than being silently ignored.
  - `--no-chat-template` forces the old flattening even when a template is present
    (A/B comparison / escape hatch); `--chat-template-file <path>` supplies one for
    a GGUF that doesn't ship its own. See the Usage section above.
  - **The literal EOS marker (e.g. `<|im_end|>`) can appear in `message.content`**
    when generation stops on it — observed on real GPU hardware. The core
    engine's decode path doesn't strip the stop token's own text before
    returning it over IPC; a real OpenAI API never includes it. Cosmetic, not a
    correctness bug (the model's actual answer is intact and `finish_reason`
    still correctly reports `"stop"`), and out of scope for this change (core
    engine decode/generate-loop behavior, not chat-template rendering) — a
    follow-up for whoever next touches `Model::generate`'s stop handling.
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
