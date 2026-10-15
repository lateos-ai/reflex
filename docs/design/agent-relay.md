# Agent relay: handoff latency between agents

Multi-agent loops make many short, strictly sequential inference calls: agent A writes
a note, agent B reads it and writes its own, and so on. What a user feels is the cost
of each **hop**, meaning one request's round trip on a warm engine, not sustained
throughput. This note covers how the hop is measured and what is planned to make it
cheaper. The work stays inside Reflex's Non-goals (see
[DEVELOPMENT.md](../DEVELOPMENT.md)):
- one request at a time;
- no batching, scheduler or queue inside the engine;
- no network socket in the core engine.

## Measuring a hop (phase 0)

`scripts/agent_relay_bench.py` runs an N-hop chain against one warm process. Each
hop's prompt is:
- a shared system prompt;
- the next agent's role;
- the tail of the previous hop's output (`--carry-chars`, default 400, so the prompt
  size stays stable).

Decoding is greedy everywhere. Targets:

| target | how | engine time comes from |
|---|---|---|
| `stdio` | spawns `reflex stdio <model>`, talks over its pipes | `IpcTimings` |
| `uds` | spawns `reflex uds <model> <socket>`, connects (Unix only) | `IpcTimings` |
| `llama-server` | an already-running llama.cpp server, `POST /completion` | its `timings` block |

Per hop it reports p50/p95/mean of the client round trip, the engine's time to first
token, the engine's total, and **overhead**. Overhead is the round trip minus engine
time: the transport, JSON and the process boundary. By default llama-server runs with
`cache_prompt: false`, since Reflex has no prefix reuse yet. Pass
`--llama-cache-prompt` to measure the gap that prefix reuse (phase A) is meant to
close.

### `"timings": true` (IPC)

The benchmark asks the engine for its own split, so overhead isn't guessed. When a
`reflex stdio`/`uds` request sets `"timings": true`, its final line gains:

```json
"timings": {"parse_ms": 0.01, "first_token_ms": 21.4, "model_ms": 63.0}
```

- `parse_ms`: parsing the request JSON.
- `first_token_ms`: model call start to the first generated token, covering
  tokenize, prefill, the LM head and the first sampling step. It is set on the
  `generate` path only.
- `model_ms`: the whole model call. On a streaming request it includes writing each
  token line, because those writes happen inside the decode loop.

When the field is omitted (the default), responses are byte-identical to before.

### Example

```sh
cargo build --release --features ipc --bin reflex
scripts/agent_relay_bench.py stdio --model Qwen3-0.6B-Q4_K_M.gguf --hops 30 \
    --json-out relay-stdio.json
scripts/agent_relay_bench.py uds --model Qwen3-0.6B-Q4_K_M.gguf --hops 30
llama-server -m Qwen3-0.6B-Q4_K_M.gguf -ngl 99 &
scripts/agent_relay_bench.py llama-server --hops 30
```

### Results (Tesla T4, 2026-10-14)

Setup:
- Qwen3-0.6B Q4_K_M, a 30-hop chain after 3 warm-up hops, 32 greedy tokens per hop.
- Reflex built with `REFLEX_CUDA_ARCH=sm_75`; driver 595.91.07.
- llama.cpp `22bdcc4`, `llama-server -ngl 99 -c 4096`.

Each hop's prompt is about 120 tokens. Decode per token is (engine total − first
token) / 31.

| target | round trip p50 | first token p50 | decode / token | overhead p50 |
|---|---:|---:|---:|---:|
| Reflex `stdio` (f16 weights) | 268.9 ms | 18.5 ms | 8.1 ms | 0.12 ms |
| Reflex `uds` (f16 weights) | 269.3 ms | 18.6 ms | 8.1 ms | 0.13 ms |
| Reflex `stdio`, `REFLEX_QUANT_RESIDENT=1` | 250.4 ms | 29.4 ms | 7.1 ms | 0.12 ms |
| llama-server, no prompt cache | 145.3 ms | 20.4 ms | 4.0 ms | 1.44 ms |
| llama-server, prompt cache | 135.7 ms | 10.9 ms | 4.0 ms | 1.43 ms |

What this shows:

- **Transport is not the bottleneck.** The local IPC overhead is about 0.13 ms per
  hop, against about 1.4 ms for llama-server's HTTP. `stdio` and `uds` are the same
  within noise. A binary wire format or another transport would save a fraction of a
  millisecond, so it is dropped from the plan.
- **Prefill is competitive.** Reflex reaches the first token sooner than llama-server
  without prompt caching (18.5 vs 20.4 ms). llama-server's prompt cache saves about
  9.5 ms per hop here. Prefix reuse (phase A/B) targets that saving, and it grows
  with the size of the shared prefix.
- **Decode dominates the hop, and Reflex is about 2× slower there** (8.1 vs 4.0 ms
  per token). At 32 output tokens, decode is about 93% of Reflex's hop time. The
  quantized-resident mode cuts decode to 7.1 ms but makes the first token slower.
  This is single-stream decode speed: it is not something prefix reuse or a
  different transport can fix.

## Planned next phases

1. **Prefix reuse through files.** Add per-request `import_kv`/`export_kv` paths over
   IPC, built on the existing `--export-kv`/`--import-kv` cache formats. The
   orchestrator owns the files (for example on tmpfs), and the engine keeps nothing
   between requests.
2. **One pinned prefix in VRAM.** Inside a warm `stdio`/`uds` session, one optional
   slot holds a prompt's KV cache, and later requests fork from it with a
   device-to-device copy. It is one slot, with no names and no eviction policy;
   anything more belongs in the orchestrator.
3. **JSON Schema constrained decoding.** Logit masking ahead of sampling, so that
   handoff payloads and tool calls parse on the first try.
4. **Buffer reuse across warm requests**, only if phase 0 shows allocation time on
   the hop.

## Swarm pattern

Priority, routing and fan-out stay outside the engine:
- Run one warm `reflex uds` (or `stdio`) process per agent role or per GPU.
- An orchestrator decides which request goes where, and in what order.
- Short-lived sub-agents can be fresh `reflex generate` processes, since cold start
  is what Reflex is built for.
