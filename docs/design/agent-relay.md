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

## Prefix reuse through files (phase A)

A `generate` request on `reflex stdio`/`uds` can name two cache files:

| field | effect |
|---|---|
| `"export_kv": "<path>"` | writes the model's state right after this request's prompt; needs `"max_tokens": 1` |
| `"import_kv": "<path>"` | resumes from that state; `prompt` is the continuation, with no BOS added |

The files use the same formats as `reflex generate --export-kv/--import-kv`: dense/MoE
(Qwen3, Qwen2, Llama, Mistral, Kolibri), the Qwen3.5 hybrid and MLA. Gemma 4 and
Granite 4.0-H have no cache format yet and return `unsupported_architecture`. A file
from a different architecture returns `kv_cache`. Both fields on a `candidates`
request return `invalid_input`.

An export captures exactly one generated token's worth of work: the state after the
prompt. It can't come from a longer generation, because the Qwen3.5 hybrid's
recurrent state can't be rolled back past tokens generated after the prompt. A request
may set both fields. It then resumes from one file and saves the longer state to
another, so the prefixes can be layered:

```jsonl
{"prompt": "<system prompt>", "export_kv": "/dev/shm/sys.cskv", "max_tokens": 1}
{"prompt": "<agent role>", "import_kv": "/dev/shm/sys.cskv", "export_kv": "/dev/shm/role.cskv", "max_tokens": 1}
{"prompt": "<this hop's input>", "import_kv": "/dev/shm/role.cskv", "max_tokens": 64}
```

The caller owns the files: the engine keeps nothing between requests, and two warm
processes can share one file. With `"timings": true`, `kv_io_ms` reports the time
spent reading and writing them. It is included in `model_ms`, and the read is also
included in `first_token_ms`. The files hold f32 values, so a long prefix makes a
large file; tmpfs keeps the read cheap.

**The continuation is tokenized as if it followed the prefix.** With the same cache
reused, the prompt resumes exactly where the joined text would have, provided the
split falls on a token boundary. That matters for SentencePiece tokenizers (Llama-2,
Mistral, TinyLlama). Like llama.cpp, they give a stretch of raw text a leading `▁`
space only when it starts the text or follows a special token. So a continuation
gets that space only when the prefix ended on a special token, such as `</s>`.
- Dense cache files record which case applies. They use KV format version 4, the
  version-1 dense layout plus one byte.
- A pinned prefix keeps the same bit.
- A version-1 file, written before this, is resumed as if its prefix ended on a
  special token, which was the earlier behavior. Re-export it to resume exactly.
- gpt2-style vocabularies (Qwen, Llama-3, DeepSeek) have no such space, so they're
  unaffected.

`scripts/agent_relay_bench.py --reuse-prefix` exports the shared system prompt once and
imports it on every hop. `--system-prompt-file` sets a longer prefix.

### Results (Tesla T4, 2026-10-15)

Correctness: on the T4, resuming through two layered files gives the same 12 greedy
ids as one request over the joined text. This holds for real Qwen3-0.6B (f16 weights)
and for the MoE, Qwen3.5-hybrid and MLA fixtures (`src/ipc/relay_tests.rs`).

Setup is the same as phase 0: Qwen3-0.6B Q4_K_M, 30 hops after 3 warm-up hops, 32
greedy tokens per hop, cache file on `/dev/shm`. The short prefix is the built-in
system prompt, about 40 tokens. The long one is a 6.3K-character team-rules prompt,
about 1,490 tokens; its cache file is 342 MB.

| prefix | target | round trip p50 | first token p50 | KV file read p50 |
|---|---|---:|---:|---:|
| short | Reflex `stdio` | 266.3 ms | 18.2 ms | |
| short | Reflex `stdio`, prefix reused | 269.9 ms | 23.0 ms | 2.9 ms |
| long | Reflex `stdio` | 1,564.4 ms | 1,214.3 ms | |
| long | Reflex `stdio`, prefix reused | 646.5 ms | 297.4 ms | 112.1 ms |
| long | llama-server, no prompt cache | 322.8 ms | 169.5 ms | |
| long | llama-server, prompt cache | 162.8 ms | 12.4 ms | |

What this shows:

- **Reuse pays off with a long prefix, not a short one.** With the long prefix it cuts
  the hop by 2.4× (1,564 → 647 ms). With a 40-token prefix it costs 3.6 ms: reading
  and uploading the file takes longer than prefilling 40 tokens.
- **The file is now the main cost of a reused hop.** The cache is stored as f32, 229 KB
  per token for this model. Reading the 342 MB file takes 112 ms, and uploading it is
  most of the rest of the 297 ms first token. Phase B's in-VRAM slot removes both.
- **Reflex's long-prompt prefill is about 7× slower than llama.cpp's here** (1,214 vs
  169 ms for about 1,550 tokens). Decode with the long context is also slower: 11.3 vs
  4.8 ms per token. Attention dominates Reflex's long-prompt prefill on this model.
  Tiled prefill attention, which is in progress, would help every long-prompt hop,
  with or without reuse.

### Results with tiled prefill attention (Tesla T4, 2026-10-15)

Same setup, rebuilt on top of tiled prefill attention
([tiled-prefill-attention.md](tiled-prefill-attention.md)), with llama-server
re-measured in the same session:

| prefix | target | round trip p50 | first token p50 | KV file read p50 | decode / token |
|---|---|---:|---:|---:|---:|
| short | Reflex `stdio` | 263.6 ms | 15.8 ms | | 8.0 ms |
| short | Reflex `stdio`, prefix reused | 269.0 ms | 22.4 ms | 2.5 ms | 8.0 ms |
| long | Reflex `stdio` | 601.5 ms | 251.4 ms | | 11.3 ms |
| long | Reflex `stdio`, prefix reused | 634.1 ms | 286.6 ms | 104.0 ms | 11.2 ms |
| long | llama-server, no prompt cache | 319.8 ms | 167.9 ms | | 4.9 ms |
| long | llama-server, prompt cache | 161.3 ms | 12.2 ms | | 4.7 ms |

What changed:

- **Prefill no longer dominates the long hop.** The first token dropped from 1,214 ms
  to 251 ms (4.8×), and the hop from 1,564 ms to 601 ms. Exporting the long prefix
  takes 982 ms instead of 1,891 ms.
- **File-based reuse now loses.** With a cheap prefill, a reused long hop (634 ms) is
  slower than recomputing the prefix (601 ms). The 104 ms file read and the upload
  cost more than prefilling about 1,490 tokens now does. Phase A's files remain
  useful when processes don't share a GPU, or for much longer prefixes. A per-hop win
  inside one warm process needs the in-VRAM slot.
- **Decode is the largest gap now.** At this context, 31 decode steps take 350 ms of
  the 601 ms hop, at 11.3 ms per token against llama.cpp's 4.9 ms. With a short
  prompt Reflex decodes at 8.0 ms per token, so about 3 ms per token grows with the
  context. That is decode attention over ~1,550 cached positions plus the per-step
  work the short-prompt numbers already show.

## One pinned prefix in VRAM (phase B)

A warm `reflex stdio`/`uds` session can keep one prefix's model state on the GPU:

| field | effect |
|---|---|
| `"pin_prefix": true` | keeps the state right after this request's prompt in the session's slot, replacing what was there; needs `"max_tokens": 1` |
| `"use_pinned": true` | resumes from the slot; `prompt` is the continuation, with no BOS added |

```jsonl
{"prompt": "<system prompt>", "pin_prefix": true, "max_tokens": 1}
{"prompt": "<agent role>", "use_pinned": true, "pin_prefix": true, "max_tokens": 1}
{"prompt": "<this hop's input>", "use_pinned": true, "max_tokens": 64}
{"prompt": "<next hop's input>", "use_pinned": true, "max_tokens": 64}
```

How it works:

- **Pinning copies nothing.** A `pin_prefix` request runs like an `export_kv` request,
  but instead of downloading its caches it hands the device buffers to the slot. The
  same max-tokens-1 rule applies, for the same reason: the Qwen3.5 hybrid's recurrent
  state can't be rolled back past later tokens.
- **Each use copies the slot.** A `use_pinned` request allocates its own cache, as any
  request does, and fills the first `seq_len` positions with a device-to-device copy
  of the slot. The hybrid's GDN conv and recurrent state are copied whole. The pinned
  state is never written, so one pin serves any number of requests.
- **Files and the slot share one code path.** Each architecture's cache allocation
  (`alloc_dense_kv_caches`, `alloc_hybrid_states`, `alloc_mla_kv_caches`) takes a
  prefix source. A cache file is uploaded host-to-device straight into the new
  cache. A pinned prefix is copied device-to-device into it. Everything after that
  is shared. The file is uploaded directly into the request's cache rather than first
  into a device copy that is then forked, so an `import_kv` request never holds the
  prefix twice in VRAM. `export_kv` is now "pin, then download".
- **Layering works like phase A's files.** Setting both fields resumes from the slot
  and pins the longer state in its place.
- **The slot's lifetime.** `reflex stdio` keeps one slot for the process. `reflex uds`
  keeps one across its connections, so a client can reconnect and still use it. A
  failed request leaves the slot as it was.

Errors:

| request | `error_kind` |
|---|---|
| `use_pinned` with nothing pinned | `invalid_input` |
| `pin_prefix` without `"max_tokens": 1` | `invalid_input` |
| `pin_prefix`/`use_pinned` together with `import_kv`/`export_kv` | `invalid_input` |
| `pin_prefix`/`use_pinned` on a `candidates` request | `invalid_input` |
| `pin_prefix`/`use_pinned` on Gemma 4 or Granite 4.0-H | `unsupported_architecture` |
| a use whose own cache can't be allocated | `out_of_memory` |

**VRAM.** A pinned prefix holds as much device memory as its KV cache: 229 KB per
token for Qwen3-0.6B (f32), so 342 MB for the 1,490-token prefix below. Each request
that uses it allocates a second copy for its own cache while it runs. If that
allocation fails, the request returns `out_of_memory` and the session and slot carry
on. The slot holds its memory until it is replaced or the process exits.

**Timings.** With `"timings": true`, a `use_pinned` request reports the copy in
`kv_io_ms`, the field phase A uses for file reads. It is included in
`first_token_ms` and `model_ms`. A `pin_prefix` request on its own reports no
`kv_io_ms`, since pinning copies nothing.

**One slot is the boundary.** There are no names, no second slot, no LRU or eviction
policy and no background work. Keeping several prefixes, choosing which to keep, or
sharing one between processes is the orchestrator's job: cache files through
`import_kv`/`export_kv`, or one warm process per prefix. See
[DEVELOPMENT.md](../DEVELOPMENT.md)'s Non-goals.

`scripts/agent_relay_bench.py --pin-prefix` pins the shared system prompt once and
uses it on every hop.

### Results (Tesla T4, 2026-10-15)

Correctness: on the T4, every use of the slot gives the same 12 greedy ids as one
request over the joined text and as the same chain through files. This holds for real
Qwen3-0.6B (f16 weights) and for the Qwen3-MoE, Qwen3.5-hybrid, MLA, Qwen2 and
Kolibri-1 fixtures. The tests pin a prefix, layer a second part on it, use the slot
three times, check it's unchanged, re-pin a shorter prefix and check that it replaced
the slot (`src/ipc/relay_tests.rs`). The GPU nightly suite passes, 83 of 83 (4
skipped for absent optional models).

Setup is the same as phases 0 and A (Qwen3-0.6B Q4_K_M, 30 hops after 3 warm-up
hops, 32 greedy tokens per hop, every hop generated all 32), on top of decode-speed
and tiled prefill attention, with llama-server in the same session. Decode per token
is (engine total − first token) / 31.

| prefix | target | round trip p50 | first token p50 | prefix read/copy p50 | decode / token |
|---|---|---:|---:|---:|---:|
| short | Reflex `stdio` | 229.3 ms | 15.6 ms | | 6.9 ms |
| short | Reflex `stdio`, file reuse | 236.4 ms | 22.4 ms | 2.6 ms | 6.9 ms |
| short | Reflex `stdio`, pinned | 229.2 ms | 15.9 ms | 0.2 ms | 6.9 ms |
| short | llama-server, no prompt cache | 141.5 ms | 19.7 ms | | 3.9 ms |
| short | llama-server, prompt cache | 132.5 ms | 10.5 ms | | 3.9 ms |
| long | Reflex `stdio` | 501.3 ms | 242.8 ms | | 8.3 ms |
| long | Reflex `stdio`, file reuse | 547.2 ms | 291.3 ms | 108.1 ms | 8.3 ms |
| long | Reflex `stdio`, pinned | 364.1 ms | 108.1 ms | 4.4 ms | 8.3 ms |
| long | Reflex `uds`, pinned | 365.1 ms | 108.2 ms | 4.5 ms | 8.3 ms |
| long | llama-server, no prompt cache | 319.0 ms | 167.1 ms | | 4.8 ms |
| long | llama-server, prompt cache | 161.8 ms | 12.3 ms | | 4.7 ms |

The same session's baseline before this change matched these within 2% (long hop
510 ms recomputed, 545 ms reused from a file).

What this shows:

- **The slot makes reuse pay off.** The long hop drops from 501 ms (recompute) to
  364 ms, and the first token from 243 to 108 ms. Copying 342 MB device-to-device
  takes 4.4 ms, against 108 ms to read the same state from a file.
- **A short prefix gains nothing, and loses nothing either.** Prefilling 40 tokens
  costs about as much as copying them, so the pinned short hop ties the recomputed
  one (229 ms each). File reuse still costs 7 ms there.
- **The resumed prefill was the slow part of the first token, and is now fixed.**
  After the copy, the first token took about 104 ms. The hop's ~45 new tokens were
  under the tiled kernel's 64-row threshold, so they ran the online attention kernel,
  which reads the whole cached prefix once per row and head. Prefills after a cached
  prefix now switch to the tiled kernel once `rows * group` fills a block
  ([tiled-prefill-attention.md](tiled-prefill-attention.md), "Resumed prefill"). In
  the same T4 session the pinned hop went from 364.5 to 295.3 ms, its first token from
  108.5 to 38.6 ms, and file reuse from 547 to 472 ms. llama-server's cached hop is
  161.7 ms, with a 12.3 ms first token.
- **Decode is still the largest gap.** 31 decode steps take 256 ms of the 295 ms
  pinned hop: 8.3 ms per token against llama.cpp's 4.7 ms.

## Planned next phases

1. **JSON Schema constrained decoding.** Logit masking ahead of sampling, so that
   handoff payloads and tool calls parse on the first try.
2. **Buffer reuse across warm requests**, only if phase 0 shows allocation time on
   the hop.

## Swarm pattern

Priority, routing and fan-out stay outside the engine:
- Run one warm `reflex uds` (or `stdio`) process per agent role or per GPU.
- An orchestrator decides which request goes where, and in what order.
- Short-lived sub-agents can be fresh `reflex generate` processes, since cold start
  is what Reflex is built for.
