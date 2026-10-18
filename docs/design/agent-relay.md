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

Later phases added `kv_io_ms` (phases A and B), `mask_ms_total` (phase C), and
`pinned_seq_len`/`pinned_bytes`, the pinned slot's size after the request (phase E).
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
from a different architecture returns `kv_cache`. `export_kv` on a `candidates`
request returns `invalid_input`; `import_kv` there resumes System-1 scoring (phase E,
below).

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
spent reading and writing them; since phase F the read includes the upload into the
request's cache. It is included in `model_ms`, and the read is also included in
`first_token_ms`. The files hold f32 values, so a long prefix makes a large file;
tmpfs keeps the read cheap, and `"export_kv_dtype": "f16"` halves it (phase F, below).

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
| `pin_prefix` on a `candidates` request (`use_pinned` is allowed since phase E) | `invalid_input` |
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

## JSON Schema constrained decoding (phase C)

A handoff payload or a tool call that doesn't parse costs the orchestrator a retry,
which is a whole extra hop. A `generate` request can name a JSON Schema, and the
output is then one JSON value that validates against it:

```jsonl
{"prompt": "<hop prompt>", "json_schema": {"type": "object", "properties": {"status": {"enum": ["done", "blocked"]}, "note": {"type": "string"}}, "required": ["status", "note"]}, "max_tokens": 64}
```

```sh
reflex generate model.gguf "<prompt>" --max-tokens 64 --json-schema handoff.json
```

`text` is the value alone; `token_ids` end with EOS. Constraining covers only the
generated text, never the prompt, so `json_schema` works with `use_pinned` and
`import_kv` like any other request.

### The supported subset

`type` (one name or an array of them), `properties`, `required`,
`additionalProperties` (`false` or omitted), `items`, `enum`, `const`, `anyOf`, and a
local `$ref` (`#` or `#/$defs/<name>`, recursion allowed). The annotations `title`,
`description`, `$comment`, `$schema`, `default` and `examples` are accepted, since
they never change which values are valid.

Every other keyword is rejected with `invalid_input`, and the message names it:
`minLength`, `pattern`, `format`, `minimum`, `oneOf`, `allOf`, `prefixItems` and so on.
A schema the engine accepts never carries a constraint it silently drops, so an
accepted schema means what it says. Other rejections:
- `additionalProperties: true` or a schema value;
- a `required` name missing from `properties`;
- `$defs` anywhere but the root;
- a `$ref` cycle that never consumes input;
- `anyOf` or `$ref` next to other validation keywords in the same object;
- an `enum` none of whose values pass the sibling keywords.

### A fixed output format

The output is compact JSON, with one allowance:
- no whitespace outside strings, except **one optional space after each `:` and
  `,`**;
- object properties in schema order, each optional one present or absent;
- `enum` and `const` values exactly as written compactly;
- generation ends at EOS as soon as the value is complete. Once the value is closed,
  EOS is the only token allowed.

"Schema order" is the order of the request's bytes: the engine parses the schema
itself and never reorders it. A client that builds the request through a JSON type
that sorts keys (`serde_json::Value` without `preserve_order`, for one) sends the
properties alphabetically, and gets them back in that order. Python's `json` keeps
insertion order.

With `properties` given, the object has no other properties, whether
`additionalProperties` is `false` or omitted. llama.cpp's `json_schema_to_grammar`
does the same. An object schema with neither `properties` nor
`additionalProperties: false` allows any object. Every output still validates against
the schema, since this is a subset of what the schema allows.

Why the format is fixed rather than free:
- **A smaller automaton.** Optional whitespace between every token of the value, and
  properties in any order, multiply the states the matcher has to track. An object
  with *n* optional properties in any order needs a state per subset already
  emitted; in schema order it needs about 2*n* rules. One optional space after a
  separator adds one nullable rule.
- **Fewer tokens.** Newlines and indentation are tokens the model has to decode, at
  about 8 ms each on the T4.
- **Output the orchestrator can predict.** There is no layout to normalize: the bytes
  differ only in those single spaces, and greedy decoding gives the same bytes for
  the same request every time.

**Why one space is allowed.** The first version allowed no whitespace at all, and
that cost content on a small model. Qwen3-0.6B writes `"key": "value"`. With its
` "` token masked after a colon, it often picked `", "` as the whole string, which
gives `{"agent":", "}`. That output validates but carries nothing. The single space
lets a model keep its usual tokens. Pretty-printing (newlines and indentation) stays
out: a model that prefers it is pushed onto one line at the first newline, which
happens before any value is written. The before/after measurements are under
"Results" below.

### How masking works

`src/grammar.rs` compiles the schema into a context-free grammar over bytes. At each
generated position it sets the logit of every token that can't continue a
conforming value to `-inf`, before greedy argmax and before temperature/top-k/top-p.
Nothing else about the logits changes. So when the unconstrained greedy output
already conforms, the constrained output is identical, token for token.

- **The matcher** runs the grammar as a set of pushdown stacks, so `anyOf` branches
  and `enum` values may overlap until the bytes tell them apart. Stacks are
  hash-consed, and each distinct set of them becomes one state of a DFA over bytes,
  built lazily. Once a state has seen a byte, stepping it is one table lookup.
- **The token index** is the vocabulary as a byte trie in preorder, with a skip
  pointer per node. Each token's bytes are exactly what it adds to the decoded text.
  For gpt2 byte-level BPE that is the byte-level decoding. For SentencePiece, `▁`
  becomes a space and a `<0xNN>` token is the raw byte. Control, unknown and unused
  tokens are never allowed. EOS is allowed only when the value is complete.
- **A mask** is one linear pass over the trie that skips every subtree whose prefix is
  already rejected. It is cached per DFA state, since the state inside a long string
  repeats token after token.
- **Tokens that split a character or an escape.** The grammar checks UTF-8 byte by
  byte (no overlongs, no surrogates, nothing past U+10FFFF), and checks escapes the
  same way: a `\u` high surrogate must be followed by a low one. A token that ends
  partway through a sequence is allowed whenever the bytes so far can still be
  completed, and the next tokens must then complete it. A gpt2 byte token `0xE4`, or
  SentencePiece's `<0xE4>`, is allowed inside a string and nowhere else.
- **U+2581 on SentencePiece models.** The decoder turns every `▁` into a space, even
  one assembled from three byte-fallback tokens. So the grammar keeps U+2581 out of
  strings for those tokenizers, and a schema literal that contains it is rejected.
- **Numbers.** The number grammar is RFC 8259's, as in llama.cpp, with no limit on
  digits. So `1e400` is accepted, although some parsers (`serde_json`) refuse it as
  out of range.

**The index is built lazily.** The first constrained request in a process builds it
and keeps it on the `Model`. That takes 66 ms for Qwen3's 152K-token vocabulary
(333,054 trie nodes) on the T4 host. Cold start and unconstrained requests never
build it.

### Errors and timings

| request | `error_kind` |
|---|---|
| a bad or unsupported schema | `invalid_input` |
| `json_schema` with `candidates` | `invalid_input` |
| `json_schema` with `pin_prefix` or `export_kv` | `invalid_input` |
| `max_tokens` ran out before the value was complete | `grammar_dead_end` |

A `grammar_dead_end` response carries the partial `text` and `token_ids`. A pinned or
exported prefix is the state right after the prompt, before any value exists, so
there is nothing for a schema to constrain there.

With `"timings": true`, a constrained request reports `mask_ms_total`. That is the
time spent computing and applying masks and following each chosen token through the
grammar, summed over the request. It is included in `first_token_ms` (the first mask)
and in `model_ms`. A failed `generate` request now also carries `timings` when asked,
so a dead end's work can be accounted for.

`scripts/agent_relay_bench.py --json-schema <file>` constrains every hop and adds the
masking time and the count of hops whose output parsed.
`scripts/relay_handoff_schema.json` is a sample schema.
`scripts/json_schema_compare.py` runs 20 schema-and-prompt cases on Reflex and
llama-server, with and without the schema. It reports the validity rate and the
decode time per token.

### Results (Tesla T4, 2026-10-16)

Setup: Qwen3-0.6B Q4_K_M (gpt2 byte-level BPE, 152K tokens) and
TinyLlama-1.1B-Chat Q4_K_M (SentencePiece, 32K tokens). The host CPU is a Xeon
Platinum 8259CL. llama.cpp is at `22bdcc4`, run as `llama-server -ngl 99 -c 4096`.
The numbers are for the format above, with the optional space. The first version,
without it, is compared where it matters.

**Correctness.**
- Host tests (`src/grammar/tests.rs`) cover schema compile and rejection, and the
  automaton on valid and invalid byte streams: UTF-8, escapes, surrogates, spaces,
  order and required properties. Random walks of the automaton always give output
  that parses and validates.
- Every mask was compared with a walk of every vocabulary token, after 21 prefixes
  that stop after a separator, mid-string, mid-escape, mid-UTF-8 and mid-number. They
  agree on the real Qwen3 and TinyLlama vocabularies.
- GPU tests (`src/ipc/json_schema_tests.rs`, 11 tests) cover a nested schema, an enum
  and an anyOf on both models. They also run a `use_pinned` and an `import_kv` hop
  with a schema, each equal to one joined request on both models, and the dead end
  and the request errors. A finite schema runs on the Qwen3-MoE, Qwen3.5, MLA,
  Granite 4.0-H and Gemma 4 fixtures. Every output parses with `serde_json`,
  validates, and is within the format.
- **Identity:** on Qwen3, the two test prompts whose unconstrained greedy output
  already conformed gave identical ids under the schema.
- The GPU nightly passes 98 tests, with 4 skipped for absent optional models.

**Mask cost.**

| | Qwen3 (152K) | TinyLlama (32K) |
|---|---:|---:|
| index build, once per process | 67 ms | 18 ms |
| first mask inside a string (whole trie) | 3.6 to 5.4 ms | |
| a mask the per-state cache already has | < 0.001 ms | |
| mask at a structural position | 0.03 ms | |
| per generated token, mean over the 20 comparison cases | 0.32 ms (max 0.53) | 0.08 ms (max 0.13) |

That is well under the ~1 ms per token budget, so nothing past the token trie was
needed.

**Cold start is unchanged.** `bench_cold_start_phases.sh`, 10 runs per round,
interleaved. p50 `process_start_to_first_token_ms`: master 655.4 / 657.3 ms against
this branch's 659.8 / 655.8 ms. A cold `reflex generate --json-schema` reaches its
first token about 66 ms later, which is the index build.

**Against llama-server** (`scripts/json_schema_compare.py`: 20 cases over 5 schemas,
greedy, `max_tokens` 192, no prompt hint). Validity means the output parses and
validates with `jsonschema`.

| model | engine | valid with schema | decode/token, free | decode/token, schema | mask/token |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Reflex | 20/20 | 6.96 ms | 7.28 ms | 0.32 ms |
| Qwen3-0.6B | llama-server | 20/20 | 4.25 ms | 8.71 ms | |
| TinyLlama | Reflex | 16/20 | 10.02 ms | 10.12 ms | 0.08 ms |
| TinyLlama | llama-server | 17/20 | 4.64 ms | 5.68 ms | |

- Without a schema, 0 of 20 outputs validate on either engine; most are wrapped in a
  `` ```json `` fence.
- Grammar costs llama-server about 4.5 ms per token on Qwen3, and Reflex 0.3 ms.
- TinyLlama's failures are the four handoff cases on Reflex and three of the same
  four on llama-server. The model keeps adding items to the `steps` array until
  `max_tokens`.

**Content, with and without the optional space** (Qwen3-0.6B, the same 20 cases):

| | no whitespace (first version) | one optional space |
|---|---|---|
| handoffs whose `agent` is a degenerate `", "` | 3 of 4 | 0 of 4 |
| tool calls pairing one tool's name with another's arguments | 2 of 4 | 0 of 4 |
| extractions that list the people in the text | 0 of 3 | 3 of 3 |

The fourth extraction case has no people, and both versions return an empty list.
Without the space, Qwen3-0.6B's ` "` token after a colon was masked, and the model
often picked `", "` as the whole string. That output was valid and empty. A prompt
hint showing compact JSON fixed most of it, but not extraction. With the space, the
outputs match llama-server's in substance, without a hint.

**Relay hops** (the pinned ~1,490-token prefix, 30 hops, Qwen3-0.6B):

| prompt | target | round trip p50 | masking p50 | tokens | valid |
|---|---|---:|---:|---:|---:|
| default (free-text note), 64 max | Reflex, free | 575.7 ms | | 64 | |
| default, 64 max | Reflex, schema | 447.5 ms | 10.9 ms | 48 | 30/30 |
| default, 64 max | llama-server cached, schema | 240.9 ms | | 28 | 30/30 |
| `--json-prompt`, 96 max | Reflex, free | 858.5 ms | | 96 | |
| `--json-prompt`, 96 max | Reflex, schema | 267.0 ms | 8.4 ms | 27 | 30/30 |
| `--json-prompt`, 96 max | llama-server cached, free | 491.3 ms | | 96 | |
| `--json-prompt`, 96 max | llama-server cached, schema | 213.7 ms | | 27 | 30/30 |

- **Masking costs 0.2 to 0.3 ms per token**, about 2% of a hop.
- **A schema makes the hop shorter, not longer.** Generation stops at EOS once the
  value closes, so the hop decodes only the handoff. That makes it 1.3× shorter with
  the default prompt and 3.2× with `--json-prompt` (267 vs 858 ms).
- **Without the optional space, no default-prompt hop completed** (0/30, even at 160
  tokens): the model filled the `note` string with `. . .`. With it, all 30 do.
- The remaining gap to llama-server is decode speed (7.3 against 4.3 ms per token
  free), not the grammar.

**Prefix reuse on SentencePiece models.** These runs found that a `use_pinned` or
`import_kv` continuation on TinyLlama got the tokenizer's dummy-prefix space where
the joined text has none (`▁<` for `<`). The phase A/B tests compared only token
counts, which happened to match. That is fixed now (see "Prefix reuse through
files", above), the tests compare ids, and the constrained hop test checks
`use_pinned` and `import_kv` against one joined request on TinyLlama too.

Raw data: `bench-results/agent-relay-t4-phase-c/` (not committed; `run4` has the
numbers above, `run1` to `run3` the first version's).

## Multi-process chains (phase D)

Phases 0 to C measured hops inside one warm process. An orchestrator can also run a
hop as a fresh process: a short-lived sub-agent, a GPU shared by turns, or no warm
process at all. Phase D measures those chains, and the GPU energy of every target.

### Bench targets

`scripts/agent_relay_bench.py` gains:

| target | one hop is | state between hops |
|---|---|---|
| `cold` | a fresh `reflex generate` over the whole prompt | the carry text only |
| `cold-chain` | a fresh `reflex generate --import-kv <prefix> --max-tokens N <rest>` | the system prompt's cache file, written once by a cold `--export-kv` process |
| `cold-warm` | a fresh `reflex generate --export-kv` over the system prompt and the previous note, then a warm `reflex stdio` request that imports that file and decodes from the role line | one new cache file per hop |
| `llama-cli` | a fresh llama.cpp completion process (`llama-completion` on current builds, where `llama-cli` became a chat UI); `--llama-prompt-cache` reuses the system prompt through llama.cpp's `--prompt-cache` file | the carry text, or that file |

**Why a cold hop is one process.** `reflex generate` takes `--import-kv` together
with `--max-tokens N`, but `--export-kv` can't be combined with `--import-kv` and
needs `--max-tokens 1`. (IPC requests can import and export at once.) So a resumed
cold hop is one process that imports and decodes. An export process followed by a
decode process would pay two cold starts per hop, and since the export can't import,
it would prefill the whole prefix every time, which costs more than prefilling it in
the decode process directly.

Cold hops report the process's wall time from spawn to exit, its first token, and the
file read or write. `reflex generate` now prints the file time as `kv_io_ms` (`--json`
schema 1.2.0). For Reflex the first token is `process_start_to_first_token_ms`, which
starts at `main()`. For llama.cpp it is the first generated byte on stdout, from
spawn. Every hop's text goes into `--json-out`, so chains on different targets can be
compared.

**`--energy`** runs each cold process under `reflex-energy`. That tool reads NVML's
device-wide energy counter, measures the idle power for `--energy-idle-ms` (default
2,000) before the process, and reports energy gross and net of that idle baseline.
Warm targets and `cold-warm` bracket the 30 measured hops once and divide by 30; their
idle baseline is measured with the warm engine loaded. Timing and energy are separate
passes, since the energy pass waits for an idle window before every cold process.

### Results (Tesla T4, 2026-10-17)

Setup is the same as phases 0 to C:
- Qwen3-0.6B Q4_K_M (f16 weights), 30 hops after 3 warm-up hops, 32 greedy tokens per
  hop, cache files on `/dev/shm`.
- The short prefix is about 40 tokens; the long one is the ~1,490-token team-rules
  prompt, whose cache file is 342 MB.
- Reflex is built with `REFLEX_CUDA_ARCH=sm_75`, driver 595.91.07.
- llama.cpp is at `22bdcc4`: `llama-completion -ngl 99 --no-warmup -no-cnv --simple-io
  -c 4096`, and `llama-server -ngl 99 -c 4096`.

**Correctness.** Within each prefix, every Reflex chain gave the same text on all 30
hops: `cold`, `cold-chain` and `cold-warm` matched warm `stdio` with recompute, file
reuse and the pinned slot. llama-completion matched llama-server on all 30 as well.

**Time per hop** (p50). A hop is the process's wall time for cold targets, and the
round trip for warm ones. For `cold-warm` it is the export process's wall time plus
the warm round trip.

| prefix | target | hop | first token | load | prompt eval | KV read | KV write |
|---|---|---:|---:|---:|---:|---:|---:|
| short | Reflex `cold` | 1,048 ms | 700 ms | 443 ms | 253 ms | | |
| short | Reflex `cold-chain` | 1,060 ms | 712 ms | 445 ms | 264 ms | 7.9 ms | |
| short | Reflex `cold-warm` (818 + 245) | 1,063 ms | 32 ms | | | 4.8 ms | 11.8 ms |
| short | Reflex `stdio`, warm | 230 ms | 15.7 ms | | | | |
| short | Reflex `stdio`, pinned | 231 ms | 16.1 ms | | | | |
| short | llama.cpp cold | 1,466 ms | 1,175 ms | | 71 ms | | |
| short | llama.cpp cold, prompt cache | 1,486 ms | 1,193 ms | | 78 ms | | |
| short | llama-server, warm | 143 ms | 19.8 ms | | | | |
| long | Reflex `cold` | 1,322 ms | 922 ms | 449 ms | 471 ms | | |
| long | Reflex `cold-chain` | 1,499 ms | 1,084 ms | 447 ms | 633 ms | 284 ms | |
| long | Reflex `cold-warm` (1,547 + 674) | 2,226 ms | 405 ms | | | 290 ms | 226 ms |
| long | Reflex `stdio`, warm | 505 ms | 246 ms | | | | |
| long | Reflex `stdio`, file reuse | 480 ms | 224 ms | | | 110 ms | |
| long | Reflex `stdio`, pinned | 295 ms | 38.6 ms | | | | |
| long | llama.cpp cold | 1,678 ms | 1,346 ms | | 210 ms | | |
| long | llama.cpp cold, prompt cache | 1,598 ms | 1,281 ms | | 81 ms | | |
| long | llama-server, warm | 319 ms | 166 ms | | | | |
| long | llama-server, prompt cache | 162 ms | 12.4 ms | | | | |

- Load is `gguf_open_ms + cuda_init_ms + model_load_ms`.
- Exporting the long prefix once, for `cold-chain`, takes a 1,566 ms process with a
  224 ms write.
- Decode is 7.1 ms per token cold and 6.9 ms warm with the short prefix, and 8.4 to
  8.7 ms with the long one. llama.cpp's is 3.7 to 4.9 ms.
- llama.cpp's prompt-cache files hold f16 values: 171 MB for the long prefix, against
  Reflex's 342 MB of f32.

**GPU energy per hop** (p50 for cold targets; the bracket's mean for warm targets and
`cold-warm`). The T4's idle power was 32 W in every bracket. That is its idle draw
right after work, before it drops to a lower power state. So the idle baseline is a
large share of every cold hop's gross energy.

| prefix | target | gross | net of idle |
|---|---|---:|---:|
| short | Reflex `cold` | 53.9 J | 18.1 J |
| short | Reflex `cold-chain` | 55.1 J | 18.1 J |
| short | Reflex `cold-warm` | 51.5 J | 16.4 J |
| short | Reflex `stdio`, warm | 16.1 J | 8.4 J |
| short | Reflex `stdio`, pinned | 16.2 J | 8.4 J |
| short | llama.cpp cold | 56.9 J | 5.9 J |
| short | llama-server, warm | 9.8 J | 5.2 J |
| long | Reflex `cold` | 72.2 J | 27.4 J |
| long | Reflex `cold-chain` | 70.2 J | 19.9 J |
| long | Reflex `cold-warm` | 99.9 J | 28.7 J |
| long | Reflex `stdio`, warm | 34.8 J | 18.4 J |
| long | Reflex `stdio`, file reuse | 26.7 J | 10.4 J |
| long | Reflex `stdio`, pinned | 20.5 J | 10.6 J |
| long | llama.cpp cold | 70.2 J | 13.4 J |
| long | llama.cpp cold, prompt cache | 61.1 J | 8.2 J |
| long | llama-server, warm | 21.9 J | 11.6 J |
| long | llama-server, prompt cache | 11.2 J | 6.0 J |

What this shows:

- **A cold hop is 4 to 5 times a warm one.** Short prefix: 1,048 ms cold against
  230 ms warm. Long prefix: 1,322 ms cold against 295 ms pinned. Each cold hop pays
  about 445 ms of load. Its first forward pass also costs about 235 ms more than the
  same prefill in a warm process (253 ms against 16 ms for the short prompt), every
  time, whatever the prefix. Keeping a process warm saves more than any cache-file
  change could.
- **Cold, Reflex is ahead of llama.cpp; warm, it's behind.** The cold hop is 1.4×
  faster than llama.cpp's with the short prefix and 1.27× with the long one, and the
  first token comes 425 to 475 ms sooner. The warm gap is decode speed, as in phases 0 to C.
- **In a cold process, a cache file loses to recomputing the prefix.** With the long
  prefix, `cold-chain` (1,499 ms) is 177 ms slower than `cold` (1,322 ms). The file
  read alone takes 284 ms, against 110 ms for the same file in a warm process.
- **The cold read is page faults.** `/usr/bin/time -v` on one cold hop counted
  193,829 minor faults with the import and 110,620 without, 3 runs each. That is
  83,209 more, one per 4 KB page of the 342 MB buffer the file is read into. A warm
  process's allocator reuses memory it has already faulted in.
- **Energy follows the GPU work, not the wall time.** Gross energy per cold hop is
  close to llama.cpp's (54 against 57 J short), because Reflex's hop is shorter. Net
  of idle, Reflex's is 3× higher (18.1 against 5.9 J). The likely causes, not
  measured separately here, are its weight load, which dequantizes on the GPU, and
  its slower decode. Avoiding the
  prefill saves energy even when it doesn't save time: warm long-prefix hops use
  18.4 J net when recomputing and 10.4 J from a file, at about the same round trip.
  A cold `cold-chain` hop uses 19.9 J against 27.4 J for `cold`.
- **`cold-warm` is the most expensive pattern.** It pays a cold start to produce the
  state, plus a file write and a read. Use it only when that state has to come from
  another process.

### Phase F's gate: file read plus upload as a share of a cold hop

The upload into the model's cache isn't timed on its own. Its size comes from the warm
runs, where the file and pinned hops prefill the same rows over the same prefix. So
upload ≈ first token (file) − first token (pinned) − read + pinned copy = 224.3 −
38.6 − 109.8 + 4.4 ≈ 80 ms. The cold numbers agree: subtracting the short-prefix
`cold-chain` prompt eval (minus its read) and the long read from the long one's prompt
eval leaves about 93 ms for the upload and the extra attention.

| chain (long prefix) | read + upload (+ write) | share of the hop | share of first token |
|---|---:|---:|---:|
| `cold-chain` | 284 + ~80 = ~365 ms | 24% | 34% |
| `cold-warm` | 226 write + 290 read + ~80 = ~596 ms | 27% | |
| `cold-chain`, short prefix | 7.9 ms + upload | < 1% | |

**Verdict: the gate passes** for the long prefix: about 24% of a `cold-chain` hop and
27% of a `cold-warm` hop, against the ~20% threshold. With the short prefix the files
cost nothing measurable. Two caveats belong with phase F:
- **f16 files alone would only tie recomputing.** Halving the read and the upload
  saves about 180 ms, which brings `cold-chain` to roughly 1,320 ms, the same as
  `cold`. The files win cold only with longer prefixes, larger models, or chains where
  the prefix can't be recomputed (the state came from another process).
- **Most of the cold read is faulting in a fresh buffer**, and that cost doesn't
  depend on the format. Reading into memory that is already faulted in, or uploading
  straight from the file's pages, would bring the 284 ms close to the warm 110 ms. It
  would also help f32 files.

Raw data: `bench-results/agent-relay-t4-phase-d/` (not committed; `time-*` and
`energy-*` JSON per target, `pf-*.time` for the page faults).

## System-1 over a resumed prefix (phase E)

System-1 scores a few candidate continuations of a prompt in one forward pass, with no
decode loop (`Model::system1_evaluate`). In a relay it answers the small questions
between hops: which agent should act next, whether a note is done, which of A to D
applies. Each question is asked after the same long shared prompt the agents use. Phase
E lets a `candidates` request resume from that prompt's cached state instead of
prefilling it again.

| field on a `candidates` request | effect |
|---|---|
| `"use_pinned": true` | scores after the session's pinned prefix; `prompt` is the continuation |
| `"import_kv": "<path>"` | scores after that cache file's state; `prompt` is the continuation |

```jsonl
{"prompt": "<team rules>", "pin_prefix": true, "max_tokens": 1}
{"prompt": "Previous note: ...\nThe agent who should act on this note next is the", "use_pinned": true,
 "candidates": [" planner", " coder", " reviewer"], "timings": true}
{"prompt": "Previous note: ...\nYou are the coder. Your note:", "use_pinned": true, "max_tokens": 64}
```

How it works:

- **One prefix path for generating and scoring.** `Model::system1_evaluate_resumed`
  maps a `Resume` onto each architecture's prefix source, the same mapping
  `generate_resumed` uses. The dense/MoE, MLA and Qwen3.5 hybrid System-1 functions
  then call the same batched prefill as generation, which seeds the request's own
  cache from the file or the slot. `system1_evaluate` is the fresh case
  (`Resume::Fresh`), so a request without a prefix runs exactly as before.
- **The pinned state is never written.** Multi-token candidates are scored by teacher
  forcing their own tokens after the prompt. Those positions go into the request's
  copy of the cache, past the prefix, so one pin serves any number of scoring and
  generating requests.
- **Candidates are tokenized as continuations.** The prompt and each `prompt +
  candidate` are encoded with the state the cached prefix ended on
  (`Tokenizer::encode_continuation`). On a SentencePiece model, that decides whether
  the continuation gets a dummy-prefix space, exactly as for a joined request.
- **The prompt must add at least one token.** The scores come from the last position
  of this request's own prefill. The final hidden state of the cached prefix isn't
  kept, so an empty continuation returns `invalid_input`.
- **The Qwen3.5 hybrid takes single-token candidates**, after a prefix as without one.
  Scoring a multi-token candidate would need the recurrent state copied again for every
  candidate, so it returns `invalid_input`.
- **Gemma 4 and Granite 4.0-H** have no prefix-state format yet. They still score fresh
  prompts and return `unsupported_architecture` with a prefix, before reading any file.

The response:

- **`chosen`** is the index of the highest-probability candidate (the first on a tie).
  It is on every `candidates` response, with or without a prefix. Generate responses
  don't have it.
- **Slot facts.** With `"timings": true`, every response, generating or scoring, reports
  the slot as it is after the request: `pinned_seq_len` (positions held, 0 when empty)
  and `pinned_bytes` (VRAM held). With them an orchestrator can weigh keeping the pin
  against recomputing the prefix or writing a cache file. Without `timings`, responses
  are byte-identical to before.
- **`kv_io_ms`** on a resumed `candidates` request is the device-to-device copy of the
  pinned prefix, or the file read and upload, as on the generate path.

Errors on a `candidates` request:

| request | `error_kind` |
|---|---|
| `pin_prefix` or `export_kv` | `invalid_input` (scoring writes past the prompt, so there's no state right after it to keep) |
| `use_pinned` together with `import_kv` | `invalid_input` |
| `use_pinned` with nothing pinned | `invalid_input` |
| an empty continuation | `invalid_input` |
| a multi-token candidate on the Qwen3.5 hybrid | `invalid_input` |
| `use_pinned` or `import_kv` on Gemma 4 or Granite 4.0-H | `unsupported_architecture` |
| `json_schema` | `invalid_input` |

`scripts/agent_relay_bench.py --system1 <labels>` starts every turn with a System-1
routing request: it scores the comma-separated labels after the previous note and
"The agent who should act on this note next is the". With `--pin-prefix` the request
resumes from the pinned system prompt, with `--reuse-prefix` from its cache file, and
otherwise it sends the whole prompt. The generate hop follows, as before.

### Results (Tesla T4, 2026-10-17)

**Correctness.** `src/ipc/relay_tests.rs` scores the same candidates after a pinned
prefix, after the same prefix's cache file, from the slot a second time, and over the
joined text. The resumed requests give the joined request's token ids and `chosen`,
and its scores up to float summation order. The resumed prefill runs fewer rows than
the joined one, so a few kernels add in a different order, as in the
batched-against-sequential prefill tests:

| model | weights | largest score difference |
|---|---|---:|
| Qwen3-MoE fixture | f32 | 0 (bit-identical) |
| MLA fixture | f32 | 5e-10 (1 ulp) |
| Qwen3.5-MoE hybrid fixture, single-token candidates | f32 | 7e-8 (10 ulps) |
| Qwen3-0.6B, candidates of 1 to 5 tokens | f16 | 4.9e-3, on scores up to 95 |
| TinyLlama-1.1B-Chat (SentencePiece), after raw text and after `</s>` | f16 | 2.6e-3 |

The f32 checks allow 1e-5 relative and the f16 ones 5e-3. The tests also cover the
hybrid's multi-token rejection, Gemma 4 and Granite 4.0-H rejecting a prefix while still
scoring fresh prompts, and the request errors, none of which changes the slot.

**Unchanged without the new fields.** The same requests (generate, streaming,
sampling, `json_schema`, fresh candidates, pin and use, errors) sent to `reflex stdio`
built from this branch and from master give byte-identical lines on Qwen3-0.6B,
TinyLlama and Qwen3.5-0.8B, apart from the new `chosen` on candidates lines. The phase 0
bench gives the same 30 texts. Cold start is unchanged: interleaved
`bench_cold_start_phases.sh` runs, 5 rounds of 2 runs per build, give a median
first token of 630 ms on master and 635 ms here, with rounds spread over 624 to
641 ms. The GPU nightly suite passes.

**The routing hop.** `agent_relay_bench.py --system1
planner,researcher,coder,reviewer,tester,summarizer`: Qwen3-0.6B Q4_K_M (f16 weights),
30 turns after 3 warm-up turns, 6 labels of one token each, 32 greedy tokens per
generate hop. p50 of the second of two identical rounds (the first agreed within
1 ms):

| prefix | System-1 hop from | routing round trip | prefix read/copy | generate round trip | turn |
|---|---|---:|---:|---:|---:|
| short (~40 tokens) | the whole prompt | 22.0 ms | | 230.7 ms | 252.6 ms |
| short | the pinned prefix | 21.4 ms | 0.2 ms | 230.8 ms | 252.2 ms |
| long (~1,490 tokens) | the whole prompt | 284.8 ms | | 509.3 ms | 793.6 ms |
| long | its cache file | 228.9 ms | 107.0 ms | 478.9 ms | 707.7 ms |
| long | the pinned prefix | 45.3 ms | 4.4 ms | 294.6 ms | 339.9 ms |

What this shows:

- **With a long shared prompt, routing against the pin costs 45 ms instead of 285.**
  The copy is 4.4 ms. The rest is prefilling the note and the question over 1,490
  cached positions, plus the LM-head gather for the 6 labels. A whole turn, routing plus
  generating, drops from 794 to 340 ms.
- **A file still pays its read.** 107 ms of the file route's 229 ms is the read; the
  upload into the cache is most of the rest, as phase D found for generation.
- **With a short prompt there's nothing to save.** Prefilling 40 tokens costs about
  what copying them does, so both routes take ~22 ms, under a tenth of the generate
  hop.
- **The decisions match.** Within each prefix the pinned, file and fresh routes chose
  the same label on all 30 turns, with probabilities within 2e-12, and the generate
  hops gave the same 30 texts. (Qwen3-0.6B picks "summarizer" on every turn; the bench
  measures the hop, not the router's quality.)

Raw data: `bench-results/agent-relay-t4-phase-e/` (not committed). `run_e.log` and
`phase-e-out/` hold the bench, byte-identity (`ident/`) and cold-start runs;
`rerun_e.log` and `phase-e-out2/` hold the relay tests and the nightly run on the
final test file.

## Faster cache files (phase F)

Phase D found a cold process reading the long prefix's 342 MB cache file in 284 ms,
against 110 ms in a warm one, and file read plus upload at about a quarter of a cold
hop. Phase F first fixed the read, which applies to every existing file, then added
f16 files as an opt-in.

### Where a cold import's time went

Measured before any change, with a standalone probe on the T4: one fresh process per
variant, CUDA initialized first as in `reflex generate`, the long prefix's file on
`/dev/shm`, 5 runs each (all within ~2 ms):

| import strategy | total | phases (ms / minor faults) |
|---|---:|---|
| the old code's shape: read into a fresh `Vec<u8>`, collect into a fresh `Vec<f32>`, pageable upload | 507 ms | read 180 / 83K, convert 232 / 83K, upload 80 |
| read straight into the final `Vec<f32>` | 259 ms | read 179 / 83K, upload 80 |
| the same with `MADV_HUGEPAGE` | 257 ms | faults unchanged (THP is `madvise`-only and didn't apply) |
| an anonymous `MAP_POPULATE` buffer, then read | 275 ms | populate 114 / 83K, read 81, upload 80 |
| mmap the file, pageable upload from its pages | 96 ms | 5.2K faults (64 KB fault-around) |
| the same with `MAP_POPULATE` | 96 ms | populate 16, upload 80 |
| `cuMemHostRegister` on the mapping | | `CUDA_ERROR_NOT_SUPPORTED` on tmpfs |
| two 64 MB pinned buffers, `read()` into them, async upload | 152 ms | `cuMemHostAlloc` 59 / 33K, stream 92 |
| two 8 MB pinned buffers, `read()` into them, async upload | **64 ms** | alloc 7.6, stream 56 |
| the same with 2 or 4 `pread` threads per chunk | 63 ms | stream 55.5: the upload (~6.2 GB/s) is the bound, not the read |

- **Every fresh page costs a fault.** The old import read the payload into one fresh
  buffer and converted it into another. The real binary freed each slice's byte buffer
  before reading the next, so it faulted about half as much as the probe's replica:
  its `kv_io_ms` was 276 to 281 ms.
- **Pinned memory is expensive to allocate in a cold process**: about 0.46 ms per MB.
  Staging through the weight loader's 64 MB chunk size would cost as much as the
  whole upload, so cache files use 8 MB chunks.
- **The read itself is cheap.** With pinned staging the upload is the bound, so more
  read threads don't help.

### The read fix

`kv_io::import_kv` now reads only the header. It checks that the file is exactly as
long as the header says and returns file ranges. Each range is then read with
positioned reads straight into the model's pinned staging buffers, 8 MB at a time
(`WeightLoadPipeline::upload_state`, the same two slots the weight load uses). The
next chunk is read while the previous one is on its way to the GPU. The copies run on
the compute stream, so they follow the zeroing of the new cache and come before the
prefill that reads it. Nothing else changes:
- The file formats and every existing version are unchanged, and what an import puts
  in the cache is bit-identical.
- A truncated, overlong or otherwise inconsistent file is rejected at import, before
  any model work (`kv_cache`).
- The caller must not change a file while a request resumes from it. A file that
  shrinks in between fails that request with `kv_cache`; it doesn't crash the
  process, as a memory-mapped file would.
- `kv_io_ms` now covers the read and the upload, since they overlap. Work already
  queued on the GPU, such as the end of a cold weight load, is waited for before
  the timing starts.
- Exports write each buffer's values directly instead of copying them into a byte
  buffer first.

### f16 files

An export can ask for f16 values: `"export_kv_dtype": "f16"` over IPC,
`--export-kv-dtype f16` on the CLI. Versions 5, 6 and 7 are the dense, hybrid and MLA
layouts with f16 values (see [reference.md](../reference.md#kv-cache-files)). Every
format imports, whichever was written.
- **Opt-in, not the default.** An f32 file resumes exactly, as before; an f16 file
  doesn't. Making f16 the default would change every existing caller's outputs and
  files. A caller that wants the smaller file asks for it. Without the field, requests,
  responses and files are byte-identical to before.
- **The conversions run on the GPU.** Export rounds each value to nearest even
  (`cast_act_f16_kernel`) and downloads f16 bits. A value beyond f16's range fails the
  export instead of becoming infinite. Import copies each chunk to the GPU once and
  widens its f16 runs in one launch (`f16_to_f32_runs_kernel`, exact).
- **The working cache stays f32.** Only the file's values are rounded.
- **The hybrid's GatedDeltaNet state stays f32** (version 6). llama.cpp also keeps
  recurrent state in f32 next to an f16 KV cache. The measured cost of rounding it
  too was small: on Qwen3.5-0.8B with f32 weights and a ~600-token prefix, rounding
  the attention K/V alone moved the next token's logits by 1.04e-4 (relative L2) and
  rounding the recurrent state as well by 1.19e-4, with the same 32 greedy tokens. The
  state is a fixed ~20 MB, so keeping it f32 makes a short hybrid prefix's f16 file
  barely smaller (20.4 against 20.5 MB). Rounding it too is an option if smaller
  hybrid files matter.

### Accuracy

f16 storage keeps 11 significant bits. `src/ipc/relay_tests.rs` exports each prefix
twice, as f32 and as f16, resumes from both over IPC, and compares with one request
over the joined text. `src/model/kv_f16_tests.rs` rounds a pinned prefix's buffers in
place to isolate the effect.

| model | weights | next-token logits, f16 file vs f32 file (rel L2 / max abs) | greedy tokens |
|---|---|---|---|
| Qwen3-0.6B | f16 | 7.0e-4 / 1.4e-2 | identical |
| Qwen3-0.6B, buffers rounded in place | f32 | 6.1e-4 / 6.5e-3 | 32/32 identical |
| TinyLlama-1.1B-Chat (SentencePiece), after raw text | f16 | 2.1e-4 / 4.6e-3 | identical |
| TinyLlama-1.1B-Chat, after `</s>` | f16 | 6.0e-4 / 7.5e-3 | identical |
| Qwen3.5-0.8B hybrid | f16 | 1.9e-4 / 4.1e-3 | identical |
| Qwen3.5-MoE hybrid fixture | f32 | 1.8e-4 / 2.1e-4 | identical |
| Qwen3-MoE fixture | f32 | 5.5e-7 / 7.5e-9 | identical |
| MLA fixture | f32 | 2.7e-6 / 5.1e-8 | identical |

- For comparison, the f32 file against the joined request (fewer prefill rows, so a
  different summation order) is 8.5e-4 on Qwen3-0.6B with f16 weights. f16 weights
  already round the activations about as much as an f16 file rounds the cache.
- The tests allow 1e-2 relative L2 against the f32 file and require the joined
  request's greedy ids. The CLI's 32-token resumes and all 30 hops of every bench chain
  gave the same text from an f16 file as from master's f32 one.

### Results (Tesla T4, 2026-10-17)

Same setup as phase D: Qwen3-0.6B Q4_K_M (f16 weights), 30 hops after 3 warm-up
hops, 32 greedy tokens per hop, files on `/dev/shm`, `REFLEX_CUDA_ARCH=sm_75`. Master
is f79231d. p50 per hop:

| prefix | target | master | read fix, f32 file | read fix, f16 file |
|---|---|---:|---:|---:|
| long | `cold` (recompute, no file) | 1,329 ms | 1,336 ms | |
| long | `cold-chain` hop | 1,530 ms | **1,203 ms** | **1,182 ms** |
| long | `cold-chain` first token | 1,108 ms | 792 ms | 777 ms |
| long | `cold-chain` read (+ upload after the fix) | 296 ms (+~80) | 67 ms | 44 ms |
| long | file size | 342 MB | 342 MB | 171 MB |
| long | `cold-warm` hop | 2,233 ms | 1,851 ms | 1,589 ms |
| long | `cold-warm` write / read | 230 / 291 ms | 153 / 76 ms | 77 / 44 ms |
| long | `stdio` with file reuse, round trip | 479 ms | 348 ms | 321 ms |
| long | `stdio` with file reuse, first token | 223 ms | 91 ms | 65 ms |
| short | `cold` | 1,071 ms | 1,067 ms | |
| short | `cold-chain` hop | 1,075 ms | 1,074 ms | 1,066 ms |
| short | `cold-warm` hop | 1,071 ms | 1,064 ms | 1,052 ms* |
| short | `stdio` with file reuse, round trip | 237 ms | 234 ms | 233 ms |

\* From the final build, run after this table's runs. The first f16 import widened
each cached buffer with its own copy and kernel launch; in a warm process that had
idled between requests, those 56 launches added ~9 ms to a short file's read (16 against
6.6 ms for f32). The final build copies each 8 MB chunk once and widens its buffers in
one launch: 6.6 ms, the same as f32. Its long-prefix rerun was 1 to 2% slower across
the board (box drift), with the same gaps: `cold-chain` 1,222 ms (f32 file) and
1,195 ms (f16) against 1,344 ms for `cold`.

GPU energy per hop, gross / net of idle (cold targets p50, warm ones the bracket's
mean):

| prefix | target | master | read fix, f32 file | read fix, f16 file |
|---|---|---:|---:|---:|
| long | `cold` | 73.4 / 25.9 J | 73.3 / 26.0 J | |
| long | `cold-chain` | 71.8 / 18.7 J | 61.9 / 18.7 J | 61.8 / 19.8 J |
| long | `cold-warm` | 103.8 / 27.8 J | 90.2 / 27.9 J | 80.8 / 27.6 J |
| long | `stdio` with file reuse | 26.7 / 10.5 J | 22.2 / 10.2 J | 21.4 / 10.3 J |
| short | `cold-chain` | 56.2 / 17.0 J | 56.2 / 17.5 J | 56.1 / 16.9 J |

What this shows:
- **A cold chain now beats recomputing the prefix.** With the long prefix, `cold-chain`
  takes 1,203 ms against 1,336 ms for `cold`, 10% less. Before the fix it was 177 to
  200 ms slower. The read fix alone does this; it also applies to f32 files written
  before it.
- **f16 adds a little on top.** It halves the file and the read (67 to 44 ms) and the
  write (153 to 77 ms): 21 ms per `cold-chain` hop, 262 ms per `cold-warm` hop, which
  writes a file every hop.
- **Warm processes gain too.** Reusing the long prefix from a file inside `reflex stdio`
  went from a 223 to a 91 ms first token, or 65 ms with an f16 file. A pinned prefix
  is still faster (38.6 ms, phase B), since nothing leaves the GPU.
- **Energy follows the GPU work.** Net energy per hop is unchanged: the fix removes
  host work, not GPU work. Gross energy drops with the shorter hop, by 10 J per long
  `cold-chain` hop, since the idle draw runs for less time.
- **Short prefixes are unchanged.** Their files are 10 MB or less.
- **Faults.** One cold long-prefix hop went from 193.8K minor faults to 112.5K, against
  about 106K to 110K without any import.

Unchanged elsewhere: the GPU nightly suite passes; requests without the new field give
byte-identical IPC responses and cache files on Qwen3-0.6B, TinyLlama and
Qwen3.5-0.8B, and the CLI's default export is byte-identical, including the 342 MB
long prefix; the phase 0 bench gives the same 30 texts; cold start is unchanged
(interleaved `bench_cold_start_phases.sh`, 5 rounds of 2 runs: per-round p50 first
token 690 to 695 ms on master, 688 to 696 ms here).

Raw data: `bench-results/agent-relay-t4-phase-f/` (not committed): `phase-f-step1/`
(the probe), `phase-f1-out/` (the read fix alone), `phase-f2-out/` (both, with tests,
nightly, byte identity, CLI checks and cold start), `phase-f3-out/` and
`phase-f4-out/` (the `kv_io_ms` and f16-upload follow-ups).

## Sidecar surface (phase G)

The OpenAI sidecar (`sidecar/openai-adapter/`) is the only HTTP surface. Phase G maps
`response_format` on `/v1/chat/completions` onto IPC `json_schema`:

| `response_format` | IPC `json_schema` |
|---|---|
| omitted, or `{"type": "text"}` | none |
| `{"type": "json_schema", "json_schema": {"schema": S}}` | `S`, as written |
| `{"type": "json_object"}` | `{"type": "object"}` (any object) |
| any other `type`, or `json_schema` without `schema` | none: the sidecar returns `400` |

The schema is kept as raw text (`serde_json::value::RawValue`) from the HTTP body to
the IPC line, with only the whitespace between tokens removed, since an IPC request is
one line. A `serde_json::Value` would sort the keys, and the properties would then come
out in alphabetical order rather than schema order. `name`, `description` and `strict`
are accepted and not used: the engine always enforces the schema.

The engine's errors map to OpenAI-style errors:
- a schema outside the subset is `400` `invalid_request_error`, code `invalid_input`;
- `grammar_dead_end` (`max_tokens` ran out before the value was complete) is `400`
  `invalid_request_error`, code `grammar_dead_end`.

Both reach a streaming client as a real status when they come before the first token,
as a schema error always does. A failure after the stream has started is one
`{"error": ...}` event followed by `data: [DONE]`, with no `finish_reason` chunk, so a
truncated value never looks complete. A completed value ends at EOS, so its
`finish_reason` is `"stop"`.

`/v1/classify` already returned what an orchestrator needs from a System-1 response:
`label`/`label_index` (the engine's `chosen`, with the same first-on-a-tie rule),
`entropy`, and each label's `probability`, `score` and token count. It is unchanged.

HTTP requests still can't reach the pinned slot or cache files: `pin_prefix`,
`use_pinned`, `import_kv` and `export_kv` are ignored, like any field the sidecar
doesn't know, and the sidecar writes only the IPC fields it names. Unit tests check
both endpoints' IPC lines byte for byte.

End to end on a Tesla T4 (2026-10-18), `reflex stdio` with Qwen3-0.6B-Q4_K_M behind
the sidecar, greedy:
- a `json_schema` request listing `name`, `age`, `city` returned
  `{"name": "Lila", "age": 25, "city": "New York"}`, keys in schema order, with
  `finish_reason: "stop"`. Streaming gave the same text in 25 events. `json_object`
  gave a parsing object. Without a schema the model starts with `<think>`;
- `pattern` came back `400` `invalid_input` naming the keyword, and so did a
  `required` name missing from `properties` on a streaming request, before any event;
- `max_tokens: 4` gave `400` `grammar_dead_end`, and streaming gave four content
  chunks, the error event and `[DONE]`;
- a request with `pin_prefix`, `use_pinned`, `import_kv` and `export_kv` ran as a plain
  request and wrote no file; `/v1/classify` gave the same probabilities as before.

## Planned next phases

Round 2 is planned in [agent-relay-round2.md](agent-relay-round2.md). Phases F and G
are done (above); next are composition examples (phase H).

## Swarm pattern

Priority, routing and fan-out stay outside the engine:
- Run one warm `reflex uds` (or `stdio`) process per agent role or per GPU.
- An orchestrator decides which request goes where, and in what order.
- Short-lived sub-agents can be fresh `reflex generate` processes, since cold start
  is what Reflex is built for.
