# Agent relay, round 2: composable hops for inter-agent communication

Status: planned (2026-10-16). Builds on agent relay phases 0 to C
([agent-relay.md](agent-relay.md)).

## Why

Multi-agent systems already need an orchestrator. Reflex helps most by making each
decision hop as cheap as possible in latency and energy, and by making hops easy to
chain through local, sequential interfaces and explicit handoff files. The Non-goals
in [DEVELOPMENT.md](../DEVELOPMENT.md) stay as they are:
- `batch_size` is 1;
- the core has no network socket;
- the warm loop holds exactly one pinned prefix;
- the engine has no scheduler or queue.

Agents talk to each other through the orchestrator. The engine never knows other
agents exist.

What already exists, so this round doesn't redo it:
- IPC responses carry System-1 `candidates[]` (`score`, `probability`), `entropy` and
  opt-in `timings` (`src/ipc.rs`, `IpcResponse`/`IpcTimings`).
- File handoff (`import_kv`/`export_kv`), one VRAM slot (`pin_prefix`/`use_pinned`,
  `ipc::PrefixSlot`) and JSON Schema output (`json_schema`).
- The sidecar's `/v1/chat/completions` and `/v1/classify`, which does System-1 scoring
  via `candidates` (`sidecar/openai-adapter/src/classify.rs`).
- `reflex-energy` (`src/bin/reflex-energy.rs`), which measures a command's GPU energy
  from outside the process.

Gaps:
1. **System-1 can't resume from a prefix.** `handle_request` rejects `candidates` when
   any prefix field is set (`kv_fields_on_candidates_err` in `src/ipc.rs`). So an agent
   that has pinned its role and context can't score a decision against it, and that is
   the most common System-1 hop.
2. **Nothing measures multi-process chains.** `scripts/agent_relay_bench.py` drives one
   warm process. There are no numbers for cold → cold or cold → warm hops through files,
   and no joules per hop.
3. **The orchestrator can't see the slot's state.** Responses don't carry its `seq_len`
   or byte size (`PrefixSlot` already has both accessors), so deciding between pin,
   recompute and file means tracking that state on the client.
4. **There is no chosen-candidate field.** Every client recomputes the argmax.
5. **The sidecar doesn't map `response_format: json_schema`** onto IPC `json_schema`.
6. **Cache files are f32.** That is 229 KB per token for Qwen3-0.6B, or 342 MB for a
   1.5K-token prefix, which makes files the slow path. Worth fixing only if
   measurement shows a chain waits on the file.

## Deliberately out of scope

- **An engine-side "decision envelope" object, carry text, blob references or a state
  summary.** The engine already returns every fact an envelope needs. A versioned
  envelope that holds carry text and file paths is a protocol between agents, so it
  belongs to the orchestrator. It ships as a documented JSON Schema and a small Python
  helper in `examples/`, which builds it from IPC responses. The engine only adds the
  facts it doesn't report yet (gaps 3 and 4). A "state summary" has no defined meaning
  beyond the KV file.
- **Zero-copy pinned use.** Each use must copy the slot: a request writes its own
  cache past the prefix, and the pinned state must stay unchanged. The copy already
  takes only 4.4 ms for 342 MB on a T4.
- **A second slot, named prefixes or eviction.**
- **A2A cards, discovery, a registry, encryption or message queues**, whether in the
  core or in the sidecar. The sidecar only grows request and response fields that map
  onto existing IPC.
- **Features whose value shows only under concurrent load.**
- **Decode speed** is a separate track (see the end). It is not bundled here.

## Phases

Each phase gets its own branch and PR, verified on a T4.

### Phase D: measure multi-process chains and energy (first; decides F)

`scripts/agent_relay_bench.py` gets two new targets:
- **`cold-chain`.** Every hop is a fresh `reflex generate` process with
  `--import-kv`/`--export-kv`. An export needs `max_tokens` 1, so if the CLI can't
  import, export and decode in one call, a hop is an export process followed by a
  decode process. The doc records whichever shape the CLI supports.
- **`cold-warm`.** A cold process exports, and a warm `stdio` process imports.

It also gets an `--energy` flag:
- Each spawned process runs under `reflex-energy`, and the bench parses its
  `REFLEX_ENERGY_OK` line.
- Warm targets bracket the whole chain once, since the counter is device-wide.

Per hop it reports wall time, first token, file write and read, and joules, both
gross and net of idle. The comparison rows are a llama.cpp `llama-cli` cold chain
(the model reloads every hop) and the existing llama-server warm target.

T4 numbers go in [agent-relay.md](agent-relay.md) under a new "Multi-process chains
(phase D)" section. The setup is the same as phases 0 to C: Qwen3-0.6B, with the
short prefix and the ~1,490-token one.

### Phase E: System-1 over a resumed prefix, and facts for the orchestrator (main deliverable)

Engine (`src/model/`):
- `system1_evaluate` takes a `Resume`. The dense/MoE path (`system1_evaluate_dense`,
  whose `prefill_dense_batched` already takes an optional prefix) reuses the prefix
  source in `src/model/prefix.rs`, so pinned and file prefixes share one path, as in
  phase B.
- MLA works the same way.
- On the Qwen3.5 hybrid, multi-token candidates would need the recurrent state copied
  again for each candidate. It starts with single-token candidates on a resumed
  prefix, and a multi-token candidate returns `invalid_input`, matching its current
  System-1 limit.
- Gemma 4 and Granite 4.0-H keep returning `unsupported_architecture`.
- Candidates are tokenized as continuations (`Tokenizer::encode_continuation`, the
  SentencePiece rule).

IPC (`src/ipc.rs`):
- `use_pinned` and `import_kv` are allowed with `candidates`. `pin_prefix` and
  `export_kv` are still rejected there, since scoring writes past the prompt.
- **`chosen`**: the index of the highest-probability candidate. It appears on the
  candidates path only.
- **Slot facts**, opt-in under `"timings": true` so default responses stay
  byte-identical: `pinned_seq_len` and `pinned_bytes` after the request. With these
  the orchestrator can choose between pin, recompute and file.
- The module doc comment and the error table are updated.

Tests (`src/ipc/relay_tests.rs`):
- A candidates request over `use_pinned`, and one over `import_kv`, must give the same
  scores and probabilities as one request over the joined text.
- Matching is bit-exact with f32 weights and within a tolerance with f16, the same rule
  as phase B.
- Coverage: dense, MoE, MLA, the hybrid with single-token candidates, and SentencePiece
  (TinyLlama).
- The GPU nightly gets requirement rules for the new tests.

The bench gets `--system1 <labels>`, which adds a scoring hop against the pinned prefix
on every agent turn.

### Phase F: smaller cache files (only if phase D's gate passes)

- The cache file payload is written as f16, as a new KV format version for each
  architecture. The current f32 versions still import. This halves the file and the
  read.
- **Gate:** in phase D's cold chains, file read plus upload is more than about 20% of a
  hop. If it isn't, the numbers get recorded and this phase is skipped.

### Phase G: sidecar surface

`sidecar/openai-adapter/`:
- `/v1/chat/completions` maps `response_format: {"type": "json_schema", "json_schema":
  {"schema": ...}}` onto IPC `json_schema`. The schema passes through as raw JSON so
  property order is kept: a `serde_json::Value` without `preserve_order` sorts keys.
- `/v1/classify` passes `chosen`, `entropy` and `probability` through. Only the ones
  it doesn't expose already get added.
- HTTP gets no pin, slot or file fields. Those stay on local IPC, where the orchestrator
  uses them.

### Phase H: composition docs and examples

- A "Composing agents" section in [agent-relay.md](agent-relay.md), covering what
  belongs in Reflex and what belongs outside it. The out-of-scope list above is the
  boundary. One line mirrors it in [DEVELOPMENT.md](../DEVELOPMENT.md)'s Non-goals.
- `examples/relay_envelope.schema.json`: the orchestrator-side decision envelope.
  - the decision: label, probability, entropy;
  - carry text;
  - an optional KV file path;
  - timings.
- `examples/relay_orchestrator.py`: a stdlib-only loop with three agents. It builds an
  envelope from each IPC response and runs against a real model. It shows:
  - one warm process per role, each with a pinned role prefix;
  - a System-1 routing decision against the pinned prefix (phase E);
  - a JSON Schema handoff;
  - a cold sub-agent resumed through `--import-kv`.
- `examples/relay_chain.sh`: the cold-chain version in shell.
- `examples/README.md` and the CHANGELOG are updated.

### Separate track: decode speed

Decode is still the largest cost in every hop: 8.3 ms per token against llama.cpp's
4.7 ms at 1.5K context ([decode-speed.md](decode-speed.md)). The options are a 4-row
Q6_K LM head, an f16 KV cache, rmsnorm fusion and CUDA graphs. They are planned
separately.

## Order

D, then E, G and H. F goes in after D only if its gate passes. E is the main
deliverable. D comes first because its numbers decide F and give H's examples real
figures.

## Verification

- **Host checks.** `cargo fmt`, `cargo clippy` and `cargo test`, including new IPC
  parse tests for `chosen`, the slot facts, and candidates combined with prefix fields.
  The sidecar's `cargo test` too.
- **GPU (T4).**
  - Rebuild with `cargo build --release --features ipc --bin reflex`, since
    `cargo test` doesn't rebuild the binary.
  - Run the new relay tests and the full nightly with requirement rules: 0 failures.
- **Byte-identity.** Requests that don't use the new fields give the same responses as
  master. The checks are the existing IPC tests plus a diff of a phase 0 bench run's
  outputs.
- **Cold start unchanged.** `bench_cold_start_phases.sh`, 10 interleaved runs against
  master. Phase E adds no work at load.
- **Bench results.** Phase D's numbers, and the `--system1` hop, are published in
  [agent-relay.md](agent-relay.md).
- **The example.** `examples/relay_orchestrator.py` runs end to end on the T4 with
  Qwen3-0.6B, and every envelope validates against `relay_envelope.schema.json`.
