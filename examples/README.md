# Examples

Small, runnable templates for common ways to use Reflex. They are scripts, not Rust
code: this project is CLI-first (see
[docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md#non-goals)'s Non-goals), and its outputs
are stable contracts (`reflex system1`'s `KEY=value` lines, the `--json` lines, and
the `reflex stdio` JSON protocol).

**Requirements**: a release build and a real GPU with a local dense/MoE Qwen3 GGUF.
There is no CPU fallback (see
[docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md#no-cpu-fallback)).

## System-1 decision gates

`system1_guardrail.sh` drives `reflex system1` (single-pass candidate scoring, no
decode loop) as a fast local decision gate. Build with `cargo build --release --bin
reflex`. See the main [README](../README.md)'s "Example: download a model from Hugging
Face, then run a System1 test" section for what `score`/`probability`/`entropy` mean.

| Script | Pattern |
|---|---|
| `system1_guardrail.sh bool` | Boolean gate: ALLOW / DENY |
| `system1_guardrail.sh multi` | Multi-action selection: pick one of N named actions |
| `system1_guardrail.sh confidence` | Calibrated confidence gate: threshold on probability/entropy, escalate when unsure |

```
examples/system1_guardrail.sh bool <path-to-gguf> "<yes/no prompt>"
examples/system1_guardrail.sh multi <path-to-gguf> "<prompt>" -- <action1> [action2 ...]
examples/system1_guardrail.sh confidence <path-to-gguf> "<prompt>" -- <candidate1> [candidate2 ...]
```

## Composing agents

Three agents chained by an orchestrator: a router, a worker and a sub-agent. Routing,
the envelope and the cache files belong to the orchestrator; the engine only answers
one request at a time. See
[docs/design/agent-relay.md](../docs/design/agent-relay.md#composing-agents-phase-h).
Build with `cargo build --release --features ipc --bin reflex`, which also enables
`--json` output.

| File | What it is |
|---|---|
| `relay_envelope.schema.json` | JSON Schema (draft 2020-12) for the envelope each hop's result is passed on in: a version, a System-1 decision, carry text, a cache file and timings |
| `relay_orchestrator.py` | Python 3.8+, stdlib only. A router and a worker as warm `reflex stdio` processes, each with a pinned role prefix: the router makes a System-1 decision against its pin, the worker writes a JSON Schema handoff and exports its context to a cache file, and a cold `reflex generate --import-kv` sub-agent resumes from that file |
| `relay_chain.sh` | POSIX sh. The cold version: a fresh process per hop, handing off through files (the shared prefix's cache file and the carry text) |

```
examples/relay_orchestrator.py --reflex target/release/reflex --model <path-to-gguf> [--task "<task>"] [--weights f32] [--kv-dtype f16]
examples/relay_chain.sh target/release/reflex <path-to-gguf> ["<task>"]
```

Both print one envelope per hop as a JSON line on stdout. Errors go to stderr with a
non-zero exit. Cache files go to `/dev/shm` when it exists and are deleted at the end
(`--keep-kv`, or `KEEP_FILES=1` for the shell script, keeps them).
