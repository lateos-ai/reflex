# Examples

Small, runnable templates showing common ways to use Reflex's `system1`
subcommand (single-pass candidate scoring, no autoregressive decode loop) as a
fast local decision gate. See the main [README](../README.md)'s
"Example: download a model from Hugging Face, then run a System1 test" section
for the underlying command and what `score`/`probability`/`entropy` mean.

These are shell scripts, not Rust code — this project is CLI-first (see
CLAUDE.md's Non-goals), and `reflex system1`'s stdout is already a stable,
parseable `KEY=value` contract (see `scripts/bench_cold_system1_vs_jev.sh` for
another script that relies on it).

**Requirements**: a release build (`cargo build --release --bin reflex` from
the repo root) and a real GPU with a local dense/MoE Qwen3 GGUF — there is no
CPU fallback for `system1` (see CLAUDE.md).

| Script | Pattern |
|---|---|
| `system1_guardrail.sh bool` | Boolean gate: ALLOW / DENY |
| `system1_guardrail.sh multi` | Multi-action selection: pick one of N named actions |
| `system1_guardrail.sh confidence` | Calibrated confidence gate: threshold on probability/entropy, escalate when unsure |

Usage:

```
examples/system1_guardrail.sh bool <path-to-gguf> "<yes/no prompt>"
examples/system1_guardrail.sh multi <path-to-gguf> "<prompt>" -- <action1> [action2 ...]
examples/system1_guardrail.sh confidence <path-to-gguf> "<prompt>" -- <candidate1> [candidate2 ...]
```
