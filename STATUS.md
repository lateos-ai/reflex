# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `README.md` — this file is the
short, current-state summary; README is the log.

_Last updated: 2026-09-18_

## MVP progress

| Step | Status |
|---|---|
| 1. Dense Qwen3 | Done — real-hardware-verified (A6000) |
| 2. Qwen3-MoE | Done — real-hardware-verified (A6000), against a synthetic (non-Qwen3) MoE fixture |
| 3. Qwen3.5 hybrid Gated DeltaNet mixer | Done — real-hardware-verified (A6000) against a real `Qwen3.5-0.8B-Q4_K_M.gguf` fixture, cross-checked byte-exact against a fresh `llama.cpp` build. Scope: dense `qwen35` only (`qwen35moe`, MTP/NextN unsupported), single-token sequential dispatch (no chunked prefill). |
| 4. DeepSeek-V2/V3 MLA | Not started (deliberately last) |

## Performance

First real cold-start A/B benchmark vs. `llama.cpp` (`972d231`, same A6000, same
`Qwen3-0.6B-Q4_K_M.gguf`, same prompt, full GPU offload, `n=3`):

| | coldstart-infer (initial) | coldstart-infer (after Phase 2 round 1) | llama.cpp |
|---|---|---|---|
| wall clock | 28–30s | 10.4–11.7s | 6.5–6.6s |
| peak RSS | 3.68 GB | 1.33 GB | 900 MB |
| sys CPU time | 14.02s | ~3.2–4.0s | 1.27s |

Root cause of the initial 4.3x gap: `model.rs`'s weight-loading path dequantized every
tensor to a host `f32` buffer *and* re-uploaded that buffer to the GPU on every single
`gemv`/`gemv_expert`/`rmsnorm` call (every layer, every token). Fixed in Phase 2 round 1
by making `Weight` hold a `CudaSlice<f32>` uploaded once. Current gap: **~1.7x slower
than llama.cpp**, believed dominated by the remaining CPU-bound host-side dequant step
itself (llama.cpp dequantizes/matmuls on-GPU, never materializing a full-`f32` host
copy) plus per-op host↔device round-tripping of activations.

## Open decision (not yet resolved)

Next work is unresolved between two options — ask the user before picking one:
1. Start MVP step 4 (DeepSeek-V2/V3 MLA).
2. Phase 2 Fast IO round 2 (on-GPU dequant kernel, and/or keeping activations
   device-resident across a whole layer) to narrow the remaining ~1.7x gap further.

## Known debt / limitations

- **Remote instance git state**: the ThunderCompute dev instance is now a fresh instance
  (created 2026-09-18, replacing the prior `dbl5elf0` instance referenced in earlier
  session history — Thunder Compute instances are ephemeral across sessions). Its
  `~/coldstart-infer` working tree has no git history at all (populated by `rsync` from
  local, not `git clone`/`scp`); local remains the only git-tracked copy. A real
  `ggml-org/llama.cpp` checkout was also built from source there (`~/llama.cpp`, CUDA
  enabled, `examples/simple`'s `llama-simple` + `tools/cli`'s `llama-cli` built) for
  cross-checking new architecture work against independent ground truth — worth reusing
  rather than rebuilding if the instance survives to the next session.
- **`llama-cli`'s newer conversational mode always applies the model's chat template**,
  even when a raw prompt is passed via `-p` — no `--no-cnv` flag exists in the current
  build. Use `examples/simple`'s `llama-simple` binary instead for true prompt-in/
  token-out comparisons with no chat wrapping.
- **MoE test fixture**: `test-data/Tiny-Moe.Q4_K_M.gguf` is a Mixtral-style synthetic
  model (`general.architecture="llama"`, `expert_count=2`, `expert_used_count=2` — top-k
  always selects every expert, so it can't prove routing actually excludes an expert; no
  QK-Norm tensors, so it doesn't exercise QK-Norm+MoE together). A real small
  `qwen3moe`-architecture fixture (or one with `expert_used_count < expert_count`)
  doesn't exist yet locally.
- **Cargo/binary staleness gotcha**: `cargo test --release` does not rebuild
  `target/release/<bin-name>` — only `target/release/deps/`. After any source change,
  run `cargo build --release --bin <name>` explicitly before trusting a binary run
  against real hardware.
