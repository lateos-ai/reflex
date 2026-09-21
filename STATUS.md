# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `README.md` — this file is the
short, current-state summary; README is the log.

_Last updated: 2026-09-21 (MLA session)_

## MVP progress

| Step | Status |
|---|---|
| 1. Dense Qwen3 | Done — real-hardware-verified (A6000) |
| 2. Qwen3-MoE | Done — real-hardware-verified (A6000), against a synthetic (non-Qwen3) MoE fixture |
| 3. Qwen3.5 hybrid Gated DeltaNet mixer | Done — real-hardware-verified (A6000) against a real `Qwen3.5-0.8B-Q4_K_M.gguf` fixture, cross-checked byte-exact against a fresh `llama.cpp` build. Scope: dense `qwen35` only (`qwen35moe`, MTP/NextN unsupported), single-token sequential dispatch (no chunked prefill). |
| 4. DeepSeek-V2/V3 MLA | Done, narrowly scoped (dense-only, no Q-LoRA/YaRN/MTP) — verified against a synthetic fixture, not a real pretrained model (see below) |

## Performance

First real cold-start A/B benchmark vs. `llama.cpp` (`972d231`, same A6000, same
`Qwen3-0.6B-Q4_K_M.gguf`, same prompt, full GPU offload, `n=3`):

| | coldstart-infer (initial) | coldstart-infer (after Phase 2 round 1) | coldstart-infer (after Phase 2 round 2) | llama.cpp |
|---|---|---|---|---|
| wall clock | 28–30s | 10.4–11.7s | 6.4–8.5s | 6.44–6.46s |
| peak RSS | 3.68 GB | 1.33 GB | 1.35 GB | 887 MB |
| user+sys CPU time | 5.13s + 14.02s | ~3.2s + ~4.0s | ~2.1s + ~3.5s | ~1.0s + ~1.2s |

Root cause of the initial 4.3x gap: `model.rs`'s weight-loading path dequantized every
tensor to a host `f32` buffer *and* re-uploaded that buffer to the GPU on every single
`gemv`/`gemv_expert`/`rmsnorm` call (every layer, every token). Fixed in Phase 2 round 1
by making `Weight` hold a `CudaSlice<f32>` uploaded once, closing the gap to ~1.7x.
Phase 2 round 2 then converted every kernel-wrapper op (`rmsnorm`/`gemv`/`rope`/
`silu_and_mul`/`attention`/the GDN mixer's kernels) to chain `CudaSlice<f32>` device
buffers through a whole layer instead of round-tripping each op's activations over
PCIe, and made the K/V cache device-resident (written via device-to-device copy)
instead of re-uploading its full history every token position. Current gap: **~1.1x
slower than llama.cpp** (peak RSS/CPU time gap unchanged from round 1 — round 2 didn't
touch weight loading — so the residual gap is still attributed to the CPU-bound
host-side dequant step, per llama.cpp's on-GPU dequant/matmul approach never
materializing a full-`f32` host copy at all).

## MLA (MVP step 4)

DeepSeek-V2/V3 support is real but narrow: dense-only (no MoE FFN/shared experts), no
Q-LoRA query decomposition, no YaRN RoPE scaling, no MTP — `parse_mla_config` in
`model.rs` hard-errors with a clear message on any of these. Verified against a fully
synthetic `deepseek2` GGUF (`test-data/deepseek-tiny-mla.gguf`), not a real pretrained
model — no small real `deepseek2`-architecture GGUF exists publicly, and the smallest
real one (DeepSeek-V2-Lite) needs ~63GB of `f32` device memory under this project's
GPU-residency design, more than the A6000 this project develops against. See
README.md's MLA section for the full writeup, including two easy-to-miss correctness
details (the attention scale's dimension, and MLA's different RoPE rotation
convention) that produced silently-wrong (non-crashing) output before being caught by
byte-exact comparison against a real llama.cpp build.

## Open decision (not yet resolved)

Next work is unresolved between two options — ask the user before picking one:
1. Phase 2 Fast IO round 3: an on-GPU dequant kernel, to close the remaining ~1.1x
   cold-start gap vs. llama.cpp.
2. Extend MLA to real DeepSeek-V2-Lite (MoE FFN + shared experts + an ~80GB H100
   instance, since it won't fit the A6000 — see above).

## Known debt / limitations

- **Remote instance git state**: the ThunderCompute dev instance is now a fresh instance
  (`2xhxwa87`, created 2026-09-21, replacing the prior `pn6lxmbv` instance — Thunder
  Compute instances are ephemeral across sessions, confirmed again this session). Its
  `~/coldstart-infer` working tree has no git history at all (populated by `rsync` from
  local, not `git clone`/`scp`); local remains the only git-tracked copy. A real
  `ggml-org/llama.cpp` checkout was built from source there (`~/llama.cpp` at `ce8caa6`,
  CUDA enabled, `examples/simple`'s `llama-simple` built) for the Phase 2 round 2 A/B
  benchmark — worth reusing rather than rebuilding if the instance survives to the next
  session. `cmake`/GNU `time` were not preinstalled on this instance and had to be
  `apt-get install`ed.
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
