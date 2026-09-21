# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `README.md` — this file is the
short, current-state summary; README is the log.

_Last updated: 2026-09-21 (Phase 3 round 1 session)_

## MVP progress

| Step | Status |
|---|---|
| 1. Dense Qwen3 | Done — real-hardware-verified (A6000) |
| 2. Qwen3-MoE | Done — real-hardware-verified (A6000), against a synthetic (non-Qwen3) MoE fixture |
| 3. Qwen3.5 hybrid Gated DeltaNet mixer | Done — real-hardware-verified (A6000) against a real `Qwen3.5-0.8B-Q4_K_M.gguf` fixture, cross-checked byte-exact against a fresh `llama.cpp` build. Scope: dense `qwen35` only (`qwen35moe`, MTP/NextN unsupported), single-token sequential dispatch (no chunked prefill). |
| 4. DeepSeek-V2/V3 MLA | Done — dense-only, MoE+shared-expert, and YaRN RoPE scaling all supported (only Q-LoRA/MTP still rejected); verified against both a synthetic fixture and the real `DeepSeek-V2-Lite` checkpoint on an 80GB A100 (see below) |

## Performance

First real cold-start A/B benchmark vs. `llama.cpp` (`972d231`, same A6000, same
`Qwen3-0.6B-Q4_K_M.gguf`, same prompt, full GPU offload, `n=3`):

| | coldstart-infer (initial) | round 1 | round 2 | round 3 | llama.cpp |
|---|---|---|---|---|---|
| wall clock | 28–30s | 10.4–11.7s | 6.4–8.5s | 6.38–6.46s | 6.44–6.67s |
| peak RSS | 3.68 GB | 1.33 GB | 1.35 GB | 1.35 GB | ~887 MB |
| user+sys CPU time | 5.13s + 14.02s | ~3.2s + ~4.0s | ~2.1s + ~3.5s | ~1.3s + ~2.2s | ~1.3s + ~1.4s |

Root cause of the initial 4.3x gap: `model.rs`'s weight-loading path dequantized every
tensor to a host `f32` buffer *and* re-uploaded that buffer to the GPU on every single
`gemv`/`gemv_expert`/`rmsnorm` call (every layer, every token). Fixed in Phase 2 round 1
by making `Weight` hold a `CudaSlice<f32>` uploaded once, closing the gap to ~1.7x.
Phase 2 round 2 then converted every kernel-wrapper op (`rmsnorm`/`gemv`/`rope`/
`silu_and_mul`/`attention`/the GDN mixer's kernels) to chain `CudaSlice<f32>` device
buffers through a whole layer instead of round-tripping each op's activations over
PCIe, and made the K/V cache device-resident (written via device-to-device copy)
instead of re-uploading its full history every token position, closing the gap to
~1.1x. Phase 2 round 3 added on-GPU dequant kernels for `Q4_K`/`Q6_K` (the block types
this project's `Q4_K_M` fixtures use for the bulk of weight bytes —
`kernels_cuda/dequant.cu`, one CUDA thread per 256-element super-block), removing the
CPU-bound host-side dequant-to-`f32` step from the load path for those types (every
other block type still uses the existing host path). **Current gap: ~1.0x — parity with
llama.cpp, within run-to-run noise** (see README.md's "Phase 2, round 3" section for
the full writeup and per-run numbers).

## MLA (MVP step 4)

DeepSeek-V2/V3 support covers dense-lead layers, routed-MoE + always-on shared-expert
FFN, and YaRN RoPE scaling — the actual shape of every real DeepSeek-V2/V3 checkpoint.
Only Q-LoRA query decomposition and MTP/NextN are still rejected with a clear error
(`parse_mla_config` in `model.rs`); no real file needing either has been seen. First
verified against a fully synthetic `deepseek2` GGUF
(`test-data/deepseek-tiny-mla.gguf`, dense-only, no real fixture exists publicly),
then extended the same session to the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint
(converted fresh from source with a current `convert_hf_to_gguf.py` — every
pre-quantized community GGUF found predates llama.cpp's MLA tensor-split format) on a
rented 80GB A100 (needed for the ~63GB of `f32` device-resident weights; doesn't fit
the A6000's 48GB). See README.md's MLA sections for the full writeup, including
several easy-to-miss correctness details (attention scale dimension, MLA's different
RoPE rotation convention, DeepSeek-V2-Lite's un-renormalized router weights, YaRN's
separate rotation-vs-attention-scale formulas) that each produced silently-wrong
(non-crashing) output before being caught by byte-exact comparison against real
llama.cpp builds.

## Phase 3 (State I/O), round 1

`--export-kv <file>`/`--import-kv <file>` added to `qwen3_coldstart`, dense/MoE Qwen3
only (`src/kv_io.rs`, new `Model::forward_prompt_capture_kv` in `model.rs`). Round 1 is
scoped to raw buffer export/import only — no resume-generation-from-cache, since there's
no per-token generation loop or `start_pos` anywhere in `model.rs` yet for a cache to
resume into (see README.md's "Phase 3, round 1" section and DECISIONS.md for the full
scope rationale). `--import-kv` proves the file round-trips byte-identical through a
device upload/download instead. Real-hardware-verified on the A6000 (`kgevfmca`
instance, still running from the Phase 2 round 3 session): `cargo test` (57 tests, incl.
2 new `kv_io` tests) plus real `--export-kv`/`--import-kv` runs against both
`Qwen3-0.6B-Q4_K_M.gguf` (dense) and `Tiny-Moe.Q4_K_M.gguf` (MoE).

## Open decision (not yet resolved)

None currently open. Phase 3 round 1 (above) is done as scoped. Remaining low-priority
follow-ups (not blocking, not actively planned): a real small `qwen3moe`-architecture
GGUF fixture (see "Known debt" below), on-device dequant coverage for the 15+ GGUF
block types still on the host path (Q4_0/1, Q5_0/1, Q8_0/1, Q2_K/Q3_K/Q5_K/Q8_K, the
IQ-family formats — none exercised by a local fixture's bulk weight bytes), MoE's
per-expert weighted-sum accumulation / the Gated Attention mixer's fused-qg gating
still round-tripping through the host (flagged, not measured as worth closing), and
Phase 3 round 2 (hybrid/MLA KV-cache export formats, plus the generation loop +
`start_pos` plumbing needed for `--import-kv` to actually resume generation — see
README.md's Phase 3 section).

## Known debt / limitations

- **Phase 2 round 3 on-device dequant scope**: only `Q4_K`/`Q6_K` dequantize on-GPU
  (`kernels_cuda/dequant.cu`). Every other GGUF block type (`Q4_0/1`, `Q5_0/1`,
  `Q8_0/1`, `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K`, all 8 IQ-family formats, plus F32/F16/Bf16/int
  passthrough) still dequantizes on the host, unchanged from before this round — correct,
  just not GPU-accelerated. This round's instance (`kgevfmca`, A6000) has no git history
  either (populated by `rsync`, not `git clone`) — local remains the only git-tracked
  copy; a fresh `ggml-org/llama.cpp` (`9655061`) was built there for the A/B benchmark.
- **Remote instance git state**: the MLA-extension work used a *second*, separate
  ThunderCompute instance for this session (`fl1uh6dt`, an 80GB A100, created
  2026-09-21 specifically for the real-DeepSeek-V2-Lite VRAM requirement — the Phase
  2 round 2 A/B benchmark earlier in the same session used a different A6000 instance,
  `2xhxwa87`; ThunderCompute instances are ephemeral and per-purpose, not assumed to
  persist or be reused across even the same session's different tasks). Its
  `~/coldstart-infer` working tree has no git history at all (populated by `rsync`
  from local, not `git clone`/`scp`); local remains the only git-tracked copy. A real
  `ggml-org/llama.cpp` checkout was built from source there (`~/llama.cpp`, CUDA
  enabled, `examples/simple`'s `llama-simple` built, `-DCMAKE_CUDA_ARCHITECTURES=80`
  for the A100) — worth reusing rather than rebuilding if the instance survives to the
  next session. `cmake`, GNU `time`, and the full `requirements-convert_hf_to_gguf.txt`
  Python stack (torch, transformers, etc. — needed to run `convert_hf_to_gguf.py`) were
  not preinstalled on this instance and had to be installed fresh.
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
- **Real DeepSeek-V2-Lite GGUF not preserved locally**: unlike every other fixture,
  the real `DeepSeek-V2-Lite.gguf` (16.7GB, `--outtype q8_0`) used to verify MLA's
  MoE/shared-expert/YaRN path was left on the A100 instance (`fl1uh6dt`) rather than
  copied to local `test-data/` — too large to be worth preserving the way the tiny
  synthetic fixtures are. Regenerating it needs: download
  `deepseek-ai/DeepSeek-V2-Lite`'s safetensors (~30GB,
  `huggingface_hub.snapshot_download`), then `python3 convert_hf_to_gguf.py
  <dir> --outfile DeepSeek-V2-Lite.gguf --outtype q8_0` with this project's pinned
  llama.cpp checkout (needs `requirements-convert_hf_to_gguf.txt` installed). Every
  pre-quantized community GGUF checked (mradermacher, tensorblock, duyntnet,
  bartowski) predates the MLA tensor-split conversion format and will be silently
  rejected by `parse_mla_config` (no `key_length_mla`/`value_length_mla` metadata) —
  don't assume a downloaded GGUF is usable without checking for those keys first.
