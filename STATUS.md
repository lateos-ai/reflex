# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `HISTORY.md` — this file is the
short, current-state summary; HISTORY.md is the log.

_Last updated: 2026-09-23 (benchmark expansion session: llama.cpp regression fix, vLLM, TypeSafe Jev citation, Ollama, multi-architecture fast_exit re-verification, dead-code cleanup)_

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

| | Reflex (initial) | round 1 | round 2 | round 3 | llama.cpp |
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
other block type still uses the existing host path). Phase 2 round 3 closed the gap to
~1.0x parity with llama.cpp, but a later re-measurement on a fresh instance (see
"Benchmarking expansion" below) found the wall-clock parity claim had regressed to
~1.4x *slower* — root-caused to CUDA-context-teardown cost, not the forward pass —
and fixed. **Current state: Reflex is ~1.3-1.4x *faster* than llama.cpp on
cold start** (see HISTORY.md's "Benchmark expansion" section for the full
investigation and per-run numbers).

## Benchmarking expansion (done: llama.cpp re-verified + fixed, vLLM, Jev citation)

Harness (`scripts/bench_cold_common.sh`, reuses the exact external-wall-clock
methodology from the original llama.cpp comparison) plus two new comparison scripts,
per DECISIONS.md's "Benchmark expansion", "TypeSafe Jev comparison framing", and
"Fast-exit after printing the benchmark result" entries. All three run; results and
caveats are in HISTORY.md's "Benchmark expansion" section, summarized here:

| Comparison | Result |
|---|---|
| llama.cpp (re-verified) | Found and fixed a real ~1.4x regression (CUDA-context-teardown cost, not the forward pass) via `reflex_engine::fast_exit`. **Now ~1.3-1.4x faster than llama.cpp**, not just parity |
| vLLM (`scripts/bench_cold_vllm.sh`) | **Reflex ~24-52x faster.** Weight-format deviation disclosed: installed vLLM 0.30.0 has no GGUF support at all, so vLLM ran against the HF safetensors checkpoint instead of the GGUF fixture |
| TypeSafe Jev latency citation, cold (`scripts/bench_cold_system1_vs_jev.sh`) | Reported honestly as a loss: Reflex's System1 cold start is ~10-60x *slower* than Jev's published figures — dominated by cold-loading the GGUF from disk, which Jev's always-resident managed service never pays. Illustrative citation only, not a benchmark claim |
| TypeSafe Jev latency citation, warm (`reflex bench --candidate`) | Fairer axis: Jev's 10-15ms figure is itself warm/compute-only. Reflex's warm System1 scoring is 19.4ms at the shortest prompt bucket (29 tokens) — within ~1.3-2x, competitive, not a loss. Published alongside the cold citation, not instead of it |
| Ollama (`scripts/bench_cold_ollama.sh`) | Wraps llama.cpp's ggml runtime, so no new AOT-vs-JIT data point — but found real, reproducible intermittent flakiness (2/7 and 3/5 runs across two scenarios hit an internal ~55-62s GPU-discovery-watchdog stall vs. ~6-11s otherwise), reported per-run rather than averaged away |
| TGI, TensorRT-LLM/Triton, other cloud/serverless vendors | Still deliberately deferred, not attempted — see DECISIONS.md for rationale |

`fast_exit` re-verified across MoE (`Tiny-Moe.Q4_K_M.gguf`), hybrid
(`Qwen3.5-0.8B-Q4_K_M.gguf`), and synthetic MLA (`deepseek-tiny-mla.gguf`) — all
match documented golden tokens, all exit cleanly (no teardown-delay regression), and
the batched-vs-sequential-prefill oracle tests pass against real fixtures for all
three architectures (MLA's real-DeepSeek-V2-Lite-only test correctly declines the
synthetic fixture rather than passing incorrectly). Also cleaned up the
`prefill_dense`/`prefill_hybrid`/`prefill_mla` dead-code warnings every build this
session showed — genuinely test-only now (`#[cfg(test)]` added), not a real bug;
`prefill_dense`'s doc comment was stale and got fixed to match the other two's.

## MLA (MVP step 4)

DeepSeek-V2/V3 support covers dense-lead layers, routed-MoE + always-on shared-expert
FFN, and YaRN RoPE scaling — the actual shape of every real DeepSeek-V2/V3 checkpoint.
Only Q-LoRA query decomposition and MTP/NextN are still rejected with a clear error
(`parse_mla_config` in `model.rs`); no real file needing either has been seen. First
verified against a fully synthetic `deepseek2` GGUF
(`test-data/deepseek-tiny-mla.gguf`, dense-only, no real fixture exists publicly),
then extended to the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint
(converted fresh from source with a current `convert_hf_to_gguf.py` — every
pre-quantized community GGUF found predates llama.cpp's MLA tensor-split format) on a
rented 80GB A100 (needed for the ~63GB of `f32` device-resident weights; doesn't fit
the A6000's 48GB). See HISTORY.md's MLA sections for the full writeup, including
several easy-to-miss correctness details (attention scale dimension, MLA's different
RoPE rotation convention, DeepSeek-V2-Lite's un-renormalized router weights, YaRN's
separate rotation-vs-attention-scale formulas) that each produced silently-wrong
(non-crashing) output before being caught by byte-exact comparison against real
llama.cpp builds.

## Phase 3 (State I/O), round 1

`--export-kv <file>`/`--import-kv <file>` added to `reflex generate`, dense/MoE Qwen3
only (`src/kv_io.rs`, new `Model::forward_prompt_capture_kv` in `model.rs`). Round 1 is
scoped to raw buffer export/import only — no resume-generation-from-cache, since there's
no per-token generation loop or `start_pos` anywhere in `model.rs` yet for a cache to
resume into (see HISTORY.md's "Phase 3, round 1" section and DECISIONS.md for the full
scope rationale). `--import-kv` proves the file round-trips byte-identical through a
device upload/download instead. Real-hardware-verified on the A6000 (the same
instance, still running from the Phase 2 round 3 round): `cargo test` (57 tests, incl.
2 new `kv_io` tests) plus real `--export-kv`/`--import-kv` runs against both
`Qwen3-0.6B-Q4_K_M.gguf` (dense) and `Tiny-Moe.Q4_K_M.gguf` (MoE).

## Phase 3 (State I/O), round 2

`--import-kv` now actually resumes generation, and `--max-tokens N` adds a real
per-token generation loop (feeding each generated id back in, stopping early on EOS) —
both pieces round 1 deliberately deferred together (see DECISIONS.md's round 1 entry).
Scope: **dense/MoE and the Qwen3.5 hybrid mixer**
(MLA stays round 3, matching this project's narrow-first precedent).

- `Model::generate` (`model.rs`) is the new top-level entry point; `forward_prompt`
  becomes a thin `max_new_tokens=1, imported=None` wrapper over the same
  `generate_dense_impl`/`generate_hybrid_impl` functions.
- `start_pos` plumbing: `k_cache`/`v_cache` buffers are now sized for
  `start_pos + prompt_len + max_new_tokens` (headroom for every token this call might
  still generate) instead of exactly the prompt length, and seeded from the imported
  cache via `htod_sync_copy_into` before the per-position loop starts (mid-buffer,
  offset by `start_pos`). The hybrid `GatedAttention` sublayers need the identical
  treatment; the `GatedDeltaNet` sublayers' `conv_state`/`recurrent` are fixed-size and
  round-trip as-is (no `start_pos` concept applies to them).
- `kv_io.rs` gained a version-2 hybrid file format (`HybridKvCache`,
  `export_hybrid_kv`/`import_hybrid_kv`) alongside the unchanged version-1 dense
  format, plus `import_kv`/`ImportedKv` to dispatch on whichever a file holds.
  `Model::architecture_kind()` lets the CLI pick the matching capture function
  (`forward_prompt_capture_kv` vs. `forward_prompt_capture_kv_hybrid`) without reaching
  into `Model`'s private fields.
- `--export-kv`/`--import-kv` can no longer be combined in one run (round-2 scope is
  resume, not chained re-export), and `--export-kv` requires `--max-tokens 1` (it only
  captures the cache after the initial prompt pass).
- **Real-hardware-verified on a fresh A6000 instance** (the round-1
  instance was gone by this round — confirms instances really
  are per-session ephemeral, not just per-purpose): `cargo test` (58 tests, incl. a new
  hybrid `kv_io` round-trip test) plus **byte-exact** export→import→continue vs. a
  single uninterrupted run, both for dense (`Qwen3-0.6B-Q4_K_M.gguf`, tokens
  `[19846,13,576,6722,315]` in both) and hybrid (`Qwen3.5-0.8B-Q4_K_M.gguf`, tokens
  `[19241,13,561,6511,314]` in both).
- **New finding, not a round-2 bug**: `Tiny-Moe.Q4_K_M.gguf` uses the SentencePiece
  encode path (`encode_sentencepiece` in `tokenizer.rs`), which unconditionally
  prepends an implicit leading-space token to *any* `encode()` call (real SentencePiece
  convention, confirmed against TinyLlama). That makes a text-level continuation prompt
  never byte-identical to the same text encoded as part of one continuous string — a
  property of the tokenizer, not of the resume path (`generate_dense_impl`/
  `forward_one_token_dense` are exactly the same code for dense and MoE; only
  `forward_layer_moe`'s FFN differs, and it has no position/cache logic at all).
  Verified instead via determinism (two resumes with identical inputs produce identical
  output) plus the code-sharing argument. Qwen3/Qwen3.5's real `gpt2`-style tokenizer
  doesn't have this property, which is why the dense/hybrid byte-exact checks above
  work as designed. See DECISIONS.md for the full writeup.

## Phase 3 (State I/O), round 3

`--import-kv` now resumes MLA models too, closing out Phase 3's architecture coverage
entirely (dense/MoE's K/V pair, hybrid's per-layer tagged state, and now MLA's single
compressed latent cache all support export/import/resume). Scope: extend round 2's
generation-loop + `start_pos` pattern to MLA's `kv_cache`, not new math — confirmed
with the user before starting (fixture and GPU-instance choice both explicitly asked).

- `generate_mla_impl` (`model.rs`, mirrors `generate_dense_impl`/`generate_hybrid_impl`)
  is the new entry point; `forward_prompt_mla` becomes a thin
  `max_new_tokens=1, imported=None` wrapper over it, matching `forward_prompt`/
  `forward_prompt_hybrid`. `forward_mla_attn_block` needed no change at all — it already
  took an absolute `position` and indexed `kv_cache` by it; only the caller needed to
  loop over positions with a preallocated, `start_pos`-offset buffer instead of running
  once per call.
- `kv_io.rs` gained a version-3 `MlaKvCache` format (one `[seq_len, kv_lora_rank +
  qk_rope_head_dim]` buffer per layer, no separate K/V pair) alongside the unchanged
  version-1/2 formats; `import_kv`/`ImportedKv`, `Model::generate`, and
  `forward_prompt_capture_kv_mla` (the `--export-kv` capture function, matching
  `forward_prompt_capture_kv`/`forward_prompt_capture_kv_hybrid`) all dispatch to it.
- **Real-hardware-verified on a fresh A6000 instance** (`tnr status --json`
  showed none running beforehand): `cargo test` (60 tests, incl. a new MLA
  `kv_io` round-trip test) plus **byte-exact** export→import→continue vs. a single
  uninterrupted run against the synthetic `test-data/deepseek-tiny-mla.gguf` fixture
  (chosen over real DeepSeek-V2-Lite — see DECISIONS.md's round 3 entry for why),
  tokens `[69344,10420,40306,145381,87488]` in both, prompt
  `"The quick brown fox jumps over the lazy dog"` + continuation `" and runs"`. Confirmed
  this fixture's `tokenizer.ggml.model` is `gpt2` (by reading the GGUF's own metadata
  bytes) before relying on the byte-exact-not-determinism-only verification bar round
  2 established for `gpt2`-tokenizer fixtures.

## Phase 4 (Embeddability), round 1

`--lora <adapter.gguf>` added to `reflex generate` (`src/lora.rs` new module,
`Model::apply_lora`/`Model::find_lora_target_mut` in `model.rs`): parses a
llama.cpp-format LoRA adapter GGUF and applies `W' = W + (alpha/rank) * (B @ A)` to
each targeted weight once, at load time, reusing the existing in-place-add kernel — no
new kernel, forward pass unchanged. Scope covers
dense/MoE Qwen3 attention+FFN and the Qwen3.5 hybrid's Gated-Attention-layer
tensors/Gated-DeltaNet-mixer FFN tensors; MLA and MoE's per-expert-stacked FFN/the
Gated DeltaNet mixer's non-Linear tensors are rejected with a clear error — see
DECISIONS.md's Phase 4 round 1 entry for the full scope rationale and format details.

- **Real-hardware-verified on the A6000** (reused from the Phase 3 round 3
  round — still running, per this project's practice of checking `tnr status --json`
  before creating a fresh instance): `cargo build --release` clean, `cargo test`
  unchanged at 59 passing.
- **Real fixture, not synthetic**: downloaded the real public
  `premjatin/qwen-linear-algebra-coder` PEFT LoRA adapter (rank 16, alpha 32) for
  `Qwen/Qwen3-1.7B`, converted both to GGUF with llama.cpp's own converters. Cross-checked
  three independent ways against a real llama.cpp build (`llama-export-lora`'s merge
  tensor count and `calculated_scale` log, plus `llama-simple`'s completion on the
  merged model) — all three matched this project's own `--lora` output exactly
  (`tensors_applied=196`, `" Paris"` for "The capital of France is"). See
  DECISIONS.md for the full verification writeup.
- **Accept/reject paths verified with hand-built synthetic adapters** (`gguf.GGUFWriter`,
  since no real adapter targeting a real MoE/hybrid checkpoint's exact modules was
  found) against the existing local `Tiny-Moe.Q4_K_M.gguf`/`Qwen3.5-0.8B-Q4_K_M.gguf`/
  `deepseek-tiny-mla.gguf` fixtures: MoE attention accepted, MoE FFN-experts rejected,
  hybrid Gated-Attention accepted, hybrid Gated-DeltaNet FFN accepted, hybrid
  Gated-DeltaNet `attn_qkv` rejected, MLA rejected outright, and a deliberately
  wrong-shaped adapter tensor rejected with a shape-mismatch error naming both shapes.
- **New finding, not a round-1 bug in the shipped code**: an initial reading of
  `convert_lora_to_gguf.py`'s Python source suggested the base tensor name is stripped
  of `.weight` before `.lora_a`/`.lora_b` is appended; the real converted file proved
  that wrong (the base name already includes `.weight`) — caught immediately by the
  first real end-to-end run (`blk.0.ffn_down.weight.weight`, a clear panic, not silent
  misbehavior) and fixed before any further verification. See DECISIONS.md.

## Phase 4 (Embeddability), round 2

`src/ffi.rs` (new module) adds a `extern "C"` surface (`reflex_load`/
`reflex_generate`/`reflex_free_generate_result`/`reflex_free`/
`reflex_last_error`) wrapping the exact same `Model::load`/`Model::generate`/
`Model::apply_lora` calls `reflex generate` itself uses — no new model-loading or
generation logic. `Cargo.toml`'s `[lib]` now emits `cdylib`/`staticlib` alongside
`rlib`; header generated via `cbindgen` into checked-in `include/reflex_engine.h`
(regenerated by hand, not wired into `build.rs`). Scope: load/generate/free only (LoRA
folds into `reflex_load` as an optional
parameter since it's load-time-only anyway; Phase 3's `--export-kv`/`--import-kv` state
I/O is *not* exposed through this FFI round), `cbindgen` over a hand-written header,
reused the still-running A6000 instance — see DECISIONS.md's Phase 4 round 2 entry.

- **Real-hardware-verified on the A6000**: `cargo build --release`/`cargo
  test --release` both clean (59 tests, unchanged), plus a real C test harness
  (`ffi-test/smoke_test.c`, plain `gcc` against the built `libreflex_engine.so`)
  exercising `load` → `generate` → `free`, cross-checked byte-exact against
  `reflex generate` on both the dense `Qwen3-0.6B-Q4_K_M.gguf` fixture
  (`token_ids=[13,576,3974,13876,38835]`) and the hybrid `Qwen3.5-0.8B-Q4_K_M.gguf`
  fixture (`token_ids=[0,353,1044]`), plus a clean (no-crash) error path for a
  nonexistent GGUF path.
- **`staticlib` follow-up (resolved)**: a follow-up debugging pass (see README's Phase 4
  round 2 section) found the originally-reported link-needs-`--allow-multiple-definition`
  / runtime-hang symptoms don't reproduce — a clean link with no extra flags produces a
  binary that runs correctly, byte-exact against `cdylib`/CLI on both fixtures. The
  original hang was very likely this project's already-documented ThunderCompute
  GPU-capacity-contention pattern (a queued GPU-driver call that clears on its own after
  some minutes), not a linking defect — the process was killed prematurely at ~90s.
  Both `cdylib` and `staticlib` are now verified working embedding paths.

This closes Phase 4 (Embeddability) entirely.

## Open decision (not yet resolved)

None currently open. Phase 4 (both rounds) is done. Remaining low-priority follow-ups
(not blocking, not actively planned): a real small `qwen3moe`-architecture GGUF
fixture with a `gpt2`-style tokenizer (see "Known debt" below — would also make MoE's
resume path byte-exact-testable at the text level, unlike `Tiny-Moe`), on-device
dequant coverage for the 15+ GGUF block types still on the host path (Q4_0/1, Q5_0/1,
Q8_0/1, Q2_K/Q3_K/Q5_K/Q8_K, the IQ-family formats — none exercised by a local
fixture's bulk weight bytes), MoE's per-expert weighted-sum accumulation / the Gated
Attention mixer's fused-qg gating still round-tripping through the host (flagged, not
measured as worth closing), Phase 3 round 3's own resume path verified only against
the dense-only synthetic MLA fixture (see DECISIONS.md's round 3 entry), and Phase 4
round 1's own LoRA MoE/hybrid accept/reject paths verified only against synthetic
hand-built adapters, not a real adapter trained against those architectures (none
found publicly — see DECISIONS.md's Phase 4 round 1 entry). No further Phase 4/post-MVP
work is currently planned — next steps, if any, are a separate confirm-before-starting
conversation.

## Known debt / limitations

- **Phase 2 round 3 on-device dequant scope**: only `Q4_K`/`Q6_K` dequantize on-GPU
  (`kernels_cuda/dequant.cu`). Every other GGUF block type (`Q4_0/1`, `Q5_0/1`,
  `Q8_0/1`, `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K`, all 8 IQ-family formats, plus F32/F16/Bf16/int
  passthrough) still dequantizes on the host, unchanged from before this round — correct,
  just not GPU-accelerated. This round's instance (an A6000) has no git history
  either (populated by `rsync`, not `git clone`) — local remains the only git-tracked
  copy; a fresh `ggml-org/llama.cpp` (`9655061`) was built there for the A/B benchmark.
- **Remote instance git state**: the MLA-extension work used a *second*, separate
  ThunderCompute instance (an 80GB A100, created
  2026-09-21 specifically for the real-DeepSeek-V2-Lite VRAM requirement — the Phase
  2 round 2 A/B benchmark earlier used a different A6000 instance;
  ThunderCompute instances are ephemeral and per-purpose, not assumed to
  persist or be reused across different tasks). Its
  `~/Reflex` working tree has no git history at all (populated by `rsync`
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
  doesn't exist yet locally. Its `llama`-architecture SentencePiece tokenizer also makes
  it unsuitable for text-level byte-exact resume verification (Phase 3 round 2's
  `--import-kv` continuation-prompt test) — see STATUS.md's Phase 3 round 2 section.
- **Cargo/binary staleness gotcha**: `cargo test --release` does not rebuild
  `target/release/<bin-name>` — only `target/release/deps/`. After any source change,
  run `cargo build --release --bin <name>` explicitly before trusting a binary run
  against real hardware.
- ~~**Phase 3 round 3 MLA resume, dense-only fixture**~~ — **closed**: originally only
  verified byte-exact against the synthetic dense-only `test-data/deepseek-tiny-mla.gguf`
  fixture. Now confirmed against real DeepSeek-V2-Lite's MoE+shared-expert+YaRN path too
  (`model::mla_batching_tests::prefill_mla_batched_import_kv_resume_matches_sequential`,
  see the batched-prefill re-verification entry below) — the resume/cache mechanism
  (`generate_mla_impl`/`forward_mla_attn_block`) does operate purely on the attention
  block's single compressed `kv_cache`, independent of the FFN tail, as originally
  reasoned; that reasoning is now backed by a real run, not just architectural inference.
- **Real DeepSeek-V2-Lite GGUF not preserved locally**: unlike every other fixture,
  the real `DeepSeek-V2-Lite.gguf` (16.7GB, `--outtype q8_0`) used to verify MLA's
  MoE/shared-expert/YaRN path was left on the A100 instance rather than
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
- **Dockerfile `docker build` now real-verified (both modes); `docker run --gpus all`
  still not, for a hardware reason this time, not an environment-access one**: a
  genuine (non-nested-container) Docker host was finally available (a
  Windows machine running Docker Desktop, WSL2 backend) — unlike every ThunderCompute
  A6000 instance used previously, which is itself a nested container
  (`systemd-detect-virt` reports `docker`) and rejects any Docker build outright
  (`unshare: operation not permitted`). `docker build --build-arg
  REFLEX_CUDA_ARCH=sm_86 -t Reflex .` and the default portable-PTX
  `docker build -t Reflex .` **both now pass cleanly** — but the portable-PTX
  mode only after a real bug this run found and fixed: `build.rs`'s
  `env::var("REFLEX_CUDA_ARCH").ok()` treated Docker's set-but-empty `ARG` (present
  even when no `--build-arg` is passed) as `Some("")` instead of `None`, silently
  taking the cubin branch with an empty `-arch=` and making `nvcc` fatal on every
  default docker build. Fixed with a one-line `.filter(|s| !s.is_empty())`; both modes
  re-verified clean post-fix. See DECISIONS.md's "Dockerfile real `docker build`/
  `docker run` verification" entry for the full root-cause writeup.
  `docker run` (no `--gpus`) against the built image with
  `test-data/deepseek-tiny-mla.gguf` bind-mounted confirmed the binary itself is
  correct inside the container — it opens and parses the GGUF, then progresses all the
  way to `cudarc`'s dynamic `libcuda`/`nvcuda` load before failing, exactly the
  expected boundary with no GPU present.
  **`docker run --rm --gpus all` remains unverified** — but now purely because this
  particular Docker host has no NVIDIA GPU at all (confirmed: `Get-CimInstance
  Win32_VideoController` → AMD Radeon only), not because of the nested-container
  access problem the previous entry described. `--gpus all` fails immediately with
  `nvidia-container-cli: initialization error: WSL environment detected but no
  adapters were found` — a hardware-absence error. (Incidental finding: Docker
  Desktop's WSL2 backend already has a working `nvidia-container-cli` wired in — the
  error is a specific "no adapter," not "toolkit missing" — so a Windows/Docker
  Desktop host with a real NVIDIA GPU would likely need no extra host-side toolkit
  setup for `--gpus all` to work.) **Still needed before this image is treated as
  release-ready**: one real `docker run --rm --gpus all` pass on a Docker host that has
  both genuine VM-level virtualization *and* an actual NVIDIA GPU/driver, producing
  real `REFLEX_GENERATE_OK process_start_to_first_token_ms=... token_text=...` output —
  neither the nested-container ThunderCompute instances nor the GPU-less
  Windows host used for the Docker build check above can provide that combination.
- **Kernel-byte-embedding refactor re-verified against the Qwen3.5 hybrid fixture,
  closing the one gap the MVP-release round's own hardware pass left open**: the
  initial verification re-ran the dense/MoE and MLA paths against
  `src/aot.rs`'s new `include_bytes!`-based kernel loading in both PTX and
  `REFLEX_CUDA_ARCH=sm_86` cubin modes, but not hybrid's `gated_deltanet.cu`
  module, since that fixture (`Qwen3.5-0.8B-Q4_K_M.gguf`) wasn't present on that
  instance. Downloaded fresh via this project's own `--model` hf-hub
  integration (`unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` — a real,
  publicly hosted GGUF, confirmed via the HF Hub search API, not a guess) on a new
  A6000 instance: `reflex generate --model ... "Once upon a time"`
  reproduced this project's own documented historical result exactly (token id `11`,
  `","`), and `model::hybrid_batching_tests::prefill_hybrid_batched_matches_sequential`
  passed in both PTX and cubin modes. All three architecture families are now
  confirmed against the kernel-embedding refactor in both build modes.
- **MLA batched-prefill (`prefill_mla_batched`) re-verified against the real
  DeepSeek-V2-Lite checkpoint, closing the gap its own test doc comments flagged**:
  the batched-prefill MLA work's own tests (`model::mla_batching_tests`) originally
  ran only against the synthetic, dense-lead-only, no-YaRN
  `test-data/deepseek-tiny-mla.gguf` fixture — explicitly noted in those tests' doc
  comments as not exercising `MlaFfn::Moe`'s per-row loop or
  `rope_norm_yarn_batch_kernel`. Re-verified on a fresh 80GB A100 instance:
  `deepseek-ai/DeepSeek-V2-Lite` downloaded (~30GB safetensors,
  `huggingface_hub.snapshot_download`) and converted fresh (`convert_hf_to_gguf.py
  --outtype q8_0`, current `ggml-org/llama.cpp`, 16.7GB output) — same recipe as the
  original MLA verification. `reflex generate` against the real checkpoint produced
  `"The capital of France is" -> " Paris"`, independently reproduced byte-exact by a
  fresh CUDA-enabled `llama-simple` build (`-DCMAKE_CUDA_ARCHITECTURES=80`) from the
  same checkpoint. All three `mla_batching_tests` (including the batched-vs-sequential
  diff and the `--import-kv` resume test) passed against the real file (573.73s
  total, mostly repeated `Model::load` cost — `Q8_0` isn't on the on-GPU dequant
  kernel path, unlike `Q4_K`/`Q6_K`). This is the first time the batched-prefill MLA
  path has been exercised against real MoE routing, the always-on shared expert, and
  YaRN RoPE scaling together, not just architecturally reasoned to be independent of
  them.
