# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `README.md` — this file is the
short, current-state summary; README is the log.

_Last updated: 2026-09-22 (Phase 4 round 2 session)_

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

## Phase 3 (State I/O), round 2

`--import-kv` now actually resumes generation, and `--max-tokens N` adds a real
per-token generation loop (feeding each generated id back in, stopping early on EOS) —
both pieces round 1 deliberately deferred together (see DECISIONS.md's round 1 entry).
Scope: **dense/MoE and the Qwen3.5 hybrid mixer**, confirmed with the user before
starting (MLA stays round 3, matching this project's narrow-first precedent).

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
- **Real-hardware-verified on a fresh A6000 instance** (`lunpulve`; the round-1
  session's `kgevfmca` instance was gone by this session — confirms instances really
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
- **Real-hardware-verified on a fresh A6000 instance** (`bkzn3giz`; `tnr status --json`
  showed none running at session start): `cargo test` (60 tests, incl. a new MLA
  `kv_io` round-trip test) plus **byte-exact** export→import→continue vs. a single
  uninterrupted run against the synthetic `test-data/deepseek-tiny-mla.gguf` fixture
  (chosen over real DeepSeek-V2-Lite — see DECISIONS.md's round 3 entry for why),
  tokens `[69344,10420,40306,145381,87488]` in both, prompt
  `"The quick brown fox jumps over the lazy dog"` + continuation `" and runs"`. Confirmed
  this fixture's `tokenizer.ggml.model` is `gpt2` (by reading the GGUF's own metadata
  bytes) before relying on the byte-exact-not-determinism-only verification bar round
  2 established for `gpt2`-tokenizer fixtures.

## Phase 4 (Embeddability), round 1

`--lora <adapter.gguf>` added to `qwen3_coldstart` (`src/lora.rs` new module,
`Model::apply_lora`/`Model::find_lora_target_mut` in `model.rs`): parses a
llama.cpp-format LoRA adapter GGUF and applies `W' = W + (alpha/rank) * (B @ A)` to
each targeted weight once, at load time, reusing the existing in-place-add kernel — no
new kernel, forward pass unchanged. Scope confirmed with the user before starting
(dense/MoE Qwen3 attention+FFN and the Qwen3.5 hybrid's Gated-Attention-layer
tensors/Gated-DeltaNet-mixer FFN tensors; MLA and MoE's per-expert-stacked FFN/the
Gated DeltaNet mixer's non-Linear tensors rejected with a clear error — see
DECISIONS.md's Phase 4 round 1 entry for the full scope rationale and format details).

- **Real-hardware-verified on the A6000** (`bkzn3giz`, reused from the Phase 3 round 3
  session — still running, per this project's practice of checking `tnr status --json`
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

`src/ffi.rs` (new module) adds a `extern "C"` surface (`coldstart_load`/
`coldstart_generate`/`coldstart_free_generate_result`/`coldstart_free`/
`coldstart_last_error`) wrapping the exact same `Model::load`/`Model::generate`/
`Model::apply_lora` calls `qwen3_coldstart` itself uses — no new model-loading or
generation logic. `Cargo.toml`'s `[lib]` now emits `cdylib`/`staticlib` alongside
`rlib`; header generated via `cbindgen` into checked-in `include/coldstart_infer.h`
(regenerated by hand, not wired into `build.rs`). Scope confirmed with the user before
starting: load/generate/free only (LoRA folds into `coldstart_load` as an optional
parameter since it's load-time-only anyway; Phase 3's `--export-kv`/`--import-kv` state
I/O is *not* exposed through this FFI round), `cbindgen` over a hand-written header,
reused the still-running `bkzn3giz` A6000 — see DECISIONS.md's Phase 4 round 2 entry.

- **Real-hardware-verified on the A6000** (`bkzn3giz`): `cargo build --release`/`cargo
  test --release` both clean (59 tests, unchanged), plus a real C test harness
  (`ffi-test/smoke_test.c`, plain `gcc` against the built `libcoldstart_infer.so`)
  exercising `load` → `generate` → `free`, cross-checked byte-exact against
  `qwen3_coldstart` on both the dense `Qwen3-0.6B-Q4_K_M.gguf` fixture
  (`token_ids=[13,576,3974,13876,38835]`) and the hybrid `Qwen3.5-0.8B-Q4_K_M.gguf`
  fixture (`token_ids=[0,353,1044]`), plus a clean (no-crash) error path for a
  nonexistent GGUF path.
- **Known limitation**: `staticlib` output builds and links (needing
  `-Wl,--allow-multiple-definition`, itself a bad sign) but the resulting binary hangs
  at runtime — not root-caused this round. `cdylib` is the verified, recommended
  embedding path; see "Known debt" below and README's Phase 4 round 2 section.

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
the dense-only synthetic MLA fixture (see DECISIONS.md's round 3 entry), Phase 4 round
1's own LoRA MoE/hybrid accept/reject paths verified only against synthetic hand-built
adapters, not a real adapter trained against those architectures (none found publicly
— see DECISIONS.md's Phase 4 round 1 entry), and Phase 4 round 2's `staticlib` hang
(above, not root-caused). No further Phase 4/post-MVP work is currently planned —
next steps, if any, are a separate confirm-before-starting conversation.

## Known debt / limitations

- **Phase 4 round 2 `staticlib` linking hangs at runtime**: `cargo build --release`
  produces `libcoldstart_infer.a` without error, and a C binary links against it, but
  needed `-Wl,--allow-multiple-definition` to resolve duplicate symbols (a sign of an
  unresolved rough edge, not a clean link) and the resulting binary hung rather than
  running or crashing when actually executed (`ffi-test/smoke_test.c`, real A6000
  hardware). Not root-caused — plausibly a `libc`/threading-runtime duplication between
  the static archive and system libs `cudarc`'s driver-loading path also pulls in.
  `cdylib` (fully verified byte-exact against `qwen3_coldstart`, see the Phase 4 round 2
  section above) is the recommended embedding path; a real host needs `libcuda.so`
  dynamically resolvable at runtime regardless of how `coldstart-infer` itself links, so
  `cdylib` has no real downside here. Revisit only if a host specifically needs static
  linking.
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
  doesn't exist yet locally. Its `llama`-architecture SentencePiece tokenizer also makes
  it unsuitable for text-level byte-exact resume verification (Phase 3 round 2's
  `--import-kv` continuation-prompt test) — see STATUS.md's Phase 3 round 2 section.
- **Cargo/binary staleness gotcha**: `cargo test --release` does not rebuild
  `target/release/<bin-name>` — only `target/release/deps/`. After any source change,
  run `cargo build --release --bin <name>` explicitly before trusting a binary run
  against real hardware.
- **Phase 3 round 3 MLA resume, dense-only fixture**: `--import-kv` resume was verified
  byte-exact only against the synthetic dense-only `test-data/deepseek-tiny-mla.gguf`
  fixture. The resume/cache mechanism (`generate_mla_impl`/`forward_mla_attn_block`)
  operates purely on the attention block's single compressed `kv_cache`, independent of
  whether a layer's FFN tail is `MlaFfn::Dense` or `MlaFfn::Moe` — the same reasoning
  round 2 used to scope hybrid's `GatedDeltaNet` state — but this hasn't been confirmed
  end to end against real DeepSeek-V2-Lite's MoE+shared-expert+YaRN path. Not planned
  as follow-up unless a specific need comes up (see DECISIONS.md's round 3 entry).
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
