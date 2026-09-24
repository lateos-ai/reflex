# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `HISTORY.md` — this file is the
short, current-state summary; HISTORY.md is the log.

_Last updated: 2026-09-23 (CI added: build/test on GitHub-hosted runners via REFLEX_SKIP_CUDA; previous update: benchmark expansion session — llama.cpp regression fix, vLLM, TypeSafe Jev citation, Ollama, multi-architecture fast_exit re-verification, dead-code cleanup)_

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

## Planned next work (post-public-release-prep, 2026-09-23)

Phase 4 (both rounds) is done and no further Phase 4/post-MVP architecture work is
planned. The user has asked to queue up three release-hardening items, in this order:

1. ~~**Add CI** (build/test workflow)~~ — **done**: `.github/workflows/ci.yml`,
   `cargo build --locked --all-targets` + `cargo test --locked` under
   `REFLEX_SKIP_CUDA=1` on an `[ubuntu-latest, windows-latest]` matrix (no
   GitHub-hosted runner has a GPU). `cargo fmt --check`/`cargo clippy` deliberately
   left out — see HISTORY.md's CI entry and "Known debt" below for why. This does
   not replace real-hardware verification, only catches non-GPU-dependent breakage.
2. **Verify `docker run --rm --gpus all`** end-to-end — the one remaining unverified
   Docker path (see "Known debt" below), needs a host with genuine VM-level
   virtualization *and* a real NVIDIA GPU/driver.
3. ~~**On-device dequant for more GGUF block types**~~ — **`Q5_K` done**, and as of
   2026-09-24, **Q4_0/1, Q5_0/1, Q8_0/1, Q2_K, Q3_K, Q8_K also done** (see "Known
   debt" below for the real-hardware verification writeup, which surfaced and
   root-caused a real Q2_K/Q3_K token-level discrepancy — verified as inherent
   quantization noise, not a kernel bug). Only the 8 IQ-family formats (IQ2_XXS/XS/S,
   IQ3_XXS/S, IQ1_S/M, IQ4_XS) remain on the host path — they need constant-memory
   lookup tables ported from `ggml-common.h`, not just this file's per-block-loop
   pattern; left for a future round.

Other remaining low-priority follow-ups (not queued, not blocking): Phase 3 round 3's
own resume path verified only against the dense-only synthetic MLA fixture (see
DECISIONS.md's round 3 entry).

**Real-adapter LoRA verification for the hybrid architecture — closed 2026-09-24**:
`Tilakoid/qwen3.5-0.8b-hoasa-lora` (targets `Qwen/Qwen3.5-0.8B` exactly) was
downloaded and inspected for real, not just via its `adapter_config.json`'s regex
`target_modules` (too imprecise to trust — see DECISIONS.md's new entry) but via its
actual safetensors header. That corrected an assumption this paragraph used to make:
the adapter targets not just `self_attn`/`mlp` (the previously-accepted subset) but
also the Gated DeltaNet mixer's `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,
in_proj_b,out_proj}` — which `find_lora_target_mut` rejected before this round. Those
five turned out to be plain 2-D `Weight`s already driven by `gemv` (mapping onto this
project's own `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`/`ssm_out`), not the
genuinely non-Linear `ssm_dt`/`ssm_a`/`ssm_conv1d`/`ssm_norm` the mixer also has (which
the adapter never targets, confirmed by its tensor list) — so widening the accept list
needed no new math or kernel, just five more match arms in `find_lora_target_mut`
(`src/model.rs`). Verified end-to-end on a real L40: `convert_lora_to_gguf.py`
produced exactly the predicted GGUF base tensor names
(`blk.N.{ssm_alpha,ssm_beta,attn_qkv,attn_gate,ssm_out}.weight`); `reflex generate
--lora` against `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` applied
`tensors_applied=186` (18 `GatedDeltaNet` layers × 8 tensors + 6 `GatedAttention`
layers × 7, exactly the expected count, confirming zero silent rejections); cross-
checked against a real llama.cpp build (`llama-export-lora` logged `merged 186
tensors with lora adapters` and `calculated_scale=2.000000`, matching both the tensor
count and `alpha/rank=32/16` exactly) and `llama-simple` on the merged GGUF produced
token-for-token identical continuation text to `reflex generate --lora`'s own output
for `"The capital of France is"` (`" the city of Paris.\nThe capital of Germany is
the"`), which also visibly diverges from the un-adapted base's `" the capital of the
country."` continuation — proof the adapter is doing something, not a no-op accept.
**Note**: freshly converting `Qwen/Qwen3.5-0.8B` from HF directly (rather than using
the pre-quantized `unsloth` GGUF) hit this project's own `nextn_predict_layers`
MTP/NextN rejection — the real upstream checkpoint now ships an MTP draft block that
didn't exist when `load_hybrid`'s doc comment was written; worked around by using the
`unsloth` GGUF as the LoRA base instead (MTP rejection is orthogonal to LoRA and still
correctly enforced, not disabled). `davidanugraha/Qwen3.5-35B-A3B-SWE-Smith-LoRA-
Adapters`/`-9B-` remain untested: confirmed (via HF page text, not yet its own
safetensors header) to target MoE's per-expert routed-expert projections, which really
does need new per-expert-slice delta math beyond a widened accept list, plus a much
larger model — left for a future round.

## Known debt / limitations

- ~~**`src/ffi.rs`'s `extern "C"` functions dereference raw pointers without being
  `unsafe fn`**~~ — **closed**: found by `cargo clippy --all-targets` while adding CI
  (see HISTORY.md's CI entry) — 12 `clippy::not_unsafe_ptr_arg_deref` errors (a
  deny-by-default correctness lint) across `reflex_load`/`reflex_generate`/
  `reflex_free`/etc. Fixed by adding `unsafe` to the 5 affected function signatures
  (`reflex_last_error` takes no pointer args, so it was never flagged) plus a `#
  Safety` doc section on each (clippy's `missing_safety_doc`, newly surfaced once the
  functions became `unsafe fn`). Confirmed the change is source-level only, not the
  wider public-API break originally feared: `cbindgen`'s regenerated
  `include/reflex_engine.h` is identical except for the new doc comments (`unsafe` is
  a Rust-only annotation with no C-side representation), `ffi-test/smoke_test.c`'s
  plain C calls need no changes, and no in-crate Rust code calls these functions
  directly. `cargo test` still 74 passed/0 failed under `REFLEX_SKIP_CUDA=1`. `cargo
  clippy` is still not wired into CI (the pre-existing `cargo fmt` non-compliance
  found the same CI session remains open), but this specific blocker is resolved.
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
- ~~**MoE test fixture**~~ — **closed**: `test-data/Tiny-Moe.Q4_K_M.gguf` is still a
  Mixtral-style synthetic model (`general.architecture="llama"`, `expert_count=2`,
  `expert_used_count=2` — top-k always selects every expert, so it can't prove routing
  actually excludes an expert; no QK-Norm tensors; `llama`-arch SentencePiece tokenizer
  blocks text-level byte-exact resume verification). Added
  `test-data/tiny-qwen3moe.gguf`, a real `qwen3moe`-architecture fixture built the same
  way as `deepseek-tiny-mla.gguf` (hand-built HF `config.json`/`safetensors`, random
  weights, run through llama.cpp's own real, unmodified `convert_hf_to_gguf.py` --
  source archived as `test-data/tiny-qwen3moe-src.tar.gz`): `expert_count=8`,
  `expert_used_count=2` (routing can now be shown to exclude experts), real Qwen3
  `attn_q_norm`/`attn_k_norm` tensors, and a `gpt2`-style tokenizer (reused verbatim
  from `deepseek-tiny-mla`'s, enabling the same byte-exact resume verification bar
  Phase 3 round 2 established). Verified locally against this project's own
  `GgufFile`/`parse_model_config` (host-only, no GPU:
  `model::moe_fixture_tests::qwen3moe_fixture_has_excluding_topk_and_qk_norm`), and
  since real-hardware-verified on a fresh L40 (sm_89) instance:
  `qwen3moe_fixture_generates_without_error` (a real `Model::generate` pass) and
  `prefill_dense_batched_matches_sequential_prefill` (batched-vs-sequential MoE
  routing comparison, byte-exact) both passed against it.
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
- ~~**On-device dequant extended to Q5_K, not yet real-hardware-verified**~~ —
  **closed**: added `dequantize_q5k_kernel` (`kernels_cuda/dequant.cu`, line-for-line
  port of `dequant.rs`'s `dequantize_block_q5_k`, reusing the same `get_scale_min_k4`
  device helper Q4_K's kernel already established) and wired it into
  `dequantize_tensor_to_device`'s dispatch at all three load sites (dense/MoE, hybrid,
  MLA). Real-hardware-verified on a fresh L40 (sm_89) instance: downloaded a real,
  publicly hosted `Q5_K_M` quant (`unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q5_K_M.gguf`),
  `reflex generate "Once upon a time"` produced `token_id=11, ","`, independently
  reproduced byte-exact (same continuation text) by a fresh CUDA-enabled
  `llama-simple` build (`-DCMAKE_CUDA_ARCHITECTURES=89`) against the same file.
- ~~**On-device dequant extended to Q4_0/1, Q5_0/1, Q8_0/1, Q2_K, Q3_K, Q8_K, not yet
  real-hardware-verified**~~ — **closed 2026-09-24**: 9 more kernels added to
  `kernels_cuda/dequant.cu`, each a line-for-line port of its `dequant.rs` host
  counterpart (all already existed and were unit-tested against hand-computed
  values, just never GPU-accelerated or exercised end-to-end against a real GGUF).
  `dequantize_tensor_to_device`'s per-kernel `&AotKernel` parameters were bundled
  into one `DequantKernels` struct (`load_dequant_kernels`, shared by all three
  load sites) rather than growing the parameter list by one arg per format.
  Real-hardware-verified on a fresh A100 (`g3jx9w64`, sm_80): quantized a real
  `Qwen/Qwen3-0.6B` checkpoint with llama.cpp's own `llama-quantize --pure` into
  all 7 real-storable target types (Q8_1/Q8_K are runtime-only quantized-dot-
  product intermediates, never a stored GGUF tensor type in practice — confirmed
  by `dequant.rs`'s own pre-existing doc comments on both, so verified by code
  review against the now-confirmed-correct Q8_0 kernel instead, same value
  semantics). `reflex generate` matched `llama-simple` byte-exact on 5/7
  (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`, all producing `token_id=12095, " Paris"`
  identically) but **diverged on `Q2_K`/`Q3_K`** (`reflex`: `" r"`/`" located"`;
  `llama-simple`: `"?"`/`" Paris"`) — investigated rather than dismissed. Root
  cause: **not a dequant bug**. Extracted the real `blk.0.attn_q.weight` tensor's
  raw block bytes from both quantized GGUFs and dequantized them three ways —
  this project's Rust (`dequant::dequantize`), llama.cpp's own Python reference
  (`gguf-py`'s `Q2_K`/`Q3_K.dequantize_blocks`), and (for the CUDA kernel
  specifically) a temporary host-path fallback to confirm the on-device kernel
  produces bit-identical output to the already-verified host function — all three
  agreed to f32 precision. The token-level divergence is inherent numerical
  instability from 2-3-bit quantization on a 0.6B model, not an implementation
  defect: proven by running `llama-simple` itself on CPU (`-ngl 0`) vs GPU
  (`-ngl 99`) for the same `Q2_K` file, which gave a *third* different answer
  (`"ising"`) — llama.cpp disagrees with its own two backends, meaning the
  top-token race is a photo finish sensitive to any implementation's floating-
  point summation order, not something a "correct" implementation is expected to
  win consistently at this quantization level. `Q3_K`'s CPU/GPU llama.cpp
  backends happened to agree with each other in this one instance, so that
  specific case is not as ironclad as `Q2_K`'s, but the same per-block dequant
  correctness evidence applies equally to both. Caught a real rsync+cargo gotcha
  along the way: `rsync -a` preserves the local mtime, which can be *older* than
  a previously-built target on the remote, causing `cargo build` to silently skip
  recompilation and report success against stale object code — `touch`ing changed
  source files after an rsync (or after any manual edit-then-revert cycle on the
  remote) before rebuilding avoids trusting a false-positive "Finished" line.
- ~~**MoE weighted-sum and Gated-Attention fused-qg gating moved on-device in the
  decode path, not yet real-hardware-verified**~~ — **closed**: `forward_layer_moe`/
  `forward_mla_moe_ffn`'s per-expert weighted accumulate now uses
  `Self::moe_scatter_add` (the same kernel `forward_layer_moe_batched`'s grouped-GEMM
  path already used, called once per selected expert with a single-row group) instead
  of a `dtoh_sync_copy`/host-sum/`htod_sync_copy` round trip per expert; the always-on
  MLA shared expert now uses `Self::add_inplace` directly. `forward_gated_attn_mixer`
  now calls `Self::split_qg_k`/`Self::sigmoid_gate_k` (the exact kernels
  `forward_gated_attn_mixer_batched` uses, both already generic over row count) with
  `rows=1` instead of two host round trips. Real-hardware-verified on the same L40
  instance plus a fresh 80GB A100: `prefill_dense_batched_matches_sequential_prefill`
  passed against `test-data/tiny-qwen3moe.gguf` (exercises `forward_layer_moe`'s
  changed weighted-sum against real MoE routing, byte-exact vs. the unmodified batched
  path); `prefill_hybrid_batched_matches_sequential` passed against a freshly
  downloaded real `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` (exercises
  `forward_gated_attn_mixer`'s changed gating, byte-exact vs. the unmodified batched
  path), and a plain `reflex generate` against the same file reproduced the
  previously-documented golden token (`token_id=11, ","`) exactly; on a fresh 80GB
  A100, `prefill_mla_batched_matches_sequential_real_moe_checkpoint` passed against a
  freshly re-downloaded/re-converted real `deepseek-ai/DeepSeek-V2-Lite` (exercises
  `forward_mla_moe_ffn`'s changed weighted-sum + shared-expert add against real MoE
  routing, the always-on shared expert, and YaRN RoPE scaling together), and `reflex
  generate` against it reproduced the documented `"The capital of France is" -> "
  Paris"` golden output exactly. All four architecture families this change touches
  are now real-hardware-verified, not just locally compiled.
