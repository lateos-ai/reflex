# STATUS.md

Current state of the project. For narrative write-ups (how each milestone was verified,
full benchmark tables, bugs found along the way), see `HISTORY.md` — this file is the
short, current-state summary; HISTORY.md is the log.

_Last updated: 2026-09-25 (items 2/3/4 of the ranked warm-latency-vs-TypeSafe-Jev plan are now real-hardware-verified on a fresh AWS EC2 `g4dn.xlarge` — see "Planned next work: warm-latency perf vs. TypeSafe Jev" below and HISTORY.md's "Real-hardware verification of the warm-latency perf plan" entry for the full numbers: item 2's warp-per-row `gemv_kernel`/`gemv_gather_kernel` rewrite measured a **~4.9x decode-throughput improvement** (15.4 → 74.8 tok/s), item 3's lazy `LmHead` dropped GPU-resident bytes by exactly the predicted ~608 MiB and improved cold-start too, item 4's phase breakdown now reports real, tight p50/p95 numbers. Golden tokens reproduced exactly for dense and hybrid; all 8 relevant internal-consistency oracle tests pass across dense/MoE/hybrid/synthetic-MLA. Only item 1 (f16 weight residency) remains unstarted, deliberately, pending a numerics-methodology decision. Previous update: TypeSafe Jev citation — both cold-start and warm-latency axes — re-verified on a real AWS EC2 `g4dn.xlarge` (Tesla T4), cross-validating the original ThunderCompute A6000 numbers on independent rented-GPU hardware; see HISTORY.md's "TypeSafe Jev re-verification on real AWS EC2 T4" entry)_

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
| TypeSafe Jev latency citation, cold (`scripts/bench_cold_system1_vs_jev.sh`) | Reported honestly as a loss: Reflex's System1 cold start is ~10-60x *slower* than Jev's published figures on ThunderCompute A6000 — dominated by cold-loading the GGUF from disk, which Jev's always-resident managed service never pays. Illustrative citation only, not a benchmark claim. **Re-verified on a real AWS EC2 `g4dn.xlarge` (Tesla T4, 2026-09-25)**: gap narrows to ~2.5-18x on real dedicated GPU hardware — same conclusion, host-dependent magnitude (see HISTORY.md) |
| TypeSafe Jev latency citation, warm (`reflex bench --candidate`) | Fairer axis: Jev's 10-15ms figure is itself warm/compute-only. Reflex's warm System1 scoring is 19.4ms at the shortest prompt bucket (29 tokens) on ThunderCompute A6000 — within ~1.3-2x, competitive, not a loss. Published alongside the cold citation, not instead of it. **Re-verified on AWS EC2 T4 (2026-09-25)**: 20.9ms at the same bucket, ~1.4-2.1x — same conclusion on a second GPU vendor/host |
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
   left out at the time — see HISTORY.md's CI entry and "Known debt" below for why.
   **Both wired in 2026-09-24** — see HISTORY.md's follow-up CI entry. This does
   not replace real-hardware verification, only catches non-GPU-dependent breakage.
2. ~~**Verify `docker run --rm --gpus all`** end-to-end~~ — **done 2026-09-24**: a real
   EC2 `g4dn.xlarge` (On-Demand; Spot capacity was exhausted in every `us-east-1` AZ at
   launch time) gave the genuine VM-level virtualization *and* real NVIDIA GPU/driver
   this needed — see "Known debt" below for the full writeup.
3. ~~**On-device dequant for more GGUF block types**~~ — **done**: `Q5_K` done first,
   then as of 2026-09-24, **Q4_0/1, Q5_0/1, Q8_0/1, Q2_K, Q3_K, Q8_K also done** (see
   "Known debt" below for the real-hardware verification writeup, which surfaced and
   root-caused a real Q2_K/Q3_K token-level discrepancy — verified as inherent
   quantization noise, not a kernel bug), and finally, same day, **the 8 IQ-family
   formats (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M, IQ4_XS) closed the set entirely** — see
   "Known debt" below for the codebook-table-generation approach and real-hardware
   verification writeup. Every *block-quantized* GGUF format this project's
   `gguf.rs` parses now dequantizes on-GPU; only the passthrough types
   (F32/F16/Bf16, a type conversion rather than a dequantization) still take the
   host `dequant::dequantize` -> `htod_sync_copy` path.

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

## Planned next work: warm-latency perf vs. TypeSafe Jev (2026-09-25)

Following the AWS EC2 T4 re-verification of the Jev citation (see HISTORY.md's
"TypeSafe Jev re-verification on real AWS EC2 T4" entry), analyzed where
System1's warm 20.9ms (shortest bucket) actually goes, using only numbers
already on record (no profiler needed):

- System1 @29 tok = 20.9ms vs. `forward_prompt` @29 tok = 27.5ms — the 6.6ms
  delta is exactly the extra full-vocab GEMV over the 594 MiB `lm_head`, so
  System1's 20.9ms is **entirely the 28-layer batched prefill**.
- ~1.84GB of layer weights read once per pass in 20.9ms ≈ **88 GB/s** — a T4
  peaks at 320 GB/s, so this is **~28% of peak, memory-bandwidth-bound**, not
  launch-overhead-bound (cudarc 0.11.9's allocator is stream-ordered async,
  ~500 kernel launches/pass is only ~2-3ms — ruled out as the bottleneck).
- The plain decode-path `gemv_kernel` (`kernels_cuda/gemv.cu`) is scalar,
  one thread per output row — decode's 65ms/token ≈ 28 GB/s ≈ **9% of peak**,
  the worst number in the whole benchmark.
- `Model::load`'s tied-embedding branch uploaded the *entire* ~594 MiB
  `token_embd`-as-`lm_head` matrix even for a `reflex system1` run that only
  ever gathers a handful of candidate rows — pure waste for that path.

Ranked plan (payoff vs. risk), **items 3+4 started 2026-09-25**:

1. **f16 weight residency + `cublasGemmEx`/tensor-core GEMM** — est. 20.9ms →
   ~10-12ms, ~halves VRAM and cold-load bytes too. Largest win, but changes
   numerics: every golden token in this file and `reflex check`'s whole
   byte-exact-vs-llama.cpp methodology assumes f32 greedy argmax. **Needs a
   DECISIONS.md entry and a deliberate call before starting** — not free,
   not started.
2. **Rewrite `gemv_kernel`**: warp-per-row + vectorized loads + shuffle
   reduction. Est. decode 65ms/tok → ~15-20ms/tok. **Done, 2026-09-25**: both
   `kernels_cuda/gemv.cu` and `kernels_cuda/gemv_gather.cu` (same anti-pattern,
   same fix) rewritten from one-thread-per-output-row (strided, uncoalesced
   access across a warp's lanes — the actual root cause of the ~9%-of-peak
   decode bandwidth) to one-warp-per-output-row (all 32 lanes read the same
   row, 32 consecutive elements apart — fully coalesced), plus `float4`
   vectorized loads when `in_features % 4 == 0` (true for every real hidden/
   FFN size this project uses; a scalar fallback keeps other sizes correct),
   plus a `__shfl_down_sync` warp-reduction tree instead of one thread
   summing serially. Every launch site of the shared `gemv_kernel` handle
   (`Model::gemv_raw`, `Model::gemv_view` — **initially missed, then found by
   grepping every call site of `self.gemv_k`/`self.gemv_gather_k` across the
   whole crate**, since a stale thread-per-row grid/block geometry against
   the new warp-per-row kernel would have silently computed wrong results
   for MLA's per-head GEMV path) and `gemv_gather_kernel` (`Model::
   gemv_gather`) updated to the matching warp-per-block launch geometry.
   Compiles clean, all 85 host-only tests pass, `cargo fmt`/`clippy` clean
   under `REFLEX_SKIP_CUDA=1`. **Real-hardware-verified, 2026-09-25** (see
   HISTORY.md's "Real-hardware verification of the warm-latency perf plan"
   entry for the full writeup): `nvcc` compiled both kernels cleanly first
   try on a real AWS EC2 T4; dense `reflex generate` reproduces the
   documented golden token (`12095`/`" Paris"`) exactly, hybrid reproduces
   the documented base continuation's first token (`279`/`" the"`), and all
   8 relevant batched-vs-sequential/gemv-gather oracle tests pass across
   dense/MoE/hybrid/synthetic-MLA fixtures (only the real-DeepSeek-V2-Lite
   MoE-MLA test was skipped, deliberately — needs an 80GB A100). **Measured
   decode throughput improved ~4.9x** (15.4 → 74.8 tok/s, 65.0 → 13.4
   ms/token @29-token bucket) — the single biggest number in this project's
   perf history outside the original llama.cpp-parity saga. System1's warm
   latency is correctly unaffected (its bottleneck is the cuBLAS-driven
   batched prefill, not `gemv_kernel` at all — confirming, not
   contradicting, the bandwidth analysis above).
3. **Lazy `lm_head` for tied dense/MoE models** — `system1_evaluate` gathers
   a handful of rows straight from host-resident `token_embd` instead of
   forcing the full matrix device-resident; the full upload is now deferred
   until an actual full-vocab call needs it (`generate`/`forward_prompt`,
   unaffected). **Done, 2026-09-25**: `LmHead` enum (`Resident`/`TiedLazy`)
   in `src/model.rs`, `Model::lm_head_resident`/`Model::gemv_gather_lm_head`,
   scoped to dense/MoE `Model::load` only (`load_hybrid`/`load_mla` untouched
   — `system1_evaluate` already rejects those architectures, so there's no
   win to have there). New test
   `gemv_gather_lm_head_matches_full_vocab_gemv_while_still_lazy` (real-
   hardware/`REFLEX_TEST_GGUF`-gated, `#[ignore]`d like its siblings) checks
   the new lazy compact-upload path against a forced-resident full-vocab
   `gemv` on real tensor data. Compiles clean and all 85 host-only tests
   pass under `REFLEX_SKIP_CUDA=1`, `cargo fmt --check`/`cargo clippy
   --all-targets -- -D warnings` both clean. **Real-hardware-verified,
   2026-09-25**: the new test passes on a real AWS EC2 T4, and
   `model_resident_mib` dropped from 2348 to **1740 MiB (−608 MiB)** exactly
   as predicted. Cold-start also improved (not this item's original target,
   but skipping a 608 MiB upload+dequant helps regardless of which
   subcommand triggers it): `model_load_ms` p50 889.5ms, cold
   `process_start_to_result_ms` p50 1104.4ms (down from ~1248.6ms
   pre-optimization) — see HISTORY.md for the full table.
4. **Phase-instrument `reflex system1`** — it had none of `reflex generate`'s
   Reddit-feedback-driven phase fields, so the cold-start split for the Jev
   citation was a guess. **Done, 2026-09-25**: `REFLEX_SYSTEM1_OK` now
   reports `gguf_open_ms`/`cuda_init_ms`/`model_load_ms`/`prompt_eval_ms`,
   same fields/bucketing convention as `generate.rs` (see `system1.rs`'s doc
   comment for the one asymmetry inherited from there: `--lora` apply time
   lands in `prompt_eval_ms`, not `model_load_ms`). New
   `scripts/bench_cold_start_phases_system1.sh` mirrors
   `bench_cold_start_phases.sh` field-for-field. **Real-hardware-verified,
   2026-09-25**, n=10 clean runs (no competing CPU load — see HISTORY.md's
   "Lesson for future sessions" paragraph, a contaminated first attempt run
   concurrently with a backgrounded `llama.cpp` compile gave misleadingly
   bad numbers): process launch p50 130.3ms/p95 132.3ms, CUDA init p50
   142.1ms/p95 146.7ms, model load p50 889.5ms/p95 895.7ms, scoring pass
   p50 35.7ms/p95 36.1ms, total p50 1104.4ms/p95 1110.9ms — tight spreads
   throughout, a real usable instrument now, not just a guess.
5. **Pipeline model load**: pinned double-buffered staging, async H2D on two
   streams; `alloc_zeros`→`alloc` for dequant kernel outputs (every element
   gets overwritten by the kernel, so zeroing first is wasted work). Est.
   -20-40% of the load phase. Medium effort, low risk — not started.

**Next real step**: items 2/3/4 are now all real-hardware-verified (see
HISTORY.md's "Real-hardware verification of the warm-latency perf plan"
entry for the full numbers and methodology). Only item 1 (f16 weight
residency) and item 5 (pipelined model load) remain — item 1 needs the
numerics-methodology decision flagged above before any code is written;
item 5 is the next low-risk pickup if more cold-load-time reduction is
wanted.

## IPC sampling + streaming (chat-completion-integration prep, done 2026-09-24)

Two gaps blocking any real chat-completion-style integration (OpenRouter, a
serverless platform, a first-party API) against `reflex stdio`/`reflex uds`, closed
in one round — see HISTORY.md's entry for the full design/verification writeup:

1. **Temperature/top-k/top-p sampling** (`src/sampling.rs`, new module). Confirmed
   first that this was unimplemented scope, not a permanent constraint — README's
   Non-goals list `batch_size`/concurrency/networking, never sampling strategy.
   Greedy argmax stays the default (`SamplingParams::temperature <= 0.0`) and is
   still what `reflex check`'s byte-exact-vs-llama.cpp methodology relies on;
   sampling is an explicit opt-in per request (`Model::generate`'s new `sampling:
   &SamplingParams` parameter; `reflex generate --temperature/--top-k/--top-p
   --seed` at the CLI; `"sampling": {...}` in the IPC JSON protocol).
2. **Per-token streaming** over the existing stdio/UDS line-delimited-JSON
   protocol (`crate::ipc::handle_request_streaming`, `"stream": true`) — one
   `IpcStreamToken` JSON line per generated token as it's produced, followed by one
   final `IpcResponse` line, both flushed immediately. `Tokenizer::decode_stream`
   (new) buffers a token's raw bytes until they form a complete UTF-8 character,
   avoiding a `U+FFFD` for a multi-byte character split across a token boundary —
   `decode`'s existing whole-buffer `from_utf8_lossy` never had to handle this
   since it only ever runs once the full id sequence is in hand.

No concurrency was introduced anywhere — token lines are written synchronously
from inside the same decode loop `Model::generate` already ran, one line at a
time, same `strictly sequential, never a thread pool` constraint the rest of this
module already holds to.

Real-hardware-verified on a fresh ThunderCompute A100 (`q82fifka`), against a real
`Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf` (downloaded via `reflex generate
--quickstart`): greedy output byte-identical across repeated runs and matching this
project's own documented golden continuation (`"The capital of France is"` ->
token ids `[12095,11,323,279,6722,315,15344,374]`, `" Paris, and the capital of
Italy is"`); `--temperature 1.2 --top-k 50` with no seed produced 4 different
continuations across 4 runs; the same `--seed` reproduced byte-identical output
across repeated runs, a different seed diverged; `reflex stdio` streaming emitted
the first token at +381ms and the final aggregate line at +1968ms for a 24-token
generation (a real ~1.6s gap across 23 decode steps, not a buffered-at-the-end
single write); a non-streaming request still produced exactly one JSON line
(regression check on the pre-streaming contract); `system1_evaluate` (the
`candidates` path) is unaffected by both `sampling` and `stream` (single-pass
score, nothing to sample or stream). `cargo test --release --features ipc`: 92
passed (host-only) + all `REFLEX_TEST_GGUF`-gated tests this round's code path
touches (`prefill_dense_batched_matches_sequential_prefill`,
`gemv_gather_matches_full_vocab_gemv_at_matching_rows`); `cargo fmt --check`/
`cargo clippy --release --features ipc,download --all-targets -- -D warnings`
both clean.

## OpenAI-compatible HTTP sidecar (`sidecar/openai-adapter`, done 2026-09-24)

Built the sidecar the IPC sampling/streaming round above was prep for, and the escape
hatch README's Non-goals section has described (but left unbuilt) since the MVP-
release round: a standalone `POST /v1/chat/completions` HTTP adapter (streaming SSE
and non-streaming JSON) in front of one managed `reflex stdio <gguf>` child process —
see HISTORY.md's entry for the full design writeup and `sidecar/openai-adapter/
README.md` for usage/scope/known limitations (no chat-template support yet — plain
role-labeled prompt concatenation; `usage.prompt_tokens`/`finish_reason` are
approximations, no exact-tokenizer/stop-reason info crosses the IPC boundary).

Deliberately its own Cargo project (own `Cargo.toml`/`Cargo.lock`, not a root-workspace
member, no dependency on `reflex-engine`) so axum/tokio never enter the core engine's
dependency graph and the sidecar builds with a plain stable Rust toolchain, no CUDA
toolkit needed. `batch_size == 1`/strictly-sequential is preserved underneath real
concurrent HTTP traffic by a single background worker task
(`src/reflex_client.rs::run_worker`) that's the only thing touching the managed
process's stdin/stdout — it never dequeues the next HTTP-originated request until it
has read the previous one's `"event": "final"` IPC line.

Real-hardware-verified on a fresh ThunderCompute A100-SXM4-80GB (`zx638gm8`, sm_80,
CUDA 12.6 toolkit installed fresh — this instance had a driver but no toolkit
preinstalled) against a real `Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf`: non-
streaming `/v1/chat/completions` returned `" The capital of France is Paris."` for
that exact prompt; SSE streaming delivered one `chat.completion.chunk` per token at
~60-70ms intervals (confirmed via timestamped `curl -N` output, not buffered until
the end); greedy (`temperature` omitted/`0`) was byte-identical across 3 repeated
requests; `temperature: 1.1`/`top_p: 0.9` produced 3 different continuations across 3
runs; two concurrent HTTP requests fired simultaneously both completed cleanly with
distinct `chatcmpl-reflex-*` ids and no interleaved/corrupted output, confirming the
worker's one-job-at-a-time queue actually serializes concurrent HTTP traffic into the
single underlying `reflex stdio` process. `cargo fmt --check`/`cargo clippy
--all-targets -- -D warnings` both clean on this crate, on both Windows (dev machine)
and Linux (the verification instance).

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
  directly. `cargo test` still 74 passed/0 failed under `REFLEX_SKIP_CUDA=1`. At the
  time, `cargo clippy` was still not wired into CI (the pre-existing `cargo fmt`
  non-compliance found the same CI session remained open), though this specific
  blocker was resolved. **Both closed 2026-09-24** — see HISTORY.md's follow-up CI
  entry: `cargo fmt` applied tree-wide (verified whitespace-only via token-stream
  diffing), every remaining `cargo clippy --all-targets` warning fixed or scoped-
  `#[allow]`ed with a documented reason, and both wired into `ci.yml` with
  `-D warnings`.
- ~~**Phase 2 round 3 on-device dequant scope**: only `Q4_K`/`Q6_K` dequantize on-GPU~~
  — **closed 2026-09-24**: every *block-quantized* GGUF format this project's
  `gguf.rs` parses now dequantizes on-GPU, closing out the IQ family (see this file's
  IQ-family entry below for the round that finished it). The passthrough types
  (F32/F16/Bf16) still take `dequantize_tensor_to_device`'s `other` fallback arm
  (host `dequant::dequantize` + `htod_sync_copy`) — unlike the block-quantized
  formats, there's no unpacking math to move on-device for these, just a type
  conversion, so this is a much smaller remaining gap than the one this entry
  originally tracked, not an oversight. This round's instance (an A6000) has no git
  history either (populated by `rsync`, not `git clone`) — local remains the only
  git-tracked copy; a fresh `ggml-org/llama.cpp` (`9655061`) was built there for the
  A/B benchmark.
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
  At the time, `docker run --rm --gpus all` remained unverified purely because this
  particular Docker host had no NVIDIA GPU at all (confirmed: `Get-CimInstance
  Win32_VideoController` → AMD Radeon only), not because of the nested-container
  access problem the previous entry described. `--gpus all` failed immediately with
  `nvidia-container-cli: initialization error: WSL environment detected but no
  adapters were found` — a hardware-absence error. (Incidental finding: Docker
  Desktop's WSL2 backend already has a working `nvidia-container-cli` wired in — the
  error is a specific "no adapter," not "toolkit missing" — so a Windows/Docker
  Desktop host with a real NVIDIA GPU would likely need no extra host-side toolkit
  setup for `--gpus all` to work.)
- ~~**`docker run --rm --gpus all` unverified**~~ — **closed 2026-09-24**: verified on
  a real EC2 `g4dn.xlarge` (On-Demand — Spot capacity was exhausted in every
  `us-east-1` AZ at launch time; the `base-oss-nvidia-driver-gpu-ubuntu-22.04` DLAMI
  already has Docker CE, the NVIDIA Container Toolkit, and the `nvidia` container
  runtime preinstalled, so no bootstrap script was actually needed on this AMI). Built
  the repo's own `Dockerfile` with `REFLEX_CUDA_ARCH=sm_75` (matching the instance's
  Tesla T4), then `docker run --rm --gpus all --entrypoint /usr/local/bin/reflex
  reflex:verify smoke` (the image's default `ENTRYPOINT` is `["reflex", "generate"]`,
  so invoking `smoke` needs an explicit entrypoint override or the binary mis-parses
  `smoke` as a GGUF path argument). **Result: `REFLEX_SMOKE_OK
  process_start_to_first_result_ms=707.599`**, GPU detected inside the container as
  `Tesla T4 (sm_75)` — confirms the AOT-compiled cubin kernel path loads and runs
  correctly via the CUDA driver API inside a real, non-nested Docker container with
  `--gpus all`. This was a one-off verification pass (throwaway IAM role/instance
  profile, temp S3 bucket used to ship the repo, and the EC2 instance itself were all
  created and torn down in the same session) rather than the persistent
  `docs/aws-deployment.md` ASG/EFS/ECR pattern, which remains unvalidated against real
  infrastructure as a separate, larger scope. This closes the Docker image's last
  unverified path — both `docker build` modes and now `docker run --gpus all` are
  real-hardware-verified.
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
- ~~**On-device dequant extended to the 8 IQ-family formats (IQ2_XXS/XS/S,
  IQ3_XXS/S, IQ1_S/M, IQ4_XS), not yet real-hardware-verified**~~ — **closed
  2026-09-24**: this was the last gap `dequantize_tensor_to_device` had — every
  other block-quantized format was already on-device (see the entry above). Unlike
  the prior round, these formats are non-uniform/codebook quant types: each block's
  raw bits index into a fixed lookup table of representative values, so the port
  needed `__constant__`-memory codebook tables in `kernels_cuda/dequant.cu`, not
  just per-block bit-unpacking. `scripts/gen_iq_tables.py` (referenced but not
  actually present in the repo before this round — written fresh) mechanically
  generates `kernels_cuda/dequant_iq_tables.cuh` from `src/dequant_iq_tables.rs`
  (itself already a byte-for-byte transcription of upstream `ggml-common.h`, per
  that file's own doc comment) rather than hand-transcribing the ~4000-line table
  set a second time from the C header — one generator, one source of truth. The 8
  new kernels (`dequantize_iq2xxs_kernel`/etc.) are line-for-line ports of their
  `dequant_iq.rs` host counterparts, same porting convention as every other kernel
  in the file, wired into `DequantKernels`/`load_dequant_kernels`/
  `dequantize_tensor_to_device` exactly like the Q2_K/Q3_K/Q8_K round. Real-
  hardware-verified on a fresh L40 (`4x8eh2ki`, sm_89): quantized a real
  `Qwen/Qwen3-0.6B` checkpoint with llama.cpp's own `llama-quantize` into all 8
  target types (the four lowest-bit types — IQ2_XXS/XS/S, IQ1_S — hard-require an
  imatrix even to run, discovered via `ggml_abort`/`GGML_ASSERT(imatrix != NULL)`
  when first attempted with `--pure` and no imatrix; generated one with
  `llama-imatrix` against a small synthetic calibration corpus, then dropped
  `--pure` for those four specifically since forcing every tensor including
  `output.weight` to a real IQ2/IQ1 type is not how any real published GGUF of
  these formats is built — the standard per-tensor type-selection strategy keeps
  `output.weight`/`token_embd` at a safer type for exactly this reason, and doing
  the same here incidentally exercised more of the new kernel set per file, since
  llama.cpp's default strategy mixes in `IQ3_S`/`IQ2_XS`/`IQ2_XXS` for specific
  tensors even when targeting a different nominal type). Added a new permanent
  `#[ignore]`d test, `model::iq_dequant_host_vs_device_tests::
  iq_dequant_kernel_matches_host_on_real_tensors` (same `REFLEX_TEST_GGUF`-env-var
  convention as this file's other real-GGUF tests), which extracts every real
  IQ-family tensor's raw block bytes from a GGUF, dequantizes them both via the
  host path and via `dequantize_tensor_to_device`'s on-device kernel, and asserts
  bit-exact equality — run against all 8 quantized files, every one of the 8
  kernels matched the host path bit-exact on real tensor data (up to 155M elements
  per tensor), not just the existing hand-computed-value unit tests' synthetic
  all-zero blocks. `reflex generate "The capital of France is"` matched
  `llama-simple` byte-exact on 3/8 (`IQ2_XS`, `IQ3_XXS`, `IQ4_XS`, each producing
  the same continuation token) with token-level divergence on the rest (`IQ2_XXS`,
  `IQ2_S`, `IQ3_S`, `IQ1_S`, `IQ1_M`) — not investigated to the same
  first-principles depth as the Q2_K/Q3_K precedent above (extracting the same
  tensor's bytes into `gguf-py` and comparing CPU-vs-GPU `llama-simple` runs for
  every diverging format), since the bit-exact host-vs-device dequant evidence
  above already directly answers the question this project's own methodology
  cares about (is the on-device kernel correct, not which implementation wins a
  given top-token race), and 1-3-bit quantization on a 0.6B model is exactly the
  regime the Q2_K/Q3_K investigation already established as inherently unstable
  top-token-race territory across any two correct implementations, not a new
  phenomenon to re-derive per format.
- **Sidecar chat-template rendering added, and it surfaced a real core-engine
  tokenizer bug that a prior round's own "no chat template" limitation had been
  hiding** (2026-09-24): `sidecar/openai-adapter` now reads and renders the
  loaded GGUF's own `tokenizer.chat_template` (via a new standalone
  `gguf_meta.rs` reader + `minijinja`) instead of always flattening messages
  into plain role-labeled text — see `sidecar/openai-adapter/README.md`'s
  updated "Known limitations" for the feature's own scope. Phase 1 (no GPU):
  the new reader's `tokenizer.chat_template` extraction matched `gguf-py`
  byte-for-byte, and `minijinja`'s render of the real Qwen3-0.6B ChatML
  template matched real Python `jinja2` (HF's own `Environment` settings)
  byte-for-byte, single- and multi-turn. Phase 2 (real GPU): a fresh AWS
  `g4dn.xlarge` (Tesla T4, `sm_75`, SSM-only access — this account still has no
  SSH key pairs) built clean and passed `reflex smoke`, but the *first* real
  chat-templated request came back visibly worse than the old flattening —
  meta-commentary, an immediate spurious `<|im_end|>`, wrong answers — the
  opposite of what this feature was supposed to buy. Root-caused with a
  throwaway host-only probe binary (no GPU needed — pure tokenizer logic;
  written, used, and deleted again, not kept in the repo) against the real
  vocab: `src/tokenizer.rs`'s `encode_gpt2` (and `encode_sentencepiece`) had
  **no special/added-token exact-match pass at all** — a rendered template's
  literal `<|im_start|>` substring was shredded by generic BPE into 6
  meaningless byte-fragment tokens (`<`/`|`/`im`/`_start`/`|`/`>`) instead of
  the model's one real reserved id, corrupting every templated prompt in a way
  the old flattening (which never contains those literal marker substrings)
  never triggered. Fixed by adding a `special_tokens` list to `Tokenizer`
  (every vocab entry whose `tokenizer.ggml.token_type` is `CONTROL` (3) or
  `USER_DEFINED` (4) — confirmed against the real Qwen3-0.6B GGUF's own
  metadata: ChatML's markers are `CONTROL`, `<think>`/`<tool_call>`/etc. are
  `USER_DEFINED`) and a longest-match literal-substring scan in `encode` that
  runs before the general regex-pretokenize/BPE path, matching how real
  llama.cpp/HF tokenizers already handle this. Two new unit tests
  (`test_encode_matches_special_token_as_single_id_not_bpe_fragments`,
  `test_encode_matches_special_token_mid_text`) cover the regression; the
  probe confirmed `<|im_start|>` now maps to its real id (151644) on the real
  GGUF, and a full end-to-end re-run showed the model producing a genuine
  `<think>...</think>` reasoning trace that explicitly referenced a
  system-prompt persona ("As a pirate, I need to keep it fun and engaging")
  before answering in character — the coherence and system-prompt steering
  this feature was meant to deliver, now real-hardware-confirmed. One small
  model-quality miss noted honestly, not a plumbing bug: the templated path
  got a 2-step arithmetic follow-up wrong (answered `2*3=6` instead of
  `4*3=12`, misreading which prior number "times 3" referred to) where the old
  flattening happened to get it right by shallow text-continuation pattern-
  matching — expected variance for a 0.6B model's actual reasoning, not
  evidence of a remaining correctness bug. One small cosmetic gap also noted,
  not fixed here (out of scope, core `generate`/decode behavior): the literal
  EOS marker text (e.g. `<|im_end|>`) can appear in the returned
  `message.content` when generation stops on it. AWS resources (the
  `g4dn.xlarge`, its throwaway SSM-only IAM role/instance profile) were fully
  torn down after verification.
- **Cold-start phase breakdown added** (2026-09-25, real Reddit feedback --
  see HISTORY.md's "Cold-start phase breakdown" entry): `reflex generate`
  reports `gguf_open_ms`/`cuda_init_ms`/`model_load_ms`/`prompt_eval_ms`; new
  `scripts/bench_cold_start_phases.sh` runs N cold processes and reports
  p50/p95 per phase. Real-hardware-verified on a ThunderCompute L40, n=30
  (10+20 runs): CUDA init small/stable (417.9ms/542.6ms p50/p95), model load
  dominates and is the least stable phase, session-to-session variance
  exceeded intra-session variance in this round's own two batches -- disclosed
  in README.md's new "Cold-start phase breakdown" subsection rather than
  smoothed over. **Still open, not yet attempted**: host-vs-container/cgroup
  rows and a persistent-vs-`exec`'d row (both specifically requested,
  deliberately scoped out of this round to avoid a multi-day detour).
