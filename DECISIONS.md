# DECISIONS.md

A log of the project's non-obvious technical and scope decisions, and why they were
made. For current state see `STATUS.md`; for full narrative/benchmark detail see
`README.md`.

## Phase 3 (State I/O) round 1 scope: dense/MoE only, export/import-the-buffers only

**Decision**: `--export-kv`/`--import-kv` round 1 supports only the dense/MoE Qwen3
forward path's `k_cache`/`v_cache` pair (not the Qwen3.5 hybrid's per-layer
`Attn`/`Gdn` split, not MLA's single compressed cache), and does not wire an imported
cache back into a forward pass — `--import-kv` proves the file round-trips
byte-identical through a device upload/download and stops there, rather than resuming
generation from it.

**Why**: confirmed with the user before implementing (two explicit open questions,
matching this project's practice of confirming scope before starting rather than after).
Dense-only matches the established "narrow first" precedent from every prior MVP step
(dense before MoE before hybrid before MLA). The export/import-only cut came from what
investigating `model.rs` actually found, not just caution: `forward_prompt` (all three
architecture paths) always starts at position 0, and there is no per-token generation
loop anywhere in `model.rs` or `qwen3_coldstart.rs` — each run does one full prompt pass
and returns exactly one token, then the process exits. "Resume generation past position
0" therefore isn't a small `start_pos: usize` plumb-through; it needs (a) a generation
loop that doesn't exist yet, and (b) each path's K/V cache allocation changed from
"sized to exactly this call's token count" (`alloc_zeros(ids.len() * ...)`, fresh per
call) to something sized for `start_pos + new_tokens`, at three separate call sites
(dense/MoE, hybrid, MLA don't share the position loop). Scoping that out of round 1
kept this round verifiable on its own terms (a lossless byte-exact round trip) instead
of half-building a resume path with no generation loop to plug it into.

**How to apply**: round 2 of this phase is where the generation loop + `start_pos`
plumbing belongs, together — building one without the other leaves either a resume
path nothing can call, or a loop nothing can resume. Do both in the same round, and
revisit whether hybrid `Gdn` state (already streaming-friendly — fixed-size, not
indexed by position) can piggyback on the same file-format version bump used for
hybrid `Attn`/MLA's cache shapes.

## Phase 3 (State I/O) round 2 scope: dense/MoE + hybrid resume, MLA still round 3

**Decision**: round 2 wires `--import-kv` up to actually resume generation and adds a
real per-token generation loop (`--max-tokens N`), together, per round 1's own "How to
apply" note. Scope is **dense/MoE and the Qwen3.5 hybrid mixer**; MLA's single
compressed cache is deliberately deferred to round 3.

**Why**: confirmed with the user before starting (three options offered: dense/MoE
only, +hybrid, or +hybrid+MLA all in one round). Dense/MoE+hybrid matches this
project's narrow-first precedent while still being the meaningfully smaller lift of
the two extensions round 1 flagged as open: the hybrid `GatedAttention` sublayers need
exactly the same `start_pos`-offset `k_cache`/`v_cache` treatment dense/MoE's single
`k_cache`/`v_cache` pair does (same `forward_attn_block`-style position-indexed
device-to-device copy), and the `GatedDeltaNet` sublayers' `conv_state`/`recurrent`
need *no* `start_pos` handling at all — they're fixed-size buffers mutated in place
regardless of position, so they round-trip by uploading them as-is. MLA's single
compressed `kv_cache` (`forward_mla_attn_block`) is a third, structurally different
shape again and was left for its own round rather than tripling this round's
verification surface.

**Implementation note**: `k_cache`/`v_cache` (and hybrid's `GatedAttention` cache) are
now allocated for `start_pos + prompt_len + max_new_tokens` up front — headroom for
every token the call might still generate — rather than exactly the prompt length.
Callers that download the cache for `--export-kv` (`forward_prompt_capture_kv`/
`forward_prompt_capture_kv_hybrid`) must slice to `0..seq_len * kv_stride` before the
download, not the whole buffer, or the exported file's `k_caches`/`v_caches` length
stops matching its own `seq_len` metadata. `kv_io.rs`'s hybrid format is a new version
(2), read-dispatched by `import_kv`/`ImportedKv` — round 1's version-1 dense format is
untouched and stays readable via the same `import_dense_kv` it always was, satisfying
round 1's own "version-gated so a later round can add per-architecture variants
without breaking round-1 files" design note.

**Finding surfaced during verification, not a round-2 defect**: byte-exact
export→import→continue-vs-single-run verification worked cleanly for dense
(`Qwen3-0.6B-Q4_K_M.gguf`) and hybrid (`Qwen3.5-0.8B-Q4_K_M.gguf`) — both use the
`gpt2`-style tokenizer path (`encode_gpt2`), whose regex pre-tokenization has no
context-dependent whitespace handling across an arbitrary split. It did *not* work for
`Tiny-Moe.Q4_K_M.gguf` (the only local MoE fixture): its `general.architecture="llama"`
GGUF routes through `encode_sentencepiece`, which unconditionally prepends an implicit
leading-space symbol to *every* `encode()` call (`tokenizer.rs`'s own doc comment,
confirmed against real TinyLlama). A continuation prompt re-encoded on its own always
picks up that phantom extra symbol, so it can never be byte-identical to the same text
encoded as a substring of one continuous prompt — a real, deliberate property of
SentencePiece tokenization, not something `--import-kv`'s resume logic does wrong.
Confirmed this is tokenizer-level, not resume-level, three ways: (1) the discrepancy is
reproducible from two independent, non-resuming `encode()` calls with no cache/model
code involved at all; (2) `generate_dense_impl`/`forward_one_token_dense` — the actual
resume machinery — is *exactly* the same code for dense and MoE, since `forward_layer`
dispatches dense vs. MoE only inside the FFN tail (`forward_layer_moe`), which has no
position or cache logic whatsoever; (3) resuming `Tiny-Moe` twice with identical
imported-cache + continuation-prompt inputs produces byte-identical output both times
(determinism holds; the surprising part is only that it disagrees with the *differently
tokenized* single-run baseline, not that it's inconsistent with itself).

**How to apply**: don't add a workaround for SentencePiece's implicit-leading-space
convention to `tokenizer.rs` — it's correct, documented, real-model-verified behavior,
not a bug. If a real `qwen3moe` fixture is ever obtained (open follow-up, see
STATUS.md), prefer it over `Tiny-Moe` for any future text-level continuation testing,
since Qwen-family models use the `gpt2` tokenizer path this property doesn't apply to.
Round 3 (MLA) should budget for verifying resume the same two ways used here: a
byte-exact text-continuation check against whichever tokenizer the MLA fixture(s) use,
plus a determinism check as a fallback if that fixture also turns out to be
SentencePiece-based.

## Phase 3 (State I/O) round 3 scope: MLA resume, synthetic fixture not real DeepSeek-V2-Lite

**Decision**: round 3 extends `--import-kv` resume + `--max-tokens` to MLA models,
closing Phase 3's architecture coverage (dense/MoE, hybrid, and now MLA all support
export/import/resume). Verification uses the synthetic `test-data/deepseek-tiny-mla.gguf`
fixture (dense-lead layers only), not the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint's
MoE+shared-expert+YaRN path.

**Why**: confirmed with the user before starting (two explicit open questions: which
fixture(s), and whether to reuse or create a ThunderCompute instance — `tnr status --json`
showed none running, so a fresh A6000 was created). The synthetic fixture was chosen over
real DeepSeek-V2-Lite because round 3's actual code delta is cache/`start_pos` plumbing at
the attention-block level (`forward_mla_attn_block` already took an absolute `position`
argument and indexed its single `kv_cache` by it, unchanged by this round), not new
math — the routed-MoE/shared-expert/YaRN correctness was already byte-exact-verified
against real DeepSeek-V2-Lite in the MVP step 4 session, and `forward_one_token_mla`'s
per-layer loop dispatches to `MlaFfn::Dense` or `MlaFfn::Moe` identically regardless of
`position`/cache state (same "cache mechanism is FFN-routing-independent" reasoning round
2 used to justify not re-verifying hybrid's `GatedDeltaNet` state against every FFN
variant). The synthetic fixture is also free to run (already on disk, needs only an
A6000) versus the real checkpoint's rented-80GB-A100 + from-source-conversion cost, which
would have been disproportionate to what this round is actually testing. See STATUS.md's
"Known debt" for the resulting gap (resume unverified specifically against the MoE+YaRN
combination) — flagged, not planned as follow-up unless a real need comes up.

**Implementation note**: mechanically identical to round 2's dense/hybrid extension --
`generate_mla_impl` preallocates each layer's `kv_cache` for `start_pos + prompt_len +
max_new_tokens` (instead of exactly the prompt length), uploads an imported cache into
the front of that buffer via `htod_sync_copy_into`/`slice_mut` before the per-position
loop starts, and `forward_prompt_capture_kv_mla` (the `--export-kv` capture path) slices
to `0..seq_len * qk_dim` before downloading, same as the dense/hybrid capture functions
already do. `kv_io.rs`'s MLA format is a new version (3, `MlaKvCache` -- one buffer per
layer, no separate K/V pair since MLA's compressed latent is shared and decompressed by
`wk_b`/`wv_b` on the fly), read-dispatched by `import_kv`/`ImportedKv` alongside the
untouched version-1/2 formats.

**Verification finding, not a defect**: unlike `Tiny-Moe` in round 2, the synthetic MLA
fixture's `tokenizer.ggml.model` is `gpt2` (confirmed by reading the GGUF's own metadata
bytes before relying on it, following the fixture-recipe memory's HTTP-range-request
pattern applied locally instead) -- the same tokenizer family dense/hybrid verified round
2's byte-exact bar with, and not SentencePiece's implicit-leading-space property that
made `Tiny-Moe` need a determinism fallback instead. Byte-exact text-continuation
verification therefore worked directly, no fallback needed: `[69344,10420,40306,145381,
87488]` in both the single uninterrupted run and the export→import→continue run, split
at `"The quick brown fox jumps over the lazy dog"` + `" and runs"`.

**How to apply**: this closes Phase 3's architecture-coverage scope entirely -- no round
4 of this phase is currently planned. If real DeepSeek-V2-Lite's MoE+YaRN resume path
ever needs verifying specifically (e.g. before relying on it for a real deployment),
budget for the rented-A100 + from-source-conversion cost the MVP step 4 MLA-extension
session already paid once (see STATUS.md's "Real DeepSeek-V2-Lite GGUF not preserved
locally" entry for the exact regeneration steps) rather than assuming the dense-fixture
verification generalizes untested.

## Hybrid Qwen3.5 MVP scope: dense `qwen35` only, single-token dispatch, no MTP

**Decision**: MVP step 3 supports only the dense `qwen35` architecture string. The
`qwen35moe` variant (same hybrid mixer interleaving, but every layer's FFN is additionally
a routed+shared-expert MoE) is explicitly rejected with a clear error rather than silently
mishandled. MTP/NextN blocks are also rejected outright (`nextn_predict_layers != 0`
errors immediately) rather than partially supported. The forward pass processes one token
at a time sequentially through the Gated DeltaNet recurrence, even during prompt
processing — no chunked/parallel-prefill kernels.

**Why**: matches this project's established naive-first precedent (see MoE's naive
per-expert dispatch decision above) — get one real, narrow architecture path fully correct
and hardware-verified before broadening scope. `qwen35moe` would combine two MVP
milestones' worth of complexity (hybrid mixer routing + MoE FFN routing) into one change,
and no small real `qwen35moe` fixture was available to verify it against anyway. Chunked
prefill is a real llama.cpp optimization (closed-form solve over a whole chunk instead of
per-token), but it's parallel-throughput-oriented — exactly the kind of optimization this
project's cold-start-latency bet doesn't need first, and adds substantial kernel
complexity (a lower-triangular solve, cumulative log-decay) for a small number of prompt
tokens in the cold-start use case.

## Cross-check new architecture work against an independent ground truth before trusting it

**Decision/lesson**: when implementing a new model architecture path, don't trust
"produces plausible-looking output without crashing" as evidence of correctness — build a
fresh copy of the reference implementation (llama.cpp) from source and compare real
generated tokens on the same prompt/fixture before considering the work done.

**Why**: the first working build of the Qwen3.5 hybrid path produced fluent but
semantically wrong completions (e.g. Chinese text following an English prompt) — no
crash, no NaN, deterministic output, which could easily have been mistaken for "probably
fine, the base model is just small/quirky." The actual bug (`forward_gdn_mixer` never
added its output back to the residual stream, unlike the Gated Attention mixer's forward
function — see `model.rs`) broke the residual stream through 18 of the fixture's 24
layers, entirely invisible from output shape/NaN-checks alone. It was only caught by
building `ggml-org/llama.cpp` from source on the same instance and comparing real
generated token ids on identical prompts — the same real-hardware-verification posture
this project already applies to its own kernels (see README's dense/MoE sections), now
extended to mean "verify against an independent implementation," not just "verify it
doesn't crash." **How to apply**: for any new architecture/kernel path, get a real
independent-implementation comparison (llama.cpp, or a pure-host CPU reference computed
from the same real dequantized weights) before calling the work done — a clean run is not
evidence of correctness.

**Related tooling note**: `llama.cpp`'s `tools/cli` (`llama-cli`)'s newer conversational
mode always applies the model's embedded chat template, even when a raw prompt is passed
via `-p`, and there is no `--no-cnv` flag in current versions to disable it — this
recreates the exact chat-template caveat already flagged in the original dense-Qwen3-vs-
llama.cpp benchmark (see README). Use `examples/simple` (built as `llama-simple`) instead
for a true prompt-in/token-out comparison with no chat wrapping.

## AOT kernel compilation, never NVRTC

**Decision**: every CUDA kernel is compiled by `nvcc` at build time (`build.rs`), loaded
via the CUDA driver API at process start (`src/aot.rs`). No runtime JIT compilation
(NVRTC) path exists or is planned.

**Why**: this project is a direct pivot from RustFeference (`rft-gpu`), which compiled
kernels via NVRTC at process start and measured a real ~4.5s JIT tax per invocation —
fatal for a cold-start-latency target. llama.cpp avoids this because `nvcc` compiles its
kernels at build time; coldstart-infer adopts the same approach as its core bet.

**Open question, not yet resolved**: whether to default to portable PTX (small
driver-side JIT-to-SASS cost) or `COLDSTART_CUDA_ARCH=sm_XX`-targeted cubins (zero JIT,
but needs a matching cubin per deployment target) is unverified on real model kernels —
`smoke_coldstart` found the two statistically indistinguishable (473–637ms vs.
480–617ms), but only for a trivial kernel where CUDA context init dominates. Worth
re-measuring against real model-sized kernels.

## No serving-platform features, ever (permanent non-goal)

**Decision**: `batch_size` is always 1. No internal request queue/scheduler, no
continuous batching, no multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager,
no concurrent HTTP/gRPC server, no autoscaling logic. If a warm-context mode is ever
built, it accepts one job at a time, strictly sequentially — never a thread pool.
Multi-tenancy and persistent state are the *host orchestrator's* job, not this engine's.

**Why**: RustFeference's own postmortem
(`../RustFeference/LESSONS_LEARNED_RUSTFEFERENCE.md`) is explicit that a broad serving
feature set re-enters the exact warm-throughput race against vLLM/SGLang that project
already lost, with no reason to expect a different outcome this time. coldstart-infer
exists to win a narrower, different bet (cold-start energy/latency) — the two goals are
in tension, and every feature that inches toward serving-platform territory dilutes the
one thing this project is trying to prove. This was explicitly considered (a
"Serverless-Native Inference Engine" feature pitch, framed as what GPU cloud providers
like Modal/Replicate/RunPod want) and rejected on 2026-09-17.

**How this plays out concretely**: instead of building these features in, the roadmap
pushes the same needs outward — Phase 3 (`--export-kv`/`--import-kv` raw file flags,
engine stays ignorant of storage backend) and Phase 4 (`--lora <path>` load-time-only
flag, Rust C-FFI so an external orchestrator can embed the engine) — see README's
"Post-architecture-MVP roadmap" section.

## MVP architecture order: dense → MoE → hybrid mixer → MLA (last)

**Decision**: Qwen3 dense first, then Qwen3-MoE, then the Qwen3.5 hybrid Gated DeltaNet
mixer, with DeepSeek-V2/V3 MLA deliberately last.

**Why**: dense Qwen3 reuses RustFeference's most mature, most-verified architecture,
proving the AOT-compilation + cold-start-benchmark harness works at all before spending
effort on anything novel. MLA is ordered last on purpose — it's a genuinely different
caching strategy (compressed latent KV), not an incremental extension of GQA, so it
carries the most implementation risk and should be tackled once the rest of the harness
is trustworthy. (Read llama.cpp PR #11446 before attempting it.)

## Naive (ungrouped) per-expert MoE dispatch, not grouped/batched

**Decision**: Qwen3-MoE's FFN dispatches one `gemv_expert` call per selected expert
(sliced out of a `[in_features, out_features, expert_count]` per-expert-stacked
tensor), rather than grouping tokens by expert or batching expert computation.

**Why**: this matches RustFeference's own documented finding that naive per-expert
dispatch is the correct starting point before optimizing. Since `batch_size` is always 1
here (see the serving non-goal above), there's only one token being routed per forward
call anyway — the grouped-dispatch optimization that matters for batched serving doesn't
apply the same way to this project's actual workload shape.

## Weights are GPU-resident once, not re-uploaded per call

**Decision**: `Weight` (in `model.rs`) holds a `CudaSlice<f32>` uploaded to the GPU once
at load time (immediately after dequantizing, with the host scratch buffer dropped).
`gemv`/`gemv_expert`/`rmsnorm` take that device buffer directly; MoE's per-expert
dispatch slices it with a zero-copy `CudaSlice::slice` view.

**Why**: the first real cold-start-vs-llama.cpp benchmark (2026-09-17) found
coldstart-infer ~4.3x slower, 4x the peak RSS, and ~11x the system CPU time of
llama.cpp. Reading `model.rs`'s kernel-calling functions directly (not just trusting the
first plausible root cause) found the *dominant* cost wasn't just load-time dequant to
`f32` — it was that every weight buffer was being re-uploaded to the GPU via a fresh
`htod_sync_copy` on **every single call** (every layer, every token). Fixing this closed
the gap from 4.3x to ~1.7x slower. **This is a decision worth defending against
regression**: don't reintroduce per-call weight upload for convenience or refactoring —
it was silently catastrophic and easy to miss without hardware benchmarking.

## Activations are device-resident across a whole layer, not round-tripped per op

**Decision**: every kernel-wrapper method in `model.rs` (`rmsnorm`, `gemv`/
`gemv_expert`, `rope`, `silu_and_mul`, `attention`, and the Qwen3.5 hybrid's `gdn_*`
kernels) takes/returns `CudaSlice<f32>` device buffers directly, chained through each
of the six forward functions without touching host memory mid-layer. Residual adds use
a new in-place `add_kernel` (`kernels_cuda/elementwise.cu`) instead of a host-side
`zip().map()` loop. `rope_kernel` takes `position` as a scalar kernel argument instead
of an uploaded one-element device array (`batch_size` is permanently 1, so there's
never more than one position to pass — see the serving non-goal above).
`silu_and_mul_kernel` takes `gate`/`up` as two separate buffers instead of requiring a
host-side concatenation into one. `k_cache`/`v_cache` are preallocated device buffers
(sized to the known prompt length before the per-position loop starts) written into via
`CudaDevice::dtod_copy`, instead of `attention()` re-uploading the entire cache history
from host on every token position.

**Why**: this is Phase 2 (Fast IO) round 2 — round 1 (above) fixed *weights* being
re-uploaded per call, but every op still separately `htod`/`dtoh`'d its small
*activation* vectors, and `attention()`'s full-K/V-cache-history re-upload scaled with
position count. Closed the cold-start gap vs. llama.cpp from ~1.7x to ~1.1x (see
[[coldstart-infer-benchmark-result]] memory / README's "Phase 2, round 2" section).
Left out of scope on purpose: MoE's per-expert weighted-sum accumulation and the Gated
Attention mixer's fused-qg head split/sigmoid gating still round-trip through the host
— small, and not on the primary dense/MoE path this benchmark measures. **This is
worth defending against regression** for the same reason round 1's decision is: it was
found by systematically removing every per-op host round-trip, not by guessing, and
reintroducing one for convenience would silently reopen part of the gap.

## MLA (MVP step 4): synthetic fixture instead of real DeepSeek-V2-Lite, narrow scope

**Decision**: DeepSeek-V2/V3 MLA support is verified against a fully synthetic
`deepseek2` GGUF (`test-data/deepseek-tiny-mla.gguf` — hand-built HF `config.json` +
random-weight `safetensors`, run through llama.cpp's own real `convert_hf_to_gguf.py`
for authentic tensor layout), not the smallest real `deepseek2` model
(DeepSeek-V2-Lite). Scope is dense-only (no MoE FFN/shared experts), no Q-LoRA query
decomposition, no YaRN RoPE scaling, no MTP — each rejected with a clear error
(`parse_mla_config` in `model.rs`), not silently mishandled.

**Why**: no small real `deepseek2`-architecture GGUF exists publicly at all (unlike
every prior MVP step's fixture problem, which was "not available locally" — this one
genuinely doesn't exist to find). DeepSeek-V2-Lite, the smallest real one, needs
~63GB of `f32` device memory for its 64-routed-expert MoE FFN alone under this
project's "dequantize every weight once, hold it GPU-resident for the model's whole
lifetime" design (`Weight` in `model.rs`, defended against regression by the "Weights
are GPU-resident once" decision above) — more than the A6000 (48GB) this project
develops against. Given the user's explicit choice between renting an ~80GB H100 for
the real model vs. a synthetic fixture on the existing A6000, the synthetic fixture
was chosen to verify the MLA *mechanism* cheaply first, deferring the real model (and
its MoE-residency implications, a separate, bigger decision) to optional future work.

**Two real bugs found via the byte-exact llama.cpp comparison**, both silently
producing plausible-looking wrong tokens rather than crashing (same lesson as the
Gated DeltaNet mixer's residual-add bug):
1. The attention softmax scale must be `1/sqrt(qk_nope_head_dim +
   qk_rope_head_dim)` (the *uncompressed* per-head dim) — not `1/sqrt(kv_lora_rank +
   qk_rope_head_dim)` (the compressed dot-product width actually used to compute the
   scores), confirmed against llama.cpp's `deepseek2.cpp` `kq_scale` directly. Every
   other attention op in this codebase scales by its own dot-product dimension, so
   this is an easy trap to fall into by pattern-matching instead of reading the
   reference source.
2. MLA's `q_pe`/`k_pe` RoPE uses `LLAMA_ROPE_TYPE_NORM` (consecutive-pair rotation),
   confirmed against llama.cpp's `llama_model_rope_type` — a different convention
   from `LLAMA_ROPE_TYPE_NEOX` (half-split), which `rope_kernel` already implements
   for Qwen3/Qwen3.5. Fixed with a new, separate `rope_norm_kernel`
   (`kernels_cuda/rope.cu`) rather than modifying the existing, hardware-verified
   `rope_kernel` — **don't assume RoPE convention is architecture-independent** when
   adding another model family later; check `llama_model_rope_type` first.

## MLA extended to real DeepSeek-V2-Lite: MoE + shared experts + YaRN, same session

**Decision**: after the synthetic-fixture-only MLA work above landed, it was
extended the same session to the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint on a
rented 80GB A100 (the VRAM estimate from the entry above held: fits 80GB with
headroom, doesn't fit the A6000's 48GB). This added routed-MoE + always-on
shared-expert FFN (`MlaFfn::Moe`) and YaRN RoPE scaling (`MlaYarnConfig`,
`rope_norm_yarn_kernel`) to the previously dense-only, no-YaRN implementation.

**Why extend in the same session rather than treat it as separate future work**: the
user explicitly chose full scope (MoE+shared-experts+YaRN together) after being told
YaRN was a real, separate chunk of work beyond the originally-scoped MoE addition —
see the "genuinely blocked, decision only they can make" judgment call this
represented; once approved, there was no reason to artificially split the work across
sessions.

**Three more real, silently-wrong-not-crashing bugs/gotchas found via byte-exact
comparison against a real llama.cpp build on the real model** (extending the pattern
from the entry above):
1. **DeepSeek-V2-Lite's router does not renormalize top-k probabilities**
   (`norm_topk_prob: false` in the source HF config) — unlike Qwen3-MoE's convention
   this codebase's `route_top_k` already assumed everywhere. The tell in the GGUF is
   subtle: llama.cpp's converter only writes `expert_weights_norm` when the source
   value is *truthy*, so **the key's absence means "don't renormalize," not "key
   missing, assume the usual true default"** — the opposite of how most other
   optional metadata keys in this codebase behave. Fixed via `route_top_k_with_norm`
   (`moe.rs`), `route_top_k` now a thin wrapper over it.
2. **The shared expert(s) are a single fused dense FFN**, not
   `expert_shared_count` separate per-expert calls — confirmed from the real
   converted GGUF's own tensor shapes (`ffn_gate_shexp`: `{hidden, n_ff_exp *
   expert_shared_count}`). Assuming a per-expert loop (the natural pattern-match
   from the routed-expert code right next to it) would have been extra unneeded
   complexity, not just a performance issue.
3. **Every pre-quantized DeepSeek-V2-Lite GGUF found publicly (mradermacher,
   tensorblock, duyntnet, bartowski's Coder-V2-Lite) predates llama.cpp's MLA
   tensor-split conversion change** — legacy unsplit `attn_kv_b`, no
   `key_length_mla`/`value_length_mla` metadata at all. **Don't assume a downloaded
   or found GGUF for an architecture with an evolving tensor format is current** —
   check for the format-defining metadata keys (an HTTP range request for just the
   header, ~20MB, is enough) before committing to a multi-GB full download. The
   reliable fix was converting fresh from the original safetensors checkpoint with
   this project's own pinned, confirmed-current `convert_hf_to_gguf.py`.

**Also required, less novel but worth noting**: YaRN's math is genuinely separate
from a simple RoPE scale tweak — it blends interpolated/extrapolated rotation angles
per frequency (a correction ramp from `beta_fast`/`beta_slow`-derived dimension
bounds) and applies a magnitude correction to both the rotation itself *and*,
independently, the attention softmax scale (a second, different formula, involving
`deepseek2`'s own `rope_yarn_log_mul` metadata — itself stored pre-multiplied by
`0.1` by the converter and undone by the loader, `[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`,
a detail this project's `parse_mla_config` had to replicate exactly). Ported by
reading `ggml`'s own CUDA `rope_yarn()`, `llama-context.cpp`'s YaRN `cparams` setup,
and `deepseek2.cpp`'s `kq_scale` computation in full, not derived from first
principles or copied from a simplified description.

## On-GPU dequant kernel: only Q4_K/Q6_K, not all 18 block types

**Decision**: Phase 2 (Fast IO) round 3's on-device dequant kernels
(`kernels_cuda/dequant.cu`) cover only `Q4_K` and `Q6_K`. Every other GGUF block type
this project supports on the host (`Q4_0/1`, `Q5_0/1`, `Q8_0/1`, `Q2_K`/`Q3_K`/`Q5_K`/
`Q8_K`, all 8 IQ-family formats) still dequantizes via the existing
`dequant::dequantize`/`dequant_iq::dequantize_block_*` host path, unchanged.
`model.rs`'s new `dequantize_tensor_to_device` dispatches on `ggml_type` per tensor and
falls back to the host path for anything that isn't `Q4_K`/`Q6_K`.

**Why**: this project's only three local `Q4_K_M` GGUF fixtures (`Qwen3-0.6B`,
`Tiny-Moe`, `Qwen3.5-0.8B`) use `Q4_K`/`Q6_K` for the overwhelming majority of weight
bytes — writing GPU kernels for the other 16 types would be real new-kernel-writing
work with no local fixture able to prove them byte-exact (the synthetic MLA fixture and
the real DeepSeek-V2-Lite checkpoint both use F16/F32 and Q8_0 respectively, neither of
which needed a new kernel to begin with — Q8_0's dequant is already a trivial `x = d*q`
per-element multiply, not a meaningful CPU-time contributor next to K-quant's bit-
unpacking). Confirmed with the user before starting (this project's usual practice for
open-ended follow-up work — see STATUS.md's prior "Open decision" entries) with this
exact scope question asked explicitly, rather than assumed.

**How to apply**: if a future fixture or real deployment target exercises another
block type for the bulk of its weight bytes, extend `dequantize_tensor_to_device`
(`model.rs`) and `dequant.cu` for that type specifically, verified byte-exact against
that fixture — don't add speculative coverage for types nothing here can verify.

## Cold-start-vs-llama.cpp benchmark methodology

**Decision**: measure with external wall-clock (`/usr/bin/time -v`, process launch to
exit — including OS exec/dynamic-linking overhead), not just the engine's own internal
`Instant::now()`-based metric, when comparing against llama.cpp. Use greedy decoding
(`--temp 0` / argmax) on both sides for determinism, and disclose (not paper over)
methodology caveats — e.g. llama.cpp's `llama-cli` REPL overhead and unconfirmed
chat-template handling in the first benchmark run.

**Why**: internal metrics only tell you about this engine in isolation; the project's
actual claim is comparative (cold-start energy/latency vs. existing engines), so the
comparison has to use a fair, external, reproducible measurement or the result isn't
trustworthy. Small honest caveats are worth stating explicitly rather than silently
assuming they don't matter — see README's benchmark section for the specific caveats
disclosed in the first run.

## Phase 4 (Embeddability) round 1 scope: `--lora`, dense/MoE + hybrid Gated-Attention tensors only, load-time-only

**Decision**: `--lora <adapter.gguf>` applies a llama.cpp-format LoRA adapter
(`src/lora.rs`) to any 2-D `nn.Linear`-shaped weight the loaded model actually has —
dense/MoE Qwen3's attention tensors, dense's FFN, and the Qwen3.5 hybrid's
Gated-Attention-layer attention/FFN tensors and the Gated DeltaNet mixer's FFN tensors
— once, at load time, in `Model::apply_lora` (`model.rs`). Explicitly rejected, not
silently mishandled: DeepSeek-V2/V3 MLA outright (checked first, before parsing the
adapter at all), MoE's per-expert-stacked `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`
tensors (3-D, no per-expert LoRA targeting), the Gated DeltaNet mixer's non-Linear
state-space tensors (`ssm_*`, `attn_qkv`, `attn_gate`), and `token_embd`/`output`/norm
tensors — all four cases fall through `Model::find_lora_target_mut`'s match arms to a
single clear "no matching 2-D weight" error rather than four separate checks, since
none of them are 2-D `Weight`s regardless of architecture.

**Why**: confirmed with the user before starting (three explicit questions: scope,
fixture, GPU instance — this project's usual practice). Scope came back "dense/MoE +
hybrid" rather than dense-only, matching the project's narrow-first precedent one step
wider than MVP-step ordering alone would suggest, but MLA was excluded by the user's
original framing (`README.md`'s Non-goals: load-time-only adapter application, no
runtime hot-swap) and every other MLA-adjacent feature in this codebase already stops
at the same line. The tensor-level accept/reject split (2-D found-and-shape-matches vs.
everything else) came from what the real format actually looks like once inspected
(llama.cpp's `convert_lora_to_gguf.py`/`src/llama-adapter.cpp`, read in full, not
assumed): a LoRA adapter has no idea what architecture its base model is, it just
targets tensor *names*, so a from-scratch reimplementation only needs to know which of
*this project's* tensor names are 2-D and GPU-resident — the MoE/hybrid-mixer-specific
rejections are a consequence of that lookup, not separate special cases to write.

**Format decision**: parsed with llama.cpp's own convention (its own GGUF adapter
container, `adapter.type="lora"` + `adapter.lora.alpha` metadata, `<name>.lora_a`/
`<name>.lora_b` tensor pairs where `<name>` already includes the base tensor's
`.weight` suffix — confirmed against a real converted adapter file, *not* the
`.weight`-stripped-then-reappended assumption an initial reading of
`convert_lora_to_gguf.py`'s Python source suggested; that assumption produced a real
bug — `blk.0.ffn_down.weight.weight`, caught immediately by the first real end-to-end
run, see `src/lora.rs`'s git history), not a hand-rolled format — this project's own
GGUF parser (`src/gguf.rs`) already reads a LoRA adapter file with zero changes, since
it's still just a GGUF container with a different tensor/metadata set. `scale = alpha /
rank` (rank read from `lora_a`'s/`lora_b`'s own shapes, never a separate metadata key),
matching llama.cpp's own formula exactly — confirmed independently by
`llama-export-lora`'s `calculated_scale` log line on the real verification run below,
not just by reading the source.

**Application mechanism**: `crate::lora::load` does the full `B @ A` matmul host-side
(low-rank factors are tiny — rank 16 in the real adapter tested — so this is a
one-time, load-time cost, not worth a device kernel) and hands back a full-size,
already-`scale`-multiplied delta per targeted tensor; `Model::apply_lora` uploads each
delta once and adds it into the already-GPU-resident base `Weight` with the existing
`add_k` in-place-add kernel (unchanged since Phase 2 round 2) — no new kernel, and the
forward pass itself needed zero changes. Matches this round's explicit instruction to
stay load-time-only, and this project's standing rule against reintroducing per-call
weight upload (see `model.rs`'s `Weight` doc comment).

**Verification**: real hardware (A6000, `bkzn3giz`), real fixture — no synthetic
stand-in needed for the dense case. Downloaded the real public
`premjatin/qwen-linear-algebra-coder` PEFT adapter (rank 16, alpha 32, targets all
7 `q/k/v/o/gate/up/down_proj` modules) for `Qwen/Qwen3-1.7B`, converted both to GGUF
with llama.cpp's own `convert_hf_to_gguf.py`/`convert_lora_to_gguf.py`. Cross-checked
three independent ways against a real llama.cpp build: (1) tensor count —
`llama-export-lora`'s merge log reports `merged 196 tensors with lora adapters`,
exactly `28 layers × 7 targeted modules`, matching this project's own
`tensors_applied=196`; (2) scale — `llama-export-lora`'s `calculated_scale=2.000000`
log line matches `alpha/rank = 32/16 = 2.0` independently; (3) output token —
`llama-simple` (raw completion, no chat template — see STATUS.md's known-debt entry on
`llama-cli`'s template-always-on behavior) on the LoRA-merged GGUF completes "The
capital of France is" with " Paris", matching this project's own `--lora`-applied
output exactly (`token_id=12095, token_text=" Paris"`). Accept/reject paths (MoE
attention accepted, MoE FFN-experts rejected, hybrid Gated-Attention accepted, hybrid
Gated-DeltaNet FFN accepted, hybrid Gated-DeltaNet `attn_qkv` rejected, MLA rejected
outright, shape-mismatch rejected) all verified against synthetic hand-built adapter
GGUFs (`gguf.GGUFWriter`, since no real LoRA adapter targeting a real MoE/hybrid
checkpoint's exact modules was found) targeting the existing local
`Tiny-Moe.Q4_K_M.gguf`/`Qwen3.5-0.8B-Q4_K_M.gguf`/`deepseek-tiny-mla.gguf` fixtures —
each produced exactly the expected accept-and-run or the expected clear error text.

**How to apply**: a future round wanting MoE per-expert LoRA or Gated-DeltaNet-mixer
LoRA needs new math (per-expert delta selection mirroring `gemv_expert`'s slicing, or
the mixer's own non-Linear parameterization), not just widening
`find_lora_target_mut`'s match arms — treat that as a new round, not a follow-on patch.
`--lora-scale` (a CLI-level multiplier on top of `alpha/rank`, which llama.cpp's own
`--lora-scaled` flag supports) was deliberately left out of round 1 as unrequested
scope; add it as a plain `f32` multiplier into `LoraTarget::delta`'s scale computation
if a future need comes up.

## Phase 4 (Embeddability) round 2 scope: C-FFI is load/generate/free only, cbindgen, reused instance

**Decision**: the Rust C-FFI surface (`src/ffi.rs`) exposes exactly three operations —
`coldstart_load` (GGUF path + optional LoRA adapter path), `coldstart_generate` (prompt
in, token ids + text out), `coldstart_free` — plus `coldstart_last_error` for the
error-string convention and `coldstart_free_generate_result` for the generate call's
output buffers. Phase 3's `--export-kv`/`--import-kv` state I/O is **not** exposed
through this FFI round. The header is generated with `cbindgen` from `src/ffi.rs`
(config in `cbindgen.toml`) into a checked-in `include/coldstart_infer.h`, regenerated
by hand rather than wired into `build.rs`. The already-running `bkzn3giz` A6000
instance (confirmed `RUNNING` via `tnr status --json`, not assumed) was reused for
real-hardware verification.

**Why**: confirmed with the user before starting (three explicit questions, matching
this project's practice: API surface, header-generation approach, GPU instance).
Load/generate/free-only matches this project's narrow-first precedent one more time —
`--lora`'s adapter path was folded into `coldstart_load` as an optional parameter
rather than given its own FFI call, since round 1 already made it load-time-only (no
separate "apply LoRA" step exists to expose); state I/O was left out because no
embedding host had asked for it yet and adding it means designing a buffer-ownership
convention across the FFI boundary for KV blobs, which is new scope beyond "wrap the
existing load/generate calls in a C-safe shell". `cbindgen` was chosen over a
hand-written header because it's the standard convention for a Rust crate exposing a C
ABI and keeps the header in sync with `src/ffi.rs` automatically as the surface
evolves, rather than risking hand-transcription drift (the kind of raw-source-vs.-
paraphrase mismatch that already caused a real bug in Phase 4 round 1's LoRA parsing,
see above) between the two.

**Error-boundary mechanism**: every `extern "C"` function wraps its body in
`std::panic::catch_unwind`, converting both a caught panic and this crate's existing
`Result<_, String>` convention (`model.rs`/`lora.rs`, unchanged) into the same
thread-local last-error string read via `coldstart_last_error`. This was necessary,
not optional caution: unwinding a Rust panic across an `extern "C"` boundary is
undefined behavior in the C caller, and this crate's existing code already reaches for
`.expect()`/panics in a few places (e.g. the `env!()` kernel-path macros, GGUF parsing
edge cases) that a naive `extern "C"` wrapper without `catch_unwind` would let escape
directly into the host process's control flow.

**Compiles as**: `Cargo.toml`'s `[lib]` section gained `crate-type = ["rlib", "cdylib",
"staticlib"]` — `rlib` had to stay in the list (not just be replaced) because Cargo
only auto-links a package's own lib target into its `src/bin/*.rs` targets when a
Rust-linkable crate-type (`lib`/`rlib`/`dylib`) is present; dropping it to `["cdylib",
"staticlib"]` alone would have broken `qwen3_coldstart`/`smoke_coldstart`. No
`build.rs` changes were needed — the AOT kernel-compilation pipeline governs `.cu` →
PTX/cubin, entirely orthogonal to which Rust crate-types `rustc` emits from the
already-built kernels.

**Verification**: real hardware (A6000, `bkzn3giz`), a real C program
(`ffi-test/smoke_test.c`, plain `gcc`, not a Rust test) linked against the built
`libcoldstart_infer.so`, exercising the full `load` → `generate` → `free` surface and
cross-checked byte-exact against `qwen3_coldstart` on the same GGUF+prompt+
`max_new_tokens` for two architectures: dense `Qwen3-0.6B-Q4_K_M.gguf`
(`token_ids=[13,576,3974,13876,38835]`, identical decoded text) and Qwen3.5 hybrid
`Qwen3.5-0.8B-Q4_K_M.gguf` (`token_ids=[0,353,1044]`, identical decoded text) — not
just "it compiles and links". The error path (a nonexistent GGUF path) was also
verified: `coldstart_load` returns `NULL`, no crash, and `coldstart_last_error()`
names the missing file. `staticlib` linking was attempted too (not part of the
confirmed scope, but cheap to try since the crate-type was already added) and *appeared*
to have a real, unresolved problem in this same session — flagged as known debt at the
time — but a follow-up debugging pass the same day found it doesn't reproduce (see the
round-2-follow-up entry below): both `cdylib` and `staticlib` are verified working.

**How to apply**: a future round wanting Phase 3 state I/O (`--export-kv`/
`--import-kv`) through the FFI needs to design an explicit buffer-ownership convention
for KV-cache blobs crossing the C boundary (who allocates, who frees, whether it's a
raw byte buffer or a path to a file this crate itself writes) — treat that as new
scope requiring its own confirm-before-starting conversation, not a small addition to
`coldstart_load`/`coldstart_generate`.

## Phase 4 round 2 follow-up: the `staticlib` "hang" was GPU-capacity contention, not a linking bug

**Decision**: no code change. The `staticlib` linking issue flagged as known debt right
after Phase 4 round 2 (`-Wl,--allow-multiple-definition` needed to link,
then a runtime hang) is retracted — a dedicated debugging session (same day, same
`bkzn3giz` instance) could not reproduce either symptom.

**What was actually found**: `gcc -I include -o smoke_test_static ffi-test/smoke_test.c
-L target/release -l:libcoldstart_infer.a -ldl -lpthread -lm` — the exact same command
as before, *minus* `-Wl,--allow-multiple-definition` — linked with zero duplicate-symbol
warnings. The resulting binary ran successfully against both the dense
(`Qwen3-0.6B-Q4_K_M.gguf`, `token_ids=[13,576,3974,13876,38835]`) and hybrid
(`Qwen3.5-0.8B-Q4_K_M.gguf`, `token_ids=[0,353,1044]`) fixtures, byte-exact against the
`cdylib`/CLI runs from the original round, completing in single-digit seconds each —
not a hang.

**Why the original session saw what it saw**: almost certainly this project's
already-documented ThunderCompute GPU-capacity-contention pattern (see
STATUS.md/memory: SSH commands touching the GPU driver can queue for tens of minutes
under `"all requested GPU capacity is currently busy"` platform-side scheduling, then
clear on their own) — the original attempt was killed after only ~90 seconds on the
assumption it was permanently stuck, which was too short a wait to distinguish
"queued" from "actually hung." The `--allow-multiple-definition` flag was added
defensively in the same original session without first testing whether the link
actually required it — it didn't.

**How to apply**: don't assume a single failed run on a shared/ephemeral GPU instance
proves a code-level bug, especially one that manifests as "stuck, not crashed" — that
signature matches GPU-capacity contention as easily as a real hang. Wait longer (or
check `nvidia-smi`/process state for whether it's still making progress) before
concluding it's broken, particularly for anything touching `CudaDevice::new` or other
first GPU-driver contact in a fresh process.

## Dockerfile real `docker build`/`docker run` verification: found and fixed a real portable-PTX build bug

**Decision**: fixed `build.rs`'s `COLDSTART_CUDA_ARCH` handling (one line) rather than
working around it in the Dockerfile.

**What was found**: the first real `docker build` pass on a genuine (non-nested-container)
Docker host — a Windows machine running Docker Desktop, the first environment available
across this project's sessions that could actually run `docker build` at all — reproduced
a real, previously-undetected bug: the default (no `--build-arg COLDSTART_CUDA_ARCH=...`)
portable-PTX build mode failed every time with `nvcc fatal: Value '' is not defined for
option 'gpu-architecture'`. Root cause: the Dockerfile's `ARG COLDSTART_CUDA_ARCH=""`
exposes that variable to the `RUN` instruction's shell as a *set-but-empty* env var even
when no `--build-arg` override is passed (that's exactly why the Dockerfile's own
`if [ -n "$COLDSTART_CUDA_ARCH" ]` shell check works as intended, correctly taking the
plain-`cargo build` else-branch) — but `build.rs`'s `env::var("COLDSTART_CUDA_ARCH").ok()`
returns `Some("")` for a set-but-empty var, not `None`, so it silently took the cubin
branch anyway with an empty `-arch=` flag. This never reproduces on a bare host shell
(where an unset var is truly absent, not present-and-empty), which is exactly why every
prior real-hardware session's plain `cargo build --release` runs never hit it — only
Docker's `ARG` mechanism exposes the gap. Fix: `.filter(|s| !s.is_empty())` after the
`.ok()`.

**Why fixed immediately rather than just documented**: one-line, unambiguous root cause,
directly blocking the one release-path check STATUS.md had explicitly flagged as
un-verified; leaving a known-broken default build mode in the shipped Dockerfile would
have defeated the point of finally getting a real Docker host to test on.

**Full verification, both build modes, real host**: `docker build --build-arg
COLDSTART_CUDA_ARCH=sm_86 -t coldstart-infer .` and `docker build -t coldstart-infer .`
(portable PTX) both pass cleanly post-fix; re-ran the `sm_86` build afterward too to
confirm the fix doesn't regress the cubin path (it doesn't — `arch` is `Some("sm_86")`
either way, unaffected by the new filter). `docker run` (no `--gpus`) against the
`ptx`-tagged image with `test-data/deepseek-tiny-mla.gguf` bind-mounted confirmed the
binary itself is fully correct inside the container: it opened the GGUF, parsed it, and
progressed all the way to `cudarc`'s dynamic `libcuda`/`nvcuda` load before failing —
exactly the expected failure point with no GPU/driver present, not a container defect.

**What could *not* be verified**: `docker run --rm --gpus all` itself, i.e. real GPU
passthrough + actual inference output (`COLDSTART_QWEN3_OK ...`) from inside the
container. This Docker host has no NVIDIA GPU at all (confirmed via `Get-CimInstance
Win32_VideoController` → AMD Radeon only) — `docker run --gpus all` fails immediately
with `nvidia-container-cli: initialization error: WSL environment detected but no
adapters were found`, a hardware-absence error, not a configuration problem this session
could fix. Interesting incidental finding: Docker Desktop's WSL2 backend does already
carry a working `nvidia-container-cli`/toolkit wiring (the error is a clean, specific
"no adapter" message, not "toolkit not installed") — so a Windows machine with Docker
Desktop *and* a real NVIDIA GPU would likely need no extra host setup for `--gpus all`
to work, unlike a from-scratch Linux Docker host.

**How to apply**: the real GPU-passthrough + inference-output check
(`COLDSTART_QWEN3_OK process_start_to_first_token_ms=... token_text=...` from inside a
container) still needs a Docker host with (a) genuine VM-level virtualization, not a
nested container, and (b) an actual NVIDIA GPU + driver — e.g. a Windows or Linux
machine with Docker Desktop/`nvidia-container-toolkit` and a real NVIDIA card, not
another ThunderCompute-style rented GPU-cloud instance (those have all reproducibly
been nested containers so far, per the original known-debt entry this session
partially closed). See STATUS.md's updated Dockerfile entry for the precise remaining
scope.
