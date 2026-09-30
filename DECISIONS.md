# DECISIONS.md

A log of the project's non-obvious technical and scope decisions, and why they were
made. For current state see `STATUS.md`; for full narrative/benchmark detail see
`HISTORY.md`.

## Phase 3 (State I/O) round 1 scope: dense/MoE only, export/import-the-buffers only

**Decision**: `--export-kv`/`--import-kv` round 1 supports only the dense/MoE Qwen3
forward path's `k_cache`/`v_cache` pair (not the Qwen3.5 hybrid's per-layer
`Attn`/`Gdn` split, not MLA's single compressed cache), and does not wire an imported
cache back into a forward pass — `--import-kv` proves the file round-trips
byte-identical through a device upload/download and stops there, rather than resuming
generation from it.

**Why**: dense-only matches the established "narrow first" precedent from every prior MVP
step (dense before MoE before hybrid before MLA). The export/import-only cut came from what
investigating `model.rs` actually found, not just caution: `forward_prompt` (all three
architecture paths) always starts at position 0, and there is no per-token generation
loop anywhere in `model.rs` or the `generate` subcommand — each run does one full prompt pass
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

**Why**: of the three possible scopes (dense/MoE only, +hybrid, or +hybrid+MLA all in
one round), dense/MoE+hybrid matches this
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

**Why**: the fixture choice was the open question here (a fresh A6000 instance was
rented, since none was running at the time). The synthetic fixture was chosen over
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

**Why**: runtime NVRTC compilation carries a real, measured multi-second JIT tax on
first kernel use — fatal for a cold-start-latency target. llama.cpp avoids this
because `nvcc` compiles its kernels at build time; Reflex adopts the same approach as
its core bet.

**Open question, not yet resolved**: whether to default to portable PTX (small
driver-side JIT-to-SASS cost) or `REFLEX_CUDA_ARCH=sm_XX`-targeted cubins (zero JIT,
but needs a matching cubin per deployment target) is unverified on real model kernels —
`reflex smoke` found the two statistically indistinguishable (473–637ms vs.
480–617ms), but only for a trivial kernel where CUDA context init dominates. Worth
re-measuring against real model-sized kernels.

## No serving-platform features, ever (permanent non-goal)

**Decision**: `batch_size` is always 1. No internal request queue/scheduler, no
continuous batching, no multi-tenant LoRA router, no internal NVMe/S3 KV-cache manager,
no concurrent HTTP/gRPC server, no autoscaling logic. If a warm-context mode is ever
built, it accepts one job at a time, strictly sequentially — never a thread pool.
Multi-tenancy and persistent state are the *host orchestrator's* job, not this engine's.

**Why**: a broad serving feature set re-enters the exact warm-throughput race against
vLLM/SGLang that's an unwinnable kernel-optimization race against projects with a
multi-year head start. Reflex
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

**Why**: dense Qwen3 is the best-understood, most well-documented architecture to build
against first, proving the AOT-compilation + cold-start-benchmark harness works at all
before spending effort on anything novel. MLA is ordered last on purpose — it's a genuinely different
caching strategy (compressed latent KV), not an incremental extension of GQA, so it
carries the most implementation risk and should be tackled once the rest of the harness
is trustworthy. (Read llama.cpp PR #11446 before attempting it.)

## Extending architecture coverage: explicit whitelist + per-arch RoPE type (Llama/Mistral, then `qwen35moe`)

**Decision**: after the four-MVP-family set (dense Qwen3 → Qwen3-MoE → Qwen3.5 hybrid →
MLA), the next architecture additions are **Llama/Mistral dense GQA first**, then
**`qwen35moe`**, and both are done under two rules:

1. **The dense-path architecture gate stays an explicit whitelist**, never "anything
   that structurally parses". `parse_model_config` accepts dense `qwen3`, `llama`,
   `mistral`, `mixtral`, plus (as before) any file reporting a nonzero
   `<arch>.expert_count`; every other architecture string is still a clear hard error.
   Silently running an unrecognized family through the Qwen3-shaped dense path would
   turn a metadata mismatch into plausible-looking-but-wrong output.
2. **RoPE convention is a per-architecture property derived from one auditable
   mapping**, `rope_type_for(architecture)`, confirmed against llama.cpp's
   `llama_model_rope_type` — `Neox` (half-split pairs `(i, i + rotary_dim/2)`,
   `rope_kernel`) for Qwen3/Qwen3.5, `Norm` (consecutive pairs `(2i, 2i+1)`,
   `rope_norm_kernel`, already shipped for MLA) for Llama/Mistral/Mixtral. RoPE
   scaling types other than `none` (e.g. Llama-3.1's `linear`/`yarn` long-context
   variants) are rejected with a clear error rather than run with wrong frequencies.
   `rope.dimension_count` is read explicitly (full rotation for these families), not
   assumed equal to `head_dim`.

**Why**: Llama/Mistral are the cheapest real addition because everything *except* the
RoPE convention already exists generically in the dense/MoE path — GQA
(`attention.head_count_kv`), RMSNorm, SwiGLU `ffn_gate`/`ffn_up`/`ffn_down`, the
SentencePiece tokenizer (`tokenizer.ggml.model == "llama"`), tied-vs-untied
`output.weight`, and even optional QK-Norm. The MLA work had already proven (the hard
way, via a silently-corrupted-output bug) that RoPE convention is *not*
architecture-independent — see this file's MLA RoPE entry and the "don't assume RoPE
convention is architecture-independent" rule stated there; this decision turns that
lesson into a single enforced mapping. The whitelist rule exists for the same reason
the family ordering does: a wrong metadata assumption elsewhere in the pipeline should
surface as a rejection, not as wrong tokens.

`qwen35moe` is deliberately *second*, not first, even though both halves exist in
isolation today (the `qwen35` hybrid mixer, and routed-MoE + shared-expert FFN from
`qwen3moe`/MLA): it combines two mechanisms in one layer, and no small real
`qwen35moe` checkpoint exists to verify against, so its blocker is a **synthetic
fixture** (the same hand-built-HF-checkpoint-through-`convert_hf_to_gguf.py` pattern
as `tiny-qwen3moe.gguf`/`deepseek-tiny-mla.gguf`), not new kernel math.

**How to apply**: breadth is llama.cpp's axis, not this engine's (see the Non-goals
framing) — add a family only when a concrete target model pulls it, and budget each
addition as *fixture + independent-implementation verification* first, forward-pass
code second. Every addition must clear the bar in the "Cross-check new architecture
work against an independent ground truth" entry above: real generated tokens compared
against a fresh llama.cpp build (or a host reference), never "it runs and looks
plausible". The Llama/Mistral path was verified this way before being declared done
(HISTORY.md's "Llama/Mistral dense-GQA support (2026-09-30)" entry): `reflex
check`/`generate` matched a fresh CUDA `llama-simple` build byte-exact on
TinyLlama-1.1B-Chat, and a secondary finding — real Mistral-7B GGUFs report
`general.architecture = "llama"`, because llama.cpp has no bare `mistral` arch string
(only the separate `mistral3`/`mistral4`) — means the `llama` arm carries Mistral-7B; the
`mistral`/`mixtral` arms are defensive aliases. Running a 7B model end-to-end additionally
needs >15GB (every weight is `f32`-resident), so the byte-exact check was done on the
smaller `llama`-arch model; the conventions that differ from Qwen3 (Norm RoPE, GQA,
SwiGLU) are scale-invariant and were exercised there.

## Naive (ungrouped) per-expert MoE dispatch, not grouped/batched

**Decision**: Qwen3-MoE's FFN dispatches one `gemv_expert` call per selected expert
(sliced out of a `[in_features, out_features, expert_count]` per-expert-stacked
tensor), rather than grouping tokens by expert or batching expert computation.

**Why**: naive per-expert dispatch is the correct starting point before optimizing.
Since `batch_size` is always 1
here (see the serving non-goal above), there's only one token being routed per forward
call anyway — the grouped-dispatch optimization that matters for batched serving doesn't
apply the same way to this project's actual workload shape.

## Weights are GPU-resident once, not re-uploaded per call

**Decision**: `Weight` (in `model.rs`) holds a `CudaSlice<f32>` uploaded to the GPU once
at load time (immediately after dequantizing, with the host scratch buffer dropped).
`gemv`/`gemv_expert`/`rmsnorm` take that device buffer directly; MoE's per-expert
dispatch slices it with a zero-copy `CudaSlice::slice` view.

**Why**: the first real cold-start-vs-llama.cpp benchmark (2026-09-17) found
Reflex ~4.3x slower, 4x the peak RSS, and ~11x the system CPU time of
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
[[Reflex-benchmark-result]] memory / README's "Phase 2, round 2" section).
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

## MLA extended to real DeepSeek-V2-Lite: MoE + shared experts + YaRN

**Decision**: after the synthetic-fixture-only MLA work above landed, it was
extended to the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint on a
rented 80GB A100 (the VRAM estimate from the entry above held: fits 80GB with
headroom, doesn't fit the A6000's 48GB). This added routed-MoE + always-on
shared-expert FFN (`MlaFfn::Moe`) and YaRN RoPE scaling (`MlaYarnConfig`,
`rope_norm_yarn_kernel`) to the previously dense-only, no-YaRN implementation.

**Why extend immediately rather than treat it as separate future work**: full scope
(MoE+shared-experts+YaRN together) was the right call once it was clear YaRN was a
real, separate chunk of work beyond the originally-scoped MoE addition — splitting the
work artificially across rounds would have meant re-verifying the same checkpoint
twice for no benefit.

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
unpacking).

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

**Why**: scope landed on "dense/MoE +
hybrid" rather than dense-only, matching the project's narrow-first precedent one step
wider than MVP-step ordering alone would suggest, but MLA was excluded per the
project's own framing (`README.md`'s Non-goals: load-time-only adapter application, no
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
`reflex_load` (GGUF path + optional LoRA adapter path), `reflex_generate` (prompt
in, token ids + text out), `reflex_free` — plus `reflex_last_error` for the
error-string convention and `reflex_free_generate_result` for the generate call's
output buffers. Phase 3's `--export-kv`/`--import-kv` state I/O is **not** exposed
through this FFI round. The header is generated with `cbindgen` from `src/ffi.rs`
(config in `cbindgen.toml`) into a checked-in `include/reflex_engine.h`, regenerated
by hand rather than wired into `build.rs`. The already-running A6000
instance (confirmed `RUNNING` via `tnr status --json`, not assumed) was reused for
real-hardware verification.

**Why**: load/generate/free-only matches this project's narrow-first precedent one more time —
`--lora`'s adapter path was folded into `reflex_load` as an optional parameter
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
thread-local last-error string read via `reflex_last_error`. This was necessary,
not optional caution: unwinding a Rust panic across an `extern "C"` boundary is
undefined behavior in the C caller, and this crate's existing code already reaches for
`.expect()`/panics in a few places (e.g. the `env!()` kernel-path macros, GGUF parsing
edge cases) that a naive `extern "C"` wrapper without `catch_unwind` would let escape
directly into the host process's control flow.

**Compiles as**: `Cargo.toml`'s `[lib]` section gained `crate-type = ["rlib", "cdylib",
"staticlib"]` — `rlib` had to stay in the list (not just be replaced) because Cargo
only auto-links a package's own lib target into its `src/bin/*.rs` targets when a
Rust-linkable crate-type (`lib`/`rlib`/`dylib`) is present; dropping it to `["cdylib",
"staticlib"]` alone would have broken `reflex generate`/`reflex smoke`. No
`build.rs` changes were needed — the AOT kernel-compilation pipeline governs `.cu` →
PTX/cubin, entirely orthogonal to which Rust crate-types `rustc` emits from the
already-built kernels.

**Verification**: real hardware (A6000, `bkzn3giz`), a real C program
(`ffi-test/smoke_test.c`, plain `gcc`, not a Rust test) linked against the built
`libreflex_engine.so`, exercising the full `load` → `generate` → `free` surface and
cross-checked byte-exact against `reflex generate` on the same GGUF+prompt+
`max_new_tokens` for two architectures: dense `Qwen3-0.6B-Q4_K_M.gguf`
(`token_ids=[13,576,3974,13876,38835]`, identical decoded text) and Qwen3.5 hybrid
`Qwen3.5-0.8B-Q4_K_M.gguf` (`token_ids=[0,353,1044]`, identical decoded text) — not
just "it compiles and links". The error path (a nonexistent GGUF path) was also
verified: `reflex_load` returns `NULL`, no crash, and `reflex_last_error()`
names the missing file. `staticlib` linking was attempted too (not part of the
originally-scoped work, but cheap to try since the crate-type was already added) and
*appeared* to have a real, unresolved problem — flagged as known debt at the
time — but a follow-up debugging pass found it doesn't reproduce (see the
round-2-follow-up entry below): both `cdylib` and `staticlib` are verified working.

**How to apply**: a future round wanting Phase 3 state I/O (`--export-kv`/
`--import-kv`) through the FFI needs to design an explicit buffer-ownership convention
for KV-cache blobs crossing the C boundary (who allocates, who frees, whether it's a
raw byte buffer or a path to a file this crate itself writes) — treat that as new
scope requiring its own confirm-before-starting conversation, not a small addition to
`reflex_load`/`reflex_generate`.

## Phase 4 round 2 follow-up: the `staticlib` "hang" was GPU-capacity contention, not a linking bug

**Decision**: no code change. The `staticlib` linking issue flagged as known debt right
after Phase 4 round 2 (`-Wl,--allow-multiple-definition` needed to link,
then a runtime hang) is retracted — a dedicated debugging pass on the same
instance could not reproduce either symptom.

**What was actually found**: `gcc -I include -o smoke_test_static ffi-test/smoke_test.c
-L target/release -l:libreflex_engine.a -ldl -lpthread -lm` — the exact same command
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

**Decision**: fixed `build.rs`'s `REFLEX_CUDA_ARCH` handling (one line) rather than
working around it in the Dockerfile.

**What was found**: the first real `docker build` pass on a genuine (non-nested-container)
Docker host — a Windows machine running Docker Desktop, the first environment available
across this project's sessions that could actually run `docker build` at all — reproduced
a real, previously-undetected bug: the default (no `--build-arg REFLEX_CUDA_ARCH=...`)
portable-PTX build mode failed every time with `nvcc fatal: Value '' is not defined for
option 'gpu-architecture'`. Root cause: the Dockerfile's `ARG REFLEX_CUDA_ARCH=""`
exposes that variable to the `RUN` instruction's shell as a *set-but-empty* env var even
when no `--build-arg` override is passed (that's exactly why the Dockerfile's own
`if [ -n "$REFLEX_CUDA_ARCH" ]` shell check works as intended, correctly taking the
plain-`cargo build` else-branch) — but `build.rs`'s `env::var("REFLEX_CUDA_ARCH").ok()`
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
REFLEX_CUDA_ARCH=sm_86 -t Reflex .` and `docker build -t Reflex .`
(portable PTX) both pass cleanly post-fix; re-ran the `sm_86` build afterward too to
confirm the fix doesn't regress the cubin path (it doesn't — `arch` is `Some("sm_86")`
either way, unaffected by the new filter). `docker run` (no `--gpus`) against the
`ptx`-tagged image with `test-data/deepseek-tiny-mla.gguf` bind-mounted confirmed the
binary itself is fully correct inside the container: it opened the GGUF, parsed it, and
progressed all the way to `cudarc`'s dynamic `libcuda`/`nvcuda` load before failing —
exactly the expected failure point with no GPU/driver present, not a container defect.

**What could *not* be verified**: `docker run --rm --gpus all` itself, i.e. real GPU
passthrough + actual inference output (`REFLEX_GENERATE_OK ...`) from inside the
container. This Docker host has no NVIDIA GPU at all (confirmed via `Get-CimInstance
Win32_VideoController` → AMD Radeon only) — `docker run --gpus all` fails immediately
with `nvidia-container-cli: initialization error: WSL environment detected but no
adapters were found`, a hardware-absence error, not a configuration problem that could
be fixed here. Interesting incidental finding: Docker Desktop's WSL2 backend does already
carry a working `nvidia-container-cli`/toolkit wiring (the error is a clean, specific
"no adapter" message, not "toolkit not installed") — so a Windows machine with Docker
Desktop *and* a real NVIDIA GPU would likely need no extra host setup for `--gpus all`
to work, unlike a from-scratch Linux Docker host.

**How to apply**: the real GPU-passthrough + inference-output check
(`REFLEX_GENERATE_OK process_start_to_first_token_ms=... token_text=...` from inside a
container) still needs a Docker host with (a) genuine VM-level virtualization, not a
nested container, and (b) an actual NVIDIA GPU + driver — e.g. a Windows or Linux
machine with Docker Desktop/`nvidia-container-toolkit` and a real NVIDIA card, not
another ThunderCompute-style rented GPU-cloud instance (those have all reproducibly
been nested containers so far, per the original known-debt entry this partially
closed). See STATUS.md's updated Dockerfile entry for the precise remaining
scope.

## Benchmark expansion: vLLM added, Ollama/TGI/TensorRT-LLM/other-cloud-vendors deferred

**Decision**: extend the cold-start benchmark suite with one new same-A6000 self-hosted
comparison this round — **vLLM** — reusing the existing external-wall-clock methodology
verbatim (`scripts/bench_cold_common.sh`, `scripts/bench_cold_vllm.sh`). Ollama, TGI,
TensorRT-LLM/Triton, and serverless cloud vendors (Modal, Baseten, RunPod Serverless,
Beam.cloud, fal.ai, Replicate) are explicitly deferred, not silently dropped.

**Why vLLM specifically**: llama.cpp — this project's only existing comparison — is,
like Reflex, AOT-compiled via `nvcc`; it never JIT-compiles CUDA kernels. That
means the llama.cpp comparison never actually tested this project's core technical bet
(AOT-compiled kernels vs. a real JIT/warmup tax at cold start — see CLAUDE.md). vLLM's
CUDA graph capture and (historically) `torch.compile`/JIT-driven kernel compilation is a
real, well-documented cold-start cost and a much better foil for that specific claim.

**Why the rest are deferred**: Ollama wraps llama.cpp's own ggml runtime, so it doesn't
add a new data point on the AOT-vs-JIT question — it would test daemon/model-pull
packaging overhead instead, a different claim, worth a future round on its own. TGI and
TensorRT-LLM/Triton are meaningfully higher setup cost on a single shared, ephemeral
ThunderCompute instance (heavier Python/Docker stack; TensorRT-LLM doesn't accept GGUF
at all and needs a separate engine-build step per GPU arch) — not attempted without a
clear reason to burn shared-instance hours on them. Other cloud/serverless vendors
answer a different question entirely ("pay-per-request serverless cold boot" vs.
"rent-your-own-GPU true process cold start") and introduce new vendor accounts and real
recurring billing — gated on explicit future budget approval, not bundled into this
round.

**How to apply**: if vLLM's GGUF loader can't load the target Qwen3 fixture, document
any fallback weight format as an explicit methodology deviation in HISTORY.md — same
standard as the existing chat-template-parity caveat — rather than silently substituting
it. When Ollama/TGI/TensorRT-LLM/other-cloud-vendor comparisons are eventually pursued,
each needs its own scoped decision entry here, not a retroactive expansion of this one.

**Update, same round**: the installed vLLM (`0.30.0`) turned out to have no `gguf`
entry in its quantization method registry at all (confirmed via
`vllm.model_executor.layers.quantization.QUANTIZATION_METHODS`) — not a config
issue, GGUF loading isn't present in this version. Per the "how to apply" note
above, this was disclosed rather than worked around silently: vLLM was pointed at
the original `Qwen/Qwen3-0.6B` HF safetensors checkpoint instead (confirmed with the
user first), and HISTORY.md's benchmark writeup states plainly that this tests the
same cold-start mechanism, not byte-identical weights/precision, across engines.

## TypeSafe Jev comparison framing: latency-only citation, not a live benchmark

**Decision**: Reflex's comparison against TypeSafe AI's "Jev" model
(`scripts/bench_cold_system1_vs_jev.sh`) cites Jev's own **published** latency figures
(10-15ms compute / 70-500ms end-to-end via TypeSafe's managed cloud API) next to a
locally **measured** cold-start latency of Reflex's System1 candidate-scoring
path (`Model::system1_evaluate` / `reflex system1`, `process_start_to_result_ms`).
This is deliberately **not** a live call to Jev's API, and deliberately **not** a
decision-quality/calibration comparison — latency only, and labeled as an illustrative
citation, never merged into the same results table as the vLLM/llama.cpp engine-vs-engine
comparisons.

**Why System1 (not `reflex generate`) is the right Reflex surface for this
comparison**: Jev is not a generative LLM — per TypeSafe's own description, it takes a
state and one or more questions and returns typed answers with probabilities, never a
sentence. A time-to-first-token comparison against Reflex's normal
autoregressive generation path would be apples-to-oranges. `Model::system1_evaluate`
(single-pass, non-autoregressive candidate scoring — see README's "System1" section) is
the same task shape: prompt + fixed candidates in, scored typed results + probabilities
out, no decode loop.

**Why latency-only, citation-based, not a live API call or quality claim**: (1) Jev's
number necessarily includes a network round-trip to TypeSafe's managed cloud, while
Reflex's is a pure local process launch with zero network dependency at all —
different deployment models, not just different numbers, so a literal head-to-head table
would misrepresent both sides; (2) Jev is presumably purpose-trained for calibrated
decision-making, while System1 is a generic instruction-tuned Qwen3 GGUF with an
efficient scoring head — this project has no basis to claim decision-quality parity, only
to measure its own latency on a comparable task shape; (3) calling Jev's live API adds a
new vendor dependency, rate-limit exposure, and a benchmark-publication ToS question that
citing already-published numbers avoids entirely.

**How to apply**: if a genuine head-to-head is ever wanted, the only fair path is
TypeSafe's enterprise on-prem tier (Docker/Kubernetes on local NVIDIA GPUs, per their own
published materials) deployed on the same A6000 instance — pursue that as a separate,
explicitly-scoped decision if/when access and pricing are confirmed, rather than
retroactively upgrading this citation into a benchmark claim.

**Addendum, same round**: the cold-start citation above reported Reflex
losing by ~10-60x, which is honest but incomplete — Jev's 10-15ms figure is itself a
*warm, compute-only* number (an always-resident service pays no cold load), so citing
it only against Reflex's *cold* number answers a different question than the
one Jev's figure is actually about. Added a second, separate citation using the
existing `reflex bench --candidate` warm-latency microbenchmark (model loaded
once, isolates just the gather-GEMV scoring step): at the shortest prompt-length
bucket (29 tokens), Reflex's warm System1 p50 is 19.4ms, within ~1.3-2x of
Jev's 10-15ms — competitive, not a loss. Both citations are published side by side in
HISTORY.md, not just the favorable one — cold-start-to-decision and warm-per-decision-
scoring answer genuinely different questions, and reporting only one would be exactly
the kind of cherry-picking this project's methodology exists to avoid.

**Second addendum**: both citations above were measured on a rented ThunderCompute
A6000 — the same shared/virtualized environment behind the `fast_exit` regression
below, raising the question of whether the conclusion was an artifact of that
specific host. Re-ran both on a real AWS EC2 `g4dn.xlarge` (Tesla T4) and got the
same qualitative result on independent hardware: cold-start-to-decision still loses
(though the gap narrows to ~2.5-18x on real dedicated hardware, vs. ~10-60x on
ThunderCompute), warm scoring stays competitive (~1.4-2.1x vs. ~1.3-2x). See
HISTORY.md's "TypeSafe Jev re-verification on real AWS EC2 T4" entry for the full
numbers and the open question of how much of the cold-start gap difference is
ThunderCompute's GPU-virtualization proxy taxing CUDA init specifically.

**Third addendum: reason (3) above (live-call risk) revisited, one axis upgraded from
citation to measurement.** Jev turned out to be reachable through OpenRouter
(`~typesafe/jev-1.13`, a standard public API gateway this project already references
elsewhere for the `sidecar/openai-adapter`), at $0.042/M input tokens with free
output — the vendor-dependency/rate-limit/ToS concern reason (3) raised assumed a
bespoke integration against TypeSafe's own infra; a handful of calls through an
existing public gateway is a materially smaller version of that risk, so it was
revisited and a real measurement was taken (cost: ~$0.0002 total for every call this
session). This upgrades the **cold-start-to-decision** axis from citation to
independent measurement (Jev side: 307.8-569.6ms, fresh HTTPS connection per call,
`n=6`). It does **not** upgrade the **warm compute-only** axis — no external caller,
including this measurement, can isolate TypeSafe's internal compute time from outside
their infra, so 10-15ms remains a citation. A third, new data point was added instead:
Jev's warm *round-trip* latency over a persistent connection (120.6-190ms, p50
141ms) — explicitly not compared 1:1 against Reflex's 19.4-20.9ms compute-only figure,
since Jev's number includes real network RTT that Reflex's local in-process call never
pays. See HISTORY.md's "Jev measured directly via OpenRouter" entry for full numbers
and the network-path caveat (measured from a dev machine, not the AWS rig Reflex's
numbers came from).

**Fourth addendum: closed the "not apples-to-apples" gap on the warm-round-trip
comparison itself, same day.** Deployed `sidecar/openai-adapter` (the existing
escape-hatch pattern, not a new endpoint written for this) on a throwaway
`g4dn.xlarge`, measured the real network floor to it from this dev machine via its
`/healthz` endpoint using the identical warm/persistent-connection method as the
Jev measurement, and added that floor to Reflex's already-measured 20.9ms compute
figure — an explicit construction, not a single live decision call, but it puts
real network cost on both sides of the comparison for the first time. Result: the
warm gap shrank from ~7x (20.9ms vs. 141ms, network-less vs. network-inclusive) to
~10% at p50 (127.0ms vs. 141.0ms), and Reflex's max was actually *worse* than
Jev's (224.9ms vs. 190.0ms) — network jitter on this measurement's path to
`us-east-1`, reported as observed. See HISTORY.md's "Making the Jev comparison
genuinely apples-to-apples" entry for full numbers and the residual caveats
(different physical endpoints, `/healthz` vs. a real decision payload not verified
identical in network-floor terms).

## Fast-exit after printing the benchmark result (`reflex_engine::fast_exit`)

**Decision**: `reflex generate`/`reflex system1`/`reflex smoke` call
`reflex_engine::fast_exit(code)` (`src/lib.rs`) instead of returning from `main`
normally, immediately after printing their result. It flushes stdout/stderr, then
calls the raw `_exit` syscall via an `extern "C"` declaration — not
`std::process::exit`, which still runs libc's `atexit` chain.

**Why**: re-running the llama.cpp comparison on a fresh ThunderCompute A6000 this
round surfaced a real regression — Reflex measured ~1.4x *slower* than
llama.cpp on full external wall-clock, even though its own internal
`process_start_to_first_token_ms` metric was unchanged (~4.8-5.0s). `strace -f -T`
showed ~4.5s of staged-backoff `futex`/`poll` waits against `/tmp/.tc_hac`
(ThunderCompute's GPU-virtualization proxy) happening entirely *after* the result was
printed. First hypothesis — too many separate device allocations (~300
`CudaSlice<f32>` buffers, one per weight tensor) each paying their own teardown
round-trip — was tested via a full arena-consolidation refactor (one shared buffer
per model) and **made no measurable difference**, so it was reverted rather than kept
for no benefit (see git history for that attempt; not present in the current tree).
The actual tell: `reflex smoke`, which allocates almost nothing, showed the *same*
~5.6s of pure post-result teardown. The cost is fixed per-process, not
allocation-count-proportional — it's the CUDA driver's own `atexit`-registered
context-teardown hook handshaking with the virtualization proxy, paid by any CUDA
program on this kind of shared/virtualized instance. Confirmed the fix actually works
by testing a raw `_exit()` diagnostically first (`reflex smoke`: 6.3s → 0.56s wall
clock) before wiring it into the real binaries.

**Why this is safe**: every one of these binaries' job (compute a result, print it)
is finished by the time `fast_exit` runs. The OS reclaims the GPU context, device
memory, and file descriptors on process death regardless of whether userspace tore
them down gracefully first — `_exit` skips only the *graceful* teardown handshake,
not actual resource reclamation. Buffered-writer flushing (the one thing `_exit`
genuinely skips that matters) is done explicitly first.

**How to apply**: any future one-shot `reflex` subcommand whose job ends by printing a
result should call `fast_exit` the same way. This is deliberately not applied to the
`check`/`stdio`/`uds` subcommands or library code — panics and multi-request
servers should keep normal Rust unwind/cleanup semantics; this is specifically for
single-shot CLI binaries measuring their own process lifetime.

## Ollama benchmark: three scenarios reported separately, flakiness disclosed not averaged

**Decision**: `scripts/bench_cold_ollama.sh` measures Ollama cold-start across three
distinct scenarios (cold daemon + cold model, warm daemon + cold model, warm daemon +
warm model) rather than one number, and reports every individual run rather than
just a mean/median when the results are inconsistent.

**Why**: Ollama wraps llama.cpp's own ggml runtime (confirmed via its logs, which show
it spawning a bundled `llama-server` subprocess) — it doesn't test the AOT-vs-JIT
question `bench_cold_vllm.sh` does, it tests real daemon/packaging overhead, per the
original "Benchmark expansion" entry's rationale for deferring it. Measuring it
surfaced a genuine, reproducible finding: Ollama's bundled `llama-server` hits its own
`"GPU discovery watchdog timed out"` error intermittently on this ThunderCompute
instance — 2/7 cold-daemon runs and 3/5 warm-daemon-cold-model runs stalled to
~55-62s instead of the ~6-11s the other runs showed. This is the same class of
GPU-virtualization-layer slowdown the `fast_exit` fix above addresses, but inside
Ollama's own process, not Reflex's — nothing here to fix on this project's
side. Averaging these runs into one number would hide a real, user-relevant
reliability difference (Reflex/llama.cpp's own cold-start runs
stayed within single-digit-percent variance; Ollama's did not) — reporting every run
individually keeps that visible.

**How to apply**: if Ollama is benchmarked again on different infrastructure, check
whether the same intermittent stall reproduces before assuming either the ~6-7s or
the ~55-60s figure is "the" number — on this environment, neither alone is honest.

## Phase 4 round 1 follow-up: widen `--lora` to the Gated DeltaNet mixer's Linear tensors, verified against a real hybrid adapter

**Decision**: `find_lora_target_mut` (`src/model.rs`) now also matches
`HybridLayerWeights::GatedDeltaNet`'s `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`/
`ssm_out` fields, not just its FFN tensors. No change to `src/lora.rs` or the
application mechanism (`add_k`, unchanged) — this is purely five more match arms.

**Why**: round 1's doc comments called the whole Gated DeltaNet mixer "non-Linear" and
rejected all of it beyond FFN, based on `model.rs`'s own field-doc language, not a real
adapter's actual target list (none was found public at the time). A fresh search this
round found `Tilakoid/qwen3.5-0.8b-hoasa-lora` (targets `Qwen/Qwen3.5-0.8B` exactly),
and — critically — its `adapter_config.json`'s `target_modules` is an unsloth-style
*regex* (`(?:...)(?:qkv|proj|...|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|...)`), not
a literal list, so paraphrasing it (even via a direct fetch) couldn't establish what it
actually touched. Read the real safetensors header instead (`curl -r` for the 8-byte
length prefix + the JSON header itself, no full download needed) and found LoRA pairs
for `linear_attn.in_proj_qkv`/`.in_proj_z`/`.in_proj_a`/`.in_proj_b`/`.out_proj` — the
mixer's own projections — alongside the already-accepted `self_attn.*`/`mlp.*`. Cross-
referencing those HF module names against `model.rs`'s own `GatedDeltaNetLayerWeights`
struct (not assumed, read directly) showed all five are declared as plain 2-D `Weight`
fields driven by `gemv` in `forward_gated_attn_mixer` — the same machinery every
already-accepted Linear projection uses. Only `ssm_dt`/`ssm_a`/`ssm_conv1d`/`ssm_norm`
(a bias vector, a decay vector, a conv1d kernel, a norm weight — none an `nn.Linear`
module PEFT could target, confirmed by their absence from the adapter's tensor list)
remain genuinely non-Linear and stay rejected. So the "non-Linear" framing was true for
four of the mixer's nine weight tensors, not all nine — a real adapter's ground-truth
tensor names caught the overbroad rejection a synthetic hand-built fixture (round 1's
only verification tool, built to whatever shape the author assumed) never could.

**How to apply**: when a llama.cpp/PEFT-format adapter's `target_modules` is a regex
(common with unsloth-trained adapters) rather than a literal list, don't infer what it
matched from the pattern text or a summarized fetch of it — read the actual
`adapter_model.safetensors` header (cheap: a ranged HTTP fetch of its first 8 bytes for
the JSON-header length, then that many more bytes, no full-file download) and treat
*that* tensor list as ground truth. The same caution applies to any base-model source
code paraphrase, not just this one — see round 1's `.weight`-suffix bug above for the
first time this exact mistake pattern (trusting a paraphrase over the raw artifact)
caused a real bug in this file's history.

**Verification**: real hardware (L40, `ae85c35a`). `convert_hf_to_gguf.py` on a fresh
`Qwen/Qwen3.5-0.8B` checkout hit this project's own `nextn_predict_layers` MTP/NextN
rejection (the real upstream checkpoint now ships an MTP draft block that didn't exist
when `load_hybrid`'s doc comment was written — an orthogonal, correctly-enforced
rejection, not a bug) — worked around by using the pre-quantized
`unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` as the LoRA base instead (MTP
absence confirmed by it loading), since the LoRA adapter's own targeted tensor
names/shapes don't depend on which GGUF build supplies the base weights.
`convert_lora_to_gguf.py` on the real adapter produced exactly the predicted GGUF base
names (`blk.N.ssm_alpha.weight`/`ssm_beta`/`attn_qkv`/`attn_gate`/`ssm_out`, confirmed
in its own conversion log before any of this project's code ran). `reflex generate
--lora` applied `tensors_applied=186`, exactly `18 GatedDeltaNet layers × 8 tensors +
6 GatedAttention layers × 7` — proof every real adapter tensor resolved, zero silent
rejections. Cross-checked against a real llama.cpp build the same three ways as round
1: (1) tensor count — `llama-export-lora` logged `merged 186 tensors with lora
adapters`, matching exactly; (2) scale — `calculated_scale=2.000000` matches
`alpha/rank = 32/16` independently; (3) output text — `llama-simple` on the merged
GGUF produced `"The capital of France is the city of Paris.\nThe capital of Germany is
the"`, token-for-token identical to `reflex generate --lora`'s own decoded continuation
for the same prompt, and visibly different from the un-adapted base's `"...the capital
of the country."` (proof the adapter changes behavior, not a no-op accept).
`davidanugraha/Qwen3.5-35B-A3B-SWE-Smith-LoRA-Adapters`/`-9B-` remain untested this
round — confirmed via the HF repo's own page text (not yet its safetensors header) to
target MoE's per-expert routed-expert projections, which per round 1's own
"How to apply" note needs new per-expert delta-selection math, not just a widened
accept list, plus a much larger model to rent for — left for a future round.

## On-device dequant extended to 9 more block types; a real Q2_K/Q3_K token divergence investigated to a non-bug conclusion, not dismissed

**Decision**: `kernels_cuda/dequant.cu` gained 9 more on-device dequant kernels
(Q4_0/1, Q5_0/1, Q8_0/1, Q2_K, Q3_K, Q8_K), each a line-for-line port of its already-
existing, already-unit-tested `dequant.rs` host function — no new math, these formats
were always correctly dequantizable, just not yet GPU-accelerated. `model.rs`'s
`dequantize_tensor_to_device`/`load_weight_device` used to take three separate
`&AotKernel` parameters (one per supported format); rather than grow that list to
twelve, they now take one `&DequantKernels` struct (`load_dequant_kernels` loads all
twelve kernels from the AOT-compiled module in one call, replacing three near-
identical manual `aot::load_kernel_module` + `.next()` blocks, one per architecture's
load site). `dequantize_on_device` gained a `block_elems` parameter (previously
hardcoded to `QK_K`=256) since the legacy formats use 32-element blocks.

**Why**: matches the precedent Q4_K/Q6_K then Q5_K set — port the common formats
first, leave the 8 IQ-family formats (which need constant-memory lookup tables from
`ggml-common.h`, not just this file's per-block-loop pattern) for a future round. The
`DequantKernels` bundling is a straightforward simplification once a fourth kernel
would have made the flat-parameter-list pattern unwieldy, not scope creep — it
doesn't change any load site's behavior, only how the kernel handles get threaded
through.

**Verification, and a real discrepancy investigated rather than shrugged off**: real
hardware (A100, `g3jx9w64`). `nvcc` compiled all 9 new kernels clean on first try.
Quantized a real `Qwen/Qwen3-0.6B` checkpoint with llama.cpp's own `llama-quantize
--pure` into every real-storable target type this round adds (`Q8_1`/`Q8_K` are
runtime-only quantized-dot-product intermediates -- `dequant.rs`'s own pre-existing
doc comments already established this, and `llama-quantize --help` offers no `Q8_1`/
`Q8_K` target type to confirm it independently -- so those two were verified by code
review against the now-confirmed-correct `Q8_0` kernel instead of a real GGUF round
trip, and that gap is disclosed rather than silently folded into "9/9 verified").
`reflex generate` matched `llama-simple` byte-exact on `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/
`Q8_0` (5/7 real-storable types, all producing `token_id=12095, " Paris"` identically)
but diverged on `Q2_K` (`" r"` vs. llama.cpp's `"?"`) and `Q3_K` (`" located"` vs.
`" Paris"`). The instinct to write this off as "expected at 2-3 bits" was deliberately
not trusted without evidence -- every other format this project has ever shipped
(`Q4_K`/`Q5_K`/`Q6_K` in earlier rounds, plus this round's other 5) matched byte-exact,
so a clean divergence isolated to exactly these two formats was treated as a likely
real bug until proven otherwise. Root-caused instead of assumed: extracted the real
`blk.0.attn_q.weight` tensor's actual raw block bytes from both quantized GGUFs and
dequantized them three independent ways -- this project's Rust `dequant::dequantize`,
llama.cpp's own Python reference implementation (`gguf-py`'s `Q2_K`/
`Q3_K.dequantize_blocks`, a from-scratch numpy reimplementation maintained by the same
upstream project, not just another copy of the same C source), and (to isolate the
CUDA port specifically) a temporary host-only fallback build to confirm the on-device
kernel and the host function produce bit-identical output -- all three agreed to f32
precision on real production bytes, not just the existing hand-crafted unit-test
inputs. That rules out the dequant math (both host and device) as the source. The
actual explanation: `llama-simple` itself, run on the *same* `Q2_K` file on CPU
(`-ngl 0`) vs. GPU (`-ngl 99`), produced a *third* different answer (`"ising"`) --
llama.cpp disagrees with its own two backends on this file, which only makes sense if
the top-token race is a photo finish that any implementation's floating-point
summation order can flip, not a race a "correct" implementation is expected to win
consistently once quantization noise gets this high (2-3 bits on a 0.6B model).
`Q3_K`'s llama.cpp CPU/GPU backends happened to agree with each other in this one
instance (both said `"Paris"`), so that specific case has one fewer independent data
point than `Q2_K`'s -- disclosed as such rather than overclaimed, though the same
per-block dequant correctness evidence (Rust/gguf-py/CUDA three-way agreement) applies
equally to both.

**A real tooling gotcha hit along the way, worth remembering past this session**:
while debugging, a temporary edit was made directly on the remote instance (removing
the `Q2K`/`Q3K` match arms to force the host fallback for comparison), then the
working tree was re-synced from the clean local copy via the project's usual `rsync
-a` recipe to restore it. The rebuild afterward reported `Finished ... in 0.05s` and
*zero* new compiler warnings where several were expected -- `cargo` had silently
skipped recompilation. Cause: `rsync -a` preserves the *local* file's mtime, which was
older than the remote's already-built target artifact from the mid-debugging state,
so `cargo`'s mtime-based fingerprinting saw "unchanged" and trusted a stale build.
**How to apply**: after any `rsync` that's meant to overwrite remote edits made
mid-session (not just the first sync of a fresh instance), `touch` the affected source
files before rebuilding, or otherwise don't trust a suspiciously-fast "Finished" line
as proof the new code actually compiled -- check for the warnings/behavior you expect
to see change, the same instinct that caught this in the first place.

## MoE per-expert LoRA: build against a synthetic fixture after both real candidate adapters turned out unusable

**Decision**: widened `--lora` to accept MoE's three per-expert-stacked FFN tensors
(`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`) for a standard PEFT adapter format
(one `nn.Linear` per expert, `experts.{i}.{gate,up,down}_proj`), verified against a
hand-built synthetic adapter (`test-data/tiny-qwen3moe-lora.gguf`,
`scripts/build_tiny_moe_lora_fixture.py`) rather than a real one, after determining
neither previously-identified `davidanugraha` candidate could actually be used.

**Why the real candidates were dropped, not just left untested**: round 1's follow-up
note said these needed "new per-expert delta-selection math" and left them for a
future round on that assumption. This round read the real safetensors headers before
writing any code (same lesson as the Tilakoid round above: read the artifact, don't
paraphrase it) and found something more fundamental. `Qwen3.5-9B-SWE-Smith-LoRA-
Adapters`' header has zero MoE-shaped tensors at all -- every target
(`self_attn.*`/`linear_attn.*`/`mlp.{gate,up,down}_proj`) is a plain dense Linear
already accepted, because the 9B checkpoint has no MoE layers (`-A3B` names the 35B
variant, not the 9B). `Qwen3.5-35B-A3B-SWE-Smith-LoRA-Adapters`' header does have
MoE-shaped LoRA tensors (`mlp.experts.lora_A`/`lora_B`), but as a *single pair per
layer* with no per-expert index anywhere in the name, and shapes
(`[8192,512]`/`[2048,8192]`) that don't factor into any `(rank, in_features)`/
`(out_features, rank)` pair for the adapter's own `r=32` -- its `adapter_config.json`
carries `"megatron_core": "megatron.core"`, and the repo's own file tree
(`training-image/verl_qwen35_replicated_gdn.patch`, FSDP/RLOO training artifacts) is
a `verl` RL-training export, not a plain HF PEFT checkpoint. Read llama.cpp's actual
`convert_lora_to_gguf.py` + `Qwen2MoeModel.modify_tensors` (the code every MoE arch's
`LoraModel` subclass inherits unchanged) to check whether this was merely an
inconvenient shape or a hard blocker: its *only* per-expert-stacking mechanism is a
`torch.stack` over a Python dict keyed by per-expert-indexed HF tensor names
(`model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight`), populated one entry at a
time as each expert's tensor streams in, triggered only once `n_experts * 3` entries
have accumulated for a layer. An adapter with one fused tensor per layer never
populates that dict in a way that reaches `n_experts * 3` distinct keys -- the
stacking branch simply never fires, so **this specific adapter cannot be converted to
GGUF by llama.cpp at all**, independent of whether this project ever added MoE-LoRA
support. That's a different and stronger conclusion than round 1's "needs new math"
note assumed, so it's called out explicitly here rather than silently reusing that
note's framing.

**Confirmed with the user before proceeding** (a real premise change mid-task, not a
minor detail): given neither real adapter is usable, asked whether to keep searching
for a real, standard-format MoE LoRA adapter targeting a compatible checkpoint, drop
the item, or build against a synthetic fixture instead (this project's own established
pattern for a feature with no real available fixture, e.g. the MLA and qwen3moe base
fixtures). User chose the synthetic-fixture path.

**Math derivation, not run through the real converter**: no local Python
`transformers`/`torch`/`peft` install on this machine (and installing them just to
verify a shape/byte-layout derivation was judged not worth the weight for this
session), so the standard-format math was derived by tracing the real, unmodified
`convert_lora_to_gguf.py`/`Qwen2MoeModel.modify_tensors` source directly rather than
executing it: `LoraTorchTensor.__torch_function__`'s `torch.stack` arm stacks a list
of per-expert `LoraTorchTensor`s' `_lora_A`/`_lora_B` fields separately along a new
leading dim, so a per-expert-Linear adapter's stacked `lora_a`/`lora_b` end up GGUF
ne-shape `[in_features, rank, expert_count]`/`[rank, out_features, expert_count]` --
one more trailing dim than the dense 2-D case this project already handled, in the
same row-major-per-expert-contiguous-chunk byte layout as the base model's own
`ffn_*_exps` tensors (cross-checked against `model.rs`'s pre-existing
`expert_weight_view` doc comment, which already documents that exact layout for the
base weights). This is a derivation, not independent execution -- flagged honestly
below rather than folded into "verified".

**Implementation**: `src/lora.rs`'s per-target parsing loop now accepts either a 2-D
or a 3-D shape for `lora_a`/`lora_b` (a new `expert_count: Option<usize>` field on
`LoraTarget`), and its delta computation gained one outer per-expert loop around the
*same* row-major math the dense case already used -- not new math, just applying the
existing math once per expert instead of once. `model.rs`'s `apply_lora` shape check
now accepts a matching 3-D base weight shape (`expert_count` must match too), and
`find_lora_target_mut` gained three more `LayerWeights::Moe` match arms
(`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`). Because the computed delta is laid
out identically to the base weight's whole device buffer regardless of expert count,
the existing single whole-buffer `add_inplace` launch in `apply_lora` needed no
change at all -- no new kernel, no per-expert slicing on the caller's side.

**Verification, and what's honestly still missing**: host-only (`REFLEX_SKIP_CUDA=1`
build/test/fmt/clippy all clean; the other 85 pre-existing host-only tests
unaffected). `scripts/build_tiny_moe_lora_fixture.py` hand-builds
`test-data/tiny-qwen3moe-lora.gguf` via `gguf.GGUFWriter` (matching
`test-data/tiny-qwen3moe.gguf`'s real `in_features=out_features=32`,
`expert_count=8`, 2 MoE layers) with a deterministic per-layer/per-tensor-kind/
per-expert/per-rank value formula distinct enough that a mixed-up layer, tensor-kind,
or expert offset would produce a visibly wrong delta, not a subtly-close one. A new
test (`lora::moe_expert_lora_fixture_tests::moe_per_expert_lora_matches_hand_computed_delta`)
loads it through the real `lora::load` and compares against an independently
recomputed expected delta (a separate triple loop in the test, sharing no code with
the implementation) -- passes, byte-exact across all 6 targets (2 layers x 3 tensor
kinds). This machine has no CUDA-capable GPU, so at the time this entry was first
written, `Model::apply_lora`'s device-side `add_inplace` launch and an actual
forward pass through the adapted model were not yet verified against real GPU
hardware -- **closed the same session**: rented a fresh AWS EC2 `g4dn.xlarge`
(Tesla T4) and confirmed `reflex generate --lora` against the synthetic fixture
applies `tensors_applied=6` (exactly matching expectation, no silent rejections)
and measurably changes the model's output vs. the unadapted baseline, with zero
regressions in the pre-existing MoE test suite (see HISTORY.md's matching entry
for the full readout, including why the LoRA-adapted run collapsing to immediate
EOS is expected given the fixture's deliberately large synthetic delta magnitudes,
not a bug).

## Tokenizer map hasher: vendored inline FxHasher (implemented, real-hardware-verified 2026-09-28)

**Decision**: replace `std::collections::HashMap`'s default SipHash hasher with a
fixed, small, non-cryptographic hasher for `Tokenizer`'s two maps (`token_to_id`,
`merge_rank`) — and only those two maps, not a crate-wide hasher-policy change. The
hasher is **vendored inline in `src/tokenizer.rs`** (the classic Firefox/rustc `FxHash`
algorithm, ~50 lines), **not** the `rustc-hash` crate: this matches the project's
no-new-default-dependency convention and the existing in-repo precedent of
`aot.rs`'s inline `fnv1a_hash`, keeps the crate's default dependency footprint at
cudarc/half/memmap2/rand, and avoids pinning another crate for what is self-contained,
testable, ~50 lines of arithmetic.

**Why**: the item-5/item-6 phase-breakdown profiling made tokenizer construction a
measured, named cost (~107ms of `model_load_ms`, ~26% of the 410.3ms p50 on dense
`Qwen3-0.6B`). The plan's hypothesis was that its dominant compute is hashing: ~151k
token strings into `token_to_id` and ~150k `(String,String)` merge keys into
`merge_rank`, every one SipHash — the std default, designed for HashDoS resistance on
untrusted input, typically 3–5x slower than a non-cryptographic hasher the same
operation doesn't need. These maps never see untrusted input — keys come from a model
file the user already chose to load, in the same trust boundary as the weights — so
HashDoS resistance buys nothing here. A hasher swap is also the lowest-risk part of
this plan: no ownership/allocation changes, no algorithm changes, byte-exact encode
output is unchanged because hashing affects only table insertion/lookup, never
iteration order in these lookups (encode resolves each symbol via a direct `get`,
never enumerates the map).

**Measured outcome, real hardware (AWS EC2 `g4dn.xlarge`/Tesla T4, `sm_75` cubin,
strict change-only A/B of `src/tokenizer.rs` alone, interleaved n=10)**: golden tokens
byte-identical on both arms (dense `12095`/`" Paris"`, hybrid `279`/`" the"`, synthetic
MLA `94216`/`" NavLink"`, MoE deterministic `[45729,22560,23860,16773,8275]`). The
timing win is **real but much smaller than the plan estimated**: `model_load_ms` p50
**415.568ms → 411.627ms (-3.9ms)**, total `process_start_to_result_ms` p50
**632.712ms → 629.671ms (-3.0ms)**. The win is consistent, not noise — every one of the
10 paired before/after runs improved (mean per-pair delta ~5.2ms on `model_load_ms`),
and `prompt_eval_ms` is untouched (39.023 → 39.048ms, within noise) — but it is ~1% of
`model_load_ms`, far below the ~60-80ms the hasher item's own plan estimated.

**The finding that reframes the tokenizer half of the plan**: the same way the AOT
rope-dedup's ~1ms-not-~8ms measurement reframed the AOT half, this says SipHash was
**not** the dominant cost of `Tokenizer::from_gguf`. The ~107ms "tokenizer
construction" is dominated by *allocation/copying* — parsing and cloning the ~151k
`tokens` strings, cloning them a second time into `token_to_id`, and the ~150k
two-`String`-per-key `merge_rank` builds — not by hashing them. The plan's item
ranking therefore inverts: the fast hasher is the *smallest* lever, not the biggest
(the duplicate-vocab-copy/arena item is the likely large one), and the load-time
worker-thread item (#4) keeps its full ~107ms value regardless, since it removes the
whole construction from the serial path rather than shrinking it.

**How to apply**: the hasher change is scoped to the two `HashMap` type aliases in
`src/tokenizer.rs` (`FxHashMap`/`FxBuildHasher` there); no `Cargo.toml` change. Land
it (zero-risk, correctness-neutral, strictly a small consistent win), but do **not**
rely on the hasher to deliver a large tokenizer win — re-rank the remaining tokenizer
items against a fresh measurement (the arena/duplicate-copy work is the one to
evaluate next), and measure any successor with the same interleaved A/B + golden-token
regression gate used here (`scripts/bench_cold_start_phases_system1.sh`, n=10, plus the
`tokenizer.rs` unit tests). Revisit the `crate-policy` question only if some other map
later shows up as hot on this path; don't grow a crate-wide hashing convention from a
two-map fix.

## Tokenizer construction on a load-time worker thread (implemented, real-hardware-verified 2026-09-28)

**Decision**: build the `Tokenizer` on a single one-shot background thread inside
`Model::load` (dense/MoE, hybrid, and MLA — all three `load*` sites), started at the top
of each load body and `join()`-ed before the `Model` is returned, so its CPU
hashing/allocation overlaps the GPU-bound weight-load pipeline instead of sitting after
it on the serial path. Exactly one extra thread, spawned per load, joined before the
caller sees the `Model`, never reused and never accepting a second unit of work.

**Why**: `Tokenizer::from_gguf` needs only `file` (already mmap'd/opened by then) and
reads pure host metadata — no dependency on the device, the loaded weights, or the
dequant pipeline, which is what gives it runnable-in-parallel status. The
piece-by-piece hasher/arena work above shrinks the *serial* cost; this item removes it
from the serial path *entirely* (the weight-load pipeline is the genuinely independent
concurrent side). The one thing that stopped this being proposed bluntly is the
project's permanent Non-goals (no thread pool, one job at a time, strictly
sequential). It goes ahead on the strength of in-repo precedent: `energy.rs`'s NVML
poll thread is already a deliberate, documented non-exception (a single internal
thread that measures but never *serves*, is torn down with the process via
`fast_exit`, and accepts no work items). A one-shot tokenizer-build thread is the same
shape — it exists only to make one load faster and is gone before any request runs.
The `batch_size`-1/no-thread-pool constraints govern *concurrent request handling*,
not whether a single load may use an internal helper thread once and join it.

**Implementation note — scoped threads, not `std::thread::spawn`**: the load functions
take `file: &GgufFile`, so the thread needs a borrow that does not outlive it rather
than a `'static` `Arc<GgufFile>`. `std::thread::scope` gives exactly that: each
`load`/`load_hybrid`/`load_mla` is now a thin wrapper that spawns
`Tokenizer::from_gguf(file)` on a scoped thread and hands the
`ScopedJoinHandle<Result<Tokenizer, String>>` to a new private `*_inner` body, which
joins it at its end (surfacing a thread panic as an ordinary `Err` via `map_err`). No
`Arc` and no metadata snapshot/copy is needed — `GgufFile` is immutable after parse
(`Mmap` + owned metadata, all `Send + Sync`), so the main load body and the tokenizer
thread only ever share immutable reads. Failure propagation is unchanged: a tokenizer
error surfaces after the join, exactly as before. `REFLEX_SKIP_CUDA=1` builds are
unaffected (the thread is unrelated to CUDA).

**Measured outcome, real hardware (AWS EC2 `g4dn.xlarge`/Tesla T4, `sm_75` cubin,
strict change-only A/B of `src/model.rs` alone against HEAD `88a1f32`, interleaved
n=10)**: golden tokens byte-identical on both arms (dense `12095`/`" Paris"`, hybrid
`279`/`" the"`, synthetic MLA `94216`/`" NavLink"`, MoE deterministic
`[45729,22560,23860,16773,8275]`). `model_load_ms` p50 **408.942ms → 307.554ms
(-101.4ms, -24.8%)**, total `process_start_to_result_ms` p50 **627.148ms → 523.686ms
(-103.5ms, -16.5%)**; `prompt_eval_ms` is flat (39.027 → 39.328ms, noise). This is the
full ~100ms tokenizer construction moved off the serial path — the item's whole design
goal. It also means the earlier hasher item (#1) is now almost entirely subsumed at the
E2E level: whether the tokenizer takes 107ms or 103ms no longer matters once it is
hidden behind the GPU load, so further *serial* tokenizer micro-optimization
(arena/merge-key) has much less value than its own measurement suggested; only
shortening the overlap window would make it matter again.

**How to apply**: don't generalize this — keep it exactly one scoped thread spawned and
joined inside a single load, never a pool and never a background thread that outlives
the call. If a future load-path item wants concurrency too, reuse this same
`thread::scope` wrapper/inner shape rather than adding a second mechanism. Because this
is a timing-only change with zero numerics impact, the golden-token check was a
regression assert, not new first-principles verification. As of the cuBLAS-overlap entry
below, this same worker thread also builds the cuBLAS handle (the helper is now
`load_background_init`); the one-thread guarantee is unchanged.

## AOT module loads are ~3ms; the "~83ms" is cuBLAS init (measured 2026-09-28)

**Decision**: stop treating "AOT kernel module load" as a cold-load cost to optimize —
every dense `cuModuleLoad` together measures ~3ms. Reject the single-module coalescing
idea (item 2 of the tokenizer+AOT plan) outright, and re-point the next cold-load work at
`CudaBlas::new` + `cublasSetMathMode` (~81ms), which the plan's "~83ms AOT module load"
figure was actually measuring.

**How this was measured**: a throwaway instrumentation build (env-gated `eprintln!`
timers in `src/aot.rs` and the dense `Model::load` body, applied only on the remote
instance and never committed — the same convention the item-5 profiling used) on an AWS
EC2 `g4dn.xlarge`/T4, `sm_75` cubin (and, for the module-load point, a second portable-PTX
build). Two results:
- Per-module: the whole dense AOT set (rmsnorm, rope, silu, gemv, gemv_gather, attention,
  attention_prefill, elementwise, dequant) is **~3ms total** (cubin) / ~3.5ms (portable
  PTX), with no first-module/context-warmup spike and the 20-fn `dequant` module the
  largest at ~1ms. The rope-dedup's earlier ~1ms-per-freed-load result already hinted at
  this; this is the direct confirmation.
- Per-region: `parse_model_config` ~0.01ms; **`CudaBlas::new` + `cublasSetMathMode`
  ~80.9ms**; kernel loads + `WeightLoadPipeline::new` ~3.2ms; weight loop ~123.8ms;
  `token_embd` construction ~102.1ms; `output_norm`/`lm_head` ~0.1ms; tokenizer-thread
  join ~0.01ms (fully hidden by the worker thread).

**Why the original figure was wrong**: the item-5 "AOT kernel module load 83.09ms" timer
almost certainly started before `CudaBlas::new` and ran through the kernel loads, so it
captured cuBLAS handle init *plus* module loads, and the plan then attributed the whole
thing to the module loads. Same class of error the rope-dedup entry already exposed (a
back-of-envelope "~8ms per call" vs. a measured ~1ms); the lesson is to instrument the
actual boundary, not infer it.

**How to apply**: don't revisit module coalescing — there is nothing there. The ~81ms
`cuBLAS` init is the next candidate (overlap it on a scoped thread like the tokenizer, or
avoid cuBLAS on the `system1`/decode path where the custom kernels already do the work),
but that is a real design decision with its own numerics/verification question, not a
follow-on patch — treat it as a new item, with its own DECISIONS entry and strict A/B,
before any code is written.

## cuBLAS handle init overlapped on the load-time worker thread (implemented, real-hardware-verified 2026-09-28)

**Decision**: move `CudaBlas::new` + `cublasSetMathMode` off the serial load path by
building them on the same one-shot scoped worker thread the tokenizer already uses, run
*after* the tokenizer on that thread, and move the handle back once the thread is joined
(both `Tokenizer` and `CudaBlas` are `Send`; cudarc 0.11.9 declares `unsafe impl Send for
CudaBlas`). Still exactly one extra thread per load, never a pool.

**Why**: the AOT-diagnostic entry above established that the plan's "~83ms AOT module
load" was really `CudaBlas::new` + `cublasSetMathMode` (~80.9ms) — the largest single
remaining cold-load cost once the tokenizer was overlapped. cuBLAS handle creation is
pure host/driver setup that needs no loaded weights, so it can run concurrently with the
GPU-bound weight loop exactly like the tokenizer. `CudaBlas::new` itself calls
`device.bind_to_thread()` (cudarc 0.11.9 `cublas/safe.rs:29`), which sets the primary
context current on whichever thread runs it, so no extra context plumbing is needed.
Running tokenizer (~103ms) then cuBLAS (~81ms) sequentially on the *same* thread keeps
the load at one extra thread and still fits inside the weight loop's ~226ms, so both are
fully hidden.

**Measured outcome, real hardware** (AWS EC2 `g4dn.xlarge`/T4, `sm_75` cubin, strict
change-only A/B of `src/model.rs` alone against HEAD `99cb544`, interleaved n=10): golden
tokens byte-identical on both arms (dense `12095`/`" Paris"`, hybrid `279`/`" the"`,
synthetic MLA `94216`/`" NavLink"`, MoE deterministic `[45729,22560,23860,16773,8275]`).
`model_load_ms` p50 **316.310ms → 236.590ms (-79.7ms, -25.2%)** (means 315.2 → 237.1ms),
total `process_start_to_result_ms` p50 **536.124ms → 456.440ms (-79.7ms, -14.9%)**;
`prompt_eval_ms` flat (39.818 → 39.793). The full ~80ms cuBLAS cost is hidden, as
predicted, on the first try — the `bind_to_thread` cross-thread concern did not
materialize (golden tokens byte-identical).

**How to apply**: this is the last of the measured host/driver setup costs on the load
path. Don't add a second background thread/mechanism for the next such item — extend the
existing `load_background_init` helper (it now carries both the tokenizer and the cuBLAS
handle; one thread, joined, no pool). The remaining `model_load_ms` is dominated by the
per-tensor weight loop (~124ms) and the lazy `token_embd` raw-byte copy (~102ms), both
memory/GPU-bound rather than host setup, so further load-path wins need a different idea
(re-rank against the new ~237ms floor).

## Multi-arch fatbin packaging (`REFLEX_CUDA_ARCHS`) and the explicit core-vs-optional feature split

**Decision (fatbin)**: add a third AOT output mode to `build.rs`, selected by the
plural `REFLEX_CUDA_ARCHS=sm_XX,sm_YY,...`, that compiles each kernel once via
`nvcc -fatbin` with one `-gencode arch=compute_XX,code=sm_XX` per listed arch plus a
trailing `-gencode arch=compute_<highest>,code=compute_<highest>` (embedded PTX
fallback for any GPU newer than the highest listed arch). `REFLEX_CUDA_ARCHS` is
mutually exclusive with the existing singular `REFLEX_CUDA_ARCH`; setting both is a
`build.rs` panic.

**Why (fatbin)**: the single-arch cubin mode (`REFLEX_CUDA_ARCH`) has zero driver-side
JIT but hard-fails on any other GPU — the Runpod `AMPERE_16` pool is documented as
mixed `sm_86`/`sm_89`, forcing the "pin to RTX A4500" SKU workaround. A fatbin covering
the pool's archs would let one undistinguished image schedule anywhere in it correctly,
while still giving zero-JIT on every listed arch. The loader needs no change:
`Ptx::from_file` → `cuModuleLoad` is format-agnostic (confirmed against cudarc 0.11.9's
`driver/safe/ptx.rs` and `src/aot.rs`'s own cubin precedent), so `aot.rs` just adds a
`"fatbin"` arm reusing the cubin temp-file path. `src/diagnostics.rs` gains
`COMPILED_ARCHS` + `fatbin_native_for_device` because a GPU *not* in the list must be a
non-fatal "will JIT from embedded PTX" report, not the hard error single-arch mode
correctly raises (that mode has no fallback). `reflex doctor` surfaces
native-vs-fallback for fatbin builds.

**Decision (features)**: make the core-vs-optional dependency boundary explicit —
`default = []` in `Cargo.toml` (previously implicit), and a README "Build features"
table documenting that only `download` (Linux `libssl-dev`+`pkg-config`) and `python`
(a Python interpreter) add host build dependencies the feature-less core build doesn't.
`--all-features` is documented as a developer matrix-testing convenience, not a release
build; the deploy Dockerfiles keep building explicit minimal feature sets as they
already did.

**Not done here, and why**: real-hardware verification of the fatbin path (a working
multi-arch `.fatbin` actually loading/launching on a real CUDA-toolkit GPU) — this was
implemented and CI/`REFLEX_SKIP_CUDA=1` checked only, since no `nvcc`/GPU is available on
the machine that wrote it; it reuses this repo's existing real-hardware-verification
bar. Also not fixed: the `python` feature does not currently compile at all
(`--all-features` fails with `Send` not implemented for `PyModel`, because
`WeightLoadPipeline` holds a `*mut u8` pinned-host pointer that isn't `Send`) — a
pre-existing incompatibility between the pyo3 `#[pyclass]` `Send` bound and the current
`Model` shape, documented here and left untouched rather than papered over with an
`unsafe impl Send`.

## Runtime-image slimming is opt-in (`REFLEX_RUNTIME_BASE` + `REFLEX_RUNTIME_SLIM`), defaults unchanged (2026-09-29)

**Context**: the root README's "Container image size is part of cold start here" section
recorded a measured finding -- the full CUDA `-runtime-` base is mostly libraries this
engine never loads (Reflex links only the driver API + cuBLAS), and `base` + `libcublas`
is much smaller -- but explicitly "not yet landed". Landing it risks silently changing an
already-deployed image's base image, which is the kind of default change this project
deliberately avoids.

**Decision**: add the slim runtime as an *opt-in* build-arg variant on every deploy
Dockerfile, mirroring the multi-arch fatbin work's "add capability, don't change
defaults" rule. A global `ARG REFLEX_RUNTIME_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04`
(declared before the first `FROM` so it can be used in the runtime `FROM`) plus
`ARG REFLEX_RUNTIME_SLIM=""`; the latter installs `libcublas-12-4` only when non-empty.
Both unset reproduce the previously-shipped image exactly. `scripts/deploy_runpod.sh
--slim` drives it and tags the result `:runtime`.

**Why build-args, not a second Dockerfile**: a near-duplicate Dockerfile per deploy target
would drift from the original; the build-arg form keeps one source of truth per target and
makes the default provably unchanged (the `RUN` is a no-op when unset, and `--check`
passes with default args). The cost is the pre-`FROM` global ARG wording, documented at
each site.

**Endpoint-creation API choice (same session)**: the load-balancing endpoint *type*
(`type: "LB"`) exists only on the GraphQL `saveEndpoint` mutation, while the exact GPU
*SKU* is expressible on the REST endpoint input (`gpuTypeIds`) but the type is not. So
`scripts/deploy_runpod.sh` deliberately combines both: REST for the template +
post-create configuration (SKU pin, flashboot), GraphQL for endpoint creation. This mirrors
the README's documented "create the endpoint, then pin the SKU with a control-plane call"
sequence and the mixed-architecture `AMPERE_16` hazard that makes the pin mandatory.

**Not verified here**: the live API calls (no Runpod credentials on the writing machine)
and GPU execution of the slim image (no local NVIDIA GPU). The image build itself was
verified locally for both variants; the slimming's original real-hardware verification is
recorded in the README. A live deploy using the script is the remaining step.
