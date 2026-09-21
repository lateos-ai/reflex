# DECISIONS.md

A log of the project's non-obvious technical and scope decisions, and why they were
made. For current state see `STATUS.md`; for full narrative/benchmark detail see
`README.md`.

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
