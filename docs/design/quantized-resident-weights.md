# Design: keep weights quantized on the GPU

**Status: prototype built and measured on a T4 (2026-10-02), opt-in behind
`REFLEX_QUANT_RESIDENT=1`. Results are in [Prototype results](#prototype-results); the
sections before them are the original proposal, kept as written.**

**Rebuilt on `f16` weight storage (2026-10-06), not yet re-measured.** `f16` became the
default weight storage after this prototype was measured, so every "f32" baseline below
predates it. The two now combine under one rule: `REFLEX_QUANT_RESIDENT=1` decides which
weights stay quantized (Q4_K layer weights and a Q6_K LM head, dense path only), and
`--weights` decides the element type of everything that is dequantized. That covers the
other matrix weights, and the scratch buffer the prefill path above the row crossover
dequantizes into: `f16` read by `cublasGemmEx` by default, `f32` read by `Sgemm` with
`--weights f32`. Under `--weights f32` the behaviour is the prototype's. `WeightData` has
three variants (`F32`, `F16`, `Quant`). The quantized kernels themselves (`gemv_q4k`,
`gemv_q6k`) take `f32` activations either way.

Today every weight tensor is dequantized once at load and kept on the GPU as an `f32`
buffer (`Weight { data: CudaSlice<f32>, .. }` in `src/model/loading.rs`). This note
proposes keeping the GGUF's quantized blocks resident instead and dequantizing inside
the matmul kernels, starting with Q4_K on the dense Qwen3 path behind a flag.

Why it matters:

- **Model load is the largest cold-start phase**: ~235 ms of ~485 ms total on a T4
  (`bench/baseline-t4.json`), ~180 ms of ~436 ms on an A10G (`reflex system1`,
  Qwen3-0.6B Q4_K_M, p50).
- **VRAM is ~4 bytes per parameter whatever the file's quantization.** Mistral 7B runs
  out of memory on both the T4 (16 GB) and the A10G (24 GB). A 27B model would need
  ~108 GB.
- **Decode is memory-bound**, and the GEMV reads about 7x more bytes than the file holds.

Nothing here touches the Non-goals: `batch_size` stays 1, and there is no server, queue
or scheduler. Weight layout and kernels change; the external surface does not.

## 1. Where the load time goes today

### What gets moved (measured from the file)

`Qwen3-0.6B-Q4_K_M.gguf`, the benchmark model (inventory from its GGUF header):

| tensors | type | count | elements | bytes on disk | bytes as `f32` |
|---|---|---|---|---|---|
| matmul weights | Q4_K | 168 | 381.7 M | 214.7 MB | 1,526.7 MB |
| matmul weights (`attn_v`, `ffn_down` in half the layers) | Q6_K | 28 | 58.7 M | 48.2 MB | 234.9 MB |
| norms | F32 | 113 | 0.1 M | 0.3 MB | 0.3 MB |
| **device-resident total** | | **309** | | **263.1 MB** | **1,761.9 MB** |
| `token_embd` (tied LM head) | Q6_K | 1 | 155.6 M | 127.6 MB | 622.3 MB |

`token_embd` stays on the host as raw bytes (`LazyTokenEmbedding`). The LM head is tied
to it, so it only becomes device-resident when something needs full-vocabulary logits
(`LmHead::TiedLazy` → `lm_head_resident`). That happens on `generate`'s first token, not
on `system1`. Then it is dequantized to 622 MB of `f32`, a cost that lands in
`prompt_eval_ms`.

Q4_K is 87% of the matmul elements, so a Q4_K-only prototype covers most of the bytes.
Q6_K is the second format to add, because the LM head is Q6_K too.

### Per-tensor work inside `model_load_ms`

For each of the 309 tensors, `WeightLoadPipeline::dequantize` does this:

1. host `memcpy` from the mmap'd file into a pinned staging buffer (263 MB in total);
2. async H2D copy of the raw bytes (263 MB);
3. `cuMemAlloc` of the `f32` output (309 allocations, 1.76 GB in total);
4. the dequant kernel (reads 263 MB, writes 1.76 GB);
5. for a partial last block, a device-to-device truncation copy (rare).

Also inside `model_load_ms`: loading about ten AOT kernel modules (the `dequant` module
alone holds 20 kernels), copying `token_embd`'s 127.6 MB into an owned `Vec`
(`LazyTokenEmbedding::new` calls `to_vec()`), and joining the background thread that
creates the tokenizer and the cuBLAS handle.

**We have no measured split of these parts.** For scale only (not measurements), here
are the bandwidth floors. The H2D copy of 263 MB is ≥ ~22 ms on the T4 (PCIe 3 x16,
~12 GB/s) and ≥ ~11 ms on the A10G (PCIe 4). Writing 1.76 GB of `f32` is ≥ ~6 ms on the
T4 (320 GB/s) and ≥ ~3 ms on the A10G (600 GB/s). Those floors add up to well under the
measured 180–235 ms, so allocation, host copies, page faults and module loading are
probably a large share. They need measuring before any load-time gain can be promised.

**Prototype step 0 adds that measurement**: sub-phase timers in `load_dense_inner`
(module loads, pinned `memcpy`, H2D, alloc, dequant kernel, `token_embd` copy, cuBLAS
join). Each one is a `cuEventRecord` pair or a host `Instant`, reported in the existing
`REFLEX_PHASE_OK` style.

### What quantized residency removes, and what it keeps

| step | f32 residency (today) | quantized residency |
|---|---|---|
| pinned `memcpy` + H2D of raw bytes | 263 MB | 263 MB (unchanged) |
| device allocations | 309, 1.76 GB | one arena, 263 MB |
| dequant kernels | 309 launches, 1.76 GB written | none at load |
| tied LM head on `generate` | 622 MB `f32` dequant in `prompt_eval_ms` | upload 127.6 MB raw Q6_K (once Q6_K kernels exist) |

The load-time gain is therefore bounded by steps 3–5 plus the allocation overhead. If
step 0 shows the load is dominated by host copies and module loading, the cold-start case
rests on `generate`'s LM head, VRAM and decode speed rather than on `model_load_ms`. The
note should be read with that in mind.

## 2. Kernel changes

### Representation

`Weight` becomes a sum type. It keeps its name, so every `&Weight` signature stays as it
is:

```rust
pub(super) enum WeightData {
    F32(CudaSlice<f32>),
    /// Raw GGUF blocks, byte-identical to the file. Rows are block-aligned
    /// (a ggml invariant), so row r starts at r * row_bytes.
    Quant { ty: GgmlType, bytes: CudaView<'static, u8> /* or (arena, offset, len) */, row_bytes: usize },
}
pub(super) struct Weight { pub(super) data: WeightData, pub(super) shape: Vec<u64> }
```

Every quantized tensor lives in one device arena, allocated once from the summed tensor
sizes, which are known from the GGUF header before any copy starts. Norms and
unsupported types stay `F32`.

### Decode: GEMV that dequantizes as it computes (`gemv_q4k_kernel`)

- Same geometry as `gemv_kernel`: one warp per output row, 8 rows per 256-thread block.
- A Q4_K row of `in_features = K` is `K/256` super-blocks of 144 bytes: `d` and `dmin`
  (f16), 12 bytes of packed 6-bit scales and mins, and 128 bytes of nibbles. Each lane
  handles one 32-element sub-block at a time. It decodes that sub-block's scale and min
  once, unpacks 32 nibbles from 16 coalesced bytes, and runs `fma` against `x` (`f32`,
  read through `float4`). Then comes the existing warp-shuffle reduction.
- **Activations stay `f32`.** Every product `w_i * x_i` uses the same dequantized `w_i`
  as today, so only the summation order changes, as it did when `gemv_kernel` went
  warp-per-row. llama.cpp's MMVQ quantizes `x` to Q8_1 and uses `dp4a` integer dot
  products, which is faster but numerically different again. That stays a later
  optimization with its own verification.
- Decode-only variants needed by the dense path: `gemv` (whole tensor) and `gemv_gather`
  (System1's candidate rows of the LM head; only once the LM head is quantized). The MoE
  `gemv_expert` and MLA `gemv_per_head*` variants are out of prototype scope.

### Prefill: batched projections

Every prompt, even System1's ~12 tokens, goes through `prefill_dense_batched` and its
cuBLAS `Sgemm` (`Model::gemm`). cuBLAS can't read Q4_K, so there are two paths:

1. **Few rows (≤ 8, tunable): a multi-row fused GEMV.** The same kernel as decode, but
   each warp keeps `R` accumulators and reads each weight block once for all `R` input
   rows. Weight bytes read: `ceil(rows/8) × 263 MB`, with no `f32` written. This is how
   llama.cpp handles small batches (MMVQ with `ncols ≤ 8`), and it covers System1's
   prompt in two passes.
2. **More rows: dequantize into a reused scratch buffer, then the existing `Sgemm`.**
   One device scratch buffer, sized to the largest tensor (`ffn_gate`/`ffn_up`:
   1024×3072 `f32` = 12 MB here), is filled by the existing `dequant.cu` kernels from
   the arena (device to device, no H2D), then `gemm_view` consumes it. This is the same
   math as today, reordered: the `f32` matrix exists for one call instead of for the
   process lifetime. Cost: 1.76 GB of transient writes per prefill, which only pays off
   above the crossover row count.

Pick the crossover by measurement on the T4 and A10G. A later option is an `f16`
scratch with `cublasGemmEx` on tensor cores, which changes numerics and is out of scope.

### Q6_K and Q8_0

Same structure: one `gemv_q6k_kernel` and its multi-row form, and the scratch path
reuses the existing `dequantize_q6k_kernel`. Q6_K matters for the LM head and for
`attn_v`/`ffn_down`. Q8_0 (34-byte blocks of 32) is the easiest of the three and covers
`--outtype q8_0` conversions such as the DeepSeek-V2-Lite GGUF.

## 3. Interactions

**`WeightLoadPipeline`.** For a quantized tensor there is no dequant kernel. Step 4's
destination becomes the tensor's offset in the arena, and the device-side staging
buffers and `kernel_done` events are skipped. The pinned double-buffer and the async copy
on the forked stream stay, so the host `memcpy` of tensor N+1 still overlaps the H2D of
tensor N, and the two existing rules still hold: no blocking per-tensor copy, and no
per-tensor staging allocation. `F32`-resident tensors (norms, any unsupported format)
keep today's dequant path unchanged. Later idea, not in the prototype: register the
mmap with `cuMemHostRegister` and skip the pinned `memcpy` entirely.

**`LazyTokenEmbedding`.** Unchanged: the embedding lookup is a host-side gather and only
needs a few rows. When the LM head is tied and quantized, the full-vocabulary upload
becomes a raw-byte H2D copy of `token_embd.raw` (127.6 MB) instead of an on-device
dequant to 622 MB. The `to_vec()` copy in `LazyTokenEmbedding::new` is a separate,
adjacent cost that step 0 will size.

**LoRA (`Model::apply_lora`).** Today it adds `scale·(B@A)` into the `f32` base weight in
place. A quantized weight can't take that without re-quantizing, which is lossy and
diverges from llama.cpp, so that option is out. Options:

- **(a) Materialize targeted tensors to `f32` (prototype).** `apply_lora` converts each
  targeted `Weight` from `Quant` to `F32` with the existing dequant kernel, reading the
  arena (device only), then adds the delta exactly as today. Results match the flag-off
  path. VRAM grows only by the targeted tensors.
- **(b) Runtime low-rank path (later).** `y += scale·B(Ax)` per targeted projection,
  which is what llama.cpp itself does. It keeps everything quantized, costs two small
  GEMVs per targeted projection per token, and adds a branch to the forward pass. The
  current design deliberately avoids that branch.

**cuBLAS prefill path.** It is kept and is still the default for `F32` weights. For
quantized weights it is reached only through the scratch path above. The
`prefill_dense_batched_matches_sequential_prefill` test stays the oracle and runs with
the flag on.

**MoE / MLA / hybrid.** Out of prototype scope. `expert_weight_view` slices by element
offset. The quantized equivalent slices by `expert_idx × expert_bytes`, which works
because each expert's chunk is whole rows. With the flag on, these architectures keep
`f32` residency (and say so in a one-line notice), so behavior can't silently differ.

**KV cache, `kv_io`, IPC, FFI, Python, sidecar.** Unaffected: none of them see weights.

## 4. Expected gains and risks

All figures below are arithmetic from the tensor inventory above, not measurements.

| | today (`f32`) | prototype (Q4_K resident, Q6_K + LM head `f32`) | Q4_K + Q6_K resident |
|---|---|---|---|
| device weight bytes, layers | 1,762 MB | 215 + 235 = 450 MB | 263 MB |
| + LM head when resident (`generate`) | +622 MB | +622 MB | +128 MB |
| weight bytes read per decode token | ~2.38 GB | ~1.07 GB | ~0.39 GB |
| decode floor on a T4 at 320 GB/s | ~7.4 ms | ~3.3 ms | ~1.2 ms |

- **Load (`model_load_ms`).** At most the dequant, allocation and truncation share of
  today's 180–235 ms. Step 0 sets the real number. Target: a measurable drop on both
  GPUs, with no regression in `gguf_open`/`cuda_init`.
- **`generate`'s first token.** With Q6_K, the tied LM head's 622 MB dequant in
  `prompt_eval_ms` becomes a 128 MB upload. That is likely the largest single
  cold-start gain on `generate`, which the README's headline comparison uses.
- **VRAM.** Mistral 7B Q4_K_M (~4.4 GB file) goes from ~29 GB `f32` (OOM on a T4) to
  roughly 8 GB in the prototype (its Q6_K tensors still `f32`) and ~5 GB with Q6_K. 27B
  models become plausible on 24 GB cards once the hybrid path is converted too; the
  prototype does not do that.
- **Decode speed.** It should approach the byte ratio on bandwidth-bound GPUs. Real
  numbers come from `reflex bench` at 128 tokens.

Risks:

- **Numerics.** Summation order changes in GEMV, and on the scratch path cuBLAS sees
  identical `f32` inputs. Greedy ids should hold, but logit checksums against the T4
  references will shift. Expect to re-baseline checksums, not token ids.
- **Short-prompt prefill.** If the multi-row kernel is slow, the dequant-to-scratch path
  would move the `f32` write cost from load into `prompt_eval_ms` rather than remove it.
  The crossover must be measured on System1's actual ~12-token prompt.
- **A slow fused kernel.** On a GPU where the in-register Q4_K unpack makes GEMV
  compute-bound (older or smaller parts), decode could regress for tiny models. Bench
  on the T4 (the weakest supported card) first.
- **Scope creep.** Every architecture's forward pass takes `&Weight`. The enum keeps
  them compiling, but each `match` on `WeightData` is a place a non-prototype path could
  panic. Make the unsupported arm a clear `ReflexError`, never `unreachable!`.
- **Flag drift.** Two residency modes double the GPU test matrix. Retire `f32` residency
  for supported formats once the quantized path is verified, as with
  `REFLEX_ATTN_KERNEL=legacy`.

## 5. Verification plan

On the stopped T4 (`g4dn.xlarge`, sm_75), then the A10G for the second compute
capability:

1. **Kernel tests (GPU, `#[ignore]`d like the others).** `gemv_q4k` against
   `dequantize_q4k_kernel` + `gemv_kernel`, on every Q4_K tensor of Qwen3-0.6B with
   random `x`: max relative error below 1e-5. The multi-row kernel for `rows` 1–8
   against the same, and the scratch + `Sgemm` path against `Model::gemm` on the `f32`
   weight.
2. **Existing GPU suite with the flag on.** `scripts/gpu_nightly_tests.sh`, including
   `prefill_dense_batched_matches_sequential_prefill`, with `REFLEX_QUANT_RESIDENT=1`.
3. **`reflex check` against llama.cpp.** Run `"The capital of France is" --max-tokens 8`
   on Qwen3-0.6B and TinyLlama (dense path). Token ids must equal llama.cpp's and the T4
   reference ids recorded for the fatbin work. Report logit-checksum deltas.
4. **New coverage.** Mistral 7B Q4_K_M on the T4: it must load (OOM today) and `check`
   must match llama.cpp's greedy ids.
5. **LoRA.** `--lora` on a quantized-resident model must give the same output as the
   flag-off run. The real hybrid adapter is out of scope; use the tiny Qwen3-MoE LoRA
   fixture's dense counterpart or a synthetic dense adapter.
6. **Bench, interleaved flag-on/flag-off, n=10 per round, 2+ rounds per GPU.**
   `bench_cold_start_phases_system1.sh` and `bench_cold_start_phases.sh` (`generate`),
   the step-0 sub-phases, `reflex bench` decode ms/token at 128 tokens, and VRAM after
   load (`cuMemGetInfo`).

Pass bar for the prototype: steps 1–5 green, plus a measured, reported result in step 6
whichever way it goes. A regression is reported, not tuned away silently.

## 6. Prototype scope

- **Path:** dense Qwen3 (`load_dense_inner`, `forward_*_dense*`, `prefill_dense_batched`).
  The dense path also serves Llama/Mistral, so Mistral 7B is in reach.
- **Format:** Q4_K only. Q6_K, Q8_0, F32 norms and the LM head stay on today's `f32`
  path.
- **Flag:** `REFLEX_QUANT_RESIDENT=1` (env var, read once at load, same pattern as
  `REFLEX_ATTN_KERNEL`). Off by default. MoE/hybrid/MLA ignore it with a notice.
- **Work items, in order:**
  0. Sub-phase load timers (useful on their own; can merge first).
  1. `WeightData` enum and arena allocation; pipeline writes raw blocks into the arena.
  2. `gemv_q4k_kernel` (decode) and its tests.
  3. Multi-row `gemv_q4k` for prefill ≤ 8 rows, plus the scratch + `Sgemm` path above that.
  4. `apply_lora` materialization for quantized targets.
  5. Verification and bench (section 5), with results written into this note.
- **Not in the prototype:** Q6_K/Q8_0 kernels, the quantized LM head, MoE/MLA/hybrid,
  Q8_1 activations / `dp4a`, `f16` tensor-core scratch, `cuMemHostRegister`.

## Docs and rules this changes

When the flag becomes the default, these statements stop being true and need rewriting.
While it is opt-in, add a note that they describe the default path:

- [docs/DEVELOPMENT.md](../DEVELOPMENT.md), "Model loading: weights stay on the GPU":
  "dequantizes every weight tensor once and uploads it once, as a device-resident `f32`
  buffer". Its "never copy weights host-to-device per call" rule still holds (the
  scratch path is device to device), but should say so explicitly. "tensor N+1's copy
  overlaps tensor N's dequant kernel" becomes "overlaps tensor N's copy" for quantized
  tensors.
- [README.md](../../README.md), supported models: "Weights are held as `f32` on the GPU
  ... about 4 bytes per parameter whatever the file's quantization".
- [docs/runpod-llamacpp-comparison.md](../runpod-llamacpp-comparison.md), caveats:
  "Reflex dequantizes every weight to `f32` on the GPU at load".
- Code comments: `Weight`'s and `WeightLoadPipeline`'s doc comments in
  `src/model/loading.rs`, `gemv_kernel`'s header in `src/kernels_cuda/gemv.cu`,
  `expert_weight_view` (once MoE is converted), and `Model::apply_lora`'s doc comment.

## Prototype results

Measured 2026-10-02 on the `g4dn.xlarge` T4 (sm_75, driver 595.91.07, CUDA 13.2,
`REFLEX_CUDA_ARCH=sm_75`), Qwen3-0.6B Q4_K_M unless stated. "master" is the build without
the prototype; the prototype with the flag off matched it everywhere (same token ids and
checksums, same phase timings within 1 ms).

### Correctness

- New GPU tests pass: per-tensor GEMV/GEMM against the `f32` path (decode, fused 2–13 rows,
  scratch + cuBLAS; max relative difference < 1e-4) on Qwen3-0.6B, TinyLlama and Mistral
  7B, and 16-token greedy generation identical with the flag on and off. The
  batched-vs-sequential prefill oracle passes with the flag on. The full GPU suite passes
  with the flag off (8 pass; 11 skipped for fixtures not on the instance).
- `reflex check "The capital of France is" --max-tokens 8`: token ids identical to master
  on Qwen3-0.6B and TinyLlama (both also match llama.cpp's `llama-simple` text). Logit
  checksums move by ~1e-6 relative.
- **Mistral 7B now loads on the T4** (master runs out of memory at layer 18) and fits in
  7,756 MiB. Its greedy text does not match llama.cpp's (`" Paris

The 1"` vs. `" a city
  of many faces"`). The kernel matches the `f32` path on Mistral's own tensors, so this is
  not the quantized path; Mistral could not run on this engine's test GPUs before, and the
  difference is still open.

### Where model load goes (step 0, `REFLEX_LOAD_PROFILE=1`, `system1`, n=5)

| part of `model_load_ms` | f32 (today) | Q4_K resident |
|---|---|---|
| host: copy `token_embd` into an owned `Vec` | **~103 ms** | ~103 ms |
| host: fill pinned staging (263 MB from the mmap) | ~54 ms | ~53 ms |
| host: other weight-loop work (allocs, launches, waits) | ~10 ms | ~4 ms |
| GPU: H2D copies (overlapped) | ~43 ms | ~43 ms |
| GPU: dequant kernels (overlapped) | ~65 ms | ~27 ms (Q6_K only) |
| module loads (+ the 263 MB arena alloc when on) | ~3.4 ms | ~14.8 ms |
| tensors dequantized to `f32` / kept quantized | 196 / 0 | 28 / 168 |

The load is **host-bound**: the GPU finishes its queued work before the host finishes
enqueueing (the end-of-load drain measured 0 ms). The single largest cost is copying the
127.6 MB `token_embd` out of the mmap (`LazyTokenEmbedding::new`'s `to_vec()`), which runs at
~1.2 GB/s and so is probably dominated by first-touch page faults. It is independent of
weight residency. The pinned fill (~4.8 GB/s) looks similar.

### Cold start, VRAM, decode

| | master (`f32`) | Q4_K resident |
|---|---|---|
| `system1` model load, p50 | 230.7–231.4 ms | 215.6–216.6 ms |
| `system1` prompt eval, p50 (12-token prompt, fused path) | 39.2 ms | 49.4 ms |
| `system1` total, p50 | 446.7–447.6 ms | **442.2–442.3 ms** |
| `generate` total, p50 | 695.4–696.6 ms | 695.7–696.2 ms |
| model-resident VRAM (`reflex bench`) | 1,708 MiB | **460 MiB** |
| decode, ms/token (cold `check`, 128 tokens, n=3) | 12.5–12.8 | **9.2–9.3** |
| warm prefill 29 / 113 / 449 tokens | 22.7 / 59.4 / 264.9 ms | 35.9 / 72.6 / 282.8 ms (scratch path) |

(`system1`/`generate`: p50 of n=10, two interleaved rounds; ranges are the two rounds.)

- Cold start improves by ~1% on `system1` and is unchanged on `generate`: the load saves
  ~15 ms, and the short-prompt prefill gives back ~10 ms.
- The fused kernel wins at 12 rows but loses badly at 29 (63.9 vs. 35.9 ms warm), because
  each warp re-reads the activations per output row. Above the crossover, the scratch path
  costs ~13–18 ms more than `f32` per prefill (dequantizing 1.5 GB per prompt). The default
  crossover is 16 rows, between the two measured points.
- The first kernel version (byte-wise, uncoalesced) was 2x slower than `f32` at decode;
  coalesced 16-byte loads fixed that. The existing `dequantize_q4k_kernel` has the same
  uncoalesced-write pattern (one thread per 256-element block), which made the first
  scratch path cost ~41 ms per prefill; a one-thread-per-element version cut it to ~13 ms.

### What this says about the next steps

1. **The cold-start win is elsewhere.** Two costs that weight residency doesn't touch are
   larger than anything it removed: the ~103 ms `token_embd` copy in model load, and, on
   `generate`, prompt eval of 288 ms against `system1`'s 39 ms. That ~250 ms gap is most
   likely the tied LM head being expanded to 622 MB of `f32` on the first token (not yet
   profiled separately). Keeping the mmap alive instead of copying `token_embd`, and a
   Q6_K LM head kept quantized, go straight at both.
2. Q6_K kernels (layers and LM head) for the rest of the VRAM and decode gains.
3. Coalesce the existing load-time dequant kernels; the GPU side of the load is hidden
   behind the host today, but this matters once the host side shrinks.
4. A multi-row kernel that reuses activations across output rows (tiling, as llama.cpp's
   MMQ does) would make prefill faster than `f32`, not just close to it.

### Follow-up: `token_embd` kept in the mmap, quantized LM head (2026-10-02)

Two changes the step-0 profile pointed at:

- **`token_embd` stays in the mmap.** `GgufFile` shares its mapping (`SharedBytes`), and
  `LazyTokenEmbedding` keeps a handle instead of copying 127.6 MB out. This applies to every
  architecture and to the default (`f32`) path. The model keeps the file mapped for its
  lifetime, as llama.cpp does by default.
- **Quantized LM head (flag on).** A tied Q6_K head is uploaded as raw blocks on its first
  full-vocab use instead of being expanded to 622 MB of `f32`, and an untied Q6_K/Q4_K
  `output.weight` goes into the arena at load. `gemv_q6k_kernel` reads Q6_K directly, with
  the same decode expression as `dequantize_q6k_kernel`.

Same T4 and model; p50 of n=10, two interleaved rounds (ranges are the two rounds):

| | master | new, flag off (default) | new, flag on |
|---|---|---|---|
| `system1` model load | 235.8–236.3 ms | 200.4–205.0 ms | 202.6–202.7 ms |
| `system1` total | 451.7–453.3 ms | **418.7–422.7 ms** | 428.1–429.2 ms |
| `generate` model load | 233.2–239.5 ms | 200.6–202.2 ms | 200.4–204.5 ms |
| `generate` prompt eval | 288.0–288.3 ms | 294.9–295.6 ms | **54.9–55.0 ms** |
| `generate` total | 696.9–706.3 ms | 672.4–675.2 ms | **430.9–437.3 ms** |
| decode, ms/token (n=3) | 12.9–13.0 | — | **7.73–7.80** |
| model-resident VRAM | 1,708 MiB | — | 460 MiB |

Mistral 7B with the flag on: 7,276 MiB resident (was 7,756 with an `f32` `output.weight`).

- The `token_embd` change alone cuts ~30–35 ms of model load on every path. Reading the
  file pages later, on `generate`'s first full-vocab use, costs only ~7 ms, so most of the
  old ~103 ms was faulting in and zeroing the freshly allocated `Vec`, not reading the file.
- With the flag on, `generate`'s first token no longer pays for an `f32` LM head:
  prompt eval drops from ~295 ms to ~55 ms and the cold start falls 38% against master.
  Decode improves 40%.
- With the flag on, `system1` is now ~7–10 ms *slower* than with it off: the load-time
  saving is gone (the load is host-bound and no longer copies `token_embd` either way),
  while the short-prompt fused prefill still costs ~10 ms more than `f32`.

Correctness: every token id is unchanged across master, flag off and flag on for Qwen3-0.6B,
TinyLlama, Qwen3.5-0.8B (hybrid), the tiny MLA fixture and the tiny Qwen3-MoE fixture.
Flag-off checksums are identical to master; flag-on checksums move ~1e-6 relative (the
hybrid/MoE/MLA paths ignore the flag). New test `quant_resident_lm_head_matches_f32` passes on
Qwen3-0.6B (tied) and Mistral 7B (untied). The GPU suite, now with the tiny fixtures
present, gives 18 pass and 3 skip with the flag off.

### On top of `f16` weight storage (2026-10-06)

Rebuilt on the `f16` default (see the status note at the top) and re-measured on the same
dedicated T4 (`g4dn.xlarge`, driver 595.91.07), portable-PTX builds, Qwen3 Q4_K_M files.

**Correctness.**

- **Flag off:** identical to master on 56 of 56 comparisons. That's `reflex check` token ids
  plus first-token logit checksum, and `system1` scores, across seven models and fixtures,
  three prompts, `--weights f32` and `f16`.
- **GPU suite:** 22 pass, 2 skip in both `--weights` modes. That includes the three
  quantized-resident tests, which now cover the `f16` scratch path.
  `quant_resident_gemv_matches_f32` and `quant_resident_lm_head_matches_f32` also pass on
  Mistral 7B.
- **Flag on, greedy tokens:** 32 greedy tokens match plain `--weights f32` on every prompt for
  Qwen3-0.6B, Qwen3-1.7B and TinyLlama, with either `--weights` mode.
- **Mistral 7B:** can't load as plain `f32` on a 16 GB card; its flag-on `f32` and `f16`
  runs agree with each other. They do not match llama.cpp, and the text is garbled
  (`", ithin ithis ithland, ithere ithwas"`). That looks like the known Mistral mismatch,
  which points at the tokenizer or detokenizer rather than these kernels. It has not been
  confirmed with the flag off, which needs ~14.5 GB in `f16`.

**Cold start**, Qwen3-0.6B, `--weights f16`, n=10, two interleaved rounds, p50:

| | flag off | flag on |
|---|---|---|
| `system1` model load | 235.2 / 237.3 ms | 236.1 / 233.4 ms |
| `system1` scoring pass | 9.9 / 9.9 ms | **27.9 / 27.9 ms** |
| `system1` total | 420.9 / 425.1 ms | 441.9 / 438.1 ms |
| `generate` prompt eval | 247.4 / 247.2 ms | **46.6 / 46.6 ms** |
| `generate` total | 661.3 / 659.9 ms | **461.3 / 459.6 ms** |

**Memory and decode** (`reflex bench`, warmup 3, 20 iterations; prompt buckets 29 / 113 / 449):

| model | flag | weights | resident MiB | model load ms | decode ms/token | warm prompt p50 ms |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | off | f32 | 1,708 | 202 | 12.7 / 13.6 / 15.9 | 23.5 / 61.4 / 274.9 |
| Qwen3-0.6B | off | f16 | 876 | 237 | 8.19 / 8.94 / 11.33 | 14.4 / 35.4 / 192.0 |
| Qwen3-0.6B | on | f32 | 460 | 213 | 7.89 / 8.83 / 11.01 | 35.6 / 74.2 / 282.3 |
| Qwen3-0.6B | on | f16 | **364** | 236 | **7.28** / 8.22 / 10.42 | 25.9 / 47.8 / 204.9 |
| Qwen3-1.7B | off | f32 | 5,452 | 699 | 30.9 / 31.5 / 33.9 | 60.0 / 169.2 / 640.0 |
| Qwen3-1.7B | off | f16 | 2,732 | 499 | 18.0 / 18.6 / 21.0 | 36.1 / 86.8 / 376.3 |
| Qwen3-1.7B | on | f32 | 1,484 | 459 | 16.9 / 17.7 / 19.8 | 99.8 / 207.2 / 684.0 |
| Qwen3-1.7B | on | f16 | **1,100** | 410 | **15.2** / 15.8 / 18.1 | 74.2 / 125.8 / 416.6 |

Mistral 7B, flag on, `--weights f16`: 5,356 MiB resident (7,276 MiB with the prototype's
`f32` default), decode 51.3 / 52.8 / 59.9 ms/token, model load 2.29 s.

What this says against the `f16` baseline rather than `f32`:

- **VRAM:** the flag still saves 58% (0.6B) and 60% (1.7B).
- **Decode:** the margin shrinks from ~40% against `f32` to 11% (0.6B) and 15% (1.7B),
  since `f16` already halved the weight bytes read per token.
- **`generate`'s cold first token:** 30% faster. A tied LM head stays Q6_K instead of being
  dequantized to the full vocab on first use.
- **`system1` gets worse:** it is ~17 ms slower than the `f16` default (was ~7–10 ms against
  `f32`). Its 12-token scoring pass runs on the fused Q4_K kernel at 27.9 ms, while `f16`'s
  cuBLAS path, now warmed at load, takes 9.8 ms.
- **Warm prefill is slower at every length:** 1.8x at 29 tokens and 1.1x at 449 for 0.6B.
  The fused kernel and the scratch path both lose to `f16` weights read directly. That's
  the "tiled multi-row Q4_K kernel" gap from the earlier results, now measured against the
  faster baseline.
