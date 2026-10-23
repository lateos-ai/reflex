# Decode speed: where a decode step's time goes

After tiled prefill attention, a long agent hop on Qwen3-0.6B spent most of its time
decoding: 11.3 ms per token at ~1,550 cached positions on a Tesla T4, against
llama.cpp's 4.9 ms (see [agent-relay.md](agent-relay.md)). This note records where
that time went and the first two fixes.

## Profile (Tesla T4, 2026-10-15)

`nsys profile -t cuda` of `reflex generate` with 256 decode steps. The model is
Qwen3-0.6B Q4_K_M with the default f16 weights; the prompt is either short or the
~1,550-token team-rules prompt from the relay benchmark. Per decode step:

| | short context (8.1 ms) | ~1,550 positions (11.3 ms) |
|---|---:|---:|
| `gemv_f16_kernel` (all matmuls and the LM head) | 4.0 ms | 4.0 ms |
| decode attention (`attention_online_kernel`) | 0.7 ms | 3.8 ms |
| `rmsnorm`, RoPE, adds, SiLU | 0.8 ms | 0.8 ms |
| memsets from zero-initialized output buffers | 0.4 ms | 0.4 ms |
| GPU idle | ~2 ms | ~2 ms |

- **The GEMVs are at the memory-bandwidth floor.** A step reads ~1.2 GB of f16
  weights (the tied 151,936-row LM head is 311 MB of it) in 4.0 ms, about 300 GB/s of
  the T4's 320. They can only get faster by reading fewer bytes. The
  quantized-resident Q4_K path is the lever for that.
- **Decode attention ran at ~28% of bandwidth.** `attention_online_kernel` gives each
  query head its own blocks, so a KV head's rows are read once per query head of its
  group (twice for Qwen3-0.6B). Each warp also walks its positions one at a time, with
  a full memory latency exposed per position: 142 µs per layer at 1,550 positions.
- **Per-op overhead starves the GPU.** Each op allocates its output, zeroes it,
  launches and frees, about 10 µs of host time and 4 driver calls. That is about as
  long as the op's GPU work, so the host never gets ahead, and the GPU idles about a
  quarter of each step. The zeroing memsets alone were 78,000 calls in 257 tokens:
  0.4 ms of GPU time and ~1.2 ms of host time per token.

## Fixes

1. **No zeroing for outputs a kernel fully writes** (`Model::alloc_out`). This covers
   RMSNorm, the f32/f16 GEMVs, SiLU-and-mul, and `attention_online`'s output and
   partials. Each of those kernels writes every element. The results are unchanged
   bit for bit.
2. **GQA decode attention** (`kernels_cuda/attention_decode.cu`, `attention_decode<DL,
   HB>`):
   - **One read per K/V row.** A block handles up to 8 query heads that share a KV
     head, and reads each K and V row once for all of them.
   - **Two loads in flight.** Each warp takes two positions per iteration and issues
     both rows' loads before using them.
   - **Splits sized to the GPU.** The launcher picks splits for about 4 blocks per SM,
     at least 32 positions each. The splits are merged by the existing
     `attention_online_combine_kernel`.
   - **Coverage.** There are kernels for head dims 64, 128 and 256 with 1, 2, 4 or 8
     heads per block. The heads per block is the most that divides the group, so
     groups of 6 or 7 use 2 or 1.
   - **Lazy loading.** The module loads on the first decode step, so time to first
     token never pays for it.
   - **Dispatch** is in `Model::attention`: dense, MoE, Kolibri-1 and Qwen3.5 decode.
     Gemma 4, Granite 4.0-H and MLA keep `attention_online`.
   - **`REFLEX_ATTN_DECODE=online`** restores the old kernel for A/B runs.

## Results (Tesla T4, 2026-10-15)

Decode milliseconds per token over 256 steps, as (last token − first token) / 256.
Three interleaved runs each, which agree within 0.3%:
- *master* is the code before this work;
- *no memset* is fix 1 alone (`REFLEX_ATTN_DECODE=online`);
- *both* is fixes 1 and 2.

| model | context | master | no memset | both | change |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B (d128, group 2) | short | 8.15 | 7.66 | 6.96 | −15% |
| | ~1,550 | 11.43 | 10.90 | 8.39 | −27% |
| TinyLlama 1.1B (d64, group 8) | short | 10.40 | 9.99 | 9.76 | −6% |
| | ~1,550 | 13.50 | 13.09 | 10.44 | −23% |
| Qwen3.5-0.8B (d256, group 4) | short | 10.20 | 9.82 | 9.53 | −7% |
| Mistral 7B, `REFLEX_QUANT_RESIDENT=1` (d128, group 4) | short | 48.11 | 47.95 | 47.26 | −2% |
| | ~1,550 | 54.88 | 54.60 | 49.36 | −10% |

Decode attention on Qwen3-0.6B at ~1,550 positions went from 142 µs to 53 µs per
layer (`attention_decode_d128_h2_kernel`, plus 4.5 µs for the combine).

The long-prompt Qwen3.5 run is left out of the table: the model stopped early, which
breaks the per-token arithmetic.

**Correctness.**
- On random Q/K/V, the GQA kernel matches `attention_online_kernel` to ≤1.7e-7 across
  11 shapes. The shapes cover d64/128/256; groups 2, 4, 6, 7, 8 and 16; 1 to 5,000
  positions; and a sliding window.
- 32 greedy tokens after an ~800-970-token prompt (f32 weights) are identical with
  either kernel on Qwen3-0.6B, TinyLlama and Qwen3.5-0.8B.
- The GPU nightly suite passes (69/69 that ran).
- No kernel spills on sm_75/86/90.

## Round 2 profile (Tesla T4, 2026-10-18)

Same method, on the code after the GQA kernel and the faster Q4_K/Q6_K decode GEMVs,
with Qwen3-0.6B and TinyLlama, each with f16 weights and `REFLEX_QUANT_RESIDENT=1`
(QR). Per decode step:

| | Qwen3 f16 short | Qwen3 QR short | Qwen3 QR ~1,550 | TinyLlama f16 short |
|---|---:|---:|---:|---:|
| wall | 7.15 ms | 5.40 ms | 6.64 ms | 10.02 ms |
| GPU busy | 6.29 ms | 4.15 ms | 5.67 ms | 9.54 ms |
| GPU idle | 0.86 ms | 1.25 ms | 0.97 ms | 0.48 ms |
| host work between logits copies | 3.9 ms | 3.9 ms | 3.8 ms | 2.7 ms |
| launches / allocations / memsets | 504 / 393 / 0 | 504 / 393 / 0 | 506 / 395 / 0 | 352 / 265 / 0 |
| `rmsnorm_kernel` | 0.75 ms | 0.76 ms | 0.79 ms | 0.93 ms |

- **RMSNorm was slow because of its serial sum, not its launch count.** One thread
  sums each row's squares in index order (that order keeps every architecture's output
  bit-identical), at ~10 ns per element: 6.7 µs per 1,024-wide row, 20.6 µs at 2,048.
- **QR at short context is the most host-sensitive case.** Its GPU work (4.15 ms) is
  close to the host's (3.9 ms), so it idles 1.25 ms per step. With f16 weights, the
  host stays about 2.4 ms ahead, and the idle time is mostly the ramp at the start of
  each step, after the logits copy.

## Round 2 fixes

3. **Faster, fused decode norms.** These are bit-identical.
   - **`rmsnorm_kernel` sums the same way, but faster.** The single summing thread
     reads the staged row four floats at a time, unrolled (`kernels_cuda/rmsnorm.cuh`).
     The additions are the same, in the same order. This applies to every architecture,
     in prefill and decode.
   - **The residual add is fused into the next norm**
     (`kernels_cuda/decode_ops.cu`, `add_rmsnorm_kernel`). Each layer returns its FFN
     output unadded, and the next layer's attention norm adds it. The O-proj add is
     fused the same way into the FFN norm.
   - **Q-Norm and K-Norm run in one launch** (`rmsnorm_pair_kernel`).
   - **Coverage.** Dense, MoE and Kolibri-1 decode use the fused ops. The module loads
     lazily on the first decode step. `REFLEX_DECODE_FUSE=0` turns the fusion off for
     A/B runs.
   - **Effect per step.** On Qwen3-0.6B, launches drop from 504 to 416 and norm plus
     add GPU time from 0.85 to 0.51 ms. On TinyLlama they drop from 352 to 304 and from
     0.99 to 0.68 ms.

Decode ms/token (as above; medians of 3 interleaved runs with prebuilt binaries;
greedy ids of all 257 tokens identical to the previous commit in every run):

| model | context | before | after | change |
|---|---|---:|---:|---:|
| Qwen3-0.6B f16 | short | 6.91 | 6.60 | −4.4% |
| | ~1,550 | 8.33 | 8.00 | −4.0% |
| Qwen3-0.6B QR | short | 4.92 | 4.62 | −6.1% |
| | ~1,550 | 6.40 | 6.10 | −4.7% |
| TinyLlama f16 | short | 9.87 | 9.57 | −3.0% |
| | ~1,550 | 10.35 | 10.06 | −2.8% |
| TinyLlama QR | short | 5.95 | 5.74 | −3.5% |
| | ~1,550 | 6.56 | 6.32 | −3.6% |
| Qwen3.5-0.8B f16 (faster sum only) | short | 9.47 | 9.35 | −1.2% |

The fused norm still takes 7.8 µs per 1,024-wide row and 15 µs at 2,048. Most of that
is the serial summation chain, which a bit-identical kernel has to keep.

4. **A reused decode workspace** (`src/model/decode_step.rs`).
   - **What changed.** `generate`'s dense decode step now keeps every intermediate
     in a `DecodeWorkspace`, allocated on the first decode step and reused after that.
     It writes K and V straight into their cache rows: the K/V GEMV outputs, the
     bias adds, K-Norm and RoPE all target the row.
   - **Unchanged.** The kernels, their launch geometry and their order are the same
     as the allocating path's.
   - **Effect per step.** About 390 allocations and frees and 56 device-to-device
     copies go away on Qwen3-0.6B.
   - **What it's for.** It leaves the step free of allocations, which CUDA graph
     capture needs.
   - **Scope.** Models whose layers are all dense (Qwen3, Qwen2, Llama, Mistral),
     with the online attention kernels. MoE and Kolibri-1 (host-side routing) keep the
     allocating path, as do system1 scoring and the sequential prefill.
   - **`REFLEX_DECODE_WS=0`** turns it off.

| model | context | before | after | change |
|---|---|---:|---:|---:|
| Qwen3-0.6B f16 | short | 6.62 | 6.42 | −2.9% |
| | ~1,550 | 8.01 | 7.83 | −2.3% |
| Qwen3-0.6B QR | short | 4.64 | 4.48 | −3.4% |
| | ~1,550 | 6.13 | 5.97 | −2.6% |
| TinyLlama f16 | short | 9.58 | 9.44 | −1.5% |
| | ~1,550 | 10.07 | 9.93 | −1.4% |
| TinyLlama QR | short | 5.80 | 5.70 | −1.7% |
| | ~1,550 | 6.42 | 6.34 | −1.3% |

TinyLlama gains less than 2%: it has fewer, larger layers, so its GPU work hides more
of the host cost. The workspace stays regardless, because CUDA graphs need it.
- **Bit-identity checks.**
  - Logits of every decode step are bit-identical to the allocating path on
    Qwen3-0.6B and TinyLlama, with f32 and f16 weights, for short and ~800-token
    prompts (split attention).
  - The qwen2 fixture (Q/K/V biases) gives the same result.
  - Greedy ids are identical in every A/B run.

**Tried and dropped: CUDA graphs.** The workspace step was captured and replayed as a
graph with one `cuGraphLaunch`.
- **How it worked.**
  - Each step was captured again and applied with `cuGraphExecUpdate`, because RoPE's
    position, the cache rows and the attention splits change every step.
  - That capture ran while the previous step was on the GPU, with two executable
    graphs alternating.
  - Capture ran on a side stream: the device's work stream is the legacy default
    stream, which can't be captured.
- **Correctness.** Logits were bit-identical to the plain path.
- **Speed.** It gained only 0.1-1.6% on the T4 (Qwen3-0.6B f16 −1.4%/−1.6%, QR
  −0.5%/−1.0%, TinyLlama ≤0.4%), below the 2% bar.
- **Why so little.** Launching a ~420-node graph costs about 0.6 ms of host time on
  this driver. That launch sits between picking the token and the GPU starting, which
  is about as long as the streamed launches took to get ahead of the GPU. GPU idle
  stayed at ~0.6 ms per step (QR, short context).
- **What might still help.** A step split into a few smaller graphs, or kernels that
  read the position from device memory, so that a step could be submitted before its
  token is known.

## f16 KV cache (opt-in, `REFLEX_KV_CACHE=f16`)

At ~1,550 positions, decode attention reads ~355 MB of f32 K/V per Qwen3-0.6B step,
1.5 ms of it. `REFLEX_KV_CACHE=f16` stores the working cache in f16 and halves that.
It changes outputs, so it is opt-in. The default stays f32 and byte-identical.

**What is f16.**
- Every attention layer's K and V: dense, MoE, Kolibri-1, and Qwen3.5's gated
  attention.
- MLA's compressed latent.
- Not Qwen3.5's GatedDeltaNet `conv_state`/`recurrent`, which stay f32 as in f16 cache
  files.
- Activations, scores, softmax and accumulators stay f32. The kernels widen each K/V
  element as they load it.
- Every cache write rounds to nearest even, saturating at ±65504, with the same
  expression as f16 cache-file export. NaN stays NaN.

**Per process.**
- The setting is read once at load.
- The one pinned prefix holds its state in the same type, at half the VRAM, and
  every request in the process uses it as is.
- Gemma 4, Granite 4.0-H and `REFLEX_ATTN_KERNEL=legacy` refuse the setting at load.

**Cache files.** No new format is needed.
- With an f16 cache, `export_kv` / `--export-kv` default to an f16 file (versions
  5/6/7), written as it is.
- `export_kv_dtype: "f32"` widens the values exactly instead.
- Importing an f16 file into an f16 cache is a plain copy.
- An f32 file is rounded on the device on the way in, the same rounding as every
  cache write.
- With the default f32 cache, nothing about files changes.

**Kernels.**
- The attention kernels are templated on the cache's element type
  (`attention_decode.cuh`, `attention_online.cuh`, `attention_tiled.cuh`, with loads
  from `kv_load.cuh`).
- The f32 instantiations compile to byte-identical cubins on sm_75/86/90.
- The f16 kernels are separate modules: `attention_decode_f16.cu`, `kv_f16.cu` (online
  attention, the rounding stores, MLA's batch write) and `attention_tiled*_f16.cu`.
  An f16 process loads them the first time it needs them, so an f32 process loads and
  JIT-compiles exactly what it did before.
- One `kv_store_kv_f16_kernel` launch writes a layer's K and V rows. With two
  launches, short-context decode was 1% slower.

**Accuracy.** Measured as for phase F's f16 cache files: f32 weights, a ~700-token
prompt, next-token logits with an f16 cache against an f32 one, and 32 greedy tokens.

| model | rel L2 | max abs | greedy ids |
|---|---:|---:|---|
| Qwen3-0.6B | 6.8e-4 | 8.3e-3 | 32/32 identical |
| TinyLlama 1.1B | 1.3e-4 | 1.8e-3 | 32/32 identical |
| Qwen3.5-0.8B (hybrid) | 3.0e-4 | 3.5e-3 | 32/32 identical |
| Qwen3-MoE fixture | 1.1e-6 | 8.8e-9 | 32/32 identical |
| MLA fixture | 9.8e-7 | 1.7e-8 | 32/32 identical |

These are about the size of what phase F's f16 files cost a resumed prefix (6e-4 on
Qwen3-0.6B), as expected: an f16 working cache rounds the same values, plus the ones
it writes while decoding.
- **f16 pinned prefix vs its own f16 file.** Resuming from the f16 file is
  bit-identical to resuming from the pinned prefix.
- **f32 file into an f16 cache vs the f16 pinned prefix.** rel L2 5.0e-4 on
  Qwen3-0.6B and 2.1e-4 on Qwen3.5, with the same 32 tokens.

**Speed.** Same binary with and without `REFLEX_KV_CACHE=f16`. Decode ms/token,
medians of 3 interleaved runs, greedy ids identical in every run:

| model | context | f32 cache | f16 cache | change |
|---|---|---:|---:|---:|
| Qwen3-0.6B f16 | short | 6.40 | 6.41 | +0.2% |
| | ~1,550 | 7.80 | 7.18 | −8.0% |
| Qwen3-0.6B QR | short | 4.43 | 4.43 | 0.0% |
| | ~1,550 | 5.90 | 5.31 | −10.1% |
| TinyLlama f16 | short / ~1,550 | 9.41 / 9.87 | 9.37 / 9.84 | −0.4% / −0.4% |
| TinyLlama QR | short / ~1,550 | 5.52 / 6.14 | 5.51 / 6.08 | −0.2% / −0.9% |

TinyLlama gains nothing. All 32 of its query heads share 4 KV heads, so each block of
its decode kernel puts 8 query heads on every K/V row it loads. That makes the kernel
compute-bound rather than bandwidth-bound, so halving the bytes doesn't shorten it.
The gain grows with context: decode attention is the part an f16 cache shrinks, and
it is the part that grows with context.

## Verification (Tesla T4, 2026-10-19)

The round 2 branch (fixes 3 and 4 and the f16 KV cache) against master:
- **GPU nightly:** 126 passed, 0 failed. 4 skipped, needing real Granite, IQ,
  DeepSeek-V2-Lite and Qwen2.5 files.
- **`reflex check`:** the same output (token ids, logit checksum) on Qwen3-0.6B,
  TinyLlama and Qwen3.5-0.8B, with f32 and f16 weights.
- **IPC requests:** a fixed set sent to `reflex stdio` gave byte-identical response
  lines and cache files on the same three models. It covers generate, streaming,
  sampling, `json_schema`, candidates, pin and use, errors, cache-file export and
  import, and a ~1,550-token prompt decoded directly and through a pinned prefix.
- **Agent relay bench** (Qwen3-0.6B, 30 hops, 32 tokens, the same texts as master):

| hop | master | round 2 | round 2 + f16 KV cache |
|---|---:|---:|---:|
| short prefix | 231.3 ms (6.95 ms/token) | 215.7 ms (6.44) | n/a |
| ~1,550-token prefix, recomputed | 505.7 ms (8.36) | 487.8 ms (7.84) | 465.3 ms (7.26) |
| ~1,550-token prefix, pinned | 296.0 ms (8.30) | 280.3 ms (7.77) | 259.3 ms (7.19) |

- **Cold start is unchanged.** `bench_cold_start_phases.sh`, 5 interleaved rounds of 2
  runs, gives a median first token of 647 ms on master and 649 ms on the branch, with
  every phase within noise. Both binaries were built with `--features ipc`.
- **Binary size.** The binary is 0.8 MB larger, from the embedded f16 kernel modules,
  which only `REFLEX_KV_CACHE=f16` loads.

## Round 3 (2026-10-20): QR as the default

Quantized-resident weights became the default (`REFLEX_QUANT_RESIDENT=0` opts out), so
this round's four decode changes were verified through that change rather than as a
standalone A/B. All four are bit-identical. The A10G cold-start, decode and VRAM table
(QR default against `REFLEX_QUANT_RESIDENT=0`) is in
[quantized-resident-weights.md](quantized-resident-weights.md#qr-as-the-default-2026-10-20);
Qwen3-0.6B decodes at 3.48 vs 3.81 ms/token and uses 270 vs 878 MiB, Mistral 7B at
15.21 vs 30.19 ms/token. The f16 opt-out path is unchanged.

- **4 rows per warp in `gemv_q6k_r1_kernel`** (`kernels_cuda/gemv_q4k.cu`). Each warp
  writes four output rows and loads each activation once per super-block for all four,
  where `gemv_q6k_kernel` handles one row per warp, so activation reads per weight fall
  4x. Q6_K layer tensors and the LM head use it in QR mode. The per-row dot product and
  reduction order are the same as `gemv_q6k_kernel`, so the output is bit-identical.
- **Async embedding upload from pinned memory** (`src/model/decode_step.rs`,
  `src/model/loading.rs`). The decode step's embedding row goes into a reused pinned
  staging slot and is sent on the load pipeline's copy stream; the compute stream waits
  on an event before its first kernel. That overlaps the ~4 KB H2D with the host
  argmax/picker work and removes the blocking `htod_sync_copy`, with no per-token
  allocation.
- **Fused Q/K/V and gate/up GEMVs** (`kernels_cuda/gemv_fused.cu`). One launch per
  projection group instead of three and two: `gemv_qkv_kernel` over separate q/k/v
  weight pointers, `gemv_gate_up_kernel` over the gate/up pair. Per-row dot products and
  reductions are the same, so outputs are bit-identical. Dense, Gemma 4, Granite 4.0-H,
  Qwen3.5 and MLA decode dispatch it by weight type (f32/f16/quantized);
  `REFLEX_DECODE_FUSE=0` restores the separate launches.
- **Streamed Q6_K block headers** (same kernel). The 4-row kernel no longer stages all
  four rows' block headers in registers before walking a super-block; each row's `d`
  and scales are read as the block is consumed. Bit-identical. This is commit `2bba001`,
  which still carries a WIP label; only the label is stale.

**Verification.** A10G 24 GB (`g5.xlarge`, sm_86): GPU suite 128 passed / 0 failed / 2
skipped in both QR-default and opt-out modes, and `reflex check` gives 47 comparisons
with 0 token-id divergences. That suite surfaced two decode-path bugs, both fixed here:
the f16 fused QKV/gate-up kernels reduced the warp sum twice (`warp_dot_f16` already
reduces), corrupting the opt-out workspace decode and JSON-schema tests; and one-row
`gemv()` used the older one-row-per-warp Q6_K kernel while `gemv_to()` used
`gemv_q6k_r1_kernel`, so the workspace and allocating paths differed with QR on. The f16
opt-out still takes the same path and produces the same results as before the round.

## What's left

Ranked by the round 2 profile and the round 3 QR-default numbers. Status as of
2026-10-20:
- **Close the QR warm-prefill gap.** *Done for Q4_K/Q6_K on Ampere and newer
  (2026-10-22/23):* prefill GEMMs now run a tensor-core kernel straight from the quantized
  blocks, see [quantized-mma-prefill.md](quantized-mma-prefill.md). On an A10G, QR warm
  prefill beats the f16 cuBLAS path (QR off) at every bucket: Qwen3-0.6B 5.5 / 9.6 /
  25.8 ms against 7.4 / 9.9 / 28.2 at 29 / 113 / 449 tokens, Mistral 7B 23.0 / 71.5 /
  186.6 ms against 39.5 / 75.0 / 195.4 at 33 / 129 / 513. Qwen3-0.6B `system1` prompt
  eval went from slower than QR off to 6.3 vs 19.1 ms. Still open:
  - On a T4, QR off stays faster from ~113 rows (Mistral 7B from 129 tokens,
    Qwen3-0.6B at 449). Long prompts there go back to the dequantize-to-scratch path
    ("Row cap" in that doc).
  - Q8_0 weights (Gemma 4) still use the fused GEMV and the scratch path.
- **Cut the remaining per-op host cost.** *Partly done:* fused norms and the reused
  workspace (round 2), and the embedding upload is now async from pinned memory
  (round 3). A single CUDA graph per step didn't pay (above). What remains of the GPU
  idle (~0.6-0.8 ms per step) is the step start: the logits copy, the host argmax and
  the first launches.
- **f16 KV cache.** *Done (opt-in, above):* −8% to −10% per token on Qwen3-0.6B at
  ~1,550 positions. Decode attention for GQA groups of 8 (TinyLlama) is compute-bound,
  so f16 storage doesn't help it; making it faster would mean less arithmetic per
  cached position.
- **Done in round 3.** These are bit-identical:
  - a 4-rows-per-warp `gemv_q6k` (QR only; the Q6_K LM head ran at about 61% of
    bandwidth);
  - fused Q/K/V and gate/up GEMVs (~2-3%).
- **Not planned.** Greedy argmax on the device (~0.2 ms on a 152K vocabulary) is still
  open.
- **Read fewer weight bytes.** *Partly done (2026-10-15):* the decode Q4_K GEMV is
  1.6-1.8x faster, and Q6_K layer tensors stay quantized. See
  [quantized-resident-weights.md](quantized-resident-weights.md), "Faster Q4_K/Q6_K
  decode".
