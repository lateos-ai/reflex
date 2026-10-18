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

## What's left

Ranked by the round 2 profile. Status as of 2026-10-18:
- **Cut the remaining per-op host cost.** Reuse output buffers across steps instead of
  allocating them, and write K/V straight into the cache. Then use CUDA graphs, which
  would remove most of the 0.5-1.25 ms of GPU idle time per step.
- **f16 KV cache** (opt-in). Long contexts only: decode attention takes 1.5 ms per
  step on Qwen3-0.6B at ~1,550 positions. It changes outputs, so it can't be the
  default.
- **Not planned this round.** These are all bit-identical:
  - a 4-rows-per-warp `gemv_q6k` (QR only, ~0.3 ms; the Q6_K LM head runs at about
    61% of bandwidth);
  - greedy argmax on the device (~0.2 ms on a 152K vocabulary);
  - fused Q/K/V and gate/up GEMVs (~2-3%).
- **Read fewer weight bytes.** *Partly done (2026-10-15):* the decode Q4_K GEMV is
  1.6-1.8x faster, and Q6_K layer tensors stay quantized. See
  [quantized-resident-weights.md](quantized-resident-weights.md), "Faster Q4_K/Q6_K
  decode".
