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

## What's left

At short context Qwen3-0.6B now decodes in 6.96 ms, of which 4.0 ms is f16 weight
reads. Next candidates, roughly in order of payoff:
- **Read fewer weight bytes.** Make the Q4_K-resident decode GEMV bandwidth-efficient;
  at ~0.56 bytes per weight it would read about a third of today's bytes.
- **Cut the remaining per-op host cost.** Reuse output buffers across steps instead of
  allocating them, fuse the Q/K/V GEMVs, and eventually use CUDA graphs.
- **Pick the greedy token on the device.** That would replace the 600 KB logits copy
  and the host argmax at the end of each step.
