# Design: tensor-core prefill straight from quantized weights

Status: 2026-10-21. Step 1 (build floor, lazy module, dispatch hook) is in. Step 2's
Q4_K kernel (`gemm_q4k_mma_m{1..4}_kernel`, no split-K) and its tests
(`quant_mma_tests.rs`) are written but **not yet compiled or run on a GPU**; with QR
and `--weights f16` they are live for Q4_K weights at 2+ rows. Nothing is measured
yet; the numbers in "Where the gap comes from" are arithmetic on the published A10G
results, not a profile. The go/no-go run is `quant_mma_q4k_prefill_timings`.

## The problem

With quantized-resident (QR) weights, now the default, warm prefill is slower than with
`REFLEX_QUANT_RESIDENT=0` at every bucket the bench reports (A10G, `--weights f16`, p50,
from [quantized-resident-weights.md](quantized-resident-weights.md#qr-as-the-default-2026-10-20)):

| model | QR on | QR off | gap |
|---|---|---|---|
| Qwen3-0.6B, 29 / 113 / 449 tokens | 11.3 / 14.4 / 32.9 ms | 7.4 / 9.9 / 28.2 ms | 3.9 / 4.5 / 4.7 ms |
| Qwen3-1.7B | 26.4 / 31.6 / 68.8 ms | 13.2 / 18.4 / 55.6 ms | 13.2 / 13.2 / 13.2 ms |
| TinyLlama-1.1B, 33 / 129 / 513 | 18.3 / 26.6 / 49.5 ms | 9.0 / 17.5 / 40.0 ms | 9.3 / 9.1 / 9.5 ms |
| Mistral-7B, 33 / 129 / 513 | 104.7 / 140.6 / 261.3 ms | 39.7 / 75.2 / 195.5 ms | 65.0 / 65.4 / 65.8 ms |

`system1`'s ~12-row scoring pass is slower too (Qwen3-0.6B 23.1 vs 19.3 ms), which is
why it is the one cold-start metric QR loses.

## Where the gap comes from

Two separate paths, with two separate causes.

**Above 16 rows (every bench bucket): the scratch round trip.** `Model::gemm` sends
anything above `quant_fused_max_rows()` (16) to `gemm_quant_scratch`: dequantize the
whole tensor to f16 in a scratch buffer, then `cublasGemmEx`. The GEMM is then the
same call the QR-off path makes, so the gap is the scratch traffic alone, and it should
be a constant per model, independent of row count. It is: 3.9-4.7 ms on Qwen3-0.6B,
13.2 ms on every 1.7B bucket, ~65 ms on every Mistral bucket. The size matches too: for
Mistral's ~7B matrix weights, the scratch writes ~14 GB of f16 and cuBLAS reads it back,
~28 GB more traffic than QR off, ~47 ms at the A10G's 600 GB/s before dequant-kernel
inefficiency. Qwen3-0.6B's ~0.44B matrix weights give ~1.8 GB, ~3 ms.

So the bench gap is not about GEMV versus tensor cores at all; the fix is to stop
materializing the f16 matrix.

**1-16 rows (`system1`, short prompts): CUDA-core FMAs.** `gemv_q4k_kernel` (and
`gemv_q8_0`) read the quantized bytes once per 8 rows but do every product as an f32
FMA, and lose to f16 cuBLAS above ~4 rows (the Q8_0 micro-benchmark in the QR doc:
137-175 vs 520-542 GB/s at 12 rows). Q6_K has no multi-row kernel and goes through the
scratch path from 2 rows up, so a Q4_K_M file's `attn_v`/`ffn_down` (Q6_K in half the
layers) pay the round trip even in `system1`.

## Proposal

One kernel family, `gemm_q_mma`: read raw GGUF blocks, dequantize in registers to f16,
multiply on tensor cores (`mma.sync`) against f16 activations, accumulate in f32, write
f32 `y`. It replaces both the scratch path and the fused GEMV for 2+ rows when
`--weights f16` (the default). Decode (1 row) keeps the `_r1` GEMVs, and `--weights
f32` keeps today's paths unchanged.

### Numerics: the same math as the scratch path

Every input to the MMA is a value today's scratch path already produces:

- **Weights.** Each element is dequantized with the expression the scratch kernels use
  (`d1 * (float)q - m1` in f32 for Q4_K, the same for Q6_K and Q8_0), then rounded with
  `__float2half_rn`. That is what `dequantize_q4k_coalesced_f16_kernel` writes, so the
  f16 weight values are bit-identical to the scratch buffer's. Marlin's `lop3`/`half2`
  dequant tricks round differently, so they're out: per-element f32 math is cheap next
  to the weight load.
- **Activations.** `x` is cast by the existing `cast_act_f16` kernel (round to nearest,
  saturate at ±65504, folds into `f16_act_stats`), as `gemm_ex_f16` does. One cast per
  input, shared by Q/K/V and by gate/up. Fusing it into the GEMM is a later
  optimization.
- **Accumulation.** f32 (`mma.sync ... .f32.f16.f16.f32`), like `CUBLAS_COMPUTE_32F`.

So above 16 rows the only change is summation order, the same class of difference
cuBLAS already has between GPUs. At 2-16 rows activations go from f32 to f16, which
does change numerics: those rows then match what the QR-off f16 path has always
computed. That is the "own numerics check" this needs (see Verification).

**Row independence.** A row's output must not depend on how many other rows are in the
batch, so the tile shape and split-K factor depend only on `(in_features,
out_features)` and the GPU, never on `rows`. Each output element then sums its products
in the same order whatever the batch, so a batched prefill gives the same bits as the
same rows run in smaller batches (decode still differs: it uses the f32-activation
GEMV).

**Determinism.** Split-K partial sums go to an f32 workspace `[splits, rows, out]` and
a fixed-order reduction kernel adds them; no float atomics, so repeated runs give the
same bits.

### Kernel shape

- **Operands.** A = activations (`[rows, K]` f16, row-major, so each 16-row m-tile is
  a ready `mma` A fragment through `ldmatrix`); B = weights, `[out, K]` row-major in the
  GGUF, which is exactly the column-major `K x n8` B fragment layout: each output row is
  contiguous along K.
- **Instruction.** `mma.sync.aligned.m16n8k16` on sm_80+; two `m16n8k8` on sm_75
  (Turing has no k16 shape). Same fragment register mapping otherwise.
- **Weight layout stays raw.** No repack at load: the decode GEMVs and the scratch
  kernels read the arena as GGUF bytes, and a second, repacked copy would undo QR's
  memory saving. A super-block is 256 elements of one output row, so a warp's
  `n8 x k256` tile is 8 super-blocks: each lane dequantizes the elements its B
  fragments need from a shared-memory copy. Q4_K's nibble order (`qs[32g + l]` holds
  elements `64g + l` and `64g + 32 + l`) puts a fragment's two consecutive K elements
  in adjacent bytes, so one 16-bit read gives a lane its pair.
- **Block tile.** `BN = 64` output rows x `BK = 256` (one super-block per row per
  stage), m-tiles 1-4 (16-64 rows) per block, 4 warps each owning 16 output rows (2
  `n8` tiles). Weight bytes come in through 16-byte loads, double-buffered in shared
  memory: `cp.async` on sm_80+, register staging on sm_75. Above 64 rows the grid adds
  a row dimension.
- **Grid fill.** Qwen3-0.6B's 1024-wide projections give only 16 `BN = 64` blocks for
  40 (T4) or 80 (A10G) SMs, so small `out_features` get split-K, chosen from
  `(K, N, SM count)` to reach ~2 blocks per SM.
- **Types.** One template over a `dequant_pair<T>(smem, row, k)` per GGUF type: Q4_K
  first, then Q6_K (needed for the bench gap to close on Q4_K_M files) and Q8_0
  (Gemma 4). Each type's dequant expression is copied from its coalesced scratch
  kernel, which is the bit-exactness reference.

### Dispatch and fallbacks

In `Model::gemm`, for a `WeightData::Quant` weight with `weights_dtype == F16` and
`rows >= 2`: `gemm_q_mma` when the module loaded, else today's path. Today's paths stay
for `--weights f32`, for devices below sm_75, and behind `REFLEX_QUANT_MMA=0` for A/B
runs. An upper row limit (`REFLEX_QUANT_MMA_MAX_ROWS`) sends very long prefills back to
the scratch path if cuBLAS wins there; at 449+ rows the scratch round trip is a small
share, so a simple kernel may lose to cuBLAS. Measure, don't assume.

### Build and module loading

- **PTX target.** The default portable-PTX build passes no `-arch`, and `mma.sync` /
  `ldmatrix` need PTX for `sm_75`+. `build.rs` gets a per-file minimum: a
  `// reflex-min-arch: 75` line in the `.cu` adds `-arch=compute_75` in PTX mode. In
  the fatbin and cubin modes, `#if __CUDA_ARCH__ >= 750` makes the sm_75-and-below
  images an empty stub that the Rust side never launches (it checks compute capability
  first).
- **Lazy or eager.** `system1`'s ~12-row pass would run this kernel on the cold path.
  A separate module loaded on the first 2+-row QR prefill keeps decode-only and
  1-row paths free of it, but in a PTX build without a warm driver cache, JIT
  compiling it costs ~200 ms (the tiled-attention modules' measurement), far more than
  the 4 ms it saves. Plan: separate lazy module (`gemm_q_mma.cu`); measure cold
  `system1` with a cubin build (where loading is ~ms) and a PTX build; in a PTX build
  without a JIT cache, keep using the fused GEMV for the first call if the
  measurement says so.

## Verification

Kernel-level (GPU tests, synthetic blocks, shapes from the decode table plus every
dense-model projection shape, rows {2, 5, 8, 12, 13, 16, 17, 29, 33, 64, 113, 449}):

1. **Weight bits.** A debug entry point writes the dequantized f16 tile instead of
   multiplying; it must be bit-identical to `dequantize_*_coalesced_f16_kernel`'s
   output for Q4_K, Q6_K and Q8_0.
2. **Against the scratch path.** Same f16 inputs through scratch + `cublasGemmEx`:
   relative L2 below 1e-5 (summation order only), max abs diff reported.
3. **Against f64.** A host f64 GEMM over the same f16-rounded inputs: the kernel's
   error is no worse than cuBLAS's on the same shapes.
4. **Row independence.** Rows 0..r of a 113-row call are bit-identical to an r-row call
   on the same inputs.
5. **Determinism.** Two runs give identical bits, including split-K shapes.
6. **Saturation.** Activations beyond ±65504 give the same result as the scratch path
   (both saturate in the cast).

Model-level, both modes (QR default, `REFLEX_QUANT_MMA=0`), `--weights f16`:

7. **GPU suite** passes in both modes. Tests that pin today's 2-16-row fused-GEMV
   output (`quant_resident_tests`) will move: their reference for `--weights f16`
   becomes the scratch path, not the f32-activation GEMV.
8. **`reflex check`** against llama.cpp on every model in the QR-default run: 0
   token-id divergences, first-token checksum deltas reported against both today's QR
   and QR off. Expected: much closer to QR off than today (same weight and activation
   rounding, different order).
9. **`prefill_dense_batched_matches_sequential_prefill`** and the f16 tolerance
   (`F16_REL_L2_TOL`) tests hold.

Performance (T4 sm_75 and A10G sm_86, native cubin; PTX build for the cold-start
row): kernel micro-benchmark against scratch + cuBLAS and against f16 cuBLAS alone,
then `reflex bench` warm prefill on Qwen3-0.6B, 1.7B, TinyLlama and Mistral 7B, cold
`system1`/`generate` p50 n=10 in two interleaved rounds as before, and decode to show
it didn't move.

**Done means:** QR warm prefill at or below QR off at every bucket on both GPUs,
Qwen3-0.6B cold `system1` no slower than QR off, and 0 token-id divergences.

## Order of work

1. `build.rs` per-file minimum arch; empty kernel module that loads lazily; dispatch
   behind `REFLEX_QUANT_MMA`. No behavior change yet.
2. Q4_K kernel, 16-64 rows, no split-K; tests 1-6. First T4/A10G micro-benchmark:
   this is the go/no-go point. If a straightforward kernel can't beat scratch + cuBLAS
   at 29 rows, stop and reassess before polishing.
3. Split-K and the >64-row grid; row cap tuned.
4. Q6_K, then Q8_0 (Gemma 4).
5. Model-level verification and benchmarks; update
   [quantized-resident-weights.md](quantized-resident-weights.md) and
   [decode-speed.md](decode-speed.md)'s "What's left".

Out of scope: MoE/Kolibri expert GEMMs (`gemv_expert`, grouped prefill), MLA per-head
GEMMs, `--weights f32`, and fusing the activation cast.
