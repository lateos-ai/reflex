# Design: tensor-core prefill straight from quantized weights

Status: 2026-10-22. Steps 1 and 2 are in: the Q4_K kernel
(`gemm_q4k_mma_m{1..4}_kernel`, no split-K) is live for Q4_K weights at 2+ rows with
QR and `--weights f16`. Step 2 passed its go/no-go on an A10G (see "Step 2 results");
not yet run on a T4. Step 4's Q6_K kernel (`gemm_q6k_mma_m{1..4}_kernel`, see
"Step 4: Q6_K") and step 3's split-K (see "Step 3: split-K") are written and
dispatched; they compile for sm_75, sm_86 and PTX but have not run on a GPU yet. The numbers in "Where the gap comes from" are arithmetic on the
published A10G results, not a profile.

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

**Row independence.** A row's output should not depend on how many other rows are in the
batch, so the tile shape depends only on `(in_features, out_features)` and the GPU,
never on `rows`. The split-K factor was meant to follow the same rule; step 3 drops
that (see "Step 3: split-K"): it depends on the grid size, so row independence holds
between calls that get the same split count. Each output element then sums its products
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
- **Types.** One template over a per-type traits struct (`QmmaQ4K`, `QmmaQ6K`: block
  size, staging, scales, B fragment): Q4_K first, then Q6_K (needed for the bench gap
  to close on Q4_K_M files) and Q8_0 (Gemma 4). Each type's dequant expression is copied from its coalesced scratch
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
   relative L2 error stays below `3e-9 * in_features` (at least 1e-6), with cuBLAS's
   printed beside it. Not "no worse than cuBLAS": at a few rows cuBLAS runs CUDA-core
   f32 FMAs, while `mma.sync` adds inside the tensor core at lower internal precision
   (A10G, 1024x136 at 2 rows: 1.5e-6 vs 1.8e-7), and the error grows about linearly
   with K. All of it is far below the f16 input rounding (~3e-4).
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

## Step 2 results (A10G, 2026-10-21)

g5.xlarge, driver 595.91, CUDA 13.2, `REFLEX_CUDA_ARCH=sm_86` cubin and the portable
PTX build (`-arch=compute_75` from the file's floor, so the two-k8 path, JIT'd for
sm_86). Not yet run on a T4.

**Tests 1, 2, 4, 5, 6 pass in both builds.** Test 3 failed as first written ("within
2x cuBLAS's error"); the bound was wrong, not the kernel (see item 3). Worst relative
L2 against the f64 product, kernel / cuBLAS: 5.4e-7 / 5.4e-7 at 256x40, 2.0e-6 /
1.7e-6 at 1024x1024, 5.5e-6 / 1.4e-6 at 3072x1024, 6.9e-6 / 3.5e-6 at 4096x4096.
Both builds give the same numbers. `quant_resident` tests pass (the Kolibri and
Qwen2.5 ones need fixtures that weren't on the box).

**Greedy tokens:** 32 generated tokens identical with MMA on, `REFLEX_QUANT_MMA=0`
and `REFLEX_QUANT_RESIDENT=0`, on Qwen3-0.6B and Mistral 7B.

**Kernel micro-benchmark** (`quant_mma_prefill_timings`, then Q4_K only, µs; mma / scratch +
cuBLAS / f16 cuBLAS alone). Go: the kernel beats today's QR path at every shape and
row count, and beats f16 cuBLAS at 12-64 rows on the wide shapes.

| shape | 12 rows | 29 rows | 113 rows | 449 rows |
|---|---|---|---|---|
| 1024x3072 | 12.5 / 56.3 / 23.8 | 16.1 / 54.4 / 25.2 | 37.0 / 65.9 / 33.1 | 84.0 / 142.7 / 104.9 |
| 3072x1024 | 23.5 / 55.1 / 26.9 | 33.2 / 56.8 / 27.5 | 54.4 / 66.1 / 33.0 | 127.0 / 139.6 / 104.6 |
| 4096x4096 | 32.3 / 289.5 / 85.1 | 44.6 / 254.1 / 86.3 | 121.1 / 268.8 / 102.3 | 412.0 / 544.8 / 381.2 |
| 4096x14336 | 90.2 / 763.1 / 264.9 | 104.3 / 836.7 / 267.2 | 326.4 / 883.2 / 313.8 | 1223.9 / 1619.7 / 1047.6 |
| 14336x4096 | 93.7 / 1029.5 / 264.7 | 137.9 / 843.5 / 267.6 | 401.0 / 878.6 / 309.2 | 1363.3 / 1753.3 / 1185.2 |

It loses to f16 cuBLAS on narrow outputs (`out_features` 1024: 16 blocks on 80 SMs)
from ~29 rows, and everywhere from ~113 rows: split-K and the larger-row tiles are
step 3.

**`reflex bench` warm prefill** (p50 ms, 5 warmup + 20 iters; MMA / today's QR /
QR off):

| model | short | medium | long |
|---|---|---|---|
| Qwen3-0.6B, 29 / 113 / 449 | **6.81** / 11.26 / 7.37 | 11.10 / 14.34 / **9.88** | **27.29** / 32.73 / 28.22 |
| Mistral 7B, 33 / 129 / 513 | **34.2** / 104.4 / 39.5 | 79.4 / 140.7 / **75.0** | 213.9 / 261.2 / **195.5** |

Mistral's 65 ms gap is gone at 33 tokens and down to 4-18 ms above it. What's left:
the Q6_K tensors (`attn_v`/`ffn_down` in half the layers of a Q4_K_M file) still take
the scratch path (step 4), and the larger-row tiles (step 3). Decode throughput
didn't move (Qwen3-0.6B 288 vs 291 tok/s, within run-to-run noise).

## Step 3: split-K (2026-10-22, not yet run on a GPU)

Done after step 4. Grid z splits each row's super-blocks into equal ranges (the last
one shorter, none empty); with more than one split, split z writes its partial sums
to a `[splits, rows, out]` f32 workspace and `gemm_q_mma_splitk_reduce_kernel` adds
them in split order. No atomics, so repeated runs give the same bits.

- **Split count from the grid, not from `(K, N)` alone.** A count fixed per shape, as
  first planned, would give Qwen3-0.6B's 1024-wide outputs ~10 splits on an A10G at
  every row count. At 449 rows that adds `splits * 8` bytes of workspace traffic per
  output element against `2K` flops, about as much time as the GEMM itself.
  `quant_mma_splits` instead aims for about two blocks per SM over the whole grid
  (output blocks x row blocks): `round(2 * SMs / blocks)`, at most `K / 512`
  (2+ super-blocks per split, so loads still overlap math), and 1 once the grid fills
  the GPU. So a row's bits match across batch sizes only while the split count
  matches. The cuBLAS paths (QR off and scratch) are no stricter: cuBLAS picks
  kernels by `m`.
- **Examples** (A10G, 80 SMs): Qwen3-0.6B `o_proj` (3072x1024) at 29 rows: 6 splits of
  2; Mistral `ffn_down` (14336x4096) at 29 rows: 3 splits of 19; any shape at 449
  rows: 1. On a T4 (40 SMs) the 3072x1024 case gets 4 splits of 3.
- **`REFLEX_QUANT_MMA_SPLITS=<n>`** forces the count (clamped to the super-block
  count), for tuning runs.
- **Tests.** `quant_mma_split_rule` (host only: examples, no empty split),
  `quant_mma_split_k_matches_unsplit` (forced 2, 3, 5 and one split per super-block
  against 1 split: relative L2 < 1e-6), and `quant_mma_rows_independent_and_deterministic`
  now pins the count (1, 3, 5) for the row-independence part and checks repeat
  determinism with the picked count. `quant_mma_prefill_timings` prints the
  unsplit time and the split count next to the split one, to tune the rule.
- Larger row tiles (above 64 rows per block) are not done: they need more than 48 KB
  of shared memory (dynamic shared memory and an opt-in attribute) or 8 warps.
  Whether they're worth it waits for the GPU numbers: at 113 and 449 rows the
  Q4_K kernel was within ~4-30% of f16 cuBLAS, and part of that is the
  narrow-output grid fill that split-K addresses.

## Step 4: Q6_K (2026-10-22, not yet run on a GPU)

Done first, then step 3. In a Q4_K_M file `attn_v` and `ffn_down` are Q6_K in about half
the layers, and they still took the scratch path: at 33 rows on the A10G, a
14336x4096 Q4_K weight costs ~840 µs through scratch against ~270 µs for f16 cuBLAS,
and Mistral has 16 such `ffn_down`s. That is most likely the bulk of the 4-18 ms
Mistral still trails QR off by (an estimate, not a profile).

- **Same kernel.** `gemm_q_mma_impl<Q, MT>` is the step-2 kernel with the per-type
  parts moved into a traits struct; the Q4_K kernels do the same math in the same
  order as before.
- **Staging an unaligned row.** A Q6_K block is 210 bytes, so rows and blocks are only
  2-byte aligned (arena offsets are 256-aligned). Each weight row stages the 14 aligned
  16-byte chunks that cover its block and reads the block at the address's offset in
  the first one (0-14, even). 14 chunks always cover it, and each of them holds at
  least one of the block's bytes, so no load touches a 16-byte granule outside the
  tensor. Rows are staged 240 bytes apart (60 words) so the 8 rows a warp reads at
  once sit on 8 different banks. With 64 m rows, shared memory is exactly 48 KB
  (33 KB activations + 15 KB weights).
- **Dequant.** One scale per k16 step (`scales[8n + s%2 + 2r]`), `d * (float)sc`
  computed once per step and `* (float)q` per element, the coalesced kernel's
  expression and order, rounded with `__float2half_rn`. The fragment-to-element
  mapping was checked against the coalesced kernel's indexing for all 256 elements
  on the host.
- **Registers.** No spills; the 4-tile kernel uses 255 registers on sm_75 and sm_86
  (Q4_K: 247/251).
- **Tests.** `quant_mma_tests.rs` now runs tests 1-6 and the micro-benchmark for both
  types (`quant_mma_dequant_bit_exact`, `quant_mma_matches_f16_gemm`,
  `quant_mma_rows_independent_and_deterministic`, `quant_mma_saturates_like_scratch`,
  `quant_mma_prefill_timings`), on synthetic Q6_K blocks with signed scales and a `d`
  that is sometimes zero or negative.

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
