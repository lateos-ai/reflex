# Tiled prefill attention for the other architectures

`attention_tiled.cu` runs Gemma 4's batched-prefill attention since 2026-10-12 (see
[gemma4.md, Long-prompt prefill](gemma4.md#long-prompt-prefill-2026-10-12)). Every
other architecture still runs prefill through `attention_online_kernel`, which streams
each query row's whole K/V range once per (row, query head), with no reuse across rows
or across a GQA group's query heads. This note records the measurements that decide which
architectures get the tiled kernel.

## Phase A: where prefill time goes (2026-10-15)

Runpod A40 (sm_86, driver 580, CUDA 12.4 toolkit), default `cargo build --release`
(PTX). Models: Qwen3-0.6B `Q8_0`, Qwen3-8B `Q8_0`, Qwen3.5-0.8B `Q4_K_M` and Granite
4.0-H 1B `Q8_0`, all with the default f16 matrix weights. The prompt is this repository's
`README.md` followed by `docs/*.md` and `docs/design/*.md`, cut so each model's token
count lands within 2% of 512, 2,048 and 8,192. Every point ran twice; the table shows the
second run, and the two agree within 1% everywhere. Scripts and raw output are archived
in `test-data/tiled-prefill-attention-src.tar.gz` (`phase-a-2026-10-15/`).

**Measured with `REFLEX_PREFILL_PROFILE=1`**, extended from Gemma 4 to every batched
prefill path for this work: dense, MoE and Kolibri-1 (`forward_layer_batched`),
Qwen3.5 (`forward_hybrid_layer_batched`), Granite 4.0-H (`forward_granite_layer`) and
MLA (`forward_mla_layer_batched`). Phases are charged everything enqueued since the
previous mark. Layer kinds are `layer` (uniform layers), `sliding`/`global`, and
`attn`/`ssm` for the hybrids; MLA adds `absorb` (the per-head `wk_b`/`wv_b` GEMVs) and
the hybrids add `mixer` (Gated DeltaNet or Mamba-2). Qwen3.5's Gated DeltaNet layers run
the mixer and the FFN one prompt row at a time, so that whole loop is one `ssm.mixer`
span. The line ends with `attn_ms`, the attention total over all layer kinds, and
`attn_pct`, its share of the GPU total.

| Model | Attention shape | Tokens | Prefill (GPU) | Attention | Share | Largest other phase |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 layers, 16 q / 8 kv, d 128 | 514 | 68 ms | 50 ms | 73% | FFN 8 ms |
| | | 2,008 | 786 ms | 717 ms | **91%** | FFN 28 ms |
| | | 8,164 | 15.6 s | 15.3 s | 98% | FFN 128 ms |
| Qwen3-8B | 36 layers, 32 q / 8 kv, d 128 | 514 | 257 ms | 127 ms | 49% | FFN 80 ms |
| | | 2,008 | 2.29 s | 1.85 s | **81%** | FFN 292 ms |
| | | 8,164 | 41.1 s | 39.5 s | 96% | FFN 1.03 s |
| Granite 4.0-H 1B | 4 of 40 layers, 12 q / 4 kv, d 128 | 509 | 96 ms | 5 ms | 5% | Mamba-2 mixer 70 ms |
| | | 2,008 | 400 ms | 77 ms | **19%** | Mamba-2 mixer 248 ms |
| | | 8,171 | 2.94 s | 1.64 s | 56% | Mamba-2 mixer 984 ms |
| Qwen3.5-0.8B | 6 of 24 layers (gated), 8 q / 2 kv, d 256 | 514 | 2.12 s | 9 ms | 0.4% | GDN loop 2.10 s |
| | | 2,025 | 8.49 s | 135 ms | **1.6%** | GDN loop 8.32 s |
| | | 8,330 | 38.4 s | 4.15 s | 11% | GDN loop 34.1 s |

Findings:

1. **Dense attention is almost the whole long prefill.** At 2K tokens it is 91% of
   Qwen3-0.6B's prefill and 81% of Qwen3-8B's, and above 95% at 8K. The online kernel's
   cost grows with rows × positions × query heads, so it is quadratic in prompt length
   and scales with the GQA group: Qwen3-8B's 32 query heads read each of its 8 KV heads
   four times over. The GEMMs and norms are small next to it (Qwen3-8B FFN 1.0 s at 8K).
   Even at 514 tokens attention is half of Qwen3-8B's prefill.
2. **Granite 4.0-H sits at the bar.** Only 4 of its 40 layers are attention, so the
   share is 19% at 2K, but it rises to 56% at 8K, where attention (1.64 s) costs more
   than all 36 Mamba-2 mixers (0.98 s). Its GQA group is 3, which the current kernel's
   block layout (M query vectors per block, whole groups only) can't pack at M = 64; it
   needs either an M that is a multiple of 3 or a padded group. Its attention also calls
   `attention_online` directly rather than `attention_prefill`, so it needs its own
   dispatch point.
3. **Qwen3.5 is bound by its Gated DeltaNet loop, not attention.** The GDN layers run
   token by token (about 4.1 ms per prompt row across 18 layers, host-launch bound:
   host enqueue time ≈ wall time). Attention is 1.6% at 2K and 11% at 8K. Its d 256
   attention already has a tiled instantiation, so dispatching it is cheap, but the
   gain is at most ~10% at 8K until the GDN prefill is batched or chunked.
4. **Not measured here:** Kolibri-1 and DeepSeek-V2-Lite (MLA) need an 80 GB A100.
   Kolibri-1's dense-attention layers have the same online kernel shape as Qwen3, plus
   a sliding window. MLA's attention reads one shared compressed row per position for
   all heads, which needs its own kernel variant.

## Cold-start cost of more tiled modules

Measured on the same A40 with `cuModuleLoadData` in a fresh context, three fresh
processes per row (they agree within 3 ms). PTX JIT uses `CUDA_CACHE_DISABLE=1`. The
d64 and d128 modules are throwaway instantiations of today's kernel with M capped at 64
(one kernel per module), so their times are estimates of what a final per-family module
costs. d256 and d512 at the cap are today's two kernels, each in its own module.

| Module | PTX JIT, no driver cache | PTX, warm driver cache | Fatbin (Docker archs) |
|---|---|---|---|
| d64 (prototype) | 208 ms | | |
| d128 (prototype) | 215 ms | 0.4 ms | |
| d256 | 247 ms | | |
| d512 | 183 ms | | |
| today's `attention_tiled` (d256 + d512) | 417 ms | | 0.1 ms |

For scale, the whole module set a dense model loads today (`modules_ms` on Qwen3-0.6B,
short prompt, `REFLEX_LOAD_PROFILE=1`, three runs): PTX without a driver cache
1,142-1,163 ms, PTX with a warm cache 5-7 ms, fatbin (`REFLEX_CUDA_ARCHS=sm_75,sm_80,
sm_86,sm_89,sm_90`, the Dockerfile default) 9-17 ms. None of these loads include
`attention_tiled` today.

So a lazily loaded per-family module costs ~210-250 ms once, on the first long prefill,
and only in a PTX build without a warm driver JIT cache. Fatbin and cubin builds pay
~0.1 ms. Short prompts and `system1` never load it, so the short-prompt cold-start
benchmarks shouldn't move; Phase B checks that.

**Register use** (`nvcc --resource-usage`, CUDA 12.4, every Dockerfile arch): the
prototype d64 and d128 kernels at M = 64 don't spill on any of sm_75/80/86/89/90 (112-128
registers; d128 is at the 128-register cap on sm_90). Today's d256 kernel spills 8 bytes
on sm_75 and sm_80 and 4 bytes on sm_90, and none on sm_86/89, which is why an A40
build shows no spill. d512 doesn't spill anywhere.

## Phase B: tiled prefill for dense, MoE, Kolibri and Qwen3.5 (2026-10-15)

**What changed.**
- **One kernel template.** `attention_tiled.cuh` holds the kernel as
  `attention_tiled<D, M>`, where M is the number of query vectors per block: 64, or 32
  for d512.
- **Padded groups.** A GQA group that doesn't divide M (3, 6, 7) now leaves
  `M % group` query slots per block unused, masked like rows past the end, instead of
  falling back to the online kernel. That covers Qwen2.5 (6 and 7) and the group-3
  shape Granite uses.
- **One module per head-dim family**, so a PTX build JIT-compiles only what its model
  uses: `attention_tiled.cu` (d256 and d512: Gemma 4, Qwen3.5), `attention_tiled_d128.cu`
  and `attention_tiled_d64.cu`.
- **One dispatch point.** `Model::attention_prefill` serves dense, MoE, Kolibri-1's
  dense layers and Qwen3.5's gated attention. It runs the tiled kernel when a prefill
  has at least `ATTN_TILED_MIN_ROWS` (64) rows and the shape fits (`attn_tiled_fits`),
  and `attention_online` otherwise.
- **Lazy loading.** Each module loads on the first prefill that needs it
  (`TiledAttnKernels`). Short prompts, decode and `system1` never load one, so
  short-prompt cold start is unchanged.
- **Gemma 4** keeps its eagerly loaded copy of the d256/d512 kernels and shares the
  launcher.
- **`REFLEX_ATTN_TILED=0`** sends every prefill back to the online kernel, for A/B runs.
- **Not covered yet:** Granite 4.0-H calls `attention_online` directly and still uses
  it, and MLA needs its own shared-latent variant. Both are deferred.

**Correctness** (Tesla T4, sm_75 cubin):
- `attention_tiled_small_dims_match_online_kernel` compares tiled against online on
  random Q/K/V, for 10 shapes across d64, d128 and d256. The shapes include groups 2,
  3, 4, 6, 7 and 8, a resumed prefill with a sliding window, and two rows. The largest
  difference is 1.1e-6.
- `attention_tiled_prefill_matches_online_end_to_end` runs a whole long-prompt
  prefill with f32 weights, tiled vs online, on Qwen3-0.6B (794 tokens), TinyLlama
  (971) and Qwen3.5-0.8B (807). The first-token logits agree to 1.8e-6 relative L2 or
  better, and 8 greedy ids match. With f16 weights the first-token logits differ by 3.6e-4 to
  7.0e-4 relative L2: every GEMM rounds its activations to f16, which amplifies the
  attention's summation-order differences.
- The GPU nightly suite passes with every fixture: Gemma 4's tiled tests, and
  batched-vs-sequential prefill for dense, MoE, Kolibri, Qwen2 and Qwen3.5.

**Registers** (CUDA 12.4, every Dockerfile arch): d64 uses 110-118 registers and d128
uses 124-127, with no spills on sm_75/80/86/89/90. d256 still spills 8 bytes on sm_75,
as before.

**Where tiled starts to win.** These are kernel times on the T4
(`attention_tiled_crossover_timings`, fresh prefill):

| shape | 16 rows | 32 rows | 64 rows | 128 rows | 512 rows | 2,048 rows |
|---|---|---|---|---|---|---|
| d128, 16 q / 8 kv: tiled | 32 us | 34 us | 35 us | 68 us | 0.62 ms | 7.0 ms |
| d128, 16 q / 8 kv: online | 17 us | 37 us | 93 us | 289 us | 4.15 ms | 63.2 ms |
| d128, 32 q / 8 kv: tiled | 34 us | 35 us | 39 us | 116 us | 1.07 ms | 13.9 ms |
| d128, 32 q / 8 kv: online | 29 us | 63 us | 162 us | 596 us | 8.32 ms | 126.4 ms |
| d64, 14 q / 2 kv: tiled | 22 us | 22 us | 23 us | 42 us | 0.26 ms | 3.5 ms |
| d64, 14 q / 2 kv: online | 14 us | 25 us | 57 us | 162 us | 2.56 ms | 39.4 ms |

Online wins below about 32 rows, where a tiled grid has too few blocks to fill the
GPU. From 64 rows up, tiled is 2.5x faster or more. The threshold of 64 also keeps the
tiled modules out of short prompts entirely.

**Whole prefill** (`REFLEX_PREFILL_PROFILE=1`, GPU total, `REFLEX_ATTN_TILED=0` vs
default). The prompts and the method are Phase A's (`profile.py`), and Mistral 7B runs
with `REFLEX_QUANT_RESIDENT=1` to fit in 16 GB:

| model | tokens | online | tiled | speedup | attention share, tiled |
|---|---|---|---|---|---|
| Qwen3-0.6B (d128, group 2) | 514 | 171 ms | 62 ms | 2.8x | 34% |
| | 2,008 | 2.03 s | 378 ms | 5.4x | 60% |
| | 8,164 | 43.1 s | 4.12 s | 10.5x | 87% |
| TinyLlama 1.1B (d64, group 8) | 511 | 222 ms | 93 ms | 2.4x | 18% |
| | 2,022 | 2.39 s | 448 ms | 5.3x | 51% |
| | 8,144 | 35.2 s | 4.19 s | 8.4x | 81% |
| Mistral 7B (d128, group 4) | 509 | 851 ms | 598 ms | 1.4x | 8% |
| | 2,023 | 6.25 s | 2.22 s | 2.8x | 34% |
| Qwen3.5-0.8B (d256, group 4) | 2,025 | 12.33 s | 12.04 s | 1.02x | 0.3% |
| | 8,330 | 58.5 s | 50.0 s | 1.17x | 1.5% |

Takeaways:
- **Dense prefill is no longer quadratic-bound up to 2K.** On Qwen3-0.6B, attention
  drops from 93% to 60% of a 2K prefill.
- **At 8K attention is still 87% of the prefill**, and this f32 SIMT kernel is the
  next limit. Tensor cores (f16 Q/K/V tiles) would be the next step.
- **Qwen3.5's attention shrinks 12x**, but its prefill stays bound by the token-by-token
  Gated DeltaNet loop, as Phase A found.

## Resumed prefill: switching on the cached length (2026-10-15)

**The problem.** A resumed prefill is a few new rows after many cached positions: a
pinned prefix or an imported cache (docs/design/agent-relay.md). The 64-row threshold
above was measured on fresh prefills, so a 45-row continuation after 1,490 cached
positions still ran `attention_online`. That kernel runs one block per (row, query
head), and each block reads every cached K/V row, so its cost grows with the prefix.
On the T4 with Qwen3-0.6B (`REFLEX_PREFILL_PROFILE=1`, attention GPU time over all 28
layers):

| new rows | 0 cached | 489 cached | 1,489 cached |
|---:|---:|---:|---:|
| 17 (online) | 0.5 ms | 12 ms | 38 ms |
| 41 (online) | 1.4 ms | 31 ms | 90 ms |
| 82 (tiled) | 1.8 ms | 8 ms | 23 ms |

The agent relay hops land in the slow row: each carries the previous hop's 32-token
note, so the continuation is about 45 tokens.

**Where tiled wins on a resumed prefill.** These are kernel times
(`attention_tiled_crossover_timings_resumed`), shown as online / tiled:

| shape | cached | 8 rows | 16 rows | 32 rows | 63 rows |
|---|---:|---:|---:|---:|---:|
| d128, 16 q / 8 kv | 64 | 0.5x | 0.9x | 1.6x | 3.4x |
| | 512 | 0.9x | 1.4x | 2.3x | 4.3x |
| | 2,048 | 1.0x | 1.9x | 3.1x | 5.1x |
| d128, 32 q / 8 kv | 64 | 0.9x | 1.5x | 2.9x | 6.0x |
| | 2,048 | 1.9x | 3.1x | 5.3x | 8.5x |
| d64, 14 q / 2 kv | 64 | 0.6x | 0.8x | 1.6x | 2.9x |
| | 2,048 | 0.8x | 1.2x | 2.3x | 4.5x |

Once anything is cached, the crossover depends mostly on how full the tiled block's
64 query slots are, `rows * group`, and hardly on the prefix length.

**The rule** (`attn_tiled_preferred` in `kernels.rs`): run the tiled kernel from 64
rows up, as before, or, with at least `ATTN_TILED_MIN_START` (64) positions cached,
once `rows * group >= ATTN_TILED_MIN_QUERIES` (64). Fresh prompts are unchanged, so short
prompts and cold start still never load a tiled module. A short resumed prompt now
loads it. With a PTX build and no warm driver cache, that adds a one-time ~200 ms JIT
to a cold `reflex generate --import-kv` process; a warm session pays it once. Prompts
with fewer rows than the rule allows (17 rows here) still use the online kernel. At
long offsets tiled would win there too, by up to 1.9x, and a finer rule could pick
that up later.

**Correctness.** `attention_tiled_small_dims_match_online_kernel` gained six resumed
shapes: 41 and 5 rows after about 1,500 cached positions, a padded group, a sliding
window, d64 and d256. They match the online kernel, like the existing shapes. The pinned-prefix relay test
still gives the same greedy ids as one joined request, and the GPU nightly suite passes
(84 passed, 0 failed, 4 skipped for absent optional models).

**Result** (T4, same session, the phase-B binary and this one run alternately):

| resumed prefill, 1,489 cached | before | after |
|---|---:|---:|
| 41 rows: attention / first token | 90 ms / 107 ms | 21 ms / 38 ms |
| 82 rows: attention / first token | 23 ms / 41 ms | 23 ms / 41 ms |

| agent relay hop, ~1,490-token prefix, 32 tokens | round trip p50 | first token p50 |
|---|---:|---:|
| pinned prefix, before | 364.5 ms | 108.5 ms |
| pinned prefix, after | 295.3 ms | 38.6 ms |
| file reuse, after (547.2 ms before) | 472.5 ms | 216.3 ms |
| recomputed prefix (unchanged) | 504.8 ms | 246.1 ms |
| llama-server, prompt cache | 161.7 ms | 12.3 ms |

The resumed first token is now about 39 ms, of which about 21 ms is attention.
llama.cpp's f16 tensor-core flash attention reaches 12 ms. The tiled kernel's f32 math
runs at about 1.3-1.4 TFLOPS whether or not anything is cached, so closing the rest of
that gap would take an f16 tensor-core kernel. Splitting the cached keys across
blocks would help prompts too short to fill the GPU.
