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
