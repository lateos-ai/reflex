# Design: Mamba-2 hybrid support

**Status (2026-10-15): done and verified on an RTX 4000 Ada.** Target: IBM Granite
4.0-H 350M (`granitehybrid`), chosen from the survey below; 1B was checked too.

- **Fixture:** greedy tokens match llama.cpp `79e2e74` on 7/7 prompts × 20 tokens, in f32
  and f16.
- **Real models:** with f32 activations in llama.cpp, Granite 4.0-H 350M and 1B match
  5/6 each, in f32 and f16. The one divergence per model is a near-tie (margin 0.0019
  and 0.033) where Reflex picks llama.cpp's own runner-up.
- **Regression:** `reflex check` is unchanged on all 48 existing rows. The GPU nightly
  passes 65, fails 0, skips 3.
- **Cold start:** 0.76 s p50 vs `llama-simple`'s 0.83 s.
- **Decode cost:** flat in context length from 57 to 8,191 tokens.

See [Results](#results-2026-10-15).

Goal: run a Mamba-2 hybrid (Mamba-2 SSM layers plus a few attention layers) with
greedy tokens identical to llama.cpp, as for every other architecture. Prefill is
naive-first: SSM layers step through the prompt one token at a time, like the Qwen3.5
Gated DeltaNet layers.

## Reference implementations

llama.cpp `79e2e74` (2026-10-09), the pin used by every other fixture. No later commit
touches the Mamba-2 graph, the recurrent memory, the CUDA SSM kernels or the
Mamba/Granite converters.

- Graphs: `src/models/mamba-base.cpp` (`build_mamba2_layer`, shared by every
  candidate), `mamba2.cpp`, `granite-hybrid.cpp`, `falcon-h1.cpp`, `nemotron-h.cpp`,
  `nemotron-h-moe.cpp`, `jamba.cpp`.
- Ops: `ggml_compute_forward_ssm_conv_f32`/`ggml_compute_forward_ssm_scan_f32`
  (`ggml/src/ggml-cpu/ops.cpp`); `ggml/src/ggml-cuda/ssm-conv.cu`, `ssm-scan.cu`.
- State: `src/llama-memory-recurrent.cpp`.
- Converters: `conversion/mamba.py` (`Mamba2Model`), `conversion/granite.py`
  (`GraniteHybridModel`, registered for `GraniteMoeHybridForCausalLM` and
  `BambaForCausalLM`), `conversion/falcon_h1.py`, `conversion/nemotron.py`,
  `conversion/jamba.py`.

## Header survey

All values below come from the first 40 MB of each GGUF (an HTTP range request) and from
the HF `config.json`. The stripped headers are in `test-data/` (names in the last
column). Sizes are the Q8_0 / Q4_K_M files.

| Candidate | arch | Q8_0 / Q4_K_M | License | Layers | `d_conv` / `d_inner` / `d_state` / `n_group` / heads (head dim) | Pre type | Other ops | Header |
|---|---|---|---|---|---|---|---|---|
| **Granite 4.0-H 350M** (IBM) | `granitehybrid` | 0.37 / 0.22 GB | Apache-2.0 | 32: 28 Mamba-2, 4 attention (10, 13, 17, 27); every layer has a SwiGLU FFN | 4 / 1536 / 128 / 1 / 48 (32) | `dbrx` | NoPE GQA 12/4, head 64; muP scales (embedding ×12, residual ×0.246, attention 1/64, logits ÷3); tied LM head | `granite-4.0-h-350m-header.gguf` |
| Granite 4.0-H 1B | `granitehybrid` | 1.56 / 0.90 GB | Apache-2.0 | 40: 36 Mamba-2, 4 attention (5, 15, 25, 35) | 4 / 3072 / 128 / 1 / 48 (64) | `dbrx` | as 350M; GQA 12/4, head 128, attention 1/128, residual ×0.22, logits ÷6 | `granite-4.0-h-1b-header.gguf` |
| Granite 4.0-H Micro (3.2B) | `granitehybrid` | 3.40 / 1.94 GB | Apache-2.0 | 40, as 1B | 4 / 4096 / 128 / 1 / 64 (64) | `dbrx` | as 350M; GQA 32/8, head 64, logits ÷8 | `granite-4.0-h-micro-header.gguf` |
| Granite 4.0-H Tiny (7B-A1B) | `granitehybrid` | 7.39 / 4.23 GB | Apache-2.0 | 40, as 1B; every FFN is MoE | 4 / 3072 / 128 / 1 / 48 (64) | `dbrx` | as 1B, plus a softmax top-6-of-64 MoE and an always-on shared expert | `granite-4.0-h-tiny-header.gguf` |
| Falcon-H1-Tiny-90M-Instruct (TII) | `falcon-h1` | 0.10 / 0.06 GB | Falcon LLM license (Apache-2.0 based, plus an acceptable-use policy) | 24, each parallel attention + Mamba-2, then SwiGLU | 4 / 768 / **64** / 1 / 24 (32) | `falcon-h1` (LLAMA3 group) | NORM RoPE, base 1e11; no `ssm_norm` (plain `silu(z)` gate); muP folded into weights by the converter; `ffn_norm` with no `.weight` suffix | `falcon-h1-tiny-90m-header.gguf` |
| Falcon-H1-0.5B-Instruct | `falcon-h1` | 0.56 / 0.32 GB | Falcon LLM license | 36, as Tiny | 4 / 1536 / 128 / 1 / 24 (64) | `falcon-h1` | as Tiny | (not saved) |
| NVIDIA-Nemotron-3-Nano-4B | `nemotron_h` | – / 2.84 GB | NVIDIA Open Model License | 42: 21 Mamba-2, 4 attention, 17 FFN-only layers | 4 / 7680 / 128 / **8** / 96 (80) | `pixtral` (LLAMA3 group) | NoPE attention; ReLU² FFN with no gate; separate FFN-only layers | `nemotron-3-nano-4b-header.gguf` |
| Nemotron-H-8B-Reasoning-128K | `nemotron_h` | 8.62 / 4.98 GB | NVIDIA internal research license | 52 | 4 / 8192 / 128 / 8 / 128 (64) | `tekken` (not ported) | as Nano-4B | (not saved) |
| mamba2-130m (community GGUF of `state-spaces/mamba2-130m`) | `mamba2` | 0.18 GB (Q8_0 only) | Apache-2.0 | 24 Mamba-2, no attention, no FFN | 4 / 1536 / 128 / 1 / 24 (64) | `olmo` (GPT2 split) | none; untied `output.weight` | `mamba2-130m-header.gguf` |
| Mamba-Codestral-7B | `mamba2` | 7.75 GB | Apache-2.0 | 64 Mamba-2 | 4 / 8192 / 128 / 8 / 128 (64) | SentencePiece (`llama`) | none | (not saved) |
| jamba-900M (pszemraj); AI21 Jamba2-3B | `jamba` | 0.95; 3.41 GB | Apache-2.0 | attention every 4th / 14th layer, MoE every other layer (900M) | 4 / 2048 / **16** / – / `dt_rank` 256 | `gpt-2`; `llama` | **Mamba-1** layers (`ssm_x`, `ssm_dt.weight`, per-element `ssm_a [16, d_inner]`, dt/B/C RMSNorms), not Mamba-2 | `jamba-900m-header.gguf` |

One Mamba-2 layer of Granite 4.0-H 350M (GGUF `ne` order, innermost first):

```
attn_norm.weight   [768]          F32
ssm_in.weight      [768, 3376]    Q8_0   3376 = 2*1536 + 2*1*128 + 48  (z, xBC, dt)
ssm_conv1d.weight  [4, 1792]      F32    1792 = 1536 + 2*1*128         (conv_dim)
ssm_conv1d.bias    [1792]         F32
ssm_dt.bias        [48]           F32
ssm_a              [1, 48]        F32    no .weight suffix; already -exp(A_log)
ssm_d              [1, 48]        F32    no .weight suffix
ssm_norm.weight    [1536]         F32    [d_inner/n_group, n_group] collapses to 1-D at n_group = 1
ssm_out.weight     [1536, 768]    Q8_0
ffn_norm.weight, ffn_gate/up [768, 2048], ffn_down [2048, 768]
```

Its attention layers carry `attn_q [768, 768]`, `attn_k`/`attn_v [768, 256]`,
`attn_output [768, 768]`, with no biases and no Q/K norm. `granitehybrid.attention.head_count_kv`
is a per-layer array whose zeros mark the SSM layers (llama.cpp:
`is_recr(i) = n_head_kv(i) == 0`). Every surveyed file has
`ssm.time_step_rank == d_inner / head_dim`, the head count; it isn't a low-rank `dt`
projection as in Mamba-1.

### Converter `transformers` versions

`GraniteMoeHybridForCausalLM` is in `transformers` 4.52 and later (Granite's own config
says 4.56.0), `FalconH1ForCausalLM` in 4.53 and later, and `Mamba2ForCausalLM` in 4.44
and later. `NemotronHForCausalLM` isn't in any release up to 5.0.0. It loads only with
`trust_remote_code`. For Granite the `transformers` version doesn't matter: converting
the fixture with 4.57.1 and with 5.19 gives byte-identical GGUFs, both with
`tokenizer.ggml.pre = dbrx`. The 4.57.1 pin in `docs/DEVELOPMENT.md` is only needed for
DeepSeek tokenizers.

## Recommendation: Granite 4.0-H 350M

1. **Smallest real first-party file with a permissive license.** 366 MB at Q8_0, from
   IBM's own GGUF repo, Apache-2.0. Falcon-H1-Tiny is smaller (98 MB), but its license
   isn't Apache-2.0, and it uses `d_state = 64`, which llama.cpp's CUDA scan doesn't
   support. That file would run its scan on the CPU in the reference build.
2. **Fewest new ops beyond the Mamba-2 mixer.** The attention layers are standard GQA
   with no RoPE, no bias and no Q/K norm. `AttnMode { rope: false }` already exists for
   Kolibri-1's NoPE layers. The FFN is plain SwiGLU. The extra pieces are four scalar
   muP scales and one attention-scale override. Falcon-H1 runs attention and SSM in
   parallel inside every layer, which needs both kinds of state per layer. Nemotron-H
   adds ReLU² FFN-only layers, `n_group = 8`, an NVIDIA license and a converter that
   needs `trust_remote_code`.
3. **The family scales without new code.** 1B and Micro (3.2B) use the identical graph.
   Tiny (7B-A1B) adds only a softmax MoE plus a shared expert, which `moe::route_top_k`
   and the MLA shared-expert FFN already cover.
4. **Tokenizer: one small port.** `dbrx` is LLAMA3's regex string (`llama-vocab.cpp`:
   "same as llama3") without llama.cpp's `ignore_merges` and without the forced BOS. It
   is a new `PreType::Dbrx` that reuses `unicode_regex_split_custom_llama3`'s splitter
   with `ignore_merges = false`. BOS follows `add_bos_token = false`, which Reflex
   already honors.

Cheap follow-ons once Granite works, each a separate decision:

- **pure `mamba2`**: the same mixer with no attention or FFN layers. Its `olmo` pre type
  is llama.cpp's GPT2 split with no flags, so it would need only an alias.
- **Granite 4.0-H Tiny**: MoE plus a shared expert.

## Scope

In:

- `granitehybrid` with `expert_count = 0` (350M, 1B, Micro). The layer kind comes from
  the `head_count_kv` array, as llama.cpp reads it.
- The Mamba-2 mixer (below), with `n_group >= 1`. The real targets all have
  `n_group = 1`; the fixture uses 2, so the group mapping is tested too.
- Granite's muP scales: `embedding_scale`, `residual_scale`, `attention.scale`,
  `logit_scale`. Each one is skipped when absent or 0, like llama.cpp's `if (scale)`.
- `--weights f32|f16` and `system1` with single-token candidates only, for the same
  reason the Qwen3.5 hybrid path has: SSM state is a running accumulator.
- Pre type `dbrx`.

Out, rejected at load with an error naming the key or tensor and its value:

- `granitehybrid.expert_count > 0` (Granite 4.0-H Tiny/Small, queued).
- `granitehybrid.rope.scaling.finetuned = true` (Bamba-style RoPE on the attention
  layers; no surveyed Granite 4.0-H file uses it).
- Attention biases (`attn_q/k/v.bias`, `attn_output.bias`) and FFN biases. The graph
  allows them, but no surveyed file has them.
- An SSM layer without `ssm_norm` (Falcon-H1 style) and an SSM layer without
  `ssm_conv1d.bias`. Both are optional in llama.cpp; neither is in a Granite file.
- `d_inner != 2 * n_embd` (llama.cpp asserts this for `granitehybrid`),
  `d_inner % head_count != 0`, and `head_count % n_group != 0`.
- A `d_state` that isn't a multiple of 32 up to 256. The scan kernel keeps 32 lanes ×
  up to 8 state values per warp in registers. llama.cpp's own CUDA scan accepts fewer
  values: 96, 128 or 256.
- Attention layers whose `head_count_kv` entries differ.
- A file with no SSM layer, i.e. a non-hybrid Granite, which llama.cpp loads as
  `granite` or `granitemoe`.
- `mamba`, `mamba2`, `jamba`, `falcon-h1`, `nemotron_h`, `nemotron_h_moe`: rejected by
  name in `parse_model_config`. `jamba` and `nemotron_h_moe` report a nonzero
  `expert_count`, so without this they reach the generic MoE path and fail on a missing
  tensor instead of saying why. `bamba` gets the generic unsupported-architecture error.
- `--export-kv`/`--import-kv`. Exporting the fixed-size SSM state is a later, separate
  decision.
- `--lora`.
- `REFLEX_QUANT_RESIDENT=1`: prints its notice and loads in the `--weights` dtype, like
  `qwen35`.

## The Mamba-2 single-step math

From `build_mamba2_layer` (`mamba-base.cpp`), `ggml_ssm_conv`, and `ggml_ssm_scan`'s
scalar-A branch (CPU `ops.cpp`; CUDA `ssm_scan_f32_group`). Notation:

- `D = d_inner`, `N = d_state`, `G = n_group`, `H = ssm.time_step_rank` (heads)
- `P = D / H` (head dim), `C_dim = D + 2·G·N` (conv dim)
- input `u` is the layer's `attn_norm` output, `[n_embd]`

1. **In-projection.** `zxBCdt = ssm_in · u`, length `2·D + 2·G·N + H`. Split it in
   this order:
   - `z = [0, D)`
   - `xBC = [D, D + C_dim)`
   - `dt_raw = [D + C_dim, D + C_dim + H)`

   There is no projection bias (`mamba_proj_bias: false`).
2. **Causal depthwise conv over `xBC`, then SiLU.** Per channel `c < C_dim`:
   - The window is the last `d_conv - 1` raw `xBC` inputs (oldest first), then the
     current one.
   - Weights are `ssm_conv1d.weight` with `ne = [d_conv, C_dim]`, so channel `c`'s
     taps are `w[c·d_conv .. c·d_conv + d_conv)`, oldest tap first.
   - `xBC'[c] = silu(Σ_k window[k][c] · w[c·d_conv + k] + conv_bias[c])`.

   llama.cpp's CUDA kernel sums the taps in order, adds the bias, then applies SiLU.
   Afterwards the conv state drops its oldest column and appends the current raw
   (pre-conv) `xBC`. That is `gdn_conv_kernel`'s weight layout and update order exactly.
   The difference is the bias: GDN's conv has none.
3. **Split the conv output.**
   - `x = xBC'[0, D)`, viewed as `[H, P]`: element `(h, p)` is at `h·P + p`.
   - `B = xBC'[D, D + G·N)`, viewed as `[G, N]`.
   - `Cm = xBC'[D + G·N, D + 2·G·N)`, viewed as `[G, N]`.
4. **Time step.** `dt_h = softplus(dt_raw_h + dt_bias_h)`, with
   `softplus(v) = v <= 20 ? log1p(exp(v)) : v` (the CUDA kernel uses `log1pf(expf(v))`).
   No `time_step_limit` clamp is applied: llama.cpp ignores the HF key, and Granite's and
   mamba2-130m's configs set it to `(0, inf)` anyway.
5. **Decay.** `A_h = ssm_a[h]`, stored by the converter as `-exp(A_log)`
   (`conversion/mamba.py`: `data_torch = -torch.exp(data_torch)`), so the engine uses it
   as is. `dA_h = exp(dt_h · A_h)`.
6. **State update and readout**, for each head `h`, its group `g = h / (H / G)` (the
   `repeat_interleave` grouping), each `p < P` and each `n < N`:

   ```
   s[h,p,n] = s[h,p,n] · dA_h + B[g,n] · (x[h,p] · dt_h)
   y[h,p]   = Σ_n s[h,p,n] · Cm[g,n]
   ```

   The state is `[H, P, N]` with `n` innermost, `s[(h·P + p)·N + n]`, which is
   llama.cpp's `{d_state, head_dim, n_head}`. CUDA evaluates `x_dt = x · dt` first, then
   `state·dA + B·x_dt`, then reduces `y` across a warp. Reflex's kernel uses the same
   expression order. The reduction order can differ in the last bit; greedy agreement
   is the test, as for every other kernel.
7. **D skip.** `y[h,p] += ssm_d[h] · x[h,p]`. `ssm_d` has one value per head (`[1, H]`).
8. **Gate, then grouped RMSNorm.** `y = y · silu(z)` (`ggml_swiglu_split(z, y)`). The
   gate comes **before** the norm (HF `norm_before_gate = False`). Then:
   - `y` is split into `G` contiguous groups of `D / G`.
   - Each group is normalized as `y_g / sqrt(mean(y_g²) + eps)`, with
     `eps = attention.layer_norm_rms_epsilon`.
   - The result is multiplied by `ssm_norm.weight` over the full `D`.

   GDN's `gdn_gated_norm_kernel` does the opposite: it normalizes first, gates after,
   and uses a per-head weight. It can't be reused.
9. **Out-projection.** `out = ssm_out · y`, back to `[n_embd]`.

### The Granite 4.0-H layer around it

From `granite-hybrid.cpp`:

```
h = token_embd[id] · embedding_scale
for each layer:
    m = rmsnorm(h, attn_norm)
    m = is_ssm ? mamba2(m) : attention(m)
    h = h + m · residual_scale
    f = ffn_down · (silu(ffn_gate · rmsnorm(h, ffn_norm)) ⊙ (ffn_up · rmsnorm(h, ffn_norm)))
    h = h + f · residual_scale
logits = (lm_head · rmsnorm(h, output_norm)) / logit_scale    (lm_head tied to token_embd)
```

Attention details:

- Standard causal GQA, no RoPE: `has_rope = rope.scaling.finetuned`, false in every
  file.
- `kq_scale = attention.scale` (1/64 at head dim 64, 1/128 at head dim 128; not
  `1/sqrt(head_dim)`).
- `undo_permute` in the converter permutes the Q/K rows of each head. With no RoPE, the
  scores are unchanged.

## What exists and what is new

The Qwen3.5 hybrid path (`src/model/hybrid.rs`, `kernels_cuda/gated_deltanet.cu`) has
the shape this needs. Each piece:

| Need | Existing | Reuse? |
|---|---|---|
| Per-layer kind from metadata | `parse_hybrid_layer_kinds` (`qwen35`: `full_attention_interval`) | Pattern only. Granite reads `head_count_kv[i] == 0` |
| Per-layer state enum (attention K/V or recurrent) | `HybridLayerState::{Attn, Gdn}` | Same shape: new `Ssm { conv_state, ssm_state }` |
| Conv state step | `gdn_conv_kernel` (same weight layout and shift order) | New `mamba2_conv_kernel`: the bias changes the math, so GDN's kernel stays bit-identical |
| Sequential recurrent prefill inside batched layers | `forward_hybrid_layer_batched`: GDN rows loop over `extract_row`/`write_row`, attention and FFN batched | Not needed: each SSM kernel walks the rows itself, so projections and FFN stay batched on SSM layers too |
| Batched attention prefill + decode | `forward_attn_block(_batched)` with `AttnMode { rope: false }`; `attention`/`attention_prefill` | Needs a scale override: both wrappers hardcode `1/sqrt(head_dim)`, and `AttnParams` already carries `scale` |
| SwiGLU FFN, batched | `forward_hybrid_ffn(_batched)` / dense FFN | Yes |
| RMSNorm, gemv/gemm, `silu_and_mul`, `add_inplace` | `kernels.rs` | Yes |
| Residual / embedding / logit scaling | Gemma 4's `scale_kernel`, host-side embedding scaling | Embedding and logit scales on the host; the residual scale needs a scale-then-add without FMA, so new `scaled_add_kernel` |
| Tied LM head, lazy `token_embd` | `LmHead::TiedLazy` | Yes |
| BPE `dbrx` split | `unicode_regex_split_custom_llama3` (`PreType::Llama3`) | New `PreType::Dbrx` with `ignore_merges = false` |

## Implementation

- **`src/model/granite_hybrid.rs`**: `Model::load_granite_hybrid`, `GraniteHybridModel`
  and the forward passes. It's a separate module like `gemma4.rs` rather than another
  `HybridLayerWeights` variant, because the Qwen3.5 attention (gated, partial RoPE,
  QK-norm) and Granite's share nothing but the kernel. One layer function serves
  prefill and decode:
  - **SSM mixer.** The in/out projections are one GEMM over all prompt rows. The
    recurrence runs sequentially over the rows inside one launch per kernel, so there is
    no per-row host loop and no `extract_row`/`write_row`.
  - **Attention mixer.** It calls `attention_online` directly with `attention.scale`,
    as Gemma 4 does with its scale of 1.0. `REFLEX_ATTN_KERNEL=legacy` is rejected,
    because the legacy kernels fix the scale.
  - **Scales.** The residual scale goes through `scaled_add_kernel` (`h + m·s`, with
    the product rounded on its own). The embedding and logit scales are f32 multiplies
    on the host.
- **`src/model/config.rs`**: `parse_granite_hybrid_config`, `GraniteHybridConfig`,
  `Mamba2Config`. `parse_model_config` now rejects the other SSM architectures by name.
- **`kernels_cuda/mamba2.cu`**: the shape is passed as one by-value `Mamba2Dims` struct,
  because cudarc launches take at most 12 arguments.
  - `mamba2_conv_kernel`: one thread per conv channel. It does the depthwise conv, the
    bias, SiLU and the state shift.
  - `mamba2_scan_kernel`: steps 4-5, with one warp per `(head, p)`. Lane `l` holds the
    state values `n = 32j + l` in registers (fully unrolled loop, no local memory). The
    warp reduces `y` with an xor butterfly. This is the layout of llama.cpp's
    `ssm_scan_f32_group`.
  - `mamba2_gate_norm_kernel`: steps 6-7, one block per `(row, group)`. The sum of
    squares is serial in index order, the same way `rmsnorm_kernel` computes it.
- **`reference/mamba2_reference.rs`**: the host reference for steps 2-7. It was written
  before the kernels and is compiled into the test build only.
- **Tokenizer**: `PreType::Dbrx`, the `llama-bpe` split without `ignore_merges`.
- **`is_matrix_weight`**: `ssm_a`, `ssm_d` and `ssm_norm` stay f32 under
  `--weights f16`. They are 2-D in the GGUF (`[1, H]` and `[D/G, G]`), but the kernels
  read them as vectors.

Per-layer state for the 350M:
- conv state: `3 × 1792` floats
- SSM state: `48 × 32 × 128` floats (768 KB)

That's 21.5 MB over 28 layers, allocated once per process and the same at any context
length. Only the 4 attention layers grow a KV cache.

## Fixture

`test-data/tiny-granitehybrid.gguf` (60 MB, f32, sha256 `e12a8bbe…08b74`) comes from a
random `GraniteMoeHybridForCausalLM` checkpoint, run through `79e2e74`'s unmodified
`convert_hf_to_gguf.py --outtype f32`. `build_tiny_granitehybrid.py` writes the
checkpoint directly as safetensors.

The source archive, `test-data/tiny-granitehybrid-src.tar.gz`, holds:
- the builder and the Dockerfile
- the real 350M's config and tokenizer files
- the prompts and `ref.txt`
- the ablation patch and its outputs

- **Shapes.** Chosen to keep llama.cpp's reference on its CUDA kernels; `supports_op`
  falls back to the CPU otherwise. The CUDA constraints are `d_state` ∈ {96, 128, 256},
  `head_dim % 16 == 0` and `conv_dim % 128 == 0`.
  - hidden 128, `d_inner` 256, 8 SSM heads of 32, `d_state` 128
  - 2 groups, so the conv dim is 768 = 6 × 128
  - `d_conv` 4
  - 6 layers: `[ssm, ssm, attn, ssm, ssm, attn]`
  - attention: 4 query / 2 KV heads of 32, no RoPE
  - FFN 256
  - muP scales: embedding 12, residual 0.25, attention 1/32, logits ÷3
  - tied embeddings
- **Weight scales.** A random model with tied embeddings has two failure modes. It
  repeats the current token when that token's own embedding dominates the residual
  stream. It gives near-tie logits when the embeddings are small. The first attempts hit
  both: top-1/top-2 margins of 0.001-0.03, or only 2-6 distinct tokens in 20.
  - Scaling every weight by one factor gives the same tokens, with the logits scaled by
    that factor.
  - So the mix was chosen on a small grid (`grid.sh`) and then doubled: embeddings
    std 0.7, `out_proj` and FFN down std 4, `o_proj` std 2.8.
  - Result: 9-18 distinct tokens per 20, and first-step margins of 0.05-2.0.
- **Sensitivity.** `ablate_mamba2.py` adds env-gated ablations to llama.cpp's
  `build_mamba2_layer`; with the variables unset the graph is unchanged. Each muP scale
  is changed by rewriting its metadata value (`gguf_set_metadata.py`). Every change but
  one alters the 20-token greedy output of 7/7 prompts, almost always from the first
  token:

  | Dropped / changed | Prompts changed | First divergence |
  |---|---|---|
  | `D` skip | 7/7 | 0,0,0,0,0,0,0 |
  | `dt_bias` | 7/7 | 0,0,0,0,0,0,0 |
  | `z` gate | 7/7 | 0,0,0,0,0,0,0 |
  | conv bias | 7/7 | 0,0,0,0,0,0,1 |
  | grouped RMSNorm | 7/7 | 0,0,0,0,0,0,0 |
  | `embedding_scale` → 1 | 7/7 | 0,0,0,0,0,0,0 |
  | `residual_scale` → 1 | 7/7 | 0,0,0,0,0,0,0 |
  | `attention.scale` → 0 (1/√32) | 7/7 | 0,0,0,0,0,0,0 |
  | `logit_scale` → 1 | 0/7 | (a uniform logit scale can't move an argmax) |

  The logit scale is checked through the first-step logit values instead
  (`granite_fixture_first_logits_match_llamacpp`). The group mapping (2 groups) is
  checked by the kernel test against the host reference.
- **Goldens.** llama.cpp `79e2e74` greedy ids: CPU backend, f32 weights and KV, stopping
  only at EOS, 7 prompts × 20 tokens (`ref.txt`). They are `FIXTURE_GOLDEN` in
  `src/model/granite_hybrid_tests.rs`. The prompts are tiny-qwen2's seven, rewritten in
  Granite's chat format; "long" is 981 tokens. The GPU session re-records them with the
  CUDA build.

`test-data/granite-tokenizer.gguf` is the tokenizer-only 350M head, used by the `dbrx`
golden test. It matches `llama-tokenize`'s ids on all 13 strings of the test set. Reflex
adds no BOS (`add_bos_token = false`).

## Tests

Host-only:
- The real Granite headers: 350M, 1B and Micro parse. Tiny is rejected, naming
  `expert_count = 64`.
- The RoPE switch set to true, or absent, is rejected.
- Unsupported SSM shapes are rejected.
- Synthetic headers cover the scale defaults, biases and incomplete SSM layers.
- The Falcon-H1, Nemotron-H, Jamba and Mamba-2 headers are rejected by name.
- The fixture's header parses.
- The reference's own unit tests run against hand-computed values.
- `dbrx` goldens, and no BOS.

GPU (`#[ignore]`d; `scripts/gpu_nightly_tests.sh` has a rule for each):
- `mamba2_kernels_match_host_reference`: conv, scan and gate+norm against the
  reference, over one 7-row prefill launch, then 3 single-row decode launches on the
  same state.
- `mamba2_scaled_add_matches_host_exactly`: bit-exact.
- The `granite_fixture_*` tests:
  - greedy ids vs llama.cpp, and prompt lengths
  - first-step logits within 2e-3
  - batched vs sequential prefill, in f32 and f16
  - the f16 first token on prompts with a clear top-1
  - system1 scores equal to the full logits, and multi-token candidates rejected
  - LoRA and KV export rejected
- `granite_real_model_batched_prefill_matches_sequential` (`GRANITE_GGUF`).

## Verification

- Host (`REFLEX_SKIP_CUDA=1`): config tests on the saved headers, the rejection tests,
  the `dbrx` split vs `llama-tokenize` goldens, and the host reference's own unit tests
  (state update and grouping against hand-computed values).
- GPU, every comparison against a fresh llama.cpp `79e2e74` CUDA build:
  - Fixture: 7 prompts × 20 greedy tokens, f32 and f16 weights, kernel-vs-host tests,
    batched-vs-sequential prefill.
  - Granite 4.0-H 350M Q8_0, and 1B if time allows: the same prompts against llama.cpp
    with f32 activations (`LLAMA_NO_Q8=1` patch). Stock Q8 activations move Q8_0 logits
    (see Gemma 4 and Kolibri-1).
  - Prompts stay at or under 128 tokens. Above that, llama.cpp's CUDA backend switches
    the Mamba-2 scan to its chunked SSD matmul path (`SSM_SSD_MIN_TOKENS`, Turing+).
    That is different arithmetic from the sequential recurrence Reflex runs. Long
    prompts are compared once separately: against llama.cpp's CPU backend, or reported
    as near-ties with margins if they differ.
  - `reflex check` base vs branch on every existing fixture plus Qwen3-0.6B (no change
    expected), and the full `scripts/gpu_nightly_tests.sh` with 0 failed. The new tests
    get `requirement()` rules in the same PR.
  - Cold start (process start to first token) vs `llama-simple`, n=30. Per-token decode
    cost at 64, 512 and 2048 context: flat on the SSM layers, linear only in the 4
    attention layers.

## Results (2026-10-15)

Run on a Runpod RTX 4000 Ada (20 GB, sm_89, secure cloud, ~1.6 h, ~$0.45). The reference
was llama.cpp `79e2e74` built with CUDA plus the `LLAMA_NO_Q8` patch, which keeps
activations in f32. Scripts and every output are in
`test-data/tiny-granitehybrid-src.tar.gz`, under `gpu-rtx4000ada/`.

**Greedy ids vs llama.cpp** (20 tokens per prompt; "f32ref" means llama.cpp run with f32
activations):

| Model, Reflex weights | Reference | Prompts identical | Divergence: step, llama.cpp's top-2 margin |
|---|---|---|---|
| fixture, f32 | CUDA | 7/7 | – |
| fixture, f32 | CUDA, flash attention off | 7/7 | – |
| fixture, f16 | CUDA | 7/7 | – |
| fixture "long" (981 tokens), f32 | CPU | 1/1 | – |
| 350M Q8_0, f32 | f32ref | 5/6 | arabic, step 17, 0.0019 |
| 350M Q8_0, f16 | f32ref | 5/6 | arabic, step 17, 0.0019 |
| 350M Q8_0, f32 | stock (8-bit activations) | 4/6 | code step 7, 0.074; arabic step 17 |
| 350M "long", f32 | f32ref, and CPU | 1/1, 1/1 | – |
| 1B Q8_0, f32 | f32ref | 5/6 | story, step 5, 0.033 |
| 1B Q8_0, f16 | f32ref | 5/6 | story, step 5, 0.033 |

- **Divergences.** At every one, Reflex picks llama.cpp's own runner-up.
- **Long prompts.** The 981-token prompts run llama.cpp's CUDA chunked-SSD scan path
  (above 128 tokens). Reflex still matches it there, and matches the CPU backend too.
- **Fixture goldens.** The CUDA build's fixture goldens are identical to the CPU ones in
  `ref.txt`.

**Tests:**
- Fixture GPU tests: 6/6. Kernel-vs-reference: 2/2.
- Real-model batched vs sequential prefill, f32: rel_l2 ≈ 1e-6 on 350M and 1B.
- Full `cargo test`: passes.
- `scripts/gpu_nightly_tests.sh`: 65 passed, 0 failed, 3 skipped. The skips are the
  optional IQ, real DeepSeek-V2-Lite and real Qwen2.5 models, which weren't provided.

**Regression.** `reflex check` ran on github/master and on the branch, same pod, 3
prompts each. All 48 existing rows are identical, token ids and logit checksums
included: every fixture, Qwen3-0.6B (f32, f16, quantized-resident) and Qwen3.5-0.8B. The
12 new Granite rows are rejected by master, as expected.

**Cold start** (`reflex generate --max-tokens 1` vs `llama-simple -n 1 --no-warmup`,
350M Q8_0, n=30): Reflex 0.76 s p50 / 0.89 s p95, `llama-simple` 0.83 s / 0.91 s.

**Decode cost** is flat in context length on the SSM layers. Reflex was timed in
process (prefill, then 128 teacher-forced steps including the full-vocab logits copy);
llama.cpp with `llama-bench tg128 -d`:

| Model | Context | Reflex f32 (ms/token) | Reflex f16 (ms/token) | llama.cpp (tok/s) |
|---|---|---|---|---|
| 350M | 57 / 505 / 2,045 / 8,191 | 7.18 / 6.55 / 6.55 / 6.99 | 5.15 / 4.52 / 4.50 / 4.96 | 394 / 395 / 390 / 375 (depth 64 / 512 / 2,048 / 8,192) |
| 1B | same | 22.8 / 21.7 / 21.7 / 22.4 | 14.0 / 12.9 / 12.9 / 13.6 | 151 / 150 / 149 / 146 |

The ~10% rise at 8K comes from the 4 attention layers. Steady-state decode is 1.8-2x
slower than llama.cpp: 222 vs 394 tok/s on the 350M, f16. Each step pays a
host-to-device embedding upload and the logits copy, and llama.cpp runs Q8_0 matmuls
under CUDA graphs. Throughput isn't this engine's target, so this is recorded rather
than chased here.

## Rejected variants

- **Falcon-H1**: parallel attention + SSM in every layer, both state kinds per layer,
  RoPE. The Tiny file's `d_state = 64` has no llama.cpp CUDA scan kernel. Falcon LLM
  license. Reconsider after Granite: the mixer is the same, only the layer wiring
  differs.
- **Nemotron-H / Nemotron-3-Nano**: ReLU² FFN-only layers, `n_group = 8`, `tekken`
  (8B) or `pixtral` pre types. The converter needs `trust_remote_code`, since no
  `transformers` release has the class. NVIDIA licenses.
- **Jamba** (all sizes): Mamba-1 SSM layers (per-element `A`, low-rank `dt`, dt/B/C
  RMSNorms). That is a different scan from Mamba-2, so it falls outside this phase.
- **Pure `mamba2`**: no first-party GGUF (130m is a community conversion; Codestral
  is 7.75 GB with a SentencePiece `.v3` tokenizer that needs renaming). A cheap
  follow-on once the mixer exists.
- **Chunked (SSD) parallel prefill**: out for now, the same naive-first decision as
  Gated DeltaNet. Prefill on SSM layers stays linear and sequential.
