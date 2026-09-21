//! Dense and MoE Qwen3 forward pass (MVP steps 1-2, see README.md's MVP
//! order): loads a GGUF file's weights, runs embedding -> every transformer
//! layer -> final RMSNorm -> LM head -> argmax for the *first* generated
//! token only. No KV cache reuse across separate calls, no batching, no
//! sampling beyond greedy argmax -- the target metric is
//! process-start-to-first-token latency, not sustained decode throughput
//! (see README.md's "Why this exists").
//!
//! Architecture and math ported from RustFeference's own most mature, most-
//! verified model code (`rft-gpu/src/generate.rs` + `dispatch.rs` +
//! `moe.rs`, git history around commits `d8ed273` "minimal end-to-end dense
//! forward pass", `1459330` "Qwen3 architecture support", and `6a70287`
//! "qwen3moe support") as the correctness oracle, not copied wholesale:
//! RustFeference's serving/paged-KV-cache/tensor-parallel machinery is all
//! out of scope here (see MVP scope discussion) -- kernels below are
//! deliberately fresh, simple, from-scratch AOT kernels, not ports of
//! RustFeference's own (far more complex, paged/batched/fused) CUDA source.
//!
//! MoE scope (MVP step 2): naive per-token expert dispatch -- one `gemv`
//! call per selected expert per FFN matrix, no batched/grouped-by-expert
//! GEMM -- which RustFeference's own docs call the correct starting point.
//! The attention block is byte-for-byte identical between dense and MoE
//! layers (shared via `Model::forward_attn_block`); MoE only replaces the
//! single shared FFN with a router (softmax + top-k, `crate::moe::route_top_k`)
//! over per-expert-stacked SwiGLU weights. No real small `qwen3moe`-
//! architecture GGUF was available to test against, so this was verified
//! against a real (Mixtral-style, `general.architecture = "llama"`, no
//! QK-Norm) `Tiny-Moe.Q4_K_M.gguf` fixture instead -- the MoE routing/dispatch
//! math is architecture-agnostic (see `crate::moe`'s doc comment), and the
//! shared attention block already covers Qwen3's QK-Norm separately (dense
//! Qwen3 MVP step 1).
//!
//! GEMM convention throughout: `y = x @ W^T` (`nn.Linear`), where a real
//! GGUF weight tensor's parsed `shape` is `[in_features, out_features]`
//! (confirmed against RustFeference's own `parse_model_config`/`gemm_shape`
//! usage of the identical, unmodified `gguf.rs` parser this crate salvaged)
//! and its flat dequantized bytes are already row-major
//! `(out_features, in_features)` -- exactly `gemv_kernel`'s expected layout,
//! no transpose needed. A per-expert-stacked MoE tensor's shape is
//! `[in_features, out_features, expert_count]` (confirmed against
//! llama.cpp's `qwen3moe.cpp`), and expert `e`'s `in_features * out_features`
//! chunk is contiguous and already in that same 2-D layout -- see
//! `Model::gemv_expert`.

use crate::aot::{self, AotKernel};
use crate::dequant;
use crate::gguf::{GgufFile, GgufValue};
use crate::moe::route_top_k;
use crate::tokenizer::Tokenizer;
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

fn u64_meta(file: &GgufFile, key: &str) -> Option<u64> {
    file.metadata.get(key).and_then(GgufValue::as_u64)
}

fn f32_meta(file: &GgufFile, key: &str) -> Option<f32> {
    file.metadata.get(key).and_then(GgufValue::as_f32)
}

/// Static shape/hyperparameter config for one dense Qwen3 transformer layer,
/// read from the GGUF file's `qwen3.*` metadata.
#[derive(Clone)]
pub struct LayerConfig {
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Read explicitly from `qwen3.attention.key_length` -- Qwen3 decouples
    /// this from `hidden_size / num_q_heads` (real finding from
    /// RustFeference's own Phase 21.14: a real Qwen3-0.6B has head_dim=128,
    /// not the 64 that division would give).
    pub head_dim: usize,
    /// Number of leading dims of each head RoPE actually rotates (GPT-NeoX
    /// half-rotation convention). Equal to `head_dim` (full rotary) for
    /// dense/MoE Qwen3; Qwen3.5's Gated Attention layers use a real partial
    /// value from `qwen35.rope.dimension_count` (see `Model::load_hybrid`).
    pub rotary_dim: usize,
    pub ffn_hidden_size: usize,
    pub rope_base: f32,
    pub rmsnorm_eps: f32,
}

/// A weight tensor, dequantized to `f32` once at load time and uploaded to
/// device memory immediately after (see `Model::load`'s `load_weight`) so
/// no forward-pass call re-uploads it -- kernels below take `&self.data`
/// (or a zero-copy `CudaView` slice of it, for per-expert MoE tensors)
/// directly. Shape is the original GGUF shape (`[in_features,
/// out_features]` for a 2-D `nn.Linear`-style weight, `[hidden_size]` for a
/// norm weight).
struct Weight {
    data: CudaSlice<f32>,
    shape: Vec<u64>,
}

struct DenseLayerWeights {
    attn_norm: Weight,
    attn_q: Weight,
    attn_k: Weight,
    attn_v: Weight,
    attn_output: Weight,
    /// Per-head RMSNorm on Q before RoPE (Qwen3's QK-Norm). `None` for
    /// architectures without it -- presence of the tensor, not a separate
    /// config flag, gates whether the forward pass applies it.
    attn_q_norm: Option<Weight>,
    attn_k_norm: Option<Weight>,
    ffn_norm: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

/// One MoE transformer layer's weights: an attention block identical in
/// shape/meaning to [`DenseLayerWeights`]'s (shared at forward time via
/// `Model::forward_attn_block`), plus a router (`ffn_gate_inp`, `[hidden_size,
/// expert_count]`) and per-expert-stacked SwiGLU weights (`ffn_gate_exps`/
/// `ffn_up_exps`/`ffn_down_exps`, each `[in_features, out_features,
/// expert_count]`) in place of the dense path's single shared FFN.
struct MoeLayerWeights {
    attn_norm: Weight,
    attn_q: Weight,
    attn_k: Weight,
    attn_v: Weight,
    attn_output: Weight,
    attn_q_norm: Option<Weight>,
    attn_k_norm: Option<Weight>,
    ffn_norm: Weight,
    ffn_gate_inp: Weight,
    ffn_gate_exps: Weight,
    ffn_up_exps: Weight,
    ffn_down_exps: Weight,
}

enum LayerWeights {
    Dense(DenseLayerWeights),
    Moe(MoeLayerWeights),
}

/// `<arch>.expert_count`/`<arch>.expert_used_count` metadata for an MoE
/// model -- present (and `expert_count` nonzero) iff the file describes an
/// MoE architecture. See `crate::moe`'s doc comment for why this is keyed by
/// the file's own architecture string rather than hardcoded to `llama.*`.
pub struct MoeMetaConfig {
    pub expert_count: usize,
    pub expert_used_count: usize,
}

/// Derive [`LayerConfig`], the layer count, and (for an MoE architecture)
/// [`MoeMetaConfig`] from a GGUF file's `<arch>.*` metadata (plus
/// `blk.0.ffn_gate[_exps].weight`'s real shape, since GGUF has no dedicated
/// "FFN hidden size" metadata key). Pure/host-only, no GPU required.
/// Supports dense `qwen3` and any architecture reporting a nonzero
/// `<arch>.expert_count` (real `qwen3moe`, or a Mixtral-style test fixture
/// under `llama.*`) -- other architectures are out of scope (see README.md's
/// MVP order).
pub fn parse_model_config(file: &GgufFile) -> Result<(LayerConfig, usize, Option<MoeMetaConfig>), String> {
    let architecture = file.metadata.get("general.architecture").and_then(GgufValue::as_str).unwrap_or("");

    let moe = match u64_meta(file, &format!("{architecture}.expert_count")).filter(|&n| n > 0) {
        Some(expert_count) => {
            let expert_used_count = u64_meta(file, &format!("{architecture}.expert_used_count"))
                .ok_or_else(|| format!("missing {architecture}.expert_used_count metadata key"))?;
            Some(MoeMetaConfig { expert_count: expert_count as usize, expert_used_count: expert_used_count as usize })
        }
        None => None,
    };

    if architecture != "qwen3" && moe.is_none() {
        return Err(format!(
            "unsupported architecture '{architecture}': only dense 'qwen3' and MoE architectures reporting a nonzero '{architecture}.expert_count' are in scope for this MVP"
        ));
    }

    let block_count =
        u64_meta(file, &format!("{architecture}.block_count")).ok_or_else(|| format!("missing {architecture}.block_count metadata key"))? as usize;
    let hidden_size = u64_meta(file, &format!("{architecture}.embedding_length"))
        .ok_or_else(|| format!("missing {architecture}.embedding_length metadata key"))? as usize;
    let num_q_heads = u64_meta(file, &format!("{architecture}.attention.head_count"))
        .ok_or_else(|| format!("missing {architecture}.attention.head_count metadata key"))? as usize;
    let num_kv_heads =
        u64_meta(file, &format!("{architecture}.attention.head_count_kv")).unwrap_or(num_q_heads as u64) as usize;
    // Qwen3 (dense and MoE) decouples head_dim from hidden_size/num_q_heads via
    // this key. Non-Qwen3 MoE fixtures (e.g. a Mixtral-style test GGUF with no
    // per-head decoupling) don't set it, so fall back to the standard derivation
    // rather than hard-failing -- real Qwen3 files always have the key, so this
    // fallback only ever engages for such fixtures.
    let head_dim = u64_meta(file, &format!("{architecture}.attention.key_length"))
        .map(|n| n as usize)
        .unwrap_or(hidden_size / num_q_heads);

    let ffn_gate_weight_name = if moe.is_some() { "blk.0.ffn_gate_exps.weight" } else { "blk.0.ffn_gate.weight" };
    let ffn_gate_info = file.tensor_info(ffn_gate_weight_name).ok_or_else(|| format!("missing {ffn_gate_weight_name} tensor"))?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        [_in_features, out_features, _expert_count] => *out_features as usize,
        other => return Err(format!("{ffn_gate_weight_name} has unexpected shape {other:?}")),
    };

    let rope_base = f32_meta(file, &format!("{architecture}.rope.freq_base")).unwrap_or(10000.0);
    let rmsnorm_eps = f32_meta(file, &format!("{architecture}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);

    Ok((
        LayerConfig { hidden_size, num_q_heads, num_kv_heads, head_dim, rotary_dim: head_dim, ffn_hidden_size, rope_base, rmsnorm_eps },
        block_count,
        moe,
    ))
}

/// Which trunk blocks of a Qwen3.5 hybrid model are Gated DeltaNet layers
/// (`true`) vs. Gated Attention layers (`false`), read from metadata --
/// never hardcoded (a wrong pattern would run the wrong mixer on real
/// weights). Ported from RustFeference's `hybrid.rs` `HybridConfig::parse`:
/// an explicit `{arch}.attention.recurrent_layers` boolean array wins when
/// present, otherwise every `full_attention_interval`-th block (the last of
/// each group) is Gated Attention and the rest are Gated DeltaNet. The real
/// `Qwen3.5-0.8B` fixture uses the interval fallback (`full_attention_interval
/// = 4`), giving Gated Attention at trunk indices `[3, 7, 11, 15, 19, 23]`.
fn parse_hybrid_layer_kinds(file: &GgufFile, architecture: &str, block_count: usize) -> Result<Vec<bool>, String> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let recurrent_key = key("attention.recurrent_layers");
    match file.metadata.get(&recurrent_key) {
        Some(GgufValue::Array(items)) => {
            if items.len() != block_count {
                return Err(format!("{recurrent_key} has {} entries but block_count is {block_count}", items.len()));
            }
            items
                .iter()
                .map(|v| match v {
                    GgufValue::Bool(b) => Ok(*b),
                    other => {
                        other.as_u64().map(|x| x != 0).ok_or_else(|| format!("{recurrent_key} has non-boolean entry {other:?}"))
                    }
                })
                .collect()
        }
        Some(other) => Err(format!("{recurrent_key} must be an array, got {other:?}")),
        None => {
            let interval_key = key("full_attention_interval");
            let interval = u64_meta(file, &interval_key)
                .ok_or_else(|| format!("{architecture} model has neither {recurrent_key} nor {interval_key}"))? as usize;
            if interval == 0 {
                return Err(format!("{interval_key} must be > 0"));
            }
            Ok((0..block_count).map(|i| (i + 1) % interval != 0).collect())
        }
    }
}

/// Derives [`MlaConfig`] and the layer count from a GGUF file's `deepseek2.*`
/// metadata. Scope deliberately narrowed to what's needed for a first, narrow MLA
/// implementation (same "naive/narrow first" precedent as every prior MVP step),
/// each rejected case matching existing precedent (`qwen35moe`/MTP rejection):
/// hard errors on Q-LoRA query decomposition (`attention.q_lora_rank` present and
/// nonzero), MoE layers (`leading_dense_block_count < block_count` -- a real
/// DeepSeek-V2/V3 file always has MoE layers past its dense-lead layers; only a
/// dense-only file, like this MVP step's synthetic test fixture, is in scope),
/// YaRN RoPE scaling, and MTP/NextN blocks. No small real `deepseek2`-architecture
/// GGUF exists publicly (see README.md) -- verified against a synthetic fixture
/// built via llama.cpp's own real `convert_hf_to_gguf.py` (authentic tensor
/// layout, random weights), cross-checked against a real llama.cpp build.
fn parse_mla_config(file: &GgufFile) -> Result<(MlaConfig, usize), String> {
    let architecture = "deepseek2";
    let key = |suffix: &str| format!("{architecture}.{suffix}");

    let block_count = u64_meta(file, &key("block_count")).ok_or_else(|| format!("missing {}", key("block_count")))? as usize;

    let nextn = u64_meta(file, &key("nextn_predict_layers")).unwrap_or(0);
    if nextn != 0 {
        return Err(format!(
            "{} MTP/NextN blocks (nextn_predict_layers={nextn}) are not supported by this MVP",
            key("nextn_predict_layers")
        ));
    }

    let leading_dense = u64_meta(file, &key("leading_dense_block_count"))
        .ok_or_else(|| format!("missing {}", key("leading_dense_block_count")))? as usize;
    if leading_dense < block_count {
        return Err(format!(
            "{} ({leading_dense}) < block_count ({block_count}): MoE layers present, not supported by this MVP (dense-only deepseek2 files only)",
            key("leading_dense_block_count")
        ));
    }

    if u64_meta(file, &key("attention.q_lora_rank")).filter(|&n| n > 0).is_some() {
        return Err(format!("{} (Q-LoRA query decomposition) is not supported by this MVP", key("attention.q_lora_rank")));
    }

    let rope_scaling_type = file.metadata.get(&key("rope.scaling.type")).and_then(GgufValue::as_str);
    if matches!(rope_scaling_type, Some(t) if t != "none") {
        return Err(format!("{} = {rope_scaling_type:?} (YaRN/RoPE scaling) is not supported by this MVP", key("rope.scaling.type")));
    }

    let hidden_size = u64_meta(file, &key("embedding_length")).ok_or_else(|| format!("missing {}", key("embedding_length")))? as usize;
    let num_heads = u64_meta(file, &key("attention.head_count")).ok_or_else(|| format!("missing {}", key("attention.head_count")))? as usize;
    let kv_lora_rank =
        u64_meta(file, &key("attention.kv_lora_rank")).ok_or_else(|| format!("missing {}", key("attention.kv_lora_rank")))? as usize;
    let n_embd_head_k_mla = u64_meta(file, &key("attention.key_length_mla"))
        .ok_or_else(|| format!("missing {}", key("attention.key_length_mla")))? as usize;
    let v_head_dim = u64_meta(file, &key("attention.value_length_mla"))
        .ok_or_else(|| format!("missing {}", key("attention.value_length_mla")))? as usize;
    let qk_rope_head_dim =
        u64_meta(file, &key("rope.dimension_count")).ok_or_else(|| format!("missing {}", key("rope.dimension_count")))? as usize;
    if n_embd_head_k_mla <= qk_rope_head_dim {
        return Err(format!(
            "{} ({n_embd_head_k_mla}) must be greater than {} ({qk_rope_head_dim})",
            key("attention.key_length_mla"),
            key("rope.dimension_count")
        ));
    }
    let qk_nope_head_dim = n_embd_head_k_mla - qk_rope_head_dim;

    let ffn_gate_info = file
        .tensor_info("blk.0.ffn_gate.weight")
        .ok_or("missing blk.0.ffn_gate.weight tensor (MoE-only deepseek2 files are not supported by this MVP)")?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        other => return Err(format!("blk.0.ffn_gate.weight has unexpected shape {other:?}")),
    };

    // Sanity-check `attention.value_length_mla` against `wv_b`'s own shape (the
    // tensor `Model::gemv_per_head` actually derives its decompressed output width
    // from) -- catches a malformed/mismatched real file early rather than silently
    // producing a wrong-sized attention output deep in the forward pass.
    let wv_b_info = file.tensor_info("blk.0.attn_v_b.weight").ok_or("missing blk.0.attn_v_b.weight tensor")?;
    match wv_b_info.shape.as_slice() {
        [_in_features, out_features, _n_head] if *out_features as usize == v_head_dim => {}
        other => return Err(format!("blk.0.attn_v_b.weight shape {other:?} doesn't match {} ({v_head_dim})", key("attention.value_length_mla"))),
    }

    let rope_base = f32_meta(file, &key("rope.freq_base")).unwrap_or(10000.0);
    let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

    Ok((
        MlaConfig { hidden_size, num_heads, qk_rope_head_dim, qk_nope_head_dim, kv_lora_rank, ffn_hidden_size, rope_base, rmsnorm_eps },
        block_count,
    ))
}

/// One Gated Attention transformer layer's weights (Qwen3.5 hybrid, see
/// `HybridModel`). Differs from [`DenseLayerWeights`]'s attention block in
/// exactly two ways (confirmed against `reference/gated_deltanet_rustfeference.rs`'s
/// `GatedAttentionWeights`/`gated_attention_step`, itself ported from real
/// llama.cpp `qwen35.cpp`): `attn_q` is a *fused* query+gate projection
/// (`[hidden, 2*num_q_heads*head_dim]`, per head `[q(head_dim),
/// gate(head_dim)]`), and the attention output is gated by `sigmoid(gate)`
/// before the output projection. QK-Norm and (partial) RoPE are otherwise
/// identical to the dense path. `post_attn_norm` is this architecture's
/// pre-FFN norm tensor -- named `post_attention_norm.weight` in the real
/// GGUF, not `ffn_norm.weight` (confirmed against a real
/// `Qwen3.5-0.8B-Q4_K_M.gguf` header).
struct GatedAttnLayerWeights {
    attn_norm: Weight,
    attn_q: Weight,
    attn_k: Weight,
    attn_v: Weight,
    attn_q_norm: Weight,
    attn_k_norm: Weight,
    attn_output: Weight,
    post_attn_norm: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

/// One Gated DeltaNet transformer layer's weights (Qwen3.5 hybrid). Tensor
/// names and shapes confirmed against `reference/gated_deltanet_rustfeference.rs`'s
/// module doc comment and a real `Qwen3.5-0.8B-Q4_K_M.gguf` header. `ssm_a`
/// has no `.weight`/`.bias` suffix in the real file (already stored as
/// `-exp(A_log)`, per the reference).
struct GatedDeltaNetLayerWeights {
    attn_norm: Weight,
    /// `[hidden, 2*key_dim + value_dim]`, fused q/k/v the causal conv runs over.
    attn_qkv: Weight,
    /// `[hidden, value_dim]`, the gated-output gate `z`.
    attn_gate: Weight,
    ssm_beta: Weight,
    ssm_alpha: Weight,
    ssm_dt: Weight,
    ssm_a: Weight,
    ssm_conv1d: Weight,
    ssm_norm: Weight,
    ssm_out: Weight,
    post_attn_norm: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

enum HybridLayerWeights {
    GatedAttention(GatedAttnLayerWeights),
    GatedDeltaNet(GatedDeltaNetLayerWeights),
}

/// Shape/hyperparameter config for a DeepSeek-V2/V3 Multi-head Latent Attention
/// (MLA) model (MVP step 4), read from the GGUF file's `deepseek2.*` metadata by
/// `parse_mla_config`. Scope deliberately narrowed (see that function's doc
/// comment): dense-only (no MoE FFN), no Q-LoRA query decomposition (`is_lite`-style
/// direct `wq` only), no YaRN RoPE scaling, no MTP/NextN.
struct MlaConfig {
    hidden_size: usize,
    num_heads: usize,
    /// Per-head dim of the RoPE-rotated part of Q/K (`qk_rope_head_dim` in real
    /// DeepSeek configs). Also `rope_dim` -- full rotation, no partial-head split
    /// like Qwen3.5's Gated Attention layers (see `LayerConfig::rotary_dim`'s doc
    /// comment) -- MLA's `q_pe`/`k_pe` are already separately-extracted buffers of
    /// exactly this width, not a slice of a wider head.
    qk_rope_head_dim: usize,
    /// Per-head dim of the non-rotated part of Q (and, after decompression via
    /// `wk_b`, of the K side too -- see `Model::forward_mla_attn_block`).
    qk_nope_head_dim: usize,
    /// Compressed KV-cache dimension shared by every head (MQA) -- the whole point
    /// of "latent" attention. Also the per-head width of `Vcur` before
    /// decompression via `wv_b`.
    kv_lora_rank: usize,
    ffn_hidden_size: usize,
    rope_base: f32,
    rmsnorm_eps: f32,
}

/// One MLA layer's weights. Tensor names/shapes confirmed against a real
/// `llama.cpp` build's `src/models/deepseek2.cpp` (`is_mla && is_lite` branch) and a
/// synthetic `deepseek2`-architecture GGUF fixture built for this MVP step (see
/// README.md -- no small real `deepseek2` GGUF exists publicly). `wk_b`/`wv_b` are
/// per-head-stacked tensors (`[in_features, out_features, n_head]`, the same
/// layout convention as MoE's per-expert tensors -- see `Model::gemv_expert`'s doc
/// comment -- just "expert" -> "head"; every head is always used here, unlike MoE's
/// top-k selection). Dense SwiGLU FFN (`ffn_gate`/`ffn_up`/`ffn_down`), identical in
/// shape/meaning to `DenseLayerWeights`'s (MoE FFN is out of scope this round).
struct MlaLayerWeights {
    attn_norm: Weight,
    /// `[hidden, n_head*(qk_nope_head_dim+qk_rope_head_dim)]` -- direct projection,
    /// no Q-LoRA decomposition (out of scope this round).
    wq: Weight,
    /// `[hidden, kv_lora_rank+qk_rope_head_dim]`, fused compressed-KV + shared
    /// rope-K projection (MQA: a single shared "head").
    wkv_a_mqa: Weight,
    attn_kv_a_norm: Weight,
    /// `[qk_nope_head_dim, kv_lora_rank, n_head]`.
    wk_b: Weight,
    /// `[kv_lora_rank, v_head_dim, n_head]`.
    wv_b: Weight,
    /// `[n_head*v_head_dim, hidden]`.
    wo: Weight,
    ffn_norm: Weight,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

/// A loaded DeepSeek-V2/V3 MLA model's extra state, layered on top of the same
/// [`Model`] every other architecture uses (shared `token_embd`/`output_norm`/
/// `lm_head`/`tokenizer`, and the same `rmsnorm_k`/`rope_k`/`silu_k`/`gemv_k`/
/// `add_k` kernels every other path reuses unchanged -- see `Model::forward_mla_attn_block`).
struct MlaModel {
    cfg: MlaConfig,
    layers: Vec<MlaLayerWeights>,
    mla_attn_k: AotKernel,
    /// `rope_norm_kernel` (`kernels_cuda/rope.cu`), **not** the shared `Model::rope_k`
    /// (`rope_kernel`) every other architecture uses -- confirmed against
    /// llama.cpp's `llama_model_rope_type`, which maps `deepseek2` to
    /// `LLAMA_ROPE_TYPE_NORM` (consecutive-pair rotation), not the
    /// `LLAMA_ROPE_TYPE_NEOX` (half-split) convention Qwen3/Qwen3.5 use.
    rope_norm_k: AotKernel,
}

/// Per-sequence recurrent state for one hybrid layer, matching
/// [`HybridLayerWeights`]'s variant for that layer index one-to-one.
/// Device-resident (Phase 2 round 2): `k_cache`/`v_cache` are preallocated to
/// the full prompt length up front (`forward_prompt_hybrid` knows the token
/// count before the per-position loop starts) and written into directly via
/// device-to-device copy each position -- no host round-trip, unlike the
/// pre-round-2 convention. Same for `conv_state`/`recurrent`, mutated in
/// place on-device by `Model::gdn_conv`/`Model::gdn_delta`.
enum HybridLayerState {
    Attn { k_cache: CudaSlice<f32>, v_cache: CudaSlice<f32> },
    Gdn { conv_state: CudaSlice<f32>, recurrent: CudaSlice<f32> },
}

/// A loaded Qwen3.5 hybrid model's extra state, layered on top of the same
/// [`Model`] every other architecture uses (shared `token_embd`/
/// `output_norm`/`lm_head`/`tokenizer`, and the same `rmsnorm_k`/`rope_k`/
/// `silu_k`/`gemv_k`/`attn_k` kernels the Gated Attention layers and every
/// FFN reuse unchanged). `attn_cfg` is the Gated Attention layers' shape
/// (its `rotary_dim` is the real partial value); `gdn_cfg` is the Gated
/// DeltaNet layers' shape. Both are uniform across every layer of that kind
/// -- a real `qwen35` file has exactly one `qwen35.ssm.*`/`qwen35.attention.*`
/// config, not a per-layer one.
struct HybridModel {
    attn_cfg: LayerConfig,
    gdn_cfg: crate::gated_deltanet::GatedDeltaNetConfig,
    layers: Vec<HybridLayerWeights>,
    gdn_conv_k: AotKernel,
    gdn_l2_norm_k: AotKernel,
    gdn_gates_k: AotKernel,
    gdn_delta_k: AotKernel,
    gdn_gated_norm_k: AotKernel,
}

/// A loaded dense Qwen3 model, ready to [`Model::forward_prompt`] from.
pub struct Model {
    device: Arc<CudaDevice>,
    rmsnorm_k: AotKernel,
    rope_k: AotKernel,
    silu_k: AotKernel,
    gemv_k: AotKernel,
    attn_k: AotKernel,
    /// In-place residual add (`a[i] += b[i]`, see `kernels_cuda/elementwise.cu`)
    /// -- keeps residual-stream adds device-resident (Phase 2 round 2)
    /// instead of downloading both operands to host just to add two vectors.
    add_k: AotKernel,
    cfg: LayerConfig,
    layers: Vec<LayerWeights>,
    /// `k` (top-k expert count), `Some` iff this is an MoE model.
    expert_used_count: Option<usize>,
    /// `[hidden_size, vocab_size]`, row-major `(vocab_size, hidden_size)`
    /// flat data -- kept host-resident (unlike every other weight) for
    /// embedding lookup (host-side gather; batch is always 1 in this MVP, so
    /// a GPU gather kernel buys nothing). When no separate `output.weight`
    /// tensor exists, its dequantized bytes are also uploaded to device
    /// memory once, as `lm_head`, rather than dequantizing them twice.
    token_embd: Vec<f32>,
    output_norm: Weight,
    lm_head: Weight,
    tokenizer: Tokenizer,
    /// `Some` iff this is a Qwen3.5 hybrid model (see `Self::load_hybrid`);
    /// `cfg`/`layers`/`expert_used_count` above are unused garbage in that
    /// case (`forward_prompt` branches on this before touching them).
    hybrid: Option<HybridModel>,
    /// `Some` iff this is a DeepSeek-V2/V3 MLA model (see `Self::load_mla`); like
    /// `hybrid`, `cfg`/`layers`/`expert_used_count` are unused garbage in that case.
    mla: Option<MlaModel>,
}

impl Model {
    pub fn load(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        let architecture = file.metadata.get("general.architecture").and_then(GgufValue::as_str).unwrap_or("");
        if architecture == "qwen35" {
            return Self::load_hybrid(device, file);
        }
        if architecture == "qwen35moe" {
            return Err(
                "qwen35moe (hybrid Gated DeltaNet + routed-MoE FFN) is not yet supported -- only the dense \
                 qwen35 hybrid architecture, and qwen3/qwen3-MoE, are in scope for this MVP"
                    .to_string(),
            );
        }
        if architecture == "deepseek2" {
            return Self::load_mla(device, file);
        }

        let (cfg, block_count, moe) = parse_model_config(file)?;
        let expert_used_count = moe.map(|m| m.expert_used_count);

        let rmsnorm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_RMSNORM"), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope", "rope_kernel")?;
        let silu_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_SILU_AND_MUL"), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_GEMV"), "gemv", "gemv_kernel")?;
        let attn_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ATTENTION"), "attention", "attention_kernel")?;
        let add_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ELEMENTWISE"), "elementwise", "add_kernel")?;

        // Dequantizes straight from the mmap'd GGUF bytes into a scratch host
        // `Vec<f32>`, uploads it to device memory, then drops the host copy
        // (goes out of scope) -- unlike before, no dequantized weight stays
        // host-resident for the model's lifetime, and no forward-pass call
        // re-uploads it (see `Weight`'s doc comment).
        let load_weight = |name: &str| -> Result<Weight, String> {
            let info = file.tensor_info(name).ok_or_else(|| format!("missing weight '{name}'"))?;
            let bytes = file.tensor_bytes(info)?;
            let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
            let data = device.htod_sync_copy(&host).map_err(|e| format!("upload weight '{name}' to device: {e}"))?;
            Ok(Weight { data, shape: info.shape.clone() })
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let attn_norm = load_weight(&format!("blk.{i}.attn_norm.weight"))?;
            let attn_q = load_weight(&format!("blk.{i}.attn_q.weight"))?;
            let attn_k = load_weight(&format!("blk.{i}.attn_k.weight"))?;
            let attn_v = load_weight(&format!("blk.{i}.attn_v.weight"))?;
            let attn_output = load_weight(&format!("blk.{i}.attn_output.weight"))?;
            let attn_q_norm = load_weight(&format!("blk.{i}.attn_q_norm.weight")).ok();
            let attn_k_norm = load_weight(&format!("blk.{i}.attn_k_norm.weight")).ok();
            let ffn_norm = load_weight(&format!("blk.{i}.ffn_norm.weight"))?;

            let layer = if expert_used_count.is_some() {
                LayerWeights::Moe(MoeLayerWeights {
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    ffn_norm,
                    ffn_gate_inp: load_weight(&format!("blk.{i}.ffn_gate_inp.weight"))?,
                    ffn_gate_exps: load_weight(&format!("blk.{i}.ffn_gate_exps.weight"))?,
                    ffn_up_exps: load_weight(&format!("blk.{i}.ffn_up_exps.weight"))?,
                    ffn_down_exps: load_weight(&format!("blk.{i}.ffn_down_exps.weight"))?,
                })
            } else {
                LayerWeights::Dense(DenseLayerWeights {
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    ffn_norm,
                    ffn_gate: load_weight(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load_weight(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load_weight(&format!("blk.{i}.ffn_down.weight"))?,
                })
            };
            layers.push(layer);
        }

        let token_embd_info =
            file.tensor_info("token_embd.weight").ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd =
            dequant::dequantize(token_embd_info.ggml_type, token_embd_bytes, token_embd_info.element_count())?;

        let output_norm = load_weight("output_norm.weight")?;

        // Tied-embedding models have no separate `output.weight` tensor --
        // reuse `token_embd`'s already-dequantized host bytes for the LM
        // head's device upload instead of dequantizing them a second time.
        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
                let data =
                    device.htod_sync_copy(&host).map_err(|e| format!("upload weight 'output.weight' to device: {e}"))?;
                Weight { data, shape: info.shape.clone() }
            }
            None => {
                let data = device
                    .htod_sync_copy(&token_embd)
                    .map_err(|e| format!("upload weight 'token_embd.weight' to device: {e}"))?;
                Weight { data, shape: token_embd_info.shape.clone() }
            }
        };

        let tokenizer = Tokenizer::from_gguf(file)?;

        Ok(Model {
            device,
            rmsnorm_k,
            rope_k,
            silu_k,
            gemv_k,
            attn_k,
            add_k,
            cfg,
            layers,
            expert_used_count,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: None,
            mla: None,
        })
    }

    /// Loads a Qwen3.5 hybrid model: per-layer mixer kind (Gated Attention vs.
    /// Gated DeltaNet) resolved from metadata (never hardcoded -- see
    /// `parse_hybrid_layer_kinds`), MTP/NextN blocks rejected outright (not
    /// in this MVP's scope; a real non-MTP file like `Qwen3.5-0.8B` reports
    /// `nextn_predict_layers` absent/zero). See `HybridModel`'s doc comment
    /// for what's shared with the dense/MoE path.
    fn load_hybrid(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        let architecture = "qwen35";
        let key = |suffix: &str| format!("{architecture}.{suffix}");

        let block_count = u64_meta(file, &key("block_count")).ok_or_else(|| format!("missing {}", key("block_count")))? as usize;
        let nextn = u64_meta(file, &key("nextn_predict_layers")).unwrap_or(0);
        if nextn != 0 {
            return Err(format!(
                "{} MTP/NextN blocks (nextn_predict_layers={nextn}) are not supported by this MVP",
                key("nextn_predict_layers")
            ));
        }

        let hidden_size = u64_meta(file, &key("embedding_length")).ok_or_else(|| format!("missing {}", key("embedding_length")))? as usize;
        let num_q_heads = u64_meta(file, &key("attention.head_count")).ok_or_else(|| format!("missing {}", key("attention.head_count")))? as usize;
        let num_kv_heads = u64_meta(file, &key("attention.head_count_kv")).unwrap_or(num_q_heads as u64) as usize;
        let head_dim = u64_meta(file, &key("attention.key_length")).ok_or_else(|| format!("missing {}", key("attention.key_length")))? as usize;
        let rotary_dim = u64_meta(file, &key("rope.dimension_count")).map(|n| n as usize).unwrap_or(head_dim);
        let rope_base = f32_meta(file, &key("rope.freq_base")).unwrap_or(10_000.0);
        let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let ffn_hidden_size =
            u64_meta(file, &key("feed_forward_length")).ok_or_else(|| format!("missing {}", key("feed_forward_length")))? as usize;
        let attn_cfg = LayerConfig { hidden_size, num_q_heads, num_kv_heads, head_dim, rotary_dim, ffn_hidden_size, rope_base, rmsnorm_eps };

        let d_state = u64_meta(file, &key("ssm.state_size")).ok_or_else(|| format!("missing {}", key("ssm.state_size")))? as usize;
        let d_inner = u64_meta(file, &key("ssm.inner_size")).ok_or_else(|| format!("missing {}", key("ssm.inner_size")))? as usize;
        let group_count = u64_meta(file, &key("ssm.group_count")).ok_or_else(|| format!("missing {}", key("ssm.group_count")))? as usize;
        let conv_kernel = u64_meta(file, &key("ssm.conv_kernel")).ok_or_else(|| format!("missing {}", key("ssm.conv_kernel")))? as usize;
        if d_state == 0 {
            return Err(format!("{} must be nonzero", key("ssm.state_size")));
        }
        let num_v_heads = u64_meta(file, &key("ssm.time_step_rank")).map(|n| n as usize).filter(|&n| n > 0).unwrap_or(d_inner / d_state);
        let gdn_cfg = crate::gated_deltanet::GatedDeltaNetConfig {
            hidden_size,
            num_k_heads: group_count,
            num_v_heads,
            head_dim: d_state,
            conv_kernel_size: conv_kernel,
            eps: rmsnorm_eps,
        };
        gdn_cfg.validate()?;

        let is_gdn = parse_hybrid_layer_kinds(file, architecture, block_count)?;

        let rmsnorm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_RMSNORM"), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope", "rope_kernel")?;
        let silu_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_SILU_AND_MUL"), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_GEMV"), "gemv", "gemv_kernel")?;
        let attn_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ATTENTION"), "attention", "attention_kernel")?;
        let add_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ELEMENTWISE"), "elementwise", "add_kernel")?;
        let mut gdn_fns = aot::load_kernel_module(
            &device,
            env!("COLDSTART_KERNEL_GATED_DELTANET"),
            "gated_deltanet",
            &["gdn_conv_kernel", "gdn_l2_norm_kernel", "gdn_gates_kernel", "gdn_delta_kernel", "gdn_gated_norm_kernel"],
        )?
        .into_iter();
        let gdn_conv_k = gdn_fns.next().ok_or("missing gdn_conv_kernel")?;
        let gdn_l2_norm_k = gdn_fns.next().ok_or("missing gdn_l2_norm_kernel")?;
        let gdn_gates_k = gdn_fns.next().ok_or("missing gdn_gates_kernel")?;
        let gdn_delta_k = gdn_fns.next().ok_or("missing gdn_delta_kernel")?;
        let gdn_gated_norm_k = gdn_fns.next().ok_or("missing gdn_gated_norm_kernel")?;

        let load_weight = |name: &str| -> Result<Weight, String> {
            let info = file.tensor_info(name).ok_or_else(|| format!("missing weight '{name}'"))?;
            let bytes = file.tensor_bytes(info)?;
            let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
            let data = device.htod_sync_copy(&host).map_err(|e| format!("upload weight '{name}' to device: {e}"))?;
            Ok(Weight { data, shape: info.shape.clone() })
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let attn_norm = load_weight(&format!("blk.{i}.attn_norm.weight"))?;
            let post_attn_norm = load_weight(&format!("blk.{i}.post_attention_norm.weight"))?;
            let ffn_gate = load_weight(&format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = load_weight(&format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = load_weight(&format!("blk.{i}.ffn_down.weight"))?;

            let layer = if is_gdn[i] {
                HybridLayerWeights::GatedDeltaNet(GatedDeltaNetLayerWeights {
                    attn_norm,
                    attn_qkv: load_weight(&format!("blk.{i}.attn_qkv.weight"))?,
                    attn_gate: load_weight(&format!("blk.{i}.attn_gate.weight"))?,
                    ssm_beta: load_weight(&format!("blk.{i}.ssm_beta.weight"))?,
                    ssm_alpha: load_weight(&format!("blk.{i}.ssm_alpha.weight"))?,
                    ssm_dt: load_weight(&format!("blk.{i}.ssm_dt.bias"))?,
                    ssm_a: load_weight(&format!("blk.{i}.ssm_a"))?,
                    ssm_conv1d: load_weight(&format!("blk.{i}.ssm_conv1d.weight"))?,
                    ssm_norm: load_weight(&format!("blk.{i}.ssm_norm.weight"))?,
                    ssm_out: load_weight(&format!("blk.{i}.ssm_out.weight"))?,
                    post_attn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                })
            } else {
                HybridLayerWeights::GatedAttention(GatedAttnLayerWeights {
                    attn_norm,
                    attn_q: load_weight(&format!("blk.{i}.attn_q.weight"))?,
                    attn_k: load_weight(&format!("blk.{i}.attn_k.weight"))?,
                    attn_v: load_weight(&format!("blk.{i}.attn_v.weight"))?,
                    attn_q_norm: load_weight(&format!("blk.{i}.attn_q_norm.weight"))?,
                    attn_k_norm: load_weight(&format!("blk.{i}.attn_k_norm.weight"))?,
                    attn_output: load_weight(&format!("blk.{i}.attn_output.weight"))?,
                    post_attn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                })
            };
            layers.push(layer);
        }

        let token_embd_info =
            file.tensor_info("token_embd.weight").ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd =
            dequant::dequantize(token_embd_info.ggml_type, token_embd_bytes, token_embd_info.element_count())?;

        let output_norm = load_weight("output_norm.weight")?;

        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
                let data =
                    device.htod_sync_copy(&host).map_err(|e| format!("upload weight 'output.weight' to device: {e}"))?;
                Weight { data, shape: info.shape.clone() }
            }
            None => {
                let data = device
                    .htod_sync_copy(&token_embd)
                    .map_err(|e| format!("upload weight 'token_embd.weight' to device: {e}"))?;
                Weight { data, shape: token_embd_info.shape.clone() }
            }
        };

        let tokenizer = Tokenizer::from_gguf(file)?;

        Ok(Model {
            device,
            rmsnorm_k,
            rope_k,
            silu_k,
            gemv_k,
            attn_k,
            add_k,
            cfg: attn_cfg.clone(),
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: Some(HybridModel { attn_cfg, gdn_cfg, layers, gdn_conv_k, gdn_l2_norm_k, gdn_gates_k, gdn_delta_k, gdn_gated_norm_k }),
            mla: None,
        })
    }

    /// Loads a DeepSeek-V2/V3 MLA model (MVP step 4). See [`parse_mla_config`] for
    /// the scope this supports (dense-only, no Q-LoRA, no YaRN, no MTP).
    /// `cfg`/`layers`/`expert_used_count` below are unused garbage (matching the
    /// `hybrid` path's own convention) -- `forward_prompt` branches on `self.mla`
    /// before touching them.
    fn load_mla(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        let (mla_cfg, block_count) = parse_mla_config(file)?;

        let rmsnorm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_RMSNORM"), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope", "rope_kernel")?;
        let silu_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_SILU_AND_MUL"), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_GEMV"), "gemv", "gemv_kernel")?;
        let attn_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ATTENTION"), "attention", "attention_kernel")?;
        let add_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ELEMENTWISE"), "elementwise", "add_kernel")?;
        let mla_attn_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_MLA_ATTENTION"), "mla_attention", "mla_attention_kernel")?;
        let rope_norm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope_norm", "rope_norm_kernel")?;

        let load_weight = |name: &str| -> Result<Weight, String> {
            let info = file.tensor_info(name).ok_or_else(|| format!("missing weight '{name}'"))?;
            let bytes = file.tensor_bytes(info)?;
            let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
            let data = device.htod_sync_copy(&host).map_err(|e| format!("upload weight '{name}' to device: {e}"))?;
            Ok(Weight { data, shape: info.shape.clone() })
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            layers.push(MlaLayerWeights {
                attn_norm: load_weight(&format!("blk.{i}.attn_norm.weight"))?,
                wq: load_weight(&format!("blk.{i}.attn_q.weight"))?,
                wkv_a_mqa: load_weight(&format!("blk.{i}.attn_kv_a_mqa.weight"))?,
                attn_kv_a_norm: load_weight(&format!("blk.{i}.attn_kv_a_norm.weight"))?,
                wk_b: load_weight(&format!("blk.{i}.attn_k_b.weight"))?,
                wv_b: load_weight(&format!("blk.{i}.attn_v_b.weight"))?,
                wo: load_weight(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: load_weight(&format!("blk.{i}.ffn_norm.weight"))?,
                ffn_gate: load_weight(&format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: load_weight(&format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: load_weight(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        let token_embd_info =
            file.tensor_info("token_embd.weight").ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd =
            dequant::dequantize(token_embd_info.ggml_type, token_embd_bytes, token_embd_info.element_count())?;

        let output_norm = load_weight("output_norm.weight")?;

        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let host = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
                let data =
                    device.htod_sync_copy(&host).map_err(|e| format!("upload weight 'output.weight' to device: {e}"))?;
                Weight { data, shape: info.shape.clone() }
            }
            None => {
                let data = device
                    .htod_sync_copy(&token_embd)
                    .map_err(|e| format!("upload weight 'token_embd.weight' to device: {e}"))?;
                Weight { data, shape: token_embd_info.shape.clone() }
            }
        };

        let tokenizer = Tokenizer::from_gguf(file)?;

        let dummy_cfg = LayerConfig {
            hidden_size: mla_cfg.hidden_size,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rotary_dim: 1,
            ffn_hidden_size: 1,
            rope_base: mla_cfg.rope_base,
            rmsnorm_eps: mla_cfg.rmsnorm_eps,
        };

        Ok(Model {
            device,
            rmsnorm_k,
            rope_k,
            silu_k,
            gemv_k,
            attn_k,
            add_k,
            cfg: dummy_cfg,
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: None,
            mla: Some(MlaModel { cfg: mla_cfg, layers, mla_attn_k, rope_norm_k }),
        })
    }

    /// `x` is already device-resident (Phase 2 round 2) -- unlike the
    /// pre-round-2 version, no `htod`/`dtoh` happens here; the caller chains
    /// this op's `CudaSlice` output straight into the next op.
    fn rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let n = x.len() as u32;
        let mut dev_out = self.device.alloc_zeros::<f32>(x.len()).map_err(|e| format!("rmsnorm alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.rmsnorm_k
                .function
                .clone()
                .launch(launch_cfg, (x, weight, &mut dev_out, rows as u32, hidden_size as u32, eps))
                .map_err(|e| format!("rmsnorm launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// `x` and `w_dev` are both already device-resident (Phase 2 round 2) --
    /// `w_dev` is either `&self.data` on a whole [`Weight`] (a
    /// `&CudaSlice<f32>`) or a zero-copy `CudaView` slice of one (see
    /// `Self::gemv_expert`).
    fn gemv_raw<W: DeviceRepr>(&self, x: &CudaSlice<f32>, w_dev: W, in_features: usize, out_features: usize) -> Result<CudaSlice<f32>, String> {
        if x.len() != in_features {
            return Err(format!("gemv: x.len()={} != in_features={in_features}", x.len()));
        }

        let mut dev_y = self.device.alloc_zeros::<f32>(out_features).map_err(|e| format!("gemv alloc y: {e}"))?;

        let threads = 256u32;
        let blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(launch_cfg, (x, w_dev, &mut dev_y, in_features as u32, out_features as u32))
                .map_err(|e| format!("gemv launch: {e}"))?;
        }
        Ok(dev_y)
    }

    fn gemv(&self, x: &CudaSlice<f32>, w: &Weight) -> Result<CudaSlice<f32>, String> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        self.gemv_raw(x, &w.data, in_features, out_features)
    }

    /// GEMV against expert `expert_idx`'s slice of a per-expert-stacked 3-D
    /// MoE tensor (shape `[in_features, out_features, expert_count]`).
    /// Expert `e`'s `in_features * out_features` elements are a contiguous
    /// chunk already in the same row-major `(out_features, in_features)`
    /// layout as a standalone 2-D weight (see this module's doc comment), so
    /// `CudaSlice::slice` gives a zero-copy device-side view -- no
    /// device-to-device copy, let alone a host round-trip.
    fn gemv_expert(&self, x: &CudaSlice<f32>, w: &Weight, expert_idx: usize) -> Result<CudaSlice<f32>, String> {
        let (in_features, out_features, expert_count) = match w.shape.as_slice() {
            [i, o, e] => (*i as usize, *o as usize, *e as usize),
            other => return Err(format!("gemv_expert: expected 3-D per-expert tensor shape, got {other:?}")),
        };
        if expert_idx >= expert_count {
            return Err(format!("gemv_expert: expert_idx {expert_idx} out of range (expert_count={expert_count})"));
        }
        let expert_len = in_features * out_features;
        let start = expert_idx * expert_len;
        let view = w.data.slice(start..start + expert_len);
        self.gemv_raw(x, &view, in_features, out_features)
    }

    /// In-place: `t` is already device-resident. `position` is a plain
    /// scalar kernel argument rather than an uploaded device array --
    /// `batch_size` is a permanent project constraint (CLAUDE.md's
    /// Non-goals), so there is never more than one token's position to pass,
    /// and the previous per-call device allocation+upload for it was pure
    /// overhead (Phase 2 round 2).
    fn rope(
        &self,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
    ) -> Result<(), String> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            self.rope_k
                .function
                .clone()
                .launch(launch_cfg, (t, position as u32, num_heads as u32, head_dim as u32, rotary_dim as u32, base))
                .map_err(|e| format!("rope launch: {e}"))?;
        }
        Ok(())
    }

    /// Like [`Self::rope`], but launches `m.rope_norm_k` (`rope_norm_kernel` --
    /// consecutive-pair rotation) instead of the shared `self.rope_k`
    /// (`rope_kernel` -- half-split rotation). Only DeepSeek-V2/V3 MLA needs this
    /// (see `MlaModel::rope_norm_k`'s doc comment); every other architecture uses
    /// `Self::rope` unchanged.
    fn rope_norm(&self, m: &MlaModel, t: &mut CudaSlice<f32>, num_heads: usize, head_dim: usize, rotary_dim: usize, position: usize, base: f32) -> Result<(), String> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            m.rope_norm_k
                .function
                .clone()
                .launch(launch_cfg, (t, position as u32, num_heads as u32, head_dim as u32, rotary_dim as u32, base))
                .map_err(|e| format!("rope_norm launch: {e}"))?;
        }
        Ok(())
    }

    /// `gate`/`up` are already device-resident, separate (not concatenated)
    /// buffers -- `silu_and_mul_kernel` takes them as two pointers, so no
    /// device-side concatenation step is needed either (Phase 2 round 2).
    fn silu_and_mul(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, hidden_size: usize) -> Result<CudaSlice<f32>, String> {
        let mut dev_out = self.device.alloc_zeros::<f32>(hidden_size).map_err(|e| format!("silu alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = (hidden_size as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.silu_k
                .function
                .clone()
                .launch(launch_cfg, (gate, up, &mut dev_out, 1u32, hidden_size as u32))
                .map_err(|e| format!("silu launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// Causal single-new-query attention against the full K/V cache so far
    /// (`k_cache`/`v_cache` views already include this position's own K/V --
    /// `seq_len = position + 1`). GQA-grouped: query head `h` reads KV head
    /// `h / (num_q_heads / num_kv_heads)`. `q`/`k_cache`/`v_cache` are all
    /// already device-resident (Phase 2 round 2) -- `k_cache`/`v_cache` are
    /// `CudaView`s into a preallocated per-layer device buffer, not a fresh
    /// upload of the whole cache history on every call.
    fn attention(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaView<f32>,
        v_cache: &CudaView<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dev_out =
            self.device.alloc_zeros::<f32>(num_q_heads * head_dim).map_err(|e| format!("attn alloc out: {e}"))?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            self.attn_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        k_cache,
                        v_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        seq_len as u32,
                        scale,
                    ),
                )
                .map_err(|e| format!("attn launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// In-place residual add: `a[i] += b[i]`, both already device-resident
    /// (Phase 2 round 2) -- replaces the host-side
    /// `a.iter().zip(b.iter()).map(|(&x,&y)| x+y)` loops every forward
    /// function used to do, which required both operands on the host.
    fn add_inplace(&self, a: &mut CudaSlice<f32>, b: &CudaSlice<f32>) -> Result<(), String> {
        let n = a.len() as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.add_k.function.clone().launch(launch_cfg, (a, b, n)).map_err(|e| format!("add launch: {e}"))?;
        }
        Ok(())
    }

    /// Like [`Self::gemv_raw`], but generic over both operands being any
    /// device-resident reference (`&CudaSlice<f32>` or `&CudaView<f32>`) instead of
    /// requiring `x` to be a whole owned `CudaSlice`. Needed for MLA's per-head
    /// absorption/decompression steps (see `Self::forward_mla_attn_block`,
    /// `Self::gemv_per_head`), where `x` is a strided per-head slice of a larger
    /// buffer. No length assertion (unlike `gemv_raw`) -- a generic view type isn't
    /// cheaply length-checked here, so correctness relies on the caller passing
    /// consistent `in_features`/`out_features`.
    fn gemv_view<X: DeviceRepr, W: DeviceRepr>(&self, x: X, w_dev: W, in_features: usize, out_features: usize) -> Result<CudaSlice<f32>, String> {
        let mut dev_y = self.device.alloc_zeros::<f32>(out_features).map_err(|e| format!("gemv_view alloc y: {e}"))?;
        let threads = 256u32;
        let blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(launch_cfg, (x, w_dev, &mut dev_y, in_features as u32, out_features as u32))
                .map_err(|e| format!("gemv_view launch: {e}"))?;
        }
        Ok(dev_y)
    }

    /// Applies a per-head-stacked weight tensor (`w`, shape `[in_features,
    /// out_features, n_head]` -- same layout convention as MoE's per-expert
    /// tensors, see `Self::gemv_expert`'s doc comment, just "expert" -> "head") to
    /// every head of a per-head-stacked input `x` (`[n_head, in_features]`
    /// row-major, contiguous per head), producing a per-head-stacked output
    /// (`[n_head, out_features]`). Unlike `gemv_expert` (which selects one of many
    /// experts per token), MLA's decompression step (`wv_b`, see
    /// `Self::forward_mla_attn_block`) always uses every head, so this loops over
    /// all of them, reusing the same `gemv_kernel` per head via zero-copy
    /// `CudaSlice::slice` views on both operands (via `Self::gemv_view`).
    fn gemv_per_head(&self, x: &CudaSlice<f32>, w: &Weight, n_head: usize) -> Result<CudaSlice<f32>, String> {
        let (in_features, out_features, head_count) = match w.shape.as_slice() {
            [i, o, h] => (*i as usize, *o as usize, *h as usize),
            other => return Err(format!("gemv_per_head: expected 3-D per-head tensor shape, got {other:?}")),
        };
        if head_count != n_head {
            return Err(format!("gemv_per_head: tensor's head dim {head_count} != n_head {n_head}"));
        }
        let mut out = self.device.alloc_zeros::<f32>(n_head * out_features).map_err(|e| format!("gemv_per_head alloc: {e}"))?;
        for h in 0..n_head {
            let w_view = w.data.slice(h * in_features * out_features..(h + 1) * in_features * out_features);
            let x_view = x.slice(h * in_features..(h + 1) * in_features);
            let y = self.gemv_view(&x_view, &w_view, in_features, out_features)?;
            let mut dst = out.slice_mut(h * out_features..(h + 1) * out_features);
            self.device.dtod_copy(&y, &mut dst).map_err(|e| format!("gemv_per_head dtod head {h}: {e}"))?;
        }
        Ok(out)
    }

    /// DeepSeek-V2/V3 MLA's MQA-style attention: `num_q_heads` query heads (each
    /// `qk_dim` wide) attend against a single shared compressed KV "head"
    /// (`kv_cache` view, `[seq_len, qk_dim]` row-major, one row per cached
    /// position -- the *same* row also serves as the value vector, using only its
    /// first `v_dim` elements, since MLA's whole point is that K and V share one
    /// compressed representation, unlike GQA's separate caches). Output is
    /// `[num_q_heads, v_dim]`, still in compressed latent space --
    /// `Self::gemv_per_head` (with `wv_b`) decompresses it afterward. `scale` must
    /// be `1/sqrt(qk_nope_head_dim + qk_rope_head_dim)` -- the *uncompressed*
    /// per-head dim, confirmed against llama.cpp's `deepseek2.cpp` (`kq_scale`) --
    /// **not** `1/sqrt(qk_dim)` (the compressed dot-product width), an easy
    /// mistake since every other op in this codebase scales by its own dot-product
    /// dimension.
    #[allow(clippy::too_many_arguments)]
    fn mla_attention(
        &self,
        m: &MlaModel,
        q: &CudaSlice<f32>,
        kv_cache: &CudaView<f32>,
        num_q_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dev_out = self.device.alloc_zeros::<f32>(num_q_heads * v_dim).map_err(|e| format!("mla_attn alloc out: {e}"))?;
        let block_dim = (qk_dim as u32).next_power_of_two();
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            m.mla_attn_k
                .function
                .clone()
                .launch(launch_cfg, (q, kv_cache, &mut dev_out, num_q_heads as u32, qk_dim as u32, v_dim as u32, seq_len as u32, scale))
                .map_err(|e| format!("mla_attn launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// RMSNorm -> QKV -> QK-Norm (if present) -> RoPE -> causal attention ->
    /// O-proj (residual), for one layer at `position`, writing this
    /// position's K/V directly into `k_cache`/`v_cache` (preallocated
    /// device buffers, see `Model::forward_prompt`). Shared byte-for-byte by
    /// dense and MoE layers -- MoE only replaces what comes after this (see
    /// `forward_layer_moe`). Takes ownership of `hidden` and mutates it
    /// in place for the final residual add, returning it back to the
    /// caller -- the whole block stays device-resident end to end (Phase 2
    /// round 2), no host round-trip.
    #[allow(clippy::too_many_arguments)]
    fn forward_attn_block(
        &self,
        attn_norm: &Weight,
        attn_q: &Weight,
        attn_k: &Weight,
        attn_v: &Weight,
        attn_output: &Weight,
        attn_q_norm: &Option<Weight>,
        attn_k_norm: &Option<Weight>,
        mut hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(&hidden, &attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let mut q = self.gemv(&normed, attn_q)?;
        let mut k = self.gemv(&normed, attn_k)?;
        let v = self.gemv(&normed, attn_v)?;

        if let Some(qn) = attn_q_norm {
            q = self.rmsnorm(&q, &qn.data, cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }
        if let Some(kn) = attn_k_norm {
            k = self.rmsnorm(&k, &kn.data, cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }

        self.rope(&mut q, cfg.num_q_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;
        self.rope(&mut k, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = position * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + kv_stride);
            self.device.dtod_copy(&k, &mut dst).map_err(|e| format!("attn kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + kv_stride);
            self.device.dtod_copy(&v, &mut dst).map_err(|e| format!("attn kv-cache dtod v: {e}"))?;
        }
        let seq_len = position + 1;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let attn_out = self.attention(&q, &k_view, &v_view, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, seq_len)?;
        let o_proj = self.gemv(&attn_out, attn_output)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    fn forward_layer_dense(
        &self,
        layer: &DenseLayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let mut post_attn = self.forward_attn_block(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            hidden,
            position,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;
        let gate = self.gemv(&ffn_normed, &layer.ffn_gate)?;
        let up = self.gemv(&ffn_normed, &layer.ffn_up)?;
        let activated = self.silu_and_mul(&gate, &up, cfg.ffn_hidden_size)?;
        let down = self.gemv(&activated, &layer.ffn_down)?;

        self.add_inplace(&mut post_attn, &down)?;
        Ok(post_attn)
    }

    /// Same attention block as [`Self::forward_layer_dense`], but the shared
    /// FFN is replaced by a router (softmax + top-k over `ffn_gate_inp`'s
    /// logits, `crate::moe::route_top_k`) dispatching to each selected
    /// expert's SwiGLU FFN (naive per-expert `gemv` calls, no batched/grouped
    /// GEMM -- see this module's MoE scope doc comment), weighted-summed by
    /// the router's renormalized combination weights. The router's top-k is
    /// an inherently host-side sort, and the per-expert weighted accumulate
    /// stays host-driven too (small expert count, already flagged as
    /// naive/unoptimized) -- not addressed by Phase 2 round 2's
    /// device-residency work, unlike everything else in this function.
    fn forward_layer_moe(
        &self,
        layer: &MoeLayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let mut post_attn = self.forward_attn_block(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            hidden,
            position,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let router_logits_dev = self.gemv(&ffn_normed, &layer.ffn_gate_inp)?;
        let router_logits = self.device.dtoh_sync_copy(&router_logits_dev).map_err(|e| format!("moe router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or("forward_layer_moe called on a model with no expert_used_count")?;
        let routed = route_top_k(&router_logits, k)?;

        let mut ffn_out = vec![0.0f32; cfg.hidden_size];
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, &layer.ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, &layer.ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, cfg.ffn_hidden_size)?;
            let down = self.gemv_expert(&activated, &layer.ffn_down_exps, expert_idx)?;
            let down_host = self.device.dtoh_sync_copy(&down).map_err(|e| format!("moe expert down dtoh: {e}"))?;
            for (o, d) in ffn_out.iter_mut().zip(down_host.iter()) {
                *o += weight * d;
            }
        }

        let ffn_out_dev = self.device.htod_sync_copy(&ffn_out).map_err(|e| format!("moe ffn_out htod: {e}"))?;
        self.add_inplace(&mut post_attn, &ffn_out_dev)?;
        Ok(post_attn)
    }

    fn forward_layer(
        &self,
        layer: &LayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        match layer {
            LayerWeights::Dense(l) => self.forward_layer_dense(l, hidden, position, k_cache, v_cache),
            LayerWeights::Moe(l) => self.forward_layer_moe(l, hidden, position, k_cache, v_cache),
        }
    }

    /// Encodes `prompt`, runs the full prompt through every layer one
    /// position at a time (real causal self-attention throughout, matching
    /// RustFeference's own documented scope choice for its minimal forward
    /// pass), and returns the argmax-sampled first generated token id plus
    /// its decoded text.
    pub fn forward_prompt(&self, prompt: &str) -> Result<(u32, String), String> {
        if let Some(h) = &self.hybrid {
            return self.forward_prompt_hybrid(h, prompt);
        }
        if let Some(m) = &self.mla {
            return self.forward_prompt_mla(m, prompt);
        }

        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_token_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        // Preallocated up front (Phase 2 round 2) since the full prompt's
        // token count is already known here -- `forward_attn_block` writes
        // each position's K/V directly into these device buffers via
        // device-to-device copy instead of the pre-round-2 pattern of
        // re-uploading the entire host-side cache history on every call.
        let kv_cache_len = ids.len() * self.cfg.num_kv_heads * self.cfg.head_dim;
        let mut k_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(kv_cache_len))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc k_cache: {e}"))?;
        let mut v_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(kv_cache_len))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc v_cache: {e}"))?;

        let hidden_size = self.cfg.hidden_size;
        let mut hidden_host = vec![0.0f32; hidden_size];
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for (position, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            hidden_host.copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
            let mut hidden = self.device.htod_sync_copy(&hidden_host).map_err(|e| format!("embedding htod: {e}"))?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer(layer, hidden, position, &mut k_caches[layer_idx], &mut v_caches[layer_idx])?;
            }
            hidden_dev = Some(hidden);
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, self.cfg.rmsnorm_eps)?;
        let logits_dev = self.gemv(&normed, &self.lm_head)?;
        let logits = self.device.dtoh_sync_copy(&logits_dev).map_err(|e| format!("logits dtoh: {e}"))?;

        let next_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or("cannot argmax an empty logits slice")?;

        let text = self.tokenizer.decode(&[next_id]);
        Ok((next_id, text))
    }

    /// Causal depthwise conv1d + SiLU over the fused qkv, advancing
    /// `conv_state` in place on-device (Phase 2 round 2 -- no
    /// upload/download per call, unlike the pre-round-2 convention referred
    /// to in `HybridLayerState`'s doc comment). Returns the post-SiLU
    /// `conv_dim` output. Ports `gdn_conv_kernel` (see
    /// `kernels_cuda/gated_deltanet.cu`).
    fn gdn_conv(&self, h: &HybridModel, qkv: &CudaSlice<f32>, conv1d: &CudaSlice<f32>, conv_state: &mut CudaSlice<f32>, conv_dim: usize) -> Result<CudaSlice<f32>, String> {
        let mut dev_out = self.device.alloc_zeros::<f32>(conv_dim).map_err(|e| format!("gdn_conv alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = (conv_dim as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_conv_k
                .function
                .clone()
                .launch(launch_cfg, (qkv, conv1d, conv_state, &mut dev_out, conv_dim as u32, h.gdn_cfg.conv_kernel_size as u32))
                .map_err(|e| format!("gdn_conv launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// In-place per-head L2-normalize `x[offset..offset + heads*head_dim]`
    /// (`x` is already device-resident -- see `Self::forward_gdn_mixer`,
    /// which calls this twice in a row on the same device buffer, once for
    /// the q heads and once for the k heads, without an intervening
    /// host round-trip). Ports `gdn_l2_norm_kernel`.
    fn gdn_l2_norm(&self, h: &HybridModel, dev_x: &mut CudaSlice<f32>, offset: usize, heads: usize, head_dim: usize, eps: f32, scale: f32) -> Result<(), String> {
        let launch_cfg =
            LaunchConfig { grid_dim: (heads as u32, 1, 1), block_dim: (h.gdn_cfg.norm_block_dim(), 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_l2_norm_k
                .function
                .clone()
                .launch(launch_cfg, (dev_x, offset as u32, head_dim as u32, eps, scale))
                .map_err(|e| format!("gdn_l2_norm launch: {e}"))?;
        }
        Ok(())
    }

    /// `beta = sigmoid(beta_raw)`, `decay = exp(softplus(alpha_raw + dt) *
    /// a)`. All inputs/outputs device-resident (Phase 2 round 2). Ports
    /// `gdn_gates_kernel`.
    fn gdn_gates(&self, h: &HybridModel, alpha_raw: &CudaSlice<f32>, beta_raw: &CudaSlice<f32>, dt: &CudaSlice<f32>, a: &CudaSlice<f32>, num_v_heads: usize) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
        let mut dev_decay = self.device.alloc_zeros::<f32>(num_v_heads).map_err(|e| format!("gdn_gates alloc decay: {e}"))?;
        let mut dev_beta = self.device.alloc_zeros::<f32>(num_v_heads).map_err(|e| format!("gdn_gates alloc beta: {e}"))?;

        let threads = 256u32;
        let blocks = (num_v_heads as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_gates_k
                .function
                .clone()
                .launch(launch_cfg, (alpha_raw, beta_raw, dt, a, &mut dev_decay, &mut dev_beta, num_v_heads as u32))
                .map_err(|e| format!("gdn_gates launch: {e}"))?;
        }
        Ok((dev_beta, dev_decay))
    }

    /// The delta rule, mutating `recurrent` (`S`, device-resident, see
    /// `HybridLayerState`) in place on-device and returning the `value_dim`
    /// mixer output (also device-resident, Phase 2 round 2 -- no
    /// upload/download per call, unlike the pre-round-2 convention).
    /// `qkv_normed` is the post-conv/SiLU/L2-norm fused buffer (q at offset
    /// 0, k at `key_dim`, v at `2*key_dim`). Ports `gdn_delta_kernel`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_delta(&self, h: &HybridModel, recurrent: &mut CudaSlice<f32>, qkv_normed: &CudaSlice<f32>, key_dim: usize, beta: &CudaSlice<f32>, decay: &CudaSlice<f32>) -> Result<CudaSlice<f32>, String> {
        let cfg = &h.gdn_cfg;
        let mut dev_o = self.device.alloc_zeros::<f32>(cfg.value_dim()).map_err(|e| format!("gdn_delta alloc o: {e}"))?;

        let launch_cfg = LaunchConfig { grid_dim: (cfg.num_v_heads as u32, 1, 1), block_dim: (cfg.head_dim as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_delta_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        recurrent,
                        qkv_normed,
                        0u32,
                        key_dim as u32,
                        (2 * key_dim) as u32,
                        beta,
                        decay,
                        &mut dev_o,
                        cfg.head_dim as u32,
                        cfg.num_k_heads as u32,
                        cfg.num_v_heads as u32,
                    ),
                )
                .map_err(|e| format!("gdn_delta launch: {e}"))?;
        }
        Ok(dev_o)
    }

    /// `y = RMSNorm(o, ssm_norm) * silu(z)`, all device-resident (Phase 2
    /// round 2). Ports `gdn_gated_norm_kernel`.
    fn gdn_gated_norm(&self, h: &HybridModel, o: &CudaSlice<f32>, z: &CudaSlice<f32>, norm_w: &CudaSlice<f32>, eps: f32) -> Result<CudaSlice<f32>, String> {
        let cfg = &h.gdn_cfg;
        let mut dev_y = self.device.alloc_zeros::<f32>(cfg.value_dim()).map_err(|e| format!("gdn_gated_norm alloc y: {e}"))?;

        let launch_cfg =
            LaunchConfig { grid_dim: (cfg.num_v_heads as u32, 1, 1), block_dim: (cfg.norm_block_dim(), 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_gated_norm_k
                .function
                .clone()
                .launch(launch_cfg, (o, z, norm_w, &mut dev_y, cfg.head_dim as u32, eps))
                .map_err(|e| format!("gdn_gated_norm launch: {e}"))?;
        }
        Ok(dev_y)
    }

    /// One token through a Gated DeltaNet mixer (see `reference/
    /// gated_deltanet_rustfeference.rs`'s `step` for the exact math this
    /// ports): RMSNorm(`attn_norm`) -> input projections (plain `gemv`) ->
    /// gates -> causal conv1d +
    /// SiLU -> per-head L2-norm (q scaled by `1/sqrt(head_dim)`, k not) ->
    /// delta rule -> gated RMSNorm -> output projection -> residual add
    /// (`x + out_proj`, matching `forward_gated_attn_mixer`'s convention).
    /// Mutates `conv_state`/`recurrent` in place; everything stays
    /// device-resident end to end (Phase 2 round 2), including the
    /// per-head L2-norm step, which used to be the only part of this
    /// function already avoiding a host round-trip.
    fn forward_gdn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedDeltaNetLayerWeights,
        mut x: CudaSlice<f32>,
        conv_state: &mut CudaSlice<f32>,
        recurrent: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &h.gdn_cfg;
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();

        let normed = self.rmsnorm(&x, &w.attn_norm.data, 1, h.attn_cfg.hidden_size, cfg.eps)?;
        let qkv = self.gemv(&normed, &w.attn_qkv)?;
        let z = self.gemv(&normed, &w.attn_gate)?;
        let beta_raw = self.gemv(&normed, &w.ssm_beta)?;
        let alpha_raw = self.gemv(&normed, &w.ssm_alpha)?;

        let (beta, decay) = self.gdn_gates(h, &alpha_raw, &beta_raw, &w.ssm_dt.data, &w.ssm_a.data, cfg.num_v_heads)?;
        let mut conv_out = self.gdn_conv(h, &qkv, &w.ssm_conv1d.data, conv_state, conv_dim)?;

        // Split q/k, L2-normalize both (q additionally scaled), v left raw,
        // in place on the same device buffer `gdn_conv` just produced.
        let q_scale = 1.0 / (cfg.head_dim as f32).sqrt();
        self.gdn_l2_norm(h, &mut conv_out, 0, cfg.num_k_heads, cfg.head_dim, cfg.eps, q_scale)?;
        self.gdn_l2_norm(h, &mut conv_out, key_dim, cfg.num_k_heads, cfg.head_dim, cfg.eps, 1.0)?;

        let o = self.gdn_delta(h, recurrent, &conv_out, key_dim, &beta, &decay)?;
        let y = self.gdn_gated_norm(h, &o, &z, &w.ssm_norm.data, cfg.eps)?;
        let out_proj = self.gemv(&y, &w.ssm_out)?;
        self.add_inplace(&mut x, &out_proj)?;
        Ok(x)
    }

    /// One token through a Gated Attention mixer (see `reference/
    /// gated_deltanet_rustfeference.rs`'s `gated_attention_step`): identical
    /// to the dense/MoE path's attention block, except `attn_q` is a fused
    /// query+gate projection (split per head into `[q(head_dim),
    /// gate(head_dim)]`) and the attention output is gated by
    /// `sigmoid(gate)` before the output projection. The head split and
    /// sigmoid gating stay host-side (small, same precedent as MoE's
    /// host-side router; Phase 2 round 2 only tackles the primary
    /// dense/MoE-benchmarked path's device residency, not this smaller,
    /// Gated-Attention-sublayer-only round trip) -- everything else in this
    /// function (RMSNorm/QKV/QK-Norm/RoPE/attention/O-proj/residual) is
    /// device-resident like `forward_attn_block`.
    fn forward_gated_attn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedAttnLayerWeights,
        mut hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &h.attn_cfg;
        let normed = self.rmsnorm(&hidden, &w.attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let qg = self.gemv(&normed, &w.attn_q)?;
        let qg_host = self.device.dtoh_sync_copy(&qg).map_err(|e| format!("gated-attn qg dtoh: {e}"))?;
        let q_dim = cfg.num_q_heads * cfg.head_dim;
        let mut q_host = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..cfg.num_q_heads {
            let base = head * 2 * cfg.head_dim;
            q_host[head * cfg.head_dim..(head + 1) * cfg.head_dim].copy_from_slice(&qg_host[base..base + cfg.head_dim]);
            gate[head * cfg.head_dim..(head + 1) * cfg.head_dim].copy_from_slice(&qg_host[base + cfg.head_dim..base + 2 * cfg.head_dim]);
        }
        let mut q = self.device.htod_sync_copy(&q_host).map_err(|e| format!("gated-attn q htod: {e}"))?;

        let mut k = self.gemv(&normed, &w.attn_k)?;
        let v = self.gemv(&normed, &w.attn_v)?;

        q = self.rmsnorm(&q, &w.attn_q_norm.data, cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        k = self.rmsnorm(&k, &w.attn_k_norm.data, cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;

        self.rope(&mut q, cfg.num_q_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;
        self.rope(&mut k, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = position * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + kv_stride);
            self.device.dtod_copy(&k, &mut dst).map_err(|e| format!("gated-attn kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + kv_stride);
            self.device.dtod_copy(&v, &mut dst).map_err(|e| format!("gated-attn kv-cache dtod v: {e}"))?;
        }
        let seq_len = position + 1;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let attn_out = self.attention(&q, &k_view, &v_view, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, seq_len)?;

        let mut attn_out_host = self.device.dtoh_sync_copy(&attn_out).map_err(|e| format!("gated-attn out dtoh: {e}"))?;
        for (a, &g) in attn_out_host.iter_mut().zip(gate.iter()) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }
        let attn_out = self.device.htod_sync_copy(&attn_out_host).map_err(|e| format!("gated-attn out htod: {e}"))?;

        let o_proj = self.gemv(&attn_out, &w.attn_output)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Shared post-mixer FFN tail for both hybrid layer kinds: RMSNorm
    /// (`post_attn_norm`) -> SwiGLU -> residual. Identical math to
    /// `forward_layer_dense`'s tail, kept as a separate small copy rather
    /// than sharing code with it -- the dense/MoE path's verified tensors
    /// are named `ffn_norm`, hybrid's is `post_attention_norm` (see
    /// `GatedAttnLayerWeights`'s doc comment), and touching the already
    /// hardware-verified dense/MoE path is not worth the risk for a few
    /// shared lines.
    fn forward_hybrid_ffn(&self, mut post_mixer: CudaSlice<f32>, norm: &Weight, ffn_gate: &Weight, ffn_up: &Weight, ffn_down: &Weight, hidden_size: usize, ffn_hidden_size: usize, eps: f32) -> Result<CudaSlice<f32>, String> {
        let normed = self.rmsnorm(&post_mixer, &norm.data, 1, hidden_size, eps)?;
        let gate = self.gemv(&normed, ffn_gate)?;
        let up = self.gemv(&normed, ffn_up)?;
        let activated = self.silu_and_mul(&gate, &up, ffn_hidden_size)?;
        let down = self.gemv(&activated, ffn_down)?;
        self.add_inplace(&mut post_mixer, &down)?;
        Ok(post_mixer)
    }

    /// One DeepSeek-V2/V3 MLA attention block (see `MlaConfig`'s doc comment for
    /// this MVP step's scope). Ported from llama.cpp's `src/models/deepseek2.cpp`
    /// `graph::graph()`, the `is_mla && is_lite` branch (read in full while
    /// planning this): RMSNorm -> `wq` (direct, no Q-LoRA) -> split into
    /// `q_nope`/`q_pe` per head -> `wkv_a_mqa` -> split into `kv_cmpr`/`k_pe` ->
    /// RoPE on `k_pe`/`q_pe` (full rotation over their own small buffers, not a
    /// slice of a wider head -- `Self::rope` applies unchanged) -> RMSNorm
    /// `kv_cmpr` -> **absorption** (`q_nope` per head times `wk_b`'s matching
    /// per-head slice, via `Self::gemv_view`) -> concat into `Qcur` per head
    /// (`kv_lora_rank + qk_rope_head_dim` wide) -> write this position's `Kcur`
    /// (`kv_cmpr_normed` concat `k_pe`, a single shared MQA "head") into the
    /// preallocated `kv_cache` via device-to-device copy (same convention Phase 2
    /// round 2 established for GQA's `k_cache`/`v_cache`) -> `Self::mla_attention`
    /// (MQA, compressed space) -> **decompression** (`Self::gemv_per_head` with
    /// `wv_b`) -> `wo` -> residual add. Takes ownership of `hidden` and mutates it
    /// in place for the residual add (same convention as `forward_attn_block`).
    fn forward_mla_attn_block(
        &self,
        m: &MlaModel,
        w: &MlaLayerWeights,
        mut hidden: CudaSlice<f32>,
        position: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &m.cfg;
        let n_head = cfg.num_heads;
        let qk_nope = cfg.qk_nope_head_dim;
        let qk_rope = cfg.qk_rope_head_dim;
        let n_embd_head_k_mla = qk_nope + qk_rope;
        let kv_lora = cfg.kv_lora_rank;
        let qk_dim = kv_lora + qk_rope;
        let v_dim = kv_lora;

        let normed = self.rmsnorm(&hidden, &w.attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        // q: [n_head, n_embd_head_k_mla] flat (plain gemv -- is_lite path, no Q-LoRA).
        let q = self.gemv(&normed, &w.wq)?;

        // kv_cmpr_pe: [kv_lora_rank + qk_rope_head_dim] flat (single shared "head").
        let kv_cmpr_pe = self.gemv(&normed, &w.wkv_a_mqa)?;

        let mut k_pe = self.device.alloc_zeros::<f32>(qk_rope).map_err(|e| format!("mla k_pe alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(kv_lora..kv_lora + qk_rope);
            self.device.dtod_copy(&src, &mut k_pe).map_err(|e| format!("mla k_pe dtod: {e}"))?;
        }
        self.rope_norm(m, &mut k_pe, 1, qk_rope, qk_rope, position, cfg.rope_base)?;

        let mut kv_cmpr_owned = self.device.alloc_zeros::<f32>(kv_lora).map_err(|e| format!("mla kv_cmpr alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(0..kv_lora);
            self.device.dtod_copy(&src, &mut kv_cmpr_owned).map_err(|e| format!("mla kv_cmpr dtod: {e}"))?;
        }
        let kv_cmpr_normed = self.rmsnorm(&kv_cmpr_owned, &w.attn_kv_a_norm.data, 1, kv_lora, cfg.rmsnorm_eps)?;

        // Gather q_pe (all heads) into its own contiguous [n_head, qk_rope_head_dim]
        // buffer before RoPE -- `Self::rope` expects one contiguous multi-head buffer,
        // and q_pe is a strided sub-slice of each head's [n_embd_head_k_mla]-wide row
        // in `q`, not itself contiguous across heads.
        let mut q_pe = self.device.alloc_zeros::<f32>(n_head * qk_rope).map_err(|e| format!("mla q_pe alloc: {e}"))?;
        for h in 0..n_head {
            let src = q.slice(h * n_embd_head_k_mla + qk_nope..h * n_embd_head_k_mla + n_embd_head_k_mla);
            let mut dst = q_pe.slice_mut(h * qk_rope..(h + 1) * qk_rope);
            self.device.dtod_copy(&src, &mut dst).map_err(|e| format!("mla q_pe dtod head {h}: {e}"))?;
        }
        self.rope_norm(m, &mut q_pe, n_head, qk_rope, qk_rope, position, cfg.rope_base)?;

        // Per head: absorb q_nope via wk_b, then concat with the (already-roped)
        // q_pe slice into Qcur's per-head [qk_dim]-wide row.
        let mut qcur = self.device.alloc_zeros::<f32>(n_head * qk_dim).map_err(|e| format!("mla qcur alloc: {e}"))?;
        for h in 0..n_head {
            let q_nope_view = q.slice(h * n_embd_head_k_mla..h * n_embd_head_k_mla + qk_nope);
            let wk_b_view = w.wk_b.data.slice(h * qk_nope * kv_lora..(h + 1) * qk_nope * kv_lora);
            let absorbed = self.gemv_view(&q_nope_view, &wk_b_view, qk_nope, kv_lora)?;

            let mut dst_nope = qcur.slice_mut(h * qk_dim..h * qk_dim + kv_lora);
            self.device.dtod_copy(&absorbed, &mut dst_nope).map_err(|e| format!("mla qcur absorbed dtod head {h}: {e}"))?;

            let pe_src = q_pe.slice(h * qk_rope..(h + 1) * qk_rope);
            let mut dst_pe = qcur.slice_mut(h * qk_dim + kv_lora..h * qk_dim + qk_dim);
            self.device.dtod_copy(&pe_src, &mut dst_pe).map_err(|e| format!("mla qcur pe dtod head {h}: {e}"))?;
        }

        // Write this position's compressed Kcur (== kv_cmpr_normed ++ k_pe) into the
        // preallocated per-layer cache -- device-resident from the start (Phase 2
        // round 2 convention), no host round-trip, ever, for this cache.
        let offset = position * qk_dim;
        {
            let mut dst = kv_cache.slice_mut(offset..offset + kv_lora);
            self.device.dtod_copy(&kv_cmpr_normed, &mut dst).map_err(|e| format!("mla kv_cache dtod cmpr: {e}"))?;
        }
        {
            let mut dst = kv_cache.slice_mut(offset + kv_lora..offset + qk_dim);
            self.device.dtod_copy(&k_pe, &mut dst).map_err(|e| format!("mla kv_cache dtod k_pe: {e}"))?;
        }
        let seq_len = position + 1;

        let kv_view = kv_cache.slice(0..seq_len * qk_dim);
        // Scale uses the *uncompressed* per-head dim (n_embd_head_k_mla), not
        // qk_dim -- see `Self::mla_attention`'s doc comment.
        let scale = 1.0 / (n_embd_head_k_mla as f32).sqrt();
        let compressed_out = self.mla_attention(m, &qcur, &kv_view, n_head, qk_dim, v_dim, seq_len, scale)?;

        let decompressed = self.gemv_per_head(&compressed_out, &w.wv_b, n_head)?;
        let o_proj = self.gemv(&decompressed, &w.wo)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Hybrid-model counterpart to [`Self::forward_prompt`]: same encode ->
    /// per-position, per-layer loop -> final norm -> LM head -> argmax
    /// shape, but each layer dispatches to [`Self::forward_gated_attn_mixer`]
    /// or [`Self::forward_gdn_mixer`] (never both) based on its
    /// [`HybridLayerWeights`] variant, carrying the matching
    /// [`HybridLayerState`] variant across positions.
    fn forward_prompt_hybrid(&self, h: &HybridModel, prompt: &str) -> Result<(u32, String), String> {
        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_token_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        // Preallocated up front (Phase 2 round 2), same rationale as
        // `forward_prompt`'s `k_caches`/`v_caches`: the full prompt's token
        // count is already known before the per-position loop starts.
        let attn_kv_cache_len = ids.len() * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
        let mut states: Vec<HybridLayerState> = h
            .layers
            .iter()
            .map(|l| -> Result<HybridLayerState, String> {
                match l {
                    HybridLayerWeights::GatedAttention(_) => {
                        let k_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| format!("alloc k_cache: {e}"))?;
                        let v_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| format!("alloc v_cache: {e}"))?;
                        Ok(HybridLayerState::Attn { k_cache, v_cache })
                    }
                    HybridLayerWeights::GatedDeltaNet(_) => {
                        let conv_state =
                            self.device.alloc_zeros::<f32>(h.gdn_cfg.conv_state_len()).map_err(|e| format!("alloc conv_state: {e}"))?;
                        let recurrent =
                            self.device.alloc_zeros::<f32>(h.gdn_cfg.recurrent_len()).map_err(|e| format!("alloc recurrent: {e}"))?;
                        Ok(HybridLayerState::Gdn { conv_state, recurrent })
                    }
                }
            })
            .collect::<Result<Vec<_>, String>>()?;

        let hidden_size = h.attn_cfg.hidden_size;
        let ffn_hidden_size = h.attn_cfg.ffn_hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let mut hidden_host = vec![0.0f32; hidden_size];
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for (position, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            hidden_host.copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
            let mut hidden = self.device.htod_sync_copy(&hidden_host).map_err(|e| format!("embedding htod: {e}"))?;

            for (layer, state) in h.layers.iter().zip(states.iter_mut()) {
                hidden = match (layer, state) {
                    (HybridLayerWeights::GatedAttention(w), HybridLayerState::Attn { k_cache, v_cache }) => {
                        let post_mixer = self.forward_gated_attn_mixer(h, w, hidden, position, k_cache, v_cache)?;
                        self.forward_hybrid_ffn(post_mixer, &w.post_attn_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down, hidden_size, ffn_hidden_size, eps)?
                    }
                    (HybridLayerWeights::GatedDeltaNet(w), HybridLayerState::Gdn { conv_state, recurrent }) => {
                        let post_mixer = self.forward_gdn_mixer(h, w, hidden, conv_state, recurrent)?;
                        self.forward_hybrid_ffn(post_mixer, &w.post_attn_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down, hidden_size, ffn_hidden_size, eps)?
                    }
                    _ => return Err("internal error: hybrid layer/state kind mismatch".to_string()),
                };
            }
            hidden_dev = Some(hidden);
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits_dev = self.gemv(&normed, &self.lm_head)?;
        let logits = self.device.dtoh_sync_copy(&logits_dev).map_err(|e| format!("logits dtoh: {e}"))?;

        let next_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or("cannot argmax an empty logits slice")?;

        let text = self.tokenizer.decode(&[next_id]);
        Ok((next_id, text))
    }

    /// MLA-model counterpart to [`Self::forward_prompt`]/[`Self::forward_prompt_hybrid`]:
    /// same encode -> per-position, per-layer loop -> final norm -> LM head ->
    /// argmax shape. Each layer runs [`Self::forward_mla_attn_block`] then the
    /// dense SwiGLU FFN tail (reuses [`Self::forward_hybrid_ffn`] unchanged -- it's
    /// already generic over which norm/gate/up/down weights it's given, not
    /// actually hybrid-specific). `kv_caches` are preallocated up front (same
    /// rationale as `forward_prompt`'s/`forward_prompt_hybrid`'s own caches: the
    /// full prompt's token count is already known before the per-position loop
    /// starts).
    fn forward_prompt_mla(&self, m: &MlaModel, prompt: &str) -> Result<(u32, String), String> {
        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_token_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        let cfg = &m.cfg;
        let qk_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let kv_cache_len = ids.len() * qk_dim;
        let mut kv_caches: Vec<CudaSlice<f32>> = (0..m.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(kv_cache_len))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc mla kv_cache: {e}"))?;

        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;
        let mut hidden_host = vec![0.0f32; hidden_size];
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for (position, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            hidden_host.copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
            let mut hidden = self.device.htod_sync_copy(&hidden_host).map_err(|e| format!("embedding htod: {e}"))?;

            for (layer_idx, layer) in m.layers.iter().enumerate() {
                let post_attn = self.forward_mla_attn_block(m, layer, hidden, position, &mut kv_caches[layer_idx])?;
                hidden = self.forward_hybrid_ffn(
                    post_attn,
                    &layer.ffn_norm,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                    hidden_size,
                    ffn_hidden_size,
                    eps,
                )?;
            }
            hidden_dev = Some(hidden);
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits_dev = self.gemv(&normed, &self.lm_head)?;
        let logits = self.device.dtoh_sync_copy(&logits_dev).map_err(|e| format!("logits dtoh: {e}"))?;

        let next_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or("cannot argmax an empty logits slice")?;

        let text = self.tokenizer.decode(&[next_id]);
        Ok((next_id, text))
    }
}
