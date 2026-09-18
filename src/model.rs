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
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
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

/// Per-sequence recurrent state for one hybrid layer, matching
/// [`HybridLayerWeights`]'s variant for that layer index one-to-one. Like
/// the dense/MoE path's `k_cache`/`v_cache`, this is plain host memory,
/// fully re-uploaded to the GPU on every kernel call that touches it (see
/// `Model::gdn_conv`/`Model::gdn_delta`) -- consistent with, not a
/// regression from, this MVP's existing per-call host<->device round-trip
/// convention for activations (only *weights* are GPU-resident, per
/// DECISIONS.md).
enum HybridLayerState {
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
    Gdn { conv_state: Vec<f32>, recurrent: Vec<f32> },
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

        let (cfg, block_count, moe) = parse_model_config(file)?;
        let expert_used_count = moe.map(|m| m.expert_used_count);

        let rmsnorm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_RMSNORM"), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope", "rope_kernel")?;
        let silu_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_SILU_AND_MUL"), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_GEMV"), "gemv", "gemv_kernel")?;
        let attn_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ATTENTION"), "attention", "attention_kernel")?;

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
            cfg,
            layers,
            expert_used_count,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: None,
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
            cfg: attn_cfg.clone(),
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: Some(HybridModel { attn_cfg, gdn_cfg, layers, gdn_conv_k, gdn_l2_norm_k, gdn_gates_k, gdn_delta_k, gdn_gated_norm_k }),
        })
    }

    fn rmsnorm(
        &self,
        x: &[f32],
        weight: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let dev_x = self.device.htod_sync_copy(x).map_err(|e| format!("rmsnorm htod x: {e}"))?;
        let mut dev_out = self.device.alloc_zeros::<f32>(x.len()).map_err(|e| format!("rmsnorm alloc out: {e}"))?;

        let n = x.len() as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.rmsnorm_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_x, weight, &mut dev_out, rows as u32, hidden_size as u32, eps))
                .map_err(|e| format!("rmsnorm launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_out).map_err(|e| format!("rmsnorm dtoh: {e}"))
    }

    /// `w_dev` is already device-resident -- either `&self.data` on a whole
    /// [`Weight`] (a `&CudaSlice<f32>`) or a zero-copy `CudaView` slice of
    /// one (see `Self::gemv_expert`) -- so, unlike `x`, it is never
    /// re-uploaded here.
    fn gemv_raw<W: DeviceRepr>(&self, x: &[f32], w_dev: W, in_features: usize, out_features: usize) -> Result<Vec<f32>, String> {
        if x.len() != in_features {
            return Err(format!("gemv: x.len()={} != in_features={in_features}", x.len()));
        }

        let dev_x = self.device.htod_sync_copy(x).map_err(|e| format!("gemv htod x: {e}"))?;
        let mut dev_y = self.device.alloc_zeros::<f32>(out_features).map_err(|e| format!("gemv alloc y: {e}"))?;

        let threads = 256u32;
        let blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_x, w_dev, &mut dev_y, in_features as u32, out_features as u32))
                .map_err(|e| format!("gemv launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_y).map_err(|e| format!("gemv dtoh: {e}"))
    }

    fn gemv(&self, x: &[f32], w: &Weight) -> Result<Vec<f32>, String> {
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
    fn gemv_expert(&self, x: &[f32], w: &Weight, expert_idx: usize) -> Result<Vec<f32>, String> {
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

    fn rope(
        &self,
        t: &mut Vec<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
    ) -> Result<(), String> {
        let positions = [position as i32];
        let mut dev_t = self.device.htod_sync_copy(t).map_err(|e| format!("rope htod t: {e}"))?;
        let dev_pos = self.device.htod_sync_copy(&positions).map_err(|e| format!("rope htod positions: {e}"))?;

        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            self.rope_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (&mut dev_t, &dev_pos, 1u32, num_heads as u32, head_dim as u32, rotary_dim as u32, base),
                )
                .map_err(|e| format!("rope launch: {e}"))?;
        }
        let result = self.device.dtoh_sync_copy(&dev_t).map_err(|e| format!("rope dtoh: {e}"))?;
        t.copy_from_slice(&result);
        Ok(())
    }

    fn silu_and_mul(&self, gate: &[f32], up: &[f32], hidden_size: usize) -> Result<Vec<f32>, String> {
        let mut gate_up = vec![0.0f32; 2 * hidden_size];
        gate_up[..hidden_size].copy_from_slice(gate);
        gate_up[hidden_size..].copy_from_slice(up);

        let dev_in = self.device.htod_sync_copy(&gate_up).map_err(|e| format!("silu htod: {e}"))?;
        let mut dev_out = self.device.alloc_zeros::<f32>(hidden_size).map_err(|e| format!("silu alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = (hidden_size as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.silu_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_in, &mut dev_out, 1u32, hidden_size as u32))
                .map_err(|e| format!("silu launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_out).map_err(|e| format!("silu dtoh: {e}"))
    }

    /// Causal single-new-query attention against the full K/V cache so far
    /// (`k_cache`/`v_cache` already include this position's own K/V --
    /// `seq_len = position + 1`). GQA-grouped: query head `h` reads KV head
    /// `h / (num_q_heads / num_kv_heads)`.
    fn attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        let dev_q = self.device.htod_sync_copy(q).map_err(|e| format!("attn htod q: {e}"))?;
        let dev_k = self.device.htod_sync_copy(k_cache).map_err(|e| format!("attn htod k: {e}"))?;
        let dev_v = self.device.htod_sync_copy(v_cache).map_err(|e| format!("attn htod v: {e}"))?;
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
                        &dev_q,
                        &dev_k,
                        &dev_v,
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
        self.device.dtoh_sync_copy(&dev_out).map_err(|e| format!("attn dtoh: {e}"))
    }

    /// RMSNorm -> QKV -> QK-Norm (if present) -> RoPE -> causal attention ->
    /// O-proj (residual), for one layer at `position`, appending this
    /// position's K/V onto `k_cache`/`v_cache`. Shared byte-for-byte by
    /// dense and MoE layers -- MoE only replaces what comes after this (see
    /// `forward_layer_moe`).
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
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(hidden, &attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

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

        k_cache.extend_from_slice(&k);
        v_cache.extend_from_slice(&v);
        let seq_len = position + 1;

        let attn_out = self.attention(&q, k_cache, v_cache, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, seq_len)?;
        let o_proj = self.gemv(&attn_out, attn_output)?;
        Ok(hidden.iter().zip(o_proj.iter()).map(|(&h, &o)| h + o).collect())
    }

    fn forward_layer_dense(
        &self,
        layer: &DenseLayerWeights,
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let post_attn = self.forward_attn_block(
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

        Ok(post_attn.iter().zip(down.iter()).map(|(&h, &d)| h + d).collect())
    }

    /// Same attention block as [`Self::forward_layer_dense`], but the shared
    /// FFN is replaced by a router (softmax + top-k over `ffn_gate_inp`'s
    /// logits, `crate::moe::route_top_k`) dispatching to each selected
    /// expert's SwiGLU FFN (naive per-expert `gemv` calls, no batched/grouped
    /// GEMM -- see this module's MoE scope doc comment), weighted-summed by
    /// the router's renormalized combination weights.
    fn forward_layer_moe(
        &self,
        layer: &MoeLayerWeights,
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let post_attn = self.forward_attn_block(
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

        let router_logits = self.gemv(&ffn_normed, &layer.ffn_gate_inp)?;
        let k = self.expert_used_count.ok_or("forward_layer_moe called on a model with no expert_used_count")?;
        let routed = route_top_k(&router_logits, k)?;

        let mut ffn_out = vec![0.0f32; cfg.hidden_size];
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, &layer.ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, &layer.ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, cfg.ffn_hidden_size)?;
            let down = self.gemv_expert(&activated, &layer.ffn_down_exps, expert_idx)?;
            for (o, d) in ffn_out.iter_mut().zip(down.iter()) {
                *o += weight * d;
            }
        }

        Ok(post_attn.iter().zip(ffn_out.iter()).map(|(&h, &f)| h + f).collect())
    }

    fn forward_layer(
        &self,
        layer: &LayerWeights,
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
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

        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.bos_token_id {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        let mut k_caches: Vec<Vec<f32>> = vec![Vec::new(); self.layers.len()];
        let mut v_caches: Vec<Vec<f32>> = vec![Vec::new(); self.layers.len()];

        let hidden_size = self.cfg.hidden_size;
        let mut hidden = vec![0.0f32; hidden_size];
        for (position, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            hidden.copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer(layer, &hidden, position, &mut k_caches[layer_idx], &mut v_caches[layer_idx])?;
            }
        }

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, self.cfg.rmsnorm_eps)?;
        let logits = self.gemv(&normed, &self.lm_head)?;

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
    /// `conv_state` in place (host-resident, fully re-uploaded/re-downloaded
    /// here every call -- same convention as `k_cache`/`v_cache`, see
    /// `HybridLayerState`'s doc comment). Returns the post-SiLU `conv_dim`
    /// output. Ports `gdn_conv_kernel` (see `kernels_cuda/gated_deltanet.cu`).
    fn gdn_conv(&self, h: &HybridModel, qkv: &[f32], conv1d: &CudaSlice<f32>, conv_state: &mut Vec<f32>, conv_dim: usize) -> Result<Vec<f32>, String> {
        let dev_qkv = self.device.htod_sync_copy(qkv).map_err(|e| format!("gdn_conv htod qkv: {e}"))?;
        let mut dev_state = self.device.htod_sync_copy(conv_state).map_err(|e| format!("gdn_conv htod state: {e}"))?;
        let mut dev_out = self.device.alloc_zeros::<f32>(conv_dim).map_err(|e| format!("gdn_conv alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = (conv_dim as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_conv_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_qkv, conv1d, &mut dev_state, &mut dev_out, conv_dim as u32, h.gdn_cfg.conv_kernel_size as u32))
                .map_err(|e| format!("gdn_conv launch: {e}"))?;
        }
        let updated_state = self.device.dtoh_sync_copy(&dev_state).map_err(|e| format!("gdn_conv dtoh state: {e}"))?;
        conv_state.copy_from_slice(&updated_state);
        self.device.dtoh_sync_copy(&dev_out).map_err(|e| format!("gdn_conv dtoh out: {e}"))
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
    /// a)`. Ports `gdn_gates_kernel`.
    fn gdn_gates(&self, h: &HybridModel, alpha_raw: &[f32], beta_raw: &[f32], dt: &CudaSlice<f32>, a: &CudaSlice<f32>, num_v_heads: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let dev_alpha = self.device.htod_sync_copy(alpha_raw).map_err(|e| format!("gdn_gates htod alpha: {e}"))?;
        let dev_beta_raw = self.device.htod_sync_copy(beta_raw).map_err(|e| format!("gdn_gates htod beta: {e}"))?;
        let mut dev_decay = self.device.alloc_zeros::<f32>(num_v_heads).map_err(|e| format!("gdn_gates alloc decay: {e}"))?;
        let mut dev_beta = self.device.alloc_zeros::<f32>(num_v_heads).map_err(|e| format!("gdn_gates alloc beta: {e}"))?;

        let threads = 256u32;
        let blocks = (num_v_heads as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_gates_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_alpha, &dev_beta_raw, dt, a, &mut dev_decay, &mut dev_beta, num_v_heads as u32))
                .map_err(|e| format!("gdn_gates launch: {e}"))?;
        }
        let decay = self.device.dtoh_sync_copy(&dev_decay).map_err(|e| format!("gdn_gates dtoh decay: {e}"))?;
        let beta = self.device.dtoh_sync_copy(&dev_beta).map_err(|e| format!("gdn_gates dtoh beta: {e}"))?;
        Ok((beta, decay))
    }

    /// The delta rule, mutating `recurrent` (`S`, host-resident, see
    /// `HybridLayerState`) in place and returning the `value_dim` mixer
    /// output. `qkv_normed` is the post-conv/SiLU/L2-norm fused buffer (q at
    /// offset 0, k at `key_dim`, v at `2*key_dim`). Ports `gdn_delta_kernel`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_delta(&self, h: &HybridModel, recurrent: &mut Vec<f32>, qkv_normed: &[f32], key_dim: usize, beta: &[f32], decay: &[f32]) -> Result<Vec<f32>, String> {
        let cfg = &h.gdn_cfg;
        let mut dev_s = self.device.htod_sync_copy(recurrent).map_err(|e| format!("gdn_delta htod S: {e}"))?;
        let dev_qkv = self.device.htod_sync_copy(qkv_normed).map_err(|e| format!("gdn_delta htod qkv: {e}"))?;
        let dev_beta = self.device.htod_sync_copy(beta).map_err(|e| format!("gdn_delta htod beta: {e}"))?;
        let dev_decay = self.device.htod_sync_copy(decay).map_err(|e| format!("gdn_delta htod decay: {e}"))?;
        let mut dev_o = self.device.alloc_zeros::<f32>(cfg.value_dim()).map_err(|e| format!("gdn_delta alloc o: {e}"))?;

        let launch_cfg = LaunchConfig { grid_dim: (cfg.num_v_heads as u32, 1, 1), block_dim: (cfg.head_dim as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_delta_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        &mut dev_s,
                        &dev_qkv,
                        0u32,
                        key_dim as u32,
                        (2 * key_dim) as u32,
                        &dev_beta,
                        &dev_decay,
                        &mut dev_o,
                        cfg.head_dim as u32,
                        cfg.num_k_heads as u32,
                        cfg.num_v_heads as u32,
                    ),
                )
                .map_err(|e| format!("gdn_delta launch: {e}"))?;
        }
        let updated_s = self.device.dtoh_sync_copy(&dev_s).map_err(|e| format!("gdn_delta dtoh S: {e}"))?;
        recurrent.copy_from_slice(&updated_s);
        self.device.dtoh_sync_copy(&dev_o).map_err(|e| format!("gdn_delta dtoh o: {e}"))
    }

    /// `y = RMSNorm(o, ssm_norm) * silu(z)`. Ports `gdn_gated_norm_kernel`.
    fn gdn_gated_norm(&self, h: &HybridModel, o: &[f32], z: &[f32], norm_w: &CudaSlice<f32>, eps: f32) -> Result<Vec<f32>, String> {
        let cfg = &h.gdn_cfg;
        let dev_o = self.device.htod_sync_copy(o).map_err(|e| format!("gdn_gated_norm htod o: {e}"))?;
        let dev_z = self.device.htod_sync_copy(z).map_err(|e| format!("gdn_gated_norm htod z: {e}"))?;
        let mut dev_y = self.device.alloc_zeros::<f32>(cfg.value_dim()).map_err(|e| format!("gdn_gated_norm alloc y: {e}"))?;

        let launch_cfg =
            LaunchConfig { grid_dim: (cfg.num_v_heads as u32, 1, 1), block_dim: (cfg.norm_block_dim(), 1, 1), shared_mem_bytes: 0 };
        unsafe {
            h.gdn_gated_norm_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_o, &dev_z, norm_w, &mut dev_y, cfg.head_dim as u32, eps))
                .map_err(|e| format!("gdn_gated_norm launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_y).map_err(|e| format!("gdn_gated_norm dtoh: {e}"))
    }

    /// One token through a Gated DeltaNet mixer (see `reference/
    /// gated_deltanet_rustfeference.rs`'s `step` for the exact math this
    /// ports): RMSNorm(`attn_norm`) -> input projections (plain `gemv`) ->
    /// gates -> causal conv1d +
    /// SiLU -> per-head L2-norm (q scaled by `1/sqrt(head_dim)`, k not) ->
    /// delta rule -> gated RMSNorm -> output projection -> residual add
    /// (`x + out_proj`, matching `forward_gated_attn_mixer`'s convention).
    /// Mutates `conv_state`/`recurrent` in place.
    fn forward_gdn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedDeltaNetLayerWeights,
        x: &[f32],
        conv_state: &mut Vec<f32>,
        recurrent: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &h.gdn_cfg;
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();

        let normed = self.rmsnorm(x, &w.attn_norm.data, 1, h.attn_cfg.hidden_size, cfg.eps)?;
        let qkv = self.gemv(&normed, &w.attn_qkv)?;
        let z = self.gemv(&normed, &w.attn_gate)?;
        let beta_raw = self.gemv(&normed, &w.ssm_beta)?;
        let alpha_raw = self.gemv(&normed, &w.ssm_alpha)?;

        let (beta, decay) = self.gdn_gates(h, &alpha_raw, &beta_raw, &w.ssm_dt.data, &w.ssm_a.data, cfg.num_v_heads)?;
        let conv_out = self.gdn_conv(h, &qkv, &w.ssm_conv1d.data, conv_state, conv_dim)?;

        // Split q/k, L2-normalize both (q additionally scaled), v left raw --
        // done as one device round trip covering both per-head norm calls.
        let mut dev_qkv = self.device.htod_sync_copy(&conv_out).map_err(|e| format!("gdn qk-norm htod: {e}"))?;
        let q_scale = 1.0 / (cfg.head_dim as f32).sqrt();
        self.gdn_l2_norm(h, &mut dev_qkv, 0, cfg.num_k_heads, cfg.head_dim, cfg.eps, q_scale)?;
        self.gdn_l2_norm(h, &mut dev_qkv, key_dim, cfg.num_k_heads, cfg.head_dim, cfg.eps, 1.0)?;
        let qkv_normed = self.device.dtoh_sync_copy(&dev_qkv).map_err(|e| format!("gdn qk-norm dtoh: {e}"))?;

        let o = self.gdn_delta(h, recurrent, &qkv_normed, key_dim, &beta, &decay)?;
        let y = self.gdn_gated_norm(h, &o, &z, &w.ssm_norm.data, cfg.eps)?;
        let out_proj = self.gemv(&y, &w.ssm_out)?;
        Ok(x.iter().zip(out_proj.iter()).map(|(&hh, &o)| hh + o).collect())
    }

    /// One token through a Gated Attention mixer (see `reference/
    /// gated_deltanet_rustfeference.rs`'s `gated_attention_step`): identical
    /// to the dense/MoE path's attention block, except `attn_q` is a fused
    /// query+gate projection (split per head into `[q(head_dim),
    /// gate(head_dim)]`) and the attention output is gated by
    /// `sigmoid(gate)` (host-side, cheap -- same precedent as MoE's
    /// host-side router) before the output projection.
    fn forward_gated_attn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedAttnLayerWeights,
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &h.attn_cfg;
        let normed = self.rmsnorm(hidden, &w.attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let qg = self.gemv(&normed, &w.attn_q)?;
        let q_dim = cfg.num_q_heads * cfg.head_dim;
        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..cfg.num_q_heads {
            let base = head * 2 * cfg.head_dim;
            q[head * cfg.head_dim..(head + 1) * cfg.head_dim].copy_from_slice(&qg[base..base + cfg.head_dim]);
            gate[head * cfg.head_dim..(head + 1) * cfg.head_dim].copy_from_slice(&qg[base + cfg.head_dim..base + 2 * cfg.head_dim]);
        }

        let mut k = self.gemv(&normed, &w.attn_k)?;
        let v = self.gemv(&normed, &w.attn_v)?;

        q = self.rmsnorm(&q, &w.attn_q_norm.data, cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        k = self.rmsnorm(&k, &w.attn_k_norm.data, cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;

        self.rope(&mut q, cfg.num_q_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;
        self.rope(&mut k, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim, position, cfg.rope_base)?;

        k_cache.extend_from_slice(&k);
        v_cache.extend_from_slice(&v);
        let seq_len = position + 1;

        let mut attn_out = self.attention(&q, k_cache, v_cache, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, seq_len)?;
        for (a, &g) in attn_out.iter_mut().zip(gate.iter()) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }

        let o_proj = self.gemv(&attn_out, &w.attn_output)?;
        Ok(hidden.iter().zip(o_proj.iter()).map(|(&hh, &o)| hh + o).collect())
    }

    /// Shared post-mixer FFN tail for both hybrid layer kinds: RMSNorm
    /// (`post_attn_norm`) -> SwiGLU -> residual. Identical math to
    /// `forward_layer_dense`'s tail, kept as a separate small copy rather
    /// than sharing code with it -- the dense/MoE path's verified tensors
    /// are named `ffn_norm`, hybrid's is `post_attention_norm` (see
    /// `GatedAttnLayerWeights`'s doc comment), and touching the already
    /// hardware-verified dense/MoE path is not worth the risk for a few
    /// shared lines.
    fn forward_hybrid_ffn(&self, post_mixer: &[f32], norm: &Weight, ffn_gate: &Weight, ffn_up: &Weight, ffn_down: &Weight, hidden_size: usize, ffn_hidden_size: usize, eps: f32) -> Result<Vec<f32>, String> {
        let normed = self.rmsnorm(post_mixer, &norm.data, 1, hidden_size, eps)?;
        let gate = self.gemv(&normed, ffn_gate)?;
        let up = self.gemv(&normed, ffn_up)?;
        let activated = self.silu_and_mul(&gate, &up, ffn_hidden_size)?;
        let down = self.gemv(&activated, ffn_down)?;
        Ok(post_mixer.iter().zip(down.iter()).map(|(&hh, &d)| hh + d).collect())
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

        let mut states: Vec<HybridLayerState> = h
            .layers
            .iter()
            .map(|l| match l {
                HybridLayerWeights::GatedAttention(_) => HybridLayerState::Attn { k_cache: Vec::new(), v_cache: Vec::new() },
                HybridLayerWeights::GatedDeltaNet(_) => {
                    HybridLayerState::Gdn { conv_state: vec![0.0; h.gdn_cfg.conv_state_len()], recurrent: vec![0.0; h.gdn_cfg.recurrent_len()] }
                }
            })
            .collect();

        let hidden_size = h.attn_cfg.hidden_size;
        let ffn_hidden_size = h.attn_cfg.ffn_hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let mut hidden = vec![0.0f32; hidden_size];
        for (position, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            hidden.copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);

            for (layer, state) in h.layers.iter().zip(states.iter_mut()) {
                hidden = match (layer, state) {
                    (HybridLayerWeights::GatedAttention(w), HybridLayerState::Attn { k_cache, v_cache }) => {
                        let post_mixer = self.forward_gated_attn_mixer(h, w, &hidden, position, k_cache, v_cache)?;
                        self.forward_hybrid_ffn(&post_mixer, &w.post_attn_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down, hidden_size, ffn_hidden_size, eps)?
                    }
                    (HybridLayerWeights::GatedDeltaNet(w), HybridLayerState::Gdn { conv_state, recurrent }) => {
                        let post_mixer = self.forward_gdn_mixer(h, w, &hidden, conv_state, recurrent)?;
                        self.forward_hybrid_ffn(&post_mixer, &w.post_attn_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down, hidden_size, ffn_hidden_size, eps)?
                    }
                    _ => return Err("internal error: hybrid layer/state kind mismatch".to_string()),
                };
            }
        }

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits = self.gemv(&normed, &self.lm_head)?;

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
