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
use crate::gguf::{GgmlType, GgufFile, GgufValue};
use crate::lora;
use crate::moe::{route_top_k, route_top_k_with_norm};
use crate::tokenizer::Tokenizer;
use cudarc::cublas::sys as cublas_sys;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
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

/// GGUF super-block sizes for the two block types dequantized on-device
/// (`src/kernels_cuda/dequant.cu`, Phase 2 round 3) -- must match
/// `dequant.rs`'s `QK_K` and the block-byte-size table
/// `gguf.rs::ggml_type_size_bytes` computes independently for the same types.
const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const Q6K_BLOCK_BYTES: usize = 210;

/// Dequantizes one tensor's raw quantized bytes straight to a device-resident
/// `f32` buffer. Q4_K/Q6_K (Phase 2 round 3: the block types this project's
/// local Q4_K_M fixtures exercise for the bulk of weight bytes) dequantize
/// on-device via `src/kernels_cuda/dequant.cu` -- no host `f32` copy is ever
/// materialized for these, closing the gap with llama.cpp's CUDA backend,
/// which never materializes one either (see README.md's Phase 2 round 3
/// writeup). Every other block type still falls back to the existing host
/// `dequant::dequantize` path (`src/dequant.rs`/`dequant_iq.rs`) -- correct
/// but not (yet) GPU-accelerated.
fn dequantize_tensor_to_device(
    device: &Arc<CudaDevice>,
    dequant_q4k_k: &AotKernel,
    dequant_q6k_k: &AotKernel,
    ggml_type: GgmlType,
    bytes: &[u8],
    element_count: u64,
) -> Result<CudaSlice<f32>, String> {
    match ggml_type {
        GgmlType::Q4K => dequantize_on_device(device, dequant_q4k_k, Q4K_BLOCK_BYTES, bytes, element_count),
        GgmlType::Q6K => dequantize_on_device(device, dequant_q6k_k, Q6K_BLOCK_BYTES, bytes, element_count),
        other => {
            let host = dequant::dequantize(other, bytes, element_count)?;
            device.htod_sync_copy(&host).map_err(|e| format!("upload weight to device: {e}"))
        }
    }
}

/// Uploads `bytes` (raw quantized block bytes, unmodified) to device memory
/// and launches `kernel` (`dequantize_q4k_kernel`/`dequantize_q6k_kernel`) to
/// unpack them into a fresh `f32` buffer, one CUDA thread per `QK_K`-element
/// block. Truncates to `element_count` if the last block is only partially
/// used (ggml's own invariant is that a quantized tensor's element count is
/// always a block-size multiple, so this is defensive, matching
/// `dequant::dequantize`'s own truncation).
fn dequantize_on_device(
    device: &Arc<CudaDevice>,
    kernel: &AotKernel,
    block_bytes: usize,
    bytes: &[u8],
    element_count: u64,
) -> Result<CudaSlice<f32>, String> {
    let num_blocks = bytes.len() / block_bytes;
    let raw = device.htod_sync_copy(bytes).map_err(|e| format!("upload raw quantized bytes: {e}"))?;
    let out_len = num_blocks * QK_K;
    let mut dev_out = device.alloc_zeros::<f32>(out_len).map_err(|e| format!("alloc dequant output: {e}"))?;

    let threads = 256u32;
    let blocks = (num_blocks as u32).div_ceil(threads).max(1);
    let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        kernel
            .function
            .clone()
            .launch(launch_cfg, (&raw, &mut dev_out, num_blocks as u32))
            .map_err(|e| format!("dequant kernel launch: {e}"))?;
    }

    if out_len as u64 == element_count {
        return Ok(dev_out);
    }
    let n = element_count as usize;
    let mut truncated = device.alloc_zeros::<f32>(n).map_err(|e| format!("alloc truncated dequant output: {e}"))?;
    let src = dev_out.slice(0..n);
    device.dtod_copy(&src, &mut truncated).map_err(|e| format!("truncate dequant output: {e}"))?;
    Ok(truncated)
}

/// Loads and dequantizes weight `name` straight to a device-resident `f32`
/// buffer -- shared by `Model::load`/`load_hybrid`/`load_mla`'s own
/// `load_weight` closures (see [`dequantize_tensor_to_device`] for the
/// on-device-vs-host dispatch).
fn load_weight_device(
    device: &Arc<CudaDevice>,
    dequant_q4k_k: &AotKernel,
    dequant_q6k_k: &AotKernel,
    file: &GgufFile,
    name: &str,
) -> Result<Weight, String> {
    let info = file.tensor_info(name).ok_or_else(|| format!("missing weight '{name}'"))?;
    let bytes = file.tensor_bytes(info)?;
    let data = dequantize_tensor_to_device(device, dequant_q4k_k, dequant_q6k_k, info.ggml_type, bytes, info.element_count())
        .map_err(|e| format!("load weight '{name}': {e}"))?;
    Ok(Weight { data, shape: info.shape.clone() })
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
/// metadata. Scope narrowed to real DeepSeek-V2/V3 files' actual shape (confirmed
/// against a real `DeepSeek-V2-Lite` GGUF's metadata while extending this from the
/// MVP-step-4 synthetic-fixture-only version): dense-lead + MoE-with-shared-expert
/// FFN, `is_lite`-style direct `wq` (no Q-LoRA), and YaRN RoPE scaling are all
/// supported now. Still hard-errors (matching existing `qwen35moe`/MTP rejection
/// precedent) on: Q-LoRA query decomposition (`attention.q_lora_rank` present and
/// nonzero -- no real small file needing this has been seen yet), MTP/NextN
/// blocks, and any RoPE scaling type other than `"none"`/`"yarn"`.
fn parse_mla_config(file: &GgufFile) -> Result<(MlaConfig, usize, usize), String> {
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

    if u64_meta(file, &key("attention.q_lora_rank")).filter(|&n| n > 0).is_some() {
        return Err(format!("{} (Q-LoRA query decomposition) is not supported by this MVP", key("attention.q_lora_rank")));
    }

    let hidden_size = u64_meta(file, &key("embedding_length")).ok_or_else(|| format!("missing {}", key("embedding_length")))? as usize;
    let num_heads = u64_meta(file, &key("attention.head_count")).ok_or_else(|| format!("missing {}", key("attention.head_count")))? as usize;
    let kv_lora_rank =
        u64_meta(file, &key("attention.kv_lora_rank")).ok_or_else(|| format!("missing {}", key("attention.kv_lora_rank")))? as usize;
    let n_embd_head_k_mla = u64_meta(file, &key("attention.key_length_mla"))
        .ok_or_else(|| format!("missing {} (a legacy pre-MLA-split deepseek2 GGUF -- unsplit attn_kv_b, no key_length_mla/value_length_mla metadata -- is not supported by this MVP; reconvert from the original checkpoint with a current convert_hf_to_gguf.py)", key("attention.key_length_mla")))? as usize;
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
    let rope_base = f32_meta(file, &key("rope.freq_base")).unwrap_or(10000.0);

    // At least one dense-lead layer is required by this MVP step (true of every
    // real DeepSeek-V2/V3 file seen, and of the synthetic all-dense test fixture) --
    // an all-MoE deepseek2 file (leading_dense == 0) is not supported.
    let ffn_gate_info = file
        .tensor_info("blk.0.ffn_gate.weight")
        .ok_or("missing blk.0.ffn_gate.weight tensor (an all-MoE deepseek2 file, leading_dense_block_count == 0, is not supported by this MVP)")?;
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

    let moe = if leading_dense < block_count {
        let expert_used_count = u64_meta(file, &key("expert_used_count"))
            .ok_or_else(|| format!("missing {} (expert_count > 0 implied by leading_dense_block_count < block_count)", key("expert_used_count")))?
            as usize;
        let ffn_gate_exps_info = file
            .tensor_info(&format!("blk.{leading_dense}.ffn_gate_exps.weight"))
            .ok_or_else(|| format!("missing blk.{leading_dense}.ffn_gate_exps.weight tensor"))?;
        let n_ff_exp = match ffn_gate_exps_info.shape.as_slice() {
            [_in_features, out_features, _expert_count] => *out_features as usize,
            other => return Err(format!("blk.{leading_dense}.ffn_gate_exps.weight has unexpected shape {other:?}")),
        };
        let routed_scaling_factor = f32_meta(file, &key("expert_weights_scale")).unwrap_or(1.0);
        // The converter only ever writes this key when the source model's
        // `norm_topk_prob` is truthy (see `MlaMoeConfig`'s doc comment) -- absence
        // means "don't renormalize", not "assume the usual true default".
        let normalize_top_k = matches!(file.metadata.get(&key("expert_weights_norm")), Some(GgufValue::Bool(true)));
        Some(MlaMoeConfig { expert_used_count, n_ff_exp, routed_scaling_factor, normalize_top_k })
    } else {
        None
    };

    let rope_scaling_type = file.metadata.get(&key("rope.scaling.type")).and_then(GgufValue::as_str);
    let yarn = match rope_scaling_type {
        None | Some("none") => None,
        Some("yarn") => {
            let factor = f32_meta(file, &key("rope.scaling.factor"))
                .ok_or_else(|| format!("missing {}", key("rope.scaling.factor")))?;
            let orig_ctx_len = u64_meta(file, &key("rope.scaling.original_context_length"))
                .ok_or_else(|| format!("missing {}", key("rope.scaling.original_context_length")))? as f32;
            // llama.cpp's own CLI-settable defaults (32.0/1.0), used when the GGUF
            // doesn't override them -- real DeepSeek-V2-Lite doesn't set these keys
            // either, relying on the same defaults.
            let beta_fast = f32_meta(file, &key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0);
            let beta_slow = f32_meta(file, &key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0);
            // Stored pre-multiplied by 0.1 by the converter; the loader undoes that
            // ([TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] in a real llama.cpp build's
            // `deepseek2.cpp` `load_arch_hparams`) before using it -- replicate that
            // exactly, since every downstream formula assumes the undone value.
            let yarn_log_mul_raw = f32_meta(file, &key("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0) / 0.1;

            let freq_scale = 1.0 / factor;
            let ext_factor = 1.0f32;
            let factor_ln = factor.ln();
            // `cparams.yarn_attn_factor` after `llama-context.cpp`'s DEEPSEEK2
            // special case (the ratio `get_mscale(factor,mscale)/get_mscale(factor,mscale_all_dims)`
            // cancels to 1.0 whenever `mscale == mscale_all_dims`, which that special
            // case forces) followed by its own `*= 1/(1+0.1*ln(factor))` cancellation.
            let attn_factor = 1.0 / (1.0 + 0.1 * factor_ln);
            // deepseek2.cpp's own "cancel the adjustment to get the original
            // attn_factor" step -- reconstructs ~1.0 by construction, but computed
            // explicitly (not hardcoded) to mirror the reference exactly.
            let attn_factor_org = attn_factor * (1.0 + 0.1 * factor_ln);
            let mscale_kq = attn_factor_org * (1.0 + 0.1 * yarn_log_mul_raw * factor_ln);
            let attention_scale = mscale_kq * mscale_kq / (n_embd_head_k_mla as f32).sqrt();

            // ggml_rope_yarn_corr_dims: start/end correction dims over the rotated
            // width (qk_rope_head_dim), from beta_fast/beta_slow.
            let corr_dim = |n_rot: f32| qk_rope_head_dim as f32 * (orig_ctx_len / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * rope_base.ln());
            let corr_dim_start = corr_dim(beta_fast).floor().max(0.0);
            let corr_dim_end = corr_dim(beta_slow).ceil().min(qk_rope_head_dim as f32 - 1.0);

            Some(MlaYarnConfig { freq_scale, ext_factor, attn_factor, corr_dim_start, corr_dim_end, attention_scale })
        }
        Some(other) => return Err(format!("{} = {other:?} (only \"none\"/\"yarn\" RoPE scaling is supported by this MVP)", key("rope.scaling.type"))),
    };

    let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

    Ok((
        MlaConfig { hidden_size, num_heads, qk_rope_head_dim, qk_nope_head_dim, kv_lora_rank, ffn_hidden_size, rope_base, rmsnorm_eps, moe, yarn },
        block_count,
        leading_dense,
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
    /// FFN hidden size of the dense-lead layers (`leading_dense_block_count`
    /// layers at the start of the model, always at least 1 for every real
    /// DeepSeek-V2/V3 file this MVP step targets). Layers past that use `moe`'s
    /// `n_ff_exp` instead (see `MlaFfn::Moe`).
    ffn_hidden_size: usize,
    rope_base: f32,
    rmsnorm_eps: f32,
    /// `Some` iff this file has MoE layers past its dense-lead layers (a real
    /// DeepSeek-V2/V3 file always does; the synthetic MVP-step-4 test fixture is
    /// dense-only, so `None` there).
    moe: Option<MlaMoeConfig>,
    /// `Some` iff `deepseek2.rope.scaling.type == "yarn"` (every real DeepSeek-V2/V3
    /// file this MVP step has seen; the synthetic test fixture sets no rope scaling
    /// at all, so `None` there).
    yarn: Option<MlaYarnConfig>,
}

/// MoE routing config for MLA layers past `leading_dense_block_count`. Mirrors
/// [`MoeMetaConfig`]'s role for the dense/GQA path, plus the two things real
/// DeepSeek-V2/V3 adds beyond Qwen3-MoE's convention (see
/// `Model::forward_mla_moe_ffn`): an always-on shared expert (not gated by the
/// router) and `normalize_top_k = false` (confirmed against a real
/// `DeepSeek-V2-Lite` GGUF: `expert_weights_norm` is absent, and the converter only
/// ever writes that key when the source `norm_topk_prob` is truthy -- so its
/// absence here means "don't renormalize", not "key missing, assume default true"
/// the way most other optional keys in this codebase work).
struct MlaMoeConfig {
    expert_used_count: usize,
    /// Routed-expert FFN hidden size (`blk.{first_moe_layer}.ffn_gate_exps.weight`'s
    /// shape) -- distinct from `MlaConfig::ffn_hidden_size` (the dense-lead layers').
    n_ff_exp: usize,
    /// `expert_weights_scale` metadata (default `1.0`, a no-op) -- see
    /// `crate::moe`'s doc comment for the identical Qwen3-MoE convention.
    routed_scaling_factor: f32,
    /// `expert_weights_norm` metadata, default `false` if absent (see this
    /// struct's own doc comment for why the default differs from most other
    /// optional keys in this codebase).
    normalize_top_k: bool,
}

/// Precomputed YaRN RoPE-scaling parameters for MLA's `q_pe`/`k_pe` rotation
/// (`Model::rope_norm_yarn`) and attention softmax scale
/// (`Model::forward_mla_attn_block`). Derived once at load time
/// (`parse_mla_config`) from `deepseek2.rope.scaling.*` metadata, mirroring
/// `llama-context.cpp`'s YaRN setup and `deepseek2.cpp`'s own `kq_scale`
/// computation (both read in full while implementing this -- see DECISIONS.md).
struct MlaYarnConfig {
    /// `1 / rope.scaling.factor`.
    freq_scale: f32,
    /// Always `1.0` when YaRN is active (matches llama.cpp's own default when no
    /// CLI override is given) -- this codebase has no CLI, so always `1.0` here.
    ext_factor: f32,
    /// The (already-`DEEPSEEK2`-special-cased) `attn_factor` fed into the rotation
    /// itself -- **not** the same value as `deepseek2.cpp`'s own `kq_scale`
    /// computation, which independently reconstructs and further adjusts it (see
    /// `attention_scale` below).
    attn_factor: f32,
    corr_dim_start: f32,
    corr_dim_end: f32,
    /// Precomputed final attention softmax scale, replacing the non-YaRN
    /// `1/sqrt(qk_nope_head_dim+qk_rope_head_dim)` -- see `Model::mla_attention`'s
    /// doc comment for why this dimension (not the compressed one) is scaled, and
    /// `parse_mla_config` for the YaRN-specific `mscale^2/sqrt(...)` derivation.
    attention_scale: f32,
}

/// One MLA layer's FFN: dense SwiGLU for the `leading_dense_block_count` lead
/// layers (identical in shape/meaning to `DenseLayerWeights`'s), or routed MoE +
/// an always-on shared expert for every layer past that (real DeepSeek-V2/V3
/// files always have both kinds; the synthetic MVP-step-4 fixture is
/// `Dense`-only). See `Model::forward_mla_moe_ffn` for the shared-expert math --
/// its weights (`ffn_{gate,up,down}_shexp`) are a *single* fused dense FFN over
/// `n_ff_exp * expert_shared_count` hidden units (every shared expert's weights
/// concatenated into one bigger matmul), not `expert_shared_count` separate
/// per-expert calls -- confirmed against `deepseek2.cpp`'s own tensor shapes.
enum MlaFfn {
    Dense { ffn_gate: Weight, ffn_up: Weight, ffn_down: Weight },
    Moe {
        ffn_gate_inp: Weight,
        ffn_gate_exps: Weight,
        ffn_up_exps: Weight,
        ffn_down_exps: Weight,
        ffn_gate_shexp: Weight,
        ffn_up_shexp: Weight,
        ffn_down_shexp: Weight,
    },
}

/// One MLA layer's weights. Tensor names/shapes confirmed against a real
/// `llama.cpp` build's `src/models/deepseek2.cpp` (`is_mla && is_lite` branch) and a
/// synthetic `deepseek2`-architecture GGUF fixture built for this MVP step (see
/// README.md -- no small real `deepseek2` GGUF exists publicly). `wk_b`/`wv_b` are
/// per-head-stacked tensors (`[in_features, out_features, n_head]`, the same
/// layout convention as MoE's per-expert tensors -- see `Model::gemv_expert`'s doc
/// comment -- just "expert" -> "head"; every head is always used here, unlike MoE's
/// top-k selection). `ffn` is dense SwiGLU for the lead layers or routed-MoE +
/// shared-expert for the rest -- see [`MlaFfn`].
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
    ffn: MlaFfn,
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
    /// `rope_norm_yarn_kernel` -- used instead of `rope_norm_k` whenever
    /// `cfg.yarn.is_some()` (see `Model::forward_mla_attn_block`). Always loaded
    /// (even for the synthetic, YaRN-free MVP-step-4 fixture) since the tiny
    /// extra load cost isn't worth an `Option`.
    rope_norm_yarn_k: AotKernel,
    /// Batched-prefill variant of `mla_attn_k` (`mla_attention_prefill_kernel`,
    /// `kernels_cuda/mla_attention_prefill.cu`) -- see
    /// `Model::forward_mla_attn_block_batched`.
    mla_attn_prefill_k: AotKernel,
    /// Batched-prefill variant of `rope_norm_k` (`rope_norm_batch_kernel`,
    /// `kernels_cuda/rope.cu`).
    rope_norm_batch_k: AotKernel,
    /// Batched-prefill variant of `rope_norm_yarn_k` (`rope_norm_yarn_batch_kernel`,
    /// `kernels_cuda/rope.cu`).
    rope_norm_yarn_batch_k: AotKernel,
    /// Batched-prefill variant of `Model::gemv_per_head`'s per-head-loop-of-`gemv_k`
    /// (`gemv_per_head_batch_kernel`, `kernels_cuda/gemv_per_head_batch.cu`) --
    /// applies MLA's per-head-stacked `wk_b`/`wv_b` weights to every head of every
    /// batched row in one launch. See `Model::gemv_per_head_batch`.
    gemv_per_head_batch_k: AotKernel,
    /// Extracts a per-head sub-slice out of a wider batched per-head buffer in one
    /// launch (`mla_extract_batch_kernel`, `kernels_cuda/elementwise.cu`) -- used for
    /// q_pe/k_pe/kv_cmpr extraction ahead of RoPE/RMSNorm in
    /// `Model::forward_mla_attn_block_batched`.
    mla_extract_batch_k: AotKernel,
    /// Merges absorbed q_nope and RoPE'd q_pe into Qcur's per-head row in one launch
    /// (`mla_concat_qcur_batch_kernel`, `kernels_cuda/elementwise.cu`).
    mla_concat_qcur_batch_k: AotKernel,
    /// Writes a batch's compressed Kcur into the preallocated `kv_cache` in one
    /// launch (`mla_write_kv_cache_batch_kernel`, `kernels_cuda/elementwise.cu`).
    mla_write_kv_cache_batch_k: AotKernel,
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
    /// Handle for the batched-prefill GEMM projections (`Self::gemm`) --
    /// created once at load time with math mode pinned to
    /// `CUBLAS_PEDANTIC_MATH` (see `Self::load`'s construction site) so
    /// cuBLAS's summation order can be trusted not to silently drift from
    /// `gemv_kernel`'s naive per-row dot product via a TF32/reduced-precision
    /// tensor-core path. Only prefill (`rows > 1`) uses this; the per-token
    /// decode loop still uses `gemv_k` (a GEMM with n=1 buys nothing).
    cublas: CudaBlas,
    rmsnorm_k: AotKernel,
    rope_k: AotKernel,
    /// Batched-prefill variant of `rope_k` (`rope_batch_kernel`,
    /// `kernels_cuda/rope.cu`) -- rotates all of a prefill batch's rows in
    /// one launch, each at its own absolute position, instead of one
    /// `rope_k` launch per row.
    rope_batch_k: AotKernel,
    silu_k: AotKernel,
    gemv_k: AotKernel,
    /// Gathers only caller-chosen output rows of a GEMV instead of every row
    /// -- System1's candidate-subset LM-head scoring (see
    /// `Self::gemv_gather`/`Self::system1_evaluate`), never used by the
    /// ordinary dense/MoE/hybrid/MLA forward paths.
    gemv_gather_k: AotKernel,
    /// Grouped-GEMM MoE batching (`Self::forward_layer_moe_batched`,
    /// `Self::forward_mla_moe_ffn_batched`): gathers one expert's assigned rows out of
    /// a batched-prefill hidden buffer into a contiguous group before running that
    /// group through the expert's weights as one real GEMM (`moe_gather_kernel`,
    /// `kernels_cuda/elementwise.cu`).
    moe_gather_k: AotKernel,
    /// Inverse of `moe_gather_k`: weighted scatter-add of one expert group's
    /// down-projected output back into each selected row's output slot
    /// (`moe_scatter_add_kernel`, `kernels_cuda/elementwise.cu`).
    moe_scatter_add_k: AotKernel,
    attn_k: AotKernel,
    /// Batched-prefill variant of `attn_k` (`attention_prefill_kernel`,
    /// `kernels_cuda/attention_prefill.cu`) -- scores every row of a prefill
    /// batch in one launch (grid gains a query-row dimension), each row
    /// causally masked to its own position, instead of one `attn_k` launch
    /// per row.
    attn_prefill_k: AotKernel,
    /// In-place residual add (`a[i] += b[i]`, see `kernels_cuda/elementwise.cu`)
    /// -- keeps residual-stream adds device-resident (Phase 2 round 2)
    /// instead of downloading both operands to host just to add two vectors.
    add_k: AotKernel,
    /// Splits Qwen3.5 hybrid Gated Attention's fused query+gate projection
    /// output into separate q/gate buffers (`split_qg_kernel`, same
    /// `kernels_cuda/elementwise.cu` module as `add_k`) -- used by
    /// `Model::forward_gated_attn_mixer`/`forward_gated_attn_mixer_batched`
    /// instead of a per-call host round trip.
    split_qg_k: AotKernel,
    /// In-place `out[i] *= sigmoid(gate[i])` (`sigmoid_gate_kernel`) -- Gated
    /// Attention's post-attention gating, row-count-agnostic like `add_k` so
    /// the same kernel serves both the decode step and batched prefill.
    sigmoid_gate_k: AotKernel,
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

/// Which forward path a loaded [`Model`] dispatches to -- used by callers
/// (currently `qwen3_coldstart`'s `--export-kv`/`--import-kv` handling) that
/// need to pick an architecture-specific KV-cache capture/resume function
/// without reaching into `Model`'s private `hybrid`/`mla` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureKind {
    Dense,
    Hybrid,
    Mla,
}

/// One candidate continuation to score against a shared prompt, for
/// [`Model::system1_evaluate`]. `text` is tokenized as `prompt + text` and
/// diffed against `encode(prompt)` -- never tokenized standalone (see
/// `Model::resolve_candidate_token_ids`).
#[derive(Debug, Clone)]
pub struct System1Candidate {
    pub text: String,
}

/// One candidate's scored result from [`Model::system1_evaluate`]. `score`
/// is the sum of each resolved token's raw gathered lm_head logit -- the
/// first token's from the shared prefill hidden state, any subsequent ones
/// (multi-token candidates only) from a teacher-forced continuation feeding
/// the KNOWN candidate token, never sampled. `score` is NOT a
/// vocab-normalized log-probability -- it is only meaningful relative to
/// other candidates in the SAME `system1_evaluate` call; see
/// [`System1Response::probabilities`] for a calibrated distribution over
/// just this candidate set.
#[derive(Debug, Clone)]
pub struct System1CandidateResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub score: f32,
}

/// Result of [`Model::system1_evaluate`].
#[derive(Debug, Clone)]
pub struct System1Response {
    /// Same order as the `candidates` slice passed to `system1_evaluate`.
    pub results: Vec<System1CandidateResult>,
    /// `crate::calibration::softmax_scores_with_temperature` over `results[i].score`.
    pub probabilities: Vec<f32>,
    /// `crate::calibration::shannon_entropy` of `probabilities` (bits) -- `0.0` when
    /// one candidate completely dominates, `log2(probabilities.len())` when every
    /// candidate is equally likely. A single scalar confidence/escalation signal
    /// alongside the raw distribution.
    pub entropy: f32,
}

impl Model {
    pub fn architecture_kind(&self) -> ArchitectureKind {
        if self.hybrid.is_some() {
            ArchitectureKind::Hybrid
        } else if self.mla.is_some() {
            ArchitectureKind::Mla
        } else {
            ArchitectureKind::Dense
        }
    }

    /// Number of tokens `prompt` encodes to with this model's tokenizer
    /// (BOS not included) -- a narrow, derived-value accessor for callers
    /// like `bench_coldstart` that need to report actual prompt length,
    /// without exposing the private `tokenizer` field itself.
    pub fn encoded_prompt_len(&self, prompt: &str) -> Result<usize, String> {
        Ok(self.tokenizer.encode(prompt)?.len())
    }

    /// Decodes each id in `ids` to its own individual text piece (unlike
    /// `Model::generate`'s combined whole-sequence `text`, which merges every
    /// generated id's bytes into one string) -- for callers like
    /// `check_correctness` that want to display/compare each generated token
    /// separately, without exposing the private `tokenizer` field itself.
    pub fn decode_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter().map(|&id| self.tokenizer.decode(&[id])).collect()
    }

    /// Applies a llama.cpp-format LoRA adapter GGUF (see `crate::lora`'s
    /// module doc comment for the file format and the `W' = W + scale * (B @
    /// A)` math) to this already-loaded model's GPU-resident weights, once,
    /// in place -- `crate::lora::load` does the host-side `B @ A` math and
    /// hands back a full-size delta per targeted tensor; this method only
    /// uploads each delta once and adds it in with the existing in-place-add
    /// kernel (`add_k`, unchanged since Phase 2 round 2). No new kernel, and
    /// the forward pass itself is completely unmodified afterward -- exactly
    /// the "load-time adapter application, no runtime hot-swap multiplexer"
    /// scope README.md's Non-goals section commits this feature to.
    ///
    /// Deliberately narrow for this round (see `Self::find_lora_target_mut`'s
    /// doc comment for the exact accepted tensor set): dense/MoE attention
    /// tensors and Qwen3.5 hybrid Gated-Attention-layer tensors only.
    /// DeepSeek-V2/V3 MLA is rejected outright below, matching every other
    /// MLA-excluded feature in this codebase. Any adapter tensor that
    /// doesn't resolve to a supported base weight, or whose shape doesn't
    /// match that weight's, is a hard error -- never a silent skip.
    pub fn apply_lora(&mut self, path: &std::path::Path) -> Result<usize, String> {
        if self.mla.is_some() {
            return Err(
                "--lora is not supported for DeepSeek-V2/V3 MLA models in this round -- only dense/MoE Qwen3 \
                 and the Qwen3.5 hybrid architecture are supported LoRA base models"
                    .to_string(),
            );
        }

        let adapter = lora::load(path)?;
        // Cloned out before the loop's per-target `&mut self` borrow (via
        // `find_lora_target_mut`) starts, so the in-place-add launch below
        // doesn't need to re-borrow `self` (which `Self::add_inplace`, a
        // `&self` method, would) while that borrow is still live.
        let device = self.device.clone();
        let add_fn = self.add_k.function.clone();

        let mut applied = 0usize;
        for target in &adapter.targets {
            let delta_dev = device
                .htod_sync_copy(&target.delta)
                .map_err(|e| format!("upload LoRA delta for '{}': {e}", target.name))?;

            let weight = self.find_lora_target_mut(&target.name).ok_or_else(|| {
                format!(
                    "LoRA adapter targets '{}' but this project's Model has no matching 2-D weight for it \
                     (dense-attention/FFN and Qwen3.5 hybrid Gated-Attention-layer tensors are the only \
                     supported LoRA targets in this round -- MoE's per-expert-stacked FFN tensors, the \
                     hybrid Gated DeltaNet mixer's non-Linear tensors, embeddings, and norms are not)",
                    target.name
                )
            })?;
            if weight.shape.len() != 2 || weight.shape[0] as usize != target.in_features || weight.shape[1] as usize != target.out_features {
                return Err(format!(
                    "LoRA adapter tensor '{}' has shape [in={}, out={}] but the base model's tensor has shape {:?} -- wrong base model?",
                    target.name, target.in_features, target.out_features, weight.shape
                ));
            }

            let n = weight.data.len() as u32;
            let threads = 256u32;
            let blocks = n.div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                add_fn
                    .clone()
                    .launch(launch_cfg, (&mut weight.data, &delta_dev, n))
                    .map_err(|e| format!("LoRA add launch for '{}': {e}", target.name))?;
            }
            applied += 1;
        }

        if applied == 0 {
            return Err("LoRA adapter matched no tensors in the base model -- check it targets a compatible architecture/checkpoint".to_string());
        }
        Ok(applied)
    }

    /// Locates the mutable device-resident 2-D [`Weight`] a LoRA adapter
    /// target name (`blk.{i}.{suffix}.weight`) refers to, across whichever
    /// architecture is loaded (dense/MoE via `self.layers`, Qwen3.5 hybrid
    /// via `self.hybrid`). Deliberately narrow: only the 2-D `nn.Linear`-
    /// shaped tensors every supported layer kind actually has are matched --
    /// MoE's per-expert-stacked FFN tensors (`ffn_gate_exps`/`ffn_up_exps`/
    /// `ffn_down_exps`, 3-D), the Gated DeltaNet mixer's non-Linear state-
    /// space tensors (`ssm_*`, `attn_qkv`, `attn_gate`), and anything outside
    /// a `blk.N.*` tensor (`token_embd`/`output`/norms) all fall through to
    /// the `None` arm and are rejected by `Self::apply_lora` with a clear
    /// error, rather than silently mismatched or misapplied.
    fn find_lora_target_mut(&mut self, name: &str) -> Option<&mut Weight> {
        let rest = name.strip_prefix("blk.")?;
        let (idx_str, rest) = rest.split_once('.')?;
        let idx: usize = idx_str.parse().ok()?;
        let suffix = rest.strip_suffix(".weight")?;

        if let Some(hybrid) = &mut self.hybrid {
            let layer = hybrid.layers.get_mut(idx)?;
            return match (layer, suffix) {
                (HybridLayerWeights::GatedAttention(l), "attn_q") => Some(&mut l.attn_q),
                (HybridLayerWeights::GatedAttention(l), "attn_k") => Some(&mut l.attn_k),
                (HybridLayerWeights::GatedAttention(l), "attn_v") => Some(&mut l.attn_v),
                (HybridLayerWeights::GatedAttention(l), "attn_output") => Some(&mut l.attn_output),
                (HybridLayerWeights::GatedAttention(l), "ffn_gate") => Some(&mut l.ffn_gate),
                (HybridLayerWeights::GatedAttention(l), "ffn_up") => Some(&mut l.ffn_up),
                (HybridLayerWeights::GatedAttention(l), "ffn_down") => Some(&mut l.ffn_down),
                (HybridLayerWeights::GatedDeltaNet(l), "ffn_gate") => Some(&mut l.ffn_gate),
                (HybridLayerWeights::GatedDeltaNet(l), "ffn_up") => Some(&mut l.ffn_up),
                (HybridLayerWeights::GatedDeltaNet(l), "ffn_down") => Some(&mut l.ffn_down),
                _ => None,
            };
        }

        let layer = self.layers.get_mut(idx)?;
        match (layer, suffix) {
            (LayerWeights::Dense(l), "attn_q") => Some(&mut l.attn_q),
            (LayerWeights::Dense(l), "attn_k") => Some(&mut l.attn_k),
            (LayerWeights::Dense(l), "attn_v") => Some(&mut l.attn_v),
            (LayerWeights::Dense(l), "attn_output") => Some(&mut l.attn_output),
            (LayerWeights::Dense(l), "ffn_gate") => Some(&mut l.ffn_gate),
            (LayerWeights::Dense(l), "ffn_up") => Some(&mut l.ffn_up),
            (LayerWeights::Dense(l), "ffn_down") => Some(&mut l.ffn_down),
            (LayerWeights::Moe(l), "attn_q") => Some(&mut l.attn_q),
            (LayerWeights::Moe(l), "attn_k") => Some(&mut l.attn_k),
            (LayerWeights::Moe(l), "attn_v") => Some(&mut l.attn_v),
            (LayerWeights::Moe(l), "attn_output") => Some(&mut l.attn_output),
            _ => None,
        }
    }

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

        // Created once per load, like every AOT kernel handle below -- see
        // the `cublas` field's doc comment on `Model` for why math mode is
        // pinned right after creation.
        let cublas = CudaBlas::new(device.clone()).map_err(|e| format!("cublas handle: {e:?}"))?;
        unsafe {
            cublas_sys::lib()
                .cublasSetMathMode(*cublas.handle(), cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)
                .result()
                .map_err(|e| format!("cublasSetMathMode: {e:?}"))?;
        }
        let rmsnorm_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_RMSNORM")), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope", "rope_kernel")?;
        let rope_batch_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_batch", "rope_batch_kernel")?;
        let silu_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_SILU_AND_MUL")), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV")), "gemv", "gemv_kernel")?;
        let gemv_gather_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV_GATHER")), "gemv_gather", "gemv_gather_kernel")?;
        let attn_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ATTENTION")), "attention", "attention_kernel")?;
        let attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ATTENTION_PREFILL")),
            "attention_prefill",
            "attention_prefill_kernel",
        )?;
        let mut elementwise_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ELEMENTWISE")),
            "elementwise",
            &["add_kernel", "split_qg_kernel", "sigmoid_gate_kernel", "moe_gather_kernel", "moe_scatter_add_kernel"],
        )?
        .into_iter();
        let add_k = elementwise_fns.next().ok_or("missing add_kernel")?;
        let split_qg_k = elementwise_fns.next().ok_or("missing split_qg_kernel")?;
        let sigmoid_gate_k = elementwise_fns.next().ok_or("missing sigmoid_gate_kernel")?;
        let moe_gather_k = elementwise_fns.next().ok_or("missing moe_gather_kernel")?;
        let moe_scatter_add_k = elementwise_fns.next().ok_or("missing moe_scatter_add_kernel")?;
        let mut dequant_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_DEQUANT")),
            "dequant",
            &["dequantize_q4k_kernel", "dequantize_q6k_kernel"],
        )?
        .into_iter();
        let dequant_q4k_k = dequant_fns.next().ok_or("missing dequantize_q4k_kernel")?;
        let dequant_q6k_k = dequant_fns.next().ok_or("missing dequantize_q6k_kernel")?;

        // Dequantizes straight from the mmap'd GGUF bytes (on-device for
        // Q4_K/Q6_K, Phase 2 round 3; host `Vec<f32>` scratch, immediately
        // dropped, for every other type) into device memory -- unlike
        // before round 1, no dequantized weight stays host-resident for the
        // model's lifetime, and no forward-pass call re-uploads it (see
        // `Weight`'s doc comment).
        let load_weight = |name: &str| -> Result<Weight, String> {
            load_weight_device(&device, &dequant_q4k_k, &dequant_q6k_k, file, name)
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
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
        eprintln!();

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
                let data = dequantize_tensor_to_device(
                    &device,
                    &dequant_q4k_k,
                    &dequant_q6k_k,
                    info.ggml_type,
                    bytes,
                    info.element_count(),
                )
                .map_err(|e| format!("load weight 'output.weight': {e}"))?;
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
            cublas,
            rmsnorm_k,
            rope_k,
            rope_batch_k,
            silu_k,
            gemv_k,
            gemv_gather_k,
            moe_gather_k,
            moe_scatter_add_k,
            attn_k,
            attn_prefill_k,
            add_k,
            split_qg_k,
            sigmoid_gate_k,
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

        // Created once per load, like every AOT kernel handle below -- see
        // the `cublas` field's doc comment on `Model` for why math mode is
        // pinned right after creation.
        let cublas = CudaBlas::new(device.clone()).map_err(|e| format!("cublas handle: {e:?}"))?;
        unsafe {
            cublas_sys::lib()
                .cublasSetMathMode(*cublas.handle(), cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)
                .result()
                .map_err(|e| format!("cublasSetMathMode: {e:?}"))?;
        }
        let rmsnorm_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_RMSNORM")), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope", "rope_kernel")?;
        let rope_batch_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_batch", "rope_batch_kernel")?;
        let silu_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_SILU_AND_MUL")), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV")), "gemv", "gemv_kernel")?;
        let gemv_gather_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV_GATHER")), "gemv_gather", "gemv_gather_kernel")?;
        let attn_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ATTENTION")), "attention", "attention_kernel")?;
        let attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ATTENTION_PREFILL")),
            "attention_prefill",
            "attention_prefill_kernel",
        )?;
        let mut elementwise_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ELEMENTWISE")),
            "elementwise",
            &["add_kernel", "split_qg_kernel", "sigmoid_gate_kernel", "moe_gather_kernel", "moe_scatter_add_kernel"],
        )?
        .into_iter();
        let add_k = elementwise_fns.next().ok_or("missing add_kernel")?;
        let split_qg_k = elementwise_fns.next().ok_or("missing split_qg_kernel")?;
        let sigmoid_gate_k = elementwise_fns.next().ok_or("missing sigmoid_gate_kernel")?;
        let moe_gather_k = elementwise_fns.next().ok_or("missing moe_gather_kernel")?;
        let moe_scatter_add_k = elementwise_fns.next().ok_or("missing moe_scatter_add_kernel")?;
        let mut gdn_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_GATED_DELTANET")),
            "gated_deltanet",
            &["gdn_conv_kernel", "gdn_l2_norm_kernel", "gdn_gates_kernel", "gdn_delta_kernel", "gdn_gated_norm_kernel"],
        )?
        .into_iter();
        let gdn_conv_k = gdn_fns.next().ok_or("missing gdn_conv_kernel")?;
        let gdn_l2_norm_k = gdn_fns.next().ok_or("missing gdn_l2_norm_kernel")?;
        let gdn_gates_k = gdn_fns.next().ok_or("missing gdn_gates_kernel")?;
        let gdn_delta_k = gdn_fns.next().ok_or("missing gdn_delta_kernel")?;
        let gdn_gated_norm_k = gdn_fns.next().ok_or("missing gdn_gated_norm_kernel")?;
        let mut dequant_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_DEQUANT")),
            "dequant",
            &["dequantize_q4k_kernel", "dequantize_q6k_kernel"],
        )?
        .into_iter();
        let dequant_q4k_k = dequant_fns.next().ok_or("missing dequantize_q4k_kernel")?;
        let dequant_q6k_k = dequant_fns.next().ok_or("missing dequantize_q6k_kernel")?;

        let load_weight = |name: &str| -> Result<Weight, String> {
            load_weight_device(&device, &dequant_q4k_k, &dequant_q6k_k, file, name)
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
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
        eprintln!();

        let token_embd_info =
            file.tensor_info("token_embd.weight").ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd =
            dequant::dequantize(token_embd_info.ggml_type, token_embd_bytes, token_embd_info.element_count())?;

        let output_norm = load_weight("output_norm.weight")?;

        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let data = dequantize_tensor_to_device(
                    &device,
                    &dequant_q4k_k,
                    &dequant_q6k_k,
                    info.ggml_type,
                    bytes,
                    info.element_count(),
                )
                .map_err(|e| format!("load weight 'output.weight': {e}"))?;
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
            cublas,
            rmsnorm_k,
            rope_k,
            rope_batch_k,
            silu_k,
            gemv_k,
            gemv_gather_k,
            moe_gather_k,
            moe_scatter_add_k,
            attn_k,
            attn_prefill_k,
            add_k,
            split_qg_k,
            sigmoid_gate_k,
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
    /// the scope this supports. `cfg`/`layers`/`expert_used_count` below are
    /// unused garbage (matching the `hybrid` path's own convention) --
    /// `forward_prompt` branches on `self.mla` before touching them.
    fn load_mla(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        let (mla_cfg, block_count, leading_dense) = parse_mla_config(file)?;

        // Created once per load, like every AOT kernel handle below -- see
        // the `cublas` field's doc comment on `Model` for why math mode is
        // pinned right after creation.
        let cublas = CudaBlas::new(device.clone()).map_err(|e| format!("cublas handle: {e:?}"))?;
        unsafe {
            cublas_sys::lib()
                .cublasSetMathMode(*cublas.handle(), cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)
                .result()
                .map_err(|e| format!("cublasSetMathMode: {e:?}"))?;
        }
        let rmsnorm_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_RMSNORM")), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope", "rope_kernel")?;
        let rope_batch_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_batch", "rope_batch_kernel")?;
        let silu_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_SILU_AND_MUL")), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV")), "gemv", "gemv_kernel")?;
        let gemv_gather_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_GEMV_GATHER")), "gemv_gather", "gemv_gather_kernel")?;
        let attn_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ATTENTION")), "attention", "attention_kernel")?;
        let attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ATTENTION_PREFILL")),
            "attention_prefill",
            "attention_prefill_kernel",
        )?;
        let mut elementwise_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ELEMENTWISE")),
            "elementwise",
            &[
                "add_kernel",
                "split_qg_kernel",
                "sigmoid_gate_kernel",
                "mla_extract_batch_kernel",
                "mla_concat_qcur_batch_kernel",
                "mla_write_kv_cache_batch_kernel",
                "moe_gather_kernel",
                "moe_scatter_add_kernel",
            ],
        )?
        .into_iter();
        let add_k = elementwise_fns.next().ok_or("missing add_kernel")?;
        let split_qg_k = elementwise_fns.next().ok_or("missing split_qg_kernel")?;
        let sigmoid_gate_k = elementwise_fns.next().ok_or("missing sigmoid_gate_kernel")?;
        let mla_extract_batch_k = elementwise_fns.next().ok_or("missing mla_extract_batch_kernel")?;
        let mla_concat_qcur_batch_k = elementwise_fns.next().ok_or("missing mla_concat_qcur_batch_kernel")?;
        let mla_write_kv_cache_batch_k = elementwise_fns.next().ok_or("missing mla_write_kv_cache_batch_kernel")?;
        let moe_gather_k = elementwise_fns.next().ok_or("missing moe_gather_kernel")?;
        let moe_scatter_add_k = elementwise_fns.next().ok_or("missing moe_scatter_add_kernel")?;
        let mla_attn_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_MLA_ATTENTION")), "mla_attention", "mla_attention_kernel")?;
        let mla_attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_MLA_ATTENTION_PREFILL")),
            "mla_attention_prefill",
            "mla_attention_prefill_kernel",
        )?;
        let rope_norm_k = aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_norm", "rope_norm_kernel")?;
        let rope_norm_yarn_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_norm_yarn", "rope_norm_yarn_kernel")?;
        let rope_norm_batch_k =
            aot::load_kernel(&device, include_bytes!(env!("COLDSTART_KERNEL_ROPE")), "rope_norm_batch", "rope_norm_batch_kernel")?;
        let rope_norm_yarn_batch_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_ROPE")),
            "rope_norm_yarn_batch",
            "rope_norm_yarn_batch_kernel",
        )?;
        let gemv_per_head_batch_k = aot::load_kernel(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_GEMV_PER_HEAD_BATCH")),
            "gemv_per_head_batch",
            "gemv_per_head_batch_kernel",
        )?;
        let mut dequant_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("COLDSTART_KERNEL_DEQUANT")),
            "dequant",
            &["dequantize_q4k_kernel", "dequantize_q6k_kernel"],
        )?
        .into_iter();
        let dequant_q4k_k = dequant_fns.next().ok_or("missing dequantize_q4k_kernel")?;
        let dequant_q6k_k = dequant_fns.next().ok_or("missing dequantize_q6k_kernel")?;

        let load_weight = |name: &str| -> Result<Weight, String> {
            load_weight_device(&device, &dequant_q4k_k, &dequant_q6k_k, file, name)
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
            let ffn = if i < leading_dense {
                MlaFfn::Dense {
                    ffn_gate: load_weight(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load_weight(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load_weight(&format!("blk.{i}.ffn_down.weight"))?,
                }
            } else {
                MlaFfn::Moe {
                    ffn_gate_inp: load_weight(&format!("blk.{i}.ffn_gate_inp.weight"))?,
                    ffn_gate_exps: load_weight(&format!("blk.{i}.ffn_gate_exps.weight"))?,
                    ffn_up_exps: load_weight(&format!("blk.{i}.ffn_up_exps.weight"))?,
                    ffn_down_exps: load_weight(&format!("blk.{i}.ffn_down_exps.weight"))?,
                    ffn_gate_shexp: load_weight(&format!("blk.{i}.ffn_gate_shexp.weight"))?,
                    ffn_up_shexp: load_weight(&format!("blk.{i}.ffn_up_shexp.weight"))?,
                    ffn_down_shexp: load_weight(&format!("blk.{i}.ffn_down_shexp.weight"))?,
                }
            };
            layers.push(MlaLayerWeights {
                attn_norm: load_weight(&format!("blk.{i}.attn_norm.weight"))?,
                wq: load_weight(&format!("blk.{i}.attn_q.weight"))?,
                wkv_a_mqa: load_weight(&format!("blk.{i}.attn_kv_a_mqa.weight"))?,
                attn_kv_a_norm: load_weight(&format!("blk.{i}.attn_kv_a_norm.weight"))?,
                wk_b: load_weight(&format!("blk.{i}.attn_k_b.weight"))?,
                wv_b: load_weight(&format!("blk.{i}.attn_v_b.weight"))?,
                wo: load_weight(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: load_weight(&format!("blk.{i}.ffn_norm.weight"))?,
                ffn,
            });
        }
        eprintln!();

        let token_embd_info =
            file.tensor_info("token_embd.weight").ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd =
            dequant::dequantize(token_embd_info.ggml_type, token_embd_bytes, token_embd_info.element_count())?;

        let output_norm = load_weight("output_norm.weight")?;

        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let data = dequantize_tensor_to_device(
                    &device,
                    &dequant_q4k_k,
                    &dequant_q6k_k,
                    info.ggml_type,
                    bytes,
                    info.element_count(),
                )
                .map_err(|e| format!("load weight 'output.weight': {e}"))?;
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
            cublas,
            rmsnorm_k,
            rope_k,
            rope_batch_k,
            silu_k,
            gemv_k,
            gemv_gather_k,
            moe_gather_k,
            moe_scatter_add_k,
            attn_k,
            attn_prefill_k,
            add_k,
            split_qg_k,
            sigmoid_gate_k,
            cfg: dummy_cfg,
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            output_norm,
            lm_head,
            tokenizer,
            hybrid: None,
            mla: Some(MlaModel {
                cfg: mla_cfg,
                layers,
                mla_attn_k,
                rope_norm_k,
                rope_norm_yarn_k,
                mla_attn_prefill_k,
                rope_norm_batch_k,
                rope_norm_yarn_batch_k,
                gemv_per_head_batch_k,
                mla_extract_batch_k,
                mla_concat_qcur_batch_k,
                mla_write_kv_cache_batch_k,
            }),
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

    /// Batched linear projection: `y[rows, out_features] = x[rows, in_features]
    /// @ w^T`, via cuBLAS Sgemm instead of `gemv_k`'s naive per-row dot
    /// product loop -- the actual win for prefill (`rows` = prompt length):
    /// cuBLAS reuses each weight byte across all `rows` output columns in one
    /// pass instead of re-reading the whole weight matrix from global memory
    /// once per row, turning a bandwidth-bound op into a compute-bound one.
    /// Not used for the `rows == 1` decode step (`Self::forward_one_token_dense`
    /// still uses `Self::gemv`) -- a GEMM with n=1 is just a slower GEMV.
    ///
    /// `w.data` is row-major `[out_features, in_features]` and `x` is
    /// row-major `[rows, in_features]`; cuBLAS is column-major. Reinterpreting
    /// each row-major buffer as column-major gives its transpose for free, so
    /// `w.data`'s raw buffer read as column-major is already `[in_features,
    /// out_features]` == w^T -- computing `y^T = w^T applied via CUBLAS_OP_T
    /// on w, CUBLAS_OP_N on x` (both operands untouched, no transpose copy)
    /// writes `y`'s raw buffer such that reading it back as row-major gives
    /// exactly `[rows, out_features]`. This is the single easiest place in
    /// the batched-prefill path to get a silently-wrong, non-crashing result,
    /// so verify it against `Self::gemv_raw` row-by-row before trusting it
    /// (`prefill_batching_tests::prefill_dense_batched_matches_sequential_prefill` does this).
    fn gemm(&self, x: &CudaSlice<f32>, w: &Weight, rows: usize) -> Result<CudaSlice<f32>, String> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != rows * in_features {
            return Err(format!("gemm: x.len()={} != rows*in_features={}", x.len(), rows * in_features));
        }

        let mut dev_y = self.device.alloc_zeros::<f32>(rows * out_features).map_err(|e| format!("gemm alloc y: {e}"))?;

        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: out_features as i32,
            n: rows as i32,
            k: in_features as i32,
            alpha: 1.0f32,
            lda: in_features as i32,
            ldb: in_features as i32,
            beta: 0.0f32,
            ldc: out_features as i32,
        };
        unsafe {
            self.cublas.gemm(cfg, &w.data, x, &mut dev_y).map_err(|e| format!("gemm launch: {e:?}"))?;
        }
        Ok(dev_y)
    }

    /// Like [`Self::gemm`], but generic over both operands being any device-resident
    /// reference (`&CudaSlice<f32>` or `&CudaView<f32>`) and taking explicit
    /// `in_features`/`out_features` instead of reading them off a `Weight` -- the
    /// same relationship [`Self::gemv_view`] has to [`Self::gemv_raw`]/[`Self::gemv`],
    /// kept as a separate sibling method rather than folded into `gemm` for the same
    /// reason that trio stays separate. Needed for grouped-GEMM MoE batching
    /// (`Self::forward_layer_moe_batched`, `Self::forward_mla_moe_ffn_batched`), where
    /// `w` is one expert's `CudaView` slice (see [`Self::expert_weight_view`]) and `x`
    /// is a freshly gathered, variable-`rows`-sized group buffer, neither of which fit
    /// `gemm`'s `&Weight` signature. No length assertion on `x` (unlike `gemm`) -- a
    /// generic view type isn't cheaply length-checked here, so correctness relies on
    /// the caller passing consistent `rows`/`in_features`.
    fn gemm_view<X: DevicePtr<f32>, W: DevicePtr<f32>>(
        &self,
        x: &X,
        w: &W,
        in_features: usize,
        out_features: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dev_y = self.device.alloc_zeros::<f32>(rows * out_features).map_err(|e| format!("gemm_view alloc y: {e}"))?;
        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: out_features as i32,
            n: rows as i32,
            k: in_features as i32,
            alpha: 1.0f32,
            lda: in_features as i32,
            ldb: in_features as i32,
            beta: 0.0f32,
            ldc: out_features as i32,
        };
        unsafe {
            self.cublas.gemm(cfg, w, x, &mut dev_y).map_err(|e| format!("gemm_view launch: {e:?}"))?;
        }
        Ok(dev_y)
    }

    /// Shared shape-validation/slicing logic behind [`Self::gemv_expert`] and grouped-
    /// GEMM MoE batching's per-expert-group GEMMs (`Self::forward_layer_moe_batched`,
    /// `Self::forward_mla_moe_ffn_batched`): resolves expert `expert_idx`'s slice of a
    /// per-expert-stacked 3-D MoE tensor (shape `[in_features, out_features,
    /// expert_count]`). Expert `e`'s `in_features * out_features` elements are a
    /// contiguous chunk already in the same row-major `(out_features, in_features)`
    /// layout as a standalone 2-D weight (see this module's doc comment), so
    /// `CudaSlice::slice` gives a zero-copy device-side view -- no device-to-device
    /// copy, let alone a host round-trip.
    fn expert_weight_view<'a>(w: &'a Weight, expert_idx: usize) -> Result<(CudaView<'a, f32>, usize, usize), String> {
        let (in_features, out_features, expert_count) = match w.shape.as_slice() {
            [i, o, e] => (*i as usize, *o as usize, *e as usize),
            other => return Err(format!("expert_weight_view: expected 3-D per-expert tensor shape, got {other:?}")),
        };
        if expert_idx >= expert_count {
            return Err(format!("expert_weight_view: expert_idx {expert_idx} out of range (expert_count={expert_count})"));
        }
        let expert_len = in_features * out_features;
        let start = expert_idx * expert_len;
        Ok((w.data.slice(start..start + expert_len), in_features, out_features))
    }

    /// GEMV against expert `expert_idx`'s slice of a per-expert-stacked 3-D MoE tensor
    /// (see [`Self::expert_weight_view`]). `x` is generic (like [`Self::gemv_view`],
    /// which this delegates to) so callers can pass either an owned `&CudaSlice<f32>`
    /// (the ordinary per-token MoE path) or a `&CudaView<f32>` row-slice of a larger
    /// buffer. Used by the single-token decode path (`Self::forward_layer_moe`,
    /// `Self::forward_mla_moe_ffn`) -- the batched-prefill MoE FFN
    /// (`Self::forward_layer_moe_batched`, `Self::forward_mla_moe_ffn_batched`) instead
    /// groups every row routed to the same expert and runs one [`Self::gemm_view`]
    /// call per expert, so it calls [`Self::expert_weight_view`] directly rather than
    /// through this single-row wrapper.
    fn gemv_expert<X: DeviceRepr>(&self, x: X, w: &Weight, expert_idx: usize) -> Result<CudaSlice<f32>, String> {
        let (view, in_features, out_features) = Self::expert_weight_view(w, expert_idx)?;
        self.gemv_view(x, &view, in_features, out_features)
    }

    /// Grouped-GEMM MoE batching's gather step (`moe_gather_kernel`,
    /// `kernels_cuda/elementwise.cu`): copies one expert group's selected rows out of
    /// `src` (a batched-prefill `[rows, hidden_size]` buffer, e.g. `ffn_normed`) into a
    /// fresh contiguous `[perm_row.len(), hidden_size]` buffer, so the group can be run
    /// through that expert's weights as one [`Self::gemm_view`] call. `perm_row` is
    /// already device-resident (uploaded once per expert group by the caller).
    fn moe_gather(&self, src: &CudaSlice<f32>, perm_row: &CudaSlice<u32>, hidden_size: usize) -> Result<CudaSlice<f32>, String> {
        let num_assignments = perm_row.len();
        let mut dst = self
            .device
            .alloc_zeros::<f32>(num_assignments * hidden_size)
            .map_err(|e| format!("moe_gather alloc: {e}"))?;
        let n = (num_assignments * hidden_size) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.moe_gather_k
                .function
                .clone()
                .launch(launch_cfg, (src, perm_row, &mut dst, num_assignments as u32, hidden_size as u32))
                .map_err(|e| format!("moe_gather launch: {e}"))?;
        }
        Ok(dst)
    }

    /// Inverse of [`Self::moe_gather`]: weighted scatter-add of one expert group's
    /// down-projected FFN output (`src`, `[dest_row.len(), hidden_size]`) back into
    /// `dst` (`[rows, hidden_size]`, already allocated/seeded by the caller --
    /// zeroed for dense/MoE, or pre-seeded with MLA's shared-expert output).
    /// `moe_scatter_add_kernel` uses a plain `+=`, not an atomic add -- safe only
    /// because `dest_row` (this expert's group of selected rows) has no duplicate
    /// entries, which top-k routing guarantees (a row never selects the same expert
    /// twice) and because CUDA kernel launches on the default stream (the only stream
    /// this project uses) run sequentially, so distinct expert groups' launches never
    /// overlap either.
    fn moe_scatter_add(
        &self,
        src: &CudaSlice<f32>,
        dest_row: &CudaSlice<u32>,
        weight: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        hidden_size: usize,
    ) -> Result<(), String> {
        let num_assignments = dest_row.len();
        let n = (num_assignments * hidden_size) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.moe_scatter_add_k
                .function
                .clone()
                .launch(launch_cfg, (src, dest_row, weight, dst, num_assignments as u32, hidden_size as u32))
                .map_err(|e| format!("moe_scatter_add launch: {e}"))?;
        }
        Ok(())
    }

    /// Like `gemv`, but computes only the output rows named by
    /// `row_indices` instead of every row `0..out_features` -- System1's
    /// candidate-subset LM-head scoring (`Self::system1_evaluate`), so a
    /// full-vocab GEMV and a vocab-sized D2H transfer are never paid when
    /// only a handful of candidate token ids' logits are needed. Returns
    /// the gathered logits already downloaded to host (unlike `gemv_raw`/
    /// `gemv_expert`, which return a device-resident `CudaSlice` for further
    /// on-device chaining) since every call site here is a leaf op wanting
    /// host floats.
    fn gemv_gather(&self, x: &CudaSlice<f32>, w: &Weight, row_indices: &[u32]) -> Result<Vec<f32>, String> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != in_features {
            return Err(format!("gemv_gather: x.len()={} != in_features={in_features}", x.len()));
        }
        if row_indices.is_empty() {
            return Err("gemv_gather: row_indices must not be empty".to_string());
        }
        if let Some(&bad) = row_indices.iter().find(|&&r| r as usize >= out_features) {
            return Err(format!("gemv_gather: row index {bad} out of range (out_features={out_features})"));
        }
        let num_rows = row_indices.len();
        let dev_indices = self.device.htod_sync_copy(row_indices).map_err(|e| format!("gemv_gather upload row_indices: {e}"))?;
        let mut dev_y = self.device.alloc_zeros::<f32>(num_rows).map_err(|e| format!("gemv_gather alloc y: {e}"))?;
        let threads = 256u32;
        let blocks = (num_rows as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.gemv_gather_k
                .function
                .clone()
                .launch(launch_cfg, (x, &w.data, &dev_indices, &mut dev_y, in_features as u32, num_rows as u32))
                .map_err(|e| format!("gemv_gather launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_y).map_err(|e| format!("gemv_gather dtoh: {e}"))
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

    /// Batched-prefill variant of [`Self::rope`]: rotates `rows` rows of `t`
    /// in one launch, row `r` at absolute position `start_pos + r` (unlike
    /// `rope`'s single scalar `position`, which only serves the `rows == 1`
    /// decode step). `t` is row-major `[rows, num_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn rope_batch(
        &self,
        t: &mut CudaSlice<f32>,
        start_pos: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rows: usize,
        base: f32,
    ) -> Result<(), String> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            self.rope_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (t, start_pos as u32, num_heads as u32, head_dim as u32, rotary_dim as u32, rows as u32, base),
                )
                .map_err(|e| format!("rope_batch launch: {e}"))?;
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

    /// Like [`Self::rope_norm`], but launches `m.rope_norm_yarn_k`
    /// (`rope_norm_yarn_kernel`) with the extra YaRN parameters from
    /// `cfg.yarn` (see `MlaYarnConfig`'s doc comment). Used instead of
    /// `Self::rope_norm` whenever `cfg.yarn.is_some()`.
    #[allow(clippy::too_many_arguments)]
    fn rope_norm_yarn(
        &self,
        m: &MlaModel,
        yarn: &MlaYarnConfig,
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
            m.rope_norm_yarn_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        position as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        base,
                        yarn.freq_scale,
                        yarn.ext_factor,
                        yarn.attn_factor,
                        yarn.corr_dim_start,
                        yarn.corr_dim_end,
                    ),
                )
                .map_err(|e| format!("rope_norm_yarn launch: {e}"))?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::rope_norm`]: rotates `rows` rows of `t` in
    /// one launch, row `r` at absolute position `start_pos + r` (`rope_norm_batch_kernel`,
    /// same batching idea as [`Self::rope_batch`] applied to the consecutive-pair
    /// rotation). `t` is row-major `[rows, num_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn rope_norm_batch(
        &self,
        m: &MlaModel,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        start_pos: usize,
        rows: usize,
        base: f32,
    ) -> Result<(), String> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            m.rope_norm_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (t, start_pos as u32, num_heads as u32, head_dim as u32, rotary_dim as u32, rows as u32, base),
                )
                .map_err(|e| format!("rope_norm_batch launch: {e}"))?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::rope_norm_yarn`]: like [`Self::rope_norm_batch`],
    /// plus the extra YaRN parameters from `cfg.yarn` (`rope_norm_yarn_batch_kernel`).
    #[allow(clippy::too_many_arguments)]
    fn rope_norm_yarn_batch(
        &self,
        m: &MlaModel,
        yarn: &MlaYarnConfig,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        start_pos: usize,
        rows: usize,
        base: f32,
    ) -> Result<(), String> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            m.rope_norm_yarn_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        start_pos as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        rows as u32,
                        base,
                        yarn.freq_scale,
                        yarn.ext_factor,
                        yarn.attn_factor,
                        yarn.corr_dim_start,
                        yarn.corr_dim_end,
                    ),
                )
                .map_err(|e| format!("rope_norm_yarn_batch launch: {e}"))?;
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

    /// Batched-prefill variant of [`Self::attention`]: scores `rows` new
    /// query rows against the shared K/V cache in one launch (grid gains a
    /// query-row dimension), each row causally masked to its own
    /// `start_pos + row + 1` positions instead of one shared `seq_len`.
    /// `k_cache`/`v_cache` must already cover `0..start_pos+rows` positions
    /// (this batch's own K/V, written by the caller before this call --
    /// `Self::forward_attn_block_batched`). `q` is row-major `[rows,
    /// num_q_heads, head_dim]`; returns row-major `[rows, num_q_heads,
    /// head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaView<f32>,
        v_cache: &CudaView<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(rows * num_q_heads * head_dim)
            .map_err(|e| format!("attn_prefill alloc out: {e}"))?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let max_seq_len = start_pos + rows;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, rows as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (max_seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            self.attn_prefill_k
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
                        start_pos as u32,
                        rows as u32,
                        scale,
                    ),
                )
                .map_err(|e| format!("attn_prefill launch: {e}"))?;
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

    /// Batched-prefill variant of [`Self::gemv_per_head`]: applies a per-head-stacked
    /// weight tensor `w` to every head of every row of a batched `rows`-row input in
    /// one launch (`gemv_per_head_batch_kernel`) instead of `rows` separate
    /// `Self::gemv_per_head` calls (each of which itself loops `n_head` times --
    /// looping this per row would mean `rows * n_head` launches, the exact
    /// launch-count blowup this kernel exists to avoid; see
    /// `Self::forward_mla_attn_block_batched`'s doc comment for the math). Unlike
    /// `gemv_per_head`, `x` need not be a contiguous `[rows, n_head, in_features]`
    /// buffer -- `x_row_stride`/`x_head_stride`/`x_head_offset` let the caller read
    /// directly out of a wider strided buffer (MLA's absorption step reads q_nope
    /// straight out of the `wq` projection output, which interleaves q_nope/q_pe per
    /// head, with no separate gather step). `out` is always a freshly allocated,
    /// contiguous `[rows, n_head, out_features]` buffer.
    #[allow(clippy::too_many_arguments)]
    fn gemv_per_head_batch(
        &self,
        m: &MlaModel,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
        n_head: usize,
        x_row_stride: usize,
        x_head_stride: usize,
        x_head_offset: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let (in_features, out_features, head_count) = match w.shape.as_slice() {
            [i, o, h] => (*i as usize, *o as usize, *h as usize),
            other => return Err(format!("gemv_per_head_batch: expected 3-D per-head tensor shape, got {other:?}")),
        };
        if head_count != n_head {
            return Err(format!("gemv_per_head_batch: tensor's head dim {head_count} != n_head {n_head}"));
        }
        let mut out =
            self.device.alloc_zeros::<f32>(rows * n_head * out_features).map_err(|e| format!("gemv_per_head_batch alloc: {e}"))?;

        let threads = 256u32;
        let out_blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg =
            LaunchConfig { grid_dim: (out_blocks, n_head as u32, rows as u32), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            m.gemv_per_head_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        &w.data,
                        &mut out,
                        rows as u32,
                        n_head as u32,
                        in_features as u32,
                        out_features as u32,
                        x_row_stride as u32,
                        x_head_stride as u32,
                        x_head_offset as u32,
                    ),
                )
                .map_err(|e| format!("gemv_per_head_batch launch: {e}"))?;
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

    /// Batched-prefill variant of [`Self::mla_attention`]: scores `rows` new query
    /// rows against the shared compressed MQA KV cache in one launch (grid gains a
    /// query-row dimension), each row causally masked to its own `start_pos + row +
    /// 1` positions -- same relationship [`Self::attention_prefill`] has to
    /// [`Self::attention`], applied to MLA's MQA/compressed-KV shape. `q` is
    /// row-major `[rows, num_q_heads, qk_dim]`; returns row-major `[rows,
    /// num_q_heads, v_dim]`. `kv_cache` must already cover `0..start_pos+rows`
    /// positions (this batch's own compressed KV, written by the caller before this
    /// call -- `Self::forward_mla_attn_block_batched`).
    #[allow(clippy::too_many_arguments)]
    fn mla_attention_prefill(
        &self,
        m: &MlaModel,
        q: &CudaSlice<f32>,
        kv_cache: &CudaView<f32>,
        num_q_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        start_pos: usize,
        rows: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dev_out =
            self.device.alloc_zeros::<f32>(rows * num_q_heads * v_dim).map_err(|e| format!("mla_attn_prefill alloc out: {e}"))?;
        let block_dim = (qk_dim as u32).next_power_of_two();
        let max_seq_len = start_pos + rows;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, rows as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (max_seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            m.mla_attn_prefill_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        kv_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        qk_dim as u32,
                        v_dim as u32,
                        start_pos as u32,
                        rows as u32,
                        scale,
                    ),
                )
                .map_err(|e| format!("mla_attn_prefill launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// Batched-prefill helper: extracts a fixed-width, fixed-offset sub-slice of
    /// every (row, head) entry of `src` into its own contiguous output
    /// (`mla_extract_batch_kernel`) -- see that kernel's doc comment
    /// (`kernels_cuda/elementwise.cu`) for the exact layout contract.
    #[allow(clippy::too_many_arguments)]
    fn mla_extract_batch(
        &self,
        m: &MlaModel,
        src: &CudaSlice<f32>,
        rows: usize,
        num_heads: usize,
        src_head_width: usize,
        dst_width: usize,
        src_head_offset: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let mut dst = self
            .device
            .alloc_zeros::<f32>(rows * num_heads * dst_width)
            .map_err(|e| format!("mla_extract_batch alloc: {e}"))?;
        let n = (rows * num_heads * dst_width) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            m.mla_extract_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (src, &mut dst, rows as u32, num_heads as u32, src_head_width as u32, dst_width as u32, src_head_offset as u32),
                )
                .map_err(|e| format!("mla_extract_batch launch: {e}"))?;
        }
        Ok(dst)
    }

    /// Batched-prefill helper: merges per-head `absorbed` (kv_lora-wide) and
    /// already-RoPE'd `q_pe` (qk_rope-wide) into Qcur's per-head row
    /// (`mla_concat_qcur_batch_kernel`).
    fn mla_concat_qcur_batch(
        &self,
        m: &MlaModel,
        absorbed: &CudaSlice<f32>,
        q_pe: &CudaSlice<f32>,
        rows: usize,
        n_head: usize,
        kv_lora: usize,
        qk_rope: usize,
    ) -> Result<CudaSlice<f32>, String> {
        let qk_dim = kv_lora + qk_rope;
        let mut out =
            self.device.alloc_zeros::<f32>(rows * n_head * qk_dim).map_err(|e| format!("mla_concat_qcur_batch alloc: {e}"))?;
        let n = (rows * n_head * qk_dim) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            m.mla_concat_qcur_batch_k
                .function
                .clone()
                .launch(launch_cfg, (absorbed, q_pe, &mut out, rows as u32, n_head as u32, kv_lora as u32, qk_rope as u32))
                .map_err(|e| format!("mla_concat_qcur_batch launch: {e}"))?;
        }
        Ok(out)
    }

    /// Batched-prefill helper: writes this batch's compressed Kcur (`kv_cmpr` concat
    /// `k_pe`) into `kv_cache` at rows `start_pos..start_pos+rows`
    /// (`mla_write_kv_cache_batch_kernel`).
    #[allow(clippy::too_many_arguments)]
    fn mla_write_kv_cache_batch(
        &self,
        m: &MlaModel,
        kv_cache: &mut CudaSlice<f32>,
        kv_cmpr: &CudaSlice<f32>,
        k_pe: &CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_lora: usize,
        qk_rope: usize,
    ) -> Result<(), String> {
        let qk_dim = kv_lora + qk_rope;
        let n = (rows * qk_dim) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            m.mla_write_kv_cache_batch_k
                .function
                .clone()
                .launch(launch_cfg, (kv_cache, kv_cmpr, k_pe, start_pos as u32, rows as u32, kv_lora as u32, qk_rope as u32))
                .map_err(|e| format!("mla_write_kv_cache_batch launch: {e}"))?;
        }
        Ok(())
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

    /// Batched-prefill variant of [`Self::forward_attn_block`]: normalizes,
    /// projects, RoPEs, and attends over `rows` positions at once (`rows *
    /// hidden_size` flat `hidden`, row-major) instead of one position per
    /// call -- see `Self::gemm`/`Self::rope_batch`/`Self::attention_prefill`.
    /// `start_pos` is this batch's first row's absolute position (row `r` is
    /// `start_pos + r`), matching `Self::prefill_dense_batched`'s resume
    /// convention (`--import-kv`'s `start_pos > 0` case).
    #[allow(clippy::too_many_arguments)]
    fn forward_attn_block_batched(
        &self,
        attn_norm: &Weight,
        attn_q: &Weight,
        attn_k: &Weight,
        attn_v: &Weight,
        attn_output: &Weight,
        attn_q_norm: &Option<Weight>,
        attn_k_norm: &Option<Weight>,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(&hidden, &attn_norm.data, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let mut q = self.gemm(&normed, attn_q, rows)?;
        let mut k = self.gemm(&normed, attn_k, rows)?;
        let v = self.gemm(&normed, attn_v, rows)?;

        if let Some(qn) = attn_q_norm {
            q = self.rmsnorm(&q, &qn.data, rows * cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }
        if let Some(kn) = attn_k_norm {
            k = self.rmsnorm(&k, &kn.data, rows * cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }

        self.rope_batch(&mut q, start_pos, cfg.num_q_heads, cfg.head_dim, cfg.rotary_dim, rows, cfg.rope_base)?;
        self.rope_batch(&mut k, start_pos, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim, rows, cfg.rope_base)?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = start_pos * kv_stride;
        let write_len = rows * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + write_len);
            self.device.dtod_copy(&k, &mut dst).map_err(|e| format!("attn_batched kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + write_len);
            self.device.dtod_copy(&v, &mut dst).map_err(|e| format!("attn_batched kv-cache dtod v: {e}"))?;
        }
        let seq_len = start_pos + rows;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let attn_out =
            self.attention_prefill(&q, &k_view, &v_view, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, start_pos, rows)?;
        let o_proj = self.gemm(&attn_out, attn_output, rows)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Batched-prefill variant of [`Self::forward_layer_dense`]: every
    /// projection in both the attention block and the FFN becomes one GEMM
    /// over all `rows` positions instead of `rows` separate GEMV launches --
    /// `Self::silu_and_mul`/`Self::add_inplace` need no change (already flat
    /// elementwise ops, see their doc comments).
    fn forward_layer_dense_batched(
        &self,
        layer: &DenseLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let mut post_attn = self.forward_attn_block_batched(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            hidden,
            start_pos,
            rows,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;
        let gate = self.gemm(&ffn_normed, &layer.ffn_gate, rows)?;
        let up = self.gemm(&ffn_normed, &layer.ffn_up, rows)?;
        let activated = self.silu_and_mul(&gate, &up, rows * cfg.ffn_hidden_size)?;
        let down = self.gemm(&activated, &layer.ffn_down, rows)?;

        self.add_inplace(&mut post_attn, &down)?;
        Ok(post_attn)
    }

    /// Grouped-GEMM MoE FFN batching core, shared by [`Self::forward_layer_moe_batched`]
    /// (dense/MoE Qwen3) and [`Self::forward_mla_moe_ffn_batched`] (DeepSeek-V2/V3
    /// MLA): each of `rows` tokens routes to a different, data-dependent top-k subset
    /// of experts (host-side `crate::moe::route_top_k`/`route_top_k_with_norm`), so
    /// there's no single shared weight matrix to run one GEMM against like the
    /// attention block's projections. Instead this groups rows by *which expert they
    /// selected* (bounded by `expert_count`, not by `rows * k`), and for every expert
    /// with at least one assigned row: gathers that group's rows out of `ffn_normed`
    /// (`Self::moe_gather`), runs the group through the expert's gate/up/down weights
    /// as one real GEMM apiece (`Self::gemm_view` against `Self::expert_weight_view`'s
    /// zero-copy per-expert slice) instead of `group_size` separate `Self::gemv_expert`
    /// launches, and weighted-scatter-adds the down-projected result straight into
    /// `ffn_out` (`Self::moe_scatter_add`) -- no host round trip anywhere in this loop,
    /// unlike the single-token/per-row path this replaces. `ffn_out` must already be
    /// allocated to `[rows, hidden_size]` and seeded with whatever this should
    /// accumulate on top of (zero for dense/MoE, MLA's always-on shared-expert output
    /// for `MlaFfn::Moe`); this function only ever adds into it. `weight_scale` folds
    /// in a caller-side scalar (MLA's `routed_scaling_factor`; `1.0` -- a no-op -- for
    /// dense/MoE, which has no such knob) so the scatter kernel itself stays
    /// architecture-agnostic.
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_grouped(
        &self,
        ffn_normed: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        router_logits: &[f32],
        expert_count: usize,
        k: usize,
        normalize_top_k: bool,
        weight_scale: f32,
        ffn_gate_exps: &Weight,
        ffn_up_exps: &Weight,
        ffn_down_exps: &Weight,
        ffn_out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let mut groups: Vec<Vec<(u32, f32)>> = vec![Vec::new(); expert_count];
        for row in 0..rows {
            let row_logits = &router_logits[row * expert_count..(row + 1) * expert_count];
            let routed = route_top_k_with_norm(row_logits, k, normalize_top_k)?;
            for (expert_idx, weight) in routed {
                groups[expert_idx].push((row as u32, weight * weight_scale));
            }
        }

        for (expert_idx, group) in groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let rows_e: Vec<u32> = group.iter().map(|&(r, _)| r).collect();
            let weights_e: Vec<f32> = group.iter().map(|&(_, w)| w).collect();

            let perm_row = self.device.htod_sync_copy(&rows_e).map_err(|e| format!("moe_ffn_grouped upload rows_e: {e}"))?;
            let weight_dev = self.device.htod_sync_copy(&weights_e).map_err(|e| format!("moe_ffn_grouped upload weights_e: {e}"))?;
            let group_size = rows_e.len();

            let x_e = self.moe_gather(ffn_normed, &perm_row, hidden_size)?;
            let (gate_w, in_features, gate_out_features) = Self::expert_weight_view(ffn_gate_exps, expert_idx)?;
            let gate = self.gemm_view(&x_e, &gate_w, in_features, gate_out_features, group_size)?;
            let (up_w, _, up_out_features) = Self::expert_weight_view(ffn_up_exps, expert_idx)?;
            let up = self.gemm_view(&x_e, &up_w, in_features, up_out_features, group_size)?;
            let activated = self.silu_and_mul(&gate, &up, group_size * gate_out_features)?;
            let (down_w, down_in_features, down_out_features) = Self::expert_weight_view(ffn_down_exps, expert_idx)?;
            let down = self.gemm_view(&activated, &down_w, down_in_features, down_out_features, group_size)?;

            self.moe_scatter_add(&down, &perm_row, &weight_dev, ffn_out, hidden_size)?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::forward_layer_moe`]: the attention block
    /// batches identically to the dense case (`Self::forward_attn_block_batched`), and
    /// the FFN tail now batches too, via grouped-GEMM MoE batching
    /// (`Self::moe_ffn_grouped`) instead of `Self::forward_layer_moe`'s per-row
    /// `Self::gemv_expert` loop.
    fn forward_layer_moe_batched(
        &self,
        layer: &MoeLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let mut post_attn = self.forward_attn_block_batched(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            hidden,
            start_pos,
            rows,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;
        let router_logits_dev = self.gemm(&ffn_normed, &layer.ffn_gate_inp, rows)?;
        let router_logits = self.device.dtoh_sync_copy(&router_logits_dev).map_err(|e| format!("moe router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or("forward_layer_moe_batched called on a model with no expert_used_count")?;
        let num_experts = router_logits.len() / rows;

        let mut ffn_out = self.device.alloc_zeros::<f32>(rows * cfg.hidden_size).map_err(|e| format!("moe ffn_out alloc: {e}"))?;
        self.moe_ffn_grouped(
            &ffn_normed,
            rows,
            cfg.hidden_size,
            &router_logits,
            num_experts,
            k,
            true, // Qwen3-MoE's convention: renormalize the selected top-k weights (crate::moe::route_top_k).
            1.0,  // no routed_scaling_factor-equivalent knob for dense/MoE.
            &layer.ffn_gate_exps,
            &layer.ffn_up_exps,
            &layer.ffn_down_exps,
            &mut ffn_out,
        )?;

        self.add_inplace(&mut post_attn, &ffn_out)?;
        Ok(post_attn)
    }

    /// Batched-prefill variant of [`Self::forward_layer`].
    fn forward_layer_batched(
        &self,
        layer: &LayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        match layer {
            LayerWeights::Dense(l) => self.forward_layer_dense_batched(l, hidden, start_pos, rows, k_cache, v_cache),
            LayerWeights::Moe(l) => self.forward_layer_moe_batched(l, hidden, start_pos, rows, k_cache, v_cache),
        }
    }

    /// Encodes `prompt`, runs it through every layer one position at a time
    /// (real causal self-attention throughout, matching RustFeference's own
    /// documented scope choice for its minimal forward pass), and returns
    /// the argmax-sampled first generated token id plus its decoded text.
    pub fn forward_prompt(&self, prompt: &str) -> Result<(u32, String), String> {
        if let Some(h) = &self.hybrid {
            return self.forward_prompt_hybrid(h, prompt);
        }
        if let Some(m) = &self.mla {
            return self.forward_prompt_mla(m, prompt);
        }
        let (generated, text, _k_caches, _v_caches, _seq_len) = self.generate_dense_impl(prompt, None, 1, |_logits| {})?;
        Ok((generated[0], text))
    }

    /// Phase 3 (State I/O) round 2's generation entry point: like
    /// `forward_prompt`, but produces up to `max_new_tokens` tokens (feeding
    /// each generated id back in as the next position's input embedding,
    /// stopping early on the tokenizer's `eos_token_id`) and, when
    /// `imported` is `Some`, resumes from a previously exported cache
    /// instead of starting at position 0 -- `prompt` is then the
    /// continuation text appended after the imported cache's positions, not
    /// a fresh prompt (no BOS is inserted). `on_first_token` is called
    /// exactly once, right after the first new token is produced, with that
    /// token's full logits vector -- callers can ignore the argument to just
    /// capture accurate "time to first token" timing (as `src/bin/qwen3_coldstart.rs`
    /// does), or inspect the logits themselves (as `src/bin/check_correctness.rs`
    /// does) -- even when `max_new_tokens > 1` keeps the call running past that
    /// point.
    ///
    /// Dense/MoE and the Qwen3.5 hybrid mixer support resume as of round 2
    /// (the hybrid `GatedDeltaNet` sublayers' `conv_state`/`recurrent` need
    /// no `start_pos` handling at all -- see `kv_io.rs`'s doc comment); MLA
    /// as of round 3, via `generate_mla_impl`.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        imported: Option<&crate::kv_io::ImportedKv>,
        on_first_token: impl FnMut(&[f32]),
    ) -> Result<(Vec<u32>, String), String> {
        match imported {
            Some(crate::kv_io::ImportedKv::Dense(cache)) => {
                if self.hybrid.is_some() || self.mla.is_some() {
                    return Err("imported KV cache file is dense/MoE format, but this model is not a dense/MoE Qwen3 model".to_string());
                }
                let (generated, text, _, _, _) = self.generate_dense_impl(prompt, Some(cache), max_new_tokens, on_first_token)?;
                Ok((generated, text))
            }
            Some(crate::kv_io::ImportedKv::Hybrid(cache)) => {
                let h = self
                    .hybrid
                    .as_ref()
                    .ok_or("imported KV cache file is hybrid format, but this model is not a Qwen3.5 hybrid model")?;
                let (generated, text, _, _) = self.generate_hybrid_impl(h, prompt, Some(cache), max_new_tokens, on_first_token)?;
                Ok((generated, text))
            }
            Some(crate::kv_io::ImportedKv::Mla(cache)) => {
                let m = self.mla.as_ref().ok_or("imported KV cache file is MLA format, but this model is not an MLA model")?;
                let (generated, text, _, _) = self.generate_mla_impl(m, prompt, Some(cache), max_new_tokens, on_first_token)?;
                Ok((generated, text))
            }
            None => {
                if let Some(h) = &self.hybrid {
                    let (generated, text, _, _) = self.generate_hybrid_impl(h, prompt, None, max_new_tokens, on_first_token)?;
                    return Ok((generated, text));
                }
                if let Some(m) = &self.mla {
                    let (generated, text, _, _) = self.generate_mla_impl(m, prompt, None, max_new_tokens, on_first_token)?;
                    return Ok((generated, text));
                }
                let (generated, text, _, _, _) = self.generate_dense_impl(prompt, None, max_new_tokens, on_first_token)?;
                Ok((generated, text))
            }
        }
    }

    /// Sequential dense prefill: encodes `prompt` (a continuation, not a fresh
    /// prompt, when `imported.is_some()` -- no BOS is inserted in that
    /// case), seeds the K/V cache from `imported` first when resuming, then
    /// runs every prompt position through `forward_one_token_dense`
    /// sequentially, exactly like a from-scratch run just offset by
    /// `imported`'s `seq_len` -- the pre-batching behavior, kept unchanged
    /// as the verification oracle for [`Self::prefill_dense_batched`]
    /// (`prefill_batching_tests` below), the same role [`Self::prefill_hybrid`]
    /// plays for `prefill_hybrid_batched`. Not used by `generate_dense_impl`/
    /// `system1_evaluate` any more (both switched to `prefill_dense_batched`
    /// -- see README's "Batched Prefill GEMM" section) -- kept only for the
    /// oracle role and any future direct caller. `extra_headroom` sizes the
    /// K/V cache with that many additional position slots beyond the
    /// encoded prompt itself. Returns the encoded prompt ids (including any
    /// inserted BOS), the final position's hidden state, the filled K/V
    /// caches, and the next absolute position a caller may write into.
    #[cfg(test)]
    fn prefill_dense(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, Vec<CudaSlice<f32>>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        let kv_stride = self.cfg.num_kv_heads * self.cfg.head_dim;
        // Preallocated up front (Phase 2 round 2 sized this to exactly the
        // prompt's token count; round 2 of Phase 3 sizes it for the whole
        // run -- imported positions, the continuation prompt, and headroom
        // for every position a caller might still write -- since
        // `forward_attn_block` indexes into it by absolute position and
        // needs the buffer to already be that big).
        let total_len = start_pos + ids.len() + extra_headroom;
        let mut k_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc k_cache: {e}"))?;
        let mut v_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc v_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.num_kv_heads != self.cfg.num_kv_heads || cache.head_dim != self.cfg.head_dim {
                return Err(format!(
                    "imported KV cache shape mismatch: file has num_kv_heads={} head_dim={}, model expects num_kv_heads={} head_dim={}",
                    cache.num_kv_heads, cache.head_dim, self.cfg.num_kv_heads, self.cfg.head_dim
                ));
            }
            if cache.k_caches.len() != self.layers.len() {
                return Err(format!("imported KV cache has {} layers, model has {}", cache.k_caches.len(), self.layers.len()));
            }
            let imported_len = cache.seq_len * kv_stride;
            for (layer_idx, (k_host, v_host)) in cache.k_caches.iter().zip(&cache.v_caches).enumerate() {
                let mut k_dst = k_caches[layer_idx].slice_mut(0..imported_len);
                self.device.htod_sync_copy_into(k_host, &mut k_dst).map_err(|e| format!("import k_cache htod layer {layer_idx}: {e}"))?;
                let mut v_dst = v_caches[layer_idx].slice_mut(0..imported_len);
                self.device.htod_sync_copy_into(v_host, &mut v_dst).map_err(|e| format!("import v_cache htod layer {layer_idx}: {e}"))?;
            }
        }

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_dense(token_id, position, &mut k_caches, &mut v_caches)?);
            position += 1;
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        Ok((ids, hidden, k_caches, v_caches, position))
    }

    /// Batched-prefill variant of [`Self::prefill_dense`]: same signature and
    /// KV-cache allocation/import logic (kept byte-identical to `prefill_dense`
    /// so the two stay directly comparable -- see
    /// `prefill_dense_batched_matches_sequential_prefill` below), but runs
    /// every prompt token through each layer in one batched pass
    /// (`Self::forward_layer_batched`, `rows = ids.len()`) instead of looping
    /// `forward_one_token_dense` once per token. Unlike `prefill_dense`,
    /// whose returned `hidden` is already the single last-position vector,
    /// this returns the *whole* `[rows, hidden_size]` batched hidden state --
    /// callers that only want the last prompt position (every current
    /// caller) must slice it out with [`Self::last_row`].
    fn prefill_dense_batched(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, Vec<CudaSlice<f32>>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }
        let rows = ids.len();

        let kv_stride = self.cfg.num_kv_heads * self.cfg.head_dim;
        let total_len = start_pos + rows + extra_headroom;
        let mut k_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc k_cache: {e}"))?;
        let mut v_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc v_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.num_kv_heads != self.cfg.num_kv_heads || cache.head_dim != self.cfg.head_dim {
                return Err(format!(
                    "imported KV cache shape mismatch: file has num_kv_heads={} head_dim={}, model expects num_kv_heads={} head_dim={}",
                    cache.num_kv_heads, cache.head_dim, self.cfg.num_kv_heads, self.cfg.head_dim
                ));
            }
            if cache.k_caches.len() != self.layers.len() {
                return Err(format!("imported KV cache has {} layers, model has {}", cache.k_caches.len(), self.layers.len()));
            }
            let imported_len = cache.seq_len * kv_stride;
            for (layer_idx, (k_host, v_host)) in cache.k_caches.iter().zip(&cache.v_caches).enumerate() {
                let mut k_dst = k_caches[layer_idx].slice_mut(0..imported_len);
                self.device.htod_sync_copy_into(k_host, &mut k_dst).map_err(|e| format!("import k_cache htod layer {layer_idx}: {e}"))?;
                let mut v_dst = v_caches[layer_idx].slice_mut(0..imported_len);
                self.device.htod_sync_copy_into(v_host, &mut v_dst).map_err(|e| format!("import v_cache htod layer {layer_idx}: {e}"))?;
            }
        }

        let hidden_size = self.cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
        }
        let mut hidden = self.device.htod_sync_copy(&host_embd).map_err(|e| format!("embedding htod: {e}"))?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_batched(layer, hidden, start_pos, rows, &mut k_caches[layer_idx], &mut v_caches[layer_idx])?;
        }

        Ok((ids, hidden, k_caches, v_caches, start_pos + rows))
    }

    /// Copies row `rows - 1` (the last prompt position) out of a batched
    /// `[rows, hidden_size]` prefill output (`Self::prefill_dense_batched`)
    /// into its own owned buffer. Every current caller only wants that row
    /// to continue generation/scoring from, but needs it as an owned
    /// `CudaSlice` rather than a borrowed view, since callers go on to
    /// reassign it from `Self::forward_one_token_dense`'s per-token decode
    /// loop.
    fn last_row(&self, hidden_batched: &CudaSlice<f32>, rows: usize, hidden_size: usize) -> Result<CudaSlice<f32>, String> {
        let offset = (rows - 1) * hidden_size;
        let mut out = self.device.alloc_zeros::<f32>(hidden_size).map_err(|e| format!("last_row alloc: {e}"))?;
        let src = hidden_batched.slice(offset..offset + hidden_size);
        self.device.dtod_copy(&src, &mut out).map_err(|e| format!("last_row dtod: {e}"))?;
        Ok(out)
    }

    /// Like [`Self::last_row`], but for any row index -- used by the hybrid
    /// model's layer-major batched prefill (`Self::forward_hybrid_layer_batched`)
    /// to pull one position's hidden vector out of a `[rows, hidden_size]`
    /// buffer before feeding it through a `GatedDeltaNet` layer's sequential
    /// per-token recurrence (`Self::forward_gdn_mixer` takes ownership of a
    /// single-row `CudaSlice`, not a view into a larger batch).
    fn extract_row(&self, batched: &CudaSlice<f32>, row: usize, hidden_size: usize) -> Result<CudaSlice<f32>, String> {
        let offset = row * hidden_size;
        let mut out = self.device.alloc_zeros::<f32>(hidden_size).map_err(|e| format!("extract_row alloc: {e}"))?;
        let src = batched.slice(offset..offset + hidden_size);
        self.device.dtod_copy(&src, &mut out).map_err(|e| format!("extract_row dtod: {e}"))?;
        Ok(out)
    }

    /// Inverse of [`Self::extract_row`]: writes `src` (one position's hidden
    /// vector) back into row `row` of a `[rows, hidden_size]` buffer -- same
    /// device-to-device copy convention `Self::forward_attn_block` already
    /// uses for kv-cache writes, never a host round trip.
    fn write_row(&self, batched: &mut CudaSlice<f32>, row: usize, hidden_size: usize, src: &CudaSlice<f32>) -> Result<(), String> {
        let offset = row * hidden_size;
        let mut dst = batched.slice_mut(offset..offset + hidden_size);
        self.device.dtod_copy(src, &mut dst).map_err(|e| format!("write_row dtod: {e}"))
    }

    /// Shared dense/MoE implementation behind `forward_prompt`,
    /// `forward_prompt_capture_kv`, and `generate` (Phase 3 round 2):
    /// decodes new tokens one at a time (via `Self::prefill_dense` for the
    /// prompt, then the same `forward_one_token_dense`/`lm_head_argmax` pair
    /// per generated token) until `max_new_tokens` have been produced or
    /// `eos_token_id` comes up. Returns the generated token ids, their
    /// concatenated decoded text, the final per-layer K/V caches (still
    /// device-resident, sized with headroom for up to `max_new_tokens`
    /// generated positions -- callers downloading them for export must
    /// slice to `0..seq_len * kv_stride`, not the whole buffer), and the
    /// total sequence length reached.
    fn generate_dense_impl(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        max_new_tokens: usize,
        mut on_first_token: impl FnMut(&[f32]),
    ) -> Result<(Vec<u32>, String, Vec<CudaSlice<f32>>, Vec<CudaSlice<f32>>, usize), String> {
        if max_new_tokens == 0 {
            return Err("max_new_tokens must be at least 1".to_string());
        }

        let (ids, hidden_batched, mut k_caches, mut v_caches, mut position) =
            self.prefill_dense_batched(prompt, imported, max_new_tokens)?;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), self.cfg.hidden_size)?;

        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits = self.lm_head_logits(&hidden, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
        let mut next_id = Self::argmax(&first_logits)?;
        on_first_token(&first_logits);
        generated.push(next_id);

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_dense(next_id, position, &mut k_caches, &mut v_caches)?;
            position += 1;
            next_id = self.lm_head_argmax(&hidden, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
            generated.push(next_id);
        }

        let text = self.tokenizer.decode(&generated);
        Ok((generated, text, k_caches, v_caches, position))
    }

    /// Resolves `candidate`'s actual continuation token ids given `prompt`,
    /// by tokenizing `prompt` and `prompt + candidate` together and diffing
    /// -- tokenizing `candidate` alone does not reliably give the token(s)
    /// the model would actually emit as a continuation, since BPE/
    /// SentencePiece merge boundaries depend on what precedes the candidate
    /// text. Errs if `encode(prompt)` is not an exact prefix of
    /// `encode(prompt + candidate)`, or the candidate contributes zero new
    /// tokens.
    fn resolve_candidate_token_ids(&self, prompt: &str, candidate: &str) -> Result<Vec<u32>, String> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        let full_ids = self.tokenizer.encode(&format!("{prompt}{candidate}"))?;
        if full_ids.len() <= prompt_ids.len() || full_ids[..prompt_ids.len()] != prompt_ids[..] {
            return Err(format!("system1: candidate {candidate:?} does not tokenize as a clean continuation of the prompt"));
        }
        Ok(full_ids[prompt_ids.len()..].to_vec())
    }

    /// System1: single-pass, non-autoregressive candidate scoring. Runs
    /// `prompt` through the same prefill path as `generate_dense_impl`
    /// (`Self::prefill_dense`) exactly once, then scores every one of
    /// `candidates` from that single prefill -- no argmax-then-feedback
    /// decode loop for single-token candidates (one batched gather-GEMV
    /// covers all of them, `Self::gemv_gather`), and only a short
    /// teacher-forced continuation for multi-token ones (feeding each
    /// candidate's own known next token, never a sampled one).
    ///
    /// `score` is relative to this candidate set only, not a vocab-wide
    /// log-probability -- computing the latter for a single-token candidate
    /// would require the exact full-vocab GEMV + D2H transfer this method
    /// exists to avoid. `temperature` (`1.0` = no-op) is passed through to
    /// `crate::calibration::softmax_scores_with_temperature` to produce
    /// `System1Response::probabilities`.
    ///
    /// Dense/MoE Qwen3 models only as of this version -- errs if
    /// `self.hybrid`/`self.mla` is set. Every candidate must resolve to at
    /// least one token (see `Self::resolve_candidate_token_ids`); a
    /// resolution failure for one candidate fails the whole call.
    pub fn system1_evaluate(&self, prompt: &str, candidates: &[System1Candidate], temperature: f32) -> Result<System1Response, String> {
        if self.hybrid.is_some() || self.mla.is_some() {
            return Err("system1_evaluate: only dense/MoE Qwen3 models are supported in this version".to_string());
        }
        if candidates.is_empty() {
            return Err("system1_evaluate: candidates must not be empty".to_string());
        }

        let resolved: Vec<Vec<u32>> =
            candidates.iter().map(|c| self.resolve_candidate_token_ids(prompt, &c.text)).collect::<Result<_, _>>()?;
        let max_len = resolved.iter().map(Vec::len).max().unwrap_or(1);

        let (ids, hidden_batched, mut k_caches, mut v_caches, base_position) =
            self.prefill_dense_batched(prompt, None, max_len.saturating_sub(1))?;
        let hidden = self.last_row(&hidden_batched, ids.len(), self.cfg.hidden_size)?;

        // Batched first-token gather: the sub-50ms win for the common
        // single-token case (Yes/No, A-D, a 1-10 scale).
        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemv_gather(&normed, &self.lm_head, &first_tokens)?;

        // Multi-token candidates: teacher-forced continuation, reusing the
        // shared post-prompt KV headroom sequentially per candidate (safe
        // since each candidate is scored to completion before the next one
        // starts).
        for (i, ids) in resolved.iter().enumerate() {
            if ids.len() < 2 {
                continue;
            }
            let mut position = base_position;
            for w in ids.windows(2) {
                let (prev, next) = (w[0], w[1]);
                let h = self.forward_one_token_dense(prev, position, &mut k_caches, &mut v_caches)?;
                position += 1;
                let normed_step = self.rmsnorm(&h, &self.output_norm.data, 1, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
                scores[i] += self.gemv_gather(&normed_step, &self.lm_head, &[next])?[0];
            }
        }

        let probabilities = crate::calibration::softmax_scores_with_temperature(&scores, temperature)?;
        let entropy = crate::calibration::shannon_entropy(&probabilities)?;
        let results = candidates
            .iter()
            .zip(resolved)
            .zip(scores)
            .map(|((c, token_ids), score)| System1CandidateResult { text: c.text.clone(), token_ids, score })
            .collect();
        Ok(System1Response { results, probabilities, entropy })
    }

    /// Embeds `token_id` and runs it through every dense/MoE layer at
    /// absolute `position`, writing this position's K/V into `k_caches`/
    /// `v_caches` (preallocated device buffers, see `generate_dense_impl`).
    fn forward_one_token_dense(
        &self,
        token_id: u32,
        position: usize,
        k_caches: &mut [CudaSlice<f32>],
        v_caches: &mut [CudaSlice<f32>],
    ) -> Result<CudaSlice<f32>, String> {
        let hidden_size = self.cfg.hidden_size;
        let embd_base = token_id as usize * hidden_size;
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd[embd_base..embd_base + hidden_size])
            .map_err(|e| format!("embedding htod: {e}"))?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer(layer, hidden, position, &mut k_caches[layer_idx], &mut v_caches[layer_idx])?;
        }
        Ok(hidden)
    }

    /// Final RMSNorm -> LM head -> argmax, shared by every architecture's
    /// generation loop (`hidden_size`/`eps` differ by architecture; the
    /// `output_norm`/`lm_head` weights are shared across all of them).
    fn lm_head_argmax(&self, hidden: &CudaSlice<f32>, hidden_size: usize, eps: f32) -> Result<u32, String> {
        let logits = self.lm_head_logits(hidden, hidden_size, eps)?;
        Self::argmax(&logits)
    }

    /// Like [`Self::lm_head_argmax`], but returns the full host-resident logits
    /// vector instead of collapsing it to an argmax index -- used by the first
    /// generated token's step only (see `Self::generate_dense_impl`/
    /// `generate_hybrid_impl`/`generate_mla_impl`'s `on_first_token` call sites),
    /// so `Model::generate`'s callers (e.g. `check_correctness`, see
    /// `src/bin/check_correctness.rs`) can inspect the real logits a byte-exact
    /// verification needs without adding a second full generation API.
    fn lm_head_logits(&self, hidden: &CudaSlice<f32>, hidden_size: usize, eps: f32) -> Result<Vec<f32>, String> {
        let normed = self.rmsnorm(hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits_dev = self.gemv(&normed, &self.lm_head)?;
        self.device.dtoh_sync_copy(&logits_dev).map_err(|e| format!("logits dtoh: {e}"))
    }

    fn argmax(logits: &[f32]) -> Result<u32, String> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "cannot argmax an empty logits slice".to_string())
    }

    /// Dense/MoE-only: runs the same forward pass as `forward_prompt` but
    /// also downloads the per-layer K/V caches (sliced to exactly the
    /// positions actually written -- `generate_dense_impl`'s buffers carry
    /// extra headroom this capture doesn't use) to host memory for
    /// `--export-kv` to serialize. Hybrid models use
    /// `forward_prompt_capture_kv_hybrid` instead, MLA models
    /// `forward_prompt_capture_kv_mla`.
    pub fn forward_prompt_capture_kv(&self, prompt: &str) -> Result<((u32, String), crate::kv_io::DenseKvCache), String> {
        if self.hybrid.is_some() {
            return Err("--export-kv on a hybrid Qwen3.5 model needs forward_prompt_capture_kv_hybrid, not this function".to_string());
        }
        if self.mla.is_some() {
            return Err("--export-kv on an MLA model needs forward_prompt_capture_kv_mla, not this function".to_string());
        }
        let (generated, text, k_caches, v_caches, seq_len) = self.generate_dense_impl(prompt, None, 1, |_logits| {})?;

        let kv_stride = self.cfg.num_kv_heads * self.cfg.head_dim;
        let per_layer_len = seq_len * kv_stride;
        let k_caches = k_caches
            .iter()
            .map(|c| self.device.dtoh_sync_copy(&c.slice(0..per_layer_len)).map_err(|e| format!("k_cache dtoh: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let v_caches = v_caches
            .iter()
            .map(|c| self.device.dtoh_sync_copy(&c.slice(0..per_layer_len)).map_err(|e| format!("v_cache dtoh: {e}")))
            .collect::<Result<Vec<_>, _>>()?;

        let cache = crate::kv_io::DenseKvCache {
            seq_len,
            num_kv_heads: self.cfg.num_kv_heads,
            head_dim: self.cfg.head_dim,
            k_caches,
            v_caches,
        };
        Ok(((generated[0], text), cache))
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

    /// Batched-prefill variant of [`Self::forward_gated_attn_mixer`]: normalizes,
    /// projects, RoPEs, and attends over `rows` positions at once (`Self::gemm`/
    /// `Self::rope_batch`/`Self::attention_prefill`, same shape as
    /// `Self::forward_attn_block_batched`), plus the two extra steps this mixer
    /// needs beyond dense's attention block -- splitting the fused query+gate
    /// projection and post-attention sigmoid gating -- done via the
    /// `Self::split_qg_k`/`Self::sigmoid_gate_k` kernels instead of a per-row
    /// host round trip. `start_pos` is this batch's first row's absolute
    /// position (row `r` is `start_pos + r`), matching
    /// `Self::forward_attn_block_batched`'s resume convention.
    #[allow(clippy::too_many_arguments)]
    fn forward_gated_attn_mixer_batched(
        &self,
        h: &HybridModel,
        w: &GatedAttnLayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &h.attn_cfg;
        let normed = self.rmsnorm(&hidden, &w.attn_norm.data, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let qg = self.gemm(&normed, &w.attn_q, rows)?;
        let q_elems = rows * cfg.num_q_heads * cfg.head_dim;
        let mut q =
            self.device.alloc_zeros::<f32>(q_elems).map_err(|e| format!("gated-attn-batched q alloc: {e}"))?;
        let mut gate =
            self.device.alloc_zeros::<f32>(q_elems).map_err(|e| format!("gated-attn-batched gate alloc: {e}"))?;
        {
            let threads = 256u32;
            let blocks = (q_elems as u32).div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                self.split_qg_k
                    .function
                    .clone()
                    .launch(launch_cfg, (&qg, &mut q, &mut gate, cfg.num_q_heads as u32, cfg.head_dim as u32, rows as u32))
                    .map_err(|e| format!("split_qg launch: {e}"))?;
            }
        }

        let mut k = self.gemm(&normed, &w.attn_k, rows)?;
        let v = self.gemm(&normed, &w.attn_v, rows)?;

        q = self.rmsnorm(&q, &w.attn_q_norm.data, rows * cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        k = self.rmsnorm(&k, &w.attn_k_norm.data, rows * cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;

        self.rope_batch(&mut q, start_pos, cfg.num_q_heads, cfg.head_dim, cfg.rotary_dim, rows, cfg.rope_base)?;
        self.rope_batch(&mut k, start_pos, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim, rows, cfg.rope_base)?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = start_pos * kv_stride;
        let write_len = rows * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + write_len);
            self.device.dtod_copy(&k, &mut dst).map_err(|e| format!("gated-attn-batched kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + write_len);
            self.device.dtod_copy(&v, &mut dst).map_err(|e| format!("gated-attn-batched kv-cache dtod v: {e}"))?;
        }
        let seq_len = start_pos + rows;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let mut attn_out =
            self.attention_prefill(&q, &k_view, &v_view, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, start_pos, rows)?;

        {
            let n = q_elems as u32;
            let threads = 256u32;
            let blocks = n.div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                self.sigmoid_gate_k
                    .function
                    .clone()
                    .launch(launch_cfg, (&mut attn_out, &gate, n))
                    .map_err(|e| format!("sigmoid_gate launch: {e}"))?;
            }
        }

        let o_proj = self.gemm(&attn_out, &w.attn_output, rows)?;
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

    /// Batched-prefill variant of [`Self::forward_hybrid_ffn`]: identical
    /// SwiGLU shape, `Self::gemv`->`Self::gemm(..., rows)` and RMSNorm's
    /// row count `1`->`rows`, same as `Self::forward_layer_dense_batched`'s
    /// FFN tail. Used for the `GatedAttention` sublayer only -- `GatedDeltaNet`
    /// still calls the unbatched `Self::forward_hybrid_ffn` once per row (see
    /// `Self::forward_hybrid_layer_batched`).
    #[allow(clippy::too_many_arguments)]
    fn forward_hybrid_ffn_batched(
        &self,
        mut post_mixer: CudaSlice<f32>,
        norm: &Weight,
        ffn_gate: &Weight,
        ffn_up: &Weight,
        ffn_down: &Weight,
        hidden_size: usize,
        ffn_hidden_size: usize,
        rows: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let normed = self.rmsnorm(&post_mixer, &norm.data, rows, hidden_size, eps)?;
        let gate = self.gemm(&normed, ffn_gate, rows)?;
        let up = self.gemm(&normed, ffn_up, rows)?;
        let activated = self.silu_and_mul(&gate, &up, rows * ffn_hidden_size)?;
        let down = self.gemm(&activated, ffn_down, rows)?;
        self.add_inplace(&mut post_mixer, &down)?;
        Ok(post_mixer)
    }

    /// Layer-major dispatcher for hybrid batched prefill (`Self::prefill_hybrid_batched`):
    /// runs all `rows` prompt positions through one layer at once, before the
    /// next layer sees any of them (unlike `Self::forward_one_token_hybrid`'s
    /// token-major loop, which runs one position through every layer before
    /// the next position). Valid because a layer's output at position `p`
    /// depends only on position `p`'s input plus that layer's own carried
    /// state (`k_cache`/`v_cache` or `conv_state`/`recurrent`), never on
    /// another position's intermediate value at the same layer -- the same
    /// reassociation `Self::prefill_dense_batched` already relies on.
    /// `GatedAttention` batches its mixer + FFN over all `rows` in one GEMM
    /// pass each (`Self::forward_gated_attn_mixer_batched`/
    /// `Self::forward_hybrid_ffn_batched`). `GatedDeltaNet` is a real
    /// recurrence (`conv_state`/`recurrent` carry position-to-position
    /// dependencies) and is **not** batched -- it loops `rows` times over the
    /// *unmodified* `Self::forward_gdn_mixer`/`Self::forward_hybrid_ffn`,
    /// extracting/writing one row at a time (`Self::extract_row`/`Self::write_row`)
    /// from the shared `[rows, hidden_size]` buffer. Total GDN work is
    /// unchanged from the token-major loop, just grouped by layer instead of
    /// interleaved.
    fn forward_hybrid_layer_batched(
        &self,
        h: &HybridModel,
        layer: &HybridLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        state: &mut HybridLayerState,
    ) -> Result<CudaSlice<f32>, String> {
        let hidden_size = h.attn_cfg.hidden_size;
        let ffn_hidden_size = h.attn_cfg.ffn_hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;

        match (layer, state) {
            (HybridLayerWeights::GatedAttention(w), HybridLayerState::Attn { k_cache, v_cache }) => {
                let post_mixer = self.forward_gated_attn_mixer_batched(h, w, hidden, start_pos, rows, k_cache, v_cache)?;
                self.forward_hybrid_ffn_batched(
                    post_mixer,
                    &w.post_attn_norm,
                    &w.ffn_gate,
                    &w.ffn_up,
                    &w.ffn_down,
                    hidden_size,
                    ffn_hidden_size,
                    rows,
                    eps,
                )
            }
            (HybridLayerWeights::GatedDeltaNet(w), HybridLayerState::Gdn { conv_state, recurrent }) => {
                let mut out = hidden;
                for row in 0..rows {
                    let row_hidden = self.extract_row(&out, row, hidden_size)?;
                    let post_mixer = self.forward_gdn_mixer(h, w, row_hidden, conv_state, recurrent)?;
                    let row_out =
                        self.forward_hybrid_ffn(post_mixer, &w.post_attn_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down, hidden_size, ffn_hidden_size, eps)?;
                    self.write_row(&mut out, row, hidden_size, &row_out)?;
                }
                Ok(out)
            }
            _ => Err("internal error: hybrid layer/state kind mismatch".to_string()),
        }
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
        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn(m, yarn, &mut k_pe, 1, qk_rope, qk_rope, position, cfg.rope_base)?,
            None => self.rope_norm(m, &mut k_pe, 1, qk_rope, qk_rope, position, cfg.rope_base)?,
        }

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
        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn(m, yarn, &mut q_pe, n_head, qk_rope, qk_rope, position, cfg.rope_base)?,
            None => self.rope_norm(m, &mut q_pe, n_head, qk_rope, qk_rope, position, cfg.rope_base)?,
        }

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
        // qk_dim -- see `Self::mla_attention`'s doc comment. YaRN adjusts this via
        // its own precomputed mscale^2/sqrt(...) (see `MlaYarnConfig`).
        let scale = match &cfg.yarn {
            Some(yarn) => yarn.attention_scale,
            None => 1.0 / (n_embd_head_k_mla as f32).sqrt(),
        };
        let compressed_out = self.mla_attention(m, &qcur, &kv_view, n_head, qk_dim, v_dim, seq_len, scale)?;

        let decompressed = self.gemv_per_head(&compressed_out, &w.wv_b, n_head)?;
        let o_proj = self.gemv(&decompressed, &w.wo)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Batched-prefill variant of [`Self::forward_mla_attn_block`]: normalizes,
    /// projects, RoPEs, absorbs/decompresses, and attends over `rows` positions at
    /// once instead of one position per call. `wq`/`wkv_a_mqa`/`wo` (the dominant
    /// FLOP cost, same role QKV/O-proj play in the dense path) become one
    /// [`Self::gemm`] call each over all `rows` rows. Absorption (`wk_b`) and
    /// decompression (`wv_b`) are NOT left as a per-row loop over the sequential
    /// per-head calls: looping `Self::gemv_per_head`/`Self::gemv_view` `rows` times
    /// would mean `rows * n_head` kernel launches for absorption alone (plus as many
    /// device-to-device copies), the same order of magnitude as the exact
    /// per-call-overhead regression CLAUDE.md documents for Phase 2 round 1 -- e.g.
    /// ~14k launches per layer at a 449-row prefill with 16 heads, ~28k counting
    /// decompression too. [`Self::gemv_per_head_batch`] does both in one launch each
    /// instead. The three small `Self::mla_extract_batch`/`Self::mla_concat_qcur_batch`/
    /// `Self::mla_write_kv_cache_batch` helpers replace the sequential path's
    /// per-head/per-row `dtod_copy` loops the same way, each in one launch. `start_pos`
    /// is this batch's first row's absolute position (row `r` is `start_pos + r`),
    /// matching [`Self::forward_attn_block_batched`]'s resume convention.
    fn forward_mla_attn_block_batched(
        &self,
        m: &MlaModel,
        w: &MlaLayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
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

        let normed = self.rmsnorm(&hidden, &w.attn_norm.data, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;

        // q_batched: [rows, n_head, n_embd_head_k_mla] flat (plain gemm -- is_lite
        // path, no Q-LoRA).
        let q_batched = self.gemm(&normed, &w.wq, rows)?;

        // kv_cmpr_pe_batched: [rows, kv_lora_rank + qk_rope_head_dim] flat (single
        // shared "head" per row).
        let kv_cmpr_pe_batched = self.gemm(&normed, &w.wkv_a_mqa, rows)?;

        // Extract k_pe/kv_cmpr into their own contiguous [rows, 1, width] buffers
        // (num_heads=1: the whole fused wkv_a_mqa row is treated as a single head).
        let mut k_pe_batched = self.mla_extract_batch(m, &kv_cmpr_pe_batched, rows, 1, kv_lora + qk_rope, qk_rope, kv_lora)?;
        let kv_cmpr_batched = self.mla_extract_batch(m, &kv_cmpr_pe_batched, rows, 1, kv_lora + qk_rope, kv_lora, 0)?;

        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn_batch(m, yarn, &mut k_pe_batched, 1, qk_rope, qk_rope, start_pos, rows, cfg.rope_base)?,
            None => self.rope_norm_batch(m, &mut k_pe_batched, 1, qk_rope, qk_rope, start_pos, rows, cfg.rope_base)?,
        }

        let kv_cmpr_normed_batched = self.rmsnorm(&kv_cmpr_batched, &w.attn_kv_a_norm.data, rows, kv_lora, cfg.rmsnorm_eps)?;

        // Extract q_pe (all heads, all rows) into its own contiguous
        // [rows, n_head, qk_rope] buffer before RoPE -- q_pe is a strided sub-slice
        // of each head's [n_embd_head_k_mla]-wide row in q_batched, not itself
        // contiguous across heads.
        let mut q_pe_batched = self.mla_extract_batch(m, &q_batched, rows, n_head, n_embd_head_k_mla, qk_rope, qk_nope)?;
        match &cfg.yarn {
            Some(yarn) => {
                self.rope_norm_yarn_batch(m, yarn, &mut q_pe_batched, n_head, qk_rope, qk_rope, start_pos, rows, cfg.rope_base)?
            }
            None => self.rope_norm_batch(m, &mut q_pe_batched, n_head, qk_rope, qk_rope, start_pos, rows, cfg.rope_base)?,
        }

        // Absorption: q_nope (read directly out of q_batched via strides -- no
        // separate gather) times wk_b's per-head slice, batched over every row and
        // head in one launch.
        let absorbed_batched = self.gemv_per_head_batch(
            m,
            &q_batched,
            &w.wk_b,
            rows,
            n_head,
            n_head * n_embd_head_k_mla,
            n_embd_head_k_mla,
            0,
        )?;

        // Qcur = absorbed (nope, now in compressed kv_lora space) ++ q_pe (roped),
        // per head, per row.
        let qcur_batched = self.mla_concat_qcur_batch(m, &absorbed_batched, &q_pe_batched, rows, n_head, kv_lora, qk_rope)?;

        // Write this batch's compressed Kcur into the preallocated per-layer cache.
        self.mla_write_kv_cache_batch(m, kv_cache, &kv_cmpr_normed_batched, &k_pe_batched, start_pos, rows, kv_lora, qk_rope)?;
        let seq_len = start_pos + rows;

        let kv_view = kv_cache.slice(0..seq_len * qk_dim);
        let scale = match &cfg.yarn {
            Some(yarn) => yarn.attention_scale,
            None => 1.0 / (n_embd_head_k_mla as f32).sqrt(),
        };
        let compressed_out_batched = self.mla_attention_prefill(m, &qcur_batched, &kv_view, n_head, qk_dim, v_dim, start_pos, rows, scale)?;

        // Decompression: already-contiguous [rows, n_head, v_dim] input, standard
        // strides.
        let decompressed_batched =
            self.gemv_per_head_batch(m, &compressed_out_batched, &w.wv_b, rows, n_head, n_head * v_dim, v_dim, 0)?;

        let o_proj = self.gemm(&decompressed_batched, &w.wo, rows)?;
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Layer dispatcher for MLA batched prefill (`Self::prefill_mla_batched`): runs
    /// the attention block batched (`Self::forward_mla_attn_block_batched`), then the
    /// FFN tail. The dense-lead layers' FFN batches too (`Self::forward_hybrid_ffn_batched`,
    /// already generic over which norm/gate/up/down weights it's given -- see
    /// `MlaLayerWeights`'s doc comment). The routed-MoE + shared-expert tail batches
    /// too, via [`Self::forward_mla_moe_ffn_batched`] (grouped-GEMM MoE batching, the
    /// same `Self::moe_ffn_grouped` core `Self::forward_layer_moe_batched` uses for
    /// dense/MoE's own MoE FFN) instead of a per-row loop over the unmodified
    /// per-token `Self::forward_mla_moe_ffn`.
    fn forward_mla_layer_batched(
        &self,
        m: &MlaModel,
        layer: &MlaLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        let post_attn = self.forward_mla_attn_block_batched(m, layer, hidden, start_pos, rows, kv_cache)?;

        let cfg = &m.cfg;
        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;

        match &layer.ffn {
            MlaFfn::Dense { ffn_gate, ffn_up, ffn_down } => {
                self.forward_hybrid_ffn_batched(post_attn, &layer.ffn_norm, ffn_gate, ffn_up, ffn_down, hidden_size, ffn_hidden_size, rows, eps)
            }
            MlaFfn::Moe { .. } => {
                let moe_cfg = cfg.moe.as_ref().ok_or("internal error: MlaFfn::Moe layer but MlaConfig::moe is None")?;
                self.forward_mla_moe_ffn_batched(layer, post_attn, hidden_size, rows, moe_cfg, eps)
            }
        }
    }

    /// Hybrid-model counterpart to [`Self::forward_prompt`]: thin wrapper
    /// over [`Self::generate_hybrid_impl`] with no import and exactly one
    /// generated token.
    fn forward_prompt_hybrid(&self, h: &HybridModel, prompt: &str) -> Result<(u32, String), String> {
        let (generated, text, _states, _seq_len) = self.generate_hybrid_impl(h, prompt, None, 1, |_logits| {})?;
        Ok((generated[0], text))
    }

    /// Shared per-layer state allocation/import behind [`Self::prefill_hybrid`]
    /// and [`Self::prefill_hybrid_batched`]: allocates each layer's
    /// `HybridLayerState` (`GatedAttention`'s `k_cache`/`v_cache` sized for
    /// `start_pos + rows + extra_headroom` positions, `GatedDeltaNet`'s
    /// fixed-size `conv_state`/`recurrent`) and seeds it from `imported` when
    /// resuming -- identical between the sequential and batched prefill paths,
    /// so factored out once rather than duplicated.
    fn alloc_hybrid_states(
        &self,
        h: &HybridModel,
        imported: Option<&crate::kv_io::HybridKvCache>,
        start_pos: usize,
        rows: usize,
        extra_headroom: usize,
    ) -> Result<Vec<HybridLayerState>, String> {
        if let Some(cache) = imported {
            if cache.attn_num_kv_heads != h.attn_cfg.num_kv_heads || cache.attn_head_dim != h.attn_cfg.head_dim {
                return Err("imported hybrid KV cache's GatedAttention shape doesn't match this model".to_string());
            }
            if cache.layers.len() != h.layers.len() {
                return Err(format!("imported hybrid KV cache has {} layers, model has {}", cache.layers.len(), h.layers.len()));
            }
        }

        let attn_kv_cache_len = (start_pos + rows + extra_headroom) * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
        h.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, l)| -> Result<HybridLayerState, String> {
                let imported_layer = imported.map(|c| &c.layers[layer_idx]);
                match l {
                    HybridLayerWeights::GatedAttention(_) => {
                        let mut k_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| format!("alloc k_cache: {e}"))?;
                        let mut v_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| format!("alloc v_cache: {e}"))?;
                        if let Some(crate::kv_io::HybridLayerCacheData::Attn { k_cache: k_host, v_cache: v_host }) = imported_layer {
                            let imported_len = start_pos * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
                            let mut k_dst = k_cache.slice_mut(0..imported_len);
                            self.device.htod_sync_copy_into(k_host, &mut k_dst).map_err(|e| format!("import hybrid k_cache layer {layer_idx}: {e}"))?;
                            let mut v_dst = v_cache.slice_mut(0..imported_len);
                            self.device.htod_sync_copy_into(v_host, &mut v_dst).map_err(|e| format!("import hybrid v_cache layer {layer_idx}: {e}"))?;
                        } else if imported_layer.is_some() {
                            return Err(format!("imported hybrid KV cache layer {layer_idx} is Gdn-kind but model layer is GatedAttention"));
                        }
                        Ok(HybridLayerState::Attn { k_cache, v_cache })
                    }
                    HybridLayerWeights::GatedDeltaNet(_) => {
                        let conv_state_len = h.gdn_cfg.conv_state_len();
                        let recurrent_len = h.gdn_cfg.recurrent_len();
                        let mut conv_state = self.device.alloc_zeros::<f32>(conv_state_len).map_err(|e| format!("alloc conv_state: {e}"))?;
                        let mut recurrent = self.device.alloc_zeros::<f32>(recurrent_len).map_err(|e| format!("alloc recurrent: {e}"))?;
                        if let Some(crate::kv_io::HybridLayerCacheData::Gdn { conv_state: c_host, recurrent: r_host }) = imported_layer {
                            if c_host.len() != conv_state_len || r_host.len() != recurrent_len {
                                return Err(format!("imported hybrid KV cache layer {layer_idx} Gdn state size mismatch"));
                            }
                            self.device.htod_sync_copy_into(c_host, &mut conv_state).map_err(|e| format!("import gdn conv_state layer {layer_idx}: {e}"))?;
                            self.device.htod_sync_copy_into(r_host, &mut recurrent).map_err(|e| format!("import gdn recurrent layer {layer_idx}: {e}"))?;
                        } else if imported_layer.is_some() {
                            return Err(format!("imported hybrid KV cache layer {layer_idx} is Attn-kind but model layer is GatedDeltaNet"));
                        }
                        Ok(HybridLayerState::Gdn { conv_state, recurrent })
                    }
                }
            })
            .collect::<Result<Vec<_>, String>>()
    }

    /// Sequential hybrid prefill: encodes `prompt`, seeds state from `imported`
    /// (see [`Self::alloc_hybrid_states`]), then runs every prompt token
    /// through every layer one position at a time via
    /// [`Self::forward_one_token_hybrid`] -- the pre-batching behavior,
    /// kept unchanged as the verification oracle for
    /// [`Self::prefill_hybrid_batched`] (`hybrid_batching_tests` below), the
    /// same role [`Self::prefill_dense`] plays for `prefill_dense_batched`.
    /// Not used by [`Self::generate_hybrid_impl`] any more (see that
    /// function's doc comment) -- kept only for the oracle role and any
    /// future direct caller.
    #[cfg(test)]
    fn prefill_hybrid(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<HybridLayerState>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        let mut states = self.alloc_hybrid_states(h, imported, start_pos, ids.len(), extra_headroom)?;

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_hybrid(h, token_id, position, &mut states)?);
            position += 1;
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        Ok((ids, hidden, states, position))
    }

    /// Batched-prefill variant of [`Self::prefill_hybrid`]: same
    /// signature/state-allocation logic (`Self::alloc_hybrid_states`), but
    /// runs every prompt token through each layer in one layer-major batched
    /// pass (`Self::forward_hybrid_layer_batched`, `rows = ids.len()`)
    /// instead of looping `forward_one_token_hybrid` once per token. Like
    /// `Self::prefill_dense_batched`, returns the *whole* `[rows,
    /// hidden_size]` batched hidden state -- callers wanting only the last
    /// prompt position must slice it out with [`Self::last_row`].
    fn prefill_hybrid_batched(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<HybridLayerState>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }
        let rows = ids.len();

        let mut states = self.alloc_hybrid_states(h, imported, start_pos, rows, extra_headroom)?;

        let hidden_size = h.attn_cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
        }
        let mut hidden = self.device.htod_sync_copy(&host_embd).map_err(|e| format!("embedding htod: {e}"))?;

        for (layer, state) in h.layers.iter().zip(states.iter_mut()) {
            hidden = self.forward_hybrid_layer_batched(h, layer, hidden, start_pos, rows, state)?;
        }

        Ok((ids, hidden, states, start_pos + rows))
    }

    /// Hybrid counterpart to [`Self::generate_dense_impl`] (Phase 3 round 2;
    /// switched to the layer-major batched prefill path in the batched-prefill
    /// round that added [`Self::prefill_hybrid_batched`], mirroring
    /// `generate_dense_impl`'s own switch to `prefill_dense_batched`): prompt
    /// positions are batched through `GatedAttention` layers and looped
    /// sequentially through `GatedDeltaNet` layers' recurrence
    /// (`Self::prefill_hybrid_batched`), then new tokens are decoded one at a
    /// time (`rows == 1`, a GEMM buys nothing there) via the unchanged
    /// per-token per-layer loop, [`Self::forward_one_token_hybrid`].
    fn generate_hybrid_impl(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        max_new_tokens: usize,
        mut on_first_token: impl FnMut(&[f32]),
    ) -> Result<(Vec<u32>, String, Vec<HybridLayerState>, usize), String> {
        if max_new_tokens == 0 {
            return Err("max_new_tokens must be at least 1".to_string());
        }

        let (ids, hidden_batched, mut states, mut position) = self.prefill_hybrid_batched(h, prompt, imported, max_new_tokens)?;
        let hidden_size = h.attn_cfg.hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), hidden_size)?;

        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits = self.lm_head_logits(&hidden, hidden_size, eps)?;
        let mut next_id = Self::argmax(&first_logits)?;
        on_first_token(&first_logits);
        generated.push(next_id);

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_hybrid(h, next_id, position, &mut states)?;
            position += 1;
            next_id = self.lm_head_argmax(&hidden, hidden_size, eps)?;
            generated.push(next_id);
        }

        let text = self.tokenizer.decode(&generated);
        Ok((generated, text, states, position))
    }

    /// Embeds `token_id` and runs it through every hybrid layer at absolute
    /// `position`, dispatching each layer to its mixer/state pair.
    fn forward_one_token_hybrid(&self, h: &HybridModel, token_id: u32, position: usize, states: &mut [HybridLayerState]) -> Result<CudaSlice<f32>, String> {
        let hidden_size = h.attn_cfg.hidden_size;
        let ffn_hidden_size = h.attn_cfg.ffn_hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let embd_base = token_id as usize * hidden_size;
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd[embd_base..embd_base + hidden_size])
            .map_err(|e| format!("embedding htod: {e}"))?;

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
        Ok(hidden)
    }

    /// Hybrid counterpart to [`Self::forward_prompt_capture_kv`]: runs the
    /// same forward pass as `forward_prompt` on a hybrid model but also
    /// downloads every layer's state (attn `k_cache`/`v_cache` sliced to
    /// exactly the positions written; GDN `conv_state`/`recurrent` in full,
    /// since they're already fixed-size) to host memory for `--export-kv`.
    pub fn forward_prompt_capture_kv_hybrid(&self, prompt: &str) -> Result<((u32, String), crate::kv_io::HybridKvCache), String> {
        let h = self.hybrid.as_ref().ok_or("forward_prompt_capture_kv_hybrid called on a non-hybrid model")?;
        let (generated, text, states, seq_len) = self.generate_hybrid_impl(h, prompt, None, 1, |_logits| {})?;

        let attn_len = seq_len * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
        let mut layers = Vec::with_capacity(states.len());
        for state in &states {
            match state {
                HybridLayerState::Attn { k_cache, v_cache } => {
                    let k_host = self.device.dtoh_sync_copy(&k_cache.slice(0..attn_len)).map_err(|e| format!("hybrid k_cache dtoh: {e}"))?;
                    let v_host = self.device.dtoh_sync_copy(&v_cache.slice(0..attn_len)).map_err(|e| format!("hybrid v_cache dtoh: {e}"))?;
                    layers.push(crate::kv_io::HybridLayerCacheData::Attn { k_cache: k_host, v_cache: v_host });
                }
                HybridLayerState::Gdn { conv_state, recurrent } => {
                    let conv_host = self.device.dtoh_sync_copy(conv_state).map_err(|e| format!("gdn conv_state dtoh: {e}"))?;
                    let rec_host = self.device.dtoh_sync_copy(recurrent).map_err(|e| format!("gdn recurrent dtoh: {e}"))?;
                    layers.push(crate::kv_io::HybridLayerCacheData::Gdn { conv_state: conv_host, recurrent: rec_host });
                }
            }
        }

        let cache = crate::kv_io::HybridKvCache {
            seq_len,
            attn_num_kv_heads: h.attn_cfg.num_kv_heads,
            attn_head_dim: h.attn_cfg.head_dim,
            gdn_conv_state_len: h.gdn_cfg.conv_state_len(),
            gdn_recurrent_len: h.gdn_cfg.recurrent_len(),
            layers,
        };
        Ok(((generated[0], text), cache))
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
    /// MLA's routed-MoE + shared-expert FFN tail (real DeepSeek-V2/V3 layers past
    /// `leading_dense_block_count` -- see `MlaFfn::Moe`). Structurally
    /// `forward_layer_moe`'s router+per-expert dispatch (`crate::moe::route_top_k_with_norm`,
    /// `Self::gemv_expert`, host-side weighted accumulate -- same "stays
    /// host-driven, small expert count, not addressed by Phase 2 round 2"
    /// convention), plus one addition real DeepSeek-V2/V3 has and Qwen3-MoE
    /// doesn't: an always-on shared expert, computed as a single dense FFN (its
    /// `ffn_{gate,up,down}_shexp` weights already fuse every shared expert into
    /// one bigger matmul -- see `MlaFfn`'s doc comment) and added to the
    /// accumulator unconditionally, not gated by the router.
    fn forward_mla_moe_ffn(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let MlaFfn::Moe { ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp } = &layer.ffn
        else {
            return Err("internal error: forward_mla_moe_ffn called on a Dense layer".to_string());
        };

        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, 1, hidden_size, eps)?;

        let router_logits_dev = self.gemv(&ffn_normed, ffn_gate_inp)?;
        let router_logits = self.device.dtoh_sync_copy(&router_logits_dev).map_err(|e| format!("mla moe router dtoh: {e}"))?;
        let routed = route_top_k_with_norm(&router_logits, moe_cfg.expert_used_count, moe_cfg.normalize_top_k)?;

        let mut ffn_out = vec![0.0f32; hidden_size];
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, moe_cfg.n_ff_exp)?;
            let down = self.gemv_expert(&activated, ffn_down_exps, expert_idx)?;
            let down_host = self.device.dtoh_sync_copy(&down).map_err(|e| format!("mla moe expert down dtoh: {e}"))?;
            for (o, d) in ffn_out.iter_mut().zip(down_host.iter()) {
                *o += weight * moe_cfg.routed_scaling_factor * d;
            }
        }

        // Always-on shared expert(s) -- a single fused dense FFN, not gated by the
        // router, added unconditionally.
        let shared_hidden_size = ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemv(&ffn_normed, ffn_gate_shexp)?;
        let shared_up = self.gemv(&ffn_normed, ffn_up_shexp)?;
        let shared_activated = self.silu_and_mul(&shared_gate, &shared_up, shared_hidden_size)?;
        let shared_down = self.gemv(&shared_activated, ffn_down_shexp)?;
        let shared_down_host = self.device.dtoh_sync_copy(&shared_down).map_err(|e| format!("mla moe shared down dtoh: {e}"))?;
        for (o, d) in ffn_out.iter_mut().zip(shared_down_host.iter()) {
            *o += d;
        }

        let ffn_out_dev = self.device.htod_sync_copy(&ffn_out).map_err(|e| format!("mla moe ffn_out htod: {e}"))?;
        self.add_inplace(&mut post_attn, &ffn_out_dev)?;
        Ok(post_attn)
    }

    /// Batched-prefill variant of [`Self::forward_mla_moe_ffn`]: the always-on shared
    /// expert has no routing (every row uses it unconditionally with the same
    /// weights), so it batches trivially with the existing [`Self::gemm`] -- no
    /// permutation needed, just `rows` instead of `1`. That shared-expert output
    /// seeds `ffn_out`; the routed experts (a different, data-dependent top-k subset
    /// per row) then batch via [`Self::moe_ffn_grouped`] (the same grouped-GEMM core
    /// [`Self::forward_layer_moe_batched`] uses for dense/MoE), scatter-adding on top
    /// of that seed instead of a zeroed buffer. `moe_cfg.routed_scaling_factor` is
    /// passed through as `moe_ffn_grouped`'s `weight_scale` so it's folded into each
    /// assignment's weight before the scatter-add, matching
    /// `Self::forward_mla_moe_ffn`'s unbatched `weight * moe_cfg.routed_scaling_factor`.
    fn forward_mla_moe_ffn_batched(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        rows: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        let MlaFfn::Moe { ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp } = &layer.ffn
        else {
            return Err("internal error: forward_mla_moe_ffn_batched called on a Dense layer".to_string());
        };

        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, rows, hidden_size, eps)?;

        // Always-on shared expert(s), batched across every row with no routing --
        // seeds ffn_out; the routed experts below accumulate `+=` on top of it.
        let shared_hidden_size = ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemm(&ffn_normed, ffn_gate_shexp, rows)?;
        let shared_up = self.gemm(&ffn_normed, ffn_up_shexp, rows)?;
        let shared_activated = self.silu_and_mul(&shared_gate, &shared_up, rows * shared_hidden_size)?;
        let mut ffn_out = self.gemm(&shared_activated, ffn_down_shexp, rows)?;

        let router_logits_dev = self.gemm(&ffn_normed, ffn_gate_inp, rows)?;
        let router_logits = self.device.dtoh_sync_copy(&router_logits_dev).map_err(|e| format!("mla moe router dtoh: {e}"))?;
        let num_experts = router_logits.len() / rows;

        self.moe_ffn_grouped(
            &ffn_normed,
            rows,
            hidden_size,
            &router_logits,
            num_experts,
            moe_cfg.expert_used_count,
            moe_cfg.normalize_top_k,
            moe_cfg.routed_scaling_factor,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            &mut ffn_out,
        )?;

        self.add_inplace(&mut post_attn, &ffn_out)?;
        Ok(post_attn)
    }

    /// MLA-model counterpart to [`Self::forward_prompt_hybrid`]: thin wrapper
    /// over [`Self::generate_mla_impl`] with no import and exactly one
    /// generated token.
    fn forward_prompt_mla(&self, m: &MlaModel, prompt: &str) -> Result<(u32, String), String> {
        let (generated, text, _kv_caches, _seq_len) = self.generate_mla_impl(m, prompt, None, 1, |_logits| {})?;
        Ok((generated[0], text))
    }

    /// Shared per-layer `kv_cache` allocation/import behind [`Self::prefill_mla`] and
    /// [`Self::prefill_mla_batched`]: allocates each layer's compressed `[total_len,
    /// qk_dim]` cache (`total_len = start_pos + rows + extra_headroom`) and seeds it
    /// from `imported` when resuming -- identical between the sequential and batched
    /// prefill paths, so factored out once rather than duplicated (same role
    /// `Self::alloc_hybrid_states` plays for the hybrid path).
    fn alloc_mla_kv_caches(
        &self,
        m: &MlaModel,
        imported: Option<&crate::kv_io::MlaKvCache>,
        start_pos: usize,
        rows: usize,
        extra_headroom: usize,
    ) -> Result<Vec<CudaSlice<f32>>, String> {
        let qk_dim = m.cfg.kv_lora_rank + m.cfg.qk_rope_head_dim;
        let total_len = start_pos + rows + extra_headroom;
        let mut kv_caches: Vec<CudaSlice<f32>> = (0..m.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * qk_dim))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("alloc mla kv_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.qk_dim != qk_dim {
                return Err(format!(
                    "imported KV cache shape mismatch: file has qk_dim={}, model expects qk_dim={}",
                    cache.qk_dim, qk_dim
                ));
            }
            if cache.kv_caches.len() != m.layers.len() {
                return Err(format!("imported KV cache has {} layers, model has {}", cache.kv_caches.len(), m.layers.len()));
            }
            let imported_len = cache.seq_len * qk_dim;
            for (layer_idx, kv_host) in cache.kv_caches.iter().enumerate() {
                let mut dst = kv_caches[layer_idx].slice_mut(0..imported_len);
                self.device.htod_sync_copy_into(kv_host, &mut dst).map_err(|e| format!("import mla kv_cache htod layer {layer_idx}: {e}"))?;
            }
        }
        Ok(kv_caches)
    }

    /// Sequential MLA prefill: encodes `prompt`, seeds `kv_caches` from `imported`
    /// (see [`Self::alloc_mla_kv_caches`]), then runs every prompt token through
    /// every layer one position at a time via [`Self::forward_one_token_mla`] -- the
    /// pre-batching behavior, kept unchanged as the verification oracle for
    /// [`Self::prefill_mla_batched`] (`mla_batching_tests` below), the same role
    /// [`Self::prefill_hybrid`] plays for `prefill_hybrid_batched`. Not used by
    /// [`Self::generate_mla_impl`] any more (see that function's doc comment) --
    /// kept only for the oracle role and any future direct caller.
    #[cfg(test)]
    fn prefill_mla(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }

        let mut kv_caches = self.alloc_mla_kv_caches(m, imported, start_pos, ids.len(), extra_headroom)?;

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_mla(m, token_id, position, &mut kv_caches)?);
            position += 1;
        }
        let hidden = hidden_dev.ok_or("no tokens processed")?;

        Ok((ids, hidden, kv_caches, position))
    }

    /// Batched-prefill variant of [`Self::prefill_mla`]: same signature/state-
    /// allocation logic (`Self::alloc_mla_kv_caches`), but runs every prompt token
    /// through each layer in one layer-major batched pass
    /// (`Self::forward_mla_layer_batched`, `rows = ids.len()`) instead of looping
    /// `Self::forward_one_token_mla` once per token. Like `Self::prefill_dense_batched`/
    /// `Self::prefill_hybrid_batched`, returns the *whole* `[rows, hidden_size]`
    /// batched hidden state -- callers wanting only the last prompt position must
    /// slice it out with [`Self::last_row`].
    fn prefill_mla_batched(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        extra_headroom: usize,
    ) -> Result<(Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, usize), String> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.bos_token_id {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err("encode produced no tokens".to_string());
        }
        let rows = ids.len();

        let mut kv_caches = self.alloc_mla_kv_caches(m, imported, start_pos, rows, extra_headroom)?;

        let hidden_size = m.cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            let embd_base = token_id as usize * hidden_size;
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd[embd_base..embd_base + hidden_size]);
        }
        let mut hidden = self.device.htod_sync_copy(&host_embd).map_err(|e| format!("embedding htod: {e}"))?;

        for (layer_idx, layer) in m.layers.iter().enumerate() {
            hidden = self.forward_mla_layer_batched(m, layer, hidden, start_pos, rows, &mut kv_caches[layer_idx])?;
        }

        Ok((ids, hidden, kv_caches, start_pos + rows))
    }

    /// MLA counterpart to [`Self::generate_dense_impl`]/[`Self::generate_hybrid_impl`]
    /// (Phase 3 round 3; switched to the layer-major batched prefill path in the
    /// batched-prefill round that added [`Self::prefill_mla_batched`], mirroring
    /// `generate_dense_impl`'s/`generate_hybrid_impl`'s own switch): prompt positions
    /// are batched through every layer (`Self::prefill_mla_batched`), then new
    /// tokens are decoded one at a time (`rows == 1`, a GEMM buys nothing there) via
    /// the unchanged per-token per-layer loop, [`Self::forward_one_token_mla`]. Each
    /// layer's single compressed `kv_cache` (`[seq_len, kv_lora_rank +
    /// qk_rope_head_dim]`, no separate K/V pair -- see
    /// [`Self::forward_mla_attn_block`]) gets the same `start_pos`-offset treatment
    /// dense/MoE's `k_cache`/`v_cache` and hybrid's `GatedAttention` sublayers
    /// already do.
    fn generate_mla_impl(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        max_new_tokens: usize,
        mut on_first_token: impl FnMut(&[f32]),
    ) -> Result<(Vec<u32>, String, Vec<CudaSlice<f32>>, usize), String> {
        if max_new_tokens == 0 {
            return Err("max_new_tokens must be at least 1".to_string());
        }

        let (ids, hidden_batched, mut kv_caches, mut position) = self.prefill_mla_batched(m, prompt, imported, max_new_tokens)?;
        let hidden_size = m.cfg.hidden_size;
        let eps = m.cfg.rmsnorm_eps;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), hidden_size)?;

        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits = self.lm_head_logits(&hidden, hidden_size, eps)?;
        let mut next_id = Self::argmax(&first_logits)?;
        on_first_token(&first_logits);
        generated.push(next_id);

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_mla(m, next_id, position, &mut kv_caches)?;
            position += 1;
            next_id = self.lm_head_argmax(&hidden, hidden_size, eps)?;
            generated.push(next_id);
        }

        let text = self.tokenizer.decode(&generated);
        Ok((generated, text, kv_caches, position))
    }

    /// Embeds `token_id` and runs it through every MLA layer at absolute
    /// `position`, writing this position's compressed `Kcur` into
    /// `kv_caches` (preallocated device buffers, see `generate_mla_impl`).
    fn forward_one_token_mla(&self, m: &MlaModel, token_id: u32, position: usize, kv_caches: &mut [CudaSlice<f32>]) -> Result<CudaSlice<f32>, String> {
        let cfg = &m.cfg;
        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;
        let embd_base = token_id as usize * hidden_size;
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd[embd_base..embd_base + hidden_size])
            .map_err(|e| format!("embedding htod: {e}"))?;

        for (layer_idx, layer) in m.layers.iter().enumerate() {
            let post_attn = self.forward_mla_attn_block(m, layer, hidden, position, &mut kv_caches[layer_idx])?;
            hidden = match &layer.ffn {
                MlaFfn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    self.forward_hybrid_ffn(post_attn, &layer.ffn_norm, ffn_gate, ffn_up, ffn_down, hidden_size, ffn_hidden_size, eps)?
                }
                MlaFfn::Moe { .. } => {
                    let moe_cfg = cfg.moe.as_ref().ok_or("internal error: MlaFfn::Moe layer but MlaConfig::moe is None")?;
                    self.forward_mla_moe_ffn(layer, post_attn, hidden_size, moe_cfg, eps)?
                }
            };
        }
        Ok(hidden)
    }

    /// MLA counterpart to [`Self::forward_prompt_capture_kv`]/
    /// [`Self::forward_prompt_capture_kv_hybrid`]: runs the same forward pass
    /// as `forward_prompt` on an MLA model but also downloads every layer's
    /// single compressed `kv_cache` (sliced to exactly the positions
    /// actually written -- `generate_mla_impl`'s buffers carry extra
    /// headroom this capture doesn't use) to host memory for `--export-kv`.
    pub fn forward_prompt_capture_kv_mla(&self, prompt: &str) -> Result<((u32, String), crate::kv_io::MlaKvCache), String> {
        let m = self.mla.as_ref().ok_or("forward_prompt_capture_kv_mla called on a non-MLA model")?;
        let (generated, text, kv_caches, seq_len) = self.generate_mla_impl(m, prompt, None, 1, |_logits| {})?;

        let qk_dim = m.cfg.kv_lora_rank + m.cfg.qk_rope_head_dim;
        let per_layer_len = seq_len * qk_dim;
        let kv_caches = kv_caches
            .iter()
            .map(|c| self.device.dtoh_sync_copy(&c.slice(0..per_layer_len)).map_err(|e| format!("mla kv_cache dtoh: {e}")))
            .collect::<Result<Vec<_>, _>>()?;

        let cache = crate::kv_io::MlaKvCache { seq_len, qk_dim, kv_caches };
        Ok(((generated[0], text), cache))
    }
}

#[cfg(test)]
mod prefill_batching_tests {
    use super::*;
    use crate::gguf::GgufFile;
    use cudarc::driver::CudaDevice;

    /// Byte-exact-ish cross-check of `prefill_dense_batched` (cuBLAS GEMM +
    /// batched RoPE/attention) against `prefill_dense` (the original
    /// sequential per-token loop) on the same prompt/weights -- this is the
    /// blocking check before trusting any batched-prefill latency number
    /// (cuBLAS's summation order, RoPE's per-row position math, and the
    /// batched attention kernel's causal masking are exactly the places a
    /// silently-wrong-but-non-crashing bug would hide). Compares every row
    /// of the batched hidden state against the
    /// corresponding sequential-path position, plus the final argmax token
    /// id. Real GGUF fixtures live outside this repo (gitignored), so this
    /// is `#[ignore]`d by default -- run with:
    /// `COLDSTART_TEST_GGUF=<path> cargo test --release -- --ignored prefill_dense_batched_matches_sequential_prefill`
    #[test]
    #[ignore]
    fn prefill_dense_batched_matches_sequential_prefill() {
        let gguf_path = std::env::var("COLDSTART_TEST_GGUF").expect("set COLDSTART_TEST_GGUF to a real local GGUF path to run this test");
        let prompt = "The capital of France is";

        let file = GgufFile::open(&gguf_path).expect("failed to open COLDSTART_TEST_GGUF");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load model");

        let (seq_ids, seq_hidden, _, _, seq_position) =
            model.prefill_dense(prompt, None, 0).expect("prefill_dense failed");
        let (batch_ids, batch_hidden, _, _, batch_position) =
            model.prefill_dense_batched(prompt, None, 0).expect("prefill_dense_batched failed");

        assert_eq!(seq_ids, batch_ids, "tokenization must match between the two prefill paths");
        assert_eq!(seq_position, batch_position, "final position must match between the two prefill paths");

        let rows = batch_ids.len();
        let hidden_size = model.cfg.hidden_size;
        let batch_last_row = model.last_row(&batch_hidden, rows, hidden_size).expect("last_row failed");

        let seq_host = model.device.dtoh_sync_copy(&seq_hidden).expect("seq hidden dtoh failed");
        let batch_host = model.device.dtoh_sync_copy(&batch_last_row).expect("batch hidden dtoh failed");
        assert_eq!(seq_host.len(), batch_host.len());
        for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "hidden[{i}]: sequential={a}, batched={b}");
        }

        let seq_argmax = model.lm_head_argmax(&seq_hidden, hidden_size, model.cfg.rmsnorm_eps).expect("seq argmax failed");
        let batch_argmax = model.lm_head_argmax(&batch_last_row, hidden_size, model.cfg.rmsnorm_eps).expect("batch argmax failed");
        assert_eq!(seq_argmax, batch_argmax, "greedy-argmax next token must match between the two prefill paths");
    }
}

#[cfg(test)]
mod hybrid_batching_tests {
    use super::*;
    use crate::gguf::GgufFile;
    use cudarc::driver::CudaDevice;

    /// Byte-exact-ish cross-check of `prefill_hybrid_batched` (layer-major
    /// GatedAttention batching, GatedDeltaNet left sequential -- see
    /// `Model::forward_hybrid_layer_batched`'s doc comment) against
    /// `prefill_hybrid` (the original token-major sequential loop) on the
    /// same prompt/weights -- the blocking check before trusting the batched
    /// hybrid prefill path, same role
    /// `prefill_dense_batched_matches_sequential_prefill` plays for dense/MoE.
    /// Real GGUF fixtures live outside this repo (gitignored), so this is
    /// `#[ignore]`d by default -- run with:
    /// `COLDSTART_TEST_GGUF=<path to a Qwen3.5 hybrid GGUF> cargo test --release -- --ignored prefill_hybrid_batched_matches_sequential`
    #[test]
    #[ignore]
    fn prefill_hybrid_batched_matches_sequential() {
        let gguf_path = std::env::var("COLDSTART_TEST_GGUF")
            .expect("set COLDSTART_TEST_GGUF to a real local Qwen3.5 hybrid GGUF path to run this test");
        let prompt = "The capital of France is";

        let file = GgufFile::open(&gguf_path).expect("failed to open COLDSTART_TEST_GGUF");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load_hybrid(device, &file).expect("failed to load hybrid model");
        let h = model.hybrid.as_ref().expect("loaded model is not hybrid");

        let (seq_ids, seq_hidden, _, seq_position) = model.prefill_hybrid(h, prompt, None, 0).expect("prefill_hybrid failed");
        let (batch_ids, batch_hidden, _, batch_position) =
            model.prefill_hybrid_batched(h, prompt, None, 0).expect("prefill_hybrid_batched failed");

        assert_eq!(seq_ids, batch_ids, "tokenization must match between the two prefill paths");
        assert_eq!(seq_position, batch_position, "final position must match between the two prefill paths");

        let rows = batch_ids.len();
        let hidden_size = h.attn_cfg.hidden_size;
        let batch_last_row = model.last_row(&batch_hidden, rows, hidden_size).expect("last_row failed");

        let seq_host = model.device.dtoh_sync_copy(&seq_hidden).expect("seq hidden dtoh failed");
        let batch_host = model.device.dtoh_sync_copy(&batch_last_row).expect("batch hidden dtoh failed");
        assert_eq!(seq_host.len(), batch_host.len());
        for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "hidden[{i}]: sequential={a}, batched={b}");
        }

        let eps = h.attn_cfg.rmsnorm_eps;
        let seq_argmax = model.lm_head_argmax(&seq_hidden, hidden_size, eps).expect("seq argmax failed");
        let batch_argmax = model.lm_head_argmax(&batch_last_row, hidden_size, eps).expect("batch argmax failed");
        assert_eq!(seq_argmax, batch_argmax, "greedy-argmax next token must match between the two prefill paths");
    }
}

#[cfg(test)]
mod mla_batching_tests {
    use super::*;
    use crate::gguf::GgufFile;
    use cudarc::driver::CudaDevice;

    const MLA_FIXTURE: &str = "test-data/deepseek-tiny-mla.gguf";

    /// Lowest-level sanity check before trusting the full attention-block test
    /// below: `Self::gemv_per_head_batch` at `rows=1` (a novel 3-D-grid kernel with
    /// no direct single-token analogue to diff row-by-row, unlike `Self::gemm`,
    /// which `prefill_dense_batched_matches_sequential_prefill` could check against
    /// `gemv_raw`) must reproduce `Self::gemv_per_head`'s existing, already-
    /// hardware-verified output exactly against the same real `wk_b` weight tensor.
    /// `test-data/deepseek-tiny-mla.gguf` (synthetic, hand-built via llama.cpp's
    /// real converter -- see README.md's MLA fixture section) is already local, so
    /// this doesn't need `COLDSTART_TEST_GGUF`; still `#[ignore]`d since it needs a
    /// real GPU -- run with `cargo test --release -- --ignored
    /// gemv_per_head_batch_matches_gemv_per_head_at_rows_one`.
    #[test]
    #[ignore]
    fn gemv_per_head_batch_matches_gemv_per_head_at_rows_one() {
        let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load MLA model");
        let m = model.mla.as_ref().expect("loaded model is not MLA");
        let layer = &m.layers[0];

        let n_head = m.cfg.num_heads;
        let in_features = m.cfg.qk_nope_head_dim;

        let host_x: Vec<f32> = (0..n_head * in_features).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let x = model.device.htod_sync_copy(&host_x).expect("x htod failed");

        let single = model.gemv_per_head(&x, &layer.wk_b, n_head).expect("gemv_per_head failed");
        let batched = model
            .gemv_per_head_batch(m, &x, &layer.wk_b, 1, n_head, n_head * in_features, in_features, 0)
            .expect("gemv_per_head_batch failed");

        let single_host = model.device.dtoh_sync_copy(&single).expect("single dtoh failed");
        let batched_host = model.device.dtoh_sync_copy(&batched).expect("batched dtoh failed");
        assert_eq!(single_host.len(), batched_host.len());
        for (i, (a, b)) in single_host.iter().zip(batched_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "out[{i}]: gemv_per_head={a}, gemv_per_head_batch={b}");
        }
    }

    /// Byte-exact-ish cross-check of `prefill_mla_batched` (layer-major batched
    /// attention block; grouped-GEMM-batched MoE FFN tail where present, see
    /// `Model::forward_mla_moe_ffn_batched`) against `prefill_mla` (the original
    /// token-major sequential loop, still calling the unmodified per-token
    /// `Model::forward_mla_moe_ffn`) on the same prompt/weights -- the blocking check
    /// before trusting the batched MLA prefill path, same role
    /// `prefill_hybrid_batched_matches_sequential` plays for the hybrid path. NOTE:
    /// `test-data/deepseek-tiny-mla.gguf` is dense-lead-only with no YaRN scaling (see
    /// README.md's MLA fixture section), so this test does not exercise
    /// `MlaFfn::Moe`'s grouped-GEMM path or `rope_norm_yarn_batch_kernel` -- see
    /// `prefill_mla_batched_matches_sequential_real_moe_checkpoint` below for that,
    /// which only the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint can exercise (see
    /// CLAUDE.md's "Known test-fixture limitation"). Run with `cargo test --release
    /// -- --ignored prefill_mla_batched_matches_sequential`.
    #[test]
    #[ignore]
    fn prefill_mla_batched_matches_sequential() {
        let prompt = "The capital of France is";

        let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load MLA model");
        let m = model.mla.as_ref().expect("loaded model is not MLA");

        let (seq_ids, seq_hidden, _, seq_position) = model.prefill_mla(m, prompt, None, 0).expect("prefill_mla failed");
        let (batch_ids, batch_hidden, _, batch_position) =
            model.prefill_mla_batched(m, prompt, None, 0).expect("prefill_mla_batched failed");

        assert_eq!(seq_ids, batch_ids, "tokenization must match between the two prefill paths");
        assert_eq!(seq_position, batch_position, "final position must match between the two prefill paths");

        let rows = batch_ids.len();
        let hidden_size = m.cfg.hidden_size;
        let batch_last_row = model.last_row(&batch_hidden, rows, hidden_size).expect("last_row failed");

        let seq_host = model.device.dtoh_sync_copy(&seq_hidden).expect("seq hidden dtoh failed");
        let batch_host = model.device.dtoh_sync_copy(&batch_last_row).expect("batch hidden dtoh failed");
        assert_eq!(seq_host.len(), batch_host.len());
        for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "hidden[{i}]: sequential={a}, batched={b}");
        }

        let eps = m.cfg.rmsnorm_eps;
        let seq_argmax = model.lm_head_argmax(&seq_hidden, hidden_size, eps).expect("seq argmax failed");
        let batch_argmax = model.lm_head_argmax(&batch_last_row, hidden_size, eps).expect("batch argmax failed");
        assert_eq!(seq_argmax, batch_argmax, "greedy-argmax next token must match between the two prefill paths");
    }

    /// Real-`MlaFfn::Moe` counterpart to `prefill_mla_batched_matches_sequential`
    /// above: that test's `test-data/deepseek-tiny-mla.gguf` fixture is dense-lead-only
    /// (see its own doc comment), so it never exercises
    /// `Model::forward_mla_moe_ffn_batched`'s grouped-GEMM routed-expert path or its
    /// batched shared-expert seeding -- both new this round, and both only reachable
    /// through a real `deepseek2` file's routed-MoE layers (no small synthetic
    /// `deepseek2` MoE fixture exists, see CLAUDE.md's "Known test-fixture
    /// limitation"). Same cross-check shape as the fixture-based test (byte-exact-ish
    /// hidden state plus matching greedy-argmax token), but reads its GGUF path from
    /// `COLDSTART_TEST_GGUF` (the same convention `prefill_dense_batched_matches_sequential_prefill`
    /// uses) instead of the hardcoded fixture constant, so it can point at the real
    /// `deepseek-ai/DeepSeek-V2-Lite` checkpoint (regenerate per STATUS.md's
    /// documented recipe). Run with:
    /// `COLDSTART_TEST_GGUF=<path to a real deepseek2 GGUF with MoE layers> cargo test --release -- --ignored prefill_mla_batched_matches_sequential_real_moe_checkpoint`.
    #[test]
    #[ignore]
    fn prefill_mla_batched_matches_sequential_real_moe_checkpoint() {
        let gguf_path = std::env::var("COLDSTART_TEST_GGUF")
            .expect("set COLDSTART_TEST_GGUF to a real local deepseek2 GGUF path (with MoE layers) to run this test");
        let prompt = "The capital of France is";

        let file = GgufFile::open(&gguf_path).expect("failed to open COLDSTART_TEST_GGUF");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load MLA model");
        let m = model.mla.as_ref().expect("loaded model is not MLA");
        assert!(m.cfg.moe.is_some(), "COLDSTART_TEST_GGUF must be a real deepseek2 checkpoint with MoE layers, not the dense-lead-only synthetic fixture");

        let (seq_ids, seq_hidden, _, seq_position) = model.prefill_mla(m, prompt, None, 0).expect("prefill_mla failed");
        let (batch_ids, batch_hidden, _, batch_position) =
            model.prefill_mla_batched(m, prompt, None, 0).expect("prefill_mla_batched failed");

        assert_eq!(seq_ids, batch_ids, "tokenization must match between the two prefill paths");
        assert_eq!(seq_position, batch_position, "final position must match between the two prefill paths");

        let rows = batch_ids.len();
        let hidden_size = m.cfg.hidden_size;
        let batch_last_row = model.last_row(&batch_hidden, rows, hidden_size).expect("last_row failed");

        let seq_host = model.device.dtoh_sync_copy(&seq_hidden).expect("seq hidden dtoh failed");
        let batch_host = model.device.dtoh_sync_copy(&batch_last_row).expect("batch hidden dtoh failed");
        assert_eq!(seq_host.len(), batch_host.len());
        for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "hidden[{i}]: sequential={a}, batched={b}");
        }

        let eps = m.cfg.rmsnorm_eps;
        let seq_argmax = model.lm_head_argmax(&seq_hidden, hidden_size, eps).expect("seq argmax failed");
        let batch_argmax = model.lm_head_argmax(&batch_last_row, hidden_size, eps).expect("batch argmax failed");
        assert_eq!(seq_argmax, batch_argmax, "greedy-argmax next token must match between the two prefill paths");
    }

    /// `--import-kv` resume equivalence: captures a short prompt's compressed
    /// `kv_cache` via the already-hardware-verified `forward_prompt_capture_kv_mla`,
    /// then continues generation from it through both `prefill_mla` and
    /// `prefill_mla_batched` (`start_pos > 0`) and diffs the two continuations --
    /// the same `imported.is_some()` case `prefill_hybrid_batched`'s test coverage
    /// doesn't separately exercise but this round's plan calls out explicitly (the
    /// batched KV-cache write path, `Self::mla_write_kv_cache_batch`, must offset by
    /// `start_pos` correctly, not just `0`). Run with `cargo test --release --
    /// --ignored prefill_mla_batched_import_kv_resume_matches_sequential`.
    #[test]
    #[ignore]
    fn prefill_mla_batched_import_kv_resume_matches_sequential() {
        let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load MLA model");
        let m = model.mla.as_ref().expect("loaded model is not MLA");

        let (_, cache) = model.forward_prompt_capture_kv_mla("The capital of France is").expect("capture_kv failed");

        let continuation = " Paris";
        let (seq_ids, seq_hidden, _, seq_position) =
            model.prefill_mla(m, continuation, Some(&cache), 0).expect("prefill_mla resume failed");
        let (batch_ids, batch_hidden, _, batch_position) =
            model.prefill_mla_batched(m, continuation, Some(&cache), 0).expect("prefill_mla_batched resume failed");

        assert_eq!(seq_ids, batch_ids, "tokenization must match between the two resumed prefill paths");
        assert_eq!(seq_position, batch_position, "final position must match between the two resumed prefill paths");

        let rows = batch_ids.len();
        let hidden_size = m.cfg.hidden_size;
        let batch_last_row = model.last_row(&batch_hidden, rows, hidden_size).expect("last_row failed");

        let seq_host = model.device.dtoh_sync_copy(&seq_hidden).expect("seq hidden dtoh failed");
        let batch_host = model.device.dtoh_sync_copy(&batch_last_row).expect("batch hidden dtoh failed");
        assert_eq!(seq_host.len(), batch_host.len());
        for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "resumed hidden[{i}]: sequential={a}, batched={b}");
        }
    }
}

#[cfg(test)]
mod system1_tests {
    use super::*;
    use crate::gguf::GgufFile;
    use cudarc::driver::CudaDevice;

    /// Exact cross-check of `gemv_gather` against the existing full-vocab
    /// `gemv` path: same weights, same math, different kernel -- gathering a
    /// handful of rows (including the model's own real argmax id) must
    /// agree with the corresponding entries of a full-vocab GEMV to float
    /// rounding. Real GGUF fixtures live outside this repo (`.gguf` is
    /// gitignored, per CLAUDE.md's "Known test-fixture limitation"), so this
    /// is `#[ignore]`d by default and reads its model path from
    /// `COLDSTART_TEST_GGUF` rather than guessing a local path -- run with:
    /// `COLDSTART_TEST_GGUF=<path> cargo test --release -- --ignored gemv_gather_matches_full_vocab_gemv`
    #[test]
    #[ignore]
    fn gemv_gather_matches_full_vocab_gemv_at_matching_rows() {
        let gguf_path = std::env::var("COLDSTART_TEST_GGUF").expect("set COLDSTART_TEST_GGUF to a real local GGUF path to run this test");
        let file = GgufFile::open(&gguf_path).expect("failed to open COLDSTART_TEST_GGUF");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load model");

        let (_, hidden, _, _, _) = model.prefill_dense("The capital of France is", None, 0).expect("prefill_dense failed");
        let normed = model.rmsnorm(&hidden, &model.output_norm.data, 1, model.cfg.hidden_size, model.cfg.rmsnorm_eps).expect("rmsnorm failed");

        let full = model.gemv(&normed, &model.lm_head).expect("gemv failed");
        let full_host = model.device.dtoh_sync_copy(&full).expect("dtoh failed");

        let vocab_size = model.lm_head.shape[1] as usize;
        let argmax_id = full_host
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .expect("full_host must not be empty");
        let row_indices = [0u32, (vocab_size / 2) as u32, (vocab_size - 1) as u32, argmax_id];

        let gathered = model.gemv_gather(&normed, &model.lm_head, &row_indices).expect("gemv_gather failed");

        for (j, &row) in row_indices.iter().enumerate() {
            let expected = full_host[row as usize];
            let got = gathered[j];
            assert!((expected - got).abs() < 1e-4, "row {row}: full_vocab={expected}, gathered={got}");
        }
    }
}
