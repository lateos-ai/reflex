//! Dense Qwen3 forward pass (MVP step 1, see README.md's MVP order): loads a
//! GGUF file's weights, runs embedding -> every transformer layer -> final
//! RMSNorm -> LM head -> argmax for the *first* generated token only. No KV
//! cache reuse across separate calls, no batching, no sampling beyond greedy
//! argmax -- the target metric is process-start-to-first-token latency, not
//! sustained decode throughput (see README.md's "Why this exists").
//!
//! Architecture and math ported from RustFeference's own most mature, most-
//! verified dense-model code (`rft-gpu/src/generate.rs` + `dispatch.rs` +
//! `kernels/{rmsnorm,rope,silu_and_mul}.rs`, git history around commits
//! `d8ed273` "minimal end-to-end dense forward pass" and `1459330` "Qwen3
//! architecture support") as the correctness oracle, not copied wholesale:
//! RustFeference's serving/paged-KV-cache/tensor-parallel machinery is all
//! out of scope here (see MVP scope discussion) -- kernels below are
//! deliberately fresh, simple, from-scratch AOT kernels, not ports of
//! RustFeference's own (far more complex, paged/batched/fused) CUDA source.
//!
//! GEMM convention throughout: `y = x @ W^T` (`nn.Linear`), where a real
//! GGUF weight tensor's parsed `shape` is `[in_features, out_features]`
//! (confirmed against RustFeference's own `parse_model_config`/`gemm_shape`
//! usage of the identical, unmodified `gguf.rs` parser this crate salvaged)
//! and its flat dequantized bytes are already row-major
//! `(out_features, in_features)` -- exactly `gemv_kernel`'s expected layout,
//! no transpose needed.

use crate::aot::{self, AotKernel};
use crate::dequant;
use crate::gguf::{GgufFile, GgufValue};
use crate::tokenizer::Tokenizer;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

fn u64_meta(file: &GgufFile, key: &str) -> Option<u64> {
    file.metadata.get(key).and_then(GgufValue::as_u64)
}

fn f32_meta(file: &GgufFile, key: &str) -> Option<f32> {
    file.metadata.get(key).and_then(GgufValue::as_f32)
}

/// Static shape/hyperparameter config for one dense Qwen3 transformer layer,
/// read from the GGUF file's `qwen3.*` metadata.
pub struct LayerConfig {
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Read explicitly from `qwen3.attention.key_length` -- Qwen3 decouples
    /// this from `hidden_size / num_q_heads` (real finding from
    /// RustFeference's own Phase 21.14: a real Qwen3-0.6B has head_dim=128,
    /// not the 64 that division would give).
    pub head_dim: usize,
    pub ffn_hidden_size: usize,
    pub rope_base: f32,
    pub rmsnorm_eps: f32,
}

/// A dequantized weight tensor plus its original GGUF shape
/// (`[in_features, out_features]` for a 2-D `nn.Linear`-style weight,
/// `[hidden_size]` for a norm weight, `[hidden_size, vocab_size]` for the
/// token embedding table).
struct Weight {
    data: Vec<f32>,
    shape: Vec<u64>,
}

struct LayerWeights {
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

/// Derive [`LayerConfig`] and the layer count from a GGUF file's `qwen3.*`
/// metadata (plus `blk.0.ffn_gate.weight`'s real shape, since GGUF has no
/// dedicated "FFN hidden size" metadata key). Pure/host-only, no GPU
/// required. Dense `qwen3` only -- `qwen3moe` and other architectures are
/// out of scope (see README.md's MVP order).
pub fn parse_model_config(file: &GgufFile) -> Result<(LayerConfig, usize), String> {
    let architecture = file.metadata.get("general.architecture").and_then(GgufValue::as_str).unwrap_or("");
    if architecture != "qwen3" {
        return Err(format!(
            "unsupported architecture '{architecture}': only dense 'qwen3' is in scope for this MVP"
        ));
    }

    let block_count = u64_meta(file, "qwen3.block_count").ok_or("missing qwen3.block_count metadata key")? as usize;
    let hidden_size =
        u64_meta(file, "qwen3.embedding_length").ok_or("missing qwen3.embedding_length metadata key")? as usize;
    let num_q_heads = u64_meta(file, "qwen3.attention.head_count")
        .ok_or("missing qwen3.attention.head_count metadata key")? as usize;
    let num_kv_heads = u64_meta(file, "qwen3.attention.head_count_kv").unwrap_or(num_q_heads as u64) as usize;
    let head_dim = u64_meta(file, "qwen3.attention.key_length")
        .ok_or("missing qwen3.attention.key_length metadata key (Qwen3's head_dim is not derivable from hidden_size/num_q_heads)")?
        as usize;

    let ffn_gate_info =
        file.tensor_info("blk.0.ffn_gate.weight").ok_or("missing blk.0.ffn_gate.weight tensor")?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        other => return Err(format!("blk.0.ffn_gate.weight has unexpected shape {other:?}, expected 2-D")),
    };

    let rope_base = f32_meta(file, "qwen3.rope.freq_base").unwrap_or(10000.0);
    let rmsnorm_eps = f32_meta(file, "qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

    Ok((LayerConfig { hidden_size, num_q_heads, num_kv_heads, head_dim, ffn_hidden_size, rope_base, rmsnorm_eps }, block_count))
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
    /// `[hidden_size, vocab_size]`, row-major `(vocab_size, hidden_size)`
    /// flat data -- used both for embedding lookup (host-side gather; batch
    /// is always 1 in this MVP, so a GPU gather kernel buys nothing) and,
    /// when no separate `output.weight` tensor exists, as the tied LM head.
    token_embd: Weight,
    output_norm: Weight,
    lm_head: Weight,
    tokenizer: Tokenizer,
}

impl Model {
    pub fn load(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        let (cfg, block_count) = parse_model_config(file)?;

        let rmsnorm_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_RMSNORM"), "rmsnorm", "rmsnorm_kernel")?;
        let rope_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ROPE"), "rope", "rope_kernel")?;
        let silu_k =
            aot::load_kernel(&device, env!("COLDSTART_KERNEL_SILU_AND_MUL"), "silu_and_mul", "silu_and_mul_kernel")?;
        let gemv_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_GEMV"), "gemv", "gemv_kernel")?;
        let attn_k = aot::load_kernel(&device, env!("COLDSTART_KERNEL_ATTENTION"), "attention", "attention_kernel")?;

        let load_weight = |name: &str| -> Result<Weight, String> {
            let info = file.tensor_info(name).ok_or_else(|| format!("missing weight '{name}'"))?;
            let bytes = file.tensor_bytes(info)?;
            let data = dequant::dequantize(info.ggml_type, bytes, info.element_count())?;
            Ok(Weight { data, shape: info.shape.clone() })
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            layers.push(LayerWeights {
                attn_norm: load_weight(&format!("blk.{i}.attn_norm.weight"))?,
                attn_q: load_weight(&format!("blk.{i}.attn_q.weight"))?,
                attn_k: load_weight(&format!("blk.{i}.attn_k.weight"))?,
                attn_v: load_weight(&format!("blk.{i}.attn_v.weight"))?,
                attn_output: load_weight(&format!("blk.{i}.attn_output.weight"))?,
                attn_q_norm: load_weight(&format!("blk.{i}.attn_q_norm.weight")).ok(),
                attn_k_norm: load_weight(&format!("blk.{i}.attn_k_norm.weight")).ok(),
                ffn_norm: load_weight(&format!("blk.{i}.ffn_norm.weight"))?,
                ffn_gate: load_weight(&format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: load_weight(&format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: load_weight(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        let token_embd = load_weight("token_embd.weight")?;
        let output_norm = load_weight("output_norm.weight")?;
        let lm_head = load_weight("output.weight").or_else(|_| load_weight("token_embd.weight"))?;

        let tokenizer = Tokenizer::from_gguf(file)?;

        Ok(Model { device, rmsnorm_k, rope_k, silu_k, gemv_k, attn_k, cfg, layers, token_embd, output_norm, lm_head, tokenizer })
    }

    fn rmsnorm(&self, x: &[f32], weight: &[f32], rows: usize, hidden_size: usize, eps: f32) -> Result<Vec<f32>, String> {
        let dev_x = self.device.htod_sync_copy(x).map_err(|e| format!("rmsnorm htod x: {e}"))?;
        let dev_w = self.device.htod_sync_copy(weight).map_err(|e| format!("rmsnorm htod weight: {e}"))?;
        let mut dev_out = self.device.alloc_zeros::<f32>(x.len()).map_err(|e| format!("rmsnorm alloc out: {e}"))?;

        let n = x.len() as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.rmsnorm_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_x, &dev_w, &mut dev_out, rows as u32, hidden_size as u32, eps))
                .map_err(|e| format!("rmsnorm launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_out).map_err(|e| format!("rmsnorm dtoh: {e}"))
    }

    fn gemv(&self, x: &[f32], w: &Weight) -> Result<Vec<f32>, String> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != in_features {
            return Err(format!("gemv: x.len()={} != in_features={in_features}", x.len()));
        }

        let dev_x = self.device.htod_sync_copy(x).map_err(|e| format!("gemv htod x: {e}"))?;
        let dev_w = self.device.htod_sync_copy(&w.data).map_err(|e| format!("gemv htod w: {e}"))?;
        let mut dev_y = self.device.alloc_zeros::<f32>(out_features).map_err(|e| format!("gemv alloc y: {e}"))?;

        let threads = 256u32;
        let blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(launch_cfg, (&dev_x, &dev_w, &mut dev_y, in_features as u32, out_features as u32))
                .map_err(|e| format!("gemv launch: {e}"))?;
        }
        self.device.dtoh_sync_copy(&dev_y).map_err(|e| format!("gemv dtoh: {e}"))
    }

    fn rope(&self, t: &mut Vec<f32>, num_heads: usize, head_dim: usize, position: usize, base: f32) -> Result<(), String> {
        let positions = [position as i32];
        let mut dev_t = self.device.htod_sync_copy(t).map_err(|e| format!("rope htod t: {e}"))?;
        let dev_pos = self.device.htod_sync_copy(&positions).map_err(|e| format!("rope htod positions: {e}"))?;

        let rotary_dim = head_dim;
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

    /// RMSNorm -> QKV -> QK-Norm (if present) -> RoPE -> attention -> O-proj
    /// (residual) -> RMSNorm -> SwiGLU FFN (residual), for one layer at
    /// `position`, appending this position's K/V onto `k_cache`/`v_cache`.
    fn forward_layer(
        &self,
        layer: &LayerWeights,
        hidden: &[f32],
        position: usize,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(hidden, &layer.attn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;

        let mut q = self.gemv(&normed, &layer.attn_q)?;
        let mut k = self.gemv(&normed, &layer.attn_k)?;
        let v = self.gemv(&normed, &layer.attn_v)?;

        if let Some(qn) = &layer.attn_q_norm {
            q = self.rmsnorm(&q, &qn.data, cfg.num_q_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }
        if let Some(kn) = &layer.attn_k_norm {
            k = self.rmsnorm(&k, &kn.data, cfg.num_kv_heads, cfg.head_dim, cfg.rmsnorm_eps)?;
        }

        self.rope(&mut q, cfg.num_q_heads, cfg.head_dim, position, cfg.rope_base)?;
        self.rope(&mut k, cfg.num_kv_heads, cfg.head_dim, position, cfg.rope_base)?;

        k_cache.extend_from_slice(&k);
        v_cache.extend_from_slice(&v);
        let seq_len = position + 1;

        let attn_out = self.attention(&q, k_cache, v_cache, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, seq_len)?;
        let o_proj = self.gemv(&attn_out, &layer.attn_output)?;
        let post_attn: Vec<f32> = hidden.iter().zip(o_proj.iter()).map(|(&h, &o)| h + o).collect();

        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;
        let gate = self.gemv(&ffn_normed, &layer.ffn_gate)?;
        let up = self.gemv(&ffn_normed, &layer.ffn_up)?;
        let activated = self.silu_and_mul(&gate, &up, cfg.ffn_hidden_size)?;
        let down = self.gemv(&activated, &layer.ffn_down)?;

        Ok(post_attn.iter().zip(down.iter()).map(|(&h, &d)| h + d).collect())
    }

    /// Encodes `prompt`, runs the full prompt through every layer one
    /// position at a time (real causal self-attention throughout, matching
    /// RustFeference's own documented scope choice for its minimal forward
    /// pass), and returns the argmax-sampled first generated token id plus
    /// its decoded text.
    pub fn forward_prompt(&self, prompt: &str) -> Result<(u32, String), String> {
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
            hidden.copy_from_slice(&self.token_embd.data[embd_base..embd_base + hidden_size]);

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
}
