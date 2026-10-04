//! Qwen3.5 hybrid models (Gated DeltaNet + gated attention, dense or routed-MoE
//! FFN): loading and forward passes.

use super::*;
use crate::error::ReflexError;

/// Hybrid-mixer prefill result: encoded prompt ids, the final position's
/// hidden state, the filled per-layer mixer states (attention K/V or GDN
/// recurrent state, see `HybridLayerState`), and the next absolute position.
/// Shared by `prefill_hybrid` and `prefill_hybrid_batched`.
pub(super) type HybridPrefillResult = (Vec<u32>, CudaSlice<f32>, Vec<HybridLayerState>, usize);

/// Hybrid-mixer generate result: generated token ids, their concatenated
/// decoded text, the final per-layer mixer states, and the total sequence
/// length reached. Returned by `generate_hybrid_impl`.
pub(super) type HybridGenerateResult = (Vec<u32>, String, Vec<HybridLayerState>, usize);

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
pub(super) struct GatedAttnLayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    pub(super) attn_v: Weight,
    pub(super) attn_q_norm: Weight,
    pub(super) attn_k_norm: Weight,
    pub(super) attn_output: Weight,
    pub(super) post_attn_norm: Weight,
    pub(super) ffn: HybridFfn,
}

/// One Gated DeltaNet transformer layer's weights (Qwen3.5 hybrid). Tensor
/// names and shapes confirmed against `reference/gated_deltanet_rustfeference.rs`'s
/// module doc comment and a real `Qwen3.5-0.8B-Q4_K_M.gguf` header. `ssm_a`
/// has no `.weight`/`.bias` suffix in the real file (already stored as
/// `-exp(A_log)`, per the reference).
pub(super) struct GatedDeltaNetLayerWeights {
    pub(super) attn_norm: Weight,
    /// `[hidden, 2*key_dim + value_dim]`, fused q/k/v the causal conv runs over.
    pub(super) attn_qkv: Weight,
    /// `[hidden, value_dim]`, the gated-output gate `z`.
    pub(super) attn_gate: Weight,
    pub(super) ssm_beta: Weight,
    pub(super) ssm_alpha: Weight,
    pub(super) ssm_dt: Weight,
    pub(super) ssm_a: Weight,
    pub(super) ssm_conv1d: Weight,
    pub(super) ssm_norm: Weight,
    pub(super) ssm_out: Weight,
    pub(super) post_attn_norm: Weight,
    pub(super) ffn: HybridFfn,
}

/// One Qwen3.5 hybrid layer's FFN: dense SwiGLU for `qwen35`, or routed MoE +
/// a sigmoid-gated shared expert for every layer of `qwen35moe` (see
/// [`HybridMoeFfn`]). Shape follows [`MlaFfn`].
// Only ever stored inside an already-boxed `GatedAttnLayerWeights`/
// `GatedDeltaNetLayerWeights` (one per layer), so the Dense/Moe size gap never
// affects a hot or large-count value; boxing Dense's fields too would just add
// indirection to the verified dense `qwen35` path.
#[allow(clippy::large_enum_variant)]
pub(super) enum HybridFfn {
    Dense {
        ffn_gate: Weight,
        ffn_up: Weight,
        ffn_down: Weight,
    },
    Moe(Box<HybridMoeFfn>),
}

impl HybridFfn {
    /// `Model::find_lora_target_mut`'s lookup for a dense FFN projection
    /// (`ffn_gate`/`ffn_up`/`ffn_down`). `None` for [`HybridFfn::Moe`] -- LoRA
    /// on per-expert-stacked tensors isn't supported for this architecture, so
    /// `Model::apply_lora` rejects such a target with its usual clear error.
    pub(super) fn dense_weight_mut(&mut self, suffix: &str) -> Option<&mut Weight> {
        match (self, suffix) {
            (HybridFfn::Dense { ffn_gate, .. }, "ffn_gate") => Some(ffn_gate),
            (HybridFfn::Dense { ffn_up, .. }, "ffn_up") => Some(ffn_up),
            (HybridFfn::Dense { ffn_down, .. }, "ffn_down") => Some(ffn_down),
            _ => None,
        }
    }
}

/// A `qwen35moe` layer's FFN weights. Tensor set confirmed against llama.cpp's
/// `src/models/qwen35moe.cpp` (`load_block_trunk`/`build_layer_ffn`): routed
/// experts (`ffn_gate_inp` router + per-expert-stacked `ffn_{gate,up,down}_exps`,
/// the same `[in_features, out_features, expert_count]` layout
/// `Model::gemv_expert` slices) plus one always-on shared expert
/// (`ffn_{gate,up,down}_shexp`). Unlike MLA's shared expert (added
/// unconditionally -- see [`MlaFfn`]), this one is scaled per token by
/// `sigmoid(ffn_gate_inp_shexp . x)` -- the Qwen3-Next convention llama.cpp
/// follows. `ffn_gate_inp_shexp` is a 1-D `[n_embd]` tensor in the GGUF; its
/// `shape` is widened to `[n_embd, 1]` at load time so `Model::gemv`/`gemm`
/// treat it as a one-output projection.
pub(super) struct HybridMoeFfn {
    pub(super) ffn_gate_inp: Weight,
    pub(super) ffn_gate_exps: Weight,
    pub(super) ffn_up_exps: Weight,
    pub(super) ffn_down_exps: Weight,
    pub(super) ffn_gate_inp_shexp: Weight,
    pub(super) ffn_gate_shexp: Weight,
    pub(super) ffn_up_shexp: Weight,
    pub(super) ffn_down_shexp: Weight,
}

pub(super) enum HybridLayerWeights {
    GatedAttention(Box<GatedAttnLayerWeights>),
    GatedDeltaNet(Box<GatedDeltaNetLayerWeights>),
}

/// Per-sequence recurrent state for one hybrid layer, matching
/// [`HybridLayerWeights`]'s variant for that layer index one-to-one.
/// Device-resident (Phase 2 round 2): `k_cache`/`v_cache` are preallocated to
/// the full prompt length up front (`forward_prompt_hybrid` knows the token
/// count before the per-position loop starts) and written into directly via
/// device-to-device copy each position -- no host round-trip, unlike the
/// pre-round-2 convention. Same for `conv_state`/`recurrent`, mutated in
/// place on-device by `Model::gdn_conv`/`Model::gdn_delta`.
pub(super) enum HybridLayerState {
    Attn {
        k_cache: CudaSlice<f32>,
        v_cache: CudaSlice<f32>,
    },
    Gdn {
        conv_state: CudaSlice<f32>,
        recurrent: CudaSlice<f32>,
    },
}

/// A loaded Qwen3.5 hybrid model's extra state, layered on top of the same
/// [`Model`] every other architecture uses (shared `token_embd`/
/// `output_norm`/`lm_head`/`tokenizer`, and the same `rmsnorm_k`/`rope_k`/
/// `silu_k`/`gemv_k`/`attn_k` kernels the Gated Attention layers and every
/// FFN reuse unchanged). `attn_cfg` is the Gated Attention layers' shape
/// (its `rotary_dim` is the real partial value); `gdn_cfg` is the Gated
/// DeltaNet layers' shape. Both are uniform across every layer of that kind
/// -- a real `qwen35` file has exactly one `qwen35.ssm.*`/`qwen35.attention.*`
/// config, not a per-layer one. `qwen35moe` shares this whole trunk, reading
/// the same keys under `qwen35moe.*`; only each layer's FFN differs (see
/// [`HybridFfn`]).
pub(super) struct HybridModel {
    pub(super) attn_cfg: LayerConfig,
    pub(super) gdn_cfg: crate::gated_deltanet::GatedDeltaNetConfig,
    /// `Some` for `qwen35moe` (every layer's FFN is [`HybridFfn::Moe`]),
    /// `None` for dense `qwen35`.
    pub(super) moe: Option<HybridMoeConfig>,
    pub(super) layers: Vec<HybridLayerWeights>,
    pub(super) gdn_conv_k: AotKernel,
    pub(super) gdn_l2_norm_k: AotKernel,
    pub(super) gdn_gates_k: AotKernel,
    pub(super) gdn_delta_k: AotKernel,
    pub(super) gdn_gated_norm_k: AotKernel,
}

impl Model {
    /// Loads a Qwen3.5 hybrid model: per-layer mixer kind (Gated Attention vs.
    /// Gated DeltaNet) resolved from metadata (never hardcoded -- see
    /// `parse_hybrid_layer_kinds`), MTP/NextN blocks rejected outright (not
    /// in this MVP's scope; a real non-MTP file like `Qwen3.5-0.8B` reports
    /// `nextn_predict_layers` absent/zero). See `HybridModel`'s doc comment
    /// for what's shared with the dense/MoE path.
    pub(super) fn load_hybrid(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
    ) -> Result<Self, ReflexError> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let warm = policy.matrix_dtype;
            let init = scope.spawn(move || Self::load_background_init(file, init_device, warm));
            Self::load_hybrid_inner(device, file, policy, init)
        })
    }

    pub(super) fn load_hybrid_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), ReflexError>>,
    ) -> Result<Self, ReflexError> {
        let architecture = file
            .metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("qwen35");
        let is_moe = architecture == "qwen35moe";
        let key = |suffix: &str| format!("{architecture}.{suffix}");

        let block_count = u64_meta(file, &key("block_count"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("block_count")))?
            as usize;
        let nextn = u64_meta(file, &key("nextn_predict_layers")).unwrap_or(0);
        if nextn != 0 {
            return Err(crate::reflex_err!(
                Other,
                "{} MTP/NextN blocks (nextn_predict_layers={nextn}) are not supported by this MVP \
                 (reconvert with convert_hf_to_gguf.py --no-mtp to drop them)",
                key("nextn_predict_layers")
            ));
        }

        let hidden_size = u64_meta(file, &key("embedding_length"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("embedding_length")))?
            as usize;
        let num_q_heads = u64_meta(file, &key("attention.head_count"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("attention.head_count")))?
            as usize;
        let num_kv_heads =
            u64_meta(file, &key("attention.head_count_kv")).unwrap_or(num_q_heads as u64) as usize;
        let head_dim = u64_meta(file, &key("attention.key_length"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("attention.key_length")))?
            as usize;
        let rotary_dim = u64_meta(file, &key("rope.dimension_count"))
            .map(|n| n as usize)
            .unwrap_or(head_dim);
        let rope_base = f32_meta(file, &key("rope.freq_base")).unwrap_or(10_000.0);
        let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        // qwen35moe has no dense FFN, so `feed_forward_length` is optional there
        // (it's only ever read by the `HybridFfn::Dense` path).
        let ffn_hidden_size = match u64_meta(file, &key("feed_forward_length")) {
            Some(n) => n as usize,
            None if is_moe => 0,
            None => {
                return Err(crate::reflex_err!(
                    Other,
                    "missing {}",
                    key("feed_forward_length")
                ))
            }
        };
        let moe = if is_moe {
            Some(parse_hybrid_moe_config(file, architecture)?)
        } else {
            None
        };
        let attn_cfg = LayerConfig {
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            ffn_hidden_size,
            rope_base,
            rmsnorm_eps,
            rope_type: RopeType::Neox,
        };

        let d_state = u64_meta(file, &key("ssm.state_size"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("ssm.state_size")))?
            as usize;
        let d_inner = u64_meta(file, &key("ssm.inner_size"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("ssm.inner_size")))?
            as usize;
        let group_count = u64_meta(file, &key("ssm.group_count"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("ssm.group_count")))?
            as usize;
        let conv_kernel = u64_meta(file, &key("ssm.conv_kernel"))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("ssm.conv_kernel")))?
            as usize;
        if d_state == 0 {
            return Err(crate::reflex_err!(
                Other,
                "{} must be nonzero",
                key("ssm.state_size")
            ));
        }
        let num_v_heads = u64_meta(file, &key("ssm.time_step_rank"))
            .map(|n| n as usize)
            .filter(|&n| n > 0)
            .unwrap_or(d_inner / d_state);
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

        let rmsnorm_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_RMSNORM")),
            "rmsnorm",
            "rmsnorm_kernel",
        )?;
        let mut rope_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ROPE")),
            "rope",
            &["rope_kernel", "rope_batch_kernel"],
        )?
        .into_iter();
        let rope_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_kernel".to_string()))?;
        let rope_batch_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_batch_kernel".to_string()))?;
        let silu_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_SILU_AND_MUL")),
            "silu_and_mul",
            "silu_and_mul_kernel",
        )?;
        let [gemv_k, gemv_f16_k] = load_kernel_pair(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV")),
            "gemv",
            ["gemv_kernel", "gemv_f16_kernel"],
        )?;
        let [gemv_gather_k, gemv_gather_f16_k] = load_kernel_pair(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV_GATHER")),
            "gemv_gather",
            ["gemv_gather_kernel", "gemv_gather_f16_kernel"],
        )?;
        let attn_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ATTENTION")),
            "attention",
            "attention_kernel",
        )?;
        let attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ATTENTION_PREFILL")),
            "attention_prefill",
            "attention_prefill_kernel",
        )?;
        let mut attn_online_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ATTENTION_ONLINE")),
            "attention_online",
            &["attention_online_kernel", "attention_online_combine_kernel"],
        )?;
        let attn_online_combine_k = attn_online_fns.pop().expect("two kernels requested");
        let attn_online_k = attn_online_fns.pop().expect("two kernels requested");
        let attn_impl = AttnImpl::from_env()?;
        let mut elementwise_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ELEMENTWISE")),
            "elementwise",
            &[
                "add_kernel",
                "split_qg_kernel",
                "sigmoid_gate_kernel",
                "moe_gather_kernel",
                "moe_scatter_add_kernel",
            ],
        )?
        .into_iter();
        let add_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing add_kernel".to_string()))?;
        let split_qg_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing split_qg_kernel".to_string()))?;
        let sigmoid_gate_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing sigmoid_gate_kernel".to_string()))?;
        let moe_gather_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing moe_gather_kernel".to_string()))?;
        let moe_scatter_add_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing moe_scatter_add_kernel".to_string()))?;
        let mut gdn_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GATED_DELTANET")),
            "gated_deltanet",
            &[
                "gdn_conv_kernel",
                "gdn_l2_norm_kernel",
                "gdn_gates_kernel",
                "gdn_delta_kernel",
                "gdn_gated_norm_kernel",
            ],
        )?
        .into_iter();
        let gdn_conv_k = gdn_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing gdn_conv_kernel".to_string()))?;
        let gdn_l2_norm_k = gdn_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing gdn_l2_norm_kernel".to_string()))?;
        let gdn_gates_k = gdn_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing gdn_gates_kernel".to_string()))?;
        let gdn_delta_k = gdn_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing gdn_delta_kernel".to_string()))?;
        let gdn_gated_norm_k = gdn_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing gdn_gated_norm_kernel".to_string()))?;
        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;

        let mut load_weight = |name: &str| -> Result<Weight, ReflexError> {
            load_weight_device(&mut pipeline, &dequant_kernels, policy, file, name)
        };

        let mut layers = Vec::with_capacity(block_count);
        for (i, &gdn) in is_gdn.iter().enumerate() {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
            let attn_norm = load_weight(&format!("blk.{i}.attn_norm.weight"))?;
            let post_attn_norm = load_weight(&format!("blk.{i}.post_attention_norm.weight"))?;
            let ffn = if is_moe {
                let mut ffn_gate_inp_shexp =
                    load_weight(&format!("blk.{i}.ffn_gate_inp_shexp.weight"))?;
                if ffn_gate_inp_shexp.shape.len() == 1 {
                    ffn_gate_inp_shexp.shape.push(1);
                }
                HybridFfn::Moe(Box::new(HybridMoeFfn {
                    ffn_gate_inp: load_weight(&format!("blk.{i}.ffn_gate_inp.weight"))?,
                    ffn_gate_exps: load_weight(&format!("blk.{i}.ffn_gate_exps.weight"))?,
                    ffn_up_exps: load_weight(&format!("blk.{i}.ffn_up_exps.weight"))?,
                    ffn_down_exps: load_weight(&format!("blk.{i}.ffn_down_exps.weight"))?,
                    ffn_gate_inp_shexp,
                    ffn_gate_shexp: load_weight(&format!("blk.{i}.ffn_gate_shexp.weight"))?,
                    ffn_up_shexp: load_weight(&format!("blk.{i}.ffn_up_shexp.weight"))?,
                    ffn_down_shexp: load_weight(&format!("blk.{i}.ffn_down_shexp.weight"))?,
                }))
            } else {
                HybridFfn::Dense {
                    ffn_gate: load_weight(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load_weight(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load_weight(&format!("blk.{i}.ffn_down.weight"))?,
                }
            };

            let layer = if gdn {
                HybridLayerWeights::GatedDeltaNet(Box::new(GatedDeltaNetLayerWeights {
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
                    ffn,
                }))
            } else {
                HybridLayerWeights::GatedAttention(Box::new(GatedAttnLayerWeights {
                    attn_norm,
                    attn_q: load_weight(&format!("blk.{i}.attn_q.weight"))?,
                    attn_k: load_weight(&format!("blk.{i}.attn_k.weight"))?,
                    attn_v: load_weight(&format!("blk.{i}.attn_v.weight"))?,
                    attn_q_norm: load_weight(&format!("blk.{i}.attn_q_norm.weight"))?,
                    attn_k_norm: load_weight(&format!("blk.{i}.attn_k_norm.weight"))?,
                    attn_output: load_weight(&format!("blk.{i}.attn_output.weight"))?,
                    post_attn_norm,
                    ffn,
                }))
            };
            layers.push(layer);
        }
        eprintln!();

        let token_embd_info = file
            .tensor_info("token_embd.weight")
            .ok_or_else(|| ReflexError::Gguf("missing weight 'token_embd.weight'".to_string()))?;
        let token_embd_bytes = file.tensor_bytes_shared(token_embd_info)?;
        let token_embd = LazyTokenEmbedding::new(
            token_embd_info.ggml_type,
            token_embd_bytes,
            &token_embd_info.shape,
        )?;

        let output_norm = load_weight("output_norm.weight")?;

        // Unlike dense/MoE `Model::load`, this path keeps the tied case
        // eager (`LmHead::Resident`, not `TiedLazy`) -- `system1_evaluate`
        // (the only caller the laziness optimization targets) already
        // rejects hybrid/MLA models outright, so there's no lazy-gather
        // win to have here, only a type to match `Model::lm_head`'s field.
        let lm_head = match file.tensor_info("output.weight") {
            Some(info) => {
                let bytes = file.tensor_bytes(info)?;
                let data = dequantize_matrix_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    policy.matrix_dtype,
                    info.ggml_type,
                    bytes,
                    info.element_count(),
                )
                .map_err(|e| e.rewrap(format!("load weight 'output.weight': {e}")))?;
                LmHead::Resident(Weight {
                    data,
                    shape: info.shape.clone(),
                })
            }
            None => {
                let data = dequantize_matrix_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    policy.matrix_dtype,
                    token_embd.ggml_type,
                    &token_embd.raw,
                    token_embd_info.element_count(),
                )
                .map_err(|e| e.rewrap(format!("load weight 'token_embd.weight': {e}")))?;
                LmHead::Resident(Weight {
                    data,
                    shape: token_embd_info.shape.clone(),
                })
            }
        };

        let (tokenizer, cublas) = init.join().map_err(|_| {
            ReflexError::Other("background load-init thread panicked".to_string())
        })??;

        let f16_act_stats = device
            .alloc_zeros::<u32>(2)
            .map_err(|e| crate::gpu_err!(e, "alloc f16 activation stats: {e}"))?;
        Ok(Model {
            device,
            cublas,
            rmsnorm_k,
            rope_k,
            rope_batch_k,
            rope_norm_k: None,
            rope_norm_batch_k: None,
            silu_k,
            gemv_k,
            gemv_f16_k,
            gemv_gather_k,
            gemv_gather_f16_k,
            moe_gather_k,
            moe_scatter_add_k,
            attn_k,
            attn_prefill_k,
            attn_online_k,
            attn_online_combine_k,
            attn_impl,
            add_k,
            split_qg_k,
            sigmoid_gate_k,
            cfg: attn_cfg.clone(),
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            dequant_kernels,
            dequant_pipeline: RefCell::new(pipeline),
            output_norm,
            lm_head,
            weights_dtype: policy.matrix_dtype,
            f16_act_stats,
            tokenizer,
            hybrid: Some(HybridModel {
                attn_cfg,
                gdn_cfg,
                moe,
                layers,
                gdn_conv_k,
                gdn_l2_norm_k,
                gdn_gates_k,
                gdn_delta_k,
                gdn_gated_norm_k,
            }),
            mla: None,
        })
    }

    /// Qwen3.5 hybrid Gated DeltaNet path -- see [`Self::system1_evaluate`]'s
    /// doc comment for the overall approach.
    ///
    /// **Single-token candidates only.** Unlike dense/MLA, `GatedDeltaNet`
    /// sublayers' `conv_state`/`recurrent` (`HybridLayerState::Gdn`) are a
    /// running recurrent accumulator, not a position-addressed slot --
    /// dense's/MLA's trick of reusing one shared cache sequentially across
    /// candidates (safe there because K/V is indexed by position, so a
    /// later candidate's continuation simply overwrites an earlier one's)
    /// would instead leave a multi-token candidate's continuation drifted
    /// past the shared post-prefill snapshot, silently corrupting every
    /// candidate scored after the first multi-token one. Rejecting
    /// multi-token candidates here avoids that hazard entirely; this still
    /// covers System1's own headline use cases (Yes/No, A-D, a 1-10 scale).
    /// State-cloning to lift this restriction is a real follow-up, not
    /// attempted here.
    pub(super) fn system1_evaluate_hybrid(
        &self,
        h: &HybridModel,
        prompt: &str,
        candidates: &[System1Candidate],
        temperature: f32,
    ) -> Result<System1Response, ReflexError> {
        if candidates.is_empty() {
            return Err(ReflexError::InvalidInput(
                "system1_evaluate: candidates must not be empty".to_string(),
            ));
        }

        let resolved: Vec<Vec<u32>> = candidates
            .iter()
            .map(|c| self.resolve_candidate_token_ids(prompt, &c.text))
            .collect::<Result<_, _>>()?;
        if resolved.iter().any(|ids| ids.len() > 1) {
            return Err(ReflexError::InvalidInput(
                "system1_evaluate: hybrid Qwen3.5 models currently support single-token candidates only"
                    .to_string(),
            ));
        }

        let (ids, hidden_batched, _states, _position) =
            self.prefill_hybrid_batched(h, prompt, None, 0)?;
        let hidden_size = h.attn_cfg.hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let hidden = self.last_row(&hidden_batched, ids.len(), hidden_size)?;

        let normed = self.rmsnorm(&hidden, self.output_norm.f32()?, 1, hidden_size, eps)?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let scores = self.gemv_gather_lm_head(&normed, &first_tokens)?;

        Self::finish_system1_response(candidates, resolved, scores, temperature)
    }

    /// Causal depthwise conv1d + SiLU over the fused qkv, advancing
    /// `conv_state` in place on-device (Phase 2 round 2 -- no
    /// upload/download per call, unlike the pre-round-2 convention referred
    /// to in `HybridLayerState`'s doc comment). Returns the post-SiLU
    /// `conv_dim` output. Ports `gdn_conv_kernel` (see
    /// `kernels_cuda/gated_deltanet.cu`).
    pub(super) fn gdn_conv(
        &self,
        h: &HybridModel,
        qkv: &CudaSlice<f32>,
        conv1d: &CudaSlice<f32>,
        conv_state: &mut CudaSlice<f32>,
        conv_dim: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(conv_dim)
            .map_err(|e| crate::gpu_err!(e, "gdn_conv alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = (conv_dim as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            h.gdn_conv_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        qkv,
                        conv1d,
                        conv_state,
                        &mut dev_out,
                        conv_dim as u32,
                        h.gdn_cfg.conv_kernel_size as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gdn_conv launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// In-place per-head L2-normalize `x[offset..offset + heads*head_dim]`
    /// (`x` is already device-resident -- see `Self::forward_gdn_mixer`,
    /// which calls this twice in a row on the same device buffer, once for
    /// the q heads and once for the k heads, without an intervening
    /// host round-trip). Ports `gdn_l2_norm_kernel`.
    // Each parameter maps 1:1 to a distinct `gdn_l2_norm_kernel` launch
    // argument; bundling them into a struct would just relocate the count,
    // not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_l2_norm(
        &self,
        h: &HybridModel,
        dev_x: &mut CudaSlice<f32>,
        offset: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
        scale: f32,
    ) -> Result<(), ReflexError> {
        let launch_cfg = LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: (h.gdn_cfg.norm_block_dim(), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            h.gdn_l2_norm_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (dev_x, offset as u32, head_dim as u32, eps, scale),
                )
                .map_err(|e| crate::gpu_err!(e, "gdn_l2_norm launch: {e}"))?;
        }
        Ok(())
    }

    /// `beta = sigmoid(beta_raw)`, `decay = exp(softplus(alpha_raw + dt) *
    /// a)`. All inputs/outputs device-resident (Phase 2 round 2). Ports
    /// `gdn_gates_kernel`.
    pub(super) fn gdn_gates(
        &self,
        h: &HybridModel,
        alpha_raw: &CudaSlice<f32>,
        beta_raw: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        num_v_heads: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), ReflexError> {
        let mut dev_decay = self
            .device
            .alloc_zeros::<f32>(num_v_heads)
            .map_err(|e| crate::gpu_err!(e, "gdn_gates alloc decay: {e}"))?;
        let mut dev_beta = self
            .device
            .alloc_zeros::<f32>(num_v_heads)
            .map_err(|e| crate::gpu_err!(e, "gdn_gates alloc beta: {e}"))?;

        let threads = 256u32;
        let blocks = (num_v_heads as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            h.gdn_gates_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        alpha_raw,
                        beta_raw,
                        dt,
                        a,
                        &mut dev_decay,
                        &mut dev_beta,
                        num_v_heads as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gdn_gates launch: {e}"))?;
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
    pub(super) fn gdn_delta(
        &self,
        h: &HybridModel,
        recurrent: &mut CudaSlice<f32>,
        qkv_normed: &CudaSlice<f32>,
        key_dim: usize,
        beta: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.gdn_cfg;
        let mut dev_o = self
            .device
            .alloc_zeros::<f32>(cfg.value_dim())
            .map_err(|e| crate::gpu_err!(e, "gdn_delta alloc o: {e}"))?;

        let launch_cfg = LaunchConfig {
            grid_dim: (cfg.num_v_heads as u32, 1, 1),
            block_dim: (cfg.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
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
                .map_err(|e| crate::gpu_err!(e, "gdn_delta launch: {e}"))?;
        }
        Ok(dev_o)
    }

    /// `y = RMSNorm(o, ssm_norm) * silu(z)`, all device-resident (Phase 2
    /// round 2). Ports `gdn_gated_norm_kernel`.
    pub(super) fn gdn_gated_norm(
        &self,
        h: &HybridModel,
        o: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        norm_w: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.gdn_cfg;
        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(cfg.value_dim())
            .map_err(|e| crate::gpu_err!(e, "gdn_gated_norm alloc y: {e}"))?;

        let launch_cfg = LaunchConfig {
            grid_dim: (cfg.num_v_heads as u32, 1, 1),
            block_dim: (cfg.norm_block_dim(), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            h.gdn_gated_norm_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (o, z, norm_w, &mut dev_y, cfg.head_dim as u32, eps),
                )
                .map_err(|e| crate::gpu_err!(e, "gdn_gated_norm launch: {e}"))?;
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
    pub(super) fn forward_gdn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedDeltaNetLayerWeights,
        mut x: CudaSlice<f32>,
        conv_state: &mut CudaSlice<f32>,
        recurrent: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.gdn_cfg;
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();

        let normed = self.rmsnorm(&x, w.attn_norm.f32()?, 1, h.attn_cfg.hidden_size, cfg.eps)?;
        let qkv = self.gemv(&normed, &w.attn_qkv)?;
        let z = self.gemv(&normed, &w.attn_gate)?;
        let beta_raw = self.gemv(&normed, &w.ssm_beta)?;
        let alpha_raw = self.gemv(&normed, &w.ssm_alpha)?;

        let (beta, decay) = self.gdn_gates(
            h,
            &alpha_raw,
            &beta_raw,
            w.ssm_dt.f32()?,
            w.ssm_a.f32()?,
            cfg.num_v_heads,
        )?;
        let mut conv_out = self.gdn_conv(h, &qkv, w.ssm_conv1d.f32()?, conv_state, conv_dim)?;

        // Split q/k, L2-normalize both (q additionally scaled), v left raw,
        // in place on the same device buffer `gdn_conv` just produced.
        let q_scale = 1.0 / (cfg.head_dim as f32).sqrt();
        self.gdn_l2_norm(
            h,
            &mut conv_out,
            0,
            cfg.num_k_heads,
            cfg.head_dim,
            cfg.eps,
            q_scale,
        )?;
        self.gdn_l2_norm(
            h,
            &mut conv_out,
            key_dim,
            cfg.num_k_heads,
            cfg.head_dim,
            cfg.eps,
            1.0,
        )?;

        let o = self.gdn_delta(h, recurrent, &conv_out, key_dim, &beta, &decay)?;
        let y = self.gdn_gated_norm(h, &o, &z, w.ssm_norm.f32()?, cfg.eps)?;
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
    /// sigmoid gating are device-resident via `Self::split_qg_k`/
    /// `Self::sigmoid_gate_k` -- the same kernels
    /// `forward_gated_attn_mixer_batched` uses, called here with `rows=1`
    /// since both are already generic over row count -- everything else in
    /// this function (RMSNorm/QKV/QK-Norm/RoPE/attention/O-proj/residual) is
    /// device-resident like `forward_attn_block`.
    pub(super) fn forward_gated_attn_mixer(
        &self,
        h: &HybridModel,
        w: &GatedAttnLayerWeights,
        mut hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.attn_cfg;
        let normed = self.rmsnorm(
            &hidden,
            w.attn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let qg = self.gemv(&normed, &w.attn_q)?;
        let q_dim = cfg.num_q_heads * cfg.head_dim;
        let mut q = self
            .device
            .alloc_zeros::<f32>(q_dim)
            .map_err(|e| crate::gpu_err!(e, "gated-attn q alloc: {e}"))?;
        let mut gate = self
            .device
            .alloc_zeros::<f32>(q_dim)
            .map_err(|e| crate::gpu_err!(e, "gated-attn gate alloc: {e}"))?;
        {
            let threads = 256u32;
            let blocks = (q_dim as u32).div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.split_qg_k
                    .function
                    .clone()
                    .launch(
                        launch_cfg,
                        (
                            &qg,
                            &mut q,
                            &mut gate,
                            cfg.num_q_heads as u32,
                            cfg.head_dim as u32,
                            1u32,
                        ),
                    )
                    .map_err(|e| crate::gpu_err!(e, "split_qg launch: {e}"))?;
            }
        }

        let mut k = self.gemv(&normed, &w.attn_k)?;
        let v = self.gemv(&normed, &w.attn_v)?;

        q = self.rmsnorm(
            &q,
            w.attn_q_norm.f32()?,
            cfg.num_q_heads,
            cfg.head_dim,
            cfg.rmsnorm_eps,
        )?;
        k = self.rmsnorm(
            &k,
            w.attn_k_norm.f32()?,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rmsnorm_eps,
        )?;

        self.rope(
            &mut q,
            cfg.num_q_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_base,
            cfg.rope_type,
        )?;
        self.rope(
            &mut k,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_base,
            cfg.rope_type,
        )?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = position * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + kv_stride);
            self.device
                .dtod_copy(&k, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "gated-attn kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + kv_stride);
            self.device
                .dtod_copy(&v, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "gated-attn kv-cache dtod v: {e}"))?;
        }
        let seq_len = position + 1;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let mut attn_out = self.attention(
            &q,
            &k_view,
            &v_view,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            seq_len,
        )?;

        {
            let n = q_dim as u32;
            let threads = 256u32;
            let blocks = n.div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.sigmoid_gate_k
                    .function
                    .clone()
                    .launch(launch_cfg, (&mut attn_out, &gate, n))
                    .map_err(|e| crate::gpu_err!(e, "sigmoid_gate launch: {e}"))?;
            }
        }

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
    pub(super) fn forward_gated_attn_mixer_batched(
        &self,
        h: &HybridModel,
        w: &GatedAttnLayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.attn_cfg;
        let normed = self.rmsnorm(
            &hidden,
            w.attn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let qg = self.gemm(&normed, &w.attn_q, rows)?;
        let q_elems = rows * cfg.num_q_heads * cfg.head_dim;
        let mut q = self
            .device
            .alloc_zeros::<f32>(q_elems)
            .map_err(|e| crate::gpu_err!(e, "gated-attn-batched q alloc: {e}"))?;
        let mut gate = self
            .device
            .alloc_zeros::<f32>(q_elems)
            .map_err(|e| crate::gpu_err!(e, "gated-attn-batched gate alloc: {e}"))?;
        {
            let threads = 256u32;
            let blocks = (q_elems as u32).div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.split_qg_k
                    .function
                    .clone()
                    .launch(
                        launch_cfg,
                        (
                            &qg,
                            &mut q,
                            &mut gate,
                            cfg.num_q_heads as u32,
                            cfg.head_dim as u32,
                            rows as u32,
                        ),
                    )
                    .map_err(|e| crate::gpu_err!(e, "split_qg launch: {e}"))?;
            }
        }

        let mut k = self.gemm(&normed, &w.attn_k, rows)?;
        let v = self.gemm(&normed, &w.attn_v, rows)?;

        q = self.rmsnorm(
            &q,
            w.attn_q_norm.f32()?,
            rows * cfg.num_q_heads,
            cfg.head_dim,
            cfg.rmsnorm_eps,
        )?;
        k = self.rmsnorm(
            &k,
            w.attn_k_norm.f32()?,
            rows * cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rmsnorm_eps,
        )?;

        self.rope_batch(
            &mut q,
            start_pos,
            cfg.num_q_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            rows,
            cfg.rope_base,
            cfg.rope_type,
        )?;
        self.rope_batch(
            &mut k,
            start_pos,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            rows,
            cfg.rope_base,
            cfg.rope_type,
        )?;

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = start_pos * kv_stride;
        let write_len = rows * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + write_len);
            self.device
                .dtod_copy(&k, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "gated-attn-batched kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + write_len);
            self.device
                .dtod_copy(&v, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "gated-attn-batched kv-cache dtod v: {e}"))?;
        }
        let seq_len = start_pos + rows;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let mut attn_out = self.attention_prefill(
            &q,
            &k_view,
            &v_view,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            start_pos,
            rows,
        )?;

        {
            let n = q_elems as u32;
            let threads = 256u32;
            let blocks = n.div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.sigmoid_gate_k
                    .function
                    .clone()
                    .launch(launch_cfg, (&mut attn_out, &gate, n))
                    .map_err(|e| crate::gpu_err!(e, "sigmoid_gate launch: {e}"))?;
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
    // Each parameter is a distinct weight tensor or shape/eps value the FFN
    // math needs; bundling them into a struct would just relocate the count,
    // not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hybrid_ffn(
        &self,
        mut post_mixer: CudaSlice<f32>,
        norm: &Weight,
        ffn_gate: &Weight,
        ffn_up: &Weight,
        ffn_down: &Weight,
        hidden_size: usize,
        ffn_hidden_size: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let normed = self.rmsnorm(&post_mixer, norm.f32()?, 1, hidden_size, eps)?;
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
    pub(super) fn forward_hybrid_ffn_batched(
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
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let normed = self.rmsnorm(&post_mixer, norm.f32()?, rows, hidden_size, eps)?;
        let gate = self.gemm(&normed, ffn_gate, rows)?;
        let up = self.gemm(&normed, ffn_up, rows)?;
        let activated = self.silu_and_mul(&gate, &up, rows * ffn_hidden_size)?;
        let down = self.gemm(&activated, ffn_down, rows)?;
        self.add_inplace(&mut post_mixer, &down)?;
        Ok(post_mixer)
    }

    /// One hybrid layer's FFN tail for a single row, dispatching on
    /// [`HybridFfn`]: dense `qwen35` -> [`Self::forward_hybrid_ffn`] (unchanged),
    /// `qwen35moe` -> [`Self::forward_hybrid_moe_ffn`].
    pub(super) fn forward_hybrid_layer_ffn(
        &self,
        h: &HybridModel,
        post_mixer: CudaSlice<f32>,
        norm: &Weight,
        ffn: &HybridFfn,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.attn_cfg;
        match ffn {
            HybridFfn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => self.forward_hybrid_ffn(
                post_mixer,
                norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                cfg.hidden_size,
                cfg.ffn_hidden_size,
                cfg.rmsnorm_eps,
            ),
            HybridFfn::Moe(w) => {
                let moe_cfg = h.moe.as_ref().ok_or_else(|| {
                    ReflexError::Other(
                        "internal error: hybrid MoE layer without a HybridMoeConfig".to_string(),
                    )
                })?;
                self.forward_hybrid_moe_ffn(
                    post_mixer,
                    norm,
                    w,
                    moe_cfg,
                    cfg.hidden_size,
                    cfg.rmsnorm_eps,
                )
            }
        }
    }

    /// Batched-prefill variant of [`Self::forward_hybrid_layer_ffn`].
    pub(super) fn forward_hybrid_layer_ffn_batched(
        &self,
        h: &HybridModel,
        post_mixer: CudaSlice<f32>,
        norm: &Weight,
        ffn: &HybridFfn,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &h.attn_cfg;
        match ffn {
            HybridFfn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => self.forward_hybrid_ffn_batched(
                post_mixer,
                norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                cfg.hidden_size,
                cfg.ffn_hidden_size,
                rows,
                cfg.rmsnorm_eps,
            ),
            HybridFfn::Moe(w) => {
                let moe_cfg = h.moe.as_ref().ok_or_else(|| {
                    ReflexError::Other(
                        "internal error: hybrid MoE layer without a HybridMoeConfig".to_string(),
                    )
                })?;
                self.forward_hybrid_moe_ffn_batched(
                    post_mixer,
                    norm,
                    w,
                    moe_cfg,
                    cfg.hidden_size,
                    rows,
                    cfg.rmsnorm_eps,
                )
            }
        }
    }

    /// `qwen35moe`'s FFN tail for a single row, ported from llama.cpp's
    /// `llama_model_qwen35moe::graph::build_layer_ffn`: routed experts exactly
    /// as [`Self::forward_mla_moe_ffn`] runs them (router `gemv` -> host
    /// [`route_top_k`] -- always renormalized, llama.cpp's `norm_w = true` --
    /// -> per-expert [`Self::gemv_expert`] SwiGLU -> weighted
    /// [`Self::moe_scatter_add`]), then the shared expert, a dense SwiGLU scaled
    /// by `sigmoid(ffn_gate_inp_shexp . x)` before being added (the per-token
    /// gate MLA's shared expert doesn't have). The sigmoid runs host-side on
    /// the single gate logit and is applied as the scatter-add's weight, so
    /// no new kernel is needed.
    pub(super) fn forward_hybrid_moe_ffn(
        &self,
        mut post_mixer: CudaSlice<f32>,
        norm: &Weight,
        w: &HybridMoeFfn,
        moe_cfg: &HybridMoeConfig,
        hidden_size: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let ffn_normed = self.rmsnorm(&post_mixer, norm.f32()?, 1, hidden_size, eps)?;

        let router_logits_dev = self.gemv(&ffn_normed, &w.ffn_gate_inp)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe router dtoh: {e}"))?;
        let routed = route_top_k(&router_logits, moe_cfg.expert_used_count)?;

        let mut ffn_out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe ffn_out alloc: {e}"))?;
        let dest_row0 = self
            .device
            .htod_sync_copy(&[0u32])
            .map_err(|e| crate::gpu_err!(e, "hybrid moe dest_row htod: {e}"))?;
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, &w.ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, &w.ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, moe_cfg.n_ff_exp)?;
            let down = self.gemv_expert(&activated, &w.ffn_down_exps, expert_idx)?;
            let weight_dev = self
                .device
                .htod_sync_copy(&[weight * moe_cfg.weights_scale])
                .map_err(|e| crate::gpu_err!(e, "hybrid moe weight htod: {e}"))?;
            self.moe_scatter_add(&down, &dest_row0, &weight_dev, &mut ffn_out, hidden_size)?;
        }

        let shared_hidden_size = w.ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemv(&ffn_normed, &w.ffn_gate_shexp)?;
        let shared_up = self.gemv(&ffn_normed, &w.ffn_up_shexp)?;
        let shared_activated = self.silu_and_mul(&shared_gate, &shared_up, shared_hidden_size)?;
        let shared_down = self.gemv(&shared_activated, &w.ffn_down_shexp)?;
        let shared_logit_dev = self.gemv(&ffn_normed, &w.ffn_gate_inp_shexp)?;
        let shared_logit = self
            .device
            .dtoh_sync_copy(&shared_logit_dev)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe shared gate dtoh: {e}"))?;
        let shared_weight: Vec<f32> = shared_logit.iter().map(|&g| sigmoid(g)).collect();
        let shared_weight_dev = self
            .device
            .htod_sync_copy(&shared_weight)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe shared gate htod: {e}"))?;
        self.moe_scatter_add(
            &shared_down,
            &dest_row0,
            &shared_weight_dev,
            &mut ffn_out,
            hidden_size,
        )?;

        self.add_inplace(&mut post_mixer, &ffn_out)?;
        Ok(post_mixer)
    }

    /// Batched-prefill variant of [`Self::forward_hybrid_moe_ffn`]: routed
    /// experts via the shared grouped-GEMM core [`Self::moe_ffn_grouped`]
    /// (renormalized, `weights_scale` folded in), accumulated into a zeroed
    /// `ffn_out` first -- llama.cpp's `moe_out + ffn_shexp` order -- then the
    /// shared expert batched with [`Self::gemm`] and scatter-added row by row
    /// with each row's own host-computed `sigmoid(gate)` weight.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hybrid_moe_ffn_batched(
        &self,
        mut post_mixer: CudaSlice<f32>,
        norm: &Weight,
        w: &HybridMoeFfn,
        moe_cfg: &HybridMoeConfig,
        hidden_size: usize,
        rows: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let ffn_normed = self.rmsnorm(&post_mixer, norm.f32()?, rows, hidden_size, eps)?;

        let router_logits_dev = self.gemm(&ffn_normed, &w.ffn_gate_inp, rows)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe router dtoh: {e}"))?;
        let num_experts = router_logits.len() / rows;

        let mut ffn_out = self
            .device
            .alloc_zeros::<f32>(rows * hidden_size)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe ffn_out alloc: {e}"))?;
        self.moe_ffn_grouped(
            &ffn_normed,
            rows,
            hidden_size,
            &router_logits,
            num_experts,
            moe_cfg.expert_used_count,
            true,
            moe_cfg.weights_scale,
            &w.ffn_gate_exps,
            &w.ffn_up_exps,
            &w.ffn_down_exps,
            &mut ffn_out,
        )?;

        let shared_hidden_size = w.ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemm(&ffn_normed, &w.ffn_gate_shexp, rows)?;
        let shared_up = self.gemm(&ffn_normed, &w.ffn_up_shexp, rows)?;
        let shared_activated =
            self.silu_and_mul(&shared_gate, &shared_up, rows * shared_hidden_size)?;
        let shared_down = self.gemm(&shared_activated, &w.ffn_down_shexp, rows)?;
        let shared_logits_dev = self.gemm(&ffn_normed, &w.ffn_gate_inp_shexp, rows)?;
        let shared_logits = self
            .device
            .dtoh_sync_copy(&shared_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe shared gate dtoh: {e}"))?;
        let shared_weights: Vec<f32> = shared_logits.iter().map(|&g| sigmoid(g)).collect();
        let dest_rows: Vec<u32> = (0..rows as u32).collect();
        let shared_weights_dev = self
            .device
            .htod_sync_copy(&shared_weights)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe shared gate htod: {e}"))?;
        let dest_rows_dev = self
            .device
            .htod_sync_copy(&dest_rows)
            .map_err(|e| crate::gpu_err!(e, "hybrid moe dest_rows htod: {e}"))?;
        self.moe_scatter_add(
            &shared_down,
            &dest_rows_dev,
            &shared_weights_dev,
            &mut ffn_out,
            hidden_size,
        )?;

        self.add_inplace(&mut post_mixer, &ffn_out)?;
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
    pub(super) fn forward_hybrid_layer_batched(
        &self,
        h: &HybridModel,
        layer: &HybridLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        state: &mut HybridLayerState,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let hidden_size = h.attn_cfg.hidden_size;

        match (layer, state) {
            (
                HybridLayerWeights::GatedAttention(w),
                HybridLayerState::Attn { k_cache, v_cache },
            ) => {
                let post_mixer = self.forward_gated_attn_mixer_batched(
                    h, w, hidden, start_pos, rows, k_cache, v_cache,
                )?;
                self.forward_hybrid_layer_ffn_batched(
                    h,
                    post_mixer,
                    &w.post_attn_norm,
                    &w.ffn,
                    rows,
                )
            }
            (
                HybridLayerWeights::GatedDeltaNet(w),
                HybridLayerState::Gdn {
                    conv_state,
                    recurrent,
                },
            ) => {
                let mut out = hidden;
                for row in 0..rows {
                    let row_hidden = self.extract_row(&out, row, hidden_size)?;
                    let post_mixer =
                        self.forward_gdn_mixer(h, w, row_hidden, conv_state, recurrent)?;
                    let row_out =
                        self.forward_hybrid_layer_ffn(h, post_mixer, &w.post_attn_norm, &w.ffn)?;
                    self.write_row(&mut out, row, hidden_size, &row_out)?;
                }
                Ok(out)
            }
            _ => Err(ReflexError::Other(
                "internal error: hybrid layer/state kind mismatch".to_string(),
            )),
        }
    }

    /// Hybrid-model counterpart to [`Self::forward_prompt`]: thin wrapper
    /// over [`Self::generate_hybrid_impl`] with no import and exactly one
    /// generated token.
    pub(super) fn forward_prompt_hybrid(
        &self,
        h: &HybridModel,
        prompt: &str,
    ) -> Result<(u32, String), ReflexError> {
        let (generated, text, _states, _seq_len) = self.generate_hybrid_impl(
            h,
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;
        Ok((generated[0], text))
    }

    /// Shared per-layer state allocation/import behind [`Self::prefill_hybrid`]
    /// and [`Self::prefill_hybrid_batched`]: allocates each layer's
    /// `HybridLayerState` (`GatedAttention`'s `k_cache`/`v_cache` sized for
    /// `start_pos + rows + extra_headroom` positions, `GatedDeltaNet`'s
    /// fixed-size `conv_state`/`recurrent`) and seeds it from `imported` when
    /// resuming -- identical between the sequential and batched prefill paths,
    /// so factored out once rather than duplicated.
    pub(super) fn alloc_hybrid_states(
        &self,
        h: &HybridModel,
        imported: Option<&crate::kv_io::HybridKvCache>,
        start_pos: usize,
        rows: usize,
        extra_headroom: usize,
    ) -> Result<Vec<HybridLayerState>, ReflexError> {
        crate::limits::check_positions_up_to(
            start_pos,
            rows,
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;
        if let Some(cache) = imported {
            if cache.attn_num_kv_heads != h.attn_cfg.num_kv_heads
                || cache.attn_head_dim != h.attn_cfg.head_dim
            {
                return Err(ReflexError::KvCache(
                    "imported hybrid KV cache's GatedAttention shape doesn't match this model"
                        .to_string(),
                ));
            }
            if cache.layers.len() != h.layers.len() {
                return Err(crate::reflex_err!(
                    Other,
                    "imported hybrid KV cache has {} layers, model has {}",
                    cache.layers.len(),
                    h.layers.len()
                ));
            }
        }

        let attn_kv_cache_len =
            (start_pos + rows + extra_headroom) * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
        h.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, l)| -> Result<HybridLayerState, ReflexError> {
                let imported_layer = imported.map(|c| &c.layers[layer_idx]);
                match l {
                    HybridLayerWeights::GatedAttention(_) => {
                        let mut k_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| crate::gpu_err!(e, "alloc k_cache: {e}"))?;
                        let mut v_cache = self.device.alloc_zeros::<f32>(attn_kv_cache_len).map_err(|e| crate::gpu_err!(e, "alloc v_cache: {e}"))?;
                        if let Some(crate::kv_io::HybridLayerCacheData::Attn { k_cache: k_host, v_cache: v_host }) = imported_layer {
                            let imported_len = start_pos * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
                            let mut k_dst = k_cache.slice_mut(0..imported_len);
                            self.device.htod_sync_copy_into(k_host, &mut k_dst).map_err(|e| crate::gpu_err!(e, "import hybrid k_cache layer {layer_idx}: {e}"))?;
                            let mut v_dst = v_cache.slice_mut(0..imported_len);
                            self.device.htod_sync_copy_into(v_host, &mut v_dst).map_err(|e| crate::gpu_err!(e, "import hybrid v_cache layer {layer_idx}: {e}"))?;
                        } else if imported_layer.is_some() {
                            return Err(crate::reflex_err!(KvCache, "imported hybrid KV cache layer {layer_idx} is Gdn-kind but model layer is GatedAttention"));
                        }
                        Ok(HybridLayerState::Attn { k_cache, v_cache })
                    }
                    HybridLayerWeights::GatedDeltaNet(_) => {
                        let conv_state_len = h.gdn_cfg.conv_state_len();
                        let recurrent_len = h.gdn_cfg.recurrent_len();
                        let mut conv_state = self.device.alloc_zeros::<f32>(conv_state_len).map_err(|e| crate::gpu_err!(e, "alloc conv_state: {e}"))?;
                        let mut recurrent = self.device.alloc_zeros::<f32>(recurrent_len).map_err(|e| crate::gpu_err!(e, "alloc recurrent: {e}"))?;
                        if let Some(crate::kv_io::HybridLayerCacheData::Gdn { conv_state: c_host, recurrent: r_host }) = imported_layer {
                            if c_host.len() != conv_state_len || r_host.len() != recurrent_len {
                                return Err(crate::reflex_err!(KvCache, "imported hybrid KV cache layer {layer_idx} Gdn state size mismatch"));
                            }
                            self.device.htod_sync_copy_into(c_host, &mut conv_state).map_err(|e| crate::gpu_err!(e, "import gdn conv_state layer {layer_idx}: {e}"))?;
                            self.device.htod_sync_copy_into(r_host, &mut recurrent).map_err(|e| crate::gpu_err!(e, "import gdn recurrent layer {layer_idx}: {e}"))?;
                        } else if imported_layer.is_some() {
                            return Err(crate::reflex_err!(KvCache, "imported hybrid KV cache layer {layer_idx} is Attn-kind but model layer is GatedDeltaNet"));
                        }
                        Ok(HybridLayerState::Gdn { conv_state, recurrent })
                    }
                }
            })
            .collect::<Result<Vec<_>, ReflexError>>()
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
    pub(super) fn prefill_hybrid(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        extra_headroom: usize,
    ) -> Result<HybridPrefillResult, ReflexError> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.prompt_bos() {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err(ReflexError::InvalidInput(
                "encode produced no tokens".to_string(),
            ));
        }

        let mut states =
            self.alloc_hybrid_states(h, imported, start_pos, ids.len(), extra_headroom)?;

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_hybrid(h, token_id, position, &mut states)?);
            position += 1;
        }
        let hidden =
            hidden_dev.ok_or_else(|| ReflexError::Other("no tokens processed".to_string()))?;

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
    pub(super) fn prefill_hybrid_batched(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        extra_headroom: usize,
    ) -> Result<HybridPrefillResult, ReflexError> {
        let start_pos = imported.map(|c| c.seq_len).unwrap_or(0);

        let mut ids = self.tokenizer.encode(prompt)?;
        if start_pos == 0 {
            if let Some(bos) = self.tokenizer.prompt_bos() {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
        }
        if ids.is_empty() {
            return Err(ReflexError::InvalidInput(
                "encode produced no tokens".to_string(),
            ));
        }
        let rows = ids.len();

        let mut states = self.alloc_hybrid_states(h, imported, start_pos, rows, extra_headroom)?;

        let hidden_size = h.attn_cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd.row(token_id)?);
        }
        let mut hidden = self
            .device
            .htod_sync_copy(&host_embd)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;

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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_hybrid_impl(
        &self,
        h: &HybridModel,
        prompt: &str,
        imported: Option<&crate::kv_io::HybridKvCache>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        mut on_first_token: impl FnMut(&[f32]),
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<HybridGenerateResult, ReflexError> {
        if max_new_tokens == 0 {
            return Err(ReflexError::InvalidInput(
                "max_new_tokens must be at least 1".to_string(),
            ));
        }

        let (ids, hidden_batched, mut states, mut position) =
            self.prefill_hybrid_batched(h, prompt, imported, max_new_tokens)?;
        let hidden_size = h.attn_cfg.hidden_size;
        let eps = h.attn_cfg.rmsnorm_eps;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), hidden_size)?;

        let mut rng = crate::sampling::make_rng(sampling.seed);
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits = self.lm_head_logits(&hidden, hidden_size, eps)?;
        let mut next_id = crate::sampling::sample(&first_logits, sampling, &mut rng)?;
        on_first_token(&first_logits);
        generated.push(next_id);
        on_token(
            next_id,
            &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
        );

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_hybrid(h, next_id, position, &mut states)?;
            position += 1;
            let logits = self.lm_head_logits(&hidden, hidden_size, eps)?;
            next_id = crate::sampling::sample(&logits, sampling, &mut rng)?;
            generated.push(next_id);
            on_token(
                next_id,
                &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
            );
        }

        let text = self.tokenizer.decode(&generated);
        Ok((generated, text, states, position))
    }

    /// Embeds `token_id` and runs it through every hybrid layer at absolute
    /// `position`, dispatching each layer to its mixer/state pair.
    pub(super) fn forward_one_token_hybrid(
        &self,
        h: &HybridModel,
        token_id: u32,
        position: usize,
        states: &mut [HybridLayerState],
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd.row(token_id)?)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;

        for (layer, state) in h.layers.iter().zip(states.iter_mut()) {
            hidden = match (layer, state) {
                (
                    HybridLayerWeights::GatedAttention(w),
                    HybridLayerState::Attn { k_cache, v_cache },
                ) => {
                    let post_mixer =
                        self.forward_gated_attn_mixer(h, w, hidden, position, k_cache, v_cache)?;
                    self.forward_hybrid_layer_ffn(h, post_mixer, &w.post_attn_norm, &w.ffn)?
                }
                (
                    HybridLayerWeights::GatedDeltaNet(w),
                    HybridLayerState::Gdn {
                        conv_state,
                        recurrent,
                    },
                ) => {
                    let post_mixer = self.forward_gdn_mixer(h, w, hidden, conv_state, recurrent)?;
                    self.forward_hybrid_layer_ffn(h, post_mixer, &w.post_attn_norm, &w.ffn)?
                }
                _ => {
                    return Err(ReflexError::Other(
                        "internal error: hybrid layer/state kind mismatch".to_string(),
                    ))
                }
            };
        }
        Ok(hidden)
    }

    /// Hybrid counterpart to [`Self::forward_prompt_capture_kv`]: runs the
    /// same forward pass as `forward_prompt` on a hybrid model but also
    /// downloads every layer's state (attn `k_cache`/`v_cache` sliced to
    /// exactly the positions written; GDN `conv_state`/`recurrent` in full,
    /// since they're already fixed-size) to host memory for `--export-kv`.
    pub fn forward_prompt_capture_kv_hybrid(
        &self,
        prompt: &str,
    ) -> Result<((u32, String), crate::kv_io::HybridKvCache), ReflexError> {
        let h = self.hybrid.as_ref().ok_or_else(|| {
            ReflexError::Other(
                "forward_prompt_capture_kv_hybrid called on a non-hybrid model".to_string(),
            )
        })?;
        let (generated, text, states, seq_len) = self.generate_hybrid_impl(
            h,
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;

        let attn_len = seq_len * h.attn_cfg.num_kv_heads * h.attn_cfg.head_dim;
        let mut layers = Vec::with_capacity(states.len());
        for state in &states {
            match state {
                HybridLayerState::Attn { k_cache, v_cache } => {
                    let k_host = self
                        .device
                        .dtoh_sync_copy(&k_cache.slice(0..attn_len))
                        .map_err(|e| crate::gpu_err!(e, "hybrid k_cache dtoh: {e}"))?;
                    let v_host = self
                        .device
                        .dtoh_sync_copy(&v_cache.slice(0..attn_len))
                        .map_err(|e| crate::gpu_err!(e, "hybrid v_cache dtoh: {e}"))?;
                    layers.push(crate::kv_io::HybridLayerCacheData::Attn {
                        k_cache: k_host,
                        v_cache: v_host,
                    });
                }
                HybridLayerState::Gdn {
                    conv_state,
                    recurrent,
                } => {
                    let conv_host = self
                        .device
                        .dtoh_sync_copy(conv_state)
                        .map_err(|e| crate::gpu_err!(e, "gdn conv_state dtoh: {e}"))?;
                    let rec_host = self
                        .device
                        .dtoh_sync_copy(recurrent)
                        .map_err(|e| crate::gpu_err!(e, "gdn recurrent dtoh: {e}"))?;
                    layers.push(crate::kv_io::HybridLayerCacheData::Gdn {
                        conv_state: conv_host,
                        recurrent: rec_host,
                    });
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
}
