//! Dense and MoE transformer models (Qwen3, Qwen3-MoE, Llama/Mistral): loading and
//! forward passes.

use super::*;
use crate::error::ReflexError;

/// Prefill result: encoded prompt ids, the final position's hidden state,
/// the filled per-layer K/V caches (separate K and V buffers), and the next
/// absolute position a caller may write into. Shared by `prefill_dense` and
/// `prefill_dense_batched`.
pub(super) type DensePrefillResult = (
    Vec<u32>,
    CudaSlice<f32>,
    Vec<CudaSlice<f32>>,
    Vec<CudaSlice<f32>>,
    usize,
);

/// Generate result: generated token ids, their concatenated decoded text,
/// the final per-layer K/V caches, and the total sequence length reached.
/// Returned by `generate_dense_impl`.
pub(super) type DenseGenerateResult = (
    Vec<u32>,
    String,
    Vec<CudaSlice<f32>>,
    Vec<CudaSlice<f32>>,
    usize,
);

pub(super) struct DenseLayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    pub(super) attn_v: Weight,
    pub(super) attn_output: Weight,
    /// Per-head RMSNorm on Q before RoPE (Qwen3's QK-Norm). `None` for
    /// architectures without it -- presence of the tensor, not a separate
    /// config flag, gates whether the forward pass applies it.
    pub(super) attn_q_norm: Option<Weight>,
    pub(super) attn_k_norm: Option<Weight>,
    pub(super) ffn_norm: Weight,
    pub(super) ffn_gate: Weight,
    pub(super) ffn_up: Weight,
    pub(super) ffn_down: Weight,
}

/// One MoE transformer layer's weights: an attention block identical in
/// shape/meaning to [`DenseLayerWeights`]'s (shared at forward time via
/// `Model::forward_attn_block`), plus a router (`ffn_gate_inp`, `[hidden_size,
/// expert_count]`) and per-expert-stacked SwiGLU weights (`ffn_gate_exps`/
/// `ffn_up_exps`/`ffn_down_exps`, each `[in_features, out_features,
/// expert_count]`) in place of the dense path's single shared FFN.
pub(super) struct MoeLayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    pub(super) attn_v: Weight,
    pub(super) attn_output: Weight,
    pub(super) attn_q_norm: Option<Weight>,
    pub(super) attn_k_norm: Option<Weight>,
    pub(super) ffn_norm: Weight,
    pub(super) ffn_gate_inp: Weight,
    pub(super) ffn_gate_exps: Weight,
    pub(super) ffn_up_exps: Weight,
    pub(super) ffn_down_exps: Weight,
}

/// One Kolibri-1 (`kolibri1`) layer: Qwen3-MoE-style attention with sandwich
/// norms around both blocks, a `SIGMOID_LOGIT_ADD` router
/// (`crate::moe::route_sigmoid_logit_add`), and one ungated shared expert added
/// to the routed output before `post_ffw_norm`:
///
/// ```text
/// h   = x + post_attention_norm( attn( attn_norm(x) ) )
/// out = h + post_ffw_norm( moe( ffn_norm(h) ) + shared_expert( ffn_norm(h) ) )
/// ```
///
/// Sliding layers apply RoPE and a sliding attention window; full-attention
/// layers apply neither (NoPE). Conventions: docs/design/kolibri.md.
pub(super) struct KolibriLayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    pub(super) attn_v: Weight,
    pub(super) attn_output: Weight,
    /// Always present in `kolibri1`; `Option` only to share
    /// `Model::forward_attn_block`'s signature.
    pub(super) attn_q_norm: Option<Weight>,
    pub(super) attn_k_norm: Option<Weight>,
    pub(super) post_attention_norm: Weight,
    pub(super) ffn_norm: Weight,
    pub(super) ffn_gate_inp: Weight,
    /// `exp_probs_b.bias`, kept host-side: it is only ever added to the router
    /// logits, which the host-side top-k already downloads.
    pub(super) exp_probs_b: Vec<f32>,
    pub(super) ffn_gate_exps: Weight,
    pub(super) ffn_up_exps: Weight,
    pub(super) ffn_down_exps: Weight,
    pub(super) ffn_gate_shexp: Weight,
    pub(super) ffn_up_shexp: Weight,
    pub(super) ffn_down_shexp: Weight,
    pub(super) post_ffw_norm: Weight,
    /// RoPE on/off and sliding window for this layer.
    pub(super) attn_mode: AttnMode,
    /// `expert_weights_norm` (off in real Kolibri-1).
    pub(super) normalize_top_k: bool,
    /// `REFLEX_LAZY_EXPERTS=1`: the expert stacks still to upload, one expert
    /// at a time as the router picks it (see [`Model::ensure_experts`]).
    /// `None` when every expert was loaded up front.
    pub(super) lazy: Option<LazyExperts>,
}

/// Kolibri-1's per-layer matmul tensors (`blk.{i}.<name>.weight`): the ones
/// `REFLEX_QUANT_RESIDENT=1` keeps as raw Q4_K/Q6_K blocks. `ffn_gate_inp`
/// (the router) is F32 in the file and stays dequantized.
pub(super) const KOLIBRI_MATMUL_TENSORS: [&str; 10] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
];

/// Per-layer attention variant for [`Model::forward_attn_block`]: whether Q/K
/// get RoPE, and the sliding window (`0` = plain causal attention). Every
/// architecture but Kolibri-1 uses [`AttnMode::STANDARD`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AttnMode {
    pub(super) rope: bool,
    pub(super) window: usize,
}

impl AttnMode {
    pub(super) const STANDARD: AttnMode = AttnMode {
        rope: true,
        window: 0,
    };
}

/// A per-row MoE router: one row's expert logits to its `(expert, weight)`
/// picks. See `Model::moe_ffn_grouped`.
pub(super) type MoeRouter<'a> = dyn Fn(&[f32]) -> Result<Vec<(usize, f32)>, ReflexError> + 'a;

pub(super) enum LayerWeights {
    Dense(DenseLayerWeights),
    Moe(MoeLayerWeights),
    Kolibri(Box<KolibriLayerWeights>),
}

impl Model {
    pub(super) fn load_dense(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
    ) -> Result<Self, ReflexError> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let warm = policy.matrix_dtype;
            let init = scope.spawn(move || Self::load_background_init(file, init_device, warm));
            Self::load_dense_inner(device, file, policy, init)
        })
    }

    pub(super) fn load_dense_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), ReflexError>>,
    ) -> Result<Self, ReflexError> {
        let kolibri = match file
            .metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
        {
            Some("kolibri1") => Some(parse_kolibri_config(file)?.0),
            _ => None,
        };
        let (cfg, block_count, expert_used_count) = match &kolibri {
            Some(kc) => (
                kc.layer.clone(),
                kc.sliding_layers.len(),
                Some(kc.expert_used_count),
            ),
            None => {
                let (cfg, block_count, moe) = parse_model_config(file)?;
                (cfg, block_count, moe.map(|m| m.expert_used_count))
            }
        };
        let profile_on = load_profile_enabled();
        let t_modules = std::time::Instant::now();

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
            &[
                "rope_kernel",
                "rope_batch_kernel",
                "rope_norm_kernel",
                "rope_norm_batch_kernel",
            ],
        )?
        .into_iter();
        let rope_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_kernel".to_string()))?;
        let rope_batch_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_batch_kernel".to_string()))?;
        let rope_norm_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_kernel".to_string()))?;
        let rope_norm_batch_k = rope_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_batch_kernel".to_string()))?;
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
        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;
        // Lazy expert upload (docs/design/kolibri.md, Phase 4 step 2):
        // Kolibri-1 with quantized-resident weights only.
        let lazy_experts = lazy_experts_enabled() && quant_resident_enabled() && kolibri.is_some();
        if lazy_experts_enabled() && !lazy_experts {
            eprintln!(
                "note: REFLEX_LAZY_EXPERTS=1 needs REFLEX_QUANT_RESIDENT=1 and a Kolibri-1 model; loading every weight up front"
            );
        }
        if lazy_experts {
            // The prefetch readers walk the data section in file order, so
            // they would read the expert bytes this load skips.
            pipeline.keep_fill_workers = true;
        } else {
            pipeline.prefetch_from(file);
        }
        if profile_on {
            pipeline.profile = Some(LoadProfile::default());
        }

        // Quantized-resident weights (docs/design/quantized-resident-weights.md):
        // dense layers and Kolibri-1's layers; other MoE models keep f32 and
        // say so.
        let quant_on = quant_resident_enabled();
        if quant_on && expert_used_count.is_some() && kolibri.is_none() {
            eprintln!("note: REFLEX_QUANT_RESIDENT=1 is not supported for this MoE model yet; using f32 weights");
        }
        // Dense layer matmuls stay quantized if Q4_K; an untied `output.weight`
        // (the LM head) if Q6_K or Q4_K. A tied LM head is handled lazily in
        // `Model::lm_head_resident`. Kolibri-1 keeps every matmul tensor,
        // stacked experts included, if Q4_K or Q6_K: its Q4_K_M file has Q6_K
        // `attn_v`/`ffn_down_exps`/`ffn_down_shexp` in half the layers.
        let quant_dense = quant_on && (expert_used_count.is_none() || kolibri.is_some());
        let layer_quant_types: &[GgmlType] = if kolibri.is_some() {
            &[GgmlType::Q4K, GgmlType::Q6K]
        } else {
            &[GgmlType::Q4K]
        };
        let mut arena = if quant_dense {
            let layer_tensors: &[&str] = if kolibri.is_some() {
                &KOLIBRI_MATMUL_TENSORS
            } else {
                &[
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_output",
                    "ffn_gate",
                    "ffn_up",
                    "ffn_down",
                ]
            };
            let mut names: Vec<String> = (0..block_count)
                .flat_map(|i| {
                    layer_tensors
                        .iter()
                        .map(move |t| format!("blk.{i}.{t}.weight"))
                })
                .collect();
            names.push("output.weight".to_string());
            // LoRA targets load in f32 for the merge, so they need no arena space.
            QuantArena::for_tensors(&device, file, &names, |name, ty| {
                !policy.keep_f32.contains(name)
                    && (layer_quant_types.contains(&ty)
                        || (name == "output.weight" && ty == GgmlType::Q6K))
            })?
        } else {
            None
        };
        let (gemv_q4k_k, dequant_q4k_coalesced_k, dequant_q4k_coalesced_f16_k, gemv_q6k_k) =
            if quant_dense {
                let mut fns = aot::load_kernel_module(
                    &device,
                    include_bytes!(env!("REFLEX_KERNEL_GEMV_Q4K")),
                    "gemv_q4k",
                    &[
                        "gemv_q4k_kernel",
                        "dequantize_q4k_coalesced_kernel",
                        "dequantize_q4k_coalesced_f16_kernel",
                        "gemv_q6k_kernel",
                    ],
                )?;
                let q6 = fns.pop().expect("four kernels requested");
                let dq16 = fns.pop().expect("four kernels requested");
                let dq = fns.pop().expect("four kernels requested");
                let gemv = fns.pop().expect("four kernels requested");
                (Some(gemv), Some(dq), Some(dq16), Some(q6))
            } else {
                (None, None, None, None)
            };
        let modules_ms = t_modules.elapsed().as_secs_f64() * 1000.0;
        let t_weights = std::time::Instant::now();

        // Dequantizes straight from the mmap'd GGUF bytes (on-device for
        // every format but the IQ family, Phase 2 round 3 + post-MVP
        // extensions; host `Vec<f32>` scratch, immediately dropped, for IQ)
        // into device memory -- unlike before round 1, no dequantized weight
        // stays host-resident for the model's lifetime, and no forward-pass
        // call re-uploads it (see `Weight`'s doc comment). Pipelined across
        // successive calls via `pipeline` (see `WeightLoadPipeline`'s doc
        // comment) instead of the old sequential blocking-H2D-copy path.
        // `matmul` = an `nn.Linear` weight that may stay quantized; norms and
        // everything else follow `policy` (f32, or the `--weights` dtype for a
        // matrix weight).
        // Expert stacks reserved but not uploaded (lazy mode), collected per
        // layer and moved into that layer's `KolibriLayerWeights::lazy`.
        let lazy_pending: RefCell<Vec<LazyStackedTensor>> = RefCell::new(Vec::new());
        let mut load_weight_kind = |name: &str, matmul: bool| -> Result<Weight, ReflexError> {
            match arena.as_mut() {
                Some(a) if matmul && lazy_experts && name.ends_with("_exps.weight") => {
                    match reserve_expert_stack_quant(policy, file, name, a, layer_quant_types)? {
                        Some((w, stack)) => {
                            lazy_pending.borrow_mut().push(stack);
                            Ok(w)
                        }
                        None => load_weight_device_quant(
                            &mut pipeline,
                            &dequant_kernels,
                            policy,
                            file,
                            name,
                            a,
                            layer_quant_types,
                        ),
                    }
                }
                Some(a) if matmul => load_weight_device_quant(
                    &mut pipeline,
                    &dequant_kernels,
                    policy,
                    file,
                    name,
                    a,
                    layer_quant_types,
                ),
                _ => load_weight_device(&mut pipeline, &dequant_kernels, policy, file, name),
            }
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
            let attn_norm = load_weight_kind(&format!("blk.{i}.attn_norm.weight"), false)?;
            let attn_q = load_weight_kind(&format!("blk.{i}.attn_q.weight"), true)?;
            let attn_k = load_weight_kind(&format!("blk.{i}.attn_k.weight"), true)?;
            let attn_v = load_weight_kind(&format!("blk.{i}.attn_v.weight"), true)?;
            let attn_output = load_weight_kind(&format!("blk.{i}.attn_output.weight"), true)?;
            let attn_q_norm = load_weight_kind(&format!("blk.{i}.attn_q_norm.weight"), false).ok();
            let attn_k_norm = load_weight_kind(&format!("blk.{i}.attn_k_norm.weight"), false).ok();
            let ffn_norm = load_weight_kind(&format!("blk.{i}.ffn_norm.weight"), false)?;

            let layer = if let Some(kc) = &kolibri {
                let bias_name = format!("blk.{i}.exp_probs_b.bias");
                let bias_info = file
                    .tensor_info(&bias_name)
                    .ok_or_else(|| crate::reflex_err!(Gguf, "missing tensor '{bias_name}'"))?;
                let exp_probs_b = dequant::dequantize(
                    bias_info.ggml_type,
                    file.tensor_bytes(bias_info)?,
                    bias_info.element_count(),
                )?;
                if exp_probs_b.len() != kc.expert_count {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{bias_name} has {} entries, expected expert_count = {}",
                        exp_probs_b.len(),
                        kc.expert_count
                    ));
                }
                let sliding = kc.sliding_layers[i];
                LayerWeights::Kolibri(Box::new(KolibriLayerWeights {
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm: Some(attn_q_norm.ok_or_else(|| {
                        crate::reflex_err!(Gguf, "missing weight 'blk.{i}.attn_q_norm.weight'")
                    })?),
                    attn_k_norm: Some(attn_k_norm.ok_or_else(|| {
                        crate::reflex_err!(Gguf, "missing weight 'blk.{i}.attn_k_norm.weight'")
                    })?),
                    post_attention_norm: load_weight_kind(
                        &format!("blk.{i}.post_attention_norm.weight"),
                        false,
                    )?,
                    ffn_norm,
                    ffn_gate_inp: load_weight_kind(&format!("blk.{i}.ffn_gate_inp.weight"), false)?,
                    exp_probs_b,
                    ffn_gate_exps: load_weight_kind(
                        &format!("blk.{i}.ffn_gate_exps.weight"),
                        true,
                    )?,
                    ffn_up_exps: load_weight_kind(&format!("blk.{i}.ffn_up_exps.weight"), true)?,
                    ffn_down_exps: load_weight_kind(
                        &format!("blk.{i}.ffn_down_exps.weight"),
                        true,
                    )?,
                    ffn_gate_shexp: load_weight_kind(
                        &format!("blk.{i}.ffn_gate_shexp.weight"),
                        true,
                    )?,
                    ffn_up_shexp: load_weight_kind(&format!("blk.{i}.ffn_up_shexp.weight"), true)?,
                    ffn_down_shexp: load_weight_kind(
                        &format!("blk.{i}.ffn_down_shexp.weight"),
                        true,
                    )?,
                    post_ffw_norm: load_weight_kind(
                        &format!("blk.{i}.post_ffw_norm.weight"),
                        false,
                    )?,
                    attn_mode: AttnMode {
                        rope: sliding,
                        window: if sliding { kc.sliding_window } else { 0 },
                    },
                    normalize_top_k: kc.normalize_top_k,
                    lazy: LazyExperts::new(
                        lazy_pending.borrow_mut().drain(..).collect(),
                        kc.expert_count,
                    ),
                }))
            } else if expert_used_count.is_some() {
                LayerWeights::Moe(MoeLayerWeights {
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_norm,
                    attn_k_norm,
                    ffn_norm,
                    ffn_gate_inp: load_weight_kind(&format!("blk.{i}.ffn_gate_inp.weight"), false)?,
                    ffn_gate_exps: load_weight_kind(
                        &format!("blk.{i}.ffn_gate_exps.weight"),
                        false,
                    )?,
                    ffn_up_exps: load_weight_kind(&format!("blk.{i}.ffn_up_exps.weight"), false)?,
                    ffn_down_exps: load_weight_kind(
                        &format!("blk.{i}.ffn_down_exps.weight"),
                        false,
                    )?,
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
                    ffn_gate: load_weight_kind(&format!("blk.{i}.ffn_gate.weight"), true)?,
                    ffn_up: load_weight_kind(&format!("blk.{i}.ffn_up.weight"), true)?,
                    ffn_down: load_weight_kind(&format!("blk.{i}.ffn_down.weight"), true)?,
                })
            };
            layers.push(layer);
        }
        eprintln!();
        let weights_enqueue_ms = t_weights.elapsed().as_secs_f64() * 1000.0;

        let t_embd = std::time::Instant::now();
        let token_embd_info = file
            .tensor_info("token_embd.weight")
            .ok_or_else(|| ReflexError::Gguf("missing weight 'token_embd.weight'".to_string()))?;
        let token_embd_bytes = file.tensor_bytes_shared(token_embd_info)?;
        let token_embd = LazyTokenEmbedding::new(
            token_embd_info.ggml_type,
            token_embd_bytes,
            &token_embd_info.shape,
        )?;
        let token_embd_ms = t_embd.elapsed().as_secs_f64() * 1000.0;

        let output_norm = load_weight_kind("output_norm.weight", false)?;

        // Tied-embedding models have no separate `output.weight` tensor --
        // rather than eagerly re-uploading `token_embd`'s already-dequantized
        // host bytes as a second full device copy (wasted work for a
        // `reflex system1` run, which only ever gathers a handful of rows --
        // see `LmHead`'s doc comment), defer that upload until something
        // actually needs the full matrix.
        let lm_head = match file.tensor_info("output.weight") {
            Some(_) if arena.is_some() => LmHead::Resident(load_weight_device_quant(
                &mut pipeline,
                &dequant_kernels,
                policy,
                file,
                "output.weight",
                arena.as_mut().expect("checked by the guard"),
                &[GgmlType::Q4K, GgmlType::Q6K],
            )?),
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
            None => LmHead::TiedLazy {
                shape: token_embd_info.shape.clone(),
                cell: std::sync::OnceLock::new(),
            },
        };

        pipeline.end_load();
        let t_join = std::time::Instant::now();
        let (tokenizer, cublas) = init.join().map_err(|_| {
            ReflexError::Other("background load-init thread panicked".to_string())
        })??;
        let init_join_wait_ms = t_join.elapsed().as_secs_f64() * 1000.0;

        if let Some(mut p) = pipeline.profile.take() {
            // Drain everything queued so the GPU-side totals are complete.
            // Profile mode only: the normal load never blocks here.
            let t_drain = std::time::Instant::now();
            device
                .synchronize()
                .map_err(|e| crate::gpu_err!(e, "load profile: synchronize: {e}"))?;
            unsafe { result::stream::synchronize(pipeline.copy_stream) }
                .map_err(|e| crate::gpu_err!(e, "load profile: sync copy stream: {e}"))?;
            let drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
            let (h2d_gpu_ms, dequant_gpu_ms) = p.gpu_ms();
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "REFLEX_LOAD_PROFILE quant_resident={} modules_ms={modules_ms:.2} weights_enqueue_ms={weights_enqueue_ms:.2}                  pinned_wait_ms={:.2} pinned_fill_ms={:.2} fill_threads={} fill_gbps={:.2} readers={} prefetched_mb={:.1} staging_grow_ms={:.2} out_alloc_ms={:.2} launch_ms={:.2}                  h2d_gpu_ms={h2d_gpu_ms:.2} dequant_gpu_ms={dequant_gpu_ms:.2} token_embd_ms={token_embd_ms:.2}                  init_join_wait_ms={init_join_wait_ms:.2} drain_ms={drain_ms:.2} tensors_f32={} tensors_quant={} tensors_host={}                  h2d_mb={:.1} f32_mb={:.1}",
                arena.is_some(),
                ms(p.pinned_wait),
                ms(p.pinned_fill),
                pipeline.fill_threads(),
                p.fill_bytes as f64 / 1e9 / p.pinned_fill.as_secs_f64().max(1e-9),
                pipeline.reader_threads(),
                p.prefetched_bytes as f64 / 1e6,
                ms(p.staging_grow),
                ms(p.out_alloc),
                ms(p.launch),
                p.tensors_f32,
                p.tensors_quant,
                p.tensors_host,
                p.h2d_bytes as f64 / 1e6,
                p.f32_bytes as f64 / 1e6,
            );
        }

        let f16_act_stats = device
            .alloc_zeros::<u32>(2)
            .map_err(|e| crate::gpu_err!(e, "alloc f16 activation stats: {e}"))?;
        Ok(Model {
            device,
            cublas,
            rmsnorm_k,
            rope_k,
            rope_batch_k,
            rope_norm_k: Some(rope_norm_k),
            rope_norm_batch_k: Some(rope_norm_batch_k),
            silu_k,
            gemv_k,
            gemv_f16_k,
            gemv_gather_k,
            gemv_gather_f16_k,
            gemv_q4k_k,
            dequant_q4k_coalesced_k,
            dequant_q4k_coalesced_f16_k,
            gemv_q6k_k,
            quant_scratch: RefCell::new(None),
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
            cfg,
            layers,
            expert_used_count,
            token_embd,
            dequant_kernels,
            dequant_pipeline: RefCell::new(pipeline),
            output_norm,
            lm_head,
            weights_dtype: policy.matrix_dtype,
            f16_act_stats,
            tokenizer,
            hybrid: None,
            mla: None,
        })
    }

    /// RMSNorm -> QKV -> QK-Norm (if present) -> RoPE -> causal attention ->
    /// O-proj (residual), for one layer at `position`, writing this
    /// position's K/V directly into `k_cache`/`v_cache` (preallocated
    /// device buffers, see `Model::forward_prompt`). Shared byte-for-byte by
    /// dense and MoE layers -- MoE only replaces what comes after this (see
    /// `forward_layer_moe`). Takes ownership of `hidden` and mutates it
    /// in place for the final residual add, returning it back to the
    /// caller -- the whole block stays device-resident end to end (Phase 2
    /// round 2), no host round-trip. `mode` turns RoPE off and/or sets a
    /// sliding window (Kolibri-1); `post_norm`, when given, is applied to the
    /// O-proj output before the residual add (Kolibri-1's sandwich norm).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attn_block(
        &self,
        attn_norm: &Weight,
        attn_q: &Weight,
        attn_k: &Weight,
        attn_v: &Weight,
        attn_output: &Weight,
        attn_q_norm: &Option<Weight>,
        attn_k_norm: &Option<Weight>,
        mode: AttnMode,
        post_norm: Option<&Weight>,
        mut hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(
            &hidden,
            attn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let mut q = self.gemv(&normed, attn_q)?;
        let mut k = self.gemv(&normed, attn_k)?;
        let v = self.gemv(&normed, attn_v)?;

        if let Some(qn) = attn_q_norm {
            q = self.rmsnorm(
                &q,
                qn.f32()?,
                cfg.num_q_heads,
                cfg.head_dim,
                cfg.rmsnorm_eps,
            )?;
        }
        if let Some(kn) = attn_k_norm {
            k = self.rmsnorm(
                &k,
                kn.f32()?,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rmsnorm_eps,
            )?;
        }

        if mode.rope {
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
        }

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = position * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + kv_stride);
            self.device
                .dtod_copy(&k, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "attn kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + kv_stride);
            self.device
                .dtod_copy(&v, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "attn kv-cache dtod v: {e}"))?;
        }
        let seq_len = position + 1;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let attn_out = self.attention(
            &q,
            &k_view,
            &v_view,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            seq_len,
            mode.window,
        )?;
        let mut o_proj = self.gemv(&attn_out, attn_output)?;
        if let Some(w) = post_norm {
            o_proj = self.rmsnorm(&o_proj, w.f32()?, 1, cfg.hidden_size, cfg.rmsnorm_eps)?;
        }
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    pub(super) fn forward_layer_dense(
        &self,
        layer: &DenseLayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            AttnMode::STANDARD,
            None,
            hidden,
            position,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;
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
    /// an inherently host-side sort (small expert count, already flagged as
    /// naive/unoptimized), but the weighted accumulate itself is device-resident
    /// via [`Self::moe_scatter_add`] (the same kernel the batched/grouped prefill
    /// path uses, called once per selected expert with a single-row group) --
    /// no expert's `down`-projected output round-trips through the host the way
    /// it did before this was wired in.
    pub(super) fn forward_layer_moe(
        &self,
        layer: &MoeLayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            AttnMode::STANDARD,
            None,
            hidden,
            position,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let router_logits_dev = self.gemv(&ffn_normed, &layer.ffn_gate_inp)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "moe router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or_else(|| {
            ReflexError::Other(
                "forward_layer_moe called on a model with no expert_used_count".to_string(),
            )
        })?;
        let routed = route_top_k(&router_logits, k)?;

        let mut ffn_out_dev = self
            .device
            .alloc_zeros::<f32>(cfg.hidden_size)
            .map_err(|e| crate::gpu_err!(e, "moe ffn_out alloc: {e}"))?;
        self.moe_ffn_routed(
            &ffn_normed,
            &routed,
            &layer.ffn_gate_exps,
            &layer.ffn_up_exps,
            &layer.ffn_down_exps,
            &mut ffn_out_dev,
        )?;

        self.add_inplace(&mut post_attn, &ffn_out_dev)?;
        Ok(post_attn)
    }

    /// Single-token routed-expert FFN: for each `(expert, weight)` in `routed`,
    /// runs that expert's SwiGLU on `ffn_normed` (`Self::gemv_expert`) and
    /// adds `weight *` its output into `ffn_out` on the device
    /// (`Self::moe_scatter_add` with a one-row group). The router itself is the
    /// caller's (softmax top-k for Qwen3-MoE, `SIGMOID_LOGIT_ADD` for
    /// Kolibri-1).
    pub(super) fn moe_ffn_routed(
        &self,
        ffn_normed: &CudaSlice<f32>,
        routed: &[(usize, f32)],
        ffn_gate_exps: &Weight,
        ffn_up_exps: &Weight,
        ffn_down_exps: &Weight,
        ffn_out: &mut CudaSlice<f32>,
    ) -> Result<(), ReflexError> {
        let hidden_size = ffn_normed.len();
        let ffn_hidden_size = ffn_gate_exps.shape[1] as usize;
        let dest_row0 = self
            .device
            .htod_sync_copy(&[0u32])
            .map_err(|e| crate::gpu_err!(e, "moe dest_row htod: {e}"))?;
        for &(expert_idx, weight) in routed {
            let gate = self.expert_gemv(ffn_normed, ffn_gate_exps, expert_idx)?;
            let up = self.expert_gemv(ffn_normed, ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, ffn_hidden_size)?;
            let down = self.expert_gemv(&activated, ffn_down_exps, expert_idx)?;
            let weight_dev = self
                .device
                .htod_sync_copy(&[weight])
                .map_err(|e| crate::gpu_err!(e, "moe weight htod: {e}"))?;
            self.moe_scatter_add(&down, &dest_row0, &weight_dev, ffn_out, hidden_size)?;
        }
        Ok(())
    }

    /// One-row [`Self::gemv_expert`] for any storage of a stacked per-expert
    /// tensor: a quantized-resident one goes through [`Self::gemv`] on that
    /// expert's slice ([`Self::quant_expert_weight`]).
    pub(super) fn expert_gemv(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        expert_idx: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        match Self::quant_expert_weight(w, expert_idx)? {
            Some(ew) => self.gemv(x, &ew),
            None => self.gemv_expert(x, w, expert_idx),
        }
    }

    /// `x[rows, in_features] @ w[expert_idx]^T` for one expert group of
    /// batched prefill: a quantized-resident tensor goes through
    /// [`Self::gemm`] on that expert's slice, an f32/f16 one through
    /// [`Self::gemm_view`] on its zero-copy view (cuBLAS even for a one-row
    /// group, as before quantized experts existed).
    pub(super) fn expert_gemm(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        expert_idx: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if let Some(ew) = Self::quant_expert_weight(w, expert_idx)? {
            return self.gemm(x, &ew, rows);
        }
        let (view, in_features, out_features) = Self::expert_weight_view(w, expert_idx)?;
        self.gemm_view(x, &view, in_features, out_features, rows)
    }

    /// One `REFLEX_EXPERT_TRACE` line (see [`expert_trace_enabled`]).
    /// `router_logits` holds `rows` rows of `bias.len()` logits. Besides the
    /// picks, prints each row's margin between the `k`-th and `(k+1)`-th
    /// biased logit: a margin near zero is a routing near-tie, where another
    /// implementation's rounding can pick a different expert.
    fn print_expert_trace(router_logits: &[f32], bias: &[f32], k: usize, normalize: bool) {
        let mut experts = Vec::new();
        let mut margins = Vec::new();
        for row in router_logits.chunks(bias.len()) {
            if let Ok(routed) = route_sigmoid_logit_add(row, bias, k, normalize) {
                experts.extend(routed.iter().map(|&(e, _)| e.to_string()));
            }
            let mut biased: Vec<f32> = row.iter().zip(bias).map(|(l, b)| l + b).collect();
            biased.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            margins.push(match biased.get(k) {
                Some(next) => format!("{:.4}", biased[k - 1] - next),
                None => "inf".to_string(),
            });
        }
        eprintln!(
            "REFLEX_EXPERT_TRACE rows={} experts={} topk_margin={}",
            router_logits.len() / bias.len(),
            experts.join(","),
            margins.join(",")
        );
    }

    /// Kolibri-1 decode step for one layer; see [`KolibriLayerWeights`] for the
    /// data flow. The shared expert seeds the FFN accumulator, the routed
    /// experts add onto it, and `post_ffw_norm` is applied to the sum before
    /// the residual add.
    /// Uploads whichever of `experts` this lazily loaded layer doesn't have
    /// on the GPU yet (all of each one's stacked slices), and marks them
    /// resident. The compute stream waits for the copies, so the expert
    /// matmuls that follow read the real weights. Byte-identical to the
    /// eager load, so the output is too (docs/design/kolibri.md, Phase 4
    /// step 2).
    pub(super) fn ensure_experts(
        &self,
        lazy: &LazyExperts,
        experts: impl Iterator<Item = usize>,
    ) -> Result<(), ReflexError> {
        let mut resident = lazy.resident.borrow_mut();
        let mut missing: Vec<usize> = Vec::new();
        for e in experts {
            if e >= resident.len() {
                return Err(crate::reflex_err!(
                    Other,
                    "ensure_experts: expert {e} of {}",
                    resident.len()
                ));
            }
            if !resident[e] && !missing.contains(&e) {
                missing.push(e);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        let mut slices = Vec::with_capacity(missing.len() * lazy.stacks.len());
        for &e in &missing {
            for st in &lazy.stacks {
                let start = e * st.expert_len;
                slices.push((
                    st.device_base + start as u64,
                    &st.bytes[start..start + st.expert_len],
                ));
            }
        }
        self.dequant_pipeline.borrow_mut().upload_slices(&slices)?;
        for e in missing {
            resident[e] = true;
        }
        Ok(())
    }

    /// `(resident, total)` experts over every lazily loaded layer, or `None`
    /// when this model loaded its experts up front.
    pub fn lazy_expert_counts(&self) -> Option<(usize, usize)> {
        let mut any = false;
        let (mut r, mut t) = (0, 0);
        for layer in &self.layers {
            if let LayerWeights::Kolibri(l) = layer {
                if let Some(lazy) = &l.lazy {
                    let (lr, lt) = lazy.counts();
                    r += lr;
                    t += lt;
                    any = true;
                }
            }
        }
        any.then_some((r, t))
    }

    pub(super) fn forward_layer_kolibri(
        &self,
        layer: &KolibriLayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            layer.attn_mode,
            Some(&layer.post_attention_norm),
            hidden,
            position,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let shared_hidden_size = layer.ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemv(&ffn_normed, &layer.ffn_gate_shexp)?;
        let shared_up = self.gemv(&ffn_normed, &layer.ffn_up_shexp)?;
        let shared_activated = self.silu_and_mul(&shared_gate, &shared_up, shared_hidden_size)?;
        let mut ffn_out = self.gemv(&shared_activated, &layer.ffn_down_shexp)?;

        let router_logits_dev = self.gemv(&ffn_normed, &layer.ffn_gate_inp)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "kolibri router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or_else(|| {
            ReflexError::Other("forward_layer_kolibri: no expert_used_count".to_string())
        })?;
        let routed =
            route_sigmoid_logit_add(&router_logits, &layer.exp_probs_b, k, layer.normalize_top_k)?;
        if let Some(lazy) = &layer.lazy {
            self.ensure_experts(lazy, routed.iter().map(|&(e, _)| e))?;
        }
        if expert_trace_enabled() {
            Self::print_expert_trace(&router_logits, &layer.exp_probs_b, k, layer.normalize_top_k);
        }
        self.moe_ffn_routed(
            &ffn_normed,
            &routed,
            &layer.ffn_gate_exps,
            &layer.ffn_up_exps,
            &layer.ffn_down_exps,
            &mut ffn_out,
        )?;

        let ffn_out = self.rmsnorm(
            &ffn_out,
            layer.post_ffw_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;
        self.add_inplace(&mut post_attn, &ffn_out)?;
        Ok(post_attn)
    }

    pub(super) fn forward_layer(
        &self,
        layer: &LayerWeights,
        hidden: CudaSlice<f32>,
        position: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        match layer {
            LayerWeights::Dense(l) => {
                self.forward_layer_dense(l, hidden, position, k_cache, v_cache)
            }
            LayerWeights::Moe(l) => self.forward_layer_moe(l, hidden, position, k_cache, v_cache),
            LayerWeights::Kolibri(l) => {
                self.forward_layer_kolibri(l, hidden, position, k_cache, v_cache)
            }
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
    pub(super) fn forward_attn_block_batched(
        &self,
        attn_norm: &Weight,
        attn_q: &Weight,
        attn_k: &Weight,
        attn_v: &Weight,
        attn_output: &Weight,
        attn_q_norm: &Option<Weight>,
        attn_k_norm: &Option<Weight>,
        mode: AttnMode,
        post_norm: Option<&Weight>,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &self.cfg;
        let normed = self.rmsnorm(
            &hidden,
            attn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let mut q = self.gemm(&normed, attn_q, rows)?;
        let mut k = self.gemm(&normed, attn_k, rows)?;
        let v = self.gemm(&normed, attn_v, rows)?;

        if let Some(qn) = attn_q_norm {
            q = self.rmsnorm(
                &q,
                qn.f32()?,
                rows * cfg.num_q_heads,
                cfg.head_dim,
                cfg.rmsnorm_eps,
            )?;
        }
        if let Some(kn) = attn_k_norm {
            k = self.rmsnorm(
                &k,
                kn.f32()?,
                rows * cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rmsnorm_eps,
            )?;
        }

        if mode.rope {
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
        }

        let kv_stride = cfg.num_kv_heads * cfg.head_dim;
        let offset = start_pos * kv_stride;
        let write_len = rows * kv_stride;
        {
            let mut dst = k_cache.slice_mut(offset..offset + write_len);
            self.device
                .dtod_copy(&k, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "attn_batched kv-cache dtod k: {e}"))?;
        }
        {
            let mut dst = v_cache.slice_mut(offset..offset + write_len);
            self.device
                .dtod_copy(&v, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "attn_batched kv-cache dtod v: {e}"))?;
        }
        let seq_len = start_pos + rows;

        let k_view = k_cache.slice(0..seq_len * kv_stride);
        let v_view = v_cache.slice(0..seq_len * kv_stride);
        let attn_out = self.attention_prefill(
            &q,
            &k_view,
            &v_view,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            start_pos,
            rows,
            mode.window,
        )?;
        let mut o_proj = self.gemm(&attn_out, attn_output, rows)?;
        if let Some(w) = post_norm {
            o_proj = self.rmsnorm(&o_proj, w.f32()?, rows, cfg.hidden_size, cfg.rmsnorm_eps)?;
        }
        self.add_inplace(&mut hidden, &o_proj)?;
        Ok(hidden)
    }

    /// Batched-prefill variant of [`Self::forward_layer_dense`]: every
    /// projection in both the attention block and the FFN becomes one GEMM
    /// over all `rows` positions instead of `rows` separate GEMV launches --
    /// `Self::silu_and_mul`/`Self::add_inplace` need no change (already flat
    /// elementwise ops, see their doc comments).
    pub(super) fn forward_layer_dense_batched(
        &self,
        layer: &DenseLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block_batched(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            AttnMode::STANDARD,
            None,
            hidden,
            start_pos,
            rows,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;
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
    /// of experts (host-side `route`, one call per row's logits), so
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
    /// for `MlaFfn::Moe`, Kolibri-1's shared expert); this function only ever adds into
    /// it. `route` maps one row's `expert_count` logits to its `(expert, weight)` picks
    /// (softmax top-k for Qwen3-MoE/MLA, `SIGMOID_LOGIT_ADD` for Kolibri-1). `weight_scale` folds
    /// in a caller-side scalar (MLA's `routed_scaling_factor`; `1.0` -- a no-op -- for
    /// dense/MoE, which has no such knob) so the scatter kernel itself stays
    /// architecture-agnostic.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_ffn_grouped(
        &self,
        ffn_normed: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        router_logits: &[f32],
        expert_count: usize,
        route: &MoeRouter,
        weight_scale: f32,
        ffn_gate_exps: &Weight,
        ffn_up_exps: &Weight,
        ffn_down_exps: &Weight,
        ffn_out: &mut CudaSlice<f32>,
    ) -> Result<(), ReflexError> {
        let mut groups: Vec<Vec<(u32, f32)>> = vec![Vec::new(); expert_count];
        for row in 0..rows {
            let row_logits = &router_logits[row * expert_count..(row + 1) * expert_count];
            let routed = route(row_logits)?;
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

            let perm_row = self
                .device
                .htod_sync_copy(&rows_e)
                .map_err(|e| crate::gpu_err!(e, "moe_ffn_grouped upload rows_e: {e}"))?;
            let weight_dev = self
                .device
                .htod_sync_copy(&weights_e)
                .map_err(|e| crate::gpu_err!(e, "moe_ffn_grouped upload weights_e: {e}"))?;
            let group_size = rows_e.len();

            let x_e = self.moe_gather(ffn_normed, &perm_row, hidden_size)?;
            let gate_out_features = ffn_gate_exps.shape[1] as usize;
            let gate = self.expert_gemm(&x_e, ffn_gate_exps, expert_idx, group_size)?;
            let up = self.expert_gemm(&x_e, ffn_up_exps, expert_idx, group_size)?;
            let activated = self.silu_and_mul(&gate, &up, group_size * gate_out_features)?;
            let down = self.expert_gemm(&activated, ffn_down_exps, expert_idx, group_size)?;

            self.moe_scatter_add(&down, &perm_row, &weight_dev, ffn_out, hidden_size)?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::forward_layer_moe`]: the attention block
    /// batches identically to the dense case (`Self::forward_attn_block_batched`), and
    /// the FFN tail now batches too, via grouped-GEMM MoE batching
    /// (`Self::moe_ffn_grouped`) instead of `Self::forward_layer_moe`'s per-row
    /// `Self::gemv_expert` loop.
    pub(super) fn forward_layer_moe_batched(
        &self,
        layer: &MoeLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block_batched(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            AttnMode::STANDARD,
            None,
            hidden,
            start_pos,
            rows,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;
        let router_logits_dev = self.gemm(&ffn_normed, &layer.ffn_gate_inp, rows)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "moe router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or_else(|| {
            ReflexError::Other(
                "forward_layer_moe_batched called on a model with no expert_used_count".to_string(),
            )
        })?;
        let num_experts = router_logits.len() / rows;

        let mut ffn_out = self
            .device
            .alloc_zeros::<f32>(rows * cfg.hidden_size)
            .map_err(|e| crate::gpu_err!(e, "moe ffn_out alloc: {e}"))?;
        self.moe_ffn_grouped(
            &ffn_normed,
            rows,
            cfg.hidden_size,
            &router_logits,
            num_experts,
            // Qwen3-MoE's convention: renormalize the selected top-k weights (crate::moe::route_top_k).
            &|logits| route_top_k_with_norm(logits, k, true),
            1.0, // no routed_scaling_factor-equivalent knob for dense/MoE.
            &layer.ffn_gate_exps,
            &layer.ffn_up_exps,
            &layer.ffn_down_exps,
            &mut ffn_out,
        )?;

        self.add_inplace(&mut post_attn, &ffn_out)?;
        Ok(post_attn)
    }

    /// Batched-prefill variant of [`Self::forward_layer_kolibri`]: the shared
    /// expert runs as one GEMM over every row and seeds the accumulator; the
    /// routed experts batch through [`Self::moe_ffn_grouped`] with the
    /// `SIGMOID_LOGIT_ADD` router.
    pub(super) fn forward_layer_kolibri_batched(
        &self,
        layer: &KolibriLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut post_attn = self.forward_attn_block_batched(
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            layer.attn_mode,
            Some(&layer.post_attention_norm),
            hidden,
            start_pos,
            rows,
            k_cache,
            v_cache,
        )?;

        let cfg = &self.cfg;
        let ffn_normed = self.rmsnorm(
            &post_attn,
            layer.ffn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        let shared_hidden_size = layer.ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemm(&ffn_normed, &layer.ffn_gate_shexp, rows)?;
        let shared_up = self.gemm(&ffn_normed, &layer.ffn_up_shexp, rows)?;
        let shared_activated =
            self.silu_and_mul(&shared_gate, &shared_up, rows * shared_hidden_size)?;
        let mut ffn_out = self.gemm(&shared_activated, &layer.ffn_down_shexp, rows)?;

        let router_logits_dev = self.gemm(&ffn_normed, &layer.ffn_gate_inp, rows)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "kolibri router dtoh: {e}"))?;
        let k = self.expert_used_count.ok_or_else(|| {
            ReflexError::Other("forward_layer_kolibri_batched: no expert_used_count".to_string())
        })?;
        let num_experts = layer.exp_probs_b.len();
        if let Some(lazy) = &layer.lazy {
            // Every expert any row routes to: the union `moe_ffn_grouped`
            // is about to read. Routing is cheap host work, so it runs twice.
            let mut needed = Vec::new();
            for row in router_logits.chunks(num_experts) {
                let routed =
                    route_sigmoid_logit_add(row, &layer.exp_probs_b, k, layer.normalize_top_k)?;
                needed.extend(routed.into_iter().map(|(e, _)| e));
            }
            self.ensure_experts(lazy, needed.into_iter())?;
        }
        if expert_trace_enabled() {
            Self::print_expert_trace(&router_logits, &layer.exp_probs_b, k, layer.normalize_top_k);
        }
        self.moe_ffn_grouped(
            &ffn_normed,
            rows,
            cfg.hidden_size,
            &router_logits,
            num_experts,
            &|logits| route_sigmoid_logit_add(logits, &layer.exp_probs_b, k, layer.normalize_top_k),
            1.0,
            &layer.ffn_gate_exps,
            &layer.ffn_up_exps,
            &layer.ffn_down_exps,
            &mut ffn_out,
        )?;

        let ffn_out = self.rmsnorm(
            &ffn_out,
            layer.post_ffw_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;
        self.add_inplace(&mut post_attn, &ffn_out)?;
        Ok(post_attn)
    }

    /// Batched-prefill variant of [`Self::forward_layer`].
    pub(super) fn forward_layer_batched(
        &self,
        layer: &LayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        match layer {
            LayerWeights::Dense(l) => {
                self.forward_layer_dense_batched(l, hidden, start_pos, rows, k_cache, v_cache)
            }
            LayerWeights::Moe(l) => {
                self.forward_layer_moe_batched(l, hidden, start_pos, rows, k_cache, v_cache)
            }
            LayerWeights::Kolibri(l) => {
                self.forward_layer_kolibri_batched(l, hidden, start_pos, rows, k_cache, v_cache)
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
    /// `system1_evaluate` any more (both switched to `prefill_dense_batched`) -- kept only
    /// for the oracle role and any future direct caller. `extra_headroom` sizes the
    /// K/V cache with that many additional position slots beyond the
    /// encoded prompt itself. Returns the encoded prompt ids (including any
    /// inserted BOS), the final position's hidden state, the filled K/V
    /// caches, and the next absolute position a caller may write into.
    #[cfg(test)]
    pub(super) fn prefill_dense(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        extra_headroom: usize,
    ) -> Result<DensePrefillResult, ReflexError> {
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

        crate::limits::check_positions_up_to(
            start_pos,
            ids.len(),
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;
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
            .map_err(|e| crate::gpu_err!(e, "alloc k_cache: {e}"))?;
        let mut v_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::gpu_err!(e, "alloc v_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.num_kv_heads != self.cfg.num_kv_heads || cache.head_dim != self.cfg.head_dim {
                return Err(crate::reflex_err!(KvCache,
                    "imported KV cache shape mismatch: file has num_kv_heads={} head_dim={}, model expects num_kv_heads={} head_dim={}",
                    cache.num_kv_heads, cache.head_dim, self.cfg.num_kv_heads, self.cfg.head_dim
                ));
            }
            if cache.k_caches.len() != self.layers.len() {
                return Err(crate::reflex_err!(
                    Other,
                    "imported KV cache has {} layers, model has {}",
                    cache.k_caches.len(),
                    self.layers.len()
                ));
            }
            let imported_len = cache.seq_len * kv_stride;
            for (layer_idx, (k_host, v_host)) in
                cache.k_caches.iter().zip(&cache.v_caches).enumerate()
            {
                let mut k_dst = k_caches[layer_idx].slice_mut(0..imported_len);
                self.device
                    .htod_sync_copy_into(k_host, &mut k_dst)
                    .map_err(|e| {
                        crate::gpu_err!(e, "import k_cache htod layer {layer_idx}: {e}")
                    })?;
                let mut v_dst = v_caches[layer_idx].slice_mut(0..imported_len);
                self.device
                    .htod_sync_copy_into(v_host, &mut v_dst)
                    .map_err(|e| {
                        crate::gpu_err!(e, "import v_cache htod layer {layer_idx}: {e}")
                    })?;
            }
        }

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_dense(
                token_id,
                position,
                &mut k_caches,
                &mut v_caches,
            )?);
            position += 1;
        }
        let hidden =
            hidden_dev.ok_or_else(|| ReflexError::Other("no tokens processed".to_string()))?;

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
    pub(super) fn prefill_dense_batched(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        extra_headroom: usize,
    ) -> Result<DensePrefillResult, ReflexError> {
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
        crate::limits::check_positions_up_to(
            start_pos,
            rows,
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;

        let kv_stride = self.cfg.num_kv_heads * self.cfg.head_dim;
        let total_len = start_pos + rows + extra_headroom;
        let mut k_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::gpu_err!(e, "alloc k_cache: {e}"))?;
        let mut v_caches: Vec<CudaSlice<f32>> = (0..self.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * kv_stride))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::gpu_err!(e, "alloc v_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.num_kv_heads != self.cfg.num_kv_heads || cache.head_dim != self.cfg.head_dim {
                return Err(crate::reflex_err!(KvCache,
                    "imported KV cache shape mismatch: file has num_kv_heads={} head_dim={}, model expects num_kv_heads={} head_dim={}",
                    cache.num_kv_heads, cache.head_dim, self.cfg.num_kv_heads, self.cfg.head_dim
                ));
            }
            if cache.k_caches.len() != self.layers.len() {
                return Err(crate::reflex_err!(
                    Other,
                    "imported KV cache has {} layers, model has {}",
                    cache.k_caches.len(),
                    self.layers.len()
                ));
            }
            let imported_len = cache.seq_len * kv_stride;
            for (layer_idx, (k_host, v_host)) in
                cache.k_caches.iter().zip(&cache.v_caches).enumerate()
            {
                let mut k_dst = k_caches[layer_idx].slice_mut(0..imported_len);
                self.device
                    .htod_sync_copy_into(k_host, &mut k_dst)
                    .map_err(|e| {
                        crate::gpu_err!(e, "import k_cache htod layer {layer_idx}: {e}")
                    })?;
                let mut v_dst = v_caches[layer_idx].slice_mut(0..imported_len);
                self.device
                    .htod_sync_copy_into(v_host, &mut v_dst)
                    .map_err(|e| {
                        crate::gpu_err!(e, "import v_cache htod layer {layer_idx}: {e}")
                    })?;
            }
        }

        let hidden_size = self.cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd.row(token_id)?);
        }
        let mut hidden = self
            .device
            .htod_sync_copy(&host_embd)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_batched(
                layer,
                hidden,
                start_pos,
                rows,
                &mut k_caches[layer_idx],
                &mut v_caches[layer_idx],
            )?;
        }

        Ok((ids, hidden, k_caches, v_caches, start_pos + rows))
    }

    /// Shared dense/MoE implementation behind `forward_prompt`,
    /// `forward_prompt_capture_kv`, and `generate` (Phase 3 round 2):
    /// decodes new tokens one at a time (via `Self::prefill_dense` for the
    /// prompt, then the same `forward_one_token_dense`/`lm_head_logits`/
    /// `crate::sampling::sample` trio per generated token) until
    /// `max_new_tokens` have been produced or `eos_token_id` comes up.
    /// `sampling.is_greedy()` (the default) makes `sample` delegate straight
    /// to the same `Self::argmax` this loop always used, so this stays
    /// byte-identical to the pre-sampling implementation in that case.
    /// Returns the generated token ids, their concatenated decoded text, the
    /// final per-layer K/V caches (still device-resident, sized with
    /// headroom for up to `max_new_tokens` generated positions -- callers
    /// downloading them for export must slice to `0..seq_len * kv_stride`,
    /// not the whole buffer), and the total sequence length reached.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_dense_impl(
        &self,
        prompt: &str,
        imported: Option<&crate::kv_io::DenseKvCache>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        mut on_first_token: impl FnMut(&[f32]),
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<DenseGenerateResult, ReflexError> {
        if max_new_tokens == 0 {
            return Err(ReflexError::InvalidInput(
                "max_new_tokens must be at least 1".to_string(),
            ));
        }

        let (ids, hidden_batched, mut k_caches, mut v_caches, mut position) =
            self.prefill_dense_batched(prompt, imported, max_new_tokens)?;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), self.cfg.hidden_size)?;

        let mut rng = crate::sampling::make_rng(sampling.seed);
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits =
            self.lm_head_logits(&hidden, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
        let mut next_id = crate::sampling::sample(&first_logits, sampling, &mut rng)?;
        on_first_token(&first_logits);
        generated.push(next_id);
        on_token(
            next_id,
            &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
        );

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden =
                self.forward_one_token_dense(next_id, position, &mut k_caches, &mut v_caches)?;
            position += 1;
            let logits =
                self.lm_head_logits(&hidden, self.cfg.hidden_size, self.cfg.rmsnorm_eps)?;
            next_id = crate::sampling::sample(&logits, sampling, &mut rng)?;
            generated.push(next_id);
            on_token(
                next_id,
                &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
            );
        }

        let text = self.tokenizer.decode(&generated);
        Ok((generated, text, k_caches, v_caches, position))
    }

    /// Dense/MoE Qwen3 path (the original, and still the only path with a
    /// committed real-GGUF test fixture) -- see [`Self::system1_evaluate`]'s
    /// doc comment for the overall approach. Supports both single- and
    /// multi-token candidates: multi-token continuation reuses the shared
    /// post-prompt KV headroom sequentially per candidate, safe since K/V is
    /// position-indexed and each candidate is scored to completion before
    /// the next one starts.
    pub(super) fn system1_evaluate_dense(
        &self,
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
        let max_len = resolved.iter().map(Vec::len).max().unwrap_or(1);

        let (ids, hidden_batched, mut k_caches, mut v_caches, base_position) =
            self.prefill_dense_batched(prompt, None, max_len.saturating_sub(1))?;
        let hidden = self.last_row(&hidden_batched, ids.len(), self.cfg.hidden_size)?;

        // Batched first-token gather: the sub-50ms win for the common
        // single-token case (Yes/No, A-D, a 1-10 scale).
        let normed = self.rmsnorm(
            &hidden,
            self.output_norm.f32()?,
            1,
            self.cfg.hidden_size,
            self.cfg.rmsnorm_eps,
        )?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemv_gather_lm_head(&normed, &first_tokens)?;

        // Multi-token candidates: teacher-forced continuation, reusing the
        // shared post-prompt KV headroom sequentially per candidate (safe
        // since each candidate is scored to completion before the next one
        // starts).
        for (i, ids) in resolved.iter().enumerate() {
            if ids.len() < 2 {
                continue;
            }
            for (position, w) in (base_position..).zip(ids.windows(2)) {
                let (prev, next) = (w[0], w[1]);
                let h =
                    self.forward_one_token_dense(prev, position, &mut k_caches, &mut v_caches)?;
                let normed_step = self.rmsnorm(
                    &h,
                    self.output_norm.f32()?,
                    1,
                    self.cfg.hidden_size,
                    self.cfg.rmsnorm_eps,
                )?;
                scores[i] += self.gemv_gather_lm_head(&normed_step, &[next])?[0];
            }
        }

        Self::finish_system1_response(candidates, resolved, scores, temperature)
    }

    /// Embeds `token_id` and runs it through every dense/MoE layer at
    /// absolute `position`, writing this position's K/V into `k_caches`/
    /// `v_caches` (preallocated device buffers, see `generate_dense_impl`).
    pub(super) fn forward_one_token_dense(
        &self,
        token_id: u32,
        position: usize,
        k_caches: &mut [CudaSlice<f32>],
        v_caches: &mut [CudaSlice<f32>],
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd.row(token_id)?)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer(
                layer,
                hidden,
                position,
                &mut k_caches[layer_idx],
                &mut v_caches[layer_idx],
            )?;
        }
        Ok(hidden)
    }

    /// Dense/MoE-only: runs the same forward pass as `forward_prompt` but
    /// also downloads the per-layer K/V caches (sliced to exactly the
    /// positions actually written -- `generate_dense_impl`'s buffers carry
    /// extra headroom this capture doesn't use) to host memory for
    /// `--export-kv` to serialize. Hybrid models use
    /// `forward_prompt_capture_kv_hybrid` instead, MLA models
    /// `forward_prompt_capture_kv_mla`.
    pub fn forward_prompt_capture_kv(
        &self,
        prompt: &str,
    ) -> Result<((u32, String), crate::kv_io::DenseKvCache), ReflexError> {
        if self.hybrid.is_some() {
            return Err(ReflexError::Other("--export-kv on a hybrid Qwen3.5 model needs forward_prompt_capture_kv_hybrid, not this function".to_string()));
        }
        if self.mla.is_some() {
            return Err(ReflexError::Other("--export-kv on an MLA model needs forward_prompt_capture_kv_mla, not this function".to_string()));
        }
        let (generated, text, k_caches, v_caches, seq_len) = self.generate_dense_impl(
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;

        let kv_stride = self.cfg.num_kv_heads * self.cfg.head_dim;
        let per_layer_len = seq_len * kv_stride;
        let k_caches = k_caches
            .iter()
            .map(|c| {
                self.device
                    .dtoh_sync_copy(&c.slice(0..per_layer_len))
                    .map_err(|e| crate::gpu_err!(e, "k_cache dtoh: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v_caches = v_caches
            .iter()
            .map(|c| {
                self.device
                    .dtoh_sync_copy(&c.slice(0..per_layer_len))
                    .map_err(|e| crate::gpu_err!(e, "v_cache dtoh: {e}"))
            })
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
}
