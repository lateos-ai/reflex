//! Gemma 4 (`gemma4`) dense text models: loading, the forward pass, and the
//! kernels only this architecture uses (with host references for their
//! tests). A separate path like `hybrid`/`mla`, because Gemma 4 changes head
//! dim, KV-head count, RoPE and the V source from layer to layer, which the
//! dense path's single `LayerConfig` can't express. Conventions and sources:
//! `docs/design/gemma4.md`.

use super::*;
use crate::error::ReflexError;

/// Gemma 4's per-layer matmul tensors (`blk.{i}.<name>.weight`): the ones
/// `REFLEX_QUANT_RESIDENT=1` keeps as raw Q8_0 blocks. Global layers have no
/// `attn_v`.
pub(super) const GEMMA4_MATMUL_TENSORS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// One Gemma 4 layer's weights (`blk.{i}.*` in the GGUF). The math is in
/// [`Model::forward_gemma4_layer`].
pub(super) struct Gemma4LayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    /// `None` on the global layers, which have no `attn_v`: V is then taken
    /// from `attn_k`'s projection (before K's norm and RoPE).
    pub(super) attn_v: Option<Weight>,
    pub(super) attn_output: Weight,
    pub(super) attn_q_norm: Weight,
    pub(super) attn_k_norm: Weight,
    pub(super) post_attention_norm: Weight,
    pub(super) ffn_norm: Weight,
    pub(super) ffn_gate: Weight,
    pub(super) ffn_up: Weight,
    pub(super) ffn_down: Weight,
    pub(super) post_ffw_norm: Weight,
    /// `layer_output_scale.weight` (one f32), read to the host at load; 1.0
    /// when the tensor is absent, as in llama.cpp.
    pub(super) output_scale: f32,
}

/// A loaded Gemma 4 model's own state, on top of the shared [`Model`] fields
/// (`token_embd`, `output_norm`, `lm_head`, `tokenizer` and the common
/// kernels). `Model::cfg`/`layers` are unused for this architecture.
pub(super) struct Gemma4Model {
    pub(super) cfg: Gemma4Config,
    pub(super) layers: Vec<Gemma4LayerWeights>,
    pub(super) kernels: Gemma4Kernels,
    /// `rope_freqs.weight`: the global layers' per-pair RoPE frequency
    /// divisors. `None` when there are no global layers.
    pub(super) rope_freqs: Option<Weight>,
    /// All ones, as long as the widest head: the sliding layers' RoPE
    /// factors (plain RoPE) and V's weightless RMSNorm.
    pub(super) ones: CudaSlice<f32>,
}

/// Per-layer `(k_cache, v_cache)`, each `[positions, num_kv_heads * head_dim]`
/// with that layer's own shape.
pub(super) type Gemma4KvCaches = Vec<(CudaSlice<f32>, CudaSlice<f32>)>;

/// Prefill result: encoded prompt ids (BOS included), the batched
/// `[rows, hidden_size]` hidden state, the filled KV caches, and the next
/// position.
pub(super) type Gemma4PrefillResult = (Vec<u32>, CudaSlice<f32>, Gemma4KvCaches, usize);

impl Model {
    /// Loads a Gemma 4 dense text model. See [`parse_gemma4_config`] for what
    /// is rejected. The tokenizer and cuBLAS handle are built on the usual
    /// background thread, overlapping the weight upload.
    pub(super) fn load_gemma4(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
    ) -> Result<Self, ReflexError> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let warm = policy.matrix_dtype;
            let init = scope.spawn(move || Self::load_background_init(file, init_device, warm));
            Self::load_gemma4_inner(device, file, policy, init)
        })
    }

    fn load_gemma4_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), ReflexError>>,
    ) -> Result<Self, ReflexError> {
        let g4_cfg = parse_gemma4_config(file)?;
        let attn_impl = AttnImpl::from_env()?;
        if attn_impl == AttnImpl::Legacy {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "REFLEX_ATTN_KERNEL=legacy is not supported for Gemma 4: the legacy kernels fix the attention scale at 1/sqrt(head_dim), and Gemma 4 uses 1.0"
            ));
        }
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
            &["rope_kernel", "rope_batch_kernel"],
        )?;
        let rope_batch_k = rope_fns.pop().expect("two kernels requested");
        let rope_k = rope_fns.pop().expect("two kernels requested");
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
        )?;
        let moe_scatter_add_k = elementwise_fns.pop().expect("five kernels requested");
        let moe_gather_k = elementwise_fns.pop().expect("five kernels requested");
        let sigmoid_gate_k = elementwise_fns.pop().expect("five kernels requested");
        let split_qg_k = elementwise_fns.pop().expect("five kernels requested");
        let add_k = elementwise_fns.pop().expect("five kernels requested");
        let kernels = Gemma4Kernels::load(&device)?;

        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;
        pipeline.prefetch_from(file);
        if profile_on {
            pipeline.profile = Some(LoadProfile::default());
        }

        let hidden_size = g4_cfg.hidden_size;
        let num_q_heads = g4_cfg.num_q_heads;
        let ffn = g4_cfg.ffn_hidden_size;
        let block_count = g4_cfg.layers.len();

        // Quantized-resident Q8_0 (docs/design/quantized-resident-weights.md,
        // "Q8_0 on the Gemma 4 path"): every Q8_0 layer matmul (and an untied
        // Q8_0 `output.weight`) stays as raw blocks in one arena; a tied Q8_0
        // LM head goes resident the same way on first use
        // (`Model::lm_head_resident`).
        let quant_on = quant_resident_enabled();
        let mut arena = if quant_on {
            let mut names: Vec<String> = (0..block_count)
                .flat_map(|i| {
                    GEMMA4_MATMUL_TENSORS
                        .iter()
                        .map(move |t| format!("blk.{i}.{t}.weight"))
                })
                .collect();
            names.push("output.weight".to_string());
            QuantArena::for_tensors(&device, file, &names, |name, ty| {
                ty == GgmlType::Q8_0 && !policy.keep_f32.contains(name)
            })?
        } else {
            None
        };
        let q8_0_embd = file
            .tensor_info("token_embd.weight")
            .is_some_and(|i| i.ggml_type == GgmlType::Q8_0);
        let q8_0_k = if quant_on && (arena.is_some() || q8_0_embd) {
            Some(Q8_0Kernels::load(&device)?)
        } else {
            if quant_on {
                eprintln!(
                    "note: REFLEX_QUANT_RESIDENT=1 keeps only Q8_0 weights on the Gemma 4 path, and this file has none; using {} weights",
                    policy.matrix_dtype
                );
            }
            None
        };
        let modules_ms = t_modules.elapsed().as_secs_f64() * 1000.0;
        let t_weights = std::time::Instant::now();

        // `matmul`: a layer matmul that may stay quantized; everything else
        // follows `policy`.
        let mut load_weight_kind = |name: &str, matmul: bool| -> Result<Weight, ReflexError> {
            match arena.as_mut() {
                Some(a) if matmul => load_weight_device_quant(
                    &mut pipeline,
                    &dequant_kernels,
                    policy,
                    file,
                    name,
                    a,
                    &[GgmlType::Q8_0],
                ),
                _ => load_weight_device(&mut pipeline, &dequant_kernels, policy, file, name),
            }
        };
        let mut layers = Vec::with_capacity(block_count);
        for (i, shape) in g4_cfg.layers.iter().enumerate() {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
            let name = |t: &str| format!("blk.{i}.{t}.weight");
            let q_dim = num_q_heads * shape.head_dim;
            let kv_dim = shape.num_kv_heads * shape.head_dim;
            let mut matrix = |t: &str, rows: usize, cols: usize| -> Result<Weight, ReflexError> {
                let w = load_weight_kind(&name(t), true)?;
                if w.shape.len() != 2 || w.shape[0] as usize != rows || w.shape[1] as usize != cols
                {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{} has shape {:?}, expected [{rows}, {cols}]",
                        name(t),
                        w.shape
                    ));
                }
                Ok(w)
            };
            let attn_q = matrix("attn_q", hidden_size, q_dim)?;
            let attn_k = matrix("attn_k", hidden_size, kv_dim)?;
            let attn_v = if file.tensor_info(&name("attn_v")).is_some() {
                Some(matrix("attn_v", hidden_size, kv_dim)?)
            } else {
                None
            };
            let attn_output = matrix("attn_output", q_dim, hidden_size)?;
            let ffn_gate = matrix("ffn_gate", hidden_size, ffn)?;
            let ffn_up = matrix("ffn_up", hidden_size, ffn)?;
            let ffn_down = matrix("ffn_down", ffn, hidden_size)?;
            let output_scale = match file.tensor_info(&name("layer_output_scale")) {
                Some(info) => {
                    let values = dequant::dequantize(info.ggml_type, file.tensor_bytes(info)?, 1)?;
                    *values.first().ok_or_else(|| {
                        crate::reflex_err!(Gguf, "{} is empty", name("layer_output_scale"))
                    })?
                }
                None => 1.0,
            };
            layers.push(Gemma4LayerWeights {
                attn_norm: load_weight_kind(&name("attn_norm"), false)?,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm: load_weight_kind(&name("attn_q_norm"), false)?,
                attn_k_norm: load_weight_kind(&name("attn_k_norm"), false)?,
                post_attention_norm: load_weight_kind(&name("post_attention_norm"), false)?,
                ffn_norm: load_weight_kind(&name("ffn_norm"), false)?,
                ffn_gate,
                ffn_up,
                ffn_down,
                post_ffw_norm: load_weight_kind(&name("post_ffw_norm"), false)?,
                output_scale,
            });
        }
        eprintln!();

        let rope_freqs = if g4_cfg.layers.iter().any(|l| !l.sliding) {
            Some(load_weight_kind("rope_freqs.weight", false)?)
        } else {
            None
        };
        let weights_enqueue_ms = t_weights.elapsed().as_secs_f64() * 1000.0;
        let max_head_dim = g4_cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(1);
        let ones = device
            .htod_sync_copy(&vec![1.0f32; max_head_dim])
            .map_err(|e| crate::gpu_err!(e, "gemma4 ones upload: {e}"))?;

        let t_embd = std::time::Instant::now();
        let token_embd_info = file
            .tensor_info("token_embd.weight")
            .ok_or_else(|| ReflexError::Gguf("missing weight 'token_embd.weight'".to_string()))?;
        let token_embd = LazyTokenEmbedding::new(
            token_embd_info.ggml_type,
            file.tensor_bytes_shared(token_embd_info)?,
            &token_embd_info.shape,
        )?;
        let token_embd_ms = t_embd.elapsed().as_secs_f64() * 1000.0;
        let output_norm = load_weight_kind("output_norm.weight", false)?;
        // Real Gemma 4 files tie the LM head to `token_embd`. Keep it lazy
        // like the dense path: `system1` only gathers the candidates' rows, so
        // a classifier never pays for the full 262,144-row upload.
        let lm_head = match file.tensor_info("output.weight") {
            Some(info) if info.ggml_type == GgmlType::Q8_0 && arena.is_some() => {
                LmHead::Resident(load_weight_device_quant(
                    &mut pipeline,
                    &dequant_kernels,
                    policy,
                    file,
                    "output.weight",
                    arena.as_mut().expect("checked by the guard"),
                    &[GgmlType::Q8_0],
                )?)
            }
            Some(info) => {
                let data = dequantize_matrix_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    policy.matrix_dtype,
                    info.ggml_type,
                    file.tensor_bytes(info)?,
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
        report_load_profile(
            &device,
            &mut pipeline,
            LoadTimes {
                quant_resident: arena.is_some(),
                modules_ms,
                weights_enqueue_ms,
                token_embd_ms,
                init_join_wait_ms,
            },
        )?;
        let f16_act_stats = device
            .alloc_zeros::<u32>(2)
            .map_err(|e| crate::gpu_err!(e, "alloc f16 activation stats: {e}"))?;
        let unused_cfg = LayerConfig {
            hidden_size,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rotary_dim: 1,
            ffn_hidden_size: 1,
            rope_base: 10_000.0,
            rmsnorm_eps: g4_cfg.rmsnorm_eps,
            rope_type: RopeType::Neox,
        };
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
            gemv_q4k_k: None,
            dequant_q4k_coalesced_k: None,
            dequant_q4k_coalesced_f16_k: None,
            gemv_q6k_k: None,
            q8_0_k,
            quant_scratch: RefCell::new(None),
            moe_gather_k,
            moe_scatter_add_k,
            attn_k,
            attn_prefill_k,
            attn_online_k,
            attn_online_combine_k,
            attn_impl,
            attn_tiled: TiledAttnKernels::from_env()?,
            attn_decode: DecodeAttnKernels::from_env()?,
            add_k,
            split_qg_k,
            sigmoid_gate_k,
            cfg: unused_cfg,
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
            hybrid: None,
            mla: None,
            gemma4: Some(Gemma4Model {
                cfg: g4_cfg,
                layers,
                kernels,
                rope_freqs,
                ones,
            }),
            granite: None,
        })
    }

    /// A projection over `rows` positions: the decode GEMV for one row (as
    /// the dense path does), a cuBLAS GEMM otherwise.
    fn gemma4_mm(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if rows == 1 {
            self.gemv(x, w)
        } else {
            self.gemm(x, w, rows)
        }
    }

    /// Token embeddings for `ids`, scaled by `sqrt(hidden_size)` (llama.cpp's
    /// `build_inp_embd(tok_embd, sqrtf(n_embd))`), uploaded as `[rows,
    /// hidden_size]`.
    fn gemma4_embed(&self, m: &Gemma4Model, ids: &[u32]) -> Result<CudaSlice<f32>, ReflexError> {
        let hidden_size = m.cfg.hidden_size;
        let scale = (hidden_size as f32).sqrt();
        let mut host = vec![0.0f32; ids.len() * hidden_size];
        for (row, &id) in ids.iter().enumerate() {
            let embd = self.token_embd.row(id)?;
            for (dst, &src) in host[row * hidden_size..(row + 1) * hidden_size]
                .iter_mut()
                .zip(embd.iter())
            {
                *dst = src * scale;
            }
        }
        self.device
            .htod_sync_copy(&host)
            .map_err(|e| crate::gpu_err!(e, "gemma4 embedding htod: {e}"))
    }

    /// Causal attention for `rows` query rows at `start_pos..`, against
    /// caches that already hold positions `0..start_pos + rows`, with
    /// Gemma 4's scale of 1.0. A sliding layer's window narrows the K/V
    /// views for a decode step (like [`Self::attention`]) and is masked per
    /// row inside the kernel for a prefill (like [`Self::attention_prefill`]).
    /// Prefill uses the tiled kernel ([`Gemma4Kernels::attention_prefill_tiled`]),
    /// decode the online one.
    #[allow(clippy::too_many_arguments)]
    fn gemma4_attention(
        &self,
        m: &Gemma4Model,
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        rows: usize,
        window: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let kv_stride = num_kv_heads * head_dim;
        let seq_len = start_pos + rows;
        let (first, window) = if rows == 1 && window > 0 {
            (seq_len.saturating_sub(window), 0)
        } else {
            (0, window)
        };
        let k = k_cache.slice(first * kv_stride..seq_len * kv_stride);
        let v = v_cache.slice(first * kv_stride..seq_len * kv_stride);
        let shape = OnlineAttnShape {
            rows,
            start_pos: start_pos - first,
            num_q_heads,
            group_size: num_q_heads / num_kv_heads,
            qk_dim: head_dim,
            v_dim: head_dim,
            k_pos_stride: kv_stride,
            k_head_stride: head_dim,
            v_pos_stride: kv_stride,
            v_head_stride: head_dim,
            scale: 1.0,
            window,
        };
        if let Some(out) = m
            .kernels
            .attention_prefill_tiled(&self.device, q, &k, &v, &shape)?
        {
            return Ok(out);
        }
        self.attention_online(q, &k, &v, shape)
    }

    /// One Gemma 4 layer over `rows` positions starting at `start_pos`
    /// (`rows == 1` is a decode step), writing their K/V into the layer's
    /// caches. llama.cpp's `llama_model_gemma4::graph`, per layer:
    ///
    /// ```text
    /// x = rmsnorm(h, attn_norm)
    /// q = rope(rmsnorm_per_head(Wq x, attn_q_norm))
    /// v = rmsnorm_per_head(Wv x  or, without attn_v, Wk x)   // no weight, no RoPE
    /// k = rope(rmsnorm_per_head(Wk x, attn_k_norm))
    /// a = rmsnorm(Wo attention(q, k, v; scale 1.0), post_attention_norm) + h
    /// f = W_down (gelu_tanh(W_gate y) * W_up y),  y = rmsnorm(a, ffn_norm)
    /// h' = (rmsnorm(f, post_ffw_norm) + a) * layer_output_scale
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_gemma4_layer(
        &self,
        m: &Gemma4Model,
        shape: &Gemma4LayerShape,
        layer: &Gemma4LayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let (hidden_size, eps) = (cfg.hidden_size, cfg.rmsnorm_eps);
        let (nq, nkv, hd) = (cfg.num_q_heads, shape.num_kv_heads, shape.head_dim);
        let freq_factors = if shape.sliding {
            &m.ones
        } else {
            m.rope_freqs
                .as_ref()
                .ok_or_else(|| {
                    ReflexError::Other("internal: gemma4 global layer without rope_freqs".into())
                })?
                .f32()?
        };

        let kind = if shape.sliding {
            LayerKind::Sliding
        } else {
            LayerKind::Global
        };
        let x = self.rmsnorm(&hidden, layer.attn_norm.f32()?, rows, hidden_size, eps)?;
        let q = self.gemma4_mm(&x, &layer.attn_q, rows)?;
        let k_raw = self.gemma4_mm(&x, &layer.attn_k, rows)?;
        let v_raw;
        let v_src = match &layer.attn_v {
            Some(w) => {
                v_raw = self.gemma4_mm(&x, w, rows)?;
                &v_raw
            }
            None => &k_raw,
        };
        prof_mark(PrefillPhase::QkvGemm, kind)?;

        let mut q = self.rmsnorm(&q, layer.attn_q_norm.f32()?, rows * nq, hd, eps)?;
        m.kernels.rope_ff(
            &mut q,
            freq_factors,
            start_pos,
            rows,
            nq,
            hd,
            shape.n_rot,
            shape.rope_base,
        )?;
        let v = self.rmsnorm(v_src, &m.ones, rows * nkv, hd, eps)?;
        let mut k = self.rmsnorm(&k_raw, layer.attn_k_norm.f32()?, rows * nkv, hd, eps)?;
        m.kernels.rope_ff(
            &mut k,
            freq_factors,
            start_pos,
            rows,
            nkv,
            hd,
            shape.n_rot,
            shape.rope_base,
        )?;
        prof_mark(PrefillPhase::NormRope, kind)?;

        let kv_stride = nkv * hd;
        let range = start_pos * kv_stride..(start_pos + rows) * kv_stride;
        self.device
            .dtod_copy(&k, &mut k_cache.slice_mut(range.clone()))
            .map_err(|e| crate::gpu_err!(e, "gemma4 kv-cache dtod k: {e}"))?;
        self.device
            .dtod_copy(&v, &mut v_cache.slice_mut(range))
            .map_err(|e| crate::gpu_err!(e, "gemma4 kv-cache dtod v: {e}"))?;
        prof_mark(PrefillPhase::KvCopy, kind)?;
        let window = if shape.sliding { cfg.sliding_window } else { 0 };
        let attn = self.gemma4_attention(
            m, &q, k_cache, v_cache, nq, nkv, hd, start_pos, rows, window,
        )?;
        prof_mark(PrefillPhase::Attention, kind)?;
        let o = self.gemma4_mm(&attn, &layer.attn_output, rows)?;
        let o = self.rmsnorm(&o, layer.post_attention_norm.f32()?, rows, hidden_size, eps)?;
        self.add_inplace(&mut hidden, &o)?;
        prof_mark(PrefillPhase::OProj, kind)?;

        let y = self.rmsnorm(&hidden, layer.ffn_norm.f32()?, rows, hidden_size, eps)?;
        let gate = self.gemma4_mm(&y, &layer.ffn_gate, rows)?;
        let up = self.gemma4_mm(&y, &layer.ffn_up, rows)?;
        let act = m.kernels.gelu_tanh_and_mul(&self.device, &gate, &up)?;
        let f = self.gemma4_mm(&act, &layer.ffn_down, rows)?;
        prof_mark(PrefillPhase::Ffn, kind)?;
        let f = self.rmsnorm(&f, layer.post_ffw_norm.f32()?, rows, hidden_size, eps)?;
        self.add_inplace(&mut hidden, &f)?;
        if layer.output_scale != 1.0 {
            m.kernels.scale(&mut hidden, layer.output_scale)?;
        }
        prof_mark(PrefillPhase::FfnTail, kind)?;
        Ok(hidden)
    }

    /// Allocates every layer's K and V cache for `total_len` positions, each
    /// sized from that layer's own shape.
    pub(super) fn alloc_gemma4_kv_caches(
        &self,
        m: &Gemma4Model,
        total_len: usize,
    ) -> Result<Gemma4KvCaches, ReflexError> {
        m.cfg
            .layers
            .iter()
            .map(|shape| {
                let n = total_len * shape.num_kv_heads * shape.head_dim;
                let k = self.device.alloc_zeros::<f32>(n)?;
                let v = self.device.alloc_zeros::<f32>(n)?;
                Ok((k, v))
            })
            .collect::<Result<_, cudarc::driver::DriverError>>()
            .map_err(|e| crate::gpu_err!(e, "alloc gemma4 kv caches: {e}"))
    }

    /// Encodes `prompt` (BOS prepended) and runs it through every layer as
    /// one batch, with KV room for `extra_headroom` more positions. Returns
    /// the whole `[rows, hidden_size]` hidden state; see [`Self::last_row`].
    pub(super) fn prefill_gemma4(
        &self,
        m: &Gemma4Model,
        prompt: &str,
        extra_headroom: usize,
    ) -> Result<Gemma4PrefillResult, ReflexError> {
        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.tokenizer.prompt_bos() {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if ids.is_empty() {
            return Err(ReflexError::InvalidInput(
                "encode produced no tokens".to_string(),
            ));
        }
        let rows = ids.len();
        crate::limits::check_positions_up_to(
            0,
            rows,
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;
        let mut caches = self.alloc_gemma4_kv_caches(m, rows + extra_headroom)?;
        let prof = prefill_profile_begin(&self.device)?;
        let mut hidden = self.gemma4_embed(m, &ids)?;
        prof_mark(PrefillPhase::Embed, LayerKind::Layer)?;
        for ((shape, layer), (k, v)) in m.cfg.layers.iter().zip(&m.layers).zip(&mut caches) {
            hidden = self.forward_gemma4_layer(m, shape, layer, hidden, 0, rows, k, v)?;
        }
        if let Some(p) = prof {
            p.finish(rows)?;
        }
        Ok((ids, hidden, caches, rows))
    }

    /// Embeds `token_id` and runs it through every layer at `position`.
    pub(super) fn forward_one_token_gemma4(
        &self,
        m: &Gemma4Model,
        token_id: u32,
        position: usize,
        caches: &mut Gemma4KvCaches,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut hidden = self.gemma4_embed(m, &[token_id])?;
        for ((shape, layer), (k, v)) in m.cfg.layers.iter().zip(&m.layers).zip(caches) {
            hidden = self.forward_gemma4_layer(m, shape, layer, hidden, position, 1, k, v)?;
        }
        Ok(hidden)
    }

    /// Final RMSNorm, the full-vocab LM head, then the logit soft-cap, as
    /// host logits (the same values llama.cpp reports).
    pub(super) fn gemma4_logits(
        &self,
        m: &Gemma4Model,
        hidden: &CudaSlice<f32>,
    ) -> Result<Vec<f32>, ReflexError> {
        let normed = self.rmsnorm(
            hidden,
            self.output_norm.f32()?,
            1,
            m.cfg.hidden_size,
            m.cfg.rmsnorm_eps,
        )?;
        let mut logits = self.gemv(&normed, self.lm_head_resident()?)?;
        if let Some(cap) = m.cfg.final_logit_softcap {
            m.kernels.softcap(&mut logits, cap)?;
        }
        self.device
            .dtoh_sync_copy(&logits)
            .map_err(|e| crate::gpu_err!(e, "gemma4 logits dtoh: {e}"))
    }

    /// The soft-capped logits of `token_ids` only, from `hidden` (System1's
    /// row gather: the tied LM head is never uploaded in full).
    fn gemma4_gather_logits(
        &self,
        m: &Gemma4Model,
        hidden: &CudaSlice<f32>,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, ReflexError> {
        let normed = self.rmsnorm(
            hidden,
            self.output_norm.f32()?,
            1,
            m.cfg.hidden_size,
            m.cfg.rmsnorm_eps,
        )?;
        let mut logits = self.gemv_gather_lm_head(&normed, token_ids)?;
        if let Some(cap) = m.cfg.final_logit_softcap {
            for l in &mut logits {
                *l = softcap_ref(*l, cap);
            }
        }
        Ok(logits)
    }

    /// Gemma 4 counterpart to `generate_mla_impl`: batched prefill, then one
    /// token at a time. No KV import (rejected by [`Model::generate`]).
    pub(super) fn generate_gemma4_impl(
        &self,
        m: &Gemma4Model,
        prompt: &str,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        mut on_first_token: impl FnMut(&[f32]),
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<(Vec<u32>, String), ReflexError> {
        if max_new_tokens == 0 {
            return Err(ReflexError::InvalidInput(
                "max_new_tokens must be at least 1".to_string(),
            ));
        }
        let (ids, hidden_batched, mut caches, mut position) =
            self.prefill_gemma4(m, prompt, max_new_tokens)?;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), m.cfg.hidden_size)?;

        let mut rng = crate::sampling::make_rng(sampling.seed);
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let first_logits = self.gemma4_logits(m, &hidden)?;
        let mut next_id = crate::sampling::sample(&first_logits, sampling, &mut rng)?;
        on_first_token(&first_logits);
        generated.push(next_id);
        on_token(
            next_id,
            &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
        );

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_gemma4(m, next_id, position, &mut caches)?;
            position += 1;
            let logits = self.gemma4_logits(m, &hidden)?;
            next_id = crate::sampling::sample(&logits, sampling, &mut rng)?;
            generated.push(next_id);
            on_token(
                next_id,
                &self.tokenizer.decode_stream(&mut pending_bytes, next_id),
            );
        }
        let text = self.tokenizer.decode(&generated);
        Ok((generated, text))
    }

    /// System1 for Gemma 4, the same way as the dense path: one batched
    /// prefill, every candidate's first token from one row gather, and a
    /// teacher-forced continuation for multi-token candidates, reusing the
    /// shared post-prompt KV headroom one candidate at a time. Each token's
    /// score is its soft-capped logit, as in llama.cpp's output.
    pub(super) fn system1_evaluate_gemma4(
        &self,
        m: &Gemma4Model,
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

        let (ids, hidden_batched, mut caches, base_position) =
            self.prefill_gemma4(m, prompt, max_len.saturating_sub(1))?;
        let hidden = self.last_row(&hidden_batched, ids.len(), m.cfg.hidden_size)?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemma4_gather_logits(m, &hidden, &first_tokens)?;

        for (i, ids) in resolved.iter().enumerate() {
            for (position, w) in (base_position..).zip(ids.windows(2)) {
                let h = self.forward_one_token_gemma4(m, w[0], position, &mut caches)?;
                scores[i] += self.gemma4_gather_logits(m, &h, &[w[1]])?[0];
            }
        }
        Self::finish_system1_response(candidates, resolved, scores, temperature)
    }
}

/// Threads per block for the elementwise kernels in `kernels_cuda/gemma4.cu`;
/// they are grid-stride loops, so the grid is capped.
const THREADS: u32 = 256;
const MAX_BLOCKS: u64 = 65_535;

fn grid_stride_cfg(n: u64) -> LaunchConfig {
    let blocks = n.div_ceil(THREADS as u64).clamp(1, MAX_BLOCKS) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The `kernels_cuda/gemma4.cu` kernels, loaded from one module, and the
/// tiled prefill attention kernels from `kernels_cuda/attention_tiled.cu`.
pub(super) struct Gemma4Kernels {
    rope_ff: AotKernel,
    gelu_tanh_and_mul: AotKernel,
    softcap: AotKernel,
    scale: AotKernel,
    attn_tiled_d256: AotKernel,
    attn_tiled_d512: AotKernel,
}

impl Gemma4Kernels {
    pub(super) fn load(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let mut fns = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_GEMMA4")),
            "gemma4",
            &[
                "rope_neox_ff_kernel",
                "gelu_tanh_and_mul_kernel",
                "softcap_kernel",
                "scale_kernel",
            ],
        )?
        .into_iter();
        let mut next = || {
            fns.next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module gemma4"))
        };
        let mut tiled = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED")),
            "attention_tiled",
            &["attention_tiled_d256_kernel", "attention_tiled_d512_kernel"],
        )?
        .into_iter();
        let mut next_tiled = || {
            tiled
                .next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module attention_tiled"))
        };
        Ok(Self {
            rope_ff: next()?,
            gelu_tanh_and_mul: next()?,
            softcap: next()?,
            scale: next()?,
            attn_tiled_d256: next_tiled()?,
            attn_tiled_d512: next_tiled()?,
        })
    }

    /// Batched-prefill attention with `attention_tiled.cu`, which reuses each
    /// K/V tile across a block of query rows and the query heads of a GQA
    /// group. `None` when the shape isn't one it covers (see
    /// [`attn_tiled_fits`]; head dims other than 256/512 too); the caller then
    /// uses [`Model::attention_online`]. Same layout, masking and output as
    /// that kernel; only the f32 summation order differs.
    pub(super) fn attention_prefill_tiled(
        &self,
        device: &Arc<CudaDevice>,
        q: &CudaSlice<f32>,
        k: &CudaView<f32>,
        v: &CudaView<f32>,
        s: &OnlineAttnShape,
    ) -> Result<Option<CudaSlice<f32>>, ReflexError> {
        let kernel = match s.qk_dim {
            256 => &self.attn_tiled_d256,
            512 => &self.attn_tiled_d512,
            _ => return Ok(None),
        };
        if !attn_tiled_fits(s) {
            return Ok(None);
        }
        launch_attention_tiled(device, kernel, q, k, v, s).map(Some)
    }

    /// NEOX RoPE with per-pair frequency factors, in place over `t`
    /// (`[rows, num_heads, head_dim]`, row `r` at position `start_pos + r`).
    /// `freq_factors` holds `n_dims / 2` divisors; all ones is plain RoPE.
    /// See [`rope_neox_ff_ref`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_ff(
        &self,
        t: &mut CudaSlice<f32>,
        freq_factors: &CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        num_heads: usize,
        head_dim: usize,
        n_dims: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        if n_dims == 0
            || !n_dims.is_multiple_of(2)
            || n_dims > head_dim
            || freq_factors.len() < n_dims / 2
        {
            return Err(crate::reflex_err!(
                Other,
                "rope_ff: n_dims {n_dims} must be even, at most head_dim {head_dim}, with {} freq factors (got {})",
                n_dims / 2,
                freq_factors.len()
            ));
        }
        let pairs = (rows * num_heads * (n_dims / 2)) as u64;
        unsafe {
            self.rope_ff
                .function
                .clone()
                .launch(
                    grid_stride_cfg(pairs),
                    (
                        t,
                        freq_factors,
                        start_pos as u32,
                        rows as u32,
                        num_heads as u32,
                        head_dim as u32,
                        n_dims as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_ff launch: {e}"))?;
        }
        Ok(())
    }

    /// GeGLU: `gelu_tanh(gate) * up`, elementwise over equal-length buffers.
    /// See [`gelu_tanh_ref`].
    pub(super) fn gelu_tanh_and_mul(
        &self,
        device: &Arc<CudaDevice>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let n = gate.len();
        if up.len() != n {
            return Err(crate::reflex_err!(
                Other,
                "gelu_tanh_and_mul: gate has {n} elements, up has {}",
                up.len()
            ));
        }
        let mut out = device
            .alloc_zeros::<f32>(n)
            .map_err(|e| crate::gpu_err!(e, "geglu alloc out: {e}"))?;
        unsafe {
            self.gelu_tanh_and_mul
                .function
                .clone()
                .launch(grid_stride_cfg(n as u64), (gate, up, &mut out, n as u64))
                .map_err(|e| crate::gpu_err!(e, "geglu launch: {e}"))?;
        }
        Ok(out)
    }

    /// In place: `x = cap * tanh(x / cap)` (the final logit soft-cap).
    pub(super) fn softcap(&self, x: &mut CudaSlice<f32>, cap: f32) -> Result<(), ReflexError> {
        let n = x.len() as u64;
        unsafe {
            self.softcap
                .function
                .clone()
                .launch(grid_stride_cfg(n), (x, n, cap))
                .map_err(|e| crate::gpu_err!(e, "softcap launch: {e}"))?;
        }
        Ok(())
    }

    /// In place: `x *= s` (a layer's `layer_output_scale`).
    pub(super) fn scale(&self, x: &mut CudaSlice<f32>, s: f32) -> Result<(), ReflexError> {
        let n = x.len() as u64;
        unsafe {
            self.scale
                .function
                .clone()
                .launch(grid_stride_cfg(n), (x, n, s))
                .map_err(|e| crate::gpu_err!(e, "scale launch: {e}"))?;
        }
        Ok(())
    }
}

/// Host reference for `rope_neox_ff_kernel`, same arithmetic in f32: pair
/// `(i, i + n_dims/2)` of each head rotates by
/// `position * (base^(-2/n_dims))^i / freq_factors[i]`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn rope_neox_ff_ref(
    t: &mut [f32],
    freq_factors: &[f32],
    start_pos: usize,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    n_dims: usize,
    base: f32,
) {
    let half = n_dims / 2;
    let theta_scale = base.powf(-2.0 / n_dims as f32);
    for row in 0..rows {
        let position = (start_pos + row) as f32;
        for head in 0..num_heads {
            let off = (row * num_heads + head) * head_dim;
            for i in 0..half {
                let theta = position * theta_scale.powf(i as f32) / freq_factors[i];
                let (sin_t, cos_t) = theta.sin_cos();
                let (x1, x2) = (t[off + i], t[off + i + half]);
                t[off + i] = x1 * cos_t - x2 * sin_t;
                t[off + i + half] = x1 * sin_t + x2 * cos_t;
            }
        }
    }
}

/// Host reference for the GELU in `gelu_tanh_and_mul_kernel` (tanh
/// approximation, ggml's constants).
#[cfg(test)]
pub(super) fn gelu_tanh_ref(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Host reference for `softcap_kernel`.
pub(super) fn softcap_ref(x: f32, cap: f32) -> f32 {
    (x * (1.0 / cap)).tanh() * cap
}
