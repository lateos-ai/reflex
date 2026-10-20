//! Granite 4.0-H (`granitehybrid`): Mamba-2 SSM layers plus a few attention
//! layers, every layer followed by a SwiGLU FFN. Loading, the forward pass and
//! the Mamba-2 kernel wrappers. A separate path like `gemma4`, because its
//! attention (no RoPE, a muP scale) and its SSM mixer share nothing with the
//! Qwen3.5 hybrid layers but the generic kernels. Math, sources and scope:
//! `docs/design/mamba2-hybrid.md` and `reference/mamba2_reference.rs`.
//!
//! Prefill runs every layer over all prompt rows at once. Projections are
//! batched GEMMs; the SSM recurrence (conv and scan) walks the rows in order
//! inside one kernel launch each. There is no chunked (SSD) parallel scan.

use super::*;
use crate::error::ReflexError;

/// One Mamba-2 mixer's weights (`blk.{i}.ssm_*`). `ssm_a` is already
/// `-exp(A_log)` (llama.cpp's converter applies it).
pub(super) struct Mamba2Weights {
    /// `[hidden, d_in_proj]`: z, xBC, dt.
    pub(super) ssm_in: Weight,
    /// `[d_conv, conv_dim]`, f32.
    pub(super) conv_w: Weight,
    pub(super) conv_b: Weight,
    pub(super) dt_bias: Weight,
    pub(super) a: Weight,
    pub(super) d: Weight,
    /// `[d_inner]` (`[d_inner / n_group, n_group]` in the GGUF), f32.
    pub(super) norm: Weight,
    /// `[d_inner, hidden]`.
    pub(super) ssm_out: Weight,
}

/// One attention mixer's weights: plain GQA, no biases, no Q/K norm.
pub(super) struct GraniteAttnWeights {
    pub(super) attn_q: Weight,
    pub(super) attn_k: Weight,
    pub(super) attn_v: Weight,
    pub(super) attn_output: Weight,
}

pub(super) enum GraniteMixer {
    Ssm(Box<Mamba2Weights>),
    Attn(Box<GraniteAttnWeights>),
}

/// One Granite 4.0-H layer: `attn_norm`, the mixer, then a SwiGLU FFN.
pub(super) struct GraniteLayerWeights {
    pub(super) attn_norm: Weight,
    pub(super) mixer: GraniteMixer,
    pub(super) ffn_norm: Weight,
    pub(super) ffn_gate: Weight,
    pub(super) ffn_up: Weight,
    pub(super) ffn_down: Weight,
}

/// A loaded Granite 4.0-H model's own state, on top of the shared [`Model`]
/// fields (`token_embd`, `output_norm`, `lm_head`, `tokenizer` and the common
/// kernels). `Model::cfg`/`layers` are unused for this architecture.
pub(super) struct GraniteHybridModel {
    pub(super) cfg: GraniteHybridConfig,
    pub(super) layers: Vec<GraniteLayerWeights>,
    pub(super) kernels: Mamba2Kernels,
}

/// One layer's carried state, matching its [`GraniteMixer`]: K/V caches
/// (`[positions, num_kv_heads * head_dim]`) for attention, the fixed-size conv
/// window (`[d_conv - 1, conv_dim]`) and SSM state (`[H, P, N]`) for Mamba-2.
pub(super) enum GraniteLayerState {
    Attn {
        k_cache: CudaSlice<f32>,
        v_cache: CudaSlice<f32>,
    },
    Ssm {
        conv_state: CudaSlice<f32>,
        ssm_state: CudaSlice<f32>,
    },
}

/// Prefill result: encoded prompt ids, the batched `[rows, hidden_size]`
/// hidden state, the per-layer states, and the next position.
pub(super) type GranitePrefillResult = (Vec<u32>, CudaSlice<f32>, Vec<GraniteLayerState>, usize);

impl Model {
    /// Loads a Granite 4.0-H model. See [`parse_granite_hybrid_config`] for
    /// what is rejected. The tokenizer and cuBLAS handle are built on the
    /// usual background thread, overlapping the weight upload.
    pub(super) fn load_granite_hybrid(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
    ) -> Result<Self, ReflexError> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let warm = policy.matrix_dtype;
            let init = scope.spawn(move || Self::load_background_init(file, init_device, warm));
            Self::load_granite_hybrid_inner(device, file, policy, init)
        })
    }

    fn load_granite_hybrid_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        policy: &WeightPolicy,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), ReflexError>>,
    ) -> Result<Self, ReflexError> {
        let cfg = parse_granite_hybrid_config(file)?;
        let attn_impl = AttnImpl::from_env()?;
        if attn_impl == AttnImpl::Legacy {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "REFLEX_ATTN_KERNEL=legacy is not supported for granitehybrid: the legacy kernels fix the attention scale at 1/sqrt(head_dim), and Granite 4.0-H uses its own attention.scale"
            ));
        }
        if quant_resident_enabled() {
            eprintln!(
                "note: REFLEX_QUANT_RESIDENT=1 is not supported for granitehybrid; using {} weights",
                policy.matrix_dtype
            );
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
        let kernels = Mamba2Kernels::load(&device)?;

        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;
        pipeline.prefetch_from(file);
        if profile_on {
            pipeline.profile = Some(LoadProfile::default());
        }
        let modules_ms = t_modules.elapsed().as_secs_f64() * 1000.0;
        let t_weights = std::time::Instant::now();

        let hidden = cfg.hidden_size;
        let ssm = cfg.ssm;
        let q_dim = cfg.num_q_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let ffn = cfg.ffn_hidden_size;
        let block_count = cfg.ssm_layers.len();
        let mut layers = Vec::with_capacity(block_count);
        for (i, &is_ssm) in cfg.ssm_layers.iter().enumerate() {
            eprint!("\rLoading weights: layer {}/{block_count}", i + 1);
            let mut load = |t: &str, elements: usize| -> Result<Weight, ReflexError> {
                let name = format!("blk.{i}.{t}");
                let w = load_weight_device(&mut pipeline, &dequant_kernels, policy, file, &name)?;
                let n: u64 = w.shape.iter().product();
                if n as usize != elements {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{name} has shape {:?} ({n} elements), expected {elements}",
                        w.shape
                    ));
                }
                Ok(w)
            };
            let attn_norm = load("attn_norm.weight", hidden)?;
            let mixer = if is_ssm {
                GraniteMixer::Ssm(Box::new(Mamba2Weights {
                    ssm_in: load("ssm_in.weight", hidden * ssm.d_in_proj())?,
                    conv_w: load("ssm_conv1d.weight", ssm.d_conv * ssm.conv_dim())?,
                    conv_b: load("ssm_conv1d.bias", ssm.conv_dim())?,
                    dt_bias: load("ssm_dt.bias", ssm.n_head)?,
                    a: load("ssm_a", ssm.n_head)?,
                    d: load("ssm_d", ssm.n_head)?,
                    norm: load("ssm_norm.weight", ssm.d_inner)?,
                    ssm_out: load("ssm_out.weight", ssm.d_inner * hidden)?,
                }))
            } else {
                GraniteMixer::Attn(Box::new(GraniteAttnWeights {
                    attn_q: load("attn_q.weight", hidden * q_dim)?,
                    attn_k: load("attn_k.weight", hidden * kv_dim)?,
                    attn_v: load("attn_v.weight", hidden * kv_dim)?,
                    attn_output: load("attn_output.weight", q_dim * hidden)?,
                }))
            };
            layers.push(GraniteLayerWeights {
                attn_norm,
                mixer,
                ffn_norm: load("ffn_norm.weight", hidden)?,
                ffn_gate: load("ffn_gate.weight", hidden * ffn)?,
                ffn_up: load("ffn_up.weight", hidden * ffn)?,
                ffn_down: load("ffn_down.weight", ffn * hidden)?,
            });
        }
        eprintln!();
        let weights_enqueue_ms = t_weights.elapsed().as_secs_f64() * 1000.0;

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
        let output_norm = load_weight_device(
            &mut pipeline,
            &dequant_kernels,
            policy,
            file,
            "output_norm.weight",
        )?;
        // Granite 4.0-H ties the LM head to `token_embd`; keep it lazy, as the
        // dense path does, so `system1` only gathers the candidates' rows.
        let lm_head = match file.tensor_info("output.weight") {
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
                quant_resident: false,
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
            hidden_size: hidden,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rotary_dim: 1,
            ffn_hidden_size: 1,
            rope_base: 10_000.0,
            rmsnorm_eps: cfg.rmsnorm_eps,
            rope_type: RopeType::Neox,
        };
        Ok(Model {
            device: device.clone(),
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
            gemv_qkv_k: aot::load_kernel(
                &device,
                include_bytes!(env!("REFLEX_KERNEL_GEMV_FUSED")),
                "gemv_fused",
                "gemv_qkv_kernel",
            )?,
            gemv_qkv_f16_k: aot::load_kernel(
                &device,
                include_bytes!(env!("REFLEX_KERNEL_GEMV_FUSED")),
                "gemv_fused",
                "gemv_qkv_f16_kernel",
            )?,
            gemv_gate_up_k: aot::load_kernel(
                &device,
                include_bytes!(env!("REFLEX_KERNEL_GEMV_FUSED")),
                "gemv_fused",
                "gemv_gate_up_kernel",
            )?,
            gemv_gate_up_f16_k: aot::load_kernel(
                &device,
                include_bytes!(env!("REFLEX_KERNEL_GEMV_FUSED")),
                "gemv_fused",
                "gemv_gate_up_f16_kernel",
             )?,
            gemv_q4k_k: None,
            gemv_q4k_r1_k: None,
            dequant_q4k_coalesced_k: None,
            dequant_q4k_coalesced_f16_k: None,
            dequant_q6k_coalesced_k: None,
            dequant_q6k_coalesced_f16_k: None,
            gemv_q6k_k: None,
            gemv_q6k_r1_k: None,
            q8_0_k: None,
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
            decode_ops: DecodeOpKernels::from_env()?,
            decode_ws: DecodeWorkspaceSlot::from_env()?,
            kv_dtype: Cell::new(kv_dtype_from_env()?),
            kv_f16: KvF16Kernels::new(),
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
            grammar_index: Default::default(),
            hybrid: None,
            mla: None,
            gemma4: None,
            granite: Some(GraniteHybridModel {
                cfg,
                layers,
                kernels,
            }),
        })
    }

    /// A projection over `rows` positions: the decode GEMV for one row, a
    /// cuBLAS GEMM otherwise (the same split every other path uses).
    fn granite_mm(
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

    /// Token embeddings for `ids` times `embedding_scale` (llama.cpp's
    /// `ggml_scale` in `build_inp_embd`), uploaded as `[rows, hidden_size]`.
    fn granite_embed(
        &self,
        m: &GraniteHybridModel,
        ids: &[u32],
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let hidden = m.cfg.hidden_size;
        let scale = m.cfg.embedding_scale;
        let mut host = vec![0.0f32; ids.len() * hidden];
        for (row, &id) in ids.iter().enumerate() {
            let embd = self.token_embd.row(id)?;
            for (dst, &src) in host[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .zip(embd.iter())
            {
                *dst = src * scale;
            }
        }
        self.device
            .htod_sync_copy(&host)
            .map_err(|e| crate::gpu_err!(e, "granite embedding htod: {e}"))
    }

    /// One Mamba-2 mixer over `rows` positions (the input already through
    /// `attn_norm`), advancing `conv_state`/`ssm_state`. Steps 1-8 of
    /// `reference/mamba2_reference.rs`.
    pub(super) fn granite_ssm_mixer(
        &self,
        m: &GraniteHybridModel,
        w: &Mamba2Weights,
        x: &CudaSlice<f32>,
        rows: usize,
        conv_state: &mut CudaSlice<f32>,
        ssm_state: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let ssm = &m.cfg.ssm;
        let dims = Mamba2Dims::new(ssm, rows, m.cfg.rmsnorm_eps);
        let zxbcdt = self.granite_mm(x, &w.ssm_in, rows)?;
        let k = &m.kernels;
        let conv = k.conv(
            &self.device,
            &zxbcdt,
            w.conv_w.f32()?,
            w.conv_b.f32()?,
            conv_state,
            dims,
        )?;
        let y = k.scan(
            &self.device,
            &conv,
            &zxbcdt,
            w.dt_bias.f32()?,
            w.a.f32()?,
            ssm_state,
            dims,
        )?;
        let gated = k.gate_norm(
            &self.device,
            &y,
            &conv,
            &zxbcdt,
            w.d.f32()?,
            w.norm.f32()?,
            dims,
        )?;
        self.granite_mm(&gated, &w.ssm_out, rows)
    }

    /// One attention mixer over `rows` positions at `start_pos..` (input
    /// already through `attn_norm`): Q/K/V projections, no RoPE, K/V written
    /// into the caches, causal GQA with `attention.scale`, output projection.
    #[allow(clippy::too_many_arguments)]
    fn granite_attn_mixer(
        &self,
        m: &GraniteHybridModel,
        w: &GraniteAttnWeights,
        x: &CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let (nq, nkv, hd) = (cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim);
        let q = self.granite_mm(x, &w.attn_q, rows)?;
        let k = self.granite_mm(x, &w.attn_k, rows)?;
        let v = self.granite_mm(x, &w.attn_v, rows)?;
        prof_mark(PrefillPhase::QkvGemm, LayerKind::Attn)?;
        let kv_stride = nkv * hd;
        let range = start_pos * kv_stride..(start_pos + rows) * kv_stride;
        self.device
            .dtod_copy(&k, &mut k_cache.slice_mut(range.clone()))
            .map_err(|e| crate::gpu_err!(e, "granite kv-cache dtod k: {e}"))?;
        self.device
            .dtod_copy(&v, &mut v_cache.slice_mut(range))
            .map_err(|e| crate::gpu_err!(e, "granite kv-cache dtod v: {e}"))?;
        prof_mark(PrefillPhase::KvCopy, LayerKind::Attn)?;
        let filled = 0..(start_pos + rows) * kv_stride;
        let attn = self.attention_online(
            &q,
            // Granite always keeps an f32 cache (REFLEX_KV_CACHE=f16 is rejected).
            &KvView::F32(k_cache.slice(filled.clone())),
            &KvView::F32(v_cache.slice(filled)),
            OnlineAttnShape {
                rows,
                start_pos,
                num_q_heads: nq,
                group_size: nq / nkv,
                qk_dim: hd,
                v_dim: hd,
                k_pos_stride: kv_stride,
                k_head_stride: hd,
                v_pos_stride: kv_stride,
                v_head_stride: hd,
                scale: cfg.attention_scale,
                window: 0,
            },
        )?;
        prof_mark(PrefillPhase::Attention, LayerKind::Attn)?;
        self.granite_mm(&attn, &w.attn_output, rows)
    }

    /// One Granite 4.0-H layer over `rows` positions starting at `start_pos`
    /// (`rows == 1` is a decode step). llama.cpp's `granite-hybrid.cpp`:
    ///
    /// ```text
    /// h = h + mixer(rmsnorm(h, attn_norm)) * residual_scale
    /// y = rmsnorm(h, ffn_norm)
    /// h = h + ffn_down(silu(ffn_gate y) * ffn_up y) * residual_scale
    /// ```
    pub(super) fn forward_granite_layer(
        &self,
        m: &GraniteHybridModel,
        layer: &GraniteLayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        state: &mut GraniteLayerState,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let (h, eps) = (cfg.hidden_size, cfg.rmsnorm_eps);
        let x = self.rmsnorm(&hidden, layer.attn_norm.f32()?, rows, h, eps)?;
        let (kind, mixer_phase) = match layer.mixer {
            GraniteMixer::Ssm(_) => (LayerKind::Ssm, PrefillPhase::Mixer),
            GraniteMixer::Attn(_) => (LayerKind::Attn, PrefillPhase::OProj),
        };
        let mixed = match (&layer.mixer, state) {
            (
                GraniteMixer::Ssm(w),
                GraniteLayerState::Ssm {
                    conv_state,
                    ssm_state,
                },
            ) => self.granite_ssm_mixer(m, w, &x, rows, conv_state, ssm_state)?,
            (GraniteMixer::Attn(w), GraniteLayerState::Attn { k_cache, v_cache }) => {
                self.granite_attn_mixer(m, w, &x, start_pos, rows, k_cache, v_cache)?
            }
            _ => {
                return Err(ReflexError::Other(
                    "internal error: granitehybrid layer/state kind mismatch".to_string(),
                ))
            }
        };
        m.kernels
            .scaled_add(&mut hidden, &mixed, cfg.residual_scale)?;
        prof_mark(mixer_phase, kind)?;

        let y = self.rmsnorm(&hidden, layer.ffn_norm.f32()?, rows, h, eps)?;
        let gate = self.granite_mm(&y, &layer.ffn_gate, rows)?;
        let up = self.granite_mm(&y, &layer.ffn_up, rows)?;
        let act = self.silu_and_mul(&gate, &up, rows * cfg.ffn_hidden_size)?;
        let f = self.granite_mm(&act, &layer.ffn_down, rows)?;
        m.kernels.scaled_add(&mut hidden, &f, cfg.residual_scale)?;
        prof_mark(PrefillPhase::Ffn, kind)?;
        Ok(hidden)
    }

    /// Fresh per-layer state: K/V caches for `total_len` positions on the
    /// attention layers, zeroed conv and SSM states on the SSM layers.
    pub(super) fn alloc_granite_states(
        &self,
        m: &GraniteHybridModel,
        total_len: usize,
    ) -> Result<Vec<GraniteLayerState>, ReflexError> {
        let kv = total_len * m.cfg.num_kv_heads * m.cfg.head_dim;
        let ssm = &m.cfg.ssm;
        m.cfg
            .ssm_layers
            .iter()
            .map(|&is_ssm| {
                Ok(if is_ssm {
                    GraniteLayerState::Ssm {
                        conv_state: self.device.alloc_zeros::<f32>(ssm.conv_state_len())?,
                        ssm_state: self.device.alloc_zeros::<f32>(ssm.ssm_state_len())?,
                    }
                } else {
                    GraniteLayerState::Attn {
                        k_cache: self.device.alloc_zeros::<f32>(kv)?,
                        v_cache: self.device.alloc_zeros::<f32>(kv)?,
                    }
                })
            })
            .collect::<Result<_, cudarc::driver::DriverError>>()
            .map_err(|e| crate::gpu_err!(e, "alloc granitehybrid layer states: {e}"))
    }

    /// Encodes `prompt` (BOS per `add_bos_token`; Granite 4.0-H adds none)
    /// for `prefill_granite`/tests.
    pub(super) fn granite_prompt_ids(&self, prompt: &str) -> Result<Vec<u32>, ReflexError> {
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
        Ok(ids)
    }

    /// Runs the whole prompt through every layer as one batch, with KV room
    /// for `extra_headroom` more positions. Returns the whole
    /// `[rows, hidden_size]` hidden state; see [`Self::last_row`].
    pub(super) fn prefill_granite(
        &self,
        m: &GraniteHybridModel,
        prompt: &str,
        extra_headroom: usize,
    ) -> Result<GranitePrefillResult, ReflexError> {
        let ids = self.granite_prompt_ids(prompt)?;
        let rows = ids.len();
        crate::limits::check_positions_up_to(
            0,
            rows,
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;
        let mut states = self.alloc_granite_states(m, rows + extra_headroom)?;
        let prof = prefill_profile_begin(&self.device)?;
        let mut hidden = self.granite_embed(m, &ids)?;
        prof_mark(PrefillPhase::Embed, LayerKind::Layer)?;
        for (layer, state) in m.layers.iter().zip(&mut states) {
            hidden = self.forward_granite_layer(m, layer, hidden, 0, rows, state)?;
        }
        if let Some(p) = prof {
            p.finish(rows)?;
        }
        Ok((ids, hidden, states, rows))
    }

    /// Embeds `token_id` and runs it through every layer at `position`.
    pub(super) fn forward_one_token_granite(
        &self,
        m: &GraniteHybridModel,
        token_id: u32,
        position: usize,
        states: &mut [GraniteLayerState],
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut hidden = self.granite_embed(m, &[token_id])?;
        for (layer, state) in m.layers.iter().zip(states) {
            hidden = self.forward_granite_layer(m, layer, hidden, position, 1, state)?;
        }
        Ok(hidden)
    }

    /// Final RMSNorm, the full-vocab LM head, then `1 / logit_scale`, as host
    /// logits (the values llama.cpp reports).
    pub(super) fn granite_logits(
        &self,
        m: &GraniteHybridModel,
        hidden: &CudaSlice<f32>,
    ) -> Result<Vec<f32>, ReflexError> {
        let mut logits = self.lm_head_logits(hidden, m.cfg.hidden_size, m.cfg.rmsnorm_eps)?;
        Self::granite_scale_logits(m, &mut logits);
        Ok(logits)
    }

    /// llama.cpp's `ggml_scale(cur, 1.0f / f_logit_scale)`: one f32 multiply
    /// per logit by the f32 reciprocal.
    fn granite_scale_logits(m: &GraniteHybridModel, logits: &mut [f32]) {
        if let Some(s) = m.cfg.logit_scale {
            let inv = 1.0f32 / s;
            for l in logits {
                *l *= inv;
            }
        }
    }

    /// Granite 4.0-H counterpart to `generate_gemma4_impl`: batched prefill,
    /// then one token at a time. No KV import (rejected by [`Model::generate`]).
    pub(super) fn generate_granite_impl(
        &self,
        m: &GraniteHybridModel,
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
        let (ids, hidden_batched, mut states, mut position) =
            self.prefill_granite(m, prompt, max_new_tokens)?;
        let mut hidden = self.last_row(&hidden_batched, ids.len(), m.cfg.hidden_size)?;

        let mut picker = self.token_picker(sampling)?;
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut first_logits = self.granite_logits(m, &hidden)?;
        let mut next_id = picker.pick(&mut first_logits)?;
        on_first_token(&first_logits);
        generated.push(next_id);
        on_token(
            next_id,
            &picker.stream_text(&self.tokenizer, &mut pending_bytes, next_id),
        );

        while generated.len() < max_new_tokens && Some(next_id) != self.tokenizer.eos_token_id {
            hidden = self.forward_one_token_granite(m, next_id, position, &mut states)?;
            position += 1;
            let mut logits = self.granite_logits(m, &hidden)?;
            next_id = picker.pick(&mut logits)?;
            generated.push(next_id);
            on_token(
                next_id,
                &picker.stream_text(&self.tokenizer, &mut pending_bytes, next_id),
            );
        }
        let text = picker.finish(&self.tokenizer, &generated)?;
        Ok((generated, text))
    }

    /// System1 for Granite 4.0-H: one batched prefill and every candidate's
    /// first token from one row gather. **Single-token candidates only**, for
    /// the reason `system1_evaluate_hybrid` gives: the SSM state is a running
    /// accumulator, so scoring one multi-token candidate would leave the
    /// shared post-prompt state advanced for the next.
    pub(super) fn system1_evaluate_granite(
        &self,
        m: &GraniteHybridModel,
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
            .map(|c| self.resolve_candidate_token_ids(prompt, &c.text, true))
            .collect::<Result<_, _>>()?;
        if resolved.iter().any(|ids| ids.len() > 1) {
            return Err(ReflexError::InvalidInput(
                "system1_evaluate: granitehybrid models currently support single-token candidates only"
                    .to_string(),
            ));
        }
        let (ids, hidden_batched, _states, _position) = self.prefill_granite(m, prompt, 0)?;
        let hidden = self.last_row(&hidden_batched, ids.len(), m.cfg.hidden_size)?;
        let normed = self.rmsnorm(
            &hidden,
            self.output_norm.f32()?,
            1,
            m.cfg.hidden_size,
            m.cfg.rmsnorm_eps,
        )?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemv_gather_lm_head(&normed, &first_tokens)?;
        Self::granite_scale_logits(m, &mut scores);
        Self::finish_system1_response(candidates, resolved, scores, temperature)
    }
}

/// `kernels_cuda/mamba2.cu`'s shape arguments. Must match `struct Mamba2Dims`
/// there field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct Mamba2Dims {
    rows: u32,
    in_stride: u32,
    c_dim: u32,
    d_inner: u32,
    n_head: u32,
    head_dim: u32,
    n_group: u32,
    d_state: u32,
    d_conv: u32,
    eps: f32,
}

// SAFETY: a plain-old-data #[repr(C)] struct, passed to the kernel by value.
unsafe impl DeviceRepr for Mamba2Dims {}

impl Mamba2Dims {
    pub(super) fn new(ssm: &Mamba2Config, rows: usize, eps: f32) -> Self {
        Self {
            rows: rows as u32,
            in_stride: ssm.d_in_proj() as u32,
            c_dim: ssm.conv_dim() as u32,
            d_inner: ssm.d_inner as u32,
            n_head: ssm.n_head as u32,
            head_dim: ssm.head_dim() as u32,
            n_group: ssm.n_group as u32,
            d_state: ssm.d_state as u32,
            d_conv: ssm.d_conv as u32,
            eps,
        }
    }
}

/// Warps per block in `mamba2_scan_kernel` (one warp per `(head, p)`).
const SCAN_WARPS_PER_BLOCK: u32 = 4;
const THREADS: u32 = 256;
const MAX_BLOCKS: u64 = 65_535;

/// The `kernels_cuda/mamba2.cu` kernels, loaded from one module.
pub(super) struct Mamba2Kernels {
    conv: AotKernel,
    scan: AotKernel,
    gate_norm: AotKernel,
    scaled_add: AotKernel,
}

impl Mamba2Kernels {
    pub(super) fn load(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let mut fns = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_MAMBA2")),
            "mamba2",
            &[
                "mamba2_conv_kernel",
                "mamba2_scan_kernel",
                "mamba2_gate_norm_kernel",
                "scaled_add_kernel",
            ],
        )?
        .into_iter();
        let mut next = || {
            fns.next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module mamba2"))
        };
        Ok(Self {
            conv: next()?,
            scan: next()?,
            gate_norm: next()?,
            scaled_add: next()?,
        })
    }

    /// Step 2 over `dims.rows` positions: `[rows, conv_dim]` post-SiLU conv
    /// output from `zxbcdt`'s xBC columns, advancing `state` in place.
    pub(super) fn conv(
        &self,
        device: &Arc<CudaDevice>,
        zxbcdt: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        dims: Mamba2Dims,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let c_dim = dims.c_dim;
        let mut out = device
            .alloc_zeros::<f32>(dims.rows as usize * c_dim as usize)
            .map_err(|e| crate::gpu_err!(e, "mamba2 conv alloc: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (c_dim.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.conv
                .function
                .clone()
                .launch(cfg, (zxbcdt, w, b, state, &mut out, dims))
                .map_err(|e| crate::gpu_err!(e, "mamba2 conv launch: {e}"))?;
        }
        Ok(out)
    }

    /// Steps 4-5 over `dims.rows` positions: `[rows, d_inner]` scan output
    /// (no D skip), advancing `state` (`[H, P, N]`) in place.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan(
        &self,
        device: &Arc<CudaDevice>,
        conv: &CudaSlice<f32>,
        zxbcdt: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        dims: Mamba2Dims,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if dims.d_state == 0 || !dims.d_state.is_multiple_of(32) || dims.d_state > 256 {
            return Err(crate::reflex_err!(
                Other,
                "mamba2 scan: d_state {} must be a multiple of 32 up to 256",
                dims.d_state
            ));
        }
        let mut y = device
            .alloc_zeros::<f32>(dims.rows as usize * dims.d_inner as usize)
            .map_err(|e| crate::gpu_err!(e, "mamba2 scan alloc: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (dims.d_inner.div_ceil(SCAN_WARPS_PER_BLOCK), 1, 1),
            block_dim: (SCAN_WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.scan
                .function
                .clone()
                .launch(cfg, (conv, zxbcdt, dt_bias, a, state, &mut y, dims))
                .map_err(|e| crate::gpu_err!(e, "mamba2 scan launch: {e}"))?;
        }
        Ok(y)
    }

    /// Steps 6-7 over `dims.rows` positions: D skip, `silu(z)` gate, grouped
    /// RMSNorm times `norm_w`. Returns `[rows, d_inner]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gate_norm(
        &self,
        device: &Arc<CudaDevice>,
        y: &CudaSlice<f32>,
        conv: &CudaSlice<f32>,
        zxbcdt: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        norm_w: &CudaSlice<f32>,
        dims: Mamba2Dims,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut out = device
            .alloc_zeros::<f32>(dims.rows as usize * dims.d_inner as usize)
            .map_err(|e| crate::gpu_err!(e, "mamba2 gate_norm alloc: {e}"))?;
        let blocks = (dims.rows as u64 * dims.n_group as u64).clamp(1, MAX_BLOCKS) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.gate_norm
                .function
                .clone()
                .launch(cfg, (y, conv, zxbcdt, d, norm_w, &mut out, dims))
                .map_err(|e| crate::gpu_err!(e, "mamba2 gate_norm launch: {e}"))?;
        }
        Ok(out)
    }

    /// In place: `h += m * s`, the product rounded on its own (Granite's
    /// `residual_scale` before the residual add).
    pub(super) fn scaled_add(
        &self,
        h: &mut CudaSlice<f32>,
        m: &CudaSlice<f32>,
        s: f32,
    ) -> Result<(), ReflexError> {
        let n = h.len() as u64;
        if m.len() as u64 != n {
            return Err(crate::reflex_err!(
                Other,
                "scaled_add: {n} vs {} elements",
                m.len()
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(THREADS as u64).clamp(1, MAX_BLOCKS) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.scaled_add
                .function
                .clone()
                .launch(cfg, (h, m, s, n))
                .map_err(|e| crate::gpu_err!(e, "scaled_add launch: {e}"))?;
        }
        Ok(())
    }
}
