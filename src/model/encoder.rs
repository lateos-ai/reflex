//! Embedding (encoder) models on the dense path: bidirectional attention, YaRN
//! RoPE, the query temperature scale and mean pooling, behind [`Model::embed`].
//! Nemotron-3.5-Embed (`mistral3`, `attention.causal = false`) is the model this
//! was built for. See docs/design/embeddings.md.

use super::*;
use crate::error::ReflexError;

/// One input's embedding, from [`Model::embed`].
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The mean-pooled, L2-normalized final hidden state (`hidden_size` floats).
    pub vector: Vec<f32>,
    /// Tokens the input encoded to: the mean runs over all of them.
    pub prompt_tokens: usize,
}

/// An encoder's settings and kernels. `Some` on [`Model`] iff the GGUF says
/// `<arch>.attention.causal = false` ([`is_encoder`]).
pub(super) struct EncoderModel {
    pub(super) cfg: EncoderConfig,
    /// `rope_norm_yarn_batch_kernel` (`kernels_cuda/rope.cu`), iff `cfg.yarn`.
    rope_yarn_batch_k: Option<AotKernel>,
    /// `kernels_cuda/encoder.cu`.
    attn_bidir_k: AotKernel,
    temp_scale_k: AotKernel,
    mean_pool_k: AotKernel,
    /// `kernels_cuda/attention_tiled_bidir.cu`, `[d64, d128]`, loaded on the first
    /// prefill long enough to use one (same rule as [`TiledAttnKernels`]).
    tiled: [std::sync::OnceLock<AotKernel>; 2],
}

impl EncoderModel {
    /// Loads `encoder.cu`. `rope_yarn_batch_k` comes from the dense loader's
    /// `rope` module (requested there iff `cfg.yarn` is set).
    pub(super) fn load(
        device: &Arc<CudaDevice>,
        cfg: EncoderConfig,
        rope_yarn_batch_k: Option<AotKernel>,
    ) -> Result<Self, ReflexError> {
        if cfg.yarn.is_some() != rope_yarn_batch_k.is_some() {
            return Err(crate::reflex_err!(
                Other,
                "internal: encoder YaRN config and kernel disagree"
            ));
        }
        let mut fns = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_ENCODER")),
            "encoder",
            &[
                "attention_online_bidir_kernel",
                "attn_temp_scale_kernel",
                "mean_pool_kernel",
            ],
        )?
        .into_iter();
        let mut next = |name: &str| {
            fns.next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing {name} in module encoder"))
        };
        Ok(Self {
            cfg,
            rope_yarn_batch_k,
            attn_bidir_k: next("attention_online_bidir_kernel")?,
            temp_scale_k: next("attn_temp_scale_kernel")?,
            mean_pool_k: next("mean_pool_kernel")?,
            tiled: Default::default(),
        })
    }

    /// The bidirectional tiled kernel for `head_dim`, loading its module on first
    /// use; `None` for head dims without one.
    fn tiled_kernel(
        &self,
        device: &Arc<CudaDevice>,
        head_dim: usize,
    ) -> Result<Option<&AotKernel>, ReflexError> {
        let (i, name) = match head_dim {
            64 => (0, "attention_tiled_bidir_d64_kernel"),
            128 => (1, "attention_tiled_bidir_d128_kernel"),
            _ => return Ok(None),
        };
        if let Some(k) = self.tiled[i].get() {
            return Ok(Some(k));
        }
        let module = "attention_tiled_bidir";
        let kernel = match device.get_func(module, name) {
            Some(function) => AotKernel { function },
            None => aot::load_kernel_module(
                device,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_BIDIR")),
                module,
                &[name],
            )?
            .pop()
            .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module {module}"))?,
        };
        Ok(Some(self.tiled[i].get_or_init(|| kernel)))
    }
}

impl Model {
    /// Whether this model is an embedding model ([`Self::embed`]) rather than a
    /// language model.
    pub fn is_embedding_model(&self) -> bool {
        self.encoder.is_some()
    }

    /// Errors on an embedding model: it has no next-token logits, so
    /// generation, System1 scoring and prefix caching don't apply.
    pub(super) fn check_generative(&self) -> Result<(), ReflexError> {
        if self.encoder.is_some() {
            return Err(ReflexError::InvalidInput(
                "this is an embedding model (bidirectional attention, pooled output): \
                 use embed (`reflex embed`, IPC `\"embed\": true`), not generation or scoring"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Encodes `text` as is (no BOS unless the tokenizer asks for one, no
    /// template: callers add the model's input prefix, e.g. Nemotron's
    /// `"query: "`/`"passage: "`), runs every token through the layers with
    /// bidirectional attention, applies the final norm, averages the rows and
    /// L2-normalizes the result.
    pub fn embed(&self, text: &str) -> Result<Embedding, ReflexError> {
        let enc = self.encoder.as_ref().ok_or_else(|| {
            ReflexError::InvalidInput(
                "embed needs an embedding model (a GGUF with <arch>.attention.causal = false); \
                 this is a language model"
                    .to_string(),
            )
        })?;
        let (ids, hidden, _k, _v, _seq_len) = self.prefill_dense_batched(text, None, 0)?;
        let rows = ids.len();
        let hidden_size = self.cfg.hidden_size;
        let normed = self.rmsnorm(
            &hidden,
            self.output_norm.f32()?,
            rows,
            hidden_size,
            self.cfg.rmsnorm_eps,
        )?;
        let pooled = match enc.cfg.pooling {
            Pooling::Mean => self.mean_pool(enc, &normed, rows, hidden_size)?,
        };
        let mut vector = self
            .device
            .dtoh_sync_copy(&pooled)
            .map_err(|e| crate::gpu_err!(e, "embedding dtoh: {e}"))?;
        l2_normalize(&mut vector);
        Ok(Embedding {
            vector,
            prompt_tokens: rows,
        })
    }

    pub(super) fn mean_pool(
        &self,
        enc: &EncoderModel,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let out = self.alloc_out(cols, "mean_pool")?;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((cols as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            enc.mean_pool_k
                .function
                .clone()
                .launch(cfg, (x, *out.device_ptr(), rows as u32, cols as u32))
                .map_err(|e| crate::gpu_err!(e, "mean_pool launch: {e}"))?;
        }
        Ok(out)
    }

    /// The encoder's YaRN rotation for a batch of rows, if it has one: `Ok(true)`
    /// when it ran (so [`Self::rope_batch`] skips the plain kernel).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_yarn_batch(
        &self,
        t: &mut CudaSlice<f32>,
        start_pos: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rows: usize,
        base: f32,
    ) -> Result<bool, ReflexError> {
        let Some(enc) = &self.encoder else {
            return Ok(false);
        };
        let (Some(y), Some(kernel)) = (enc.cfg.yarn, &enc.rope_yarn_batch_k) else {
            return Ok(false);
        };
        let total_pairs = (rows * num_heads * (rotary_dim / 2)) as u32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total_pairs.div_ceil(threads).max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    cfg,
                    (
                        t,
                        start_pos as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        rows as u32,
                        base,
                        y.freq_scale,
                        y.ext_factor,
                        y.attn_factor,
                        y.corr_dim_start,
                        y.corr_dim_end,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_norm_yarn_batch launch: {e}"))?;
        }
        Ok(true)
    }

    /// The encoder's query temperature scale on `q` (`[rows, row_len]`, already
    /// RoPE'd). A no-op for other models and while every row is below the floor
    /// scale, where the factor is exactly 1.
    pub(super) fn encoder_temp_scale(
        &self,
        q: &mut CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        row_len: usize,
    ) -> Result<(), ReflexError> {
        let Some(enc) = &self.encoder else {
            return Ok(());
        };
        let Some(t) = enc.cfg.temperature else {
            return Ok(());
        };
        if ((start_pos + rows) as f32) <= t.floor_scale {
            return Ok(());
        }
        let total = (rows * row_len) as u32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(threads).clamp(1, 65_535), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            enc.temp_scale_k
                .function
                .clone()
                .launch(
                    cfg,
                    (
                        q,
                        start_pos as u32,
                        rows as u32,
                        row_len as u32,
                        t.beta,
                        t.floor_scale,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn_temp_scale launch: {e}"))?;
        }
        Ok(())
    }

    /// [`Self::attention_prefill`] for an encoder: every row sees all
    /// `start_pos + rows` positions. Tiled kernels from the same row count up
    /// as the causal path, the online kernel below. f32 cache only (checked at
    /// load).
    pub(super) fn attention_prefill_bidir(
        &self,
        enc: &EncoderModel,
        q: &CudaSlice<f32>,
        k_cache: &KvView,
        v_cache: &KvView,
        shape: OnlineAttnShape,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if shape.window != 0 {
            return Err(crate::reflex_err!(
                Other,
                "internal: bidirectional attention has no sliding window"
            ));
        }
        if attn_tiled_preferred(shape.rows, shape.start_pos, shape.group_size)
            && self.attn_tiled.enabled()
            && attn_tiled_fits(&shape)
        {
            if let Some(kernel) = enc.tiled_kernel(&self.device, shape.qk_dim)? {
                return launch_attention_tiled(&self.device, kernel, q, k_cache, v_cache, &shape);
            }
        }
        if shape.qk_dim > ATTN_ONLINE_MAX_DIM || shape.v_dim > ATTN_ONLINE_MAX_DIM {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "attention head dims qk={} v={} exceed the online attention kernel's limit of {ATTN_ONLINE_MAX_DIM}",
                shape.qk_dim,
                shape.v_dim
            ));
        }
        let out = self.alloc_out(shape.rows * shape.num_q_heads * shape.v_dim, "attn_bidir")?;
        let cfg = LaunchConfig {
            grid_dim: (shape.rows as u32, shape.num_q_heads as u32, 1),
            block_dim: ((ATTN_ONLINE_WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: ((shape.qk_dim + ATTN_ONLINE_WARPS * (shape.v_dim + 2))
                * std::mem::size_of::<f32>()) as u32,
        };
        let (k, v) = (k_cache.f32()?, v_cache.f32()?);
        unsafe {
            enc.attn_bidir_k
                .function
                .clone()
                .launch(cfg, (q, k, v, *out.device_ptr(), shape.unsplit_params()))
                .map_err(|e| crate::gpu_err!(e, "attn_bidir launch: {e}"))?;
        }
        Ok(out)
    }
}

/// Scales `v` to unit L2 norm (the sum of squares in f64); leaves an all-zero
/// vector as is.
pub(super) fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
}
