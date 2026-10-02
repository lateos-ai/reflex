//! DeepSeek-V2/V3 Multi-head Latent Attention models: loading and forward passes.

use super::*;
use crate::error::ReflexError;

/// MLA prefill result: encoded prompt ids, the final position's hidden
/// state, the filled per-layer compressed-latent K/V caches, and the next
/// absolute position. Shared by `prefill_mla_batched` and its non-batched
/// counterpart.
pub(super) type MlaPrefillResult = (Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, usize);

/// MLA generate result: generated token ids, their concatenated decoded
/// text, the final per-layer compressed-latent K/V caches, and the total
/// sequence length reached. Returned by `generate_mla_impl`.
pub(super) type MlaGenerateResult = (Vec<u32>, String, Vec<CudaSlice<f32>>, usize);

/// One MLA layer's FFN: dense SwiGLU for the `leading_dense_block_count` lead
/// layers (identical in shape/meaning to `DenseLayerWeights`'s), or routed MoE +
/// an always-on shared expert for every layer past that (real DeepSeek-V2/V3
/// files always have both kinds; the synthetic MVP-step-4 fixture is
/// `Dense`-only). See `Model::forward_mla_moe_ffn` for the shared-expert math --
/// its weights (`ffn_{gate,up,down}_shexp`) are a *single* fused dense FFN over
/// `n_ff_exp * expert_shared_count` hidden units (every shared expert's weights
/// concatenated into one bigger matmul), not `expert_shared_count` separate
/// per-expert calls -- confirmed against `deepseek2.cpp`'s own tensor shapes.
pub(super) enum MlaFfn {
    Dense {
        ffn_gate: Weight,
        ffn_up: Weight,
        ffn_down: Weight,
    },
    Moe {
        ffn_gate_inp: Weight,
        ffn_gate_exps: Weight,
        ffn_up_exps: Weight,
        ffn_down_exps: Weight,
        ffn_gate_shexp: Weight,
        ffn_up_shexp: Box<Weight>,
        ffn_down_shexp: Box<Weight>,
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
pub(super) struct MlaLayerWeights {
    pub(super) attn_norm: Weight,
    /// `[hidden, n_head*(qk_nope_head_dim+qk_rope_head_dim)]` -- direct projection,
    /// no Q-LoRA decomposition (out of scope this round).
    pub(super) wq: Weight,
    /// `[hidden, kv_lora_rank+qk_rope_head_dim]`, fused compressed-KV + shared
    /// rope-K projection (MQA: a single shared "head").
    pub(super) wkv_a_mqa: Weight,
    pub(super) attn_kv_a_norm: Weight,
    /// `[qk_nope_head_dim, kv_lora_rank, n_head]`.
    pub(super) wk_b: Weight,
    /// `[kv_lora_rank, v_head_dim, n_head]`.
    pub(super) wv_b: Weight,
    /// `[n_head*v_head_dim, hidden]`.
    pub(super) wo: Weight,
    pub(super) ffn_norm: Weight,
    pub(super) ffn: MlaFfn,
}

/// A loaded DeepSeek-V2/V3 MLA model's extra state, layered on top of the same
/// [`Model`] every other architecture uses (shared `token_embd`/`output_norm`/
/// `lm_head`/`tokenizer`, and the same `rmsnorm_k`/`rope_k`/`silu_k`/`gemv_k`/
/// `add_k` kernels every other path reuses unchanged -- see `Model::forward_mla_attn_block`).
pub(super) struct MlaModel {
    pub(super) cfg: MlaConfig,
    pub(super) layers: Vec<MlaLayerWeights>,
    pub(super) mla_attn_k: AotKernel,
    /// `rope_norm_kernel` (`kernels_cuda/rope.cu`), **not** the shared `Model::rope_k`
    /// (`rope_kernel`) every other architecture uses -- confirmed against
    /// llama.cpp's `llama_model_rope_type`, which maps `deepseek2` to
    /// `LLAMA_ROPE_TYPE_NORM` (consecutive-pair rotation), not the
    /// `LLAMA_ROPE_TYPE_NEOX` (half-split) convention Qwen3/Qwen3.5 use.
    pub(super) rope_norm_k: AotKernel,
    /// `rope_norm_yarn_kernel` -- used instead of `rope_norm_k` whenever
    /// `cfg.yarn.is_some()` (see `Model::forward_mla_attn_block`). Always loaded
    /// (even for the synthetic, YaRN-free MVP-step-4 fixture) since the tiny
    /// extra load cost isn't worth an `Option`.
    pub(super) rope_norm_yarn_k: AotKernel,
    /// Batched-prefill variant of `mla_attn_k` (`mla_attention_prefill_kernel`,
    /// `kernels_cuda/mla_attention_prefill.cu`) -- see
    /// `Model::forward_mla_attn_block_batched`.
    pub(super) mla_attn_prefill_k: AotKernel,
    /// Batched-prefill variant of `rope_norm_k` (`rope_norm_batch_kernel`,
    /// `kernels_cuda/rope.cu`).
    pub(super) rope_norm_batch_k: AotKernel,
    /// Batched-prefill variant of `rope_norm_yarn_k` (`rope_norm_yarn_batch_kernel`,
    /// `kernels_cuda/rope.cu`).
    pub(super) rope_norm_yarn_batch_k: AotKernel,
    /// Batched-prefill variant of `Model::gemv_per_head`'s per-head-loop-of-`gemv_k`
    /// (`gemv_per_head_batch_kernel`, `kernels_cuda/gemv_per_head_batch.cu`) --
    /// applies MLA's per-head-stacked `wk_b`/`wv_b` weights to every head of every
    /// batched row in one launch. See `Model::gemv_per_head_batch`.
    pub(super) gemv_per_head_batch_k: AotKernel,
    /// Extracts a per-head sub-slice out of a wider batched per-head buffer in one
    /// launch (`mla_extract_batch_kernel`, `kernels_cuda/elementwise.cu`) -- used for
    /// q_pe/k_pe/kv_cmpr extraction ahead of RoPE/RMSNorm in
    /// `Model::forward_mla_attn_block_batched`.
    pub(super) mla_extract_batch_k: AotKernel,
    /// Merges absorbed q_nope and RoPE'd q_pe into Qcur's per-head row in one launch
    /// (`mla_concat_qcur_batch_kernel`, `kernels_cuda/elementwise.cu`).
    pub(super) mla_concat_qcur_batch_k: AotKernel,
    /// Writes a batch's compressed Kcur into the preallocated `kv_cache` in one
    /// launch (`mla_write_kv_cache_batch_kernel`, `kernels_cuda/elementwise.cu`).
    pub(super) mla_write_kv_cache_batch_k: AotKernel,
}

impl Model {
    /// Loads a DeepSeek-V2/V3 MLA model (MVP step 4). See [`parse_mla_config`] for
    /// the scope this supports. `cfg`/`layers`/`expert_used_count` below are
    /// unused garbage (matching the `hybrid` path's own convention) --
    /// `forward_prompt` branches on `self.mla` before touching them.
    pub(super) fn load_mla(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, ReflexError> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let init = scope.spawn(move || Self::load_background_init(file, init_device));
            Self::load_mla_inner(device, file, init)
        })
    }

    pub(super) fn load_mla_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), ReflexError>>,
    ) -> Result<Self, ReflexError> {
        let (mla_cfg, block_count, leading_dense) = parse_mla_config(file)?;

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
        let gemv_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV")),
            "gemv",
            "gemv_kernel",
        )?;
        let gemv_gather_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV_GATHER")),
            "gemv_gather",
            "gemv_gather_kernel",
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
                "mla_extract_batch_kernel",
                "mla_concat_qcur_batch_kernel",
                "mla_write_kv_cache_batch_kernel",
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
        let mla_extract_batch_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing mla_extract_batch_kernel".to_string()))?;
        let mla_concat_qcur_batch_k = elementwise_fns.next().ok_or_else(|| {
            ReflexError::Other("missing mla_concat_qcur_batch_kernel".to_string())
        })?;
        let mla_write_kv_cache_batch_k = elementwise_fns.next().ok_or_else(|| {
            ReflexError::Other("missing mla_write_kv_cache_batch_kernel".to_string())
        })?;
        let moe_gather_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing moe_gather_kernel".to_string()))?;
        let moe_scatter_add_k = elementwise_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing moe_scatter_add_kernel".to_string()))?;
        let mla_attn_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_MLA_ATTENTION")),
            "mla_attention",
            "mla_attention_kernel",
        )?;
        let mla_attn_prefill_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_MLA_ATTENTION_PREFILL")),
            "mla_attention_prefill",
            "mla_attention_prefill_kernel",
        )?;
        let mut rope_norm_fns = aot::load_kernel_module(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_ROPE")),
            "rope_norm",
            &[
                "rope_norm_kernel",
                "rope_norm_yarn_kernel",
                "rope_norm_batch_kernel",
                "rope_norm_yarn_batch_kernel",
            ],
        )?
        .into_iter();
        let rope_norm_k = rope_norm_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_kernel".to_string()))?;
        let rope_norm_yarn_k = rope_norm_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_yarn_kernel".to_string()))?;
        let rope_norm_batch_k = rope_norm_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_batch_kernel".to_string()))?;
        let rope_norm_yarn_batch_k = rope_norm_fns
            .next()
            .ok_or_else(|| ReflexError::Other("missing rope_norm_yarn_batch_kernel".to_string()))?;
        let gemv_per_head_batch_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV_PER_HEAD_BATCH")),
            "gemv_per_head_batch",
            "gemv_per_head_batch_kernel",
        )?;
        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;

        let mut load_weight = |name: &str| -> Result<Weight, ReflexError> {
            load_weight_device(&mut pipeline, &dequant_kernels, file, name)
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
                    ffn_up_shexp: Box::new(load_weight(&format!("blk.{i}.ffn_up_shexp.weight"))?),
                    ffn_down_shexp: Box::new(load_weight(&format!(
                        "blk.{i}.ffn_down_shexp.weight"
                    ))?),
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

        let token_embd_info = file
            .tensor_info("token_embd.weight")
            .ok_or_else(|| ReflexError::Gguf("missing weight 'token_embd.weight'".to_string()))?;
        let token_embd_bytes = file.tensor_bytes(token_embd_info)?;
        let token_embd = LazyTokenEmbedding::new(
            token_embd_info.ggml_type,
            token_embd_bytes.to_vec(),
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
                let data = dequantize_tensor_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    info.ggml_type,
                    bytes,
                    info.element_count(),
                )
                .map_err(|e| e.rewrap(format!("load weight 'output.weight': {e}")))?;
                LmHead::Resident(Weight::from_f32(data, info.shape.clone()))
            }
            None => {
                let data = dequantize_tensor_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    token_embd.ggml_type,
                    &token_embd.raw,
                    token_embd_info.element_count(),
                )
                .map_err(|e| e.rewrap(format!("load weight 'token_embd.weight': {e}")))?;
                LmHead::Resident(Weight::from_f32(data, token_embd_info.shape.clone()))
            }
        };

        let (tokenizer, cublas) = init.join().map_err(|_| {
            ReflexError::Other("background load-init thread panicked".to_string())
        })??;

        let dummy_cfg = LayerConfig {
            hidden_size: mla_cfg.hidden_size,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rotary_dim: 1,
            ffn_hidden_size: 1,
            rope_base: mla_cfg.rope_base,
            rmsnorm_eps: mla_cfg.rmsnorm_eps,
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
            gemv_gather_k,
            gemv_q4k_k: None,
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
            cfg: dummy_cfg,
            layers: Vec::new(),
            expert_used_count: None,
            token_embd,
            dequant_kernels,
            dequant_pipeline: RefCell::new(pipeline),
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

    /// DeepSeek-V2/V3 MLA path -- see [`Self::system1_evaluate`]'s doc
    /// comment for the overall approach. Supports multi-token candidates
    /// the same way dense does: `MlaPrefillResult`'s per-layer `kv_cache` is
    /// compressed latent-KV (`[seq_len, kv_lora_rank + qk_rope_head_dim]`)
    /// but still **position-indexed**, not an unaddressed recurrence like
    /// hybrid's `GatedDeltaNet` state -- so the same shared-cache,
    /// score-to-completion-before-the-next-candidate reuse dense relies on
    /// carries over unmodified.
    pub(super) fn system1_evaluate_mla(
        &self,
        m: &MlaModel,
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

        let (ids, hidden_batched, mut kv_caches, base_position) =
            self.prefill_mla_batched(m, prompt, None, max_len.saturating_sub(1))?;
        let hidden_size = m.cfg.hidden_size;
        let eps = m.cfg.rmsnorm_eps;
        let hidden = self.last_row(&hidden_batched, ids.len(), hidden_size)?;

        let normed = self.rmsnorm(&hidden, self.output_norm.f32()?, 1, hidden_size, eps)?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemv_gather_lm_head(&normed, &first_tokens)?;

        for (i, ids) in resolved.iter().enumerate() {
            if ids.len() < 2 {
                continue;
            }
            for (position, w) in (base_position..).zip(ids.windows(2)) {
                let (prev, next) = (w[0], w[1]);
                let h = self.forward_one_token_mla(m, prev, position, &mut kv_caches)?;
                let normed_step = self.rmsnorm(&h, self.output_norm.f32()?, 1, hidden_size, eps)?;
                scores[i] += self.gemv_gather_lm_head(&normed_step, &[next])?[0];
            }
        }

        Self::finish_system1_response(candidates, resolved, scores, temperature)
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
    pub(super) fn forward_mla_attn_block(
        &self,
        m: &MlaModel,
        w: &MlaLayerWeights,
        mut hidden: CudaSlice<f32>,
        position: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let n_head = cfg.num_heads;
        let qk_nope = cfg.qk_nope_head_dim;
        let qk_rope = cfg.qk_rope_head_dim;
        let n_embd_head_k_mla = qk_nope + qk_rope;
        let kv_lora = cfg.kv_lora_rank;
        let qk_dim = kv_lora + qk_rope;
        let v_dim = kv_lora;

        let normed = self.rmsnorm(
            &hidden,
            w.attn_norm.f32()?,
            1,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        // q: [n_head, n_embd_head_k_mla] flat (plain gemv -- is_lite path, no Q-LoRA).
        let q = self.gemv(&normed, &w.wq)?;

        // kv_cmpr_pe: [kv_lora_rank + qk_rope_head_dim] flat (single shared "head").
        let kv_cmpr_pe = self.gemv(&normed, &w.wkv_a_mqa)?;

        let mut k_pe = self
            .device
            .alloc_zeros::<f32>(qk_rope)
            .map_err(|e| crate::gpu_err!(e, "mla k_pe alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(kv_lora..kv_lora + qk_rope);
            self.device
                .dtod_copy(&src, &mut k_pe)
                .map_err(|e| crate::gpu_err!(e, "mla k_pe dtod: {e}"))?;
        }
        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn(
                m,
                yarn,
                &mut k_pe,
                1,
                qk_rope,
                qk_rope,
                position,
                cfg.rope_base,
            )?,
            None => self.rope_norm(m, &mut k_pe, 1, qk_rope, qk_rope, position, cfg.rope_base)?,
        }

        let mut kv_cmpr_owned = self
            .device
            .alloc_zeros::<f32>(kv_lora)
            .map_err(|e| crate::gpu_err!(e, "mla kv_cmpr alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(0..kv_lora);
            self.device
                .dtod_copy(&src, &mut kv_cmpr_owned)
                .map_err(|e| crate::gpu_err!(e, "mla kv_cmpr dtod: {e}"))?;
        }
        let kv_cmpr_normed = self.rmsnorm(
            &kv_cmpr_owned,
            w.attn_kv_a_norm.f32()?,
            1,
            kv_lora,
            cfg.rmsnorm_eps,
        )?;

        // Gather q_pe (all heads) into its own contiguous [n_head, qk_rope_head_dim]
        // buffer before RoPE -- `Self::rope` expects one contiguous multi-head buffer,
        // and q_pe is a strided sub-slice of each head's [n_embd_head_k_mla]-wide row
        // in `q`, not itself contiguous across heads.
        let mut q_pe = self
            .device
            .alloc_zeros::<f32>(n_head * qk_rope)
            .map_err(|e| crate::gpu_err!(e, "mla q_pe alloc: {e}"))?;
        for h in 0..n_head {
            let src =
                q.slice(h * n_embd_head_k_mla + qk_nope..h * n_embd_head_k_mla + n_embd_head_k_mla);
            let mut dst = q_pe.slice_mut(h * qk_rope..(h + 1) * qk_rope);
            self.device
                .dtod_copy(&src, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "mla q_pe dtod head {h}: {e}"))?;
        }
        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn(
                m,
                yarn,
                &mut q_pe,
                n_head,
                qk_rope,
                qk_rope,
                position,
                cfg.rope_base,
            )?,
            None => self.rope_norm(
                m,
                &mut q_pe,
                n_head,
                qk_rope,
                qk_rope,
                position,
                cfg.rope_base,
            )?,
        }

        // Per head: absorb q_nope via wk_b, then concat with the (already-roped)
        // q_pe slice into Qcur's per-head [qk_dim]-wide row.
        let mut qcur = self
            .device
            .alloc_zeros::<f32>(n_head * qk_dim)
            .map_err(|e| crate::gpu_err!(e, "mla qcur alloc: {e}"))?;
        for h in 0..n_head {
            let q_nope_view = q.slice(h * n_embd_head_k_mla..h * n_embd_head_k_mla + qk_nope);
            let wk_b_view = w
                .wk_b
                .f32()?
                .slice(h * qk_nope * kv_lora..(h + 1) * qk_nope * kv_lora);
            let absorbed = self.gemv_view(&q_nope_view, &wk_b_view, qk_nope, kv_lora)?;

            let mut dst_nope = qcur.slice_mut(h * qk_dim..h * qk_dim + kv_lora);
            self.device
                .dtod_copy(&absorbed, &mut dst_nope)
                .map_err(|e| crate::gpu_err!(e, "mla qcur absorbed dtod head {h}: {e}"))?;

            let pe_src = q_pe.slice(h * qk_rope..(h + 1) * qk_rope);
            let mut dst_pe = qcur.slice_mut(h * qk_dim + kv_lora..h * qk_dim + qk_dim);
            self.device
                .dtod_copy(&pe_src, &mut dst_pe)
                .map_err(|e| crate::gpu_err!(e, "mla qcur pe dtod head {h}: {e}"))?;
        }

        // Write this position's compressed Kcur (== kv_cmpr_normed ++ k_pe) into the
        // preallocated per-layer cache -- device-resident from the start (Phase 2
        // round 2 convention), no host round-trip, ever, for this cache.
        let offset = position * qk_dim;
        {
            let mut dst = kv_cache.slice_mut(offset..offset + kv_lora);
            self.device
                .dtod_copy(&kv_cmpr_normed, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "mla kv_cache dtod cmpr: {e}"))?;
        }
        {
            let mut dst = kv_cache.slice_mut(offset + kv_lora..offset + qk_dim);
            self.device
                .dtod_copy(&k_pe, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "mla kv_cache dtod k_pe: {e}"))?;
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
        let compressed_out =
            self.mla_attention(m, &qcur, &kv_view, n_head, qk_dim, v_dim, seq_len, scale)?;

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
    /// per-call-overhead regression docs/DEVELOPMENT.md's model-loading section documents -- e.g.
    /// ~14k launches per layer at a 449-row prefill with 16 heads, ~28k counting
    /// decompression too. [`Self::gemv_per_head_batch`] does both in one launch each
    /// instead. The three small `Self::mla_extract_batch`/`Self::mla_concat_qcur_batch`/
    /// `Self::mla_write_kv_cache_batch` helpers replace the sequential path's
    /// per-head/per-row `dtod_copy` loops the same way, each in one launch. `start_pos`
    /// is this batch's first row's absolute position (row `r` is `start_pos + r`),
    /// matching [`Self::forward_attn_block_batched`]'s resume convention.
    pub(super) fn forward_mla_attn_block_batched(
        &self,
        m: &MlaModel,
        w: &MlaLayerWeights,
        mut hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let n_head = cfg.num_heads;
        let qk_nope = cfg.qk_nope_head_dim;
        let qk_rope = cfg.qk_rope_head_dim;
        let n_embd_head_k_mla = qk_nope + qk_rope;
        let kv_lora = cfg.kv_lora_rank;
        let qk_dim = kv_lora + qk_rope;
        let v_dim = kv_lora;

        let normed = self.rmsnorm(
            &hidden,
            w.attn_norm.f32()?,
            rows,
            cfg.hidden_size,
            cfg.rmsnorm_eps,
        )?;

        // q_batched: [rows, n_head, n_embd_head_k_mla] flat (plain gemm -- is_lite
        // path, no Q-LoRA).
        let q_batched = self.gemm(&normed, &w.wq, rows)?;

        // kv_cmpr_pe_batched: [rows, kv_lora_rank + qk_rope_head_dim] flat (single
        // shared "head" per row).
        let kv_cmpr_pe_batched = self.gemm(&normed, &w.wkv_a_mqa, rows)?;

        // Extract k_pe/kv_cmpr into their own contiguous [rows, 1, width] buffers
        // (num_heads=1: the whole fused wkv_a_mqa row is treated as a single head).
        let mut k_pe_batched = self.mla_extract_batch(
            m,
            &kv_cmpr_pe_batched,
            rows,
            1,
            kv_lora + qk_rope,
            qk_rope,
            kv_lora,
        )?;
        let kv_cmpr_batched = self.mla_extract_batch(
            m,
            &kv_cmpr_pe_batched,
            rows,
            1,
            kv_lora + qk_rope,
            kv_lora,
            0,
        )?;

        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn_batch(
                m,
                yarn,
                &mut k_pe_batched,
                1,
                qk_rope,
                qk_rope,
                start_pos,
                rows,
                cfg.rope_base,
            )?,
            None => self.rope_norm_batch(
                m,
                &mut k_pe_batched,
                1,
                qk_rope,
                qk_rope,
                start_pos,
                rows,
                cfg.rope_base,
            )?,
        }

        let kv_cmpr_normed_batched = self.rmsnorm(
            &kv_cmpr_batched,
            w.attn_kv_a_norm.f32()?,
            rows,
            kv_lora,
            cfg.rmsnorm_eps,
        )?;

        // Extract q_pe (all heads, all rows) into its own contiguous
        // [rows, n_head, qk_rope] buffer before RoPE -- q_pe is a strided sub-slice
        // of each head's [n_embd_head_k_mla]-wide row in q_batched, not itself
        // contiguous across heads.
        let mut q_pe_batched = self.mla_extract_batch(
            m,
            &q_batched,
            rows,
            n_head,
            n_embd_head_k_mla,
            qk_rope,
            qk_nope,
        )?;
        match &cfg.yarn {
            Some(yarn) => self.rope_norm_yarn_batch(
                m,
                yarn,
                &mut q_pe_batched,
                n_head,
                qk_rope,
                qk_rope,
                start_pos,
                rows,
                cfg.rope_base,
            )?,
            None => self.rope_norm_batch(
                m,
                &mut q_pe_batched,
                n_head,
                qk_rope,
                qk_rope,
                start_pos,
                rows,
                cfg.rope_base,
            )?,
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
        let qcur_batched = self.mla_concat_qcur_batch(
            m,
            &absorbed_batched,
            &q_pe_batched,
            rows,
            n_head,
            kv_lora,
            qk_rope,
        )?;

        // Write this batch's compressed Kcur into the preallocated per-layer cache.
        self.mla_write_kv_cache_batch(
            m,
            kv_cache,
            &kv_cmpr_normed_batched,
            &k_pe_batched,
            start_pos,
            rows,
            kv_lora,
            qk_rope,
        )?;
        let seq_len = start_pos + rows;

        let kv_view = kv_cache.slice(0..seq_len * qk_dim);
        let scale = match &cfg.yarn {
            Some(yarn) => yarn.attention_scale,
            None => 1.0 / (n_embd_head_k_mla as f32).sqrt(),
        };
        let compressed_out_batched = self.mla_attention_prefill(
            m,
            &qcur_batched,
            &kv_view,
            n_head,
            qk_dim,
            v_dim,
            start_pos,
            rows,
            scale,
        )?;

        // Decompression: already-contiguous [rows, n_head, v_dim] input, standard
        // strides.
        let decompressed_batched = self.gemv_per_head_batch(
            m,
            &compressed_out_batched,
            &w.wv_b,
            rows,
            n_head,
            n_head * v_dim,
            v_dim,
            0,
        )?;

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
    pub(super) fn forward_mla_layer_batched(
        &self,
        m: &MlaModel,
        layer: &MlaLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let post_attn =
            self.forward_mla_attn_block_batched(m, layer, hidden, start_pos, rows, kv_cache)?;

        let cfg = &m.cfg;
        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;

        match &layer.ffn {
            MlaFfn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => self.forward_hybrid_ffn_batched(
                post_attn,
                &layer.ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                hidden_size,
                ffn_hidden_size,
                rows,
                eps,
            ),
            MlaFfn::Moe { .. } => {
                let moe_cfg = cfg.moe.as_ref().ok_or_else(|| {
                    ReflexError::Other(
                        "internal error: MlaFfn::Moe layer but MlaConfig::moe is None".to_string(),
                    )
                })?;
                self.forward_mla_moe_ffn_batched(layer, post_attn, hidden_size, rows, moe_cfg, eps)
            }
        }
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
    /// `Self::gemv_expert`, device-resident weighted accumulate via
    /// `Self::moe_scatter_add` -- same convention `forward_layer_moe` uses), plus
    /// one addition real DeepSeek-V2/V3 has and Qwen3-MoE doesn't: an always-on
    /// shared expert, computed as a single dense FFN (its
    /// `ffn_{gate,up,down}_shexp` weights already fuse every shared expert into
    /// one bigger matmul -- see `MlaFfn`'s doc comment) and added to the
    /// accumulator unconditionally, not gated by the router.
    pub(super) fn forward_mla_moe_ffn(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let MlaFfn::Moe {
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
        } = &layer.ffn
        else {
            return Err(ReflexError::Other(
                "internal error: forward_mla_moe_ffn called on a Dense layer".to_string(),
            ));
        };

        let ffn_normed = self.rmsnorm(&post_attn, layer.ffn_norm.f32()?, 1, hidden_size, eps)?;

        let router_logits_dev = self.gemv(&ffn_normed, ffn_gate_inp)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "mla moe router dtoh: {e}"))?;
        let routed = route_top_k_with_norm(
            &router_logits,
            moe_cfg.expert_used_count,
            moe_cfg.normalize_top_k,
        )?;

        let mut ffn_out_dev = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| crate::gpu_err!(e, "mla moe ffn_out alloc: {e}"))?;
        let dest_row0 = self
            .device
            .htod_sync_copy(&[0u32])
            .map_err(|e| crate::gpu_err!(e, "mla moe dest_row htod: {e}"))?;
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, moe_cfg.n_ff_exp)?;
            let down = self.gemv_expert(&activated, ffn_down_exps, expert_idx)?;
            let weight_dev = self
                .device
                .htod_sync_copy(&[weight * moe_cfg.routed_scaling_factor])
                .map_err(|e| crate::gpu_err!(e, "mla moe weight htod: {e}"))?;
            self.moe_scatter_add(
                &down,
                &dest_row0,
                &weight_dev,
                &mut ffn_out_dev,
                hidden_size,
            )?;
        }

        // Always-on shared expert(s) -- a single fused dense FFN, not gated by the
        // router, added unconditionally (device-resident: a plain vector add, no
        // per-row weighting needed, so `Self::add_inplace` covers it directly).
        let shared_hidden_size = ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemv(&ffn_normed, ffn_gate_shexp)?;
        let shared_up = self.gemv(&ffn_normed, ffn_up_shexp)?;
        let shared_activated = self.silu_and_mul(&shared_gate, &shared_up, shared_hidden_size)?;
        let shared_down = self.gemv(&shared_activated, ffn_down_shexp)?;
        self.add_inplace(&mut ffn_out_dev, &shared_down)?;

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
    pub(super) fn forward_mla_moe_ffn_batched(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        rows: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let MlaFfn::Moe {
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
        } = &layer.ffn
        else {
            return Err(ReflexError::Other(
                "internal error: forward_mla_moe_ffn_batched called on a Dense layer".to_string(),
            ));
        };

        let ffn_normed = self.rmsnorm(&post_attn, layer.ffn_norm.f32()?, rows, hidden_size, eps)?;

        // Always-on shared expert(s), batched across every row with no routing --
        // seeds ffn_out; the routed experts below accumulate `+=` on top of it.
        let shared_hidden_size = ffn_gate_shexp.shape[1] as usize;
        let shared_gate = self.gemm(&ffn_normed, ffn_gate_shexp, rows)?;
        let shared_up = self.gemm(&ffn_normed, ffn_up_shexp, rows)?;
        let shared_activated =
            self.silu_and_mul(&shared_gate, &shared_up, rows * shared_hidden_size)?;
        let mut ffn_out = self.gemm(&shared_activated, ffn_down_shexp, rows)?;

        let router_logits_dev = self.gemm(&ffn_normed, ffn_gate_inp, rows)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| crate::gpu_err!(e, "mla moe router dtoh: {e}"))?;
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
    pub(super) fn forward_prompt_mla(
        &self,
        m: &MlaModel,
        prompt: &str,
    ) -> Result<(u32, String), ReflexError> {
        let (generated, text, _kv_caches, _seq_len) = self.generate_mla_impl(
            m,
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;
        Ok((generated[0], text))
    }

    /// Shared per-layer `kv_cache` allocation/import behind [`Self::prefill_mla`] and
    /// [`Self::prefill_mla_batched`]: allocates each layer's compressed `[total_len,
    /// qk_dim]` cache (`total_len = start_pos + rows + extra_headroom`) and seeds it
    /// from `imported` when resuming -- identical between the sequential and batched
    /// prefill paths, so factored out once rather than duplicated (same role
    /// `Self::alloc_hybrid_states` plays for the hybrid path).
    pub(super) fn alloc_mla_kv_caches(
        &self,
        m: &MlaModel,
        imported: Option<&crate::kv_io::MlaKvCache>,
        start_pos: usize,
        rows: usize,
        extra_headroom: usize,
    ) -> Result<Vec<CudaSlice<f32>>, ReflexError> {
        crate::limits::check_positions_up_to(
            start_pos,
            rows,
            extra_headroom,
            self.attn_impl.max_positions(),
        )?;
        let qk_dim = m.cfg.kv_lora_rank + m.cfg.qk_rope_head_dim;
        let total_len = start_pos + rows + extra_headroom;
        let mut kv_caches: Vec<CudaSlice<f32>> = (0..m.layers.len())
            .map(|_| self.device.alloc_zeros::<f32>(total_len * qk_dim))
            .collect::<Result<_, _>>()
            .map_err(|e| crate::gpu_err!(e, "alloc mla kv_cache: {e}"))?;

        if let Some(cache) = imported {
            if cache.qk_dim != qk_dim {
                return Err(crate::reflex_err!(
                    Other,
                    "imported KV cache shape mismatch: file has qk_dim={}, model expects qk_dim={}",
                    cache.qk_dim,
                    qk_dim
                ));
            }
            if cache.kv_caches.len() != m.layers.len() {
                return Err(crate::reflex_err!(
                    Other,
                    "imported KV cache has {} layers, model has {}",
                    cache.kv_caches.len(),
                    m.layers.len()
                ));
            }
            let imported_len = cache.seq_len * qk_dim;
            for (layer_idx, kv_host) in cache.kv_caches.iter().enumerate() {
                let mut dst = kv_caches[layer_idx].slice_mut(0..imported_len);
                self.device
                    .htod_sync_copy_into(kv_host, &mut dst)
                    .map_err(|e| {
                        crate::gpu_err!(e, "import mla kv_cache htod layer {layer_idx}: {e}")
                    })?;
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
    pub(super) fn prefill_mla(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        extra_headroom: usize,
    ) -> Result<MlaPrefillResult, ReflexError> {
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

        let mut kv_caches =
            self.alloc_mla_kv_caches(m, imported, start_pos, ids.len(), extra_headroom)?;

        let mut position = start_pos;
        let mut hidden_dev: Option<CudaSlice<f32>> = None;
        for &token_id in &ids {
            hidden_dev = Some(self.forward_one_token_mla(m, token_id, position, &mut kv_caches)?);
            position += 1;
        }
        let hidden =
            hidden_dev.ok_or_else(|| ReflexError::Other("no tokens processed".to_string()))?;

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
    pub(super) fn prefill_mla_batched(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        extra_headroom: usize,
    ) -> Result<MlaPrefillResult, ReflexError> {
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

        let mut kv_caches =
            self.alloc_mla_kv_caches(m, imported, start_pos, rows, extra_headroom)?;

        let hidden_size = m.cfg.hidden_size;
        let mut host_embd = vec![0.0f32; rows * hidden_size];
        for (row, &token_id) in ids.iter().enumerate() {
            host_embd[row * hidden_size..(row + 1) * hidden_size]
                .copy_from_slice(&self.token_embd.row(token_id)?);
        }
        let mut hidden = self
            .device
            .htod_sync_copy(&host_embd)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;

        for (layer_idx, layer) in m.layers.iter().enumerate() {
            hidden = self.forward_mla_layer_batched(
                m,
                layer,
                hidden,
                start_pos,
                rows,
                &mut kv_caches[layer_idx],
            )?;
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mla_impl(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        mut on_first_token: impl FnMut(&[f32]),
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<MlaGenerateResult, ReflexError> {
        if max_new_tokens == 0 {
            return Err(ReflexError::InvalidInput(
                "max_new_tokens must be at least 1".to_string(),
            ));
        }

        let (ids, hidden_batched, mut kv_caches, mut position) =
            self.prefill_mla_batched(m, prompt, imported, max_new_tokens)?;
        let hidden_size = m.cfg.hidden_size;
        let eps = m.cfg.rmsnorm_eps;
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
            hidden = self.forward_one_token_mla(m, next_id, position, &mut kv_caches)?;
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
        Ok((generated, text, kv_caches, position))
    }

    /// Embeds `token_id` and runs it through every MLA layer at absolute
    /// `position`, writing this position's compressed `Kcur` into
    /// `kv_caches` (preallocated device buffers, see `generate_mla_impl`).
    pub(super) fn forward_one_token_mla(
        &self,
        m: &MlaModel,
        token_id: u32,
        position: usize,
        kv_caches: &mut [CudaSlice<f32>],
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let cfg = &m.cfg;
        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd.row(token_id)?)
            .map_err(|e| crate::gpu_err!(e, "embedding htod: {e}"))?;

        for (layer_idx, layer) in m.layers.iter().enumerate() {
            let post_attn =
                self.forward_mla_attn_block(m, layer, hidden, position, &mut kv_caches[layer_idx])?;
            hidden = match &layer.ffn {
                MlaFfn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => self.forward_hybrid_ffn(
                    post_attn,
                    &layer.ffn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    hidden_size,
                    ffn_hidden_size,
                    eps,
                )?,
                MlaFfn::Moe { .. } => {
                    let moe_cfg = cfg.moe.as_ref().ok_or_else(|| {
                        ReflexError::Other(
                            "internal error: MlaFfn::Moe layer but MlaConfig::moe is None"
                                .to_string(),
                        )
                    })?;
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
    pub fn forward_prompt_capture_kv_mla(
        &self,
        prompt: &str,
    ) -> Result<((u32, String), crate::kv_io::MlaKvCache), ReflexError> {
        let m = self.mla.as_ref().ok_or_else(|| {
            ReflexError::Other(
                "forward_prompt_capture_kv_mla called on a non-MLA model".to_string(),
            )
        })?;
        let (generated, text, kv_caches, seq_len) = self.generate_mla_impl(
            m,
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;

        let qk_dim = m.cfg.kv_lora_rank + m.cfg.qk_rope_head_dim;
        let per_layer_len = seq_len * qk_dim;
        let kv_caches = kv_caches
            .iter()
            .map(|c| {
                self.device
                    .dtoh_sync_copy(&c.slice(0..per_layer_len))
                    .map_err(|e| crate::gpu_err!(e, "mla kv_cache dtoh: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let cache = crate::kv_io::MlaKvCache {
            seq_len,
            qk_dim,
            kv_caches,
        };
        Ok(((generated[0], text), cache))
    }
}
