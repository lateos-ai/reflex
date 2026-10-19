//! The dense decode step on reused device buffers. [`Model::forward_one_token_dense`]
//! allocates every op's output (about 400 stream-ordered allocations and frees per
//! step on Qwen3-0.6B) and copies each layer's K and V into the cache. Here every
//! intermediate lives in a [`DecodeWorkspace`], allocated on the first decode step
//! and reused for the rest of the process, and K/V are written straight into their
//! cache rows. The kernels, their launch geometry and their order of operations are
//! the same as the allocating path's, so the logits are the same bits
//! (`decode_step_tests.rs`). It also leaves the step free of allocations, which a
//! captured CUDA graph needs.
//!
//! Used by `generate_dense_impl` for models whose layers are all dense, with the
//! online attention kernels; MoE and Kolibri-1 layers (host-side routing) and
//! `REFLEX_ATTN_KERNEL=legacy` keep the allocating path. `REFLEX_DECODE_WS=0` turns
//! it off for A/B runs.

use super::*;
use crate::error::ReflexError;

/// Whether the workspace path is on (`REFLEX_DECODE_WS`), and the workspace once
/// the first decode step has allocated it.
pub(super) struct DecodeWorkspaceSlot {
    enabled: Cell<bool>,
    ws: RefCell<Option<DecodeWorkspace>>,
}

impl DecodeWorkspaceSlot {
    pub(super) fn from_env() -> Result<Self, ReflexError> {
        let enabled = match std::env::var("REFLEX_DECODE_WS").as_deref() {
            Err(_) | Ok("") | Ok("1") => true,
            Ok("0") => false,
            Ok(other) => {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "REFLEX_DECODE_WS must be 1 (the default) or 0, got {other:?}"
                ))
            }
        };
        Ok(Self {
            enabled: Cell::new(enabled),
            ws: RefCell::new(None),
        })
    }

    /// For tests that compare the two paths on one loaded model.
    #[cfg(test)]
    pub(super) fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    /// Whether the workspace has been allocated (tests).
    #[cfg(test)]
    pub(super) fn allocated(&self) -> bool {
        self.ws.borrow().is_some()
    }
}

/// Every intermediate of one dense decode step, sized from the model's config.
/// The output buffers are written in full by their kernels before they are read,
/// so none is zeroed.
pub(super) struct DecodeWorkspace {
    /// The residual stream: the token's embedding, then each layer's output.
    hidden: CudaSlice<f32>,
    /// The current layer's normed input to Q/K/V.
    normed: CudaSlice<f32>,
    q: CudaSlice<f32>,
    /// Q after Q-Norm.
    q_normed: CudaSlice<f32>,
    /// K before K-Norm, for models with one (otherwise K goes straight into the cache).
    k: CudaSlice<f32>,
    /// This position's final K and V rows with an f16 cache, stored into it by
    /// `kv_store` (an f32 cache gets them written straight into its row).
    k_row: CudaSlice<f32>,
    v_row: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    /// Split-decode partials for the attention kernels.
    attn_partial: CudaSlice<f32>,
    o_proj: CudaSlice<f32>,
    ffn_normed: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    /// The FFN output, added to `hidden` by the next layer's fused norm.
    down: CudaSlice<f32>,
    final_normed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

impl DecodeWorkspace {
    fn new(model: &Model) -> Result<Self, ReflexError> {
        let cfg = &model.cfg;
        let q_len = cfg.num_q_heads * cfg.head_dim;
        let kv_len = cfg.num_kv_heads * cfg.head_dim;
        let vocab = model.lm_head_resident()?.shape[1] as usize;
        let buf = |len: usize| model.alloc_out(len, "decode workspace");
        Ok(Self {
            hidden: buf(cfg.hidden_size)?,
            normed: buf(cfg.hidden_size)?,
            q: buf(q_len)?,
            q_normed: buf(q_len)?,
            k: buf(kv_len)?,
            k_row: buf(kv_len)?,
            v_row: buf(kv_len)?,
            attn: buf(q_len)?,
            attn_partial: buf(cfg.num_q_heads * ATTN_ONLINE_MAX_SPLITS * (cfg.head_dim + 2))?,
            o_proj: buf(cfg.hidden_size)?,
            ffn_normed: buf(cfg.hidden_size)?,
            gate: buf(cfg.ffn_hidden_size)?,
            up: buf(cfg.ffn_hidden_size)?,
            act: buf(cfg.ffn_hidden_size)?,
            down: buf(cfg.hidden_size)?,
            final_normed: buf(cfg.hidden_size)?,
            logits: buf(vocab)?,
        })
    }
}

/// A device pointer to the start of `s`.
fn ptr(s: &CudaSlice<f32>) -> u64 {
    *s.device_ptr()
}

impl Model {
    /// Whether [`Self::decode_step_dense_ws`] serves this model's decode steps.
    pub(super) fn decode_ws_enabled(&self) -> bool {
        self.decode_ws.enabled.get()
            && self.attn_impl == AttnImpl::Online
            && self
                .layers
                .iter()
                .all(|l| matches!(l, LayerWeights::Dense(_)))
    }

    /// One decode step through every dense layer at `position`, then the final
    /// norm and LM head: the same ops as [`Self::forward_one_token_dense`] and
    /// [`Self::lm_head_logits`], on workspace buffers, with this position's K/V
    /// written straight into `k_caches`/`v_caches` (an f32 cache), or rounded into
    /// them from two workspace rows (an f16 cache, `REFLEX_KV_CACHE=f16`). The
    /// logits land in `logits`.
    pub(super) fn decode_step_dense_ws(
        &self,
        token_id: u32,
        position: usize,
        k_caches: &mut [KvBuf],
        v_caches: &mut [KvBuf],
        logits: &mut Vec<f32>,
    ) -> Result<(), ReflexError> {
        let mut slot = self.decode_ws.ws.borrow_mut();
        if slot.is_none() {
            *slot = Some(DecodeWorkspace::new(self)?);
        }
        let ws = slot.as_mut().expect("just set");
        let cfg = &self.cfg;
        let (hs, eps) = (cfg.hidden_size, cfg.rmsnorm_eps);
        let (nq, nkv, hd) = (cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim);
        let kv_stride = nkv * hd;
        let row = position * kv_stride;
        let seq_len = position + 1;

        // Async embedding upload: copy to pinned buffer and launch H2D on
        // a side stream, then wait for it before launching kernels. This
        // overlaps the ~4 KB embedding upload with host-side work between
        // tokens (argmax/picker).
        let embd = self.token_embd.row(token_id)?;
        {
            let mut pipeline = self.dequant_pipeline.borrow_mut();
            pipeline.htod_async_embedding(&embd, *ws.hidden.device_ptr())?;
        }
        {
            let mut pipeline = self.dequant_pipeline.borrow_mut();
            pipeline.wait_embedding_copy()?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let LayerWeights::Dense(l) = layer else {
                return Err(crate::reflex_err!(
                    Other,
                    "internal: decode workspace on a non-dense layer"
                ));
            };
            if let Some(cache) = [&k_caches[i], &v_caches[i]]
                .into_iter()
                .find(|c| c.len() < seq_len * kv_stride)
            {
                return Err(crate::reflex_err!(
                    Other,
                    "internal: KV cache of {} values has no room for position {position}",
                    cache.len()
                ));
            }
            // Where this position's final K and V go: the cache row itself, or the
            // workspace rows that `kv_store` rounds into an f16 cache.
            let direct = k_caches[i].dtype() == crate::kv_io::KvDtype::F32;
            let (k_row, v_row) = if direct {
                (k_caches[i].ptr_at(row), v_caches[i].ptr_at(row))
            } else {
                (ptr(&ws.k_row), ptr(&ws.v_row))
            };

            // Attention norm, adding the previous layer's FFN output first.
            if i == 0 {
                self.rmsnorm_to(&ws.hidden, l.attn_norm.f32()?, 1, hs, eps, ptr(&ws.normed))?;
            } else {
                self.residual_rmsnorm_to(
                    &mut ws.hidden,
                    &ws.down,
                    l.attn_norm.f32()?,
                    hs,
                    eps,
                    ptr(&ws.normed),
                )?;
            }

            // Q/K/V. K goes to the cache row unless K-Norm still has to read it.
            let k_out = if l.attn_k_norm.is_some() {
                ptr(&ws.k)
            } else {
                k_row
            };
            self.gemv_qkv_to(&ws.normed, &l.attn_q, &l.attn_k, &l.attn_v, ptr(&ws.q), k_out, v_row)?;
            if let Some(b) = &l.attn_bias {
                self.add_at(ptr(&ws.q), nq * hd, b.q.f32()?, nq * hd)?;
                self.add_at(k_out, kv_stride, b.k.f32()?, kv_stride)?;
                self.add_at(v_row, kv_stride, b.v.f32()?, kv_stride)?;
            }
            let q_normed = match (&l.attn_q_norm, &l.attn_k_norm) {
                (Some(qn), Some(kn)) => {
                    self.rmsnorm_pair_to(
                        &ws.q,
                        qn.f32()?,
                        nq,
                        &ws.k,
                        kn.f32()?,
                        nkv,
                        hd,
                        eps,
                        (ptr(&ws.q_normed), k_row),
                    )?;
                    true
                }
                (qn, kn) => {
                    if let Some(kn) = kn {
                        self.rmsnorm_to(&ws.k, kn.f32()?, nkv, hd, eps, k_row)?;
                    }
                    if let Some(qn) = qn {
                        self.rmsnorm_to(&ws.q, qn.f32()?, nq, hd, eps, ptr(&ws.q_normed))?;
                    }
                    qn.is_some()
                }
            };
            let q = if q_normed { &ws.q_normed } else { &ws.q };
            self.rope_at(
                ptr(q),
                nq,
                hd,
                cfg.rotary_dim,
                position,
                cfg.rope_base,
                cfg.rope_type,
            )?;
            self.rope_at(
                k_row,
                nkv,
                hd,
                cfg.rotary_dim,
                position,
                cfg.rope_base,
                cfg.rope_type,
            )?;

            if !direct {
                self.kv_store_kv(k_row, v_row, &k_caches[i], &v_caches[i], row, kv_stride)?;
            }
            self.attention_to(
                q,
                &k_caches[i].view(0..seq_len * kv_stride),
                &v_caches[i].view(0..seq_len * kv_stride),
                nq,
                nkv,
                hd,
                seq_len,
                0,
                ptr(&ws.attn),
                Some(ptr(&ws.attn_partial)),
            )?;
            self.gemv_to(&ws.attn, &l.attn_output, ptr(&ws.o_proj))?;
            self.residual_rmsnorm_to(
                &mut ws.hidden,
                &ws.o_proj,
                l.ffn_norm.f32()?,
                hs,
                eps,
                ptr(&ws.ffn_normed),
            )?;

            self.gemv_gate_up_to(&ws.ffn_normed, &l.ffn_gate, &l.ffn_up, ptr(&ws.gate), ptr(&ws.up))?;
            self.silu_and_mul_to(&ws.gate, &ws.up, cfg.ffn_hidden_size, ptr(&ws.act))?;
            self.gemv_to(&ws.act, &l.ffn_down, ptr(&ws.down))?;
        }
        if !self.layers.is_empty() {
            self.add_inplace(&mut ws.hidden, &ws.down)?;
        }

        self.rmsnorm_to(
            &ws.hidden,
            self.output_norm.f32()?,
            1,
            hs,
            eps,
            ptr(&ws.final_normed),
        )?;
        self.gemv_to(&ws.final_normed, self.lm_head_resident()?, ptr(&ws.logits))?;
        logits.resize(ws.logits.len(), 0.0);
        self.device
            .dtoh_sync_copy_into(&ws.logits, logits)
            .map_err(|e| crate::gpu_err!(e, "logits dtoh: {e}"))
    }
}
