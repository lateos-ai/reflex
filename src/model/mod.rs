//! Dense and MoE Qwen3 forward pass (MVP steps 1-2, see README.md's MVP
//! order): loads a GGUF file's weights, runs embedding -> every transformer
//! layer -> final RMSNorm -> LM head -> next-token choice for the *first*
//! generated token only. No KV cache reuse across separate calls, no
//! batching -- those stay permanent constraints (see README.md's
//! Non-goals) -- but next-token choice is no longer greedy-only: greedy
//! argmax is still the default and is what this project's own byte-exact-
//! vs-llama.cpp verification methodology depends on, but `crate::sampling`
//! adds temperature/top-k/top-p sampling as an explicit opt-in (sampling
//! strategy was never in README.md's permanent-constraints list, only
//! unimplemented scope). The target metric is still process-start-to-
//! first-token latency, not sustained decode throughput (see README.md's
//! "Why this exists").
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
//!
//! Layout: this file holds the `Model` struct, its public API and the dispatch to
//! each architecture. `config` parses GGUF metadata into the config types,
//! `loading` uploads and dequantizes weights, `kernels` wraps the AOT CUDA kernels
//! and cuBLAS calls, and `dense` (dense and MoE Qwen3, Llama/Mistral), `hybrid`
//! (Qwen3.5 Gated DeltaNet) and `mla` (DeepSeek-V2/V3) hold each architecture's
//! loading and forward passes. All of them are `impl Model` blocks or types private
//! to this module.

use crate::aot::{self, AotKernel};
use crate::dequant;
use crate::diagnostics;
use crate::gguf::{GgmlType, GgufFile, GgufValue};
use crate::lora;
use crate::moe::{route_top_k, route_top_k_with_norm};
use crate::sampling::SamplingParams;
use crate::tokenizer::Tokenizer;
use cudarc::cublas::sys as cublas_sys;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    result, sys, CudaDevice, CudaSlice, CudaView, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync,
    LaunchConfig,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::ScopedJoinHandle;

mod config;
mod dense;
mod hybrid;
mod kernels;
mod loading;
mod mla;

use self::config::*;
pub use self::config::{parse_model_config, LayerConfig, MoeMetaConfig, RopeType};
use self::dense::*;
use self::hybrid::*;
use self::loading::*;
use self::mla::*;
use crate::error::ReflexError;

#[cfg(test)]
mod hybrid_batching_tests;
#[cfg(test)]
mod iq_dequant_host_vs_device_tests;
#[cfg(test)]
mod mla_batching_tests;
#[cfg(test)]
mod moe_fixture_tests;
#[cfg(test)]
mod prefill_batching_tests;
#[cfg(test)]
mod rope_type_tests;
#[cfg(test)]
mod system1_tests;

/// Host-side logistic sigmoid, for `qwen35moe`'s per-token shared-expert
/// gate (a single scalar per row -- see `Model::forward_hybrid_moe_ffn`).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    /// `rope_norm_kernel` (`kernels_cuda/rope.cu`) -- the consecutive-pair
    /// ([`RopeType::Norm`]) rotation Llama/Mistral need, loaded from the same
    /// `REFLEX_KERNEL_ROPE` module as `rope_k` (unlike `MlaModel`'s separate
    /// copy). `None` for models that never use the Norm convention (hybrid,
    /// MLA), so no module lookup is paid where it isn't used.
    rope_norm_k: Option<AotKernel>,
    /// Batched-prefill variant of `rope_norm_k` (`rope_norm_batch_kernel`,
    /// `kernels_cuda/rope.cu`).
    rope_norm_batch_k: Option<AotKernel>,
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
    /// Kept host-resident (unlike every other weight) for embedding lookup
    /// (host-side gather; batch is always 1 in this MVP, so a GPU gather
    /// kernel buys nothing) -- lazily dequantized row-by-row, see
    /// [`LazyTokenEmbedding`]. When no separate `output.weight` tensor
    /// exists, `lm_head` reuses this same lazy source instead of
    /// dequantizing it twice (see [`LmHead::TiedLazy`]/
    /// `Model::lm_head_resident`/`Model::gemv_gather_lm_head`).
    token_embd: LazyTokenEmbedding,
    /// Kept alive past load solely so [`Self::lm_head_resident`] can
    /// dequantize a tied `token_embd`/`lm_head` on-device on first use
    /// (`dequantize_tensor_to_device`, the same on-device path every other
    /// weight tensor uses) instead of falling back to a slow host-side
    /// dequant loop. `RefCell` because `lm_head_resident` takes `&self`; safe
    /// without further synchronization for the same reason `LmHead::TiedLazy`'s
    /// `OnceLock` is -- this project never runs more than one request at a
    /// time (see docs/DEVELOPMENT.md's Non-goals).
    dequant_kernels: DequantKernels,
    dequant_pipeline: RefCell<WeightLoadPipeline>,
    output_norm: Weight,
    lm_head: LmHead,
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
/// (currently `reflex generate`'s `--export-kv`/`--import-kv` handling) that
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
    /// like `reflex bench` that need to report actual prompt length,
    /// without exposing the private `tokenizer` field itself.
    pub fn encoded_prompt_len(&self, prompt: &str) -> Result<usize, ReflexError> {
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
    /// Deliberately narrow (see `Self::find_lora_target_mut`'s doc comment
    /// for the exact accepted tensor set): dense/MoE attention tensors, MoE's
    /// three per-expert-stacked FFN tensors (`ffn_gate_exps`/`ffn_up_exps`/
    /// `ffn_down_exps` -- added after this round; see `lora.rs`'s module doc
    /// comment for the exact adapter format this expects and why a
    /// fused/grouped-expert LoRA representation some training frameworks
    /// produce doesn't qualify), Qwen3.5 hybrid Gated-Attention-layer
    /// tensors, and the hybrid Gated DeltaNet mixer's five 2-D
    /// Linear-shaped tensors (`attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`/
    /// `ssm_out`) only. DeepSeek-V2/V3 MLA is rejected outright below,
    /// matching every other MLA-excluded feature in this codebase. Any
    /// adapter tensor that doesn't resolve to a supported base weight, or
    /// whose shape doesn't match that weight's, is a hard error -- never a
    /// silent skip.
    pub fn apply_lora(&mut self, path: &std::path::Path) -> Result<usize, ReflexError> {
        if self.mla.is_some() {
            return Err(ReflexError::Lora(
                "--lora is not supported for DeepSeek-V2/V3 MLA models in this round -- only dense/MoE Qwen3 \
                 and the Qwen3.5 hybrid architecture are supported LoRA base models"
                    .to_string(),
            ));
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
                .map_err(|e| crate::gpu_err!(e, "upload LoRA delta for '{}': {e}", target.name))?;

            let weight = self.find_lora_target_mut(&target.name).ok_or_else(|| {
                crate::reflex_err!(Lora,
                    "LoRA adapter targets '{}' but this project's Model has no matching weight for it \
                     (dense/MoE attention+FFN including MoE's per-expert-stacked ffn_*_exps tensors, Qwen3.5 \
                     hybrid Gated-Attention-layer tensors, and the hybrid Gated DeltaNet mixer's \
                     attn_qkv/attn_gate/ssm_alpha/ssm_beta/ssm_out are the only supported LoRA targets in this \
                     round -- the mixer's remaining non-Linear state-space tensors (ssm_dt/ssm_a/ssm_conv1d/ \
                     ssm_norm), embeddings, norms, and MLA are not)",
                    target.name
                )
            })?;
            let shape_ok = match weight.shape.as_slice() {
                [in_f, out_f] => {
                    target.expert_count.is_none()
                        && *in_f as usize == target.in_features
                        && *out_f as usize == target.out_features
                }
                [in_f, out_f, expert_count] => {
                    target.expert_count == Some(*expert_count as usize)
                        && *in_f as usize == target.in_features
                        && *out_f as usize == target.out_features
                }
                _ => false,
            };
            if !shape_ok {
                return Err(crate::reflex_err!(Lora,
                    "LoRA adapter tensor '{}' has shape [in={}, out={}, experts={:?}] but the base model's tensor has shape {:?} -- wrong base model?",
                    target.name, target.in_features, target.out_features, target.expert_count, weight.shape
                ));
            }

            let n = weight.data.len() as u32;
            let threads = 256u32;
            let blocks = n.div_ceil(threads).max(1);
            let launch_cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                add_fn
                    .clone()
                    .launch(launch_cfg, (&mut weight.data, &delta_dev, n))
                    .map_err(|e| {
                        crate::gpu_err!(e, "LoRA add launch for '{}': {e}", target.name)
                    })?;
            }
            applied += 1;
        }

        if applied == 0 {
            return Err(ReflexError::Lora("LoRA adapter matched no tensors in the base model -- check it targets a compatible architecture/checkpoint".to_string()));
        }
        Ok(applied)
    }

    /// Locates the mutable device-resident [`Weight`] a LoRA adapter target
    /// name (`blk.{i}.{suffix}.weight`) refers to, across whichever
    /// architecture is loaded (dense/MoE via `self.layers`, Qwen3.5 hybrid
    /// via `self.hybrid`). Matches every 2-D `nn.Linear`-shaped tensor each
    /// supported layer kind has, plus MoE's three per-expert-stacked 3-D FFN
    /// tensors (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` -- see
    /// `lora.rs`'s module doc comment for how a standard per-expert-`Linear`
    /// PEFT adapter's `lora_a`/`lora_b` end up as matching 3-D tensors after
    /// llama.cpp's `convert_lora_to_gguf.py`, and `Self::apply_lora`'s shape
    /// check for how the two are told apart). Anything outside a `blk.N.*`
    /// tensor (`token_embd`/`output`/norms), and the mixer's non-Linear
    /// state-space tensors, fall through to the `None` arm and are rejected
    /// by `Self::apply_lora` with a clear error, rather than silently
    /// mismatched or misapplied.
    ///
    /// The Gated DeltaNet mixer's `attn_qkv`/`attn_gate`/`ssm_alpha`/
    /// `ssm_beta`/`ssm_out` *are* matched here despite an earlier version of
    /// this comment calling the whole mixer "non-Linear": those five are
    /// plain 2-D `Weight`s driven by the same `gemv` every other Linear
    /// projection uses (see `forward_gated_attn_mixer`) -- only
    /// `ssm_dt`/`ssm_a`/`ssm_conv1d`/`ssm_norm` (a bias vector, a decay
    /// vector, a conv1d kernel, and a norm weight -- none of them a PEFT
    /// `nn.Linear` module, confirmed by a real downloaded adapter's tensor
    /// names never including any of the four) are genuinely non-Linear and
    /// stay unmatched. Confirmed against a real adapter
    /// (`Tilakoid/qwen3.5-0.8b-hoasa-lora`, targets `Qwen/Qwen3.5-0.8B`
    /// exactly): its safetensors header's `linear_attn.in_proj_qkv` /
    /// `.in_proj_z` / `.in_proj_a` / `.in_proj_b` / `.out_proj` module names
    /// map onto `attn_qkv` / `attn_gate` / `ssm_alpha` / `ssm_beta` /
    /// `ssm_out` respectively (llama.cpp's `convert_lora_to_gguf.py` reuses
    /// the base model's own name-mapping table, so the GGUF base tensor
    /// names line up with this project's own base-model loader above).
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
                (HybridLayerWeights::GatedAttention(l), "ffn_gate" | "ffn_up" | "ffn_down") => {
                    l.ffn.dense_weight_mut(suffix)
                }
                (HybridLayerWeights::GatedDeltaNet(l), "ffn_gate" | "ffn_up" | "ffn_down") => {
                    l.ffn.dense_weight_mut(suffix)
                }
                (HybridLayerWeights::GatedDeltaNet(l), "attn_qkv") => Some(&mut l.attn_qkv),
                (HybridLayerWeights::GatedDeltaNet(l), "attn_gate") => Some(&mut l.attn_gate),
                (HybridLayerWeights::GatedDeltaNet(l), "ssm_alpha") => Some(&mut l.ssm_alpha),
                (HybridLayerWeights::GatedDeltaNet(l), "ssm_beta") => Some(&mut l.ssm_beta),
                (HybridLayerWeights::GatedDeltaNet(l), "ssm_out") => Some(&mut l.ssm_out),
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
            (LayerWeights::Moe(l), "ffn_gate_exps") => Some(&mut l.ffn_gate_exps),
            (LayerWeights::Moe(l), "ffn_up_exps") => Some(&mut l.ffn_up_exps),
            (LayerWeights::Moe(l), "ffn_down_exps") => Some(&mut l.ffn_down_exps),
            _ => None,
        }
    }

    pub fn load(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, ReflexError> {
        diagnostics::check_kernel_compute_capability(&device)?;
        let architecture = file
            .metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .unwrap_or("");
        if architecture == "qwen35" || architecture == "qwen35moe" {
            return Self::load_hybrid(device, file);
        }
        if architecture == "deepseek2" {
            return Self::load_mla(device, file);
        }

        Self::load_dense(device, file)
    }

    /// Builds the `Tokenizer` and the cuBLAS handle (with `CUBLAS_PEDANTIC_MATH`
    /// pinned) for the one scoped background thread each `load_*` wrapper spawns,
    /// so their combined host/driver setup overlaps the GPU-bound weight load
    /// instead of sitting on the serial path. Both are pure setup needing no loaded
    /// weights; `CudaBlas::new` binds the primary context to this thread itself
    /// (`device.bind_to_thread`). Run sequentially on one thread rather than two so
    /// the load still spawns exactly one extra thread -- see DECISIONS.md's entries
    /// on the load-time worker thread and the cuBLAS-init overlap.
    fn load_background_init(
        file: &GgufFile,
        device: Arc<CudaDevice>,
    ) -> Result<(Tokenizer, CudaBlas), ReflexError> {
        let tokenizer = Tokenizer::from_gguf(file)?;
        let cublas =
            CudaBlas::new(device).map_err(|e| crate::gpu_err!(e, "cublas handle: {e:?}"))?;
        unsafe {
            cublas_sys::lib()
                .cublasSetMathMode(
                    *cublas.handle(),
                    cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                )
                .result()
                .map_err(|e| crate::gpu_err!(e, "cublasSetMathMode: {e:?}"))?;
        }
        Ok((tokenizer, cublas))
    }

    /// Encodes `prompt`, runs it through every layer one position at a time
    /// (real causal self-attention throughout, matching RustFeference's own
    /// documented scope choice for its minimal forward pass), and returns
    /// the argmax-sampled first generated token id plus its decoded text.
    pub fn forward_prompt(&self, prompt: &str) -> Result<(u32, String), ReflexError> {
        if let Some(h) = &self.hybrid {
            return self.forward_prompt_hybrid(h, prompt);
        }
        if let Some(m) = &self.mla {
            return self.forward_prompt_mla(m, prompt);
        }
        let (generated, text, _k_caches, _v_caches, _seq_len) = self.generate_dense_impl(
            prompt,
            None,
            1,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )?;
        Ok((generated[0], text))
    }

    /// Phase 3 (State I/O) round 2's generation entry point: like
    /// `forward_prompt`, but produces up to `max_new_tokens` tokens (feeding
    /// each generated id back in as the next position's input embedding,
    /// stopping early on the tokenizer's `eos_token_id`) and, when
    /// `imported` is `Some`, resumes from a previously exported cache
    /// instead of starting at position 0 -- `prompt` is then the
    /// continuation text appended after the imported cache's positions, not
    /// a fresh prompt (no BOS is inserted). `sampling` selects the next-token
    /// choice strategy every generated position uses -- `&SamplingParams::default()`
    /// (greedy argmax) is byte-identical to this function's behavior before
    /// `crate::sampling` existed; see that module's doc comment for
    /// temperature/top-k/top-p. `on_first_token` is called exactly once,
    /// right after the first new token is produced, with that token's full
    /// logits vector -- callers can ignore the argument to just capture
    /// accurate "time to first token" timing (as `src/bin/reflex/generate.rs`
    /// does), or inspect the logits themselves (as `src/bin/check_correctness.rs`
    /// does) -- even when `max_new_tokens > 1` keeps the call running past
    /// that point. `on_token` is called once per generated token (including
    /// the first, alongside `on_first_token`), with that token's id and its
    /// incrementally-decoded text (`Tokenizer::decode_stream`) -- this is
    /// the hook `crate::ipc::handle_request_streaming` uses to emit each
    /// token over the wire as soon as it's ready, instead of buffering the
    /// whole response.
    ///
    /// Dense/MoE and the Qwen3.5 hybrid mixer support resume as of round 2
    /// (the hybrid `GatedDeltaNet` sublayers' `conv_state`/`recurrent` need
    /// no `start_pos` handling at all -- see `kv_io.rs`'s doc comment); MLA
    /// as of round 3, via `generate_mla_impl`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        imported: Option<&crate::kv_io::ImportedKv>,
        sampling: &SamplingParams,
        on_first_token: impl FnMut(&[f32]),
        on_token: impl FnMut(u32, &str),
    ) -> Result<(Vec<u32>, String), ReflexError> {
        match imported {
            Some(crate::kv_io::ImportedKv::Dense(cache)) => {
                if self.hybrid.is_some() || self.mla.is_some() {
                    return Err(ReflexError::KvCache("imported KV cache file is dense/MoE format, but this model is not a dense/MoE Qwen3 model".to_string()));
                }
                let (generated, text, _, _, _) = self.generate_dense_impl(
                    prompt,
                    Some(cache),
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?;
                Ok((generated, text))
            }
            Some(crate::kv_io::ImportedKv::Hybrid(cache)) => {
                let h = self
                    .hybrid
                    .as_ref()
                    .ok_or_else(|| ReflexError::KvCache("imported KV cache file is hybrid format, but this model is not a Qwen3.5 hybrid model".to_string()))?;
                let (generated, text, _, _) = self.generate_hybrid_impl(
                    h,
                    prompt,
                    Some(cache),
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?;
                Ok((generated, text))
            }
            Some(crate::kv_io::ImportedKv::Mla(cache)) => {
                let m = self.mla.as_ref().ok_or_else(|| {
                    ReflexError::KvCache(
                        "imported KV cache file is MLA format, but this model is not an MLA model"
                            .to_string(),
                    )
                })?;
                let (generated, text, _, _) = self.generate_mla_impl(
                    m,
                    prompt,
                    Some(cache),
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?;
                Ok((generated, text))
            }
            None => {
                if let Some(h) = &self.hybrid {
                    let (generated, text, _, _) = self.generate_hybrid_impl(
                        h,
                        prompt,
                        None,
                        max_new_tokens,
                        sampling,
                        on_first_token,
                        on_token,
                    )?;
                    return Ok((generated, text));
                }
                if let Some(m) = &self.mla {
                    let (generated, text, _, _) = self.generate_mla_impl(
                        m,
                        prompt,
                        None,
                        max_new_tokens,
                        sampling,
                        on_first_token,
                        on_token,
                    )?;
                    return Ok((generated, text));
                }
                let (generated, text, _, _, _) = self.generate_dense_impl(
                    prompt,
                    None,
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?;
                Ok((generated, text))
            }
        }
    }

    /// Copies row `rows - 1` (the last prompt position) out of a batched
    /// `[rows, hidden_size]` prefill output (`Self::prefill_dense_batched`)
    /// into its own owned buffer. Every current caller only wants that row
    /// to continue generation/scoring from, but needs it as an owned
    /// `CudaSlice` rather than a borrowed view, since callers go on to
    /// reassign it from `Self::forward_one_token_dense`'s per-token decode
    /// loop.
    fn last_row(
        &self,
        hidden_batched: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let offset = (rows - 1) * hidden_size;
        let mut out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| crate::gpu_err!(e, "last_row alloc: {e}"))?;
        let src = hidden_batched.slice(offset..offset + hidden_size);
        self.device
            .dtod_copy(&src, &mut out)
            .map_err(|e| crate::gpu_err!(e, "last_row dtod: {e}"))?;
        Ok(out)
    }

    /// Like [`Self::last_row`], but for any row index -- used by the hybrid
    /// model's layer-major batched prefill (`Self::forward_hybrid_layer_batched`)
    /// to pull one position's hidden vector out of a `[rows, hidden_size]`
    /// buffer before feeding it through a `GatedDeltaNet` layer's sequential
    /// per-token recurrence (`Self::forward_gdn_mixer` takes ownership of a
    /// single-row `CudaSlice`, not a view into a larger batch).
    fn extract_row(
        &self,
        batched: &CudaSlice<f32>,
        row: usize,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let offset = row * hidden_size;
        let mut out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| crate::gpu_err!(e, "extract_row alloc: {e}"))?;
        let src = batched.slice(offset..offset + hidden_size);
        self.device
            .dtod_copy(&src, &mut out)
            .map_err(|e| crate::gpu_err!(e, "extract_row dtod: {e}"))?;
        Ok(out)
    }

    /// Inverse of [`Self::extract_row`]: writes `src` (one position's hidden
    /// vector) back into row `row` of a `[rows, hidden_size]` buffer -- same
    /// device-to-device copy convention `Self::forward_attn_block` already
    /// uses for kv-cache writes, never a host round trip.
    fn write_row(
        &self,
        batched: &mut CudaSlice<f32>,
        row: usize,
        hidden_size: usize,
        src: &CudaSlice<f32>,
    ) -> Result<(), ReflexError> {
        let offset = row * hidden_size;
        let mut dst = batched.slice_mut(offset..offset + hidden_size);
        self.device
            .dtod_copy(src, &mut dst)
            .map_err(|e| crate::gpu_err!(e, "write_row dtod: {e}"))
    }

    /// Resolves `candidate`'s actual continuation token ids given `prompt`,
    /// by tokenizing `prompt` and `prompt + candidate` together and diffing
    /// -- tokenizing `candidate` alone does not reliably give the token(s)
    /// the model would actually emit as a continuation, since BPE/
    /// SentencePiece merge boundaries depend on what precedes the candidate
    /// text. Errs if `encode(prompt)` is not an exact prefix of
    /// `encode(prompt + candidate)`, or the candidate contributes zero new
    /// tokens.
    fn resolve_candidate_token_ids(
        &self,
        prompt: &str,
        candidate: &str,
    ) -> Result<Vec<u32>, ReflexError> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        let full_ids = self.tokenizer.encode(&format!("{prompt}{candidate}"))?;
        if full_ids.len() <= prompt_ids.len() || full_ids[..prompt_ids.len()] != prompt_ids[..] {
            return Err(crate::reflex_err!(InvalidInput, "system1: candidate {candidate:?} does not tokenize as a clean continuation of the prompt"));
        }
        Ok(full_ids[prompt_ids.len()..].to_vec())
    }

    /// System1: single-pass, non-autoregressive candidate scoring. Runs
    /// `prompt` through the same prefill path `generate` uses exactly once,
    /// then scores every one of `candidates` from that single prefill -- no
    /// argmax-then-feedback decode loop for single-token candidates (one
    /// batched gather-GEMV covers all of them, `Self::gemv_gather`), and
    /// only a short teacher-forced continuation for multi-token ones
    /// (feeding each candidate's own known next token, never a sampled
    /// one) where that's supported -- see [`Self::system1_evaluate_dense`]/
    /// [`Self::system1_evaluate_hybrid`]/[`Self::system1_evaluate_mla`],
    /// dispatched to here exactly the way `forward_prompt`/`generate`
    /// dispatch to their own `_dense`/`_hybrid`/`_mla` impls.
    ///
    /// `score` is relative to this candidate set only, not a vocab-wide
    /// log-probability -- computing the latter for a single-token candidate
    /// would require the exact full-vocab GEMV + D2H transfer this method
    /// exists to avoid. `temperature` (`1.0` = no-op) is passed through to
    /// `crate::calibration::softmax_scores_with_temperature` to produce
    /// `System1Response::probabilities`.
    ///
    /// Every candidate must resolve to at least one token (see
    /// `Self::resolve_candidate_token_ids`); a resolution failure for one
    /// candidate fails the whole call.
    pub fn system1_evaluate(
        &self,
        prompt: &str,
        candidates: &[System1Candidate],
        temperature: f32,
    ) -> Result<System1Response, ReflexError> {
        if let Some(h) = &self.hybrid {
            return self.system1_evaluate_hybrid(h, prompt, candidates, temperature);
        }
        if let Some(m) = &self.mla {
            return self.system1_evaluate_mla(m, prompt, candidates, temperature);
        }
        self.system1_evaluate_dense(prompt, candidates, temperature)
    }

    /// Shared tail of every `system1_evaluate_*` impl: turns raw per-candidate
    /// `scores` into a [`System1Response`] (calibrated `probabilities` +
    /// `entropy`). Factored out since it's identical across all three
    /// architectures -- unlike the prefill/continuation logic above it,
    /// which differs enough per architecture (different state shapes,
    /// different hazards) to be worth keeping separate.
    fn finish_system1_response(
        candidates: &[System1Candidate],
        resolved: Vec<Vec<u32>>,
        scores: Vec<f32>,
        temperature: f32,
    ) -> Result<System1Response, ReflexError> {
        let probabilities =
            crate::calibration::softmax_scores_with_temperature(&scores, temperature)?;
        let entropy = crate::calibration::shannon_entropy(&probabilities)?;
        let results = candidates
            .iter()
            .zip(resolved)
            .zip(scores)
            .map(|((c, token_ids), score)| System1CandidateResult {
                text: c.text.clone(),
                token_ids,
                score,
            })
            .collect();
        Ok(System1Response {
            results,
            probabilities,
            entropy,
        })
    }

    /// Final RMSNorm -> LM head -> argmax. `crate::sampling::sample`'s greedy
    /// path now covers this same computation for every architecture's
    /// generation loop (`lm_head_logits` + `Self::argmax`, same as here) --
    /// this wrapper is kept only as `prefill_batching_tests`' batched-vs-
    /// sequential-prefill oracle (`hidden_size`/`eps` differ by architecture;
    /// the `output_norm`/`lm_head` weights are shared across all of them),
    /// hence `#[cfg(test)]`.
    #[cfg(test)]
    fn lm_head_argmax(
        &self,
        hidden: &CudaSlice<f32>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<u32, ReflexError> {
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
    fn lm_head_logits(
        &self,
        hidden: &CudaSlice<f32>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>, ReflexError> {
        let normed = self.rmsnorm(hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits_dev = self.gemv(&normed, self.lm_head_resident()?)?;
        self.device
            .dtoh_sync_copy(&logits_dev)
            .map_err(|e| crate::gpu_err!(e, "logits dtoh: {e}"))
    }

    /// Forces the LM head fully device-resident, on-device-dequantizing
    /// `token_embd`'s raw quantized bytes if it hasn't been already (see
    /// [`LmHead`]'s doc comment) -- needed by [`Self::lm_head_logits`], which
    /// (unlike [`Self::gemv_gather_lm_head`]) genuinely needs every vocab
    /// row. Uses the same on-device `dequantize_tensor_to_device` path (and
    /// the `dequant_kernels`/`dequant_pipeline` kept alive on `Model` for
    /// exactly this) every other weight tensor's dequant goes through,
    /// instead of a slow single-threaded host dequant loop -- see
    /// HISTORY.md's "Lazy `token_embd` dequant (item 6)" for why the host
    /// path this replaced was a real, measured regression. A no-op past the
    /// first call (`Resident`, or a `TiedLazy` some earlier call already
    /// forced): `OnceLock::get`/`set` rather than the still-unstable
    /// `get_or_try_init`, safe without a race check because this project
    /// never runs more than one request at a time (`batch_size` is a
    /// permanent constraint, see docs/DEVELOPMENT.md's Non-goals) -- there is never a
    /// second caller to race against.
    fn lm_head_resident(&self) -> Result<&Weight, ReflexError> {
        match &self.lm_head {
            LmHead::Resident(w) => Ok(w),
            LmHead::TiedLazy { shape, cell } => {
                if let Some(w) = cell.get() {
                    return Ok(w);
                }
                let element_count =
                    self.token_embd.vocab_size as u64 * self.token_embd.hidden_size as u64;
                let data = dequantize_tensor_to_device(
                    &mut self.dequant_pipeline.borrow_mut(),
                    &self.dequant_kernels,
                    self.token_embd.ggml_type,
                    &self.token_embd.raw,
                    element_count,
                )
                .map_err(|e| {
                    e.rewrap(format!("upload weight 'token_embd.weight' to device: {e}"))
                })?;
                let _ = cell.set(Weight {
                    data,
                    shape: shape.clone(),
                });
                Ok(cell.get().expect("just set"))
            }
        }
    }

    /// [`Self::gemv_gather`], but for the LM head specifically -- avoids
    /// forcing a still-lazy tied LM head fully device-resident just to
    /// gather a handful of rows (System1's whole reason for existing; see
    /// [`LmHead`]'s doc comment). If the LM head is already fully resident
    /// (a real `output.weight` tensor, or a tied one some earlier full-vocab
    /// call already forced), this is exactly [`Self::gemv_gather`] with no
    /// extra cost. Otherwise, uploads only the requested rows -- typically a
    /// handful of candidate tokens, a few KB, not the full matrix's hundreds
    /// of MB -- straight from the host-resident `token_embd` bytes (same
    /// row-major `(vocab_size, hidden_size)` layout `output.weight` would
    /// have, since they're the same tensor when tied), then reuses
    /// `gemv_gather_kernel` against that compact buffer with trivial indices
    /// `0..row_indices.len()` (the buffer already IS exactly the selected
    /// rows, in order) -- same kernel, same math, just a much smaller upload.
    fn gemv_gather_lm_head(
        &self,
        x: &CudaSlice<f32>,
        row_indices: &[u32],
    ) -> Result<Vec<f32>, ReflexError> {
        let (shape, cell) = match &self.lm_head {
            LmHead::Resident(w) => return self.gemv_gather(x, w, row_indices),
            LmHead::TiedLazy { shape, cell } => (shape, cell),
        };
        if let Some(w) = cell.get() {
            return self.gemv_gather(x, w, row_indices);
        }

        let hidden_size = shape[0] as usize;
        let vocab_size = shape[1] as usize;
        if let Some(&bad) = row_indices.iter().find(|&&r| r as usize >= vocab_size) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_gather_lm_head: row index {bad} out of range (vocab_size={vocab_size})"
            ));
        }
        let mut compact = Vec::with_capacity(row_indices.len() * hidden_size);
        for &r in row_indices {
            compact.extend_from_slice(&self.token_embd.row(r)?);
        }
        let dev_compact = self
            .device
            .htod_sync_copy(&compact)
            .map_err(|e| crate::gpu_err!(e, "gemv_gather_lm_head upload compact rows: {e}"))?;
        let compact_w = Weight {
            data: dev_compact,
            shape: vec![hidden_size as u64, row_indices.len() as u64],
        };
        let trivial_indices: Vec<u32> = (0..row_indices.len() as u32).collect();
        self.gemv_gather(x, &compact_w, &trivial_indices)
    }

    /// `pub(crate)` (not private) so [`crate::sampling::sample`]'s greedy
    /// path can delegate straight here instead of duplicating this scan.
    pub(crate) fn argmax(logits: &[f32]) -> Result<u32, ReflexError> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| {
                ReflexError::InvalidInput("cannot argmax an empty logits slice".to_string())
            })
    }
}
