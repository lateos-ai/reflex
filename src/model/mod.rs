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
use self::config::*;
mod loading;
pub use self::config::{parse_model_config, LayerConfig, MoeMetaConfig, RopeType};
use self::loading::*;
mod dense;
use self::dense::*;
mod hybrid;
use self::hybrid::*;
#[cfg(test)]
mod hybrid_batching_tests;
#[cfg(test)]
mod iq_dequant_host_vs_device_tests;
mod kernels;
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

/// MLA prefill result: encoded prompt ids, the final position's hidden
/// state, the filled per-layer compressed-latent K/V caches, and the next
/// absolute position. Shared by `prefill_mla_batched` and its non-batched
/// counterpart.
type MlaPrefillResult = (Vec<u32>, CudaSlice<f32>, Vec<CudaSlice<f32>>, usize);

/// MLA generate result: generated token ids, their concatenated decoded
/// text, the final per-layer compressed-latent K/V caches, and the total
/// sequence length reached. Returned by `generate_mla_impl`.
type MlaGenerateResult = (Vec<u32>, String, Vec<CudaSlice<f32>>, usize);

/// Host-side logistic sigmoid, for `qwen35moe`'s per-token shared-expert
/// gate (a single scalar per row -- see `Model::forward_hybrid_moe_ffn`).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
                return Err(format!(
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
                    .map_err(|e| format!("LoRA add launch for '{}': {e}", target.name))?;
            }
            applied += 1;
        }

        if applied == 0 {
            return Err("LoRA adapter matched no tensors in the base model -- check it targets a compatible architecture/checkpoint".to_string());
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

    pub fn load(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
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
    ) -> Result<(Tokenizer, CudaBlas), String> {
        let tokenizer = Tokenizer::from_gguf(file)?;
        let cublas = CudaBlas::new(device).map_err(|e| format!("cublas handle: {e:?}"))?;
        unsafe {
            cublas_sys::lib()
                .cublasSetMathMode(
                    *cublas.handle(),
                    cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                )
                .result()
                .map_err(|e| format!("cublasSetMathMode: {e:?}"))?;
        }
        Ok((tokenizer, cublas))
    }

    /// Loads a DeepSeek-V2/V3 MLA model (MVP step 4). See [`parse_mla_config`] for
    /// the scope this supports. `cfg`/`layers`/`expert_used_count` below are
    /// unused garbage (matching the `hybrid` path's own convention) --
    /// `forward_prompt` branches on `self.mla` before touching them.
    fn load_mla(device: Arc<CudaDevice>, file: &GgufFile) -> Result<Self, String> {
        std::thread::scope(|scope| {
            let init_device = device.clone();
            let init = scope.spawn(move || Self::load_background_init(file, init_device));
            Self::load_mla_inner(device, file, init)
        })
    }

    fn load_mla_inner<'scope>(
        device: Arc<CudaDevice>,
        file: &GgufFile,
        init: ScopedJoinHandle<'scope, Result<(Tokenizer, CudaBlas), String>>,
    ) -> Result<Self, String> {
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
        let rope_k = rope_fns.next().ok_or("missing rope_kernel")?;
        let rope_batch_k = rope_fns.next().ok_or("missing rope_batch_kernel")?;
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
        let add_k = elementwise_fns.next().ok_or("missing add_kernel")?;
        let split_qg_k = elementwise_fns.next().ok_or("missing split_qg_kernel")?;
        let sigmoid_gate_k = elementwise_fns
            .next()
            .ok_or("missing sigmoid_gate_kernel")?;
        let mla_extract_batch_k = elementwise_fns
            .next()
            .ok_or("missing mla_extract_batch_kernel")?;
        let mla_concat_qcur_batch_k = elementwise_fns
            .next()
            .ok_or("missing mla_concat_qcur_batch_kernel")?;
        let mla_write_kv_cache_batch_k = elementwise_fns
            .next()
            .ok_or("missing mla_write_kv_cache_batch_kernel")?;
        let moe_gather_k = elementwise_fns.next().ok_or("missing moe_gather_kernel")?;
        let moe_scatter_add_k = elementwise_fns
            .next()
            .ok_or("missing moe_scatter_add_kernel")?;
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
        let rope_norm_k = rope_norm_fns.next().ok_or("missing rope_norm_kernel")?;
        let rope_norm_yarn_k = rope_norm_fns
            .next()
            .ok_or("missing rope_norm_yarn_kernel")?;
        let rope_norm_batch_k = rope_norm_fns
            .next()
            .ok_or("missing rope_norm_batch_kernel")?;
        let rope_norm_yarn_batch_k = rope_norm_fns
            .next()
            .ok_or("missing rope_norm_yarn_batch_kernel")?;
        let gemv_per_head_batch_k = aot::load_kernel(
            &device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV_PER_HEAD_BATCH")),
            "gemv_per_head_batch",
            "gemv_per_head_batch_kernel",
        )?;
        let dequant_kernels = load_dequant_kernels(&device)?;
        let mut pipeline = WeightLoadPipeline::new(&device)?;

        let mut load_weight = |name: &str| -> Result<Weight, String> {
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
            .ok_or_else(|| "missing weight 'token_embd.weight'".to_string())?;
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
                .map_err(|e| format!("load weight 'output.weight': {e}"))?;
                LmHead::Resident(Weight {
                    data,
                    shape: info.shape.clone(),
                })
            }
            None => {
                let data = dequantize_tensor_to_device(
                    &mut pipeline,
                    &dequant_kernels,
                    token_embd.ggml_type,
                    &token_embd.raw,
                    token_embd_info.element_count(),
                )
                .map_err(|e| format!("load weight 'token_embd.weight': {e}"))?;
                LmHead::Resident(Weight {
                    data,
                    shape: token_embd_info.shape.clone(),
                })
            }
        };

        let (tokenizer, cublas) = init
            .join()
            .map_err(|_| "background load-init thread panicked".to_string())??;

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
    ) -> Result<(Vec<u32>, String), String> {
        match imported {
            Some(crate::kv_io::ImportedKv::Dense(cache)) => {
                if self.hybrid.is_some() || self.mla.is_some() {
                    return Err("imported KV cache file is dense/MoE format, but this model is not a dense/MoE Qwen3 model".to_string());
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
                    .ok_or("imported KV cache file is hybrid format, but this model is not a Qwen3.5 hybrid model")?;
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
                let m = self.mla.as_ref().ok_or(
                    "imported KV cache file is MLA format, but this model is not an MLA model",
                )?;
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
    ) -> Result<CudaSlice<f32>, String> {
        let offset = (rows - 1) * hidden_size;
        let mut out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| format!("last_row alloc: {e}"))?;
        let src = hidden_batched.slice(offset..offset + hidden_size);
        self.device
            .dtod_copy(&src, &mut out)
            .map_err(|e| format!("last_row dtod: {e}"))?;
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
    ) -> Result<CudaSlice<f32>, String> {
        let offset = row * hidden_size;
        let mut out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| format!("extract_row alloc: {e}"))?;
        let src = batched.slice(offset..offset + hidden_size);
        self.device
            .dtod_copy(&src, &mut out)
            .map_err(|e| format!("extract_row dtod: {e}"))?;
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
    ) -> Result<(), String> {
        let offset = row * hidden_size;
        let mut dst = batched.slice_mut(offset..offset + hidden_size);
        self.device
            .dtod_copy(src, &mut dst)
            .map_err(|e| format!("write_row dtod: {e}"))
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
    ) -> Result<Vec<u32>, String> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        let full_ids = self.tokenizer.encode(&format!("{prompt}{candidate}"))?;
        if full_ids.len() <= prompt_ids.len() || full_ids[..prompt_ids.len()] != prompt_ids[..] {
            return Err(format!("system1: candidate {candidate:?} does not tokenize as a clean continuation of the prompt"));
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
    ) -> Result<System1Response, String> {
        if let Some(h) = &self.hybrid {
            return self.system1_evaluate_hybrid(h, prompt, candidates, temperature);
        }
        if let Some(m) = &self.mla {
            return self.system1_evaluate_mla(m, prompt, candidates, temperature);
        }
        self.system1_evaluate_dense(prompt, candidates, temperature)
    }

    /// DeepSeek-V2/V3 MLA path -- see [`Self::system1_evaluate`]'s doc
    /// comment for the overall approach. Supports multi-token candidates
    /// the same way dense does: `MlaPrefillResult`'s per-layer `kv_cache` is
    /// compressed latent-KV (`[seq_len, kv_lora_rank + qk_rope_head_dim]`)
    /// but still **position-indexed**, not an unaddressed recurrence like
    /// hybrid's `GatedDeltaNet` state -- so the same shared-cache,
    /// score-to-completion-before-the-next-candidate reuse dense relies on
    /// carries over unmodified.
    fn system1_evaluate_mla(
        &self,
        m: &MlaModel,
        prompt: &str,
        candidates: &[System1Candidate],
        temperature: f32,
    ) -> Result<System1Response, String> {
        if candidates.is_empty() {
            return Err("system1_evaluate: candidates must not be empty".to_string());
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

        let normed = self.rmsnorm(&hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let first_tokens: Vec<u32> = resolved.iter().map(|ids| ids[0]).collect();
        let mut scores = self.gemv_gather_lm_head(&normed, &first_tokens)?;

        for (i, ids) in resolved.iter().enumerate() {
            if ids.len() < 2 {
                continue;
            }
            for (position, w) in (base_position..).zip(ids.windows(2)) {
                let (prev, next) = (w[0], w[1]);
                let h = self.forward_one_token_mla(m, prev, position, &mut kv_caches)?;
                let normed_step = self.rmsnorm(&h, &self.output_norm.data, 1, hidden_size, eps)?;
                scores[i] += self.gemv_gather_lm_head(&normed_step, &[next])?[0];
            }
        }

        Self::finish_system1_response(candidates, resolved, scores, temperature)
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
    ) -> Result<System1Response, String> {
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
    ) -> Result<u32, String> {
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
    ) -> Result<Vec<f32>, String> {
        let normed = self.rmsnorm(hidden, &self.output_norm.data, 1, hidden_size, eps)?;
        let logits_dev = self.gemv(&normed, self.lm_head_resident()?)?;
        self.device
            .dtoh_sync_copy(&logits_dev)
            .map_err(|e| format!("logits dtoh: {e}"))
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
    fn lm_head_resident(&self) -> Result<&Weight, String> {
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
                .map_err(|e| format!("upload weight 'token_embd.weight' to device: {e}"))?;
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
    ) -> Result<Vec<f32>, String> {
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
            return Err(format!(
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
            .map_err(|e| format!("gemv_gather_lm_head upload compact rows: {e}"))?;
        let compact_w = Weight {
            data: dev_compact,
            shape: vec![hidden_size as u64, row_indices.len() as u64],
        };
        let trivial_indices: Vec<u32> = (0..row_indices.len() as u32).collect();
        self.gemv_gather(x, &compact_w, &trivial_indices)
    }

    /// `pub(crate)` (not private) so [`crate::sampling::sample`]'s greedy
    /// path can delegate straight here instead of duplicating this scan.
    pub(crate) fn argmax(logits: &[f32]) -> Result<u32, String> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "cannot argmax an empty logits slice".to_string())
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

        let normed = self.rmsnorm(
            &hidden,
            &w.attn_norm.data,
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
            .map_err(|e| format!("mla k_pe alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(kv_lora..kv_lora + qk_rope);
            self.device
                .dtod_copy(&src, &mut k_pe)
                .map_err(|e| format!("mla k_pe dtod: {e}"))?;
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
            .map_err(|e| format!("mla kv_cmpr alloc: {e}"))?;
        {
            let src = kv_cmpr_pe.slice(0..kv_lora);
            self.device
                .dtod_copy(&src, &mut kv_cmpr_owned)
                .map_err(|e| format!("mla kv_cmpr dtod: {e}"))?;
        }
        let kv_cmpr_normed = self.rmsnorm(
            &kv_cmpr_owned,
            &w.attn_kv_a_norm.data,
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
            .map_err(|e| format!("mla q_pe alloc: {e}"))?;
        for h in 0..n_head {
            let src =
                q.slice(h * n_embd_head_k_mla + qk_nope..h * n_embd_head_k_mla + n_embd_head_k_mla);
            let mut dst = q_pe.slice_mut(h * qk_rope..(h + 1) * qk_rope);
            self.device
                .dtod_copy(&src, &mut dst)
                .map_err(|e| format!("mla q_pe dtod head {h}: {e}"))?;
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
            .map_err(|e| format!("mla qcur alloc: {e}"))?;
        for h in 0..n_head {
            let q_nope_view = q.slice(h * n_embd_head_k_mla..h * n_embd_head_k_mla + qk_nope);
            let wk_b_view = w
                .wk_b
                .data
                .slice(h * qk_nope * kv_lora..(h + 1) * qk_nope * kv_lora);
            let absorbed = self.gemv_view(&q_nope_view, &wk_b_view, qk_nope, kv_lora)?;

            let mut dst_nope = qcur.slice_mut(h * qk_dim..h * qk_dim + kv_lora);
            self.device
                .dtod_copy(&absorbed, &mut dst_nope)
                .map_err(|e| format!("mla qcur absorbed dtod head {h}: {e}"))?;

            let pe_src = q_pe.slice(h * qk_rope..(h + 1) * qk_rope);
            let mut dst_pe = qcur.slice_mut(h * qk_dim + kv_lora..h * qk_dim + qk_dim);
            self.device
                .dtod_copy(&pe_src, &mut dst_pe)
                .map_err(|e| format!("mla qcur pe dtod head {h}: {e}"))?;
        }

        // Write this position's compressed Kcur (== kv_cmpr_normed ++ k_pe) into the
        // preallocated per-layer cache -- device-resident from the start (Phase 2
        // round 2 convention), no host round-trip, ever, for this cache.
        let offset = position * qk_dim;
        {
            let mut dst = kv_cache.slice_mut(offset..offset + kv_lora);
            self.device
                .dtod_copy(&kv_cmpr_normed, &mut dst)
                .map_err(|e| format!("mla kv_cache dtod cmpr: {e}"))?;
        }
        {
            let mut dst = kv_cache.slice_mut(offset + kv_lora..offset + qk_dim);
            self.device
                .dtod_copy(&k_pe, &mut dst)
                .map_err(|e| format!("mla kv_cache dtod k_pe: {e}"))?;
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

        let normed = self.rmsnorm(
            &hidden,
            &w.attn_norm.data,
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
            &w.attn_kv_a_norm.data,
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
    fn forward_mla_layer_batched(
        &self,
        m: &MlaModel,
        layer: &MlaLayerWeights,
        hidden: CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_cache: &mut CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
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
                let moe_cfg = cfg
                    .moe
                    .as_ref()
                    .ok_or("internal error: MlaFfn::Moe layer but MlaConfig::moe is None")?;
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
    fn forward_mla_moe_ffn(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
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
            return Err("internal error: forward_mla_moe_ffn called on a Dense layer".to_string());
        };

        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, 1, hidden_size, eps)?;

        let router_logits_dev = self.gemv(&ffn_normed, ffn_gate_inp)?;
        let router_logits = self
            .device
            .dtoh_sync_copy(&router_logits_dev)
            .map_err(|e| format!("mla moe router dtoh: {e}"))?;
        let routed = route_top_k_with_norm(
            &router_logits,
            moe_cfg.expert_used_count,
            moe_cfg.normalize_top_k,
        )?;

        let mut ffn_out_dev = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| format!("mla moe ffn_out alloc: {e}"))?;
        let dest_row0 = self
            .device
            .htod_sync_copy(&[0u32])
            .map_err(|e| format!("mla moe dest_row htod: {e}"))?;
        for (expert_idx, weight) in routed {
            let gate = self.gemv_expert(&ffn_normed, ffn_gate_exps, expert_idx)?;
            let up = self.gemv_expert(&ffn_normed, ffn_up_exps, expert_idx)?;
            let activated = self.silu_and_mul(&gate, &up, moe_cfg.n_ff_exp)?;
            let down = self.gemv_expert(&activated, ffn_down_exps, expert_idx)?;
            let weight_dev = self
                .device
                .htod_sync_copy(&[weight * moe_cfg.routed_scaling_factor])
                .map_err(|e| format!("mla moe weight htod: {e}"))?;
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
    fn forward_mla_moe_ffn_batched(
        &self,
        layer: &MlaLayerWeights,
        mut post_attn: CudaSlice<f32>,
        hidden_size: usize,
        rows: usize,
        moe_cfg: &MlaMoeConfig,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
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
            return Err(
                "internal error: forward_mla_moe_ffn_batched called on a Dense layer".to_string(),
            );
        };

        let ffn_normed = self.rmsnorm(&post_attn, &layer.ffn_norm.data, rows, hidden_size, eps)?;

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
            .map_err(|e| format!("mla moe router dtoh: {e}"))?;
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
    fn alloc_mla_kv_caches(
        &self,
        m: &MlaModel,
        imported: Option<&crate::kv_io::MlaKvCache>,
        start_pos: usize,
        rows: usize,
        extra_headroom: usize,
    ) -> Result<Vec<CudaSlice<f32>>, String> {
        crate::limits::check_positions(start_pos, rows, extra_headroom)?;
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
                return Err(format!(
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
                    .map_err(|e| format!("import mla kv_cache htod layer {layer_idx}: {e}"))?;
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
    ) -> Result<MlaPrefillResult, String> {
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
            return Err("encode produced no tokens".to_string());
        }

        let mut kv_caches =
            self.alloc_mla_kv_caches(m, imported, start_pos, ids.len(), extra_headroom)?;

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
    ) -> Result<MlaPrefillResult, String> {
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
            return Err("encode produced no tokens".to_string());
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
            .map_err(|e| format!("embedding htod: {e}"))?;

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
    fn generate_mla_impl(
        &self,
        m: &MlaModel,
        prompt: &str,
        imported: Option<&crate::kv_io::MlaKvCache>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        mut on_first_token: impl FnMut(&[f32]),
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<MlaGenerateResult, String> {
        if max_new_tokens == 0 {
            return Err("max_new_tokens must be at least 1".to_string());
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
    fn forward_one_token_mla(
        &self,
        m: &MlaModel,
        token_id: u32,
        position: usize,
        kv_caches: &mut [CudaSlice<f32>],
    ) -> Result<CudaSlice<f32>, String> {
        let cfg = &m.cfg;
        let hidden_size = cfg.hidden_size;
        let ffn_hidden_size = cfg.ffn_hidden_size;
        let eps = cfg.rmsnorm_eps;
        let mut hidden = self
            .device
            .htod_sync_copy(&self.token_embd.row(token_id)?)
            .map_err(|e| format!("embedding htod: {e}"))?;

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
                    let moe_cfg = cfg
                        .moe
                        .as_ref()
                        .ok_or("internal error: MlaFfn::Moe layer but MlaConfig::moe is None")?;
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
    ) -> Result<((u32, String), crate::kv_io::MlaKvCache), String> {
        let m = self
            .mla
            .as_ref()
            .ok_or("forward_prompt_capture_kv_mla called on a non-MLA model")?;
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
                    .map_err(|e| format!("mla kv_cache dtoh: {e}"))
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
