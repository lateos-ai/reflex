//! Model configuration parsed from GGUF metadata: per-architecture layer, MoE,
//! hybrid and MLA settings, and the RoPE convention.

use super::*;
use crate::error::ReflexError;

pub(super) fn u64_meta(file: &GgufFile, key: &str) -> Option<u64> {
    file.metadata.get(key).and_then(GgufValue::as_u64)
}

pub(super) fn f32_meta(file: &GgufFile, key: &str) -> Option<f32> {
    file.metadata.get(key).and_then(GgufValue::as_f32)
}

/// Which rotary-embedding convention a model's attention RoPE must use.
/// `Neox` is llama.cpp's `LLAMA_ROPE_TYPE_NEOX` -- half-split pairs `(i, i +
/// rotary_dim/2)` -- implemented by `rope_kernel`/`rope_batch_kernel`, used by
/// Qwen3/Qwen3.5. `Norm` is `LLAMA_ROPE_TYPE_NORM` -- consecutive pairs
/// `(2i, 2i+1)` -- implemented by `rope_norm_kernel`/`rope_norm_batch_kernel`,
/// used by Llama/Mistral (and DeepSeek-V2/V3 MLA, handled on `MlaModel`).
/// **Confirmed per-architecture against llama.cpp's `llama_model_rope_type`,
/// never assumed** -- the MLA work found this is a real, easy-to-miss
/// per-architecture difference that silently corrupts output when wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    Neox,
    Norm,
}

/// Maps a GGUF `general.architecture` string to its RoPE convention for the
/// dense/MoE path. `qwen3`/`qwen3moe` (and any other architecture reaching
/// [`parse_model_config`] today) use the half-split Neox convention this
/// project has always applied; `llama` (and the `mistral`/`mixtral` aliases)
/// use the consecutive-pair Norm convention (`llama_model_rope_type` maps the
/// Llama family, including Mistral, to `LLAMA_ROPE_TYPE_NORM`). Kept a pure
/// function of the architecture string so the mapping is auditable in one
/// place.
///
/// Note: real Mistral-7B GGUFs report `general.architecture = "llama"`, not
/// `"mistral"` -- llama.cpp has no bare `mistral` arch string (its newer
/// `mistral3`/`mistral4` are separate architectures outside this scope), so
/// the `mistral`/`mixtral` arms are defensive aliases; `llama` is the arm that
/// actually carries Mistral-7B.
///
/// `kolibri1` is listed explicitly even though it would fall through to Neox
/// anyway: the community patch the published GGUFs are made with adds it to
/// llama.cpp's NEOX list, and vLLM's `get_rope` (Aleph Alpha's official
/// plugin) defaults to NEOX. Its full-attention layers use no RoPE at all (see
/// [`KolibriConfig::sliding_layers`]). `qwen2` is listed for the same reason:
/// `llama_model_rope_type` at `79e2e74` has `LLM_ARCH_QWEN2` in its NEOX
/// group (docs/design/qwen2.md). `mistral3` (Ministral 3, and the
/// Nemotron-3.5-Embed encoders built on it) is in its NORM group.
pub(super) fn rope_type_for(architecture: &str) -> RopeType {
    match architecture {
        "llama" | "mistral" | "mixtral" | "mistral3" => RopeType::Norm,
        "kolibri1" | "qwen2" => RopeType::Neox,
        _ => RopeType::Neox,
    }
}

/// Tensors llama.cpp applies when present but the dense/MoE path doesn't
/// implement, checked on layer 0 and the LM head by [`parse_model_config`].
/// Without the check a file carrying one would load and run without it:
/// wrong output, no error. `attn_qkv` is the fused projection
/// `create_tensor_qkv` prefers over separate `attn_q`/`attn_k`/`attn_v`.
/// No surveyed Qwen2.5, Qwen3 or Llama GGUF has any of them
/// (docs/design/qwen2.md).
const UNSUPPORTED_DENSE_TENSORS: [&str; 7] = [
    "blk.0.attn_output.bias",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_qkv.bias",
    "output.bias",
    // `mistral3`'s optional MLP biases (and Llama's, in llama.cpp).
    "blk.0.ffn_gate.bias",
    "blk.0.ffn_up.bias",
    "blk.0.ffn_down.bias",
];

/// Shared-expert tensors (`qwen2moe`, `qwen35moe`, DeepSeek). The generic MoE
/// path has no shared expert, so a file with one is rejected rather than run
/// with it silently dropped.
const SHARED_EXPERT_TENSORS: [&str; 4] = [
    "blk.0.ffn_gate_inp_shexp.weight",
    "blk.0.ffn_gate_shexp.weight",
    "blk.0.ffn_up_shexp.weight",
    "blk.0.ffn_down_shexp.weight",
];

/// The `blk.{layer}.attn_{q,k,v}.bias` names to load for one dense/MoE layer
/// (Qwen2's Q/K/V biases, `create_tensor_qkv` in llama.cpp), or `None` when
/// the layer has none. All three or none: a partial set is rejected, naming
/// the missing tensor, and each length must equal its projection's output
/// width. Host-only (metadata and tensor table), so loading never starts on a
/// file that would fail here.
pub(super) fn qkv_bias_names(
    file: &GgufFile,
    layer: usize,
    cfg: &LayerConfig,
) -> Result<Option<[String; 3]>, ReflexError> {
    let names = ["attn_q", "attn_k", "attn_v"].map(|t| format!("blk.{layer}.{t}.bias"));
    let widths = [
        cfg.num_q_heads * cfg.head_dim,
        cfg.num_kv_heads * cfg.head_dim,
        cfg.num_kv_heads * cfg.head_dim,
    ];
    let present = names
        .iter()
        .filter(|n| file.tensor_info(n).is_some())
        .count();
    if present == 0 {
        return Ok(None);
    }
    for (name, width) in names.iter().zip(widths) {
        let info = file.tensor_info(name).ok_or_else(|| {
            crate::reflex_err!(
                Gguf,
                "layer {layer} has {present} of the 3 Q/K/V biases but no '{name}'; \
                 a partial bias set is not supported"
            )
        })?;
        if info.shape.as_slice() != [width as u64] {
            return Err(crate::reflex_err!(
                Gguf,
                "'{name}' has shape {:?}, expected [{width}]",
                info.shape
            ));
        }
    }
    Ok(Some(names))
}

/// Static shape/hyperparameter config for one dense Qwen3 transformer layer,
/// read from the GGUF file's `qwen3.*` metadata.
#[derive(Clone)]
pub struct LayerConfig {
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Read explicitly from `qwen3.attention.key_length` -- Qwen3 decouples
    /// this from `hidden_size / num_q_heads` (real finding from
    /// RustFeference's own Phase 21.14: a real Qwen3-0.6B has head_dim=128,
    /// not the 64 that division would give). Files without the key (`qwen2`,
    /// Llama) use `hidden_size / num_q_heads`, as llama.cpp does.
    pub head_dim: usize,
    /// Number of leading dims of each head RoPE actually rotates (GPT-NeoX
    /// half-rotation convention). Equal to `head_dim` (full rotary) for
    /// dense/MoE Qwen3; Qwen3.5's Gated Attention layers use a real partial
    /// value from `qwen35.rope.dimension_count` (see `Model::load_hybrid`).
    pub rotary_dim: usize,
    pub ffn_hidden_size: usize,
    pub rope_base: f32,
    pub rmsnorm_eps: f32,
    /// Which RoPE convention this model's attention uses (see [`RopeType`]).
    /// `Neox` for Qwen3/Qwen3.5, `Norm` for Llama/Mistral.
    pub rope_type: RopeType,
}

/// `<arch>.expert_count`/`<arch>.expert_used_count` metadata for an MoE
/// model -- present (and `expert_count` nonzero) iff the file describes an
/// MoE architecture. See `crate::moe`'s doc comment for why this is keyed by
/// the file's own architecture string rather than hardcoded to `llama.*`.
pub struct MoeMetaConfig {
    pub expert_count: usize,
    pub expert_used_count: usize,
}

/// Derive [`LayerConfig`], the layer count, and (for an MoE architecture)
/// [`MoeMetaConfig`] from a GGUF file's `<arch>.*` metadata (plus
/// `blk.0.ffn_gate[_exps].weight`'s real shape, since GGUF has no dedicated
/// "FFN hidden size" metadata key). Pure/host-only, no GPU required.
/// Supports dense `qwen3`/`qwen2`/`llama` and any architecture reporting a
/// nonzero `<arch>.expert_count` (real `qwen3moe`, or a Mixtral-style test
/// fixture under `llama.*`) -- other architectures are out of scope (see
/// README.md's MVP order). Also rejects files carrying tensors this path
/// doesn't implement (`UNSUPPORTED_DENSE_TENSORS`, and shared experts on an
/// MoE file); Qwen2's Q/K/V biases are checked per layer by [`qkv_bias_names`].
pub fn parse_model_config(
    file: &GgufFile,
) -> Result<(LayerConfig, usize, Option<MoeMetaConfig>), ReflexError> {
    let architecture = file
        .metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("");

    // `kolibri1` reports a nonzero `expert_count`, so without this check it
    // would be accepted as a generic MoE model and run with the wrong router,
    // no sandwich norms and RoPE on every layer -- wrong output, no crash.
    if architecture == "kolibri1" {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "'kolibri1' has its own config (parse_kolibri_config), not the generic dense/MoE one"
        ));
    }

    // SSM architectures llama.cpp supports and this engine doesn't. Some
    // report a nonzero `expert_count` (Jamba, Nemotron-H MoE), so without
    // this check they would reach the generic MoE path and fail on a missing
    // tensor instead of saying why. `granitehybrid` has its own loader.
    if let Some(why) = match architecture {
        "mamba" | "jamba" => Some("Mamba-1 SSM layers"),
        "mamba2" | "falcon-h1" | "nemotron_h" | "nemotron_h_moe" => {
            Some("a Mamba-2 layout other than Granite 4.0-H's")
        }
        _ => None,
    } {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "unsupported architecture '{architecture}' ({why}): the only SSM hybrid supported is \
             'granitehybrid' (Granite 4.0-H); see docs/design/mamba2-hybrid.md"
        ));
    }

    let moe = match u64_meta(file, &format!("{architecture}.expert_count")).filter(|&n| n > 0) {
        Some(expert_count) => {
            let expert_used_count = u64_meta(file, &format!("{architecture}.expert_used_count"))
                .ok_or_else(|| {
                    crate::reflex_err!(
                        Gguf,
                        "missing {architecture}.expert_used_count metadata key"
                    )
                })?;
            Some(MoeMetaConfig {
                expert_count: expert_count as usize,
                expert_used_count: expert_used_count as usize,
            })
        }
        None => None,
    };

    if !matches!(
        architecture,
        "qwen3" | "qwen2" | "llama" | "mistral" | "mixtral" | "mistral3"
    ) && moe.is_none()
    {
        return Err(crate::reflex_err!(UnsupportedArchitecture,
            "unsupported architecture '{architecture}': only dense 'qwen3'/'qwen2'/'llama'/'mistral', 'mistral3' embedding models and MoE architectures reporting a nonzero '{architecture}.expert_count' are in scope for this MVP"
        ));
    }
    // Ministral 3 chat models need YaRN and the query temperature scale on the
    // decode path too; only the encoder (embedding) form is implemented.
    if architecture == "mistral3" && !is_encoder(file, architecture) {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "'mistral3' is supported only as an embedding model ({architecture}.attention.causal = \
             false, e.g. Nemotron-3.5-Embed); Ministral 3 chat models are not supported yet"
        ));
    }

    if let Some(name) = UNSUPPORTED_DENSE_TENSORS
        .iter()
        .find(|n| file.tensor_info(n).is_some())
    {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "'{architecture}' file has tensor '{name}', which this engine doesn't implement \
             (running without it would give wrong output); see docs/design/qwen2.md"
        ));
    }
    if moe.is_some() {
        if let Some(name) = SHARED_EXPERT_TENSORS
            .iter()
            .find(|n| file.tensor_info(n).is_some())
        {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "'{architecture}' MoE file has a shared expert ('{name}'), which the generic \
                 MoE path doesn't implement (e.g. qwen2moe); only routed experts are supported"
            ));
        }
    }

    // RoPE scaling types other than "none" (e.g. Llama-3.1's "linear"/"yarn"
    // long-context variants) change the per-dimension rotation frequencies;
    // this path implements only the unscaled convention, so reject rather than
    // silently run wrong frequencies (same rejection posture as MLA's Q-LoRA/
    // MTP handling). `deepseek2`'s "yarn" is handled separately on
    // the MLA path, never here, and an encoder's by `parse_encoder_config`.
    if let Some(scaling) = file
        .metadata
        .get(&format!("{architecture}.rope.scaling.type"))
        .and_then(GgufValue::as_str)
    {
        if scaling != "none" && !(scaling == "yarn" && is_encoder(file, architecture)) {
            return Err(crate::reflex_err!(UnsupportedArchitecture,
                "{architecture}.rope.scaling.type = {scaling:?} is not supported by this MVP (only unscaled RoPE, i.e. no scaling or \"none\", is implemented)"
            ));
        }
    }

    let block_count = u64_meta(file, &format!("{architecture}.block_count")).ok_or_else(|| {
        crate::reflex_err!(Gguf, "missing {architecture}.block_count metadata key")
    })? as usize;
    let hidden_size =
        u64_meta(file, &format!("{architecture}.embedding_length")).ok_or_else(|| {
            crate::reflex_err!(Gguf, "missing {architecture}.embedding_length metadata key")
        })? as usize;
    let num_q_heads =
        u64_meta(file, &format!("{architecture}.attention.head_count")).ok_or_else(|| {
            crate::reflex_err!(
                Gguf,
                "missing {architecture}.attention.head_count metadata key"
            )
        })? as usize;
    let num_kv_heads = u64_meta(file, &format!("{architecture}.attention.head_count_kv"))
        .unwrap_or(num_q_heads as u64) as usize;
    // Qwen3 (dense and MoE) decouples head_dim from hidden_size/num_q_heads via
    // this key. Non-Qwen3 MoE fixtures (e.g. a Mixtral-style test GGUF with no
    // per-head decoupling) don't set it, so fall back to the standard derivation
    // rather than hard-failing -- real Qwen3 files always have the key, so this
    // fallback only ever engages for such fixtures.
    let head_dim = u64_meta(file, &format!("{architecture}.attention.key_length"))
        .map(|n| n as usize)
        .unwrap_or(hidden_size / num_q_heads);
    // Number of leading head dims RoPE rotates. Full (`head_dim`) for
    // Qwen3/Llama/Mistral; read explicitly so a partial-rotation file isn't
    // silently run with the wrong count. Falls back to full rotation when the
    // key is absent (all of the above, plus the Mixtral-style fixture).
    let rotary_dim = u64_meta(file, &format!("{architecture}.rope.dimension_count"))
        .map(|n| n as usize)
        .unwrap_or(head_dim);

    let ffn_gate_weight_name = if moe.is_some() {
        "blk.0.ffn_gate_exps.weight"
    } else {
        "blk.0.ffn_gate.weight"
    };
    let ffn_gate_info = file
        .tensor_info(ffn_gate_weight_name)
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {ffn_gate_weight_name} tensor"))?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        [_in_features, out_features, _expert_count] => *out_features as usize,
        other => {
            return Err(crate::reflex_err!(
                Gguf,
                "{ffn_gate_weight_name} has unexpected shape {other:?}"
            ))
        }
    };

    let rope_base = f32_meta(file, &format!("{architecture}.rope.freq_base")).unwrap_or(10000.0);
    let rmsnorm_eps = f32_meta(
        file,
        &format!("{architecture}.attention.layer_norm_rms_epsilon"),
    )
    .unwrap_or(1e-5);

    Ok((
        LayerConfig {
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            ffn_hidden_size,
            rope_base,
            rmsnorm_eps,
            rope_type: rope_type_for(architecture),
        },
        block_count,
        moe,
    ))
}

/// Whether `file` describes an encoder: bidirectional attention
/// (`<arch>.attention.causal = false`, llama.cpp's `LLM_KV_ATTENTION_CAUSAL`,
/// default true). Such a model produces embeddings (`Model::embed`), never
/// tokens. A causal model with a pooling type (Qwen3-Embedding's last-token
/// pooling) is not an encoder here and loads as an ordinary language model.
pub(super) fn is_encoder(file: &GgufFile, architecture: &str) -> bool {
    matches!(
        file.metadata
            .get(&format!("{architecture}.attention.causal")),
        Some(GgufValue::Bool(false))
    )
}

/// YaRN RoPE scaling for a dense encoder's consecutive-pair rotation
/// (`rope_norm_yarn*_kernel`, the kernels MLA uses), as the kernel's launch
/// arguments. See [`parse_encoder_config`] for how they are derived.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RopeYarn {
    pub(super) freq_scale: f32,
    pub(super) ext_factor: f32,
    /// The kernel multiplies this by `1 + 0.1 * ln(factor)` to get the
    /// cos/sin magnitude, as ggml's `rope_yarn` does.
    pub(super) attn_factor: f32,
    pub(super) corr_dim_start: f32,
    pub(super) corr_dim_end: f32,
}

/// Llama-4-style query temperature scaling: Q at absolute position `pos` is
/// multiplied by `1 + beta * ln(floor(pos / floor_scale) + 1)` after RoPE
/// (llama.cpp's `build_inp_attn_scale`; HF's `_get_llama_4_attn_scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AttnTemperature {
    pub(super) beta: f32,
    pub(super) floor_scale: f32,
}

/// What an encoder reduces its final hidden states to. Only mean pooling
/// (llama.cpp's `LLAMA_POOLING_TYPE_MEAN`, 1) is implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pooling {
    Mean,
}

/// An encoder's settings beyond [`LayerConfig`], from [`parse_encoder_config`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EncoderConfig {
    pub(super) yarn: Option<RopeYarn>,
    pub(super) temperature: Option<AttnTemperature>,
    pub(super) pooling: Pooling,
}

/// Reads an encoder's ([`is_encoder`]) pooling, YaRN and query temperature
/// settings; `None` for a causal model. Host-only.
///
/// YaRN follows llama.cpp's `llama-context.cpp` setup and ggml's
/// `ggml_rope_yarn_corr_dims` over the rotated width, which match HF
/// `transformers`' `_compute_yarn_parameters` (floor/ceil-truncated
/// correction range, same linear ramp). The cos/sin magnitude: with
/// `rope.scaling.yarn_log_multiplier` (HF's `mscale_all_dim`; `mscale` taken
/// as 1, as llama.cpp does outside DeepSeek-V2) it is
/// `get_mscale(f, 1) / get_mscale(f, log_mul)`, 1.0 for Nemotron-3.5-Embed
/// whose HF config has `mscale = mscale_all_dim = 1`; without the key it is
/// `get_mscale(f, 1)`. Times `rope.scaling.attn_factor` (default 1).
/// `attention.temperature_scale` turns on the query temperature, with
/// `original_context_length` as the floor scale (llama.cpp's
/// `n_attn_temp_floor_scale = n_ctx_orig_yarn` for `mistral3`).
pub(super) fn parse_encoder_config(
    file: &GgufFile,
    architecture: &str,
    cfg: &LayerConfig,
) -> Result<Option<EncoderConfig>, ReflexError> {
    if !is_encoder(file, architecture) {
        return Ok(None);
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let pooling = match u64_meta(file, &key("pooling_type")) {
        Some(1) => Pooling::Mean,
        other => {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {other:?}: only mean pooling (1) is supported for embedding models",
                key("pooling_type")
            ))
        }
    };
    if cfg.rope_type != RopeType::Norm {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "embedding models are supported only with the consecutive-pair RoPE convention \
             ('{architecture}' uses {:?})",
            cfg.rope_type
        ));
    }
    let orig_ctx_len = u64_meta(file, &key("rope.scaling.original_context_length"));
    let scaling = file
        .metadata
        .get(&key("rope.scaling.type"))
        .and_then(GgufValue::as_str);
    let yarn = match scaling {
        None | Some("none") => None,
        Some("yarn") => {
            let factor = f32_meta(file, &key("rope.scaling.factor")).ok_or_else(|| {
                crate::reflex_err!(Gguf, "missing {}", key("rope.scaling.factor"))
            })?;
            let orig = orig_ctx_len.ok_or_else(|| {
                crate::reflex_err!(
                    Gguf,
                    "missing {}",
                    key("rope.scaling.original_context_length")
                )
            })? as f32;
            if factor.is_nan() || factor <= 1.0 {
                return Err(crate::reflex_err!(
                    Gguf,
                    "{} = {factor}: YaRN needs a factor above 1",
                    key("rope.scaling.factor")
                ));
            }
            let beta_fast = f32_meta(file, &key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0);
            let beta_slow = f32_meta(file, &key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0);
            let log_mul = f32_meta(file, &key("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0);
            let rope_attn_factor = f32_meta(file, &key("rope.scaling.attn_factor")).unwrap_or(1.0);
            let get_mscale = |scale: f32, mscale: f32| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * scale.ln() + 1.0
                }
            };
            let magnitude = if log_mul != 0.0 {
                get_mscale(factor, 1.0) / get_mscale(factor, log_mul)
            } else {
                get_mscale(factor, 1.0)
            } * rope_attn_factor;
            let n_dims = cfg.rotary_dim as f32;
            let corr_dim = |n_rot: f32| {
                n_dims * (orig / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                    / (2.0 * cfg.rope_base.ln())
            };
            Some(RopeYarn {
                freq_scale: 1.0 / factor,
                ext_factor: 1.0,
                attn_factor: magnitude / (1.0 + 0.1 * factor.ln()),
                corr_dim_start: corr_dim(beta_fast).floor().max(0.0),
                corr_dim_end: corr_dim(beta_slow).ceil().min(n_dims - 1.0),
            })
        }
        Some(other) => {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {other:?}: embedding models support only \"none\" or \"yarn\"",
                key("rope.scaling.type")
            ))
        }
    };
    let temperature =
        match f32_meta(file, &key("attention.temperature_scale")).filter(|&b| b != 0.0) {
            None => None,
            Some(beta) => {
                let floor_scale = orig_ctx_len.filter(|&n| n > 0).ok_or_else(|| {
                    crate::reflex_err!(
                        Gguf,
                        "{} is set but {} is missing",
                        key("attention.temperature_scale"),
                        key("rope.scaling.original_context_length")
                    )
                })? as f32;
                Some(AttnTemperature { beta, floor_scale })
            }
        };
    Ok(Some(EncoderConfig {
        yarn,
        temperature,
        pooling,
    }))
}

/// Which trunk blocks of a Qwen3.5 hybrid model are Gated DeltaNet layers
/// (`true`) vs. Gated Attention layers (`false`), read from metadata --
/// never hardcoded (a wrong pattern would run the wrong mixer on real
/// weights). Ported from RustFeference's `hybrid.rs` `HybridConfig::parse`:
/// an explicit `{arch}.attention.recurrent_layers` boolean array wins when
/// present, otherwise every `full_attention_interval`-th block (the last of
/// each group) is Gated Attention and the rest are Gated DeltaNet. The real
/// `Qwen3.5-0.8B` fixture uses the interval fallback (`full_attention_interval
/// = 4`), giving Gated Attention at trunk indices `[3, 7, 11, 15, 19, 23]`.
pub(super) fn parse_hybrid_layer_kinds(
    file: &GgufFile,
    architecture: &str,
    block_count: usize,
) -> Result<Vec<bool>, ReflexError> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let recurrent_key = key("attention.recurrent_layers");
    match file.metadata.get(&recurrent_key) {
        Some(GgufValue::Array(items)) => {
            if items.len() != block_count {
                return Err(crate::reflex_err!(
                    Gguf,
                    "{recurrent_key} has {} entries but block_count is {block_count}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|v| match v {
                    GgufValue::Bool(b) => Ok(*b),
                    other => other.as_u64().map(|x| x != 0).ok_or_else(|| {
                        crate::reflex_err!(Gguf, "{recurrent_key} has non-boolean entry {other:?}")
                    }),
                })
                .collect()
        }
        Some(other) => Err(crate::reflex_err!(
            Gguf,
            "{recurrent_key} must be an array, got {other:?}"
        )),
        None => {
            let interval_key = key("full_attention_interval");
            let interval = u64_meta(file, &interval_key).ok_or_else(|| {
                crate::reflex_err!(
                    Gguf,
                    "{architecture} model has neither {recurrent_key} nor {interval_key}"
                )
            })? as usize;
            if interval == 0 {
                return Err(crate::reflex_err!(Gguf, "{interval_key} must be > 0"));
            }
            Ok((0..block_count).map(|i| (i + 1) % interval != 0).collect())
        }
    }
}

/// Reads a `qwen35moe` file's [`HybridMoeConfig`] from `{architecture}.*`
/// metadata, mirroring llama.cpp's `llama_model_qwen35moe::load_arch_hparams`/
/// `load_block_trunk`: `expert_feed_forward_length` falls back to
/// `blk.0.ffn_gate_exps.weight`'s shape when absent. Rejects a fused
/// `ffn_gate_up_exps` tensor -- llama.cpp can load one, but its own
/// `convert_hf_to_gguf.py` always splits HF's fused `gate_up_proj` into
/// separate `ffn_gate_exps`/`ffn_up_exps`, and this path only implements the
/// split form.
pub(super) fn parse_hybrid_moe_config(
    file: &GgufFile,
    architecture: &str,
) -> Result<HybridMoeConfig, ReflexError> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let expert_count = u64_meta(file, &key("expert_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("expert_count")))?
        as usize;
    let expert_used_count = u64_meta(file, &key("expert_used_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("expert_used_count")))?
        as usize;
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(crate::reflex_err!(
            Gguf,
            "{} ({expert_used_count}) must be in 1..={} ({expert_count})",
            key("expert_used_count"),
            key("expert_count")
        ));
    }
    if file.tensor_info("blk.0.ffn_gate_up_exps.weight").is_some() {
        return Err(crate::reflex_err!(UnsupportedArchitecture,
            "{architecture} file has a fused ffn_gate_up_exps tensor, which is not supported \
             (reconvert with llama.cpp's convert_hf_to_gguf.py, which emits split ffn_gate_exps/ffn_up_exps)"
        ));
    }
    let n_ff_exp = match u64_meta(file, &key("expert_feed_forward_length")) {
        Some(n) => n as usize,
        None => file
            .tensor_info("blk.0.ffn_gate_exps.weight")
            .and_then(|info| info.shape.get(1).copied())
            .ok_or_else(|| {
                crate::reflex_err!(
                    Gguf,
                    "missing {} and blk.0.ffn_gate_exps.weight to derive it from",
                    key("expert_feed_forward_length")
                )
            })? as usize,
    };
    let weights_scale = f32_meta(file, &key("expert_weights_scale"))
        .filter(|&s| s != 0.0)
        .unwrap_or(1.0);
    Ok(HybridMoeConfig {
        expert_used_count,
        n_ff_exp,
        weights_scale,
    })
}

/// Derives [`MlaConfig`] and the layer count from a GGUF file's `deepseek2.*`
/// metadata, following llama.cpp `79e2e74`'s `llama_model_deepseek2::load_arch_hparams`/
/// `load_arch_tensors` and `build_moe_ffn`. Supported: dense-lead + MoE-with-shared-expert
/// FFN, direct `wq` or Q-LoRA queries, softmax or sigmoid (DeepSeek-V3) gating with an
/// optional `exp_probs_b` selection bias, `expert_weights_scale`/`expert_weights_norm`,
/// and YaRN RoPE scaling. Hard-errors on: MTP/NextN blocks, all-MoE files,
/// group-limited routing (`expert_group_count > 1`), other gating functions, attention
/// temperature scaling, legacy unsplit `attn_kv_b` files, and any RoPE scaling type
/// other than `"none"`/`"yarn"`. Survey and sources: `docs/design/mla-widening.md`.
pub(super) fn parse_mla_config(file: &GgufFile) -> Result<(MlaConfig, usize, usize), ReflexError> {
    let architecture = "deepseek2";
    let key = |suffix: &str| format!("{architecture}.{suffix}");

    let block_count = u64_meta(file, &key("block_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("block_count")))?
        as usize;

    let nextn = u64_meta(file, &key("nextn_predict_layers")).unwrap_or(0);
    if nextn != 0 {
        return Err(crate::reflex_err!(
            Gguf,
            "{} MTP/NextN blocks (nextn_predict_layers={nextn}) are not supported by this MVP",
            key("nextn_predict_layers")
        ));
    }

    let leading_dense = u64_meta(file, &key("leading_dense_block_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("leading_dense_block_count")))?
        as usize;

    if f32_meta(file, &key("attention.temperature_scale")).is_some_and(|t| t != 0.0) {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "{} (attention temperature tuning) is not supported",
            key("attention.temperature_scale")
        ));
    }

    // llama.cpp's vocab size for its per-model heuristics: `vocab_size`, else the
    // token list's length.
    let n_vocab = u64_meta(file, &key("vocab_size"))
        .map(|n| n as usize)
        .or_else(|| match file.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(items)) => Some(items.len()),
            _ => None,
        });

    let q_lora_rank = parse_mla_q_lora_rank(file, block_count, n_vocab)?;

    let hidden_size = u64_meta(file, &key("embedding_length"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("embedding_length")))?
        as usize;
    let num_heads = u64_meta(file, &key("attention.head_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("attention.head_count")))?
        as usize;
    let kv_lora_rank = u64_meta(file, &key("attention.kv_lora_rank"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("attention.kv_lora_rank")))?
        as usize;
    let n_embd_head_k_mla = u64_meta(file, &key("attention.key_length_mla"))
        .ok_or_else(|| crate::reflex_err!(UnsupportedArchitecture, "missing {} (a legacy pre-MLA-split deepseek2 GGUF -- unsplit attn_kv_b, no key_length_mla/value_length_mla metadata -- is not supported by this MVP; reconvert from the original checkpoint with a current convert_hf_to_gguf.py)", key("attention.key_length_mla")))? as usize;
    let v_head_dim = u64_meta(file, &key("attention.value_length_mla"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("attention.value_length_mla")))?
        as usize;
    let qk_rope_head_dim = u64_meta(file, &key("rope.dimension_count"))
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing {}", key("rope.dimension_count")))?
        as usize;
    if n_embd_head_k_mla <= qk_rope_head_dim {
        return Err(crate::reflex_err!(
            Gguf,
            "{} ({n_embd_head_k_mla}) must be greater than {} ({qk_rope_head_dim})",
            key("attention.key_length_mla"),
            key("rope.dimension_count")
        ));
    }
    let qk_nope_head_dim = n_embd_head_k_mla - qk_rope_head_dim;
    let rope_base = f32_meta(file, &key("rope.freq_base")).unwrap_or(10000.0);

    // At least one dense-lead layer is required by this MVP step (true of every
    // real DeepSeek-V2/V3 file seen, and of the synthetic all-dense test fixture) --
    // an all-MoE deepseek2 file (leading_dense == 0) is not supported.
    let ffn_gate_info = file
        .tensor_info("blk.0.ffn_gate.weight")
        .ok_or_else(|| ReflexError::UnsupportedArchitecture("missing blk.0.ffn_gate.weight tensor (an all-MoE deepseek2 file, leading_dense_block_count == 0, is not supported by this MVP)".to_string()))?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        other => {
            return Err(crate::reflex_err!(
                Gguf,
                "blk.0.ffn_gate.weight has unexpected shape {other:?}"
            ))
        }
    };

    // Sanity-check `attention.value_length_mla` against `wv_b`'s own shape (the
    // tensor `Model::gemv_per_head` actually derives its decompressed output width
    // from) -- catches a malformed/mismatched real file early rather than silently
    // producing a wrong-sized attention output deep in the forward pass.
    let wv_b_info = file
        .tensor_info("blk.0.attn_v_b.weight")
        .ok_or_else(|| ReflexError::Gguf("missing blk.0.attn_v_b.weight tensor".to_string()))?;
    match wv_b_info.shape.as_slice() {
        [_in_features, out_features, _n_head] if *out_features as usize == v_head_dim => {}
        other => {
            return Err(crate::reflex_err!(
                Gguf,
                "blk.0.attn_v_b.weight shape {other:?} doesn't match {} ({v_head_dim})",
                key("attention.value_length_mla")
            ))
        }
    }

    let moe = if leading_dense < block_count {
        let expert_used_count = u64_meta(file, &key("expert_used_count")).ok_or_else(|| {
            crate::reflex_err!(
                Gguf,
                "missing {} (expert_count > 0 implied by leading_dense_block_count < block_count)",
                key("expert_used_count")
            )
        })? as usize;
        let ffn_gate_exps_info = file
            .tensor_info(&format!("blk.{leading_dense}.ffn_gate_exps.weight"))
            .ok_or_else(|| {
                crate::reflex_err!(
                    Gguf,
                    "missing blk.{leading_dense}.ffn_gate_exps.weight tensor"
                )
            })?;
        let n_ff_exp = match ffn_gate_exps_info.shape.as_slice() {
            [_in_features, out_features, _expert_count] => *out_features as usize,
            other => {
                return Err(crate::reflex_err!(
                    Gguf,
                    "blk.{leading_dense}.ffn_gate_exps.weight has unexpected shape {other:?}"
                ))
            }
        };
        // `build_moe_ffn` skips the scaling when the factor is 0 or 1, so both
        // are stored as the no-op 1.0.
        let routed_scaling_factor = f32_meta(file, &key("expert_weights_scale"))
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        // The converter only ever writes this key when the source model's
        // `norm_topk_prob` is truthy (see `MlaMoeConfig`'s doc comment) -- absence
        // means "don't renormalize", not "assume the usual true default".
        let normalize_top_k = matches!(
            file.metadata.get(&key("expert_weights_norm")),
            Some(GgufValue::Bool(true))
        );
        let gating = parse_mla_gating(file, block_count, n_vocab)?;

        let groups = u64_meta(file, &key("expert_group_count")).unwrap_or(0);
        if groups > 1 {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {groups} (group-limited expert routing) is not supported",
                key("expert_group_count")
            ));
        }

        let has_exp_probs_b = file
            .tensor_info(&format!("blk.{leading_dense}.exp_probs_b.bias"))
            .is_some();
        if has_exp_probs_b && gating == MlaGating::Softmax {
            // llama.cpp would add it to the softmax probabilities; no file does that.
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "deepseek2 file has exp_probs_b with softmax gating, which is not supported"
            ));
        }
        Some(MlaMoeConfig {
            expert_used_count,
            n_ff_exp,
            routed_scaling_factor,
            normalize_top_k,
            gating,
            has_exp_probs_b,
        })
    } else {
        None
    };

    let rope_scaling_type = file
        .metadata
        .get(&key("rope.scaling.type"))
        .and_then(GgufValue::as_str);
    let yarn = match rope_scaling_type {
        None | Some("none") => None,
        Some("yarn") => {
            let factor = f32_meta(file, &key("rope.scaling.factor")).ok_or_else(|| {
                crate::reflex_err!(Gguf, "missing {}", key("rope.scaling.factor"))
            })?;
            let orig_ctx_len = u64_meta(file, &key("rope.scaling.original_context_length"))
                .ok_or_else(|| {
                    crate::reflex_err!(
                        Gguf,
                        "missing {}",
                        key("rope.scaling.original_context_length")
                    )
                })? as f32;
            // llama.cpp's own CLI-settable defaults (32.0/1.0), used when the GGUF
            // doesn't override them -- real DeepSeek-V2-Lite doesn't set these keys
            // either, relying on the same defaults.
            let beta_fast = f32_meta(file, &key("rope.scaling.yarn_beta_fast")).unwrap_or(32.0);
            let beta_slow = f32_meta(file, &key("rope.scaling.yarn_beta_slow")).unwrap_or(1.0);
            // Stored pre-multiplied by 0.1 by the converter; the loader undoes that
            // ([TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] in a real llama.cpp build's
            // `deepseek2.cpp` `load_arch_hparams`) before using it -- replicate that
            // exactly, since every downstream formula assumes the undone value.
            let yarn_log_mul_raw =
                f32_meta(file, &key("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0) / 0.1;

            let freq_scale = 1.0 / factor;
            let ext_factor = 1.0f32;
            let factor_ln = factor.ln();
            // `cparams.yarn_attn_factor` after `llama-context.cpp`'s DEEPSEEK2
            // special case (the ratio `get_mscale(factor,mscale)/get_mscale(factor,mscale_all_dims)`
            // cancels to 1.0 whenever `mscale == mscale_all_dims`, which that special
            // case forces) followed by its own `*= 1/(1+0.1*ln(factor))` cancellation.
            let attn_factor = 1.0 / (1.0 + 0.1 * factor_ln);
            // deepseek2.cpp's own "cancel the adjustment to get the original
            // attn_factor" step -- reconstructs ~1.0 by construction, but computed
            // explicitly (not hardcoded) to mirror the reference exactly.
            let attn_factor_org = attn_factor * (1.0 + 0.1 * factor_ln);
            let mscale_kq = attn_factor_org * (1.0 + 0.1 * yarn_log_mul_raw * factor_ln);
            let attention_scale = mscale_kq * mscale_kq / (n_embd_head_k_mla as f32).sqrt();

            // ggml_rope_yarn_corr_dims: start/end correction dims over the rotated
            // width (qk_rope_head_dim), from beta_fast/beta_slow.
            let corr_dim = |n_rot: f32| {
                qk_rope_head_dim as f32 * (orig_ctx_len / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                    / (2.0 * rope_base.ln())
            };
            let corr_dim_start = corr_dim(beta_fast).floor().max(0.0);
            let corr_dim_end = corr_dim(beta_slow)
                .ceil()
                .min(qk_rope_head_dim as f32 - 1.0);

            Some(MlaYarnConfig {
                freq_scale,
                ext_factor,
                attn_factor,
                corr_dim_start,
                corr_dim_end,
                attention_scale,
            })
        }
        Some(other) => {
            return Err(crate::reflex_err!(
                Gguf,
                "{} = {other:?} (only \"none\"/\"yarn\" RoPE scaling is supported by this MVP)",
                key("rope.scaling.type")
            ))
        }
    };

    let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

    Ok((
        MlaConfig {
            hidden_size,
            num_heads,
            qk_rope_head_dim,
            qk_nope_head_dim,
            kv_lora_rank,
            ffn_hidden_size,
            rope_base,
            rmsnorm_eps,
            q_lora_rank,
            moe,
            yarn,
        },
        block_count,
        leading_dense,
    ))
}

/// llama.cpp's `is_lite` test in `llama_model_deepseek2::load_arch_hparams`: for
/// these layer counts (DeepSeek-V2-Lite, GigaChat3-10B, Kanana-2-30B) it never reads
/// `attention.q_lora_rank`.
fn mla_is_lite(n_layer: usize, n_vocab: Option<usize>) -> bool {
    n_layer == 27 || n_layer == 26 || (n_layer == 48 && n_vocab == Some(128256))
}

/// Which query projection an MLA file uses: `Some(rank)` for Q-LoRA (`attn_q_a` ->
/// `attn_q_a_norm` -> `attn_q_b`), `None` for a direct `attn_q`. Decided by which
/// tensors layer 0 has, as llama.cpp's graph does (`is_lite = layers[il].wq`),
/// and cross-checked against `attention.q_lora_rank` where llama.cpp reads it.
fn parse_mla_q_lora_rank(
    file: &GgufFile,
    block_count: usize,
    n_vocab: Option<usize>,
) -> Result<Option<usize>, ReflexError> {
    let rank_key = "deepseek2.attention.q_lora_rank";
    let key_rank = u64_meta(file, rank_key).filter(|&n| n > 0);
    let q_a = file.tensor_info("blk.0.attn_q_a.weight");
    let q_direct = file.tensor_info("blk.0.attn_q.weight").is_some();
    match (q_a, q_direct) {
        (Some(_), true) => Err(crate::reflex_err!(
            Gguf,
            "deepseek2 file has both attn_q and attn_q_a/attn_q_b query projections"
        )),
        (None, false) => Err(crate::reflex_err!(
            Gguf,
            "deepseek2 file has neither attn_q nor attn_q_a/attn_q_b query projections"
        )),
        (None, true) => {
            if let Some(rank) = key_rank {
                if !mla_is_lite(block_count, n_vocab) {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{rank_key} = {rank} but the file has a direct attn_q, not attn_q_a/attn_q_b"
                    ));
                }
            }
            Ok(None)
        }
        (Some(info), false) => {
            let rank = match info.shape.as_slice() {
                [_hidden, rank] => *rank as usize,
                other => {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "blk.0.attn_q_a.weight has unexpected shape {other:?}"
                    ))
                }
            };
            if let Some(k) = key_rank {
                if k as usize != rank {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{rank_key} = {k} doesn't match blk.0.attn_q_a.weight's width {rank}"
                    ));
                }
            }
            Ok(Some(rank))
        }
    }
}

/// `deepseek2.expert_gating_func`, defaulted the way `load_arch_hparams` does when
/// the key is absent: sigmoid for "GLM 4.7 Lite" (47 or 48 layers with a 154,880
/// vocab -- ggml-org's and bartowski's GLM-4.7-Flash files carry no key), softmax
/// for everything else (DeepSeek-V2/V2.5).
fn parse_mla_gating(
    file: &GgufFile,
    block_count: usize,
    n_vocab: Option<usize>,
) -> Result<MlaGating, ReflexError> {
    match expert_gating_func(file, "deepseek2") {
        Some(GATING_SOFTMAX) => Ok(MlaGating::Softmax),
        Some(GATING_SIGMOID) => Ok(MlaGating::Sigmoid),
        Some(other) => Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "deepseek2.expert_gating_func = {other} is not supported (only {GATING_SOFTMAX}, \
             softmax, and {GATING_SIGMOID}, sigmoid, are implemented)"
        )),
        None if (block_count == 47 || block_count == 48) && n_vocab == Some(154880) => {
            Ok(MlaGating::Sigmoid)
        }
        None => Ok(MlaGating::Softmax),
    }
}

/// `qwen35moe`'s routing config, read from `qwen35moe.*` metadata by
/// `Model::load_hybrid_inner`. llama.cpp's `build_layer_ffn` hardcodes softmax
/// gating with `norm_w = true`, so top-k weights are always renormalized
/// (`crate::moe::route_top_k`) -- unlike DeepSeek's `expert_weights_norm`-driven
/// choice ([`MlaMoeConfig::normalize_top_k`]).
pub(super) struct HybridMoeConfig {
    pub(super) expert_used_count: usize,
    /// Routed-expert FFN hidden size (`expert_feed_forward_length`, falling
    /// back to `blk.0.ffn_gate_exps.weight`'s shape).
    pub(super) n_ff_exp: usize,
    /// `expert_weights_scale`, applied only when present and not `0`/`1`
    /// (llama.cpp's `build_moe_ffn` treats both as no-ops) -- stored as `1.0`
    /// in that case.
    pub(super) weights_scale: f32,
}

/// Shape/hyperparameter config for a DeepSeek-V2/V3 Multi-head Latent Attention
/// (MLA) model (MVP step 4), read from the GGUF file's `deepseek2.*` metadata by
/// `parse_mla_config` (see that function's doc comment for what is supported and
/// what is rejected).
pub(super) struct MlaConfig {
    pub(super) hidden_size: usize,
    pub(super) num_heads: usize,
    /// Per-head dim of the RoPE-rotated part of Q/K (`qk_rope_head_dim` in real
    /// DeepSeek configs). Also `rope_dim` -- full rotation, no partial-head split
    /// like Qwen3.5's Gated Attention layers (see `LayerConfig::rotary_dim`'s doc
    /// comment) -- MLA's `q_pe`/`k_pe` are already separately-extracted buffers of
    /// exactly this width, not a slice of a wider head.
    pub(super) qk_rope_head_dim: usize,
    /// Per-head dim of the non-rotated part of Q (and, after decompression via
    /// `wk_b`, of the K side too -- see `Model::forward_mla_attn_block`).
    pub(super) qk_nope_head_dim: usize,
    /// Compressed KV-cache dimension shared by every head (MQA) -- the whole point
    /// of "latent" attention. Also the per-head width of `Vcur` before
    /// decompression via `wv_b`.
    pub(super) kv_lora_rank: usize,
    /// FFN hidden size of the dense-lead layers (`leading_dense_block_count`
    /// layers at the start of the model, always at least 1 for every real
    /// DeepSeek-V2/V3 file this MVP step targets). Layers past that use `moe`'s
    /// `n_ff_exp` instead (see `MlaFfn::Moe`).
    pub(super) ffn_hidden_size: usize,
    pub(super) rope_base: f32,
    pub(super) rmsnorm_eps: f32,
    /// `Some(q_lora_rank)` when queries go through Q-LoRA (`attn_q_a` ->
    /// `attn_q_a_norm` -> `attn_q_b`, GLM-4.7-Flash and the full-size V2/V3),
    /// `None` for a direct `attn_q` (the "lite" files).
    pub(super) q_lora_rank: Option<usize>,
    /// `Some` iff this file has MoE layers past its dense-lead layers (a real
    /// DeepSeek-V2/V3 file always does; the synthetic MVP-step-4 test fixture is
    /// dense-only, so `None` there).
    pub(super) moe: Option<MlaMoeConfig>,
    /// `Some` iff `deepseek2.rope.scaling.type == "yarn"` (every real DeepSeek-V2/V3
    /// file this MVP step has seen; the synthetic test fixture sets no rope scaling
    /// at all, so `None` there).
    pub(super) yarn: Option<MlaYarnConfig>,
}

/// MoE routing config for MLA layers past `leading_dense_block_count`. Mirrors
/// [`MoeMetaConfig`]'s role for the dense/GQA path, plus the two things real
/// DeepSeek-V2/V3 adds beyond Qwen3-MoE's convention (see
/// `Model::forward_mla_moe_ffn`): an always-on shared expert (not gated by the
/// router) and `normalize_top_k = false` (confirmed against a real
/// `DeepSeek-V2-Lite` GGUF: `expert_weights_norm` is absent, and the converter only
/// ever writes that key when the source `norm_topk_prob` is truthy -- so its
/// absence here means "don't renormalize", not "key missing, assume default true"
/// the way most other optional keys in this codebase work).
pub(super) struct MlaMoeConfig {
    pub(super) expert_used_count: usize,
    /// Routed-expert FFN hidden size (`blk.{first_moe_layer}.ffn_gate_exps.weight`'s
    /// shape) -- distinct from `MlaConfig::ffn_hidden_size` (the dense-lead layers').
    pub(super) n_ff_exp: usize,
    /// `expert_weights_scale` metadata (default `1.0`, a no-op) -- see
    /// `crate::moe`'s doc comment for the identical Qwen3-MoE convention.
    pub(super) routed_scaling_factor: f32,
    /// `expert_weights_norm` metadata, default `false` if absent (see this
    /// struct's own doc comment for why the default differs from most other
    /// optional keys in this codebase).
    pub(super) normalize_top_k: bool,
    /// Router gating function (see [`MlaGating`]).
    pub(super) gating: MlaGating,
    /// Whether the MoE layers carry an `exp_probs_b.bias` selection bias (sigmoid
    /// gating only). Loaded host-side per layer (`MlaFfn::Moe::exp_probs_b`).
    pub(super) has_exp_probs_b: bool,
}

/// An MLA file's router gating (`expert_gating_func`, defaulted as in
/// `parse_mla_gating`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MlaGating {
    /// DeepSeek-V2/V2.5: softmax over all experts, `crate::moe::route_top_k_with_norm`.
    Softmax,
    /// DeepSeek-V3, GigaChat3, GLM-4.7-Flash, Moonlight: sigmoid with an optional
    /// selection bias, `crate::moe::route_sigmoid_bias`.
    Sigmoid,
}

/// Precomputed YaRN RoPE-scaling parameters for MLA's `q_pe`/`k_pe` rotation
/// (`Model::rope_norm_yarn`) and attention softmax scale
/// (`Model::forward_mla_attn_block`). Derived once at load time
/// (`parse_mla_config`) from `deepseek2.rope.scaling.*` metadata, mirroring
/// `llama-context.cpp`'s YaRN setup and `deepseek2.cpp`'s own `kq_scale`
/// computation (both read in full while implementing this).
pub(super) struct MlaYarnConfig {
    /// `1 / rope.scaling.factor`.
    pub(super) freq_scale: f32,
    /// Always `1.0` when YaRN is active (matches llama.cpp's own default when no
    /// CLI override is given) -- this codebase has no CLI, so always `1.0` here.
    pub(super) ext_factor: f32,
    /// The (already-`DEEPSEEK2`-special-cased) `attn_factor` fed into the rotation
    /// itself -- **not** the same value as `deepseek2.cpp`'s own `kq_scale`
    /// computation, which independently reconstructs and further adjusts it (see
    /// `attention_scale` below).
    pub(super) attn_factor: f32,
    pub(super) corr_dim_start: f32,
    pub(super) corr_dim_end: f32,
    /// Precomputed final attention softmax scale, replacing the non-YaRN
    /// `1/sqrt(qk_nope_head_dim+qk_rope_head_dim)` -- see `Model::mla_attention`'s
    /// doc comment for why this dimension (not the compressed one) is scaled, and
    /// `parse_mla_config` for the YaRN-specific `mscale^2/sqrt(...)` derivation.
    pub(super) attention_scale: f32,
}

/// llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID_LOGIT_ADD`, added by the
/// community `kolibri1` patch (not in mainline llama.cpp): select the top-k
/// experts on `logits + exp_probs_b`, weight them by the *unbiased*
/// `sigmoid(logits)`. Not DeepSeek-V3's `SIGMOID` (2), which selects on
/// `sigmoid(logits) + bias` and picks different experts whenever the bias is
/// nonzero.
pub(super) const KOLIBRI_GATING_SIGMOID_LOGIT_ADD: u64 = 5;

/// llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (`src/llama-hparams.h`).
pub(super) const GATING_SOFTMAX: u64 = 1;
/// llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`: DeepSeek-V3's rule,
/// `crate::moe::route_sigmoid_bias`.
pub(super) const GATING_SIGMOID: u64 = 2;

/// `{architecture}.expert_gating_func`, or `None` when the key is absent
/// (llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_NONE`, which each architecture
/// then defaults its own way). Shared by `parse_kolibri_config` and
/// `parse_mla_config`.
pub(super) fn expert_gating_func(file: &GgufFile, architecture: &str) -> Option<u64> {
    u64_meta(file, &format!("{architecture}.expert_gating_func")).filter(|&g| g != 0)
}

/// Shape/hyperparameter config for Aleph Alpha's Kolibri-1 (`kolibri1`), read
/// from `kolibri1.*` metadata by [`parse_kolibri_config`]. Every layer is MoE
/// (routed experts plus one always-on shared expert, no dense-lead layers),
/// with sandwich norms around both the attention and FFN blocks.
/// Conventions and sources: `docs/design/kolibri.md`.
pub(super) struct KolibriConfig {
    /// Attention shape, RoPE and norm settings. `rotary_dim == head_dim`
    /// (full rotation on sliding layers), `rope_type` is `Neox`, and
    /// `ffn_hidden_size` is the routed-expert width (`n_ff_exp`).
    pub(super) layer: LayerConfig,
    pub(super) expert_count: usize,
    pub(super) expert_used_count: usize,
    /// Routed-expert FFN hidden size (`expert_feed_forward_length`). This and
    /// `n_ff_shexp` are checked against the tensors here; the forward pass
    /// reads the widths from the tensor shapes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) n_ff_exp: usize,
    /// Shared-expert FFN hidden size (`expert_shared_feed_forward_length`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) n_ff_shexp: usize,
    /// `expert_weights_norm`, default `false` when absent (real Kolibri-1 sets
    /// it to `false`; the official plugin never renormalizes).
    pub(super) normalize_top_k: bool,
    /// `attention.sliding_window`: a query at position `i` sees key `j` iff
    /// `i - j < sliding_window` (llama.cpp's `LLAMA_SWA_TYPE_STANDARD`).
    pub(super) sliding_window: usize,
    /// `attention.sliding_window_pattern`, one entry per layer: `true` = a
    /// sliding-window layer with RoPE, `false` = a full-attention layer with no
    /// positional encoding at all (NoPE). Real Kolibri-1 has `false` at every
    /// 5th layer (indices 4, 9, ..., 49).
    pub(super) sliding_layers: Vec<bool>,
}

/// Reads `file`'s [`KolibriConfig`] and layer count. Rejects anything outside
/// the shape real Kolibri-1 GGUFs have (see `docs/design/kolibri.md`) rather
/// than guessing: a gating function other than
/// [`KOLIBRI_GATING_SIGMOID_LOGIT_ADD`], a shared-expert count other than 1,
/// RoPE scaling, a missing or malformed sliding-window pattern, or
/// `value_length != key_length`.
pub(super) fn parse_kolibri_config(file: &GgufFile) -> Result<(KolibriConfig, usize), ReflexError> {
    let architecture = "kolibri1";
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let required = |suffix: &str| {
        u64_meta(file, &key(suffix))
            .map(|n| n as usize)
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {} metadata key", key(suffix)))
    };

    let block_count = required("block_count")?;
    let hidden_size = required("embedding_length")?;
    let num_q_heads = required("attention.head_count")?;
    let num_kv_heads = required("attention.head_count_kv")?;
    if num_q_heads == 0 || num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
        return Err(crate::reflex_err!(
            Gguf,
            "{} ({num_q_heads}) must be a nonzero multiple of {} ({num_kv_heads})",
            key("attention.head_count"),
            key("attention.head_count_kv")
        ));
    }
    let head_dim = u64_meta(file, &key("attention.key_length"))
        .map(|n| n as usize)
        .unwrap_or(hidden_size / num_q_heads);
    if let Some(v) = u64_meta(file, &key("attention.value_length")) {
        if v as usize != head_dim {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} ({v}) != {} ({head_dim}) is not supported",
                key("attention.value_length"),
                key("attention.key_length")
            ));
        }
    }

    let gating = expert_gating_func(file, architecture).ok_or_else(|| {
        crate::reflex_err!(Gguf, "missing {} metadata key", key("expert_gating_func"))
    })?;
    if gating != KOLIBRI_GATING_SIGMOID_LOGIT_ADD {
        return Err(crate::reflex_err!(UnsupportedArchitecture,
            "{} = {gating} is not supported (only {KOLIBRI_GATING_SIGMOID_LOGIT_ADD}, SIGMOID_LOGIT_ADD, is implemented)",
            key("expert_gating_func")
        ));
    }

    let expert_count = required("expert_count")?;
    let expert_used_count = required("expert_used_count")?;
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(crate::reflex_err!(
            Gguf,
            "{} ({expert_used_count}) must be in 1..={} ({expert_count})",
            key("expert_used_count"),
            key("expert_count")
        ));
    }
    let shared = required("expert_shared_count")?;
    if shared != 1 {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "{} = {shared} is not supported (only 1 shared expert is implemented)",
            key("expert_shared_count")
        ));
    }

    // Cross-check the declared widths against layer 0's tensors, so a
    // mislabeled file fails here instead of deep in a GEMV.
    let tensor_dim = |name: &str, dim: usize| {
        file.tensor_info(name)
            .and_then(|info| info.shape.get(dim).copied())
            .map(|n| n as usize)
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {name} tensor"))
    };
    let n_ff_exp = required("expert_feed_forward_length")?;
    let n_ff_shexp = required("expert_shared_feed_forward_length")?;
    for (name, declared) in [
        ("blk.0.ffn_gate_exps.weight", n_ff_exp),
        ("blk.0.ffn_gate_shexp.weight", n_ff_shexp),
    ] {
        let actual = tensor_dim(name, 1)?;
        if actual != declared {
            return Err(crate::reflex_err!(
                Gguf,
                "{name} has {actual} output features but the metadata declares {declared}"
            ));
        }
    }
    let stacked_experts = tensor_dim("blk.0.ffn_gate_exps.weight", 2)?;
    if stacked_experts != expert_count {
        return Err(crate::reflex_err!(
            Gguf,
            "blk.0.ffn_gate_exps.weight stacks {stacked_experts} experts but {} is {expert_count}",
            key("expert_count")
        ));
    }

    if let Some(scaling) = file
        .metadata
        .get(&key("rope.scaling.type"))
        .and_then(GgufValue::as_str)
    {
        if scaling != "none" {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {scaling:?} is not supported (Kolibri-1 uses unscaled RoPE)",
                key("rope.scaling.type")
            ));
        }
    }

    let sliding_window = required("attention.sliding_window")?;
    if sliding_window == 0 {
        return Err(crate::reflex_err!(
            Gguf,
            "{} must be > 0",
            key("attention.sliding_window")
        ));
    }
    let pattern_key = key("attention.sliding_window_pattern");
    let sliding_layers: Vec<bool> = match file.metadata.get(&pattern_key) {
        Some(GgufValue::Array(items)) if items.len() == block_count => items
            .iter()
            .map(|v| match v {
                GgufValue::Bool(b) => Ok(*b),
                other => Err(crate::reflex_err!(
                    Gguf,
                    "{pattern_key} has non-boolean entry {other:?}"
                )),
            })
            .collect::<Result<_, _>>()?,
        Some(GgufValue::Array(items)) => {
            return Err(crate::reflex_err!(
                Gguf,
                "{pattern_key} has {} entries but block_count is {block_count}",
                items.len()
            ))
        }
        Some(other) => {
            return Err(crate::reflex_err!(
                Gguf,
                "{pattern_key} must be a per-layer bool array, got {other:?}"
            ))
        }
        None => {
            return Err(crate::reflex_err!(
                Gguf,
                "missing {pattern_key} metadata key"
            ))
        }
    };

    let normalize_top_k = matches!(
        file.metadata.get(&key("expert_weights_norm")),
        Some(GgufValue::Bool(true))
    );
    // The patch's graph passes a hardcoded `w_scale = 0` (no scaling) to
    // `build_moe_ffn`, so a file that sets a real scale would mean something
    // the reference ignores. Real Kolibri-1 doesn't set it.
    if let Some(scale) = f32_meta(file, &key("expert_weights_scale")) {
        if scale != 0.0 && scale != 1.0 {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {scale} is not supported (the reference implementation ignores it)",
                key("expert_weights_scale")
            ));
        }
    }
    // Only sliding layers apply RoPE, and the patch's `load_arch_hparams`
    // gives them `rope.freq_base_swa` when present (real Kolibri-1 doesn't set
    // it, so both are 10000).
    let rope_base = f32_meta(file, &key("rope.freq_base_swa"))
        .or_else(|| f32_meta(file, &key("rope.freq_base")))
        .unwrap_or(10000.0);
    let rmsnorm_eps = f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

    Ok((
        KolibriConfig {
            layer: LayerConfig {
                hidden_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim: head_dim,
                ffn_hidden_size: n_ff_exp,
                rope_base,
                rmsnorm_eps,
                rope_type: rope_type_for(architecture),
            },
            expert_count,
            expert_used_count,
            n_ff_exp,
            n_ff_shexp,
            normalize_top_k,
            sliding_window,
            sliding_layers,
        },
        block_count,
    ))
}

/// One Gemma 4 layer's attention shape. Sliding-window and global layers
/// differ in every field (docs/design/gemma4.md's "Confirmed conventions").
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Gemma4LayerShape {
    /// `attention.sliding_window_pattern[i]`: `true` = sliding-window layer.
    pub(super) sliding: bool,
    /// `attention.key_length_swa` (sliding) or `attention.key_length`
    /// (global); `value_length` is required to match.
    pub(super) head_dim: usize,
    /// `attention.head_count_kv[i]` (a per-layer array in real files).
    pub(super) num_kv_heads: usize,
    /// `rope.dimension_count_swa` / `rope.dimension_count`: how many of each
    /// head's dims RoPE covers.
    pub(super) n_rot: usize,
    /// `rope.freq_base_swa` / `rope.freq_base`.
    pub(super) rope_base: f32,
}

/// Shape and hyperparameters of a Gemma 4 (`gemma4`) text model, read by
/// [`parse_gemma4_config`].
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Gemma4Config {
    pub(super) hidden_size: usize,
    pub(super) num_q_heads: usize,
    pub(super) ffn_hidden_size: usize,
    pub(super) rmsnorm_eps: f32,
    /// `attention.sliding_window`, llama.cpp's `LLAMA_SWA_TYPE_STANDARD`:
    /// query `i` sees key `j` iff `i - j < sliding_window`.
    pub(super) sliding_window: usize,
    /// `final_logit_softcapping`: logits become `cap * tanh(logits / cap)`.
    pub(super) final_logit_softcap: Option<f32>,
    pub(super) layers: Vec<Gemma4LayerShape>,
}

/// Reads `file`'s [`Gemma4Config`]. Rejects, rather than guesses at, every
/// variant this path does not implement: per-layer token embeddings and
/// cross-layer KV sharing (E2B/E4B), MoE blocks (26B-A4B), RoPE scaling,
/// differing K/V head lengths, a FFN width that changes per layer, and a
/// `rope_freqs.weight` that doesn't cover the global layers' RoPE dims.
pub(super) fn parse_gemma4_config(file: &GgufFile) -> Result<Gemma4Config, ReflexError> {
    let key = |suffix: &str| format!("gemma4.{suffix}");
    let required = |suffix: &str| {
        u64_meta(file, &key(suffix))
            .map(|n| n as usize)
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {} metadata key", key(suffix)))
    };
    let required_f32 = |suffix: &str| {
        f32_meta(file, &key(suffix))
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {} metadata key", key(suffix)))
    };
    let unsupported_if_nonzero = |suffix: &str, what: &str| -> Result<(), ReflexError> {
        match u64_meta(file, &key(suffix)) {
            Some(n) if n > 0 => Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {n} is not supported: {what} (Gemma 4 E2B/E4B/26B-A4B) are out of scope; only the dense 12B/31B text models are",
                key(suffix)
            )),
            _ => Ok(()),
        }
    };

    let block_count = required("block_count")?;
    let hidden_size = required("embedding_length")?;
    let num_q_heads = required("attention.head_count")?;
    unsupported_if_nonzero(
        "embedding_length_per_layer_input",
        "per-layer token embeddings",
    )?;
    unsupported_if_nonzero(
        "attention.shared_kv_layers",
        "KV caches shared across layers",
    )?;
    unsupported_if_nonzero("expert_count", "mixture-of-experts blocks")?;
    if let Some(scaling) = file
        .metadata
        .get(&key("rope.scaling.type"))
        .and_then(GgufValue::as_str)
    {
        if scaling != "none" {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} = {scaling:?} is not supported",
                key("rope.scaling.type")
            ));
        }
    }

    // A scalar applies to every layer; an array gives one value per layer.
    let per_layer = |suffix: &str| -> Result<Vec<usize>, ReflexError> {
        match file.metadata.get(&key(suffix)) {
            Some(GgufValue::Array(items)) => {
                if items.len() != block_count {
                    return Err(crate::reflex_err!(
                        Gguf,
                        "{} has {} entries but block_count is {block_count}",
                        key(suffix),
                        items.len()
                    ));
                }
                items
                    .iter()
                    .map(|v| {
                        v.as_u64().map(|n| n as usize).ok_or_else(|| {
                            crate::reflex_err!(Gguf, "{} has non-integer entry {v:?}", key(suffix))
                        })
                    })
                    .collect()
            }
            Some(v) => v
                .as_u64()
                .map(|n| vec![n as usize; block_count])
                .ok_or_else(|| crate::reflex_err!(Gguf, "{} is not an integer", key(suffix))),
            None => Err(crate::reflex_err!(
                Gguf,
                "missing {} metadata key",
                key(suffix)
            )),
        }
    };
    let kv_heads = per_layer("attention.head_count_kv")?;
    let ffn = per_layer("feed_forward_length")?;
    let ffn_hidden_size = ffn[0];
    if ffn.iter().any(|&n| n != ffn_hidden_size) {
        return Err(crate::reflex_err!(
            UnsupportedArchitecture,
            "{} varies per layer, which is not supported",
            key("feed_forward_length")
        ));
    }

    let pattern_key = key("attention.sliding_window_pattern");
    let sliding: Vec<bool> = match file.metadata.get(&pattern_key) {
        Some(GgufValue::Array(items)) if items.len() == block_count => items
            .iter()
            .map(|v| match v {
                GgufValue::Bool(b) => Ok(*b),
                other => Err(crate::reflex_err!(
                    Gguf,
                    "{pattern_key} has non-boolean entry {other:?}"
                )),
            })
            .collect::<Result<_, _>>()?,
        Some(GgufValue::Array(items)) => {
            return Err(crate::reflex_err!(
                Gguf,
                "{pattern_key} has {} entries but block_count is {block_count}",
                items.len()
            ))
        }
        _ => {
            return Err(crate::reflex_err!(
                Gguf,
                "missing or malformed {pattern_key} (expected a per-layer bool array)"
            ))
        }
    };
    let sliding_window = required("attention.sliding_window")?;
    if sliding_window == 0 {
        return Err(crate::reflex_err!(
            Gguf,
            "{} must be > 0",
            key("attention.sliding_window")
        ));
    }

    // (head_dim, n_rot, rope_base) for one layer type; `swa` picks the
    // sliding-layer keys.
    let layer_type = |swa: bool| -> Result<(usize, usize, f32), ReflexError> {
        let s = if swa { "_swa" } else { "" };
        let head_dim = required(&format!("attention.key_length{s}"))?;
        let v = required(&format!("attention.value_length{s}"))?;
        if v != head_dim {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "{} ({v}) != {} ({head_dim}) is not supported",
                key(&format!("attention.value_length{s}")),
                key(&format!("attention.key_length{s}"))
            ));
        }
        let n_rot = required(&format!("rope.dimension_count{s}"))?;
        if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
            return Err(crate::reflex_err!(
                Gguf,
                "{} = {n_rot} must be even and in 2..={head_dim}",
                key(&format!("rope.dimension_count{s}"))
            ));
        }
        let rope_base = required_f32(&format!("rope.freq_base{s}"))?;
        Ok((head_dim, n_rot, rope_base))
    };
    let swa_shape = if sliding.iter().any(|&s| s) {
        Some(layer_type(true)?)
    } else {
        None
    };
    let global_shape = if sliding.iter().any(|&s| !s) {
        Some(layer_type(false)?)
    } else {
        None
    };

    let mut layers = Vec::with_capacity(block_count);
    for (i, (&is_sliding, &num_kv_heads)) in sliding.iter().zip(&kv_heads).enumerate() {
        let (head_dim, n_rot, rope_base) = if is_sliding { swa_shape } else { global_shape }
            .expect("computed for every layer type present");
        if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(crate::reflex_err!(
                Gguf,
                "layer {i}: {} ({num_q_heads}) must be a nonzero multiple of {} ({num_kv_heads})",
                key("attention.head_count"),
                key("attention.head_count_kv")
            ));
        }
        layers.push(Gemma4LayerShape {
            sliding: is_sliding,
            head_dim,
            num_kv_heads,
            n_rot,
            rope_base,
        });
    }

    if let Some((_, n_rot, _)) = global_shape {
        let info = file.tensor_info("rope_freqs.weight").ok_or_else(|| {
            crate::reflex_err!(
                Gguf,
                "missing rope_freqs.weight (Gemma 4's global layers need it)"
            )
        })?;
        if info.element_count() as usize != n_rot / 2 {
            return Err(crate::reflex_err!(
                Gguf,
                "rope_freqs.weight has {} entries, expected {} ({}/2)",
                info.element_count(),
                n_rot / 2,
                key("rope.dimension_count")
            ));
        }
    }

    Ok(Gemma4Config {
        hidden_size,
        num_q_heads,
        ffn_hidden_size,
        rmsnorm_eps: f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
        sliding_window,
        final_logit_softcap: f32_meta(file, &key("final_logit_softcapping")).filter(|&c| c != 0.0),
        layers,
    })
}

/// The Mamba-2 SSM layers' shape, shared by every SSM layer of a model
/// (`{arch}.ssm.*`). The math is in `reference/mamba2_reference.rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Mamba2Config {
    /// `ssm.inner_size` (`D`).
    pub(super) d_inner: usize,
    /// `ssm.state_size` (`N`).
    pub(super) d_state: usize,
    /// `ssm.group_count` (`G`): head `h` reads `B`/`C` group `h / (H / G)`.
    pub(super) n_group: usize,
    /// `ssm.time_step_rank` (`H`): the head count, not a low-rank `dt` as in
    /// Mamba-1.
    pub(super) n_head: usize,
    /// `ssm.conv_kernel`.
    pub(super) d_conv: usize,
}

impl Mamba2Config {
    /// `P`, each head's slice of `d_inner`.
    pub(super) fn head_dim(&self) -> usize {
        self.d_inner / self.n_head
    }

    /// `C = D + 2·G·N`: the channels the conv runs over (x, B, C).
    pub(super) fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// `ssm_in`'s output length: z (`D`), xBC (`C`), dt (`H`).
    pub(super) fn d_in_proj(&self) -> usize {
        self.d_inner + self.conv_dim() + self.n_head
    }

    /// Floats in one layer's conv state, `[d_conv - 1, C]`.
    pub(super) fn conv_state_len(&self) -> usize {
        (self.d_conv - 1) * self.conv_dim()
    }

    /// Floats in one layer's SSM state, `[H, P, N]`.
    pub(super) fn ssm_state_len(&self) -> usize {
        self.d_inner * self.d_state
    }
}

/// A Granite 4.0-H (`granitehybrid`) model's configuration, read by
/// [`parse_granite_hybrid_config`].
#[derive(Clone, Debug, PartialEq)]
pub(super) struct GraniteHybridConfig {
    pub(super) hidden_size: usize,
    /// One entry per layer: `true` = Mamba-2 SSM mixer, `false` = attention.
    /// llama.cpp: a layer is recurrent iff its `attention.head_count_kv` is 0.
    pub(super) ssm_layers: Vec<bool>,
    pub(super) num_q_heads: usize,
    /// The attention layers' KV head count (the nonzero entries of
    /// `attention.head_count_kv`, required to agree).
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) ffn_hidden_size: usize,
    pub(super) rmsnorm_eps: f32,
    /// `embedding_scale`: token embeddings are multiplied by it (1.0 when the
    /// key is absent or 0, which llama.cpp skips).
    pub(super) embedding_scale: f32,
    /// `residual_scale`: each sublayer's output is multiplied by it before
    /// its residual add (1.0 when absent or 0).
    pub(super) residual_scale: f32,
    /// `attention.scale`: the attention's `kq_scale`, or `1/sqrt(head_dim)`
    /// when absent or 0.
    pub(super) attention_scale: f32,
    /// `logit_scale`: logits are multiplied by `1 / logit_scale` (`None` when
    /// absent or 0).
    pub(super) logit_scale: Option<f32>,
    pub(super) ssm: Mamba2Config,
}

/// The tensors a Granite 4.0-H layer may carry in llama.cpp that this path
/// doesn't implement (no surveyed file has them).
const GRANITE_UNSUPPORTED_BIASES: [&str; 7] = [
    "attn_q.bias",
    "attn_k.bias",
    "attn_v.bias",
    "attn_output.bias",
    "ffn_gate.bias",
    "ffn_up.bias",
    "ffn_down.bias",
];

/// Reads `file`'s [`GraniteHybridConfig`] from `granitehybrid.*` metadata,
/// following llama.cpp `79e2e74`'s `llama_model_granite_hybrid`
/// (`src/models/granite-hybrid.cpp`). Rejects, naming the key or tensor,
/// every variant this path doesn't implement: MoE FFNs (Granite 4.0-H
/// Tiny/Small), RoPE on the attention layers (`rope.scaling.finetuned` true,
/// or absent, which llama.cpp reads as true), a file with no SSM layer,
/// attention/FFN biases, an SSM layer without `ssm_norm` or a conv bias, and
/// SSM shapes the kernels don't cover. Survey and sources:
/// `docs/design/mamba2-hybrid.md`.
pub(super) fn parse_granite_hybrid_config(
    file: &GgufFile,
) -> Result<GraniteHybridConfig, ReflexError> {
    let arch = "granitehybrid";
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let required = |suffix: &str| {
        u64_meta(file, &key(suffix))
            .map(|n| n as usize)
            .ok_or_else(|| crate::reflex_err!(Gguf, "missing {} metadata key", key(suffix)))
    };
    let unsupported = |what: String| crate::reflex_err!(UnsupportedArchitecture, "{what}");

    if let Some(n) = u64_meta(file, &key("expert_count")).filter(|&n| n > 0) {
        return Err(unsupported(format!(
            "{} = {n} is not supported: MoE Granite 4.0-H models (Tiny, Small) are out of \
             scope; only the dense-FFN ones (350M, 1B, Micro) are",
            key("expert_count")
        )));
    }
    // llama.cpp defaults this switch to true when the key is missing.
    match file.metadata.get(&key("rope.scaling.finetuned")) {
        Some(GgufValue::Bool(false)) => {}
        other => {
            let value = match other {
                Some(v) => format!("{v:?}"),
                None => "absent (llama.cpp reads that as true)".to_string(),
            };
            return Err(unsupported(format!(
                "{} = {value} is not supported: only attention without RoPE (the key set to \
                 false, as in Granite 4.0-H) is implemented",
                key("rope.scaling.finetuned")
            )));
        }
    }
    if let Some(scaling) = file
        .metadata
        .get(&key("rope.scaling.type"))
        .and_then(GgufValue::as_str)
        .filter(|s| *s != "none")
    {
        return Err(unsupported(format!(
            "{} = {scaling:?} is not supported",
            key("rope.scaling.type")
        )));
    }

    let block_count = required("block_count")?;
    let hidden_size = required("embedding_length")?;
    let num_q_heads = required("attention.head_count")?;
    let ffn_hidden_size = required("feed_forward_length")?;

    let kv_key = key("attention.head_count_kv");
    let kv_heads: Vec<usize> = match file.metadata.get(&kv_key) {
        Some(GgufValue::Array(items)) if items.len() == block_count => items
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| crate::reflex_err!(Gguf, "{kv_key} has entry {v:?}"))
            })
            .collect::<Result<_, _>>()?,
        Some(GgufValue::Array(items)) => {
            return Err(crate::reflex_err!(
                Gguf,
                "{kv_key} has {} entries but block_count is {block_count}",
                items.len()
            ))
        }
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| crate::reflex_err!(Gguf, "{kv_key} is not an integer"))?;
            vec![n as usize; block_count]
        }
        None => return Err(crate::reflex_err!(Gguf, "missing {kv_key} metadata key")),
    };
    let ssm_layers: Vec<bool> = kv_heads.iter().map(|&n| n == 0).collect();
    if !ssm_layers.iter().any(|&s| s) {
        return Err(unsupported(format!(
            "{kv_key} has no zero entry, so this file has no SSM layer (a non-hybrid Granite \
             model, which llama.cpp loads as 'granite'/'granitemoe'); not supported"
        )));
    }
    let mut attn_kv = kv_heads.iter().copied().filter(|&n| n > 0);
    let num_kv_heads = attn_kv.next().unwrap_or(1);
    if let Some(other) = attn_kv.find(|&n| n != num_kv_heads) {
        return Err(unsupported(format!(
            "{kv_key} has differing nonzero entries ({num_kv_heads} and {other}); not supported"
        )));
    }
    if num_q_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(crate::reflex_err!(
            Gguf,
            "{} ({num_q_heads}) must be a nonzero multiple of the KV head count ({num_kv_heads})",
            key("attention.head_count")
        ));
    }
    let head_dim = u64_meta(file, &key("attention.key_length"))
        .map(|n| n as usize)
        .unwrap_or(hidden_size / num_q_heads);
    if let Some(v) = u64_meta(file, &key("attention.value_length")) {
        if v as usize != head_dim {
            return Err(unsupported(format!(
                "{} ({v}) != head dim ({head_dim}) is not supported",
                key("attention.value_length")
            )));
        }
    }

    let ssm = Mamba2Config {
        d_inner: required("ssm.inner_size")?,
        d_state: required("ssm.state_size")?,
        n_group: required("ssm.group_count")?,
        n_head: required("ssm.time_step_rank")?,
        d_conv: required("ssm.conv_kernel")?,
    };
    if ssm.d_inner != 2 * hidden_size {
        return Err(unsupported(format!(
            "{} = {} must be 2 * {} = {}, as llama.cpp asserts for {arch}",
            key("ssm.inner_size"),
            ssm.d_inner,
            key("embedding_length"),
            2 * hidden_size
        )));
    }
    if ssm.n_head == 0 || !ssm.d_inner.is_multiple_of(ssm.n_head) {
        return Err(crate::reflex_err!(
            Gguf,
            "{} = {} must divide {} = {}",
            key("ssm.time_step_rank"),
            ssm.n_head,
            key("ssm.inner_size"),
            ssm.d_inner
        ));
    }
    if ssm.n_group == 0 || !ssm.n_head.is_multiple_of(ssm.n_group) {
        return Err(crate::reflex_err!(
            Gguf,
            "{} = {} must divide {} = {}",
            key("ssm.group_count"),
            ssm.n_group,
            key("ssm.time_step_rank"),
            ssm.n_head
        ));
    }
    if ssm.d_state == 0 || !ssm.d_state.is_multiple_of(32) || ssm.d_state > 256 {
        return Err(unsupported(format!(
            "{} = {} is not supported: the scan kernel needs a multiple of 32 up to 256",
            key("ssm.state_size"),
            ssm.d_state
        )));
    }
    if ssm.d_conv < 2 {
        return Err(crate::reflex_err!(
            Gguf,
            "{} = {} must be at least 2",
            key("ssm.conv_kernel"),
            ssm.d_conv
        ));
    }

    for (i, &is_ssm) in ssm_layers.iter().enumerate() {
        let present = |t: &str| file.tensor_info(&format!("blk.{i}.{t}")).is_some();
        if let Some(b) = GRANITE_UNSUPPORTED_BIASES.iter().find(|t| present(t)) {
            return Err(unsupported(format!(
                "tensor 'blk.{i}.{b}' is not supported: no surveyed Granite 4.0-H file has \
                 attention or FFN biases"
            )));
        }
        if is_ssm {
            for t in ["ssm_norm.weight", "ssm_conv1d.bias"] {
                if !present(t) {
                    return Err(unsupported(format!(
                        "SSM layer {i} has no 'blk.{i}.{t}'; layers without it are not supported"
                    )));
                }
            }
        }
    }

    let nonzero = |suffix: &str| f32_meta(file, &key(suffix)).filter(|&s| s != 0.0);
    Ok(GraniteHybridConfig {
        hidden_size,
        ssm_layers,
        num_q_heads,
        num_kv_heads,
        head_dim,
        ffn_hidden_size,
        rmsnorm_eps: f32_meta(file, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5),
        embedding_scale: nonzero("embedding_scale").unwrap_or(1.0),
        residual_scale: nonzero("residual_scale").unwrap_or(1.0),
        attention_scale: nonzero("attention.scale").unwrap_or(1.0 / (head_dim as f32).sqrt()),
        logit_scale: nonzero("logit_scale"),
        ssm,
    })
}
