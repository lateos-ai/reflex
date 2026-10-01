//! Model configuration parsed from GGUF metadata: per-architecture layer/MoE/hybrid/MLA settings and the RoPE convention.

use super::*;

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
/// per-architecture difference that silently corrupts output when wrong (see
/// DECISIONS.md's MLA RoPE entry).
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
pub(super) fn rope_type_for(architecture: &str) -> RopeType {
    match architecture {
        "llama" | "mistral" | "mixtral" => RopeType::Norm,
        _ => RopeType::Neox,
    }
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
    /// not the 64 that division would give).
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
/// Supports dense `qwen3` and any architecture reporting a nonzero
/// `<arch>.expert_count` (real `qwen3moe`, or a Mixtral-style test fixture
/// under `llama.*`) -- other architectures are out of scope (see README.md's
/// MVP order).
pub fn parse_model_config(
    file: &GgufFile,
) -> Result<(LayerConfig, usize, Option<MoeMetaConfig>), String> {
    let architecture = file
        .metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("");

    let moe = match u64_meta(file, &format!("{architecture}.expert_count")).filter(|&n| n > 0) {
        Some(expert_count) => {
            let expert_used_count = u64_meta(file, &format!("{architecture}.expert_used_count"))
                .ok_or_else(|| format!("missing {architecture}.expert_used_count metadata key"))?;
            Some(MoeMetaConfig {
                expert_count: expert_count as usize,
                expert_used_count: expert_used_count as usize,
            })
        }
        None => None,
    };

    if architecture != "qwen3"
        && !matches!(architecture, "llama" | "mistral" | "mixtral")
        && moe.is_none()
    {
        return Err(format!(
            "unsupported architecture '{architecture}': only dense 'qwen3'/'llama'/'mistral' and MoE architectures reporting a nonzero '{architecture}.expert_count' are in scope for this MVP"
        ));
    }

    // RoPE scaling types other than "none" (e.g. Llama-3.1's "linear"/"yarn"
    // long-context variants) change the per-dimension rotation frequencies;
    // this path implements only the unscaled convention, so reject rather than
    // silently run wrong frequencies (same rejection posture as MLA's Q-LoRA/
    // MTP handling). `deepseek2`'s "yarn" is handled separately on
    // the MLA path, never here.
    if let Some(scaling) = file
        .metadata
        .get(&format!("{architecture}.rope.scaling.type"))
        .and_then(GgufValue::as_str)
    {
        if scaling != "none" {
            return Err(format!(
                "{architecture}.rope.scaling.type = {scaling:?} is not supported by this MVP (only unscaled RoPE, i.e. no scaling or \"none\", is implemented)"
            ));
        }
    }

    let block_count = u64_meta(file, &format!("{architecture}.block_count"))
        .ok_or_else(|| format!("missing {architecture}.block_count metadata key"))?
        as usize;
    let hidden_size = u64_meta(file, &format!("{architecture}.embedding_length"))
        .ok_or_else(|| format!("missing {architecture}.embedding_length metadata key"))?
        as usize;
    let num_q_heads = u64_meta(file, &format!("{architecture}.attention.head_count"))
        .ok_or_else(|| format!("missing {architecture}.attention.head_count metadata key"))?
        as usize;
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
        .ok_or_else(|| format!("missing {ffn_gate_weight_name} tensor"))?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        [_in_features, out_features, _expert_count] => *out_features as usize,
        other => {
            return Err(format!(
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
) -> Result<Vec<bool>, String> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let recurrent_key = key("attention.recurrent_layers");
    match file.metadata.get(&recurrent_key) {
        Some(GgufValue::Array(items)) => {
            if items.len() != block_count {
                return Err(format!(
                    "{recurrent_key} has {} entries but block_count is {block_count}",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|v| match v {
                    GgufValue::Bool(b) => Ok(*b),
                    other => other
                        .as_u64()
                        .map(|x| x != 0)
                        .ok_or_else(|| format!("{recurrent_key} has non-boolean entry {other:?}")),
                })
                .collect()
        }
        Some(other) => Err(format!("{recurrent_key} must be an array, got {other:?}")),
        None => {
            let interval_key = key("full_attention_interval");
            let interval = u64_meta(file, &interval_key).ok_or_else(|| {
                format!("{architecture} model has neither {recurrent_key} nor {interval_key}")
            })? as usize;
            if interval == 0 {
                return Err(format!("{interval_key} must be > 0"));
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
) -> Result<HybridMoeConfig, String> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let expert_count = u64_meta(file, &key("expert_count"))
        .ok_or_else(|| format!("missing {}", key("expert_count")))? as usize;
    let expert_used_count = u64_meta(file, &key("expert_used_count"))
        .ok_or_else(|| format!("missing {}", key("expert_used_count")))?
        as usize;
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(format!(
            "{} ({expert_used_count}) must be in 1..={} ({expert_count})",
            key("expert_used_count"),
            key("expert_count")
        ));
    }
    if file.tensor_info("blk.0.ffn_gate_up_exps.weight").is_some() {
        return Err(format!(
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
                format!(
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
/// metadata. Scope narrowed to real DeepSeek-V2/V3 files' actual shape (confirmed
/// against a real `DeepSeek-V2-Lite` GGUF's metadata while extending this from the
/// MVP-step-4 synthetic-fixture-only version): dense-lead + MoE-with-shared-expert
/// FFN, `is_lite`-style direct `wq` (no Q-LoRA), and YaRN RoPE scaling are all
/// supported now. Still hard-errors (matching the hybrid path's MTP rejection
/// precedent) on: Q-LoRA query decomposition (`attention.q_lora_rank` present and
/// nonzero -- no real small file needing this has been seen yet), MTP/NextN
/// blocks, and any RoPE scaling type other than `"none"`/`"yarn"`.
pub(super) fn parse_mla_config(file: &GgufFile) -> Result<(MlaConfig, usize, usize), String> {
    let architecture = "deepseek2";
    let key = |suffix: &str| format!("{architecture}.{suffix}");

    let block_count = u64_meta(file, &key("block_count"))
        .ok_or_else(|| format!("missing {}", key("block_count")))? as usize;

    let nextn = u64_meta(file, &key("nextn_predict_layers")).unwrap_or(0);
    if nextn != 0 {
        return Err(format!(
            "{} MTP/NextN blocks (nextn_predict_layers={nextn}) are not supported by this MVP",
            key("nextn_predict_layers")
        ));
    }

    let leading_dense = u64_meta(file, &key("leading_dense_block_count"))
        .ok_or_else(|| format!("missing {}", key("leading_dense_block_count")))?
        as usize;

    if u64_meta(file, &key("attention.q_lora_rank"))
        .filter(|&n| n > 0)
        .is_some()
    {
        return Err(format!(
            "{} (Q-LoRA query decomposition) is not supported by this MVP",
            key("attention.q_lora_rank")
        ));
    }

    let hidden_size = u64_meta(file, &key("embedding_length"))
        .ok_or_else(|| format!("missing {}", key("embedding_length")))?
        as usize;
    let num_heads = u64_meta(file, &key("attention.head_count"))
        .ok_or_else(|| format!("missing {}", key("attention.head_count")))?
        as usize;
    let kv_lora_rank = u64_meta(file, &key("attention.kv_lora_rank"))
        .ok_or_else(|| format!("missing {}", key("attention.kv_lora_rank")))?
        as usize;
    let n_embd_head_k_mla = u64_meta(file, &key("attention.key_length_mla"))
        .ok_or_else(|| format!("missing {} (a legacy pre-MLA-split deepseek2 GGUF -- unsplit attn_kv_b, no key_length_mla/value_length_mla metadata -- is not supported by this MVP; reconvert from the original checkpoint with a current convert_hf_to_gguf.py)", key("attention.key_length_mla")))? as usize;
    let v_head_dim = u64_meta(file, &key("attention.value_length_mla"))
        .ok_or_else(|| format!("missing {}", key("attention.value_length_mla")))?
        as usize;
    let qk_rope_head_dim = u64_meta(file, &key("rope.dimension_count"))
        .ok_or_else(|| format!("missing {}", key("rope.dimension_count")))?
        as usize;
    if n_embd_head_k_mla <= qk_rope_head_dim {
        return Err(format!(
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
        .ok_or("missing blk.0.ffn_gate.weight tensor (an all-MoE deepseek2 file, leading_dense_block_count == 0, is not supported by this MVP)")?;
    let ffn_hidden_size = match ffn_gate_info.shape.as_slice() {
        [_in_features, out_features] => *out_features as usize,
        other => {
            return Err(format!(
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
        .ok_or("missing blk.0.attn_v_b.weight tensor")?;
    match wv_b_info.shape.as_slice() {
        [_in_features, out_features, _n_head] if *out_features as usize == v_head_dim => {}
        other => {
            return Err(format!(
                "blk.0.attn_v_b.weight shape {other:?} doesn't match {} ({v_head_dim})",
                key("attention.value_length_mla")
            ))
        }
    }

    let moe = if leading_dense < block_count {
        let expert_used_count = u64_meta(file, &key("expert_used_count")).ok_or_else(|| {
            format!(
                "missing {} (expert_count > 0 implied by leading_dense_block_count < block_count)",
                key("expert_used_count")
            )
        })? as usize;
        let ffn_gate_exps_info = file
            .tensor_info(&format!("blk.{leading_dense}.ffn_gate_exps.weight"))
            .ok_or_else(|| format!("missing blk.{leading_dense}.ffn_gate_exps.weight tensor"))?;
        let n_ff_exp = match ffn_gate_exps_info.shape.as_slice() {
            [_in_features, out_features, _expert_count] => *out_features as usize,
            other => {
                return Err(format!(
                    "blk.{leading_dense}.ffn_gate_exps.weight has unexpected shape {other:?}"
                ))
            }
        };
        let routed_scaling_factor = f32_meta(file, &key("expert_weights_scale")).unwrap_or(1.0);
        // The converter only ever writes this key when the source model's
        // `norm_topk_prob` is truthy (see `MlaMoeConfig`'s doc comment) -- absence
        // means "don't renormalize", not "assume the usual true default".
        let normalize_top_k = matches!(
            file.metadata.get(&key("expert_weights_norm")),
            Some(GgufValue::Bool(true))
        );
        Some(MlaMoeConfig {
            expert_used_count,
            n_ff_exp,
            routed_scaling_factor,
            normalize_top_k,
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
            let factor = f32_meta(file, &key("rope.scaling.factor"))
                .ok_or_else(|| format!("missing {}", key("rope.scaling.factor")))?;
            let orig_ctx_len = u64_meta(file, &key("rope.scaling.original_context_length"))
                .ok_or_else(|| format!("missing {}", key("rope.scaling.original_context_length")))?
                as f32;
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
            return Err(format!(
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
            moe,
            yarn,
        },
        block_count,
        leading_dense,
    ))
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
/// `parse_mla_config`. Scope deliberately narrowed (see that function's doc
/// comment): dense-only (no MoE FFN), no Q-LoRA query decomposition (`is_lite`-style
/// direct `wq` only), no YaRN RoPE scaling, no MTP/NextN.
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
}

/// Precomputed YaRN RoPE-scaling parameters for MLA's `q_pe`/`k_pe` rotation
/// (`Model::rope_norm_yarn`) and attention softmax scale
/// (`Model::forward_mla_attn_block`). Derived once at load time
/// (`parse_mla_config`) from `deepseek2.rope.scaling.*` metadata, mirroring
/// `llama-context.cpp`'s YaRN setup and `deepseek2.cpp`'s own `kq_scale`
/// computation (both read in full while implementing this -- see DECISIONS.md).
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
