//! Phase 4 (Embeddability) round 1: parses a llama.cpp-format LoRA adapter
//! GGUF and computes the full-size, host-resident delta each targeted tensor
//! needs (`crate::model::Model::apply_lora` uploads and adds each delta into
//! the already-loaded, GPU-resident base weight exactly once, reusing the
//! existing in-place-add kernel -- no new kernel, no runtime branching in the
//! forward pass afterward).
//!
//! **Format** (confirmed against a real llama.cpp build's
//! `convert_lora_to_gguf.py` and `src/llama-adapter.cpp`, not assumed): a
//! LoRA adapter is its own self-contained GGUF file (produced from a HF PEFT
//! checkpoint's `adapter_config.json` + `adapter_model.safetensors` by that
//! converter), readable with the same `crate::gguf::GgufFile` parser this
//! project already uses for base model files -- a LoRA GGUF is not a
//! different container format, just a different set of metadata keys and
//! tensors. Required metadata: `adapter.type` (string, must be `"lora"`) and
//! `adapter.lora.alpha` (`f32`, required -- `llama-adapter.cpp`'s
//! `get_kv_f32` has no fallback-to-rank default, it silently reads `0.0` if
//! absent, which would zero every delta, so this parser treats a missing key
//! as a hard error instead). Every targeted base tensor `<name>` (e.g.
//! `"blk.0.ffn_down.weight"`, `.weight` suffix included -- confirmed against
//! a real converted adapter file; the base name is *not* stripped before the
//! suffix is appended) gets a `<name>.lora_a`/`<name>.lora_b` pair. Per this
//! project's own `model.rs` GGUF
//! shape convention (`shape = [in_features, out_features]`, flat bytes
//! row-major `(out_features, in_features)`, confirmed against
//! `llama-adapter.cpp`'s own shape-validation checks): `lora_a`'s shape is
//! `[in_features, rank]` (flat bytes row-major `(rank, in_features)`) and
//! `lora_b`'s is `[rank, out_features]` (flat bytes row-major
//! `(out_features, rank)`). Rank is read from these shapes, never a separate
//! metadata key. The update is `W' = W + scale * (B @ A)` with
//! `scale = alpha / rank` (llama.cpp's own formula, confirmed against
//! `llama-adapter.cpp`; this project has no `--lora-scale` CLI multiplier in
//! round 1, matching llama.cpp's own default `adapter_scale = 1.0`).
//!
//! **Scope, deliberately narrow** (see `Model::apply_lora`'s doc comment for
//! the full list of rejected cases): this module only *parses* the adapter
//! and computes deltas -- it has no idea which base-model tensors actually
//! exist or are supported (that lookup, and every architecture-specific
//! accept/reject decision, lives in `model.rs`, the only place that already
//! knows each architecture's real tensor set). A tensor pair this module
//! cannot make sense of on its own terms (missing partner, rank mismatch,
//! non-2-D shape) is still a hard parse error here, since those are never
//! architecture-dependent.

use crate::dequant;
use crate::gguf::{GgufFile, GgufValue};
use std::path::Path;

/// One `<name>.lora_a`/`<name>.lora_b` pair, resolved to the full-size
/// `[in_features, out_features]` delta the caller adds into the matching
/// base weight. `name` is the base tensor's own GGUF name (e.g.
/// `"blk.3.attn_q.weight"`), taken directly from the adapter's
/// `.lora_a`-stripped tensor name (already includes `.weight`).
pub struct LoraTarget {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    /// Row-major `(out_features, in_features)`, length
    /// `in_features * out_features` -- already scaled by `alpha / rank`, so
    /// the caller only ever needs to add it in, never scale it again.
    pub delta: Vec<f32>,
}

pub struct LoraAdapter {
    pub alpha: f32,
    pub targets: Vec<LoraTarget>,
}

/// Parses `path` as a llama.cpp-format LoRA adapter GGUF and computes every
/// targeted tensor's delta. Pure host-side work (dequantizes `lora_a`/
/// `lora_b` via the existing `dequant::dequantize` host path -- these are
/// small low-rank factors, not worth a device round trip) -- no CUDA device
/// is touched here; `Model::apply_lora` does the one-time upload.
pub fn load(path: &Path) -> Result<LoraAdapter, String> {
    let file =
        GgufFile::open(path).map_err(|e| format!("failed to open LoRA adapter {path:?}: {e}"))?;

    let adapter_type = file
        .metadata
        .get("adapter.type")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| format!("{path:?} is missing the required 'adapter.type' metadata key -- not a llama.cpp-format LoRA adapter GGUF"))?;
    if adapter_type != "lora" {
        return Err(format!(
            "{path:?} has adapter.type={adapter_type:?} (only \"lora\" is supported)"
        ));
    }
    let alpha = file
        .metadata
        .get("adapter.lora.alpha")
        .and_then(GgufValue::as_f32)
        .ok_or_else(|| {
            format!("{path:?} is missing the required 'adapter.lora.alpha' metadata key")
        })?;

    let mut targets = Vec::new();
    for info in &file.tensors {
        let Some(base_name) = info.name.strip_suffix(".lora_a") else {
            continue;
        };
        let lora_b_name = format!("{base_name}.lora_b");
        let b_info = file
            .tensor_info(&lora_b_name)
            .ok_or_else(|| format!("{path:?}: '{}' has no matching '{lora_b_name}'", info.name))?;

        let a_bytes = file.tensor_bytes(info)?;
        let b_bytes = file.tensor_bytes(b_info)?;
        let a_host = dequant::dequantize(info.ggml_type, a_bytes, info.element_count())
            .map_err(|e| format!("{path:?}: dequantize '{}': {e}", info.name))?;
        let b_host = dequant::dequantize(b_info.ggml_type, b_bytes, b_info.element_count())
            .map_err(|e| format!("{path:?}: dequantize '{lora_b_name}': {e}"))?;

        let (in_features, rank) = match info.shape.as_slice() {
            [in_features, rank] => (*in_features as usize, *rank as usize),
            other => {
                return Err(format!(
                "{path:?}: '{}' has unexpected shape {other:?} (expected 2-D [in_features, rank])",
                info.name
            ))
            }
        };
        let (rank_b, out_features) = match b_info.shape.as_slice() {
            [rank_b, out_features] => (*rank_b as usize, *out_features as usize),
            other => return Err(format!("{path:?}: '{lora_b_name}' has unexpected shape {other:?} (expected 2-D [rank, out_features])")),
        };
        if rank == 0 || rank_b != rank {
            return Err(format!(
                "{path:?}: rank mismatch between '{}' (rank={rank}) and '{lora_b_name}' (rank={rank_b})",
                info.name
            ));
        }

        let scale = alpha / rank as f32;
        let mut delta = vec![0f32; in_features * out_features];
        for o in 0..out_features {
            let b_row = &b_host[o * rank..(o + 1) * rank];
            let delta_row = &mut delta[o * in_features..(o + 1) * in_features];
            for (r, &b_val) in b_row.iter().enumerate() {
                let b_val = b_val * scale;
                let a_row = &a_host[r * in_features..(r + 1) * in_features];
                for i in 0..in_features {
                    delta_row[i] += b_val * a_row[i];
                }
            }
        }

        targets.push(LoraTarget {
            name: base_name.to_string(),
            in_features,
            out_features,
            rank,
            delta,
        });
    }

    if targets.is_empty() {
        return Err(format!("{path:?} has no 'lora_a'/'lora_b' tensor pairs"));
    }

    Ok(LoraAdapter { alpha, targets })
}
