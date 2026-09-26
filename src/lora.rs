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
//! **MoE per-expert-stacked targets** (e.g. `blk.0.ffn_gate_exps.weight`):
//! confirmed against llama.cpp's own `Qwen2MoeModel.modify_tensors` (which
//! `Qwen3MoeModel`/every other MoE arch's `LoraModel` subclass in
//! `convert_lora_to_gguf.py` inherits unchanged) -- for a standard PEFT
//! adapter with one `nn.Linear` per expert (`experts.{i}.gate_proj` etc.,
//! *not* a fused/grouped-GEMM expert representation some training
//! frameworks use, which that script can't convert at all: no per-expert
//! index in the tensor name means its expert-stacking `torch.stack` loop
//! never fires), the converter collects each expert's `lora_A`/`lora_B` into
//! `LoraTorchTensor`s and `torch.stack`s them along a new leading dim
//! (`LoraTorchTensor.__torch_function__`'s `torch.stack` arm stacks `_lora_A`
//! and `_lora_B` separately), producing final PyTorch shapes `(n_experts,
//! rank, in_features)` for `lora_a` and `(n_experts, out_features, rank)` for
//! `lora_b` -- reversed into GGUF ne order (this project's `shape` field) as
//! `[in_features, rank, expert_count]` and `[rank, out_features,
//! expert_count]`, exactly one more trailing dim than the dense 2-D case.
//! Byte layout follows directly: expert `e`'s slice is a contiguous
//! `rank * in_features` (resp. `out_features * rank`) run, in the same
//! row-major layout as the dense case -- so the dense math below only needs
//! an outer per-expert loop, not new math.
//!
//! **Scope, deliberately narrow** (see `Model::apply_lora`'s doc comment for
//! the full list of rejected cases): this module only *parses* the adapter
//! and computes deltas -- it has no idea which base-model tensors actually
//! exist or are supported (that lookup, and every architecture-specific
//! accept/reject decision, lives in `model.rs`, the only place that already
//! knows each architecture's real tensor set). A tensor pair this module
//! cannot make sense of on its own terms (missing partner, rank mismatch,
//! neither a 2-D nor a 3-D per-expert-stacked shape, or a 3-D shape whose
//! expert counts disagree between `lora_a`/`lora_b`) is still a hard parse
//! error here, since those are never architecture-dependent.

use crate::dequant;
use crate::gguf::{GgufFile, GgufValue};
use std::path::Path;

/// One `<name>.lora_a`/`<name>.lora_b` pair, resolved to the full-size delta
/// the caller adds into the matching base weight. `name` is the base
/// tensor's own GGUF name (e.g. `"blk.3.attn_q.weight"`), taken directly
/// from the adapter's `.lora_a`-stripped tensor name (already includes
/// `.weight`).
pub struct LoraTarget {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    /// `Some(expert_count)` for a per-expert-stacked MoE target (a 3-D base
    /// tensor, `[in_features, out_features, expert_count]`), `None` for a
    /// plain 2-D `nn.Linear` target -- lets `Model::apply_lora` check this
    /// adapter tensor was meant for the kind of base weight it's about to
    /// apply to, not just that the raw element count happens to match.
    pub expert_count: Option<usize>,
    /// Flattened delta, already scaled by `alpha / rank`, so the caller only
    /// ever needs to add it in, never scale it again. For a dense 2-D
    /// target: row-major `(out_features, in_features)`, length
    /// `in_features * out_features`. For a per-expert-stacked target:
    /// `expert_count` contiguous chunks of that same layout back-to-back
    /// (expert `e`'s chunk at `[e * in_features * out_features..]`) --
    /// identical to how the base model's own per-expert-stacked `Weight`
    /// buffer is laid out (see `model.rs`'s `expert_weight_view` doc
    /// comment), so it can be added into that whole buffer in one launch
    /// with no per-expert slicing needed on the caller's side.
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

        let (in_features, rank, expert_count_a) = match info.shape.as_slice() {
            [in_features, rank] => (*in_features as usize, *rank as usize, None),
            [in_features, rank, expert_count] => (
                *in_features as usize,
                *rank as usize,
                Some(*expert_count as usize),
            ),
            other => return Err(format!(
                "{path:?}: '{}' has unexpected shape {other:?} (expected 2-D [in_features, rank] or 3-D per-expert-stacked [in_features, rank, expert_count])",
                info.name
            )),
        };
        let (rank_b, out_features, expert_count_b) = match b_info.shape.as_slice() {
            [rank_b, out_features] => (*rank_b as usize, *out_features as usize, None),
            [rank_b, out_features, expert_count] => (
                *rank_b as usize,
                *out_features as usize,
                Some(*expert_count as usize),
            ),
            other => return Err(format!("{path:?}: '{lora_b_name}' has unexpected shape {other:?} (expected 2-D [rank, out_features] or 3-D per-expert-stacked [rank, out_features, expert_count])")),
        };
        if rank == 0 || rank_b != rank {
            return Err(format!(
                "{path:?}: rank mismatch between '{}' (rank={rank}) and '{lora_b_name}' (rank={rank_b})",
                info.name
            ));
        }
        if expert_count_a != expert_count_b {
            return Err(format!(
                "{path:?}: expert-count mismatch between '{}' ({expert_count_a:?}) and '{lora_b_name}' ({expert_count_b:?})",
                info.name
            ));
        }
        let expert_count = expert_count_a;

        let scale = alpha / rank as f32;
        let experts = expert_count.unwrap_or(1);
        let expert_a_len = rank * in_features;
        let expert_b_len = out_features * rank;
        let expert_delta_len = in_features * out_features;
        let mut delta = vec![0f32; expert_delta_len * experts];
        for e in 0..experts {
            let a_expert = &a_host[e * expert_a_len..(e + 1) * expert_a_len];
            let b_expert = &b_host[e * expert_b_len..(e + 1) * expert_b_len];
            let delta_expert = &mut delta[e * expert_delta_len..(e + 1) * expert_delta_len];
            for o in 0..out_features {
                let b_row = &b_expert[o * rank..(o + 1) * rank];
                let delta_row = &mut delta_expert[o * in_features..(o + 1) * in_features];
                for (r, &b_val) in b_row.iter().enumerate() {
                    let b_val = b_val * scale;
                    let a_row = &a_expert[r * in_features..(r + 1) * in_features];
                    for i in 0..in_features {
                        delta_row[i] += b_val * a_row[i];
                    }
                }
            }
        }

        targets.push(LoraTarget {
            name: base_name.to_string(),
            in_features,
            out_features,
            rank,
            expert_count,
            delta,
        });
    }

    if targets.is_empty() {
        return Err(format!("{path:?} has no 'lora_a'/'lora_b' tensor pairs"));
    }

    Ok(LoraAdapter { alpha, targets })
}

/// Host-only (pure parsing/math, no CUDA device needed) verification of the
/// per-expert-stacked delta math against `test-data/tiny-qwen3moe-lora.gguf`
/// -- a hand-built synthetic adapter (see
/// `scripts/build_tiny_moe_lora_fixture.py`'s doc comment for why no real
/// MoE LoRA adapter could be used as a fixture instead, and for the exact
/// value formula this test recomputes independently below). Still
/// `#[ignore]`d like every other local-fixture test in this codebase, since
/// `test-data/*.gguf` is gitignored and won't exist on a fresh checkout/CI
/// runner -- run with:
/// `cargo test -- --ignored moe_per_expert_lora_matches_hand_computed_delta`.
#[cfg(test)]
mod moe_expert_lora_fixture_tests {
    use super::*;
    use std::path::Path;

    const FIXTURE: &str = "test-data/tiny-qwen3moe-lora.gguf";
    const IN_FEATURES: usize = 32;
    const OUT_FEATURES: usize = 32;
    const EXPERT_COUNT: usize = 8;
    const RANK: usize = 2;
    const ALPHA: f32 = 8.0;
    const KINDS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

    /// Independently recomputes the delta the fixture-building script's
    /// `A[e,r,i] = 100000*layer + 10000*kind_id + 1000*e + 10*r + i` /
    /// `B[e,r,o] = 100000*layer + 10000*kind_id + 1000*e + 100*r + o`
    /// formula should produce, via a plain triple loop -- deliberately not
    /// sharing any code with `load`'s own per-expert loop above, so this
    /// only passes if that loop reads the right expert/rank/output/input
    /// offsets out of the file's raw bytes, not just because both sides run
    /// the same formula.
    fn expected_delta(layer: usize, kind_id: usize) -> Vec<f32> {
        let scale = ALPHA / RANK as f32;
        let mut delta = vec![0f32; EXPERT_COUNT * OUT_FEATURES * IN_FEATURES];
        for e in 0..EXPERT_COUNT {
            for o in 0..OUT_FEATURES {
                for i in 0..IN_FEATURES {
                    let mut acc = 0f32;
                    for r in 0..RANK {
                        let a = (100000 * layer + 10000 * kind_id + 1000 * e + 10 * r + i) as f32;
                        let b = (100000 * layer + 10000 * kind_id + 1000 * e + 100 * r + o) as f32;
                        acc += b * a;
                    }
                    delta[e * OUT_FEATURES * IN_FEATURES + o * IN_FEATURES + i] = acc * scale;
                }
            }
        }
        delta
    }

    #[test]
    #[ignore]
    fn moe_per_expert_lora_matches_hand_computed_delta() {
        let adapter = load(Path::new(FIXTURE)).expect("failed to load synthetic MoE LoRA fixture");
        assert_eq!(adapter.alpha, ALPHA);
        assert_eq!(
            adapter.targets.len(),
            6,
            "2 layers x 3 MoE FFN tensor kinds"
        );

        for layer in 0..2usize {
            for (kind_id, kind) in KINDS.iter().enumerate() {
                let name = format!("blk.{layer}.{kind}.weight");
                let target = adapter
                    .targets
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap_or_else(|| panic!("missing target '{name}'"));
                assert_eq!(target.in_features, IN_FEATURES);
                assert_eq!(target.out_features, OUT_FEATURES);
                assert_eq!(target.rank, RANK);
                assert_eq!(target.expert_count, Some(EXPERT_COUNT));
                assert_eq!(
                    target.delta,
                    expected_delta(layer, kind_id),
                    "delta mismatch for '{name}'"
                );
            }
        }
    }
}
