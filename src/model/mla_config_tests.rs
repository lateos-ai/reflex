//! Host-only `deepseek2` (MLA) config tests on the real headers from the
//! round-2 phase-3 survey: what each file needs (gating, selection bias,
//! Q-LoRA, scaling), what is still rejected, and that every new key is read
//! the way llama.cpp `79e2e74` reads it. Design, survey and sources:
//! docs/design/mla-widening.md.
//!
//! Each header is the first 40 MB of a real GGUF with the tensor table kept
//! and the tensor data dropped (`scripts/gguf_tokenizer_only.py
//! --keep-tensor-table`); sources are listed in the design note.

use super::config::{parse_mla_config, MlaConfig, MlaGating, MlaMoeConfig};
use crate::gguf::{GgufFile, GgufValue};
use crate::tokenizer::Tokenizer;

fn header(name: &str) -> Option<GgufFile> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-data")
        .join(name);
    if !path.exists() {
        eprintln!("Skipping: test-data/{name} not present (see docs/design/mla-widening.md)");
        return None;
    }
    Some(GgufFile::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}")))
}

fn parse(file: &GgufFile) -> (MlaConfig, usize, usize) {
    match parse_mla_config(file) {
        Ok(parsed) => parsed,
        Err(e) => panic!("parse_mla_config failed: {e}"),
    }
}

fn moe(cfg: &MlaConfig) -> &MlaMoeConfig {
    cfg.moe.as_ref().expect("MoE layers")
}

fn err_text(file: &GgufFile) -> String {
    match parse_mla_config(file) {
        Ok(_) => panic!("expected parse_mla_config to fail"),
        Err(e) => e.to_string(),
    }
}

/// Every header has these, and the tokenizer's pre type decides whether the
/// file can be used at all.
fn tokenizer_result(file: &GgufFile) -> Result<Tokenizer, String> {
    Tokenizer::from_gguf(file).map_err(|e| e.to_string())
}

#[test]
fn deepseek_coder_v2_lite_needs_nothing_new() {
    let Some(file) = header("deepseek-coder-v2-lite-header.gguf") else {
        return;
    };
    let (cfg, blocks, lead) = parse(&file);
    assert_eq!((blocks, lead), (27, 1));
    assert_eq!(cfg.q_lora_rank, None, "direct attn_q");
    assert_eq!(
        (cfg.kv_lora_rank, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim),
        (512, 128, 64)
    );
    let m = moe(&cfg);
    assert_eq!(m.gating, MlaGating::Softmax, "expert_gating_func = 1");
    assert!(!m.has_exp_probs_b);
    assert_eq!(m.expert_used_count, 6);
    assert_eq!(m.routed_scaling_factor, 1.0);
    assert!(!m.normalize_top_k, "no expert_weights_norm key");
    assert!(cfg.yarn.is_some());
    let tok = tokenizer_result(&file).expect("deepseek-llm is supported");
    assert_eq!(tok.pre_type, Some(crate::tokenizer::PreType::DeepseekLlm));
}

#[test]
fn legacy_unsplit_deepseek2_file_is_still_rejected() {
    let Some(file) = header("deepseek-coder-v2-lite-legacy-header.gguf") else {
        return;
    };
    let err = err_text(&file);
    assert!(err.contains("legacy pre-MLA-split"), "{err}");
}

#[test]
fn gigachat3_uses_sigmoid_gating_with_a_selection_bias() {
    let Some(file) = header("gigachat3.1-10b-a1.8b-header.gguf") else {
        return;
    };
    let (cfg, blocks, lead) = parse(&file);
    assert_eq!((blocks, lead), (26, 1));
    assert_eq!(
        cfg.q_lora_rank, None,
        "26 layers: llama.cpp's is_lite, direct attn_q"
    );
    assert_eq!(cfg.num_heads, 32);
    let m = moe(&cfg);
    assert_eq!(m.gating, MlaGating::Sigmoid);
    assert!(m.has_exp_probs_b);
    assert_eq!(m.expert_used_count, 4);
    assert_eq!(m.routed_scaling_factor, 1.0);
    assert!(m.normalize_top_k);
    assert!(cfg.yarn.is_some(), "yarn x64");
    let tok = tokenizer_result(&file).expect("gigachat is supported");
    assert_eq!(tok.pre_type, Some(crate::tokenizer::PreType::Gpt2));
    assert_eq!(tok.prompt_bos(), Some(1), "add_bos_token = true");
}

fn assert_glm_4_7_flash(file: &GgufFile) {
    let (cfg, blocks, lead) = parse(file);
    assert_eq!((blocks, lead), (47, 1));
    assert_eq!(cfg.q_lora_rank, Some(768), "Q-LoRA");
    assert_eq!(cfg.num_heads, 20);
    assert_eq!((cfg.qk_nope_head_dim, cfg.qk_rope_head_dim), (192, 64));
    let m = moe(&cfg);
    assert_eq!(m.gating, MlaGating::Sigmoid);
    assert!(m.has_exp_probs_b);
    assert_eq!(m.expert_used_count, 4);
    assert!((m.routed_scaling_factor - 1.8).abs() < 1e-6);
    assert!(m.normalize_top_k);
    assert!(cfg.yarn.is_none());
    let tok = tokenizer_result(file).expect("glm4 is supported");
    assert_eq!(tok.pre_type, Some(crate::tokenizer::PreType::Llama3));
    assert_eq!(tok.prompt_bos(), None, "llama.cpp nulls glm4's BOS");
}

#[test]
fn glm_4_7_flash_uses_q_lora_and_sigmoid_gating() {
    let Some(file) = header("glm-4.7-flash-header.gguf") else {
        return;
    };
    assert_eq!(
        file.metadata
            .get("deepseek2.expert_gating_func")
            .and_then(GgufValue::as_u64),
        Some(2)
    );
    assert_glm_4_7_flash(&file);
}

/// ggml-org's (and bartowski's) GLM-4.7-Flash files have no
/// `expert_gating_func` key; llama.cpp's `load_arch_hparams` turns 47 layers
/// with a 154,880 vocab into sigmoid.
#[test]
fn glm_4_7_flash_without_a_gating_key_still_gets_sigmoid() {
    let Some(file) = header("glm-4.7-flash-ggml-org-header.gguf") else {
        return;
    };
    assert!(!file.metadata.contains_key("deepseek2.expert_gating_func"));
    assert_glm_4_7_flash(&file);
}

/// The fallback is that exact rule: with any other vocab size a key-less file
/// stays softmax (DeepSeek-V2/V2.5), as does any other layer count.
#[test]
fn missing_gating_key_defaults_to_softmax_outside_glm_4_7() {
    let Some(mut file) = header("glm-4.7-flash-ggml-org-header.gguf") else {
        return;
    };
    // Softmax with exp_probs_b is rejected, so drop the bias tensors too.
    file.tensors
        .retain(|t| !t.name.ends_with("exp_probs_b.bias"));
    file.metadata
        .insert("deepseek2.vocab_size".into(), GgufValue::U32(154_881));
    assert_eq!(moe(&parse(&file).0).gating, MlaGating::Softmax);
    file.metadata
        .insert("deepseek2.vocab_size".into(), GgufValue::U32(154_880));
    file.metadata
        .insert("deepseek2.block_count".into(), GgufValue::U32(46));
    assert_eq!(moe(&parse(&file).0).gating, MlaGating::Softmax);
    file.metadata
        .insert("deepseek2.block_count".into(), GgufValue::U32(48));
    assert_eq!(moe(&parse(&file).0).gating, MlaGating::Sigmoid);
}

/// Moonlight and Kanana-2 parse (sigmoid, bias, scale), but their pre types
/// aren't ported yet, so the tokenizer rejects them by name.
#[test]
fn moonlight_and_kanana2_parse_but_their_pre_types_are_rejected() {
    for (name, pre, scale, experts) in [
        ("moonlight-16b-a3b-header.gguf", "kimi-k2", 2.446, 64),
        ("kanana-2-30b-a3b-header.gguf", "kanana2", 2.448, 128),
    ] {
        let Some(file) = header(name) else {
            continue;
        };
        let (cfg, _, _) = parse(&file);
        assert_eq!(cfg.q_lora_rank, None, "{name}");
        let m = moe(&cfg);
        assert_eq!(m.gating, MlaGating::Sigmoid, "{name}");
        assert!(m.has_exp_probs_b, "{name}");
        assert!((m.routed_scaling_factor - scale).abs() < 1e-3, "{name}");
        let first_moe = file.tensor_info("blk.1.ffn_gate_inp.weight").unwrap();
        assert_eq!(first_moe.shape[1], experts, "{name}");
        let err = tokenizer_result(&file).err().expect("pre type not ported");
        assert!(err.contains(&format!("'{pre}'")), "{name}: {err}");
    }
}

#[test]
fn expert_weights_scale_of_zero_is_a_no_op() {
    let Some(mut file) = header("gigachat3.1-10b-a1.8b-header.gguf") else {
        return;
    };
    file.metadata
        .insert("deepseek2.expert_weights_scale".into(), GgufValue::F32(0.0));
    assert_eq!(moe(&parse(&file).0).routed_scaling_factor, 1.0);
}

#[test]
fn unsupported_routing_and_attention_variants_are_rejected() {
    if header("gigachat3.1-10b-a1.8b-header.gguf").is_none() {
        return;
    }
    let reopen = || header("gigachat3.1-10b-a1.8b-header.gguf").unwrap();

    let mut file = reopen();
    file.metadata
        .insert("deepseek2.expert_group_count".into(), GgufValue::U32(8));
    let err = err_text(&file);
    assert!(err.contains("expert_group_count = 8"), "{err}");

    for gating in [3u32, 4] {
        let mut file = reopen();
        file.metadata.insert(
            "deepseek2.expert_gating_func".into(),
            GgufValue::U32(gating),
        );
        let err = err_text(&file);
        assert!(
            err.contains(&format!("expert_gating_func = {gating}")),
            "{err}"
        );
    }

    let mut file = reopen();
    file.metadata
        .insert("deepseek2.expert_gating_func".into(), GgufValue::U32(1));
    let err = err_text(&file);
    assert!(err.contains("exp_probs_b with softmax"), "{err}");

    let mut file = reopen();
    file.metadata.insert(
        "deepseek2.attention.temperature_scale".into(),
        GgufValue::F32(0.1),
    );
    let err = err_text(&file);
    assert!(err.contains("temperature_scale"), "{err}");

    let mut file = reopen();
    file.metadata
        .insert("deepseek2.nextn_predict_layers".into(), GgufValue::U32(1));
    let err = err_text(&file);
    assert!(err.contains("nextn_predict_layers"), "{err}");
}

#[test]
fn query_projection_must_be_exactly_one_of_attn_q_and_q_lora() {
    let reopen = || header("glm-4.7-flash-header.gguf");
    let Some(mut file) = reopen() else {
        return;
    };
    let q_a = file.tensor_info("blk.0.attn_q_a.weight").unwrap().clone();
    let mut direct = q_a.clone();
    direct.name = "blk.0.attn_q.weight".into();
    file.tensors.push(direct.clone());
    assert!(err_text(&file).contains("both attn_q and attn_q_a"));

    let mut file = reopen().unwrap();
    file.tensors.retain(|t| t.name != "blk.0.attn_q_a.weight");
    assert!(err_text(&file).contains("neither attn_q nor attn_q_a"));

    let mut file = reopen().unwrap();
    file.metadata.insert(
        "deepseek2.attention.q_lora_rank".into(),
        GgufValue::U32(512),
    );
    let err = err_text(&file);
    assert!(err.contains("q_lora_rank = 512 doesn't match"), "{err}");

    // A non-lite file declaring a Q-LoRA rank but shipping a direct attn_q
    // (llama.cpp would fail to find attn_q_a).
    let mut file = reopen().unwrap();
    file.tensors.retain(|t| t.name != "blk.0.attn_q_a.weight");
    file.tensors.push(direct);
    let err = err_text(&file);
    assert!(err.contains("direct attn_q"), "{err}");

    // GigaChat3 (26 layers, lite): llama.cpp never reads q_lora_rank there, so a
    // stray key next to attn_q is ignored.
    if let Some(mut file) = header("gigachat3.1-10b-a1.8b-header.gguf") {
        file.metadata.insert(
            "deepseek2.attention.q_lora_rank".into(),
            GgufValue::U32(768),
        );
        assert_eq!(parse(&file).0.q_lora_rank, None);
    }
}

/// Every MLA config parse of `deepseek-tiny-mla` (the MVP-step-4 fixture,
/// dense-only, direct `wq`) is unchanged by the widening.
#[test]
fn deepseek_tiny_mla_fixture_config_is_unchanged() {
    let Some(file) = header("deepseek-tiny-mla.gguf") else {
        return;
    };
    let (cfg, blocks, lead) = parse(&file);
    assert_eq!(blocks, lead, "dense only");
    assert!(cfg.moe.is_none());
    assert_eq!(cfg.q_lora_rank, None);
    assert!(cfg.yarn.is_none());
}
