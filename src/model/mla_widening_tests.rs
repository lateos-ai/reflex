//! `deepseek2` (MLA) widening tests: host-only config tests on the real
//! headers from the round-2 phase-3 survey, then GPU tests of the
//! `tiny-mla-v3` fixture against llama.cpp. The host tests cover what each
//! file needs (gating, selection bias,
//! Q-LoRA, scaling), what is still rejected, and that every new key is read
//! the way llama.cpp `79e2e74` reads it. Design, survey and sources:
//! docs/design/mla-widening.md.
//!
//! Each header is the first 40 MB of a real GGUF with the tensor table kept
//! and the tensor data dropped (`scripts/gguf_tokenizer_only.py
//! --keep-tensor-table`); sources are listed in the design note.

use super::config::{parse_mla_config, MlaConfig, MlaGating, MlaMoeConfig};
use super::prefill_batching_tests::assert_prefill_hidden_close;
use super::qwen2_tests::prompts;
use super::*;
use crate::gguf::{GgufFile, GgufValue};
use crate::tokenizer::Tokenizer;
use cudarc::driver::CudaDevice;

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

// ---------------------------------------------------------------------------
// The tiny-mla-v3 fixture
// ---------------------------------------------------------------------------

/// `test-data/tiny-mla-v3.gguf`: a random-weight `DeepseekV3ForCausalLM` (4
/// layers, hidden 64, 4 heads, Q-LoRA rank 24, kv_lora 32, qk nope/rope 16/8,
/// v head 24, one dense lead layer, 8 routed experts with 2 used plus one shared
/// expert, sigmoid gating with a nonzero `exp_probs_b`, `expert_weights_norm`,
/// scale 2.5, the Qwen2.5 tokenizer), converted by llama.cpp `79e2e74`'s
/// unmodified `convert_hf_to_gguf.py --outtype f32`. Source, the llama.cpp
/// reference runs and the sensitivity check (each new feature, dropped, changes
/// llama.cpp's greedy output on all 7 prompts): `test-data/tiny-mla-v3-src.tar.gz`.
const FIXTURE: &str = "tiny-mla-v3.gguf";

#[test]
fn tiny_mla_v3_fixture_has_the_features_it_was_built_for() {
    let Some(file) = header(FIXTURE) else {
        return;
    };
    let (cfg, blocks, lead) = parse(&file);
    assert_eq!((blocks, lead), (4, 1));
    assert_eq!(cfg.q_lora_rank, Some(24));
    assert_eq!(
        (cfg.kv_lora_rank, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim),
        (32, 16, 8)
    );
    let m = moe(&cfg);
    assert_eq!(m.gating, MlaGating::Sigmoid);
    assert!(m.has_exp_probs_b);
    assert_eq!(m.expert_used_count, 2);
    assert_eq!(m.routed_scaling_factor, 2.5);
    assert!(m.normalize_top_k);
    assert!(cfg.yarn.is_none());
    let v_b = file.tensor_info("blk.0.attn_v_b.weight").unwrap();
    assert_eq!(v_b.shape[1], 24, "v_head_dim != qk_nope_head_dim");
}

/// llama.cpp `79e2e74` greedy ids for [`prompts`] on the fixture, 20 tokens,
/// f32 weights and KV cache (`ref.cpp` in the archive, CPU backend, EOS-only
/// stopping like Reflex).
const FIXTURE_GOLDEN: [&[u32]; 7] = [
    &[
        150509, 34215, 101849, 141107, 145837, 143148, 60012, 32803, 38898, 136060, 66313, 148273,
        1168, 2682, 107291, 148951, 20377, 57922, 55981, 145479,
    ],
    &[
        58726, 138723, 117248, 133788, 3535, 90706, 115159, 150060, 83094, 142644, 113533, 89839,
        141960, 89147, 49629, 94481, 67841, 144374, 77253, 113831,
    ],
    &[
        35376, 35535, 120234, 8051, 62153, 46602, 27677, 17676, 21723, 27646, 82326, 63505, 144177,
        111186, 147625, 37097, 16062, 107548, 115589, 33967,
    ],
    &[
        137202, 68618, 82700, 15654, 118211, 51439, 149540, 143894, 69938, 13435, 114348, 34194,
        2668, 13841, 144223, 35474, 2850, 114351, 64144, 20530,
    ],
    &[
        46145, 149069, 32228, 47487, 35840, 2767, 91756, 127058, 40776, 136975, 111620, 68574,
        6738, 80212, 140132, 19814, 8096, 99367, 25639, 44752,
    ],
    &[
        61932, 61312, 70580, 24325, 67420, 137331, 21483, 136060, 80021, 131678, 23473, 77540,
        126286, 112709, 17354, 73610, 56358, 8900, 45690, 60669,
    ],
    &[
        114608, 33974, 39399, 21996, 6335, 34616, 66545, 70721, 89308, 50160, 78506, 42135, 97414,
        50997, 121425, 19187, 61797, 13289, 43219, 137386,
    ],
];

fn load(weights: WeightsDtype) -> Option<Model> {
    let file = header(FIXTURE)?;
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    let model = Model::load_with_options(device, &file, &opts).expect("load tiny-mla-v3");
    assert!(model.mla.is_some(), "tiny-mla-v3 loads as MLA");
    Some(model)
}

fn greedy(model: &Model, prompt: &str, n: usize) -> Vec<u32> {
    model
        .generate(
            prompt,
            n,
            None,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("generate")
        .0
}

/// Token-for-token greedy agreement with llama.cpp, f32 weights: Q-LoRA,
/// sigmoid routing with the selection bias, normalization and scaling all
/// together.
#[test]
#[ignore]
fn tiny_mla_v3_fixture_matches_llamacpp() {
    let Some(model) = load(WeightsDtype::F32) else {
        return;
    };
    let mut failures = Vec::new();
    for ((name, prompt), want) in prompts().iter().zip(FIXTURE_GOLDEN) {
        let got = greedy(&model, prompt, want.len());
        let agree = got.iter().zip(want).take_while(|(a, b)| a == b).count();
        eprintln!("{name}: {agree}/{} agree", want.len());
        if got != want {
            failures.push(format!(
                "{name}\n  reflex:    {got:?}\n  llama.cpp: {want:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "mismatches:\n{}", failures.join("\n"));
}

/// Batched prefill (GEMM Q-LoRA, grouped-GEMM sigmoid-routed experts) ==
/// sequential prefill (GEMV), in both weight dtypes, on every prompt.
#[test]
#[ignore]
fn tiny_mla_v3_batched_prefill_matches_sequential() {
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let Some(model) = load(dtype) else {
            return;
        };
        let m = model.mla.as_ref().unwrap();
        let hidden_size = m.cfg.hidden_size;
        let eps = m.cfg.rmsnorm_eps;
        for (name, prompt) in prompts() {
            let (seq_ids, seq_hidden, _, seq_pos) =
                model.prefill_mla(m, &prompt, None, 0).expect("prefill_mla");
            let (batch_ids, batch_hidden, _, batch_pos) = model
                .prefill_mla_batched(m, &prompt, None, 0)
                .expect("prefill_mla_batched");
            assert_eq!(seq_ids, batch_ids, "{name}");
            assert_eq!(seq_pos, batch_pos, "{name}");
            let last = model
                .last_row(&batch_hidden, batch_ids.len(), hidden_size)
                .expect("last_row");
            let a = model.device.dtoh_sync_copy(&seq_hidden).unwrap();
            let b = model.device.dtoh_sync_copy(&last).unwrap();
            assert_prefill_hidden_close(&a, &b, model.weights_dtype(), name);
            assert_eq!(
                model.lm_head_argmax(&seq_hidden, hidden_size, eps).unwrap(),
                model.lm_head_argmax(&last, hidden_size, eps).unwrap(),
                "{name} ({dtype:?}): greedy next token"
            );
        }
    }
}

/// f16 weight storage stays close to f32: first-step logits within a relative
/// L2 error, and the same greedy first token wherever f32's top two are clearly
/// apart.
#[test]
#[ignore]
fn tiny_mla_v3_f16_weights_track_f32() {
    const REL_L2_TOL: f64 = 1e-2;
    const CLEAR_MARGIN: f32 = 0.5;
    let (Some(f32_model), Some(f16_model)) = (load(WeightsDtype::F32), load(WeightsDtype::F16))
    else {
        return;
    };
    let first = |model: &Model, prompt: &str| {
        let mut logits = Vec::new();
        model
            .generate(
                prompt,
                1,
                None,
                &SamplingParams::default(),
                |l| logits = l.to_vec(),
                |_, _| {},
            )
            .expect("generate");
        logits
    };
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap();
    let mut clear = 0;
    for (name, prompt) in prompts() {
        let a = first(&f32_model, &prompt);
        let b = first(&f16_model, &prompt);
        let diff: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum();
        let norm: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
        let rel_l2 = (diff / norm).sqrt();
        let best = argmax(&a);
        let second = a
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != best)
            .fold(f32::MIN, |m, (_, &v)| m.max(v));
        let margin = a[best] - second;
        eprintln!("{name}: rel_l2={rel_l2:e} f32 top-2 margin={margin}");
        assert!(rel_l2 < REL_L2_TOL, "{name}: rel_l2 {rel_l2:e}");
        if margin > CLEAR_MARGIN {
            clear += 1;
            assert_eq!(argmax(&b), best, "{name}: f16 changed a clear greedy pick");
        }
    }
    assert!(clear >= 4, "too few clear picks ({clear}) to test anything");
}

/// `--export-kv` then `--import-kv`: resuming from the exported compressed
/// cache gives the same greedy ids as one uninterrupted run over the joined
/// text.
#[test]
#[ignore]
fn tiny_mla_v3_kv_export_import_resumes_exactly() {
    let Some(model) = load(WeightsDtype::F32) else {
        return;
    };
    let (_, cache) = model
        .forward_prompt_capture_kv_mla("The capital of France is")
        .expect("capture kv");
    let imported = crate::kv_io::ImportedKv::Mla(cache);
    let (resumed, _) = model
        .generate(
            " Paris",
            10,
            Some(&imported),
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("resume");
    assert_eq!(
        resumed,
        greedy(&model, "The capital of France is Paris", 10)
    );
}

/// `system1` scores (batched prefill, then gathered LM-head rows per candidate
/// step) equal the sum of the full-vocab logits along each candidate, computed
/// step by step with the sequential path.
#[test]
#[ignore]
fn tiny_mla_v3_system1_scores_match_full_logits() {
    let Some(model) = load(WeightsDtype::F32) else {
        return;
    };
    let m = model.mla.as_ref().unwrap();
    let prompt = "The capital of France is";
    let candidates: Vec<System1Candidate> = [" Paris", " London", " the city of Paris"]
        .iter()
        .map(|t| System1Candidate {
            text: t.to_string(),
        })
        .collect();
    let response = model
        .system1_evaluate(prompt, &candidates, 1.0)
        .expect("system1");
    assert!(response.results.iter().any(|r| r.token_ids.len() > 1));
    let hidden_size = m.cfg.hidden_size;
    let eps = m.cfg.rmsnorm_eps;
    for result in &response.results {
        let (_, mut h, mut caches, mut position) = model
            .prefill_mla(m, prompt, None, result.token_ids.len())
            .expect("prefill");
        let mut want = 0.0f32;
        for (step, &id) in result.token_ids.iter().enumerate() {
            if step > 0 {
                h = model
                    .forward_one_token_mla(m, result.token_ids[step - 1], position, &mut caches)
                    .expect("step");
                position += 1;
            }
            want += model.lm_head_logits(&h, hidden_size, eps).expect("logits")[id as usize];
        }
        let tol = 1e-3 * (1.0 + want.abs());
        eprintln!(
            "{:?}: tokens {:?} score {} full-logit sum {want}",
            result.text, result.token_ids, result.score
        );
        assert!(
            (result.score - want).abs() <= tol,
            "{:?}: system1 score {} vs full-logit sum {want}",
            result.text,
            result.score
        );
    }
}
