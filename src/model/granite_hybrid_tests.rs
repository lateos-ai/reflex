//! `granitehybrid` (Granite 4.0-H) tests: config parsing and rejections on the
//! real Granite 4.0-H headers and synthetic ones (host-only), then the
//! synthetic fixture on a GPU against llama.cpp. Design and sources:
//! docs/design/mamba2-hybrid.md.

use super::config::{parse_granite_hybrid_config, parse_model_config, Mamba2Config};
use super::kolibri_config_tests::{Header, Val};
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// `test-data/tiny-granitehybrid.gguf`: a random-weight
/// `GraniteMoeHybridForCausalLM` (6 layers `[ssm, ssm, attn, ssm, ssm, attn]`,
/// hidden 128, 8 SSM heads of 32, d_state 128, 2 groups, attention 4 / 2 KV
/// heads of 32 without RoPE, FFN 256, tied embeddings, non-default muP scales,
/// the real Granite 4.0-H 350M tokenizer) converted by llama.cpp `79e2e74`'s
/// unmodified `convert_hf_to_gguf.py --outtype f32`. Source, prompts, the
/// llama.cpp references and the sensitivity check:
/// `test-data/tiny-granitehybrid-src.tar.gz`.
const FIXTURE: &str = "test-data/tiny-granitehybrid.gguf";

fn fixture() -> Option<GgufFile> {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("Skipping: {FIXTURE} not present (see docs/DEVELOPMENT.md's test fixtures)");
        return None;
    }
    Some(GgufFile::open(FIXTURE).expect("open tiny-granitehybrid fixture"))
}

/// A saved real header (`scripts/gguf_tokenizer_only.py --keep-tensor-table`
/// on the first 40 MB of the file), or `None` when absent.
fn header(name: &str) -> Option<GgufFile> {
    let path = format!("test-data/{name}");
    if !std::path::Path::new(&path).exists() {
        eprintln!("Skipping {name}: not present");
        return None;
    }
    Some(GgufFile::open(&path).expect("open header"))
}

fn err_text<T>(r: Result<T, ReflexError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Host-only
// ---------------------------------------------------------------------------

/// `ibm-granite/granite-4.0-h-350m-GGUF`'s `granite-4.0-h-350m-Q8_0.gguf`.
#[test]
fn granite_350m_header_parses() {
    let Some(file) = header("granite-4.0-h-350m-header.gguf") else {
        return;
    };
    let cfg = parse_granite_hybrid_config(&file).expect("350M accepted");
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.ssm_layers.len(), 32);
    let attn: Vec<usize> = (0..32).filter(|&i| !cfg.ssm_layers[i]).collect();
    assert_eq!(attn, vec![10, 13, 17, 27]);
    assert_eq!(
        (cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim),
        (12, 4, 64)
    );
    assert_eq!(cfg.ffn_hidden_size, 2048);
    assert_eq!(cfg.embedding_scale, 12.0);
    assert_eq!(cfg.residual_scale, 0.246);
    assert_eq!(cfg.attention_scale, 0.015625);
    assert_eq!(cfg.logit_scale, Some(3.0));
    assert_eq!(
        cfg.ssm,
        Mamba2Config {
            d_inner: 1536,
            d_state: 128,
            n_group: 1,
            n_head: 48,
            d_conv: 4,
        }
    );
    assert_eq!(cfg.ssm.head_dim(), 32);
    assert_eq!(cfg.ssm.conv_dim(), 1792);
    assert_eq!(cfg.ssm.d_in_proj(), 3376);
    // The tensor table agrees with the derived sizes.
    let shape = |n: &str| file.tensor_info(n).expect(n).shape.clone();
    assert_eq!(shape("blk.0.ssm_in.weight"), vec![768, 3376]);
    assert_eq!(shape("blk.0.ssm_conv1d.weight"), vec![4, 1792]);
    assert_eq!(shape("blk.10.attn_k.weight"), vec![768, 256]);
}

/// Granite 4.0-H 1B and Micro: the same graph, other sizes.
#[test]
fn granite_1b_and_micro_headers_parse() {
    for (name, hidden, d_inner, heads, head_dim, logit) in [
        ("granite-4.0-h-1b-header.gguf", 1536, 3072, 48, 128, 6.0),
        ("granite-4.0-h-micro-header.gguf", 2048, 4096, 64, 64, 8.0),
    ] {
        let Some(file) = header(name) else {
            continue;
        };
        let cfg = parse_granite_hybrid_config(&file).expect(name);
        assert_eq!(cfg.hidden_size, hidden, "{name}");
        assert_eq!(cfg.ssm.d_inner, d_inner, "{name}");
        assert_eq!(cfg.ssm.n_head, heads, "{name}");
        assert_eq!(cfg.ssm.head_dim(), 64, "{name}");
        assert_eq!(cfg.head_dim, head_dim, "{name}");
        assert_eq!(cfg.logit_scale, Some(logit), "{name}");
        let attn: Vec<usize> = (0..40).filter(|&i| !cfg.ssm_layers[i]).collect();
        assert_eq!(attn, vec![5, 15, 25, 35], "{name}");
    }
}

/// Granite 4.0-H Tiny is MoE (64 experts, 6 used, plus a shared expert).
#[test]
fn granite_tiny_moe_header_is_rejected_naming_expert_count() {
    let Some(file) = header("granite-4.0-h-tiny-header.gguf") else {
        return;
    };
    let e = err_text(parse_granite_hybrid_config(&file));
    assert!(e.contains("granitehybrid.expert_count = 64"), "{e}");
}

/// llama.cpp reads a missing `rope.scaling.finetuned` as true (RoPE on).
#[test]
fn granite_rope_switch_true_or_absent_is_rejected() {
    let Some(mut file) = header("granite-4.0-h-350m-header.gguf") else {
        return;
    };
    let k = "granitehybrid.rope.scaling.finetuned";
    file.metadata.insert(k.to_string(), GgufValue::Bool(true));
    let e = err_text(parse_granite_hybrid_config(&file));
    assert!(e.contains(k) && e.contains("true"), "{e}");
    file.metadata.remove(k);
    let e = err_text(parse_granite_hybrid_config(&file));
    assert!(e.contains(k) && e.contains("absent"), "{e}");
}

/// SSM shapes outside what llama.cpp asserts or the kernels cover.
#[test]
fn granite_unsupported_ssm_shapes_are_rejected() {
    if header("granite-4.0-h-350m-header.gguf").is_none() {
        return;
    }
    let cases: [(&str, u32, &str); 4] = [
        ("ssm.state_size", 48, "multiple of 32 up to 256"),
        ("ssm.state_size", 512, "multiple of 32 up to 256"),
        (
            "ssm.inner_size",
            1024,
            "must be 2 * granitehybrid.embedding_length",
        ),
        (
            "ssm.group_count",
            5,
            "granitehybrid.ssm.group_count = 5 must divide",
        ),
    ];
    for (key, value, want) in cases {
        let mut file = header("granite-4.0-h-350m-header.gguf").expect("present above");
        file.metadata
            .insert(format!("granitehybrid.{key}"), GgufValue::U32(value));
        let e = err_text(parse_granite_hybrid_config(&file));
        assert!(e.contains(want), "{key} = {value}: {e}");
    }
}

/// A minimal valid synthetic `granitehybrid` header: layer 0 SSM, layer 1
/// attention.
fn granite_header() -> Header {
    let k = |s: &str| format!("granitehybrid.{s}");
    Header {
        kv: vec![
            (
                "general.architecture".to_string(),
                Val::Str("granitehybrid"),
            ),
            (k("block_count"), Val::U32(2)),
            (k("embedding_length"), Val::U32(64)),
            (k("feed_forward_length"), Val::U32(128)),
            (k("attention.head_count"), Val::U32(4)),
            (k("attention.head_count_kv"), Val::U32s(vec![0, 2])),
            (k("rope.scaling.finetuned"), Val::Bool(false)),
            (k("ssm.conv_kernel"), Val::U32(4)),
            (k("ssm.inner_size"), Val::U32(128)),
            (k("ssm.state_size"), Val::U32(128)),
            (k("ssm.group_count"), Val::U32(1)),
            (k("ssm.time_step_rank"), Val::U32(4)),
        ],
        tensors: vec![
            ("blk.0.ssm_norm.weight".to_string(), vec![128]),
            ("blk.0.ssm_conv1d.bias".to_string(), vec![384]),
        ],
    }
}

#[test]
fn granite_synthetic_header_defaults_scales_like_llama_cpp() {
    let cfg = parse_granite_hybrid_config(&granite_header().open()).expect("accepted");
    assert_eq!(cfg.ssm_layers, vec![true, false]);
    assert_eq!(cfg.head_dim, 16);
    // Absent muP keys: no scaling, and llama.cpp's default kq_scale.
    assert_eq!(cfg.embedding_scale, 1.0);
    assert_eq!(cfg.residual_scale, 1.0);
    assert_eq!(cfg.attention_scale, 0.25);
    assert_eq!(cfg.logit_scale, None);
    // A zero value is skipped too (llama.cpp's `if (scale)`).
    let mut h = granite_header();
    h.set("granitehybrid.logit_scale", Val::F32(0.0));
    h.set("granitehybrid.attention.scale", Val::F32(0.0));
    let cfg = parse_granite_hybrid_config(&h.open()).expect("accepted");
    assert_eq!(cfg.logit_scale, None);
    assert_eq!(cfg.attention_scale, 0.25);
}

#[test]
fn granite_biases_and_incomplete_ssm_layers_are_rejected() {
    for bias in [
        "blk.1.attn_q.bias",
        "blk.1.attn_output.bias",
        "blk.0.ffn_up.bias",
    ] {
        let mut h = granite_header();
        h.tensors.push((bias.to_string(), vec![64]));
        let e = err_text(parse_granite_hybrid_config(&h.open()));
        assert!(e.contains(bias), "{bias}: {e}");
    }
    for missing in ["blk.0.ssm_norm.weight", "blk.0.ssm_conv1d.bias"] {
        let mut h = granite_header();
        h.tensors.retain(|(n, _)| n != missing);
        let e = err_text(parse_granite_hybrid_config(&h.open()));
        assert!(e.contains(missing), "{missing}: {e}");
    }
    // No zero in head_count_kv: no SSM layer at all.
    let mut h = granite_header();
    h.set(
        "granitehybrid.attention.head_count_kv",
        Val::U32s(vec![2, 2]),
    );
    let e = err_text(parse_granite_hybrid_config(&h.open()));
    assert!(e.contains("no SSM layer"), "{e}");
}

/// The other SSM architectures llama.cpp supports are rejected by name,
/// not run as a generic MoE (Jamba and Nemotron-H MoE report experts).
#[test]
fn other_ssm_architectures_are_rejected_by_name() {
    for (name, arch) in [
        ("falcon-h1-tiny-90m-header.gguf", "falcon-h1"),
        ("nemotron-3-nano-4b-header.gguf", "nemotron_h"),
        ("jamba-900m-header.gguf", "jamba"),
        ("mamba2-130m-header.gguf", "mamba2"),
    ] {
        let Some(file) = header(name) else {
            continue;
        };
        let e = err_text(parse_model_config(&file));
        assert!(e.contains(&format!("'{arch}'")), "{name}: {e}");
        assert!(e.contains("granitehybrid"), "{name}: {e}");
    }
}

#[test]
fn granite_fixture_header_parses() {
    let Some(file) = fixture() else {
        return;
    };
    let cfg = parse_granite_hybrid_config(&file).expect("fixture accepted");
    assert_eq!(cfg.ssm_layers, vec![true, true, false, true, true, false]);
    assert_eq!(
        cfg.ssm,
        Mamba2Config {
            d_inner: 256,
            d_state: 128,
            n_group: 2,
            n_head: 8,
            d_conv: 4,
        }
    );
    assert_eq!(
        (cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim),
        (4, 2, 32)
    );
    assert_eq!(cfg.attention_scale, 0.03125);
    assert_eq!(
        (cfg.embedding_scale, cfg.residual_scale, cfg.logit_scale),
        (12.0, 0.25, Some(3.0))
    );
}

// ---------------------------------------------------------------------------
// GPU (fixture)
// ---------------------------------------------------------------------------

/// The comparison prompts (`prompts.py` in the source archive): tiny-qwen2's
/// seven with Granite 4.0's chat format. "long" (981 tokens) is above
/// llama.cpp's 128-token switch to the chunked SSD scan on CUDA.
fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("plain", "The capital of France is".to_string()),
        ("story", "Once upon a time".to_string()),
        (
            "chat",
            "<|start_of_role|>system<|end_of_role|>You are a helpful assistant.<|end_of_text|>\n\
             <|start_of_role|>user<|end_of_role|>What is the capital of France? Answer in one word.<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>"
                .to_string(),
        ),
        (
            "math",
            "Find the value of $x$ such that $3x + 7 = 22$. Answer:".to_string(),
        ),
        (
            "code",
            "def fibonacci(n):\n    if n < 2:\n        return n\n".to_string(),
        ),
        (
            "arabic",
            "\u{0628}\u{0650}\u{0633}\u{0652}\u{0645}\u{0650} \u{0627}\u{0644}\u{0644}\u{0651}\u{064e}\u{0647}\u{0650} \
             \u{0627}\u{0644}\u{0631}\u{0651}\u{064e}\u{062d}\u{0652}\u{0645}\u{064e}\u{0670}\u{0646}\u{0650} \
             \u{0627}\u{0644}\u{0631}\u{0651}\u{064e}\u{062d}\u{0650}\u{064a}\u{0645}\u{0650}"
                .to_string(),
        ),
        (
            "long",
            "The quick brown fox jumps over the lazy dog while the cat watches. ".repeat(70),
        ),
    ]
}

/// Prompt token counts from llama.cpp's `PROMPT_IDS` (no BOS: Granite's
/// `add_bos_token` is false).
const FIXTURE_PROMPT_LENS: [usize; 7] = [5, 4, 31, 21, 15, 33, 981];

/// llama.cpp `79e2e74` greedy ids (CPU backend, f32 weights and KV, EOS-only
/// stopping; `ref.txt` in the source archive), 20 tokens per prompt.
const FIXTURE_GOLDEN: [[u32; 20]; 7] = [
    [
        48126, 51221, 71160, 21222, 99877, 99877, 76712, 55044, 11336, 52116, 52116, 28882, 6509,
        12366, 3090, 40411, 40411, 1561, 29105, 29105,
    ],
    [
        24412, 60309, 60309, 88447, 25552, 71394, 95970, 99315, 91604, 29233, 29233, 90928, 53344,
        51089, 58995, 31499, 54861, 13915, 41637, 41372,
    ],
    [
        16456, 6571, 98708, 99783, 5335, 5335, 5335, 5335, 88579, 22183, 55149, 55149, 55149,
        46642, 14603, 2410, 22164, 22164, 11288, 23956,
    ],
    [
        74109, 10213, 10213, 56231, 8099, 76385, 38256, 72393, 19123, 19123, 48449, 88206, 3549,
        60651, 9139, 51819, 17288, 3439, 98027, 98027,
    ],
    [
        99338, 50027, 70336, 69156, 24963, 70842, 12713, 100124, 67442, 67442, 7271, 79522, 88847,
        60029, 24886, 30702, 72531, 28338, 28338, 28338,
    ],
    [
        84518, 8590, 44984, 83632, 35912, 76984, 94940, 68650, 84408, 84408, 84408, 61949, 36900,
        66961, 66961, 66961, 49935, 88556, 61430, 98558,
    ],
    [
        44738, 92593, 47396, 25225, 25955, 40230, 93824, 79939, 26029, 26029, 26029, 26029, 26029,
        26029, 26029, 26029, 26029, 26029, 26029, 26029,
    ],
];

/// llama.cpp's first-step top-1 and top-2 `(id, logit)` per prompt (`TOP5`
/// in `ref.txt`). These logits include the `1 / logit_scale` multiply, which
/// greedy ids alone can't check.
const FIXTURE_TOP2: [[(u32, f32); 2]; 7] = [
    [(48126, 12.7056), (99502, 10.7053)],
    [(24412, 10.9696), (2016, 10.9204)],
    [(16456, 11.2426), (97009, 10.4058)],
    [(74109, 10.2924), (21553, 10.1246)],
    [(99338, 10.4752), (22373, 9.7143)],
    [(84518, 10.6733), (55663, 10.2540)],
    [(44738, 12.2488), (22899, 11.6975)],
];

fn load(file: &GgufFile, weights: WeightsDtype) -> Model {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, file, &opts).expect("load fixture")
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

#[test]
#[ignore]
fn granite_fixture_matches_llamacpp_greedy() {
    let Some(file) = fixture() else {
        return;
    };
    let model = load(&file, WeightsDtype::F32);
    assert_eq!(model.architecture_kind(), ArchitectureKind::GraniteHybrid);
    let mut agree = 0;
    for (i, (name, prompt)) in prompts().iter().enumerate() {
        let ids = model.granite_prompt_ids(prompt).expect("encode");
        assert_eq!(ids.len(), FIXTURE_PROMPT_LENS[i], "{name}: prompt ids");
        let got = greedy(&model, prompt, 20);
        eprintln!("{name}: {got:?}");
        if got == FIXTURE_GOLDEN[i] {
            agree += 1;
        } else {
            eprintln!("{name}: differs from llama.cpp {:?}", FIXTURE_GOLDEN[i]);
        }
    }
    assert_eq!(
        agree, 7,
        "greedy ids identical to llama.cpp on {agree}/7 prompts"
    );
}

/// First-step logits against llama.cpp's, which checks the logit scale and
/// the absolute values, not just the argmax.
#[test]
#[ignore]
fn granite_fixture_first_logits_match_llamacpp() {
    let Some(file) = fixture() else {
        return;
    };
    let model = load(&file, WeightsDtype::F32);
    for (i, (name, prompt)) in prompts().iter().enumerate() {
        let mut first = Vec::new();
        model
            .generate(
                prompt,
                1,
                None,
                &SamplingParams::default(),
                |l| first = l.to_vec(),
                |_, _| {},
            )
            .expect("generate");
        for &(id, want) in &FIXTURE_TOP2[i] {
            let got = first[id as usize];
            assert!(
                (got - want).abs() < 2e-3,
                "{name}: logit[{id}] = {got}, llama.cpp {want}"
            );
        }
    }
}

/// Prefill of the whole prompt at once (batched projections, the SSM
/// recurrence walking the rows inside each kernel) against one token at a
/// time through the decode path from position 0.
#[test]
#[ignore]
fn granite_fixture_batched_prefill_matches_sequential() {
    let Some(file) = fixture() else {
        return;
    };
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let model = load(&file, dtype);
        assert_granite_batched_matches_sequential(&model, &prompts());
    }
}

/// The check behind `granite_fixture_batched_prefill_matches_sequential`,
/// shared with the real-model test.
fn assert_granite_batched_matches_sequential(model: &Model, prompts: &[(&str, String)]) {
    let g = model.granite.as_ref().expect("granite model");
    let hidden_size = g.cfg.hidden_size;
    for (name, prompt) in prompts {
        let (ids, batched, _, _) = model.prefill_granite(g, prompt, 0).expect("prefill");
        let batch_last = model
            .last_row(&batched, ids.len(), hidden_size)
            .expect("last_row");
        let mut states = model
            .alloc_granite_states(g, ids.len())
            .expect("alloc states");
        let mut seq_last = None;
        for (pos, &id) in ids.iter().enumerate() {
            seq_last = Some(
                model
                    .forward_one_token_granite(g, id, pos, &mut states)
                    .expect("decode step"),
            );
        }
        let seq_last = seq_last.expect("non-empty prompt");
        let a = model.device.dtoh_sync_copy(&seq_last).expect("dtoh");
        let b = model.device.dtoh_sync_copy(&batch_last).expect("dtoh");
        super::prefill_batching_tests::assert_prefill_hidden_close(
            &a,
            &b,
            model.weights_dtype(),
            name,
        );
        let la = model.granite_logits(g, &seq_last).expect("logits");
        let lb = model.granite_logits(g, &batch_last).expect("logits");
        assert_eq!(
            Model::argmax(&la).unwrap(),
            Model::argmax(&lb).unwrap(),
            "{name}: greedy next token differs between batched and sequential prefill"
        );
    }
}

/// `--weights f16`: the first token agrees with f32 wherever llama.cpp's
/// top-1/top-2 margin is clear (all prompts but "story", 0.05).
#[test]
#[ignore]
fn granite_fixture_f16_weights_agree_on_clear_picks() {
    let Some(file) = fixture() else {
        return;
    };
    let f16 = load(&file, WeightsDtype::F16);
    for (i, (name, prompt)) in prompts().iter().enumerate() {
        let [(top1, l1), (_, l2)] = FIXTURE_TOP2[i];
        if l1 - l2 < 0.1 {
            continue;
        }
        let got = greedy(&f16, prompt, 1)[0];
        assert_eq!(got, top1, "{name}: f16 first token");
    }
}

/// System1's single-token scores equal the full logits at those ids; a
/// multi-token candidate is rejected (the SSM state can't be rewound).
#[test]
#[ignore]
fn granite_fixture_system1_scores_match_full_logits() {
    let Some(file) = fixture() else {
        return;
    };
    let model = load(&file, WeightsDtype::F32);
    let prompt = "The capital of France is";
    let candidates: Vec<System1Candidate> = [" Paris", " London", " Rome"]
        .iter()
        .map(|t| System1Candidate {
            text: t.to_string(),
        })
        .collect();
    let resp = model
        .system1_evaluate(prompt, &candidates, 1.0)
        .expect("system1");
    let mut full = Vec::new();
    model
        .generate(
            prompt,
            1,
            None,
            &SamplingParams::default(),
            |l| full = l.to_vec(),
            |_, _| {},
        )
        .expect("generate");
    for r in &resp.results {
        assert_eq!(r.token_ids.len(), 1, "{}", r.text);
        let want = full[r.token_ids[0] as usize];
        assert!(
            (r.score - want).abs() < 1e-4,
            "{}: system1 {} vs full logits {want}",
            r.text,
            r.score
        );
    }
    let multi = [System1Candidate {
        text: " Paris is lovely".to_string(),
    }];
    let e = err_text(model.system1_evaluate(prompt, &multi, 1.0));
    assert!(e.contains("single-token"), "{e}");
}

/// The rejected options on a loaded model: LoRA and KV export.
#[test]
#[ignore]
fn granite_fixture_rejects_lora_and_kv_export() {
    let Some(file) = fixture() else {
        return;
    };
    let mut model = load(&file, WeightsDtype::F32);
    let e = err_text(model.forward_prompt_capture_kv("hello"));
    assert!(e.contains("granitehybrid"), "{e}");
    let e = err_text(model.apply_lora(std::path::Path::new("test-data/tiny-qwen2-lora.gguf")));
    assert!(e.contains("granitehybrid"), "{e}");
}

// ---------------------------------------------------------------------------
// GPU (real model, REFLEX_TEST_GGUF = a Granite 4.0-H GGUF)
// ---------------------------------------------------------------------------

/// A real Granite 4.0-H file (e.g. `granite-4.0-h-350m-Q8_0.gguf`): loads,
/// generates, and its batched prefill agrees with the sequential one. The
/// greedy comparison against llama.cpp is a separate scripted run
/// (docs/design/mamba2-hybrid.md, "Results").
#[test]
#[ignore]
fn granite_real_model_batched_prefill_matches_sequential() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a Granite 4.0-H (granitehybrid) GGUF");
    let file = GgufFile::open(&path).expect("open REFLEX_TEST_GGUF");
    let model = load(&file, WeightsDtype::F32);
    let short: Vec<(&str, String)> = prompts().into_iter().take(6).collect();
    assert_granite_batched_matches_sequential(&model, &short);
    // A real instruct model may stop at EOS before 8 tokens (the 350M does
    // after 3 here, as llama.cpp does).
    let ids = greedy(&model, "The capital of France is", 8);
    assert!(!ids.is_empty() && ids.len() <= 8, "{ids:?}");
}
