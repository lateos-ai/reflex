//! `qwen2` (Qwen2 / Qwen2.5) tests: config parsing and rejections on synthetic
//! headers and the real Qwen2.5 headers (host-only), then the synthetic
//! fixture on a GPU against llama.cpp. Design and sources:
//! docs/design/qwen2.md.

use super::config::{parse_model_config, qkv_bias_names};
use super::kolibri_config_tests::{Header, Val};
use super::prefill_batching_tests::assert_prefill_dense_batched_matches_sequential;
use super::quant_resident_tests::assert_quant_resident_generate_matches_f32;
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// `test-data/tiny-qwen2.gguf`: a random-weight `Qwen2ForCausalLM` (2 layers,
/// hidden 64, 4 query / 2 KV heads of dim 16, FFN 128, tied embeddings,
/// nonzero Q/K/V biases, the real Qwen2.5-0.5B-Instruct tokenizer) converted
/// by llama.cpp `79e2e74`'s unmodified `convert_hf_to_gguf.py --outtype f32`.
/// Source and the llama.cpp reference runs: `test-data/tiny-qwen2-src.tar.gz`.
const FIXTURE: &str = "test-data/tiny-qwen2.gguf";

/// `test-data/tiny-qwen2-lora.gguf`: a random PEFT LoRA (rank 4, alpha 8) on
/// every layer's `q_proj`/`v_proj`/`down_proj` of the fixture, converted by
/// llama.cpp `79e2e74`'s `convert_lora_to_gguf.py`. Same archive.
const LORA_FIXTURE: &str = "test-data/tiny-qwen2-lora.gguf";

fn fixture() -> Option<GgufFile> {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("Skipping: {FIXTURE} not present (see docs/DEVELOPMENT.md's test fixtures)");
        return None;
    }
    Some(GgufFile::open(FIXTURE).expect("open tiny-qwen2 fixture"))
}

// ---------------------------------------------------------------------------
// Host-only
// ---------------------------------------------------------------------------

/// A minimal valid `qwen2` header: one layer, hidden 64, 4 query / 2 KV heads
/// (head dim 16), with Q/K/V biases.
fn qwen2_header() -> Header {
    let k = |s: &str| format!("qwen2.{s}");
    Header {
        kv: vec![
            ("general.architecture".to_string(), Val::Str("qwen2")),
            (k("block_count"), Val::U32(1)),
            (k("embedding_length"), Val::U32(64)),
            (k("feed_forward_length"), Val::U32(128)),
            (k("attention.head_count"), Val::U32(4)),
            (k("attention.head_count_kv"), Val::U32(2)),
            (k("rope.freq_base"), Val::F32(1_000_000.0)),
            (k("attention.layer_norm_rms_epsilon"), Val::F32(1e-6)),
        ],
        tensors: vec![
            ("blk.0.ffn_gate.weight".to_string(), vec![64, 128]),
            ("blk.0.attn_q.bias".to_string(), vec![64]),
            ("blk.0.attn_k.bias".to_string(), vec![32]),
            ("blk.0.attn_v.bias".to_string(), vec![32]),
        ],
    }
}

fn err_text<T>(r: Result<T, ReflexError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn qwen2_header_parses_with_neox_rope_and_derived_head_dim() {
    let file = qwen2_header().open();
    let (cfg, blocks, moe) = parse_model_config(&file).expect("qwen2 accepted");
    assert_eq!(blocks, 1);
    assert!(moe.is_none());
    assert_eq!(cfg.head_dim, 16, "no key_length: hidden / heads");
    assert_eq!(cfg.rotary_dim, 16, "full rotary");
    assert_eq!(cfg.rope_type, RopeType::Neox);
    assert_eq!(cfg.rope_base, 1_000_000.0);
    let names = qkv_bias_names(&file, 0, &cfg).expect("biases valid");
    assert_eq!(
        names,
        Some(
            [
                "blk.0.attn_q.bias",
                "blk.0.attn_k.bias",
                "blk.0.attn_v.bias"
            ]
            .map(String::from)
        )
    );
}

#[test]
fn qwen2_without_biases_has_none() {
    let mut h = qwen2_header();
    h.tensors.retain(|(n, _)| !n.ends_with(".bias"));
    let file = h.open();
    let (cfg, _, _) = parse_model_config(&file).expect("qwen2 accepted");
    assert_eq!(qkv_bias_names(&file, 0, &cfg).expect("ok"), None);
}

#[test]
fn qwen2_partial_or_misshapen_bias_is_rejected() {
    let mut h = qwen2_header();
    h.tensors.retain(|(n, _)| n != "blk.0.attn_v.bias");
    let file = h.open();
    let (cfg, _, _) = parse_model_config(&file).expect("config itself is fine");
    let e = err_text(qkv_bias_names(&file, 0, &cfg));
    assert!(e.contains("blk.0.attn_v.bias"), "{e}");

    let mut h = qwen2_header();
    for t in &mut h.tensors {
        if t.0 == "blk.0.attn_k.bias" {
            t.1 = vec![64];
        }
    }
    let file = h.open();
    let (cfg, _, _) = parse_model_config(&file).expect("config itself is fine");
    let e = err_text(qkv_bias_names(&file, 0, &cfg));
    assert!(e.contains("blk.0.attn_k.bias") && e.contains("[32]"), "{e}");
}

/// Tensors llama.cpp would apply but this path doesn't: rejected by name, on
/// any dense architecture, instead of running without them.
#[test]
fn unimplemented_dense_tensors_are_rejected() {
    for (arch, name) in [
        ("qwen2", "blk.0.attn_output.bias"),
        ("qwen2", "blk.0.attn_qkv.weight"),
        ("qwen2", "blk.0.attn_qkv.bias"),
        ("qwen2", "output.bias"),
        ("qwen3", "blk.0.attn_output.bias"),
    ] {
        let mut h = qwen2_header();
        if arch != "qwen2" {
            h.kv =
                h.kv.into_iter()
                    .map(|(k, v)| (k.replace("qwen2.", &format!("{arch}.")), v))
                    .collect();
            h.set("general.architecture", Val::Str("qwen3"));
        }
        h.tensors.push((name.to_string(), vec![64]));
        let e = err_text(parse_model_config(&h.open()));
        assert!(e.contains(name), "{arch}/{name}: {e}");
    }
}

#[test]
fn qwen2_rope_scaling_and_other_qwen2_family_archs_are_rejected() {
    let mut h = qwen2_header();
    h.set("qwen2.rope.scaling.type", Val::Str("yarn"));
    let e = err_text(parse_model_config(&h.open()));
    assert!(e.contains("rope.scaling.type"), "{e}");

    let mut h = qwen2_header();
    h.set("general.architecture", Val::Str("qwen2vl"));
    let e = err_text(parse_model_config(&h.open()));
    assert!(e.contains("qwen2vl"), "{e}");
}

/// `qwen2moe` reports a nonzero `expert_count`, so it used to pass the generic
/// MoE check and would have run without its shared expert.
#[test]
fn qwen2moe_shared_expert_is_rejected() {
    let k = |s: &str| format!("qwen2moe.{s}");
    let h = Header {
        kv: vec![
            ("general.architecture".to_string(), Val::Str("qwen2moe")),
            (k("block_count"), Val::U32(1)),
            (k("embedding_length"), Val::U32(64)),
            (k("attention.head_count"), Val::U32(4)),
            (k("attention.head_count_kv"), Val::U32(2)),
            (k("expert_count"), Val::U32(4)),
            (k("expert_used_count"), Val::U32(2)),
        ],
        tensors: vec![
            ("blk.0.ffn_gate_exps.weight".to_string(), vec![64, 32, 4]),
            ("blk.0.ffn_gate_inp_shexp.weight".to_string(), vec![64]),
            ("blk.0.ffn_gate_shexp.weight".to_string(), vec![64, 128]),
        ],
    };
    let e = err_text(parse_model_config(&h.open()));
    assert!(e.contains("shared expert"), "{e}");
}

/// The synthetic fixture parses like a real Qwen2.5 file: biases on every
/// layer, tied LM head.
#[test]
fn qwen2_fixture_config() {
    let Some(file) = fixture() else { return };
    let (cfg, blocks, moe) = parse_model_config(&file).expect("parse fixture");
    assert!(moe.is_none());
    assert_eq!(blocks, 2);
    assert_eq!(
        (
            cfg.hidden_size,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim
        ),
        (64, 4, 2, 16)
    );
    assert_eq!(cfg.ffn_hidden_size, 128);
    assert_eq!(cfg.rope_type, RopeType::Neox);
    for i in 0..blocks {
        assert!(qkv_bias_names(&file, i, &cfg).expect("biases").is_some());
    }
    assert!(file.tensor_info("output.weight").is_none(), "tied LM head");
}

/// The real Qwen2.5 headers from the survey (docs/design/qwen2.md): metadata
/// and tensor table, no tensor data (`scripts/gguf_tokenizer_only.py
/// --keep-tensor-table` on the first 40 MB of each file). Each skips if
/// absent.
#[test]
fn qwen2_real_headers_config() {
    // (file, layers, hidden, heads, kv heads, head dim, ffn, rope base, tied)
    let cases = [
        (
            "qwen2.5-0.5b-instruct-header.gguf",
            24,
            896,
            14,
            2,
            64,
            4864,
            1e6,
            false,
        ),
        (
            "qwen2.5-1.5b-instruct-header.gguf",
            28,
            1536,
            12,
            2,
            128,
            8960,
            1e6,
            false,
        ),
        (
            "qwen2.5-math-1.5b-instruct-header.gguf",
            28,
            1536,
            12,
            2,
            128,
            8960,
            1e4,
            true,
        ),
        (
            "qwen2.5-coder-1.5b-instruct-header.gguf",
            28,
            1536,
            12,
            2,
            128,
            8960,
            1e6,
            false,
        ),
        (
            "qwen2.5-coder-1.5b-instruct-q8_0-header.gguf",
            28,
            1536,
            12,
            2,
            128,
            8960,
            1e6,
            true,
        ),
    ];
    for (name, layers, hidden, heads, kv, hd, ffn, base, tied) in cases {
        let path = std::path::Path::new("test-data").join(name);
        if !path.exists() {
            eprintln!("Skipping {name}: not present");
            continue;
        }
        let file = GgufFile::open(&path).expect("open header");
        let (cfg, blocks, moe) =
            parse_model_config(&file).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(moe.is_none(), "{name}");
        assert_eq!(blocks, layers, "{name}");
        assert_eq!(
            (
                cfg.hidden_size,
                cfg.num_q_heads,
                cfg.num_kv_heads,
                cfg.head_dim
            ),
            (hidden, heads, kv, hd),
            "{name}"
        );
        assert_eq!(cfg.rotary_dim, hd, "{name}");
        assert_eq!(cfg.ffn_hidden_size, ffn, "{name}");
        assert_eq!(cfg.rope_base, base, "{name}");
        assert_eq!(cfg.rope_type, RopeType::Neox, "{name}");
        assert_eq!(cfg.rmsnorm_eps, 1e-6, "{name}");
        for i in 0..blocks {
            assert!(
                qkv_bias_names(&file, i, &cfg)
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                    .is_some(),
                "{name}: layer {i} has no Q/K/V biases"
            );
            assert!(
                file.tensor_info(&format!("blk.{i}.attn_q_norm.weight"))
                    .is_none(),
                "{name}: unexpected QK-Norm"
            );
        }
        assert_eq!(file.tensor_info("output.weight").is_none(), tied, "{name}");
    }
}

// ---------------------------------------------------------------------------
// GPU (`#[ignore]`; `scripts/gpu_nightly_tests.sh` maps each to its file)
// ---------------------------------------------------------------------------

/// The comparison prompts (`prompts.py` in the fixture archive): plain text,
/// ChatML, a Qwen2.5-Math chain-of-thought prompt, Coder FIM, Arabic with
/// harakat, and a 980-token prompt.
pub(super) fn prompts() -> Vec<(&'static str, String)> {
    let long = "The quick brown fox jumps over the lazy dog while the cat watches. ".repeat(70);
    vec![
        ("plain", "The capital of France is".to_string()),
        ("story", "Once upon a time".to_string()),
        ("chat", "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n".to_string()),
        ("math", "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nFind the value of $x$ such that $3x + 7 = 22$.<|im_end|>\n<|im_start|>assistant\n".to_string()),
        ("fim", "<|fim_prefix|>def fibonacci(n):\n    if n < 2:\n        return n\n<|fim_suffix|>\n\nprint(fibonacci(10))\n<|fim_middle|>".to_string()),
        ("arabic", "\u{628}\u{650}\u{633}\u{652}\u{645}\u{650} \u{627}\u{644}\u{644}\u{651}\u{64e}\u{647}\u{650} \u{627}\u{644}\u{631}\u{651}\u{64e}\u{62d}\u{652}\u{645}\u{64e}\u{670}\u{646}\u{650} \u{627}\u{644}\u{631}\u{651}\u{64e}\u{62d}\u{650}\u{64a}\u{645}\u{650}".to_string()),
        ("long", long),
    ]
}

/// llama.cpp `79e2e74` greedy ids for [`prompts`] on the fixture, 20 tokens,
/// f32 weights and KV cache (`ref.cpp` in the archive, EOS-only stopping like
/// Reflex). Recorded with llama.cpp's CUDA build on an RTX 3090; its CPU backend gives the same ids.
const FIXTURE_GOLDEN: [&[u32]; 7] = [
    &[
        31006, 102672, 42659, 41759, 126425, 140910, 145599, 83756, 92974, 725, 62523, 149752,
        75365, 80956, 26138, 10272, 103364, 94526, 84564, 76987,
    ],
    &[
        111334, 100383, 46662, 91624, 69942, 15634, 91428, 74667, 108353, 61223, 72902, 134741,
        36937, 63557, 90433, 102608, 76377, 116185, 22531, 73043,
    ],
    &[
        48255, 15691, 4083, 72789, 115766, 109702, 67203, 148621, 74667, 57092, 77694, 3092, 55756,
        10035, 135561, 111430, 106175, 29129, 65555, 32035,
    ],
    &[
        91450, 65256, 60495, 81913, 138549, 50131, 108994, 47215, 103793, 73161, 13237, 136466,
        122692, 21619, 146468, 39197, 38241, 112407, 31503, 7046,
    ],
    &[
        125770, 1672, 71292, 61706, 59932, 109212, 30204, 82586, 107506, 145938, 116482, 62802,
        58896, 11744, 110245, 108963, 137142, 21510, 125522, 47887,
    ],
    &[
        62201, 81677, 95485, 2786, 19211, 1920, 78635, 129842, 21070, 62103, 110580, 81913, 60242,
        95905, 40846, 123294, 62297, 128277, 66611, 92548,
    ],
    &[
        77817, 102450, 56653, 81607, 64795, 68337, 13360, 39896, 8918, 6441, 138203, 109218, 23839,
        128004, 17780, 38129, 36178, 133023, 150848, 50574,
    ],
];

/// The same with the LoRA fixture applied (llama.cpp's own LoRA path,
/// `REF_LORA`), from its CUDA build. Its CPU backend picks a different token
/// at step 8 of `story` (a near-tie); HF `transformers` with the LoRA merged
/// by hand (`hf_ref.py` in the archive) agrees with the CUDA ids there and on
/// every prompt but `arabic`, where it already differs without the LoRA.
const FIXTURE_LORA_GOLDEN: [&[u32]; 7] = [
    &[
        23200, 80708, 20487, 113586, 35241, 113910, 6746, 81233, 8018, 70434, 68513, 66003, 81776,
        47879, 102731, 28733, 141598, 55080, 28248, 31246,
    ],
    &[
        79584, 33347, 125342, 148479, 108994, 149914, 30831, 63111, 21042, 108994, 108471, 39423,
        91450, 91450, 91450, 102723, 112305, 131433, 80738, 39622,
    ],
    &[
        128497, 31310, 104564, 52999, 111755, 44311, 144854, 58648, 38624, 24220, 58048, 68700,
        37654, 80001, 93560, 141794, 132719, 131911, 91719, 42740,
    ],
    &[
        13358, 77669, 101228, 101228, 59718, 47981, 141022, 101365, 101389, 113252, 16399, 40742,
        134079, 56087, 149636, 97334, 48235, 7536, 63394, 28637,
    ],
    &[
        118215, 28729, 92694, 9861, 14940, 129965, 44757, 79648, 99650, 39730, 101941, 11094,
        74667, 50451, 14221, 62782, 37654, 134942, 32522, 114386,
    ],
    &[
        123314, 39398, 129309, 121930, 13912, 39494, 149160, 146240, 74492, 34795, 109400, 72649,
        86682, 7787, 77909, 109554, 53926, 26573, 148042, 16389,
    ],
    &[
        88075, 6037, 137933, 60170, 418, 119877, 140426, 22695, 71358, 103590, 143874, 8728, 8728,
        10507, 127795, 88075, 6037, 15258, 88881, 47120,
    ],
];

fn load(file: &GgufFile, weights: WeightsDtype, lora: Option<&str>) -> Model {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: lora.map(std::path::PathBuf::from),
    };
    let mut model = Model::load_with_options(device, file, &opts).expect("load fixture");
    if let Some(path) = lora {
        let n = model
            .apply_lora(std::path::Path::new(path))
            .expect("apply LoRA");
        assert_eq!(n, 6, "2 layers x q_proj/v_proj/down_proj");
    }
    model
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

fn assert_matches_golden(model: &Model, golden: &[&[u32]; 7], label: &str) {
    let mut failures = Vec::new();
    for ((name, prompt), want) in prompts().iter().zip(golden) {
        let got = greedy(model, prompt, want.len());
        let agree = got.iter().zip(*want).take_while(|(a, b)| a == b).count();
        eprintln!("{label} {name}: {agree}/{} agree", want.len());
        if got != *want {
            failures.push(format!(
                "{name}\n  reflex:    {got:?}\n  llama.cpp: {want:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{label} mismatches:\n{}",
        failures.join("\n")
    );
}

/// Loads and generates end to end (a smoke test independent of the golden
/// ids).
#[test]
#[ignore]
fn qwen2_fixture_generates_without_error() {
    let Some(file) = fixture() else { return };
    let model = load(&file, WeightsDtype::F32, None);
    assert!(
        model
            .layers
            .iter()
            .all(|l| matches!(l, LayerWeights::Dense(d) if d.attn_bias.is_some())),
        "every fixture layer loads its Q/K/V biases"
    );
    assert!(!greedy(&model, "Once upon a time", 5).is_empty());
}

/// The workspace decode step (`decode_step.rs`, Q/K/V biases added in place in
/// the workspace and the K/V cache rows) gives the allocating path's logits, bit
/// for bit, with f32 and f16 weights.
#[test]
#[ignore]
fn qwen2_fixture_decode_workspace_matches_allocating() {
    let Some(file) = fixture() else { return };
    for weights in [WeightsDtype::F32, WeightsDtype::F16] {
        let model = load(&file, weights, None);
        super::decode_step_tests::assert_workspace_step_matches_allocating(
            &model,
            "Once upon a time",
            10,
        );
    }
}

/// Token-for-token greedy agreement with llama.cpp, f32 weights.
#[test]
#[ignore]
fn qwen2_fixture_matches_llamacpp() {
    let Some(file) = fixture() else { return };
    let model = load(&file, WeightsDtype::F32, None);
    assert_matches_golden(&model, &FIXTURE_GOLDEN, "f32");
}

/// `--lora` on `qwen2`: the adapter merges into `attn_q`/`attn_v`/`ffn_down`
/// and generation matches llama.cpp running the same adapter.
#[test]
#[ignore]
fn qwen2_fixture_lora_matches_llamacpp() {
    let Some(file) = fixture() else { return };
    if !std::path::Path::new(LORA_FIXTURE).exists() {
        eprintln!("Skipping: {LORA_FIXTURE} not present");
        return;
    }
    let model = load(&file, WeightsDtype::F32, Some(LORA_FIXTURE));
    assert_matches_golden(&model, &FIXTURE_LORA_GOLDEN, "f32+lora");
}

/// Batched prefill (GEMM, row-broadcast bias add) == sequential prefill
/// (GEMV, plain bias add), in both weight dtypes.
#[test]
#[ignore]
fn qwen2_fixture_batched_prefill_matches_sequential() {
    let Some(file) = fixture() else { return };
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let model = load(&file, dtype, None);
        for (_, prompt) in prompts() {
            assert_prefill_dense_batched_matches_sequential(&model, &prompt);
        }
    }
}

/// f16 weight storage stays close to f32: first-step logits within a
/// relative L2 error, and the same greedy first token wherever f32's own top
/// two are clearly apart (the fixture's closest first-step pair is 0.09 apart).
#[test]
#[ignore]
fn qwen2_fixture_f16_weights_track_f32() {
    const REL_L2_TOL: f64 = 1e-2;
    const CLEAR_MARGIN: f32 = 0.5;
    let Some(file) = fixture() else { return };
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
    let f32_model = load(&file, WeightsDtype::F32, None);
    let f16_model = load(&file, WeightsDtype::F16, None);
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

/// `--export-kv` then `--import-kv`: resuming from the exported cache gives
/// the same greedy ids as one uninterrupted run over the joined text (byte
/// exact: the gpt2 tokenizer splits "... is" + " Paris" the same as "... is
/// Paris").
#[test]
#[ignore]
fn qwen2_fixture_kv_export_import_resumes_exactly() {
    let Some(file) = fixture() else { return };
    let model = load(&file, WeightsDtype::F32, None);
    let (_, cache) = model
        .forward_prompt_capture_kv("The capital of France is")
        .expect("capture kv");
    let imported = crate::kv_io::ImportedKv::Dense(cache);
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

/// A real Qwen2.5 Q4_K_M file (`REFLEX_TEST_GGUF`, e.g.
/// qwen2.5-1.5b-instruct-q4_k_m.gguf; the 0.5B has no Q4_K tensors):
/// quantized-resident weights give the f32 path's greedy ids, and batched
/// prefill matches sequential. Run single-threaded.
#[test]
#[ignore]
fn qwen2_real_model_quant_resident_and_batched_prefill() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real Qwen2.5 Q4_K_M GGUF");
    let file = GgufFile::open(&path).expect("open REFLEX_TEST_GGUF");
    assert_eq!(
        file.metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str),
        Some("qwen2")
    );
    let chat = "<|im_start|>user\nWhat is 12 * 7?<|im_end|>\n<|im_start|>assistant\n";
    assert_quant_resident_generate_matches_f32(&file, chat);
    let model = load(&file, WeightsDtype::F32, None);
    assert_prefill_dense_batched_matches_sequential(&model, chat);
}
