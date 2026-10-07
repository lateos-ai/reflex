//! Host-only tests for [`parse_kolibri_config`]: synthetic GGUF headers (no
//! tensor data) carrying real Kolibri-1's metadata and tensor shapes, as read
//! from the header of `Kolibri-1-Q4_K_M.gguf` (docs/design/kolibri.md's
//! "Confirmed conventions").

use super::config::{parse_kolibri_config, KOLIBRI_GATING_SIGMOID_LOGIT_ADD};
use super::*;
use crate::gguf::GgufFile;
use std::io::Write;

enum Val {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
    Bools(Vec<bool>),
}

struct Header {
    kv: Vec<(String, Val)>,
    /// (name, shape); every tensor is declared F32 with no data -- the config
    /// parser only reads shapes.
    tensors: Vec<(String, Vec<u64>)>,
}

impl Header {
    fn set(&mut self, key: &str, val: Val) {
        self.kv.retain(|(k, _)| k != key);
        self.kv.push((key.to_string(), val));
    }

    fn remove(&mut self, key: &str) {
        self.kv.retain(|(k, _)| k != key);
    }

    fn open(&self) -> GgufFile {
        fn string(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (key, val) in &self.kv {
            string(&mut buf, key);
            match val {
                Val::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Val::F32(v) => {
                    buf.extend_from_slice(&6u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Val::Bool(v) => {
                    buf.extend_from_slice(&7u32.to_le_bytes());
                    buf.push(*v as u8);
                }
                Val::Str(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut buf, s);
                }
                Val::Bools(items) => {
                    buf.extend_from_slice(&9u32.to_le_bytes());
                    buf.extend_from_slice(&7u32.to_le_bytes());
                    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    buf.extend(items.iter().map(|&b| b as u8));
                }
            }
        }
        for (name, shape) in &self.tensors {
            string(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "reflex_kolibri_cfg_test_{}_{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&buf))
            .expect("write temp gguf");
        let file = GgufFile::open(&path).expect("parse temp gguf");
        std::fs::remove_file(&path).ok();
        file
    }
}

/// Real Kolibri-1's header: 50 layers, every 5th one full attention.
fn real_kolibri1() -> Header {
    let k = |s: &str| format!("kolibri1.{s}");
    let kv = vec![
        ("general.architecture".to_string(), Val::Str("kolibri1")),
        (k("block_count"), Val::U32(50)),
        (k("embedding_length"), Val::U32(2560)),
        (k("attention.head_count"), Val::U32(48)),
        (k("attention.head_count_kv"), Val::U32(4)),
        (k("attention.key_length"), Val::U32(128)),
        (k("attention.value_length"), Val::U32(128)),
        (k("attention.layer_norm_rms_epsilon"), Val::F32(1e-6)),
        (k("attention.sliding_window"), Val::U32(513)),
        (
            k("attention.sliding_window_pattern"),
            Val::Bools((0..50).map(|i| (i + 1) % 5 != 0).collect()),
        ),
        (k("rope.freq_base"), Val::F32(10000.0)),
        (k("expert_count"), Val::U32(384)),
        (k("expert_used_count"), Val::U32(6)),
        (k("expert_shared_count"), Val::U32(1)),
        (k("expert_feed_forward_length"), Val::U32(512)),
        (k("expert_shared_feed_forward_length"), Val::U32(512)),
        (
            k("expert_gating_func"),
            Val::U32(KOLIBRI_GATING_SIGMOID_LOGIT_ADD as u32),
        ),
        (k("expert_weights_norm"), Val::Bool(false)),
    ];
    let tensors = vec![
        (
            "blk.0.ffn_gate_exps.weight".to_string(),
            vec![2560, 512, 384],
        ),
        ("blk.0.ffn_gate_shexp.weight".to_string(), vec![2560, 512]),
    ];
    Header { kv, tensors }
}

fn err_text(h: &Header) -> String {
    match parse_kolibri_config(&h.open()) {
        Ok(_) => panic!("expected parse_kolibri_config to reject this header"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn parses_real_kolibri1_header() {
    let (cfg, block_count) = parse_kolibri_config(&real_kolibri1().open()).unwrap();
    assert_eq!(block_count, 50);
    assert_eq!(cfg.layer.hidden_size, 2560);
    assert_eq!(cfg.layer.num_q_heads, 48);
    assert_eq!(cfg.layer.num_kv_heads, 4);
    assert_eq!(cfg.layer.head_dim, 128);
    assert_eq!(cfg.layer.rotary_dim, 128);
    assert_eq!(cfg.layer.rope_type, RopeType::Neox);
    assert_eq!(cfg.layer.rope_base, 10000.0);
    assert_eq!(cfg.layer.rmsnorm_eps, 1e-6);
    assert_eq!(cfg.expert_count, 384);
    assert_eq!(cfg.expert_used_count, 6);
    assert_eq!(cfg.n_ff_exp, 512);
    assert_eq!(cfg.n_ff_shexp, 512);
    assert!(!cfg.normalize_top_k);
    assert_eq!(cfg.sliding_window, 513);
    let full: Vec<usize> = (0..50).filter(|&i| !cfg.sliding_layers[i]).collect();
    assert_eq!(full, vec![4, 9, 14, 19, 24, 29, 34, 39, 44, 49]);
}

/// `kolibri1` reports a nonzero `expert_count`; the generic dense/MoE parser
/// must refuse it rather than run it with the wrong router and layer math.
#[test]
fn generic_parser_rejects_kolibri1() {
    let err = match parse_model_config(&real_kolibri1().open()) {
        Ok(_) => panic!("parse_model_config accepted a kolibri1 file"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("parse_kolibri_config"), "{err}");
}

/// DeepSeek-V3's `SIGMOID` gating (2) selects different experts; only
/// `SIGMOID_LOGIT_ADD` is implemented, and a missing key is not a default.
#[test]
fn rejects_other_gating_functions() {
    for gating in [1u32, 2, 3] {
        let mut h = real_kolibri1();
        h.set("kolibri1.expert_gating_func", Val::U32(gating));
        let err = err_text(&h);
        assert!(err.contains("expert_gating_func"), "{err}");
    }
    let mut h = real_kolibri1();
    h.remove("kolibri1.expert_gating_func");
    assert!(err_text(&h).contains("missing kolibri1.expert_gating_func"));
}

#[test]
fn rejects_sliding_window_pattern_problems() {
    let mut h = real_kolibri1();
    h.remove("kolibri1.attention.sliding_window_pattern");
    assert!(err_text(&h).contains("missing kolibri1.attention.sliding_window_pattern"));

    let mut h = real_kolibri1();
    h.set(
        "kolibri1.attention.sliding_window_pattern",
        Val::Bools(vec![true; 49]),
    );
    assert!(err_text(&h).contains("49 entries but block_count is 50"));

    // A scalar period (Gemma-3 style) is not what kolibri1 files carry.
    let mut h = real_kolibri1();
    h.set("kolibri1.attention.sliding_window_pattern", Val::U32(5));
    assert!(err_text(&h).contains("per-layer bool array"));
}

/// Sliding layers are the only ones with RoPE, and the patch gives them
/// `rope.freq_base_swa` over `rope.freq_base` when the file sets it.
#[test]
fn sliding_rope_base_prefers_freq_base_swa() {
    let mut h = real_kolibri1();
    h.set("kolibri1.rope.freq_base_swa", Val::F32(500_000.0));
    let (cfg, _) = parse_kolibri_config(&h.open()).unwrap();
    assert_eq!(cfg.layer.rope_base, 500_000.0);
}

#[test]
fn rejects_unsupported_shapes_and_scaling() {
    let mut h = real_kolibri1();
    h.set("kolibri1.expert_shared_count", Val::U32(2));
    assert!(err_text(&h).contains("expert_shared_count = 2"));

    let mut h = real_kolibri1();
    h.set("kolibri1.expert_used_count", Val::U32(385));
    assert!(err_text(&h).contains("expert_used_count (385)"));

    let mut h = real_kolibri1();
    h.set("kolibri1.expert_feed_forward_length", Val::U32(1024));
    assert!(err_text(&h).contains("blk.0.ffn_gate_exps.weight has 512 output features"));

    let mut h = real_kolibri1();
    h.set("kolibri1.expert_count", Val::U32(256));
    assert!(err_text(&h).contains("stacks 384 experts"));

    let mut h = real_kolibri1();
    h.set("kolibri1.rope.scaling.type", Val::Str("yarn"));
    assert!(err_text(&h).contains("rope.scaling.type"));

    let mut h = real_kolibri1();
    h.set("kolibri1.attention.value_length", Val::U32(64));
    assert!(err_text(&h).contains("value_length (64)"));

    let mut h = real_kolibri1();
    h.set("kolibri1.expert_weights_scale", Val::F32(2.5));
    assert!(err_text(&h).contains("expert_weights_scale = 2.5"));
}

/// `test-data/tiny-kolibri1.gguf`: synthetic Kolibri-1 (random weights, real
/// tensor names and tokenizer), converted and quantized to Q4_K_M with
/// llama.cpp `836d571` plus the community `kolibri1-llama.cpp.patch`; source
/// in `test-data/tiny-kolibri1-src.tar.gz` (docs/DEVELOPMENT.md's test-fixture
/// section). Host-only, but `#[ignore]`d because `test-data/*.gguf` is
/// gitignored: `cargo test -- --ignored kolibri1_fixture_has_the_properties_it_was_built_for`.
#[test]
#[ignore]
fn kolibri1_fixture_has_the_properties_it_was_built_for() {
    use crate::gguf::GgmlType;
    let file = GgufFile::open("test-data/tiny-kolibri1.gguf").expect("open tiny-kolibri1.gguf");
    let (cfg, block_count) = parse_kolibri_config(&file).expect("parse_kolibri_config");

    assert_eq!(block_count, 6);
    assert_eq!(
        cfg.sliding_layers,
        vec![true, true, true, true, false, true]
    );
    assert_eq!(
        cfg.sliding_window, 16,
        "small window so short prompts exercise it"
    );
    assert_eq!((cfg.expert_count, cfg.expert_used_count), (16, 4));
    assert!(!cfg.normalize_top_k);

    // A zero bias would make SIGMOID_LOGIT_ADD indistinguishable from
    // selecting on the raw logits.
    let bias = file
        .tensor_info("blk.0.exp_probs_b.bias")
        .expect("exp_probs_b");
    assert_eq!(bias.ggml_type, GgmlType::F32);
    let raw = file.tensor_bytes(bias).unwrap();
    assert!(raw
        .as_chunks::<4>()
        .0
        .iter()
        .any(|c| f32::from_le_bytes(*c).abs() > 0.1));

    // Same Q4_K/Q6_K mix as the real Q4_K_M file.
    let ty = |name: &str| file.tensor_info(name).expect(name).ggml_type;
    assert_eq!(ty("blk.0.ffn_gate_exps.weight"), GgmlType::Q4K);
    assert_eq!(ty("blk.0.ffn_down_exps.weight"), GgmlType::Q4K);
    assert_eq!(ty("blk.2.ffn_down_exps.weight"), GgmlType::Q6K);
    assert_eq!(ty("blk.2.attn_v.weight"), GgmlType::Q6K);
    assert_eq!(ty("output.weight"), GgmlType::Q6K);
    assert_eq!(ty("token_embd.weight"), GgmlType::Q4K);
    for i in 0..block_count {
        for t in [
            "post_attention_norm",
            "post_ffw_norm",
            "ffn_gate_shexp",
            "attn_q_norm",
        ] {
            assert!(
                file.tensor_info(&format!("blk.{i}.{t}.weight")).is_some(),
                "blk.{i}.{t}"
            );
        }
    }
}
