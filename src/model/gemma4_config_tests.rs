//! Host-only tests for [`parse_gemma4_config`]: synthetic GGUF headers (no
//! tensor data) carrying the metadata of the real `ggml-org/gemma-4-12B-it-GGUF`
//! and `gemma-4-31B-it-GGUF` files (docs/design/gemma4.md's "Confirmed
//! conventions").

use super::config::{parse_gemma4_config, Gemma4LayerShape};
use crate::gguf::GgufFile;
use std::io::Write;

enum Val {
    U32(u32),
    F32(f32),
    Str(&'static str),
    Bools(Vec<bool>),
    U32s(Vec<u32>),
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
                Val::U32s(items) => {
                    buf.extend_from_slice(&9u32.to_le_bytes());
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for v in items {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
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
            "reflex_gemma4_cfg_test_{}_{}.gguf",
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

/// A real Gemma 4 dense header: `blocks` layers, every 6th one global, with
/// the given sizes. Other values are shared by the 12B and 31B files.
fn real_gemma4(
    blocks: usize,
    hidden: u32,
    ffn: u32,
    heads: u32,
    kv_swa: u32,
    kv_global: u32,
) -> Header {
    let k = |s: &str| format!("gemma4.{s}");
    let pattern: Vec<bool> = (0..blocks).map(|i| i % 6 != 5).collect();
    let kv_heads: Vec<u32> = pattern
        .iter()
        .map(|&s| if s { kv_swa } else { kv_global })
        .collect();
    let kv = vec![
        ("general.architecture".to_string(), Val::Str("gemma4")),
        (k("block_count"), Val::U32(blocks as u32)),
        (k("context_length"), Val::U32(262_144)),
        (k("embedding_length"), Val::U32(hidden)),
        (k("feed_forward_length"), Val::U32(ffn)),
        (k("attention.head_count"), Val::U32(heads)),
        (k("attention.head_count_kv"), Val::U32s(kv_heads)),
        (k("rope.freq_base"), Val::F32(1_000_000.0)),
        (k("rope.freq_base_swa"), Val::F32(10_000.0)),
        (k("attention.layer_norm_rms_epsilon"), Val::F32(1e-6)),
        (k("attention.key_length"), Val::U32(512)),
        (k("attention.value_length"), Val::U32(512)),
        (k("final_logit_softcapping"), Val::F32(30.0)),
        (k("attention.sliding_window"), Val::U32(1024)),
        (k("attention.shared_kv_layers"), Val::U32(0)),
        (k("embedding_length_per_layer_input"), Val::U32(0)),
        (k("attention.sliding_window_pattern"), Val::Bools(pattern)),
        (k("attention.key_length_swa"), Val::U32(256)),
        (k("attention.value_length_swa"), Val::U32(256)),
        (k("rope.dimension_count"), Val::U32(512)),
        (k("rope.dimension_count_swa"), Val::U32(256)),
    ];
    Header {
        kv,
        tensors: vec![("rope_freqs.weight".to_string(), vec![256])],
    }
}

fn real_12b() -> Header {
    real_gemma4(48, 3840, 15360, 16, 8, 1)
}

fn real_31b() -> Header {
    real_gemma4(60, 5376, 21504, 32, 16, 4)
}

fn err_text(h: &Header) -> String {
    parse_gemma4_config(&h.open())
        .expect_err("config should be rejected")
        .to_string()
}

#[test]
fn parses_real_gemma4_12b_header() {
    let cfg = parse_gemma4_config(&real_12b().open()).expect("parse 12B");
    assert_eq!(cfg.hidden_size, 3840);
    assert_eq!(cfg.num_q_heads, 16);
    assert_eq!(cfg.ffn_hidden_size, 15360);
    assert_eq!(cfg.sliding_window, 1024);
    assert_eq!(cfg.final_logit_softcap, Some(30.0));
    assert_eq!(cfg.rmsnorm_eps, 1e-6);
    assert_eq!(cfg.layers.len(), 48);
    let sliding = Gemma4LayerShape {
        sliding: true,
        head_dim: 256,
        num_kv_heads: 8,
        n_rot: 256,
        rope_base: 10_000.0,
    };
    let global = Gemma4LayerShape {
        sliding: false,
        head_dim: 512,
        num_kv_heads: 1,
        n_rot: 512,
        rope_base: 1_000_000.0,
    };
    for (i, layer) in cfg.layers.iter().enumerate() {
        let want = if i % 6 == 5 { &global } else { &sliding };
        assert_eq!(layer, want, "layer {i}");
    }
}

#[test]
fn parses_real_gemma4_31b_header() {
    let cfg = parse_gemma4_config(&real_31b().open()).expect("parse 31B");
    assert_eq!(cfg.layers.len(), 60);
    assert_eq!(cfg.num_q_heads, 32);
    assert_eq!(cfg.layers[0].num_kv_heads, 16);
    assert_eq!(cfg.layers[5].num_kv_heads, 4);
    assert_eq!(cfg.layers.iter().filter(|l| !l.sliding).count(), 10);
}

#[test]
fn scalar_head_count_kv_applies_to_every_layer() {
    let mut h = real_12b();
    h.set("gemma4.attention.head_count_kv", Val::U32(4));
    let cfg = parse_gemma4_config(&h.open()).expect("parse");
    assert!(cfg.layers.iter().all(|l| l.num_kv_heads == 4));
}

#[test]
fn rejects_out_of_scope_variants() {
    for (key, what) in [
        (
            "gemma4.embedding_length_per_layer_input",
            "per-layer token embeddings",
        ),
        (
            "gemma4.attention.shared_kv_layers",
            "KV caches shared across layers",
        ),
        ("gemma4.expert_count", "mixture-of-experts"),
    ] {
        let mut h = real_12b();
        h.set(key, Val::U32(18));
        let err = err_text(&h);
        assert!(err.contains(key) && err.contains(what), "{key}: {err}");
    }

    let mut h = real_12b();
    h.set("gemma4.rope.scaling.type", Val::Str("yarn"));
    assert!(err_text(&h).contains("rope.scaling.type"));
}

#[test]
fn rejects_shape_problems() {
    let mut h = real_12b();
    h.set("gemma4.attention.value_length_swa", Val::U32(128));
    assert!(err_text(&h).contains("value_length_swa"));

    let mut h = real_12b();
    h.remove("gemma4.attention.sliding_window_pattern");
    assert!(err_text(&h).contains("sliding_window_pattern"));

    let mut h = real_12b();
    h.set(
        "gemma4.attention.sliding_window_pattern",
        Val::Bools(vec![true; 10]),
    );
    assert!(err_text(&h).contains("10 entries"));

    let mut h = real_12b();
    h.set("gemma4.attention.head_count_kv", Val::U32s(vec![3; 48]));
    assert!(err_text(&h).contains("multiple"));

    let mut h = real_12b();
    h.set(
        "gemma4.feed_forward_length",
        Val::U32s((0..48).map(|i| 15360 + i).collect()),
    );
    assert!(err_text(&h).contains("varies per layer"));

    let mut h = real_12b();
    h.set("gemma4.rope.dimension_count_swa", Val::U32(512));
    assert!(err_text(&h).contains("rope.dimension_count_swa"));
}

#[test]
fn global_layers_need_matching_rope_freqs() {
    let mut h = real_12b();
    h.tensors.clear();
    assert!(err_text(&h).contains("missing rope_freqs.weight"));

    let mut h = real_12b();
    h.tensors = vec![("rope_freqs.weight".to_string(), vec![128])];
    assert!(err_text(&h).contains("128 entries, expected 256"));

    // All-sliding models don't need it.
    let mut h = real_12b();
    h.tensors.clear();
    h.set(
        "gemma4.attention.sliding_window_pattern",
        Val::Bools(vec![true; 48]),
    );
    assert!(parse_gemma4_config(&h.open()).is_ok());
}
