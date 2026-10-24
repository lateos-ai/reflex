//! Embedding-model config tests (host-only), on the real Nemotron-3.5-Embed-1B
//! header (`test-data/nemotron-3.5-embed-1b-header.gguf`: metadata and tensor
//! table of `maternion/Nemotron-3.5-Embed-1B-GGUF`'s Q8_0 file, no tensor data,
//! via `scripts/gguf_tokenizer_only.py --keep-tensor-table`).
//! See docs/design/embeddings.md.

use super::config::{is_encoder, parse_encoder_config, parse_model_config};
use super::*;
use crate::gguf::GgufFile;

const HEADER: &str = "test-data/nemotron-3.5-embed-1b-header.gguf";

fn header() -> Option<GgufFile> {
    if !std::path::Path::new(HEADER).exists() {
        eprintln!("Skipping: {HEADER} not present (see docs/DEVELOPMENT.md's test fixtures)");
        return None;
    }
    Some(GgufFile::open(HEADER).expect("open Nemotron embed header"))
}

/// The real file parses as a dense `mistral3` encoder: the Ministral 3 text
/// tower's shape, consecutive-pair RoPE, mean pooling, YaRN and the query
/// temperature.
#[test]
fn nemotron_embed_header_parses_as_encoder() {
    let Some(file) = header() else { return };
    let (cfg, block_count, moe) = parse_model_config(&file).expect("parse_model_config");
    assert_eq!(block_count, 16);
    assert!(moe.is_none());
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(
        (cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim),
        (24, 8, 128)
    );
    assert_eq!(cfg.rotary_dim, 128);
    assert_eq!(cfg.ffn_hidden_size, 6144);
    assert_eq!(cfg.rope_base, 1_000_000.0);
    assert_eq!(cfg.rope_type, RopeType::Norm);

    assert!(is_encoder(&file, "mistral3"));
    let enc = parse_encoder_config(&file, "mistral3", &cfg)
        .expect("parse_encoder_config")
        .expect("an encoder");
    assert_eq!(enc.pooling, Pooling::Mean);

    let y = enc.yarn.expect("YaRN");
    assert_eq!(y.freq_scale, 1.0 / 16.0);
    assert_eq!(y.ext_factor, 1.0);
    // HF `_compute_yarn_parameters` (beta_fast 32, beta_slow 1, dim 128, base
    // 1e6, original 16384): find_correction_range = (floor(20.38), ceil(36.44)).
    assert_eq!((y.corr_dim_start, y.corr_dim_end), (20.0, 37.0));
    // yarn_log_multiplier = 1 (HF mscale = mscale_all_dim = 1): magnitude 1, so
    // the kernel's `attn_factor * (1 + 0.1 ln 16)` is 1.
    let magnitude = y.attn_factor * (1.0 + 0.1 * 16f32.ln());
    assert!(
        (magnitude - 1.0).abs() < 1e-6,
        "cos/sin magnitude {magnitude}"
    );

    let t = enc.temperature.expect("query temperature");
    assert!((t.beta - 0.1).abs() < 1e-7);
    assert_eq!(t.floor_scale, 16384.0);
}

/// Without `attention.causal = false` the same file is a Ministral 3 chat model,
/// which is rejected by name; a causal model is never an encoder.
#[test]
fn causal_mistral3_is_rejected() {
    let Some(mut file) = header() else { return };
    for causal in [None, Some(true)] {
        match causal {
            None => file.metadata.remove("mistral3.attention.causal"),
            Some(b) => file
                .metadata
                .insert("mistral3.attention.causal".to_string(), GgufValue::Bool(b)),
        };
        assert!(!is_encoder(&file, "mistral3"));
        let err = parse_model_config(&file).err().expect("must be rejected");
        assert!(
            err.to_string().contains("Ministral 3 chat"),
            "unexpected error: {err}"
        );
    }
}

/// Encoders other than mean-pooled ones are rejected, naming the key.
#[test]
fn encoder_without_mean_pooling_is_rejected() {
    let Some(mut file) = header() else { return };
    let (cfg, _, _) = parse_model_config(&file).expect("parse_model_config");
    for pooling in [None, Some(0u32), Some(2), Some(3)] {
        match pooling {
            None => file.metadata.remove("mistral3.pooling_type"),
            Some(p) => file
                .metadata
                .insert("mistral3.pooling_type".to_string(), GgufValue::U32(p)),
        };
        let err = parse_encoder_config(&file, "mistral3", &cfg).expect_err("must be rejected");
        assert!(err.to_string().contains("pooling_type"), "{err}");
    }
}

/// No YaRN keys and no temperature: plain RoPE, no query scaling.
#[test]
fn encoder_without_yarn_or_temperature() {
    let Some(mut file) = header() else { return };
    for k in [
        "mistral3.rope.scaling.type",
        "mistral3.attention.temperature_scale",
    ] {
        file.metadata.remove(k);
    }
    let (cfg, _, _) = parse_model_config(&file).expect("parse_model_config");
    let enc = parse_encoder_config(&file, "mistral3", &cfg)
        .expect("parse_encoder_config")
        .expect("an encoder");
    assert_eq!(enc.yarn, None);
    assert_eq!(enc.temperature, None);
}

/// A non-YaRN scaling type is rejected for an encoder, and YaRN stays rejected
/// for causal dense models.
#[test]
fn rope_scaling_rejections() {
    let Some(mut file) = header() else { return };
    file.metadata.insert(
        "mistral3.rope.scaling.type".to_string(),
        GgufValue::String("linear".to_string()),
    );
    assert!(parse_model_config(&file).is_err());
}

/// `l2_normalize` gives a unit vector and leaves zeros alone.
#[test]
fn l2_normalize_unit_and_zero() {
    let mut v = vec![3.0f32, 4.0];
    l2_normalize(&mut v);
    assert_eq!(v, vec![0.6, 0.8]);
    let mut z = vec![0.0f32; 3];
    l2_normalize(&mut z);
    assert_eq!(z, vec![0.0; 3]);
}

/// The tokenizer reproduces HF's token ids for every reference input (the fixture's
/// and the real model's: retrieval prefixes, the `[`/`<` + U+200C markers, five
/// scripts, and passages up to ~17K tokens), host-only.
#[test]
fn tokenizer_matches_hf_reference_ids() {
    for (gguf, refs) in [
        (
            "test-data/tiny-mistral3-embed.gguf",
            "test-data/tiny-mistral3-embed-ref.tsv",
        ),
        (HEADER, "test-data/nemotron-3.5-embed-ref.tsv"),
    ] {
        if !std::path::Path::new(gguf).exists() || !std::path::Path::new(refs).exists() {
            eprintln!("Skipping {gguf}: it or {refs} is not present");
            continue;
        }
        let file = GgufFile::open(gguf).expect("open");
        let tok = crate::tokenizer::Tokenizer::from_gguf(&file).expect("tokenizer");
        let refs = super::encoder_tests::read_refs(refs);
        assert!(!refs.is_empty());
        for (i, r) in refs.iter().enumerate() {
            let ids = tok.encode(&r.text).expect("encode");
            assert_eq!(ids, r.ids, "{gguf}: input {i} token ids differ from HF");
        }
        eprintln!("{gguf}: {} inputs match HF", refs.len());
    }
}
