//! GPU tests for embedding models (`encoder.rs`): the bidirectional attention
//! kernels, YaRN RoPE, query temperature and mean pooling against host
//! references, then whole embeddings against HF `transformers`. `#[ignore]`d:
//! they need a GPU.
//!
//! `test-data/tiny-mistral3-embed.gguf` is a random-weight `Ministral3Model` text
//! encoder (`is_causal: false`, mean pooling, 2 layers, hidden 128, 4 query / 2 KV
//! heads of dim 64, FFN 256, the real Nemotron-3.5-Embed tokenizer) with YaRN
//! (factor 4, original context 64) and the query temperature (beta 0.1, floor 64),
//! so prompts past 64 tokens exercise both, converted by llama.cpp's unmodified
//! `convert_hf_to_gguf.py --outtype f32`. `tiny-mistral3-embed-ref.tsv` holds HF's
//! embeddings of the comparison inputs. Source, scripts and references:
//! `test-data/tiny-mistral3-embed-src.tar.gz`. See docs/design/embeddings.md.

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;
use rand::{Rng, SeedableRng};

const FIXTURE: &str = "test-data/tiny-mistral3-embed.gguf";
const FIXTURE_REF: &str = "test-data/tiny-mistral3-embed-ref.tsv";

fn load(path: &str, weights: WeightsDtype) -> Model {
    let file = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("load model")
}

fn fixture(weights: WeightsDtype) -> Model {
    load(FIXTURE, weights)
}

fn upload(model: &Model, v: &[f32]) -> CudaSlice<f32> {
    model.device.htod_sync_copy(v).expect("htod")
}

fn download(model: &Model, v: &CudaSlice<f32>) -> Vec<f32> {
    model.device.dtoh_sync_copy(v).expect("dtoh")
}

/// `||a - b|| / ||b||` (L2 norms, in f64).
fn rel_l2_err(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - y).powi(2))
        .sum();
    let norm: f64 = b.iter().map(|y| y * y).sum();
    (diff / norm).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let na: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// Bidirectional GQA attention in f64: every row sees all `kv_len` positions.
#[allow(clippy::too_many_arguments)]
fn host_bidir_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    kv_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    d: usize,
) -> Vec<f64> {
    let group = num_q_heads / num_kv_heads;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0; rows * num_q_heads * d];
    for r in 0..rows {
        for h in 0..num_q_heads {
            let kvh = h / group;
            let qv = &q[(r * num_q_heads + h) * d..][..d];
            let scores: Vec<f64> = (0..kv_len)
                .map(|j| {
                    let kr = &k[(j * num_kv_heads + kvh) * d..][..d];
                    qv.iter()
                        .zip(kr)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = w.iter().sum();
            let o = &mut out[(r * num_q_heads + h) * d..][..d];
            for (j, wj) in w.iter().enumerate() {
                let vr = &v[(j * num_kv_heads + kvh) * d..][..d];
                for (oi, &vi) in o.iter_mut().zip(vr) {
                    *oi += wj / l * f64::from(vi);
                }
            }
        }
    }
    out
}

/// The online and tiled bidirectional kernels against a host f64 reference, at
/// the fixture's shape and the real model's (head dim 128, group 3), across the
/// tiled crossover (64 rows) and tile boundaries, with and without a cached
/// prefix (`start_pos`).
#[test]
#[ignore]
fn encoder_bidir_attention_matches_host_reference() {
    let model = fixture(WeightsDtype::F32);
    let enc = model.encoder.as_ref().expect("fixture is an encoder");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xb1d1);
    // (rows, start_pos, num_q_heads, num_kv_heads, head_dim)
    let cases = [
        (1, 0, 4, 2, 64),
        (5, 0, 4, 2, 64),
        (63, 0, 4, 2, 64),
        (64, 0, 4, 2, 64),
        (130, 0, 4, 2, 64),
        (65, 40, 4, 2, 64),
        (200, 0, 24, 8, 128),
        (33, 0, 24, 8, 128),
        (129, 7, 24, 8, 128),
    ];
    for tiled in [true, false] {
        model.attn_tiled.set_enabled(tiled);
        for &(rows, start_pos, nq, nkv, d) in &cases {
            let kv_len = start_pos + rows;
            let mut gen =
                |n: usize| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect() };
            let q = gen(rows * nq * d);
            let k = gen(kv_len * nkv * d);
            let v = gen(kv_len * nkv * d);
            let (qd, kd, vd) = (upload(&model, &q), upload(&model, &k), upload(&model, &v));
            let shape = OnlineAttnShape {
                rows,
                start_pos,
                num_q_heads: nq,
                group_size: nq / nkv,
                qk_dim: d,
                v_dim: d,
                k_pos_stride: nkv * d,
                k_head_stride: d,
                v_pos_stride: nkv * d,
                v_head_stride: d,
                scale: 1.0 / (d as f32).sqrt(),
                window: 0,
            };
            let out = model
                .attention_prefill_bidir(
                    enc,
                    &qd,
                    &KvView::F32(kd.slice(..)),
                    &KvView::F32(vd.slice(..)),
                    shape,
                )
                .expect("attention_prefill_bidir");
            let expected = host_bidir_attention(&q, &k, &v, rows, kv_len, nq, nkv, d);
            let err = rel_l2_err(&download(&model, &out), &expected);
            eprintln!("tiled={tiled} rows={rows} start={start_pos} heads={nq}/{nkv} d={d}: rel L2 err {err:.2e}");
            assert!(
                err < 1e-5,
                "tiled={tiled} rows={rows} d={d}: rel L2 err {err}"
            );
        }
    }
}

/// The YaRN rotation (`rope_norm_yarn_batch_kernel` with the fixture's
/// parameters) against ggml's formula in f64, at positions past the original
/// context, and the query temperature against llama.cpp's formula, on both sides
/// of its floor scale.
#[test]
#[ignore]
fn encoder_yarn_rope_and_temperature_match_host() {
    let model = fixture(WeightsDtype::F32);
    let enc = model.encoder.as_ref().expect("fixture is an encoder");
    let y = enc.cfg.yarn.expect("fixture has YaRN");
    let t = enc.cfg.temperature.expect("fixture has the temperature");
    let cfg = &model.cfg;
    let (heads, d, rows, start) = (cfg.num_q_heads, cfg.head_dim, 150usize, 10usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x7a2e);
    let x: Vec<f32> = (0..rows * heads * d)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let mut xd = upload(&model, &x);
    assert!(model
        .rope_yarn_batch(
            &mut xd,
            start,
            heads,
            d,
            cfg.rotary_dim,
            rows,
            cfg.rope_base
        )
        .expect("rope_yarn_batch"));
    let got = download(&model, &xd);
    let mut expected = vec![0.0f64; x.len()];
    let half = cfg.rotary_dim / 2;
    for r in 0..rows {
        let pos = (start + r) as f64;
        for h in 0..heads {
            let base_idx = (r * heads + h) * d;
            for i in 0..half {
                let extrap =
                    pos / f64::from(cfg.rope_base).powf(2.0 * i as f64 / cfg.rotary_dim as f64);
                let interp = f64::from(y.freq_scale) * extrap;
                let ramp_y = (i as f64 - f64::from(y.corr_dim_start))
                    / f64::from(y.corr_dim_end - y.corr_dim_start).max(0.001);
                let ramp = (1.0 - ramp_y.clamp(0.0, 1.0)) * f64::from(y.ext_factor);
                let theta = interp * (1.0 - ramp) + extrap * ramp;
                let mscale =
                    f64::from(y.attn_factor) * (1.0 + 0.1 * (1.0 / f64::from(y.freq_scale)).ln());
                let (x1, x2) = (
                    f64::from(x[base_idx + 2 * i]),
                    f64::from(x[base_idx + 2 * i + 1]),
                );
                expected[base_idx + 2 * i] = (x1 * theta.cos() - x2 * theta.sin()) * mscale;
                expected[base_idx + 2 * i + 1] = (x1 * theta.sin() + x2 * theta.cos()) * mscale;
            }
        }
    }
    let abs_err = got
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    eprintln!("yarn rope: max abs err {abs_err:.2e}");
    assert!(abs_err < 1e-4, "yarn rope max abs err {abs_err}");

    for (start_pos, rows) in [(0usize, 50usize), (0, 200), (100, 40)] {
        let row_len = 8;
        let q: Vec<f32> = (0..rows * row_len)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let mut qd = upload(&model, &q);
        model
            .encoder_temp_scale(&mut qd, start_pos, rows, row_len)
            .expect("encoder_temp_scale");
        let got = download(&model, &qd);
        for (i, (&g, &orig)) in got.iter().zip(&q).enumerate() {
            let pos = (start_pos + i / row_len) as f64;
            let s = 1.0 + f64::from(t.beta) * ((pos / f64::from(t.floor_scale)).floor() + 1.0).ln();
            let want = f64::from(orig) * s;
            assert!(
                (f64::from(g) - want).abs() <= 1e-6 * want.abs().max(1.0),
                "temperature at pos {pos}: got {g}, want {want}"
            );
        }
    }
}

/// Mean pooling sums each column in row order in f32, so it matches the same
/// host loop exactly.
#[test]
#[ignore]
fn encoder_mean_pool_matches_host_exactly() {
    let model = fixture(WeightsDtype::F32);
    let enc = model.encoder.as_ref().expect("fixture is an encoder");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x3ea7);
    for (rows, cols) in [(1usize, 128usize), (37, 128), (513, 2048)] {
        let x: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let pooled = model
            .mean_pool(enc, &upload(&model, &x), rows, cols)
            .expect("mean_pool");
        let got = download(&model, &pooled);
        let want: Vec<f32> = (0..cols)
            .map(|c| (0..rows).fold(0.0f32, |s, r| s + x[r * cols + c]) / rows as f32)
            .collect();
        assert_eq!(got, want, "rows={rows} cols={cols}");
    }
}

/// One reference line: the exact text HF tokenized (hex UTF-8), its token ids and
/// HF's embedding.
pub(super) struct RefLine {
    pub(super) text: String,
    pub(super) ids: Vec<u32>,
    pub(super) embedding: Vec<f32>,
}

pub(super) fn read_refs(path: &str) -> Vec<RefLine> {
    let body = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    body.lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert_eq!(cols.len(), 3, "bad reference line in {path}");
            let bytes: Vec<u8> = (0..cols[0].len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&cols[0][i..i + 2], 16).expect("hex"))
                .collect();
            RefLine {
                text: String::from_utf8(bytes).expect("utf-8"),
                ids: cols[1].split(',').map(|s| s.parse().expect("id")).collect(),
                embedding: cols[2]
                    .split(',')
                    .map(|s| s.parse().expect("float"))
                    .collect(),
            }
        })
        .collect()
}

/// Embeds every reference text and checks the token ids match HF's exactly and
/// the embedding's cosine similarity to HF's is at least `min_cosine`.
fn assert_matches_refs(model: &Model, refs: &[RefLine], min_cosine: f64, tag: &str) {
    let mut worst = 1.0f64;
    for (i, r) in refs.iter().enumerate() {
        let ids = model.tokenizer.encode(&r.text).expect("encode");
        assert_eq!(ids, r.ids, "{tag}: input {i} token ids differ from HF");
        let e = model.embed(&r.text).expect("embed");
        assert_eq!(e.prompt_tokens, r.ids.len());
        let c = cosine(&e.vector, &r.embedding);
        worst = worst.min(c);
        eprintln!("{tag}: input {i} ({} tokens): cosine {c:.7}", r.ids.len());
        assert!(
            c >= min_cosine,
            "{tag}: input {i}: cosine {c} < {min_cosine}"
        );
    }
    eprintln!("{tag}: worst cosine {worst:.7} over {} inputs", refs.len());
}

/// The fixture against HF `transformers` (fp32, the model card's mean pooling)
/// with f32 weights, tiled and online attention, on inputs from 1 to ~200
/// tokens, the longer ones past the original context (YaRN and temperature).
#[test]
#[ignore]
fn encoder_fixture_matches_hf_reference() {
    let refs = read_refs(FIXTURE_REF);
    let model = fixture(WeightsDtype::F32);
    assert!(model.is_embedding_model());
    for tiled in [true, false] {
        model.attn_tiled.set_enabled(tiled);
        assert_matches_refs(&model, &refs, 0.99999, &format!("f32 tiled={tiled}"));
    }
    let model = fixture(WeightsDtype::F16);
    assert_matches_refs(&model, &refs, 0.999, "f16");
}

/// An embedding model refuses generation and scoring, and a language model
/// refuses `embed`, each with `invalid_input`.
#[test]
#[ignore]
fn encoder_fixture_rejects_generation_and_lm_rejects_embed() {
    let model = fixture(WeightsDtype::F32);
    let sampling = crate::sampling::SamplingParams::default();
    let err = model
        .generate("query: hi", 4, None, &sampling, |_| {}, |_, _| {})
        .expect_err("generate must fail on an encoder");
    assert_eq!(err.kind(), "invalid_input", "{err}");
    let err = model
        .system1_evaluate("query: hi", &[System1Candidate { text: " a".into() }], 1.0)
        .expect_err("system1 must fail on an encoder");
    assert_eq!(err.kind(), "invalid_input", "{err}");
    assert_eq!(
        model
            .forward_prompt("hi")
            .expect_err("forward_prompt")
            .kind(),
        "invalid_input"
    );
    drop(model);

    let lm = "test-data/tiny-qwen2.gguf";
    if std::path::Path::new(lm).exists() {
        let model = load(lm, WeightsDtype::F32);
        assert!(!model.is_embedding_model());
        let err = model.embed("hi").expect_err("embed must fail on an LM");
        assert_eq!(err.kind(), "invalid_input", "{err}");
    }
}

/// The real Nemotron-3.5-Embed-1B (`EMBED_GGUF`) against HF `transformers` fp32
/// embeddings of the reference inputs (`EMBED_REF`, produced by the archive's
/// `hf_reference.py`): token ids exact, and a cosine floor by weight type: an
/// unquantized file (BF16/F16/F32) >= 0.9999 with f32 weights and >= 0.999 with f16,
/// Q8_0 >= 0.999, anything smaller >= 0.97: a 4-bit file moves the embedding by more
/// than the engine does (Q4_K_M: 0.975 on a 3-token input, where llama.cpp gives 0.977
/// on the same file, and ~0.99 on longer ones).
#[test]
#[ignore]
fn nemotron_embed_real_model_matches_hf_reference() {
    let path = std::env::var("EMBED_GGUF").expect("set EMBED_GGUF");
    let refs = read_refs(&std::env::var("EMBED_REF").expect("set EMBED_REF"));
    let ggml_type = GgufFile::open(&path)
        .expect("open")
        .tensor_info("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight")
        .ggml_type;
    let (f32_floor, f16_floor) = match ggml_type {
        GgmlType::Bf16 | GgmlType::F16 | GgmlType::F32 => (0.9999, 0.999),
        GgmlType::Q8_0 => (0.999, 0.999),
        _ => (0.97, 0.97),
    };
    eprintln!("{path}: attn_q is {ggml_type:?}");
    let model = load(&path, WeightsDtype::F32);
    assert_matches_refs(&model, &refs, f32_floor, "f32 weights");
    drop(model);
    let model = load(&path, WeightsDtype::F16);
    assert_matches_refs(&model, &refs, f16_floor, "f16 weights");
}
