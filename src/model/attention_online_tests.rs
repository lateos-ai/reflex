//! The online-softmax attention kernels (`kernels_cuda/attention_online.cu`) against
//! the legacy kernels they replace, on one loaded model with `attn_impl` toggled.
//! `#[ignore]`d: they need a real GPU (and, for the end-to-end test, a model in
//! `REFLEX_TEST_GGUF`; for the MLA test, the synthetic MLA fixture).

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

const MLA_FIXTURE: &str = "test-data/deepseek-tiny-mla.gguf";

/// Kernel-level tolerance: both kernels compute the same softmax-weighted average of
/// values in [-1, 1]; only the f32 summation order differs.
const KERNEL_TOL: f32 = 1e-5;

/// Deterministic values in [-1, 1) (an LCG, so the test needs no RNG setup).
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn load(path: &str) -> Model {
    let file = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    Model::load(device, &file).expect("failed to load model")
}

fn upload(model: &Model, host: &[f32]) -> CudaSlice<f32> {
    model.device.htod_sync_copy(host).expect("htod")
}

fn download(model: &Model, dev: &CudaSlice<f32>) -> Vec<f32> {
    model.device.dtoh_sync_copy(dev).expect("dtoh")
}

/// GQA decode and prefill on random Q/K/V with the model's real head layout,
/// including decode lengths on both sides of the split-K threshold (256 positions per
/// block) and a prefill resumed at a non-zero `start_pos`.
#[test]
#[ignore]
fn online_attention_matches_legacy_kernels_gqa() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a dense/MoE Qwen3 or Llama GGUF");
    let mut model = load(&path);
    let (hq, hkv, d) = (
        model.cfg.num_q_heads,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
    );

    let run = |model: &mut Model, imp: AttnImpl, start_pos: usize, rows: usize, seed: u64| {
        model.attn_impl = imp;
        let positions = start_pos + rows;
        let q = upload(model, &pseudo_random(rows * hq * d, seed));
        let k = upload(model, &pseudo_random(positions * hkv * d, seed + 1));
        let v = upload(model, &pseudo_random(positions * hkv * d, seed + 2));
        let out = if rows == 1 {
            model.attention(&q, &k.slice(..), &v.slice(..), hq, hkv, d, positions)
        } else {
            model.attention_prefill(&q, &k.slice(..), &v.slice(..), hq, hkv, d, start_pos, rows)
        }
        .expect("attention");
        download(model, &out)
    };

    let mut worst = 0.0f32;
    // (start_pos, rows): rows == 1 is decode over start_pos + 1 positions.
    for (i, &(start_pos, rows)) in [
        (0, 1),
        (36, 1),
        (254, 1),
        (255, 1),
        (511, 1),
        (999, 1),
        (4096, 1),
        (0, 50),
        (100, 37),
        (700, 300),
    ]
    .iter()
    .enumerate()
    {
        let legacy = run(&mut model, AttnImpl::Legacy, start_pos, rows, i as u64 * 7);
        let online = run(&mut model, AttnImpl::Online, start_pos, rows, i as u64 * 7);
        let diff = max_abs_diff(&legacy, &online);
        println!("gqa start_pos={start_pos} rows={rows}: max abs diff {diff:e}");
        assert!(
            diff < KERNEL_TOL,
            "start_pos={start_pos} rows={rows}: {diff:e}"
        );
        worst = worst.max(diff);
    }
    println!("gqa worst max abs diff: {worst:e}");
}

/// MLA decode and prefill on random inputs with the MLA fixture's real head dims
/// (compressed qk_dim/v_dim, not powers of two).
#[test]
#[ignore]
fn online_attention_matches_legacy_kernels_mla() {
    let mut model = load(MLA_FIXTURE);
    let (hq, qk_dim, v_dim, scale) = {
        let m = model.mla.as_ref().expect("fixture is not MLA");
        let qk_dim = m.cfg.kv_lora_rank + m.cfg.qk_rope_head_dim;
        let scale = 1.0 / ((m.cfg.qk_nope_head_dim + m.cfg.qk_rope_head_dim) as f32).sqrt();
        (m.cfg.num_heads, qk_dim, m.cfg.kv_lora_rank, scale)
    };

    let run = |model: &mut Model, imp: AttnImpl, start_pos: usize, rows: usize, seed: u64| {
        model.attn_impl = imp;
        let positions = start_pos + rows;
        let q = upload(model, &pseudo_random(rows * hq * qk_dim, seed));
        let kv = upload(model, &pseudo_random(positions * qk_dim, seed + 1));
        let m = model.mla.as_ref().unwrap();
        let out = if rows == 1 {
            model.mla_attention(m, &q, &kv.slice(..), hq, qk_dim, v_dim, positions, scale)
        } else {
            model.mla_attention_prefill(
                m,
                &q,
                &kv.slice(..),
                hq,
                qk_dim,
                v_dim,
                start_pos,
                rows,
                scale,
            )
        }
        .expect("mla attention");
        download(model, &out)
    };

    for (i, &(start_pos, rows)) in [(0, 1), (300, 1), (2047, 1), (0, 40), (64, 129)]
        .iter()
        .enumerate()
    {
        let legacy = run(
            &mut model,
            AttnImpl::Legacy,
            start_pos,
            rows,
            100 + i as u64,
        );
        let online = run(
            &mut model,
            AttnImpl::Online,
            start_pos,
            rows,
            100 + i as u64,
        );
        let diff = max_abs_diff(&legacy, &online);
        println!("mla qk_dim={qk_dim} v_dim={v_dim} start_pos={start_pos} rows={rows}: max abs diff {diff:e}");
        assert!(
            diff < KERNEL_TOL,
            "start_pos={start_pos} rows={rows}: {diff:e}"
        );
    }
}

/// End to end on whatever model `REFLEX_TEST_GGUF` points at (any architecture):
/// greedy tokens from the legacy and online kernels must be identical, and the
/// first-token logits must agree to within 1e-4. The prompt is long enough that
/// decode crosses the split-K path.
///
/// Always loads `f32` weights, whatever `REFLEX_WEIGHTS` says: the attention
/// kernels read f32 Q/K/V either way, but with f16 weights every prefill GEMM
/// rounds its input to f16, which turns the kernels' ~1e-6 summation-order
/// difference into whole f16 ulps downstream. Measured on a T4 with Qwen3-0.6B
/// (541 tokens): legacy vs. online 2.1e-5 with f32 weights, 8.2e-3 with f16,
/// where f16 vs. f32 weights alone already differ by 1.3e-2.
#[test]
#[ignore]
fn online_attention_matches_legacy_end_to_end() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to any GGUF");
    let file = GgufFile::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F32,
        lora_adapter: None,
    };
    let mut model = Model::load_with_options(device, &file, &opts).expect("failed to load model");
    let prompt =
        "The quick brown fox jumps over the lazy dog while the river runs past the old mill. "
            .repeat(30);

    let mut run = |imp: AttnImpl| {
        model.attn_impl = imp;
        let mut first_logits = Vec::new();
        let (ids, _text) = model
            .generate(
                &prompt,
                16,
                None,
                &crate::sampling::SamplingParams::default(),
                |logits| first_logits = logits.to_vec(),
                |_, _| {},
            )
            .expect("generate");
        (ids, first_logits)
    };
    let (legacy_ids, legacy_logits) = run(AttnImpl::Legacy);
    let (online_ids, online_logits) = run(AttnImpl::Online);

    let diff = max_abs_diff(&legacy_logits, &online_logits);
    println!(
        "{path}: prompt tokens {}, first-token max abs logit diff {diff:e}, ids {online_ids:?}",
        model.encoded_prompt_len(&prompt).unwrap()
    );
    assert_eq!(legacy_ids, online_ids, "greedy tokens differ");
    assert!(diff < 1e-4, "first-token max abs logit diff {diff:e}");
}
