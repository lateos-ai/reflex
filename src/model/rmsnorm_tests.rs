//! `rmsnorm_kernel` (`kernels_cuda/rmsnorm.cu`) against a host emulation of its
//! arithmetic, bit for bit: the row's sum of squares as fused multiply-adds in index
//! order (what nvcc emits for `sum_sq += v * v`), then `x * rms_inv * weight`. The
//! one-block-per-row kernel must reproduce the one-thread-per-element kernel it
//! replaced exactly, so every architecture's outputs stay the same. `#[ignore]`d: it
//! needs a GPU and the tiny qwen3moe fixture (any model works; only its kernels are
//! used). The fused decode norms (`decode_ops.cu`) are checked against the unfused
//! kernels the same way, op by op and over whole decode steps.

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;
use rand::{Rng, SeedableRng};

const FIXTURE: &str = "test-data/tiny-qwen3moe.gguf";

fn rmsnorm_ref(x: &[f32], w: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let sum_sq = row.iter().fold(0.0f32, |acc, &v| v.mul_add(v, acc));
        let rms_inv = 1.0f32 / (sum_sq / hidden as f32 + eps).sqrt();
        for c in 0..hidden {
            out[r * hidden + c] = row[c] * rms_inv * w[c];
        }
    }
    out
}

/// Shapes on both sides of the shared-memory staging limit (47 KB = 12,032
/// floats), widths that aren't multiples of 32 or of the 256-thread block, Gemma 4's
/// hidden sizes and head dims, and a decode-sized single row.
#[test]
#[ignore]
fn rmsnorm_kernel_matches_host_bit_for_bit() {
    let file = GgufFile::open(FIXTURE).unwrap_or_else(|e| panic!("open {FIXTURE}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4e0e);
    for (rows, hidden) in [
        (1, 3840),
        (37, 3840),
        (300, 256),
        (9, 512),
        (5, 5376),
        (64, 33),
        (3, 1000),
        (2, 12_032),
        (2, 13_000),
        (1, 1024),
        (1, 2048),
        (16, 128),
        (3, 1002),
    ] {
        let x: Vec<f32> = (0..rows * hidden)
            .map(|_| rng.gen_range(-40.0..40.0))
            .collect();
        let w: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let eps = 1e-6;
        let x_dev = model.device.htod_sync_copy(&x).expect("htod x");
        let w_dev = model.device.htod_sync_copy(&w).expect("htod w");
        let got_dev = model
            .rmsnorm(&x_dev, &w_dev, rows, hidden, eps)
            .expect("rmsnorm");
        let got = model.device.dtoh_sync_copy(&got_dev).expect("dtoh");
        let want = rmsnorm_ref(&x, &w, rows, hidden, eps);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "rows={rows} hidden={hidden} element {i}: got {g}, want {e}"
            );
        }
    }
}

fn load_fixture() -> Model {
    let file = GgufFile::open(FIXTURE).unwrap_or_else(|e| panic!("open {FIXTURE}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    Model::load(device, &file).expect("failed to load model")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The fused decode norms (`decode_ops.cu`) against the unfused kernels they
/// replace, bit for bit: `add_rmsnorm_kernel` vs `add_kernel` + `rmsnorm_kernel`
/// (both the norm and the updated residual), `rmsnorm_pair_kernel` vs two
/// `rmsnorm_kernel` launches.
#[test]
#[ignore]
fn decode_fused_norms_match_unfused_bit_for_bit() {
    let model = load_fixture();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xf05e);
    let mut rand_vec =
        |n: usize, r: f32| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-r..r)).collect() };
    for hidden in [64, 1024, 1002, 2048, 5120] {
        let x = rand_vec(hidden, 40.0);
        let d = rand_vec(hidden, 40.0);
        let w = rand_vec(hidden, 2.0);
        let w_dev = model.device.htod_sync_copy(&w).expect("htod w");
        let d_dev = model.device.htod_sync_copy(&d).expect("htod d");
        let run = |on: bool| {
            model.decode_ops.set_enabled(on);
            let mut h = model.device.htod_sync_copy(&x).expect("htod x");
            let out = model
                .residual_rmsnorm(&mut h, &d_dev, &w_dev, hidden, 1e-6)
                .expect("residual_rmsnorm");
            (
                model.device.dtoh_sync_copy(&out).expect("dtoh"),
                model.device.dtoh_sync_copy(&h).expect("dtoh"),
            )
        };
        let (plain_out, plain_h) = run(false);
        let (fused_out, fused_h) = run(true);
        assert_eq!(bits(&fused_h), bits(&plain_h), "residual, hidden={hidden}");
        assert_eq!(bits(&fused_out), bits(&plain_out), "norm, hidden={hidden}");
    }
    for (rows1, rows2, width) in [(16, 8, 128), (32, 4, 64), (8, 8, 256), (3, 1, 30)] {
        let x1 = model
            .device
            .htod_sync_copy(&rand_vec(rows1 * width, 40.0))
            .expect("htod");
        let x2 = model
            .device
            .htod_sync_copy(&rand_vec(rows2 * width, 40.0))
            .expect("htod");
        let w1 = model
            .device
            .htod_sync_copy(&rand_vec(width, 2.0))
            .expect("htod");
        let w2 = model
            .device
            .htod_sync_copy(&rand_vec(width, 2.0))
            .expect("htod");
        let run = |on: bool| {
            model.decode_ops.set_enabled(on);
            let (a, b) = model
                .rmsnorm_pair(&x1, &w1, rows1, &x2, &w2, rows2, width, 1e-6)
                .expect("rmsnorm_pair");
            (
                model.device.dtoh_sync_copy(&a).expect("dtoh"),
                model.device.dtoh_sync_copy(&b).expect("dtoh"),
            )
        };
        let (plain1, plain2) = run(false);
        let (fused1, fused2) = run(true);
        assert_eq!(bits(&fused1), bits(&plain1), "pair first, width={width}");
        assert_eq!(bits(&fused2), bits(&plain2), "pair second, width={width}");
    }
}

/// Whole decode steps with and without the fused norms (the residual add carried
/// into the next layer's attention norm, Q/K-Norm in one launch): the full-vocab
/// logits of 8 greedy steps are bit-identical. A one-token generate (prefill only)
/// must not load the module.
#[test]
#[ignore]
fn decode_fused_steps_match_unfused_bit_for_bit() {
    let model = load_fixture();
    let prompt = "The lighthouse keeper climbed the stairs";
    model
        .generate(
            prompt,
            1,
            None,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("one-token generate");
    assert!(
        !model.decode_ops.loaded(),
        "a one-token generate loaded the decode-ops module"
    );
    let hidden_size = model.cfg.hidden_size;
    let eps = model.cfg.rmsnorm_eps;
    let run = |on: bool| -> Vec<Vec<u32>> {
        model.decode_ops.set_enabled(on);
        let (ids, hidden_batched, mut k, mut v, mut position) = model
            .prefill_dense_batched(prompt, None, 8)
            .expect("prefill");
        let hidden = model
            .last_row(&hidden_batched, ids.len(), hidden_size)
            .expect("last row");
        let mut logits = model
            .lm_head_logits(&hidden, hidden_size, eps)
            .expect("logits");
        let mut steps = Vec::new();
        for _ in 0..8 {
            let next = Model::argmax(&logits).expect("argmax");
            let h = model
                .forward_one_token_dense(next, position, &mut k, &mut v)
                .expect("decode step");
            position += 1;
            logits = model.lm_head_logits(&h, hidden_size, eps).expect("logits");
            steps.push(bits(&logits));
        }
        steps
    };
    let plain = run(false);
    let fused = run(true);
    assert!(model.decode_ops.loaded());
    for (i, (f, p)) in fused.iter().zip(&plain).enumerate() {
        assert!(f == p, "decode step {i}: fused logits differ from unfused");
    }
}
