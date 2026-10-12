//! `rmsnorm_kernel` (`kernels_cuda/rmsnorm.cu`) against a host emulation of its
//! arithmetic, bit for bit: the row's sum of squares as fused multiply-adds in index
//! order (what nvcc emits for `sum_sq += v * v`), then `x * rms_inv * weight`. The
//! one-block-per-row kernel must reproduce the one-thread-per-element kernel it
//! replaced exactly, so every architecture's outputs stay the same. `#[ignore]`d: it
//! needs a GPU and the tiny qwen3moe fixture (any model works; only its kernels are
//! used).

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
