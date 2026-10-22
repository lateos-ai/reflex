//! Kernel-level tests for the quantized-weight tensor-core prefill GEMM
//! (`kernels_cuda/gemm_q_mma.cu`, docs/design/quantized-mma-prefill.md), on
//! synthetic Q4_K weights. They launch through a real dense Q4_K GGUF loaded
//! quantized-resident with `--weights f16` and set `REFLEX_QUANT_RESIDENT` for
//! the load, so run them single-threaded:
//! `REFLEX_TEST_GGUF=<dense Q4_K_M gguf> cargo test --release -- --ignored --test-threads=1 quant_mma`

use super::quant_resident_tests::{
    activations, host_matmul, load_model, rel_max_diff, synthetic_q4k,
};
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Shapes `[in_features, out_features]`: the smallest legal row, outputs that
/// leave the last 64-row block and the last 8-row tile partly empty, and the
/// Qwen3-0.6B and Mistral 7B layer shapes.
const SHAPES: [(usize, usize); 9] = [
    (256, 40),
    (1024, 136),
    (512, 37),
    (1024, 1024),
    (1024, 2048),
    (1024, 3072),
    (3072, 1024),
    (4096, 1024),
    (4096, 4096),
];

/// Row counts around every m-tile boundary (16, 32, 48, 64) and the fused-GEMV
/// threshold, plus the bench buckets.
const ROWS: [usize; 14] = [2, 5, 8, 12, 13, 16, 17, 29, 33, 48, 64, 65, 113, 129];

/// The test model: `REFLEX_TEST_GGUF` loaded quantized-resident with f16
/// weights, with the tensor-core path available. `None` (with a skip message)
/// on a device below sm_75.
fn test_model(device: &Arc<CudaDevice>) -> Option<Model> {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a dense Q4_K GGUF to run this test");
    let file = GgufFile::open(&path).expect("open REFLEX_TEST_GGUF");
    let model = load_model(device.clone(), &file, WeightsDtype::F16, true);
    if !model.quant_mma.applies(device, GgmlType::Q4K) {
        eprintln!("Skipping: the quantized MMA path needs sm_75 or newer");
        return None;
    }
    assert!(
        model.dequant_q4k_coalesced_f16_k.is_some(),
        "QR model must load the Q4_K scratch dequant"
    );
    Some(model)
}

/// A quantized-resident Q4_K [`Weight`] in its own device buffer (256-byte
/// aligned, like an arena).
fn q4k_weight(device: &Arc<CudaDevice>, bytes: &[u8], in_f: usize, out_f: usize) -> Weight {
    Weight {
        data: WeightData::Quant {
            ty: GgmlType::Q4K,
            arena: Arc::new(device.htod_sync_copy(bytes).expect("upload Q4_K blocks")),
            offset: 0,
            len: bytes.len(),
        },
        shape: vec![in_f as u64, out_f as u64],
    }
}

/// The scratch path's f16 weights: `dequantize_q4k_coalesced_f16_kernel` on
/// `wq`, as an f16 [`Weight`], so `Model::gemm` on it is the scratch path's
/// `cublasGemmEx` call for any row count.
fn scratch_weight(model: &Model, wq: &Weight) -> Weight {
    let WeightData::Quant { arena, len, .. } = &wq.data else {
        panic!("not quantized");
    };
    let num_blocks = len / Q4K_BLOCK_BYTES;
    let mut out = model
        .device
        .alloc_zeros::<half::f16>(num_blocks * QK_K)
        .expect("alloc scratch");
    let k = model.dequant_q4k_coalesced_f16_k.as_ref().expect("loaded");
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (QK_K as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        k.function
            .clone()
            .launch(cfg, (&**arena, &mut out, num_blocks as u32))
            .expect("launch scratch dequant");
    }
    Weight {
        data: WeightData::F16(out),
        shape: wq.shape.clone(),
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64 - *y as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(f64::MIN_POSITIVE)).sqrt()
}

fn gemm_host(model: &Model, x: &[f32], w: &Weight, rows: usize) -> Vec<f32> {
    let dev = &model.device;
    let x = dev.htod_sync_copy(x).expect("upload x");
    dev.dtoh_sync_copy(&model.gemm(&x, w, rows).expect("gemm"))
        .expect("dtoh")
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The f16 weights the GEMM's lanes build (`q4k_mma_dequant_check_kernel`,
/// which runs the GEMM's B-fragment code and writes each value where its
/// fragment position says it belongs) are bit-identical to the scratch
/// path's `dequantize_q4k_coalesced_f16_kernel` output: same values, and every
/// fragment maps to the right `(row, k)`.
#[test]
#[ignore]
fn quant_mma_q4k_dequant_bit_exact() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let check = &model.quant_mma.all(&device).expect("load module")[QUANT_MMA_Q4K_CHECK];
    for (k, &(in_f, out_f)) in SHAPES.iter().enumerate() {
        let bytes = synthetic_q4k(in_f, out_f, k as u64 + 11);
        let wq = q4k_weight(&device, &bytes, in_f, out_f);
        let want = scratch_weight(&model, &wq);
        let WeightData::F16(want) = &want.data else {
            unreachable!()
        };
        let want = device.dtoh_sync_copy(want).expect("dtoh");
        let src = device.htod_sync_copy(&bytes).expect("upload");
        let mut out = device
            .alloc_zeros::<u16>(in_f * out_f)
            .expect("alloc check out");
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32).div_ceil(8), (in_f / QK_K) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            check
                .function
                .clone()
                .launch(cfg, (&src, &mut out, in_f as u32, out_f as u32))
                .expect("launch check");
        }
        let got = device.dtoh_sync_copy(&out).expect("dtoh");
        for i in 0..in_f * out_f {
            assert_eq!(
                got[i],
                want[i].to_bits(),
                "{in_f}x{out_f}: weight row {} k {}",
                i / in_f,
                i % in_f
            );
        }
    }
}

/// `Model::gemm` on a Q4_K weight with the tensor-core path against the same
/// call on the scratch path's f16 weights (`cublasGemmEx`, f32 compute): the
/// inputs are identical, so only summation order may differ. For the smaller
/// shapes, both are also compared with an f64 host product of those same f16
/// inputs: the kernel's relative L2 error must stay below `3e-9 * in_features`
/// (at least 1e-6; the worst case per shape is printed next to cuBLAS's).
#[test]
#[ignore]
fn quant_mma_q4k_matches_f16_gemm() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    for (k, &(in_f, out_f)) in SHAPES.iter().enumerate() {
        let bytes = synthetic_q4k(in_f, out_f, k as u64 + 21);
        let wq = q4k_weight(&device, &bytes, in_f, out_f);
        let wd = scratch_weight(&model, &wq);
        let w16 = match &wd.data {
            WeightData::F16(s) => device.dtoh_sync_copy(s).expect("dtoh"),
            _ => unreachable!(),
        };
        let w32: Vec<f32> = w16.iter().map(|v| v.to_f32()).collect();
        let (mut worst, mut worst_cublas) = (0.0f64, 0.0f64);
        for rows in ROWS {
            let x = activations(rows * in_f, (k * 1000 + rows) as u64);
            let got = gemm_host(&model, &x, &wq, rows);
            let want = gemm_host(&model, &x, &wd, rows);
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{in_f}x{out_f} rows={rows}: non-finite"
            );
            let e = rel_l2(&got, &want);
            assert!(
                e < 1e-5,
                "{in_f}x{out_f} rows={rows}: relative L2 vs cuBLAS {e:e} (max rel diff {})",
                rel_max_diff(&got, &want)
            );
            if in_f * out_f * rows <= 1024 * 3072 * 33 {
                let x16: Vec<f32> = x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
                let exact = host_matmul(&x16, &w32, rows, in_f, out_f);
                let (e_mma, e_cublas) = (rel_l2(&got, &exact), rel_l2(&want, &exact));
                worst = worst.max(e_mma);
                worst_cublas = worst_cublas.max(e_cublas);
                // Not a multiple of cuBLAS's error: at a few rows cuBLAS picks
                // CUDA-core kernels with plain f32 FMAs, while mma.sync adds
                // inside the tensor core at lower internal precision (1.5e-6
                // vs 1.8e-7 at 1024x136, 2 rows, on an A10G). That error grows
                // about linearly with in_features (A10G worst: 2.0e-6 at 1024,
                // 5.5e-6 at 3072, 6.9e-6 at 4096), hence a bound in K. All of
                // it sits far below the f16 rounding of the inputs (~3e-4).
                let bound = (3e-9 * in_f as f64).max(1e-6);
                assert!(
                    e_mma < bound,
                    "{in_f}x{out_f} rows={rows}: kernel error {e_mma:e} vs cuBLAS {e_cublas:e}"
                );
            }
        }
        if worst > 0.0 {
            eprintln!(
                "{in_f}x{out_f}: worst relative L2 vs f64: kernel {worst:e}, cuBLAS {worst_cublas:e}"
            );
        }
    }
}

/// A row's output doesn't depend on how many rows share the call (the tile
/// shape and summation order don't depend on `rows`), and repeated calls give
/// the same bits.
#[test]
#[ignore]
fn quant_mma_q4k_rows_independent_and_deterministic() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let (in_f, out_f, all) = (1024usize, 3072usize, 129usize);
    let wq = q4k_weight(&device, &synthetic_q4k(in_f, out_f, 5), in_f, out_f);
    let x = activations(all * in_f, 77);
    let full = gemm_host(&model, &x, &wq, all);
    assert!(
        bits_equal(&full, &gemm_host(&model, &x, &wq, all)),
        "repeated call differs"
    );
    for rows in [2usize, 16, 17, 33, 64, 65, 128] {
        let part = gemm_host(&model, &x[..rows * in_f], &wq, rows);
        assert!(
            bits_equal(&part, &full[..rows * out_f]),
            "first {rows} rows differ from the same rows in a {all}-row call"
        );
    }
}

/// Activations beyond f16's range saturate in the shared cast, so the result
/// stays finite and matches the scratch path.
#[test]
#[ignore]
fn quant_mma_q4k_saturates_like_scratch() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let (in_f, out_f, rows) = (1024usize, 1024usize, 13usize);
    let wq = q4k_weight(&device, &synthetic_q4k(in_f, out_f, 9), in_f, out_f);
    let wd = scratch_weight(&model, &wq);
    let mut x = activations(rows * in_f, 3);
    x[5] = 1.0e5;
    x[in_f + 17] = -2.0e6;
    let got = gemm_host(&model, &x, &wq, rows);
    let want = gemm_host(&model, &x, &wd, rows);
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    let e = rel_l2(&got, &want);
    assert!(e < 1e-5, "relative L2 vs scratch path {e:e}");
}

/// The step-2 go/no-go numbers (docs/design/quantized-mma-prefill.md): per
/// layer shape and row count, the tensor-core kernel against today's QR path
/// (fused GEMV up to `quant_fused_max_rows()`, dequantize-to-scratch + cuBLAS
/// above) and against f16 cuBLAS on pre-dequantized weights (the
/// `REFLEX_QUANT_RESIDENT=0` path). Each timing cycles through enough weight
/// copies to miss the L2, as a prefill's layers do. Prints only.
#[test]
#[ignore]
fn quant_mma_q4k_prefill_timings() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let dev = &model.device;
    let time = |f: &dyn Fn(usize)| {
        for i in 0..3 {
            f(i);
        }
        dev.synchronize().unwrap();
        let iters = 50;
        let t = std::time::Instant::now();
        for i in 0..iters {
            f(i);
        }
        dev.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    eprintln!("shape rows: mma us | QR today us | f16 cuBLAS us (Q4_K MB)");
    for (in_f, out_f) in [
        (1024usize, 2048usize),
        (1024, 1024),
        (1024, 3072),
        (3072, 1024),
        (4096, 4096),
        (4096, 1024),
        (4096, 14336),
        (14336, 4096),
    ] {
        let bytes = synthetic_q4k(in_f, out_f, (in_f + out_f) as u64);
        let copies = (48_000_000 / (in_f * out_f * 2)).clamp(2, 64);
        let wqs: Vec<Weight> = (0..copies)
            .map(|_| q4k_weight(dev, &bytes, in_f, out_f))
            .collect();
        let wds: Vec<Weight> = wqs.iter().map(|w| scratch_weight(&model, w)).collect();
        for rows in [12usize, 16, 29, 33, 64, 113, 449] {
            let x = dev
                .htod_sync_copy(&activations(rows * in_f, rows as u64))
                .unwrap();
            let mma = time(&|i| {
                model.gemm(&x, &wqs[i % copies], rows).unwrap();
            });
            model.quant_mma.set_enabled(false);
            let today = time(&|i| {
                model.gemm(&x, &wqs[i % copies], rows).unwrap();
            });
            model.quant_mma.set_enabled(true);
            let f16 = time(&|i| {
                model.gemm(&x, &wds[i % copies], rows).unwrap();
            });
            eprintln!(
                "{in_f}x{out_f} {rows}: {mma:.1} | {today:.1} | {f16:.1} ({:.2} MB)",
                bytes.len() as f64 / 1e6
            );
        }
    }
}
