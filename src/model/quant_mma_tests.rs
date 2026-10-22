//! Kernel-level tests for the quantized-weight tensor-core prefill GEMM
//! (`kernels_cuda/gemm_q_mma.cu`, docs/design/quantized-mma-prefill.md), on
//! synthetic Q4_K and Q6_K weights. They launch through a real dense Q4_K GGUF
//! loaded quantized-resident with `--weights f16` and set
//! `REFLEX_QUANT_RESIDENT` for the load, so run them single-threaded:
//! `REFLEX_TEST_GGUF=<dense Q4_K_M gguf> cargo test --release -- --ignored --test-threads=1 quant_mma`

use super::quant_resident_tests::{
    activations, host_matmul, load_model, rel_max_diff, synthetic_q4k,
};
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// The weight types the tensor-core path covers.
const TYPES: [GgmlType; 2] = [GgmlType::Q4K, GgmlType::Q6K];

/// Shapes `[in_features, out_features]`: the smallest legal row, outputs that
/// leave the last 64-row block and the last 8-row tile partly empty, and the
/// Qwen3-0.6B and Mistral 7B layer shapes. Q6_K's 210-byte blocks start at
/// every even offset within a 16-byte chunk in each of them.
const SHAPES: [(usize, usize); 10] = [
    (256, 40),
    (1024, 136),
    (512, 37),
    (768, 24),
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
/// weights, with the tensor-core path available at every row count (no row
/// cap, [`quant_mma_max_rows`]). `None` (with a skip message) on a device
/// below sm_75.
fn test_model(device: &Arc<CudaDevice>) -> Option<Model> {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a dense Q4_K GGUF to run this test");
    let file = GgufFile::open(&path).expect("open REFLEX_TEST_GGUF");
    let model = load_model(device.clone(), &file, WeightsDtype::F16, true);
    if !TYPES.iter().all(|&ty| model.quant_mma.applies(device, ty)) {
        eprintln!("Skipping: the quantized MMA path needs sm_75 or newer");
        return None;
    }
    assert!(
        model.dequant_q4k_coalesced_f16_k.is_some() && model.dequant_q6k_coalesced_f16_k.is_some(),
        "QR model must load the Q4_K and Q6_K scratch dequant kernels"
    );
    model.quant_mma.set_max_rows(Some(usize::MAX));
    Some(model)
}

/// Synthetic Q6_K blocks for an `[in_f, out_f]` weight: random quants and
/// signed 8-bit scales, a small f16 `d` that is sometimes negative or zero.
fn synthetic_q6k(in_f: usize, out_f: usize, seed: u64) -> Vec<u8> {
    let blocks = in_f / QK_K * out_f;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut out = Vec::with_capacity(blocks * Q6K_BLOCK_BYTES);
    for b in 0..blocks {
        out.extend((0..Q6K_BLOCK_BYTES - 2).map(|_| next() as u8));
        let d = match b % 16 {
            0 => 0.0,
            1 => -2e-4,
            _ => ((next() >> 40) as f32 / (1u64 << 24) as f32) * 4e-4 + 1e-6,
        };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    out
}

fn synthetic(ty: GgmlType, in_f: usize, out_f: usize, seed: u64) -> Vec<u8> {
    match ty {
        GgmlType::Q4K => synthetic_q4k(in_f, out_f, seed),
        GgmlType::Q6K => synthetic_q6k(in_f, out_f, seed),
        _ => unreachable!("{ty:?}"),
    }
}

/// A quantized-resident [`Weight`] in its own device buffer (256-byte aligned,
/// like an arena).
fn quant_weight(
    device: &Arc<CudaDevice>,
    ty: GgmlType,
    bytes: &[u8],
    in_f: usize,
    out_f: usize,
) -> Weight {
    Weight {
        data: WeightData::Quant {
            ty,
            arena: Arc::new(device.htod_sync_copy(bytes).expect("upload blocks")),
            offset: 0,
            len: bytes.len(),
        },
        shape: vec![in_f as u64, out_f as u64],
    }
}

/// The scratch path's f16 weights: `dequantize_q4k/q6k_coalesced_f16_kernel` on
/// `wq`, as an f16 [`Weight`], so `Model::gemm` on it is the scratch path's
/// `cublasGemmEx` call for any row count.
fn scratch_weight(model: &Model, wq: &Weight) -> Weight {
    let WeightData::Quant { ty, arena, len, .. } = &wq.data else {
        panic!("not quantized");
    };
    let (k, block_bytes) = match ty {
        GgmlType::Q4K => (&model.dequant_q4k_coalesced_f16_k, Q4K_BLOCK_BYTES),
        GgmlType::Q6K => (&model.dequant_q6k_coalesced_f16_k, Q6K_BLOCK_BYTES),
        _ => unreachable!("{ty:?}"),
    };
    let k = k.as_ref().expect("loaded");
    let num_blocks = len / block_bytes;
    let mut out = model
        .device
        .alloc_zeros::<half::f16>(num_blocks * QK_K)
        .expect("alloc scratch");
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

/// The f16 weights the GEMM's lanes build (`q4k/q6k_mma_dequant_check_kernel`,
/// which run the GEMM's B-fragment code and write each value where its
/// fragment position says it belongs) are bit-identical to the scratch path's
/// `dequantize_*_coalesced_f16_kernel` output: same values, and every fragment
/// maps to the right `(row, k)`.
#[test]
#[ignore]
fn quant_mma_dequant_bit_exact() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let kernels = model.quant_mma.all(&device).expect("load module");
    for ty in TYPES {
        let check = match ty {
            GgmlType::Q4K => &kernels[QUANT_MMA_Q4K_CHECK],
            _ => &kernels[QUANT_MMA_Q6K_CHECK],
        };
        for (k, &(in_f, out_f)) in SHAPES.iter().enumerate() {
            let bytes = synthetic(ty, in_f, out_f, k as u64 + 11);
            let wq = quant_weight(&device, ty, &bytes, in_f, out_f);
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
                    "{ty:?} {in_f}x{out_f}: weight row {} k {}",
                    i / in_f,
                    i % in_f
                );
            }
        }
    }
}

/// `Model::gemm` on a quantized weight with the tensor-core path against the
/// same call on the scratch path's f16 weights (`cublasGemmEx`, f32 compute):
/// the inputs are identical, so only summation order may differ. For the
/// smaller shapes, both are also compared with an f64 host product of those
/// same f16 inputs: the kernel's relative L2 error must stay below
/// `3e-9 * in_features` (at least 1e-6; the worst case per shape is printed
/// next to cuBLAS's).
#[test]
#[ignore]
fn quant_mma_matches_f16_gemm() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    for ty in TYPES {
        for (k, &(in_f, out_f)) in SHAPES.iter().enumerate() {
            let bytes = synthetic(ty, in_f, out_f, k as u64 + 21);
            let wq = quant_weight(&device, ty, &bytes, in_f, out_f);
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
                    "{ty:?} {in_f}x{out_f} rows={rows}: non-finite"
                );
                let e = rel_l2(&got, &want);
                assert!(
                    e < 1e-5,
                    "{ty:?} {in_f}x{out_f} rows={rows}: relative L2 vs cuBLAS {e:e} (max rel diff {})",
                    rel_max_diff(&got, &want)
                );
                if in_f * out_f * rows <= 1024 * 3072 * 33 {
                    let x16: Vec<f32> =
                        x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
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
                        "{ty:?} {in_f}x{out_f} rows={rows}: kernel error {e_mma:e} vs cuBLAS {e_cublas:e}"
                    );
                }
            }
            if worst > 0.0 {
                eprintln!(
                    "{ty:?} {in_f}x{out_f}: worst relative L2 vs f64: kernel {worst:e}, cuBLAS {worst_cublas:e}"
                );
            }
        }
    }
}

/// For a fixed split-K count, a row's output doesn't depend on how many rows
/// share the call (the tile shape and summation order don't depend on `rows`),
/// and repeated calls give the same bits, with the picked split count too. The
/// shape has 12 super-blocks per row, so 3 splits of 4 and 5 forced splits
/// (4 of 3) both run.
#[test]
#[ignore]
fn quant_mma_rows_independent_and_deterministic() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let (in_f, out_f, all) = (3072usize, 1024usize, 129usize);
    for ty in TYPES {
        let wq = quant_weight(&device, ty, &synthetic(ty, in_f, out_f, 5), in_f, out_f);
        let x = activations(all * in_f, 77);
        for rows in [2usize, 29, all] {
            let part = &x[..rows * in_f];
            assert!(
                bits_equal(
                    &gemm_host(&model, part, &wq, rows),
                    &gemm_host(&model, part, &wq, rows)
                ),
                "{ty:?} rows={rows}: repeated call differs"
            );
        }
        for splits in [1usize, 3, 5] {
            model.quant_mma.set_splits(Some(splits));
            let full = gemm_host(&model, &x, &wq, all);
            for rows in [2usize, 16, 17, 33, 64, 65, 128] {
                let part = gemm_host(&model, &x[..rows * in_f], &wq, rows);
                assert!(
                    bits_equal(&part, &full[..rows * out_f]),
                    "{ty:?} splits={splits}: first {rows} rows differ from the same rows in a {all}-row call"
                );
            }
        }
        model.quant_mma.set_splits(None);
    }
}

/// Split-K only changes the summation order: every forced split count matches
/// the unsplit kernel within the tensor cores' rounding (relative L2 1.1e-6 at
/// K = 1024 on an A10G; the same bound as against cuBLAS), including counts that don't divide the
/// super-blocks evenly and one per super-block.
#[test]
#[ignore]
fn quant_mma_split_k_matches_unsplit() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    for ty in TYPES {
        for (k, &(in_f, out_f)) in [(1024usize, 1024usize), (3072, 1024), (4096, 136)]
            .iter()
            .enumerate()
        {
            let wq = quant_weight(
                &device,
                ty,
                &synthetic(ty, in_f, out_f, k as u64 + 31),
                in_f,
                out_f,
            );
            for rows in [2usize, 29, 113] {
                let x = activations(rows * in_f, (k * 100 + rows) as u64);
                model.quant_mma.set_splits(Some(1));
                let want = gemm_host(&model, &x, &wq, rows);
                for splits in [2usize, 3, 5, in_f / QK_K] {
                    model.quant_mma.set_splits(Some(splits));
                    let got = gemm_host(&model, &x, &wq, rows);
                    let e = rel_l2(&got, &want);
                    assert!(
                        e < 1e-5,
                        "{ty:?} {in_f}x{out_f} rows={rows} splits={splits}: relative L2 vs unsplit {e:e}"
                    );
                }
            }
        }
        model.quant_mma.set_splits(None);
    }
}

/// The split-K rule ([`quant_mma_splits`]): no empty split, about two blocks
/// per SM for narrow outputs or a long K, none once the grid covers half the
/// SMs, and forced counts clamped to the super-block count. Host only.
#[test]
fn quant_mma_split_rule() {
    // A10G (80 SMs): Qwen3-0.6B's o_proj at 29 rows has 16 blocks, so 10
    // splits are wanted, capped at 12 / 2.
    assert_eq!(quant_mma_splits(3072, 1024, 29, 80, None), (6, 2));
    assert_eq!(quant_mma_splits(1024, 1024, 12, 80, None), (2, 2));
    // Mistral's ffn_down at 29 rows: 64 blocks, 3 splits of 19, 19, 18.
    assert_eq!(quant_mma_splits(14336, 4096, 29, 80, None), (3, 19));
    // T4 (40 SMs): 5 splits wanted, 6 super-blocks max; ceil(12 / 5) = 3 per
    // split leaves 4 splits.
    assert_eq!(quant_mma_splits(3072, 1024, 29, 40, None), (4, 3));
    // 64 blocks cover half of 80 SMs: no split with a short K (Mistral's
    // q/o projections), or at 16 rows or fewer.
    assert_eq!(quant_mma_splits(4096, 4096, 29, 80, None), (1, 16));
    assert_eq!(quant_mma_splits(14336, 4096, 12, 80, None), (1, 56));
    assert_eq!(quant_mma_splits(1024, 3072, 29, 80, None), (1, 4));
    // Long prefills and wide outputs fill the GPU unsplit.
    assert_eq!(quant_mma_splits(4096, 14336, 449, 80, None), (1, 16));
    assert_eq!(quant_mma_splits(3072, 1024, 449, 80, None), (1, 12));
    assert_eq!(quant_mma_splits(4096, 4096, 113, 80, None), (1, 16));
    // Forced counts.
    assert_eq!(quant_mma_splits(3072, 1024, 29, 80, Some(5)), (4, 3));
    assert_eq!(quant_mma_splits(1024, 1024, 29, 80, Some(100)), (4, 1));
    assert_eq!(quant_mma_splits(256, 40, 2, 80, None), (1, 1));
    for in_f in (256..=14336).step_by(256) {
        for out_f in [8usize, 40, 1024, 4096, 14336] {
            for rows in [2usize, 13, 64, 65, 449] {
                for sms in [40usize, 80, 132] {
                    let nkb = in_f / QK_K;
                    let (s, kps) = quant_mma_splits(in_f, out_f, rows, sms, None);
                    assert!(s >= 1 && kps >= 1, "{in_f}x{out_f} {rows} {sms}");
                    assert!(
                        s * kps >= nkb && (s - 1) * kps < nkb,
                        "{in_f}x{out_f} {rows} {sms}: ({s}, {kps}) empty split"
                    );
                }
            }
        }
    }
}

/// The row cap ([`quant_mma_max_rows`]): per type on sm_75, none from sm_80
/// up, and `REFLEX_QUANT_MMA_MAX_ROWS` on every device. Host only.
#[test]
fn quant_mma_row_cap_rule() {
    let (q4, q6) = (GgmlType::Q4K, GgmlType::Q6K);
    // T4: Q4_K through 384 rows, Q6_K through 256.
    assert_eq!(quant_mma_max_rows(q4, (7, 5), None), 384);
    assert_eq!(quant_mma_max_rows(q6, (7, 5), None), 256);
    // Ampere and newer: no cap.
    for cc in [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)] {
        assert_eq!(quant_mma_max_rows(q4, cc, None), usize::MAX);
        assert_eq!(quant_mma_max_rows(q6, cc, None), usize::MAX);
    }
    // The override wins everywhere.
    assert_eq!(quant_mma_max_rows(q6, (7, 5), Some(1000)), 1000);
    assert_eq!(quant_mma_max_rows(q4, (8, 6), Some(128)), 128);
    // The bench's long buckets take the scratch path on a T4, the medium ones
    // (113 / 129 rows) stay on the kernel.
    for ty in [q4, q6] {
        let cap = quant_mma_max_rows(ty, (7, 5), None);
        assert!((129..449).contains(&cap), "{ty:?}: {cap}");
    }
}

/// Activations beyond f16's range saturate in the shared cast, so the result
/// stays finite and matches the scratch path.
#[test]
#[ignore]
fn quant_mma_saturates_like_scratch() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = test_model(&device) else {
        return;
    };
    let (in_f, out_f, rows) = (1024usize, 1024usize, 13usize);
    for ty in TYPES {
        let wq = quant_weight(&device, ty, &synthetic(ty, in_f, out_f, 9), in_f, out_f);
        let wd = scratch_weight(&model, &wq);
        let mut x = activations(rows * in_f, 3);
        x[5] = 1.0e5;
        x[in_f + 17] = -2.0e6;
        let got = gemm_host(&model, &x, &wq, rows);
        let want = gemm_host(&model, &x, &wd, rows);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{ty:?}: non-finite output"
        );
        let e = rel_l2(&got, &want);
        assert!(e < 1e-5, "{ty:?}: relative L2 vs scratch path {e:e}");
    }
}

/// The go/no-go numbers (docs/design/quantized-mma-prefill.md): per weight
/// type, layer shape and row count, the tensor-core kernel against today's QR
/// path (Q4_K: fused GEMV up to `quant_fused_max_rows()`, then dequantize to
/// scratch and cuBLAS; Q6_K: always scratch) and against f16 cuBLAS on
/// pre-dequantized weights (the `REFLEX_QUANT_RESIDENT=0` path). Each timing
/// cycles through enough weight copies to miss the L2, as a prefill's layers
/// do. Prints only.
#[test]
#[ignore]
fn quant_mma_prefill_timings() {
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
    eprintln!(
        "type shape rows: mma us (unsplit us) | QR today us | f16 cuBLAS us (quantized MB, splits)"
    );
    for ty in TYPES {
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
            let bytes = synthetic(ty, in_f, out_f, (in_f + out_f) as u64);
            let copies = (48_000_000 / (in_f * out_f * 2)).clamp(2, 64);
            let wqs: Vec<Weight> = (0..copies)
                .map(|_| quant_weight(dev, ty, &bytes, in_f, out_f))
                .collect();
            let wds: Vec<Weight> = wqs.iter().map(|w| scratch_weight(&model, w)).collect();
            for rows in [12usize, 16, 29, 33, 64, 113, 449] {
                let x = dev
                    .htod_sync_copy(&activations(rows * in_f, rows as u64))
                    .unwrap();
                let mma = time(&|i| {
                    model.gemm(&x, &wqs[i % copies], rows).unwrap();
                });
                model.quant_mma.set_splits(Some(1));
                let unsplit = time(&|i| {
                    model.gemm(&x, &wqs[i % copies], rows).unwrap();
                });
                model.quant_mma.set_splits(None);
                model.quant_mma.set_enabled(false);
                let today = time(&|i| {
                    model.gemm(&x, &wqs[i % copies], rows).unwrap();
                });
                model.quant_mma.set_enabled(true);
                let f16 = time(&|i| {
                    model.gemm(&x, &wds[i % copies], rows).unwrap();
                });
                let (splits, _) = model.quant_mma.splits(dev, in_f, out_f, rows);
                eprintln!(
                    "{ty:?} {in_f}x{out_f} {rows}: {mma:.1} ({unsplit:.1}) | {today:.1} | {f16:.1} ({:.2} MB, {splits} splits)",
                    bytes.len() as f64 / 1e6
                );
            }
        }
    }
}

/// Where the tensor-core kernel stops beating the scratch path as rows grow
/// (the row cap, [`quant_mma_max_rows`]): per type and the Qwen3-0.6B and
/// Mistral 7B layer shapes, the kernel past any cap against the scratch path
/// (`REFLEX_QUANT_MMA=0`), and which one `Model::gemm` picks with the built-in
/// cap. Prints only.
#[test]
#[ignore]
fn quant_mma_row_crossover() {
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
        let iters = 30;
        let t = std::time::Instant::now();
        for i in 0..iters {
            f(i);
        }
        dev.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    eprintln!("crossover: type shape rows: mma us | scratch us | mma/scratch | capped path");
    let caps = TYPES.map(|ty| {
        model.quant_mma.set_max_rows(None);
        let cap = model.quant_mma.max_rows(dev, ty);
        model.quant_mma.set_max_rows(Some(usize::MAX));
        cap
    });
    for (ty, cap) in TYPES.into_iter().zip(caps) {
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
            let bytes = synthetic(ty, in_f, out_f, (in_f + out_f) as u64);
            let copies = (48_000_000 / (in_f * out_f * 2)).clamp(2, 64);
            let wqs: Vec<Weight> = (0..copies)
                .map(|_| quant_weight(dev, ty, &bytes, in_f, out_f))
                .collect();
            for rows in [113usize, 160, 192, 256, 320, 384, 449, 513] {
                let x = dev
                    .htod_sync_copy(&activations(rows * in_f, rows as u64))
                    .unwrap();
                let mma = time(&|i| {
                    model.gemm(&x, &wqs[i % copies], rows).unwrap();
                });
                model.quant_mma.set_enabled(false);
                let scratch = time(&|i| {
                    model.gemm(&x, &wqs[i % copies], rows).unwrap();
                });
                model.quant_mma.set_enabled(true);
                eprintln!(
                    "crossover: {ty:?} {in_f}x{out_f} {rows}: {mma:.1} | {scratch:.1} | {:.2} | {}",
                    mma / scratch,
                    if rows > cap { "scratch" } else { "mma" }
                );
            }
        }
    }
}
