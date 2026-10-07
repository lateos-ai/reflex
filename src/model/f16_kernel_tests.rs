//! GPU checks for the `--weights f16` kernels against host references, on
//! synthetic data so they need no GGUF file. `#[ignore]`d like every other
//! GPU test; run with `cargo test --release -- --ignored f16_`.

use super::*;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use rand::{Rng, SeedableRng};

fn rng() -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(0xf16)
}

/// Every `_f16` dequant kernel writes exactly `__float2half_rn` of what its
/// f32 twin writes, for every format, on random block bytes (random scales
/// included, so the comparison also covers overflow to inf and subnormals).
#[test]
#[ignore]
fn f16_dequant_kernels_round_the_f32_output_once() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let kernels = load_dequant_kernels(&device).expect("failed to load dequant kernels");
    let mut pipeline = WeightLoadPipeline::new(&device).expect("pipeline");
    let mut rng = rng();
    let formats = [
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q8_1,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q8K,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ2S,
        GgmlType::IQ3XXS,
        GgmlType::IQ3S,
        GgmlType::IQ1S,
        GgmlType::IQ1M,
        GgmlType::IQ4XS,
    ];
    for ggml_type in formats {
        let (_, block_bytes, block_elems) = kernels.f32.for_type(ggml_type).expect("format");
        let num_blocks = 97;
        let bytes: Vec<u8> = (0..num_blocks * block_bytes).map(|_| rng.gen()).collect();
        let count = (num_blocks * block_elems) as u64;
        let out32 = dequantize_tensor_to_device(&mut pipeline, &kernels, ggml_type, &bytes, count)
            .expect("f32 dequant");
        let out16 =
            dequantize_tensor_to_device_f16(&mut pipeline, &kernels, ggml_type, &bytes, count)
                .expect("f16 dequant");
        let host32 = device.dtoh_sync_copy(&out32).expect("dtoh");
        let host16 = device.dtoh_sync_copy(&out16).expect("dtoh");
        assert_eq!(host32.len(), host16.len());
        for (i, (&a, &b)) in host32.iter().zip(&host16).enumerate() {
            let want = half::f16::from_f32(a);
            if a.is_nan() {
                assert!(b.is_nan(), "{ggml_type:?} element {i}: f32 NaN, f16 {b}");
            } else {
                assert_eq!(
                    want.to_bits(),
                    b.to_bits(),
                    "{ggml_type:?} element {i}: f32 {a} rounds to {want}, kernel wrote {b}"
                );
            }
        }
        eprintln!("OK {ggml_type:?}: {} elements", host16.len());
    }
}

/// Host reference for one f16 row against an f32 vector, in f64.
fn dot_ref(x: &[f32], w: &[half::f16]) -> (f64, f64) {
    let mut sum = 0f64;
    let mut abs = 0f64;
    for (&a, &b) in x.iter().zip(w) {
        let p = a as f64 * b.to_f32() as f64;
        sum += p;
        abs += p.abs();
    }
    (sum, abs)
}

fn warp_per_row_cfg(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(8).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `gemv_f16_kernel` and `gemv_gather_f16_kernel` match a host f64 reference on
/// each load path: 16-byte (`in % 8 == 0`, aligned), `half2` (even `in`, or a
/// view offset that leaves rows 4- but not 16-byte aligned) and scalar (odd
/// `in`, or a 2-byte-aligned view).
#[test]
#[ignore]
fn f16_gemv_kernels_match_host_reference_on_every_load_path() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let [_, gemv_f16] = load_kernel_pair(
        &device,
        include_bytes!(env!("REFLEX_KERNEL_GEMV")),
        "gemv",
        ["gemv_kernel", "gemv_f16_kernel"],
    )
    .expect("gemv module");
    let [_, gather_f16] = load_kernel_pair(
        &device,
        include_bytes!(env!("REFLEX_KERNEL_GEMV_GATHER")),
        "gemv_gather",
        ["gemv_gather_kernel", "gemv_gather_f16_kernel"],
    )
    .expect("gemv_gather module");
    let mut rng = rng();

    // (in_features, out_features, element offset of the weight view)
    for (in_f, out_f, offset) in [
        (1024usize, 37usize, 0usize),
        (1026, 19, 0),
        (1023, 11, 0),
        (1024, 9, 1),
        (1024, 9, 2),
        (8, 3, 0),
    ] {
        let x: Vec<f32> = (0..in_f).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let w: Vec<half::f16> = (0..offset + in_f * out_f)
            .map(|_| half::f16::from_f32(rng.gen_range(-1.0..1.0)))
            .collect();
        let x_dev = device.htod_sync_copy(&x).expect("htod x");
        let w_dev = device.htod_sync_copy(&w).expect("htod w");
        let w_view = w_dev.slice(offset..);

        let mut y_dev = device.alloc_zeros::<f32>(out_f).expect("alloc y");
        unsafe {
            gemv_f16
                .function
                .clone()
                .launch(
                    warp_per_row_cfg(out_f),
                    (&x_dev, &w_view, &mut y_dev, in_f as u32, out_f as u32),
                )
                .expect("gemv_f16 launch");
        }
        let y = device.dtoh_sync_copy(&y_dev).expect("dtoh y");

        let rows: Vec<u32> = vec![(out_f - 1) as u32, 0, (out_f / 2) as u32];
        let rows_dev = device.htod_sync_copy(&rows).expect("htod rows");
        let mut g_dev = device.alloc_zeros::<f32>(rows.len()).expect("alloc g");
        unsafe {
            gather_f16
                .function
                .clone()
                .launch(
                    warp_per_row_cfg(rows.len()),
                    (
                        &x_dev,
                        &w_view,
                        &rows_dev,
                        &mut g_dev,
                        in_f as u32,
                        rows.len() as u32,
                    ),
                )
                .expect("gemv_gather_f16 launch");
        }
        let g = device.dtoh_sync_copy(&g_dev).expect("dtoh g");

        let w_rows = &w[offset..];
        for r in 0..out_f {
            let (want, abs) = dot_ref(&x, &w_rows[r * in_f..(r + 1) * in_f]);
            let tol = 1e-5 * abs + 1e-6;
            assert!(
                (y[r] as f64 - want).abs() <= tol,
                "gemv_f16 in={in_f} offset={offset} row {r}: got {} want {want}",
                y[r]
            );
        }
        for (j, &r) in rows.iter().enumerate() {
            assert_eq!(
                g[j].to_bits(),
                y[r as usize].to_bits(),
                "gemv_gather_f16 row {r} differs from gemv_f16 (same reduction, must be identical)"
            );
        }
        eprintln!("OK in={in_f} out={out_f} offset={offset}");
    }
}

/// `gemv_per_head_batch_f16_kernel` matches a host reference, with a strided
/// input like MLA's absorption step reads.
#[test]
#[ignore]
fn f16_gemv_per_head_batch_matches_host_reference() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let [_, k] = load_kernel_pair(
        &device,
        include_bytes!(env!("REFLEX_KERNEL_GEMV_PER_HEAD_BATCH")),
        "gemv_per_head_batch",
        [
            "gemv_per_head_batch_kernel",
            "gemv_per_head_batch_f16_kernel",
        ],
    )
    .expect("gemv_per_head_batch module");
    let mut rng = rng();
    let (rows, n_head, in_f, out_f) = (3usize, 4usize, 40usize, 24usize);
    // Each head's input is `in_f` wide inside a wider per-head stride, at an offset.
    let (head_stride, head_offset) = (in_f + 8, 5usize);
    let row_stride = n_head * head_stride;
    let x: Vec<f32> = (0..rows * row_stride)
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();
    let w: Vec<half::f16> = (0..n_head * in_f * out_f)
        .map(|_| half::f16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();
    let x_dev = device.htod_sync_copy(&x).expect("htod x");
    let w_dev = device.htod_sync_copy(&w).expect("htod w");
    let mut out_dev = device
        .alloc_zeros::<f32>(rows * n_head * out_f)
        .expect("alloc");
    let cfg = LaunchConfig {
        grid_dim: (1, n_head as u32, rows as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        k.function
            .clone()
            .launch(
                cfg,
                (
                    &x_dev,
                    &w_dev,
                    &mut out_dev,
                    rows as u32,
                    n_head as u32,
                    in_f as u32,
                    out_f as u32,
                    row_stride as u32,
                    head_stride as u32,
                    head_offset as u32,
                ),
            )
            .expect("launch");
    }
    let out = device.dtoh_sync_copy(&out_dev).expect("dtoh");
    for r in 0..rows {
        for h in 0..n_head {
            let xs = &x[r * row_stride + h * head_stride + head_offset..][..in_f];
            for j in 0..out_f {
                let wr = &w[(h * out_f + j) * in_f..][..in_f];
                let (want, abs) = dot_ref(xs, wr);
                let got = out[(r * n_head + h) * out_f + j] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * abs + 1e-6,
                    "row {r} head {h} out {j}: got {got} want {want}"
                );
            }
        }
    }
}

/// The f16 prefill GEMM (`cast_act_f16_kernel` + `cublasGemmEx`) against
/// row-by-row `gemv_f16_kernel` on a real dense/MoE model's layer-0 `attn_q`,
/// loaded with `--weights f16`. The reference rows are pre-rounded to f16 on
/// the host, so both paths multiply identical values and may differ only in
/// summation order: a transposed or mis-strided GemmEx call fails this by
/// orders of magnitude. Then a single out-of-range activation must show up in
/// `f16_activation_stats` as saturated.
/// `REFLEX_TEST_GGUF=<dense or MoE GGUF> cargo test --release -- --ignored f16_gemm_ex`
#[test]
#[ignore]
fn f16_gemm_ex_matches_rowwise_gemv_and_reports_saturation() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a dense or MoE GGUF to run this test");
    let file = crate::gguf::GgufFile::open(&gguf_path).expect("open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F16,
        lora_adapter: None,
    };
    let model = Model::load_with_options(device.clone(), &file, &opts).expect("load f16");
    assert!(
        model.hybrid.is_none() && model.mla.is_none(),
        "needs a dense/MoE model"
    );
    let attn_q = match &model.layers[0] {
        LayerWeights::Dense(l) => &l.attn_q,
        LayerWeights::Moe(l) => &l.attn_q,
        LayerWeights::Kolibri(l) => &l.attn_q,
    };
    assert_eq!(attn_q.dtype(), WeightsDtype::F16);
    let (in_f, out_f) = (attn_q.shape[0] as usize, attn_q.shape[1] as usize);

    let rows = 7;
    let mut rng = rng();
    let x: Vec<f32> = (0..rows * in_f)
        .map(|_| half::f16::from_f32(rng.gen_range(-4.0..4.0)).to_f32())
        .collect();
    let x_dev = device.htod_sync_copy(&x).expect("htod x");
    let y = device
        .dtoh_sync_copy(&model.gemm(&x_dev, attn_q, rows).expect("gemm"))
        .expect("dtoh y");
    for r in 0..rows {
        let row_dev = device
            .htod_sync_copy(&x[r * in_f..(r + 1) * in_f])
            .expect("htod row");
        let want = device
            .dtoh_sync_copy(&model.gemv(&row_dev, attn_q).expect("gemv"))
            .expect("dtoh row");
        for j in 0..out_f {
            let (a, b) = (y[r * out_f + j], want[j]);
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "row {r} out {j}: gemm_ex {a} vs gemv {b}"
            );
        }
    }
    let before = model
        .f16_activation_stats()
        .expect("stats")
        .expect("f16 model");
    assert_eq!(before.saturated, 0, "in-range inputs must not saturate");
    assert!(
        before.max_abs <= 4.0 && before.max_abs > 3.0,
        "max_abs {}",
        before.max_abs
    );

    let mut hot = x.clone();
    hot[3] = 1.0e5;
    let hot_dev = device.htod_sync_copy(&hot).expect("htod hot");
    let y_hot = device
        .dtoh_sync_copy(&model.gemm(&hot_dev, attn_q, rows).expect("gemm hot"))
        .expect("dtoh hot");
    assert!(
        y_hot.iter().all(|v| v.is_finite()),
        "saturating cast must not produce inf"
    );
    let after = model
        .f16_activation_stats()
        .expect("stats")
        .expect("f16 model");
    assert_eq!(after.saturated, 1);
    assert_eq!(after.max_abs, 1.0e5);
}
