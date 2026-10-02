use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Max |a - b| over max |b|, the comparison the GEMV tests below use.
fn rel_max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// Deterministic pseudo-random activations in [-1, 1).
fn activations(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Quantized-resident GEMV/GEMM (`REFLEX_QUANT_RESIDENT=1`) against the f32
/// path, on every Q4_K matmul tensor of the first two layers of a real dense
/// GGUF: decode (`rows == 1`), the fused multi-row kernel (2..=8 and 13 rows,
/// exercising the 8-row chunking), and the dequantize-to-scratch + cuBLAS
/// path (rows above the fused threshold). Weights are bit-identical by
/// construction (same decode expression as `dequantize_q4k_kernel`), so only
/// summation order differs. Run (single-threaded) with:
/// `REFLEX_TEST_GGUF=<dense Q4_K_M gguf> cargo test --release -- --ignored quant_resident_gemv_matches_f32`
#[test]
#[ignore]
fn quant_resident_gemv_matches_f32() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    // Loaded quantized-resident so a model too big for f32 (Mistral 7B on a
    // 16 GB card) still fits; the f32 references below are loaded per tensor.
    // Run single-threaded: this sets the env var for the load only.
    std::env::set_var("REFLEX_QUANT_RESIDENT", "1");
    let model = Model::load(device.clone(), &file);
    std::env::remove_var("REFLEX_QUANT_RESIDENT");
    let model = model.expect("failed to load model");
    assert!(
        model.gemv_q4k_k.is_some(),
        "fixture must have Q4_K dense matmul weights"
    );

    let names: Vec<String> = (0..2)
        .flat_map(|i| {
            [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
            .map(|t| format!("blk.{i}.{t}.weight"))
        })
        .collect();
    let kernels = load_dequant_kernels(&device).expect("dequant kernels");
    let mut pipeline = WeightLoadPipeline::new(&device).expect("pipeline");
    let mut arena = QuantArena::for_tensors(&device, &file, &names)
        .expect("arena")
        .expect("fixture must contain Q4_K matmul tensors");

    let scratch_rows = quant_fused_max_rows() + 7;
    let mut checked = 0;
    for name in &names {
        if file.tensor_info(name).map(|i| i.ggml_type) != Some(GgmlType::Q4K) {
            continue;
        }
        let wf = load_weight_device(&mut pipeline, &kernels, &file, name).expect("f32 weight");
        let wq = load_weight_device_quant(&mut pipeline, &kernels, &file, name, &mut arena)
            .expect("quant weight");
        assert!(
            wq.quant_ptr().is_some(),
            "{name} should be quantized-resident"
        );
        let in_f = wf.shape[0] as usize;

        for rows in [1usize, 2, 5, 8, 13, scratch_rows] {
            let x_host = activations(rows * in_f, rows as u64 + checked as u64);
            let x = device.htod_sync_copy(&x_host).expect("upload x");
            let (yf, yq) = if rows == 1 {
                (model.gemv(&x, &wf), model.gemv(&x, &wq))
            } else {
                (model.gemm(&x, &wf, rows), model.gemm(&x, &wq, rows))
            };
            let yf = device.dtoh_sync_copy(&yf.expect("f32 path")).expect("dtoh");
            let yq = device
                .dtoh_sync_copy(&yq.expect("quant path"))
                .expect("dtoh");
            let d = rel_max_diff(&yq, &yf);
            assert!(d < 1e-4, "{name} rows={rows}: relative max diff {d}");
        }
        checked += 1;
    }
    assert!(checked > 0, "no Q4_K tensors checked");
}

/// End to end: greedy generation with `REFLEX_QUANT_RESIDENT=1` must produce
/// the same token ids as the f32 path. Sets the env var for the second load
/// only, so run it single-threaded (`--test-threads=1`, as
/// `scripts/gpu_nightly_tests.sh` does).
#[test]
#[ignore]
fn quant_resident_generate_matches_f32() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let prompt = "The capital of France is";
    let run = |quant: bool| {
        if quant {
            std::env::set_var("REFLEX_QUANT_RESIDENT", "1");
        } else {
            std::env::remove_var("REFLEX_QUANT_RESIDENT");
        }
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = Model::load(device, &file).expect("failed to load model");
        std::env::remove_var("REFLEX_QUANT_RESIDENT");
        let quant_layers = model
            .layers
            .iter()
            .filter(|l| matches!(l, LayerWeights::Dense(d) if d.attn_q.quant_ptr().is_some()))
            .count();
        let (ids, _) = model
            .generate(
                prompt,
                16,
                None,
                &SamplingParams::default(),
                |_| {},
                |_, _| {},
            )
            .expect("generate");
        (ids, quant_layers)
    };
    let (ids_f32, q0) = run(false);
    let (ids_q, q1) = run(true);
    assert_eq!(q0, 0, "flag off must not keep weights quantized");
    assert!(q1 > 0, "flag on must keep Q4_K weights quantized");
    assert_eq!(
        ids_q, ids_f32,
        "greedy ids differ between quantized-resident and f32"
    );
}
