use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Max |a - b| over max |b|, the comparison the GEMV tests below use.
fn rel_max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// A load policy with matrix weights in `dtype` and no LoRA targets.
fn policy(dtype: WeightsDtype) -> WeightPolicy {
    WeightPolicy::from_options(&LoadOptions {
        weights: dtype,
        lora_adapter: None,
    })
    .expect("policy")
}

/// [`Model::load_with_options`] with `REFLEX_QUANT_RESIDENT` set to `quant`
/// for the load only (so run these tests single-threaded).
fn load_model(device: Arc<CudaDevice>, file: &GgufFile, dtype: WeightsDtype, quant: bool) -> Model {
    if quant {
        std::env::set_var("REFLEX_QUANT_RESIDENT", "1");
    } else {
        std::env::remove_var("REFLEX_QUANT_RESIDENT");
    }
    let opts = LoadOptions {
        weights: dtype,
        lora_adapter: None,
    };
    let model = Model::load_with_options(device, file, &opts);
    std::env::remove_var("REFLEX_QUANT_RESIDENT");
    model.expect("failed to load model")
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
/// path (rows above the fused threshold), for both `--weights` modes. The
/// decode and fused paths read the exact Q4_K values, so they are checked
/// against the `f32` weight; the scratch path dequantizes into the model's
/// `--weights` dtype by design, so it is checked against a weight stored in
/// that dtype. Either way the weights are bit-identical by construction (same
/// decode expression and rounding as the load-time dequant), so only
/// summation order differs. Run (single-threaded) with:
/// `REFLEX_TEST_GGUF=<dense Q4_K_M gguf> cargo test --release -- --ignored quant_resident_gemv_matches_f32`
#[test]
#[ignore]
fn quant_resident_gemv_matches_f32() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        quant_resident_gemv_matches(&device, &file, dtype);
    }
}

fn quant_resident_gemv_matches(device: &Arc<CudaDevice>, file: &GgufFile, dtype: WeightsDtype) {
    // Loaded quantized-resident so a model too big for f32 (Mistral 7B on a
    // 16 GB card) still fits; the references below are loaded per tensor.
    let model = load_model(device.clone(), file, dtype, true);
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
    let kernels = load_dequant_kernels(device).expect("dequant kernels");
    let mut pipeline = WeightLoadPipeline::new(device).expect("pipeline");
    let mut arena = QuantArena::for_tensors(device, file, &names, |_, ty| ty == GgmlType::Q4K)
        .expect("arena")
        .expect("fixture must contain Q4_K matmul tensors");

    let scratch_rows = quant_fused_max_rows() + 7;
    let mut checked = 0;
    for name in &names {
        if file.tensor_info(name).map(|i| i.ggml_type) != Some(GgmlType::Q4K) {
            continue;
        }
        let wf = load_weight_device(
            &mut pipeline,
            &kernels,
            &policy(WeightsDtype::F32),
            file,
            name,
        )
        .expect("f32 weight");
        let wd = load_weight_device(&mut pipeline, &kernels, &policy(dtype), file, name)
            .expect("weight in the model's dtype");
        let wq = load_weight_device_quant(
            &mut pipeline,
            &kernels,
            &policy(dtype),
            file,
            name,
            &mut arena,
            &[GgmlType::Q4K],
        )
        .expect("quant weight");
        assert!(
            wq.quant_ptr().is_some(),
            "{name} should be quantized-resident"
        );
        let in_f = wf.shape[0] as usize;

        for rows in [1usize, 2, 5, 8, 13, scratch_rows] {
            let x_host = activations(rows * in_f, rows as u64 + checked as u64);
            let x = device.htod_sync_copy(&x_host).expect("upload x");
            // Above the fused threshold the quantized path computes from the
            // `dtype` scratch copy; below it, from the exact Q4_K values.
            let reference = if rows > quant_fused_max_rows() {
                &wd
            } else {
                &wf
            };
            let (yf, yq) = if rows == 1 {
                (model.gemv(&x, reference), model.gemv(&x, &wq))
            } else {
                (model.gemm(&x, reference, rows), model.gemm(&x, &wq, rows))
            };
            let yf = device.dtoh_sync_copy(&yf.expect("f32 path")).expect("dtoh");
            let yq = device
                .dtoh_sync_copy(&yq.expect("quant path"))
                .expect("dtoh");
            let d = rel_max_diff(&yq, &yf);
            assert!(
                d < 1e-4,
                "{dtype} {name} rows={rows}: relative max diff {d}"
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no Q4_K tensors checked");
}

/// End to end: greedy generation with `REFLEX_QUANT_RESIDENT=1` must produce
/// the same token ids as the plain `--weights f32` path, with either
/// `--weights` mode for the weights that aren't kept quantized. Sets the env
/// var for one load at a time, so run it single-threaded (`--test-threads=1`,
/// as `scripts/gpu_nightly_tests.sh` does).
#[test]
#[ignore]
fn quant_resident_generate_matches_f32() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let prompt = "The capital of France is";
    let run = |dtype: WeightsDtype, quant: bool| {
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let model = load_model(device, &file, dtype, quant);
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
    let (ids_f32, q0) = run(WeightsDtype::F32, false);
    assert_eq!(q0, 0, "flag off must not keep weights quantized");
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let (ids_q, q1) = run(dtype, true);
        assert!(q1 > 0, "flag on must keep Q4_K weights quantized");
        assert_eq!(
            ids_q, ids_f32,
            "greedy ids differ between quantized-resident ({dtype}) and f32"
        );
    }
}

/// The quantized-resident LM head (`gemv_q6k_kernel` on the raw Q6_K
/// `token_embd`/`output.weight` blocks) against the f32 path, full vocab, plus
/// `gemv_gather` on it (full GEMV + host pick). Run single-threaded:
/// `REFLEX_TEST_GGUF=<Q4_K_M gguf> cargo test --release -- --ignored --test-threads=1 quant_resident_lm_head`
#[test]
#[ignore]
fn quant_resident_lm_head_matches_f32() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = load_model(device.clone(), &file, WeightsDtype::default(), true);

    let lm = model.lm_head_resident().expect("lm head");
    assert!(
        lm.quant_ptr().is_some(),
        "LM head should be quantized-resident"
    );
    let (in_f, vocab) = (lm.shape[0] as usize, lm.shape[1] as usize);

    let name = if file.tensor_info("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let kernels = load_dequant_kernels(&device).expect("dequant kernels");
    let mut pipeline = WeightLoadPipeline::new(&device).expect("pipeline");
    let wf = load_weight_device(
        &mut pipeline,
        &kernels,
        &policy(WeightsDtype::F32),
        &file,
        name,
    )
    .expect("f32 LM head");

    let x = device
        .htod_sync_copy(&activations(in_f, 7))
        .expect("upload x");
    let yf = device
        .dtoh_sync_copy(&model.gemv(&x, &wf).expect("f32 gemv"))
        .expect("dtoh");
    let yq = device
        .dtoh_sync_copy(&model.gemv(&x, lm).expect("q6k gemv"))
        .expect("dtoh");
    assert_eq!(yq.len(), vocab);
    let d = rel_max_diff(&yq, &yf);
    assert!(d < 1e-4, "LM head relative max diff {d}");

    let rows = [0u32, 1, (vocab / 2) as u32, (vocab - 1) as u32];
    let gathered = model.gemv_gather(&x, lm, &rows).expect("gather");
    for (g, &r) in gathered.iter().zip(&rows) {
        assert_eq!(*g, yq[r as usize], "gather row {r}");
    }
}

/// The pipeline's chunked, multi-threaded staging against the host bytes, byte
/// for byte: sizes below the parallel-fill threshold, one chunk split across
/// the fill threads, and several 64 MB chunks with a ragged tail, through
/// `upload_raw`, `dequantize` (Q8_0, against the host dequant) and
/// `upload_host_bytes` (F32 and F16 tensors).
/// Real fixtures are too small to reach the multi-chunk path, and the
/// LM-head test above would pass with a staging bug that corrupts both its
/// sides the same way. Run: `cargo test --release -- --ignored pipeline_stages`
#[test]
#[ignore]
fn pipeline_stages_large_tensors_byte_exact() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let mut pipeline = WeightLoadPipeline::new(&device).expect("pipeline");

    let sizes = [1000usize, 3 << 20, (9 << 20) + 5, (150 << 20) + 12345];
    let bufs: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            (0..n as u64)
                .map(|i| (i.wrapping_mul(2654435761).wrapping_add(k as u64 * 977) >> 7) as u8)
                .collect()
        })
        .collect();
    let total: usize = sizes
        .iter()
        .map(|n| n.next_multiple_of(QUANT_ARENA_ALIGN))
        .sum();
    let mut arena = QuantArena {
        buf: Arc::new(unsafe { device.alloc::<u8>(total) }.expect("alloc arena")),
        used: 0,
    };
    let placed: Vec<(usize, usize)> = bufs
        .iter()
        .map(|b| pipeline.upload_raw(&mut arena, b).expect("upload_raw"))
        .collect();
    device.synchronize().expect("sync");
    let back = device
        .dtoh_sync_copy(arena.buf.as_ref())
        .expect("dtoh arena");
    for (b, &(off, len)) in bufs.iter().zip(&placed) {
        assert_eq!(len, b.len());
        assert!(back[off..off + len] == b[..], "upload_raw of {len} bytes");
    }

    // ~80 MB of Q8_0 blocks: two chunks into one device staging slot.
    let blocks = (80 << 20) / 34 + 3;
    let mut q8 = Vec::with_capacity(blocks * 34);
    for b in 0..blocks {
        let d = half::f16::from_f32(((b % 97) as f32 + 1.0) / 64.0);
        q8.extend_from_slice(&d.to_le_bytes());
        q8.extend((0..32).map(|j| ((b * 7 + j * 13) % 251) as u8));
    }
    let n = (blocks * 32) as u64;
    let kernels = load_dequant_kernels(&device).expect("dequant kernels");
    let got = dequantize_tensor_to_device(&mut pipeline, &kernels, GgmlType::Q8_0, &q8, n)
        .expect("dequantize");
    let got = device.dtoh_sync_copy(&got).expect("dtoh dequant");
    let want_q8 = dequant::dequantize(GgmlType::Q8_0, &q8, n).expect("host dequant");
    assert_eq!(got.len(), want_q8.len());
    let bad = got
        .iter()
        .zip(&want_q8)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(bad.is_none(), "dequantize mismatch at element {bad:?}");

    // Tensors with no dequant kernel (`upload_host_bytes`): F32 as is, F16
    // through the host fallback, small (a norm) and multi-chunk, alternating
    // with Q8_0 dequants so both device slots are reused across kinds.
    for &elems in &[4096usize, (20 << 20) + 3] {
        let f32s: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.37).sin()).collect();
        let f32_bytes: Vec<u8> = f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = f32s
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        for (ty, raw) in [(GgmlType::F32, &f32_bytes), (GgmlType::F16, &f16_bytes)] {
            let got = dequantize_tensor_to_device(&mut pipeline, &kernels, ty, raw, elems as u64)
                .expect("host-path upload");
            let q = dequantize_tensor_to_device(&mut pipeline, &kernels, GgmlType::Q8_0, &q8, n)
                .expect("dequantize");
            let got = device.dtoh_sync_copy(&got).expect("dtoh host-path");
            let want = dequant::dequantize(ty, raw, elems as u64).expect("host dequant");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && got.len() == want.len(),
                "{ty:?} upload of {elems} elements"
            );
            let q = device.dtoh_sync_copy(&q).expect("dtoh dequant");
            assert!(
                q.iter()
                    .zip(&want_q8)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "Q8_0 dequant after {ty:?} upload"
            );
        }
    }
}

const KOLIBRI_Q4KM: &str = "test-data/tiny-kolibri1.gguf";

/// Kolibri-1 with `REFLEX_QUANT_RESIDENT=1`: every matmul tensor of layer 0
/// (all Q4_K) and layer 2 (Q6_K `attn_v`/`ffn_down_exps`/`ffn_down_shexp`) of
/// the Q4_K_M fixture, against the same tensor dequantized. Stacked expert
/// tensors are checked on their first, a middle and the last expert, through
/// `expert_gemv` (decode) and `expert_gemm` (an expert group of batched
/// prefill). Row counts cover decode, Q4_K's fused kernel and the
/// dequantize-to-scratch path (always taken by Q6_K above one row), whose
/// reference is the weight in the model's `--weights` dtype. Run
/// single-threaded: `cargo test --release -- --ignored --test-threads=1 kolibri1_quant`
#[test]
#[ignore]
fn kolibri1_quant_resident_matmuls_match_dequantized() {
    let file = GgufFile::open(KOLIBRI_Q4KM).expect("open Kolibri fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let model = load_model(device.clone(), &file, dtype, true);
        let kernels = load_dequant_kernels(&device).expect("dequant kernels");
        let mut pipeline = WeightLoadPipeline::new(&device).expect("pipeline");
        let scratch_rows = quant_fused_max_rows() + 7;
        let mut seen: Vec<GgmlType> = Vec::new();
        for layer in [0usize, 2] {
            let LayerWeights::Kolibri(l) = &model.layers[layer] else {
                panic!("layer {layer} is not a Kolibri layer");
            };
            let tensors: [(&str, &Weight); 10] = [
                ("attn_q", &l.attn_q),
                ("attn_k", &l.attn_k),
                ("attn_v", &l.attn_v),
                ("attn_output", &l.attn_output),
                ("ffn_gate_exps", &l.ffn_gate_exps),
                ("ffn_up_exps", &l.ffn_up_exps),
                ("ffn_down_exps", &l.ffn_down_exps),
                ("ffn_gate_shexp", &l.ffn_gate_shexp),
                ("ffn_up_shexp", &l.ffn_up_shexp),
                ("ffn_down_shexp", &l.ffn_down_shexp),
            ];
            for (t, wq) in tensors {
                let name = format!("blk.{layer}.{t}.weight");
                let ty = file.tensor_info(&name).expect("tensor").ggml_type;
                assert!(
                    wq.quant_ptr().is_some(),
                    "{name} ({ty:?}) should be quantized-resident"
                );
                if !seen.contains(&ty) {
                    seen.push(ty);
                }
                let wf = load_weight_device(
                    &mut pipeline,
                    &kernels,
                    &policy(WeightsDtype::F32),
                    &file,
                    &name,
                )
                .expect("f32 weight");
                let wd = load_weight_device(&mut pipeline, &kernels, &policy(dtype), &file, &name)
                    .expect("weight in the model's dtype");
                let in_f = wf.shape[0] as usize;
                let experts: Vec<Option<usize>> = match wf.shape.len() {
                    3 => {
                        let n = wf.shape[2] as usize;
                        vec![Some(0), Some(n / 2), Some(n - 1)]
                    }
                    _ => vec![None],
                };
                for expert in experts {
                    for rows in [1usize, 3, 8, 13, scratch_rows] {
                        let fused =
                            rows == 1 || (ty == GgmlType::Q4K && rows <= quant_fused_max_rows());
                        let reference = if fused { &wf } else { &wd };
                        let x = device
                            .htod_sync_copy(&activations(rows * in_f, rows as u64 + layer as u64))
                            .expect("upload x");
                        let (yf, yq) = match (expert, rows) {
                            (Some(e), 1) => (
                                model.gemv_expert(&x, reference, e),
                                model.expert_gemv(&x, wq, e),
                            ),
                            (Some(e), _) => (
                                model.expert_gemm(&x, reference, e, rows),
                                model.expert_gemm(&x, wq, e, rows),
                            ),
                            (None, 1) => (model.gemv(&x, reference), model.gemv(&x, wq)),
                            (None, _) => {
                                (model.gemm(&x, reference, rows), model.gemm(&x, wq, rows))
                            }
                        };
                        let yf = device
                            .dtoh_sync_copy(&yf.expect("dequantized path"))
                            .expect("dtoh");
                        let yq = device
                            .dtoh_sync_copy(&yq.expect("quantized path"))
                            .expect("dtoh");
                        let d = rel_max_diff(&yq, &yf);
                        assert!(
                            d < 1e-4,
                            "{dtype} {name} {ty:?} expert={expert:?} rows={rows}: relative max diff {d}"
                        );
                    }
                }
            }
        }
        assert!(
            seen.contains(&GgmlType::Q4K) && seen.contains(&GgmlType::Q6K),
            "fixture must exercise both Q4_K and Q6_K, saw {seen:?}"
        );
        let lm = model.lm_head_resident().expect("lm head");
        assert!(
            lm.quant_ptr().is_some(),
            "Q6_K LM head should be quantized-resident"
        );
    }
}

/// End to end on the Kolibri-1 Q4_K_M fixture: greedy ids with
/// `REFLEX_QUANT_RESIDENT=1` equal the flag-off path in the same `--weights`
/// mode, on a short prompt and one past the fixture's 16-token window, and
/// quantized-resident batched prefill matches sequential prefill. Run
/// single-threaded (`--test-threads=1 kolibri1_quant`).
#[test]
#[ignore]
fn kolibri1_quant_resident_generate_matches_dequantized() {
    let file = GgufFile::open(KOLIBRI_Q4KM).expect("open Kolibri fixture");
    let prompts = [
        "Die Hauptstadt von Deutschland ist",
        "The quick brown fox jumps over the lazy dog while the river runs past the old mill, and the miller counts his sacks of flour one by one before the sun goes down over the hills.",
    ];
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let run = |quant: bool| {
            let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
            let model = load_model(device, &file, dtype, quant);
            let ids: Vec<Vec<u32>> = prompts
                .iter()
                .map(|p| {
                    model
                        .generate(p, 12, None, &SamplingParams::default(), |_| {}, |_, _| {})
                        .expect("generate")
                        .0
                })
                .collect();
            (model, ids)
        };
        let (_, plain) = run(false);
        let (model, quant) = run(true);
        assert!(
            model.layers.iter().all(
                |l| matches!(l, LayerWeights::Kolibri(k) if k.ffn_down_exps.quant_ptr().is_some())
            ),
            "every layer's experts should be quantized-resident"
        );
        for ((p, a), b) in prompts.iter().zip(&quant).zip(&plain) {
            eprintln!(
                "{dtype} {:?}: quant {a:?}",
                p.chars().take(24).collect::<String>()
            );
            assert_eq!(
                a, b,
                "{dtype}: greedy ids differ for {p:?} (quantized-resident vs dequantized)"
            );
        }

        let (seq_ids, seq_hidden, _, _, _) = model
            .prefill_dense(prompts[1], None, 0)
            .expect("prefill_dense");
        let (batch_ids, batch_hidden, _, _, _) = model
            .prefill_dense_batched(prompts[1], None, 0)
            .expect("prefill_dense_batched");
        assert_eq!(seq_ids, batch_ids);
        let last = model
            .last_row(&batch_hidden, batch_ids.len(), model.cfg.hidden_size)
            .expect("last_row");
        super::prefill_batching_tests::assert_prefill_hidden_close(
            &model.device.dtoh_sync_copy(&seq_hidden).unwrap(),
            &model.device.dtoh_sync_copy(&last).unwrap(),
            dtype,
            "kolibri quant-resident prefill",
        );
    }
}

/// [`load_model`] with `REFLEX_QUANT_RESIDENT=1` and `REFLEX_LAZY_EXPERTS=1`
/// for the load only (run single-threaded, as above).
fn load_model_lazy(device: Arc<CudaDevice>, file: &GgufFile, dtype: WeightsDtype) -> Model {
    std::env::set_var("REFLEX_LAZY_EXPERTS", "1");
    let model = load_model(device, file, dtype, true);
    std::env::remove_var("REFLEX_LAZY_EXPERTS");
    model
}

/// Kolibri-1 with `REFLEX_LAZY_EXPERTS=1` (docs/design/kolibri.md, Phase 4
/// step 2): no expert is on the GPU after the load, a 1–2 token prompt
/// uploads only some of them, and lazily uploaded experts give exactly the
/// eager model's results: greedy ids (batched prefill, then decode),
/// sequential prefill (the per-token layer path) and System1 scores. Run
/// single-threaded (`--test-threads=1 kolibri1_`).
#[test]
#[ignore]
fn kolibri1_lazy_experts_match_eager() {
    let file = GgufFile::open(KOLIBRI_Q4KM).expect("open Kolibri fixture");
    let prompts = [
        "Die Hauptstadt von Deutschland ist",
        "The quick brown fox jumps over the lazy dog while the river runs past the old mill, and the miller counts his sacks of flour one by one before the sun goes down over the hills.",
    ];
    let candidates: Vec<System1Candidate> = [" Berlin", " Paris", " die Stadt"]
        .iter()
        .map(|t| System1Candidate {
            text: t.to_string(),
        })
        .collect();
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        let eager = load_model(device.clone(), &file, dtype, true);
        assert_eq!(
            eager.lazy_expert_counts(),
            None,
            "eager load has no lazy layers"
        );
        let gen = |m: &Model, p: &str| {
            m.generate(p, 12, None, &SamplingParams::default(), |_| {}, |_, _| {})
                .expect("generate")
                .0
        };

        // Residency: nothing after the load, some but not all after a short prompt.
        let lazy = load_model_lazy(device.clone(), &file, dtype);
        let (resident, total) = lazy.lazy_expert_counts().expect("lazy layers");
        assert_eq!(
            resident, 0,
            "{dtype}: no expert should be uploaded by the load"
        );
        let LayerWeights::Kolibri(l0) = &lazy.layers[0] else {
            panic!("layer 0 is not a Kolibri layer");
        };
        assert_eq!(total, lazy.layers.len() * l0.exp_probs_b.len());
        assert_eq!(gen(&lazy, "Hi").first(), gen(&eager, "Hi").first());
        let (after_short, _) = lazy.lazy_expert_counts().unwrap();
        eprintln!("{dtype}: {after_short}/{total} experts resident after a short prompt");
        assert!(
            after_short > 0 && after_short < total,
            "{dtype}: a short prompt should upload some experts, not all ({after_short}/{total})"
        );

        // Same greedy ids, on the model that already has some experts.
        for p in prompts {
            assert_eq!(
                gen(&lazy, p),
                gen(&eager, p),
                "{dtype}: greedy ids differ for {p:?}"
            );
        }

        // Sequential prefill on a fresh lazy model (only the per-token layer
        // path uploads experts there): the same last hidden state, bit for bit.
        let lazy = load_model_lazy(device.clone(), &file, dtype);
        let (ids_l, hidden_l, _, _, _) = lazy
            .prefill_dense(prompts[1], None, 0)
            .expect("prefill_dense lazy");
        let (ids_e, hidden_e, _, _, _) = eager
            .prefill_dense(prompts[1], None, 0)
            .expect("prefill_dense eager");
        assert_eq!(ids_l, ids_e);
        assert_eq!(
            lazy.device.dtoh_sync_copy(&hidden_l).unwrap(),
            eager.device.dtoh_sync_copy(&hidden_e).unwrap(),
            "{dtype}: sequential prefill hidden state differs (lazy vs eager)"
        );

        // System1 on a fresh lazy model: identical scores.
        let lazy = load_model_lazy(device, &file, dtype);
        let s_l = lazy
            .system1_evaluate(prompts[0], &candidates, 1.0)
            .expect("system1 lazy");
        let s_e = eager
            .system1_evaluate(prompts[0], &candidates, 1.0)
            .expect("system1 eager");
        let scores = |r: &System1Response| r.results.iter().map(|c| c.score).collect::<Vec<_>>();
        assert_eq!(scores(&s_l), scores(&s_e), "{dtype}: System1 scores differ");
    }
}

const TINY_GEMMA4: &str = "test-data/tiny-gemma4.gguf";

/// Synthetic Q8_0 blocks for an `[in_f, out_f]` weight (`out_f` rows of
/// `in_f / 32` blocks): f16 scales that are negative, zero, subnormal or
/// normal, and int8 quants covering the whole range, -128 included.
fn synthetic_q8_0(in_f: usize, out_f: usize, seed: u64) -> Vec<u8> {
    let blocks = in_f / QK_LEGACY * out_f;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut out = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
    for b in 0..blocks {
        let d = match b % 16 {
            0 => half::f16::from_bits(0),
            1 => half::f16::from_bits((next() % 0x3FF + 1) as u16), // subnormal
            _ => {
                let v = ((next() >> 40) as f32 / (1u64 << 24) as f32) * 0.02 + 1e-4;
                half::f16::from_f32(if next() & 1 == 0 { v } else { -v })
            }
        };
        out.extend_from_slice(&d.to_le_bytes());
        out.extend((0..32).map(|_| next() as u8));
    }
    out
}

/// A quantized-resident Q8_0 [`Weight`] holding `bytes` (its own device
/// buffer, 256-byte aligned like an arena).
fn q8_0_weight(device: &Arc<CudaDevice>, bytes: &[u8], in_f: usize, out_f: usize) -> Weight {
    Weight {
        data: WeightData::Quant {
            ty: GgmlType::Q8_0,
            arena: Arc::new(device.htod_sync_copy(bytes).expect("upload Q8_0 blocks")),
            offset: 0,
            len: bytes.len(),
        },
        shape: vec![in_f as u64, out_f as u64],
    }
}

/// `x[rows, in_f] @ w^T` on the host, in f64, `w` row-major `[out_f, in_f]`.
fn host_matmul(x: &[f32], w: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_f];
    for r in 0..rows {
        let xr = &x[r * in_f..(r + 1) * in_f];
        for o in 0..out_f {
            let wo = &w[o * in_f..(o + 1) * in_f];
            y[r * out_f + o] = xr
                .iter()
                .zip(wo)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>() as f32;
        }
    }
    y
}

/// The tiny Gemma 4 fixture (f32 weights) loaded flag off in `dtype`, with
/// the Q8_0 kernels attached by hand: a `Model` to launch them through on
/// synthetic weights. `None` (with a skip message) if the fixture is absent.
fn q8_0_test_model(device: &Arc<CudaDevice>, dtype: WeightsDtype) -> Option<Model> {
    if !std::path::Path::new(TINY_GEMMA4).exists() {
        eprintln!("Skipping: {TINY_GEMMA4} not present");
        return None;
    }
    let file = GgufFile::open(TINY_GEMMA4).expect("open tiny Gemma 4 fixture");
    let mut model = load_model(device.clone(), &file, dtype, false);
    assert!(
        model.q8_0_k.is_none(),
        "flag off must not load Q8_0 kernels"
    );
    model.q8_0_k = Some(Q8_0Kernels::load(device).expect("load gemv_q8_0 module"));
    Some(model)
}

/// `gemv_q8_0_kernel` (decode and the fused multi-row path) against a host
/// f64 reference on the exact Q8_0 values, on synthetic weights: the tiny
/// fixture's row length (64), a ragged last shared-memory tile (800 = 512 +
/// 288), real Gemma 4 widths (3840, 5376), output counts that leave the last
/// thread block partly idle, and 1-33 input rows (16-row chunking). Run:
/// `cargo test --release -- --ignored --test-threads=1 quant_resident_q8_0`
#[test]
#[ignore]
fn quant_resident_q8_0_gemv_matches_host() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = q8_0_test_model(&device, WeightsDtype::F16) else {
        return;
    };
    let shapes = [(64usize, 40usize), (800, 37), (3840, 72), (5376, 24)];
    for (k, &(in_f, out_f)) in shapes.iter().enumerate() {
        let bytes = synthetic_q8_0(in_f, out_f, k as u64 + 1);
        let w_host = dequant::dequantize(GgmlType::Q8_0, &bytes, (in_f * out_f) as u64)
            .expect("host dequant");
        let wq = q8_0_weight(&device, &bytes, in_f, out_f);
        for rows in [1usize, 2, 7, 8, 15, 16, 17, 33] {
            let x_host = activations(rows * in_f, (k * 100 + rows) as u64);
            let x = device.htod_sync_copy(&x_host).expect("upload x");
            let want = host_matmul(&x_host, &w_host, rows, in_f, out_f);
            let got = model.gemv_q8_0(&x, &wq, rows).expect("gemv_q8_0");
            let got = device.dtoh_sync_copy(&got).expect("dtoh");
            let d = rel_max_diff(&got, &want);
            assert!(
                d < 1e-4,
                "in={in_f} out={out_f} rows={rows}: relative max diff {d}"
            );
            if rows == 1 {
                let via_gemv = device
                    .dtoh_sync_copy(&model.gemv(&x, &wq).expect("Model::gemv"))
                    .expect("dtoh");
                assert_eq!(via_gemv, got, "Model::gemv must take gemv_q8_0");
            }
        }
    }
}

/// The coalesced Q8_0 dequant kernels: f32 output bit-identical to the host
/// dequant (itself byte-exact with gguf-py, and with `dequantize_q8_0_kernel`
/// per `pipeline_stages_large_tensors_byte_exact`), f16 output equal to its
/// round-to-nearest-even (`dequantize_q8_0_f16_kernel`'s rounding). 37 rows
/// of 864 elements: 999 blocks, not a multiple of the 8 one CUDA block covers.
#[test]
#[ignore]
fn quant_resident_q8_0_dequant_coalesced_bit_exact() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let Some(model) = q8_0_test_model(&device, WeightsDtype::F16) else {
        return;
    };
    let k = model.q8_0_k.as_ref().expect("attached");
    let bytes = synthetic_q8_0(864, 37, 99);
    let num_blocks = bytes.len() / Q8_0_BLOCK_BYTES;
    let n = num_blocks * QK_LEGACY;
    let want = dequant::dequantize(GgmlType::Q8_0, &bytes, n as u64).expect("host dequant");
    let src = device.htod_sync_copy(&bytes).expect("upload");
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out32 = device.alloc_zeros::<f32>(n).expect("alloc");
    let mut out16 = device.alloc_zeros::<half::f16>(n).expect("alloc");
    unsafe {
        k.dequant_f32
            .function
            .clone()
            .launch(cfg, (&src, &mut out32, num_blocks as u32))
            .expect("launch f32");
        k.dequant_f16
            .function
            .clone()
            .launch(cfg, (&src, &mut out16, num_blocks as u32))
            .expect("launch f16");
    }
    let got32 = device.dtoh_sync_copy(&out32).expect("dtoh");
    let got16 = device.dtoh_sync_copy(&out16).expect("dtoh");
    for i in 0..n {
        assert_eq!(got32[i].to_bits(), want[i].to_bits(), "f32 element {i}");
        assert_eq!(
            got16[i].to_bits(),
            half::f16::from_f32(want[i]).to_bits(),
            "f16 element {i}"
        );
    }
}

/// `Model::gemm` on a Q8_0 weight: up to the fused threshold it equals
/// `gemv_q8_0` exactly; above it (dequantize into the `--weights` scratch,
/// then cuBLAS) it is bit-identical to `Model::gemm` on the same weight
/// stored in that dtype -- the same values and the same cuBLAS call, so a
/// long prefill matches the flag-off path exactly. Both `--weights` modes.
#[test]
#[ignore]
fn quant_resident_q8_0_gemm_matches_dequantized() {
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    for dtype in [WeightsDtype::F32, WeightsDtype::F16] {
        let Some(model) = q8_0_test_model(&device, dtype) else {
            return;
        };
        let (in_f, out_f) = (3840usize, 136usize);
        let bytes = synthetic_q8_0(in_f, out_f, 7);
        let w_host = dequant::dequantize(GgmlType::Q8_0, &bytes, (in_f * out_f) as u64)
            .expect("host dequant");
        let wq = q8_0_weight(&device, &bytes, in_f, out_f);
        let data = match dtype {
            WeightsDtype::F32 => WeightData::F32(device.htod_sync_copy(&w_host).expect("upload")),
            WeightsDtype::F16 => {
                let h: Vec<half::f16> = w_host.iter().map(|&v| half::f16::from_f32(v)).collect();
                WeightData::F16(device.htod_sync_copy(&h).expect("upload"))
            }
        };
        let wd = Weight {
            data,
            shape: vec![in_f as u64, out_f as u64],
        };
        let fused_max = quant_fused_max_rows();
        for rows in [2usize, fused_max, fused_max + 1, fused_max + 40] {
            let x = device
                .htod_sync_copy(&activations(rows * in_f, rows as u64))
                .expect("upload x");
            let got = device
                .dtoh_sync_copy(&model.gemm(&x, &wq, rows).expect("gemm on Q8_0"))
                .expect("dtoh");
            let want = if rows <= fused_max {
                model.gemv_q8_0(&x, &wq, rows).expect("gemv_q8_0")
            } else {
                model
                    .gemm(&x, &wd, rows)
                    .expect("gemm on dequantized weight")
            };
            let want = device.dtoh_sync_copy(&want).expect("dtoh");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{dtype} rows={rows}: Q8_0 gemm differs from its reference (max rel diff {})",
                rel_max_diff(&got, &want)
            );
        }
    }
}

/// Every Q8_0 row of the real Gemma 4 12B/31B shapes is a multiple of 16
/// bytes, so arena offsets (256-aligned) keep every row 16-byte aligned and
/// every block 2-byte aligned -- what `gemv_q8_0_kernel`'s loads assume.
#[test]
fn gemma4_q8_0_rows_are_16_byte_aligned() {
    for in_f in [3840usize, 4096, 5376, 8192, 15360, 16384, 21504] {
        assert_eq!(in_f % QK_LEGACY, 0, "{in_f}");
        assert_eq!((in_f / QK_LEGACY * Q8_0_BLOCK_BYTES) % 16, 0, "{in_f}");
    }
}
