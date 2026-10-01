use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

const MLA_FIXTURE: &str = "test-data/deepseek-tiny-mla.gguf";

/// Lowest-level sanity check before trusting the full attention-block test
/// below: `Self::gemv_per_head_batch` at `rows=1` (a novel 3-D-grid kernel with
/// no direct single-token analogue to diff row-by-row, unlike `Self::gemm`,
/// which `prefill_dense_batched_matches_sequential_prefill` could check against
/// `gemv_raw`) must reproduce `Self::gemv_per_head`'s existing, already-
/// hardware-verified output exactly against the same real `wk_b` weight tensor.
/// `test-data/deepseek-tiny-mla.gguf` (synthetic, hand-built via llama.cpp's
/// real converter -- see README.md's MLA fixture section) is already local, so
/// this doesn't need `REFLEX_TEST_GGUF`; still `#[ignore]`d since it needs a
/// real GPU -- run with `cargo test --release -- --ignored
/// gemv_per_head_batch_matches_gemv_per_head_at_rows_one`.
#[test]
#[ignore]
fn gemv_per_head_batch_matches_gemv_per_head_at_rows_one() {
    let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load MLA model");
    let m = model.mla.as_ref().expect("loaded model is not MLA");
    let layer = &m.layers[0];

    let n_head = m.cfg.num_heads;
    let in_features = m.cfg.qk_nope_head_dim;

    let host_x: Vec<f32> = (0..n_head * in_features)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let x = model.device.htod_sync_copy(&host_x).expect("x htod failed");

    let single = model
        .gemv_per_head(&x, &layer.wk_b, n_head)
        .expect("gemv_per_head failed");
    let batched = model
        .gemv_per_head_batch(
            m,
            &x,
            &layer.wk_b,
            1,
            n_head,
            n_head * in_features,
            in_features,
            0,
        )
        .expect("gemv_per_head_batch failed");

    let single_host = model
        .device
        .dtoh_sync_copy(&single)
        .expect("single dtoh failed");
    let batched_host = model
        .device
        .dtoh_sync_copy(&batched)
        .expect("batched dtoh failed");
    assert_eq!(single_host.len(), batched_host.len());
    for (i, (a, b)) in single_host.iter().zip(batched_host.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "out[{i}]: gemv_per_head={a}, gemv_per_head_batch={b}"
        );
    }
}

/// Byte-exact-ish cross-check of `prefill_mla_batched` (layer-major batched
/// attention block; grouped-GEMM-batched MoE FFN tail where present, see
/// `Model::forward_mla_moe_ffn_batched`) against `prefill_mla` (the original
/// token-major sequential loop, still calling the unmodified per-token
/// `Model::forward_mla_moe_ffn`) on the same prompt/weights -- the blocking check
/// before trusting the batched MLA prefill path, same role
/// `prefill_hybrid_batched_matches_sequential` plays for the hybrid path. NOTE:
/// `test-data/deepseek-tiny-mla.gguf` is dense-lead-only with no YaRN scaling (see
/// README.md's MLA fixture section), so this test does not exercise
/// `MlaFfn::Moe`'s grouped-GEMM path or `rope_norm_yarn_batch_kernel` -- see
/// `prefill_mla_batched_matches_sequential_real_moe_checkpoint` below for that,
/// which only the real `deepseek-ai/DeepSeek-V2-Lite` checkpoint can exercise (see
/// docs/DEVELOPMENT.md's "Known test-fixture limitations"). Run with `cargo test --release
/// -- --ignored prefill_mla_batched_matches_sequential`.
#[test]
#[ignore]
fn prefill_mla_batched_matches_sequential() {
    let prompt = "The capital of France is";

    let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load MLA model");
    let m = model.mla.as_ref().expect("loaded model is not MLA");

    let (seq_ids, seq_hidden, _, seq_position) = model
        .prefill_mla(m, prompt, None, 0)
        .expect("prefill_mla failed");
    let (batch_ids, batch_hidden, _, batch_position) = model
        .prefill_mla_batched(m, prompt, None, 0)
        .expect("prefill_mla_batched failed");

    assert_eq!(
        seq_ids, batch_ids,
        "tokenization must match between the two prefill paths"
    );
    assert_eq!(
        seq_position, batch_position,
        "final position must match between the two prefill paths"
    );

    let rows = batch_ids.len();
    let hidden_size = m.cfg.hidden_size;
    let batch_last_row = model
        .last_row(&batch_hidden, rows, hidden_size)
        .expect("last_row failed");

    let seq_host = model
        .device
        .dtoh_sync_copy(&seq_hidden)
        .expect("seq hidden dtoh failed");
    let batch_host = model
        .device
        .dtoh_sync_copy(&batch_last_row)
        .expect("batch hidden dtoh failed");
    assert_eq!(seq_host.len(), batch_host.len());
    for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "hidden[{i}]: sequential={a}, batched={b}"
        );
    }

    let eps = m.cfg.rmsnorm_eps;
    let seq_argmax = model
        .lm_head_argmax(&seq_hidden, hidden_size, eps)
        .expect("seq argmax failed");
    let batch_argmax = model
        .lm_head_argmax(&batch_last_row, hidden_size, eps)
        .expect("batch argmax failed");
    assert_eq!(
        seq_argmax, batch_argmax,
        "greedy-argmax next token must match between the two prefill paths"
    );
}

/// Real-`MlaFfn::Moe` counterpart to `prefill_mla_batched_matches_sequential`
/// above: that test's `test-data/deepseek-tiny-mla.gguf` fixture is dense-lead-only
/// (see its own doc comment), so it never exercises
/// `Model::forward_mla_moe_ffn_batched`'s grouped-GEMM routed-expert path or its
/// batched shared-expert seeding -- both new this round, and both only reachable
/// through a real `deepseek2` file's routed-MoE layers (no small synthetic
/// `deepseek2` MoE fixture exists, see docs/DEVELOPMENT.md's "Known test-fixture
/// limitations"). Same cross-check shape as the fixture-based test (byte-exact-ish
/// hidden state plus matching greedy-argmax token), but reads its GGUF path from
/// `REFLEX_TEST_GGUF` (the same convention `prefill_dense_batched_matches_sequential_prefill`
/// uses) instead of the hardcoded fixture constant, so it can point at the real
/// `deepseek-ai/DeepSeek-V2-Lite` checkpoint (see docs/DEVELOPMENT.md's "Known
/// test-fixture limitations" for how to produce it). Run with:
/// `REFLEX_TEST_GGUF=<path to a real deepseek2 GGUF with MoE layers> cargo test --release -- --ignored prefill_mla_batched_matches_sequential_real_moe_checkpoint`.
#[test]
#[ignore]
fn prefill_mla_batched_matches_sequential_real_moe_checkpoint() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local deepseek2 GGUF path (with MoE layers) to run this test");
    let prompt = "The capital of France is";

    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load MLA model");
    let m = model.mla.as_ref().expect("loaded model is not MLA");
    assert!(m.cfg.moe.is_some(), "REFLEX_TEST_GGUF must be a real deepseek2 checkpoint with MoE layers, not the dense-lead-only synthetic fixture");

    let (seq_ids, seq_hidden, _, seq_position) = model
        .prefill_mla(m, prompt, None, 0)
        .expect("prefill_mla failed");
    let (batch_ids, batch_hidden, _, batch_position) = model
        .prefill_mla_batched(m, prompt, None, 0)
        .expect("prefill_mla_batched failed");

    assert_eq!(
        seq_ids, batch_ids,
        "tokenization must match between the two prefill paths"
    );
    assert_eq!(
        seq_position, batch_position,
        "final position must match between the two prefill paths"
    );

    let rows = batch_ids.len();
    let hidden_size = m.cfg.hidden_size;
    let batch_last_row = model
        .last_row(&batch_hidden, rows, hidden_size)
        .expect("last_row failed");

    let seq_host = model
        .device
        .dtoh_sync_copy(&seq_hidden)
        .expect("seq hidden dtoh failed");
    let batch_host = model
        .device
        .dtoh_sync_copy(&batch_last_row)
        .expect("batch hidden dtoh failed");
    assert_eq!(seq_host.len(), batch_host.len());
    for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "hidden[{i}]: sequential={a}, batched={b}"
        );
    }

    let eps = m.cfg.rmsnorm_eps;
    let seq_argmax = model
        .lm_head_argmax(&seq_hidden, hidden_size, eps)
        .expect("seq argmax failed");
    let batch_argmax = model
        .lm_head_argmax(&batch_last_row, hidden_size, eps)
        .expect("batch argmax failed");
    assert_eq!(
        seq_argmax, batch_argmax,
        "greedy-argmax next token must match between the two prefill paths"
    );
}

/// `--import-kv` resume equivalence: captures a short prompt's compressed
/// `kv_cache` via the already-hardware-verified `forward_prompt_capture_kv_mla`,
/// then continues generation from it through both `prefill_mla` and
/// `prefill_mla_batched` (`start_pos > 0`) and diffs the two continuations --
/// the same `imported.is_some()` case `prefill_hybrid_batched`'s test coverage
/// doesn't separately exercise but this round's plan calls out explicitly (the
/// batched KV-cache write path, `Self::mla_write_kv_cache_batch`, must offset by
/// `start_pos` correctly, not just `0`). Run with `cargo test --release --
/// --ignored prefill_mla_batched_import_kv_resume_matches_sequential`.
#[test]
#[ignore]
fn prefill_mla_batched_import_kv_resume_matches_sequential() {
    let file = GgufFile::open(MLA_FIXTURE).expect("failed to open MLA fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load MLA model");
    let m = model.mla.as_ref().expect("loaded model is not MLA");

    let (_, cache) = model
        .forward_prompt_capture_kv_mla("The capital of France is")
        .expect("capture_kv failed");

    let continuation = " Paris";
    let (seq_ids, seq_hidden, _, seq_position) = model
        .prefill_mla(m, continuation, Some(&cache), 0)
        .expect("prefill_mla resume failed");
    let (batch_ids, batch_hidden, _, batch_position) = model
        .prefill_mla_batched(m, continuation, Some(&cache), 0)
        .expect("prefill_mla_batched resume failed");

    assert_eq!(
        seq_ids, batch_ids,
        "tokenization must match between the two resumed prefill paths"
    );
    assert_eq!(
        seq_position, batch_position,
        "final position must match between the two resumed prefill paths"
    );

    let rows = batch_ids.len();
    let hidden_size = m.cfg.hidden_size;
    let batch_last_row = model
        .last_row(&batch_hidden, rows, hidden_size)
        .expect("last_row failed");

    let seq_host = model
        .device
        .dtoh_sync_copy(&seq_hidden)
        .expect("seq hidden dtoh failed");
    let batch_host = model
        .device
        .dtoh_sync_copy(&batch_last_row)
        .expect("batch hidden dtoh failed");
    assert_eq!(seq_host.len(), batch_host.len());
    for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "resumed hidden[{i}]: sequential={a}, batched={b}"
        );
    }
}
