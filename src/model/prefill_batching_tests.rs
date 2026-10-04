use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Compares a sequential-prefill hidden state against the batched-prefill one
/// for the same position. With f32 weights the two paths differ only in
/// summation order, so every element must agree to 1e-3. With f16 weights the
/// batched path also rounds each GEMM's activations to f16 (`cublasGemmEx`)
/// while the sequential path's gemv reads them in f32, so the states differ by
/// accumulated activation rounding, not just order: that mode checks the
/// relative L2 error instead, against `F16_REL_L2_TOL`. Both modes print the
/// measured error.
pub(super) fn assert_prefill_hidden_close(
    seq: &[f32],
    batch: &[f32],
    dtype: WeightsDtype,
    label: &str,
) {
    // Measured on a T4 (2026-10-04): 5.6e-4 dense Qwen3-0.6B, 2.2e-4 hybrid
    // Qwen3.5-0.8B, 1.2e-5 MLA fixture. ~9x headroom over the largest.
    const F16_REL_L2_TOL: f64 = 5e-3;
    assert_eq!(seq.len(), batch.len());
    let max_abs = seq
        .iter()
        .zip(batch)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let diff_sq: f64 = seq
        .iter()
        .zip(batch)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let norm_sq: f64 = seq.iter().map(|&a| (a as f64).powi(2)).sum();
    let rel_l2 = (diff_sq / norm_sq.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("{label} ({dtype}): max_abs_diff={max_abs:e} rel_l2={rel_l2:e}");
    match dtype {
        WeightsDtype::F32 => {
            for (i, (a, b)) in seq.iter().zip(batch).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "{label}[{i}]: sequential={a}, batched={b}"
                );
            }
        }
        WeightsDtype::F16 => assert!(
            rel_l2 < F16_REL_L2_TOL,
            "{label}: relative L2 error {rel_l2:e} >= {F16_REL_L2_TOL:e} (max abs diff {max_abs:e})"
        ),
    }
}

/// Byte-exact-ish cross-check of `prefill_dense_batched` (cuBLAS GEMM +
/// batched RoPE/attention) against `prefill_dense` (the original
/// sequential per-token loop) on the same prompt/weights -- this is the
/// blocking check before trusting any batched-prefill latency number
/// (cuBLAS's summation order, RoPE's per-row position math, and the
/// batched attention kernel's causal masking are exactly the places a
/// silently-wrong-but-non-crashing bug would hide). Compares every row
/// of the batched hidden state against the
/// corresponding sequential-path position, plus the final argmax token
/// id. Real GGUF fixtures live outside this repo (gitignored), so this
/// is `#[ignore]`d by default -- run with:
/// `REFLEX_TEST_GGUF=<path> cargo test --release -- --ignored prefill_dense_batched_matches_sequential_prefill`
#[test]
#[ignore]
fn prefill_dense_batched_matches_sequential_prefill() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let prompt = "The capital of France is";

    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");

    let (seq_ids, seq_hidden, _, _, seq_position) = model
        .prefill_dense(prompt, None, 0)
        .expect("prefill_dense failed");
    let (batch_ids, batch_hidden, _, _, batch_position) = model
        .prefill_dense_batched(prompt, None, 0)
        .expect("prefill_dense_batched failed");

    assert_eq!(
        seq_ids, batch_ids,
        "tokenization must match between the two prefill paths"
    );
    assert_eq!(
        seq_position, batch_position,
        "final position must match between the two prefill paths"
    );

    let rows = batch_ids.len();
    let hidden_size = model.cfg.hidden_size;
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
    assert_prefill_hidden_close(&seq_host, &batch_host, model.weights_dtype(), "hidden");

    let seq_argmax = model
        .lm_head_argmax(&seq_hidden, hidden_size, model.cfg.rmsnorm_eps)
        .expect("seq argmax failed");
    let batch_argmax = model
        .lm_head_argmax(&batch_last_row, hidden_size, model.cfg.rmsnorm_eps)
        .expect("batch argmax failed");
    assert_eq!(
        seq_argmax, batch_argmax,
        "greedy-argmax next token must match between the two prefill paths"
    );
}
