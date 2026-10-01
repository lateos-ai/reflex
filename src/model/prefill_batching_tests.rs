use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

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
    for (i, (a, b)) in seq_host.iter().zip(batch_host.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "hidden[{i}]: sequential={a}, batched={b}"
        );
    }

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
