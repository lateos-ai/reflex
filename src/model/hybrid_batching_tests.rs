use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Byte-exact-ish cross-check of `prefill_hybrid_batched` (layer-major
/// GatedAttention batching, GatedDeltaNet left sequential -- see
/// `Model::forward_hybrid_layer_batched`'s doc comment) against
/// `prefill_hybrid` (the original token-major sequential loop) on the
/// same prompt/weights -- the blocking check before trusting the batched
/// hybrid prefill path, same role
/// `prefill_dense_batched_matches_sequential_prefill` plays for dense/MoE.
/// Real GGUF fixtures live outside this repo (gitignored), so this is
/// `#[ignore]`d by default -- run with:
/// `REFLEX_TEST_GGUF=<path to a Qwen3.5 hybrid GGUF> cargo test --release -- --ignored prefill_hybrid_batched_matches_sequential`
#[test]
#[ignore]
fn prefill_hybrid_batched_matches_sequential() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local Qwen3.5 hybrid GGUF path to run this test");
    let prompt = "The capital of France is";

    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load_hybrid(device, &file).expect("failed to load hybrid model");
    let h = model.hybrid.as_ref().expect("loaded model is not hybrid");

    let (seq_ids, seq_hidden, _, seq_position) = model
        .prefill_hybrid(h, prompt, None, 0)
        .expect("prefill_hybrid failed");
    let (batch_ids, batch_hidden, _, batch_position) = model
        .prefill_hybrid_batched(h, prompt, None, 0)
        .expect("prefill_hybrid_batched failed");

    assert_eq!(
        seq_ids, batch_ids,
        "tokenization must match between the two prefill paths"
    );
    assert_eq!(
        seq_position, batch_position,
        "final position must match between the two prefill paths"
    );

    let rows = batch_ids.len();
    let hidden_size = h.attn_cfg.hidden_size;
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

    let eps = h.attn_cfg.rmsnorm_eps;
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
