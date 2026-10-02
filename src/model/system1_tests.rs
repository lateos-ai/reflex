use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Exact cross-check of `gemv_gather` against the existing full-vocab
/// `gemv` path: same weights, same math, different kernel -- gathering a
/// handful of rows (including the model's own real argmax id) must
/// agree with the corresponding entries of a full-vocab GEMV to float
/// rounding. Real GGUF fixtures live outside this repo (`.gguf` is
/// gitignored, per docs/DEVELOPMENT.md's "Known test-fixture limitations"), so this
/// is `#[ignore]`d by default and reads its model path from
/// `REFLEX_TEST_GGUF` rather than guessing a local path -- run with:
/// `REFLEX_TEST_GGUF=<path> cargo test --release -- --ignored gemv_gather_matches_full_vocab_gemv`
#[test]
#[ignore]
fn gemv_gather_matches_full_vocab_gemv_at_matching_rows() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");

    let (_, hidden, _, _, _) = model
        .prefill_dense("The capital of France is", None, 0)
        .expect("prefill_dense failed");
    let normed = model
        .rmsnorm(
            &hidden,
            model.output_norm.f32().expect("f32 norm"),
            1,
            model.cfg.hidden_size,
            model.cfg.rmsnorm_eps,
        )
        .expect("rmsnorm failed");

    let lm_head = model.lm_head_resident().expect("lm_head_resident failed");
    let full = model.gemv(&normed, lm_head).expect("gemv failed");
    let full_host = model.device.dtoh_sync_copy(&full).expect("dtoh failed");

    let vocab_size = lm_head.shape[1] as usize;
    let argmax_id = full_host
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .expect("full_host must not be empty");
    let row_indices = [
        0u32,
        (vocab_size / 2) as u32,
        (vocab_size - 1) as u32,
        argmax_id,
    ];

    let gathered = model
        .gemv_gather(&normed, lm_head, &row_indices)
        .expect("gemv_gather failed");

    for (j, &row) in row_indices.iter().enumerate() {
        let expected = full_host[row as usize];
        let got = gathered[j];
        assert!(
            (expected - got).abs() < 1e-4,
            "row {row}: full_vocab={expected}, gathered={got}"
        );
    }
}

/// Byte-exact (to `gemv`'s own tolerance) check for the actual new code
/// path the lazy-`LmHead` optimization added: `gemv_gather_lm_head`
/// called on a *freshly loaded* model, before anything has forced the
/// tied LM head fully device-resident, must still return the same
/// values a full-vocab `gemv` against the forced-resident matrix would.
/// This is the case `gemv_gather_matches_full_vocab_gemv_at_matching_rows`
/// above can no longer exercise on its own, since it (correctly) calls
/// `lm_head_resident()` first to get the comparison baseline -- by the
/// time it calls `gemv_gather`, the lazy path has already been forced
/// resident once, so `gemv_gather_lm_head` would take the fast
/// already-resident branch instead of the still-lazy compact-upload one
/// this test targets. Requires a real *tied-embedding* GGUF (no separate
/// `output.weight` tensor) to actually exercise `LmHead::TiedLazy` at
/// all -- `#[ignore]`d, `REFLEX_TEST_GGUF`-gated like its sibling above.
#[test]
#[ignore]
fn gemv_gather_lm_head_matches_full_vocab_gemv_while_still_lazy() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a real local GGUF path to run this test");
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");

    let (_, hidden, _, _, _) = model
        .prefill_dense("The capital of France is", None, 0)
        .expect("prefill_dense failed");
    let normed = model
        .rmsnorm(
            &hidden,
            model.output_norm.f32().expect("f32 norm"),
            1,
            model.cfg.hidden_size,
            model.cfg.rmsnorm_eps,
        )
        .expect("rmsnorm failed");

    let vocab_size = model.token_embd.vocab_size();
    let row_indices = [0u32, 1u32, (vocab_size / 2) as u32, (vocab_size - 1) as u32];

    // Exercise the still-lazy path FIRST -- calling `lm_head_resident()`
    // (directly or via `gemv`) before this would force residency and
    // make `gemv_gather_lm_head` silently take its already-resident fast
    // path instead, defeating the point of this test.
    let gathered_lazy = model
        .gemv_gather_lm_head(&normed, &row_indices)
        .expect("gemv_gather_lm_head (lazy) failed");

    let lm_head = model.lm_head_resident().expect("lm_head_resident failed");
    let full = model.gemv(&normed, lm_head).expect("gemv failed");
    let full_host = model.device.dtoh_sync_copy(&full).expect("dtoh failed");

    for (j, &row) in row_indices.iter().enumerate() {
        let expected = full_host[row as usize];
        let got = gathered_lazy[j];
        assert!(
            (expected - got).abs() < 1e-4,
            "row {row}: full_vocab={expected}, gathered_lazy={got}"
        );
    }
}
