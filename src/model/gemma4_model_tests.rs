//! GPU tests for the Gemma 4 model path, `#[ignore]`d like every other GPU
//! test. They run against `REFLEX_GEMMA4_GGUF` (e.g. the real
//! `gemma-4-12B-it-Q8_0.gguf`) or, when that is unset, the synthetic
//! `test-data/tiny-gemma4.gguf`, and skip if neither exists:
//! `REFLEX_GEMMA4_GGUF=<path> cargo test --release -- --ignored gemma4_model_`.

use super::prefill_batching_tests::assert_prefill_hidden_close;
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

const TINY_FIXTURE: &str = "test-data/tiny-gemma4.gguf";

/// Loads the Gemma 4 model under test with `REFLEX_WEIGHTS` (default f16),
/// or `None` (with a skip message) if no model file is available.
fn load_gemma4(test: &str) -> Option<Model> {
    let path = std::env::var("REFLEX_GEMMA4_GGUF").unwrap_or_else(|_| TINY_FIXTURE.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("Skipping {test}: set REFLEX_GEMMA4_GGUF or add {TINY_FIXTURE}");
        return None;
    }
    let file = GgufFile::open(&path).expect("open Gemma 4 GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("load Gemma 4 model");
    assert_eq!(model.architecture_kind(), ArchitectureKind::Gemma4);
    Some(model)
}

/// A prompt long enough that a sliding layer's window cuts off its first
/// tokens (`window + 8` tokens), capped so the sequential side stays cheap.
fn prompt_past_window(model: &Model, window: usize) -> String {
    let sentence = "The quick brown fox jumps over the lazy dog while the cat watches. ";
    let mut prompt = String::new();
    while model.encoded_prompt_len(&prompt).unwrap_or(0) < (window + 8).min(1200) {
        prompt.push_str(sentence);
    }
    prompt
}

/// The batched prefill (one GEMM per projection; the sliding window masked
/// per row inside the attention kernel) must reach the same last hidden state
/// and the same greedy token as feeding the prompt one token at a time (GEMV,
/// window applied by narrowing the K/V views).
#[test]
#[ignore]
fn gemma4_model_batched_prefill_matches_sequential() {
    let Some(model) = load_gemma4("gemma4_model_batched_prefill_matches_sequential") else {
        return;
    };
    let m = model.gemma4.as_ref().expect("gemma4 model");
    let prompt = prompt_past_window(&model, m.cfg.sliding_window);

    let (ids, batched, _, _) = model
        .prefill_gemma4(m, &prompt, 0)
        .expect("batched prefill");
    let rows = ids.len();
    eprintln!(
        "prompt: {rows} tokens, sliding window {}",
        m.cfg.sliding_window
    );
    let batched_last = model
        .last_row(&batched, rows, m.cfg.hidden_size)
        .expect("last row");

    let mut caches = model.alloc_gemma4_kv_caches(m, rows).expect("alloc caches");
    let mut hidden = None;
    for (position, &id) in ids.iter().enumerate() {
        hidden = Some(
            model
                .forward_one_token_gemma4(m, id, position, &mut caches)
                .expect("sequential step"),
        );
    }
    let sequential_last = hidden.expect("at least one token");

    let seq = model.device.dtoh_sync_copy(&sequential_last).expect("dtoh");
    let bat = model.device.dtoh_sync_copy(&batched_last).expect("dtoh");
    assert_prefill_hidden_close(&seq, &bat, model.weights_dtype(), "gemma4 last hidden");

    let seq_logits = model.gemma4_logits(m, &sequential_last).expect("logits");
    let bat_logits = model.gemma4_logits(m, &batched_last).expect("logits");
    assert_eq!(
        Model::argmax(&seq_logits).unwrap(),
        Model::argmax(&bat_logits).unwrap(),
        "greedy token differs between sequential and batched prefill"
    );
    if let Some(cap) = m.cfg.final_logit_softcap {
        assert!(
            seq_logits.iter().all(|l| l.abs() <= cap),
            "soft-cap not applied"
        );
    }
}

/// `system1` scores (row gather + host soft-cap, teacher-forced
/// continuation) must equal the full-vocab soft-capped logits of the same
/// tokens computed independently, for 1-token and multi-token candidates.
#[test]
#[ignore]
fn gemma4_model_system1_scores_match_full_logits() {
    let Some(model) = load_gemma4("gemma4_model_system1_scores_match_full_logits") else {
        return;
    };
    let m = model.gemma4.as_ref().expect("gemma4 model");
    let prompt = "The capital of France is";
    let texts = [" Paris", " London", " the city of Paris", " Rome, Italy"];
    let candidates: Vec<System1Candidate> = texts
        .iter()
        .map(|t| System1Candidate {
            text: t.to_string(),
        })
        .collect();
    let response = model
        .system1_evaluate(prompt, &candidates, 1.0)
        .expect("system1");
    assert!(
        response.results.iter().any(|r| r.token_ids.len() > 1),
        "expected at least one multi-token candidate"
    );

    for result in &response.results {
        let (ids, hidden, mut caches, mut position) = model
            .prefill_gemma4(m, prompt, result.token_ids.len())
            .expect("prefill");
        let mut h = model
            .last_row(&hidden, ids.len(), m.cfg.hidden_size)
            .expect("last row");
        let mut want = 0.0f32;
        for (step, &id) in result.token_ids.iter().enumerate() {
            if step > 0 {
                h = model
                    .forward_one_token_gemma4(m, result.token_ids[step - 1], position, &mut caches)
                    .expect("step");
                position += 1;
            }
            want += model.gemma4_logits(m, &h).expect("logits")[id as usize];
        }
        let tol = 1e-3 * (1.0 + want.abs());
        eprintln!(
            "{:?}: tokens {:?} score {} full-logit sum {want}",
            result.text, result.token_ids, result.score
        );
        assert!(
            (result.score - want).abs() <= tol,
            "{:?}: system1 score {} vs full-logit sum {want}",
            result.text,
            result.score
        );
    }
}

/// Greedy generation runs end to end and stops within `max_new_tokens`.
#[test]
#[ignore]
fn gemma4_model_generates_without_error() {
    let Some(model) = load_gemma4("gemma4_model_generates_without_error") else {
        return;
    };
    let (ids, text) = model
        .generate(
            "Once upon a time",
            8,
            None,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("generate");
    eprintln!("generated {ids:?}: {text:?}");
    assert!(!ids.is_empty() && ids.len() <= 8);
}
