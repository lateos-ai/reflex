//! GPU tests for `system1` with multi-token candidates on the Qwen3.5 hybrid
//! (`Model::system1_evaluate_hybrid`): every candidate's score must equal the
//! sum of its tokens' full-vocab logits from its own, independent prefill and
//! teacher-forced continuation, on a fresh prompt and after a cached prefix
//! (pinned, and read back from a cache file). The real model runs from
//! `REFLEX_TEST_GGUF` (the nightly's `HYBRID_GGUF`, e.g.
//! `Qwen3.5-0.8B-Q4_K_M.gguf`), the MoE fixture from
//! `test-data/tiny-qwen35moe.gguf`.

use super::*;
use crate::kv_io::ImportedKv;
use crate::sampling::SamplingParams;
use cudarc::driver::CudaDevice;

const QWEN35MOE_FIXTURE: &str = "test-data/tiny-qwen35moe.gguf";
const PREFIX: &str = "You are a careful assistant in a team of agents. Answer briefly.";
const PROMPT: &str = " The capital of France is";
/// Two multi-token candidates in a row (the second one sees whatever the
/// first left in the recurrent state), single-token ones before and after.
const CANDIDATES: [&str; 5] = [
    " Paris",
    " the city of Paris",
    " Rome, Italy",
    " London",
    " a",
];

fn load(path: &str, weights: WeightsDtype) -> Model {
    let file = crate::gguf::GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    let model = Model::load_with_options(device, &file, &opts).expect("load model");
    assert!(
        model.hybrid.is_some(),
        "{path} is not a Qwen3.5 hybrid model"
    );
    model
}

fn candidates(texts: &[&str]) -> Vec<System1Candidate> {
    texts
        .iter()
        .map(|t| System1Candidate {
            text: t.to_string(),
        })
        .collect()
}

/// `resume` as the hybrid path's prefix source.
fn hybrid_src<'a>(resume: Resume<'a>, copy_ms: &'a Cell<Option<f64>>) -> Option<HybridPrefix<'a>> {
    match resume {
        Resume::Fresh => None,
        Resume::File(ImportedKv::Hybrid(cache)) => Some(PrefixSource::Host { cache, copy_ms }),
        Resume::Pinned(PinnedPrefix {
            state: DeviceState::Hybrid(state),
        }) => Some(PrefixSource::Device { state, copy_ms }),
        _ => panic!("not a hybrid prefix"),
    }
}

/// The sum of `ids`' full-vocab logits after `prompt`, from a prefill of its
/// own and a teacher-forced continuation: the score `system1` must give.
fn reference_score(model: &Model, prompt: &str, resume: Resume<'_>, ids: &[u32]) -> f32 {
    let h = model.hybrid.as_ref().unwrap();
    let copy_ms = Cell::new(None);
    let (prompt_ids, hidden, mut states, mut position) = model
        .prefill_hybrid_batched(h, prompt, hybrid_src(resume, &copy_ms), ids.len())
        .expect("reference prefill");
    let (hidden_size, eps) = (h.attn_cfg.hidden_size, h.attn_cfg.rmsnorm_eps);
    let mut hidden = model
        .last_row(&hidden, prompt_ids.len(), hidden_size)
        .expect("last row");
    let mut want = 0.0f32;
    for (step, &id) in ids.iter().enumerate() {
        if step > 0 {
            hidden = model
                .forward_one_token_hybrid(h, ids[step - 1], position, &mut states)
                .expect("reference step");
            position += 1;
        }
        want += model
            .lm_head_logits(&hidden, hidden_size, eps)
            .expect("logits")[id as usize];
    }
    want
}

/// Scores [`CANDIDATES`] after `prompt` (resumed from `resume`) and checks each
/// against [`reference_score`] within `rel * (1 + |want|)`; also checks that the
/// single-token scores are bit-identical to a request with only those
/// candidates (the snapshot and restores leave them alone). Returns the largest
/// difference, for the log.
fn check_multi_token(model: &Model, prompt: &str, resume: Resume<'_>, rel: f32, tag: &str) -> f32 {
    let (resp, _) = model
        .system1_evaluate_resumed(prompt, resume, &candidates(&CANDIDATES), 1.0)
        .expect("system1");
    let multi = resp
        .results
        .iter()
        .filter(|r| r.token_ids.len() > 1)
        .count();
    assert!(
        multi >= 2,
        "{tag}: expected at least two multi-token candidates"
    );

    let mut max_diff = 0.0f32;
    for r in &resp.results {
        let want = reference_score(model, prompt, resume, &r.token_ids);
        let diff = (r.score - want).abs();
        max_diff = max_diff.max(diff);
        eprintln!(
            "{tag} {:?}: tokens {:?} score {} reference {want} diff {diff:e}",
            r.text, r.token_ids, r.score
        );
        assert!(
            diff <= rel * (1.0 + want.abs()),
            "{tag} {:?}: system1 score {} vs reference {want}",
            r.text,
            r.score
        );
    }

    let singles: Vec<&str> = resp
        .results
        .iter()
        .filter(|r| r.token_ids.len() == 1)
        .map(|r| r.text.as_str())
        .collect();
    let (alone, _) = model
        .system1_evaluate_resumed(prompt, resume, &candidates(&singles), 1.0)
        .expect("system1, single-token candidates");
    for a in &alone.results {
        let mixed = resp.results.iter().find(|r| r.text == a.text).unwrap();
        assert_eq!(
            mixed.score.to_bits(),
            a.score.to_bits(),
            "{tag} {:?}: single-token score changed by multi-token candidates",
            a.text
        );
    }
    eprintln!("{tag}: max |score - reference| = {max_diff:e}");
    max_diff
}

/// Pins [`PREFIX`], exports it, and checks [`PROMPT`]'s candidates after it
/// from the pinned prefix and from the file. Scoring twice from the same pinned
/// prefix must give bit-identical results: the prefix is copied, never written.
fn check_after_prefix(model: &Model, rel: f32, tag: &str) {
    let (_, _, pinned, _) = model
        .generate_pin_prefix(
            PREFIX,
            Resume::Fresh,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("pin prefix");
    let file = model.download_prefix(&pinned).expect("download prefix");

    check_multi_token(
        model,
        PROMPT,
        Resume::Pinned(&pinned),
        rel,
        &format!("{tag} pinned"),
    );
    check_multi_token(
        model,
        PROMPT,
        Resume::File(&file),
        rel,
        &format!("{tag} file"),
    );

    let score = |resume| {
        let (resp, _) = model
            .system1_evaluate_resumed(PROMPT, resume, &candidates(&CANDIDATES), 1.0)
            .expect("system1");
        resp.results
            .iter()
            .map(|r| r.score.to_bits())
            .collect::<Vec<_>>()
    };
    let first = score(Resume::Pinned(&pinned));
    assert_eq!(
        first,
        score(Resume::Pinned(&pinned)),
        "{tag}: scoring changed the pinned prefix"
    );
    assert_eq!(
        first,
        score(Resume::File(&file)),
        "{tag}: pinned and file prefixes score differently"
    );
}

/// Real Qwen3.5 (`REFLEX_TEST_GGUF`), default weights, fresh prompt. The
/// reference sums full-vocab GEMV logits where `system1` gathers rows, so the
/// two differ by summation order only.
#[test]
#[ignore]
fn hybrid_system1_multi_token_matches_full_prefills() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a Qwen3.5 GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    check_multi_token(
        &model,
        &format!("{PREFIX}{PROMPT}"),
        Resume::Fresh,
        1e-3,
        "qwen35",
    );
}

/// Real Qwen3.5 (`REFLEX_TEST_GGUF`), after a pinned and an imported prefix.
#[test]
#[ignore]
fn hybrid_system1_multi_token_after_prefix_matches_full_prefills() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a Qwen3.5 GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    check_after_prefix(&model, 1e-3, "qwen35");
}

/// The Qwen3.5-MoE fixture, f32 weights, fresh prompt and after a prefix.
#[test]
#[ignore]
fn qwen35moe_fixture_system1_multi_token_matches_full_prefills() {
    let model = load(QWEN35MOE_FIXTURE, WeightsDtype::F32);
    check_multi_token(
        &model,
        &format!("{PREFIX}{PROMPT}"),
        Resume::Fresh,
        1e-4,
        "qwen35moe",
    );
    check_after_prefix(&model, 1e-4, "qwen35moe");
}
