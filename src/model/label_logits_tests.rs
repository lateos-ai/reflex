//! `Model::label_logits` on the decision-model fixtures (GPU). Design:
//! docs/design/decision-models.md.

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// `test-data/tinyopenjev-for-testing-Q8_0.gguf`: llama.cpp's own OpenJev test
/// model, https://huggingface.co/ggml-org/tinyopenjev-for-testing-gguf (the first
/// 4 of OpenJev's 64 layers at hidden size 64, CC BY-NC 4.0, output not
/// meaningful). `qwen35` with OpenJev's decision metadata.
const FIXTURE: &str = "test-data/tinyopenjev-for-testing-Q8_0.gguf";

/// f32 weights: the comparisons below are about summation order, not rounding.
fn load() -> Option<Model> {
    if !std::path::Path::new(FIXTURE).exists() {
        eprintln!("Skipping: {FIXTURE} not present (see docs/DEVELOPMENT.md's test fixtures)");
        return None;
    }
    let file = GgufFile::open(FIXTURE).expect("open tinyopenjev");
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F32,
        lora_adapter: None,
    };
    Some(Model::load_with_options(device, &file, &opts).expect("load tinyopenjev"))
}

/// An OpenJev-style prompt for one question; the questions share the state.
fn prompt(question: &str, options: &[&str]) -> String {
    let mut p = String::from(
        "<|im_start|>user\nState:\nI was charged twice for my order last week and nobody has replied.\n\nQuestion: ",
    );
    p.push_str(question);
    p.push_str("\nOptions:\n");
    for (i, o) in options.iter().enumerate() {
        p.push_str(&format!("[{}] {o}: \n", (b'A' + i as u8) as char));
    }
    p.push_str("\nAnswer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    p
}

fn labels(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect()
}

/// One prompt, nothing shared: the logits are exactly System1's raw scores for
/// the same single-token candidates (same prefill, same gather).
#[test]
#[ignore]
fn label_logits_single_prompt_matches_system1_scores() {
    let Some(model) = load() else { return };
    let p = prompt(
        "Which team should handle this?",
        &["billing", "shipping", "technical"],
    );
    let out = model
        .label_logits(&[LabelPrompt {
            prompt: p.clone(),
            labels: labels(3),
        }])
        .expect("label_logits");
    assert_eq!(out.shared_tokens, 0);
    let r = &out.results[0];
    assert_eq!(r.label_token_ids, [32, 33, 34]);
    assert_eq!(r.tokens, model.tokenizer.encode(&p).unwrap().len());

    let candidates: Vec<System1Candidate> = labels(3)
        .into_iter()
        .map(|text| System1Candidate { text })
        .collect();
    let s1 = model
        .system1_evaluate(&p, &candidates, 1.0)
        .expect("system1");
    for (i, c) in s1.results.iter().enumerate() {
        assert_eq!(c.token_ids, [r.label_token_ids[i]]);
        assert_eq!(c.score.to_bits(), r.logits[i].to_bits(), "label {i}");
    }
}

/// Several prompts with a common start: the prefix is evaluated once, and each
/// prompt's logits match scoring it alone up to summation order (a different
/// split of the prefill into batches).
#[test]
#[ignore]
fn label_logits_shared_prefix_matches_separate_prompts() {
    let Some(model) = load() else { return };
    let prompts = [
        LabelPrompt {
            prompt: prompt(
                "Which team should handle this?",
                &["billing", "shipping", "technical"],
            ),
            labels: labels(3),
        },
        LabelPrompt {
            prompt: prompt(
                "How urgent is this?",
                &["can wait", "this week", "today", "right now"],
            ),
            labels: labels(4),
        },
        LabelPrompt {
            prompt: prompt("Is the customer angry?", &["yes", "no"]),
            labels: labels(2),
        },
    ];
    let shared = model.label_logits(&prompts).expect("shared");
    assert!(shared.shared_tokens > 10, "shared {}", shared.shared_tokens);
    for (p, r) in prompts.iter().zip(&shared.results) {
        let alone = model.label_logits(std::slice::from_ref(p)).expect("alone");
        assert_eq!(alone.shared_tokens, 0);
        let a = &alone.results[0];
        assert_eq!(a.tokens, r.tokens);
        assert_eq!(a.label_token_ids, r.label_token_ids);
        for (x, y) in a.logits.iter().zip(&r.logits) {
            assert!((x - y).abs() <= 1e-4 * x.abs().max(1.0), "{x} vs {y}");
        }
    }
}

/// The same prompt twice (lev's variants can coincide): every prompt still
/// evaluates at least its last token, and both results agree exactly.
#[test]
#[ignore]
fn label_logits_identical_prompts() {
    let Some(model) = load() else { return };
    let p = LabelPrompt {
        prompt: prompt("Which team?", &["billing", "shipping"]),
        labels: labels(2),
    };
    let out = model.label_logits(&[p.clone(), p]).expect("label_logits");
    assert_eq!(out.shared_tokens, out.results[0].tokens - 1);
    assert_eq!(out.results[0].logits, out.results[1].logits);
}

#[test]
#[ignore]
fn label_logits_rejects_bad_labels() {
    let Some(model) = load() else { return };
    let p = prompt("Which team?", &["billing", "shipping"]);
    for (labels, what) in [
        (vec![], "1 to"),
        (
            vec!["A".to_string(), "billing department".to_string()],
            "must be one token",
        ),
        (vec!["<|im_end|>".to_string()], "must be one token"),
    ] {
        let err = model
            .label_logits(&[LabelPrompt {
                prompt: p.clone(),
                labels,
            }])
            .unwrap_err();
        assert_eq!(err.kind(), "invalid_input");
        assert!(err.to_string().contains(what), "{err}");
    }
    let err = model.label_logits(&[]).unwrap_err();
    assert_eq!(err.kind(), "invalid_input");
}

/// `test-data/tiny-lev.gguf`: tinyopenjev's weights with lev's decision metadata
/// and template (`test-data/tiny-lev-src.tar.gz`). lev reads up to 255 labels,
/// `A`-`Z` then the two-letter codes that are one token; the engine must accept
/// every one the sidecar builds, and read them all in one gather.
const TINY_LEV: &str = "test-data/tiny-lev.gguf";

#[test]
#[ignore]
fn label_logits_tiny_lev_reads_all_255_labels() {
    if !std::path::Path::new(TINY_LEV).exists() {
        eprintln!("Skipping: {TINY_LEV} not present");
        return;
    }
    let file = GgufFile::open(TINY_LEV).expect("open tiny-lev");
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F32,
        lora_adapter: None,
    };
    let model = Model::load_with_options(device, &file, &opts).expect("load tiny-lev");
    let codes = ('A'..='Z')
        .map(|a| a.to_string())
        .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))));
    let labels: Vec<String> = codes
        .filter(|c| model.tokenizer.encode_no_special(c).unwrap().len() == 1)
        .take(255)
        .collect();
    assert_eq!(labels.len(), 255);
    assert_eq!(labels[254], "JT");
    let p = prompt("Which team?", &["billing", "shipping"]);
    let out = model
        .label_logits(&[LabelPrompt {
            prompt: p.clone(),
            labels: labels.clone(),
        }])
        .expect("label_logits");
    let r = &out.results[0];
    assert_eq!(r.logits.len(), 255);
    assert!(r.logits.iter().all(|x| x.is_finite()));
    // The same rows read in any subset give the same values.
    let some = model
        .label_logits(&[LabelPrompt {
            prompt: p,
            labels: vec![labels[254].clone(), labels[0].clone()],
        }])
        .expect("subset");
    assert_eq!(some.results[0].logits, [r.logits[254], r.logits[0]]);
}
