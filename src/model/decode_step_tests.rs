//! The workspace decode step (`decode_step.rs`) against the allocating one
//! (`forward_one_token_dense` + `lm_head_logits`): the full-vocab logits of every
//! step must be the same bits, since only the buffers differ. `#[ignore]`d: they
//! need a GPU and a dense model (`REFLEX_TEST_GGUF`, e.g. Qwen3-0.6B or TinyLlama);
//! `qwen2_tests.rs` runs the same check on the qwen2 fixture for the Q/K/V biases.

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

fn load(weights: WeightsDtype) -> Model {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let file = GgufFile::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("failed to load model")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Prefills `prompt`, then runs `steps` greedy decode steps through the workspace
/// and through the allocating path, each from its own prefill, and asserts every
/// step's logits are bit-identical.
pub(super) fn assert_workspace_step_matches_allocating(model: &Model, prompt: &str, steps: usize) {
    assert!(
        model.decode_ws_enabled(),
        "the workspace path does not serve this model"
    );
    let hidden_size = model.cfg.hidden_size;
    let eps = model.cfg.rmsnorm_eps;
    let run = |ws: bool| -> Vec<Vec<u32>> {
        let (ids, hidden_batched, mut k, mut v, mut position) = model
            .prefill_dense_batched(prompt, None, steps)
            .expect("prefill");
        let hidden = model
            .last_row(&hidden_batched, ids.len(), hidden_size)
            .expect("last row");
        let mut logits = model
            .lm_head_logits(&hidden, hidden_size, eps)
            .expect("logits");
        let mut out = Vec::new();
        for _ in 0..steps {
            let next = Model::argmax(&logits).expect("argmax");
            if ws {
                model
                    .decode_step_dense_ws(next, position, &mut k, &mut v, &mut logits)
                    .expect("workspace step");
            } else {
                let h = model
                    .forward_one_token_dense(next, position, &mut k, &mut v)
                    .expect("decode step");
                logits = model.lm_head_logits(&h, hidden_size, eps).expect("logits");
            }
            position += 1;
            out.push(bits(&logits));
        }
        out
    };
    let allocating = run(false);
    let workspace = run(true);
    for (i, (w, a)) in workspace.iter().zip(&allocating).enumerate() {
        assert!(
            w == a,
            "decode step {i}: workspace logits differ from the allocating path"
        );
    }
}

/// Short and ~800-token prompts (the long one splits decode attention, so the
/// workspace's partials buffer is used), f32 and f16 weights. A one-token
/// generate (prefill only) must not allocate the workspace.
#[test]
#[ignore]
fn decode_workspace_step_matches_allocating_bit_for_bit() {
    let readme = std::fs::read_to_string("README.md").expect("read README.md");
    let long: String = readme.chars().take(3000).collect();
    for weights in [WeightsDtype::F32, WeightsDtype::F16] {
        let model = load(weights);
        model
            .generate(
                "Hello",
                1,
                None,
                &SamplingParams::default(),
                |_| {},
                |_, _| {},
            )
            .expect("one-token generate");
        assert!(
            !model.decode_ws.allocated(),
            "a one-token generate allocated the decode workspace"
        );
        assert_workspace_step_matches_allocating(
            &model,
            "The lighthouse keeper climbed the stairs",
            12,
        );
        assert_workspace_step_matches_allocating(&model, &long, 6);
        assert!(model.decode_ws.allocated());
    }
}

/// `generate` with the workspace on and off: identical greedy ids over 32 tokens.
#[test]
#[ignore]
fn decode_workspace_generation_matches_allocating() {
    let model = load(WeightsDtype::F16);
    let run = |on: bool| {
        model.decode_ws.set_enabled(on);
        model
            .generate(
                "Write a long story about a lighthouse keeper. Once upon a time",
                32,
                None,
                &SamplingParams::default(),
                |_| {},
                |_, _| {},
            )
            .expect("generate")
            .0
    };
    let allocating = run(false);
    let workspace = run(true);
    assert_eq!(workspace.len(), 32, "generation stopped early");
    assert_eq!(workspace, allocating);
}
