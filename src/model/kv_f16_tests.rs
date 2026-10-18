//! GPU measurement behind the f16 cache-file format (`crate::kv_io`, versions
//! 5-7): what storing a prefix's state in f16 does to the output after it, per
//! kind of buffer. A pinned prefix's buffers are rounded to f16 in place
//! (`f16_roundtrip_kernel`, the same round-to-nearest-even an f16 export uses;
//! importing widens exactly), and the run resumed from it is compared with the
//! run resumed from the untouched prefix. f32 weights, so the cache rounding is
//! the only difference.
//!
//! The hybrid case decides version 6's layout: rounding only the attention K/V
//! against also rounding the GatedDeltaNet `conv_state`/`recurrent`.

use super::*;
use crate::sampling::SamplingParams;
use cudarc::driver::CudaDevice;

const QWEN35MOE_FIXTURE: &str = "test-data/tiny-qwen35moe.gguf";
const CONTINUATION: &str = " Summarize the rules above in one sentence:";
const N: usize = 32;

fn load_f32(path: &str) -> Model {
    let file = crate::gguf::GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F32,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("load model")
}

/// A ~600-token system prompt, long enough that the recurrent state has
/// folded in many tokens.
fn prefix() -> String {
    (1..=20)
        .map(|i| {
            format!(
                "Rule {i}: when you receive a note about step {i}, check its inputs, state your assumptions and keep the output format stable.\n"
            )
        })
        .collect()
}

/// Which buffers to round.
#[derive(Clone, Copy, Debug)]
enum Round {
    None,
    Attention,
    AttentionAndRecurrent,
}

fn round_in_place(model: &Model, p: &mut PinnedPrefix, what: Round) {
    let round = |b: &mut CudaSlice<f32>| {
        f16_roundtrip_in_place(&model.dequant_kernels, b).expect("f16 roundtrip")
    };
    match (&mut p.state, what) {
        (_, Round::None) => {}
        (DeviceState::Dense(d), _) => d.k_caches.iter_mut().chain(&mut d.v_caches).for_each(round),
        (DeviceState::Mla(d), _) => d.kv_caches.iter_mut().for_each(round),
        (DeviceState::Hybrid(h), what) => {
            for layer in &mut h.layers {
                match layer {
                    HybridLayerState::Attn { k_cache, v_cache } => {
                        round(k_cache);
                        round(v_cache);
                    }
                    HybridLayerState::Gdn {
                        conv_state,
                        recurrent,
                    } => {
                        if matches!(what, Round::AttentionAndRecurrent) {
                            round(conv_state);
                            round(recurrent);
                        }
                    }
                }
            }
        }
    }
    model.device.synchronize().expect("sync");
}

/// First-token logits and `N` greedy ids after `CONTINUATION`, resumed from
/// the prefix with `what` rounded.
fn resume(model: &Model, what: Round) -> (Vec<f32>, Vec<u32>) {
    let greedy = SamplingParams::default();
    let (_, _, mut pinned, _) = model
        .generate_pin_prefix(&prefix(), Resume::Fresh, &greedy, |_| {}, |_, _| {})
        .expect("pin prefix");
    round_in_place(model, &mut pinned, what);
    let mut logits = Vec::new();
    let (ids, _, _) = model
        .generate_resumed(
            CONTINUATION,
            N,
            Resume::Pinned(&pinned),
            &greedy,
            |l| logits = l.to_vec(),
            |_, _| {},
        )
        .expect("resume");
    (logits, ids)
}

/// `(max |a - b|, ||a - b|| / ||b||)`.
fn diff(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max = 0f32;
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        max = max.max((x - y).abs());
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (max, (num / den).sqrt())
}

/// Prints each rounding's effect; asserts only that every run is finite and
/// that rounding K/V alone stays within f16 storage's expected size (rel L2
/// under 1e-2), which the f16 relay tests then check through real files.
fn report(model: &Model, tag: &str, cases: &[Round]) {
    let (base, base_ids) = resume(model, Round::None);
    for &what in cases {
        let (logits, ids) = resume(model, what);
        assert!(logits.iter().all(|x| x.is_finite()), "{tag} {what:?}");
        let (max, rel) = diff(&logits, &base);
        let same = ids
            .iter()
            .zip(&base_ids)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "F16_STATE {tag} {what:?}: first-token logits max |d| {max:.3e}, rel L2 {rel:.3e}; greedy ids identical for the first {same}/{N}"
        );
        if matches!(what, Round::Attention) {
            assert!(
                rel < 1e-2,
                "{tag}: K/V rounding moved logits by rel L2 {rel}"
            );
        }
    }
}

/// `REFLEX_TEST_GGUF` (the nightly's `HYBRID_GGUF`, Qwen3.5-0.8B), else the
/// qwen35moe fixture.
#[test]
#[ignore]
fn hybrid_f16_state_rounding_errors() {
    let path = std::env::var("REFLEX_TEST_GGUF").unwrap_or_else(|_| QWEN35MOE_FIXTURE.into());
    let model = load_f32(&path);
    assert!(model.hybrid.is_some(), "{path} is not a Qwen3.5 hybrid");
    report(
        &model,
        "hybrid",
        &[Round::Attention, Round::AttentionAndRecurrent],
    );
}

/// `REFLEX_TEST_GGUF` (the nightly's `DENSE_GGUF`): the dense K/V's share.
#[test]
#[ignore]
fn dense_f16_state_rounding_errors() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load_f32(&path);
    report(&model, "dense", &[Round::Attention]);
}
