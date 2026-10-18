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

/// `WeightLoadPipeline::upload_state` widens f16 values exactly and lands
/// every value where it belongs: more than one launch's worth of runs in one
/// chunk, a buffer spanning several chunks, f32 and f16 jobs interleaved, and
/// the same values read from an f16 cache file's ranges.
#[test]
#[ignore]
fn upload_state_widens_f16_runs_exactly() {
    use crate::kv_io::{KvDtype, KvValues};
    let model = load_f32(QWEN35MOE_FIXTURE);
    let value = |i: usize, salt: usize| ((i * 7 + salt * 131) % 4093) as f32 / 97.0 - 21.0;
    // 150 small buffers (3 launches in one chunk), one of 5M values (10 MB in
    // f16, two chunks), and f32 buffers between them.
    let mut lens: Vec<usize> = (0..150).map(|k| 1 + (k * 37) % 900).collect();
    lens.insert(75, 5_000_000);
    let host: Vec<KvValues> = lens
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let xs: Vec<f32> = (0..n).map(|i| value(i, k)).collect();
            if k % 5 == 4 {
                KvValues::Host(xs)
            } else {
                KvValues::HostF16(
                    xs.iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect(),
                )
            }
        })
        .collect();
    let expect: Vec<Vec<f32>> = host.iter().map(|v| v.to_f32().unwrap()).collect();
    let check = |values: &[KvValues], what: &str| {
        let bufs: Vec<CudaSlice<f32>> = values
            .iter()
            .map(|v| model.device.alloc_zeros::<f32>(v.len() + 3).unwrap())
            .collect();
        let jobs: Vec<_> = bufs.iter().map(|b| *b.device_ptr()).zip(values).collect();
        model
            .dequant_pipeline
            .borrow_mut()
            .upload_state(&jobs, &model.dequant_kernels.f16_to_f32_runs)
            .expect("upload_state");
        for (k, (b, want)) in bufs.iter().zip(&expect).enumerate() {
            let got = model.device.dtoh_sync_copy(b).unwrap();
            assert_eq!(&got[..want.len()], &want[..], "{what}: buffer {k} values");
            assert_eq!(
                &got[want.len()..],
                &[0.0; 3],
                "{what}: buffer {k} wrote past its end"
            );
        }
    };
    check(&host, "host values");

    // The same values through a version-5 file: one dense layer per pair.
    let n_pairs = host.len() / 2;
    let path = std::env::temp_dir().join(format!("reflex_upload_runs_{}.cskv", std::process::id()));
    let p = path.to_str().unwrap();
    let f16_only: Vec<KvValues> = host
        .iter()
        .take(n_pairs * 2)
        .map(|v| match v {
            KvValues::Host(xs) => KvValues::HostF16(
                xs.iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            ),
            other => other.clone(),
        })
        .collect();
    // Dense files need equal K/V lengths per layer: pad every buffer to one length.
    let len = 1024;
    let padded: Vec<KvValues> = f16_only
        .iter()
        .map(|v| {
            let mut xs = v.to_f32().unwrap();
            xs.resize(len, 0.5);
            KvValues::HostF16(
                xs.iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            )
        })
        .collect();
    let cache = crate::kv_io::DenseKvCache {
        seq_len: 1,
        num_kv_heads: 1,
        head_dim: len,
        k_caches: padded.iter().step_by(2).cloned().collect(),
        v_caches: padded.iter().skip(1).step_by(2).cloned().collect(),
        space_prefix_next: Some(true),
    };
    crate::kv_io::export_dense_kv(p, &cache).unwrap();
    let back = crate::kv_io::import_dense_kv(p).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(back.k_caches.iter().all(|v| v.dtype() == KvDtype::F16));
    let from_file: Vec<KvValues> = back
        .k_caches
        .iter()
        .zip(&back.v_caches)
        .flat_map(|(k, v)| [k.clone(), v.clone()])
        .collect();
    let bufs: Vec<CudaSlice<f32>> = from_file
        .iter()
        .map(|_| model.device.alloc_zeros::<f32>(len).unwrap())
        .collect();
    let jobs: Vec<_> = bufs
        .iter()
        .map(|b| *b.device_ptr())
        .zip(&from_file)
        .collect();
    model
        .dequant_pipeline
        .borrow_mut()
        .upload_state(&jobs, &model.dequant_kernels.f16_to_f32_runs)
        .expect("upload_state from a file");
    for (k, (b, v)) in bufs.iter().zip(&padded).enumerate() {
        assert_eq!(
            model.device.dtoh_sync_copy(b).unwrap(),
            v.to_f32().unwrap(),
            "file buffer {k}"
        );
    }
}
