//! Kolibri-1 forward pass on the synthetic fixture (`test-data/tiny-kolibri1*.gguf`,
//! see `kolibri_config_tests` and docs/design/kolibri.md). `#[ignore]`d: they
//! need a real GPU and the gitignored fixtures. Run with
//! `cargo test --release -- --ignored kolibri1_`.
//!
//! The fixture's sliding window is 16 tokens (not real Kolibri-1's 513), so the
//! longer prompts below exercise the window in prefill and decode.

use super::prefill_batching_tests::assert_prefill_hidden_close;
use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

const FIXTURE_F32: &str = "test-data/tiny-kolibri1-f32.gguf";
const FIXTURE_Q4KM: &str = "test-data/tiny-kolibri1.gguf";

/// Prompt, its token ids, and the patched llama.cpp's greedy continuation on
/// `tiny-kolibri1-f32.gguf` (CPU, f32 KV cache): llama.cpp `836d571` +
/// `kolibri1-llama.cpp.patch` (sha256 `e0d17c26...`, docs/design/kolibri.md),
/// via a 20-token greedy loop over the llama.h API, 2026-10-06. First-step
/// top-1/top-2 logit margins were 0.13 to 0.32, wide enough that summation-
/// order differences shouldn't flip a token.
struct Golden {
    prompt: &'static str,
    prompt_ids: &'static [u32],
    generated: &'static [u32],
}

const GOLDEN_F32: &[Golden] = &[
    Golden {
        prompt: "Once upon a time",
        prompt_ids: &[4571, 5019, 941, 1584],
        generated: &[
            63842, 62934, 91628, 65059, 10912, 92457, 12330, 1162, 12330, 117080, 52174, 117157,
            98955, 74835, 39807, 5632, 64818, 71904, 108212, 101249,
        ],
    },
    Golden {
        prompt: "Die Hauptstadt von Deutschland ist",
        prompt_ids: &[452, 22090, 493, 1678, 2459],
        generated: &[
            97901, 66073, 37198, 94906, 16925, 20141, 117817, 67412, 21023, 73137, 36140, 62141,
            78846, 127829, 30790, 17670, 60409, 56997, 17670, 42402,
        ],
    },
    Golden {
        prompt: "The quick brown fox jumps over the lazy dog while the river runs past the old mill, and the miller counts his sacks of flour one by one before the sun goes down over the hills.",
        prompt_ids: &[
            325, 13040, 10633, 39671, 61041, 1637, 262, 25423, 5926, 777, 262, 15437, 21681, 4917,
            262, 14191, 11344, 44, 286, 262, 11344, 268, 33417, 966, 109072, 279, 11618, 2403, 467,
            2403, 1042, 262, 14795, 7732, 3663, 1637, 262, 49927, 46,
        ],
        generated: &[
            82506, 117157, 33055, 123875, 108874, 45293, 11202, 69149, 120687, 89712, 53519,
            102950, 56127, 74592, 1527, 2528, 60690, 17323, 2678, 3557,
        ],
    },
    Golden {
        prompt: "<|im_start|>user\nErkläre bitte kurz, warum der Himmel blau ist, und nenne zwei Beispiele aus dem Alltag, in denen man Streuung beobachten kann.<|im_end|>\n<|im_start|>assistant\n",
        prompt_ids: &[
            127904, 1646, 10, 891, 98146, 15196, 14460, 44, 12484, 948, 19866, 26116, 2459, 44,
            420, 14163, 44798, 5617, 15346, 1953, 826, 10020, 44, 622, 18404, 836, 78568, 289,
            22915, 1826, 46, 127906, 10, 127904, 64090, 10,
        ],
        generated: &[
            127739, 29818, 51593, 26088, 117937, 48343, 85062, 85065, 116108, 14713, 7783, 120000,
            35513, 39532, 29037, 84367, 79224, 84677, 120824, 57941,
        ],
    },
];

fn load(path: &str, weights: WeightsDtype) -> Model {
    let file = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("failed to load model")
}

fn greedy(model: &Model, prompt: &str, n: usize) -> Vec<u32> {
    model
        .generate(
            prompt,
            n,
            None,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("generate")
        .0
}

/// The fixture loads as Kolibri layers with the pattern it was built with:
/// layer 4 full attention (NoPE, no window), the rest sliding with RoPE.
#[test]
#[ignore]
fn kolibri1_fixture_loads_as_kolibri_layers() {
    let model = load(FIXTURE_F32, WeightsDtype::F32);
    let modes: Vec<AttnMode> = model
        .layers
        .iter()
        .map(|l| match l {
            LayerWeights::Kolibri(k) => k.attn_mode,
            _ => panic!("expected only Kolibri layers"),
        })
        .collect();
    let sliding = AttnMode {
        rope: true,
        window: 16,
    };
    let full = AttnMode {
        rope: false,
        window: 0,
    };
    assert_eq!(
        modes,
        vec![sliding, sliding, sliding, sliding, full, sliding]
    );
    assert_eq!(model.expert_used_count, Some(4));
    assert_eq!(model.architecture_kind(), ArchitectureKind::Dense);
}

/// Byte-exact greedy agreement with the patched llama.cpp on the f32 fixture,
/// including prompts past the 16-token window (prefill) and 20 decode steps
/// past it.
#[test]
#[ignore]
fn kolibri1_f32_fixture_matches_patched_llamacpp() {
    let model = load(FIXTURE_F32, WeightsDtype::F32);
    let mut failures = Vec::new();
    for g in GOLDEN_F32 {
        let ids = model.tokenizer.encode(g.prompt).expect("encode");
        assert_eq!(ids, g.prompt_ids, "tokenization of {:?}", g.prompt);
        let got = greedy(&model, g.prompt, g.generated.len());
        let agree = got
            .iter()
            .zip(g.generated)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "{} prompt tokens: {agree}/{} generated tokens agree",
            ids.len(),
            g.generated.len()
        );
        if got != g.generated {
            failures.push(format!(
                "{:?}\n  reflex:    {got:?}\n  llama.cpp: {:?}",
                g.prompt, g.generated
            ));
        }
    }
    assert!(failures.is_empty(), "mismatches:\n{}", failures.join("\n"));
}

/// Sequential (per-token decode kernels, window applied by narrowing the K/V
/// views) vs batched (prefill kernels, window masked per row) prefill on a
/// prompt longer than the window, for both attention implementations and both
/// fixture files.
#[test]
#[ignore]
fn kolibri1_batched_prefill_matches_sequential() {
    let prompt = GOLDEN_F32[2].prompt;
    for (path, dtype) in [
        (FIXTURE_F32, WeightsDtype::F32),
        (FIXTURE_Q4KM, WeightsDtype::F32),
        (FIXTURE_Q4KM, WeightsDtype::F16),
    ] {
        let mut model = load(path, dtype);
        for imp in [AttnImpl::Online, AttnImpl::Legacy] {
            model.attn_impl = imp;
            let (seq_ids, seq_hidden, _, _, _) =
                model.prefill_dense(prompt, None, 0).expect("prefill_dense");
            let (batch_ids, batch_hidden, _, _, _) = model
                .prefill_dense_batched(prompt, None, 0)
                .expect("prefill_dense_batched");
            assert_eq!(seq_ids, batch_ids);
            assert!(
                seq_ids.len() > 16,
                "prompt must exceed the fixture's window"
            );
            let hidden_size = model.cfg.hidden_size;
            let last = model
                .last_row(&batch_hidden, batch_ids.len(), hidden_size)
                .expect("last_row");
            let seq_host = model.device.dtoh_sync_copy(&seq_hidden).unwrap();
            let batch_host = model.device.dtoh_sync_copy(&last).unwrap();
            assert_prefill_hidden_close(&seq_host, &batch_host, dtype, &format!("{path} {imp:?}"));
        }
    }
}

/// Host reference for one query row of sliding-window GQA attention: the
/// oracle for both attention kernels' new window masking (they were changed
/// together, so comparing them with each other alone would not catch a
/// shared off-by-one).
#[allow(clippy::too_many_arguments)]
fn host_attention_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hq: usize,
    hkv: usize,
    d: usize,
    qpos: usize,
    window: usize,
) -> Vec<f32> {
    let group = hq / hkv;
    let lo = if window > 0 {
        (qpos + 1).saturating_sub(window)
    } else {
        0
    };
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; hq * d];
    for h in 0..hq {
        let kvh = h / group;
        let qh = &q[h * d..(h + 1) * d];
        let scores: Vec<f32> = (lo..=qpos)
            .map(|p| {
                let kp = &k[(p * hkv + kvh) * d..(p * hkv + kvh + 1) * d];
                qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        for (j, p) in (lo..=qpos).enumerate() {
            let vp = &v[(p * hkv + kvh) * d..(p * hkv + kvh + 1) * d];
            for i in 0..d {
                out[h * d + i] += e[j] / sum * vp[i];
            }
        }
    }
    out
}

/// Both attention implementations, decode and prefill, with a sliding window,
/// against [`host_attention_row`]. Covers windows larger than, equal to and
/// smaller than the context, a resumed prefill (`start_pos > 0`), and a decode
/// long enough for the online kernel's split-K path.
#[test]
#[ignore]
fn kolibri1_sliding_window_attention_matches_host_reference() {
    let mut model = load(FIXTURE_F32, WeightsDtype::F32);
    let (hq, hkv, d) = (
        model.cfg.num_q_heads,
        model.cfg.num_kv_heads,
        model.cfg.head_dim,
    );
    let rand = |len: usize, seed: u64| -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    let mut worst = 0f32;
    // (start_pos, rows, window); rows == 1 is a decode step.
    for (case, &(start_pos, rows, window)) in [
        (0, 1, 16),
        (40, 1, 16),
        (15, 1, 16),
        (16, 1, 16),
        (1500, 1, 513),
        (0, 40, 16),
        (30, 25, 16),
        (0, 10, 16),
        (0, 40, 0),
        (5, 30, 1),
    ]
    .iter()
    .enumerate()
    {
        let positions = start_pos + rows;
        let q = rand(rows * hq * d, case as u64 * 3 + 1);
        let k = rand(positions * hkv * d, case as u64 * 3 + 2);
        let v = rand(positions * hkv * d, case as u64 * 3 + 3);
        let mut expected = Vec::with_capacity(rows * hq * d);
        for r in 0..rows {
            expected.extend(host_attention_row(
                &q[r * hq * d..(r + 1) * hq * d],
                &k,
                &v,
                hq,
                hkv,
                d,
                start_pos + r,
                window,
            ));
        }
        for imp in [AttnImpl::Online, AttnImpl::Legacy] {
            model.attn_impl = imp;
            let qd = model.device.htod_sync_copy(&q).unwrap();
            let kd = model.device.htod_sync_copy(&k).unwrap();
            let vd = model.device.htod_sync_copy(&v).unwrap();
            let out = if rows == 1 {
                model.attention(
                    &qd,
                    &kd.slice(..),
                    &vd.slice(..),
                    hq,
                    hkv,
                    d,
                    positions,
                    window,
                )
            } else {
                model.attention_prefill(
                    &qd,
                    &kd.slice(..),
                    &vd.slice(..),
                    hq,
                    hkv,
                    d,
                    start_pos,
                    rows,
                    window,
                )
            }
            .expect("attention");
            let got = model.device.dtoh_sync_copy(&out).unwrap();
            let diff = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{imp:?} start_pos={start_pos} rows={rows} window={window}: {diff:e}");
            assert!(
                diff < 1e-5,
                "{imp:?} start_pos={start_pos} rows={rows} window={window}: {diff:e}"
            );
            worst = worst.max(diff);
        }
    }
    eprintln!("worst max abs diff {worst:e}");
}

/// The Q4_K_M fixture runs end to end with the default (f16) weights and
/// stops at the requested length. Not compared with llama.cpp: its CPU path
/// quantizes activations to Q8_K for Q4_K/Q6_K dot products, and this
/// random-weight fixture's first-step top-1/top-2 margins (0.008 to 0.3) are
/// too narrow for token-exact agreement to mean much.
#[test]
#[ignore]
fn kolibri1_q4km_fixture_generates() {
    let model = load(FIXTURE_Q4KM, WeightsDtype::F16);
    for g in GOLDEN_F32 {
        let got = greedy(&model, g.prompt, 8);
        assert!(!got.is_empty() && got.len() <= 8);
        eprintln!(
            "{:?}: {got:?}",
            g.prompt.chars().take(30).collect::<String>()
        );
    }
}
