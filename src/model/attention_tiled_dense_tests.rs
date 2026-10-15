//! The tiled prefill attention kernels behind `Model::attention_prefill` (head dims 64,
//! 128 and 256, `TiledAttnKernels`) against `attention_online_kernel`, on random Q/K/V
//! at the GQA groups real files use, including the ones that don't divide the 64
//! queries per block (3, 6, 7) and leave padded slots. Then a whole long-prompt prefill
//! with the tiled kernels on and off. `#[ignore]`d: they need a GPU and a dense model
//! (`REFLEX_TEST_GGUF`, e.g. Qwen3-0.6B); the kernel test only uses its kernels.

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;
use rand::{Rng, SeedableRng};

fn load_test_model(weights: WeightsDtype) -> Model {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let file = GgufFile::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("failed to load model")
}

struct Case {
    rows: usize,
    start_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window: usize,
}

#[test]
#[ignore]
fn attention_tiled_small_dims_match_online_kernel() {
    let model = load_test_model(WeightsDtype::DEFAULT);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x7a12);
    let case = |rows, start_pos, num_q_heads, num_kv_heads, head_dim, window| Case {
        rows,
        start_pos,
        num_q_heads,
        num_kv_heads,
        head_dim,
        window,
    };
    let cases = [
        // Qwen3-0.6B (group 2) and Qwen3-8B / Llama 3 8B (group 4), rows not a
        // multiple of the block.
        case(530, 0, 16, 8, 128, 0),
        case(301, 0, 32, 8, 128, 0),
        // Padded groups: Qwen2.5-1.5B (6), Granite 4.0-H (3), Qwen2.5-7B (7).
        case(257, 0, 12, 2, 128, 0),
        case(200, 0, 12, 4, 128, 0),
        case(150, 0, 28, 4, 128, 0),
        // A resumed prefill (imported KV) with a sliding window: Kolibri/Mistral style.
        case(129, 300, 32, 8, 128, 96),
        // Head dim 64: Qwen2.5-0.5B (group 7) and TinyLlama (group 8).
        case(333, 0, 14, 2, 64, 0),
        case(140, 17, 32, 4, 64, 0),
        // Head dim 256: Qwen3.5's gated attention (group 4).
        case(260, 0, 8, 2, 256, 0),
        // The smallest shape the tiled path accepts.
        case(2, 0, 16, 8, 128, 0),
        // Few rows after a long cached prefix (a pinned or imported prefix), which
        // `ATTN_TILED_MIN_START` sends to the tiled kernel: every head dim, a padded
        // group and a sliding window.
        case(41, 1489, 16, 8, 128, 0),
        case(5, 1500, 16, 8, 128, 0),
        case(9, 600, 12, 2, 128, 0),
        case(7, 900, 32, 8, 128, 96),
        case(3, 700, 14, 2, 64, 0),
        case(17, 400, 8, 2, 256, 0),
    ];
    for c in cases {
        let kv_len = c.start_pos + c.rows;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut gen =
            |n: usize, amp: f32| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-amp..amp)).collect() };
        let q = gen(c.rows * c.num_q_heads * c.head_dim, 2.0);
        let k = gen(kv_len * c.num_kv_heads * c.head_dim, 2.0);
        let v = gen(kv_len * c.num_kv_heads * c.head_dim, 1.0);
        let (q_d, k_d, v_d) = (
            model.device.htod_sync_copy(&q).unwrap(),
            model.device.htod_sync_copy(&k).unwrap(),
            model.device.htod_sync_copy(&v).unwrap(),
        );
        let kv_stride = c.num_kv_heads * c.head_dim;
        let shape = || OnlineAttnShape {
            rows: c.rows,
            start_pos: c.start_pos,
            num_q_heads: c.num_q_heads,
            group_size: c.num_q_heads / c.num_kv_heads,
            qk_dim: c.head_dim,
            v_dim: c.head_dim,
            k_pos_stride: kv_stride,
            k_head_stride: c.head_dim,
            v_pos_stride: kv_stride,
            v_head_stride: c.head_dim,
            scale,
            window: c.window,
        };
        assert!(attn_tiled_fits(&shape()), "shape should be supported");
        let kernel = model
            .attn_tiled
            .kernel(&model.device, c.head_dim)
            .expect("load tiled module")
            .expect("a kernel for this head dim");
        let (k_v, v_v) = (k_d.slice(..), v_d.slice(..));
        let tiled = launch_attention_tiled(&model.device, kernel, &q_d, &k_v, &v_v, &shape())
            .expect("tiled launch");
        let online = model
            .attention_online(&q_d, &k_v, &v_v, shape())
            .expect("online launch");
        let (t, o) = (
            model.device.dtoh_sync_copy(&tiled).unwrap(),
            model.device.dtoh_sync_copy(&online).unwrap(),
        );
        let max_diff = t
            .iter()
            .zip(&o)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let label = format!(
            "rows={} start={} heads={}/{} d={} window={}",
            c.rows, c.start_pos, c.num_q_heads, c.num_kv_heads, c.head_dim, c.window
        );
        assert!(
            t.iter().all(|x| x.is_finite()),
            "{label}: non-finite output"
        );
        assert!(
            max_diff <= 2e-5,
            "{label}: max |tiled - online| = {max_diff}"
        );
        eprintln!("{label}: max diff {max_diff:e}");
    }
}

/// A whole prefill of a prompt above `ATTN_TILED_MIN_ROWS`, tiled vs online: the
/// first token's logits agree closely and the greedy ids match. Also checks the tiled
/// module was really loaded by the tiled run and not by a short prompt. f32 weights:
/// with f16 ones, every GEMM rounds its activations to f16, which turns the
/// attention's ~1e-6 summation-order differences into ~5e-4 relative L2 on the
/// logits (measured on a T4), hiding a real kernel error of that size.
#[test]
#[ignore]
fn attention_tiled_prefill_matches_online_end_to_end() {
    let model = load_test_model(WeightsDtype::F32);
    let readme = std::fs::read_to_string("README.md").expect("read README.md");
    let prompt: String = readme.chars().take(3000).collect();
    let rows = model.encoded_prompt_len(&prompt).unwrap();
    assert!(
        rows >= 4 * ATTN_TILED_MIN_ROWS,
        "prompt too short: {rows} tokens"
    );
    let run = |on: bool| {
        model.attn_tiled.set_enabled(on);
        let mut logits = Vec::new();
        let (ids, _) = model
            .generate(
                &prompt,
                8,
                None,
                &SamplingParams::default(),
                |l| logits = l.to_vec(),
                |_, _| {},
            )
            .expect("generate");
        (ids, logits)
    };
    model
        .generate("Hi", 2, None, &SamplingParams::default(), |_| {}, |_, _| {})
        .expect("short generate");
    let head_dim = model.cfg.head_dim;
    assert!(
        !model.attn_tiled.loaded(head_dim),
        "a short prompt loaded the tiled module"
    );
    let (online_ids, online_logits) = run(false);
    let (tiled_ids, tiled_logits) = run(true);
    assert!(
        model.attn_tiled.loaded(head_dim),
        "the long prompt didn't use the tiled path"
    );
    let diff: f64 = online_logits
        .iter()
        .zip(&tiled_logits)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let norm: f64 = online_logits.iter().map(|&a| (a as f64).powi(2)).sum();
    let rel_l2 = (diff / norm).sqrt();
    eprintln!(
        "{rows} tokens, d {head_dim}: first-token logits rel_l2={rel_l2:e}, ids {tiled_ids:?}"
    );
    assert!(rel_l2 < 1e-4, "rel_l2 {rel_l2:e}");
    assert_eq!(tiled_ids, online_ids);
}

/// Prints the tiled and online kernels' times from 8 to 2,048 rows (fresh prefill,
/// d 128 at groups 2 and 4, d 64 at group 7), the data behind `ATTN_TILED_MIN_ROWS`.
/// Asserts only that both ran.
#[test]
#[ignore]
fn attention_tiled_crossover_timings() {
    let model = load_test_model(WeightsDtype::DEFAULT);
    let dev = &model.device;
    for (num_q_heads, num_kv_heads, head_dim) in [(16, 8, 128), (32, 8, 128), (14, 2, 64)] {
        for rows in [8, 16, 32, 64, 128, 256, 512, 2048] {
            let kv_stride = num_kv_heads * head_dim;
            let q = dev
                .alloc_zeros::<f32>(rows * num_q_heads * head_dim)
                .unwrap();
            let k = dev.alloc_zeros::<f32>(rows * kv_stride).unwrap();
            let v = dev.alloc_zeros::<f32>(rows * kv_stride).unwrap();
            let shape = || OnlineAttnShape {
                rows,
                start_pos: 0,
                num_q_heads,
                group_size: num_q_heads / num_kv_heads,
                qk_dim: head_dim,
                v_dim: head_dim,
                k_pos_stride: kv_stride,
                k_head_stride: head_dim,
                v_pos_stride: kv_stride,
                v_head_stride: head_dim,
                scale: 0.1,
                window: 0,
            };
            let kernel = model.attn_tiled.kernel(dev, head_dim).unwrap().unwrap();
            let (kv, vv) = (k.slice(..), v.slice(..));
            let time = |f: &dyn Fn()| {
                f();
                dev.synchronize().unwrap();
                let iters = 20;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                dev.synchronize().unwrap();
                t.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            let tiled_us = time(&|| {
                launch_attention_tiled(dev, kernel, &q, &kv, &vv, &shape()).unwrap();
            });
            let online_us = time(&|| {
                model.attention_online(&q, &kv, &vv, shape()).unwrap();
            });
            eprintln!(
                "heads {num_q_heads}/{num_kv_heads} d {head_dim} rows {rows}: tiled {tiled_us:.1} us, online {online_us:.1} us"
            );
        }
    }
}

/// Like [`attention_tiled_crossover_timings`], but for a resumed prefill: few rows
/// after `start_pos` cached positions (a pinned prefix or an imported cache). The
/// online kernel reads every cached K/V row once per (row, query head), so its cost
/// grows with `start_pos` much faster than the tiled kernel's. The data behind
/// `ATTN_TILED_MIN_START`. Asserts only that both ran.
#[test]
#[ignore]
fn attention_tiled_crossover_timings_resumed() {
    let model = load_test_model(WeightsDtype::DEFAULT);
    let dev = &model.device;
    for (num_q_heads, num_kv_heads, head_dim) in [(16, 8, 128), (32, 8, 128), (14, 2, 64)] {
        let kernel = model.attn_tiled.kernel(dev, head_dim).unwrap().unwrap();
        for start_pos in [0, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            for rows in [2, 4, 8, 16, 32, 48, 63] {
                let kv_stride = num_kv_heads * head_dim;
                let q = dev
                    .alloc_zeros::<f32>(rows * num_q_heads * head_dim)
                    .unwrap();
                let k = dev
                    .alloc_zeros::<f32>((start_pos + rows) * kv_stride)
                    .unwrap();
                let v = dev
                    .alloc_zeros::<f32>((start_pos + rows) * kv_stride)
                    .unwrap();
                let shape = || OnlineAttnShape {
                    rows,
                    start_pos,
                    num_q_heads,
                    group_size: num_q_heads / num_kv_heads,
                    qk_dim: head_dim,
                    v_dim: head_dim,
                    k_pos_stride: kv_stride,
                    k_head_stride: head_dim,
                    v_pos_stride: kv_stride,
                    v_head_stride: head_dim,
                    scale: 0.1,
                    window: 0,
                };
                let (kv, vv) = (k.slice(..), v.slice(..));
                let time = |f: &dyn Fn()| {
                    f();
                    dev.synchronize().unwrap();
                    let iters = 20;
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        f();
                    }
                    dev.synchronize().unwrap();
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                };
                let tiled_us = time(&|| {
                    launch_attention_tiled(dev, kernel, &q, &kv, &vv, &shape()).unwrap();
                });
                let online_us = time(&|| {
                    model.attention_online(&q, &kv, &vv, shape()).unwrap();
                });
                eprintln!(
                    "RESUMED heads {num_q_heads}/{num_kv_heads} d {head_dim} start {start_pos} rows {rows}: tiled {tiled_us:.1} us, online {online_us:.1} us"
                );
            }
        }
    }
}

/// The dispatch rule from the `attention_tiled_crossover_timings*` data: fresh
/// prompts switch at `ATTN_TILED_MIN_ROWS`, so short ones never load a tiled module;
/// resumed ones switch once `rows * group_size` fills a block.
#[test]
fn attn_tiled_preferred_follows_rows_and_cached_prefix() {
    assert!(!attn_tiled_preferred(41, 0, 2));
    assert!(attn_tiled_preferred(64, 0, 2));
    assert!(attn_tiled_preferred(41, 1489, 2));
    assert!(attn_tiled_preferred(32, 64, 2));
    assert!(!attn_tiled_preferred(31, 1489, 2));
    assert!(!attn_tiled_preferred(41, 63, 2));
    assert!(attn_tiled_preferred(16, 64, 4));
    assert!(!attn_tiled_preferred(8, 2048, 4));
}
