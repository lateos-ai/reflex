//! The GQA decode attention kernels (`kernels_cuda/attention_decode.cu`, behind
//! `Model::attention`) against `attention_online_kernel` on random Q/K/V, then a whole
//! long-context generation with each. `#[ignore]`d: they need a GPU and a dense model
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

#[test]
#[ignore]
fn attention_decode_gqa_matches_online_kernel() {
    let model = load_test_model(WeightsDtype::DEFAULT);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xdec0);
    // (query heads, KV heads, head dim, cached positions, window)
    let cases = [
        (16, 8, 128, 1, 0),
        (16, 8, 128, 31, 0),
        (16, 8, 128, 1550, 0),
        (32, 8, 128, 5000, 0),
        (12, 2, 128, 700, 0), // group 6: two heads per block
        (28, 4, 128, 333, 0), // group 7: one head per block
        (32, 4, 64, 2049, 0),
        (14, 2, 64, 90, 0),
        (8, 2, 256, 1200, 0),
        (16, 1, 256, 400, 0), // group 16: four chunks of four
        (32, 8, 128, 900, 256),
    ];
    for (num_q_heads, num_kv_heads, head_dim, seq_len, window) in cases {
        let group_size = num_q_heads / num_kv_heads;
        let kv_stride = num_kv_heads * head_dim;
        let mut gen =
            |n: usize, amp: f32| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-amp..amp)).collect() };
        let q = gen(num_q_heads * head_dim, 2.0);
        let k = gen(seq_len * kv_stride, 2.0);
        let v = gen(seq_len * kv_stride, 1.0);
        let (q_d, k_d, v_d) = (
            model.device.htod_sync_copy(&q).unwrap(),
            model.device.htod_sync_copy(&k).unwrap(),
            model.device.htod_sync_copy(&v).unwrap(),
        );
        let shape = || OnlineAttnShape {
            rows: 1,
            start_pos: seq_len - 1,
            num_q_heads,
            group_size,
            qk_dim: head_dim,
            v_dim: head_dim,
            k_pos_stride: kv_stride,
            k_head_stride: head_dim,
            v_pos_stride: kv_stride,
            v_head_stride: head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            window,
        };
        let index = attn_decode_kernel_index(head_dim, group_size).expect("a kernel");
        let (k_v, v_v) = (k_d.slice(..), v_d.slice(..));
        let gqa = model
            .attention_decode_gqa(&q_d, &k_v, &v_v, &shape(), index)
            .expect("decode launch");
        let online = model
            .attention_online(&q_d, &k_v, &v_v, shape())
            .expect("online launch");
        let (a, b) = (
            model.device.dtoh_sync_copy(&gqa).unwrap(),
            model.device.dtoh_sync_copy(&online).unwrap(),
        );
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let label = format!(
            "heads={num_q_heads}/{num_kv_heads} d={head_dim} seq={seq_len} window={window}"
        );
        assert!(
            a.iter().all(|x| x.is_finite()),
            "{label}: non-finite output"
        );
        assert!(max_diff <= 2e-5, "{label}: max |gqa - online| = {max_diff}");
        eprintln!("{label}: max diff {max_diff:e}");
    }
}

/// 32 greedy tokens after a long prompt, decode attention through the GQA kernels
/// vs the online kernel, with f32 weights (so only attention's summation order
/// differs): identical ids. A one-token generate must not load the module.
#[test]
#[ignore]
fn attention_decode_gqa_generation_matches_online() {
    let model = load_test_model(WeightsDtype::F32);
    let readme = std::fs::read_to_string("README.md").expect("read README.md");
    let prompt: String = readme.chars().take(3000).collect();
    model
        .generate(
            &prompt,
            1,
            None,
            &SamplingParams::default(),
            |_| {},
            |_, _| {},
        )
        .expect("one-token generate");
    assert!(
        !model.attn_decode.loaded(),
        "a one-token generate loaded the decode module"
    );
    let run = |on: bool| {
        model.attn_decode.set_enabled(on);
        let start = std::time::Instant::now();
        let (ids, _) = model
            .generate(
                &prompt,
                32,
                None,
                &SamplingParams::default(),
                |_| {},
                |_, _| {},
            )
            .expect("generate");
        (ids, start.elapsed().as_secs_f64() * 1000.0)
    };
    let (online_ids, online_ms) = run(false);
    let (gqa_ids, gqa_ms) = run(true);
    assert!(model.attn_decode.loaded());
    eprintln!("online {online_ms:.1} ms, gqa {gqa_ms:.1} ms (incl. prefill), ids {gqa_ids:?}");
    assert_eq!(
        gqa_ids.len(),
        32,
        "generation stopped early; pick a prompt without EOS"
    );
    assert_eq!(gqa_ids, online_ids);
}
