//! The tiled prefill attention kernels (`kernels_cuda/attention_tiled.cu`) against
//! `attention_online_kernel` on random Q/K/V, at Gemma 4's head layouts (12B and 31B,
//! sliding and global), across the sliding window's edge and for a prefill resumed at
//! a non-zero `start_pos`. `#[ignore]`d: it needs a GPU and the tiny Gemma 4 fixture
//! (only its kernels are used).

use super::*;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;
use rand::{Rng, SeedableRng};

const FIXTURE: &str = "test-data/tiny-gemma4.gguf";

struct Case {
    rows: usize,
    start_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scale: f32,
}

#[test]
#[ignore]
fn attention_tiled_matches_online_kernel() {
    let file = GgufFile::open(FIXTURE).unwrap_or_else(|e| panic!("open {FIXTURE}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");
    let kernels = &model.gemma4.as_ref().expect("Gemma 4 fixture").kernels;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x7a11);
    let cases = [
        // 12B sliding layer past the 1,024 window; rows not a multiple of the block.
        Case {
            rows: 1100,
            start_pos: 0,
            num_q_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            window: 1024,
            scale: 1.0,
        },
        // 12B global layer: one KV head shared by 16 query heads.
        Case {
            rows: 301,
            start_pos: 0,
            num_q_heads: 16,
            num_kv_heads: 1,
            head_dim: 512,
            window: 0,
            scale: 1.0,
        },
        // 31B global (group 8) and sliding (group 2) layers.
        Case {
            rows: 257,
            start_pos: 0,
            num_q_heads: 32,
            num_kv_heads: 4,
            head_dim: 512,
            window: 0,
            scale: 1.0,
        },
        Case {
            rows: 130,
            start_pos: 0,
            num_q_heads: 32,
            num_kv_heads: 16,
            head_dim: 256,
            window: 1024,
            scale: 1.0,
        },
        // Resumed prefill with a small window (most tiles partly masked), group 64 > the
        // 32 queries per block (heads split across blocks), and a 1/sqrt(d) scale.
        Case {
            rows: 129,
            start_pos: 37,
            num_q_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            window: 50,
            scale: 0.0625,
        },
        Case {
            rows: 70,
            start_pos: 5,
            num_q_heads: 64,
            num_kv_heads: 1,
            head_dim: 512,
            window: 0,
            scale: 0.044,
        },
        Case {
            rows: 2,
            start_pos: 0,
            num_q_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            window: 1024,
            scale: 1.0,
        },
    ];
    for c in cases {
        let kv_len = c.start_pos + c.rows;
        // Gemma 4's Q and K are RMS-normed per head (unit RMS); scale them so scores
        // stay in a realistic range at scale 1.0.
        let qk_amp = 1.0 / (c.head_dim as f32).sqrt().sqrt();
        let mut gen =
            |n: usize, amp: f32| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-amp..amp)).collect() };
        let q = gen(c.rows * c.num_q_heads * c.head_dim, qk_amp * 3.0);
        let k = gen(kv_len * c.num_kv_heads * c.head_dim, qk_amp * 3.0);
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
            scale: c.scale,
            window: c.window,
        };
        let (k_v, v_v) = (KvView::F32(k_d.slice(..)), KvView::F32(v_d.slice(..)));
        let tiled = kernels
            .attention_prefill_tiled(&model.device, &q_d, &k_v, &v_v, &shape())
            .expect("tiled launch")
            .expect("shape should be supported");
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
        assert!(
            t.iter().all(|x| x.is_finite()),
            "rows={} d={}: non-finite output",
            c.rows,
            c.head_dim
        );
        assert!(
            max_diff <= 2e-5,
            "rows={} start={} heads={}/{} d={} window={}: max |tiled - online| = {max_diff}",
            c.rows,
            c.start_pos,
            c.num_q_heads,
            c.num_kv_heads,
            c.head_dim,
            c.window
        );
        eprintln!(
            "rows={} start={} heads={}/{} d={} window={}: max diff {max_diff:e}",
            c.rows, c.start_pos, c.num_q_heads, c.num_kv_heads, c.head_dim, c.window
        );
    }
}
