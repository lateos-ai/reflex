//! The batched-prefill Gated DeltaNet kernels (`gdn_*_rows_kernel` in
//! `kernels_cuda/gated_deltanet.cu`) against looping their single-token
//! kernels over the same rows. They must match **bit for bit**: each `_rows`
//! kernel does the single-token kernel's per-row arithmetic in the same order.
//! The kernels are loaded straight from the embedded module, so no GGUF is
//! needed. GPU tests are `#[ignore]`d like every other; run with
//! `cargo test --release -- --ignored gdn_rows_`.

use super::*;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const NAMES: [&str; 8] = [
    "gdn_conv_kernel",
    "gdn_l2_norm_kernel",
    "gdn_gates_kernel",
    "gdn_delta_kernel",
    "gdn_conv_rows_kernel",
    "gdn_l2_norm_rows_kernel",
    "gdn_gates_rows_kernel",
    "gdn_delta_rows_kernel",
];

struct Kernels {
    conv: AotKernel,
    l2: AotKernel,
    gates: AotKernel,
    delta: AotKernel,
    conv_rows: AotKernel,
    l2_rows: AotKernel,
    gates_rows: AotKernel,
    delta_rows: AotKernel,
}

fn load(device: &Arc<CudaDevice>) -> Kernels {
    let mut k = aot::load_kernel_module(
        device,
        include_bytes!(env!("REFLEX_KERNEL_GATED_DELTANET")),
        "gdn_rows_tests",
        &NAMES,
    )
    .expect("load gated_deltanet module")
    .into_iter();
    let mut next = || k.next().expect("kernel");
    Kernels {
        conv: next(),
        l2: next(),
        gates: next(),
        delta: next(),
        conv_rows: next(),
        l2_rows: next(),
        gates_rows: next(),
        delta_rows: next(),
    }
}

/// (head_dim, k heads, v heads). 128 is every real Qwen3.5 release and the
/// register path's limit; 32 leaves most of the unrolled loop guarded off;
/// 160 takes the global-memory fallback.
const SHAPES: [(usize, usize, usize); 3] = [(128, 2, 4), (32, 1, 2), (160, 1, 2)];
const ROWS: [usize; 3] = [1, 7, 149];
const EPS: f32 = 1e-6;

fn random(rng: &mut rand::rngs::StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn up(device: &Arc<CudaDevice>, v: &[f32]) -> CudaSlice<f32> {
    device.htod_sync_copy(v).expect("htod")
}

fn down(device: &Arc<CudaDevice>, v: &CudaSlice<f32>) -> Vec<f32> {
    device.dtoh_sync_copy(v).expect("dtoh")
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {i}: rows {g} vs loop {w}"
        );
    }
}

fn linear(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn norm_block(hd: usize) -> u32 {
    (hd as u32).next_power_of_two().clamp(32, 1024)
}

/// Conv (k = 4) and the gates over `rows` rows, against the single-token
/// kernels launched once per row.
#[test]
#[ignore]
fn gdn_rows_conv_and_gates_match_single_token_loop() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let k = load(&device);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d4e);
    let taps = 4usize;
    for &(hd, hk, hv) in &SHAPES {
        let conv_dim = 2 * hk * hd + hv * hd;
        for &rows in &ROWS {
            let qkv = random(&mut rng, rows * conv_dim, -2.0, 2.0);
            let w = random(&mut rng, conv_dim * taps, -0.6, 0.6);
            let state0 = random(&mut rng, (taps - 1) * conv_dim, -1.0, 1.0);
            let dev_w = up(&device, &w);

            let dev_qkv = up(&device, &qkv);
            let mut state_rows = up(&device, &state0);
            let mut out_rows = device.alloc_zeros::<f32>(rows * conv_dim).unwrap();
            unsafe {
                k.conv_rows.function.clone().launch(
                    linear(conv_dim, 256),
                    (
                        &dev_qkv,
                        &dev_w,
                        &mut state_rows,
                        &mut out_rows,
                        conv_dim as u32,
                        taps as u32,
                        rows as u32,
                    ),
                )
            }
            .unwrap();

            let mut state_loop = up(&device, &state0);
            let mut out_loop = Vec::with_capacity(rows * conv_dim);
            for t in 0..rows {
                let row = up(&device, &qkv[t * conv_dim..(t + 1) * conv_dim]);
                let mut out = device.alloc_zeros::<f32>(conv_dim).unwrap();
                unsafe {
                    k.conv.function.clone().launch(
                        linear(conv_dim, 256),
                        (
                            &row,
                            &dev_w,
                            &mut state_loop,
                            &mut out,
                            conv_dim as u32,
                            taps as u32,
                        ),
                    )
                }
                .unwrap();
                out_loop.extend(down(&device, &out));
            }
            let what = format!("conv hd={hd} rows={rows}");
            assert_bits(&down(&device, &out_rows), &out_loop, &format!("{what} out"));
            assert_bits(
                &down(&device, &state_rows),
                &down(&device, &state_loop),
                &format!("{what} state"),
            );

            // Gates, with some alpha past softplus's identity threshold of 20.
            let alpha = random(&mut rng, rows * hv, -8.0, 24.0);
            let beta_raw = random(&mut rng, rows * hv, -4.0, 4.0);
            let dt = random(&mut rng, hv, -1.0, 1.0);
            let a: Vec<f32> = random(&mut rng, hv, 0.0, 2.0)
                .iter()
                .map(|v| -v.exp())
                .collect();
            let (dev_dt, dev_a) = (up(&device, &dt), up(&device, &a));
            let (dev_alpha, dev_beta_raw) = (up(&device, &alpha), up(&device, &beta_raw));
            let mut decay_rows = device.alloc_zeros::<f32>(rows * hv).unwrap();
            let mut beta_rows = device.alloc_zeros::<f32>(rows * hv).unwrap();
            unsafe {
                k.gates_rows.function.clone().launch(
                    linear(rows * hv, 256),
                    (
                        &dev_alpha,
                        &dev_beta_raw,
                        &dev_dt,
                        &dev_a,
                        &mut decay_rows,
                        &mut beta_rows,
                        hv as u32,
                        rows as u32,
                    ),
                )
            }
            .unwrap();
            let (mut decay_loop, mut beta_loop) = (Vec::new(), Vec::new());
            for t in 0..rows {
                let al = up(&device, &alpha[t * hv..(t + 1) * hv]);
                let br = up(&device, &beta_raw[t * hv..(t + 1) * hv]);
                let mut dc = device.alloc_zeros::<f32>(hv).unwrap();
                let mut bt = device.alloc_zeros::<f32>(hv).unwrap();
                unsafe {
                    k.gates.function.clone().launch(
                        linear(hv, 256),
                        (&al, &br, &dev_dt, &dev_a, &mut dc, &mut bt, hv as u32),
                    )
                }
                .unwrap();
                decay_loop.extend(down(&device, &dc));
                beta_loop.extend(down(&device, &bt));
            }
            let what = format!("gates hd={hd} rows={rows}");
            assert_bits(
                &down(&device, &decay_rows),
                &decay_loop,
                &format!("{what} decay"),
            );
            assert_bits(
                &down(&device, &beta_rows),
                &beta_loop,
                &format!("{what} beta"),
            );
        }
    }
}

/// The q/k L2 norm and the delta rule over `rows` rows, against the
/// single-token kernels launched once per row, on the same S.
#[test]
#[ignore]
fn gdn_rows_l2_norm_and_delta_match_single_token_loop() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let k = load(&device);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x9a31);
    for &(hd, hk, hv) in &SHAPES {
        let key_dim = hk * hd;
        let conv_dim = 2 * key_dim + hv * hd;
        let q_scale = 1.0 / (hd as f32).sqrt();
        for &rows in &ROWS {
            let qkv = random(&mut rng, rows * conv_dim, -2.0, 2.0);

            // L2 norm: one rows launch vs two single-token launches per row
            // (q heads scaled, then k heads unscaled), as forward_gdn_mixer does.
            let mut x_rows = up(&device, &qkv);
            let cfg = LaunchConfig {
                grid_dim: ((2 * hk) as u32, rows as u32, 1),
                block_dim: (norm_block(hd), 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                k.l2_rows.function.clone().launch(
                    cfg,
                    (
                        &mut x_rows,
                        conv_dim as u32,
                        hk as u32,
                        hd as u32,
                        EPS,
                        q_scale,
                        1.0f32,
                    ),
                )
            }
            .unwrap();
            let mut x_loop = up(&device, &qkv);
            let head_cfg = LaunchConfig {
                grid_dim: (hk as u32, 1, 1),
                block_dim: (norm_block(hd), 1, 1),
                shared_mem_bytes: 0,
            };
            for t in 0..rows {
                for (offset, scale) in [(t * conv_dim, q_scale), (t * conv_dim + key_dim, 1.0)] {
                    unsafe {
                        k.l2.function.clone().launch(
                            head_cfg,
                            (&mut x_loop, offset as u32, hd as u32, EPS, scale),
                        )
                    }
                    .unwrap();
                }
            }
            let normed = down(&device, &x_rows);
            assert_bits(
                &normed,
                &down(&device, &x_loop),
                &format!("l2 hd={hd} rows={rows}"),
            );

            // Delta rule on the normalized rows, from a nonzero S.
            let s0 = random(&mut rng, hv * hd * hd, -0.3, 0.3);
            let beta = random(&mut rng, rows * hv, 0.05, 0.95);
            let decay = random(&mut rng, rows * hv, 0.5, 1.0);
            let dev_qkv = up(&device, &normed);
            let (dev_beta, dev_decay) = (up(&device, &beta), up(&device, &decay));
            let mut s_rows = up(&device, &s0);
            let mut o_rows = device.alloc_zeros::<f32>(rows * hv * hd).unwrap();
            let delta_cfg = LaunchConfig {
                grid_dim: (hv as u32, 1, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                k.delta_rows.function.clone().launch(
                    delta_cfg,
                    (
                        &mut s_rows,
                        &dev_qkv,
                        conv_dim as u32,
                        key_dim as u32,
                        &dev_beta,
                        &dev_decay,
                        &mut o_rows,
                        hd as u32,
                        hk as u32,
                        hv as u32,
                        rows as u32,
                    ),
                )
            }
            .unwrap();

            let mut s_loop = up(&device, &s0);
            let mut o_loop = Vec::with_capacity(rows * hv * hd);
            for t in 0..rows {
                let b = up(&device, &beta[t * hv..(t + 1) * hv]);
                let dc = up(&device, &decay[t * hv..(t + 1) * hv]);
                let mut o = device.alloc_zeros::<f32>(hv * hd).unwrap();
                let base = t * conv_dim;
                unsafe {
                    k.delta.function.clone().launch(
                        delta_cfg,
                        (
                            &mut s_loop,
                            &dev_qkv,
                            base as u32,
                            (base + key_dim) as u32,
                            (base + 2 * key_dim) as u32,
                            &b,
                            &dc,
                            &mut o,
                            hd as u32,
                            hk as u32,
                            hv as u32,
                        ),
                    )
                }
                .unwrap();
                o_loop.extend(down(&device, &o));
            }
            let what = format!("delta hd={hd} rows={rows}");
            assert_bits(&down(&device, &o_rows), &o_loop, &format!("{what} o"));
            assert_bits(
                &down(&device, &s_rows),
                &down(&device, &s_loop),
                &format!("{what} S"),
            );
        }
    }
}
