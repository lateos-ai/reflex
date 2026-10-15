//! Tests for the Mamba-2 kernels (`kernels_cuda/mamba2.cu`) against the host
//! reference `reference/mamba2_reference.rs` (included into the test build as
//! `super::mamba2_reference`, whose own unit tests check it against
//! hand-computed values). GPU tests are `#[ignore]`d like every other; run
//! with `cargo test --release -- --ignored mamba2_`.

use super::config::Mamba2Config;
use super::granite_hybrid::{Mamba2Dims, Mamba2Kernels};
use super::mamba2_reference as reference;
use super::*;
use rand::{Rng, SeedableRng};

/// Two groups of two heads, `d_state` 64 (two state values per lane), head
/// dim 16: small, but every index mapping the real files use is exercised.
const SHAPE: Mamba2Config = Mamba2Config {
    d_inner: 64,
    d_state: 64,
    n_group: 2,
    n_head: 4,
    d_conv: 4,
};
const EPS: f32 = 1e-5;

fn ref_shape() -> reference::Mamba2Shape {
    reference::Mamba2Shape {
        d_inner: SHAPE.d_inner,
        d_state: SHAPE.d_state,
        n_group: SHAPE.n_group,
        n_head: SHAPE.n_head,
        d_conv: SHAPE.d_conv,
        eps: EPS,
    }
}

fn random(rng: &mut rand::rngs::StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= tol * (1.0 + w.abs()),
            "{what}: element {i}: got {g}, want {w} (err {err})"
        );
    }
}

/// The reference's shape and the engine's config agree on every derived size.
#[test]
fn mamba2_reference_and_config_sizes_agree() {
    let r = ref_shape();
    assert_eq!(r.head_dim(), SHAPE.head_dim());
    assert_eq!(r.conv_dim(), SHAPE.conv_dim());
    assert_eq!(r.d_in_proj(), SHAPE.d_in_proj());
    assert_eq!(r.conv_state_len(), SHAPE.conv_state_len());
    assert_eq!(r.ssm_state_len(), SHAPE.ssm_state_len());
}

/// Conv, scan and gate+norm on the GPU against the reference, first over a
/// 7-row prefill in one launch each, then 3 more single-row decode steps
/// carrying the same states.
#[test]
#[ignore]
fn mamba2_kernels_match_host_reference() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let k = Mamba2Kernels::load(&device).expect("load mamba2 kernels");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x3a2b);
    let s = ref_shape();
    let (d_in, c_dim, d_inner, h) = (s.d_in_proj(), s.conv_dim(), s.d_inner, s.n_head);

    let conv_w = random(&mut rng, c_dim * s.d_conv, -0.6, 0.6);
    let conv_b = random(&mut rng, c_dim, -0.3, 0.3);
    // dt_bias as HF initializes it (small dt), and one head pushed past the
    // softplus identity threshold of 20.
    let mut dt_bias = random(&mut rng, h, -5.0, -2.0);
    dt_bias[3] = 25.0;
    let a: Vec<f32> = random(&mut rng, h, 0.0, 2.7)
        .iter()
        .map(|v| -v.exp())
        .collect();
    let d = random(&mut rng, h, 0.0, 2.0);
    let norm_w = random(&mut rng, d_inner, 0.5, 1.5);
    let up = |v: &[f32]| device.htod_sync_copy(v).expect("htod");
    let (g_conv_w, g_conv_b, g_dt_bias, g_a, g_d, g_norm) = (
        up(&conv_w),
        up(&conv_b),
        up(&dt_bias),
        up(&a),
        up(&d),
        up(&norm_w),
    );

    let mut conv_state = vec![0.0f32; s.conv_state_len()];
    let mut ssm_state = vec![0.0f32; s.ssm_state_len()];
    let mut g_conv_state = up(&conv_state);
    let mut g_ssm_state = up(&ssm_state);

    for (step, rows) in [7usize, 1, 1, 1].into_iter().enumerate() {
        let zxbcdt = random(&mut rng, rows * d_in, -2.0, 2.0);
        let dims = Mamba2Dims::new(&SHAPE, rows, EPS);
        let g_zx = up(&zxbcdt);
        let g_conv = k
            .conv(
                &device,
                &g_zx,
                &g_conv_w,
                &g_conv_b,
                &mut g_conv_state,
                dims,
            )
            .expect("conv");
        let g_y = k
            .scan(
                &device,
                &g_conv,
                &g_zx,
                &g_dt_bias,
                &g_a,
                &mut g_ssm_state,
                dims,
            )
            .expect("scan");
        let g_out = k
            .gate_norm(&device, &g_y, &g_conv, &g_zx, &g_d, &g_norm, dims)
            .expect("gate_norm");
        let got_conv = device.dtoh_sync_copy(&g_conv).expect("dtoh");
        let got_y = device.dtoh_sync_copy(&g_y).expect("dtoh");
        let got_out = device.dtoh_sync_copy(&g_out).expect("dtoh");

        for r in 0..rows {
            let row = &zxbcdt[r * d_in..(r + 1) * d_in];
            let xbc = &row[d_inner..d_inner + c_dim];
            let conv = reference::conv_step(xbc, &conv_w, &conv_b, &mut conv_state, s.d_conv);
            let gn = s.n_group * s.d_state;
            let x = &conv[..d_inner];
            let y = reference::scan_step(
                &s,
                x,
                &conv[d_inner..d_inner + gn],
                &conv[d_inner + gn..],
                &row[d_inner + c_dim..],
                &dt_bias,
                &a,
                &mut ssm_state,
            );
            let out = reference::skip_gate_norm(&s, &y, x, &d, &row[..d_inner], &norm_w);
            let what = |p: &str| format!("step {step} row {r}: {p}");
            assert_close(
                &got_conv[r * c_dim..(r + 1) * c_dim],
                &conv,
                1e-5,
                &what("conv"),
            );
            assert_close(
                &got_y[r * d_inner..(r + 1) * d_inner],
                &y,
                1e-4,
                &what("scan"),
            );
            assert_close(
                &got_out[r * d_inner..(r + 1) * d_inner],
                &out,
                1e-4,
                &what("gate_norm"),
            );
        }
        let got_cs = device.dtoh_sync_copy(&g_conv_state).expect("dtoh");
        let got_ss = device.dtoh_sync_copy(&g_ssm_state).expect("dtoh");
        assert_eq!(got_cs, conv_state, "step {step}: conv state (exact copies)");
        assert_close(
            &got_ss,
            &ssm_state,
            1e-4,
            &format!("step {step}: ssm state"),
        );
    }
}

/// `scaled_add_kernel` rounds `m * s` on its own before the add, so it is
/// bit-identical to the host's `h + m * s` (no fused multiply-add).
#[test]
#[ignore]
fn mamba2_scaled_add_matches_host_exactly() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let k = Mamba2Kernels::load(&device).expect("load mamba2 kernels");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5ca1);
    let h = random(&mut rng, 4099, -50.0, 50.0);
    let m = random(&mut rng, 4099, -50.0, 50.0);
    let s = 0.246f32;
    let mut g_h = device.htod_sync_copy(&h).expect("htod");
    let g_m = device.htod_sync_copy(&m).expect("htod");
    k.scaled_add(&mut g_h, &g_m, s).expect("scaled_add");
    let got = device.dtoh_sync_copy(&g_h).expect("dtoh");
    let want: Vec<f32> = h.iter().zip(&m).map(|(&a, &b)| a + b * s).collect();
    assert_eq!(got, want);
}
