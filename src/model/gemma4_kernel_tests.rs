//! Tests for the Gemma 4 kernels (`kernels_cuda/gemma4.cu`). The host
//! references are checked on the CPU; the kernels are checked against them on
//! the GPU (`#[ignore]`d like every other GPU test; run with
//! `cargo test --release -- --ignored gemma4_`).

use super::gemma4::*;
use super::*;
use rand::{Rng, SeedableRng};

fn rng() -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(0x9e44)
}

/// Gemma 4's real `rope_freqs.weight` (read from the 12B GGUF): 1.0 for the
/// first 64 pairs, 1e30 for the other 192.
fn gemma4_global_freq_factors() -> Vec<f32> {
    (0..256).map(|i| if i < 64 { 1.0 } else { 1e30 }).collect()
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

/// With all-ones factors and `n_dims == head_dim`, the reference is the
/// existing NEOX RoPE (`rope_kernel`'s `position / base^(2i/head_dim)`),
/// written the way ggml writes it.
#[test]
fn gemma4_rope_ref_with_unit_factors_is_plain_neox_rope() {
    let (head_dim, num_heads, base) = (8, 2, 10_000.0f32);
    let mut rng = rng();
    let orig: Vec<f32> = (0..num_heads * head_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    for position in [0usize, 1, 7, 1000] {
        let mut got = orig.clone();
        rope_neox_ff_ref(
            &mut got, &[1.0; 4], position, 1, num_heads, head_dim, head_dim, base,
        );
        let mut want = orig.clone();
        let half = head_dim / 2;
        for h in 0..num_heads {
            for i in 0..half {
                let theta = position as f32 / base.powf(2.0 * i as f32 / head_dim as f32);
                let (x1, x2) = (orig[h * head_dim + i], orig[h * head_dim + i + half]);
                want[h * head_dim + i] = x1 * theta.cos() - x2 * theta.sin();
                want[h * head_dim + i + half] = x1 * theta.sin() + x2 * theta.cos();
            }
        }
        assert_close(&got, &want, 1e-4, &format!("position {position}"));
    }
}

/// Gemma 4's global layers: only the first 64 of 256 pairs rotate; the 1e30
/// factors leave the rest unchanged.
#[test]
fn gemma4_rope_ref_global_factors_rotate_only_the_first_64_pairs() {
    let (head_dim, n_dims) = (512, 512);
    let mut rng = rng();
    let orig: Vec<f32> = (0..head_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut t = orig.clone();
    rope_neox_ff_ref(
        &mut t,
        &gemma4_global_freq_factors(),
        5000,
        1,
        1,
        head_dim,
        n_dims,
        1_000_000.0,
    );
    let half = n_dims / 2;
    for i in 64..half {
        assert!((t[i] - orig[i]).abs() < 1e-6, "pair {i} moved (x1)");
        assert!(
            (t[i + half] - orig[i + half]).abs() < 1e-6,
            "pair {i} moved (x2)"
        );
    }
    let rotated = (1..64).filter(|&i| (t[i] - orig[i]).abs() > 1e-3).count();
    assert!(
        rotated > 50,
        "expected most of pairs 1..64 to rotate, {rotated} did"
    );
}

#[test]
fn gemma4_gelu_and_softcap_refs_match_known_values() {
    assert_eq!(gelu_tanh_ref(0.0), 0.0);
    assert!((gelu_tanh_ref(1.0) - 0.841_192).abs() < 1e-5);
    assert!((gelu_tanh_ref(-1.0) + 0.158_808).abs() < 1e-5);
    assert!((gelu_tanh_ref(10.0) - 10.0).abs() < 1e-5);
    assert!(gelu_tanh_ref(-10.0).abs() < 1e-5);

    assert_eq!(softcap_ref(0.0, 30.0), 0.0);
    assert!((softcap_ref(0.3, 30.0) - 0.3).abs() < 1e-4);
    assert!(softcap_ref(1e6, 30.0) <= 30.0);
    assert!(softcap_ref(-1e6, 30.0) >= -30.0);
    assert!((softcap_ref(30.0, 30.0) - 30.0 * 1f32.tanh()).abs() < 1e-4);
}

type RopeCase = (&'static str, usize, usize, usize, Vec<f32>, f32);

fn device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("failed to init CUDA device 0")
}

/// `rope_neox_ff_kernel` against the host reference: a 3-row batch at a
/// late start position and a single decode row, for a sliding-layer shape
/// (unit factors, head_dim 256) and a global-layer shape (Gemma's real
/// factors, head_dim 512), plus a partial `n_dims` that must leave the tail
/// of each head untouched.
#[test]
#[ignore]
fn gemma4_rope_kernel_matches_host_reference() {
    let device = device();
    let kernels = Gemma4Kernels::load(&device).expect("load gemma4 kernels");
    let mut rng = rng();
    // (name, num_heads, head_dim, n_dims, freq_factors, base)
    let cases: [RopeCase; 3] = [
        ("sliding", 8, 256, 256, vec![1.0; 128], 10_000.0),
        (
            "global",
            4,
            512,
            512,
            gemma4_global_freq_factors(),
            1_000_000.0,
        ),
        ("partial", 2, 64, 32, vec![1.0; 16], 10_000.0),
    ];
    for (name, num_heads, head_dim, n_dims, factors, base) in cases {
        for (start_pos, rows) in [(0usize, 3usize), (60_000, 3), (12_345, 1)] {
            let orig: Vec<f32> = (0..rows * num_heads * head_dim)
                .map(|_| rng.gen_range(-2.0..2.0))
                .collect();
            let mut want = orig.clone();
            rope_neox_ff_ref(
                &mut want, &factors, start_pos, rows, num_heads, head_dim, n_dims, base,
            );
            let mut t = device.htod_sync_copy(&orig).expect("htod");
            let ff = device.htod_sync_copy(&factors).expect("htod");
            kernels
                .rope_ff(
                    &mut t, &ff, start_pos, rows, num_heads, head_dim, n_dims, base,
                )
                .expect("rope_ff");
            let got = device.dtoh_sync_copy(&t).expect("dtoh");
            // Large positions make theta large; device and host argument
            // reduction can then differ by a few ulps of theta.
            assert_close(
                &got,
                &want,
                2e-3,
                &format!("{name} start_pos={start_pos} rows={rows}"),
            );
            if n_dims < head_dim {
                for r in 0..rows * num_heads {
                    let tail = r * head_dim + n_dims..(r + 1) * head_dim;
                    assert_eq!(&got[tail.clone()], &orig[tail], "{name}: tail changed");
                }
            }
        }
    }
}

/// `gelu_tanh_and_mul_kernel`, `softcap_kernel` and `scale_kernel` against
/// the host references, on a size that needs the grid-stride loop
/// (more than 65,535 blocks' worth would; this covers a full vocab row).
#[test]
#[ignore]
fn gemma4_elementwise_kernels_match_host_reference() {
    let device = device();
    let kernels = Gemma4Kernels::load(&device).expect("load gemma4 kernels");
    let mut rng = rng();
    let n = 262_144;
    let gate: Vec<f32> = (0..n).map(|_| rng.gen_range(-8.0..8.0)).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.gen_range(-4.0..4.0)).collect();

    let d_gate = device.htod_sync_copy(&gate).expect("htod");
    let d_up = device.htod_sync_copy(&up).expect("htod");
    let out = kernels
        .gelu_tanh_and_mul(&device, &d_gate, &d_up)
        .expect("geglu");
    let got = device.dtoh_sync_copy(&out).expect("dtoh");
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| gelu_tanh_ref(g) * u)
        .collect();
    assert_close(&got, &want, 1e-5, "gelu_tanh_and_mul");

    let logits: Vec<f32> = (0..n).map(|_| rng.gen_range(-200.0..200.0)).collect();
    let mut d = device.htod_sync_copy(&logits).expect("htod");
    kernels.softcap(&mut d, 30.0).expect("softcap");
    let got = device.dtoh_sync_copy(&d).expect("dtoh");
    let want: Vec<f32> = logits.iter().map(|&x| softcap_ref(x, 30.0)).collect();
    assert_close(&got, &want, 1e-5, "softcap");
    assert!(
        got.iter().all(|x| x.abs() <= 30.0),
        "softcap exceeded the cap"
    );

    let mut d = device.htod_sync_copy(&logits).expect("htod");
    kernels.scale(&mut d, 0.375).expect("scale");
    let got = device.dtoh_sync_copy(&d).expect("dtoh");
    let want: Vec<f32> = logits.iter().map(|&x| x * 0.375).collect();
    assert_eq!(got, want, "scale must be exact");
}
