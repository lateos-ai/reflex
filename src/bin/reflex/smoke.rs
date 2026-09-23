//! First real thing to run on a fresh GPU instance. Proves the AOT compilation
//! pipeline works at all (build.rs -> nvcc -> PTX -> cuModuleLoad, zero NVRTC) and
//! reports wall-clock time from process start to first kernel launch completing --
//! the project's actual target metric, measured on the simplest possible kernel
//! before any model-architecture work begins.
//!
//! Only builds/runs where a CUDA toolchain + GPU are present (build.rs requires nvcc
//! unless REFLEX_SKIP_CUDA=1 is set, in which case this subcommand has nothing to do).
//!
//! Usage: `reflex smoke`
//!
//! Exits via `reflex_engine::fast_exit` after printing the result instead
//! of returning from `run` normally -- see that function's doc comment for
//! why a graceful return costs several extra seconds of CUDA-context-
//! teardown wall-clock time on GPU-virtualized rented instances.

use reflex_engine::{aot, diagnostics};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::time::Instant;

pub fn run(_args: Vec<String>) {
    let t0 = Instant::now();

    // Uses the same underlying `CudaDevice::new` call as before on the success path
    // (no added cost to the timed metric below) -- only the error message improves.
    // The GPU diagnostic line itself is deliberately printed *after* `elapsed` is
    // captured, since `diagnostics::probe`'s own driver queries would otherwise
    // skew this subcommand's whole reason for existing: the smallest possible
    // process-start-to-first-kernel-result measurement.
    let device = diagnostics::init_device_with_diagnostics(0).unwrap_or_else(|e| panic!("{e}"));
    let kernel = aot::load_kernel(&device, include_bytes!(env!("REFLEX_KERNEL_SMOKE")), "smoke", "axpy_f32")
        .expect("failed to load AOT smoke kernel");

    let n = 1024usize;
    let x = device.htod_copy(vec![1.0f32; n]).unwrap();
    let y = device.htod_copy(vec![2.0f32; n]).unwrap();
    let mut out = device.alloc_zeros::<f32>(n).unwrap();

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        kernel
            .function
            .launch(cfg, (3.0f32, &x, &y, &mut out, n as i32))
    }
    .expect("kernel launch failed");
    device.synchronize().expect("sync failed");

    let elapsed = t0.elapsed();

    let result = device.dtoh_sync_copy(&out).unwrap();
    assert!((result[0] - 5.0).abs() < 1e-5, "wrong result: {}", result[0]);

    println!("REFLEX_SMOKE_OK process_start_to_first_result_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    if let Ok(diag) = diagnostics::probe(&device) {
        eprintln!("{diag}");
    }
    reflex_engine::fast_exit(0);
}
