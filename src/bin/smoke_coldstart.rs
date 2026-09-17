//! First real thing to run on a fresh GPU instance. Proves the AOT compilation
//! pipeline works at all (build.rs -> nvcc -> PTX -> cuModuleLoad, zero NVRTC) and
//! reports wall-clock time from process start to first kernel launch completing --
//! the project's actual target metric, measured on the simplest possible kernel
//! before any model-architecture work begins. Compare this number directly against
//! rft-gpu's own measured ~4.5s NVRTC warm-up tax (see LESSONS_LEARNED_RUSTFEFERENCE.md
//! in the RustFeference repo) and against llama.cpp's cold start on the same hardware.
//!
//! Only builds/runs where a CUDA toolchain + GPU are present (build.rs requires nvcc
//! unless COLDSTART_SKIP_CUDA=1 is set, in which case this binary has nothing to do).

use coldstart_infer::aot;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::time::Instant;

const SMOKE_PTX: &str = include_str!(env!("COLDSTART_KERNEL_SMOKE"));

fn main() {
    let t0 = Instant::now();

    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let kernel = aot::load_kernel(&device, SMOKE_PTX, "smoke", "axpy_f32")
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

    println!("COLDSTART_SMOKE_OK process_start_to_first_result_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}
