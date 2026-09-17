//! Loads kernels compiled ahead-of-time by `build.rs` (see repo root `build.rs`).
//! No NVRTC call happens anywhere on this path -- that's the entire point of this
//! project vs. RustFeference/rft-gpu (see LESSONS_LEARNED_RUSTFEFERENCE.md there).
//!
//! UNVERIFIED ON REAL HARDWARE YET: this loads PTX text via `Ptx::from_src`, which
//! still costs the driver a JIT-to-SASS step at load time (smaller than a full NVRTC
//! source compile, but nonzero -- see plan risk #1 in the pivot plan). Compiling
//! straight to a `cubin` for a known compute capability (COLDSTART_CUDA_ARCH in
//! build.rs) is the true zero-JIT path and needs to be measured against this one
//! before picking a default.

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct AotKernel {
    pub function: CudaFunction,
}

/// Loads a single kernel from a PTX file embedded at compile time via `include_str!`.
/// `module_name` and `function_name` must match the `extern "C" __global__` symbol
/// name in the original `.cu` source (see `src/kernels_cuda/smoke.cu`).
pub fn load_kernel(
    device: &Arc<CudaDevice>,
    ptx_src: &str,
    module_name: &'static str,
    function_name: &'static str,
) -> Result<AotKernel, String> {
    let ptx = Ptx::from_src(ptx_src);
    device
        .load_ptx(ptx, module_name, &[function_name])
        .map_err(|e| format!("failed to load AOT-compiled module {module_name}: {e}"))?;
    let function = device
        .get_func(module_name, function_name)
        .ok_or_else(|| format!("function {function_name} not found in module {module_name}"))?;
    Ok(AotKernel { function })
}
