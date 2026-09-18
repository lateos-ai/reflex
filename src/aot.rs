//! Loads kernels compiled ahead-of-time by `build.rs` (see repo root `build.rs`).
//! No NVRTC call happens anywhere on this path -- that's the entire point of this
//! project vs. RustFeference/rft-gpu (see LESSONS_LEARNED_RUSTFEFERENCE.md there).
//!
//! Loads by file path via `Ptx::from_file`, which maps to the driver's `cuModuleLoad`.
//! Per the CUDA driver API docs that call accepts a cubin, PTX, or fatbin file
//! transparently, so this same code path covers both of build.rs's output modes:
//! the default PTX build (driver JITs to SASS at load time) and the
//! COLDSTART_CUDA_ARCH=sm_XX cubin build (no JIT at all -- the true
//! zero-runtime-compilation path). Which one wins on real hardware is measured by
//! `src/bin/smoke_coldstart.rs`, not assumed (see plan risk #1).

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct AotKernel {
    pub function: CudaFunction,
}

/// Loads a single kernel from the PTX or cubin file produced by `build.rs`, given
/// its path. `module_name` and `function_name` must match the `extern "C" __global__`
/// symbol name in the original `.cu` source (see `src/kernels_cuda/smoke.cu`).
pub fn load_kernel(
    device: &Arc<CudaDevice>,
    kernel_path: &str,
    module_name: &'static str,
    function_name: &'static str,
) -> Result<AotKernel, String> {
    let ptx = Ptx::from_file(kernel_path);
    device
        .load_ptx(ptx, module_name, &[function_name])
        .map_err(|e| format!("failed to load AOT-compiled module {module_name}: {e}"))?;
    let function = device
        .get_func(module_name, function_name)
        .ok_or_else(|| format!("function {function_name} not found in module {module_name}"))?;
    Ok(AotKernel { function })
}

/// Like [`load_kernel`], but for a `.cu` file that defines several
/// `extern "C" __global__` kernels (e.g. `gated_deltanet.cu`) -- loads the
/// module once, then resolves every named function against it, in the same
/// order as `function_names`.
pub fn load_kernel_module(
    device: &Arc<CudaDevice>,
    kernel_path: &str,
    module_name: &'static str,
    function_names: &[&'static str],
) -> Result<Vec<AotKernel>, String> {
    let ptx = Ptx::from_file(kernel_path);
    device
        .load_ptx(ptx, module_name, function_names)
        .map_err(|e| format!("failed to load AOT-compiled module {module_name}: {e}"))?;
    function_names
        .iter()
        .map(|&function_name| {
            device
                .get_func(module_name, function_name)
                .map(|function| AotKernel { function })
                .ok_or_else(|| format!("function {function_name} not found in module {module_name}"))
        })
        .collect()
}
