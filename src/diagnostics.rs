//! Runtime GPU diagnostics: a human-readable startup line (device name, compute
//! capability, VRAM) and a plain-English wrapper around the common `CudaDevice::new`
//! failure modes, replacing a raw CUDA driver error code with a message a user who
//! has never seen a `CUresult` can act on. Every binary's `CudaDevice::new(0)` call
//! site should go through [`init_device_with_diagnostics`] instead.

use cudarc::driver::sys::{CUdevice_attribute, CUresult};
use cudarc::driver::{result, CudaDevice, DriverError};
use std::sync::Arc;

/// One GPU's identity/capacity, queried once at process start.
pub struct GpuDiagnostics {
    pub name: String,
    /// (major, minor), e.g. `(8, 6)` for an A6000 ("sm_86").
    pub compute_capability: (i32, i32),
    pub vram_free_bytes: usize,
    pub vram_total_bytes: usize,
}

impl std::fmt::Display for GpuDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU: {} (sm_{}{}), VRAM: {}/{} MiB free",
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.vram_free_bytes / (1024 * 1024),
            self.vram_total_bytes / (1024 * 1024),
        )
    }
}

/// Queries `device`'s name, compute capability, and current free/total VRAM via the
/// CUDA driver API (`CudaDevice::name`/`attribute`;
/// `cudarc::driver::result::mem_get_info` is a free function that reports the
/// calling thread's *current* CUDA context, which `CudaDevice::new` already made
/// current for `device` -- there is no per-device-handle overload in this cudarc
/// version).
pub fn probe(device: &Arc<CudaDevice>) -> Result<GpuDiagnostics, String> {
    let name = device.name().map_err(|e| format!("querying GPU name: {e}"))?;
    let major = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| format!("querying compute capability major: {e}"))?;
    let minor = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| format!("querying compute capability minor: {e}"))?;
    let (vram_free_bytes, vram_total_bytes) = result::mem_get_info().map_err(|e| format!("querying VRAM: {e}"))?;
    Ok(GpuDiagnostics { name, compute_capability: (major, minor), vram_free_bytes, vram_total_bytes })
}

/// Wraps `CudaDevice::new(ordinal)`, turning the common failure modes (no GPU found,
/// invalid device index, driver/toolkit mismatch, driver not initialized) into a
/// plain-English message instead of a raw `CUresult`. Falls back to the driver's own
/// `error_string()` for anything else, so no failure mode is silently swallowed.
pub fn init_device_with_diagnostics(ordinal: usize) -> Result<Arc<CudaDevice>, String> {
    CudaDevice::new(ordinal).map_err(|e| explain_driver_error(ordinal, &e))
}

fn explain_driver_error(ordinal: usize, e: &DriverError) -> String {
    match e.0 {
        CUresult::CUDA_ERROR_NO_DEVICE => {
            "no CUDA-capable GPU was found. Check that an NVIDIA GPU is installed and visible to this process \
             (in a container, this usually means `--gpus all`/`nvidia-container-toolkit` is missing)."
                .to_string()
        }
        CUresult::CUDA_ERROR_INVALID_DEVICE => {
            format!("device index {ordinal} does not exist. Check how many GPUs are actually present (e.g. `nvidia-smi -L`).")
        }
        CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => {
            "the installed NVIDIA driver is too old for this build's CUDA toolkit version. \
             Update the GPU driver, or rebuild against an older CUDA toolkit."
                .to_string()
        }
        CUresult::CUDA_ERROR_NOT_INITIALIZED => {
            "the CUDA driver failed to initialize. Check that the NVIDIA driver is actually loaded \
             (`nvidia-smi` should work) and that this process has permission to access the GPU device nodes."
                .to_string()
        }
        other => format!(
            "CUDA device init failed ({other:?}): {}",
            e.error_string().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|_| "<no driver error string available>".to_string())
        ),
    }
}
