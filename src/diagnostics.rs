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

/// The `REFLEX_CUDA_ARCH` build.rs was invoked with (e.g. `"sm_86"`), or empty in the
/// default portable-PTX build. See `build.rs`'s matching `cargo:rustc-env` line.
const COMPILED_ARCH: &str = env!("REFLEX_CUDA_ARCH");

/// Parses a `sm_XY`/`compute_XY` (or bare `XY`) arch string into `(major, minor)`,
/// matching CUDA's own convention (all digits but the last are major, the last digit
/// is minor -- e.g. `sm_86` -> `(8, 6)`, `sm_90a` -> `(9, 0)`, the trailing `a`
/// "family-specific" suffix stripped since it doesn't affect binary compatibility here).
fn parse_arch(arch: &str) -> Result<(i32, i32), String> {
    let digits: String = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))
        .unwrap_or(arch)
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() < 2 {
        return Err(format!("couldn't parse compute capability out of REFLEX_CUDA_ARCH={arch:?}"));
    }
    let (major, minor) = digits.split_at(digits.len() - 1);
    let major = major.parse::<i32>().map_err(|e| format!("parsing major compute capability from {arch:?}: {e}"))?;
    let minor = minor.parse::<i32>().map_err(|e| format!("parsing minor compute capability from {arch:?}: {e}"))?;
    Ok((major, minor))
}

/// Confirms the running GPU's real compute capability matches the one this binary's
/// CUDA kernels were AOT-compiled for -- only meaningful when `build.rs` was given
/// `REFLEX_CUDA_ARCH` (a single-arch cubin build); the default portable-PTX build is
/// JIT'd by the driver to whatever's present, so no mismatch is possible and this
/// short-circuits immediately. Called once from `Model::load`, before any kernel is
/// loaded, so a mismatch surfaces as this plain-English error instead of an opaque
/// CUDA driver load/launch failure.
pub fn check_kernel_compute_capability(device: &Arc<CudaDevice>) -> Result<(), String> {
    if COMPILED_ARCH.is_empty() {
        return Ok(());
    }
    let (compiled_major, compiled_minor) = parse_arch(COMPILED_ARCH)?;
    let diag = probe(device)?;
    let (major, minor) = diag.compute_capability;
    if (major, minor) != (compiled_major, compiled_minor) {
        return Err(format!(
            "this binary's CUDA kernels were compiled for compute capability {compiled_major}.{compiled_minor} \
             (sm_{compiled_major}{compiled_minor}), but the detected GPU ({}) has compute capability {major}.{minor} \
             -- rebuild with REFLEX_CUDA_ARCH=sm_{major}{minor}, or omit REFLEX_CUDA_ARCH for a portable PTX build.",
            diag.name
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::parse_arch;

    #[test]
    fn parse_arch_sm_prefix() {
        assert_eq!(parse_arch("sm_86"), Ok((8, 6)));
    }

    #[test]
    fn parse_arch_compute_prefix() {
        assert_eq!(parse_arch("compute_75"), Ok((7, 5)));
    }

    #[test]
    fn parse_arch_strips_family_specific_suffix() {
        assert_eq!(parse_arch("sm_90a"), Ok((9, 0)));
    }

    #[test]
    fn parse_arch_bare_digits() {
        assert_eq!(parse_arch("120"), Ok((12, 0)));
    }

    #[test]
    fn parse_arch_rejects_garbage() {
        assert!(parse_arch("sm_").is_err());
        assert!(parse_arch("nonsense").is_err());
    }
}
