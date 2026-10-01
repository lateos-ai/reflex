//! Runtime GPU diagnostics: a human-readable startup line (device name, compute
//! capability, VRAM) and a plain-English wrapper around the common `CudaDevice::new`
//! failure modes, replacing a raw CUDA driver error code with a message a user who
//! has never seen a `CUresult` can act on. Every binary's `CudaDevice::new(0)` call
//! site should go through [`init_device_with_diagnostics`] instead.

use crate::error::ReflexError;
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
pub fn probe(device: &Arc<CudaDevice>) -> Result<GpuDiagnostics, ReflexError> {
    let name = device
        .name()
        .map_err(|e| crate::gpu_err!(e, "querying GPU name: {e}"))?;
    let major = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| crate::gpu_err!(e, "querying compute capability major: {e}"))?;
    let minor = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| crate::gpu_err!(e, "querying compute capability minor: {e}"))?;
    let (vram_free_bytes, vram_total_bytes) =
        result::mem_get_info().map_err(|e| crate::gpu_err!(e, "querying VRAM: {e}"))?;
    Ok(GpuDiagnostics {
        name,
        compute_capability: (major, minor),
        vram_free_bytes,
        vram_total_bytes,
    })
}

/// Wraps `CudaDevice::new(ordinal)`, turning the common failure modes (no GPU found,
/// invalid device index, driver/toolkit mismatch, driver not initialized) into a
/// plain-English message instead of a raw `CUresult`. Falls back to the driver's own
/// `error_string()` for anything else, so no failure mode is silently swallowed.
pub fn init_device_with_diagnostics(ordinal: usize) -> Result<Arc<CudaDevice>, ReflexError> {
    CudaDevice::new(ordinal).map_err(|e| ReflexError::gpu(&e, explain_driver_error(ordinal, &e)))
}

/// The `REFLEX_CUDA_ARCH` build.rs was invoked with (e.g. `"sm_86"`), or empty in the
/// default portable-PTX build. See `build.rs`'s matching `cargo:rustc-env` line.
const COMPILED_ARCH: &str = env!("REFLEX_CUDA_ARCH");

/// The `REFLEX_CUDA_ARCHS` build.rs was invoked with (a comma-separated `sm_XX` list
/// in the multi-arch fatbin mode), or empty otherwise.
pub const COMPILED_ARCHS: &str = env!("REFLEX_CUDA_ARCHS");

/// Parses a `sm_XY`/`compute_XY` (or bare `XY`) arch string into `(major, minor)`,
/// matching CUDA's own convention (all digits but the last are major, the last digit
/// is minor -- e.g. `sm_86` -> `(8, 6)`, `sm_90a` -> `(9, 0)`, the trailing `a`
/// "family-specific" suffix stripped since it doesn't affect binary compatibility here).
fn parse_arch(arch: &str) -> Result<(i32, i32), ReflexError> {
    let digits: String = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))
        .unwrap_or(arch)
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() < 2 {
        return Err(crate::reflex_err!(
            Cuda,
            "couldn't parse compute capability out of REFLEX_CUDA_ARCH={arch:?}"
        ));
    }
    let (major, minor) = digits.split_at(digits.len() - 1);
    let major = major.parse::<i32>().map_err(|e| {
        crate::reflex_err!(Other, "parsing major compute capability from {arch:?}: {e}")
    })?;
    let minor = minor.parse::<i32>().map_err(|e| {
        crate::reflex_err!(Other, "parsing minor compute capability from {arch:?}: {e}")
    })?;
    Ok((major, minor))
}

/// Confirms the running GPU's real compute capability matches the one this binary's
/// CUDA kernels were AOT-compiled for -- only meaningful when `build.rs` was given
/// `REFLEX_CUDA_ARCH` (a single-arch cubin build); the default portable-PTX build is
/// JIT'd by the driver to whatever's present, so no mismatch is possible and this
/// short-circuits immediately. Called once from `Model::load`, before any kernel is
/// loaded, so a mismatch surfaces as this plain-English error instead of an opaque
/// CUDA driver load/launch failure.
pub fn check_kernel_compute_capability(device: &Arc<CudaDevice>) -> Result<(), ReflexError> {
    if COMPILED_ARCH.is_empty() {
        if COMPILED_ARCHS.is_empty() {
            // Portable PTX: the driver JITs it for whatever GPU is present.
            return Ok(());
        }
        // Fatbin: the embedded PTX only helps GPUs at least as new as its target, so a
        // GPU older than every listed arch can't load the kernels at all. Catch that
        // here rather than as CUDA_ERROR_NO_BINARY_FOR_GPU from the first module load.
        let diag = probe(device)?;
        let (major, minor) = diag.compute_capability;
        if let FatbinCoverage::Unsupported { oldest, ptx } =
            fatbin_coverage(&compiled_archs()?, diag.compute_capability)
        {
            return Err(crate::reflex_err!(Cuda,
                "this binary's fatbin kernels cover REFLEX_CUDA_ARCHS={COMPILED_ARCHS} (native images, the \
                 oldest sm_{}{}) plus PTX for compute_{}{}, but the detected GPU ({}) has compute capability \
                 {major}.{minor}, which none of them can run on -- rebuild with sm_{major}{minor} in \
                 REFLEX_CUDA_ARCHS, or omit it for a portable PTX build.",
                oldest.0, oldest.1, ptx.0, ptx.1, diag.name
            ));
        }
        return Ok(());
    }
    let (compiled_major, compiled_minor) = parse_arch(COMPILED_ARCH)?;
    let diag = probe(device)?;
    let (major, minor) = diag.compute_capability;
    if (major, minor) != (compiled_major, compiled_minor) {
        return Err(crate::reflex_err!(Cuda,
            "this binary's CUDA kernels were compiled for compute capability {compiled_major}.{compiled_minor} \
             (sm_{compiled_major}{compiled_minor}), but the detected GPU ({}) has compute capability {major}.{minor} \
             -- rebuild with REFLEX_CUDA_ARCH=sm_{major}{minor}, or omit REFLEX_CUDA_ARCH for a portable PTX build.",
            diag.name
        ));
    }
    Ok(())
}

/// How a fatbin build's kernels load on a GPU of a given compute capability. `build.rs`
/// embeds one plain `sm_XY` SASS image per `REFLEX_CUDA_ARCHS` entry plus PTX for the
/// highest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatbinCoverage {
    /// A SASS image loads with zero JIT: an exact match, or -- per CUDA's binary
    /// compatibility rule -- an image with the same major version and an equal or lower
    /// minor (an `sm_86` image runs on `sm_87`/`sm_89`). `image` is the one the driver
    /// picks: the highest compatible.
    Native { image: (i32, i32) },
    /// No compatible SASS image, but the GPU is at least as new as the embedded PTX's
    /// target, so the driver JIT-compiles that PTX at load time.
    PtxJit { ptx: (i32, i32) },
    /// No compatible SASS image and the GPU is older than the PTX target: the kernels
    /// cannot load on this GPU at all.
    Unsupported { oldest: (i32, i32), ptx: (i32, i32) },
}

/// Pure coverage decision behind [`fatbin_coverage_for_device`] and the fatbin arm of
/// [`check_kernel_compute_capability`]. `compiled` must be non-empty (see
/// [`compiled_archs`]).
pub fn fatbin_coverage(compiled: &[(i32, i32)], device: (i32, i32)) -> FatbinCoverage {
    let native = compiled
        .iter()
        .filter(|&&(major, minor)| major == device.0 && minor <= device.1)
        .max();
    if let Some(&image) = native {
        return FatbinCoverage::Native { image };
    }
    let ptx = compiled.iter().max().copied().unwrap_or((0, 0));
    if device >= ptx {
        FatbinCoverage::PtxJit { ptx }
    } else {
        FatbinCoverage::Unsupported {
            oldest: compiled.iter().min().copied().unwrap_or((0, 0)),
            ptx,
        }
    }
}

/// `REFLEX_CUDA_ARCHS` parsed into `(major, minor)` pairs. Errs if it's empty, i.e. this
/// isn't a fatbin build.
pub fn compiled_archs() -> Result<Vec<(i32, i32)>, ReflexError> {
    let archs = COMPILED_ARCHS
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(parse_arch)
        .collect::<Result<Vec<_>, _>>()?;
    if archs.is_empty() {
        return Err(ReflexError::Cuda(
            "not a fatbin build: REFLEX_CUDA_ARCHS was empty at build time".to_string(),
        ));
    }
    Ok(archs)
}

/// [`fatbin_coverage`] for the detected GPU. Only meaningful in fatbin mode -- a
/// portable-PTX build always JITs and a single-arch cubin build is already enforced as
/// an exact match by [`check_kernel_compute_capability`].
pub fn fatbin_coverage_for_device(device: &Arc<CudaDevice>) -> Result<FatbinCoverage, ReflexError> {
    let diag = probe(device)?;
    Ok(fatbin_coverage(&compiled_archs()?, diag.compute_capability))
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
    use super::{fatbin_coverage, parse_arch, FatbinCoverage};

    /// The default multi-arch list the README suggests.
    const DEFAULT_LIST: [(i32, i32); 5] = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0)];

    #[test]
    fn fatbin_exact_arch_is_native() {
        assert_eq!(
            fatbin_coverage(&DEFAULT_LIST, (7, 5)),
            FatbinCoverage::Native { image: (7, 5) }
        );
    }

    #[test]
    fn fatbin_same_major_lower_minor_image_is_native() {
        // sm_87 (Jetson Orin) isn't listed; the sm_86 image runs on it with no JIT.
        assert_eq!(
            fatbin_coverage(&DEFAULT_LIST, (8, 7)),
            FatbinCoverage::Native { image: (8, 6) }
        );
        // Ada (sm_89) on a list without sm_89 still gets the sm_86 image, not PTX.
        assert_eq!(
            fatbin_coverage(&[(8, 0), (8, 6)], (8, 9)),
            FatbinCoverage::Native { image: (8, 6) }
        );
    }

    #[test]
    fn fatbin_newer_gpu_jits_the_embedded_ptx() {
        assert_eq!(
            fatbin_coverage(&DEFAULT_LIST, (12, 0)),
            FatbinCoverage::PtxJit { ptx: (9, 0) }
        );
    }

    #[test]
    fn fatbin_older_gpu_is_unsupported_not_ptx_fallback() {
        // Measured on a real T4 (sm_75) with REFLEX_CUDA_ARCHS=sm_80,sm_86: the module
        // load fails with CUDA_ERROR_NO_BINARY_FOR_GPU; compute_86 PTX can't run on 7.5.
        assert_eq!(
            fatbin_coverage(&[(8, 0), (8, 6)], (7, 5)),
            FatbinCoverage::Unsupported {
                oldest: (8, 0),
                ptx: (8, 6)
            }
        );
        // Same major but only a *higher* minor listed: SASS isn't backward compatible.
        assert_eq!(
            fatbin_coverage(&[(8, 6), (9, 0)], (8, 0)),
            FatbinCoverage::Unsupported {
                oldest: (8, 6),
                ptx: (9, 0)
            }
        );
    }

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
