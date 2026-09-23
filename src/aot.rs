//! Loads kernels compiled ahead-of-time by `build.rs` (see repo root `build.rs`).
//! No NVRTC call happens anywhere on this path -- that's the entire point of this
//! project vs. RustFeference/rft-gpu (see LESSONS_LEARNED_RUSTFEFERENCE.md there).
//!
//! Kernel bytes are embedded directly into this binary at compile time
//! (`include_bytes!(env!("REFLEX_KERNEL_<NAME>"))` at each call site in
//! `model.rs`/`reflex smoke`, passed in here as `kernel_bytes: &'static
//! [u8]`) rather than loaded from a filesystem path at runtime. The previous
//! design (`Ptx::from_file(kernel_path)`, `kernel_path` an absolute *build-time*
//! `OUT_DIR` path baked in via `env!(...)`) meant the compiled binary was not
//! actually self-contained -- it required that exact build machine's path to still
//! exist at runtime, which silently breaks under a naive multi-stage Docker build
//! (the runtime stage never has the builder stage's `target/.../out/*.ptx` files at
//! that same absolute path) and is generally fragile for any "copy the binary
//! elsewhere and run it" deployment. Embedding the bytes makes the binary genuinely
//! self-contained, which is also a more literal reading of this project's own "AOT,
//! no runtime dependency" thesis.
//!
//! `build.rs` emits one crate-wide `REFLEX_KERNEL_FORMAT` env var (`"ptx"` or
//! `"cubin"`, matching whichever single output mode that build produced for every
//! kernel) alongside each kernel's own `REFLEX_KERNEL_<NAME>` path var. PTX is
//! ASCII text, so it round-trips through `Ptx::from_src` directly from the embedded
//! bytes -- fully self-contained, no unsafe, no temp file. Cubin is arbitrary
//! binary, and the pinned `cudarc` version (0.11.9) has no public bytes-based `Ptx`
//! constructor (that lands in `cudarc` 0.19.x, not worth the pervasive-API-churn
//! risk of bumping a crate this codebase's numerics-verification credibility
//! depends on) -- so cubin mode writes the embedded bytes to a
//! content-hashed-filename temp file at process start, then calls the original
//! `Ptx::from_file` on that freshly-materialized path. Either way, every byte the
//! CUDA driver ultimately loads came from *this process's own binary*, never a
//! path that only existed on the machine that built it.

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub struct AotKernel {
    pub function: CudaFunction,
}

const KERNEL_FORMAT: &str = env!("REFLEX_KERNEL_FORMAT");

/// FNV-1a (64-bit) -- a small, dependency-free non-cryptographic hash, used only to
/// give each embedded cubin's materialized temp file a content-derived name so
/// different kernels (or different builds of the same kernel) never collide on the
/// same path. Not a security boundary; collision resistance at this scale is
/// incidental, not a design requirement.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Turns embedded kernel bytes into a `cudarc::nvrtc::Ptx` the driver can load,
/// branching on the crate-wide `REFLEX_KERNEL_FORMAT` build.rs set. See this
/// module's doc comment for why the two formats need different handling.
fn ptx_from_embedded_bytes(kernel_bytes: &'static [u8], module_name: &str) -> Result<Ptx, String> {
    match KERNEL_FORMAT {
        "ptx" => {
            let src = std::str::from_utf8(kernel_bytes)
                .map_err(|e| format!("embedded PTX for module {module_name} is not valid UTF-8: {e}"))?;
            Ok(Ptx::from_src(src.to_string()))
        }
        "cubin" => {
            let hash = fnv1a_hash(kernel_bytes);
            let tmp_path = std::env::temp_dir().join(format!("reflex-engine-kernel-{module_name}-{hash:016x}.cubin"));
            // Always (re)write, never skip-if-exists: a stale file left over from a
            // different build with the same module name (but different content --
            // the hash is content-derived, so this is only a concern if two builds
            // somehow hashed to the same value, astronomically unlikely for FNV-1a
            // at kernel-file sizes) must never be silently reused.
            std::fs::write(&tmp_path, kernel_bytes)
                .map_err(|e| format!("failed to materialize embedded cubin for module {module_name} at {tmp_path:?}: {e}"))?;
            let path_str = tmp_path
                .to_str()
                .ok_or_else(|| format!("materialized cubin temp path for module {module_name} is not valid UTF-8: {tmp_path:?}"))?;
            Ok(Ptx::from_file(path_str))
        }
        other => Err(format!("unknown REFLEX_KERNEL_FORMAT {other:?} (build.rs should only ever emit \"ptx\" or \"cubin\")")),
    }
}

/// Loads a single kernel from bytes embedded at compile time (`kernel_bytes`, built
/// via `include_bytes!(env!("REFLEX_KERNEL_<NAME>"))` at the call site -- see
/// this module's doc comment for why that has to happen at the call site rather
/// than inside this function). `module_name` and `function_name` must match the
/// `extern "C" __global__` symbol name in the original `.cu` source (see
/// `src/kernels_cuda/smoke.cu`).
pub fn load_kernel(
    device: &Arc<CudaDevice>,
    kernel_bytes: &'static [u8],
    module_name: &'static str,
    function_name: &'static str,
) -> Result<AotKernel, String> {
    let ptx = ptx_from_embedded_bytes(kernel_bytes, module_name)?;
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
    kernel_bytes: &'static [u8],
    module_name: &'static str,
    function_names: &[&'static str],
) -> Result<Vec<AotKernel>, String> {
    let ptx = ptx_from_embedded_bytes(kernel_bytes, module_name)?;
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
