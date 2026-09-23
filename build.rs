// Ahead-of-time CUDA kernel compilation. This is the project's core technical bet:
// every kernel is compiled by `nvcc` at BUILD time (like llama.cpp), never by NVRTC
// at process start -- avoiding a real multi-second JIT tax paid on every cold start.
//
// Default output is PTX (portable across compute capabilities, small driver-side JIT
// cost at load time). Set REFLEX_CUDA_ARCH=sm_XX to compile straight to a cubin for
// that exact architecture instead, which the driver loads with no JIT at all -- the
// true zero-runtime-compilation path, at the cost of needing a matching cubin per
// deployment target.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        let candidate = Path::new(&cuda_path).join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let which = if cfg!(windows) { "where" } else { "which" };
    Command::new(which)
        .arg("nvcc")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(PathBuf::from))
}

fn main() {
    let src_dir = Path::new("src/kernels_cuda");
    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-env-changed=REFLEX_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=REFLEX_SKIP_CUDA");

    let skip_cuda = env::var("REFLEX_SKIP_CUDA").is_ok();
    if skip_cuda {
        println!("cargo:warning=REFLEX_SKIP_CUDA set, skipping AOT kernel compilation (dev-machine-without-CUDA path)");
    }

    let nvcc = if skip_cuda {
        None
    } else {
        match find_nvcc() {
            Some(p) => Some(p),
            None => {
                panic!(
                    "nvcc not found (checked CUDA_PATH/CUDA_HOME and PATH). Install the CUDA toolkit, \
                     or set REFLEX_SKIP_CUDA=1 to build without GPU kernels (dev-only, no inference)."
                );
            }
        }
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // `.filter(|s| !s.is_empty())`: Docker's `ARG REFLEX_CUDA_ARCH=""` exposes this
    // as a set-but-empty env var to `RUN` even when no `--build-arg` override is passed
    // (unlike a bare host shell, where it's truly unset) -- without the filter this
    // reads as `Some("")`, taking the cubin branch below with an empty `-arch=` and
    // making nvcc fatal on every default (portable-PTX) `docker build`.
    let arch = env::var("REFLEX_CUDA_ARCH").ok().filter(|s| !s.is_empty());

    // `src/aot.rs` embeds every kernel's bytes at compile time
    // (`include_bytes!(env!("REFLEX_KERNEL_<NAME>"))` at each call site) and
    // needs to know, once, crate-wide, which of build.rs's two output modes
    // produced those bytes -- both modes always agree for a single build (the
    // `-cubin`/`-ptx` flag below is chosen once, not per file).
    println!("cargo:rustc-env=REFLEX_KERNEL_FORMAT={}", if arch.is_some() { "cubin" } else { "ptx" });
    // Threaded forward the same way, so `src/diagnostics.rs` can compare the arch a
    // cubin was actually compiled for against the running GPU's real compute
    // capability at startup, instead of letting a mismatch surface as an opaque
    // driver load/launch error. Empty string in the default (portable PTX) build,
    // where no such mismatch is possible.
    println!("cargo:rustc-env=REFLEX_CUDA_ARCH={}", arch.as_deref().unwrap_or(""));

    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => {
            println!("cargo:warning=no {} directory yet, nothing to compile", src_dir.display());
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();

        let (out_ext, mode_flag) = match &arch {
            Some(_) => ("cubin", "-cubin"),
            None => ("ptx", "-ptx"),
        };
        let out_file = out_dir.join(format!("{stem}.{out_ext}"));

        // In REFLEX_SKIP_CUDA mode, still expose the REFLEX_KERNEL_<NAME>
        // env var every `include_bytes!(env!(...))` call needs to compile, but
        // skip the nvcc invocation itself and write an empty placeholder file in
        // its place instead -- `include_bytes!` (unlike the old design's runtime
        // `Ptx::from_file`) needs *some* file to exist at compile time, but its
        // contents are never a real kernel in this mode, so no inference binary
        // can actually load/launch kernels here (type-check only, per CLAUDE.md's
        // "REFLEX_SKIP_CUDA=1 cargo build" doc).
        if let Some(nvcc) = &nvcc {
            let mut cmd = Command::new(nvcc);
            cmd.arg(mode_flag).arg(&path).arg("-o").arg(&out_file);
            if let Some(a) = &arch {
                cmd.arg(format!("-arch={a}"));
            }

            let status = cmd.status().unwrap_or_else(|e| panic!("failed to invoke nvcc at {}: {e}", nvcc.display()));
            if !status.success() {
                panic!("nvcc failed compiling {}", path.display());
            }
        } else {
            std::fs::write(&out_file, []).unwrap_or_else(|e| panic!("failed to write placeholder kernel file {}: {e}", out_file.display()));
        }
        println!("cargo:rustc-env=REFLEX_KERNEL_{}={}", stem.to_uppercase(), out_file.display());
    }
}
