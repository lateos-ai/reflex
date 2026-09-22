// Ahead-of-time CUDA kernel compilation. This is the project's core technical bet:
// every kernel is compiled by `nvcc` at BUILD time (like llama.cpp), never by NVRTC
// at process start (like rft-gpu was). See LESSONS_LEARNED_RUSTFEFERENCE.md in the
// RustFeference repo for why that distinction is the whole point of this project.
//
// Default output is PTX (portable across compute capabilities, small driver-side JIT
// cost at load time). Set COLDSTART_CUDA_ARCH=sm_XX to compile straight to a cubin for
// that exact architecture instead, which the driver loads with no JIT at all -- the
// true zero-runtime-compilation path, at the cost of needing a matching cubin per
// deployment target. Which of these actually wins on real hardware is an open
// question this project needs to measure, not assume (see plan risk #1).

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
    println!("cargo:rerun-if-env-changed=COLDSTART_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=COLDSTART_SKIP_CUDA");

    let skip_cuda = env::var("COLDSTART_SKIP_CUDA").is_ok();
    if skip_cuda {
        println!("cargo:warning=COLDSTART_SKIP_CUDA set, skipping AOT kernel compilation (dev-machine-without-CUDA path)");
    }

    let nvcc = if skip_cuda {
        None
    } else {
        match find_nvcc() {
            Some(p) => Some(p),
            None => {
                panic!(
                    "nvcc not found (checked CUDA_PATH/CUDA_HOME and PATH). Install the CUDA toolkit, \
                     or set COLDSTART_SKIP_CUDA=1 to build without GPU kernels (dev-only, no inference)."
                );
            }
        }
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("COLDSTART_CUDA_ARCH").ok();

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

        // In COLDSTART_SKIP_CUDA mode, still expose the COLDSTART_KERNEL_<NAME>
        // env var every model.rs `env!(...)` call needs to compile, but skip the
        // nvcc invocation itself -- the referenced file is never created, so no
        // inference binary can actually load kernels in this mode (type-check
        // only, per CLAUDE.md's "COLDSTART_SKIP_CUDA=1 cargo build" doc).
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
        }
        println!("cargo:rustc-env=COLDSTART_KERNEL_{}={}", stem.to_uppercase(), out_file.display());
    }
}
