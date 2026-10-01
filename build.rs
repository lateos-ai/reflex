// Ahead-of-time CUDA kernel compilation. This is the project's core technical bet:
// every kernel is compiled by `nvcc` at BUILD time (like llama.cpp), never by NVRTC
// at process start -- avoiding a real multi-second JIT tax paid on every cold start.
//
// Three output modes, chosen by env var (mutually exclusive):
//   1. Default (neither var set): portable PTX. Portable across compute
//      capabilities, small driver-side JIT cost at load time.
//   2. REFLEX_CUDA_ARCH=sm_XX (singular): compile straight to a *cubin* for that one
//      exact architecture -- the driver loads it with no JIT at all (true zero-runtime-
//      compilation), at the cost of a failing hard on any other GPU.
//   3. REFLEX_CUDA_ARCHS=sm_XX,sm_YY,... (plural): compile a *fatbin* -- a single
//      container with one native cubin (SASS) per listed architecture, plus an
//      embedded forward-compatible PTX (for the highest listed arch, `code=compute_XX`)
//      so a GPU newer than any listed arch still loads via driver JIT instead of
//      failing. This is the multi-deployment-target answer to mode 2's single-arch
//      limitation (e.g. a mixed sm_86/sm_89 serverless GPU pool).
//
// Setting both REFLEX_CUDA_ARCH and REFLEX_CUDA_ARCHS is an error: the "one cubin
// for one exact arch" and "fatbin with many archs + PTX fallback" intents contradict
// each other, and silently picking one would surprise the other half of the request.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        let candidate =
            Path::new(&cuda_path)
                .join("bin")
                .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
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

/// Extracts the numeric core of a `sm_XX`/`compute_XX`/bare-`XX` arch string for
/// `nvcc -gencode` (`sm_86` -> `86`, `compute_90` -> `90`). Drops any non-numeric
/// tail (so an accidental `sm_90a` would read as `90`, at the cost of ignoring the
/// family-specific `a` variant -- documented as unsupported for the fatbin mode,
/// which targets plain `sm_XX` forms). Returns `None` for a non-numeric string.
fn arch_numeric(arch: &str) -> Option<String> {
    let s = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))
        .unwrap_or(arch);
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        Some(digits)
    }
}

fn main() {
    let src_dir = Path::new("src/kernels_cuda");
    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-env-changed=REFLEX_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=REFLEX_CUDA_ARCHS");
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
    // making nvcc fatal on every default (portable-PTX) `docker build`. The same
    // filter applies to the plural `REFLEX_CUDA_ARCHS`.
    let arch = env::var("REFLEX_CUDA_ARCH").ok().filter(|s| !s.is_empty());
    let archs = env::var("REFLEX_CUDA_ARCHS").ok().filter(|s| !s.is_empty());

    // Mutually exclusive: a single-arch cubin and a multi-arch fatbin are different
    // intents, and silently preferring one over the other would half-satisfy the
    // request. Fail loudly instead (see this module's doc comment).
    if arch.is_some() && archs.is_some() {
        panic!(
            "REFLEX_CUDA_ARCH (single-arch cubin) and REFLEX_CUDA_ARCHS (multi-arch fatbin) \
             are mutually exclusive -- set exactly one, or neither for portable PTX."
        );
    }

    // The comma-separated arch list, normalized (trimmed, empties dropped), preserving
    // the user's order so the embedded PTX fallback can target the *last* (assumed
    // highest) arch deterministically.
    let arch_list: Vec<String> = archs
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    // `src/aot.rs` embeds every kernel's bytes at compile time
    // (`include_bytes!(env!("REFLEX_KERNEL_<NAME>"))` at each call site) and
    // needs to know, once, crate-wide, which of build.rs's three output modes
    // produced those bytes -- all modes always agree for a single build (the
    // `-cubin`/`-fatbin`/`-ptx` flag below is chosen once, not per file).
    let kernel_format = if arch.is_some() {
        "cubin"
    } else if !arch_list.is_empty() {
        "fatbin"
    } else {
        "ptx"
    };
    println!("cargo:rustc-env=REFLEX_KERNEL_FORMAT={kernel_format}");
    // Threaded forward the same way, so `src/diagnostics.rs` can compare the arch a
    // cubin was actually compiled for against the running GPU's real compute
    // capability at startup, instead of letting a mismatch surface as an opaque
    // driver load/launch error. `REFLEX_CUDA_ARCH` is empty in the portable-PTX and
    // fatbin modes (no single arch to pin); `REFLEX_CUDA_ARCHS` carries the
    // comma-joined list in fatbin mode and is empty otherwise.
    println!(
        "cargo:rustc-env=REFLEX_CUDA_ARCH={}",
        arch.as_deref().unwrap_or("")
    );
    println!("cargo:rustc-env=REFLEX_CUDA_ARCHS={}", arch_list.join(","));

    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => {
            println!(
                "cargo:warning=no {} directory yet, nothing to compile",
                src_dir.display()
            );
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();

        let (out_ext, mode_flag) = if arch.is_some() {
            ("cubin", "-cubin")
        } else if !arch_list.is_empty() {
            ("fatbin", "-fatbin")
        } else {
            ("ptx", "-ptx")
        };
        let out_file = out_dir.join(format!("{stem}.{out_ext}"));

        // In REFLEX_SKIP_CUDA mode, still expose the REFLEX_KERNEL_<NAME>
        // env var every `include_bytes!(env!(...))` call needs to compile, but
        // skip the nvcc invocation itself and write an empty placeholder file in
        // its place instead -- `include_bytes!` (unlike the old design's runtime
        // `Ptx::from_file`) needs *some* file to exist at compile time, but its
        // contents are never a real kernel in this mode, so no inference binary
        // can actually load/launch kernels here (type-check only, per docs/DEVELOPMENT.md's
        // "Working without CUDA" section).
        if let Some(nvcc) = &nvcc {
            let mut cmd = Command::new(nvcc);
            cmd.arg(mode_flag).arg(&path).arg("-o").arg(&out_file);
            match (&arch, arch_list.is_empty()) {
                // Single-arch cubin: one exact target, no PTX fallback.
                (Some(a), _) => {
                    cmd.arg(format!("-arch={a}"));
                }
                // Fatbin: one native SASS image per listed arch, plus an embedded
                // forward-compatible PTX (`code=compute_<highest>`, not `code=sm_<highest>`)
                // for the *last* (assumed highest) arch so a GPU newer than every listed
                // arch still loads via driver JIT rather than failing outright.
                (None, false) => {
                    for a in &arch_list {
                        let n = arch_numeric(a)
                            .unwrap_or_else(|| panic!("REFLEX_CUDA_ARCHS entry {a:?} is not a valid sm_XX/compute_XX arch"));
                        cmd.arg(format!("-gencode=arch=compute_{n},code=sm_{n}"));
                    }
                    let highest = arch_list.last().unwrap();
                    let highest_n = arch_numeric(highest).unwrap();
                    cmd.arg(format!(
                        "-gencode=arch=compute_{highest_n},code=compute_{highest_n}"
                    ));
                }
                // Portable PTX: no arch flag at all.
                (None, true) => {}
            }

            let status = cmd
                .status()
                .unwrap_or_else(|e| panic!("failed to invoke nvcc at {}: {e}", nvcc.display()));
            if !status.success() {
                panic!("nvcc failed compiling {}", path.display());
            }
        } else {
            std::fs::write(&out_file, []).unwrap_or_else(|e| {
                panic!(
                    "failed to write placeholder kernel file {}: {e}",
                    out_file.display()
                )
            });
        }
        println!(
            "cargo:rustc-env=REFLEX_KERNEL_{}={}",
            stem.to_uppercase(),
            out_file.display()
        );
    }
}
