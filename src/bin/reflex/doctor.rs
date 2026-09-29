//! `reflex doctor`: a one-shot health report for the local CUDA/GPU
//! toolchain. Wires together checks that already existed as side effects of
//! other subcommands (`src/diagnostics.rs`'s GPU probe, compute-capability-
//! vs-build-time-`REFLEX_CUDA_ARCH` match, driver-error translation) plus a
//! real AOT-kernel load+launch check (`aot::verify_kernel_launch`) and an
//! NVML-availability check (`energy::probe_availability`), so a user can ask
//! "is this machine set up correctly?" directly instead of inferring it from
//! `reflex smoke`/`generate` failing partway through with a raw driver error.
//!
//! No GGUF/model loading -- hardware/toolchain check only, same scope class
//! as `reflex smoke`.
//!
//! **No per-phase energy here, deliberately.** `doctor` runs no timed
//! execution phase and does not even emit a total `joules` figure -- it only
//! *probes* whether energy sampling would work (`energy::probe_availability`,
//! surfaced as the `nvml_energy` check reporting the method a real run would
//! use). There is therefore no phase boundary to bracket and no
//! `REFLEX_PHASE_OK`/`PhaseTimingJson` output to add, unlike `smoke` (whose
//! CUDA-init / kernel-load / kernel-launch boundaries are real timed phases)
//! and `bench` (which loads a model and reports its load phases additively).
//! Inventing a "phase" for a set of independent, sub-millisecond boolean
//! checks would be exactly the kind of fabricated measurement this project's
//! energy work avoids.
//!
//! Usage: `reflex doctor [--json]`
//!
//! Exit codes (mirroring `reflex check`'s 0/1/2 contract, and using
//! `std::process::exit` directly rather than `reflex_engine::fast_exit` for
//! the same reason `check.rs` does -- this exit-code contract is the whole
//! point of the subcommand, not a cold-start latency measurement `fast_exit`
//! would otherwise be protecting): `0` = every check passed (a `warn`, e.g.
//! missing NVML, is fine -- doctor must stay usable on machines that will
//! never have NVML), `1` = at least one check failed, `2` = usage error.
//!
//! `--json` (needs `cargo build --features json-output`) prints each check
//! and the summary as one line of JSON instead of the plain
//! `REFLEX_DOCTOR_*` key=value text -- see `reflex_engine::cli_output`'s doc
//! comment for the exact shapes.

use reflex_engine::{aot, diagnostics, energy};

/// Build-time constants `build.rs` already exposes crate-wide (see
/// `src/diagnostics.rs`'s `COMPILED_ARCH`/`src/aot.rs`'s `KERNEL_FORMAT` --
/// duplicated here rather than made `pub` there, since this is the only
/// other place that wants to *display* them rather than act on them).
const COMPILED_ARCH: &str = env!("REFLEX_CUDA_ARCH");
const KERNEL_FORMAT: &str = env!("REFLEX_KERNEL_FORMAT");

enum Status {
    Pass,
    Warn,
    Fail,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "fail",
        }
    }
}

struct Check {
    name: &'static str,
    status: Status,
    detail: String,
}

fn print_check(json: bool, check: &Check) {
    if !json {
        println!(
            "REFLEX_DOCTOR_CHECK name={} status={} detail={:?}",
            check.name,
            check.status.as_str(),
            check.detail,
        );
        return;
    }
    #[cfg(feature = "json-output")]
    reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::DoctorCheckJson {
        name: check.name,
        status: check.status.as_str(),
        detail: check.detail.clone(),
    });
    #[cfg(not(feature = "json-output"))]
    panic!("--json requires this binary to be built with `cargo build --features json-output`");
}

pub fn run(args: Vec<String>) {
    let mut json = false;
    for arg in &args {
        match arg.as_str() {
            "--json" => json = true,
            other => {
                eprintln!("usage: reflex doctor [--json]\nunexpected argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut checks: Vec<Check> = Vec::new();

    let device = match diagnostics::init_device_with_diagnostics(0) {
        Ok(device) => {
            let detail = diagnostics::probe(&device)
                .map(|diag| diag.to_string())
                .unwrap_or_else(|e| e);
            checks.push(Check {
                name: "gpu_probe",
                status: Status::Pass,
                detail,
            });
            Some(device)
        }
        Err(e) => {
            checks.push(Check {
                name: "gpu_probe",
                status: Status::Fail,
                detail: e,
            });
            None
        }
    };

    let arch_display = if COMPILED_ARCH.is_empty() {
        "portable-ptx".to_string()
    } else {
        COMPILED_ARCH.to_string()
    };
    match &device {
        Some(device) => match diagnostics::check_kernel_compute_capability(device) {
            Err(e) => checks.push(Check {
                name: "compute_capability",
                status: Status::Fail,
                detail: e,
            }),
            Ok(()) => {
                // A fatbin build embeds a PTX fallback, so there's no hard mismatch
                // to fail on -- but the native-vs-fallback distinction is still worth
                // surfacing (a fallback means the driver JITs, i.e. slower first load
                // than a native zero-JIT image in the fatbin would be).
                let (status, detail) = if KERNEL_FORMAT == "fatbin" {
                    match diagnostics::fatbin_native_for_device(device) {
                        Ok(true) => (
                            Status::Pass,
                            format!(
                                "kernel_format={KERNEL_FORMAT} compiled_archs={} native (zero-JIT) match for detected GPU",
                                diagnostics::COMPILED_ARCHS
                            ),
                        ),
                        Ok(false) => (
                            Status::Warn,
                            format!(
                                "kernel_format={KERNEL_FORMAT} compiled_archs={} does not include the detected GPU -- will fall back to the embedded PTX and be driver-JIT'd",
                                diagnostics::COMPILED_ARCHS
                            ),
                        ),
                        Err(e) => (Status::Fail, e),
                    }
                } else {
                    (
                        Status::Pass,
                        format!("kernel_format={KERNEL_FORMAT} compiled_arch={arch_display} matches detected GPU"),
                    )
                };
                checks.push(Check {
                    name: "compute_capability",
                    status,
                    detail,
                });
            }
        },
        None => checks.push(Check {
            name: "compute_capability",
            status: Status::Fail,
            detail: "skipped: gpu_probe failed".to_string(),
        }),
    }

    match &device {
        Some(device) => match aot::verify_kernel_launch(device) {
            Ok(()) => checks.push(Check {
                name: "aot_kernel_launch",
                status: Status::Pass,
                detail: format!(
                    "kernel_format={KERNEL_FORMAT}: loaded and launched the AOT smoke kernel, result correct"
                ),
            }),
            Err(e) => checks.push(Check {
                name: "aot_kernel_launch",
                status: Status::Fail,
                detail: e,
            }),
        },
        None => checks.push(Check {
            name: "aot_kernel_launch",
            status: Status::Fail,
            detail: "skipped: gpu_probe failed".to_string(),
        }),
    }

    match energy::probe_availability(0) {
        Ok(method) => checks.push(Check {
            name: "nvml_energy",
            status: Status::Pass,
            detail: if energy::force_polled_requested() {
                format!(
                    "method={} (forced by REFLEX_NVML_FORCE_POLLED; the total-energy counter would otherwise be preferred)",
                    method.as_str()
                )
            } else {
                format!("method={}", method.as_str())
            },
        }),
        Err(reason) => checks.push(Check {
            name: "nvml_energy",
            status: Status::Warn,
            detail: reason,
        }),
    }

    for check in &checks {
        print_check(json, check);
    }

    let checks_passed = checks
        .iter()
        .filter(|c| matches!(c.status, Status::Pass))
        .count();
    let checks_warned = checks
        .iter()
        .filter(|c| matches!(c.status, Status::Warn))
        .count();
    let checks_failed = checks
        .iter()
        .filter(|c| matches!(c.status, Status::Fail))
        .count();
    let ok = checks_failed == 0;

    if json {
        #[cfg(feature = "json-output")]
        reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::DoctorSummaryJson {
            ok,
            checks_passed,
            checks_warned,
            checks_failed,
        });
        #[cfg(not(feature = "json-output"))]
        panic!("--json requires this binary to be built with `cargo build --features json-output`");
    } else if ok {
        println!(
            "REFLEX_DOCTOR_OK checks_passed={checks_passed} checks_warned={checks_warned} checks_failed={checks_failed}"
        );
    } else {
        println!(
            "REFLEX_DOCTOR_FAIL checks_passed={checks_passed} checks_warned={checks_warned} checks_failed={checks_failed}"
        );
    }

    if ok {
        std::process::exit(0);
    }
    std::process::exit(1);
}
