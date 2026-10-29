//! Shared per-phase energy + timing emission, used by every subcommand that
//! has a real phase boundary (`generate`, `system1`, `smoke`, `bench`).
//!
//! Before M4 these four helpers lived twice -- identical copies in
//! `generate.rs` and `system1.rs` -- so the M1 follow-up ("the same per-phase
//! treatment for smoke/bench") would have added a third and fourth copy.
//! Hoisted here instead of into the `reflex_engine` library because they are
//! pure CLI-output glue (they format `REFLEX_PHASE_OK` lines and
//! `PhaseTimingJson` objects), not engine measurement logic -- `energy.rs`'s
//! `EnergySampler` remains the library-side primitive, untouched.
//!
//! Semantics are unchanged from the copies they replace: `measure()` returns
//! *cumulative* joules since `start()`, so a phase's figure is the delta
//! between the checkpoint at its end and the checkpoint at its start
//! ([`phase_energy_delta`]). Purely additive output: the aggregate
//! `REFLEX_*_OK` lines and their fields are never touched, and the `energy_*`
//! fields appear only when a measurement is available. See `src/energy.rs`'s
//! doc comment for the counter-vs-polled granularity caveat.
//!
//! [`RunReport`] is the run's context (GPU, driver, kernel image, weight mode,
//! model file), printed as one `REFLEX_RUN_REPORT` line after the aggregate
//! line by `generate` and `system1`. It is collected only after the last timer
//! has been read and the aggregate line printed, and it reuses the GPU probe
//! and NVML handle the run already has, so it adds nothing to any reported
//! phase.

use reflex_engine::diagnostics::{self, GpuDiagnostics, KernelImage};
use reflex_engine::energy;
use reflex_engine::gguf::{GgufFile, GgufValue};
use reflex_engine::model::Model;

/// The `schema_version` on the `REFLEX_RUN_REPORT` line: the `--json`
/// contract's `SCHEMA_VERSION`, repeated here because `cli_output` only exists
/// in `json-output` builds (a test keeps the two equal).
pub const RUN_REPORT_SCHEMA_VERSION: &str = "1.5.0";

/// Builds the `" joules=... energy_method=..."` suffix to append to a
/// `REFLEX_*_OK` line, or an empty string when no energy measurement is
/// available (compiled without `--features nvml`, or NVML unavailable on
/// this machine) -- see `energy::EnergySampler::measure`'s doc comment.
pub fn energy_suffix(measurement: Option<&energy::EnergyMeasurement>) -> String {
    match measurement {
        Some(m) => format!(
            " joules={:.3} energy_method={}",
            m.joules,
            m.method.as_str()
        ),
        None => String::new(),
    }
}

/// Per-phase energy is the *delta* between two cumulative
/// [`energy::EnergyMeasurement`] readings (`measure()` returns cumulative
/// joules since `start()`, and the underlying counter/accumulator is
/// monotonic). `None` if either reading is unavailable.
pub fn phase_energy_delta(
    after: &Option<energy::EnergyMeasurement>,
    before: &Option<energy::EnergyMeasurement>,
) -> Option<energy::EnergyMeasurement> {
    match (after, before) {
        (Some(a), Some(b)) => Some(energy::EnergyMeasurement {
            joules: a.joules - b.joules,
            method: a.method,
        }),
        _ => None,
    }
}

/// Prints one `REFLEX_PHASE_OK` line (or its `--json` form) for a single
/// phase. Purely additive: the aggregate `REFLEX_*_OK` lines and their
/// existing fields are unchanged.
pub fn print_phase_ok(
    json: bool,
    phase: &'static str,
    duration_ms: f64,
    energy: Option<&energy::EnergyMeasurement>,
) {
    if !json {
        let suffix = match energy {
            Some(m) => format!(
                " energy_joules={:.3} energy_method={}",
                m.joules,
                m.method.as_str()
            ),
            None => String::new(),
        };
        println!("REFLEX_PHASE_OK phase={phase} duration_ms={duration_ms:.3}{suffix}");
        return;
    }
    #[cfg(feature = "json-output")]
    reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::PhaseTimingJson {
        schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
        phase,
        duration_ms,
        energy_joules: energy.map(|m| m.joules),
        energy_method: energy.map(|m| m.method.as_str()),
    });
    #[cfg(not(feature = "json-output"))]
    json_output_unavailable();
}

/// Emits the four bracketed cold-start phases' ms + joules, given the
/// cumulative energy readings captured at each boundary (in order). The
/// per-phase energy is each boundary's delta from the previous one; the
/// `gguf_open` phase is the first cumulative reading measured from
/// `EnergySampler::start`.
#[allow(clippy::too_many_arguments)]
pub fn print_phase_report(
    json: bool,
    gguf_open_ms: f64,
    cuda_init_ms: f64,
    model_load_ms: f64,
    prompt_eval_ms: f64,
    e_gguf_open: &Option<energy::EnergyMeasurement>,
    e_cuda_init: &Option<energy::EnergyMeasurement>,
    e_model_load: &Option<energy::EnergyMeasurement>,
    e_prompt_eval: &Option<energy::EnergyMeasurement>,
) {
    print_phase_ok(json, "gguf_open", gguf_open_ms, e_gguf_open.as_ref());
    let cuda = phase_energy_delta(e_cuda_init, e_gguf_open);
    print_phase_ok(json, "cuda_init", cuda_init_ms, cuda.as_ref());
    let load = phase_energy_delta(e_model_load, e_cuda_init);
    print_phase_ok(json, "model_load", model_load_ms, load.as_ref());
    let eval = phase_energy_delta(e_prompt_eval, e_model_load);
    print_phase_ok(json, "prompt_eval", prompt_eval_ms, eval.as_ref());
}

/// The four bracketed phases, in ms, as printed on the `REFLEX_PHASE_OK` lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phases {
    pub gguf_open_ms: f64,
    pub cuda_init_ms: f64,
    pub model_load_ms: f64,
    pub prompt_eval_ms: f64,
}

/// One run's context next to its timings; see this module's doc comment.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub command: &'static str,
    pub first_result_ms: f64,
    /// `generate` only.
    pub last_result_ms: Option<f64>,
    pub phases: Phases,
    pub gpu_name: Option<String>,
    pub compute_capability: Option<(i32, i32)>,
    pub vram_total_bytes: Option<usize>,
    pub cuda_driver_version: Option<(i32, i32)>,
    pub driver_version: Option<String>,
    pub kernel_format: &'static str,
    pub kernel_image: Option<KernelImage>,
    /// `REFLEX_CUDA_ARCHS` (fatbin builds), else empty.
    pub compiled_archs: &'static str,
    /// `qr`, `f16` or `f32`.
    pub weights: &'static str,
    pub weights_dtype: &'static str,
    pub kv_cache: &'static str,
    pub arch: Option<String>,
    pub model_file: String,
    pub model_bytes: u64,
    pub energy: Option<energy::EnergyMeasurement>,
}

impl RunReport {
    /// Gathers the context. Call it after the run's last timer read: the only
    /// new work is `cuDriverGetVersion` and, with `REFLEX_ENERGY=1`, one NVML driver
    /// query on the sampler's existing handle. `gpu` is the probe the run
    /// already printed to stderr; `energy` is the aggregate line's reading.
    #[allow(clippy::too_many_arguments)]
    pub fn collect(
        command: &'static str,
        first_result_ms: f64,
        last_result_ms: Option<f64>,
        phases: Phases,
        gpu: Option<&GpuDiagnostics>,
        sampler: &energy::EnergySampler,
        model: &Model,
        file: &GgufFile,
        gguf_path: &str,
        energy: Option<&energy::EnergyMeasurement>,
    ) -> RunReport {
        let compute_capability = gpu.map(|g| g.compute_capability);
        let weights_dtype = model.weights_dtype().as_str();
        RunReport {
            command,
            first_result_ms,
            last_result_ms,
            phases,
            gpu_name: gpu.map(|g| g.name.clone()),
            compute_capability,
            vram_total_bytes: gpu.map(|g| g.vram_total_bytes),
            cuda_driver_version: diagnostics::cuda_driver_version(),
            driver_version: sampler.driver_version(),
            kernel_format: diagnostics::KERNEL_FORMAT,
            kernel_image: compute_capability.map(diagnostics::kernel_image_for),
            compiled_archs: if diagnostics::KERNEL_FORMAT == "fatbin" {
                diagnostics::COMPILED_ARCHS
            } else {
                ""
            },
            weights: if model.quant_resident() {
                "qr"
            } else {
                weights_dtype
            },
            weights_dtype,
            kv_cache: model.kv_cache_dtype().as_str(),
            arch: file
                .metadata
                .get("general.architecture")
                .and_then(GgufValue::as_str)
                .map(str::to_string),
            model_file: std::path::Path::new(gguf_path).file_name().map_or_else(
                || gguf_path.to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            model_bytes: file.file_len() as u64,
            energy: energy.cloned(),
        }
    }

    /// The plain-text `REFLEX_RUN_REPORT` line. Keys shared with the
    /// aggregate line (`gguf_open_ms`, ...) carry the same value in the same
    /// format, so a reader taking the first or the last `key=` match gets the
    /// same number. Unknown fields are left out.
    pub fn line(&self) -> String {
        use std::fmt::Write;
        let p = &self.phases;
        let mut s = format!(
            "REFLEX_RUN_REPORT schema_version={RUN_REPORT_SCHEMA_VERSION} command={} process_start_to_first_result_ms={:.3}",
            self.command, self.first_result_ms
        );
        if let Some(ms) = self.last_result_ms {
            let _ = write!(s, " process_start_to_last_result_ms={ms:.3}");
        }
        let _ = write!(
            s,
            " gguf_open_ms={:.3} cuda_init_ms={:.3} model_load_ms={:.3} prompt_eval_ms={:.3}",
            p.gguf_open_ms, p.cuda_init_ms, p.model_load_ms, p.prompt_eval_ms
        );
        if let Some(name) = &self.gpu_name {
            let _ = write!(s, " gpu_name={name:?}");
        }
        if let Some((major, minor)) = self.compute_capability {
            let _ = write!(s, " compute_capability={major}.{minor}");
        }
        if let Some(bytes) = self.vram_total_bytes {
            let _ = write!(s, " vram_total_mib={}", bytes / (1024 * 1024));
        }
        if let Some((major, minor)) = self.cuda_driver_version {
            let _ = write!(s, " cuda_driver_version={major}.{minor}");
        }
        if let Some(v) = &self.driver_version {
            let _ = write!(s, " driver_version={v}");
        }
        let _ = write!(s, " kernel_format={}", self.kernel_format);
        if let Some(image) = &self.kernel_image {
            let _ = write!(s, " kernel_image={}", image.mode);
            if let Some(arch) = &image.arch {
                let _ = write!(s, " kernel_image_arch={arch}");
            }
        }
        if !self.compiled_archs.is_empty() {
            let _ = write!(s, " compiled_archs={}", self.compiled_archs);
        }
        let _ = write!(
            s,
            " weights={} weights_dtype={} kv_cache={}",
            self.weights, self.weights_dtype, self.kv_cache
        );
        if let Some(arch) = &self.arch {
            let _ = write!(s, " arch={arch:?}");
        }
        let _ = write!(
            s,
            " model_file={:?} model_bytes={}",
            self.model_file, self.model_bytes
        );
        if let Some(m) = &self.energy {
            let _ = write!(
                s,
                " energy_joules={:.3} energy_method={}",
                m.joules,
                m.method.as_str()
            );
        }
        s
    }

    #[cfg(feature = "json-output")]
    pub fn json(&self) -> reflex_engine::cli_output::RunReportJson {
        use reflex_engine::cli_output::*;
        let p = &self.phases;
        let major_minor = |(a, b): (i32, i32)| format!("{a}.{b}");
        RunReportJson {
            schema_version: SCHEMA_VERSION,
            report: "run",
            command: self.command,
            process_start_to_first_result_ms: self.first_result_ms,
            process_start_to_last_result_ms: self.last_result_ms,
            phases_ms: RunPhasesJson {
                gguf_open: p.gguf_open_ms,
                cuda_init: p.cuda_init_ms,
                model_load: p.model_load_ms,
                prompt_eval: p.prompt_eval_ms,
            },
            gpu: RunGpuJson {
                name: self.gpu_name.clone(),
                compute_capability: self.compute_capability.map(major_minor),
                vram_total_mib: self.vram_total_bytes.map(|b| b / (1024 * 1024)),
                cuda_driver_version: self.cuda_driver_version.map(major_minor),
                driver_version: self.driver_version.clone(),
            },
            kernels: RunKernelsJson {
                format: self.kernel_format,
                image: self.kernel_image.as_ref().map(|i| i.mode),
                image_arch: self.kernel_image.as_ref().and_then(|i| i.arch.clone()),
                compiled_archs: (!self.compiled_archs.is_empty())
                    .then(|| self.compiled_archs.split(',').map(str::to_string).collect()),
            },
            weights: RunWeightsJson {
                mode: self.weights,
                dtype: self.weights_dtype,
            },
            kv_cache: self.kv_cache,
            model: RunModelJson {
                arch: self.arch.clone(),
                file: self.model_file.clone(),
                bytes: self.model_bytes,
            },
            energy: self.energy.as_ref().map(|m| RunEnergyJson {
                joules: m.joules,
                method: m.method.as_str(),
            }),
        }
    }
}

/// Prints `report` as its `REFLEX_RUN_REPORT` line, or its `--json` object.
pub fn print_run_report(json: bool, report: &RunReport) {
    if !json {
        println!("{}", report.line());
        return;
    }
    #[cfg(feature = "json-output")]
    reflex_engine::cli_output::print_json_line(&report.json());
    #[cfg(not(feature = "json-output"))]
    json_output_unavailable();
}

// Only ever called from a `#[cfg(not(feature = "json-output"))]` arm -- the
// `#[allow(dead_code)]` keeps a build *with* that feature from warning.
#[allow(dead_code)]
pub fn json_output_unavailable() -> ! {
    panic!("--json requires this binary to be built with `cargo build --features json-output`");
}

#[cfg(test)]
mod tests {
    use super::*;
    use reflex_engine::energy::{EnergyMeasurement, EnergyMethod};

    /// A T4-shaped report with every field known.
    fn full() -> RunReport {
        RunReport {
            command: "system1",
            first_result_ms: 348.4004,
            last_result_ms: None,
            phases: Phases {
                gguf_open_ms: 1.25,
                cuda_init_ms: 120.5,
                model_load_ms: 210.0,
                prompt_eval_ms: 16.65,
            },
            gpu_name: Some("Tesla T4".to_string()),
            compute_capability: Some((7, 5)),
            vram_total_bytes: Some(15_828_320_256),
            cuda_driver_version: Some((13, 0)),
            driver_version: Some("595.91.07".to_string()),
            kernel_format: "fatbin",
            kernel_image: Some(KernelImage {
                mode: "native",
                arch: Some("sm_75".to_string()),
            }),
            compiled_archs: "sm_75,sm_80,sm_86",
            weights: "qr",
            weights_dtype: "f16",
            kv_cache: "f32",
            arch: Some("qwen3".to_string()),
            model_file: "Qwen3-0.6B-Q4_K_M.gguf".to_string(),
            model_bytes: 396_705_472,
            energy: Some(EnergyMeasurement {
                joules: 12.3456,
                method: EnergyMethod::TotalEnergyCounter,
            }),
        }
    }

    #[test]
    fn run_report_line_has_every_known_field_in_order() {
        assert_eq!(
            full().line(),
            "REFLEX_RUN_REPORT schema_version=1.5.0 command=system1 process_start_to_first_result_ms=348.400 \
             gguf_open_ms=1.250 cuda_init_ms=120.500 model_load_ms=210.000 prompt_eval_ms=16.650 \
             gpu_name=\"Tesla T4\" compute_capability=7.5 vram_total_mib=15095 cuda_driver_version=13.0 \
             driver_version=595.91.07 kernel_format=fatbin kernel_image=native kernel_image_arch=sm_75 \
             compiled_archs=sm_75,sm_80,sm_86 weights=qr weights_dtype=f16 kv_cache=f32 arch=\"qwen3\" \
             model_file=\"Qwen3-0.6B-Q4_K_M.gguf\" model_bytes=396705472 energy_joules=12.346 \
             energy_method=total_energy_counter"
        );
    }

    #[test]
    fn run_report_line_leaves_out_unknown_fields() {
        let mut r = full();
        r.command = "generate";
        r.last_result_ms = Some(400.0);
        r.gpu_name = None;
        r.compute_capability = None;
        r.vram_total_bytes = None;
        r.cuda_driver_version = None;
        r.driver_version = None;
        r.kernel_format = "ptx";
        r.kernel_image = None;
        r.compiled_archs = "";
        r.weights = "f32";
        r.weights_dtype = "f32";
        r.arch = None;
        r.energy = None;
        assert_eq!(
            r.line(),
            "REFLEX_RUN_REPORT schema_version=1.5.0 command=generate process_start_to_first_result_ms=348.400 \
             process_start_to_last_result_ms=400.000 gguf_open_ms=1.250 cuda_init_ms=120.500 \
             model_load_ms=210.000 prompt_eval_ms=16.650 kernel_format=ptx weights=f32 weights_dtype=f32 \
             kv_cache=f32 model_file=\"Qwen3-0.6B-Q4_K_M.gguf\" model_bytes=396705472"
        );
    }

    #[test]
    fn run_report_line_escapes_free_text() {
        let mut r = full();
        r.gpu_name = Some("GPU \"x\" y".to_string());
        r.model_file = "a b=c.gguf".to_string();
        let line = r.line();
        assert!(line.contains(r#" gpu_name="GPU \"x\" y" "#), "{line}");
        assert!(line.contains(r#" model_file="a b=c.gguf" "#), "{line}");
        assert!(!line.contains('\n'));
    }

    /// The bench scripts' `extract_total_joules` reads `' joules='` off the
    /// aggregate line; the report must not add a key ending in ` joules=`.
    #[test]
    fn run_report_line_has_no_bare_joules_key() {
        assert!(!full().line().contains(" joules="));
    }

    #[cfg(feature = "json-output")]
    #[test]
    fn run_report_schema_version_matches_the_json_contract() {
        assert_eq!(
            RUN_REPORT_SCHEMA_VERSION,
            reflex_engine::cli_output::SCHEMA_VERSION
        );
    }

    #[cfg(feature = "json-output")]
    #[test]
    fn run_report_json_shape() {
        assert_eq!(
            serde_json::to_string(&full().json()).unwrap(),
            concat!(
                r#"{"schema_version":"1.5.0","report":"run","command":"system1","#,
                r#""process_start_to_first_result_ms":348.4004,"#,
                r#""phases_ms":{"gguf_open":1.25,"cuda_init":120.5,"model_load":210.0,"prompt_eval":16.65},"#,
                r#""gpu":{"name":"Tesla T4","compute_capability":"7.5","vram_total_mib":15095,"#,
                r#""cuda_driver_version":"13.0","driver_version":"595.91.07"},"#,
                r#""kernels":{"format":"fatbin","image":"native","image_arch":"sm_75","#,
                r#""compiled_archs":["sm_75","sm_80","sm_86"]},"#,
                r#""weights":{"mode":"qr","dtype":"f16"},"kv_cache":"f32","#,
                r#""model":{"arch":"qwen3","file":"Qwen3-0.6B-Q4_K_M.gguf","bytes":396705472},"#,
                r#""energy":{"joules":12.3456,"method":"total_energy_counter"}}"#
            )
        );
    }

    /// `examples/relay_chain.sh` and `relay_orchestrator.py` take the last
    /// JSON line with `token_text`/`best_idx`; phase readers key on `phase`.
    #[cfg(feature = "json-output")]
    #[test]
    fn run_report_json_shares_no_selector_key_with_result_objects() {
        let mut r = full();
        r.gpu_name = None;
        r.energy = None;
        let v: serde_json::Value = serde_json::to_value(r.json()).unwrap();
        let top: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        for key in [
            "token_text",
            "best_idx",
            "best_text",
            "phase",
            "duration_ms",
            "gguf_open_ms",
            "process_start_to_first_token_ms",
            "process_start_to_result_ms",
            "weights_dtype",
            "joules",
        ] {
            assert!(!top.contains(&key), "top-level key {key} in {top:?}");
        }
    }
}
