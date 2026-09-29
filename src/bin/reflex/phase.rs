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

use reflex_engine::energy;

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

// Only ever called from a `#[cfg(not(feature = "json-output"))]` arm -- the
// `#[allow(dead_code)]` keeps a build *with* that feature from warning.
#[allow(dead_code)]
pub fn json_output_unavailable() -> ! {
    panic!("--json requires this binary to be built with `cargo build --features json-output`");
}
