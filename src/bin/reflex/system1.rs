//! System1 cold-start measurement: loads a real GGUF file, scores a fixed
//! set of candidate continuations of `prompt` in a single pass via
//! `model::Model::system1_evaluate` (no argmax-then-feedback decode loop),
//! and reports wall-clock time from process start to the scored result.
//!
//! **`score`/`probability` are relative to this run's candidate set only**,
//! not vocab-normalized log-probabilities/probabilities comparable across
//! different prompts or runs -- see `Model::system1_evaluate`'s doc comment.
//!
//! Usage: `reflex system1 <path-to-gguf> <prompt> --candidate <text>
//! [--candidate <text> ...] [--temperature T] [--lora <adapter.gguf>] [--json]`
//!
//! `--json` (needs `cargo build --features json-output`) prints each result
//! as one line of JSON instead of the plain `REFLEX_*_OK key=value` text --
//! see `reflex_engine::cli_output`'s doc comment for the exact shapes.
//!
//! **Phase breakdown** (same fields/rationale as `reflex generate`'s -- see
//! that binary's doc comment; added here so the TypeSafe Jev warm-vs-cold
//! comparison in HISTORY.md can be re-run with a real per-phase split
//! instead of guessing from the aggregate number): the `REFLEX_SYSTEM1_OK`
//! line also reports `gguf_open_ms`, `cuda_init_ms`, `model_load_ms`, and
//! `prompt_eval_ms` (the single-pass scoring forward pass, including any
//! `--lora` application time -- same bucketing asymmetry `generate.rs`
//! already has, not a new inconsistency). `scripts/
//! bench_cold_start_phases_system1.sh` runs this N times and reports
//! per-phase p50/p95 across runs, mirroring `bench_cold_start_phases.sh`.
//!
//! Each phase is additionally emitted as its own additive `REFLEX_PHASE_OK
//! phase=<name> duration_ms=<ms> energy_joules=<j> energy_method=<m>` line,
//! printed *before* the aggregate `REFLEX_SYSTEM1_OK` line (the `energy_*`
//! fields appear only when a measurement is available) -- the existing
//! aggregate line and its fields are unchanged. See `src/energy.rs`'s doc
//! comment for the per-phase (delta-from-cumulative) semantics and the
//! counter-vs-polled granularity caveat.
//!
//! Exits via `reflex_engine::fast_exit` after printing the result instead
//! of returning from `run` normally -- see that function's doc comment for
//! why a graceful return costs several extra seconds of CUDA-context-
//! teardown wall-clock time on GPU-virtualized rented instances.
//!
//! Supports dense/MoE Qwen3, Qwen3.5 hybrid Gated DeltaNet, and DeepSeek-V2/V3
//! MLA models (see `Model::system1_evaluate`'s doc comment for how each
//! architecture is dispatched). **Hybrid models currently only support
//! single-token candidates** -- a multi-token `--candidate` on a hybrid
//! model is rejected with a clear error (see
//! `Model::system1_evaluate_hybrid`'s doc comment for why); dense/MoE and
//! MLA support both single- and multi-token candidates.

use reflex_engine::diagnostics;
use reflex_engine::energy;
use reflex_engine::gguf::GgufFile;
use reflex_engine::model::{Model, System1Candidate};
use std::time::Instant;

fn energy_suffix(measurement: Option<&energy::EnergyMeasurement>) -> String {
    match measurement {
        Some(m) => format!(" joules={:.3} energy_method={}", m.joules, m.method.as_str()),
        None => String::new(),
    }
}

/// Per-phase energy is the *delta* between two cumulative
/// [`energy::EnergyMeasurement`] readings (`measure()` returns cumulative
/// joules since `start()`, and the underlying counter/accumulator is
/// monotonic). `None` if either reading is unavailable. See the identical
/// helper in `generate.rs`.
fn phase_energy_delta(
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
/// phase. Purely additive: the aggregate `REFLEX_SYSTEM1_OK` line and its
/// existing fields are unchanged. See the identical helper in `generate.rs`.
fn print_phase_ok(
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
/// cumulative energy readings captured at each boundary (in order). See the
/// identical helper in `generate.rs`.
#[allow(clippy::too_many_arguments)]
fn print_phase_report(
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

// Only called from a `#[cfg(not(feature = "json-output"))]` arm below --
// `#[allow(dead_code)]` since a build *with* that feature never reaches it.
#[allow(dead_code)]
fn json_output_unavailable() -> ! {
    panic!("--json requires this binary to be built with `cargo build --features json-output`");
}

fn print_lora_ok(json: bool, path: &str, tensors_applied: usize) {
    if !json {
        println!("REFLEX_LORA_OK path={path:?} tensors_applied={tensors_applied}");
        return;
    }
    #[cfg(feature = "json-output")]
    reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::LoraAppliedJson {
        path: path.to_string(),
        tensors_applied,
    });
    #[cfg(not(feature = "json-output"))]
    json_output_unavailable();
}

pub fn run(args: Vec<String>) {
    let t0 = Instant::now();
    let sampler = energy::EnergySampler::start(0);

    let mut gguf_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut candidate_texts: Vec<String> = Vec::new();
    let mut temperature: f32 = 1.0;
    let mut lora_path: Option<String> = None;
    let mut json = false;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--candidate" => candidate_texts.push(args.next().expect("--candidate requires text")),
            "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
            "--temperature" => {
                let raw = args.next().expect("--temperature requires a number");
                temperature = raw.parse().unwrap_or_else(|_| {
                    panic!("--temperature must be a positive number, got {raw:?}")
                });
            }
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            _ if prompt.is_none() => prompt = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| {
        panic!(
            "usage: reflex system1 <path-to-gguf> <prompt> --candidate <text> [--candidate <text> ...] \
             [--temperature T] [--lora <adapter.gguf>]"
        )
    });
    let prompt = prompt.unwrap_or_else(|| panic!("a prompt is required"));
    if candidate_texts.is_empty() {
        panic!("at least one --candidate is required");
    }

    let file =
        GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let gguf_open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let e_gguf_open = sampler.measure();
    let device = diagnostics::init_device_with_diagnostics(0).unwrap_or_else(|e| panic!("{e}"));
    let cuda_init_ms = t0.elapsed().as_secs_f64() * 1000.0 - gguf_open_ms;
    let e_cuda_init = sampler.measure();
    if let Ok(diag) = diagnostics::probe(&device) {
        eprintln!("{diag}");
    }
    let mut model = Model::load(device, &file).expect("failed to load model");
    let model_load_ms = t0.elapsed().as_secs_f64() * 1000.0 - gguf_open_ms - cuda_init_ms;
    let e_model_load = sampler.measure();
    let model_ready_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Some(lora_path) = &lora_path {
        let applied = model
            .apply_lora(std::path::Path::new(lora_path))
            .expect("failed to apply LoRA adapter");
        print_lora_ok(json, lora_path, applied);
    }

    let candidates: Vec<System1Candidate> = candidate_texts
        .iter()
        .map(|text| System1Candidate { text: text.clone() })
        .collect();
    let response = model
        .system1_evaluate(&prompt, &candidates, temperature)
        .expect("system1_evaluate failed");
    let prompt_eval_ms = t0.elapsed().as_secs_f64() * 1000.0 - model_ready_ms;

    for (idx, (result, &probability)) in response
        .results
        .iter()
        .zip(&response.probabilities)
        .enumerate()
    {
        if json {
            #[cfg(feature = "json-output")]
            reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::System1CandidateJson {
                idx,
                text: result.text.clone(),
                token_ids: result.token_ids.clone(),
                score: result.score,
                probability,
            });
            #[cfg(not(feature = "json-output"))]
            json_output_unavailable();
        } else {
            let token_ids: Vec<String> = result.token_ids.iter().map(|t| t.to_string()).collect();
            println!(
                "REFLEX_SYSTEM1_CANDIDATE_OK idx={idx} text={:?} token_ids=[{}] score={:.6} probability={:.6}",
                result.text,
                token_ids.join(","),
                result.score,
                probability,
            );
        }
    }

    let (best_idx, best) = response
        .probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| (idx, &response.results[idx]))
        .expect("system1_evaluate returned no results");

    let elapsed = t0.elapsed();
    let energy_measurement = sampler.measure();
    let process_start_to_result_ms = elapsed.as_secs_f64() * 1000.0;
    print_phase_report(
        json,
        gguf_open_ms,
        cuda_init_ms,
        model_load_ms,
        prompt_eval_ms,
        &e_gguf_open,
        &e_cuda_init,
        &e_model_load,
        &energy_measurement,
    );
    if json {
        #[cfg(feature = "json-output")]
        reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::System1ResultJson {
            schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
            process_start_to_result_ms,
            gguf_open_ms,
            cuda_init_ms,
            model_load_ms,
            prompt_eval_ms,
            num_candidates: response.results.len(),
            best_idx,
            best_text: best.text.clone(),
            entropy: response.entropy,
            joules: energy_measurement.as_ref().map(|m| m.joules),
            energy_method: energy_measurement.as_ref().map(|m| m.method.as_str()),
        });
        #[cfg(not(feature = "json-output"))]
        json_output_unavailable();
    } else {
        println!(
            "REFLEX_SYSTEM1_OK process_start_to_result_ms={process_start_to_result_ms:.3} gguf_open_ms={gguf_open_ms:.3} cuda_init_ms={cuda_init_ms:.3} model_load_ms={model_load_ms:.3} prompt_eval_ms={prompt_eval_ms:.3} num_candidates={} best_idx={best_idx} best_text={:?} entropy={:.6}{}",
            response.results.len(),
            best.text,
            response.entropy,
            energy_suffix(energy_measurement.as_ref()),
        );
    }
    reflex_engine::fast_exit(0);
}
