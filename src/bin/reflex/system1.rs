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
//! [--candidate <text> ...] [--temperature T] [--lora <adapter.gguf>]`
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
//! Exits via `reflex_engine::fast_exit` after printing the result instead
//! of returning from `run` normally -- see that function's doc comment for
//! why a graceful return costs several extra seconds of CUDA-context-
//! teardown wall-clock time on GPU-virtualized rented instances.
//!
//! Dense/MoE Qwen3 models only (see `Model::system1_evaluate`'s doc
//! comment) -- hybrid Qwen3.5 and DeepSeek-V2/V3 MLA are rejected with a
//! clear error, same as every other unsupported-architecture case in this
//! project.

use reflex_engine::diagnostics;
use reflex_engine::gguf::GgufFile;
use reflex_engine::model::{Model, System1Candidate};
use std::time::Instant;

pub fn run(args: Vec<String>) {
    let t0 = Instant::now();

    let mut gguf_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut candidate_texts: Vec<String> = Vec::new();
    let mut temperature: f32 = 1.0;
    let mut lora_path: Option<String> = None;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
    let device = diagnostics::init_device_with_diagnostics(0).unwrap_or_else(|e| panic!("{e}"));
    let cuda_init_ms = t0.elapsed().as_secs_f64() * 1000.0 - gguf_open_ms;
    if let Ok(diag) = diagnostics::probe(&device) {
        eprintln!("{diag}");
    }
    let mut model = Model::load(device, &file).expect("failed to load model");
    let model_load_ms = t0.elapsed().as_secs_f64() * 1000.0 - gguf_open_ms - cuda_init_ms;
    let model_ready_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Some(lora_path) = &lora_path {
        let applied = model
            .apply_lora(std::path::Path::new(lora_path))
            .expect("failed to apply LoRA adapter");
        println!("REFLEX_LORA_OK path={lora_path:?} tensors_applied={applied}");
    }

    let candidates: Vec<System1Candidate> = candidate_texts
        .iter()
        .map(|text| System1Candidate { text: text.clone() })
        .collect();
    let response = model
        .system1_evaluate(&prompt, &candidates, temperature)
        .expect("system1_evaluate failed");
    let prompt_eval_ms = t0.elapsed().as_secs_f64() * 1000.0 - model_ready_ms;

    for (idx, (result, probability)) in response
        .results
        .iter()
        .zip(&response.probabilities)
        .enumerate()
    {
        let token_ids: Vec<String> = result.token_ids.iter().map(|t| t.to_string()).collect();
        println!(
            "REFLEX_SYSTEM1_CANDIDATE_OK idx={idx} text={:?} token_ids=[{}] score={:.6} probability={:.6}",
            result.text,
            token_ids.join(","),
            result.score,
            probability,
        );
    }

    let (best_idx, best) = response
        .probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| (idx, &response.results[idx]))
        .expect("system1_evaluate returned no results");

    let elapsed = t0.elapsed();
    println!(
        "REFLEX_SYSTEM1_OK process_start_to_result_ms={:.3} gguf_open_ms={gguf_open_ms:.3} cuda_init_ms={cuda_init_ms:.3} model_load_ms={model_load_ms:.3} prompt_eval_ms={prompt_eval_ms:.3} num_candidates={} best_idx={best_idx} best_text={:?} entropy={:.6}",
        elapsed.as_secs_f64() * 1000.0,
        response.results.len(),
        best.text,
        response.entropy,
    );
    reflex_engine::fast_exit(0);
}
