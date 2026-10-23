//! Embedding cold-start measurement: loads an embedding GGUF (an encoder:
//! `<arch>.attention.causal = false`, e.g. Nemotron-3.5-Embed-1B), embeds one
//! text via `model::Model::embed` and reports wall-clock time from process
//! start to the embedding.
//!
//! Usage: `reflex embed <path-to-gguf> <text> [--weights f16|f32] [--json]`
//!
//! The text is embedded as given, without a template: add the model's input
//! prefix yourself (Nemotron-3.5-Embed: `"query: "` for a search query,
//! `"passage: "` for a document). Prints a `REFLEX_EMBEDDING_OK dims=<n>
//! values=[...]` line (the mean-pooled, L2-normalized vector) and then
//! `REFLEX_EMBED_OK` with the same phase fields as `reflex system1`
//! (`gguf_open_ms`, `cuda_init_ms`, `model_load_ms`, `prompt_eval_ms`, the
//! last being the embedding forward pass), preceded by one `REFLEX_PHASE_OK`
//! line per phase. `--json` (needs `--features json-output`) prints
//! `EmbedResultJson` instead, with the vector in `embedding`.
//!
//! Exits via `reflex_engine::fast_exit`, as every model-loading subcommand
//! does (see that function's doc comment).

#[cfg(not(feature = "json-output"))]
use crate::phase::json_output_unavailable;
use crate::phase::{energy_suffix, print_phase_report};
use reflex_engine::diagnostics;
use reflex_engine::energy;
use reflex_engine::gguf::GgufFile;
use std::time::Instant;

pub fn run(args: Vec<String>) {
    let t0 = Instant::now();
    let sampler = energy::EnergySampler::start(0);

    let mut gguf_path: Option<String> = None;
    let mut text: Option<String> = None;
    let mut weights_flag: Option<String> = None;
    let mut json = false;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--weights" => {
                weights_flag = Some(args.next().expect("--weights requires `f16` or `f32`"))
            }
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            _ if text.is_none() => text = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| {
        panic!("usage: reflex embed <path-to-gguf> <text> [--weights f16|f32] [--json]")
    });
    let text = text.unwrap_or_else(|| panic!("a text to embed is required"));

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
    let model = crate::load_model(
        device,
        &file,
        weights_flag.as_deref(),
        reflex_engine::model::WeightsDtype::DEFAULT,
        None,
    )
    .unwrap_or_else(|e| crate::fail("failed to load model", e));
    let weights_dtype = model.weights_dtype().as_str();
    let model_load_ms = t0.elapsed().as_secs_f64() * 1000.0 - gguf_open_ms - cuda_init_ms;
    let e_model_load = sampler.measure();
    let model_ready_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let embedding = model
        .embed(&text)
        .unwrap_or_else(|e| crate::fail("embed failed", e));
    let prompt_eval_ms = t0.elapsed().as_secs_f64() * 1000.0 - model_ready_ms;
    let elapsed = t0.elapsed();
    let energy_measurement = sampler.measure();
    let process_start_to_embedding_ms = elapsed.as_secs_f64() * 1000.0;

    let dims = embedding.vector.len();
    if !json {
        let values: Vec<String> = embedding.vector.iter().map(|v| format!("{v:e}")).collect();
        println!(
            "REFLEX_EMBEDDING_OK dims={dims} values=[{}]",
            values.join(",")
        );
    }
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
        reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::EmbedResultJson {
            schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
            process_start_to_embedding_ms,
            gguf_open_ms,
            cuda_init_ms,
            model_load_ms,
            prompt_eval_ms,
            prompt_tokens: embedding.prompt_tokens,
            dims,
            embedding: embedding.vector,
            joules: energy_measurement.as_ref().map(|m| m.joules),
            energy_method: energy_measurement.as_ref().map(|m| m.method.as_str()),
            weights_dtype,
        });
        #[cfg(not(feature = "json-output"))]
        json_output_unavailable();
    } else {
        println!(
            "REFLEX_EMBED_OK process_start_to_embedding_ms={process_start_to_embedding_ms:.3} gguf_open_ms={gguf_open_ms:.3} cuda_init_ms={cuda_init_ms:.3} model_load_ms={model_load_ms:.3} prompt_eval_ms={prompt_eval_ms:.3} prompt_tokens={} dims={dims}{} weights_dtype={weights_dtype}",
            embedding.prompt_tokens,
            energy_suffix(energy_measurement.as_ref()),
        );
    }
    crate::report_f16_activation_stats(&model);
    reflex_engine::fast_exit(0);
}
