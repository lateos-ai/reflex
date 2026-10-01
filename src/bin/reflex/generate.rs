//! Dense/MoE/hybrid Qwen3 cold-start measurement: loads a real GGUF file,
//! runs the prompt through `model::Model::generate`, and reports wall-clock
//! time from process start to the first generated token -- the project's
//! actual target metric.
//!
//! **Phase breakdown** (requested by real user feedback -- a single aggregate
//! number hides where the time actually goes): the `REFLEX_GENERATE_OK` line
//! also reports `gguf_open_ms` (mmap + header/metadata parse), `cuda_init_ms`
//! (`CudaDevice::new` -- driver init + primary context creation), `model_load_ms`
//! (dequantize-and-upload every weight tensor), and `prompt_eval_ms` (forward
//! pass through the prompt to the first sampled token) -- each a delta between
//! consecutive `Instant::now()` checkpoints around the corresponding call, not
//! an independently-measured wall clock. **Deliberately not included: a
//! "process launch" phase.** That covers OS `exec`/dynamic-linking/CRT init
//! *before* `main()` runs at all, which nothing inside this process can
//! observe -- get it the same way this project's own README benchmark table
//! already does, by wrapping the whole process in an external timer
//! (`/usr/bin/time -v`, `hyperfine`, etc.) and subtracting
//! `process_start_to_first_token_ms` from that external total.
//! `scripts/bench_cold_start_phases.sh` runs this N times and reports
//! per-phase p50/p95 across runs.
//!
//! Each phase is additionally emitted as its own additive `REFLEX_PHASE_OK
//! phase=<name> duration_ms=<ms> energy_joules=<j> energy_method=<m>` line,
//! printed *before* the aggregate `REFLEX_GENERATE_OK` line (the `energy_*`
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
//! Also the entry point for KV-cache export/import: `--export-kv <file>`
//! downloads the K/V cache produced by this run's initial prompt pass and
//! writes it to `<file>` (dense/MoE, hybrid Qwen3.5, and DeepSeek-V2/V3 MLA
//! models). `--import-kv <file>` loads a previously-exported cache, uploads
//! it as the starting state, and resumes generation from it -- `prompt` is
//! then the continuation text appended after the cached positions, not a
//! fresh prompt. `--max-tokens N` (default 1) generates up to N tokens,
//! feeding each one back in, stopping early on the tokenizer's EOS.
//!
//! Usage: `reflex generate <path-to-gguf> [prompt] [--max-tokens N] [--export-kv <file>] [--json]`
//!        `reflex generate <path-to-gguf> [continuation-prompt] [--max-tokens N] --import-kv <file>`
//!
//! `--json` (needs `cargo build --features json-output`) prints each result
//! as one line of JSON instead of the plain `REFLEX_*_OK key=value` text --
//! see `reflex_engine::cli_output`'s doc comment for the exact shapes.
//!
//! `--temperature F` (omitted, or `0.0`, keeps this project's original greedy-argmax
//! behavior -- the byte-exact-reproducible default `reflex check`'s
//! methodology depends on). Any positive value switches to temperature/top-k/top-p
//! sampling (`--top-k N`, `--top-p F`, `--seed N` for a reproducible draw) -- see
//! `reflex_engine::sampling`'s doc comment.
//!
//! `--lora <adapter.gguf>` applies a llama.cpp-format LoRA adapter to the
//! loaded model's weights once, at load time, before any forward pass runs
//! (see `model::Model::apply_lora` and `lora`'s module doc comment for the
//! file format and scope).
//!
//! `--model <repo_id[:filename]>` and `--quickstart` (both require `cargo
//! build --features download`) resolve a Hugging Face repo spec to a local
//! GGUF path via `reflex_engine::hf::resolve_gguf_path`/`resolve_quickstart`
//! *before* the usual `GgufFile::open` -- hf-hub is used strictly as a
//! downloader/cache here, never a new tensor-format ingestion path; see
//! `src/hf.rs`'s module doc comment. Exactly one of a positional
//! `<path-to-gguf>`, `--model`, or `--quickstart` must be given.

#[cfg(not(feature = "json-output"))]
use crate::phase::json_output_unavailable;
use crate::phase::{energy_suffix, print_phase_report};
use reflex_engine::diagnostics;
use reflex_engine::energy;
use reflex_engine::gguf::GgufFile;
use reflex_engine::kv_io;
use reflex_engine::model::{ArchitectureKind, Model};
use std::time::Instant;

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

fn print_kv_export_ok(
    json: bool,
    path: &str,
    kind: &'static str,
    seq_len: usize,
    num_layers: usize,
) {
    if !json {
        println!("REFLEX_GENERATE_KV_EXPORT_OK path={path:?} kind={kind} seq_len={seq_len} num_layers={num_layers}");
        return;
    }
    #[cfg(feature = "json-output")]
    reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::GenerateKvExportJson {
        path: path.to_string(),
        kind,
        seq_len,
        num_layers,
    });
    #[cfg(not(feature = "json-output"))]
    json_output_unavailable();
}

/// Resolves `--model`/`--quickstart` to a local GGUF path, or returns `None` if
/// neither was passed (the caller falls back to the positional `<path-to-gguf>`
/// argument in that case). Panics with a clear "rebuild with --features download"
/// message if either flag is used on a binary built without that feature, rather
/// than failing to compile at all (the flags themselves always parse).
fn resolve_model_flag(model_spec: Option<&str>, quickstart: bool) -> Option<String> {
    if !quickstart && model_spec.is_none() {
        return None;
    }
    #[cfg(feature = "download")]
    {
        let resolved = if quickstart {
            reflex_engine::hf::resolve_quickstart()
        } else {
            reflex_engine::hf::resolve_gguf_path(model_spec.unwrap())
        };
        Some(
            resolved
                .unwrap_or_else(|e| panic!("{e}"))
                .to_string_lossy()
                .into_owned(),
        )
    }
    #[cfg(not(feature = "download"))]
    {
        let _ = (model_spec, quickstart);
        panic!("--model/--quickstart require this binary to be built with `cargo build --features download`");
    }
}

pub fn run(args: Vec<String>) {
    let t0 = Instant::now();
    let sampler = energy::EnergySampler::start(0);

    let mut positional: Vec<String> = Vec::new();
    let mut export_kv: Option<String> = None;
    let mut import_kv: Option<String> = None;
    let mut max_tokens: usize = 1;
    let mut lora_path: Option<String> = None;
    let mut model_spec: Option<String> = None;
    let mut quickstart = false;
    let mut temperature: f32 = 0.0;
    let mut top_k: Option<usize> = None;
    let mut top_p: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut json = false;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--export-kv" => {
                export_kv = Some(args.next().expect("--export-kv requires a file path"))
            }
            "--import-kv" => {
                import_kv = Some(args.next().expect("--import-kv requires a file path"))
            }
            "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
            "--model" => {
                model_spec = Some(
                    args.next()
                        .expect("--model requires a repo spec, e.g. org/repo:file.gguf"),
                )
            }
            "--quickstart" => quickstart = true,
            "--max-tokens" => {
                let raw = args.next().expect("--max-tokens requires a number");
                max_tokens = raw.parse().unwrap_or_else(|_| {
                    panic!("--max-tokens must be a positive integer, got {raw:?}")
                });
            }
            "--temperature" => {
                let raw = args.next().expect("--temperature requires a number");
                temperature = raw
                    .parse()
                    .unwrap_or_else(|_| panic!("--temperature must be a number, got {raw:?}"));
            }
            "--top-k" => {
                let raw = args.next().expect("--top-k requires a number");
                top_k =
                    Some(raw.parse().unwrap_or_else(|_| {
                        panic!("--top-k must be a positive integer, got {raw:?}")
                    }));
            }
            "--top-p" => {
                let raw = args.next().expect("--top-p requires a number");
                top_p = Some(
                    raw.parse()
                        .unwrap_or_else(|_| panic!("--top-p must be a number, got {raw:?}")),
                );
            }
            "--seed" => {
                let raw = args.next().expect("--seed requires a number");
                seed = Some(raw.parse().unwrap_or_else(|_| {
                    panic!("--seed must be a non-negative integer, got {raw:?}")
                }));
            }
            other => positional.push(other.to_string()),
        }
    }
    let sampling = reflex_engine::sampling::SamplingParams {
        temperature,
        top_k,
        top_p,
        seed,
    };
    if quickstart && model_spec.is_some() {
        panic!("--quickstart and --model cannot be combined in the same run");
    }
    let mut positional = positional.into_iter();
    let (gguf_path, prompt) = match resolve_model_flag(model_spec.as_deref(), quickstart) {
        Some(resolved_path) => (resolved_path, positional.next()),
        None => {
            let gguf_path = positional.next().unwrap_or_else(|| {
                panic!(
                    "usage: reflex generate <path-to-gguf> [prompt] [--max-tokens N] [--export-kv <file>] [--lora <adapter.gguf>] | \
                     reflex generate --model <org/repo:file.gguf> [prompt] | reflex generate --quickstart [prompt]"
                )
            });
            (gguf_path, positional.next())
        }
    };
    let prompt = prompt.unwrap_or_else(|| "Once upon a time".to_string());
    if max_tokens == 0 {
        panic!("--max-tokens must be at least 1");
    }
    if export_kv.is_some() && import_kv.is_some() {
        panic!("--export-kv and --import-kv cannot be combined in the same run");
    }
    if export_kv.is_some() && max_tokens != 1 {
        panic!("--export-kv only captures the cache after the initial prompt pass -- omit --max-tokens (defaults to 1) when exporting");
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

    if let Some(export_path) = &export_kv {
        let (token_id, text) = match model.architecture_kind() {
            ArchitectureKind::Hybrid => {
                let ((token_id, text), cache) = model
                    .forward_prompt_capture_kv_hybrid(&prompt)
                    .unwrap_or_else(|e| crate::fail("forward_prompt_capture_kv_hybrid failed", e));
                kv_io::export_hybrid_kv(export_path, &cache)
                    .expect("failed to export hybrid KV cache");
                print_kv_export_ok(
                    json,
                    export_path,
                    "hybrid",
                    cache.seq_len,
                    cache.layers.len(),
                );
                (token_id, text)
            }
            ArchitectureKind::Dense => {
                let ((token_id, text), cache) = model
                    .forward_prompt_capture_kv(&prompt)
                    .unwrap_or_else(|e| crate::fail("forward_prompt_capture_kv failed", e));
                print_kv_export_ok(
                    json,
                    export_path,
                    "dense",
                    cache.seq_len,
                    cache.k_caches.len(),
                );
                kv_io::export_dense_kv(export_path, &cache).expect("failed to export KV cache");
                (token_id, text)
            }
            ArchitectureKind::Mla => {
                let ((token_id, text), cache) = model
                    .forward_prompt_capture_kv_mla(&prompt)
                    .unwrap_or_else(|e| crate::fail("forward_prompt_capture_kv_mla failed", e));
                print_kv_export_ok(
                    json,
                    export_path,
                    "mla",
                    cache.seq_len,
                    cache.kv_caches.len(),
                );
                kv_io::export_mla_kv(export_path, &cache).expect("failed to export MLA KV cache");
                (token_id, text)
            }
        };
        let elapsed = t0.elapsed();
        let energy_measurement = sampler.measure();
        let prompt_eval_ms = elapsed.as_secs_f64() * 1000.0 - model_ready_ms;
        let process_start_to_first_token_ms = elapsed.as_secs_f64() * 1000.0;
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
            reflex_engine::cli_output::print_json_line(
                &reflex_engine::cli_output::GenerateResultJson {
                    schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
                    process_start_to_first_token_ms,
                    process_start_to_last_token_ms: None,
                    gguf_open_ms,
                    cuda_init_ms,
                    model_load_ms,
                    prompt_eval_ms,
                    num_generated: None,
                    token_id,
                    token_ids: None,
                    token_text: text,
                    joules: energy_measurement.as_ref().map(|m| m.joules),
                    energy_method: energy_measurement.as_ref().map(|m| m.method.as_str()),
                },
            );
            #[cfg(not(feature = "json-output"))]
            json_output_unavailable();
        } else {
            println!(
                "REFLEX_GENERATE_OK process_start_to_first_token_ms={process_start_to_first_token_ms:.3} gguf_open_ms={gguf_open_ms:.3} cuda_init_ms={cuda_init_ms:.3} model_load_ms={model_load_ms:.3} prompt_eval_ms={prompt_eval_ms:.3} token_id={token_id} token_text={text:?}{}",
                energy_suffix(energy_measurement.as_ref()),
            );
        }
        reflex_engine::fast_exit(0);
    }

    let imported = import_kv.as_ref().map(|path| {
        kv_io::import_kv(path).unwrap_or_else(|e| crate::fail("failed to import KV cache", e))
    });

    let mut first_token_ms: Option<f64> = None;
    let mut first_token_energy: Option<energy::EnergyMeasurement> = None;
    let (tokens, text) = model
        .generate(
            &prompt,
            max_tokens,
            imported.as_ref(),
            &sampling,
            |_logits| {
                first_token_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                first_token_energy = sampler.measure();
            },
            |_id, _text| {},
        )
        .unwrap_or_else(|e| crate::fail("generate failed", e));
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let energy_measurement = sampler.measure();
    let prompt_eval_ms = first_token_ms.unwrap_or(total_ms) - model_ready_ms;

    print_phase_report(
        json,
        gguf_open_ms,
        cuda_init_ms,
        model_load_ms,
        prompt_eval_ms,
        &e_gguf_open,
        &e_cuda_init,
        &e_model_load,
        &first_token_energy,
    );

    if json {
        #[cfg(feature = "json-output")]
        reflex_engine::cli_output::print_json_line(
            &reflex_engine::cli_output::GenerateResultJson {
                schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
                process_start_to_first_token_ms: first_token_ms.unwrap_or(total_ms),
                process_start_to_last_token_ms: Some(total_ms),
                gguf_open_ms,
                cuda_init_ms,
                model_load_ms,
                prompt_eval_ms,
                num_generated: Some(tokens.len()),
                token_id: tokens[0],
                token_ids: Some(tokens.clone()),
                token_text: text,
                joules: energy_measurement.as_ref().map(|m| m.joules),
                energy_method: energy_measurement.as_ref().map(|m| m.method.as_str()),
            },
        );
        #[cfg(not(feature = "json-output"))]
        json_output_unavailable();
    } else {
        let token_ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        println!(
            "REFLEX_GENERATE_OK process_start_to_first_token_ms={:.3} process_start_to_last_token_ms={:.3} gguf_open_ms={gguf_open_ms:.3} cuda_init_ms={cuda_init_ms:.3} model_load_ms={model_load_ms:.3} prompt_eval_ms={prompt_eval_ms:.3} num_generated={} token_id={} token_ids=[{}] token_text={text:?}{}",
            first_token_ms.unwrap_or(total_ms),
            total_ms,
            tokens.len(),
            tokens[0],
            token_ids.join(","),
            energy_suffix(energy_measurement.as_ref()),
        );
    }
    reflex_engine::fast_exit(0);
}
