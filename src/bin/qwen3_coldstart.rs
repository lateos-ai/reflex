//! Dense/MoE/hybrid Qwen3 cold-start measurement (MVP steps 1-3, see
//! README.md's MVP order): loads a real GGUF file, runs the prompt through
//! `model::Model::generate`, and reports wall-clock time from process start
//! to the first generated token -- the project's actual target metric.
//!
//! Also the CLI entry point for Phase 3 (State I/O)'s KV-cache export/
//! import: `--export-kv <file>` downloads the K/V cache produced by this
//! run's initial prompt pass and writes it to `<file>` (dense/MoE, hybrid
//! Qwen3.5, and DeepSeek-V2/V3 MLA models). `--import-kv <file>`
//! loads a previously-exported cache, uploads it as the starting state, and
//! resumes generation from it -- `prompt` is then the continuation text
//! appended after the cached positions, not a fresh prompt. `--max-tokens N`
//! (default 1) generates up to N tokens, feeding each one back in, stopping
//! early on the tokenizer's EOS.
//!
//! Usage: `qwen3_coldstart <path-to-gguf> [prompt] [--max-tokens N] [--export-kv <file>]`
//!        `qwen3_coldstart <path-to-gguf> [continuation-prompt] [--max-tokens N] --import-kv <file>`
//!
//! Phase 4 (Embeddability) round 1: `--lora <adapter.gguf>` applies a
//! llama.cpp-format LoRA adapter to the loaded model's weights once, at load
//! time, before any forward pass runs (see `model::Model::apply_lora` and
//! `lora`'s module doc comment for the file format and scope).

use coldstart_infer::gguf::GgufFile;
use coldstart_infer::kv_io;
use coldstart_infer::model::{ArchitectureKind, Model};
use cudarc::driver::CudaDevice;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    let mut gguf_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut export_kv: Option<String> = None;
    let mut import_kv: Option<String> = None;
    let mut max_tokens: usize = 1;
    let mut lora_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--export-kv" => export_kv = Some(args.next().expect("--export-kv requires a file path")),
            "--import-kv" => import_kv = Some(args.next().expect("--import-kv requires a file path")),
            "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
            "--max-tokens" => {
                let raw = args.next().expect("--max-tokens requires a number");
                max_tokens = raw.parse().unwrap_or_else(|_| panic!("--max-tokens must be a positive integer, got {raw:?}"));
            }
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            _ if prompt.is_none() => prompt = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| {
        panic!(
            "usage: qwen3_coldstart <path-to-gguf> [prompt] [--max-tokens N] [--export-kv <file>] [--lora <adapter.gguf>] | \
             qwen3_coldstart <path-to-gguf> [continuation-prompt] [--max-tokens N] --import-kv <file> [--lora <adapter.gguf>]"
        )
    });
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

    let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let mut model = Model::load(device, &file).expect("failed to load model");

    if let Some(lora_path) = &lora_path {
        let applied = model.apply_lora(std::path::Path::new(lora_path)).expect("failed to apply LoRA adapter");
        println!("COLDSTART_QWEN3_LORA_OK path={lora_path:?} tensors_applied={applied}");
    }

    if let Some(export_path) = &export_kv {
        let (token_id, text) = match model.architecture_kind() {
            ArchitectureKind::Hybrid => {
                let ((token_id, text), cache) = model.forward_prompt_capture_kv_hybrid(&prompt).expect("forward_prompt_capture_kv_hybrid failed");
                kv_io::export_hybrid_kv(export_path, &cache).expect("failed to export hybrid KV cache");
                println!(
                    "COLDSTART_QWEN3_KV_EXPORT_OK path={export_path:?} kind=hybrid seq_len={} num_layers={}",
                    cache.seq_len,
                    cache.layers.len()
                );
                (token_id, text)
            }
            ArchitectureKind::Dense => {
                let ((token_id, text), cache) = model.forward_prompt_capture_kv(&prompt).expect("forward_prompt_capture_kv failed");
                println!(
                    "COLDSTART_QWEN3_KV_EXPORT_OK path={export_path:?} kind=dense seq_len={} num_layers={}",
                    cache.seq_len,
                    cache.k_caches.len()
                );
                kv_io::export_dense_kv(export_path, &cache).expect("failed to export KV cache");
                (token_id, text)
            }
            ArchitectureKind::Mla => {
                let ((token_id, text), cache) = model.forward_prompt_capture_kv_mla(&prompt).expect("forward_prompt_capture_kv_mla failed");
                println!(
                    "COLDSTART_QWEN3_KV_EXPORT_OK path={export_path:?} kind=mla seq_len={} num_layers={}",
                    cache.seq_len,
                    cache.kv_caches.len()
                );
                kv_io::export_mla_kv(export_path, &cache).expect("failed to export MLA KV cache");
                (token_id, text)
            }
        };
        let elapsed = t0.elapsed();
        println!(
            "COLDSTART_QWEN3_OK process_start_to_first_token_ms={:.3} token_id={token_id} token_text={text:?}",
            elapsed.as_secs_f64() * 1000.0
        );
        return;
    }

    let imported = import_kv.as_ref().map(|path| kv_io::import_kv(path).expect("failed to import KV cache"));

    let mut first_token_ms: Option<f64> = None;
    let (tokens, text) = model
        .generate(&prompt, max_tokens, imported.as_ref(), || {
            first_token_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
        })
        .expect("generate failed");
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let token_ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    println!(
        "COLDSTART_QWEN3_OK process_start_to_first_token_ms={:.3} process_start_to_last_token_ms={:.3} num_generated={} token_id={} token_ids=[{}] token_text={text:?}",
        first_token_ms.unwrap_or(total_ms),
        total_ms,
        tokens.len(),
        tokens[0],
        token_ids.join(","),
    );
}
