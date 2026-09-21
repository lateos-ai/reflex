//! Dense Qwen3 cold-start measurement (MVP step 1, see README.md's MVP
//! order): loads a real Qwen3 GGUF file, runs the prompt through
//! `model::Model::forward_prompt`, and reports wall-clock time from process
//! start to the first generated token -- the project's actual target
//! metric, now proven against a real model instead of just the smoke
//! kernel.
//!
//! Also the CLI entry point for Phase 3 (State I/O) round 1's raw KV-cache
//! dump/load (`kv_io.rs`): `--export-kv <file>` downloads the K/V cache
//! produced by this run's forward pass and writes it to `<file>`.
//! `--import-kv <file>` loads a previously-exported file and proves the
//! bytes survive an upload-to-device-and-back round trip unchanged; it does
//! *not* resume generation from the cache (no per-token generation loop or
//! `start_pos` exists yet -- see kv_io.rs's doc comment and STATUS.md) so it
//! skips loading a GGUF/model entirely.
//!
//! Usage: `qwen3_coldstart <path-to-gguf> [prompt] [--export-kv <file>]`
//!        `qwen3_coldstart <path-to-gguf> --import-kv <file>`

use coldstart_infer::gguf::GgufFile;
use coldstart_infer::kv_io;
use coldstart_infer::model::Model;
use cudarc::driver::CudaDevice;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    let mut gguf_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut export_kv: Option<String> = None;
    let mut import_kv: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--export-kv" => export_kv = Some(args.next().expect("--export-kv requires a file path")),
            "--import-kv" => import_kv = Some(args.next().expect("--import-kv requires a file path")),
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            _ if prompt.is_none() => prompt = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| {
        panic!("usage: qwen3_coldstart <path-to-gguf> [prompt] [--export-kv <file>] | qwen3_coldstart <path-to-gguf> --import-kv <file>")
    });
    let prompt = prompt.unwrap_or_else(|| "Once upon a time".to_string());

    if let Some(import_path) = &import_kv {
        let cache = kv_io::import_dense_kv(import_path).expect("failed to import KV cache");
        let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
        kv_io::verify_roundtrip_on_device(&device, &cache).expect("KV cache device round-trip failed");

        let elapsed = t0.elapsed();
        println!(
            "COLDSTART_QWEN3_KV_IMPORT_OK path={import_path:?} seq_len={} num_layers={} num_kv_heads={} head_dim={} process_start_to_import_verified_ms={:.3}",
            cache.seq_len,
            cache.k_caches.len(),
            cache.num_kv_heads,
            cache.head_dim,
            elapsed.as_secs_f64() * 1000.0
        );
        return;
    }

    let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");

    let (token_id, text) = if let Some(export_path) = &export_kv {
        let ((token_id, text), cache) = model.forward_prompt_capture_kv(&prompt).expect("forward_prompt_capture_kv failed");
        let num_layers = cache.k_caches.len();
        let seq_len = cache.seq_len;
        kv_io::export_dense_kv(export_path, &cache).expect("failed to export KV cache");
        println!("COLDSTART_QWEN3_KV_EXPORT_OK path={export_path:?} seq_len={seq_len} num_layers={num_layers}");
        (token_id, text)
    } else {
        model.forward_prompt(&prompt).expect("forward_prompt failed")
    };

    let elapsed = t0.elapsed();
    println!(
        "COLDSTART_QWEN3_OK process_start_to_first_token_ms={:.3} token_id={token_id} token_text={text:?}",
        elapsed.as_secs_f64() * 1000.0
    );
}
