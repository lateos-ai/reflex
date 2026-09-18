//! Dense Qwen3 cold-start measurement (MVP step 1, see README.md's MVP
//! order): loads a real Qwen3 GGUF file, runs the prompt through
//! `model::Model::forward_prompt`, and reports wall-clock time from process
//! start to the first generated token -- the project's actual target
//! metric, now proven against a real model instead of just the smoke
//! kernel.
//!
//! Usage: `qwen3_coldstart <path-to-gguf> [prompt]`

use coldstart_infer::gguf::GgufFile;
use coldstart_infer::model::Model;
use cudarc::driver::CudaDevice;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().expect("usage: qwen3_coldstart <path-to-gguf> [prompt]");
    let prompt = args.next().unwrap_or_else(|| "Once upon a time".to_string());

    let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load model");

    let (token_id, text) = model.forward_prompt(&prompt).expect("forward_prompt failed");

    let elapsed = t0.elapsed();
    println!(
        "COLDSTART_QWEN3_OK process_start_to_first_token_ms={:.3} token_id={token_id} token_text={text:?}",
        elapsed.as_secs_f64() * 1000.0
    );
}
