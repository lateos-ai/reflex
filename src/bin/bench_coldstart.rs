//! Warm-latency microbenchmark: loads a real GGUF file once, then measures
//! isolated forward-pass latency (no process-launch/GGUF-load/dequant cost,
//! unlike every other measurement in README.md) across a few prompt-length
//! buckets, to establish a real baseline before System1's gather-GEMV path
//! is compared against it.
//!
//! Also benchmarks `Model::system1_evaluate` (single-token candidates) the
//! same way when `--candidate` is passed at least once, so the two numbers
//! -- ordinary `forward_prompt` vs. System1 -- can be compared directly on
//! the same hardware/model/prompt-length buckets. Both pay the identical
//! sequential per-token prefill cost and differ only in the final step
//! (full-vocab GEMV+D2H vs. gather-GEMV+small D2H); the delta between the
//! two is System1's actual, measured win.
//!
//! Usage: `bench_coldstart <path-to-gguf> [--warmup N] [--iters N]
//! [--candidate <text> ...] [--lora <adapter.gguf>]`

use coldstart_infer::gguf::GgufFile;
use coldstart_infer::model::{Model, System1Candidate};
use cudarc::driver::CudaDevice;
use std::time::Instant;

/// Approximate token-count buckets this bench reports latency for. Built by
/// repeating a filler phrase rather than targeting an exact token count --
/// there's no "encode to exactly N tokens" primitive -- so the printed
/// `prompt_tokens` is the real `Model::encoded_prompt_len` measurement, not
/// this constant.
const PROMPT_WORD_COUNTS: [usize; 3] = [24, 96, 384];

fn build_prompt(word_count: usize) -> String {
    "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(word_count.div_ceil(12))
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

fn print_stats(prefix: &str, prompt_tokens: usize, warmup: usize, iters: usize, mut samples_ms: Vec<f64>) {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_ms = samples_ms[0];
    let max_ms = samples_ms[samples_ms.len() - 1];
    println!(
        "{prefix} prompt_tokens={prompt_tokens} warmup={warmup} iters={iters} \
         p50_ms={:.3} p90_ms={:.3} p99_ms={:.3} min_ms={:.3} max_ms={:.3}",
        percentile(&samples_ms, 0.50),
        percentile(&samples_ms, 0.90),
        percentile(&samples_ms, 0.99),
        min_ms,
        max_ms,
    );
}

fn main() {
    let mut gguf_path: Option<String> = None;
    let mut warmup: usize = 5;
    let mut iters: usize = 50;
    let mut candidate_texts: Vec<String> = Vec::new();
    let mut lora_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--candidate" => candidate_texts.push(args.next().expect("--candidate requires text")),
            "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
            "--warmup" => {
                let raw = args.next().expect("--warmup requires a number");
                warmup = raw.parse().unwrap_or_else(|_| panic!("--warmup must be a non-negative integer, got {raw:?}"));
            }
            "--iters" => {
                let raw = args.next().expect("--iters requires a number");
                iters = raw.parse().unwrap_or_else(|_| panic!("--iters must be a positive integer, got {raw:?}"));
            }
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path =
        gguf_path.unwrap_or_else(|| panic!("usage: bench_coldstart <path-to-gguf> [--warmup N] [--iters N] [--candidate <text> ...] [--lora <adapter.gguf>]"));
    if iters == 0 {
        panic!("--iters must be at least 1");
    }

    let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let mut model = Model::load(device, &file).expect("failed to load model");

    if let Some(lora_path) = &lora_path {
        let applied = model.apply_lora(std::path::Path::new(lora_path)).expect("failed to apply LoRA adapter");
        println!("COLDSTART_QWEN3_LORA_OK path={lora_path:?} tensors_applied={applied}");
    }

    let candidates: Vec<System1Candidate> = candidate_texts.iter().map(|text| System1Candidate { text: text.clone() }).collect();

    for &word_count in &PROMPT_WORD_COUNTS {
        let prompt = build_prompt(word_count);
        let prompt_tokens = model.encoded_prompt_len(&prompt).expect("encoded_prompt_len failed");

        for _ in 0..warmup {
            model.forward_prompt(&prompt).expect("forward_prompt failed (warmup)");
        }
        let mut samples_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.forward_prompt(&prompt).expect("forward_prompt failed");
            samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        print_stats("COLDSTART_BENCH_WARM_OK", prompt_tokens, warmup, iters, samples_ms);

        if !candidates.is_empty() {
            for _ in 0..warmup {
                model.system1_evaluate(&prompt, &candidates, 1.0).expect("system1_evaluate failed (warmup)");
            }
            let mut samples_ms = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                model.system1_evaluate(&prompt, &candidates, 1.0).expect("system1_evaluate failed");
                samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            print_stats("COLDSTART_BENCH_SYSTEM1_OK", prompt_tokens, warmup, iters, samples_ms);
        }
    }
}
