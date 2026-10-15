//! GPU tests for IPC prefix reuse (`import_kv`/`export_kv`), driven through
//! [`run_request_loop`] so the wire protocol is covered end to end. Each test
//! checks that resuming from exported files gives the same greedy ids as one
//! uninterrupted request over the joined text. The prompts split where the
//! tokenizer splits the joined text anyway; `assert_split_tokenizes_alike`
//! checks that before comparing, so a failure is about the cache, not the
//! tokenizer.

use super::*;
use crate::gguf::GgufFile;
use crate::model::{LoadOptions, WeightsDtype};
use cudarc::driver::CudaDevice;
use serde_json::Value;

const QWEN3MOE_FIXTURE: &str = "test-data/tiny-qwen3moe.gguf";
const QWEN35MOE_FIXTURE: &str = "test-data/tiny-qwen35moe.gguf";
const MLA_FIXTURE: &str = "test-data/deepseek-tiny-mla.gguf";
const GRANITE_FIXTURE: &str = "test-data/tiny-granitehybrid.gguf";

/// A shared system prompt, an agent's task, and a follow-up turn: the three
/// pieces an orchestrator would cache one after another.
const PARTS: [&str; 3] = [
    "You are a careful assistant in a team of agents. Answer briefly.",
    " The capital of France is",
    " Paris, and the capital of Italy is",
];

fn load(path: &str, weights: WeightsDtype) -> Model {
    let file = GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("load model")
}

/// Sends `requests` through one [`run_request_loop`] session and returns every
/// line it wrote, parsed.
fn session(model: &Model, requests: &[Value]) -> Vec<Value> {
    let input: String = requests.iter().map(|r| format!("{r}\n")).collect();
    let mut out = Vec::new();
    run_request_loop(model, input.as_bytes(), &mut out).expect("request loop");
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).expect("response line is JSON"))
        .collect()
}

fn final_ok(resp: &Value) -> Vec<u64> {
    assert_eq!(resp["event"], "final", "{resp}");
    assert_eq!(resp["ok"], true, "{resp}");
    resp["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}

fn assert_split_tokenizes_alike(model: &Model, parts: &[&str]) {
    let split: usize = parts
        .iter()
        .map(|p| model.encoded_prompt_len(p).unwrap())
        .sum();
    let joined = model.encoded_prompt_len(&parts.concat()).unwrap();
    assert_eq!(
        split, joined,
        "test prompts must split on a token boundary for this tokenizer"
    );
}

fn tmp_path(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "reflex_ipc_relay_{tag}_{}.cskv",
            std::process::id()
        ))
        .to_str()
        .unwrap()
        .to_string()
}

/// Exports `PARTS[0]`, imports it to run `PARTS[1]` while exporting the
/// combined state, then imports that to generate after `PARTS[2]` (streamed,
/// with timings). The result must match one request over all three parts.
fn assert_relay_resumes_exactly(model: &Model, tag: &str) {
    const N: usize = 12;
    assert_split_tokenizes_alike(model, &PARTS);
    let sys = tmp_path(&format!("{tag}_sys"));
    let task = tmp_path(&format!("{tag}_task"));
    let lines = session(
        model,
        &[
            serde_json::json!({"prompt": PARTS[0], "export_kv": sys, "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[1], "import_kv": sys, "export_kv": task,
                               "max_tokens": 1}),
            serde_json::json!({"id": "resume", "prompt": PARTS[2], "import_kv": task,
                               "max_tokens": N, "stream": true, "timings": true}),
            serde_json::json!({"id": "joined", "prompt": PARTS.concat(), "max_tokens": N}),
        ],
    );
    std::fs::remove_file(&sys).ok();
    std::fs::remove_file(&task).ok();

    final_ok(&lines[0]);
    final_ok(&lines[1]);
    let finals: Vec<&Value> = lines.iter().filter(|l| l["event"] == "final").collect();
    assert_eq!(finals.len(), 4, "{lines:?}");
    let resumed = final_ok(finals[2]);
    let joined = final_ok(finals[3]);
    let streamed = lines.iter().filter(|l| l["event"] == "token").count();
    assert_eq!(
        streamed,
        resumed.len(),
        "one token line per generated token"
    );
    let timings = &finals[2]["timings"];
    assert!(timings["kv_io_ms"].as_f64().unwrap() >= 0.0, "{timings}");
    eprintln!("{tag}: resumed={resumed:?} timings={timings}");
    assert_eq!(
        resumed, joined,
        "{tag}: resumed ids differ from one joined run"
    );
}

/// Real dense Qwen3 (`REFLEX_TEST_GGUF`, the nightly's `DENSE_GGUF`), default
/// weights as `reflex stdio` loads them. Also covers the request errors.
#[test]
#[ignore]
fn ipc_kv_relay_dense_resumes_exactly() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_relay_resumes_exactly(&model, "dense");

    let out = tmp_path("dense_bad");
    let lines = session(
        &model,
        &[
            serde_json::json!({"prompt": "p", "export_kv": out}),
            serde_json::json!({"prompt": "p", "candidates": ["a"], "import_kv": out}),
            serde_json::json!({"prompt": "p", "import_kv": tmp_path("does_not_exist")}),
        ],
    );
    assert!(
        !std::path::Path::new(&out).exists(),
        "rejected export wrote a file"
    );
    assert_eq!(lines[0]["error_kind"], "invalid_input", "{}", lines[0]);
    assert_eq!(lines[1]["error_kind"], "invalid_input", "{}", lines[1]);
    assert_eq!(lines[2]["ok"], false, "{}", lines[2]);
}

#[test]
#[ignore]
fn qwen3moe_fixture_ipc_kv_relay_resumes_exactly() {
    let model = load(QWEN3MOE_FIXTURE, WeightsDtype::F32);
    assert_relay_resumes_exactly(&model, "qwen3moe");
}

/// The Qwen3.5 hybrid layout (attention K/V plus GDN recurrent state).
#[test]
#[ignore]
fn qwen35moe_fixture_ipc_kv_relay_resumes_exactly() {
    let model = load(QWEN35MOE_FIXTURE, WeightsDtype::F32);
    assert_relay_resumes_exactly(&model, "qwen35moe");
}

#[test]
#[ignore]
fn ipc_kv_relay_mla_fixture_resumes_exactly() {
    let model = load(MLA_FIXTURE, WeightsDtype::F32);
    assert_relay_resumes_exactly(&model, "mla");
}

/// Granite has no cache file format yet: the export must fail cleanly, and
/// the session must keep serving.
#[test]
#[ignore]
fn ipc_kv_relay_granite_fixture_rejects_export() {
    let model = load(GRANITE_FIXTURE, WeightsDtype::F32);
    let out = tmp_path("granite");
    let lines = session(
        &model,
        &[
            serde_json::json!({"prompt": "Hello", "export_kv": out, "max_tokens": 1}),
            serde_json::json!({"prompt": "Hello", "max_tokens": 2}),
        ],
    );
    assert!(!std::path::Path::new(&out).exists());
    assert_eq!(
        lines[0]["error_kind"], "unsupported_architecture",
        "{}",
        lines[0]
    );
    final_ok(&lines[1]);
}
