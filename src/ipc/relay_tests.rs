//! GPU tests for IPC prefix reuse (`import_kv`/`export_kv` files and the
//! `pin_prefix`/`use_pinned` VRAM slot), driven through [`run_request_loop`] so the
//! wire protocol is covered end to end. Each test checks that resuming from a
//! cached prefix gives the same greedy ids as one uninterrupted request over the
//! joined text. The prompts split where the
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
const GEMMA4_FIXTURE: &str = "test-data/tiny-gemma4.gguf";
const QWEN2_FIXTURE: &str = "test-data/tiny-qwen2.gguf";
const KOLIBRI1_FIXTURE: &str = "test-data/tiny-kolibri1-f32.gguf";

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
    session_with(model, &mut PrefixSlot::new(), requests)
}

/// [`session`] with a caller-owned pinned-prefix slot, which outlives the call
/// the way `reflex uds` keeps one across connections.
fn session_with(model: &Model, slot: &mut PrefixSlot, requests: &[Value]) -> Vec<Value> {
    let input: String = requests.iter().map(|r| format!("{r}\n")).collect();
    let mut out = Vec::new();
    run_request_loop(model, slot, input.as_bytes(), &mut out).expect("request loop");
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

/// Checks that a chain of requests resuming part by part encodes `parts` to the
/// same ids as one request over the joined text. Ids, not counts: on a
/// SentencePiece model a continuation encoded with a stray dummy-prefix space
/// (`▁<` where the joined text has `<`) can have the same length.
fn assert_split_tokenizes_alike(model: &Model, parts: &[&str]) {
    let split: Vec<u32> = model.encode_split(parts).unwrap().concat();
    assert_eq!(
        split,
        model.encode_prompt(&parts.concat()).unwrap(),
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

/// Pins `PARTS[0]`, layers `PARTS[1]` on top of it (`use_pinned` plus
/// `pin_prefix`), then generates after `PARTS[2]` from the slot three times.
/// Every use must match one request over the joined text and the same chain
/// through files, and the slot must be unchanged afterwards, since each use
/// copies it. Re-pinning `PARTS[0]` alone must replace the slot.
fn assert_pinned_resumes_exactly(model: &Model, tag: &str) {
    const N: usize = 12;
    assert_split_tokenizes_alike(model, &PARTS);
    let rest = [PARTS[1], PARTS[2]].concat();
    assert_split_tokenizes_alike(model, &[PARTS[0], &rest]);
    let mut slot = PrefixSlot::new();

    let lines = session_with(
        model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[0], "pin_prefix": true, "max_tokens": 1,
                             "timings": true}),
        ],
    );
    final_ok(&lines[0]);
    assert!(
        lines[0]["timings"].get("kv_io_ms").is_none(),
        "{}",
        lines[0]
    );
    let sys_len = slot.seq_len().expect("PARTS[0] pinned");

    let lines = session_with(
        model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[1], "use_pinned": true, "pin_prefix": true,
                             "max_tokens": 1, "timings": true}),
        ],
    );
    final_ok(&lines[0]);
    assert!(
        lines[0]["timings"]["kv_io_ms"].as_f64().unwrap() >= 0.0,
        "{}",
        lines[0]
    );
    let task_len = slot.seq_len().unwrap();
    assert_eq!(
        task_len,
        sys_len + model.encode_split(&PARTS[..2]).unwrap()[1].len(),
        "{tag}: layered pin holds both parts"
    );
    let bytes = slot.device_bytes();

    let use_req = |id: &str, stream: bool| {
        serde_json::json!({"id": id, "prompt": PARTS[2], "use_pinned": true, "max_tokens": N,
                           "stream": stream, "timings": true})
    };
    let lines = session_with(
        model,
        &mut slot,
        &[
            use_req("pin1", true),
            use_req("pin2", false),
            use_req("pin3", false),
        ],
    );
    assert_eq!(
        slot.seq_len(),
        Some(task_len),
        "{tag}: a use changed the slot"
    );
    assert_eq!(slot.device_bytes(), bytes);
    let finals: Vec<&Value> = lines.iter().filter(|l| l["event"] == "final").collect();
    assert_eq!(finals.len(), 3, "{lines:?}");
    let uses: Vec<Vec<u64>> = finals.iter().map(|f| final_ok(f)).collect();
    let streamed = lines.iter().filter(|l| l["event"] == "token").count();
    assert_eq!(
        streamed,
        uses[0].len(),
        "one token line per generated token"
    );
    for f in &finals {
        assert!(f["timings"]["kv_io_ms"].as_f64().unwrap() >= 0.0, "{f}");
    }

    let sys = tmp_path(&format!("{tag}_pin_sys"));
    let task = tmp_path(&format!("{tag}_pin_task"));
    let lines = session(
        model,
        &[
            serde_json::json!({"prompt": PARTS[0], "export_kv": sys, "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[1], "import_kv": sys, "export_kv": task,
                               "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[2], "import_kv": task, "max_tokens": N}),
            serde_json::json!({"prompt": PARTS.concat(), "max_tokens": N}),
        ],
    );
    std::fs::remove_file(&sys).ok();
    std::fs::remove_file(&task).ok();
    let from_file = final_ok(&lines[2]);
    let joined = final_ok(&lines[3]);
    eprintln!(
        "{tag}: pinned={:?} file={from_file:?} joined={joined:?} slot={bytes} B copy={}",
        uses[0], finals[1]["timings"]["kv_io_ms"]
    );
    for (i, u) in uses.iter().enumerate() {
        assert_eq!(
            u, &joined,
            "{tag}: pinned use {i} differs from one joined run"
        );
    }
    assert_eq!(
        from_file, joined,
        "{tag}: file resume differs from one joined run"
    );

    let lines = session_with(
        model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[0], "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": rest, "use_pinned": true, "max_tokens": N}),
        ],
    );
    final_ok(&lines[0]);
    assert_eq!(
        slot.seq_len(),
        Some(sys_len),
        "{tag}: re-pin didn't replace the slot"
    );
    assert_eq!(
        final_ok(&lines[1]),
        joined,
        "{tag}: resume from the re-pinned slot differs from one joined run"
    );
}

/// Pinning and using the slot must fail with `unsupported_architecture` on a
/// model without a prefix-state format, and the session must keep serving.
fn assert_pinning_rejected(model: &Model) {
    let mut slot = PrefixSlot::new();
    let lines = session_with(
        model,
        &mut slot,
        &[
            serde_json::json!({"prompt": "Hello", "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": "Hello", "use_pinned": true, "max_tokens": 2}),
            serde_json::json!({"prompt": "Hello", "max_tokens": 2}),
        ],
    );
    for l in &lines[..2] {
        assert_eq!(l["error_kind"], "unsupported_architecture", "{l}");
    }
    final_ok(&lines[2]);
    assert_eq!(slot.seq_len(), None);
}

/// Real dense Qwen3 (`REFLEX_TEST_GGUF`). Also covers the request errors, each
/// of which must leave the slot as it was.
#[test]
#[ignore]
fn ipc_pinned_prefix_dense_resumes_exactly() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_pinned_resumes_exactly(&model, "dense");

    let mut slot = PrefixSlot::new();
    let out = tmp_path("dense_pin_bad");
    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": "p", "use_pinned": true}),
            serde_json::json!({"prompt": "p", "pin_prefix": true, "max_tokens": 2}),
        ],
    );
    for l in &lines {
        assert_eq!(l["error_kind"], "invalid_input", "{l}");
    }
    assert_eq!(slot.seq_len(), None);

    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[0], "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": "p", "pin_prefix": true, "export_kv": out,
                               "max_tokens": 1}),
            serde_json::json!({"prompt": "p", "use_pinned": true, "import_kv": out}),
            serde_json::json!({"prompt": "p", "candidates": ["a"], "use_pinned": true}),
            serde_json::json!({"prompt": "p", "candidates": ["a"], "pin_prefix": true}),
            serde_json::json!({"prompt": "p", "use_pinned": true, "pin_prefix": true,
                               "max_tokens": 4}),
            serde_json::json!({"prompt": PARTS[1], "use_pinned": true, "max_tokens": 2}),
        ],
    );
    final_ok(&lines[0]);
    for l in &lines[1..6] {
        assert_eq!(l["error_kind"], "invalid_input", "{l}");
    }
    final_ok(&lines[6]);
    assert!(
        !std::path::Path::new(&out).exists(),
        "rejected export wrote a file"
    );
    assert_eq!(
        slot.seq_len(),
        Some(model.encoded_prompt_len(PARTS[0]).unwrap()),
        "a rejected request changed the slot"
    );
}

/// A use whose cache can't be allocated must fail with `out_of_memory`, keep
/// the slot, and leave the session working once memory is free again.
#[test]
#[ignore]
fn ipc_pinned_prefix_dense_fork_oom_fails_cleanly() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    let mut slot = PrefixSlot::new();
    let long = PARTS[0].repeat(40);
    let lines = session_with(
        &model,
        &mut slot,
        &[serde_json::json!({"prompt": long, "pin_prefix": true, "max_tokens": 1})],
    );
    final_ok(&lines[0]);
    let pinned_len = slot.seq_len();
    let need = slot.device_bytes();

    // Fill VRAM until less than half the slot's size is free: the use's own
    // cache, at least as large as the slot, then can't be allocated.
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let target = need / 2;
    let mut filler = Vec::new();
    loop {
        let (free, _) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
        if free < target {
            break;
        }
        let chunk = if free > target + (64 << 20) {
            64 << 20
        } else {
            1 << 20
        };
        match dev.alloc_zeros::<u8>(chunk) {
            Ok(buf) => filler.push(buf),
            Err(_) => break,
        }
    }
    let use_req = serde_json::json!({"prompt": PARTS[1], "use_pinned": true, "max_tokens": 4});
    let lines = session_with(&model, &mut slot, std::slice::from_ref(&use_req));
    assert_eq!(lines[0]["error_kind"], "out_of_memory", "{}", lines[0]);

    drop(filler);
    let lines = session_with(&model, &mut slot, &[use_req]);
    final_ok(&lines[0]);
    assert_eq!(slot.seq_len(), pinned_len);
}

#[test]
#[ignore]
fn qwen3moe_fixture_ipc_pinned_prefix_resumes_exactly() {
    let model = load(QWEN3MOE_FIXTURE, WeightsDtype::F32);
    assert_pinned_resumes_exactly(&model, "qwen3moe");
}

/// The Qwen3.5 hybrid: attention K/V plus GDN recurrent and conv state.
#[test]
#[ignore]
fn qwen35moe_fixture_ipc_pinned_prefix_resumes_exactly() {
    let model = load(QWEN35MOE_FIXTURE, WeightsDtype::F32);
    assert_pinned_resumes_exactly(&model, "qwen35moe");
}

#[test]
#[ignore]
fn ipc_pinned_prefix_mla_fixture_resumes_exactly() {
    let model = load(MLA_FIXTURE, WeightsDtype::F32);
    assert_pinned_resumes_exactly(&model, "mla");
}

/// Qwen2 adds Q/K/V biases to the dense path.
#[test]
#[ignore]
fn ipc_pinned_prefix_qwen2_fixture_resumes_exactly() {
    let model = load(QWEN2_FIXTURE, WeightsDtype::F32);
    assert_pinned_resumes_exactly(&model, "qwen2");
}

/// Kolibri-1's stacked experts on the dense/MoE path.
#[test]
#[ignore]
fn ipc_pinned_prefix_kolibri1_fixture_resumes_exactly() {
    let model = load(KOLIBRI1_FIXTURE, WeightsDtype::F32);
    assert_pinned_resumes_exactly(&model, "kolibri1");
}

/// Resumes `parts[1]` after `parts[0]` through a file and through the slot; both
/// must match one request over the joined text.
fn assert_two_part_resume(model: &Model, parts: [&str; 2], tag: &str) {
    const N: usize = 12;
    assert_split_tokenizes_alike(model, &parts);
    let file = tmp_path(&format!("{tag}_two_part"));
    let lines = session(
        model,
        &[
            serde_json::json!({"prompt": parts[0], "export_kv": file, "max_tokens": 1}),
            serde_json::json!({"prompt": parts[1], "import_kv": file, "max_tokens": N}),
            serde_json::json!({"prompt": parts[0], "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": parts[1], "use_pinned": true, "max_tokens": N}),
            serde_json::json!({"prompt": parts.concat(), "max_tokens": N}),
        ],
    );
    std::fs::remove_file(&file).ok();
    final_ok(&lines[0]);
    final_ok(&lines[2]);
    let (imported, pinned, joined) = (
        final_ok(&lines[1]),
        final_ok(&lines[3]),
        final_ok(&lines[4]),
    );
    eprintln!("{tag}: joined={joined:?}");
    assert_eq!(
        imported, joined,
        "{tag}: import_kv differs from one joined request"
    );
    assert_eq!(
        pinned, joined,
        "{tag}: use_pinned differs from one joined request"
    );
}

/// A real SentencePiece chat model (`SPM_GGUF` in the nightly, e.g.
/// TinyLlama-1.1B-Chat). Its tokenizer gives a raw-text span a dummy-prefix
/// space only at the start of the text or after a special token, so a
/// continuation must get one exactly when the joined text would: not after
/// raw text (`PARTS`, a chat turn ending in a newline), but after `</s>`.
#[test]
#[ignore]
fn ipc_kv_relay_spm_model_resumes_exactly() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a SentencePiece GGUF (the nightly's SPM_GGUF)");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_relay_resumes_exactly(&model, "spm");
    assert_pinned_resumes_exactly(&model, "spm");
    let user = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    assert_two_part_resume(
        &model,
        ["<|system|>\nYou are a careful assistant.</s>\n", user],
        "spm after raw text",
    );
    assert_two_part_resume(
        &model,
        [
            "<|system|>\nYou are a careful assistant.</s>",
            &format!("\n{user}"),
        ],
        "spm after a special token",
    );
}

#[test]
#[ignore]
fn ipc_pinned_prefix_granite_fixture_rejects() {
    assert_pinning_rejected(&load(GRANITE_FIXTURE, WeightsDtype::F32));
}

#[test]
#[ignore]
fn ipc_pinned_prefix_gemma4_fixture_rejects() {
    assert_pinning_rejected(&load(GEMMA4_FIXTURE, WeightsDtype::F32));
}
