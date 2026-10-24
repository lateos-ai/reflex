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

pub(super) fn load(path: &str, weights: WeightsDtype) -> Model {
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

pub(super) fn final_ok(resp: &Value) -> Vec<u64> {
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
pub(super) fn assert_split_tokenizes_alike(model: &Model, parts: &[&str]) {
    let split: Vec<u32> = model.encode_split(parts).unwrap().concat();
    assert_eq!(
        split,
        model.encode_prompt(&parts.concat()).unwrap(),
        "test prompts must split on a token boundary for this tokenizer"
    );
}

pub(super) fn tmp_path(tag: &str) -> String {
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
            serde_json::json!({"prompt": "p", "candidates": ["a"], "export_kv": out}),
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
            serde_json::json!({"prompt": "p", "candidates": ["a"], "use_pinned": true,
                               "import_kv": out}),
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

/// How closely a resumed System-1 request must match one over the joined text.
/// The resumed prefill runs fewer rows than the joined one, so some kernels sum
/// in a different order (as in `model::prefill_batching_tests`). Token ids and
/// `chosen` must always match exactly.
#[derive(Clone, Copy)]
enum Match {
    /// f32 weights: summation order only. Measured on a T4 (2026-10-17): bit-
    /// identical on the Qwen3-MoE fixture, 1 ulp on the MLA fixture, 10 ulps
    /// (7e-8) on the Qwen3.5 hybrid fixture.
    F32,
    /// f16 weights: the GEMMs also round activations to f16, so differences
    /// accumulate. Measured: 4.9e-3 on scores near 95 (Qwen3-0.6B), 2.6e-3
    /// (TinyLlama).
    F16,
}

/// One candidate's `(token_ids, score, probability)` from a candidates response.
fn scored(resp: &Value) -> Vec<(Vec<u64>, f64, f64)> {
    assert_eq!(resp["ok"], true, "{resp}");
    resp["candidates"]
        .as_array()
        .expect("candidates")
        .iter()
        .map(|c| {
            let ids = c["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            (
                ids,
                c["score"].as_f64().unwrap(),
                c["probability"].as_f64().unwrap(),
            )
        })
        .collect()
}

/// Checks that a resumed candidates response gives the joined one's token ids,
/// `chosen`, scores and probabilities, to `how`'s precision. Returns the largest
/// score difference, for the log.
fn assert_scores_match(resumed: &Value, joined: &Value, how: Match, what: &str) -> f64 {
    let (r, j) = (scored(resumed), scored(joined));
    assert_eq!(r.len(), j.len(), "{what}");
    let (score_rel, prob_abs) = match how {
        Match::F32 => (1e-5, 1e-5),
        Match::F16 => (5e-3, 5e-3),
    };
    let mut max_diff = 0.0f64;
    for ((r_ids, r_score, r_prob), (j_ids, j_score, j_prob)) in r.iter().zip(&j) {
        assert_eq!(r_ids, j_ids, "{what}: candidate token ids differ");
        max_diff = max_diff.max((r_score - j_score).abs());
        assert!(
            (r_score - j_score).abs() <= score_rel * j_score.abs().max(1.0),
            "{what}: score {r_score} vs joined {j_score}"
        );
        assert!(
            (r_prob - j_prob).abs() <= prob_abs,
            "{what}: probability {r_prob} vs joined {j_prob}"
        );
    }
    assert_eq!(resumed["chosen"], joined["chosen"], "{what}: chosen");
    max_diff
}

/// Pins `parts[0]` and exports it to a file, then scores `candidates` after
/// `parts[1]` from the slot, from the file, and from the slot again. Each must
/// match one candidates request over the joined text (see [`Match`]), report
/// the prefix copy or read in `kv_io_ms` and the slot in `pinned_seq_len`/
/// `pinned_bytes`, and leave the slot unchanged.
fn assert_system1_resumes(
    model: &Model,
    parts: [&str; 2],
    candidates: &[&str],
    how: Match,
    tag: &str,
) {
    assert_split_tokenizes_alike(model, &parts);
    let file = tmp_path(&format!("{tag}_s1").replace(' ', "_"));
    let mut slot = PrefixSlot::new();
    let lines = session_with(
        model,
        &mut slot,
        &[
            serde_json::json!({"prompt": parts[0], "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": parts[0], "export_kv": file, "max_tokens": 1}),
        ],
    );
    final_ok(&lines[0]);
    final_ok(&lines[1]);
    let pinned_len = slot.seq_len().expect("parts[0] pinned");
    let pinned_bytes = slot.device_bytes();

    let resumed = |field: &str, value: Value| {
        let mut req = serde_json::json!({"prompt": parts[1], "candidates": candidates,
                                         "timings": true});
        req[field] = value;
        req
    };
    let lines = session_with(
        model,
        &mut slot,
        &[
            resumed("use_pinned", Value::Bool(true)),
            resumed("import_kv", Value::String(file.clone())),
            resumed("use_pinned", Value::Bool(true)),
            serde_json::json!({"prompt": parts.concat(), "candidates": candidates,
                               "timings": true}),
        ],
    );
    std::fs::remove_file(&file).ok();
    let (pinned, from_file, pinned_again, joined) = (&lines[0], &lines[1], &lines[2], &lines[3]);
    assert_eq!(
        slot.seq_len(),
        Some(pinned_len),
        "{tag}: a scoring request changed the slot"
    );
    assert_eq!(slot.device_bytes(), pinned_bytes);
    for resp in [pinned, from_file, pinned_again, joined] {
        let t = &resp["timings"];
        assert_eq!(t["pinned_seq_len"].as_u64(), Some(pinned_len as u64), "{t}");
        assert_eq!(t["pinned_bytes"].as_u64(), Some(pinned_bytes as u64), "{t}");
        assert!(resp["chosen"].as_u64().is_some(), "{resp}");
    }
    for resp in [pinned, from_file, pinned_again] {
        assert!(
            resp["timings"]["kv_io_ms"].as_f64().unwrap() >= 0.0,
            "{resp}"
        );
    }
    assert!(joined["timings"].get("kv_io_ms").is_none(), "{joined}");

    let d_pin = assert_scores_match(pinned, joined, how, &format!("{tag} use_pinned"));
    let d_file = assert_scores_match(from_file, joined, how, &format!("{tag} import_kv"));
    assert_scores_match(
        pinned_again,
        joined,
        how,
        &format!("{tag} second use_pinned"),
    );
    eprintln!(
        "{tag}: chosen={} max |score diff| pinned={d_pin:e} file={d_file:e} copy={} read={} joined={}",
        joined["chosen"],
        pinned["timings"]["kv_io_ms"],
        from_file["timings"]["kv_io_ms"],
        joined["candidates"]
    );
}

/// Candidates spanning one token and several, for the paths that score
/// multi-token candidates by teacher forcing after the prompt.
const MULTI: [&str; 4] = [" Paris", " Rome", " Paris, and the capital", " the"];

/// Real dense Qwen3 (`REFLEX_TEST_GGUF`), f16 weights. Also covers the request
/// errors, none of which may change the slot.
#[test]
#[ignore]
fn ipc_system1_resumed_dense_matches_joined() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_system1_resumes(&model, [PARTS[0], PARTS[1]], &MULTI, Match::F16, "dense");

    let mut slot = PrefixSlot::new();
    let missing = tmp_path("dense_s1_missing");
    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris"], "use_pinned": true}),
            serde_json::json!({"prompt": PARTS[0], "pin_prefix": true, "max_tokens": 1}),
        ],
    );
    assert_eq!(lines[0]["error_kind"], "invalid_input", "{}", lines[0]);
    final_ok(&lines[1]);
    let pinned_len = slot.seq_len();

    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris"], "use_pinned": true,
                               "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris"], "use_pinned": true,
                               "export_kv": missing}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris"], "use_pinned": true,
                               "import_kv": missing}),
            serde_json::json!({"prompt": "", "candidates": [" Paris"], "use_pinned": true}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris"],
                               "import_kv": missing}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" Paris", " Rome"],
                               "use_pinned": true}),
        ],
    );
    for l in &lines[..4] {
        assert_eq!(l["error_kind"], "invalid_input", "{l}");
    }
    assert_eq!(lines[4]["ok"], false, "{}", lines[4]);
    assert!(
        !std::path::Path::new(&missing).exists(),
        "rejected export wrote a file"
    );
    assert_eq!(lines[5]["ok"], true, "{}", lines[5]);
    assert!(lines[5]["chosen"].as_u64().unwrap() < 2, "{}", lines[5]);
    assert_eq!(
        slot.seq_len(),
        pinned_len,
        "a scoring or rejected request changed the slot"
    );
}

#[test]
#[ignore]
fn qwen3moe_fixture_ipc_system1_resumed_matches_joined() {
    let model = load(QWEN3MOE_FIXTURE, WeightsDtype::F32);
    assert_system1_resumes(&model, [PARTS[0], PARTS[1]], &MULTI, Match::F32, "qwen3moe");
}

#[test]
#[ignore]
fn ipc_system1_resumed_mla_fixture_matches_joined() {
    let model = load(MLA_FIXTURE, WeightsDtype::F32);
    assert_system1_resumes(&model, [PARTS[0], PARTS[1]], &MULTI, Match::F32, "mla");
}

/// Qwen3.5 hybrid: single-token candidates after a resumed prefix, and a
/// multi-token one rejected as it is without a prefix.
#[test]
#[ignore]
fn qwen35moe_fixture_ipc_system1_resumed_single_token() {
    let model = load(QWEN35MOE_FIXTURE, WeightsDtype::F32);
    let single = [" the", " a", " and"];
    assert_system1_resumes(
        &model,
        [PARTS[0], PARTS[1]],
        &single,
        Match::F32,
        "qwen35moe",
    );

    let mut slot = PrefixSlot::new();
    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": PARTS[0], "pin_prefix": true, "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" the", " Paris, and Rome"],
                               "use_pinned": true}),
            serde_json::json!({"prompt": PARTS[1], "candidates": [" the", " Paris, and Rome"]}),
        ],
    );
    final_ok(&lines[0]);
    for l in &lines[1..] {
        assert_eq!(l["error_kind"], "invalid_input", "{l}");
    }
}

/// A real SentencePiece model (`SPM_GGUF` in the nightly): the continuation and
/// the candidates must be encoded with the dummy-prefix rule a joined request
/// applies, after raw text and after a special token.
#[test]
#[ignore]
fn ipc_system1_resumed_spm_model_matches_joined() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a SentencePiece GGUF (the nightly's SPM_GGUF)");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_system1_resumes(&model, [PARTS[0], PARTS[1]], &MULTI, Match::F16, "spm");
    let user = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\nThe capital is";
    let answers = [" Paris", " Rome", " Berlin"];
    assert_system1_resumes(
        &model,
        ["<|system|>\nYou are a careful assistant.</s>\n", user],
        &answers,
        Match::F16,
        "spm after raw text",
    );
    assert_system1_resumes(
        &model,
        [
            "<|system|>\nYou are a careful assistant.</s>",
            &format!("\n{user}"),
        ],
        &answers,
        Match::F16,
        "spm after a special token",
    );
}

/// Gemma 4 and granitehybrid have no prefix-state format: a resumed candidates
/// request fails with `unsupported_architecture` (before reading any file), and
/// a fresh one still works.
fn assert_system1_resume_rejected(model: &Model) {
    let missing = tmp_path("s1_reject_missing");
    let lines = session(
        model,
        &[
            serde_json::json!({"prompt": "Hello", "candidates": [" world"], "use_pinned": true}),
            serde_json::json!({"prompt": "Hello", "candidates": [" world"],
                               "import_kv": missing}),
            serde_json::json!({"prompt": "Hello", "candidates": [" world", " there"]}),
        ],
    );
    for l in &lines[..2] {
        assert_eq!(l["error_kind"], "unsupported_architecture", "{l}");
    }
    assert_eq!(lines[2]["ok"], true, "{}", lines[2]);
    assert!(lines[2]["chosen"].as_u64().is_some(), "{}", lines[2]);
}

#[test]
#[ignore]
fn ipc_system1_resumed_granite_fixture_rejects() {
    assert_system1_resume_rejected(&load(GRANITE_FIXTURE, WeightsDtype::F32));
}

#[test]
#[ignore]
fn ipc_system1_resumed_gemma4_fixture_rejects() {
    assert_system1_resume_rejected(&load(GEMMA4_FIXTURE, WeightsDtype::F32));
}

/// The version field of a cache file.
fn file_version(path: &str) -> u32 {
    let bytes = std::fs::read(path).unwrap();
    u32::from_le_bytes(bytes[4..8].try_into().unwrap())
}

/// First-token logits of `prompt` resumed from the cache file at `path` (or a
/// fresh prompt), at the model API: the first token's full logit vector.
fn first_logits(model: &Model, prompt: &str, path: Option<&str>) -> Vec<f32> {
    let kv = path.map(|p| crate::kv_io::import_kv(p).expect("import"));
    let resume = kv.as_ref().map_or(Resume::Fresh, Resume::File);
    let mut logits = Vec::new();
    model
        .generate_resumed(
            prompt,
            1,
            resume,
            &crate::sampling::SamplingParams::default(),
            |l| logits = l.to_vec(),
            |_, _| {},
        )
        .expect("generate");
    logits
}

/// `||a - b|| / ||b||` and `max |a - b|`.
fn logit_diff(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut num, mut den, mut max) = (0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
        max = max.max((x - y).abs());
    }
    ((num / den).sqrt(), max)
}

/// How far an f16 cache file may move the first-token logits from the f32
/// file's, as relative L2. f16 keeps 11 significant bits, so each cached value
/// moves by at most ~2.4e-4 of itself; see docs/design/agent-relay.md, "Faster
/// cache files (phase F)", for what that came to on a T4. The bound leaves
/// room above it.
const F16_FILE_REL_L2: f64 = 1e-2;

/// Phase F's f16 files: exports `parts[0]` as an f32 file and as an f16 file
/// (`export_kv_dtype`), resumes `parts[1]` from each over IPC, and compares
/// with one request over the joined text: the same greedy ids from both files,
/// an f16 file of the f16 version and smaller, and first-token logits from
/// the f16 file within [`F16_FILE_REL_L2`] of the f32 file's.
fn assert_f16_file_tracks_f32(model: &Model, parts: [&str; 2], tag: &str, version: u32) {
    const N: usize = 12;
    assert_split_tokenizes_alike(model, &parts);
    let f32_file = tmp_path(&format!("{tag}_f32").replace(' ', "_"));
    let f16_file = tmp_path(&format!("{tag}_f16").replace(' ', "_"));
    let lines = session(
        model,
        &[
            serde_json::json!({"prompt": parts[0], "export_kv": f32_file, "max_tokens": 1}),
            serde_json::json!({"prompt": parts[0], "export_kv": f16_file,
                               "export_kv_dtype": "f16", "max_tokens": 1, "timings": true}),
            serde_json::json!({"prompt": parts[1], "import_kv": f32_file, "max_tokens": N,
                               "timings": true}),
            serde_json::json!({"prompt": parts[1], "import_kv": f16_file, "max_tokens": N,
                               "timings": true}),
            serde_json::json!({"prompt": parts.concat(), "max_tokens": N}),
        ],
    );
    final_ok(&lines[0]);
    final_ok(&lines[1]);
    let (from_f32, from_f16, joined) = (
        final_ok(&lines[2]),
        final_ok(&lines[3]),
        final_ok(&lines[4]),
    );
    let (s32, s16) = (
        std::fs::metadata(&f32_file).unwrap().len(),
        std::fs::metadata(&f16_file).unwrap().len(),
    );
    assert_eq!(file_version(&f16_file), version, "{tag}: f16 file version");
    assert!(s16 < s32, "{tag}: f16 file {s16} bytes, f32 {s32}");
    let l_joined = first_logits(model, &parts.concat(), None);
    let l32 = first_logits(model, parts[1], Some(&f32_file));
    let l16 = first_logits(model, parts[1], Some(&f16_file));
    std::fs::remove_file(&f32_file).ok();
    std::fs::remove_file(&f16_file).ok();
    let (rel32, max32) = logit_diff(&l32, &l_joined);
    let (rel16, max16) = logit_diff(&l16, &l32);
    let (rel16j, max16j) = logit_diff(&l16, &l_joined);
    eprintln!(
        "F16_FILE {tag}: bytes f32 {s32} f16 {s16}; logits rel L2 / max |d|: f32 file vs joined {rel32:.2e}/{max32:.2e}, f16 file vs f32 file {rel16:.2e}/{max16:.2e}, f16 file vs joined {rel16j:.2e}/{max16j:.2e}; kv_io_ms f32 {} f16 {}",
        lines[2]["timings"]["kv_io_ms"], lines[3]["timings"]["kv_io_ms"]
    );
    assert_eq!(from_f32, joined, "{tag}: f32 file differs from joined");
    assert_eq!(from_f16, joined, "{tag}: f16 file differs from joined");
    assert!(
        rel16 < F16_FILE_REL_L2,
        "{tag}: f16 file moved logits by rel L2 {rel16}"
    );
}

/// Real dense Qwen3 (`REFLEX_TEST_GGUF`, `DENSE_GGUF`), default f16 weights.
/// Also layers f16 files (import one, export another) through the three-part
/// chain, and checks the request errors.
#[test]
#[ignore]
fn ipc_kv_f16_file_dense_tracks_f32() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "dense", 5);

    let sys = tmp_path("dense_f16_sys");
    let task = tmp_path("dense_f16_task");
    let bad = tmp_path("dense_f16_bad");
    let lines = session(
        &model,
        &[
            serde_json::json!({"prompt": PARTS[0], "export_kv": sys,
                               "export_kv_dtype": "f16", "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[1], "import_kv": sys, "export_kv": task,
                               "export_kv_dtype": "f16", "max_tokens": 1}),
            serde_json::json!({"prompt": PARTS[2], "import_kv": task, "max_tokens": 12}),
            serde_json::json!({"prompt": PARTS.concat(), "max_tokens": 12}),
            serde_json::json!({"prompt": "p", "export_kv": bad, "export_kv_dtype": "bf16",
                               "max_tokens": 1}),
            serde_json::json!({"prompt": "p", "export_kv_dtype": "f16", "max_tokens": 1}),
        ],
    );
    let layered = final_ok(&lines[2]);
    assert_eq!(file_version(&task), 5);
    std::fs::remove_file(&sys).ok();
    std::fs::remove_file(&task).ok();
    assert_eq!(layered, final_ok(&lines[3]), "layered f16 chain differs");
    assert!(
        !std::path::Path::new(&bad).exists(),
        "rejected export wrote a file"
    );
    assert_eq!(lines[4]["error_kind"], "invalid_input", "{}", lines[4]);
    assert_eq!(lines[5]["error_kind"], "invalid_input", "{}", lines[5]);
}

#[test]
#[ignore]
fn qwen3moe_fixture_ipc_kv_f16_file_tracks_f32() {
    let model = load(QWEN3MOE_FIXTURE, WeightsDtype::F32);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "qwen3moe", 5);
}

/// The hybrid's version 6: f16 attention K/V, f32 GatedDeltaNet state.
#[test]
#[ignore]
fn qwen35moe_fixture_ipc_kv_f16_file_tracks_f32() {
    let model = load(QWEN35MOE_FIXTURE, WeightsDtype::F32);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "qwen35moe", 6);
}

/// A real Qwen3.5 hybrid (`REFLEX_TEST_GGUF`, the nightly's `HYBRID_GGUF`).
#[test]
#[ignore]
fn ipc_kv_f16_file_hybrid_model_tracks_f32() {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a hybrid GGUF");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "hybrid", 6);
}

#[test]
#[ignore]
fn ipc_kv_f16_file_mla_fixture_tracks_f32() {
    let model = load(MLA_FIXTURE, WeightsDtype::F32);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "mla", 7);
}

/// SentencePiece (`REFLEX_TEST_GGUF`, the nightly's `SPM_GGUF`): the version-5
/// header keeps `space_prefix_next`, so a continuation after `</s>` still gets
/// its dummy-prefix space.
#[test]
#[ignore]
fn ipc_kv_f16_file_spm_model_tracks_f32() {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a SentencePiece GGUF (the nightly's SPM_GGUF)");
    let model = load(&path, WeightsDtype::DEFAULT);
    assert_f16_file_tracks_f32(&model, [PARTS[0], PARTS[1]], "spm", 5);
    let user = "\n<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    assert_f16_file_tracks_f32(
        &model,
        ["<|system|>\nYou are a careful assistant.</s>", user],
        "spm after a special token",
        5,
    );
}

/// Granite 4.0-H and Gemma 4 have no cache format in either dtype.
#[test]
#[ignore]
fn ipc_kv_f16_file_granite_and_gemma4_fixtures_reject() {
    for fixture in [GRANITE_FIXTURE, GEMMA4_FIXTURE] {
        let model = load(fixture, WeightsDtype::F32);
        let out = tmp_path("f16_reject");
        let lines = session(
            &model,
            &[
                serde_json::json!({"prompt": "Hello", "export_kv": out,
                                   "export_kv_dtype": "f16", "max_tokens": 1}),
                serde_json::json!({"prompt": "Hello", "max_tokens": 2}),
            ],
        );
        assert!(!std::path::Path::new(&out).exists(), "{fixture}");
        assert_eq!(
            lines[0]["error_kind"], "unsupported_architecture",
            "{fixture}: {}",
            lines[0]
        );
        final_ok(&lines[1]);
    }
}

const TINYOPENJEV_FIXTURE: &str = "test-data/tinyopenjev-for-testing-Q8_0.gguf";

/// `label_prompts` over IPC: the prompts share their state, the response carries
/// one result per prompt, and the session's pinned prefix is left as it was.
#[test]
#[ignore]
fn ipc_label_logits_tinyopenjev_fixture_shares_prefix_and_keeps_pin() {
    let model = load(TINYOPENJEV_FIXTURE, WeightsDtype::F32);
    let state = "<|im_start|>user\nState:\nI was charged twice for my order.\n\nQuestion: ";
    let tail = "\nOptions:\n[A] yes: \n[B] no: \n\nAnswer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let label_req = serde_json::json!({
        "id": "d",
        "timings": true,
        "label_prompts": [
            {"prompt": format!("{state}Is the customer angry?{tail}"), "labels": ["A", "B"]},
            {"prompt": format!("{state}Is a refund requested?{tail}"), "labels": ["A", "B"]},
        ],
    });
    let mut slot = PrefixSlot::new();
    let lines = session_with(
        &model,
        &mut slot,
        &[
            serde_json::json!({"prompt": "Hello there", "pin_prefix": true, "max_tokens": 1}),
            label_req.clone(),
            serde_json::json!({"label_prompts": [{"prompt": "x", "labels": ["A"]}], "use_pinned": true}),
        ],
    );
    final_ok(&lines[0]);
    let pinned = slot.seq_len().expect("pinned");

    let r = &lines[1];
    assert_eq!(r["ok"], true, "{r}");
    assert_eq!(r["id"], "d");
    let shared = r["shared_tokens"].as_u64().unwrap();
    assert!(shared > 10, "{r}");
    let labels = r["labels"].as_array().unwrap();
    assert_eq!(labels.len(), 2);
    for l in labels {
        assert_eq!(l["token_ids"], serde_json::json!([32, 33]), "{l}");
        assert_eq!(l["logits"].as_array().unwrap().len(), 2);
        assert!(l["tokens"].as_u64().unwrap() > shared);
    }
    assert_eq!(r["timings"]["pinned_seq_len"].as_u64(), Some(pinned as u64));
    assert!(
        r.get("token_ids").is_none() && r.get("candidates").is_none(),
        "{r}"
    );

    assert_eq!(lines[2]["error_kind"], "invalid_input", "{}", lines[2]);
    assert_eq!(slot.seq_len(), Some(pinned));

    // Same answer without the pinned prefix around.
    let again = session(&model, &[label_req]);
    assert_eq!(again[0]["labels"], r["labels"]);
}

/// Architectures other than the Qwen3.5 hybrid say so.
#[test]
#[ignore]
fn ipc_label_logits_granite_fixture_unsupported() {
    let model = load(GRANITE_FIXTURE, WeightsDtype::F32);
    let lines = session(
        &model,
        &[serde_json::json!({"label_prompts": [{"prompt": "Hello", "labels": ["A"]}]})],
    );
    assert_eq!(
        lines[0]["error_kind"], "unsupported_architecture",
        "{}",
        lines[0]
    );
}
