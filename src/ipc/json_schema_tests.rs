//! GPU tests for `json_schema` constrained decoding over IPC (agent relay phase C).
//! Every constrained output must parse (with `serde_json`, independently of
//! `crate::grammar`'s own parser) and validate against its schema. The real-model
//! tests cover a nested schema, an enum, an anyOf, a `use_pinned` and an `import_kv`
//! hop, the dead end and the request errors, and the identity property: when the
//! unconstrained greedy output already conforms, the constrained one is the same.
//! The fixture tests run a small schema on every other architecture, since masking
//! sits after each one's own decode loop.

use super::relay_tests::{assert_split_tokenizes_alike, final_ok, load, tmp_path};
use super::*;
use crate::grammar::JsonValue;
use crate::model::WeightsDtype;
use serde_json::{json, Value};

/// An agent handoff record: nested objects and arrays, an enum, a const, a type
/// list, a `$ref`, required and optional properties.
const HANDOFF: &str = r##"{
  "type": "object",
  "properties": {
    "agent": {"type": "string"},
    "status": {"enum": ["done", "blocked", "needs_review"]},
    "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
    "confidence": {"type": "number"},
    "notes": {"type": ["string", "null"]}
  },
  "required": ["agent", "status", "steps"],
  "additionalProperties": false,
  "$defs": {
    "step": {
      "type": "object",
      "properties": {
        "tool": {"enum": ["search", "read_file", "run_tests"]},
        "args": {"type": "object", "properties": {"query": {"type": "string"}}},
        "ok": {"type": "boolean"}
      },
      "required": ["tool", "ok"],
      "additionalProperties": false
    }
  }
}"##;

const COLOR: &str = r##"{"enum": ["red", "green", "blue"]}"##;

const ANSWER_OR_ERROR: &str = r##"{"anyOf": [
  {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false},
  {"type": "object", "properties": {"error": {"type": "string"}}, "required": ["error"], "additionalProperties": false}
]}"##;

/// A finite language (every value is at most a few dozen bytes), so even a model
/// with random weights always finishes it within `max_tokens`.
const SMALL: &str = r##"{"type": "object", "properties": {"ok": {"type": "boolean"}, "pick": {"enum": ["a", "bb", "ccc"]}, "n": {"type": "null"}}, "required": ["ok", "pick"]}"##;

/// Chat-prompt wrapper for the model under test.
type Chat = fn(&str) -> String;

fn qwen3_chat(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// TinyLlama-Chat's template. The hint shows compact JSON: left to itself, the model
/// writes `"key": "value"`, whose space-prefixed SentencePiece tokens (`▁"`) the
/// compact grammar masks, and without them it tends to loop.
fn zephyr_chat(user: &str) -> String {
    format!(
        "<|user|>\n{user} Answer with compact JSON with no spaces, for example {{\"a\":\"x\",\"b\":[1,2]}}.</s>\n<|assistant|>\n"
    )
}

std::thread_local! {
    static RAW_SCHEMAS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A placeholder for `schema` in a `json!` request, which [`session`] replaces with
/// the schema's own text. A `serde_json::Value` sorts object keys, so building the
/// schema as one would reorder its properties, and with them the output.
fn schema_value(schema: &str) -> Value {
    serde_json::from_str::<Value>(schema).expect("test schema is JSON");
    RAW_SCHEMAS.with(|r| {
        let mut r = r.borrow_mut();
        r.push(schema.replace(['\n', '\r'], " "));
        Value::String(format!("__raw_schema_{}__", r.len() - 1))
    })
}

/// Like `relay_tests::session_with`, with [`schema_value`] placeholders replaced
/// by the raw schema text on the wire.
fn session_with(model: &Model, slot: &mut PrefixSlot, requests: &[Value]) -> Vec<Value> {
    let input: String = requests
        .iter()
        .map(|r| {
            let mut line = r.to_string();
            RAW_SCHEMAS.with(|raw| {
                for (i, s) in raw.borrow().iter().enumerate() {
                    line = line.replace(&format!("\"__raw_schema_{i}__\""), s);
                }
            });
            line + "\n"
        })
        .collect();
    let mut out = Vec::new();
    run_request_loop(model, slot, input.as_bytes(), &mut out).expect("request loop");
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).expect("response line is JSON"))
        .collect()
}

fn session(model: &Model, requests: &[Value]) -> Vec<Value> {
    session_with(model, &mut PrefixSlot::new(), requests)
}

/// Asserts a final line is a successful constrained generation and returns its
/// text after checking it parses with `serde_json` and validates against `schema`.
fn assert_valid(resp: &Value, schema: &str) -> String {
    let ids = final_ok(resp);
    let text = resp["text"].as_str().expect("text").to_string();
    serde_json::from_str::<Value>(&text).unwrap_or_else(|e| panic!("serde_json: {text:?}: {e}"));
    let compiled = crate::grammar::JsonSchema::compile(schema).unwrap();
    let parsed = JsonValue::parse(&text).unwrap();
    compiled
        .validate(&parsed)
        .unwrap_or_else(|e| panic!("{text}: {e}"));
    assert!(
        compiled.accepts(&text),
        "output is outside the format: {text}"
    );
    assert!(!ids.is_empty());
    text
}

/// A nested schema (an enum, then an anyOf of objects), but with no array a weak
/// model can repeat items in until `max_tokens` runs out.
const TOOL_CALL: &str = r##"{
  "type": "object",
  "properties": {
    "name": {"enum": ["get_weather", "search_web", "send_email"]},
    "arguments": {"anyOf": [
      {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}}, "required": ["city"], "additionalProperties": false},
      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": false},
      {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"], "additionalProperties": false}
    ]}
  },
  "required": ["name", "arguments"],
  "additionalProperties": false
}"##;

const HANDOFF_PROMPT: &str = "Write the hand-off record, as JSON, for a planner agent that finished planning a command-line tool which deduplicates lines in a file. It ran one search step for \"dedupe lines\" and it worked.";
const TOOL_CALL_PROMPT: &str =
    "Call the right tool, as JSON: what's the weather in Lisbon in celsius?";

/// A nested schema (`nested`, asked for with `nested_prompt`), an enum and an anyOf
/// on `model`, plus streaming and timings.
fn assert_schemas_validate(
    model: &Model,
    chat: Chat,
    tag: &str,
    nested: &str,
    nested_prompt: &str,
) {
    let cases = [
        (nested, nested_prompt),
        (COLOR, "What color is the sky on a clear day? Answer with one of red, green or blue, as a JSON string."),
        (ANSWER_OR_ERROR, "What is 12 * 7? Reply as JSON with the answer."),
        (ANSWER_OR_ERROR, "What is the square root of a banana? Reply as JSON; if there is no answer, give an error message."),
    ];
    for (schema, user) in cases {
        let lines = session(
            model,
            &[
                json!({"id": "c", "prompt": chat(user), "json_schema": schema_value(schema),
                     "max_tokens": 192, "stream": true, "timings": true}),
            ],
        );
        let fin = lines.last().unwrap();
        let text = assert_valid(fin, schema);
        let streamed: String = lines
            .iter()
            .filter(|l| l["event"] == "token")
            .map(|l| l["text"].as_str().unwrap())
            .collect();
        assert_eq!(
            streamed, text,
            "{tag}: streamed pieces join up to the value"
        );
        let n = fin["token_ids"].as_array().unwrap().len();
        let t = &fin["timings"];
        let mask = t["mask_ms_total"].as_f64().expect("mask_ms_total");
        eprintln!(
            "{tag}: {n} tokens, mask {mask:.2} ms total ({:.3} ms/token), model {:.1} ms: {text}",
            mask / n as f64,
            t["model_ms"].as_f64().unwrap()
        );
    }
}

/// For each prompt, an unconstrained greedy run; whenever its output already
/// conforms to the schema (the value then EOS), the constrained run must produce
/// the same ids. Returns how many prompts conformed.
fn assert_identity_where_conforming(model: &Model, cases: &[(&str, String)], tag: &str) -> usize {
    let mut conforming = 0;
    for (schema, prompt) in cases {
        let lines = session(
            model,
            &[
                json!({"prompt": prompt, "max_tokens": 48}),
                json!({"prompt": prompt, "max_tokens": 48, "json_schema": schema_value(schema)}),
            ],
        );
        let free = final_ok(&lines[0]);
        let free_text = model.decode_tokens(&free.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let compiled = crate::grammar::JsonSchema::compile(schema).unwrap();
        // Conforms: everything before the last id is one value the output format
        // allows, and the last id ends the generation.
        let body: String = free_text[..free_text.len().saturating_sub(1)].concat();
        let ends = free.len() < 48;
        let fits = ends && compiled.accepts(&body);
        eprintln!("{tag}: unconstrained {body:?} conforms={fits}");
        if fits {
            conforming += 1;
            assert_eq!(
                final_ok(&lines[1]),
                free,
                "{tag}: constrained ids differ from a conforming greedy output for {prompt:?}"
            );
            assert_eq!(lines[1]["text"].as_str().unwrap(), body);
        } else {
            assert_valid(&lines[1], schema);
        }
    }
    conforming
}

/// A pinned system prompt, then a constrained hop through `use_pinned`, and the
/// same through an exported/imported file: both must validate and equal one joined
/// constrained request.
fn assert_constrained_hops_resume_exactly(
    model: &Model,
    chat: Chat,
    system: &str,
    tag: &str,
    (raw_schema, prompt): (&str, &str),
) {
    let user = chat(prompt);
    let parts = [system, user.as_str()];
    assert_split_tokenizes_alike(model, &parts);
    let schema = schema_value(raw_schema);
    let file = tmp_path(&format!("{tag}_schema_sys"));
    let mut slot = PrefixSlot::new();
    let lines = session_with(
        model,
        &mut slot,
        &[
            json!({"prompt": system, "pin_prefix": true, "max_tokens": 1}),
            json!({"prompt": user, "use_pinned": true, "json_schema": schema, "max_tokens": 160, "timings": true}),
            json!({"prompt": system, "export_kv": file, "max_tokens": 1}),
            json!({"prompt": user, "import_kv": file, "json_schema": schema, "max_tokens": 160}),
            json!({"prompt": parts.concat(), "json_schema": schema, "max_tokens": 160}),
        ],
    );
    std::fs::remove_file(&file).ok();
    final_ok(&lines[0]);
    final_ok(&lines[2]);
    let pinned = final_ok(&lines[1]);
    let imported = final_ok(&lines[3]);
    let joined = final_ok(&lines[4]);
    let text = assert_valid(&lines[1], raw_schema);
    assert_valid(&lines[3], raw_schema);
    assert_valid(&lines[4], raw_schema);
    eprintln!(
        "{tag}: pinned hop {} tokens, timings {}: {text}",
        pinned.len(),
        lines[1]["timings"]
    );
    assert_eq!(
        pinned, joined,
        "{tag}: use_pinned + json_schema differs from one joined request"
    );
    assert_eq!(
        imported, joined,
        "{tag}: import_kv + json_schema differs from one joined request"
    );
    assert!(lines[1]["timings"]["kv_io_ms"].as_f64().is_some());
}

/// The request errors, and running out of `max_tokens`.
fn assert_errors_and_dead_end(model: &Model, chat: Chat) {
    let schema = schema_value(HANDOFF);
    let prompt = chat("Write a hand-off record as JSON.");
    let out = tmp_path("schema_err");
    let lines = session(
        model,
        &[
            json!({"prompt": prompt, "json_schema": schema, "max_tokens": 4, "timings": true}),
            json!({"prompt": prompt, "json_schema": schema, "candidates": ["a", "b"]}),
            json!({"prompt": prompt, "json_schema": schema, "pin_prefix": true, "max_tokens": 1}),
            json!({"prompt": prompt, "json_schema": schema, "export_kv": out, "max_tokens": 1}),
            json!({"prompt": prompt, "json_schema": {"type": "string", "minLength": 3}}),
            json!({"prompt": prompt, "json_schema": "not a schema"}),
            json!({"prompt": prompt, "max_tokens": 4}),
        ],
    );
    assert!(
        !std::path::Path::new(&out).exists(),
        "rejected export wrote a file"
    );
    let dead = &lines[0];
    assert_eq!(dead["ok"], false, "{dead}");
    assert_eq!(dead["error_kind"], "grammar_dead_end", "{dead}");
    let partial = dead["text"].as_str().expect("partial text");
    assert_eq!(dead["token_ids"].as_array().unwrap().len(), 4, "{dead}");
    assert!(partial.starts_with('{'), "{dead}");
    assert!(
        dead["timings"]["mask_ms_total"].as_f64().is_some(),
        "{dead}"
    );
    for (i, needle) in [
        (1, "candidates"),
        (2, "pin_prefix"),
        (3, "export_kv"),
        (4, "\"minLength\""),
        (5, "expected a schema object"),
    ] {
        assert_eq!(lines[i]["error_kind"], "invalid_input", "{}", lines[i]);
        assert!(
            lines[i]["error"].as_str().unwrap().contains(needle),
            "{needle}: {}",
            lines[i]
        );
    }
    // The session carries on after every error.
    final_ok(&lines[6]);
}

/// Real dense Qwen3 (`REFLEX_TEST_GGUF`, the nightly's `DENSE_GGUF`).
fn dense_model() -> Model {
    let path = std::env::var("REFLEX_TEST_GGUF").expect("set REFLEX_TEST_GGUF to a dense GGUF");
    load(&path, WeightsDtype::DEFAULT)
}

#[test]
#[ignore]
fn ipc_json_schema_dense_outputs_validate() {
    let model = dense_model();
    assert!(
        model.grammar_index_stats().is_none(),
        "built before any constrained request"
    );
    assert_schemas_validate(&model, qwen3_chat, "qwen3", HANDOFF, HANDOFF_PROMPT);
    let (build_ms, nodes) = model.grammar_index_stats().expect("built by now");
    eprintln!("qwen3: token index built in {build_ms:.1} ms, {nodes} trie nodes");
}

#[test]
#[ignore]
fn ipc_json_schema_dense_matches_unconstrained_greedy() {
    let model = dense_model();
    let cases = [
        (r#"{"type": "integer"}"#, qwen3_chat("What is 2 + 3? Reply with just the number.")),
        (r#"{"type": "integer"}"#, qwen3_chat("How many legs does a spider have? Reply with just the number.")),
        (
            r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#,
            qwen3_chat("Return compact JSON with no spaces, exactly like {\"name\":\"Bob\",\"age\":30}, for Alice who is 25."),
        ),
        (COLOR, qwen3_chat("Reply with the JSON string \"green\" and nothing else.")),
    ];
    let n = assert_identity_where_conforming(&model, &cases, "qwen3");
    assert!(
        n >= 2,
        "only {n} of the prompts' greedy outputs conformed; pick prompts that do"
    );
}

#[test]
#[ignore]
fn ipc_json_schema_dense_pinned_and_imported_hops() {
    let model = dense_model();
    assert_constrained_hops_resume_exactly(
        &model,
        qwen3_chat,
        "<|im_start|>system\nYou are one agent in a team. You report to the orchestrator as JSON.<|im_end|>\n",
        "qwen3",
        (
            HANDOFF,
            "Give the hand-off record, as JSON, for a reviewer agent that is blocked waiting for test results.",
        ),
    );
}

#[test]
#[ignore]
fn ipc_json_schema_dense_errors_and_dead_end() {
    assert_errors_and_dead_end(&dense_model(), qwen3_chat);
}

/// A SentencePiece model (`SPM_GGUF` in the nightly, e.g. TinyLlama-1.1B-Chat or
/// Mistral-7B-Instruct): `▁` spaces and `<0xNN>` byte-fallback tokens.
fn spm_model() -> Model {
    let path = std::env::var("REFLEX_TEST_GGUF")
        .expect("set REFLEX_TEST_GGUF to a SentencePiece GGUF (the nightly's SPM_GGUF)");
    load(&path, WeightsDtype::DEFAULT)
}

#[test]
#[ignore]
fn ipc_json_schema_spm_model_outputs_validate() {
    let model = spm_model();
    assert_schemas_validate(&model, zephyr_chat, "spm", TOOL_CALL, TOOL_CALL_PROMPT);
    let cases = [
        (
            r#"{"type": "integer"}"#,
            zephyr_chat("What is 2 + 3? Reply with just the number."),
        ),
        (
            COLOR,
            zephyr_chat("Reply with the JSON string \"green\" and nothing else."),
        ),
    ];
    let n = assert_identity_where_conforming(&model, &cases, "spm");
    eprintln!("spm: {n} of {} greedy outputs conformed", cases.len());
}

#[test]
#[ignore]
fn ipc_json_schema_spm_model_pinned_hop_and_errors() {
    let model = spm_model();
    assert_constrained_hops_resume_exactly(
        &model,
        zephyr_chat,
        "<|system|>\nYou are one agent in a team. You report to the orchestrator as JSON.</s>\n",
        "spm",
        (
            TOOL_CALL,
            "Call the right tool, as JSON: email bob@example.com that the build is green.",
        ),
    );
    assert_errors_and_dead_end(&model, zephyr_chat);
}

/// [`SMALL`] on a fixture with random weights: the value must be complete and valid,
/// with EOS forced right after it.
fn assert_fixture_constrained(path: &str, tag: &str) {
    let model = load(path, WeightsDtype::DEFAULT);
    let lines = session(
        &model,
        &[
            json!({"prompt": "Hello", "json_schema": schema_value(SMALL), "max_tokens": 64}),
            json!({"prompt": "Hello", "json_schema": schema_value(r#"{"type":"array","items":{"enum":[true,false,null]}}"#),
                   "max_tokens": 24}),
        ],
    );
    let text = assert_valid(&lines[0], SMALL);
    eprintln!("{tag}: {text}");
    let second = &lines[1];
    if second["ok"] == true {
        assert_valid(
            second,
            r#"{"type":"array","items":{"enum":[true,false,null]}}"#,
        );
    } else {
        assert_eq!(second["error_kind"], "grammar_dead_end", "{tag}: {second}");
    }
}

#[test]
#[ignore]
fn qwen3moe_fixture_json_schema_validates() {
    assert_fixture_constrained("test-data/tiny-qwen3moe.gguf", "qwen3moe");
}

#[test]
#[ignore]
fn qwen35moe_fixture_json_schema_validates() {
    assert_fixture_constrained("test-data/tiny-qwen35moe.gguf", "qwen35moe");
}

#[test]
#[ignore]
fn ipc_json_schema_mla_fixture_validates() {
    assert_fixture_constrained("test-data/deepseek-tiny-mla.gguf", "mla");
}

#[test]
#[ignore]
fn ipc_json_schema_granite_fixture_validates() {
    assert_fixture_constrained("test-data/tiny-granitehybrid.gguf", "granite");
}

#[test]
#[ignore]
fn ipc_json_schema_gemma4_fixture_validates() {
    assert_fixture_constrained("test-data/tiny-gemma4.gguf", "gemma4");
}
