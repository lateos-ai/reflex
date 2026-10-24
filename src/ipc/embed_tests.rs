//! GPU tests for IPC `"embed": true` on the embedding fixture
//! (`test-data/tiny-mistral3-embed.gguf`, see `model::encoder_tests`), driven
//! through [`run_request_loop`].

use super::relay_tests::load;
use super::*;
use crate::model::WeightsDtype;
use serde_json::Value;

const EMBED_FIXTURE: &str = "test-data/tiny-mistral3-embed.gguf";

fn session(model: &Model, requests: &[Value]) -> Vec<Value> {
    let input: String = requests.iter().map(|r| format!("{r}\n")).collect();
    let mut out = Vec::new();
    run_request_loop(model, &mut PrefixSlot::new(), input.as_bytes(), &mut out)
        .expect("request loop");
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).expect("response line is JSON"))
        .collect()
}

/// An embed request returns `Model::embed`'s vector and token count; embed with a
/// generation field, and generation on an embedding model, fail with
/// `invalid_input`.
#[test]
#[ignore]
fn ipc_embed_fixture_round_trip_and_rejections() {
    let model = load(EMBED_FIXTURE, WeightsDtype::F32);
    let text = "query: What is NVIDIA NeMo Framework?";
    let direct = model.embed(text).expect("embed");
    let lines = session(
        &model,
        &[
            serde_json::json!({"id": "a", "prompt": text, "embed": true, "timings": true}),
            serde_json::json!({"prompt": text, "embed": true, "stream": true}),
            serde_json::json!({"prompt": text, "embed": true, "candidates": [" a"]}),
            serde_json::json!({"prompt": text, "embed": true, "use_pinned": true}),
            serde_json::json!({"prompt": text, "max_tokens": 2}),
            serde_json::json!({"prompt": text, "candidates": [" a", " b"]}),
        ],
    );
    assert_eq!(lines.len(), 6, "{lines:?}");

    let first = &lines[0];
    assert_eq!(first["ok"], true, "{first}");
    assert_eq!(first["id"], "a");
    assert_eq!(first["prompt_tokens"], direct.prompt_tokens);
    let got: Vec<f32> = first["embedding"]
        .as_array()
        .expect("embedding array")
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(got, direct.vector);
    assert!(first["timings"]["model_ms"].as_f64().is_some());
    assert!(first.get("token_ids").is_none() && first.get("text").is_none());

    // `stream` is a generation field: an embed request takes none of them.
    for (i, line) in lines.iter().enumerate().skip(1) {
        assert_eq!(line["ok"], false, "line {i}: {line}");
        assert_eq!(line["error_kind"], "invalid_input", "line {i}: {line}");
    }
}
