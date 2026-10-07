//! `POST /v1/classify`: Reflex's System1 candidate scoring over HTTP. Not an OpenAI
//! endpoint (OpenAI has no classification API); the shape is this sidecar's own.
//!
//! A request names a prompt (raw `prompt`, or chat `messages` rendered the same way
//! `/v1/chat/completions` renders them) and a list of `labels`. Each label is sent
//! to the engine as an IPC `candidates` entry, so the engine runs the prompt
//! through one prefill and scores every label as a continuation of it
//! (`Model::system1_evaluate` in the core crate). Nothing is generated. The
//! response carries each label's probability, relative to this label set only, and
//! the most probable label.
//!
//! Labels are scored as literal continuations of the prompt text, so they usually
//! want a leading space (`" urgent"`, not `"urgent"`) after a prompt that ends in a
//! word or a colon.

use crate::openai::RawMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Most labels one request may score. Every label beyond the first token costs a
/// teacher-forced step in the engine, so an unbounded list would let one request
/// hold the single engine for a long time.
pub const MAX_LABELS: usize = 64;

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// The prompt text, used exactly as given. Exactly one of `prompt` and
    /// `messages` must be set.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Chat messages, rendered through the GGUF's chat template with the assistant
    /// turn opened, so labels are scored as the start of the assistant's reply.
    #[serde(default)]
    pub messages: Option<Vec<RawMessage>>,
    pub labels: Vec<String>,
    /// Softmax temperature over the label scores (the engine's System1
    /// `temperature`, default `1.0`): below 1 sharpens the probabilities, above 1
    /// flattens them. Changes no label's rank.
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// Where the prompt comes from, after [`validate`].
#[derive(Debug)]
pub enum PromptSource<'a> {
    Raw(&'a str),
    Messages(&'a [RawMessage]),
}

/// Checks the request's shape (everything that doesn't need the engine), or a
/// human-readable reason for a `400`.
pub fn validate(req: &ClassifyRequest) -> Result<PromptSource<'_>, String> {
    let source = match (&req.prompt, &req.messages) {
        (Some(p), None) => PromptSource::Raw(p),
        (None, Some(m)) => PromptSource::Messages(m),
        (Some(_), Some(_)) => {
            return Err("set exactly one of `prompt` and `messages`, not both".to_string())
        }
        (None, None) => return Err("one of `prompt` or `messages` is required".to_string()),
    };
    if req.labels.is_empty() {
        return Err("`labels` must be a non-empty array".to_string());
    }
    if req.labels.len() > MAX_LABELS {
        return Err(format!(
            "`labels` has {} entries, over this server's limit of {MAX_LABELS}",
            req.labels.len()
        ));
    }
    if let Some(i) = req.labels.iter().position(|l| l.is_empty()) {
        return Err(format!("labels[{i}] is empty"));
    }
    if let Some(t) = req.temperature {
        if !(t.is_finite() && t > 0.0) {
            return Err(format!("`temperature` must be a positive number, got {t}"));
        }
    }
    Ok(source)
}

/// Bytes the engine is asked to tokenize: the prompt plus its longest label (each
/// label is tokenized appended to the prompt). Checked against
/// `--max-prompt-bytes`.
pub fn request_bytes(prompt: &str, labels: &[String]) -> usize {
    prompt.len() + labels.iter().map(String::len).max().unwrap_or(0)
}

/// The IPC request line's JSON: non-empty `candidates` selects System1 scoring in
/// `src/ipc.rs`.
pub fn ipc_request(prompt: &str, req: &ClassifyRequest) -> Value {
    serde_json::json!({
        "prompt": prompt,
        "candidates": req.labels,
        "temperature": req.temperature.unwrap_or(1.0),
    })
}

#[derive(Debug, Serialize, PartialEq)]
pub struct LabelScore {
    pub label: String,
    /// Probability within this request's label set (sums to 1 over `labels`).
    pub probability: f32,
    /// The engine's raw score for this label, the input to the softmax. Comparable
    /// only within one request.
    pub score: f32,
    /// How many tokens the label took as a continuation of this prompt.
    pub tokens: usize,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ClassifyResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    /// The most probable label (the first one, on a tie).
    pub label: String,
    pub label_index: usize,
    /// Every label, in request order.
    pub labels: Vec<LabelScore>,
    /// Entropy of `labels`' probabilities, in nats: 0 is certain, `ln(labels.len())`
    /// is uniform.
    pub entropy: f32,
}

/// Builds the response from the engine's successful `final` event, or an error
/// message if the event doesn't have the shape `src/ipc.rs` promises.
pub fn build_response(
    v: &Value,
    id: String,
    created: u64,
    model: String,
) -> Result<ClassifyResponse, String> {
    let candidates = v
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or("engine response has no `candidates` array")?;
    let labels = candidates
        .iter()
        .map(|c| {
            Some(LabelScore {
                label: c.get("text")?.as_str()?.to_string(),
                probability: c.get("probability")?.as_f64()? as f32,
                score: c.get("score")?.as_f64()? as f32,
                tokens: c.get("token_ids")?.as_array()?.len(),
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("engine response has a malformed `candidates` entry")?;
    let label_index = labels
        .iter()
        .enumerate()
        .fold(None, |best: Option<(usize, f32)>, (i, l)| match best {
            Some((_, p)) if p >= l.probability => best,
            _ => Some((i, l.probability)),
        })
        .map(|(i, _)| i)
        .ok_or("engine response has an empty `candidates` array")?;
    Ok(ClassifyResponse {
        id,
        object: "classification",
        created,
        model,
        label: labels[label_index].label.clone(),
        label_index,
        entropy: v.get("entropy").and_then(Value::as_f64).unwrap_or(0.0) as f32,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: Value) -> ClassifyRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn validate_requires_exactly_one_prompt_source() {
        let ok = req(serde_json::json!({"prompt": "Q:", "labels": [" a"]}));
        assert!(matches!(validate(&ok), Ok(PromptSource::Raw("Q:"))));
        let ok = req(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "labels": [" a"]
        }));
        assert!(matches!(validate(&ok), Ok(PromptSource::Messages(_))));
        let both = req(serde_json::json!({
            "prompt": "Q:", "messages": [{"role": "user", "content": "hi"}], "labels": [" a"]
        }));
        assert!(validate(&both).unwrap_err().contains("not both"));
        let neither = req(serde_json::json!({"labels": [" a"]}));
        assert!(validate(&neither).unwrap_err().contains("required"));
    }

    #[test]
    fn validate_checks_labels_and_temperature() {
        let empty = req(serde_json::json!({"prompt": "Q:", "labels": []}));
        assert!(validate(&empty).unwrap_err().contains("non-empty"));
        let blank = req(serde_json::json!({"prompt": "Q:", "labels": [" a", ""]}));
        assert!(validate(&blank).unwrap_err().contains("labels[1]"));
        let many: Vec<String> = (0..=MAX_LABELS).map(|i| format!(" {i}")).collect();
        let too_many = req(serde_json::json!({"prompt": "Q:", "labels": many}));
        assert!(validate(&too_many).unwrap_err().contains("limit"));
        let at_limit: Vec<String> = (0..MAX_LABELS).map(|i| format!(" {i}")).collect();
        assert!(validate(&req(
            serde_json::json!({"prompt": "Q:", "labels": at_limit})
        ))
        .is_ok());
        for t in [0.0, -1.0] {
            let bad = req(serde_json::json!({"prompt": "Q:", "labels": [" a"], "temperature": t}));
            assert!(validate(&bad).unwrap_err().contains("temperature"), "{t}");
        }
    }

    #[test]
    fn request_bytes_counts_prompt_plus_longest_label() {
        let labels = vec![" no".to_string(), " maybe".to_string()];
        assert_eq!(request_bytes("Q: ok?", &labels), 6 + 6);
    }

    #[test]
    fn ipc_request_sends_labels_as_candidates() {
        let r = req(serde_json::json!({"prompt": "Q:", "labels": [" yes", " no"]}));
        let v = ipc_request("Q:", &r);
        assert_eq!(v["candidates"], serde_json::json!([" yes", " no"]));
        assert_eq!(v["temperature"], 1.0);
        assert!(v.get("stream").is_none() && v.get("max_tokens").is_none());
    }

    #[test]
    fn build_response_picks_the_most_probable_label_in_request_order() {
        let engine = serde_json::json!({
            "event": "final", "ok": true, "entropy": 0.5,
            "candidates": [
                {"text": " low", "token_ids": [1], "score": -2.0, "probability": 0.2},
                {"text": " high", "token_ids": [2, 3], "score": -0.5, "probability": 0.8},
            ]
        });
        let r = build_response(&engine, "c1".into(), 7, "m".into()).unwrap();
        assert_eq!((r.label.as_str(), r.label_index), (" high", 1));
        assert_eq!(r.labels[0].label, " low");
        assert_eq!(r.labels[1].tokens, 2);
        assert_eq!(r.entropy, 0.5);
        assert_eq!(r.object, "classification");
    }

    #[test]
    fn build_response_breaks_ties_toward_the_first_label() {
        let engine = serde_json::json!({"candidates": [
            {"text": " a", "token_ids": [1], "score": 0.0, "probability": 0.5},
            {"text": " b", "token_ids": [2], "score": 0.0, "probability": 0.5},
        ]});
        let r = build_response(&engine, "c".into(), 0, "m".into()).unwrap();
        assert_eq!(r.label_index, 0);
    }

    #[test]
    fn build_response_rejects_malformed_engine_output() {
        let none = serde_json::json!({"event": "final", "ok": true});
        assert!(build_response(&none, "c".into(), 0, "m".into()).is_err());
        let bad = serde_json::json!({"candidates": [{"text": " a"}]});
        assert!(build_response(&bad, "c".into(), 0, "m".into()).is_err());
        let empty = serde_json::json!({"candidates": []});
        assert!(build_response(&empty, "c".into(), 0, "m".into()).is_err());
    }
}
