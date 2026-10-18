//! OpenAI `/v1/chat/completions` request/response shapes, and the translation
//! between them and Reflex's `src/ipc.rs` protocol. Deliberately supports only the
//! subset real callers (OpenRouter, the OpenAI Python/JS SDKs, curl) actually need
//! to drive a single-turn or multi-turn chat completion against a locally loaded
//! GGUF -- not the full OpenAI API surface (no function/tool calling, no logprobs,
//! no `n > 1`, no multimodal content parts). See this crate's README for the full
//! list of known limitations.

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<RawMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Not part of the OpenAI Chat Completions schema -- a Reflex-specific
    /// extension so a caller who wants top-k sampling (which `IpcSamplingParams`
    /// supports) isn't locked out just because OpenAI's own API doesn't expose it.
    /// Ignored unless `temperature` is also set to a positive value (same rule
    /// `IpcRequest::sampling_params` applies).
    #[serde(default)]
    pub top_k: Option<usize>,
    /// `{"type": "text"}` (the default), `{"type": "json_schema", ...}` or
    /// `{"type": "json_object"}` -- see [`ipc_json_schema`].
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    // Every other field (including the IPC-only `pin_prefix`, `use_pinned`,
    // `import_kv`, `export_kv`) is ignored: serde drops unknown fields, and
    // `build_ipc_line` writes only the fields named in `IpcGenerateRequest`.
}

/// OpenAI's `response_format`. A plain struct rather than an enum tagged on `type`:
/// serde buffers an internally tagged enum's content, which a [`RawValue`] can't go
/// through.
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaFormat>,
}

/// `response_format.json_schema`. `name`, `description` and `strict` are accepted
/// and not used: the engine always enforces the schema strictly.
#[derive(Debug, Deserialize)]
pub struct JsonSchemaFormat {
    /// Kept as the request's own text, so its properties reach the engine in the
    /// order they're written (a `Value` without `preserve_order` sorts keys).
    #[serde(default)]
    pub schema: Option<Box<RawValue>>,
}

/// The schema `{"type": "json_object"}` maps to: any JSON object. In the engine's
/// subset, an object schema without `properties` and without
/// `additionalProperties: false` allows any object.
pub const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

/// The IPC `json_schema` for `response_format`, compacted to one line, or `None`
/// for no constraint (`text`, or no `response_format`). Errs with a message for a
/// `400` on an unknown `type` or a `json_schema` without a `schema`. The schema's
/// contents are the engine's to check: an unsupported keyword comes back as an
/// `invalid_input` error.
pub fn ipc_json_schema(format: Option<&ResponseFormat>) -> Result<Option<Box<RawValue>>, String> {
    let Some(format) = format else {
        return Ok(None);
    };
    let raw = match format.kind.as_str() {
        "text" => return Ok(None),
        "json_object" => JSON_OBJECT_SCHEMA,
        "json_schema" => format
            .json_schema
            .as_ref()
            .and_then(|f| f.schema.as_deref())
            .ok_or("response_format.json_schema.schema is required with type \"json_schema\"")?
            .get(),
        other => {
            return Err(format!(
                "unsupported response_format type {other:?}: expected \"text\", \
                 \"json_schema\" or \"json_object\""
            ))
        }
    };
    RawValue::from_string(compact_json(raw))
        .map(Some)
        .map_err(|e| format!("response_format schema: {e}"))
}

/// `raw` (valid JSON) without the whitespace between tokens. A schema sent
/// pretty-printed would otherwise put newlines into the IPC request, which is one
/// JSON value per line. Strings, and so key order and every value, are untouched.
fn compact_json(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in raw.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if !matches!(c, ' ' | '\t' | '\n' | '\r') {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
        }
    }
    out
}

/// A `generate` IPC request, as `src/ipc.rs`'s `IpcRequest` reads it. A struct, not
/// a `json!` value, so the line's keys come out in this order and `json_schema` is
/// written as raw text.
#[derive(Serialize)]
struct IpcGenerateRequest<'a> {
    prompt: &'a str,
    max_tokens: usize,
    stream: bool,
    sampling: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<&'a RawValue>,
}

/// The IPC request line for a chat completion: the rendered `prompt`, the
/// already-checked `max_tokens`, the request's `stream` and sampling fields, and
/// `json_schema` from [`ipc_json_schema`].
pub fn build_ipc_line(
    req: &ChatCompletionRequest,
    prompt: &str,
    max_tokens: usize,
    json_schema: Option<&RawValue>,
) -> serde_json::Result<String> {
    serde_json::to_string(&IpcGenerateRequest {
        prompt,
        max_tokens,
        stream: req.stream,
        sampling: build_sampling(req),
        json_schema,
    })
}

/// `content` is accepted as a plain JSON `Value` at parse time (not `String`
/// directly) so a request using OpenAI's multimodal content-parts array shape fails
/// with a clear 400 from [`extract_text_messages`] instead of a generic serde parse
/// error -- this adapter only supports plain string content.
#[derive(Debug, Deserialize)]
pub struct RawMessage {
    pub role: String,
    pub content: Value,
}

pub struct TextMessage {
    pub role: String,
    pub content: String,
}

/// Extracts plain-string content from every message, or a human-readable error
/// naming the first offending message index -- e.g. a multimodal content-parts
/// array, which this adapter doesn't support (see [`RawMessage::content`]'s doc
/// comment).
pub fn extract_text_messages(messages: &[RawMessage]) -> Result<Vec<TextMessage>, String> {
    if messages.is_empty() {
        return Err("`messages` must be a non-empty array".to_string());
    }
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| match m.content.as_str() {
            Some(s) => Ok(TextMessage {
                role: m.role.clone(),
                content: s.to_string(),
            }),
            None => Err(format!(
                "messages[{i}].content must be a string -- this adapter doesn't support \
                 multimodal content-parts arrays"
            )),
        })
        .collect()
}

/// Flattens a chat message list into a single prompt string via plain role-labeled
/// concatenation, since Reflex itself has no chat-template support (`src/model.rs`'s
/// `forward_prompt` takes a plain prompt string, not a message list). This is the
/// fallback path: `main.rs::chat_completions` prefers rendering the GGUF's own
/// `tokenizer.chat_template` (see `chat_template.rs`) when one loaded successfully at
/// startup, and only calls this when no template is available, `--no-chat-template`
/// was passed, or the template fails to render for a given request.
pub fn build_prompt(messages: &[TextMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        let label = match m.role.as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };
        prompt.push_str(label);
        prompt.push_str(": ");
        prompt.push_str(&m.content);
        prompt.push('\n');
    }
    prompt.push_str("Assistant:");
    prompt
}

/// Builds the `sampling` field of an `IpcRequest` (see `src/ipc.rs::IpcSamplingParams`)
/// from the OpenAI-shaped request fields, or `None` to omit it entirely (greedy) --
/// mirrors `IpcRequest::sampling_params`'s own rule that only a positive temperature
/// opts into sampling.
pub fn build_sampling(req: &ChatCompletionRequest) -> Option<Value> {
    match req.temperature {
        Some(t) if t > 0.0 => Some(serde_json::json!({
            "temperature": t,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "seed": req.seed,
        })),
        _ => None,
    }
}

/// Very rough token-count estimate (whitespace-split word count) for the `usage`
/// field's `prompt_tokens` -- this adapter has no tokenizer of its own and the IPC
/// protocol doesn't report an exact prompt token count, so this is an approximation,
/// not an exact figure. Flagged in the README's known limitations.
pub fn estimate_prompt_tokens(prompt: &str) -> u32 {
    prompt.split_whitespace().count() as u32
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Default)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// Pricing strings for [`ModelInfo`] -- kept as strings (not numbers) to match
/// OpenRouter's own `/models` pricing convention (per-token cost as a decimal
/// string), since that's the metadata shape this endpoint exists to satisfy.
#[derive(Debug, Serialize, Clone)]
pub struct ModelPricing {
    pub prompt: String,
    pub completion: String,
}

/// One entry of `GET /v1/models`'s `data` array. This sidecar only ever manages one
/// model (the single `reflex stdio` child it spawned at startup), so `data` always
/// has exactly one element -- see `main.rs`. `context_length`/`datacenter_location`
/// are OpenAI-schema extensions, not part of the standard `/v1/models` shape, kept
/// here anyway since they're exactly what a future OpenRouter provider-listing
/// application needs (see this crate's README).
#[derive(Debug, Serialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    pub pricing: ModelPricing,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datacenter_location: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

/// `"length"` if the final response's `token_ids` reached (or exceeded) the
/// `max_tokens` actually requested from Reflex, `"stop"` otherwise. This is a
/// heuristic, not information Reflex's IPC protocol reports directly (it has no
/// explicit stop-reason field) -- documented as approximate in the README.
pub fn finish_reason(completion_tokens: usize, requested_max_tokens: usize) -> &'static str {
    if completion_tokens >= requested_max_tokens {
        "length"
    } else {
        "stop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses an HTTP body the way axum's `Json` extractor does (`from_slice` on
    /// the raw bytes) and builds its IPC line.
    fn ipc_line(body: &str) -> Result<String, String> {
        let req: ChatCompletionRequest = serde_json::from_slice(body.as_bytes()).unwrap();
        let schema = ipc_json_schema(req.response_format.as_ref())?;
        Ok(build_ipc_line(&req, "P", 5, schema.as_deref()).unwrap())
    }

    const MESSAGES: &str = r#""messages": [{"role": "user", "content": "hi"}]"#;

    #[test]
    fn json_schema_passes_through_in_schema_order_compacted_to_one_line() {
        // Properties deliberately not in alphabetical order, pretty-printed, with
        // whitespace, an escaped quote and a backslash inside strings.
        let body = format!(
            r#"{{{MESSAGES}, "response_format": {{
                "type": "json_schema",
                "json_schema": {{
                    "name": "handoff",
                    "strict": true,
                    "schema": {{
                        "type": "object",
                        "properties": {{
                            "zeta": {{"type": "string", "description": "a \"quoted\" note"}},
                            "alpha": {{"enum": ["x y", "back\\slash"]}},
                            "mid": {{"type": "integer"}}
                        }},
                        "required": ["zeta", "alpha"]
                    }}
                }}
            }}}}"#
        );
        let line = ipc_line(&body).unwrap();
        assert_eq!(
            line,
            r#"{"prompt":"P","max_tokens":5,"stream":false,"sampling":null,"json_schema":{"type":"object","properties":{"zeta":{"type":"string","description":"a \"quoted\" note"},"alpha":{"enum":["x y","back\\slash"]},"mid":{"type":"integer"}},"required":["zeta","alpha"]}}"#
        );
        assert!(!line.contains('\n'));
        // A `Value` round trip would have sorted the properties.
        let sorted = serde_json::to_string(&serde_json::from_str::<Value>(&line).unwrap()).unwrap();
        assert!(
            sorted.find("\"alpha\"") < sorted.find("\"zeta\""),
            "{sorted}"
        );
        assert!(line.find("\"zeta\"") < line.find("\"alpha\""), "{line}");
    }

    #[test]
    fn text_or_no_response_format_sends_no_schema() {
        let expected = r#"{"prompt":"P","max_tokens":5,"stream":false,"sampling":null}"#;
        assert_eq!(ipc_line(&format!("{{{MESSAGES}}}")).unwrap(), expected);
        let text = format!(r#"{{{MESSAGES}, "response_format": {{"type": "text"}}}}"#);
        assert_eq!(ipc_line(&text).unwrap(), expected);
    }

    #[test]
    fn json_object_maps_to_any_object() {
        let body = format!(r#"{{{MESSAGES}, "response_format": {{"type": "json_object"}}}}"#);
        assert_eq!(
            ipc_line(&body).unwrap(),
            r#"{"prompt":"P","max_tokens":5,"stream":false,"sampling":null,"json_schema":{"type":"object"}}"#
        );
    }

    #[test]
    fn bad_response_formats_are_rejected_before_the_engine() {
        for (format, needle) in [
            (r#"{"type": "json_schema"}"#, "schema is required"),
            (
                r#"{"type": "json_schema", "json_schema": {"name": "x"}}"#,
                "schema is required",
            ),
            (
                r#"{"type": "json_schema", "json_schema": {"schema": null}}"#,
                "schema is required",
            ),
            (r#"{"type": "regex"}"#, "unsupported response_format type"),
        ] {
            let body = format!(r#"{{{MESSAGES}, "response_format": {format}}}"#);
            let err = ipc_line(&body).unwrap_err();
            assert!(err.contains(needle), "{format}: {err}");
        }
    }

    #[test]
    fn sampling_and_stream_sit_next_to_the_schema() {
        let body = format!(
            r#"{{{MESSAGES}, "stream": true, "temperature": 0.5, "seed": 7,
                "response_format": {{"type": "json_schema", "json_schema": {{"schema": {{"type": "boolean"}}}}}}}}"#
        );
        assert_eq!(
            ipc_line(&body).unwrap(),
            r#"{"prompt":"P","max_tokens":5,"stream":true,"sampling":{"seed":7,"temperature":0.5,"top_k":null,"top_p":null},"json_schema":{"type":"boolean"}}"#
        );
    }

    /// HTTP has no pin, slot or cache-file fields: a request carrying the IPC ones
    /// is accepted (unknown fields are ignored, as for any other) and none of them
    /// reaches the engine.
    #[test]
    fn ipc_only_fields_never_reach_the_engine() {
        let body = format!(
            r#"{{{MESSAGES}, "pin_prefix": true, "use_pinned": true,
                "import_kv": "/tmp/a.kv", "export_kv": "/tmp/b.kv", "export_kv_dtype": "f16",
                "slot": 0, "file": "/etc/passwd", "candidates": [" a"], "timings": true,
                "json_schema": {{"type": "object"}}, "id": "x"}}"#
        );
        assert_eq!(
            ipc_line(&body).unwrap(),
            r#"{"prompt":"P","max_tokens":5,"stream":false,"sampling":null}"#
        );
    }

    #[test]
    fn compact_json_keeps_strings_byte_for_byte() {
        assert_eq!(
            compact_json("{ \"a b\" : [ 1 ,\n\t\"\\\\\" , \"\\\" x\" ] }"),
            r#"{"a b":[1,"\\","\" x"]}"#
        );
        assert_eq!(compact_json("\"é \\u00e9\""), "\"é \\u00e9\"");
    }
}
