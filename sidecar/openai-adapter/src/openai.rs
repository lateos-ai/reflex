//! OpenAI `/v1/chat/completions` request/response shapes, and the translation
//! between them and Reflex's `src/ipc.rs` protocol. Deliberately supports only the
//! subset real callers (OpenRouter, the OpenAI Python/JS SDKs, curl) actually need
//! to drive a single-turn or multi-turn chat completion against a locally loaded
//! GGUF -- not the full OpenAI API surface (no function/tool calling, no logprobs,
//! no `n > 1`, no multimodal content parts). See this crate's README for the full
//! list of known limitations.

use serde::{Deserialize, Serialize};
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
