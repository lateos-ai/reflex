//! Local, non-network IPC protocol shared by the `reflex stdio`/`reflex uds`
//! subcommands (`src/bin/reflex/stdio.rs`/`uds.rs`) -- see CLAUDE.md/README.md's
//! Non-goals: no HTTP/gRPC server, ever; this is the sequential, non-thread-pool
//! local-ergonomics surface that stands in for one. One line of JSON in, one line of
//! JSON out, one request fully processed before the next is read -- both binaries
//! that use this module share that same rule, just over a different transport
//! (stdin/stdout vs. a Unix Domain Socket).
//!
//! Empty `candidates` dispatches to [`crate::model::Model::generate`] (ordinary
//! decode); a non-empty `candidates` list dispatches to
//! [`crate::model::Model::system1_evaluate`] (single-pass candidate scoring) instead
//! -- never both, and never a batch of more than one prompt per request, matching
//! this engine's permanent `batch_size == 1` constraint.

use crate::model::{Model, System1Candidate};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

fn default_max_tokens() -> usize {
    32
}

fn default_temperature() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct IpcRequest {
    /// Caller-supplied correlation id, echoed back unchanged on
    /// [`IpcResponse::id`] -- lets a caller matching requests to responses over a
    /// connection that also allows concurrent-looking client code, even though
    /// this engine itself only ever processes one request at a time.
    #[serde(default)]
    pub id: Option<String>,
    pub prompt: String,
    /// Empty (the default) selects `Model::generate`; non-empty selects
    /// `Model::system1_evaluate` over these candidate continuations.
    #[serde(default)]
    pub candidates: Vec<String>,
    /// Only used by the `Model::generate` path (empty `candidates`).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Only used by the `Model::system1_evaluate` path (non-empty `candidates`).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

#[derive(Debug, Serialize)]
pub struct IpcCandidateResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub score: f32,
    pub probability: f32,
}

#[derive(Debug, Serialize)]
pub struct IpcResponse {
    pub id: Option<String>,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// `Model::generate` path only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
    /// `Model::generate` path only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// `Model::system1_evaluate` path only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<IpcCandidateResult>>,
    /// `Model::system1_evaluate` path only -- see
    /// `crate::model::System1Response::entropy`'s doc comment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entropy: Option<f32>,
}

impl IpcResponse {
    fn err(id: Option<String>, error: String) -> Self {
        IpcResponse { id, ok: false, error: Some(error), token_ids: None, text: None, candidates: None, entropy: None }
    }
}

/// Dispatches one already-parsed request to `Model::generate` or
/// `Model::system1_evaluate` and wraps the result (or error) as an
/// [`IpcResponse`]. Never panics -- every failure path from the underlying model
/// call becomes `ok: false` with `error` set, not a process exit, since a bad
/// request on a long-lived stdio/UDS session must not kill the whole session.
pub fn handle_request(model: &Model, req: IpcRequest) -> IpcResponse {
    let id = req.id.clone();
    if req.candidates.is_empty() {
        match model.generate(&req.prompt, req.max_tokens.max(1), None, |_logits| {}) {
            Ok((token_ids, text)) => {
                IpcResponse { id, ok: true, error: None, token_ids: Some(token_ids), text: Some(text), candidates: None, entropy: None }
            }
            Err(e) => IpcResponse::err(id, e),
        }
    } else {
        let candidates: Vec<System1Candidate> = req.candidates.iter().map(|text| System1Candidate { text: text.clone() }).collect();
        match model.system1_evaluate(&req.prompt, &candidates, req.temperature) {
            Ok(response) => {
                let candidates = response
                    .results
                    .into_iter()
                    .zip(response.probabilities)
                    .map(|(r, probability)| IpcCandidateResult { text: r.text, token_ids: r.token_ids, score: r.score, probability })
                    .collect();
                IpcResponse {
                    id,
                    ok: true,
                    error: None,
                    token_ids: None,
                    text: None,
                    candidates: Some(candidates),
                    entropy: Some(response.entropy),
                }
            }
            Err(e) => IpcResponse::err(id, e),
        }
    }
}

/// Reads one JSON request per line from `input`, dispatches it via
/// [`handle_request`], and writes one JSON response line to `output` (flushed
/// immediately -- a caller piping this over a pipe/socket must see each response
/// as soon as it's ready, not buffered until process exit). A line that fails to
/// parse as JSON becomes an `ok: false` response with no `id` (the request itself
/// was unreadable) instead of terminating the loop -- one malformed line must not
/// kill an otherwise-healthy long-lived session. Returns only on EOF or a hard I/O
/// error; strictly sequential, never spawns a thread.
pub fn run_request_loop<R: BufRead, W: Write>(model: &Model, mut input: R, mut output: W) -> Result<(), String> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = input.read_line(&mut line).map_err(|e| format!("ipc: reading request line: {e}"))?;
        if bytes_read == 0 {
            return Ok(()); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<IpcRequest>(trimmed) {
            Ok(req) => handle_request(model, req),
            Err(e) => IpcResponse::err(None, format!("ipc: malformed request JSON: {e}")),
        };

        let response_json = serde_json::to_string(&response).map_err(|e| format!("ipc: serializing response: {e}"))?;
        writeln!(output, "{response_json}").map_err(|e| format!("ipc: writing response line: {e}"))?;
        output.flush().map_err(|e| format!("ipc: flushing response: {e}"))?;
    }
}

/// [`run_request_loop`] over `std::io::stdin()`/`std::io::stdout()` specifically --
/// the shape `reflex stdio`'s `run` needs.
pub fn run_stdio_loop(model: &Model) -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run_request_loop(model, stdin.lock(), stdout.lock())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_request_defaults_max_tokens_and_temperature_when_omitted() {
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "hello"}"#).expect("should parse");
        assert_eq!(req.prompt, "hello");
        assert!(req.candidates.is_empty());
        assert_eq!(req.max_tokens, 32);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.id, None);
    }

    #[test]
    fn test_ipc_request_parses_all_fields() {
        let req: IpcRequest = serde_json::from_str(
            r#"{"id": "abc", "prompt": "p", "candidates": ["a", "b"], "max_tokens": 5, "temperature": 0.5}"#,
        )
        .expect("should parse");
        assert_eq!(req.id.as_deref(), Some("abc"));
        assert_eq!(req.candidates, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(req.max_tokens, 5);
        assert_eq!(req.temperature, 0.5);
    }

    #[test]
    fn test_ipc_response_error_variant_omits_result_fields() {
        let resp = IpcResponse::err(Some("id1".to_string()), "boom".to_string());
        let json = serde_json::to_string(&resp).expect("should serialize");
        assert!(json.contains("\"ok\":false"));
        assert!(json.contains("\"error\":\"boom\""));
        assert!(!json.contains("token_ids"), "error responses should omit null result fields: {json}");
        assert!(!json.contains("candidates"), "error responses should omit null result fields: {json}");
    }

    #[test]
    fn test_malformed_json_line_produces_error_response_not_a_loop_abort() {
        let model_free_response = match serde_json::from_str::<IpcRequest>("not json") {
            Ok(_) => panic!("expected parse failure"),
            Err(e) => IpcResponse::err(None, format!("ipc: malformed request JSON: {e}")),
        };
        assert!(!model_free_response.ok);
        assert!(model_free_response.error.unwrap().contains("malformed request JSON"));
    }
}
