//! Local, non-network IPC protocol shared by the `reflex stdio`/`reflex uds`
//! subcommands (`src/bin/reflex/stdio.rs`/`uds.rs`) -- see CLAUDE.md/README.md's
//! Non-goals: no HTTP/gRPC server, ever; this is the sequential, non-thread-pool
//! local-ergonomics surface that stands in for one. One line of JSON in; one or
//! more lines of JSON out (see `stream` below), one request fully processed before
//! the next is read -- both binaries that use this module share that same rule,
//! just over a different transport (stdin/stdout vs. a Unix Domain Socket). Nothing
//! here spawns a thread: a streaming request's token events are written and
//! flushed one at a time from the same synchronous decode loop
//! [`crate::model::Model::generate`] already runs, not from a background task.
//!
//! Empty `candidates` dispatches to [`crate::model::Model::generate`] (ordinary
//! decode); a non-empty `candidates` list dispatches to
//! [`crate::model::Model::system1_evaluate`] (single-pass candidate scoring) instead
//! -- never both, and never a batch of more than one prompt per request, matching
//! this engine's permanent `batch_size == 1` constraint.
//!
//! **Streaming** (`generate` path only -- `system1_evaluate` is a single forward
//! pass, not an autoregressive decode loop, so there's nothing to stream): a
//! request with `"stream": true` gets one [`IpcStreamToken`] line (`"event":
//! "token"`) per generated token, flushed as soon as it's produced, followed by one
//! final [`IpcResponse`] line (`"event": "final"`) carrying the same aggregate
//! result a non-streaming request would have gotten as its only line. A
//! non-streaming request (`stream` omitted or `false`, the default) gets exactly
//! the one final line, unchanged from this protocol's pre-streaming shape aside
//! from the added `event` field.
//!
//! **Sampling** (`generate` path only): `sampling` omitted (the default) is greedy
//! argmax, byte-identical to this protocol's pre-sampling behavior. See
//! [`IpcSamplingParams`]/`crate::sampling::SamplingParams`.

use crate::model::{Model, System1Candidate};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

fn default_max_tokens() -> usize {
    32
}

fn default_temperature() -> f32 {
    1.0
}

/// `crate::sampling::SamplingParams`'s wire form -- see that type's doc comment
/// for what each field does. `temperature <= 0.0` (the default when this whole
/// object is omitted from a request, via [`IpcRequest::sampling_params`]) is
/// greedy argmax; only a positive `temperature` activates sampling.
#[derive(Debug, Deserialize)]
pub struct IpcSamplingParams {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct IpcRequest {
    /// Caller-supplied correlation id, echoed back unchanged on
    /// [`IpcResponse::id`]/[`IpcStreamToken::id`] -- lets a caller matching
    /// requests to responses over a connection that also allows concurrent-looking
    /// client code, even though this engine itself only ever processes one request
    /// at a time.
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
    /// Only used by the `Model::system1_evaluate` path (non-empty `candidates`) --
    /// Platt-style softmax temperature over the candidate scores, unrelated to
    /// `sampling.temperature` below. Kept as its own field, at its own long-
    /// standing default of `1.0`, rather than reusing `sampling` for this, since
    /// the two "temperature" concepts (softmax-over-a-handful-of-candidate-scores
    /// vs. next-token-choice-over-the-whole-vocabulary) are different operations
    /// that happen to share a name.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Only used by the `Model::generate` path (empty `candidates`). Omitted (the
    /// default) is greedy argmax, byte-identical to this protocol's pre-sampling
    /// behavior -- see [`IpcSamplingParams`].
    #[serde(default)]
    pub sampling: Option<IpcSamplingParams>,
    /// Only used by the `Model::generate` path (empty `candidates`); ignored (as
    /// if `false`) for `system1_evaluate` (non-empty `candidates`). See this
    /// module's doc comment's "Streaming" section and
    /// [`handle_request_streaming`].
    #[serde(default)]
    pub stream: bool,
}

impl IpcRequest {
    /// Builds the `crate::sampling::SamplingParams` this request's `generate` call
    /// should use -- `crate::sampling::SamplingParams::default()` (greedy) when
    /// `sampling` is omitted.
    fn sampling_params(&self) -> crate::sampling::SamplingParams {
        match &self.sampling {
            None => crate::sampling::SamplingParams::default(),
            Some(s) => crate::sampling::SamplingParams {
                temperature: s.temperature,
                top_k: s.top_k,
                top_p: s.top_p,
                seed: s.seed,
            },
        }
    }
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
    /// `"final"` on every `IpcResponse` -- distinguishes this, the one-and-only
    /// line a non-streaming request gets, from an [`IpcStreamToken`]'s `"token"`
    /// on a streaming request's preceding lines. A client that only ever sends
    /// `"stream": false` (or omits it) can ignore this field entirely; it exists
    /// for a client that handles both request shapes over the same connection.
    pub event: &'static str,
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
        IpcResponse {
            id,
            event: "final",
            ok: false,
            error: Some(error),
            token_ids: None,
            text: None,
            candidates: None,
            entropy: None,
        }
    }
}

/// One incremental token emitted by a streaming (`"stream": true`)
/// `Model::generate` request -- see this module's doc comment's "Streaming"
/// section. `text` is that token's incrementally-decoded text
/// (`Tokenizer::decode_stream`), which may be empty (a multi-byte character's
/// leading byte(s), held back until the character completes on a later token --
/// see `Tokenizer::decode_stream`'s doc comment) even though a real token was
/// produced.
#[derive(Debug, Serialize)]
pub struct IpcStreamToken {
    pub id: Option<String>,
    /// `"token"` on every `IpcStreamToken`; see [`IpcResponse::event`].
    pub event: &'static str,
    pub token_id: u32,
    pub text: String,
}

/// Dispatches one already-parsed request to `Model::generate` or
/// `Model::system1_evaluate` and wraps the result (or error) as an
/// [`IpcResponse`]. Never panics -- every failure path from the underlying model
/// call becomes `ok: false` with `error` set, not a process exit, since a bad
/// request on a long-lived stdio/UDS session must not kill the whole session.
/// Ignores `req.stream` entirely -- this is the single-final-line path; see
/// [`handle_request_streaming`] for the streaming path.
pub fn handle_request(model: &Model, req: IpcRequest) -> IpcResponse {
    let id = req.id.clone();
    if req.candidates.is_empty() {
        let sampling = req.sampling_params();
        match model.generate(
            &req.prompt,
            req.max_tokens.max(1),
            None,
            &sampling,
            |_logits| {},
            |_token_id, _text| {},
        ) {
            Ok((token_ids, text)) => IpcResponse {
                id,
                event: "final",
                ok: true,
                error: None,
                token_ids: Some(token_ids),
                text: Some(text),
                candidates: None,
                entropy: None,
            },
            Err(e) => IpcResponse::err(id, e),
        }
    } else {
        let candidates: Vec<System1Candidate> = req
            .candidates
            .iter()
            .map(|text| System1Candidate { text: text.clone() })
            .collect();
        match model.system1_evaluate(&req.prompt, &candidates, req.temperature) {
            Ok(response) => {
                let candidates = response
                    .results
                    .into_iter()
                    .zip(response.probabilities)
                    .map(|(r, probability)| IpcCandidateResult {
                        text: r.text,
                        token_ids: r.token_ids,
                        score: r.score,
                        probability,
                    })
                    .collect();
                IpcResponse {
                    id,
                    event: "final",
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

/// Serializes `value` as one line of JSON to `output`, flushed immediately -- a
/// caller piping this over a pipe/socket must see each line as soon as it's
/// ready, not buffered until process exit or until a later line is written.
/// Shared by [`run_request_loop`]'s non-streaming write and
/// [`handle_request_streaming`]'s per-token writes, so both go through the same
/// flush-every-line discipline.
fn write_json_line<W: Write, T: Serialize>(output: &mut W, value: &T) -> Result<(), String> {
    let json =
        serde_json::to_string(value).map_err(|e| format!("ipc: serializing response: {e}"))?;
    writeln!(output, "{json}").map_err(|e| format!("ipc: writing response line: {e}"))?;
    output
        .flush()
        .map_err(|e| format!("ipc: flushing response: {e}"))
}

/// Like [`handle_request`], but honors `req.stream`: for the `Model::generate`
/// path (empty `candidates`) with `stream: true`, writes one [`IpcStreamToken`]
/// line per generated token as it's produced (via `Model::generate`'s `on_token`
/// hook -- see that method's doc comment), flushed immediately so a caller sees
/// each token as soon as it's ready rather than buffered until the whole
/// response is ready, then writes the same final [`IpcResponse`] line
/// [`handle_request`] would have written alone. Falls back to writing exactly
/// [`handle_request`]'s single line for every other case (`stream: false`/
/// omitted, or the `system1_evaluate` path regardless of `stream` -- a
/// single-pass score has nothing to stream incrementally). Still strictly
/// sequential and single-threaded: token lines are written from inside the same
/// synchronous decode loop `Model::generate` already runs, not from a
/// background task.
pub fn handle_request_streaming<W: Write>(
    model: &Model,
    req: IpcRequest,
    mut output: W,
) -> Result<(), String> {
    if !req.stream || !req.candidates.is_empty() {
        let response = handle_request(model, req);
        return write_json_line(&mut output, &response);
    }

    let id = req.id.clone();
    let sampling = req.sampling_params();
    let max_tokens = req.max_tokens.max(1);
    let mut write_err: Option<String> = None;
    let result = model.generate(
        &req.prompt,
        max_tokens,
        None,
        &sampling,
        |_logits| {},
        |token_id, text| {
            if write_err.is_some() {
                return; // A prior token's write already failed this call; stop trying.
            }
            let event = IpcStreamToken {
                id: id.clone(),
                event: "token",
                token_id,
                text: text.to_string(),
            };
            if let Err(e) = write_json_line(&mut output, &event) {
                write_err = Some(e);
            }
        },
    );
    if let Some(e) = write_err {
        return Err(e);
    }

    let response = match result {
        Ok((token_ids, text)) => IpcResponse {
            id,
            event: "final",
            ok: true,
            error: None,
            token_ids: Some(token_ids),
            text: Some(text),
            candidates: None,
            entropy: None,
        },
        Err(e) => IpcResponse::err(id, e),
    };
    write_json_line(&mut output, &response)
}

/// Reads one JSON request per line from `input`, dispatches it via
/// [`handle_request_streaming`] (which writes one or more JSON response lines to
/// `output` itself, each flushed immediately as it's written -- see that
/// function's doc comment for when it's one line vs. several). A line that fails
/// to parse as JSON writes a single `ok: false` response with no `id` (the
/// request itself was unreadable) instead of terminating the loop -- one
/// malformed line must not kill an otherwise-healthy long-lived session. Returns
/// only on EOF or a hard I/O error; strictly sequential, never spawns a thread.
pub fn run_request_loop<R: BufRead, W: Write>(
    model: &Model,
    mut input: R,
    mut output: W,
) -> Result<(), String> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = input
            .read_line(&mut line)
            .map_err(|e| format!("ipc: reading request line: {e}"))?;
        if bytes_read == 0 {
            return Ok(()); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<IpcRequest>(trimmed) {
            Ok(req) => handle_request_streaming(model, req, &mut output)?,
            Err(e) => {
                let response = IpcResponse::err(None, format!("ipc: malformed request JSON: {e}"));
                write_json_line(&mut output, &response)?;
            }
        }
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
        assert!(
            !json.contains("token_ids"),
            "error responses should omit null result fields: {json}"
        );
        assert!(
            !json.contains("candidates"),
            "error responses should omit null result fields: {json}"
        );
    }

    #[test]
    fn test_malformed_json_line_produces_error_response_not_a_loop_abort() {
        let model_free_response = match serde_json::from_str::<IpcRequest>("not json") {
            Ok(_) => panic!("expected parse failure"),
            Err(e) => IpcResponse::err(None, format!("ipc: malformed request JSON: {e}")),
        };
        assert!(!model_free_response.ok);
        assert!(model_free_response
            .error
            .unwrap()
            .contains("malformed request JSON"));
    }

    #[test]
    fn test_ipc_request_defaults_sampling_to_none_and_stream_to_false() {
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "hello"}"#).expect("should parse");
        assert!(req.sampling.is_none());
        assert!(!req.stream);
        // Omitted `sampling` must build a greedy `SamplingParams` -- byte-identical
        // to this protocol's pre-sampling behavior.
        assert!(req.sampling_params().is_greedy());
    }

    #[test]
    fn test_ipc_request_parses_sampling_and_stream_fields() {
        let req: IpcRequest = serde_json::from_str(
            r#"{"prompt": "hello", "stream": true, "sampling": {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "seed": 42}}"#,
        )
        .expect("should parse");
        assert!(req.stream);
        let sampling = req.sampling_params();
        assert!(!sampling.is_greedy());
        assert_eq!(sampling.temperature, 0.8);
        assert_eq!(sampling.top_k, Some(40));
        assert_eq!(sampling.top_p, Some(0.9));
        assert_eq!(sampling.seed, Some(42));
    }

    #[test]
    fn test_sampling_temperature_zero_is_still_greedy_even_when_object_present() {
        let req: IpcRequest =
            serde_json::from_str(r#"{"prompt": "hello", "sampling": {"top_k": 40}}"#)
                .expect("should parse");
        // `sampling` present but `temperature` omitted (defaults to 0.0) must
        // still select greedy -- presence of the object alone doesn't opt in.
        assert!(req.sampling_params().is_greedy());
    }

    #[test]
    fn test_ipc_response_final_event_and_stream_token_event_are_distinguishable() {
        let final_resp = IpcResponse::err(Some("id1".to_string()), "boom".to_string());
        assert_eq!(final_resp.event, "final");

        let token = IpcStreamToken {
            id: Some("id1".to_string()),
            event: "token",
            token_id: 7,
            text: "hi".to_string(),
        };
        let json = serde_json::to_string(&token).expect("should serialize");
        assert!(json.contains("\"event\":\"token\""));
        assert!(json.contains("\"token_id\":7"));
    }

    #[test]
    fn test_write_json_line_writes_one_flushed_line() {
        let mut buf: Vec<u8> = Vec::new();
        let resp = IpcResponse::err(None, "boom".to_string());
        write_json_line(&mut buf, &resp).expect("should write");
        let text = String::from_utf8(buf).expect("should be valid UTF-8");
        assert_eq!(
            text.matches('\n').count(),
            1,
            "should write exactly one line: {text:?}"
        );
        assert!(text.trim_end().starts_with('{') && text.trim_end().ends_with('}'));
    }
}
