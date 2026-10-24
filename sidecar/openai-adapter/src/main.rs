//! `reflex-openai-adapter`: a standalone OpenAI-compatible HTTP sidecar in front of
//! one managed `reflex stdio <gguf>` process. Implements `POST /v1/chat/completions`
//! (streaming via SSE and non-streaming JSON), `POST /v1/classify` (System1 label
//! scoring, not an OpenAI endpoint -- see `classify.rs`), `POST /v1/embeddings` (for
//! an embedding GGUF -- see `embeddings.rs`), `POST /v1/systemone` (a decision
//! model's typed answers, llama.cpp's TypeSafe-compatible API -- see `systemone.rs`),
//! `GET /v1/models`, and a health check
//! served at both `/healthz` and `/ping` (identical handler -- `/ping` exists because
//! Runpod Serverless load-balancing endpoints hard-poll that exact path, confirmed
//! against a real deployment) -- see this crate's README for usage, scope, and known
//! limitations, and the root README.md's Non-goals section for why this
//! lives in its own crate/process rather than inside the core engine.
//!
//! Usage: `reflex-openai-adapter <path-to-gguf> [--reflex-bin <path>] [--host
//! <addr>] [--port <port>] [--lora <adapter.gguf>] [--weights f16|f32] [--model-name <name>]
//! [--default-max-tokens <n>] [--no-chat-template] [--chat-template-file <path>]
//! [--owned-by <name>] [--pricing-prompt <str>] [--pricing-completion <str>]
//! [--region <str>] [--max-tokens-cap <n>] [--max-prompt-bytes <n>]
//! [--max-queue-depth <n>] [--request-timeout-secs <n>]`

mod chat_template;
mod classify;
mod embeddings;
mod gguf_meta;
mod openai;
mod reflex_client;
mod stop_tokens;
mod systemone;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chat_template::ChatTemplate;
use futures::Stream;
use openai::{
    build_ipc_line, build_prompt, estimate_prompt_tokens, extract_text_messages, finish_reason,
    ipc_json_schema, now_unix, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessageOut, Choice, ChunkChoice, Delta, ModelInfo, ModelPricing, ModelsResponse,
    RawMessage, TextMessage, Usage,
};
use reflex_client::ReflexClient;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

struct AppState {
    client: ReflexClient,
    model_label: String,
    default_max_tokens: usize,
    limits: RequestLimits,
    request_counter: AtomicU64,
    chat_template: Option<ChatTemplate>,
    /// The model's end-of-generation tokens: sent with every chat request, and
    /// their text kept out of the reply.
    stop_tokens: stop_tokens::StopTokens,
    model_info: ModelInfo,
    /// What `/v1/systemone` can do with this GGUF, decided at startup.
    systemone: systemone::Support,
}

impl AppState {
    fn next_chat_id(&self) -> String {
        let n = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-reflex-{n:x}")
    }

    fn next_classify_id(&self) -> String {
        let n = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("classify-reflex-{n:x}")
    }

    /// Renders one `/v1/embeddings` input: the GGUF's chat template over a system
    /// message holding the retrieval prefix (`query:`/`passage:`) and a user message
    /// holding the text (Nemotron-3.5-Embed's template gives `"query: <text>"` with
    /// its bracket markers), else [`embeddings::fallback_render`].
    fn render_embedding_input(&self, prefix: &str, text: &str) -> String {
        let messages = [
            TextMessage {
                role: "system".to_string(),
                content: format!("{prefix}:"),
            },
            TextMessage {
                role: "user".to_string(),
                content: text.to_string(),
            },
        ];
        match &self.chat_template {
            Some(ct) => match ct.render(&messages, false) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "[adapter] chat template render failed for an embeddings input ({e}); \
                         using \"{prefix}: <text>\" for it"
                    );
                    embeddings::fallback_render(prefix, text)
                }
            },
            None => embeddings::fallback_render(prefix, text),
        }
    }

    /// Renders chat messages into the engine's prompt: the GGUF's chat template
    /// (assistant turn opened) when one loaded, else the generic role-labeled
    /// flattening. Errs (for a `400`) on messages this adapter can't take.
    fn render_messages(&self, raw: &[RawMessage]) -> Result<String, String> {
        let messages = extract_text_messages(raw)?;
        Ok(match &self.chat_template {
            Some(ct) => match ct.render(&messages, true) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "[adapter] chat template render failed for this request ({e}); \
                         falling back to generic prompt flattening for this request"
                    );
                    build_prompt(&messages)
                }
            },
            None => build_prompt(&messages),
        })
    }
}

/// Per-request guards against one caller monopolizing the single, strictly
/// sequential engine behind this sidecar -- see the README's "Request limits".
#[derive(Clone, Copy, Debug, PartialEq)]
struct RequestLimits {
    /// Largest `max_tokens` a request may ask for; above it -> `400` (rejected, never
    /// silently clamped). The real protection against a long job blocking everyone.
    max_tokens_cap: usize,
    /// Largest rendered prompt, in bytes (this sidecar has no tokenizer, so bytes are
    /// the honest unit); above it -> `400`.
    max_prompt_bytes: usize,
    /// Most jobs queued or running in the engine at once; above it -> `429`.
    max_queue_depth: usize,
    /// How long the HTTP response waits for the engine. On expiry the client gets a
    /// `504` (or an error event mid-stream), but the engine job keeps running to
    /// completion -- there is no cancel operation.
    request_timeout: Duration,
}

impl Default for RequestLimits {
    fn default() -> Self {
        RequestLimits {
            max_tokens_cap: 2048,
            max_prompt_bytes: 262_144,
            max_queue_depth: 16,
            request_timeout: Duration::from_secs(300),
        }
    }
}

impl RequestLimits {
    /// Request-body cap handed to axum's `DefaultBodyLimit`, derived from
    /// `max_prompt_bytes`: the JSON body carries the messages plus role/field
    /// overhead and string escaping, so it gets twice the prompt budget plus 64 KiB.
    /// Bodies over it are rejected with `413` before being parsed at all.
    fn body_limit_bytes(&self) -> usize {
        self.max_prompt_bytes
            .saturating_mul(2)
            .saturating_add(64 * 1024)
    }
}

fn parse_positive(flag: &str, raw: &str) -> Result<usize, String> {
    match raw.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("{flag}: expected a positive integer, got {raw}")),
    }
}

struct Opts {
    gguf_path: String,
    reflex_bin: String,
    lora_path: Option<String>,
    /// Passed through to the child as `reflex stdio --weights <dtype>`; `None`
    /// leaves it to the child's own default (or its inherited `REFLEX_WEIGHTS`).
    weights: Option<String>,
    host: String,
    port: u16,
    model_label: Option<String>,
    default_max_tokens: usize,
    no_chat_template: bool,
    chat_template_file: Option<String>,
    owned_by: String,
    pricing_prompt: String,
    pricing_completion: String,
    region: Option<String>,
    limits: RequestLimits,
}

impl Opts {
    fn parse(args: Vec<String>) -> Result<Opts, String> {
        let mut gguf_path: Option<String> = None;
        let mut reflex_bin = "reflex".to_string();
        let mut lora_path: Option<String> = None;
        let mut weights: Option<String> = None;
        let mut host = "127.0.0.1".to_string();
        let mut port: u16 = 8000;
        let mut model_label: Option<String> = None;
        let mut default_max_tokens: usize = 256;
        let mut no_chat_template = false;
        let mut chat_template_file: Option<String> = None;
        let mut owned_by = "reflex".to_string();
        let mut pricing_prompt = "0".to_string();
        let mut pricing_completion = "0".to_string();
        let mut region: Option<String> = None;
        let mut limits = RequestLimits::default();

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--reflex-bin" => {
                    reflex_bin = args.next().ok_or("--reflex-bin requires a path")?;
                }
                "--lora" => {
                    lora_path = Some(args.next().ok_or("--lora requires a file path")?);
                }
                "--weights" => {
                    let raw = args.next().ok_or("--weights requires `f16` or `f32`")?;
                    if raw != "f16" && raw != "f32" {
                        return Err(format!("--weights must be `f16` or `f32`, got {raw:?}"));
                    }
                    weights = Some(raw);
                }
                "--host" => {
                    host = args.next().ok_or("--host requires an address")?;
                }
                "--port" => {
                    let raw = args.next().ok_or("--port requires a number")?;
                    port = raw
                        .parse()
                        .map_err(|_| format!("--port: not a valid port number: {raw}"))?;
                }
                "--model-name" => {
                    model_label = Some(args.next().ok_or("--model-name requires a value")?);
                }
                "--default-max-tokens" => {
                    let raw = args
                        .next()
                        .ok_or("--default-max-tokens requires a number")?;
                    default_max_tokens = raw
                        .parse()
                        .map_err(|_| format!("--default-max-tokens: not a valid number: {raw}"))?;
                }
                "--no-chat-template" => {
                    no_chat_template = true;
                }
                "--chat-template-file" => {
                    chat_template_file =
                        Some(args.next().ok_or("--chat-template-file requires a path")?);
                }
                "--owned-by" => {
                    owned_by = args.next().ok_or("--owned-by requires a value")?;
                }
                "--pricing-prompt" => {
                    pricing_prompt = args.next().ok_or("--pricing-prompt requires a value")?;
                }
                "--pricing-completion" => {
                    pricing_completion =
                        args.next().ok_or("--pricing-completion requires a value")?;
                }
                "--region" => {
                    region = Some(args.next().ok_or("--region requires a value")?);
                }
                "--max-tokens-cap" => {
                    let raw = args.next().ok_or("--max-tokens-cap requires a number")?;
                    limits.max_tokens_cap = parse_positive("--max-tokens-cap", &raw)?;
                }
                "--max-prompt-bytes" => {
                    let raw = args.next().ok_or("--max-prompt-bytes requires a number")?;
                    limits.max_prompt_bytes = parse_positive("--max-prompt-bytes", &raw)?;
                }
                "--max-queue-depth" => {
                    let raw = args.next().ok_or("--max-queue-depth requires a number")?;
                    limits.max_queue_depth = parse_positive("--max-queue-depth", &raw)?;
                }
                "--request-timeout-secs" => {
                    let raw = args
                        .next()
                        .ok_or("--request-timeout-secs requires a number")?;
                    limits.request_timeout =
                        Duration::from_secs(parse_positive("--request-timeout-secs", &raw)? as u64);
                }
                _ if gguf_path.is_none() => gguf_path = Some(arg),
                other => return Err(format!("unexpected argument: {other}")),
            }
        }

        let gguf_path = gguf_path.ok_or(
            "usage: reflex-openai-adapter <path-to-gguf> [--reflex-bin <path>] [--host <addr>] \
             [--port <port>] [--lora <adapter.gguf>] [--weights f16|f32] [--model-name <name>] \
             [--default-max-tokens <n>] [--no-chat-template] [--chat-template-file <path>] \
             [--owned-by <name>] [--pricing-prompt <str>] [--pricing-completion <str>] \
             [--region <str>] [--max-tokens-cap <n>] [--max-prompt-bytes <n>] \
             [--max-queue-depth <n>] [--request-timeout-secs <n>]",
        )?;

        if no_chat_template && chat_template_file.is_some() {
            return Err(
                "--no-chat-template and --chat-template-file are mutually exclusive".to_string(),
            );
        }
        if default_max_tokens > limits.max_tokens_cap {
            return Err(format!(
                "--default-max-tokens ({default_max_tokens}) exceeds --max-tokens-cap ({}); \
                 every request that omits max_tokens would be rejected",
                limits.max_tokens_cap
            ));
        }

        Ok(Opts {
            gguf_path,
            reflex_bin,
            lora_path,
            weights,
            host,
            port,
            model_label,
            default_max_tokens,
            no_chat_template,
            chat_template_file,
            owned_by,
            pricing_prompt,
            pricing_completion,
            region,
            limits,
        })
    }
}

/// Reads `<arch>.context_length` from the GGUF's own metadata (the same
/// arch-prefixed-key convention the core engine's `parse_model_config` uses, per the
/// root docs/DEVELOPMENT.md) via the standalone `gguf_meta` reader -- best-effort: returns
/// `None` rather than failing startup if the architecture or key is missing, since
/// `context_length` is an informational extra for `/v1/models`, not required for the
/// adapter to function.
fn read_context_length(gguf_path: &str) -> Option<u64> {
    let meta = gguf_meta::GgufMeta::open(gguf_path).ok()?;
    let architecture = meta.get_str("general.architecture")?;
    meta.get_u64(&format!("{architecture}.context_length"))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match Opts::parse(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let model_label = opts.model_label.clone().unwrap_or_else(|| {
        std::path::Path::new(&opts.gguf_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| opts.gguf_path.clone())
    });

    eprintln!(
        "[adapter] launching `{} stdio {}`...",
        opts.reflex_bin, opts.gguf_path
    );
    let client = match ReflexClient::spawn(
        &opts.reflex_bin,
        &opts.gguf_path,
        opts.lora_path.as_deref(),
        opts.weights.as_deref(),
        opts.limits.max_queue_depth,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[adapter] failed to launch reflex: {e:#}");
            std::process::exit(1);
        }
    };

    // Loaded/render-tested once at boot, not per-request -- see
    // `chat_template::build`'s doc comment for the fallback rules.
    let chat_template = if opts.no_chat_template {
        eprintln!("[adapter] chat template: disabled via --no-chat-template; using generic prompt flattening");
        None
    } else {
        chat_template::build(&opts.gguf_path, opts.chat_template_file.as_deref())
    };
    let stop_tokens = stop_tokens::StopTokens::from_gguf(&opts.gguf_path);
    let systemone = systemone::load(&opts.gguf_path);

    let model_info = ModelInfo {
        id: model_label.clone(),
        object: "model",
        created: now_unix(),
        owned_by: opts.owned_by,
        context_length: read_context_length(&opts.gguf_path),
        pricing: ModelPricing {
            prompt: opts.pricing_prompt,
            completion: opts.pricing_completion,
        },
        datacenter_location: opts.region,
    };

    let state = Arc::new(AppState {
        client,
        model_label,
        default_max_tokens: opts.default_max_tokens,
        limits: opts.limits,
        request_counter: AtomicU64::new(0),
        chat_template,
        stop_tokens,
        model_info,
        systemone,
    });

    // Fails the process fast and observably the moment the managed `reflex` child is
    // gone, instead of a sidecar that silently keeps returning "ok" from `/healthz`
    // forever -- see `reflex_client::ReflexClient::is_child_alive`'s doc comment and
    // this crate's README "Known limitations". Recovery is deliberately left to the
    // outer process supervisor (Docker `--restart`, or ALB/ASG health-check instance
    // replacement), not reimplemented here.
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if !state.client.is_child_alive() {
                    eprintln!(
                        "[adapter] managed reflex process is gone; exiting so the process \
                         supervisor can restart this sidecar"
                    );
                    std::process::exit(1);
                }
            }
        });
    }

    let body_limit = opts.limits.body_limit_bytes();
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(DefaultBodyLimit::max(body_limit)),
        )
        .route(
            "/v1/classify",
            post(classify_handler).layer(DefaultBodyLimit::max(body_limit)),
        )
        .route(
            "/v1/embeddings",
            post(embeddings_handler).layer(DefaultBodyLimit::max(body_limit)),
        )
        .route(
            "/v1/systemone",
            post(systemone_handler).layer(DefaultBodyLimit::max(body_limit)),
        )
        .route("/v1/models", get(list_models))
        .route("/healthz", get(healthz))
        // Alias of /healthz, same handler: Runpod Serverless load-balancing endpoints
        // hard-poll `/ping` for worker health regardless of the documented
        // `HEALTH_CHECK_PATH` override -- confirmed against a real deployment, where
        // a worker that loaded correctly and reported /healthz=200 never received any
        // traffic because Runpod's gateway was polling /ping (404, unhandled) the
        // whole time. Costs nothing to serve both paths from the same state.
        .route("/ping", get(healthz))
        .with_state(state);

    let addr = format!("{}:{}", opts.host, opts.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[adapter] failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("REFLEX_ADAPTER_READY addr={addr}");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("[adapter] server error: {e}");
        std::process::exit(1);
    }
}

/// Three-state health check backed by `ReflexClient`'s liveness/readiness flags,
/// instead of an unconditional `"ok"` -- see the watchdog installed in `main()` and
/// `reflex_client.rs`'s doc comment:
/// - `503`/JSON: the managed `reflex` process is gone (dead child).
/// - `204` (no body): the process is alive but still loading the model -- distinct
///   from "dead" so a caller that measures cold-start duration from this endpoint
///   (e.g. a RunPod Serverless load-balancing endpoint, which explicitly treats `204`
///   as "initializing" and `200` as "healthy") gets an accurate readiness signal
///   instead of a false-positive "ready" the moment this HTTP server's port is bound,
///   which happens well before `reflex stdio`'s own model load finishes.
/// - `200`/plain "ok": alive and ready to serve a request.
async fn healthz(State(state): State<Arc<AppState>>) -> Response {
    if !state.client.is_child_alive() {
        error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "managed reflex process is not running",
        )
    } else if !state.client.is_ready() {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (StatusCode::OK, "ok").into_response()
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![state.model_info.clone()],
    })
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    error_response_with(status, message, "reflex_adapter_error", None)
}

fn error_response_with(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &str,
    code: Option<&str>,
) -> Response {
    (status, Json(error_body(message, error_type, code))).into_response()
}

fn error_body(message: impl Into<String>, error_type: &str, code: Option<&str>) -> Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "code": code,
        }
    })
}

/// `400` for a request that breaks one of the configured [`RequestLimits`].
fn limit_exceeded_response(message: String, code: &str) -> Response {
    error_response_with(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        Some(code),
    )
}

/// The `400` rejection for a request over the size limits, if any (queue depth is
/// checked separately, at enqueue time). `prompt_bytes` is the rendered prompt's
/// length.
fn request_limit_violation(
    limits: &RequestLimits,
    prompt_bytes: usize,
    max_tokens: usize,
) -> Option<Response> {
    if prompt_bytes > limits.max_prompt_bytes {
        return Some(limit_exceeded_response(
            format!(
                "rendered prompt is {prompt_bytes} bytes, over this server's limit of {} bytes \
                 (--max-prompt-bytes)",
                limits.max_prompt_bytes
            ),
            "context_length_exceeded",
        ));
    }
    if max_tokens > limits.max_tokens_cap {
        return Some(limit_exceeded_response(
            format!(
                "max_tokens is {max_tokens}, over this server's limit of {} (--max-tokens-cap)",
                limits.max_tokens_cap
            ),
            "max_tokens_exceeded",
        ));
    }
    None
}

fn queue_full_response(max_in_flight: usize) -> Response {
    let mut response = error_response_with(
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "server busy: {max_in_flight} requests are already queued or running \
             (--max-queue-depth); retry shortly"
        ),
        "rate_limit_error",
        Some("queue_full"),
    );
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
    response
}

fn timeout_message(timeout: Duration) -> String {
    format!(
        "reflex did not finish within {}s (--request-timeout-secs); the job may still \
         be running in the engine",
        timeout.as_secs()
    )
}

fn timeout_response(timeout: Duration) -> Response {
    error_response_with(
        StatusCode::GATEWAY_TIMEOUT,
        timeout_message(timeout),
        "timeout_error",
        Some("request_timeout"),
    )
}

/// Stable prefix of the engine's context-length error
/// (`reflex_engine::limits::CONTEXT_LENGTH_EXCEEDED_PREFIX` -- mirrored here, not
/// imported, since this crate deliberately doesn't depend on `reflex-engine`).
const CONTEXT_LENGTH_EXCEEDED_PREFIX: &str = "context length exceeded";

/// An engine failure, from an IPC `final` event with `ok: false`.
#[derive(Debug, PartialEq)]
struct EngineError<'a> {
    message: &'a str,
    /// The engine's stable error category (IPC `error_kind`, e.g.
    /// `"context_overflow"`). `None` from an engine build that predates it.
    kind: Option<&'a str>,
}

/// Maps an engine failure to the HTTP status and OpenAI-style error `type`/`code`
/// the client should see, by the engine's `error_kind`:
/// - `context_overflow` -> `400` `context_length_exceeded`, as OpenAI returns
///   (also recognized by message prefix, for engines without `error_kind`);
/// - `invalid_input` -> `400` (the adapter already validates `max_tokens`, so this
///   is a bad request parameter such as a sampling value, or a `response_format`
///   schema outside the engine's subset);
/// - `grammar_dead_end` -> `400`: a `response_format` request's `max_tokens` ran out
///   before the JSON value was complete (the engine fails the request rather than
///   return a value that doesn't parse);
/// - `out_of_memory` -> `503`: the server is temporarily out of GPU memory;
/// - anything else -> `500`, with the kind as `code` when the engine sent one.
fn classify_engine_error(err: &EngineError) -> (StatusCode, &'static str, Option<String>) {
    let kind = err.kind.or_else(|| {
        err.message
            .starts_with(CONTEXT_LENGTH_EXCEEDED_PREFIX)
            .then_some("context_overflow")
    });
    match kind {
        Some("context_overflow") => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded".to_string()),
        ),
        Some("invalid_input") => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("invalid_input".to_string()),
        ),
        Some("grammar_dead_end") => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("grammar_dead_end".to_string()),
        ),
        Some("out_of_memory") => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            Some("out_of_memory".to_string()),
        ),
        other => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "reflex_adapter_error",
            other.map(str::to_string),
        ),
    }
}

fn engine_error_response(err: &EngineError) -> Response {
    let (status, error_type, code) = classify_engine_error(err);
    error_response_with(status, err.message, error_type, code.as_deref())
}

/// The SSE data for an engine failure after the stream has started: the body
/// [`engine_error_response`] would have sent, without the status.
fn stream_error_body(err: &EngineError) -> Value {
    let (_, error_type, code) = classify_engine_error(err);
    error_body(err.message, error_type, code.as_deref())
}

/// `Some` if `v` is the engine's terminal `final` event reporting failure.
fn engine_error(v: &Value) -> Option<EngineError<'_>> {
    let is_final = v.get("event").and_then(Value::as_str) == Some("final");
    let ok = v.get("ok").and_then(Value::as_bool).unwrap_or(false);
    (is_final && !ok).then(|| EngineError {
        message: v
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown reflex error"),
        kind: v.get("error_kind").and_then(Value::as_str),
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        // The rejection's own status: 400/422 for bad JSON, 413 over the body limit.
        Err(e) => return error_response(e.status(), e.body_text()),
    };

    let prompt = match state.render_messages(&req.messages) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens).max(1);
    if let Some(response) = request_limit_violation(&state.limits, prompt.len(), max_tokens) {
        return response;
    }
    let json_schema = match ipc_json_schema(req.response_format.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                e,
                "invalid_request_error",
                Some("invalid_input"),
            )
        }
    };
    let stream = req.stream;
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_label.clone());

    let line = match build_ipc_line(
        &req,
        &prompt,
        max_tokens,
        json_schema.as_deref(),
        &state.stop_tokens.ids(),
    ) {
        Ok(l) => l,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sidecar: failed to serialize IPC request: {e}"),
            )
        }
    };

    let mut rx = match state.client.request(line).await {
        Ok(rx) => rx,
        Err(full) => return queue_full_response(full.max_in_flight),
    };
    let chat_id = state.next_chat_id();
    let created = now_unix();
    // Dropping `rx` on timeout is safe: the worker keeps draining the job's output
    // (see `reflex_client::ReflexClient::request`).
    let timeout = state.limits.request_timeout;
    let deadline = Instant::now() + timeout;

    if stream {
        // Wait for the engine's first event before committing to a `200` SSE
        // response, so an error raised before any token is produced (e.g. the
        // prompt is over the engine's context-length limit) still reaches the
        // client as a real HTTP status instead of an empty stream.
        let first = match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(first) => first,
            Err(_) => return timeout_response(timeout),
        };
        if let Some(err) = first.as_ref().and_then(engine_error) {
            return engine_error_response(&err);
        }
        let sse_stream = build_sse_stream(
            first,
            rx,
            chat_id,
            created,
            model_label,
            max_tokens,
            deadline,
            timeout,
            state.stop_tokens.clone(),
        );
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response = non_streaming_response(
            rx,
            chat_id,
            created,
            model_label,
            prompt,
            max_tokens,
            state.stop_tokens.clone(),
        );
        match tokio::time::timeout_at(deadline, response).await {
            Ok(response) => response,
            Err(_) => timeout_response(timeout),
        }
    }
}

/// `POST /v1/classify`: scores `labels` as continuations of the prompt with one
/// engine pass and returns their probabilities. See `classify.rs` for the request
/// and response shapes. Shares the chat endpoint's limits: `--max-prompt-bytes`
/// (prompt plus longest label), `--max-queue-depth` and `--request-timeout-secs`.
async fn classify_handler(
    State(state): State<Arc<AppState>>,
    body: Result<Json<classify::ClassifyRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return error_response(e.status(), e.body_text()),
    };
    let prompt = match classify::validate(&req) {
        Ok(classify::PromptSource::Raw(p)) => p.to_string(),
        Ok(classify::PromptSource::Messages(m)) => match state.render_messages(m) {
            Ok(p) => p,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
        },
        Err(e) => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                e,
                "invalid_request_error",
                Some("invalid_input"),
            )
        }
    };
    let bytes = classify::request_bytes(&prompt, &req.labels);
    if let Some(response) = request_limit_violation(&state.limits, bytes, 0) {
        return response;
    }
    let line = match serde_json::to_string(&classify::ipc_request(&prompt, &req)) {
        Ok(l) => l,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sidecar: failed to serialize IPC request: {e}"),
            )
        }
    };
    let mut rx = match state.client.request(line).await {
        Ok(rx) => rx,
        Err(full) => return queue_full_response(full.max_in_flight),
    };
    let id = state.next_classify_id();
    let model_label = req.model.unwrap_or_else(|| state.model_label.clone());
    let timeout = state.limits.request_timeout;
    // Dropping `rx` on timeout is safe, as for chat completions.
    let response = async move {
        while let Some(v) = rx.recv().await {
            if v.get("event").and_then(Value::as_str) != Some("final") {
                continue;
            }
            if let Some(err) = engine_error(&v) {
                return engine_error_response(&err);
            }
            return match classify::build_response(&v, id, now_unix(), model_label) {
                Ok(r) => (StatusCode::OK, Json(r)).into_response(),
                Err(e) => {
                    error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("sidecar: {e}"))
                }
            };
        }
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sidecar: reflex closed the connection without a final response",
        )
    };
    match tokio::time::timeout_at(Instant::now() + timeout, response).await {
        Ok(response) => response,
        Err(_) => timeout_response(timeout),
    }
}

/// `POST /v1/systemone`: answers a decision model's typed questions; see
/// `systemone.rs`. Every variant of every question goes to the engine in one IPC
/// request, which evaluates their shared token prefix once. `--max-prompt-bytes`
/// applies to each rendered prompt; the queue depth and request timeout as for
/// the other endpoints.
async fn systemone_handler(
    State(state): State<Arc<AppState>>,
    body: Result<Json<systemone::Json>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let model = match &state.systemone {
        systemone::Support::Ready(m) => m,
        systemone::Support::NotDecisionModel => {
            return not_supported_response("This model is not a decision model")
        }
        systemone::Support::Unsupported(why) => return not_supported_response(why),
    };
    let Json(body) = match body {
        Ok(j) => j,
        Err(e) => return error_response(e.status(), e.body_text()),
    };
    let parsed = match model.parse(&body) {
        Ok(p) => p,
        Err(systemone::RequestError::Invalid(e)) => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                e,
                "invalid_request_error",
                Some("invalid_input"),
            )
        }
        Err(systemone::RequestError::ImagesUnsupported) => {
            return not_supported_response(systemone::IMAGES_UNSUPPORTED)
        }
    };
    let plan = match model.plan(&parsed) {
        Ok(p) => p,
        Err(e) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("sidecar: {e}"))
        }
    };
    for p in &plan.prompts {
        if let Some(response) = request_limit_violation(&state.limits, p.len(), 0) {
            return response;
        }
    }
    let line = match serde_json::to_string(&systemone::DecisionModel::ipc_request(&plan)) {
        Ok(l) => l,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sidecar: failed to serialize IPC request: {e}"),
            )
        }
    };
    let mut rx = match state.client.request(line).await {
        Ok(rx) => rx,
        Err(full) => return queue_full_response(full.max_in_flight),
    };
    let model_label = body
        .get("model")
        .and_then(|m| match m {
            systemone::Json::Str(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| state.model_label.clone());
    let timeout = state.limits.request_timeout;
    let response = async move {
        while let Some(v) = rx.recv().await {
            if v.get("event").and_then(Value::as_str) != Some("final") {
                continue;
            }
            if let Some(err) = engine_error(&v) {
                return engine_error_response(&err);
            }
            return match model.build_response(&parsed, &v, model_label) {
                Ok(r) => (StatusCode::OK, Json(r)).into_response(),
                Err(e) => {
                    error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("sidecar: {e}"))
                }
            };
        }
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sidecar: reflex closed the connection without a final response",
        )
    };
    match tokio::time::timeout_at(Instant::now() + timeout, response).await {
        Ok(response) => response,
        Err(_) => timeout_response(timeout),
    }
}

/// `501`, as llama.cpp answers `/v1/systemone` for a model or input it can't take.
fn not_supported_response(message: &str) -> Response {
    error_response_with(
        StatusCode::NOT_IMPLEMENTED,
        message,
        "not_supported_error",
        Some("not_supported"),
    )
}

/// `POST /v1/embeddings`; see `embeddings.rs`. Inputs go to the engine one at a
/// time, in order, each as its own IPC request; the request timeout covers all of
/// them.
async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    body: Result<Json<embeddings::EmbeddingsRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return error_response(e.status(), e.body_text()),
    };
    let valid = match embeddings::validate(&req) {
        Ok(v) => v,
        Err(e) => {
            return error_response_with(
                StatusCode::BAD_REQUEST,
                e,
                "invalid_request_error",
                Some("invalid_input"),
            )
        }
    };
    let rendered: Vec<String> = valid
        .texts
        .iter()
        .map(|t| state.render_embedding_input(valid.prefix, t))
        .collect();
    for r in &rendered {
        if let Some(response) = request_limit_violation(&state.limits, r.len(), 0) {
            return response;
        }
    }
    let model_label = req.model.unwrap_or_else(|| state.model_label.clone());
    let timeout = state.limits.request_timeout;
    let work = async move {
        let mut data = Vec::with_capacity(rendered.len());
        let mut prompt_tokens = 0u32;
        for (index, text) in rendered.iter().enumerate() {
            let line = match serde_json::to_string(&embeddings::ipc_request(text)) {
                Ok(l) => l,
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("sidecar: failed to serialize IPC request: {e}"),
                    )
                }
            };
            let mut rx = match state.client.request(line).await {
                Ok(rx) => rx,
                Err(full) => return queue_full_response(full.max_in_flight),
            };
            let mut final_event = None;
            while let Some(v) = rx.recv().await {
                if v.get("event").and_then(Value::as_str) == Some("final") {
                    final_event = Some(v);
                    break;
                }
            }
            let Some(v) = final_event else {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "sidecar: reflex closed the connection without a final response",
                );
            };
            if let Some(err) = engine_error(&v) {
                return engine_error_response(&err);
            }
            let (vector, tokens) = match embeddings::parse_engine_embedding(&v) {
                Ok(x) => x,
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("sidecar: {e}"),
                    )
                }
            };
            let vector = match embeddings::truncate(vector, valid.dimensions) {
                Ok(v) => v,
                Err(e) => {
                    return error_response_with(
                        StatusCode::BAD_REQUEST,
                        e,
                        "invalid_request_error",
                        Some("invalid_input"),
                    )
                }
            };
            prompt_tokens += tokens;
            data.push(embeddings::EmbeddingData {
                object: "embedding",
                index,
                embedding: if valid.base64 {
                    embeddings::EmbeddingValue::Base64(embeddings::base64_f32(&vector))
                } else {
                    embeddings::EmbeddingValue::Float(vector)
                },
            });
        }
        let response = embeddings::EmbeddingsResponse {
            object: "list",
            data,
            model: model_label,
            usage: embeddings::EmbeddingsUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
        };
        (StatusCode::OK, Json(response)).into_response()
    };
    match tokio::time::timeout_at(Instant::now() + timeout, work).await {
        Ok(response) => response,
        Err(_) => timeout_response(timeout),
    }
}

async fn non_streaming_response(
    mut rx: tokio::sync::mpsc::Receiver<Value>,
    chat_id: String,
    created: u64,
    model_label: String,
    prompt: String,
    requested_max_tokens: usize,
    stop_tokens: stop_tokens::StopTokens,
) -> Response {
    while let Some(v) = rx.recv().await {
        let event = v.get("event").and_then(Value::as_str).unwrap_or("");
        if event != "final" {
            continue; // shouldn't happen for a non-streaming IPC request, but ignore rather than choke on it
        }
        if let Some(err) = engine_error(&v) {
            return engine_error_response(&err);
        }
        let token_ids = v.get("token_ids").and_then(Value::as_array);
        let last_id = token_ids
            .and_then(|a| a.last())
            .and_then(Value::as_u64)
            .map(|id| id as u32);
        let text = stop_tokens
            .strip(v.get("text").and_then(Value::as_str).unwrap_or(""), last_id)
            .to_string();
        let completion_tokens = token_ids.map(|a| a.len()).unwrap_or(0);
        let prompt_tokens = estimate_prompt_tokens(&prompt);
        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion",
            created,
            model: model_label,
            choices: vec![Choice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant",
                    content: text,
                },
                finish_reason: finish_reason(completion_tokens, requested_max_tokens),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: completion_tokens as u32,
                total_tokens: prompt_tokens + completion_tokens as u32,
            },
        };
        return (StatusCode::OK, Json(response)).into_response();
    }
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "sidecar: reflex closed the connection without a final response",
    )
}

/// `first` is the engine event `chat_completions` already received (to rule out
/// an up-front error) before starting the stream; it's replayed ahead of `rx`.
/// Past `deadline` the stream sends one OpenAI-style `{"error": ...}` event and
/// ends (headers are already sent, so a status code is no longer possible).
#[allow(clippy::too_many_arguments)]
fn build_sse_stream(
    first: Option<Value>,
    mut rx: tokio::sync::mpsc::Receiver<Value>,
    chat_id: String,
    created: u64,
    model_label: String,
    requested_max_tokens: usize,
    deadline: Instant,
    timeout: Duration,
    stop_tokens: stop_tokens::StopTokens,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let role_chunk = ChatCompletionChunk {
            id: chat_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_label.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap()));

        let mut completion_tokens = 0usize;
        let mut pending = first;
        loop {
            let v = match pending.take() {
                Some(v) => v,
                None => match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(v)) => v,
                    Ok(None) => break,
                    Err(_) => {
                        eprintln!("[adapter] stream timed out after {}s", timeout.as_secs());
                        let body = error_body(
                            timeout_message(timeout),
                            "timeout_error",
                            Some("request_timeout"),
                        );
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                },
            };
            let event = v.get("event").and_then(Value::as_str).unwrap_or("");
            if event == "token" {
                completion_tokens += 1;
                let token_id = v.get("token_id").and_then(Value::as_u64).map(|id| id as u32);
                if token_id.is_some_and(|id| stop_tokens.contains(id)) {
                    continue; // the stop token: its text isn't part of the reply
                }
                let text = v.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                let chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_label.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta { role: None, content: Some(text) },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            } else if event == "final" {
                if let Some(err) = engine_error(&v) {
                    // Headers are already sent, so the failure goes out as an
                    // OpenAI-style error event, as on a timeout, not as a finish
                    // chunk: a `response_format` stream that hit `max_tokens`
                    // (`grammar_dead_end`) must not look like a complete value.
                    eprintln!("[adapter] reflex error mid-stream: {}", err.message);
                    yield Ok(Event::default().data(stream_error_body(&err).to_string()));
                    break;
                }
                let token_ids = v
                    .get("token_ids")
                    .and_then(Value::as_array)
                    .map(|a| a.len())
                    .unwrap_or(completion_tokens);
                let reason = finish_reason(token_ids, requested_max_tokens);
                let final_chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_label.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta { role: None, content: None },
                        finish_reason: Some(reason),
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
                break;
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Opts, String> {
        let mut args = vec!["model.gguf".to_string()];
        args.extend(extra.iter().map(|s| s.to_string()));
        Opts::parse(args)
    }

    async fn error_json(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn request_limit_flags_default_and_parse() {
        assert_eq!(parse(&[]).unwrap().limits, RequestLimits::default());
        let o = parse(&[
            "--max-tokens-cap",
            "512",
            "--max-prompt-bytes",
            "1000",
            "--max-queue-depth",
            "4",
            "--request-timeout-secs",
            "30",
        ])
        .unwrap();
        assert_eq!(
            o.limits,
            RequestLimits {
                max_tokens_cap: 512,
                max_prompt_bytes: 1000,
                max_queue_depth: 4,
                request_timeout: Duration::from_secs(30),
            }
        );
    }

    #[test]
    fn request_limit_flags_reject_zero_garbage_and_missing_values() {
        for flag in [
            "--max-tokens-cap",
            "--max-prompt-bytes",
            "--max-queue-depth",
            "--request-timeout-secs",
        ] {
            assert!(parse(&[flag, "0"]).is_err(), "{flag} 0");
            assert!(parse(&[flag, "-3"]).is_err(), "{flag} -3");
            assert!(parse(&[flag, "lots"]).is_err(), "{flag} lots");
            assert!(parse(&[flag]).is_err(), "{flag} with no value");
        }
    }

    #[test]
    fn default_max_tokens_above_cap_is_a_startup_error() {
        let err = parse(&["--default-max-tokens", "600", "--max-tokens-cap", "512"])
            .err()
            .unwrap();
        assert!(err.contains("--max-tokens-cap"), "{err}");
        assert!(parse(&["--default-max-tokens", "512", "--max-tokens-cap", "512"]).is_ok());
    }

    #[test]
    fn body_limit_covers_a_max_size_prompt() {
        let limits = RequestLimits::default();
        assert!(limits.body_limit_bytes() > limits.max_prompt_bytes);
        let huge = RequestLimits {
            max_prompt_bytes: usize::MAX,
            ..limits
        };
        assert_eq!(huge.body_limit_bytes(), usize::MAX);
    }

    #[tokio::test]
    async fn requests_at_the_limits_pass_and_one_over_gets_openai_shaped_400() {
        let limits = RequestLimits {
            max_tokens_cap: 100,
            max_prompt_bytes: 1000,
            ..RequestLimits::default()
        };
        assert!(request_limit_violation(&limits, 1000, 100).is_none());

        let response = request_limit_violation(&limits, 1001, 100).unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = error_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "context_length_exceeded");

        let response = request_limit_violation(&limits, 1000, 101).unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = error_json(response).await;
        assert_eq!(body["error"]["code"], "max_tokens_exceeded");
        assert!(body["error"]["message"].as_str().unwrap().contains("101"));
    }

    #[tokio::test]
    async fn queue_full_is_429_with_retry_after() {
        let response = queue_full_response(16);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body = error_json(response).await;
        assert_eq!(body["error"]["code"], "queue_full");
    }

    #[tokio::test]
    async fn timeout_is_504_openai_shaped() {
        let response = timeout_response(Duration::from_secs(300));
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = error_json(response).await;
        assert_eq!(body["error"]["code"], "request_timeout");
        assert!(body["error"]["message"].as_str().unwrap().contains("300s"));
    }

    fn engine(message: &'static str, kind: Option<&'static str>) -> EngineError<'static> {
        EngineError { message, kind }
    }

    #[test]
    fn context_overflow_maps_to_400_context_length_exceeded() {
        let expected = (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded".to_string()),
        );
        let msg = "context length exceeded: request needs 12000 positions";
        assert_eq!(
            classify_engine_error(&engine(msg, Some("context_overflow"))),
            expected
        );
        // An engine build without `error_kind`: recognized by the message prefix.
        assert_eq!(classify_engine_error(&engine(msg, None)), expected);
    }

    #[test]
    fn out_of_memory_maps_to_503_and_invalid_input_to_400() {
        assert_eq!(
            classify_engine_error(&engine("attn alloc out: OOM", Some("out_of_memory"))).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            classify_engine_error(&engine("top_p must be in (0, 1]", Some("invalid_input"))),
            (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("invalid_input".to_string())
            )
        );
    }

    #[test]
    fn other_engine_errors_stay_500_with_the_kind_as_code() {
        assert_eq!(
            classify_engine_error(&engine(
                "attn launch: CUDA_ERROR_LAUNCH_FAILED",
                Some("cuda")
            )),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "reflex_adapter_error",
                Some("cuda".to_string())
            )
        );
        assert_eq!(
            classify_engine_error(&engine("boom", None)),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "reflex_adapter_error",
                None
            )
        );
    }

    #[test]
    fn schema_errors_map_to_400_invalid_request_error() {
        // What the engine sends for a schema outside its subset, and for a
        // `response_format` request whose `max_tokens` ran out mid-value.
        assert_eq!(
            classify_engine_error(&engine(
                "invalid input: json_schema: unsupported keyword \"pattern\" at #/properties/a",
                Some("invalid_input")
            )),
            (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("invalid_input".to_string())
            )
        );
        assert_eq!(
            classify_engine_error(&engine(
                "json_schema: max_tokens ran out after 8 tokens, before the value was complete",
                Some("grammar_dead_end")
            )),
            (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("grammar_dead_end".to_string())
            )
        );
    }

    /// Runs `build_sse_stream` over canned engine events and returns its `data:`
    /// payloads.
    async fn sse_data(events: Vec<Value>, max_tokens: usize) -> Vec<String> {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let mut events = events.into_iter();
        let first = events.next();
        for v in events {
            tx.send(v).await.unwrap();
        }
        drop(tx);
        let stream = build_sse_stream(
            first,
            rx,
            "c".into(),
            0,
            "m".into(),
            max_tokens,
            Instant::now() + Duration::from_secs(5),
            Duration::from_secs(5),
            stop_tokens::StopTokens::from_pairs(&[(2, "</s>")]),
        );
        let bytes = axum::body::to_bytes(Sse::new(stream).into_response().into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    #[tokio::test]
    async fn a_failure_mid_stream_is_an_error_event_not_a_finish() {
        let data = sse_data(
            vec![
                serde_json::json!({"event": "token", "text": "{\"a\":"}),
                serde_json::json!({
                    "event": "final", "ok": false, "error_kind": "grammar_dead_end",
                    "error": "json_schema: max_tokens ran out after 1 tokens, before the value was complete",
                    "text": "{\"a\":", "token_ids": [1],
                }),
            ],
            1,
        )
        .await;
        assert_eq!(data.len(), 4, "{data:?}"); // role, token, error, [DONE]
        assert!(data[1].contains("{\\\"a\\\":"), "{}", data[1]);
        let err: Value = serde_json::from_str(&data[2]).unwrap();
        assert_eq!(err["error"]["type"], "invalid_request_error");
        assert_eq!(err["error"]["code"], "grammar_dead_end");
        assert!(!data.iter().any(|d| d.contains("finish_reason\":\"")));
        assert_eq!(data[3], "[DONE]");
    }

    #[tokio::test]
    async fn a_successful_stream_still_ends_with_its_finish_reason() {
        let data = sse_data(
            vec![
                serde_json::json!({"event": "token", "text": "{}"}),
                serde_json::json!({"event": "token", "text": ""}),
                serde_json::json!({"event": "final", "ok": true, "text": "{}", "token_ids": [1, 2]}),
            ],
            8,
        )
        .await;
        let finish: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert_eq!(data.last().unwrap(), "[DONE]");
    }

    #[tokio::test]
    async fn a_streamed_stop_token_is_not_sent() {
        let data = sse_data(
            vec![
                serde_json::json!({"event": "token", "token_id": 7, "text": "Paris"}),
                serde_json::json!({"event": "token", "token_id": 2, "text": "</s>"}),
                serde_json::json!({"event": "final", "ok": true, "text": "Paris</s>", "token_ids": [7, 2]}),
            ],
            8,
        )
        .await;
        assert_eq!(data.len(), 4, "{data:?}"); // role, "Paris", finish, [DONE]
        assert!(data[1].contains("\"Paris\""), "{}", data[1]);
        assert!(!data.iter().any(|d| d.contains("</s>")), "{data:?}");
        let finish: Value = serde_json::from_str(&data[2]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn engine_error_only_matches_failed_final_events() {
        let failed = serde_json::json!(
            {"event": "final", "ok": false, "error": "boom", "error_kind": "other"}
        );
        let old_engine = serde_json::json!({"event": "final", "ok": false, "error": "boom"});
        let succeeded = serde_json::json!({"event": "final", "ok": true, "text": "hi"});
        let token = serde_json::json!({"event": "token", "text": "hi"});
        assert_eq!(engine_error(&failed), Some(engine("boom", Some("other"))));
        assert_eq!(engine_error(&old_engine), Some(engine("boom", None)));
        assert_eq!(engine_error(&succeeded), None);
        assert_eq!(engine_error(&token), None);
    }
}
