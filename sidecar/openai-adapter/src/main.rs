//! `reflex-openai-adapter`: a standalone OpenAI-compatible HTTP sidecar in front of
//! one managed `reflex stdio <gguf>` process. Implements `POST /v1/chat/completions`
//! (streaming via SSE and non-streaming JSON) -- see this crate's README for usage,
//! scope, and known limitations, and the root CLAUDE.md/README.md's Non-goals
//! section for why this lives in its own crate/process rather than inside the core
//! engine.
//!
//! Usage: `reflex-openai-adapter <path-to-gguf> [--reflex-bin <path>] [--host
//! <addr>] [--port <port>] [--lora <adapter.gguf>] [--model-name <name>]
//! [--default-max-tokens <n>]`

mod openai;
mod reflex_client;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use openai::{
    build_prompt, build_sampling, estimate_prompt_tokens, extract_text_messages, finish_reason,
    now_unix, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessageOut,
    Choice, ChunkChoice, Delta, Usage,
};
use reflex_client::ReflexClient;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

struct AppState {
    client: ReflexClient,
    model_label: String,
    default_max_tokens: usize,
    request_counter: AtomicU64,
}

impl AppState {
    fn next_chat_id(&self) -> String {
        let n = self.request_counter.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-reflex-{n:x}")
    }
}

struct Opts {
    gguf_path: String,
    reflex_bin: String,
    lora_path: Option<String>,
    host: String,
    port: u16,
    model_label: Option<String>,
    default_max_tokens: usize,
}

impl Opts {
    fn parse(args: Vec<String>) -> Result<Opts, String> {
        let mut gguf_path: Option<String> = None;
        let mut reflex_bin = "reflex".to_string();
        let mut lora_path: Option<String> = None;
        let mut host = "127.0.0.1".to_string();
        let mut port: u16 = 8000;
        let mut model_label: Option<String> = None;
        let mut default_max_tokens: usize = 256;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--reflex-bin" => {
                    reflex_bin = args.next().ok_or("--reflex-bin requires a path")?;
                }
                "--lora" => {
                    lora_path = Some(args.next().ok_or("--lora requires a file path")?);
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
                _ if gguf_path.is_none() => gguf_path = Some(arg),
                other => return Err(format!("unexpected argument: {other}")),
            }
        }

        let gguf_path = gguf_path.ok_or(
            "usage: reflex-openai-adapter <path-to-gguf> [--reflex-bin <path>] [--host <addr>] \
             [--port <port>] [--lora <adapter.gguf>] [--model-name <name>] \
             [--default-max-tokens <n>]",
        )?;

        Ok(Opts {
            gguf_path,
            reflex_bin,
            lora_path,
            host,
            port,
            model_label,
            default_max_tokens,
        })
    }
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
    let client =
        match ReflexClient::spawn(&opts.reflex_bin, &opts.gguf_path, opts.lora_path.as_deref())
            .await
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[adapter] failed to launch reflex: {e:#}");
                std::process::exit(1);
            }
        };

    let state = Arc::new(AppState {
        client,
        model_label,
        default_max_tokens: opts.default_max_tokens,
        request_counter: AtomicU64::new(0),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/healthz", get(|| async { "ok" }))
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

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message.into(),
            "type": "reflex_adapter_error",
            "code": Value::Null,
        }
    });
    (status, Json(body)).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e.to_string()),
    };

    let messages = match extract_text_messages(&req.messages) {
        Ok(m) => m,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    let prompt = build_prompt(&messages);
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens).max(1);
    let sampling = build_sampling(&req);
    let stream = req.stream;
    let model_label = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_label.clone());

    let ipc_request = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "sampling": sampling,
    });
    let line = match serde_json::to_string(&ipc_request) {
        Ok(l) => l,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sidecar: failed to serialize IPC request: {e}"),
            )
        }
    };

    let rx = state.client.request(line).await;
    let chat_id = state.next_chat_id();
    let created = now_unix();

    if stream {
        let sse_stream = build_sse_stream(rx, chat_id, created, model_label, max_tokens);
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        non_streaming_response(rx, chat_id, created, model_label, prompt, max_tokens).await
    }
}

async fn non_streaming_response(
    mut rx: tokio::sync::mpsc::Receiver<Value>,
    chat_id: String,
    created: u64,
    model_label: String,
    prompt: String,
    requested_max_tokens: usize,
) -> Response {
    while let Some(v) = rx.recv().await {
        let event = v.get("event").and_then(Value::as_str).unwrap_or("");
        if event != "final" {
            continue; // shouldn't happen for a non-streaming IPC request, but ignore rather than choke on it
        }
        let ok = v.get("ok").and_then(Value::as_bool).unwrap_or(false);
        if !ok {
            let msg = v
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown reflex error");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, msg);
        }
        let text = v
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let completion_tokens = v
            .get("token_ids")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
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

fn build_sse_stream(
    mut rx: tokio::sync::mpsc::Receiver<Value>,
    chat_id: String,
    created: u64,
    model_label: String,
    requested_max_tokens: usize,
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
        while let Some(v) = rx.recv().await {
            let event = v.get("event").and_then(Value::as_str).unwrap_or("");
            if event == "token" {
                completion_tokens += 1;
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
                let ok = v.get("ok").and_then(Value::as_bool).unwrap_or(false);
                if !ok {
                    let msg = v.get("error").and_then(Value::as_str).unwrap_or("unknown reflex error");
                    eprintln!("[adapter] reflex error mid-stream: {msg}");
                }
                let token_ids = v
                    .get("token_ids")
                    .and_then(Value::as_array)
                    .map(|a| a.len())
                    .unwrap_or(completion_tokens);
                let reason = if ok {
                    finish_reason(token_ids, requested_max_tokens)
                } else {
                    "stop"
                };
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
