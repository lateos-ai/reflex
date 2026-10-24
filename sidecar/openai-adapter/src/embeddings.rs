//! `POST /v1/embeddings`: OpenAI's embeddings API over Reflex's embedding mode
//! (IPC `"embed": true`, `Model::embed` in the core crate), for an embedding GGUF
//! such as Nemotron-3.5-Embed-1B.
//!
//! `input` is one string or a list of strings; a list is sent to the engine one
//! text at a time, in order (the engine takes one request at a time and never
//! batches). Each text is rendered with the GGUF's own chat template as a system
//! message holding the retrieval prefix plus a user message holding the text,
//! which for Nemotron gives `"query: <text>"` or `"passage: <text>"` with the
//! template's bracket markers; without a template it is `"<prefix>: <text>"`.
//! `input_type` (NVIDIA's field) picks the prefix: `"query"` (the default) for
//! search queries, `"passage"` for documents. `dimensions` keeps the first n
//! values and re-normalizes them (the model is trained for truncation).
//! `encoding_format: "base64"` returns little-endian f32 bytes, base64-encoded.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Most texts one request may embed. Each is a full forward pass in the single
/// engine, so an unbounded list would let one request hold it for a long time.
pub const MAX_INPUTS: usize = 256;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: Value,
    /// `"float"` (the default) or `"base64"`.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Keep only the first `dimensions` values, re-normalized.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// `"query"` (the default) or `"passage"`.
    #[serde(default)]
    pub input_type: Option<String>,
    /// Accepted for OpenAI compatibility and ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
}

/// A request after [`validate`].
#[derive(Debug, PartialEq)]
pub struct Validated {
    pub texts: Vec<String>,
    pub base64: bool,
    pub dimensions: Option<usize>,
    /// The retrieval prefix without its colon: `query` or `passage`.
    pub prefix: &'static str,
}

/// Checks the request's shape (everything that doesn't need the engine), or a
/// human-readable reason for a `400`.
pub fn validate(req: &EmbeddingsRequest) -> Result<Validated, String> {
    let texts = match serde_json::from_value::<EmbeddingInput>(req.input.clone()) {
        Ok(EmbeddingInput::One(s)) => vec![s],
        Ok(EmbeddingInput::Many(v)) => v,
        Err(_) => {
            return Err(
                "`input` must be a string or an array of strings (token-id input is not \
                 supported)"
                    .to_string(),
            )
        }
    };
    if texts.is_empty() {
        return Err("`input` must not be an empty array".to_string());
    }
    if texts.len() > MAX_INPUTS {
        return Err(format!(
            "`input` has {} entries, over this server's limit of {MAX_INPUTS}",
            texts.len()
        ));
    }
    if let Some(i) = texts.iter().position(String::is_empty) {
        return Err(format!("input[{i}] is empty"));
    }
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(format!(
                "`encoding_format` must be \"float\" or \"base64\", got {other:?}"
            ))
        }
    };
    if req.dimensions == Some(0) {
        return Err("`dimensions` must be at least 1".to_string());
    }
    let prefix = match req.input_type.as_deref() {
        None | Some("query") => "query",
        Some("passage") | Some("document") => "passage",
        Some(other) => {
            return Err(format!(
                "`input_type` must be \"query\" or \"passage\", got {other:?}"
            ))
        }
    };
    Ok(Validated {
        texts,
        base64,
        dimensions: req.dimensions,
        prefix,
    })
}

/// The IPC request line's JSON for one rendered text.
pub fn ipc_request(rendered: &str) -> Value {
    serde_json::json!({ "prompt": rendered, "embed": true })
}

/// The text the engine embeds when the GGUF has no chat template.
pub fn fallback_render(prefix: &str, text: &str) -> String {
    format!("{prefix}: {text}")
}

/// Keeps the first `dims` values of `v` and scales them back to unit L2 norm.
/// Errs (for a `400`) when `dims` is above the model's width.
pub fn truncate(mut v: Vec<f32>, dims: Option<usize>) -> Result<Vec<f32>, String> {
    let Some(d) = dims else { return Ok(v) };
    if d > v.len() {
        return Err(format!(
            "`dimensions` is {d}, above this model's embedding width of {}",
            v.len()
        ));
    }
    if d == v.len() {
        return Ok(v);
    }
    v.truncate(d);
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
    Ok(v)
}

/// Standard base64 (with padding) of `v`'s little-endian f32 bytes, the format
/// OpenAI's `encoding_format: "base64"` returns.
pub fn base64_f32(v: &[f32]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> shift) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The embedding and token count out of the engine's successful `final` event, or
/// an error message if the event doesn't have the shape `src/ipc.rs` promises.
pub fn parse_engine_embedding(v: &Value) -> Result<(Vec<f32>, u32), String> {
    let embedding = v
        .get("embedding")
        .and_then(Value::as_array)
        .ok_or("engine response has no `embedding` array")?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32))
        .collect::<Option<Vec<f32>>>()
        .ok_or("engine response has a non-numeric `embedding` value")?;
    let tokens = v
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .ok_or("engine response has no `prompt_tokens`")?;
    Ok((embedding, tokens as u32))
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Serialize, PartialEq)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingValue,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: Value) -> EmbeddingsRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn validate_accepts_string_and_list_with_defaults() {
        let v = validate(&req(serde_json::json!({"input": "hi"}))).unwrap();
        assert_eq!(
            v,
            Validated {
                texts: vec!["hi".to_string()],
                base64: false,
                dimensions: None,
                prefix: "query",
            }
        );
        let v = validate(&req(serde_json::json!({
            "input": ["a", "b"],
            "input_type": "passage",
            "encoding_format": "base64",
            "dimensions": 256,
        })))
        .unwrap();
        assert_eq!(v.texts, vec!["a", "b"]);
        assert!(v.base64);
        assert_eq!(v.dimensions, Some(256));
        assert_eq!(v.prefix, "passage");
    }

    #[test]
    fn validate_rejects_bad_shapes() {
        for (body, needle) in [
            (serde_json::json!({"input": []}), "empty array"),
            (serde_json::json!({"input": [1, 2]}), "token-id"),
            (serde_json::json!({"input": ["a", ""]}), "input[1] is empty"),
            (
                serde_json::json!({"input": "a", "encoding_format": "hex"}),
                "encoding_format",
            ),
            (
                serde_json::json!({"input": "a", "dimensions": 0}),
                "dimensions",
            ),
            (
                serde_json::json!({"input": "a", "input_type": "code"}),
                "input_type",
            ),
        ] {
            let err = validate(&req(body)).unwrap_err();
            assert!(err.contains(needle), "{err}");
        }
        let many: Vec<String> = (0..=MAX_INPUTS).map(|i| i.to_string()).collect();
        assert!(validate(&req(serde_json::json!({ "input": many }))).is_err());
    }

    #[test]
    fn truncate_renormalizes_and_rejects_too_many_dims() {
        let v = truncate(vec![0.6, 0.0, 0.8, 0.0], Some(1)).unwrap();
        assert_eq!(v, vec![1.0]);
        let v = truncate(vec![0.0, 0.6, 0.8], Some(2)).unwrap();
        assert_eq!(v, vec![0.0, 1.0]);
        assert_eq!(truncate(vec![0.6, 0.8], None).unwrap(), vec![0.6, 0.8]);
        assert_eq!(truncate(vec![0.6, 0.8], Some(2)).unwrap(), vec![0.6, 0.8]);
        assert!(truncate(vec![0.6, 0.8], Some(3)).is_err());
    }

    #[test]
    fn base64_matches_known_encodings() {
        // 1.0f32 = 00 00 80 3f; -2.0f32 = 00 00 00 c0.
        assert_eq!(base64_f32(&[1.0]), "AACAPw==");
        assert_eq!(base64_f32(&[1.0, -2.0]), "AACAPwAAAMA=");
        assert_eq!(base64_f32(&[]), "");
    }

    #[test]
    fn parses_engine_response() {
        let v = serde_json::json!({"event": "final", "ok": true, "embedding": [0.5, -0.5], "prompt_tokens": 7});
        assert_eq!(parse_engine_embedding(&v).unwrap(), (vec![0.5, -0.5], 7));
        assert!(parse_engine_embedding(&serde_json::json!({"ok": true})).is_err());
    }

    /// Nemotron-3.5-Embed's own template (from the header fixture, when present)
    /// renders a system prefix plus user text as `"<prefix>: <text>"`, with a
    /// U+200C after every `[` and `<` of the user text, as HF renders it.
    #[test]
    fn nemotron_template_renders_prefix_and_markers() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-data/nemotron-3.5-embed-1b-header.gguf"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not present");
            return;
        }
        let ct = crate::chat_template::build(path, None).expect("template builds");
        let msg = |role: &str, content: &str| crate::openai::TextMessage {
            role: role.to_string(),
            content: content.to_string(),
        };
        for (system, prefix) in [("query:", "query: "), ("passage: ", "passage: ")] {
            let out = ct
                .render(
                    &[msg("system", system), msg("user", "see [INST] and <s>")],
                    false,
                )
                .unwrap();
            assert_eq!(
                out,
                format!("{prefix}see [\u{200c}INST] and <\u{200c}s>"),
                "system {system:?}"
            );
        }
    }

    #[test]
    fn response_serializes_in_openai_shape() {
        let r = EmbeddingsResponse {
            object: "list",
            data: vec![EmbeddingData {
                object: "embedding",
                index: 0,
                embedding: EmbeddingValue::Float(vec![1.0]),
            }],
            model: "m".to_string(),
            usage: EmbeddingsUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[1.0]}],"model":"m","usage":{"prompt_tokens":3,"total_tokens":3}}"#
        );
    }
}
