//! Renders a GGUF's `tokenizer.chat_template` metadata (a Jinja2 template string,
//! the same mechanism HF's `AutoTokenizer.apply_chat_template` uses) via `minijinja`,
//! instead of `openai::build_prompt`'s generic role-labeled flattening. See this
//! crate's README "Known limitations" for what's still out of scope (tool-calling
//! templates, multimodal template blocks).
//!
//! Degrades to `None` (never panics, never returns a bad prompt silently) whenever a
//! template can't be found, doesn't compile, or fails a render self-test at startup --
//! the caller (`main.rs`) falls back to `build_prompt` in that case and logs exactly
//! one clear line to stderr explaining why.

use crate::gguf_meta::GgufMeta;
use crate::openai::TextMessage;
use minijinja::{Environment, ErrorKind};

const TEMPLATE_NAME: &str = "chat";

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    fn compile(source: String, bos_token: String, eos_token: String) -> Result<Self, String> {
        let mut env = Environment::new();
        // Real HF chat templates commonly call `raise_exception(msg)` for validation
        // (e.g. "system message must come first") -- without this registered, a
        // template that expects it fails with an "unknown function" error instead of
        // the template's own intended message, and legitimate templates that never
        // hit that branch would otherwise be rejected at load for no reason.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| format!("template failed to compile: {e}"))?;
        Ok(ChatTemplate {
            env,
            bos_token,
            eos_token,
        })
    }

    pub fn render(
        &self,
        messages: &[TextMessage],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| format!("template lookup failed: {e}"))?;
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        let ctx = serde_json::json!({
            "messages": messages_json,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        });
        tmpl.render(ctx).map_err(|e| format!("render failed: {e}"))
    }
}

/// Best-effort `bos_token`/`eos_token` string lookup from `tokenizer.ggml.bos_token_id`
/// / `eos_token_id` + the `tokenizer.ggml.tokens` array -- some real-world templates
/// reference `{{ bos_token }}`/`{{ eos_token }}` directly. Missing metadata (either key
/// absent, or a GGUF with no embedded tokenizer at all) degrades to `""`, not an error
/// -- plenty of real templates never reference these variables.
fn resolve_bos_eos(meta: &GgufMeta) -> (String, String) {
    let tokens = meta.get_string_array("tokenizer.ggml.tokens");
    let lookup = |id_key: &str| -> String {
        let id = meta.get_u64(id_key);
        match (id, &tokens) {
            (Some(id), Some(toks)) => toks
                .get(id as usize)
                .map(|s| s.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        }
    };
    (
        lookup("tokenizer.ggml.bos_token_id"),
        lookup("tokenizer.ggml.eos_token_id"),
    )
}

/// One self-test message list, rendered once at startup to catch a template that
/// compiles but blows up at render time (e.g. references an unsupported construct)
/// before it's ever used against a real request.
fn self_test_messages() -> Vec<TextMessage> {
    vec![
        TextMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        TextMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        },
    ]
}

/// Tries to build a working [`ChatTemplate`] for `gguf_path`, honoring
/// `chat_template_file_override` (the `--chat-template-file` flag) when given.
/// Always logs exactly one clear line to stderr explaining the outcome -- which
/// source was used, or why it fell back to `None` (no template found, failed to
/// compile, or failed its render self-test). Never panics.
pub fn build(gguf_path: &str, chat_template_file_override: Option<&str>) -> Option<ChatTemplate> {
    let meta = match GgufMeta::open(gguf_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "[adapter] chat template: failed to read GGUF metadata ({e}); \
                 falling back to generic prompt flattening"
            );
            return None;
        }
    };
    let (bos_token, eos_token) = resolve_bos_eos(&meta);

    let (source_desc, template_src) = match chat_template_file_override {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(s) => (format!("--chat-template-file {path}"), s),
            Err(e) => {
                eprintln!(
                    "[adapter] chat template: failed to read --chat-template-file {path} ({e}); \
                     falling back to generic prompt flattening"
                );
                return None;
            }
        },
        None => match meta.get_str("tokenizer.chat_template") {
            Some(s) => (
                "the GGUF's tokenizer.chat_template metadata".to_string(),
                s.to_string(),
            ),
            None => {
                eprintln!(
                    "[adapter] chat template: no tokenizer.chat_template in GGUF metadata (and \
                     no --chat-template-file given); using generic prompt flattening"
                );
                return None;
            }
        },
    };

    let template = match ChatTemplate::compile(template_src, bos_token, eos_token) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[adapter] chat template: {source_desc} failed to compile ({e}); \
                 falling back to generic prompt flattening"
            );
            return None;
        }
    };

    if let Err(e) = template.render(&self_test_messages(), true) {
        eprintln!(
            "[adapter] chat template: {source_desc} failed a render self-test ({e}); \
             falling back to generic prompt flattening"
        );
        return None;
    }

    eprintln!("[adapter] chat template: using {source_desc}");
    Some(template)
}
