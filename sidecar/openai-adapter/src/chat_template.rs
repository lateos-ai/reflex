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
    /// Whether the engine prepends BOS to every prompt (see [`engine_adds_bos`]).
    /// Templates that open with `{{ bos_token }}` (Llama, Mistral, Gemma) then
    /// have that leading BOS removed, so the prompt gets one, as HF's
    /// `apply_chat_template` + tokenize (no special tokens added) gives.
    strip_bos: bool,
}

impl ChatTemplate {
    fn compile(
        source: String,
        bos_token: String,
        eos_token: String,
        strip_bos: bool,
    ) -> Result<Self, String> {
        let mut env = Environment::new();
        // Hugging Face renders chat templates with `trim_blocks` and
        // `lstrip_blocks` on (`apply_chat_template`'s Jinja environment), and
        // with Python's string/dict/list methods available: templates such as
        // Qwen3.5's (`.startswith()`) and Gemma 4's (`.get()`) need them.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
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
            strip_bos,
        })
    }

    /// Renders `messages`. Some templates (Mistral's) only accept strictly
    /// alternating user/assistant turns and reject a system message; when one
    /// fails on a leading system message, the system text is folded into the
    /// first user turn (`"{system}\n\n{user}"`) and rendering retried.
    pub fn render(
        &self,
        messages: &[TextMessage],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        match self.render_as_is(messages, add_generation_prompt) {
            Err(e)
                if messages.len() >= 2
                    && messages[0].role == "system"
                    && messages[1].role == "user" =>
            {
                let mut merged = vec![TextMessage {
                    role: "user".to_string(),
                    content: format!("{}\n\n{}", messages[0].content, messages[1].content),
                }];
                merged.extend(messages[2..].iter().map(|m| TextMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                }));
                self.render_as_is(&merged, add_generation_prompt)
                    .map_err(|_| e)
            }
            result => result,
        }
    }

    fn render_as_is(
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
        let rendered = tmpl
            .render(ctx)
            .map_err(|e| format!("render failed: {e}"))?;
        Ok(match rendered.strip_prefix(self.bos_token.as_str()) {
            Some(rest) if self.strip_bos && !self.bos_token.is_empty() => rest.to_string(),
            _ => rendered,
        })
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

/// Whether the engine prepends BOS to a prompt, by its tokenizer's rule
/// (`src/tokenizer.rs`): `tokenizer.ggml.add_bos_token` when present, else
/// whether a `bos_token_id` exists, and never for the `glm4` pre-tokenizer.
fn engine_adds_bos(meta: &GgufMeta) -> bool {
    meta.get_bool("tokenizer.ggml.add_bos_token")
        .unwrap_or_else(|| meta.get_u64("tokenizer.ggml.bos_token_id").is_some())
        && meta.get_str("tokenizer.ggml.pre") != Some("glm4")
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

    let template =
        match ChatTemplate::compile(template_src, bos_token, eos_token, engine_adds_bos(&meta)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A template opening with `{{ bos_token }}` loses that BOS when the engine
    /// adds its own, and keeps it otherwise.
    #[test]
    fn leading_bos_is_dropped_only_when_the_engine_adds_one() {
        let src = "{{ bos_token }}[INST] {{ messages[0].content }} [/INST]".to_string();
        let msg = [TextMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let adds = ChatTemplate::compile(src.clone(), "<s>".into(), "</s>".into(), true).unwrap();
        assert_eq!(adds.render(&msg, true).unwrap(), "[INST] Hi [/INST]");
        let keeps = ChatTemplate::compile(src, "<s>".into(), "</s>".into(), false).unwrap();
        assert_eq!(keeps.render(&msg, true).unwrap(), "<s>[INST] Hi [/INST]");
    }

    /// The conversations [`render_templates_in_dir`] renders, as `(role, content)`.
    const CONVERSATIONS: &[&[(&str, &str)]] = &[
        &[("user", "What is the capital of France?")],
        &[("system", "You are terse."), ("user", "Hello")],
        &[
            ("system", "You are terse."),
            ("user", "Hi"),
            ("assistant", "Hello."),
            ("user", "Count to 3."),
        ],
    ];

    /// Renders every `<name>.jinja` in `$TEMPLATE_DIR` (with `<name>.bos` /
    /// `<name>.eos` holding the special tokens, and `<name>.addbos` `true` when the
    /// engine prepends BOS for that model) on [`CONVERSATIONS`], opening the
    /// assistant turn, into `<name>.<i>.rendered`, or `<name>.<i>.error`. A script
    /// compares the files with Python `jinja2` under Hugging Face's settings.
    #[test]
    #[ignore]
    fn render_templates_in_dir() {
        let dir = std::env::var("TEMPLATE_DIR").expect("set TEMPLATE_DIR");
        let read = |p: std::path::PathBuf| std::fs::read_to_string(p).unwrap_or_default();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("jinja") {
                continue;
            }
            let template = ChatTemplate::compile(
                read(path.clone()),
                read(path.with_extension("bos")),
                read(path.with_extension("eos")),
                read(path.with_extension("addbos")) == "true",
            )
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            for (i, conv) in CONVERSATIONS.iter().enumerate() {
                let messages: Vec<TextMessage> = conv
                    .iter()
                    .map(|(role, content)| TextMessage {
                        role: role.to_string(),
                        content: content.to_string(),
                    })
                    .collect();
                let (ext, body) = match template.render(&messages, true) {
                    Ok(s) => (format!("{i}.rendered"), s),
                    Err(e) => (format!("{i}.error"), e),
                };
                std::fs::write(path.with_extension(ext), body).unwrap();
            }
        }
    }
}
