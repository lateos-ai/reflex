//! `POST /v1/systemone`: typed questions answered by a decision model, in the shape
//! of TypeSafe's System One API, as llama.cpp's `llama-server` serves it (PR #29818,
//! merge commit `a4cb4c6`, `tools/server/server-decision.cpp`). Not an OpenAI
//! endpoint. Design: `docs/design/decision-models.md` in the repository.
//!
//! This module owns everything specific to the API: the GGUF's `{arch}.decision.*`
//! metadata and `systemone` chat template, request validation, prompt rendering
//! (one prompt per question *variant*), and the answer math. The engine only reads
//! raw label-token logits: all of a request's prompts go to it in one IPC
//! `label_prompts` request, which evaluates their shared token prefix once.
//!
//! Phase 1 supports the label-token model types, `openjev` and `lev`. A `kev` or
//! `laya` model, a model with no `decision.type`, and images get `501`.
//!
//! Request JSON is parsed into [`Json`], which keeps object keys in the order they
//! were written (`serde_json::Value` sorts them): questions are answered, options
//! listed and the state's keys printed in request order, as llama.cpp does.

use crate::gguf_meta::GgufMeta;
use minijinja::value::{Enumerator, Object, ObjectRepr, Value as MjValue};
use minijinja::Environment;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Most questions one request may ask. Each one is one or two prompts for the
/// single engine, so an unbounded list would let one request hold it for long.
pub const MAX_QUESTIONS: usize = 64;

/// llama.cpp's limit on images per request (`DECISION_MAX_IMAGES`).
const MAX_IMAGES: usize = 8;

/// lev reads `noul` from a rating scale: 0 = certainly no, 8 = certainly yes.
const LEV_N_RATINGS: usize = 9;

const TEMPLATE_NAME: &str = "systemone";

// ---------------------------------------------------------------------------
// Order-preserving JSON
// ---------------------------------------------------------------------------

/// A JSON value with object keys in document order. Integers and floats stay
/// apart, as in nlohmann::json (`1` and `1.0` print differently in a prompt).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `Some(value)` unless the key is missing or null (nlohmann's
    /// `contains(key) && !at(key).is_null()`).
    fn get_non_null(&self, key: &str) -> Option<&Json> {
        self.get(key).filter(|v| !matches!(v, Json::Null))
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Every object's keys sorted, recursively (lev was trained on sorted keys;
    /// llama.cpp's `decision_sort_keys`, `std::map` byte order). A key that
    /// appears twice keeps its last value, as in a `std::map`.
    fn sorted(&self) -> Json {
        match self {
            Json::Arr(items) => Json::Arr(items.iter().map(Json::sorted).collect()),
            Json::Obj(items) => {
                let mut map = std::collections::BTreeMap::new();
                for (k, v) in items {
                    map.insert(k.as_bytes().to_vec(), (k.clone(), v.sorted()));
                }
                Json::Obj(map.into_values().collect())
            }
            other => other.clone(),
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v))
            }
            // nlohmann reads an integer above i64::MAX as unsigned, and the
            // template's `get<int64_t>()` wraps it.
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Int(v as i64))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(v) = seq.next_element()? {
                    items.push(v);
                }
                Ok(Json::Arr(items))
            }
            // nlohmann's ordered_json keeps the first position of a repeated key
            // and its last value.
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut items: Vec<(String, Json)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Json>()? {
                    match items.iter_mut().find(|(key, _)| *key == k) {
                        Some(slot) => slot.1 = v,
                        None => items.push((k, v)),
                    }
                }
                Ok(Json::Obj(items))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Int(i) => s.serialize_i64(*i),
            Json::Float(f) => s.serialize_f64(*f),
            Json::Str(v) => s.serialize_str(v),
            Json::Arr(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for v in items {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Json::Obj(items) => {
                let mut map = s.serialize_map(Some(items.len()))?;
                for (k, v) in items {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// llama.cpp's `tojson`
// ---------------------------------------------------------------------------

/// A float as C++'s default stream format writes it: `%g` with 6 significant
/// digits (`19.99`, `0.333333`, `1e-07`, `1e+20`).
pub fn format_float_g6(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".to_string()
        } else if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    // The exponent after rounding to 6 significant digits.
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    if !(-4..6).contains(&exp) {
        let mantissa = strip_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        let decimals = (5 - exp) as usize;
        strip_zeros(&format!("{x:.decimals$}")).to_string()
    }
}

/// `%g`'s trailing-zero removal: zeros after the point, then a bare point.
fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// A string the way llama.cpp's `tojson` quotes it: only `"`, `\`, `\b`, `\f`,
/// `\n`, `\r`, `\t` and other bytes below 0x20 are escaped; non-ASCII stays UTF-8.
fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `value | tojson` as llama.cpp `a4cb4c6`'s Jinja engine prints it
/// (`common/jinja/value.cpp`, `value_to_json_internal`): separators `", "` and
/// `": "`, keys in insertion order, see [`write_json_str`] and
/// [`format_float_g6`].
pub fn to_json_text(v: &Json) -> String {
    let mut out = String::new();
    write_json(&mut out, v);
    out
}

fn write_json(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => out.push_str(&format_float_g6(*f)),
        Json::Str(s) => write_json_str(out, s),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(out, item);
            }
            out.push(']');
        }
        Json::Obj(items) => {
            out.push('{');
            for (i, (k, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_str(out, k);
                out.push_str(": ");
                write_json(out, item);
            }
            out.push('}');
        }
    }
}

/// A JSON array or object handed to the template: keeps its key order for
/// iteration and for the `tojson` filter.
#[derive(Debug)]
struct JsonObject(Json);

impl Object for JsonObject {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        match &self.0 {
            Json::Arr(_) => ObjectRepr::Seq,
            _ => ObjectRepr::Map,
        }
    }

    fn get_value(self: &Arc<Self>, key: &MjValue) -> Option<MjValue> {
        match &self.0 {
            Json::Arr(items) => items.get(key.as_usize()?).map(to_template_value),
            Json::Obj(_) => self.0.get(key.as_str()?).map(to_template_value),
            _ => None,
        }
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        match &self.0 {
            Json::Arr(items) => Enumerator::Seq(items.len()),
            Json::Obj(items) => Enumerator::Values(
                items
                    .iter()
                    .map(|(k, _)| MjValue::from(k.as_str()))
                    .collect(),
            ),
            _ => Enumerator::Empty,
        }
    }
}

/// A request value as a template value: scalars as themselves (so `is string`
/// and truthiness behave as in Jinja), arrays and objects as [`JsonObject`].
fn to_template_value(v: &Json) -> MjValue {
    match v {
        Json::Null => MjValue::from(()),
        Json::Bool(b) => MjValue::from(*b),
        Json::Int(i) => MjValue::from(*i),
        Json::Float(f) => MjValue::from(*f),
        Json::Str(s) => MjValue::from(s.as_str()),
        other => MjValue::from_object(JsonObject(other.clone())),
    }
}

/// The inverse of [`to_template_value`], for the `tojson` filter. A value the
/// template built itself (a plain sequence or map) is walked generically.
fn from_template_value(v: &MjValue) -> Json {
    use minijinja::value::ValueKind;
    if let Some(obj) = v.downcast_object_ref::<JsonObject>() {
        return obj.0.clone();
    }
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => Json::Null,
        ValueKind::Bool => Json::Bool(v.is_true()),
        ValueKind::String => Json::Str(v.as_str().unwrap_or_default().to_string()),
        ValueKind::Number => match i64::try_from(v.clone()) {
            Ok(i) if v.is_integer() => Json::Int(i),
            _ => Json::Float(f64::try_from(v.clone()).unwrap_or(f64::NAN)),
        },
        ValueKind::Seq | ValueKind::Iterable => Json::Arr(
            v.try_iter()
                .map(|it| it.map(|x| from_template_value(&x)).collect())
                .unwrap_or_default(),
        ),
        ValueKind::Map => Json::Obj(
            v.try_iter()
                .map(|it| {
                    it.map(|k| {
                        let item = v.get_item(&k).unwrap_or_default();
                        (k.to_string(), from_template_value(&item))
                    })
                    .collect()
                })
                .unwrap_or_default(),
        ),
        _ => Json::Str(v.to_string()),
    }
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionType {
    /// Logits of one label token per option (letters A-Z, a-z).
    OpenJev,
    /// Like `OpenJev` with lev's labels, two variants for a choice, and `noul`
    /// read from a 9-step rating scale.
    Lev,
}

impl DecisionType {
    fn name(self) -> &'static str {
        match self {
            DecisionType::OpenJev => "openjev",
            DecisionType::Lev => "lev",
        }
    }
}

/// What the loaded GGUF supports, decided once at startup.
pub enum Support {
    /// No `{arch}.decision.type`: not a decision model (`501`).
    NotDecisionModel,
    /// A decision model this sidecar can't answer for (`501`, with the reason).
    Unsupported(String),
    Ready(Box<DecisionModel>),
}

pub struct DecisionModel {
    pub kind: DecisionType,
    env: Environment<'static>,
    /// `"<type>"` or `"<type>.<bucket>"` to temperature.
    temperatures: HashMap<String, f32>,
    /// The label strings the engine reads, one per option position.
    labels: Vec<String>,
    /// lev only: the label of each option position, given to the template.
    label_texts: Vec<String>,
    /// OpenJev lists a `noul` question's options as `[true, false]`.
    noul_true_first: bool,
}

/// The letters OpenJev's template labels options with.
const OPENJEV_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// lev's label codes, `A`-`Z` then `AA`-`ZZ`, keeping those that are one token, up
/// to 255 (llama.cpp's `server_decision_context::init`). For a byte-level BPE
/// vocabulary an ASCII code is one token exactly when it is in the vocabulary and,
/// for two letters, the merge of its two letters exists. The engine tokenizes each
/// label again and rejects any that isn't one token.
pub fn lev_labels(tokens: &[&str], merges: &[&str]) -> Vec<String> {
    let vocab: std::collections::HashSet<&str> = tokens.iter().copied().collect();
    let merges: std::collections::HashSet<&str> = merges.iter().copied().collect();
    let letters = ('A'..='Z').map(|c| c.to_string());
    let pairs = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    letters
        .chain(pairs)
        .filter(|code| {
            vocab.contains(code.as_str()) && {
                let mut c = code.chars();
                match (c.next(), c.next()) {
                    (Some(a), Some(b)) => merges.contains(format!("{a} {b}").as_str()),
                    _ => true,
                }
            }
        })
        .take(255)
        .collect()
}

/// A temperature as llama.cpp gets it: the metadata value is turned into text
/// with `std::to_string(float)` (6 decimals) and parsed back with `strtof`.
fn llama_cpp_temperature(v: f64) -> f32 {
    format!("{:.6}", v as f32 as f64)
        .parse::<f32>()
        .unwrap_or(0.0)
}

/// Reads the decision metadata of the GGUF at `gguf_path`. Logs one line.
pub fn load(gguf_path: &str) -> Support {
    let meta = match GgufMeta::open(gguf_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[adapter] systemone: failed to read GGUF metadata ({e})");
            return Support::Unsupported(format!("failed to read the model's metadata: {e}"));
        }
    };
    let support = from_meta(&meta);
    match &support {
        Support::NotDecisionModel => {}
        Support::Unsupported(why) => eprintln!("[adapter] systemone: unavailable: {why}"),
        Support::Ready(m) => eprintln!(
            "[adapter] systemone: decision model type {} ({} labels)",
            m.kind.name(),
            m.labels.len()
        ),
    }
    support
}

/// [`load`] on already-read metadata.
pub fn from_meta(meta: &GgufMeta) -> Support {
    let Some(arch) = meta.get_str("general.architecture") else {
        return Support::NotDecisionModel;
    };
    let prefix = format!("{arch}.decision.");
    let Some(type_name) = meta.get_str(&format!("{prefix}type")) else {
        return Support::NotDecisionModel;
    };
    let kind = match type_name {
        "openjev" => DecisionType::OpenJev,
        "lev" => DecisionType::Lev,
        "kev" | "laya" => {
            return Support::Unsupported(format!(
                "decision model type \"{type_name}\" is not supported yet"
            ))
        }
        other => return Support::Unsupported(format!("unsupported decision model type: {other}")),
    };
    let Some(source) = meta.get_str("tokenizer.chat_template.systemone") else {
        return Support::Unsupported("decision model has no \"systemone\" template".to_string());
    };

    let mut temperatures = HashMap::new();
    let temp_prefix = format!("{prefix}temperature.");
    for (key, value) in &meta.metadata {
        let Some(name) = key.strip_prefix(&temp_prefix) else {
            continue;
        };
        let t = value.as_f64().map(llama_cpp_temperature).unwrap_or(0.0);
        if t <= 0.0 {
            return Support::Unsupported(format!("invalid decision temperature: {key}"));
        }
        temperatures.insert(name.to_string(), t);
    }

    let (labels, label_texts, noul_true_first) = match kind {
        DecisionType::OpenJev => (
            OPENJEV_LETTERS.chars().map(|c| c.to_string()).collect(),
            Vec::new(),
            true,
        ),
        DecisionType::Lev => {
            let tokens = meta
                .get_string_array("tokenizer.ggml.tokens")
                .unwrap_or_default();
            let merges = meta
                .get_string_array("tokenizer.ggml.merges")
                .unwrap_or_default();
            let labels = lev_labels(&tokens, &merges);
            if labels.len() < LEV_N_RATINGS {
                return Support::Unsupported(
                    "lev model: too few single-token labels in the vocabulary".to_string(),
                );
            }
            (labels.clone(), labels, false)
        }
    };

    let mut env = Environment::new();
    // llama.cpp's Jinja engine defaults, as for chat templates.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_filter("tojson", |v: MjValue| {
        to_json_text(&from_template_value(&v))
    });
    if let Err(e) = env.add_template_owned(TEMPLATE_NAME, source.to_string()) {
        return Support::Unsupported(format!("the systemone template failed to compile: {e}"));
    }
    Support::Ready(Box::new(DecisionModel {
        kind,
        env,
        temperatures,
        labels,
        label_texts,
        noul_true_first,
    }))
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    fn name(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOption {
    pub key: String,
    /// `Json::Null` when not given.
    pub description: Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: QuestionType,
    pub instructions: Json,
    /// In the order of the model's outputs.
    pub options: Vec<DecisionOption>,
}

/// Why a request fails before reaching the engine.
#[derive(Debug, PartialEq)]
pub enum RequestError {
    /// `400`, with llama.cpp's message.
    Invalid(String),
    /// `501`: images, which Phase 1 doesn't take.
    ImagesUnsupported,
}

/// llama.cpp's message for a request with images on a server without them.
pub const IMAGES_UNSUPPORTED: &str = "This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`";

/// A validated request: its questions in request order, and the state.
#[derive(Debug)]
pub struct Parsed {
    pub questions: Vec<Question>,
    pub state: Json,
}

impl DecisionModel {
    fn n_options_max(&self) -> usize {
        self.labels.len()
    }

    /// Validates a request body the way llama.cpp's `parse_questions` and
    /// `parse_state` do, then refuses images.
    pub fn parse(&self, body: &Json) -> Result<Parsed, RequestError> {
        let invalid = |m: String| RequestError::Invalid(m);
        // A body that is not an object has no "state" (as in llama.cpp, whose
        // `contains` is false on a non-object).
        let Some(state) = body.get_non_null("state") else {
            return Err(invalid("\"state\" must be provided".to_string()));
        };
        let questions_json = match body.get("questions") {
            Some(Json::Obj(q)) if !q.is_empty() => q,
            _ => {
                return Err(invalid(
                    "\"questions\" must be a non-empty object".to_string(),
                ))
            }
        };
        if questions_json.len() > MAX_QUESTIONS {
            return Err(invalid(format!(
                "\"questions\" has {} entries, over this server's limit of {MAX_QUESTIONS}",
                questions_json.len()
            )));
        }

        let mut questions = Vec::with_capacity(questions_json.len());
        for (id, q) in questions_json {
            let err = |m: &str| invalid(format!("questions.{id}: {m}"));
            if !matches!(q, Json::Obj(_)) {
                return Err(err("must be an object"));
            }
            let Some(instructions) = q.get_non_null("instructions") else {
                return Err(err("\"instructions\" must be provided"));
            };
            let criteria = q.get("criteria").cloned().unwrap_or(Json::Null);
            let (kind, options) = match q.get("type").and_then(Json::as_str).unwrap_or("") {
                "choice" => match &criteria {
                    Json::Obj(items) if !items.is_empty() => (
                        QuestionType::Choice,
                        items
                            .iter()
                            .map(|(k, d)| DecisionOption {
                                key: k.clone(),
                                description: d.clone(),
                            })
                            .collect::<Vec<_>>(),
                    ),
                    _ => return Err(err("\"criteria\" must be a non-empty object")),
                },
                "score" => match &criteria {
                    Json::Arr(levels) if (2..=10).contains(&levels.len()) => (
                        QuestionType::Score,
                        levels
                            .iter()
                            .enumerate()
                            .map(|(i, d)| DecisionOption {
                                key: i.to_string(),
                                description: d.clone(),
                            })
                            .collect(),
                    ),
                    _ => return Err(err("\"criteria\" must be an array of 2 to 10 levels")),
                },
                "noul" => {
                    if !matches!(criteria, Json::Null | Json::Obj(_)) {
                        return Err(err("\"criteria\" must be an object"));
                    }
                    let mut options: Vec<DecisionOption> = ["false", "true"]
                        .iter()
                        .map(|k| DecisionOption {
                            key: k.to_string(),
                            description: criteria.get(k).cloned().unwrap_or(Json::Null),
                        })
                        .collect();
                    if self.noul_true_first {
                        options.swap(0, 1);
                    }
                    (QuestionType::Noul, options)
                }
                _ => return Err(err("\"type\" must be one of: choice, score, noul")),
            };
            if options.len() > self.n_options_max() {
                return Err(err(&format!(
                    "too many options ({}), this model supports at most {}",
                    options.len(),
                    self.n_options_max()
                )));
            }
            questions.push(Question {
                id: id.clone(),
                kind,
                instructions: instructions.clone(),
                options,
            });
        }

        let n_images = count_images(body, state)?;
        if n_images > 0 {
            return Err(RequestError::ImagesUnsupported);
        }
        Ok(Parsed {
            questions,
            state: state.clone(),
        })
    }

    /// Prompts evaluated for this question, each showing the options in a
    /// different order.
    pub fn n_variants(&self, q: &Question) -> usize {
        // lev shows a choice twice, the second time reversed, to cancel its
        // preference for the first label.
        if self.kind == DecisionType::Lev && q.kind == QuestionType::Choice && q.options.len() > 1 {
            2
        } else {
            1
        }
    }

    /// Outputs read per variant: one per option, or lev's 9 ratings for a noul.
    pub fn n_outputs(&self, q: &Question) -> usize {
        if self.kind == DecisionType::Lev && q.kind == QuestionType::Noul {
            LEV_N_RATINGS
        } else {
            q.options.len()
        }
    }

    /// The labels the engine reads for one variant: the first [`Self::n_outputs`]
    /// labels (lev reads a noul's ratings at its first 9 labels, not at digits).
    pub fn labels_for(&self, q: &Question) -> Vec<String> {
        self.labels[..self.n_outputs(q)].to_vec()
    }

    /// The prompt of one variant (llama.cpp's `server_decision_context::render`).
    pub fn render(&self, state: &Json, q: &Question, variant: usize) -> Result<String, String> {
        let n = q.options.len();
        let options: Vec<Json> = (0..n)
            .map(|i| {
                let opt = &q.options[if variant == 0 { i } else { n - 1 - i }];
                let mut o = vec![
                    ("key".to_string(), Json::Str(opt.key.clone())),
                    ("description".to_string(), opt.description.clone()),
                ];
                if !self.label_texts.is_empty() {
                    o.push(("label".to_string(), Json::Str(self.label_texts[i].clone())));
                }
                Json::Obj(o)
            })
            .collect();
        let mut inp = Json::Obj(vec![
            ("id".to_string(), Json::Str(q.id.clone())),
            ("type".to_string(), Json::Str(q.kind.name().to_string())),
            ("instructions".to_string(), q.instructions.clone()),
            ("state".to_string(), state.clone()),
            ("options".to_string(), Json::Arr(options)),
        ]);
        if self.kind == DecisionType::Lev {
            inp = inp.sorted();
        }
        let Json::Obj(vars) = inp else {
            unreachable!("built as an object")
        };
        let mut ctx: std::collections::BTreeMap<String, MjValue> = vars
            .iter()
            .map(|(k, v)| (k.clone(), to_template_value(v)))
            .collect();
        ctx.insert("images".to_string(), MjValue::from(Vec::<MjValue>::new()));
        self.env
            .get_template(TEMPLATE_NAME)
            .and_then(|t| t.render(MjValue::from(ctx)))
            .map_err(|e| format!("systemone template failed to render: {e}"))
    }

    /// The temperature for this question: `"<type>.<bucket>"`, then `"<type>"`,
    /// then 1. The bucket depends on the number of options.
    pub fn temperature(&self, q: &Question) -> f32 {
        let n = q.options.len();
        let bucket = if self.kind == DecisionType::Lev {
            if n <= 8 {
                "small"
            } else if n <= 26 {
                "mid"
            } else {
                "large"
            }
        } else if n <= 2 {
            "2"
        } else if n <= 5 {
            "3_5"
        } else if n <= 10 {
            "6_10"
        } else {
            "11"
        };
        let name = q.kind.name();
        self.temperatures
            .get(&format!("{name}.{bucket}"))
            .or_else(|| self.temperatures.get(name))
            .copied()
            .unwrap_or(1.0)
    }

    /// One question's answer from the raw logits of each variant
    /// (llama.cpp's `format_answer`).
    pub fn answer(&self, q: &Question, scores: &[Vec<f32>]) -> Result<Answer, String> {
        let n = self.n_outputs(q);
        if scores.len() != self.n_variants(q) {
            return Err("decision result does not match the number of variants".to_string());
        }
        let probs = average_softmax(scores, n, self.temperature(q))?;
        let ty = q.kind.name();

        if q.kind == QuestionType::Noul {
            let noul = if self.kind == DecisionType::Lev {
                (0..n).map(|i| probs[i] * i as f64 / (n - 1) as f64).sum()
            } else {
                q.options
                    .iter()
                    .zip(&probs)
                    .find(|(o, _)| o.key == "true")
                    .map_or(0.0, |(_, p)| *p)
            };
            return Ok(Answer::Noul { ty, noul });
        }

        let probabilities = Ordered(
            q.options
                .iter()
                .zip(&probs)
                .map(|(o, p)| (o.key.clone(), *p))
                .collect(),
        );
        if q.kind == QuestionType::Choice {
            let best = argmax(&probs);
            return Ok(Answer::Choice {
                ty,
                choice: q.options[best].key.clone(),
                probabilities,
                confidence: confidence_choice(&probs),
            });
        }
        let score = (0..n).map(|i| i as f64 * probs[i]).sum();
        let legend = Ordered(
            q.options
                .iter()
                .map(|o| (o.key.clone(), o.description.clone()))
                .collect(),
        );
        Ok(Answer::Score {
            ty,
            score,
            legend,
            probabilities,
            confidence: confidence_score(&probs),
        })
    }
}

/// Images in `images` and in `image_url` parts of a chat-message state, checked
/// as llama.cpp checks them (data URLs only, at most 8).
fn count_images(body: &Json, state: &Json) -> Result<usize, RequestError> {
    let mut n = 0usize;
    let mut add = |url: &Json| -> Result<(), RequestError> {
        if !url.as_str().is_some_and(|u| u.starts_with("data:image/")) {
            return Err(RequestError::Invalid(
                "images must be data URLs (data:image/...;base64,...)".to_string(),
            ));
        }
        if n >= MAX_IMAGES {
            return Err(RequestError::Invalid(format!(
                "too many images, the maximum is {MAX_IMAGES}"
            )));
        }
        n += 1;
        Ok(())
    };
    match body.get("images") {
        None | Some(Json::Null) => {}
        Some(Json::Arr(urls)) => {
            for u in urls {
                add(u)?;
            }
        }
        Some(_) => {
            return Err(RequestError::Invalid(
                "\"images\" must be an array".to_string(),
            ))
        }
    }
    let messages = match state {
        Json::Obj(_) if state.get("messages").is_some() => state.get("messages").unwrap(),
        other => other,
    };
    if let Json::Arr(msgs) = messages {
        for msg in msgs {
            let Some(Json::Arr(parts)) = msg.get("content") else {
                continue;
            };
            for part in parts {
                let is_image = part.get("type").and_then(Json::as_str) == Some("image_url");
                if let (true, Some(image_url)) = (is_image, part.get("image_url")) {
                    add(image_url.get("url").unwrap_or(image_url))?;
                }
            }
        }
    }
    Ok(n)
}

/// A softmax with temperature per variant, averaged over the variants; the
/// second variant's outputs are in reverse order. `s - max` in `f32`, the rest
/// in `f64`, as llama.cpp computes it.
pub fn average_softmax(
    scores: &[Vec<f32>],
    n: usize,
    temperature: f32,
) -> Result<Vec<f64>, String> {
    let mut probs = vec![0.0f64; n];
    for (v, s) in scores.iter().enumerate() {
        if s.len() != n {
            return Err("decision result does not match the number of options".to_string());
        }
        let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f64> = s
            .iter()
            .map(|&x| ((x - max) as f64 / temperature as f64).exp())
            .collect();
        let sum: f64 = p.iter().sum();
        for (i, pi) in p.iter().enumerate() {
            let at = if v == 0 { i } else { n - 1 - i };
            probs[at] += pi / sum / scores.len() as f64;
        }
    }
    Ok(probs)
}

/// The first index of the largest value (`std::max_element`).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// TypeSafe's choice confidence: how far the top probability is above uniform.
pub fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let p_max = probs[argmax(probs)];
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// TypeSafe's score confidence: the mean distance to the mode, relative to a
/// uniform distribution's distance to its center.
pub fn confidence_score(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let mode = argmax(probs) as f64;
    let mut dist = 0.0f64;
    let mut dist_uniform = 0.0f64;
    for (i, p) in probs.iter().enumerate() {
        dist += p * (i as f64 - mode).abs();
        dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// A JSON object written in this order.
#[derive(Debug, Clone, PartialEq)]
pub struct Ordered<T>(pub Vec<(String, T)>);

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Answer {
    Choice {
        #[serde(rename = "type")]
        ty: &'static str,
        choice: String,
        probabilities: Ordered<f64>,
        confidence: f64,
    },
    Score {
        #[serde(rename = "type")]
        ty: &'static str,
        score: f64,
        legend: Ordered<Json>,
        probabilities: Ordered<f64>,
        confidence: f64,
    },
    Noul {
        #[serde(rename = "type")]
        ty: &'static str,
        noul: f64,
    },
}

#[derive(Debug, serde::Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, serde::Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: Ordered<Answer>,
    pub usage: Usage,
}

/// Every variant of every question, in order: what goes to the engine.
pub struct Plan {
    pub prompts: Vec<String>,
    pub labels: Vec<Vec<String>>,
}

impl DecisionModel {
    pub fn plan(&self, parsed: &Parsed) -> Result<Plan, String> {
        let mut prompts = Vec::new();
        let mut labels = Vec::new();
        for q in &parsed.questions {
            for variant in 0..self.n_variants(q) {
                prompts.push(self.render(&parsed.state, q, variant)?);
                labels.push(self.labels_for(q));
            }
        }
        Ok(Plan { prompts, labels })
    }

    /// The IPC `label_prompts` request for `plan`.
    pub fn ipc_request(plan: &Plan) -> serde_json::Value {
        let prompts: Vec<serde_json::Value> = plan
            .prompts
            .iter()
            .zip(&plan.labels)
            .map(|(p, l)| serde_json::json!({"prompt": p, "labels": l}))
            .collect();
        serde_json::json!({ "label_prompts": prompts })
    }

    /// The response from the engine's final IPC line.
    pub fn build_response(
        &self,
        parsed: &Parsed,
        engine: &serde_json::Value,
        model: String,
    ) -> Result<SystemOneResponse, String> {
        let results = engine
            .get("labels")
            .and_then(serde_json::Value::as_array)
            .ok_or("engine response has no `labels`")?;
        let mut next = results.iter();
        let mut input_tokens = 0u64;
        let mut answers = Vec::with_capacity(parsed.questions.len());
        for q in &parsed.questions {
            let mut scores = Vec::with_capacity(self.n_variants(q));
            for _ in 0..self.n_variants(q) {
                let r = next.next().ok_or("engine returned too few results")?;
                input_tokens += r
                    .get("tokens")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("engine result has no `tokens`")?;
                let logits = r
                    .get("logits")
                    .and_then(serde_json::Value::as_array)
                    .ok_or("engine result has no `logits`")?
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .map(|f| f as f32)
                            .ok_or("a logit is not a number")
                    })
                    .collect::<Result<Vec<f32>, _>>()?;
                scores.push(logits);
            }
            answers.push((q.id.clone(), self.answer(q, &scores)?));
        }
        if next.next().is_some() {
            return Err("engine returned too many results".to_string());
        }
        Ok(SystemOneResponse {
            model,
            answers: Ordered(answers),
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests;
