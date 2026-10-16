//! JSON Schema constrained decoding (agent relay phase C, see
//! docs/design/agent-relay.md). A request's `json_schema` is compiled once into a
//! context-free grammar over bytes; at every generated position the tokens whose bytes
//! can't continue a value that conforms to the schema get their logits set to `-inf`,
//! before greedy argmax or temperature/top-k/top-p sampling. Nothing else about the
//! logits changes, so when the unconstrained greedy output already conforms the
//! constrained output is the same.
//!
//! Host-side and sampling-time only, like `crate::sampling`: it reads the logits
//! `Model::lm_head_logits` has already downloaded and knows nothing about the
//! architecture.
//!
//! **Supported subset.** `type` (one name or an array of them), `properties`,
//! `required`, `additionalProperties` (`false` or absent), `items`, `enum`, `const`,
//! `anyOf` and a local `$ref` into the root's `$defs`. The annotation keywords
//! `title`, `description`, `$comment`, `$schema`, `default` and `examples` are
//! accepted, since they never change which values are valid. Any other keyword is
//! rejected with `invalid_input` naming it: a schema this module accepts never has a
//! constraint it silently drops.
//!
//! **Fixed output format.** The value is compact JSON with properties in schema
//! order, each optional one present or absent. The only whitespace outside strings
//! is one optional space after a `:` or a `,`, so a model can keep its usual
//! `"key": "value"` tokens; `enum`/`const` values are matched exactly as written
//! compactly.
//! When `properties` is given, the object has no other properties, whether
//! `additionalProperties` is `false` or absent (llama.cpp's `json_schema_to_grammar`
//! does the same); an object schema without `properties` and without
//! `additionalProperties: false` allows any object. Every output therefore validates
//! against the schema.
//!
//! **How masking works.** The grammar is run as a set of pushdown stacks
//! (nondeterministic, so `anyOf` branches and `enum` values may overlap). Stacks are
//! hash-consed and each distinct set of them becomes one state of a lazily built DFA
//! over bytes, so a byte step is one table lookup once seen. The vocabulary is a byte
//! trie in preorder ([`TokenIndex`], built on the first constrained request and kept
//! on the `Model`); a mask is one linear pass over the trie that skips every subtree
//! whose prefix is already rejected, and is cached per DFA state, since the state
//! inside a long string repeats token after token. A token is judged on the exact
//! bytes it adds to the decoded text, so a token that ends partway through a UTF-8
//! sequence or an escape is allowed when the bytes so far can still be completed, and
//! the automaton then waits for the rest.

use crate::error::ReflexError;
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

fn schema_err(msg: impl std::fmt::Display) -> ReflexError {
    ReflexError::InvalidInput(format!("json_schema: {msg}"))
}

// ---------------------------------------------------------------------------
// JSON values
// ---------------------------------------------------------------------------

/// A parsed JSON value that keeps object members in document order, which the
/// schema compiler needs (properties come out in schema order) and `serde_json`'s
/// default map doesn't do. Numbers keep their source text.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

/// Deepest nesting [`JsonValue::parse`] accepts, so a hostile schema can't overflow
/// the stack of this recursive parser.
const MAX_PARSE_DEPTH: usize = 128;

impl JsonValue {
    /// Parses strict RFC 8259 JSON (one value, optional surrounding whitespace).
    /// Lone UTF-16 surrogate escapes are rejected, as `serde_json` does.
    pub fn parse(text: &str) -> Result<JsonValue, String> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing characters at byte {}", p.i));
        }
        Ok(v)
    }

    fn kind(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// JSON Schema equality (`enum`/`const`): numbers compare by value, so `1` equals
    /// `1.0`; objects compare as unordered maps.
    fn json_eq(&self, other: &JsonValue) -> bool {
        match (self, other) {
            (JsonValue::Number(a), JsonValue::Number(b)) => number_value(a) == number_value(b),
            (JsonValue::Array(a), JsonValue::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.json_eq(y))
            }
            (JsonValue::Object(a), JsonValue::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| other.get(k).is_some_and(|w| v.json_eq(w)))
                    && b.iter().all(|(k, _)| self.get(k).is_some())
            }
            _ => self == other,
        }
    }

    /// Compact serialization, the form `enum`/`const` values are generated in: no whitespace,
    /// members in their stored order, strings escaped minimally (`"`, `\` and
    /// control characters; everything else as raw UTF-8).
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Number(n) => out.push_str(n),
            JsonValue::String(s) => write_json_string(s, out),
            JsonValue::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            JsonValue::Object(members) => {
                out.push('{');
                for (i, (k, v)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(k, out);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn number_value(text: &str) -> f64 {
    text.parse().unwrap_or(f64::NAN)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.s.get(self.i) == Some(&b) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > MAX_PARSE_DEPTH {
            return self.err("nesting too deep");
        }
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut members = Vec::new();
                self.ws();
                if self.eat(b'}') {
                    return Ok(JsonValue::Object(members));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return self.err("expected a string key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if !self.eat(b':') {
                        return self.err("expected ':'");
                    }
                    self.ws();
                    let v = self.value(depth + 1)?;
                    members.push((k, v));
                    self.ws();
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b'}') {
                        return Ok(JsonValue::Object(members));
                    }
                    return self.err("expected ',' or '}'");
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat(b']') {
                    return Ok(JsonValue::Array(items));
                }
                loop {
                    self.ws();
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b']') {
                        return Ok(JsonValue::Array(items));
                    }
                    return self.err("expected ',' or ']'");
                }
            }
            Some(b'"') => Ok(JsonValue::String(self.string()?)),
            Some(b't') => self.word("true", JsonValue::Bool(true)),
            Some(b'f') => self.word("false", JsonValue::Bool(false)),
            Some(b'n') => self.word("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => self.err("expected a JSON value"),
        }
    }

    fn word(&mut self, w: &str, v: JsonValue) -> Result<JsonValue, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("invalid literal")
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        self.i - start
    }

    fn number(&mut self) -> Result<JsonValue, String> {
        let start = self.i;
        self.eat(b'-');
        if self.eat(b'0') {
        } else if matches!(self.s.get(self.i), Some(b'1'..=b'9')) {
            self.digits();
        } else {
            return self.err("invalid number");
        }
        if self.eat(b'.') && self.digits() == 0 {
            return self.err("invalid number fraction");
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            if self.digits() == 0 {
                return self.err("invalid number exponent");
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII");
        Ok(JsonValue::Number(text.to_string()))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .ok_or("truncated \\u escape")?;
        let h = std::str::from_utf8(h).map_err(|_| "invalid \\u escape".to_string())?;
        let v = u32::from_str_radix(h, 16)
            .map_err(|_| format!("invalid \\u escape at byte {}", self.i))?;
        if !h.bytes().all(|b| b.is_ascii_hexdigit()) {
            return self.err("invalid \\u escape");
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated escape");
                    };
                    self.i += 1;
                    let c = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !(self.eat(b'\\') && self.eat(b'u')) {
                                    return self.err("lone leading surrogate");
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("invalid trailing surrogate");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return self.err("lone trailing surrogate");
                            } else {
                                hi
                            };
                            char::from_u32(cp).expect("surrogates handled above")
                        }
                        _ => return self.err("invalid escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                0x00..=0x1F => return self.err("control character in string"),
                _ => out.push(b),
            }
        }
        String::from_utf8(out).map_err(|_| "invalid UTF-8 in string".to_string())
    }
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "$comment",
    "$schema",
    "default",
    "examples",
];
const VALIDATION: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "$ref",
];
const TYPE_NAMES: &[&str] = &[
    "object", "array", "string", "number", "integer", "boolean", "null",
];

/// A compiled `json_schema`: checked against the supported subset, ready to build a
/// [`Constrained`] matcher per generation. Also counts the time spent masking, which
/// IPC reports as `timings.mask_ms_total`.
#[derive(Debug)]
pub struct JsonSchema {
    root: JsonValue,
    mask_ns: AtomicU64,
}

impl JsonSchema {
    /// Parses and checks `text`. Errs with `InvalidInput` on malformed JSON, an
    /// unsupported keyword (named in the message), a bad `$ref`, a `$ref` cycle that
    /// never consumes input, or a schema no value can satisfy.
    pub fn compile(text: &str) -> Result<JsonSchema, ReflexError> {
        let root =
            JsonValue::parse(text).map_err(|e| schema_err(format!("not valid JSON: {e}")))?;
        let schema = JsonSchema {
            root,
            mask_ns: AtomicU64::new(0),
        };
        schema.check(&schema.root, "#")?;
        schema.grammar(false)?;
        Ok(schema)
    }

    /// Checks that `tok` can produce every literal the schema requires (a
    /// SentencePiece-style tokenizer can't produce U+2581).
    pub fn check_tokenizer(&self, tok: &Tokenizer) -> Result<(), ReflexError> {
        self.grammar(tok.architecture != "gpt2").map(|_| ())
    }

    /// Whether `text` is exactly a value this schema's output format can produce
    /// (with `forbid_u2581` false). Bytes, not semantics: a valid value with
    /// whitespace the format doesn't allow is not accepted.
    pub fn accepts(&self, text: &str) -> bool {
        let Ok(g) = self.grammar(false) else {
            return false;
        };
        let (mut dfa, mut st) = Dfa::new(g);
        for &b in text.as_bytes() {
            st = dfa.step(st, b);
            if st == DEAD {
                return false;
            }
        }
        dfa.accepting[st as usize]
    }

    /// Total time spent computing and applying token masks with this schema.
    pub fn mask_ms_total(&self) -> f64 {
        self.mask_ns.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Checks the keywords of every schema object reachable from `s` (structure only;
    /// [`Self::grammar`] does the rest).
    fn check(&self, s: &JsonValue, path: &str) -> Result<(), ReflexError> {
        let members = match s {
            JsonValue::Bool(true) => return Ok(()),
            JsonValue::Bool(false) => {
                return Err(schema_err(format!(
                    "the schema `false` at {path} is not supported"
                )))
            }
            JsonValue::Object(m) => m,
            other => {
                return Err(schema_err(format!(
                    "expected a schema object at {path}, got {}",
                    other.kind()
                )))
            }
        };
        for (i, (k, _)) in members.iter().enumerate() {
            if members[..i].iter().any(|(k2, _)| k2 == k) {
                return Err(schema_err(format!("duplicate key \"{k}\" at {path}")));
            }
            let known = ANNOTATIONS.contains(&k.as_str())
                || VALIDATION.contains(&k.as_str())
                || k == "$defs";
            if !known {
                return Err(schema_err(format!("unsupported keyword \"{k}\" at {path}")));
            }
        }
        if s.get("$defs").is_some() && path != "#" {
            return Err(schema_err(format!(
                "\"$defs\" is only supported at the schema root, found at {path}"
            )));
        }
        for exclusive in ["anyOf", "$ref"] {
            if s.get(exclusive).is_none() {
                continue;
            }
            if let Some((k, _)) = members
                .iter()
                .find(|(k, _)| VALIDATION.contains(&k.as_str()) && k != exclusive)
            {
                return Err(schema_err(format!(
                    "\"{exclusive}\" can't be combined with \"{k}\" in the same schema object (at {path}); move \"{k}\" into each branch"
                )));
            }
        }
        if let Some(t) = s.get("type") {
            let names: Vec<&JsonValue> = match t {
                JsonValue::Array(a) if !a.is_empty() => a.iter().collect(),
                JsonValue::Array(_) => {
                    return Err(schema_err(format!("empty \"type\" array at {path}")))
                }
                other => vec![other],
            };
            for n in names {
                match n {
                    JsonValue::String(n) if TYPE_NAMES.contains(&n.as_str()) => {}
                    other => {
                        return Err(schema_err(format!(
                            "unsupported \"type\" {} at {path}",
                            other.to_compact()
                        )))
                    }
                }
            }
        }
        if let Some(props) = s.get("properties") {
            let JsonValue::Object(props) = props else {
                return Err(schema_err(format!(
                    "\"properties\" must be an object at {path}"
                )));
            };
            for (i, (name, sub)) in props.iter().enumerate() {
                if props[..i].iter().any(|(n2, _)| n2 == name) {
                    return Err(schema_err(format!(
                        "duplicate property \"{name}\" at {path}"
                    )));
                }
                self.check(sub, &format!("{path}/properties/{name}"))?;
            }
        }
        if let Some(req) = s.get("required") {
            let JsonValue::Array(req) = req else {
                return Err(schema_err(format!(
                    "\"required\" must be an array at {path}"
                )));
            };
            for (i, r) in req.iter().enumerate() {
                let JsonValue::String(name) = r else {
                    return Err(schema_err(format!(
                        "\"required\" entries must be strings at {path}"
                    )));
                };
                if req[..i].contains(r) {
                    return Err(schema_err(format!(
                        "\"required\" lists \"{name}\" twice at {path}"
                    )));
                }
                if s.get("properties").and_then(|p| p.get(name)).is_none() {
                    return Err(schema_err(format!(
                        "required property \"{name}\" is not in \"properties\" at {path} (this subset never generates properties the schema doesn't list)"
                    )));
                }
            }
        }
        match s.get("additionalProperties") {
            None | Some(JsonValue::Bool(false)) => {}
            Some(other) => {
                return Err(schema_err(format!(
                    "\"additionalProperties\" must be false or omitted at {path}, got {}",
                    other.to_compact()
                )))
            }
        }
        if let Some(items) = s.get("items") {
            if matches!(items, JsonValue::Array(_)) {
                return Err(schema_err(format!(
                    "array-form \"items\" (tuple validation) is not supported at {path}"
                )));
            }
            self.check(items, &format!("{path}/items"))?;
        }
        if let Some(e) = s.get("enum") {
            match e {
                JsonValue::Array(a) if !a.is_empty() => {}
                _ => {
                    return Err(schema_err(format!(
                        "\"enum\" must be a non-empty array at {path}"
                    )))
                }
            }
        }
        if let Some(any) = s.get("anyOf") {
            let JsonValue::Array(branches) = any else {
                return Err(schema_err(format!("\"anyOf\" must be an array at {path}")));
            };
            if branches.is_empty() {
                return Err(schema_err(format!("empty \"anyOf\" at {path}")));
            }
            for (i, b) in branches.iter().enumerate() {
                self.check(b, &format!("{path}/anyOf/{i}"))?;
            }
        }
        if let Some(r) = s.get("$ref") {
            let JsonValue::String(r) = r else {
                return Err(schema_err(format!("\"$ref\" must be a string at {path}")));
            };
            self.resolve(r, path)?;
        }
        if let Some(defs) = s.get("$defs") {
            let JsonValue::Object(defs) = defs else {
                return Err(schema_err("\"$defs\" must be an object"));
            };
            for (name, d) in defs {
                self.check(d, &format!("#/$defs/{name}"))?;
            }
        }
        Ok(())
    }

    /// Resolves a local `$ref`: `#` (the root) or `#/$defs/<name>`, with JSON
    /// Pointer's `~0`/`~1` escapes.
    fn resolve<'s>(&'s self, r: &str, path: &str) -> Result<(&'s JsonValue, String), ReflexError> {
        if r == "#" {
            return Ok((&self.root, "#".to_string()));
        }
        let Some(name) = r.strip_prefix("#/$defs/") else {
            return Err(schema_err(format!(
                "unsupported \"$ref\" \"{r}\" at {path}: only \"#\" and \"#/$defs/<name>\" are supported"
            )));
        };
        if name.contains('/') || name.contains('%') {
            return Err(schema_err(format!(
                "unsupported \"$ref\" \"{r}\" at {path}: only \"#/$defs/<name>\" is supported"
            )));
        }
        let name = name.replace("~1", "/").replace("~0", "~");
        let target = self
            .root
            .get("$defs")
            .and_then(|d| d.get(&name))
            .ok_or_else(|| {
                schema_err(format!(
                    "\"$ref\" \"{r}\" at {path} names no entry in \"$defs\""
                ))
            })?;
        Ok((target, format!("#/$defs/{name}")))
    }

    /// Builds the byte grammar. `forbid_u2581` excludes U+2581 (`▁`) from strings,
    /// for SentencePiece-style tokenizers whose decoder turns it into a space.
    fn grammar(&self, forbid_u2581: bool) -> Result<Grammar, ReflexError> {
        let mut c = Compiler {
            schema: self,
            b: Builder::default(),
            forbid_u2581,
            refs: HashMap::new(),
            prims: HashMap::new(),
        };
        let root = c.value(&self.root, "#")?;
        c.b.finish(root)
    }

    /// Checks `v` against this schema with JSON Schema semantics (restricted to the
    /// supported keywords). `Err` names the first violation. Used to filter `enum`
    /// values by their sibling keywords, and by tests.
    pub fn validate(&self, v: &JsonValue) -> Result<(), String> {
        self.validate_at(&self.root, v, "$", 0)
    }

    fn validate_at(
        &self,
        s: &JsonValue,
        v: &JsonValue,
        at: &str,
        depth: usize,
    ) -> Result<(), String> {
        if depth > 64 {
            return Err(format!("{at}: $ref nesting too deep to validate"));
        }
        if matches!(s, JsonValue::Bool(true)) {
            return Ok(());
        }
        if let Some(JsonValue::String(r)) = s.get("$ref") {
            let (target, _) = self.resolve(r, at).map_err(|e| e.to_string())?;
            return self.validate_at(target, v, at, depth + 1);
        }
        if let Some(JsonValue::Array(branches)) = s.get("anyOf") {
            if branches
                .iter()
                .any(|b| self.validate_at(b, v, at, depth + 1).is_ok())
            {
                return Ok(());
            }
            return Err(format!("{at}: matches no anyOf branch"));
        }
        if let Some(JsonValue::Array(values)) = s.get("enum") {
            if !values.iter().any(|e| e.json_eq(v)) {
                return Err(format!("{at}: not one of the enum values"));
            }
        }
        if let Some(c) = s.get("const") {
            if !c.json_eq(v) {
                return Err(format!("{at}: not the const value"));
            }
        }
        if let Some(t) = s.get("type") {
            let names: Vec<&str> = match t {
                JsonValue::Array(a) => a
                    .iter()
                    .filter_map(|n| match n {
                        JsonValue::String(n) => Some(n.as_str()),
                        _ => None,
                    })
                    .collect(),
                JsonValue::String(n) => vec![n.as_str()],
                _ => Vec::new(),
            };
            let ok = names.iter().any(|&n| match (n, v) {
                ("integer", JsonValue::Number(x)) => number_value(x).fract() == 0.0,
                (n, v) => n == v.kind(),
            });
            if !ok {
                return Err(format!(
                    "{at}: a {} is not of type {}",
                    v.kind(),
                    t.to_compact()
                ));
            }
        }
        if let JsonValue::Object(members) = v {
            let props = s.get("properties");
            for (k, val) in members {
                match props.and_then(|p| p.get(k)) {
                    Some(sub) => self.validate_at(sub, val, &format!("{at}.{k}"), depth + 1)?,
                    None if s.get("additionalProperties") == Some(&JsonValue::Bool(false)) => {
                        return Err(format!("{at}: additional property \"{k}\""))
                    }
                    None => {}
                }
            }
            if let Some(JsonValue::Array(req)) = s.get("required") {
                for r in req {
                    if let JsonValue::String(r) = r {
                        if !members.iter().any(|(k, _)| k == r) {
                            return Err(format!("{at}: missing required property \"{r}\""));
                        }
                    }
                }
            }
        }
        if let (JsonValue::Array(items), Some(item_schema)) = (v, s.get("items")) {
            for (i, item) in items.iter().enumerate() {
                self.validate_at(item_schema, item, &format!("{at}[{i}]"), depth + 1)?;
            }
        }
        Ok(())
    }
}

/// Turns a checked schema into [`Builder`] rules, one rule per schema position, each
/// matching exactly the serializations this module's output format allows for it.
struct Compiler<'s> {
    schema: &'s JsonSchema,
    b: Builder,
    forbid_u2581: bool,
    /// `$ref` target path -> its rule, reserved before compiling the target so a
    /// recursive schema refers back to it.
    refs: HashMap<String, u32>,
    /// Shared primitive rules (`string`, `number`, the generic value, ...).
    prims: HashMap<&'static str, u32>,
}

/// A symbol in a rule alternative: a byte class or another rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sym {
    Class(u32),
    Rule(u32),
}

impl Compiler<'_> {
    fn value(&mut self, s: &JsonValue, path: &str) -> Result<u32, ReflexError> {
        if matches!(s, JsonValue::Bool(true)) {
            return Ok(self.prim("value"));
        }
        if let Some(JsonValue::String(r)) = s.get("$ref") {
            let (target, key) = self.schema.resolve(r, path)?;
            if let Some(&rule) = self.refs.get(&key) {
                return Ok(rule);
            }
            let rule = self.b.reserve();
            self.refs.insert(key.clone(), rule);
            let body = self.value(target, &key)?;
            self.b.define(rule, vec![vec![Sym::Rule(body)]]);
            return Ok(rule);
        }
        if let Some(JsonValue::Array(branches)) = s.get("anyOf") {
            let mut alts = Vec::new();
            for (i, br) in branches.iter().enumerate() {
                alts.push(vec![Sym::Rule(
                    self.value(br, &format!("{path}/anyOf/{i}"))?,
                )]);
            }
            return Ok(self.b.rule(alts));
        }
        if s.get("enum").is_some() || s.get("const").is_some() {
            return self.literals(s, path);
        }

        let types: Vec<&str> = match s.get("type") {
            Some(JsonValue::String(t)) => vec![t.as_str()],
            Some(JsonValue::Array(a)) => a
                .iter()
                .filter_map(|t| match t {
                    JsonValue::String(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect(),
            _ => {
                let object_kw = ["properties", "required", "additionalProperties"]
                    .iter()
                    .any(|k| s.get(k).is_some());
                let array_kw = s.get("items").is_some();
                match (object_kw, array_kw) {
                    (false, false) => return Ok(self.prim("value")),
                    (true, false) => vec!["object"],
                    (false, true) => vec!["array"],
                    (true, true) => vec!["object", "array"],
                }
            }
        };
        let mut alts = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for t in types {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let rule = match t {
                "object" => self.object(s, path)?,
                "array" => {
                    let item = match s.get("items") {
                        Some(items) => self.value(items, &format!("{path}/items"))?,
                        None => self.prim("value"),
                    };
                    self.array(item)
                }
                "string" => self.prim("string"),
                "number" => self.prim("number"),
                "integer" => self.prim("integer"),
                "boolean" => self.prim("boolean"),
                "null" => self.prim("null"),
                _ => unreachable!("type names are checked"),
            };
            alts.push(vec![Sym::Rule(rule)]);
        }
        Ok(self.b.rule(alts))
    }

    /// `enum`/`const`: the literal values that also satisfy the sibling keywords,
    /// each as its compact serialization.
    fn literals(&mut self, s: &JsonValue, path: &str) -> Result<u32, ReflexError> {
        let mut values: Vec<&JsonValue> = match s.get("enum") {
            Some(JsonValue::Array(a)) => a.iter().collect(),
            _ => vec![s.get("const").expect("enum or const")],
        };
        if let (Some(_), Some(c)) = (s.get("enum"), s.get("const")) {
            values.retain(|v| v.json_eq(c));
        }
        let siblings = JsonValue::Object(match s {
            JsonValue::Object(m) => m
                .iter()
                .filter(|(k, _)| k != "enum" && k != "const")
                .cloned()
                .collect(),
            _ => Vec::new(),
        });
        values.retain(|v| self.schema.validate_at(&siblings, v, "$", 0).is_ok());
        if values.is_empty() {
            return Err(schema_err(format!(
                "no \"enum\"/\"const\" value at {path} satisfies the rest of its schema"
            )));
        }
        let mut alts = Vec::new();
        for v in values {
            let text = v.to_compact();
            if self.forbid_u2581 && text.contains('\u{2581}') {
                return Err(schema_err(format!(
                    "the value {text} at {path} contains U+2581, which this model's tokenizer decodes as a space, so it can't be generated"
                )));
            }
            alts.push(self.b.lit(text.as_bytes()));
        }
        Ok(self.b.rule(alts))
    }

    fn object(&mut self, s: &JsonValue, path: &str) -> Result<u32, ReflexError> {
        let props: &[(String, JsonValue)] = match s.get("properties") {
            Some(JsonValue::Object(p)) => p,
            _ => &[],
        };
        if props.is_empty() && s.get("additionalProperties").is_none() {
            return Ok(self.prim("object"));
        }
        let required: Vec<&str> = match s.get("required") {
            Some(JsonValue::Array(r)) => r
                .iter()
                .filter_map(|v| match v {
                    JsonValue::String(n) => Some(n.as_str()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        // rest[i][c]: the properties from index i on, then `}`; c = whether a comma
        // must come before the next emitted property. Built back to front.
        let close = self.b.lit(b"}");
        let end = self.b.rule(vec![close]);
        let mut next = [end, end];
        for (name, sub) in props.iter().rev() {
            let key = JsonValue::String(name.clone()).to_compact();
            if self.forbid_u2581 && key.contains('\u{2581}') {
                return Err(schema_err(format!(
                    "the property name \"{name}\" at {path} contains U+2581, which this model's tokenizer decodes as a space, so it can't be generated"
                )));
            }
            let value = self.value(sub, &format!("{path}/properties/{name}"))?;
            let mut kv = self.b.lit(key.as_bytes());
            kv.extend(self.sep(b':'));
            kv.push(Sym::Rule(value));
            let kv = self.b.rule(vec![kv]);
            let mut cur = [0u32; 2];
            for comma in [false, true] {
                let mut emit = if comma { self.sep(b',') } else { Vec::new() };
                emit.extend([Sym::Rule(kv), Sym::Rule(next[1])]);
                let mut alts = vec![emit];
                if !required.contains(&name.as_str()) {
                    alts.push(vec![Sym::Rule(next[comma as usize])]);
                }
                cur[comma as usize] = self.b.rule(alts);
            }
            next = cur;
        }
        let mut open = self.b.lit(b"{");
        open.push(Sym::Rule(next[0]));
        Ok(self.b.rule(vec![open]))
    }

    /// `byte` (`:` or `,`) and the one optional space that may follow it, the only
    /// whitespace the output format allows.
    fn sep(&mut self, byte: u8) -> Vec<Sym> {
        let mut out = self.b.lit(&[byte]);
        out.push(Sym::Rule(self.prim("space")));
        out
    }

    /// `[` `]` or `[` item (`,` item)* `]`.
    fn array(&mut self, item: u32) -> u32 {
        let tail = self.b.reserve();
        let close = self.b.lit(b"]");
        let mut more = self.sep(b',');
        more.extend([Sym::Rule(item), Sym::Rule(tail)]);
        self.b.define(tail, vec![close.clone(), more]);
        let rest = self
            .b
            .rule(vec![close, vec![Sym::Rule(item), Sym::Rule(tail)]]);
        let mut open = self.b.lit(b"[");
        open.push(Sym::Rule(rest));
        self.b.rule(vec![open])
    }

    fn prim(&mut self, name: &'static str) -> u32 {
        if let Some(&r) = self.prims.get(name) {
            return r;
        }
        let r = self.b.reserve();
        self.prims.insert(name, r);
        let alts = match name {
            "null" => vec![self.b.lit(b"null")],
            "space" => vec![vec![], self.b.lit(b" ")],
            "boolean" => vec![self.b.lit(b"true"), self.b.lit(b"false")],
            "digits" => {
                // [0-9]*
                let d = self.b.class(&[(b'0', b'9')]);
                vec![vec![], vec![d, Sym::Rule(r)]]
            }
            "integer" => {
                let digits = self.prim("digits");
                let zero = self.b.lit(b"0");
                let nz = self.b.class(&[(b'1', b'9')]);
                let int_part = self.b.rule(vec![zero, vec![nz, Sym::Rule(digits)]]);
                let mut neg = self.b.lit(b"-");
                neg.push(Sym::Rule(int_part));
                vec![neg, vec![Sym::Rule(int_part)]]
            }
            "number" => {
                let int = self.prim("integer");
                let digits = self.prim("digits");
                let d = self.b.class(&[(b'0', b'9')]);
                let mut frac = self.b.lit(b".");
                frac.extend([d, Sym::Rule(digits)]);
                let frac = self.b.rule(vec![vec![], frac]);
                let e = self.b.class(&[(b'e', b'e'), (b'E', b'E')]);
                let sign = self.b.class(&[(b'+', b'+'), (b'-', b'-')]);
                let sign = self.b.rule(vec![vec![], vec![sign]]);
                let exp = self
                    .b
                    .rule(vec![vec![], vec![e, Sym::Rule(sign), d, Sym::Rule(digits)]]);
                vec![vec![Sym::Rule(int), Sym::Rule(frac), Sym::Rule(exp)]]
            }
            "string" => {
                let chars = self.prim("chars");
                let mut s = self.b.lit(b"\"");
                s.push(Sym::Rule(chars));
                s.extend(self.b.lit(b"\""));
                vec![s]
            }
            "chars" => {
                // char*
                let ch = self.prim("char");
                vec![vec![], vec![Sym::Rule(ch), Sym::Rule(r)]]
            }
            "char" => self.char_alts(),
            "value" => {
                let mut alts = Vec::new();
                for p in ["object", "array_any", "string", "number", "boolean", "null"] {
                    alts.push(vec![Sym::Rule(self.prim(p))]);
                }
                alts
            }
            "array_any" => {
                let v = self.prim("value");
                let a = self.array(v);
                vec![vec![Sym::Rule(a)]]
            }
            "object" => {
                // `{}` or `{` member (`,` member)* `}`, any keys and values.
                let string = self.prim("string");
                let value = self.prim("value");
                let mut member = vec![Sym::Rule(string)];
                member.extend(self.sep(b':'));
                member.push(Sym::Rule(value));
                let member = self.b.rule(vec![member]);
                let tail = self.b.reserve();
                let close = self.b.lit(b"}");
                let mut more = self.sep(b',');
                more.extend([Sym::Rule(member), Sym::Rule(tail)]);
                self.b.define(tail, vec![close.clone(), more]);
                let rest = self
                    .b
                    .rule(vec![close, vec![Sym::Rule(member), Sym::Rule(tail)]]);
                let mut open = self.b.lit(b"{");
                open.push(Sym::Rule(rest));
                vec![open]
            }
            _ => unreachable!("unknown primitive {name}"),
        };
        self.b.define(r, alts);
        r
    }

    /// One character inside a JSON string: unescaped printable ASCII, an escape, or
    /// one well-formed multi-byte UTF-8 sequence (no overlongs, no surrogates, nothing
    /// past U+10FFFF). `\u` escapes of surrogates must come as a high-low pair.
    fn char_alts(&mut self) -> Vec<Vec<Sym>> {
        let b = &mut self.b;
        let cont = b.class(&[(0x80, 0xBF)]);
        let hex = b.class(&[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')]);
        let plain = b.class(&[(0x20, 0x21), (0x23, 0x5B), (0x5D, 0x7F)]);
        let simple = b.class(&[
            (b'"', b'"'),
            (b'\\', b'\\'),
            (b'/', b'/'),
            (b'b', b'b'),
            (b'f', b'f'),
            (b'n', b'n'),
            (b'r', b'r'),
            (b't', b't'),
        ]);
        let u = b.class(&[(b'u', b'u')]);
        let bs = b.class(&[(b'\\', b'\\')]);
        let d = b.class(&[(b'D', b'D'), (b'd', b'd')]);
        // \uXXXX that is not a surrogate, or a high surrogate then a low one.
        let not_d = b.class(&[
            (b'0', b'9'),
            (b'A', b'C'),
            (b'a', b'c'),
            (b'E', b'F'),
            (b'e', b'f'),
        ]);
        let d_low = b.class(&[(b'0', b'7')]);
        let d_high = b.class(&[(b'8', b'9'), (b'A', b'B'), (b'a', b'b')]);
        let d_trail = b.class(&[(b'C', b'F'), (b'c', b'f')]);
        let u_body = b.rule(vec![
            vec![not_d, hex, hex, hex],
            vec![d, d_low, hex, hex],
            vec![d, d_high, hex, hex, bs, u, d, d_trail, hex, hex],
        ]);
        let esc = b.rule(vec![vec![simple], vec![u, Sym::Rule(u_body)]]);
        let mut alts = vec![vec![plain], vec![bs, Sym::Rule(esc)]];
        alts.push(vec![b.class(&[(0xC2, 0xDF)]), cont]);
        alts.push(vec![
            b.class(&[(0xE0, 0xE0)]),
            b.class(&[(0xA0, 0xBF)]),
            cont,
        ]);
        if self.forbid_u2581 {
            // U+2581 is E2 96 81: E2 is split out of E1-EC.
            alts.push(vec![b.class(&[(0xE1, 0xE1), (0xE3, 0xEC)]), cont, cont]);
            let e2 = b.class(&[(0xE2, 0xE2)]);
            alts.push(vec![e2, b.class(&[(0x80, 0x95), (0x97, 0xBF)]), cont]);
            alts.push(vec![
                e2,
                b.class(&[(0x96, 0x96)]),
                b.class(&[(0x80, 0x80), (0x82, 0xBF)]),
            ]);
        } else {
            alts.push(vec![b.class(&[(0xE1, 0xEC)]), cont, cont]);
        }
        alts.push(vec![
            b.class(&[(0xED, 0xED)]),
            b.class(&[(0x80, 0x9F)]),
            cont,
        ]);
        alts.push(vec![b.class(&[(0xEE, 0xEF)]), cont, cont]);
        alts.push(vec![
            b.class(&[(0xF0, 0xF0)]),
            b.class(&[(0x90, 0xBF)]),
            cont,
            cont,
        ]);
        alts.push(vec![b.class(&[(0xF1, 0xF3)]), cont, cont, cont]);
        alts.push(vec![
            b.class(&[(0xF4, 0xF4)]),
            b.class(&[(0x80, 0x8F)]),
            cont,
            cont,
        ]);
        alts
    }
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Builder {
    rules: Vec<Option<Vec<Vec<Sym>>>>,
    classes: Vec<[u64; 4]>,
    class_ids: HashMap<[u64; 4], u32>,
}

impl Builder {
    fn reserve(&mut self) -> u32 {
        self.rules.push(None);
        (self.rules.len() - 1) as u32
    }

    fn define(&mut self, r: u32, alts: Vec<Vec<Sym>>) {
        self.rules[r as usize] = Some(alts);
    }

    fn rule(&mut self, alts: Vec<Vec<Sym>>) -> u32 {
        let r = self.reserve();
        self.define(r, alts);
        r
    }

    fn class(&mut self, ranges: &[(u8, u8)]) -> Sym {
        let mut bits = [0u64; 4];
        for &(lo, hi) in ranges {
            for b in lo..=hi {
                bits[(b >> 6) as usize] |= 1 << (b & 63);
            }
        }
        let next = self.classes.len() as u32;
        let id = *self.class_ids.entry(bits).or_insert(next);
        if id == next {
            self.classes.push(bits);
        }
        Sym::Class(id)
    }

    fn lit(&mut self, bytes: &[u8]) -> Vec<Sym> {
        bytes.iter().map(|&b| self.class(&[(b, b)])).collect()
    }

    /// Flattens the rules into [`Grammar`]'s element array and rejects left recursion
    /// (a `$ref` cycle that reaches itself without consuming a byte), which would make
    /// the matcher expand forever.
    fn finish(self, root: u32) -> Result<Grammar, ReflexError> {
        let rules: Vec<Vec<Vec<Sym>>> = self
            .rules
            .into_iter()
            .map(|r| r.expect("every reserved rule is defined"))
            .collect();
        let n = rules.len();
        let mut nullable = vec![false; n];
        loop {
            let mut changed = false;
            for (r, alts) in rules.iter().enumerate() {
                if !nullable[r]
                    && alts.iter().any(|a| {
                        a.iter()
                            .all(|s| matches!(s, Sym::Rule(x) if nullable[*x as usize]))
                    })
                {
                    nullable[r] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Edges r -> x where x can come first in r; a cycle is left recursion.
        let mut first: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (r, alts) in rules.iter().enumerate() {
            for a in alts {
                for s in a {
                    match s {
                        Sym::Class(_) => break,
                        Sym::Rule(x) => {
                            first[r].push(*x);
                            if !nullable[*x as usize] {
                                break;
                            }
                        }
                    }
                }
            }
        }
        let mut color = vec![0u8; n]; // 0 new, 1 on the DFS path, 2 done
        for start in 0..n {
            if color[start] != 0 {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            color[start] = 1;
            while let Some(&mut (r, ref mut next)) = stack.last_mut() {
                if let Some(&x) = first[r].get(*next) {
                    *next += 1;
                    match color[x as usize] {
                        0 => {
                            color[x as usize] = 1;
                            stack.push((x as usize, 0));
                        }
                        1 => {
                            return Err(schema_err(
                                "the schema has a \"$ref\" cycle that never consumes input (left recursion)",
                            ))
                        }
                        _ => {}
                    }
                } else {
                    color[r] = 2;
                    stack.pop();
                }
            }
        }

        let mut elems = Vec::new();
        let mut alt_starts = Vec::with_capacity(n);
        for alts in &rules {
            let mut starts = Vec::with_capacity(alts.len());
            for a in alts {
                starts.push(elems.len() as u32);
                elems.extend(a.iter().map(|s| match *s {
                    Sym::Class(c) => Elem::Class(c),
                    Sym::Rule(r) => Elem::Rule(r),
                }));
                elems.push(Elem::End);
            }
            alt_starts.push(starts);
        }
        let root_pos = elems.len() as u32;
        elems.push(Elem::Rule(root));
        elems.push(Elem::End);
        Ok(Grammar {
            elems,
            classes: self.classes,
            alt_starts,
            root_pos,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    Class(u32),
    Rule(u32),
    End,
}

/// A context-free grammar over bytes, flattened: each rule alternative is a run of
/// [`Elem`]s ending in `End`. `root_pos` is a synthetic `[Rule(root), End]`.
struct Grammar {
    elems: Vec<Elem>,
    classes: Vec<[u64; 4]>,
    alt_starts: Vec<Vec<u32>>,
    root_pos: u32,
}

impl Grammar {
    fn class_has(&self, c: u32, b: u8) -> bool {
        self.classes[c as usize][(b >> 6) as usize] & (1 << (b & 63)) != 0
    }
}

const DEAD: u32 = u32::MAX;
const UNKNOWN: u32 = u32::MAX - 1;
/// The empty stack: the value is complete.
const EMPTY: u32 = 0;

/// The pushdown matcher, determinized lazily. A stack is a hash-consed linked list
/// of grammar positions (`nodes[id] = (position, rest)`, [`EMPTY`] at the bottom)
/// whose top is always a byte class; a DFA state is an interned sorted set of stacks.
struct Dfa {
    g: Grammar,
    nodes: Vec<(u32, u32)>,
    node_ids: HashMap<(u32, u32), u32>,
    sets: Vec<Box<[u32]>>,
    set_ids: HashMap<Box<[u32]>, u32>,
    /// `trans[state * 256 + byte]`: the next state, [`DEAD`] or [`UNKNOWN`].
    trans: Vec<u32>,
    accepting: Vec<bool>,
    /// Allowed-token bitsets, per state, once computed.
    masks: Vec<Option<Box<[u64]>>>,
}

impl Dfa {
    fn new(g: Grammar) -> (Dfa, u32) {
        let mut dfa = Dfa {
            g,
            nodes: vec![(u32::MAX, u32::MAX)],
            node_ids: HashMap::new(),
            sets: Vec::new(),
            set_ids: HashMap::new(),
            trans: Vec::new(),
            accepting: Vec::new(),
            masks: Vec::new(),
        };
        let mut start = Vec::new();
        dfa.expand(dfa.g.root_pos, EMPTY, &mut start);
        let s = dfa.intern_set(start);
        (dfa, s)
    }

    fn intern_node(&mut self, pos: u32, rest: u32) -> u32 {
        let next = self.nodes.len() as u32;
        let id = *self.node_ids.entry((pos, rest)).or_insert(next);
        if id == next {
            self.nodes.push((pos, rest));
        }
        id
    }

    fn intern_set(&mut self, mut stacks: Vec<u32>) -> u32 {
        stacks.sort_unstable();
        stacks.dedup();
        if stacks.is_empty() {
            return DEAD;
        }
        let key = stacks.into_boxed_slice();
        if let Some(&id) = self.set_ids.get(&key) {
            return id;
        }
        let id = self.sets.len() as u32;
        self.accepting.push(key.contains(&EMPTY));
        self.sets.push(key.clone());
        self.set_ids.insert(key, id);
        self.trans.extend(std::iter::repeat_n(UNKNOWN, 256));
        self.masks.push(None);
        id
    }

    /// Pushes onto `out` every stack (with a byte class on top, or [`EMPTY`]) reached
    /// from grammar position `pos` above stack `rest`, without consuming input.
    fn expand(&mut self, mut pos: u32, mut rest: u32, out: &mut Vec<u32>) {
        loop {
            match self.g.elems[pos as usize] {
                Elem::End => {
                    if rest == EMPTY {
                        out.push(EMPTY);
                        return;
                    }
                    (pos, rest) = self.nodes[rest as usize];
                }
                Elem::Class(_) => {
                    let id = self.intern_node(pos, rest);
                    out.push(id);
                    return;
                }
                Elem::Rule(r) => {
                    let cont = pos + 1;
                    let inner = if self.g.elems[cont as usize] == Elem::End {
                        rest // tail call: nothing to come back to
                    } else {
                        self.intern_node(cont, rest)
                    };
                    let n = self.g.alt_starts[r as usize].len();
                    for i in 0..n {
                        let start = self.g.alt_starts[r as usize][i];
                        self.expand(start, inner, out);
                    }
                    return;
                }
            }
        }
    }

    #[inline]
    fn step(&mut self, state: u32, b: u8) -> u32 {
        let t = self.trans[((state as usize) << 8) | b as usize];
        if t != UNKNOWN {
            return t;
        }
        self.step_slow(state, b)
    }

    #[cold]
    fn step_slow(&mut self, state: u32, b: u8) -> u32 {
        let mut out = Vec::new();
        for i in 0..self.sets[state as usize].len() {
            let s = self.sets[state as usize][i];
            if s == EMPTY {
                continue;
            }
            let (pos, rest) = self.nodes[s as usize];
            let Elem::Class(c) = self.g.elems[pos as usize] else {
                unreachable!("stack tops are byte classes");
            };
            if self.g.class_has(c, b) {
                self.expand(pos + 1, rest, &mut out);
            }
        }
        let t = self.intern_set(out);
        self.trans[((state as usize) << 8) | b as usize] = t;
        t
    }

    /// The tokens allowed in `state`, as a bitset over the vocabulary: every token
    /// whose bytes all step to a live state, plus EOS if the value is complete.
    fn mask(&mut self, state: u32, index: &TokenIndex) -> &[u64] {
        if self.masks[state as usize].is_none() {
            let mut bits = vec![0u64; index.vocab.div_ceil(64)].into_boxed_slice();
            let n = index.byte.len();
            let mut at_depth = vec![state; index.max_depth + 1];
            let mut i = 0;
            while i < n {
                let d = index.depth[i] as usize;
                let t = self.step(at_depth[d - 1], index.byte[i]);
                if t == DEAD {
                    i = index.skip[i] as usize;
                    continue;
                }
                at_depth[d] = t;
                for &tok in
                    &index.toks[index.tok_start[i] as usize..index.tok_start[i + 1] as usize]
                {
                    bits[(tok >> 6) as usize] |= 1 << (tok & 63);
                }
                i += 1;
            }
            if self.accepting[state as usize] {
                if let Some(eos) = index.eos {
                    bits[(eos >> 6) as usize] |= 1 << (eos & 63);
                }
            }
            self.masks[state as usize] = Some(bits);
        }
        self.masks[state as usize]
            .as_deref()
            .expect("just computed")
    }
}

// ---------------------------------------------------------------------------
// Vocabulary index
// ---------------------------------------------------------------------------

/// The vocabulary as a byte trie in preorder, built once per `Model` on its first
/// constrained request (see `Model::token_picker`). Node `i` holds the byte at
/// `depth[i]` (1-based) of every token below it; `skip[i]` is the first node after
/// its subtree; `toks[tok_start[i]..tok_start[i + 1]]` are the tokens whose bytes end
/// exactly there. Each token's bytes are what it adds to the decoded text
/// (`Tokenizer::token_text_bytes`). Control, unknown and unused tokens are left out
/// (never allowed), and EOS is allowed only when the value is complete.
pub struct TokenIndex {
    byte: Vec<u8>,
    depth: Vec<u16>,
    skip: Vec<u32>,
    tok_start: Vec<u32>,
    toks: Vec<u32>,
    max_depth: usize,
    /// Per token, its text bytes (`token_bytes[off[id]..off[id + 1]]`); empty for a
    /// token left out of the trie.
    off: Vec<u32>,
    token_bytes: Vec<u8>,
    vocab: usize,
    eos: Option<u32>,
    /// SentencePiece-style decoding (`▁` becomes a space), so the grammar must keep
    /// U+2581 out of strings.
    space_marker: bool,
    /// How long [`Self::build`] took.
    pub build_ms: f64,
}

impl TokenIndex {
    pub fn build(tok: &Tokenizer) -> TokenIndex {
        let start = std::time::Instant::now();
        let vocab = tok.vocab_size();
        let mut off = Vec::with_capacity(vocab + 1);
        let mut token_bytes = Vec::new();
        off.push(0u32);
        for id in 0..vocab as u32 {
            if tok.is_text_token(id) && Some(id) != tok.eos_token_id {
                token_bytes.extend(tok.token_text_bytes(id));
            }
            off.push(token_bytes.len() as u32);
        }
        let bytes_of =
            |id: u32| &token_bytes[off[id as usize] as usize..off[id as usize + 1] as usize];
        let mut order: Vec<u32> = (0..vocab as u32)
            .filter(|&id| !bytes_of(id).is_empty())
            .collect();
        order.sort_unstable_by(|&a, &b| bytes_of(a).cmp(bytes_of(b)));

        let (mut byte, mut depth, mut node_of_tok) = (Vec::new(), Vec::new(), Vec::new());
        let mut path: Vec<u32> = Vec::new(); // node index per depth of the current token
        let mut prev: &[u8] = &[];
        for &id in &order {
            let cur = bytes_of(id);
            let common = prev.iter().zip(cur).take_while(|(a, b)| a == b).count();
            path.truncate(common);
            for (d, &b) in cur.iter().enumerate().skip(common) {
                path.push(byte.len() as u32);
                byte.push(b);
                depth.push((d + 1) as u16);
            }
            node_of_tok.push((*path.last().expect("non-empty token"), id));
            prev = cur;
        }
        let n = byte.len();
        let mut skip = vec![n as u32; n];
        let mut open: Vec<usize> = Vec::new();
        for i in 0..n {
            while let Some(&j) = open.last() {
                if depth[j] >= depth[i] {
                    skip[j] = i as u32;
                    open.pop();
                } else {
                    break;
                }
            }
            open.push(i);
        }
        // `order` is sorted by bytes, so tokens arrive grouped by node, in node order.
        let mut tok_start = Vec::with_capacity(n + 1);
        let mut toks = Vec::with_capacity(node_of_tok.len());
        let mut k = 0;
        for i in 0..n {
            tok_start.push(toks.len() as u32);
            while k < node_of_tok.len() && node_of_tok[k].0 == i as u32 {
                toks.push(node_of_tok[k].1);
                k += 1;
            }
        }
        tok_start.push(toks.len() as u32);
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        TokenIndex {
            byte,
            depth,
            skip,
            tok_start,
            toks,
            max_depth,
            off,
            token_bytes,
            vocab,
            eos: tok.eos_token_id,
            space_marker: tok.architecture != "gpt2",
            build_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn bytes_of(&self, id: u32) -> &[u8] {
        match self.off.get(id as usize..id as usize + 2) {
            Some(w) => &self.token_bytes[w[0] as usize..w[1] as usize],
            None => &[],
        }
    }

    /// Trie nodes, for reports.
    pub fn node_count(&self) -> usize {
        self.byte.len()
    }
}

// ---------------------------------------------------------------------------
// Per-generation matcher
// ---------------------------------------------------------------------------

/// One generation's progress through a [`JsonSchema`]: masks logits, then follows
/// each chosen token. The text so far is kept for the `grammar_dead_end` error.
pub struct Constrained<'a> {
    schema: &'a JsonSchema,
    index: &'a TokenIndex,
    dfa: Dfa,
    state: u32,
    finished: bool,
    text: Vec<u8>,
}

impl<'a> Constrained<'a> {
    pub fn new(schema: &'a JsonSchema, index: &'a TokenIndex) -> Result<Self, ReflexError> {
        if index.eos.is_none() {
            return Err(schema_err(
                "this model's tokenizer has no EOS token, so a constrained value can't end",
            ));
        }
        let (dfa, state) = Dfa::new(schema.grammar(index.space_marker)?);
        Ok(Constrained {
            schema,
            index,
            dfa,
            state,
            finished: false,
            text: Vec::new(),
        })
    }

    /// Whether the generated text is a complete value.
    pub fn is_complete(&self) -> bool {
        self.finished || self.dfa.accepting[self.state as usize]
    }

    /// The generated text so far (the value, without EOS).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.text).into_owned()
    }

    /// Sets the logit of every token that can't continue the value to `-inf`.
    /// Errs with `GrammarDeadEnd` if no token can (a vocabulary missing a byte the
    /// grammar needs next).
    pub fn mask(&mut self, logits: &mut [f32], generated: &[u32]) -> Result<(), ReflexError> {
        let start = std::time::Instant::now();
        let bits = self.dfa.mask(self.state, self.index);
        let mut any = false;
        for (w, chunk) in logits.chunks_mut(64).enumerate() {
            let m = bits.get(w).copied().unwrap_or(0);
            any |= m != 0;
            if m == u64::MAX && chunk.len() == 64 {
                continue;
            }
            for (j, l) in chunk.iter_mut().enumerate() {
                if m & (1 << j) == 0 {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        self.schema
            .mask_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !any {
            return Err(self.dead_end(
                "no token in this model's vocabulary can continue the value".to_string(),
                generated,
            ));
        }
        Ok(())
    }

    /// Advances past the chosen token, which [`Self::mask`] allowed.
    pub fn accept(&mut self, id: u32) -> Result<(), ReflexError> {
        if Some(id) == self.index.eos {
            if !self.dfa.accepting[self.state as usize] {
                return Err(ReflexError::Other(
                    "internal error: EOS chosen before the value was complete".to_string(),
                ));
            }
            self.finished = true;
            return Ok(());
        }
        let start = std::time::Instant::now();
        let index = self.index;
        let mut s = self.state;
        for &b in index.bytes_of(id) {
            s = self.dfa.step(s, b);
            if s == DEAD {
                return Err(ReflexError::Other(format!(
                    "internal error: token {id} was chosen but the grammar rejects it"
                )));
            }
        }
        if index.bytes_of(id).is_empty() {
            return Err(ReflexError::Other(format!(
                "internal error: token {id} has no text but was chosen"
            )));
        }
        self.state = s;
        self.text.extend_from_slice(index.bytes_of(id));
        self.schema
            .mask_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The `grammar_dead_end` error, carrying the text and ids generated so far.
    pub fn dead_end(&self, why: String, generated: &[u32]) -> ReflexError {
        ReflexError::GrammarDeadEnd {
            message: format!("json_schema: {why}"),
            text: self.text(),
            token_ids: generated.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests;
