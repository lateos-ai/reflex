//! Host-only tests for `crate::grammar`: the JSON parser, schema compilation and
//! rejection, the automaton on valid and invalid byte streams, randomized soundness
//! (everything the automaton accepts parses and validates), and token masks against
//! a brute-force walk of every vocabulary token on a real Qwen3 (gpt2 byte-level BPE)
//! and TinyLlama (SentencePiece) tokenizer from `test-data/`.

use super::*;
use crate::gguf::GgufFile;
use crate::sampling::{SamplingParams, TokenPicker};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const QWEN3_TOKENIZER: &str = "qwen3-0.6b-tokenizer.gguf";
const TINYLLAMA_TOKENIZER: &str = "tinyllama-1.1b-chat-tokenizer.gguf";

/// A nested schema in the shape of an agent handoff, used by several tests.
const HANDOFF: &str = r##"{
  "type": "object",
  "properties": {
    "agent": {"type": "string"},
    "status": {"enum": ["done", "blocked", "needs_review"]},
    "confidence": {"type": "number"},
    "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
    "notes": {"type": ["string", "null"]}
  },
  "required": ["agent", "status", "steps"],
  "additionalProperties": false,
  "$defs": {
    "step": {
      "type": "object",
      "properties": {
        "tool": {"const": "search"},
        "args": {"type": "object"},
        "ok": {"type": "boolean"},
        "count": {"type": "integer"}
      },
      "required": ["tool", "ok"]
    }
  }
}"##;

fn tokenizer(name: &str) -> Option<Tokenizer> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-data")
        .join(name);
    if !path.exists() {
        eprintln!("Skipping: test-data/{name} not present");
        return None;
    }
    let file = GgufFile::open(&path).expect("open tokenizer GGUF");
    Some(Tokenizer::from_gguf(&file).expect("Tokenizer::from_gguf"))
}

fn dfa(schema: &str, forbid_u2581: bool) -> (Dfa, u32) {
    let s = JsonSchema::compile(schema).unwrap_or_else(|e| panic!("{schema}: {e}"));
    Dfa::new(s.grammar(forbid_u2581).expect("grammar"))
}

fn run(d: &mut Dfa, mut st: u32, bytes: &[u8]) -> Option<u32> {
    for &b in bytes {
        st = d.step(st, b);
        if st == DEAD {
            return None;
        }
    }
    Some(st)
}

/// Whether `text` is a complete value under `schema`.
fn accepts(schema: &str, text: &[u8]) -> bool {
    let (mut d, st) = dfa(schema, false);
    run(&mut d, st, text).is_some_and(|s| d.accepting[s as usize])
}

/// Whether `text` can still be completed into a value under `schema`.
fn viable(schema: &str, text: &[u8]) -> bool {
    let (mut d, st) = dfa(schema, false);
    run(&mut d, st, text).is_some()
}

fn rejected_keyword(schema: &str) -> String {
    match JsonSchema::compile(schema) {
        Ok(_) => panic!("{schema} should be rejected"),
        Err(e) => {
            assert_eq!(e.kind(), "invalid_input", "{e}");
            e.to_string()
        }
    }
}

// --- JSON parser -----------------------------------------------------------

#[test]
fn parser_keeps_member_order_and_number_text() {
    let v =
        JsonValue::parse(r##" {"b": 1.50, "a": [true, null, "x\u00e9\ud83d\ude00"]} "##).unwrap();
    let JsonValue::Object(m) = &v else { panic!() };
    assert_eq!(m[0].0, "b");
    assert_eq!(m[1].0, "a");
    assert_eq!(
        v.to_compact(),
        "{\"b\":1.50,\"a\":[true,null,\"x\u{e9}\u{1f600}\"]}"
    );
}

#[test]
fn parser_rejects_malformed_json() {
    for bad in [
        "",
        "{",
        "[1,]",
        "{\"a\":}",
        "01",
        "1.",
        "1e",
        "-",
        "\"\\x\"",
        "\"\u{1}\"",
        "\"\\ud800\"",
        "\"\\udc00\"",
        "\"\\ud800\\u0041\"",
        "tru",
        "{} x",
        "{\"a\" 1}",
        "[1 2]",
        "+1",
        ".5",
    ] {
        assert!(JsonValue::parse(bad).is_err(), "{bad:?} should not parse");
    }
}

#[test]
fn compact_serialization_escapes_minimally() {
    let v = JsonValue::String("q\"b\\n\n\t\u{1}é".to_string());
    assert_eq!(v.to_compact(), "\"q\\\"b\\\\n\\n\\t\\u0001é\"");
    assert_eq!(JsonValue::parse(&v.to_compact()).unwrap(), v);
}

// --- schema compile and rejection ------------------------------------------

#[test]
fn supported_schemas_compile() {
    for ok in [
        HANDOFF,
        "{}",
        "true",
        r##"{"type":"string","title":"t","description":"d","$comment":"c","default":"x","examples":["y"]}"##,
        r##"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"integer"}"##,
        r##"{"type":["integer","null"]}"##,
        r##"{"anyOf":[{"type":"string"},{"type":"number"}],"description":"x"}"##,
        r##"{"enum":[1,"a",null,[1,2],{"k":true}]}"##,
        r##"{"type":"object","properties":{"next":{"anyOf":[{"$ref":"#"},{"type":"null"}]}},"required":["next"]}"##,
        r##"{"$ref":"#/$defs/a","$defs":{"a":{"type":"array","items":{"$ref":"#/$defs/a"}}}}"##,
        r##"{"$defs":{"a/b":{"type":"null"}},"$ref":"#/$defs/a~1b"}"##,
        r##"{"type":"object","additionalProperties":false}"##,
        r##"{"items":{"type":"boolean"}}"##,
    ] {
        JsonSchema::compile(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
}

#[test]
fn unsupported_keywords_are_rejected_by_name() {
    for kw in [
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "prefixItems",
        "contains",
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
        "propertyNames",
        "minProperties",
        "dependentRequired",
        "definitions",
        "$id",
        "$anchor",
        "$dynamicRef",
        "unevaluatedProperties",
        "nullable",
    ] {
        let msg = rejected_keyword(&format!(r##"{{"type":"string","{kw}":1}}"##));
        assert!(msg.contains(&format!("\"{kw}\"")), "{kw}: {msg}");
        // Nested too.
        let msg = rejected_keyword(&format!(
            r##"{{"type":"object","properties":{{"a":{{"type":"string","{kw}":1}}}}}}"##
        ));
        assert!(msg.contains(&format!("\"{kw}\"")), "{kw}: {msg}");
        assert!(msg.contains("#/properties/a"), "{msg}");
    }
}

#[test]
fn bad_schemas_are_rejected() {
    for (schema, needle) in [
        ("not json", "not valid JSON"),
        ("[]", "expected a schema object"),
        ("false", "`false`"),
        (r##"{"type":"float"}"##, "unsupported \"type\""),
        (r##"{"type":[]}"##, "empty \"type\""),
        (
            r##"{"type":"object","additionalProperties":true}"##,
            "additionalProperties",
        ),
        (
            r##"{"type":"object","additionalProperties":{"type":"string"}}"##,
            "additionalProperties",
        ),
        (
            r##"{"type":"object","properties":{"a":{}},"required":["b"]}"##,
            "\"b\" is not in \"properties\"",
        ),
        (
            r##"{"type":"object","required":["a","a"],"properties":{"a":{}}}"##,
            "twice",
        ),
        (r##"{"items":[{"type":"string"}]}"##, "tuple"),
        (r##"{"enum":[]}"##, "non-empty"),
        (r##"{"type":"string","enum":[1,2]}"##, "satisfies"),
        (r##"{"enum":[1,2],"const":3}"##, "satisfies"),
        (r##"{"$ref":"https://example.com/s.json"}"##, "only \"#\""),
        (r##"{"$ref":"#/$defs/missing"}"##, "names no entry"),
        (r##"{"$ref":"#/definitions/a"}"##, "only \"#\""),
        (
            r##"{"$ref":"#/$defs/a","type":"string","$defs":{"a":{}}}"##,
            "can't be combined with \"type\"",
        ),
        (
            r##"{"anyOf":[{"type":"string"}],"type":"string"}"##,
            "can't be combined with \"type\"",
        ),
        (r##"{"anyOf":[]}"##, "empty \"anyOf\""),
        (
            r##"{"type":"object","properties":{"a":{"$defs":{}}}}"##,
            "only supported at the schema root",
        ),
        (
            r##"{"$defs":{"a":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
            "left recursion",
        ),
        (
            r##"{"$defs":{"a":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/b"}]},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
            "left recursion",
        ),
        (r##"{"type":"string","type":"null"}"##, "duplicate key"),
        (r##"{"properties":{"a":{},"a":{}}}"##, "duplicate"),
    ] {
        let msg = rejected_keyword(schema);
        assert!(msg.contains(needle), "{schema}: {msg}");
    }
}

#[test]
fn u2581_literals_are_rejected_for_sentencepiece_tokenizers() {
    let s = JsonSchema::compile(r##"{"const":"a\u2581b"}"##).unwrap();
    assert!(s.grammar(false).is_ok());
    let e = s.grammar(true).err().expect("rejected");
    assert!(e.to_string().contains("U+2581"), "{e}");
    let s = JsonSchema::compile(r##"{"type":"object","properties":{"▁k":{}}}"##).unwrap();
    assert!(s.grammar(true).is_err());
}

// --- the automaton ---------------------------------------------------------

#[test]
fn primitives_accept_exactly_compact_json() {
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            r##"{"type":"integer"}"##,
            &["0", "-0", "7", "-12", "1234567890"],
            &["", "01", "-", "1.0", "1e3", " 1", "1 ", "+1", "--1"],
        ),
        (
            r##"{"type":"number"}"##,
            &["0", "-1.5", "3.14e-10", "1E+2", "2e5", "0.0"],
            &["1.", ".5", "1e", "1e+", "00", "-.1", "NaN", "Infinity"],
        ),
        (
            r##"{"type":"boolean"}"##,
            &["true", "false"],
            &["True", "1", "null", "tru"],
        ),
        (
            r##"{"type":"null"}"##,
            &["null"],
            &["nul", "None", "\"null\""],
        ),
        (
            r##"{"type":"string"}"##,
            &[
                "\"\"",
                "\"a b\"",
                "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"",
                "\"\\u00e9\\uD83D\\uDE00\"",
                "\"é😀中\"",
                "\"\u{7f}\"",
            ],
            &[
                "\"",
                "\"\n\"",
                "\"\t\"",
                "\"\\x\"",
                "\"\\uD800\"",
                "\"\\uDC00\"",
                "\"\\uD800\\u0041\"",
                "\"\\u12\"",
                "'a'",
                "\"a\"\"",
            ],
        ),
        (
            r##"{"type":"array","items":{"type":"integer"}}"##,
            &["[]", "[1]", "[1,2,-3]", "[1, 2]", "[1,2, 3]"],
            &[
                "[", "[,]", "[1,]", "[ 1]", "[1 ]", "[1 ,2]", "[1,  2]", "[1,\n2]", "[\"1\"]",
                "[1]]", "[1] ",
            ],
        ),
        (
            r##"{"type":"object"}"##,
            &[
                "{}",
                "{\"a\":1}",
                "{\"a\":{\"b\":[null,{}]},\"c\":\"d\"}",
                "{\"a\": 1, \"b\": [1, 2]}",
            ],
            &[
                "{",
                "{a:1}",
                "{\"a\":1,}",
                "{\"a\" :1}",
                "{\"a\":}",
                "{ }",
                "{ \"a\":1}",
                "{\"a\":1 }",
                "{\"a\":  1}",
                "{\"a\":\t1}",
                "{\"a\":\n1}",
            ],
        ),
        (
            "{}",
            &["1", "\"s\"", "[[],{}]", "null", "{\"k\":[true]}"],
            &["", " ", "{]", "nan"],
        ),
    ];
    for &(schema, good, bad) in cases {
        for g in good {
            assert!(
                accepts(schema, g.as_bytes()),
                "{schema} should accept {g:?}"
            );
        }
        for b in bad {
            assert!(
                !accepts(schema, b.as_bytes()),
                "{schema} should reject {b:?}"
            );
        }
    }
}

#[test]
fn strings_check_utf8_byte_by_byte() {
    let s = r##"{"type":"string"}"##;
    for good in [
        &b"\"\xc3\xa9\""[..],
        b"\"\xe2\x82\xac\"",
        b"\"\xf0\x9f\x98\x80\"",
        b"\"\xed\x9f\xbf\"",
        b"\"\xf4\x8f\xbf\xbf\"",
    ] {
        assert!(accepts(s, good), "{good:?}");
    }
    for bad in [
        &b"\"\xc0\xaf\""[..], // overlong
        b"\"\xc1\xbf\"",
        b"\"\xe0\x80\xaf\"",     // overlong
        b"\"\xed\xa0\x80\"",     // UTF-16 surrogate
        b"\"\xf4\x90\x80\x80\"", // past U+10FFFF
        b"\"\xf5\x80\x80\x80\"",
        b"\"\x80\"", // stray continuation
        b"\"\xc3\"", // truncated before the quote
        b"\"\xe2\x82\"",
        b"\"\xff\"",
    ] {
        assert!(!accepts(s, bad), "{bad:?}");
    }
    // Partway through a sequence or an escape is viable; the rest must follow.
    for partial in [
        &b"\"\xe2"[..],
        b"\"\xe2\x82",
        b"\"\\",
        b"\"\\u",
        b"\"\\uD83D",
        b"\"\\uD83D\\u",
    ] {
        assert!(viable(s, partial), "{partial:?}");
        assert!(!accepts(s, partial), "{partial:?}");
    }
    assert!(
        !viable(s, b"\"\\uD83D\\u0"),
        "a low surrogate must follow a high one"
    );
}

#[test]
fn u2581_is_kept_out_of_strings_for_sentencepiece() {
    let (mut d, st) = dfa(r##"{"type":"string"}"##, true);
    assert!(run(&mut d, st, "\"a\u{2581}\"".as_bytes()).is_none());
    for ok in ["\"\u{2580}\u{2582}\u{2481}\u{2591}€\"", "\"a b\""] {
        let s = run(&mut d, st, ok.as_bytes()).expect(ok);
        assert!(d.accepting[s as usize], "{ok}");
    }
}

#[test]
fn objects_follow_schema_order_and_required() {
    let s = r##"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"},"c":{"type":"boolean"}},"required":["b"]}"##;
    for good in [
        r##"{"b":""}"##,
        r##"{"a":1,"b":"x"}"##,
        r##"{"b":"x","c":true}"##,
        r##"{"a":1,"b":"x","c":false}"##,
        r##"{"a": 1, "b": "x"}"##,
        r##"{"b": "x","c":true}"##,
    ] {
        assert!(accepts(s, good.as_bytes()), "{good}");
    }
    for bad in [
        "{}",
        r##"{"a":1}"##,
        r##"{"b":"x","a":1}"##,
        r##"{"a":1,"c":true}"##,
        r##"{"b":"x","d":1}"##,
        r##"{"b":"x",}"##,
        r##"{,"b":"x"}"##,
        r##"{"b":"x","b":"y"}"##,
        r##"{ "b":"x"}"##,
        r##"{"b" :"x"}"##,
        r##"{"b":  "x"}"##,
        r##"{"b":"x" }"##,
        r##"{"b":"x" ,"c":true}"##,
        "{\n\"b\":\"x\"}",
    ] {
        assert!(!accepts(s, bad.as_bytes()), "{bad}");
    }
    assert!(accepts(
        r##"{"type":"object","additionalProperties":false}"##,
        b"{}"
    ));
    assert!(!accepts(
        r##"{"type":"object","additionalProperties":false}"##,
        b"{\"a\":1}"
    ));
}

#[test]
fn enum_const_and_anyof() {
    let e = r##"{"enum":["done","do",12,1,{"k":[1]},null]}"##;
    for good in ["\"done\"", "\"do\"", "12", "1", "{\"k\":[1]}", "null"] {
        assert!(accepts(e, good.as_bytes()), "{good}");
    }
    for bad in ["\"d\"", "\"don\"", "2", "123", "{\"k\":[1,2]}", "\"null\""] {
        assert!(!accepts(e, bad.as_bytes()), "{bad}");
    }
    // `type` filters the enum values it doesn't allow.
    let typed = r##"{"type":"string","enum":["a",1,"b"]}"##;
    assert!(accepts(typed, b"\"a\""));
    assert!(!accepts(typed, b"1"));
    // Overlapping branches: the matcher keeps both until they part.
    let any = r##"{"anyOf":[{"type":"integer"},{"type":"number"},{"type":"object","properties":{"x":{"type":"null"}},"required":["x"]},{"type":"object","properties":{"y":{"type":"null"}},"required":["y"]}]}"##;
    for good in ["12", "12.5", "{\"x\":null}", "{\"y\":null}"] {
        assert!(accepts(any, good.as_bytes()), "{good}");
    }
    for bad in ["{}", "{\"x\":null,\"y\":null}", "\"s\""] {
        assert!(!accepts(any, bad.as_bytes()), "{bad}");
    }
}

#[test]
fn recursive_refs_nest() {
    let tree = r##"{"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},"$ref":"#/$defs/node"}"##;
    assert!(accepts(
        tree,
        br##"{"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4,"kids":[]}]}]}"##
    ));
    assert!(!accepts(tree, br##"{"v":1,"kids":[{"kids":[]}]}"##));
    let list = r##"{"type":"object","properties":{"next":{"anyOf":[{"$ref":"#"},{"type":"null"}]}},"required":["next"]}"##;
    assert!(accepts(list, br##"{"next":{"next":{"next":null}}}"##));
}

#[test]
fn handoff_schema_accepts_and_rejects() {
    assert!(accepts(
        HANDOFF,
        br##"{"agent":"planner","status":"done","confidence":0.9,"steps":[{"tool":"search","args":{"q":"x"},"ok":true,"count":3}],"notes":null}"##
    ));
    assert!(accepts(
        HANDOFF,
        br##"{"agent":"a","status":"blocked","steps":[]}"##
    ));
    assert!(!accepts(
        HANDOFF,
        br##"{"agent":"a","status":"finished","steps":[]}"##
    ));
    assert!(!accepts(
        HANDOFF,
        br##"{"agent":"a","status":"done","steps":[{"tool":"fetch","ok":true}]}"##
    ));
    assert!(!accepts(
        HANDOFF,
        br##"{"agent":"a","status":"done","steps":[{"tool":"search","ok":true,"count":1.5}]}"##
    ));
}

/// Walks the automaton with random bytes (biased towards closing structure so most
/// walks finish) and checks every complete output parses, with this module's parser
/// and with `serde_json` when built, and validates against the schema.
#[test]
fn random_walks_always_produce_valid_json() {
    let schemas = [
        HANDOFF,
        "{}",
        r##"{"type":"array","items":{"type":["number","string","boolean"]}}"##,
        r##"{"anyOf":[{"type":"integer"},{"enum":["x",[1,{"a":null}]]},{"type":"object","properties":{"s":{"type":"string"}}}]}"##,
        r##"{"$defs":{"n":{"type":"object","properties":{"kids":{"type":"array","items":{"$ref":"#/$defs/n"}}}}},"$ref":"#/$defs/n"}"##,
    ];
    let mut rng = StdRng::seed_from_u64(7);
    for schema in schemas {
        let compiled = JsonSchema::compile(schema).unwrap();
        for forbid in [false, true] {
            let (mut d, start) = Dfa::new(compiled.grammar(forbid).unwrap());
            let mut finished = 0;
            for _ in 0..300 {
                let mut st = start;
                let mut out = Vec::new();
                let mut done = false;
                for _ in 0..300 {
                    let live: Vec<u8> = (0..=255u8).filter(|&b| d.step(st, b) != DEAD).collect();
                    let accepting = d.accepting[st as usize];
                    assert!(
                        accepting || !live.is_empty(),
                        "{schema}: stuck after {out:?}"
                    );
                    if accepting && (live.is_empty() || rng.gen_bool(0.3)) {
                        done = true;
                        break;
                    }
                    let weight = |b: u8| if b"\"]},:0tfn".contains(&b) { 30 } else { 1 };
                    let total: u32 = live.iter().map(|&b| weight(b)).sum();
                    let mut pick = rng.gen_range(0..total);
                    let b = *live
                        .iter()
                        .find(|&&b| {
                            let w = weight(b);
                            if pick < w {
                                true
                            } else {
                                pick -= w;
                                false
                            }
                        })
                        .unwrap();
                    st = d.step(st, b);
                    out.push(b);
                }
                if !done {
                    continue;
                }
                finished += 1;
                let text = String::from_utf8(out).expect("the automaton only accepts UTF-8");
                if forbid {
                    assert!(!text.contains('\u{2581}'), "{text}");
                }
                let v = JsonValue::parse(&text).unwrap_or_else(|e| panic!("{schema}: {text}: {e}"));
                compiled
                    .validate(&v)
                    .unwrap_or_else(|e| panic!("{schema}: {text}: {e}"));
                // serde_json also parses it, except a number past f64's range
                // (`1e400`), which RFC 8259 allows and serde_json refuses.
                #[cfg(feature = "json-output")]
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&text) {
                    assert!(
                        e.to_string().contains("number out of range"),
                        "serde_json: {text}: {e}"
                    );
                }
            }
            assert!(finished > 50, "{schema}: only {finished} walks finished");
        }
    }
}

#[test]
fn validator_matches_json_schema_semantics() {
    let s = JsonSchema::compile(HANDOFF).unwrap();
    let ok = JsonValue::parse(
        r##"{"steps":[{"ok":false,"tool":"search","count":2.0}],"status":"done","agent":"x"}"##,
    )
    .unwrap();
    s.validate(&ok).unwrap();
    for bad in [
        r##"{"agent":"x","status":"done"}"##,
        r##"{"agent":"x","status":"done","steps":[],"extra":1}"##,
        r##"{"agent":1,"status":"done","steps":[]}"##,
        r##"{"agent":"x","status":"done","steps":[{"tool":"search","ok":true,"count":1.5}]}"##,
    ] {
        assert!(
            s.validate(&JsonValue::parse(bad).unwrap()).is_err(),
            "{bad}"
        );
    }
}

// --- vocabulary index and masks --------------------------------------------

fn brute_force_mask(d: &mut Dfa, st: u32, index: &TokenIndex) -> Vec<u64> {
    let mut bits = vec![0u64; index.vocab.div_ceil(64)];
    for id in 0..index.vocab as u32 {
        let allowed = if Some(id) == index.eos {
            d.accepting[st as usize]
        } else {
            let b = index.bytes_of(id);
            !b.is_empty() && run(d, st, b).is_some()
        };
        if allowed {
            bits[(id >> 6) as usize] |= 1 << (id & 63);
        }
    }
    bits
}

fn assert_masks_match_brute_force(tok: &Tokenizer) {
    let index = TokenIndex::build(tok);
    eprintln!(
        "{}: {} tokens, {} trie nodes, built in {:.1} ms",
        tok.architecture,
        index.vocab,
        index.node_count(),
        index.build_ms
    );
    let schema = JsonSchema::compile(HANDOFF).unwrap();
    let (mut d, start) = Dfa::new(schema.grammar(index.space_marker).unwrap());
    let prefixes: &[&[u8]] = &[
        b"",
        b"{",
        b"{\"agent\":",
        b"{\"agent\": ",
        b"{\"agent\": \"x\", ",
        b"{\"agent\":\"",
        b"{\"agent\":\"pl",
        b"{\"agent\":\"a\\",
        b"{\"agent\":\"a\\u",
        b"{\"agent\":\"a\\uD83D",
        b"{\"agent\":\"a\\uD83D\\u",
        b"{\"agent\":\"\xe4",
        b"{\"agent\":\"\xf0\x9f",
        b"{\"agent\":\"x\",\"status\":\"",
        b"{\"agent\":\"x\",\"status\":\"do",
        b"{\"agent\":\"x\",\"status\":\"done\",\"confidence\":1",
        b"{\"agent\":\"x\",\"status\":\"done\",\"confidence\":1.",
        b"{\"agent\":\"x\",\"status\":\"done\",\"steps\":[",
        b"{\"agent\":\"x\",\"status\":\"done\",\"steps\":[{\"tool\":\"search\",\"args\":{",
        b"{\"agent\":\"x\",\"status\":\"done\",\"steps\":[]",
        b"{\"agent\":\"x\",\"status\":\"done\",\"steps\":[]}",
    ];
    for &p in prefixes {
        let st = run(&mut d, start, p).unwrap_or_else(|| panic!("prefix {p:?} rejected"));
        let fast = d.mask(st, &index).to_vec();
        let slow = brute_force_mask(&mut d, st, &index);
        let n: u32 = fast.iter().map(|w| w.count_ones()).sum();
        assert_eq!(fast, slow, "{}: mask differs after {p:?}", tok.architecture);
        assert!(n > 0, "{p:?}: nothing allowed");
        let complete = d.accepting[st as usize];
        let eos = tok.eos_token_id.unwrap();
        assert_eq!(
            fast[(eos >> 6) as usize] >> (eos & 63) & 1 == 1,
            complete,
            "{p:?}"
        );
    }
}

#[test]
fn qwen3_masks_match_brute_force() {
    let Some(tok) = tokenizer(QWEN3_TOKENIZER) else {
        return;
    };
    assert_eq!(tok.architecture, "gpt2");
    assert_masks_match_brute_force(&tok);
}

#[test]
fn sentencepiece_masks_match_brute_force() {
    let Some(tok) = tokenizer(TINYLLAMA_TOKENIZER) else {
        return;
    };
    assert_eq!(tok.architecture, "llama");
    assert_masks_match_brute_force(&tok);
}

/// The index build before the sort and byte buffer were sped up: one `Vec` per
/// token from `Tokenizer::token_text_bytes` and a byte-slice comparison sort
/// (equal bytes ordered by id, which the old unstable sort left unspecified).
fn reference_index(tok: &Tokenizer) -> TokenIndex {
    let vocab = tok.vocab_size();
    let mut off = vec![0u32];
    let mut token_bytes = Vec::new();
    for id in 0..vocab as u32 {
        if tok.is_text_token(id) && Some(id) != tok.eos_token_id {
            token_bytes.extend(tok.token_text_bytes(id));
        }
        off.push(token_bytes.len() as u32);
    }
    let bytes_of = |id: u32| &token_bytes[off[id as usize] as usize..off[id as usize + 1] as usize];
    let mut order: Vec<u32> = (0..vocab as u32)
        .filter(|&id| !bytes_of(id).is_empty())
        .collect();
    order.sort_by(|&a, &b| bytes_of(a).cmp(bytes_of(b)).then(a.cmp(&b)));
    let (mut byte, mut depth, mut node_of_tok) = (Vec::new(), Vec::<u16>::new(), Vec::new());
    let mut path: Vec<u32> = Vec::new();
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
        node_of_tok.push((*path.last().unwrap(), id));
        prev = cur;
    }
    let n = byte.len();
    let mut skip = vec![n as u32; n];
    for i in 0..n {
        // Quadratic but obviously right: the first later node at depth <= this one.
        skip[i] = (i + 1..n).find(|&j| depth[j] <= depth[i]).unwrap_or(n) as u32;
    }
    let mut at_node = vec![Vec::new(); n];
    for (node, id) in node_of_tok {
        at_node[node as usize].push(id);
    }
    let mut tok_start = Vec::new();
    let mut toks = Vec::new();
    for ids in at_node {
        tok_start.push(toks.len() as u32);
        toks.extend(ids);
    }
    tok_start.push(toks.len() as u32);
    TokenIndex {
        max_depth: depth.iter().copied().max().unwrap_or(0) as usize,
        byte,
        depth,
        skip,
        tok_start,
        toks,
        off,
        token_bytes,
        vocab,
        eos: tok.eos_token_id,
        space_marker: tok.architecture != "gpt2",
        build_ms: 0.0,
    }
}

/// The index build gives the same arrays as [`reference_index`] on a gpt2 byte-level
/// BPE (Qwen3), a SentencePiece (TinyLlama) and the Gemma 4 vocabulary.
/// `REFLEX_INDEX_REF_ALL=1` checks every GGUF in `test-data/` with a tokenizer instead
/// (slow in a debug build; use `--release`).
#[test]
fn index_build_matches_reference() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let names: Vec<String> = if std::env::var("REFLEX_INDEX_REF_ALL").is_ok_and(|v| v == "1") {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".gguf"))
            .collect();
        names.sort();
        names
    } else {
        [
            QWEN3_TOKENIZER,
            TINYLLAMA_TOKENIZER,
            "gemma4-tokenizer.gguf",
        ]
        .map(String::from)
        .to_vec()
    };
    let mut checked = 0;
    for name in names {
        let Ok(file) = GgufFile::open(dir.join(&name)) else {
            eprintln!("Skipping: test-data/{name} not present");
            continue;
        };
        let Ok(tok) = Tokenizer::from_gguf(&file) else {
            continue;
        };
        let (new, old) = (TokenIndex::build(&tok), reference_index(&tok));
        assert!(new.arrays() == old.arrays(), "{name}");
        assert_eq!(new.max_depth, old.max_depth, "{name}");
        eprintln!(
            "{name}: {} tokens, {} nodes, built in {:.1} ms",
            new.vocab,
            new.node_count(),
            new.build_ms
        );
        checked += 1;
    }
    eprintln!("{checked} tokenizers checked");
}

fn id_of(tok: &Tokenizer, piece: &str) -> u32 {
    tok.tokens
        .iter()
        .position(|t| t == piece)
        .unwrap_or_else(|| panic!("{piece:?} not in vocab")) as u32
}

fn allowed(d: &mut Dfa, index: &TokenIndex, st: u32, id: u32) -> bool {
    d.mask(st, index)[(id >> 6) as usize] >> (id & 63) & 1 == 1
}

/// A byte token in the middle of a UTF-8 character is allowed inside a string and
/// nowhere else, and special tokens never are.
#[test]
fn partial_utf8_and_special_tokens_on_real_vocabs() {
    if let Some(tok) = tokenizer(QWEN3_TOKENIZER) {
        let index = TokenIndex::build(&tok);
        let (mut d, start) = dfa(
            r##"{"type":"object","properties":{"s":{"type":"string"}}}"##,
            false,
        );
        // gpt2's byte alphabet: 0xE4 is 'ä' (U+00E4), a lone lead byte as a token.
        let lead = id_of(&tok, "ä");
        assert_eq!(index.bytes_of(lead), &[0xE4]);
        let in_string = run(&mut d, start, b"{\"s\":\"").unwrap();
        assert!(allowed(&mut d, &index, in_string, lead));
        assert!(!allowed(&mut d, &index, start, lead));
        let mid = run(&mut d, in_string, &[0xE4]).unwrap();
        assert!(
            !allowed(&mut d, &index, mid, id_of(&tok, "\"")),
            "must finish the character first"
        );
        for special in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
            assert!(
                !allowed(&mut d, &index, in_string, id_of(&tok, special)),
                "{special}"
            );
        }
        // USER_DEFINED text tokens (`<think>`) are text, as in llama.cpp.
        assert!(allowed(&mut d, &index, in_string, id_of(&tok, "<think>")));
        assert!(!allowed(&mut d, &index, start, id_of(&tok, "<think>")));
    }
    if let Some(tok) = tokenizer(TINYLLAMA_TOKENIZER) {
        let index = TokenIndex::build(&tok);
        let (mut d, start) = dfa(
            r##"{"type":"object","properties":{"s":{"type":"string"}}}"##,
            true,
        );
        let lead = id_of(&tok, "<0xE4>");
        assert_eq!(index.bytes_of(lead), &[0xE4]);
        let in_string = run(&mut d, start, b"{\"s\":\"").unwrap();
        assert!(allowed(&mut d, &index, in_string, lead));
        assert!(!allowed(&mut d, &index, start, lead));
        // `▁` is a space in decoded text: allowed inside the string, not before `{`.
        let space_word = id_of(&tok, "\u{2581}the");
        assert_eq!(index.bytes_of(space_word), b" the");
        assert!(allowed(&mut d, &index, in_string, space_word));
        assert!(!allowed(&mut d, &index, start, id_of(&tok, "\u{2581}{")));
        // U+2581 can't be assembled from byte tokens either.
        let st = run(&mut d, in_string, &[0xE2, 0x96]).unwrap();
        assert!(!allowed(&mut d, &index, st, id_of(&tok, "<0x81>")));
        for special in ["<s>", "</s>", "<unk>"] {
            assert!(
                !allowed(&mut d, &index, in_string, id_of(&tok, special)),
                "{special}"
            );
        }
    }
}

// --- whole generations through TokenPicker ---------------------------------

/// Runs a constrained generation over fake logits: `prefer(step)` names the token
/// to give the highest logit at each step; the rest get small random values plus
/// `bias`.
fn fake_generate(
    tok: &Tokenizer,
    index: &TokenIndex,
    schema: &Arc<JsonSchema>,
    max_tokens: usize,
    mut prefer: impl FnMut(usize) -> Option<u32>,
    bias: Option<&[f32]>,
    seed: u64,
) -> Result<(Vec<u32>, String), ReflexError> {
    let params = SamplingParams {
        json_schema: Some(schema.clone()),
        ..Default::default()
    };
    let constraint = Constrained::new(schema, index)?;
    let mut picker = TokenPicker::new(&params, Some(constraint), tok.eos_token_id);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut generated = Vec::new();
    let mut pending = Vec::new();
    let mut streamed = String::new();
    while generated.len() < max_tokens && generated.last().copied() != tok.eos_token_id {
        let mut logits: Vec<f32> = (0..tok.vocab_size() + 7)
            .map(|_| rng.gen::<f32>())
            .collect();
        if let Some(bias) = bias {
            for (l, b) in logits.iter_mut().zip(bias) {
                *l += b;
            }
        }
        if let Some(p) = prefer(generated.len()) {
            logits[p as usize] = 100.0;
        }
        let id = picker.pick(&mut logits)?;
        streamed.push_str(&picker.stream_text(tok, &mut pending, id));
        generated.push(id);
    }
    let text = picker.finish(tok, &generated)?;
    assert_eq!(streamed, text, "streamed pieces join up to the value");
    Ok((generated, text))
}

#[test]
fn constrained_generations_decode_to_valid_json() {
    for name in [QWEN3_TOKENIZER, TINYLLAMA_TOKENIZER] {
        let Some(tok) = tokenizer(name) else { continue };
        let index = TokenIndex::build(&tok);
        let schema = Arc::new(JsonSchema::compile(HANDOFF).unwrap());
        let eos = tok.eos_token_id.unwrap();
        // Random logits, with EOS strongly preferred so values end soon after they
        // become complete, and tokens that close strings, arrays and objects favored
        // so they do.
        let bias: Vec<f32> = (0..tok.vocab_size() as u32)
            .map(|id| {
                let b = index.bytes_of(id);
                if b.iter().any(|c| b"\"]}".contains(c)) {
                    3.0
                } else {
                    0.0
                }
            })
            .collect();
        let mut ok = 0;
        for seed in 0..20 {
            match fake_generate(&tok, &index, &schema, 400, |_| Some(eos), Some(&bias), seed) {
                Ok((ids, text)) => {
                    ok += 1;
                    assert_eq!(*ids.last().unwrap(), eos);
                    let body = &ids[..ids.len() - 1];
                    assert_eq!(
                        tok.decode(body),
                        text,
                        "{name}: decode differs from the value"
                    );
                    let v =
                        JsonValue::parse(&text).unwrap_or_else(|e| panic!("{name}: {text}: {e}"));
                    schema
                        .validate(&v)
                        .unwrap_or_else(|e| panic!("{name}: {text}: {e}"));
                }
                Err(e) => assert_eq!(e.kind(), "grammar_dead_end", "{e}"),
            }
        }
        assert!(ok >= 10, "{name}: only {ok}/20 random generations finished");
        eprintln!(
            "{name}: mask_ms_total over 20 random generations = {:.1}",
            schema.mask_ms_total()
        );
    }
}

/// When the preferred (greedy) tokens already spell a conforming value, the
/// constrained output is exactly those tokens; when they don't, it still conforms.
/// Runs `value`'s own tokens (then EOS) as the greedy choices.
fn greedy_through(
    tok: &Tokenizer,
    index: &TokenIndex,
    schema: &Arc<JsonSchema>,
    value: &str,
    seed: u64,
) -> Result<(Vec<u32>, String, Vec<u32>), ReflexError> {
    let mut want = tok.encode(value).unwrap();
    if tok.decode(&want).starts_with(' ') {
        // SentencePiece's dummy-prefix space: a model continuing a prompt emits
        // the bare `{` token.
        let piece = tok.tokens[want[0] as usize].clone();
        want[0] = id_of(tok, piece.strip_prefix('\u{2581}').expect("space marker"));
    }
    assert_eq!(tok.decode(&want), value, "test value must round-trip");
    want.push(tok.eos_token_id.unwrap());
    let w = want.clone();
    let (ids, text) = fake_generate(tok, index, schema, 400, |i| w.get(i).copied(), None, seed)?;
    Ok((ids, text, want))
}

/// When the preferred (greedy) tokens already spell a value the format allows,
/// compact or with one space after each `:` and `,`, the constrained output is
/// exactly those tokens; when they don't (a newline), it still conforms.
#[test]
fn greedy_conforming_output_is_unchanged() {
    for name in [QWEN3_TOKENIZER, TINYLLAMA_TOKENIZER] {
        let Some(tok) = tokenizer(name) else { continue };
        let index = TokenIndex::build(&tok);
        let schema = Arc::new(JsonSchema::compile(HANDOFF).unwrap());
        for value in [
            r##"{"agent":"planner","status":"needs_review","confidence":0.75,"steps":[{"tool":"search","args":{"q":"naïve café ☕ 😀"},"ok":true}],"notes":"tab\tquote\" done"}"##,
            r##"{"agent": "planner", "status": "done", "steps": [{"tool": "search", "ok": true}, {"tool": "search", "args": {}, "ok": false}]}"##,
        ] {
            assert!(schema.accepts(value), "{value}");
            let (ids, text, want) = greedy_through(&tok, &index, &schema, value, 1).unwrap();
            assert_eq!(
                ids, want,
                "{name}: masking changed a conforming greedy output"
            );
            assert_eq!(text, value);
        }

        // Pretty-printed: the newline isn't allowed, so the output leaves the greedy
        // path there and stays within the format.
        let pretty = "{\n  \"agent\": \"a\",\n  \"status\": \"done\",\n  \"steps\": []\n}";
        assert!(!schema.accepts(pretty));
        let text = match greedy_through(&tok, &index, &schema, pretty, 2) {
            Ok((_, t, _)) => t,
            Err(ReflexError::GrammarDeadEnd { text, .. }) => text,
            Err(e) => panic!("{e}"),
        };
        assert!(
            text.starts_with('{') && !text.contains('\n'),
            "{name}: {text}"
        );
    }
}

#[test]
fn running_out_of_tokens_is_a_dead_end_with_the_partial_text() {
    let Some(tok) = tokenizer(QWEN3_TOKENIZER) else {
        return;
    };
    let index = TokenIndex::build(&tok);
    let schema = Arc::new(JsonSchema::compile(HANDOFF).unwrap());
    let want = tok
        .encode(r##"{"agent":"planner","status":"done""##)
        .unwrap();
    let err = fake_generate(
        &tok,
        &index,
        &schema,
        want.len(),
        |i| want.get(i).copied(),
        None,
        3,
    )
    .unwrap_err();
    assert_eq!(err.kind(), "grammar_dead_end");
    let ReflexError::GrammarDeadEnd {
        text, token_ids, ..
    } = &err
    else {
        panic!()
    };
    assert_eq!(text, r##"{"agent":"planner","status":"done""##);
    assert_eq!(token_ids, &want);
    assert!(err.to_string().contains("max_tokens"), "{err}");

    // A top-level number is complete without EOS once max_tokens runs out.
    let num = Arc::new(JsonSchema::compile(r##"{"type":"integer"}"##).unwrap());
    let digits = tok.encode("12").unwrap();
    let (ids, text) = fake_generate(
        &tok,
        &index,
        &num,
        digits.len(),
        |i| digits.get(i).copied(),
        None,
        4,
    )
    .unwrap();
    assert_eq!((ids, text.as_str()), (digits, "12"));
}

/// Prints the per-token mask cost on the Qwen3 vocabulary (152K tokens): the first
/// mask in a string-body state walks nearly the whole trie, later ones hit the
/// per-state cache. `cargo test --release grammar::tests::mask_cost -- --nocapture`.
#[test]
fn mask_cost_on_qwen3_vocab() {
    let Some(tok) = tokenizer(QWEN3_TOKENIZER) else {
        return;
    };
    let index = TokenIndex::build(&tok);
    let schema = JsonSchema::compile(HANDOFF).unwrap();
    let (mut d, start) = Dfa::new(schema.grammar(false).unwrap());
    let in_string = run(&mut d, start, b"{\"agent\":\"").unwrap();
    let t = std::time::Instant::now();
    let n: u32 = d
        .mask(in_string, &index)
        .iter()
        .map(|w| w.count_ones())
        .sum();
    let first = t.elapsed().as_secs_f64() * 1000.0;
    let t = std::time::Instant::now();
    d.mask(in_string, &index);
    let cached = t.elapsed().as_secs_f64() * 1000.0;
    let t = std::time::Instant::now();
    let n0: u32 = d.mask(start, &index).iter().map(|w| w.count_ones()).sum();
    let structural = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "qwen3 mask: string body {n} allowed, first {first:.3} ms, cached {cached:.4} ms; value start {n0} allowed, {structural:.3} ms; index build {:.1} ms, {} nodes",
        index.build_ms,
        index.node_count()
    );
}
