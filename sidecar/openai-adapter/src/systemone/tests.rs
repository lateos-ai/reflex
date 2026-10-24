use super::*;

/// A real decision-model header from the repository's `test-data/` (metadata and
/// tensor table, no weights; see docs/DEVELOPMENT.md's test fixtures), or `None`.
fn header(name: &str) -> Option<GgufMeta> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data")
        .join(name);
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    Some(GgufMeta::open(&path).expect("open header"))
}

fn model(name: &str) -> Option<DecisionModel> {
    match from_meta(&header(name)?) {
        Support::Ready(m) => Some(*m),
        Support::Unsupported(why) => panic!("{name}: {why}"),
        Support::NotDecisionModel => panic!("{name}: not a decision model"),
    }
}

fn openjev() -> Option<DecisionModel> {
    model("openjev-q4_k_m-header.gguf")
}

fn lev() -> Option<DecisionModel> {
    model("lev-q8_0-header.gguf")
}

fn json(text: &str) -> Json {
    serde_json::from_str(text).expect("valid JSON")
}

/// llama.cpp's own test request (`tools/server/tests/unit/test_systemone.py`).
const TICKET: &str = r#"{
    "state": "I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"}
    }
}"#;

const STATE: &str = "I was charged twice for my order last week and nobody has replied.";
const OPENJEV_END: &str = "\nAnswer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const LEV_HEAD: &str = "<|im_start|>system\nYou are a System One decision model. You read the Evidence and answer each Criterion by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>\n<|im_start|>user\n# Evidence\n";
const LEV_END: &str = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
}

// --- JSON ---------------------------------------------------------------------

#[test]
fn json_keeps_key_order_and_number_kinds() {
    let v = json(r#"{"zeta": 1, "alpha": 1.0, "mid": [true, null, "x"], "zeta": 2}"#);
    assert_eq!(
        v,
        Json::Obj(vec![
            ("zeta".into(), Json::Int(2)),
            ("alpha".into(), Json::Float(1.0)),
            (
                "mid".into(),
                Json::Arr(vec![Json::Bool(true), Json::Null, Json::Str("x".into())])
            ),
        ])
    );
}

#[test]
fn json_sorted_sorts_every_object_by_bytes() {
    let v = json(r#"{"b": {"z": 1, "a": 2}, "B": [{"y": 1, "x": 2}], "a": 3}"#);
    assert_eq!(
        to_json_text(&v.sorted()),
        r#"{"B": [{"x": 2, "y": 1}], "a": 3, "b": {"a": 2, "z": 1}}"#
    );
}

#[test]
fn float_format_is_c_percent_g() {
    for (x, s) in [
        (19.99, "19.99"),
        (1.0 / 3.0, "0.333333"),
        (1e-7, "1e-07"),
        (1e20, "1e+20"),
        (100000.0, "100000"),
        (1e6, "1e+06"),
        (123456.7, "123457"),
        (999999.5, "1e+06"),
        (0.0001, "0.0001"),
        (0.00001234, "1.234e-05"),
        (-2.5, "-2.5"),
        (1.0, "1"),
        (0.0, "0"),
        (-0.0, "-0"),
        (2.5e-300, "2.5e-300"),
    ] {
        assert_eq!(format_float_g6(x), s, "{x}");
    }
}

#[test]
fn tojson_matches_llama_cpp() {
    let v = json(
        r#"{"ticket": "Hé said \"hi\"\n\t<b>&'", "n": 3, "price": 19.99, "ok": false, "tags": [], "o": {}, "ctl": "\u0001"}"#,
    );
    assert_eq!(
        to_json_text(&v),
        "{\"ticket\": \"Hé said \\\"hi\\\"\\n\\t<b>&'\", \"n\": 3, \"price\": 19.99, \"ok\": false, \"tags\": [], \"o\": {}, \"ctl\": \"\\u0001\"}"
    );
}

// --- Metadata -----------------------------------------------------------------

#[test]
fn openjev_header_metadata() {
    let Some(m) = openjev() else { return };
    assert_eq!(m.kind, DecisionType::OpenJev);
    assert_eq!(m.labels.len(), 52);
    assert_eq!(m.labels[0], "A");
    assert_eq!(m.labels[51], "z");
    assert!(m.label_texts.is_empty());
    assert!(m.noul_true_first);
    // 0.85 and 0.85 * 1.829074, through llama.cpp's text round trip.
    assert_eq!(m.temperatures["choice"], 0.85);
    assert_eq!(m.temperatures["score"], 0.85);
    assert_eq!(m.temperatures["noul"], 1.554713);
}

#[test]
fn lev_header_metadata() {
    let Some(m) = lev() else { return };
    assert_eq!(m.kind, DecisionType::Lev);
    assert_eq!(m.labels.len(), 255);
    assert_eq!(m.labels[..3], ["A", "B", "C"]);
    assert_eq!(m.labels[26], "AA");
    assert_eq!(m.labels[254], "JT");
    assert_eq!(m.label_texts, m.labels);
    assert!(!m.noul_true_first);
    assert_eq!(m.temperatures.len(), 6);
    assert_eq!(m.temperatures["choice.small"], 1.466032);
    assert_eq!(m.temperatures["noul"], 1.282162);
}

#[test]
fn non_decision_and_later_phase_models_are_refused() {
    let mut meta = GgufMeta {
        metadata: HashMap::new(),
    };
    meta.metadata.insert(
        "general.architecture".into(),
        crate::gguf_meta::GgufMetaValue::String("qwen35".into()),
    );
    assert!(matches!(from_meta(&meta), Support::NotDecisionModel));
    for ty in ["kev", "laya", "mystery"] {
        meta.metadata.insert(
            "qwen35.decision.type".into(),
            crate::gguf_meta::GgufMetaValue::String(ty.into()),
        );
        let Support::Unsupported(why) = from_meta(&meta) else {
            panic!("{ty} accepted")
        };
        assert!(why.contains(ty), "{why}");
    }
    meta.metadata.insert(
        "qwen35.decision.type".into(),
        crate::gguf_meta::GgufMetaValue::String("openjev".into()),
    );
    let Support::Unsupported(why) = from_meta(&meta) else {
        panic!("accepted without a template")
    };
    assert!(why.contains("systemone"), "{why}");
}

#[test]
fn temperature_rounds_like_std_to_string() {
    assert_eq!(llama_cpp_temperature(1.4606453_f32 as f64), 1.460645);
    assert_eq!(llama_cpp_temperature(0.85_f32 as f64), 0.85);
}

#[test]
fn temperature_buckets() {
    let Some(m) = lev() else { return };
    let q = |kind, n: usize| Question {
        id: "q".into(),
        kind,
        instructions: Json::Str("x".into()),
        options: (0..n)
            .map(|i| DecisionOption {
                key: i.to_string(),
                description: Json::Null,
            })
            .collect(),
    };
    assert_eq!(m.temperature(&q(QuestionType::Choice, 8)), 1.466032);
    assert_eq!(m.temperature(&q(QuestionType::Choice, 9)), 1.322849);
    assert_eq!(m.temperature(&q(QuestionType::Choice, 26)), 1.322849);
    assert_eq!(m.temperature(&q(QuestionType::Choice, 27)), 1.498329);
    assert_eq!(m.temperature(&q(QuestionType::Score, 4)), 1.868717);
    assert_eq!(m.temperature(&q(QuestionType::Noul, 2)), 1.282162);
    let Some(o) = openjev() else { return };
    assert_eq!(o.temperature(&q(QuestionType::Choice, 12)), 0.85);
}

// --- Requests -----------------------------------------------------------------

#[test]
fn parse_keeps_request_order() {
    let Some(m) = openjev() else { return };
    let p = m.parse(&json(TICKET)).expect("valid");
    let ids: Vec<_> = p.questions.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(ids, ["route", "urgency", "angry"]);
    let keys: Vec<_> = p.questions[0]
        .options
        .iter()
        .map(|o| o.key.as_str())
        .collect();
    assert_eq!(keys, ["billing", "shipping", "technical"]);
    let keys: Vec<_> = p.questions[1]
        .options
        .iter()
        .map(|o| o.key.as_str())
        .collect();
    assert_eq!(keys, ["0", "1", "2", "3"]);
    // OpenJev lists true first; lev false first.
    let keys: Vec<_> = p.questions[2]
        .options
        .iter()
        .map(|o| o.key.as_str())
        .collect();
    assert_eq!(keys, ["true", "false"]);
    let Some(l) = lev() else { return };
    let p = l.parse(&json(TICKET)).expect("valid");
    let keys: Vec<_> = p.questions[2]
        .options
        .iter()
        .map(|o| o.key.as_str())
        .collect();
    assert_eq!(keys, ["false", "true"]);
}

/// llama.cpp's `test_systemone_invalid_request` cases, plus a few more.
#[test]
fn parse_rejects_what_llama_cpp_rejects() {
    let Some(m) = openjev() else { return };
    let q = |body: &str| format!(r#"{{"state": "s", "questions": {{"q": {body}}}}}"#);
    let too_many: String = (0..53)
        .map(|i| format!(r#""o{i}": null"#))
        .collect::<Vec<_>>()
        .join(", ");
    for (body, message) in [
        (r#"{"questions": {"q": {"type": "noul", "instructions": "x"}}}"#.to_string(), "\"state\" must be provided"),
        (r#"{"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}}"#.to_string(), "\"state\" must be provided"),
        (r#"{"state": "s"}"#.to_string(), "\"questions\" must be a non-empty object"),
        (r#"{"state": "s", "questions": {}}"#.to_string(), "\"questions\" must be a non-empty object"),
        (r#"{"state": "s", "questions": []}"#.to_string(), "\"questions\" must be a non-empty object"),
        (q(r#""x""#), "questions.q: must be an object"),
        (q(r#"{"type": "unknown", "instructions": "x"}"#), "questions.q: \"type\" must be one of: choice, score, noul"),
        (q(r#"{"type": 3, "instructions": "x"}"#), "\"type\" must be one of"),
        (q(r#"{"type": "noul"}"#), "questions.q: \"instructions\" must be provided"),
        (q(r#"{"type": "choice", "instructions": "x"}"#), "\"criteria\" must be a non-empty object"),
        (q(r#"{"type": "choice", "instructions": "x", "criteria": {}}"#), "\"criteria\" must be a non-empty object"),
        (q(r#"{"type": "score", "instructions": "x", "criteria": ["only one"]}"#), "\"criteria\" must be an array of 2 to 10 levels"),
        (q(r#"{"type": "score", "instructions": "x", "criteria": ["1","2","3","4","5","6","7","8","9","10","11"]}"#), "2 to 10 levels"),
        (q(r#"{"type": "noul", "instructions": "x", "criteria": []}"#), "\"criteria\" must be an object"),
        (q(&format!(r#"{{"type": "choice", "instructions": "x", "criteria": {{{too_many}}}}}"#)), "too many options (53), this model supports at most 52"),
        (r#"{"state": "s", "images": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}}"#.to_string(), "\"images\" must be an array"),
        (r#"{"state": "s", "images": ["https://example.com/a.png"], "questions": {"q": {"type": "noul", "instructions": "x"}}}"#.to_string(), "images must be data URLs"),
    ] {
        match m.parse(&json(&body)) {
            Err(RequestError::Invalid(got)) => assert!(got.contains(message), "{body}: {got}"),
            other => panic!("{body}: {other:?}"),
        }
    }
    let nine = [r#""data:image/png;base64,AAAA""#; 9].join(", ");
    let body = format!(
        r#"{{"state": "s", "images": [{nine}], "questions": {{"q": {{"type": "noul", "instructions": "x"}}}}}}"#
    );
    assert!(
        matches!(m.parse(&json(&body)), Err(RequestError::Invalid(e)) if e.contains("too many images"))
    );
}

#[test]
fn images_are_not_supported_yet() {
    let Some(m) = openjev() else { return };
    let noul = r#""questions": {"q": {"type": "noul", "instructions": "x"}}"#;
    for body in [
        format!(r#"{{"state": "s", "images": ["data:image/png;base64,AAAA"], {noul}}}"#),
        format!(
            r#"{{"state": [{{"role": "user", "content": [{{"type": "image_url", "image_url": {{"url": "data:image/png;base64,AAAA"}}}}]}}], {noul}}}"#
        ),
        format!(
            r#"{{"state": {{"messages": [{{"role": "user", "content": [{{"type": "image_url", "image_url": "data:image/png;base64,AAAA"}}]}}]}}, {noul}}}"#
        ),
    ] {
        assert_eq!(
            m.parse(&json(&body)).unwrap_err(),
            RequestError::ImagesUnsupported,
            "{body}"
        );
    }
    // Chat messages without images are just a state.
    let body = format!(
        r#"{{"state": [{{"role": "user", "content": [{{"type": "text", "text": "hi"}}]}}], "images": [], {noul}}}"#
    );
    assert!(m.parse(&json(&body)).is_ok());
}

// --- Rendering ----------------------------------------------------------------

#[test]
fn openjev_renders_the_ticket() {
    let Some(m) = openjev() else { return };
    let p = m.parse(&json(TICKET)).unwrap();
    assert!(p.questions.iter().all(|q| m.n_variants(q) == 1));
    let head = format!("<|im_start|>user\nState:\n{STATE}\n\nQuestion: ");
    assert_eq!(
        m.render(&p.state, &p.questions[0], 0).unwrap(),
        format!("{head}Which team should handle this?\nOptions:\n[A] billing: payments and refunds\n[B] shipping: \n[C] technical: \n{OPENJEV_END}")
    );
    assert_eq!(
        m.render(&p.state, &p.questions[1], 0).unwrap(),
        format!("{head}How urgent is this? Rate along the ordered levels below (lowest first).\nOptions:\n[A] 0: can wait\n[B] 1: this week\n[C] 2: today\n[D] 3: right now\n{OPENJEV_END}")
    );
    assert_eq!(
        m.render(&p.state, &p.questions[2], 0).unwrap(),
        format!("{head}Is the customer angry?\nOptions:\n[A] yes: The statement is true.\n[B] no: The statement is false.\n{OPENJEV_END}")
    );
    assert_eq!(m.labels_for(&p.questions[1]), ["A", "B", "C", "D"]);
}

#[test]
fn openjev_prints_non_string_values_as_llama_cpp_json() {
    let Some(m) = openjev() else { return };
    let body = r#"{"state": {"ticket": "Hé <b>", "plan": "pro", "order": {"id": 4471, "total": 19.99, "items": ["case", "charger"]}},
                   "questions": {"refund": {"type": "noul", "instructions": {"ask": "refund?"},
                                 "criteria": {"true": "money back is asked", "false": {"k": 1.0}}}}}"#;
    let p = m.parse(&json(body)).unwrap();
    assert_eq!(
        m.render(&p.state, &p.questions[0], 0).unwrap(),
        format!("<|im_start|>user\nState:\n{{\"ticket\": \"Hé <b>\", \"plan\": \"pro\", \"order\": {{\"id\": 4471, \"total\": 19.99, \"items\": [\"case\", \"charger\"]}}}}\n\nQuestion: {{\"ask\": \"refund?\"}}\nOptions:\n[A] yes: money back is asked\n[B] no: {{\"k\": 1}}\n{OPENJEV_END}")
    );
}

#[test]
fn lev_renders_both_choice_variants_with_labels_in_place() {
    let Some(m) = lev() else { return };
    let p = m.parse(&json(TICKET)).unwrap();
    let route = &p.questions[0];
    assert_eq!(m.n_variants(route), 2);
    let crit =
        format!("{LEV_HEAD}{STATE}\n\n# Criterion\nWhich team should handle this?\n\n# Options\n");
    let tail = format!("\nRespond with only the letter of the best option.{LEV_END}");
    assert_eq!(
        m.render(&p.state, route, 0).unwrap(),
        format!("{crit}A. billing: payments and refunds\nB. shipping\nC. technical\n{tail}")
    );
    assert_eq!(
        m.render(&p.state, route, 1).unwrap(),
        format!("{crit}A. technical\nB. shipping\nC. billing: payments and refunds\n{tail}")
    );
    let plan = m.plan(&p).unwrap();
    assert_eq!(plan.prompts.len(), 4, "2 route variants + urgency + angry");
    assert_eq!(plan.labels[0], ["A", "B", "C"]);
    assert_eq!(
        plan.labels[3],
        ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    );
}

#[test]
fn lev_renders_score_and_noul() {
    let Some(m) = lev() else { return };
    let body = r#"{"state": "s", "questions": {
        "u": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", null]},
        "a": {"type": "noul", "instructions": "", "criteria": {"true": "yes it is", "false": "no it is not"}},
        "b": {"type": "noul", "instructions": "Angry?"}}}"#;
    let p = m.parse(&json(body)).unwrap();
    let head = format!("{LEV_HEAD}s\n\n# Criterion\n");
    assert_eq!(m.n_variants(&p.questions[0]), 1);
    assert_eq!(
        m.render(&p.state, &p.questions[0], 0).unwrap(),
        format!("{head}How urgent?\n\n# Options\nA. (level 0 of 1) can wait\nB. (level 1 of 1) null\n\nRespond with only the letter of the level that best matches.{LEV_END}")
    );
    // Empty instructions fall back to the question id.
    assert_eq!(
        m.render(&p.state, &p.questions[1], 0).unwrap(),
        format!("{head}a\n\n# Scale\n0 = certainly no ... 8 = certainly yes\nyes: yes it is\nno: no it is not\n\nRespond with only a digit from 0 to 8.{LEV_END}")
    );
    assert_eq!(
        m.render(&p.state, &p.questions[2], 0).unwrap(),
        format!("{head}Angry?\n\n# Scale\n0 = certainly no ... 8 = certainly yes\n\nRespond with only a digit from 0 to 8.{LEV_END}")
    );
}

#[test]
fn lev_sorts_the_keys_of_object_values() {
    let Some(m) = lev() else { return };
    let body = r#"{"state": {"zeta": 1, "alpha": {"y": 2, "x": 3}},
                   "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"b": {"z": 1, "a": 2}, "a": null}}}}"#;
    let p = m.parse(&json(body)).unwrap();
    let out = m.render(&p.state, &p.questions[0], 0).unwrap();
    assert!(
        out.contains("# Evidence\n{\"alpha\": {\"x\": 3, \"y\": 2}, \"zeta\": 1}\n"),
        "{out}"
    );
    // Options keep the request's order; only object values are sorted.
    assert!(out.contains("A. b: {\"a\": 2, \"z\": 1}\nB. a\n"), "{out}");
}

// --- Answers ------------------------------------------------------------------

fn question(kind: QuestionType, keys: &[&str]) -> Question {
    Question {
        id: "q".into(),
        kind,
        instructions: Json::Str("x".into()),
        options: keys
            .iter()
            .map(|k| DecisionOption {
                key: k.to_string(),
                description: Json::Null,
            })
            .collect(),
    }
}

fn softmax(s: &[f64], t: f64) -> Vec<f64> {
    let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| ((x - max) / t).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}

#[test]
fn openjev_answers() {
    let Some(m) = openjev() else { return };
    let t = 0.85f32 as f64;

    let q = question(QuestionType::Choice, &["billing", "shipping", "technical"]);
    let a = m.answer(&q, &[vec![2.0, 1.0, 0.5]]).unwrap();
    let p = softmax(&[2.0, 1.0, 0.5], t);
    let Answer::Choice {
        choice,
        probabilities,
        confidence,
        ty,
    } = a
    else {
        panic!()
    };
    assert_eq!((ty, choice.as_str()), ("choice", "billing"));
    for (i, (k, v)) in probabilities.0.iter().enumerate() {
        assert_eq!(k, &q.options[i].key);
        assert_close(*v, p[i]);
    }
    assert_close(confidence, (p[0] - 1.0 / 3.0) / (1.0 - 1.0 / 3.0));

    let q = question(QuestionType::Score, &["0", "1", "2", "3"]);
    let s = [0.0, 1.0, 3.0, 2.0];
    let p = softmax(&s, t);
    let Answer::Score {
        score,
        confidence,
        legend,
        ..
    } = m
        .answer(&q, &[s.iter().map(|&x| x as f32).collect()])
        .unwrap()
    else {
        panic!()
    };
    assert_close(score, (0..4).map(|i| i as f64 * p[i]).sum());
    let dist: f64 = (0..4).map(|i| p[i] * (i as f64 - 2.0).abs()).sum();
    let du: f64 = (0..4).map(|i| (i as f64 - 1.5).abs() / 4.0).sum();
    assert_close(confidence, (1.0 - dist / du).max(0.0));
    assert_eq!(legend.0.len(), 4);

    // OpenJev's noul options are [true, false].
    let q = question(QuestionType::Noul, &["true", "false"]);
    let Answer::Noul { noul, .. } = m.answer(&q, &[vec![1.0, 0.0]]).unwrap() else {
        panic!()
    };
    assert_close(noul, softmax(&[1.0, 0.0], 1.554713f32 as f64)[0]);
}

#[test]
fn lev_answers_average_the_reversed_variant() {
    let Some(m) = lev() else { return };
    let q = question(QuestionType::Choice, &["a", "b", "c"]);
    let t = 1.466032f32 as f64;
    // Variant 1 lists [c, b, a].
    let a = m
        .answer(&q, &[vec![3.0, 1.0, 0.0], vec![0.5, 0.0, 2.0]])
        .unwrap();
    let p0 = softmax(&[3.0, 1.0, 0.0], t);
    let p1 = softmax(&[0.5, 0.0, 2.0], t);
    let Answer::Choice {
        probabilities,
        choice,
        ..
    } = a
    else {
        panic!()
    };
    assert_eq!(choice, "a");
    for i in 0..3 {
        assert_close(probabilities.0[i].1, (p0[i] + p1[2 - i]) / 2.0);
    }

    // noul: the expected rating over 9 labels, divided by 8.
    let q = question(QuestionType::Noul, &["false", "true"]);
    let s: Vec<f32> = (0..9).map(|i| i as f32 * 0.25).collect();
    let p = softmax(
        &s.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        1.282162f32 as f64,
    );
    let Answer::Noul { noul, .. } = m.answer(&q, &[s]).unwrap() else {
        panic!()
    };
    assert_close(noul, (0..9).map(|i| p[i] * i as f64 / 8.0).sum());

    assert!(
        m.answer(&q, &[vec![0.0; 2]]).is_err(),
        "noul reads 9 outputs"
    );
    let choice = question(QuestionType::Choice, &["a", "b"]);
    assert!(
        m.answer(&choice, &[vec![0.0; 2]]).is_err(),
        "choice has 2 variants"
    );
}

#[test]
fn confidences_at_the_extremes() {
    assert_eq!(confidence_choice(&[1.0]), 1.0);
    assert_eq!(confidence_choice(&[0.5, 0.5]), 0.0);
    assert_eq!(confidence_choice(&[1.0, 0.0, 0.0]), 1.0);
    assert_eq!(confidence_score(&[0.0, 1.0, 0.0]), 1.0);
    assert!(confidence_score(&[0.25; 4]) < 1e-12);
    // Ties pick the first maximum, as std::max_element.
    assert_eq!(argmax(&[0.2, 0.4, 0.4]), 1);
}

#[test]
fn build_response_maps_results_back_in_order() {
    let Some(m) = lev() else { return };
    let p = m.parse(&json(TICKET)).unwrap();
    let plan = m.plan(&p).unwrap();
    let ipc = DecisionModel::ipc_request(&plan);
    assert_eq!(ipc["label_prompts"].as_array().unwrap().len(), 4);
    assert!(ipc.get("prompt").is_none());
    let engine = serde_json::json!({
        "event": "final", "ok": true, "shared_tokens": 60,
        "labels": [
            {"tokens": 100, "token_ids": [32, 33, 34], "logits": [3.0, 1.0, 0.0]},
            {"tokens": 100, "token_ids": [32, 33, 34], "logits": [0.0, 1.0, 3.0]},
            {"tokens": 110, "token_ids": [32, 33, 34, 35], "logits": [0.0, 1.0, 2.0, 0.5]},
            {"tokens": 90, "token_ids": [], "logits": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
        ],
    });
    let r = m.build_response(&p, &engine, "lev".into()).unwrap();
    assert_eq!(r.usage.input_tokens, 400);
    let text = serde_json::to_string(&r).unwrap();
    let route = text.find("\"route\"").unwrap();
    let urgency = text.find("\"urgency\"").unwrap();
    let angry = text.find("\"angry\"").unwrap();
    assert!(route < urgency && urgency < angry, "{text}");
    assert!(text.starts_with(r#"{"model":"lev","answers":{"route":{"type":"choice","choice":"billing","probabilities":{"billing":"#), "{text}");
    assert!(
        text.contains(r#""legend":{"0":"can wait","1":"this week","2":"today","3":"right now"}"#),
        "{text}"
    );
    assert!(
        text.ends_with(r#""usage":{"input_tokens":400,"output_tokens":0}}"#),
        "{text}"
    );

    let mut short = engine.clone();
    short["labels"].as_array_mut().unwrap().pop();
    assert!(m.build_response(&p, &short, "lev".into()).is_err());
    let mut long = engine;
    long["labels"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"tokens": 1, "logits": [0.0]}));
    assert!(m.build_response(&p, &long, "lev".into()).is_err());
}

// --- llama.cpp goldens ----------------------------------------------------------

/// Renders a request set and checks it against goldens written by llama.cpp's own
/// `server_decision_context` (a small harness on `a4cb4c6`'s server-decision.cpp):
/// the same 400 messages, the same image refusals, and byte for byte the same
/// prompts, labels and temperatures. Runs only when `SYSTEMONE_GOLDEN_GGUF` (the
/// decision GGUF), `SYSTEMONE_GOLDEN_REQUESTS` (`[{"name", "body_text"}]`) and
/// `SYSTEMONE_GOLDENS` (the harness's JSON lines) are set.
#[test]
fn render_matches_llama_cpp_goldens_if_configured() {
    let (Ok(gguf), Ok(requests), Ok(goldens)) = (
        std::env::var("SYSTEMONE_GOLDEN_GGUF"),
        std::env::var("SYSTEMONE_GOLDEN_REQUESTS"),
        std::env::var("SYSTEMONE_GOLDENS"),
    ) else {
        eprintln!("skipping: SYSTEMONE_GOLDEN_GGUF, _REQUESTS or SYSTEMONE_GOLDENS not set");
        return;
    };
    let Support::Ready(m) = load(&gguf) else {
        panic!("{gguf}: not a supported decision model");
    };
    let requests: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&requests).unwrap()).unwrap();
    let bodies: HashMap<String, String> = requests
        .iter()
        .map(|r| {
            (
                r["name"].as_str().unwrap().to_string(),
                r["body_text"].as_str().unwrap().to_string(),
            )
        })
        .collect();

    let mut failures = Vec::new();
    let mut n_prompts = 0;
    let goldens = std::fs::read_to_string(&goldens).unwrap();
    for line in goldens.lines().filter(|l| !l.trim().is_empty()) {
        let g: serde_json::Value = serde_json::from_str(line).unwrap();
        let name = g["name"].as_str().unwrap();
        let mut fail = |what: String| failures.push(format!("{name}: {what}"));
        let parsed = match serde_json::from_str::<Json>(&bodies[name]) {
            Ok(body) => m.parse(&body),
            Err(e) => {
                if g.get("exception").is_none() {
                    fail(format!(
                        "body does not parse here ({e}) but does in llama.cpp"
                    ));
                }
                continue;
            }
        };
        if g.get("exception").is_some() {
            fail(format!(
                "llama.cpp threw {}, here {parsed:?}",
                g["exception"]
            ));
            continue;
        }
        if let Some(want) = g.get("error").and_then(|e| e.as_str()) {
            match parsed {
                Err(RequestError::Invalid(got)) if got == want => {}
                other => fail(format!("want 400 {want:?}, got {other:?}")),
            }
            continue;
        }
        if g.get("n_images").is_some() {
            if !matches!(parsed, Err(RequestError::ImagesUnsupported)) {
                fail(format!("want images 501, got {parsed:?}"));
            }
            continue;
        }
        let parsed = match parsed {
            Ok(p) => p,
            Err(e) => {
                fail(format!("rejected here: {e:?}"));
                continue;
            }
        };
        let plan = m.plan(&parsed).unwrap();
        let want = g["prompts"].as_array().unwrap();
        if plan.prompts.len() != want.len() {
            fail(format!(
                "{} prompts, llama.cpp {}",
                plan.prompts.len(),
                want.len()
            ));
            continue;
        }
        let questions = parsed
            .questions
            .iter()
            .flat_map(|q| (0..m.n_variants(q)).map(move |_| q));
        for (i, ((prompt, labels), q)) in plan
            .prompts
            .iter()
            .zip(&plan.labels)
            .zip(questions)
            .enumerate()
        {
            n_prompts += 1;
            let w = &want[i];
            if w["q"].as_str() != Some(q.id.as_str()) {
                fail(format!(
                    "prompt {i}: question {}, llama.cpp {}",
                    q.id, w["q"]
                ));
            }
            if w["prompt"].as_str() != Some(prompt.as_str()) {
                fail(format!(
                    "prompt {i} differs:\n--- here\n{prompt}\n--- llama.cpp\n{}",
                    w["prompt"].as_str().unwrap_or("")
                ));
            }
            let pieces: Vec<&str> = w["label_pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap())
                .collect();
            if *labels != pieces {
                fail(format!(
                    "prompt {i}: labels {labels:?}, llama.cpp {pieces:?}"
                ));
            }
            let t = w["temperature"].as_f64().unwrap() as f32;
            if m.temperature(q) != t {
                fail(format!(
                    "prompt {i}: temperature {} vs {t}",
                    m.temperature(q)
                ));
            }
        }
    }
    eprintln!("{n_prompts} prompts compared");
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
