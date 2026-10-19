//! Local, non-network IPC protocol shared by the `reflex stdio`/`reflex uds`
//! subcommands (`src/bin/reflex/stdio.rs`/`uds.rs`) -- see README.md's
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
//!
//! **Timings** (opt-in, `"timings": true`): the final line gains an [`IpcTimings`]
//! object, so a caller chaining requests (an agent handing off to another agent)
//! can split its measured round trip into engine time and transport overhead.
//! Omitted (the default), the response is byte-identical to before.
//!
//! **Prefix reuse**: `export_kv` names a file to write the
//! model's state after this request's prompt to, and `import_kv` names one to resume
//! from, in the same cache formats as `reflex generate --export-kv/--import-kv`
//! (`crate::kv_io`). A request that sets `import_kv` treats `prompt` as the
//! continuation after the file's positions. An `export_kv` request must set
//! `max_tokens` to 1, since the state after a later generated token is no longer
//! the prompt's (see `Model::generate_capture_kv`). `export_kv_dtype: "f16"`
//! writes the file in f16, half the size, at the cost of an exact resume; the
//! default is the working cache's type (`"f32"` unless the engine runs with
//! `REFLEX_KV_CACHE=f16`, whose cache is already f16), and `import_kv` reads
//! either. The caller owns
//! the files: the engine keeps nothing between requests.
//!
//! **Pinned prefix**: a request with `pin_prefix` keeps the
//! model's state after its prompt in VRAM, in the session's one [`PrefixSlot`],
//! replacing whatever was there; `max_tokens` must be 1, as for `export_kv`. A
//! request with `use_pinned` resumes from that slot, treating `prompt` as the
//! continuation. Each use copies the slot device-to-device into a fresh cache, so
//! the pinned state never changes. Setting both resumes from the slot and pins the
//! longer state in its place. Neither combines with `import_kv`/`export_kv`. There
//! is exactly one slot, with no names and no eviction: anything more belongs in the
//! orchestrator (docs/DEVELOPMENT.md).
//!
//! **System-1 over a resumed prefix**: a `candidates` request may set `use_pinned`
//! or `import_kv` (not both), and then scores the candidates after the cached
//! prefix plus `prompt`, which must add at least one token. Its scores and
//! probabilities are those of one request over the joined text. `pin_prefix` and
//! `export_kv` stay `generate`-only, since scoring writes past the prompt. On the
//! Qwen3.5 hybrid every candidate must be one token, as without a prefix; Gemma 4
//! and granitehybrid reject a prefix with `unsupported_architecture`. Every
//! `candidates` response carries `chosen`, the index of the most probable
//! candidate.
//!
//! **Slot facts**: with `"timings": true`, every response's timings also report
//! the slot as it is after the request: `pinned_seq_len` positions and
//! `pinned_bytes` of VRAM (both 0 when nothing is pinned).
//!
//! **JSON Schema** (`generate` path only): `json_schema` constrains the output to
//! one JSON value that validates against the schema, compact but for an optional
//! space after each `:` and `,` (`crate::grammar` has
//! the supported subset). `text` is then the value alone, and `token_ids` end with
//! EOS. If `max_tokens` runs out first, the request fails with `grammar_dead_end`,
//! and the error line carries the partial `text` and `token_ids`. It combines with
//! `import_kv` and `use_pinned` (only the generated text is constrained, never the
//! prompt), not with `pin_prefix`/`export_kv` or `candidates`.

use crate::error::ReflexError;
use crate::model::{Model, PinnedPrefix, Resume, System1Candidate};
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
    /// Opt-in: attach an [`IpcTimings`] to the final response line.
    #[serde(default)]
    pub timings: bool,
    /// Resume from this KV cache file, on either path. See this module's doc
    /// comment's "Prefix reuse" and "System-1 over a resumed prefix" sections.
    #[serde(default)]
    pub import_kv: Option<String>,
    /// `generate` path only: write the state after this request's prompt to this
    /// file. Requires `max_tokens: 1`.
    #[serde(default)]
    pub export_kv: Option<String>,
    /// With `export_kv`: the file's element type, `"f32"` (the default) or
    /// `"f16"` (half the size; resuming from it is no longer exact). See
    /// `crate::kv_io`'s f16 files.
    #[serde(default)]
    pub export_kv_dtype: Option<String>,
    /// `generate` path only: keep the state after this request's prompt in the
    /// session's [`PrefixSlot`], replacing its contents. Requires `max_tokens: 1`.
    #[serde(default)]
    pub pin_prefix: bool,
    /// Resume from the session's [`PrefixSlot`], on either path.
    #[serde(default)]
    pub use_pinned: bool,
    /// `generate` path only: a JSON Schema the output must conform to. Kept as raw
    /// text so `crate::grammar` sees the properties in the order they're written.
    #[serde(default)]
    pub json_schema: Option<Box<serde_json::value::RawValue>>,
    /// Milliseconds spent parsing this request's JSON line, filled in by
    /// [`run_request_loop`] (not a wire field) so [`IpcTimings::parse_ms`] can
    /// report it.
    #[serde(skip)]
    parse_ms: f64,
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
                json_schema: None,
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
    /// Present exactly when `error` is: the failure's stable category, one of
    /// `crate::error::ReflexErrorCode::as_str`'s names (e.g. `"context_overflow"`,
    /// `"out_of_memory"`, `"invalid_input"`). Branch on this, not on `error`'s text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<&'static str>,
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
    /// `Model::system1_evaluate` path only: the index into `candidates` of the
    /// highest `probability` (the first one on a tie).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen: Option<usize>,
    /// Present only when the request set `"timings": true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<IpcTimings>,
}

/// Per-request engine-side timing breakdown, in milliseconds (`"timings": true`).
/// A caller's measured round trip minus `parse_ms + model_ms` is what the
/// transport (pipe or socket, plus serializing the response) costs it.
#[derive(Debug, Serialize)]
pub struct IpcTimings {
    /// Parsing the request line's JSON.
    pub parse_ms: f64,
    /// From the start of the model call to its first generated token: tokenize,
    /// prefill, LM head and the first sampling step, plus reading (and
    /// uploading) the `import_kv` file or copying the pinned prefix, if any.
    /// `generate` path only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<f64>,
    /// The whole model call (`Model::generate` or `Model::system1_evaluate`). On
    /// a streaming request this includes writing and flushing each token line,
    /// since those happen inside the decode loop. Includes `kv_io_ms`.
    pub model_ms: f64,
    /// Reading the `import_kv` file plus writing the `export_kv` file, when the
    /// request named either (on a `candidates` request too). The file's payload
    /// is read while it is uploaded into the request's cache, so the read
    /// includes that upload (`crate::kv_io`). On a `use_pinned`
    /// request, generating or scoring, the device-to-device copy
    /// of the pinned prefix into the request's own cache instead. Pinning itself
    /// copies nothing, so a `pin_prefix` request alone reports none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_io_ms: Option<f64>,
    /// `json_schema` requests only: computing and applying the token masks, and
    /// following each chosen token through the grammar, summed over the request.
    /// Included in `first_token_ms` (the first mask) and `model_ms`. Building the
    /// vocabulary index on the session's first constrained request is not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_ms_total: Option<f64>,
    /// Positions held by the session's [`PrefixSlot`] after this request (0 if
    /// nothing is pinned). With `pinned_bytes`, what an orchestrator needs to
    /// choose between keeping the pin, recomputing and a cache file.
    pub pinned_seq_len: usize,
    /// VRAM held by the session's [`PrefixSlot`] after this request, in bytes.
    pub pinned_bytes: usize,
}

fn elapsed_ms(since: std::time::Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// Builds a request's [`IpcTimings`] if it asked for one. `slot` is read as it
/// is after the request.
fn timings_if(
    requested: bool,
    parse_ms: f64,
    model_start: std::time::Instant,
    first_token_ms: Option<f64>,
    kv_io_ms: Option<f64>,
    mask_ms_total: Option<f64>,
    slot: &PrefixSlot,
) -> Option<IpcTimings> {
    requested.then(|| IpcTimings {
        parse_ms,
        first_token_ms,
        model_ms: elapsed_ms(model_start),
        kv_io_ms,
        mask_ms_total,
        pinned_seq_len: slot.seq_len().unwrap_or(0),
        pinned_bytes: slot.device_bytes(),
    })
}

/// The one pinned prefix a `reflex stdio`/`uds` session may hold (`pin_prefix`/
/// `use_pinned`). [`run_stdio_loop`] keeps one for the process; `reflex uds` keeps
/// one across its connections. It holds VRAM until it is replaced or the process
/// exits: a pinned prefix costs as much device memory as its KV cache, and each
/// request that uses it allocates a second copy of that for its own cache.
#[derive(Default)]
pub struct PrefixSlot {
    pinned: Option<PinnedPrefix>,
}

impl PrefixSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions held, or `None` if nothing is pinned.
    pub fn seq_len(&self) -> Option<usize> {
        self.pinned.as_ref().map(PinnedPrefix::seq_len)
    }

    /// VRAM held by the pinned prefix, in bytes (0 if empty).
    pub fn device_bytes(&self) -> usize {
        self.pinned.as_ref().map_or(0, PinnedPrefix::device_bytes)
    }
}

/// Runs a `generate`-path request, honoring `import_kv`/`export_kv` and
/// `pin_prefix`/`use_pinned`. Shared by [`handle_request`] and
/// [`handle_request_streaming`]. Records the time spent on cache files, or on
/// copying the pinned prefix, in `kv_io_ms`. A failed request leaves `slot`
/// unchanged. A `json_schema` request records its masking time in `mask_ms`.
fn run_generate(
    model: &Model,
    slot: &mut PrefixSlot,
    req: &IpcRequest,
    kv_io_ms: &mut Option<f64>,
    mask_ms: &mut Option<f64>,
    on_first_token: impl FnMut(&[f32]),
    on_token: impl FnMut(u32, &str),
) -> Result<(Vec<u32>, String), ReflexError> {
    let mut sampling = req.sampling_params();
    if let Some(raw) = &req.json_schema {
        if req.pin_prefix || req.export_kv.is_some() {
            let field = if req.pin_prefix {
                "pin_prefix"
            } else {
                "export_kv"
            };
            return Err(ReflexError::InvalidInput(format!(
                "json_schema can't be combined with {field}: that keeps the state after the prompt, before any value is generated"
            )));
        }
        sampling.json_schema = Some(model.compile_json_schema(raw.get())?);
    }
    let result = run_generate_with(
        model,
        slot,
        req,
        &sampling,
        kv_io_ms,
        on_first_token,
        on_token,
    );
    *mask_ms = sampling.json_schema.as_ref().map(|s| s.mask_ms_total());
    result
}

/// [`run_generate`] after the request's sampling parameters are built.
#[allow(clippy::too_many_arguments)]
fn run_generate_with(
    model: &Model,
    slot: &mut PrefixSlot,
    req: &IpcRequest,
    sampling: &crate::sampling::SamplingParams,
    kv_io_ms: &mut Option<f64>,
    on_first_token: impl FnMut(&[f32]),
    on_token: impl FnMut(u32, &str),
) -> Result<(Vec<u32>, String), ReflexError> {
    let export_dtype = export_kv_dtype(req, model.kv_cache_dtype())?;
    let uses_slot = req.pin_prefix || req.use_pinned;
    if uses_slot && (req.import_kv.is_some() || req.export_kv.is_some()) {
        return Err(ReflexError::InvalidInput(
            "pin_prefix/use_pinned can't be combined with import_kv/export_kv".to_string(),
        ));
    }
    if (req.export_kv.is_some() || req.pin_prefix) && req.max_tokens != 1 {
        let field = if req.pin_prefix {
            "pin_prefix"
        } else {
            "export_kv"
        };
        return Err(ReflexError::InvalidInput(format!(
            "{field} keeps the state right after the prompt, so it needs \"max_tokens\": 1"
        )));
    }
    if uses_slot {
        model.check_prefix_state_supported()?;
    }
    if req.use_pinned && slot.pinned.is_none() {
        return Err(ReflexError::InvalidInput(
            "use_pinned: no prefix is pinned in this session; send a pin_prefix request first"
                .to_string(),
        ));
    }
    let pinned_resume = || match &slot.pinned {
        Some(p) if req.use_pinned => Resume::Pinned(p),
        _ => Resume::Fresh,
    };
    if req.pin_prefix {
        let (token_ids, text, pinned, copy_ms) = model.generate_pin_prefix(
            &req.prompt,
            pinned_resume(),
            sampling,
            on_first_token,
            on_token,
        )?;
        *kv_io_ms = copy_ms;
        slot.pinned = Some(pinned);
        return Ok((token_ids, text));
    }
    if req.use_pinned {
        let (token_ids, text, copy_ms) = model.generate_resumed(
            &req.prompt,
            req.max_tokens.max(1),
            pinned_resume(),
            sampling,
            on_first_token,
            on_token,
        )?;
        *kv_io_ms = copy_ms;
        return Ok((token_ids, text));
    }

    let imported = match &req.import_kv {
        Some(path) => {
            let start = std::time::Instant::now();
            let kv = crate::kv_io::import_kv(path)?;
            *kv_io_ms = Some(elapsed_ms(start));
            Some(kv)
        }
        None => None,
    };
    // The file's payload is read while it is uploaded (`kv_io::import_kv`
    // reads only the header), so the upload counts as part of the read.
    let resume = imported.as_ref().map_or(Resume::Fresh, Resume::File);
    let Some(export_path) = &req.export_kv else {
        let (token_ids, text, upload_ms) = model.generate_resumed(
            &req.prompt,
            req.max_tokens.max(1),
            resume,
            sampling,
            on_first_token,
            on_token,
        )?;
        add_ms(kv_io_ms, upload_ms);
        return Ok((token_ids, text));
    };
    // `Model::generate_capture_kv`, keeping the upload time.
    let (token_ids, text, state, upload_ms) =
        model.generate_pin_prefix(&req.prompt, resume, sampling, on_first_token, on_token)?;
    add_ms(kv_io_ms, upload_ms);
    let cache = model.download_prefix_as(&state, export_dtype)?;
    let start = std::time::Instant::now();
    crate::kv_io::export_kv(export_path, &cache)?;
    add_ms(kv_io_ms, Some(elapsed_ms(start)));
    Ok((token_ids, text))
}

/// Adds `ms` (if any) to the request's `kv_io_ms`.
fn add_ms(kv_io_ms: &mut Option<f64>, ms: Option<f64>) {
    if let Some(ms) = ms {
        *kv_io_ms = Some(kv_io_ms.unwrap_or(0.0) + ms);
    }
}

/// A `generate` request's `export_kv_dtype`, only together with `export_kv`; by
/// default the working cache's type (`cache`: f32 unless `REFLEX_KV_CACHE=f16`).
fn export_kv_dtype(
    req: &IpcRequest,
    cache: crate::kv_io::KvDtype,
) -> Result<crate::kv_io::KvDtype, ReflexError> {
    match &req.export_kv_dtype {
        None => Ok(cache),
        Some(_) if req.export_kv.is_none() => Err(ReflexError::InvalidInput(
            "export_kv_dtype applies only with export_kv".to_string(),
        )),
        Some(d) => crate::kv_io::KvDtype::parse(d),
    }
}

/// The error for a field that keeps a prefix (`export_kv`/`pin_prefix`) on a
/// `system1_evaluate` request: scoring writes past the prompt, so there is no
/// state right after it to keep.
fn kv_fields_on_candidates_err() -> ReflexError {
    ReflexError::InvalidInput(
        "export_kv/pin_prefix apply only to generate requests (empty candidates); a candidates request can resume from use_pinned or import_kv".to_string(),
    )
}

/// Checks a `system1_evaluate` request's fields before any model work: it may
/// resume from `use_pinned` or `import_kv` (not both), and can't keep a prefix
/// or take a `json_schema`.
fn check_candidates_fields(req: &IpcRequest) -> Result<(), ReflexError> {
    if req.export_kv.is_some() || req.export_kv_dtype.is_some() || req.pin_prefix {
        return Err(kv_fields_on_candidates_err());
    }
    if req.use_pinned && req.import_kv.is_some() {
        return Err(ReflexError::InvalidInput(
            "use_pinned can't be combined with import_kv".to_string(),
        ));
    }
    if req.json_schema.is_some() {
        return Err(ReflexError::InvalidInput(
            "json_schema applies only to generate requests (empty candidates)".to_string(),
        ));
    }
    Ok(())
}

/// Runs a `system1_evaluate` request, resuming from the session's [`PrefixSlot`]
/// (`use_pinned`) or a cache file (`import_kv`) when it names one. Records the
/// file read, or the copy of the pinned prefix, in `kv_io_ms`. Never changes
/// `slot`.
fn run_system1(
    model: &Model,
    slot: &PrefixSlot,
    req: &IpcRequest,
    kv_io_ms: &mut Option<f64>,
) -> Result<crate::model::System1Response, ReflexError> {
    check_candidates_fields(req)?;
    let candidates: Vec<System1Candidate> = req
        .candidates
        .iter()
        .map(|text| System1Candidate { text: text.clone() })
        .collect();
    if !req.use_pinned && req.import_kv.is_none() {
        return model.system1_evaluate(&req.prompt, &candidates, req.temperature);
    }
    model.check_prefix_state_supported()?;
    let imported = match &req.import_kv {
        Some(path) => {
            let start = std::time::Instant::now();
            let kv = crate::kv_io::import_kv(path)?;
            *kv_io_ms = Some(elapsed_ms(start));
            Some(kv)
        }
        None => None,
    };
    let resume =
        match (&imported, &slot.pinned) {
            (Some(kv), _) => Resume::File(kv),
            (None, Some(p)) => Resume::Pinned(p),
            (None, None) => return Err(ReflexError::InvalidInput(
                "use_pinned: no prefix is pinned in this session; send a pin_prefix request first"
                    .to_string(),
            )),
        };
    // The pinned prefix's copy, or the file's upload (its read, see
    // `run_generate`), on top of the header read above.
    let (response, copy_ms) =
        model.system1_evaluate_resumed(&req.prompt, resume, &candidates, req.temperature)?;
    add_ms(kv_io_ms, copy_ms);
    Ok(response)
}

/// The index of the highest of `probabilities` (the first one on a tie): a
/// `candidates` response's `chosen`.
fn chosen_index(probabilities: &[f32]) -> usize {
    let mut best = 0;
    for (i, &p) in probabilities.iter().enumerate() {
        if p > probabilities[best] {
            best = i;
        }
    }
    best
}

impl IpcResponse {
    /// An `ok: false` line. A `grammar_dead_end` error also carries the partial
    /// `text` and `token_ids`.
    fn err(id: Option<String>, error: ReflexError) -> Self {
        let (token_ids, text) = match &error {
            ReflexError::GrammarDeadEnd {
                text, token_ids, ..
            } => (Some(token_ids.clone()), Some(text.clone())),
            _ => (None, None),
        };
        IpcResponse {
            id,
            event: "final",
            ok: false,
            error_kind: Some(error.kind()),
            error: Some(error.to_string()),
            token_ids,
            text,
            candidates: None,
            entropy: None,
            chosen: None,
            timings: None,
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
/// [`handle_request_streaming`] for the streaming path. `slot` is the session's
/// pinned prefix (`pin_prefix`/`use_pinned`).
pub fn handle_request(model: &Model, slot: &mut PrefixSlot, req: IpcRequest) -> IpcResponse {
    let id = req.id.clone();
    let model_start = std::time::Instant::now();
    if req.candidates.is_empty() {
        let mut first_token_ms = None;
        let mut kv_io_ms = None;
        let mut mask_ms = None;
        match run_generate(
            model,
            slot,
            &req,
            &mut kv_io_ms,
            &mut mask_ms,
            |_logits| first_token_ms = Some(elapsed_ms(model_start)),
            |_token_id, _text| {},
        ) {
            Ok((token_ids, text)) => IpcResponse {
                id,
                event: "final",
                ok: true,
                error: None,
                error_kind: None,
                token_ids: Some(token_ids),
                text: Some(text),
                candidates: None,
                entropy: None,
                chosen: None,
                timings: timings_if(
                    req.timings,
                    req.parse_ms,
                    model_start,
                    first_token_ms,
                    kv_io_ms,
                    mask_ms,
                    slot,
                ),
            },
            // A failed generate request still reports its timings when asked, so a
            // caller can account for a `grammar_dead_end`'s work too.
            Err(e) => IpcResponse {
                timings: timings_if(
                    req.timings,
                    req.parse_ms,
                    model_start,
                    first_token_ms,
                    kv_io_ms,
                    mask_ms,
                    slot,
                ),
                ..IpcResponse::err(id, e)
            },
        }
    } else {
        let mut kv_io_ms = None;
        match run_system1(model, slot, &req, &mut kv_io_ms) {
            Ok(response) => {
                let chosen = chosen_index(&response.probabilities);
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
                    error_kind: None,
                    token_ids: None,
                    text: None,
                    candidates: Some(candidates),
                    entropy: Some(response.entropy),
                    chosen: Some(chosen),
                    timings: timings_if(
                        req.timings,
                        req.parse_ms,
                        model_start,
                        None,
                        kv_io_ms,
                        None,
                        slot,
                    ),
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
    slot: &mut PrefixSlot,
    req: IpcRequest,
    mut output: W,
) -> Result<(), String> {
    if !req.stream || !req.candidates.is_empty() {
        let response = handle_request(model, slot, req);
        return write_json_line(&mut output, &response);
    }

    let id = req.id.clone();
    let mut write_err: Option<String> = None;
    let model_start = std::time::Instant::now();
    let mut first_token_ms = None;
    let mut kv_io_ms = None;
    let mut mask_ms = None;
    let result = run_generate(
        model,
        slot,
        &req,
        &mut kv_io_ms,
        &mut mask_ms,
        |_logits| first_token_ms = Some(elapsed_ms(model_start)),
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
            error_kind: None,
            token_ids: Some(token_ids),
            text: Some(text),
            candidates: None,
            entropy: None,
            chosen: None,
            timings: timings_if(
                req.timings,
                req.parse_ms,
                model_start,
                first_token_ms,
                kv_io_ms,
                mask_ms,
                slot,
            ),
        },
        // A failed generate request still reports its timings when asked, so a
        // caller can account for a `grammar_dead_end`'s work too.
        Err(e) => IpcResponse {
            timings: timings_if(
                req.timings,
                req.parse_ms,
                model_start,
                first_token_ms,
                kv_io_ms,
                mask_ms,
                slot,
            ),
            ..IpcResponse::err(id, e)
        },
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
/// `slot` is the pinned prefix (`pin_prefix`/`use_pinned`), owned by the caller
/// so it can outlive one connection.
pub fn run_request_loop<R: BufRead, W: Write>(
    model: &Model,
    slot: &mut PrefixSlot,
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

        let parse_start = std::time::Instant::now();
        match serde_json::from_str::<IpcRequest>(trimmed) {
            Ok(mut req) => {
                req.parse_ms = elapsed_ms(parse_start);
                handle_request_streaming(model, slot, req, &mut output)?
            }
            Err(e) => {
                let response = IpcResponse::err(
                    None,
                    crate::reflex_err!(InvalidInput, "ipc: malformed request JSON: {e}"),
                );
                write_json_line(&mut output, &response)?;
            }
        }
    }
}

/// [`run_request_loop`] over `std::io::stdin()`/`std::io::stdout()` specifically --
/// the shape `reflex stdio`'s `run` needs. The session's [`PrefixSlot`] lives as
/// long as the process.
pub fn run_stdio_loop(model: &Model) -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run_request_loop(model, &mut PrefixSlot::new(), stdin.lock(), stdout.lock())
}

#[cfg(test)]
mod json_schema_tests;
#[cfg(test)]
mod relay_tests;

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
        let resp = IpcResponse::err(
            Some("id1".to_string()),
            ReflexError::Other("boom".to_string()),
        );
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
    fn test_ipc_error_response_carries_error_kind() {
        let overflow = crate::limits::check_positions(0, crate::limits::ATTN_MAX_POSITIONS + 1, 0)
            .unwrap_err();
        let json = serde_json::to_string(&IpcResponse::err(None, overflow)).unwrap();
        assert!(
            json.contains("\"error_kind\":\"context_overflow\""),
            "{json}"
        );
        assert!(
            json.contains(&format!(
                "\"error\":\"{}",
                crate::limits::CONTEXT_LENGTH_EXCEEDED_PREFIX
            )),
            "{json}"
        );
    }

    #[test]
    fn test_ipc_success_response_omits_error_kind() {
        let resp = IpcResponse {
            id: None,
            event: "final",
            ok: true,
            error: None,
            error_kind: None,
            token_ids: Some(vec![1]),
            text: Some("x".to_string()),
            candidates: None,
            entropy: None,
            chosen: None,
            timings: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("error_kind"), "{json}");
        assert!(!json.contains("timings"), "{json}");
    }

    #[test]
    fn test_malformed_json_line_produces_error_response_not_a_loop_abort() {
        let model_free_response = match serde_json::from_str::<IpcRequest>("not json") {
            Ok(_) => panic!("expected parse failure"),
            Err(e) => IpcResponse::err(
                None,
                crate::reflex_err!(InvalidInput, "ipc: malformed request JSON: {e}"),
            ),
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
        let final_resp = IpcResponse::err(
            Some("id1".to_string()),
            ReflexError::Other("boom".to_string()),
        );
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
        let resp = IpcResponse::err(None, ReflexError::Other("boom".to_string()));
        write_json_line(&mut buf, &resp).expect("should write");
        let text = String::from_utf8(buf).expect("should be valid UTF-8");
        assert_eq!(
            text.matches('\n').count(),
            1,
            "should write exactly one line: {text:?}"
        );
        assert!(text.trim_end().starts_with('{') && text.trim_end().ends_with('}'));
    }

    #[test]
    fn test_ipc_request_timings_defaults_off_and_parses_on() {
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "p"}"#).unwrap();
        assert!(!req.timings);
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "p", "timings": true}"#).unwrap();
        assert!(req.timings);
        // `parse_ms` is filled in by the request loop, never read off the wire.
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "p", "parse_ms": 5.0}"#).unwrap();
        assert_eq!(req.parse_ms, 0.0);
    }

    #[test]
    fn test_timings_if_only_builds_when_requested() {
        let start = std::time::Instant::now();
        let slot = PrefixSlot::new();
        assert!(timings_if(false, 1.0, start, Some(2.0), None, None, &slot).is_none());
        let t = timings_if(true, 1.0, start, None, None, None, &slot).expect("requested");
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"parse_ms\":1.0"), "{json}");
        assert!(json.contains("\"model_ms\""), "{json}");
        assert!(!json.contains("first_token_ms"), "{json}");
        assert!(!json.contains("kv_io_ms"), "{json}");
        let t = timings_if(true, 1.0, start, None, Some(3.0), Some(0.5), &slot).expect("requested");
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"kv_io_ms\":3.0"), "{json}");
        assert!(json.contains("\"mask_ms_total\":0.5"), "{json}");
    }

    #[test]
    fn test_ipc_request_kv_paths_default_off_and_parse() {
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "p"}"#).unwrap();
        assert!(req.import_kv.is_none() && req.export_kv.is_none());
        let req: IpcRequest = serde_json::from_str(
            r#"{"prompt": "p", "import_kv": "/dev/shm/sys.cskv", "export_kv": "out.cskv", "max_tokens": 1}"#,
        )
        .unwrap();
        assert_eq!(req.import_kv.as_deref(), Some("/dev/shm/sys.cskv"));
        assert_eq!(req.export_kv.as_deref(), Some("out.cskv"));
        assert_eq!(req.max_tokens, 1);
    }

    #[test]
    fn test_export_kv_dtype_defaults_to_the_cache_type_and_needs_export_kv() {
        use crate::kv_io::KvDtype;
        let parse = |s: &str| serde_json::from_str::<IpcRequest>(s).unwrap();
        let dtype = |s: &str| export_kv_dtype(&parse(s), KvDtype::F32);
        assert_eq!(
            dtype(r#"{"prompt": "p", "export_kv": "o", "max_tokens": 1}"#).unwrap(),
            KvDtype::F32
        );
        // An f16 working cache exports f16 unless the request asks for f32.
        let req = parse(r#"{"prompt": "p", "export_kv": "o", "max_tokens": 1}"#);
        assert_eq!(export_kv_dtype(&req, KvDtype::F16).unwrap(), KvDtype::F16);
        let req = parse(r#"{"prompt": "p", "export_kv": "o", "export_kv_dtype": "f32"}"#);
        assert_eq!(export_kv_dtype(&req, KvDtype::F16).unwrap(), KvDtype::F32);
        assert_eq!(dtype(r#"{"prompt": "p"}"#).unwrap(), KvDtype::F32);
        for (d, want) in [("f32", KvDtype::F32), ("f16", KvDtype::F16)] {
            let req = format!(r#"{{"prompt": "p", "export_kv": "o", "export_kv_dtype": "{d}"}}"#);
            assert_eq!(dtype(&req).unwrap(), want);
        }
        for bad in [
            r#"{"prompt": "p", "export_kv": "o", "export_kv_dtype": "bf16"}"#,
            r#"{"prompt": "p", "export_kv": "o", "export_kv_dtype": "F16"}"#,
            r#"{"prompt": "p", "export_kv_dtype": "f16"}"#,
            r#"{"prompt": "p", "import_kv": "i", "export_kv_dtype": "f16"}"#,
        ] {
            let err = dtype(bad).unwrap_err();
            assert_eq!(err.kind(), "invalid_input", "{bad}: {err}");
        }
        assert!(serde_json::from_str::<IpcRequest>(
            r#"{"prompt": "p", "export_kv": "o", "export_kv_dtype": 16}"#
        )
        .is_err());
    }

    #[test]
    fn test_ipc_request_pin_fields_default_off_and_parse() {
        let req: IpcRequest = serde_json::from_str(r#"{"prompt": "p"}"#).unwrap();
        assert!(!req.pin_prefix && !req.use_pinned);
        let req: IpcRequest = serde_json::from_str(
            r#"{"prompt": "p", "pin_prefix": true, "use_pinned": true, "max_tokens": 1}"#,
        )
        .unwrap();
        assert!(req.pin_prefix && req.use_pinned);
        assert_eq!(PrefixSlot::new().seq_len(), None);
        assert_eq!(PrefixSlot::new().device_bytes(), 0);
    }

    fn candidates_req(extra: &str) -> IpcRequest {
        serde_json::from_str(&format!(
            r#"{{"prompt": "p", "candidates": [" a", " b"]{extra}}}"#
        ))
        .expect("should parse")
    }

    #[test]
    fn test_candidates_allow_resuming_from_a_prefix() {
        for extra in [
            "",
            r#", "use_pinned": true"#,
            r#", "import_kv": "sys.cskv""#,
            r#", "use_pinned": true, "timings": true"#,
        ] {
            let req = candidates_req(extra);
            assert!(check_candidates_fields(&req).is_ok(), "{extra}");
        }
        let req = candidates_req(r#", "use_pinned": true, "import_kv": "sys.cskv""#);
        assert!(req.use_pinned && req.import_kv.is_some());
    }

    #[test]
    fn test_candidates_reject_keeping_a_prefix_and_other_fields() {
        for extra in [
            r#", "pin_prefix": true"#,
            r#", "pin_prefix": true, "max_tokens": 1"#,
            r#", "export_kv": "out.cskv""#,
            r#", "use_pinned": true, "pin_prefix": true"#,
            r#", "import_kv": "sys.cskv", "export_kv": "out.cskv""#,
            r#", "use_pinned": true, "import_kv": "sys.cskv""#,
            r#", "json_schema": {"type": "string"}"#,
            r#", "use_pinned": true, "json_schema": {"type": "string"}"#,
            r#", "export_kv_dtype": "f16""#,
        ] {
            let err = check_candidates_fields(&candidates_req(extra)).unwrap_err();
            assert_eq!(err.kind(), "invalid_input", "{extra}: {err}");
        }
        let err = check_candidates_fields(&candidates_req(r#", "pin_prefix": true"#)).unwrap_err();
        assert!(err.to_string().contains("export_kv/pin_prefix"), "{err}");
        assert!(!err.to_string().contains("use_pinned apply"), "{err}");
    }

    #[test]
    fn test_chosen_index_picks_the_first_highest_probability() {
        assert_eq!(chosen_index(&[0.7, 0.2, 0.1]), 0);
        assert_eq!(chosen_index(&[0.1, 0.2, 0.7]), 2);
        assert_eq!(chosen_index(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(chosen_index(&[1.0]), 0);
    }

    fn candidates_response(chosen: Option<usize>, timings: Option<IpcTimings>) -> IpcResponse {
        IpcResponse {
            id: None,
            event: "final",
            ok: true,
            error: None,
            error_kind: None,
            token_ids: None,
            text: None,
            candidates: Some(vec![IpcCandidateResult {
                text: " a".to_string(),
                token_ids: vec![1],
                score: 2.0,
                probability: 1.0,
            }]),
            entropy: Some(0.0),
            chosen,
            timings,
        }
    }

    #[test]
    fn test_chosen_serializes_on_candidates_only() {
        let json = serde_json::to_string(&candidates_response(Some(0), None)).unwrap();
        assert!(json.contains("\"chosen\":0"), "{json}");
        let err =
            serde_json::to_string(&IpcResponse::err(None, ReflexError::Other("x".into()))).unwrap();
        assert!(!err.contains("chosen"), "{err}");
    }

    #[test]
    fn test_slot_facts_appear_only_with_timings() {
        let json = serde_json::to_string(&candidates_response(Some(0), None)).unwrap();
        assert!(!json.contains("pinned_"), "{json}");
        let start = std::time::Instant::now();
        let t = timings_if(true, 0.5, start, None, None, None, &PrefixSlot::new());
        let json = serde_json::to_string(&candidates_response(Some(0), t)).unwrap();
        assert!(json.contains("\"pinned_seq_len\":0"), "{json}");
        assert!(json.contains("\"pinned_bytes\":0"), "{json}");
    }
}
