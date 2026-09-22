//! Rust C-FFI surface for Phase 4 (Embeddability) round 2 (see README.md's
//! "Phase 4 -- Embeddability" roadmap entry and DECISIONS.md for the exact
//! scope confirmed with the user before starting). Lets an external
//! orchestrator process link against this crate directly (`cdylib`/
//! `staticlib`, see Cargo.toml's `[lib]` section) and call
//! load -> generate -> free, instead of `exec`-ing `qwen3_coldstart` and
//! parsing its stdout.
//!
//! Deliberately narrow, matching this project's round-by-round precedent:
//! `coldstart_load` (GGUF path, optional load-time LoRA adapter path --
//! `--lora` is load-time-only already per Phase 4 round 1, so it folds into
//! the load call rather than needing its own FFI entry point),
//! `coldstart_generate` (prompt in, token ids + text out),
//! `coldstart_system1_evaluate` (prompt + candidate strings in, per-candidate
//! scores/probabilities out -- see `crate::model::Model::system1_evaluate`'s
//! doc comment), `coldstart_free`. Phase 3's `--export-kv`/`--import-kv`
//! state I/O is deliberately **not** exposed here -- confirmed out of scope
//! for this round, a separable capability an embedding host may not need
//! yet.
//!
//! **Panics never cross the FFI boundary.** Every entry point wraps its body
//! in `catch_unwind` and converts both an `Err(String)` (this crate's usual
//! `Result<_, String>` convention) and a caught panic into the same
//! thread-local last-error-string convention (`coldstart_last_error`) --
//! unwinding a Rust panic across an `extern "C"` boundary is undefined
//! behavior in the C caller.
//!
//! **Concurrency**: matches README's Non-goals -- `batch_size` is still
//! always 1 and there is still no request queue/scheduler inside this
//! engine. A single `ColdstartModel` handle must not be used from more than
//! one thread concurrently (nothing here is `Sync`-safe); a host wanting
//! concurrency runs multiple processes/handles itself.
//!
//! **Header generation**: the C header (`include/coldstart_infer.h`) is
//! generated from this file with `cbindgen` and checked in, not regenerated
//! on every build (this file changes far less often than `.cu` kernels, and
//! build.rs's nvcc invocation should not gain a second toolchain
//! dependency). Regenerate it after editing this file's public surface with:
//! `cbindgen --config cbindgen.toml --crate coldstart-infer --output include/coldstart_infer.h`

use crate::gguf::GgufFile;
use crate::model::{Model, System1Candidate};
use cudarc::driver::CudaDevice;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

/// Opaque handle to a loaded model, obtained from `coldstart_load` and freed
/// with `coldstart_free`. C callers only ever see a pointer to this type --
/// never its layout.
pub struct ColdstartModel {
    model: Model,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: String) {
    // An embedded NUL would make CString::new fail outright; strip rather
    // than lose the message entirely (error text is host-controlled data --
    // a GGUF path, a Result<_, String> from model.rs/lora.rs -- not
    // attacker input, but still not guaranteed NUL-free).
    let sanitized = if msg.contains('\0') { msg.replace('\0', "") } else { msg };
    let c = CString::new(sanitized).unwrap_or_else(|_| CString::new("coldstart-infer: error message unrepresentable").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(c));
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic: <non-string panic payload>".to_string()
    }
}

/// Returns the last error message set by this thread's most recent failing
/// `coldstart_*` call, or NULL if the most recent call succeeded (or none
/// has run yet on this thread). The returned pointer is valid only until the
/// next `coldstart_*` call on this thread -- copy it if it needs to outlive
/// that.
#[no_mangle]
pub extern "C" fn coldstart_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

/// Loads a GGUF model from `gguf_path` onto CUDA device 0, optionally
/// applying a llama.cpp-format LoRA adapter from `lora_path` first (pass
/// NULL to skip -- see `Model::apply_lora`'s doc comment in `model.rs` for
/// which architectures/tensors are accepted; DeepSeek-V2/V3 MLA rejects any
/// LoRA path).
///
/// Returns an opaque handle to free with `coldstart_free`, or NULL on
/// failure (call `coldstart_last_error` for why). `gguf_path` must be
/// non-NULL; `lora_path` may be NULL. Both, when non-NULL, must be
/// NUL-terminated UTF-8 C strings; this function does not take ownership of
/// either and they may be freed by the caller as soon as it returns.
#[no_mangle]
pub extern "C" fn coldstart_load(gguf_path: *const c_char, lora_path: *const c_char) -> *mut ColdstartModel {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<ColdstartModel, String> {
        if gguf_path.is_null() {
            return Err("coldstart_load: gguf_path must not be NULL".to_string());
        }
        let gguf_path_str = unsafe { CStr::from_ptr(gguf_path) }
            .to_str()
            .map_err(|e| format!("coldstart_load: gguf_path is not valid UTF-8: {e}"))?;
        let lora_path_str = if lora_path.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(lora_path) }
                    .to_str()
                    .map_err(|e| format!("coldstart_load: lora_path is not valid UTF-8: {e}"))?,
            )
        };

        let file = GgufFile::open(gguf_path_str).map_err(|e| format!("coldstart_load: failed to open {gguf_path_str:?}: {e}"))?;
        let device = CudaDevice::new(0).map_err(|e| format!("coldstart_load: failed to init CUDA device 0: {e}"))?;
        let mut model = Model::load(device, &file)?;
        if let Some(lora_path_str) = lora_path_str {
            model.apply_lora(Path::new(lora_path_str))?;
        }
        Ok(ColdstartModel { model })
    }));

    match result {
        Ok(Ok(handle)) => Box::into_raw(Box::new(handle)),
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            ptr::null_mut()
        }
    }
}

/// Output of `coldstart_generate`. `token_ids`/`text` are heap buffers owned
/// by this crate -- free them (and reset this struct to all-zero/NULL) with
/// `coldstart_free_generate_result`, never with the C caller's own
/// `free`/`libc::free`.
#[repr(C)]
pub struct ColdstartGenerateResult {
    /// Array of `num_tokens` generated token ids.
    pub token_ids: *mut u32,
    pub num_tokens: usize,
    /// NUL-terminated UTF-8 decoded text of the generated tokens.
    pub text: *mut c_char,
}

/// Runs a forward pass and up to `max_new_tokens` greedy-decoded generation
/// steps (feeding each generated id back in, stopping early on the
/// tokenizer's EOS) starting fresh from `prompt` -- the same behavior as
/// `qwen3_coldstart <gguf> <prompt> --max-tokens <max_new_tokens>` with
/// neither `--export-kv` nor `--import-kv`. Phase 3's `--import-kv` resume
/// path is not exposed through this FFI round (see module doc comment).
///
/// `handle` must come from `coldstart_load` and not have been freed yet.
/// `prompt` must be a non-NULL, NUL-terminated UTF-8 C string; `max_new_tokens`
/// must be at least 1. `out` must be a non-NULL pointer to a
/// `ColdstartGenerateResult` the caller owns (its initial contents are
/// ignored, not read).
///
/// On success, fills `*out` and returns 0 -- the caller must eventually pass
/// `out` to `coldstart_free_generate_result`. On failure, leaves `*out`
/// untouched and returns -1 (call `coldstart_last_error` for why).
#[no_mangle]
pub extern "C" fn coldstart_generate(
    handle: *mut ColdstartModel,
    prompt: *const c_char,
    max_new_tokens: usize,
    out: *mut ColdstartGenerateResult,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<ColdstartGenerateResult, String> {
        if handle.is_null() {
            return Err("coldstart_generate: handle must not be NULL".to_string());
        }
        if prompt.is_null() {
            return Err("coldstart_generate: prompt must not be NULL".to_string());
        }
        if out.is_null() {
            return Err("coldstart_generate: out must not be NULL".to_string());
        }
        if max_new_tokens == 0 {
            return Err("coldstart_generate: max_new_tokens must be at least 1".to_string());
        }
        let prompt_str = unsafe { CStr::from_ptr(prompt) }
            .to_str()
            .map_err(|e| format!("coldstart_generate: prompt is not valid UTF-8: {e}"))?;
        let model = unsafe { &(*handle).model };

        let (tokens, text) = model.generate(prompt_str, max_new_tokens, None, |_logits| {})?;

        let mut boxed_tokens = tokens.into_boxed_slice();
        let token_ids = boxed_tokens.as_mut_ptr();
        let num_tokens = boxed_tokens.len();
        std::mem::forget(boxed_tokens);

        let text_c = CString::new(text.replace('\0', "")).unwrap_or_else(|_| CString::new("").unwrap());

        Ok(ColdstartGenerateResult { token_ids, num_tokens, text: text_c.into_raw() })
    }));

    match result {
        Ok(Ok(r)) => {
            unsafe { ptr::write(out, r) };
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            -1
        }
    }
}

/// Frees the buffers inside a `ColdstartGenerateResult` previously filled by
/// `coldstart_generate`, and zeroes the struct out. Safe to call on a
/// zeroed/all-NULL struct (no-op), and safe to call more than once on the
/// same struct for that reason -- but never on two different copies of the
/// same non-zeroed struct (double free).
#[no_mangle]
pub extern "C" fn coldstart_free_generate_result(result: *mut ColdstartGenerateResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &mut *result };
    if !r.token_ids.is_null() {
        unsafe { drop(Box::from_raw(std::slice::from_raw_parts_mut(r.token_ids, r.num_tokens))) };
        r.token_ids = ptr::null_mut();
        r.num_tokens = 0;
    }
    if !r.text.is_null() {
        unsafe { drop(CString::from_raw(r.text)) };
        r.text = ptr::null_mut();
    }
}

/// One candidate's scored result within a `ColdstartSystem1Result`. `text`/
/// `token_ids` are heap buffers owned by this crate, freed only via the
/// outer struct's `coldstart_free_system1_result` -- never individually.
/// `score`/`probability` are relative to the candidate set in this one
/// call, not vocab-normalized log-probabilities -- see
/// `crate::model::Model::system1_evaluate`'s doc comment.
#[repr(C)]
pub struct ColdstartSystem1CandidateResult {
    pub text: *mut c_char,
    pub token_ids: *mut u32,
    pub num_token_ids: usize,
    pub score: f32,
    pub probability: f32,
}

/// Output of `coldstart_system1_evaluate`. `candidates` is a heap buffer
/// owned by this crate -- free it (and reset this struct to all-zero/NULL)
/// with `coldstart_free_system1_result`, never with the C caller's own
/// `free`/`libc::free`. `entropy` (Shannon entropy, bits, of the `probability`
/// distribution across `candidates`) is a single confidence/escalation signal
/// alongside the per-candidate probabilities -- see
/// `crate::model::System1Response::entropy`'s doc comment.
#[repr(C)]
pub struct ColdstartSystem1Result {
    pub candidates: *mut ColdstartSystem1CandidateResult,
    pub num_candidates: usize,
    pub entropy: f32,
}

/// Runs System1 (single-pass, non-autoregressive candidate scoring, see
/// `crate::model::Model::system1_evaluate`'s doc comment) against `prompt`
/// for each of `num_candidates` candidate strings in `candidate_texts`. No
/// argmax-then-feedback decode loop runs -- single-token candidates are
/// scored in one batched gather-GEMV, multi-token candidates via a short
/// teacher-forced continuation. Dense/MoE Qwen3 models only; hybrid Qwen3.5
/// and DeepSeek-V2/V3 MLA are rejected with an error (call
/// `coldstart_last_error` for why).
///
/// `handle` must come from `coldstart_load` and not have been freed yet.
/// `prompt` must be a non-NULL, NUL-terminated UTF-8 C string. `candidate_texts`
/// must be a non-NULL pointer to `num_candidates` non-NULL, NUL-terminated
/// UTF-8 C string pointers, with `num_candidates` at least 1. `temperature`
/// (`1.0` = no-op) is passed through to the calibrated `probability` field
/// on each result. `out` must be a non-NULL pointer to a
/// `ColdstartSystem1Result` the caller owns (its initial contents are
/// ignored, not read).
///
/// On success, fills `*out` and returns 0 -- the caller must eventually pass
/// `out` to `coldstart_free_system1_result`. On failure, leaves `*out`
/// untouched and returns -1 (call `coldstart_last_error` for why).
#[no_mangle]
pub extern "C" fn coldstart_system1_evaluate(
    handle: *mut ColdstartModel,
    prompt: *const c_char,
    candidate_texts: *const *const c_char,
    num_candidates: usize,
    temperature: f32,
    out: *mut ColdstartSystem1Result,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<ColdstartSystem1Result, String> {
        if handle.is_null() {
            return Err("coldstart_system1_evaluate: handle must not be NULL".to_string());
        }
        if prompt.is_null() {
            return Err("coldstart_system1_evaluate: prompt must not be NULL".to_string());
        }
        if candidate_texts.is_null() {
            return Err("coldstart_system1_evaluate: candidate_texts must not be NULL".to_string());
        }
        if num_candidates == 0 {
            return Err("coldstart_system1_evaluate: num_candidates must be at least 1".to_string());
        }
        if out.is_null() {
            return Err("coldstart_system1_evaluate: out must not be NULL".to_string());
        }
        let prompt_str = unsafe { CStr::from_ptr(prompt) }
            .to_str()
            .map_err(|e| format!("coldstart_system1_evaluate: prompt is not valid UTF-8: {e}"))?;

        let candidate_ptrs = unsafe { std::slice::from_raw_parts(candidate_texts, num_candidates) };
        let candidates: Vec<System1Candidate> = candidate_ptrs
            .iter()
            .enumerate()
            .map(|(i, &ptr)| {
                if ptr.is_null() {
                    return Err(format!("coldstart_system1_evaluate: candidate_texts[{i}] must not be NULL"));
                }
                let text = unsafe { CStr::from_ptr(ptr) }
                    .to_str()
                    .map_err(|e| format!("coldstart_system1_evaluate: candidate_texts[{i}] is not valid UTF-8: {e}"))?
                    .to_string();
                Ok(System1Candidate { text })
            })
            .collect::<Result<_, String>>()?;

        let model = unsafe { &(*handle).model };
        let response = model.system1_evaluate(prompt_str, &candidates, temperature)?;
        let entropy = response.entropy;

        let mut boxed_candidates: Vec<ColdstartSystem1CandidateResult> = response
            .results
            .into_iter()
            .zip(response.probabilities)
            .map(|(r, probability)| {
                let text_c = CString::new(r.text.replace('\0', "")).unwrap_or_else(|_| CString::new("").unwrap());
                let mut boxed_ids = r.token_ids.into_boxed_slice();
                let token_ids = boxed_ids.as_mut_ptr();
                let num_token_ids = boxed_ids.len();
                std::mem::forget(boxed_ids);
                ColdstartSystem1CandidateResult { text: text_c.into_raw(), token_ids, num_token_ids, score: r.score, probability }
            })
            .collect();
        let candidates_ptr = boxed_candidates.as_mut_ptr();
        let num_candidates = boxed_candidates.len();
        std::mem::forget(boxed_candidates);

        Ok(ColdstartSystem1Result { candidates: candidates_ptr, num_candidates, entropy })
    }));

    match result {
        Ok(Ok(r)) => {
            unsafe { ptr::write(out, r) };
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(payload) => {
            set_last_error(panic_message(payload));
            -1
        }
    }
}

/// Frees the buffers inside a `ColdstartSystem1Result` previously filled by
/// `coldstart_system1_evaluate` (including every candidate's own `text`/
/// `token_ids`), and zeroes the struct out. Safe to call on a zeroed/
/// all-NULL struct (no-op), and safe to call more than once on the same
/// struct for that reason -- but never on two different copies of the same
/// non-zeroed struct (double free).
#[no_mangle]
pub extern "C" fn coldstart_free_system1_result(result: *mut ColdstartSystem1Result) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &mut *result };
    if !r.candidates.is_null() {
        let candidates = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(r.candidates, r.num_candidates)) };
        for c in candidates.into_vec() {
            if !c.text.is_null() {
                unsafe { drop(CString::from_raw(c.text)) };
            }
            if !c.token_ids.is_null() {
                unsafe { drop(Box::from_raw(std::slice::from_raw_parts_mut(c.token_ids, c.num_token_ids))) };
            }
        }
        r.candidates = ptr::null_mut();
        r.num_candidates = 0;
    }
}

/// Frees a handle obtained from `coldstart_load`. Safe to call with NULL
/// (no-op). The handle must not be used again afterward.
#[no_mangle]
pub extern "C" fn coldstart_free(handle: *mut ColdstartModel) {
    if handle.is_null() {
        return;
    }
    let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle));
    }));
    if let Err(payload) = result {
        set_last_error(panic_message(payload));
    }
}
