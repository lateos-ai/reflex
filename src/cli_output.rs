//! `--json` output for `generate`/`system1`/`bench`/`smoke`/`check`/`doctor`.
//! `#[cfg(feature = "json-output")]` for the whole module: deliberately its
//! own feature rather than reusing `ipc` (see Cargo.toml), since JSON output
//! on these subcommands has nothing to do with the `stdio`/`uds` IPC
//! transport `ipc` gates.
//!
//! One JSON object per existing `REFLEX_*_OK`-style stdout line, printed at
//! the exact same call site the plain-text `println!` was, not one
//! aggregated end-of-run object -- mirrors each subcommand's existing
//! multi-line shape (e.g. `generate --lora` prints a `REFLEX_LORA_OK` line
//! *and* a `REFLEX_GENERATE_OK` line; `--json` prints one JSON object per
//! line in the same order, not a merged object). Field names match each
//! plain-text line's `key=value` names 1:1, using the same `Option<T>` +
//! `#[serde(skip_serializing_if = "Option::is_none")]` convention
//! `src/ipc.rs`'s `IpcResponse` already established for fields that are only
//! sometimes present.
//!
//! **Versioning**: the versioned result objects ([`GenerateResultJson`],
//! [`System1ResultJson`], [`SmokeResultJson`], [`PhaseTimingJson`]) also carry
//! a `schema_version` field (the current [`SCHEMA_VERSION`]) -- the one field
//! with no plain-text `key=value` counterpart, added so a JSON consumer can
//! detect a shape change. It's additive; readers that ignore unknown fields
//! keep working across a minor bump. See README's "`--json` output contract"
//! section.
//!
//! **Scope boundary**: only the stdout success-path contract lines
//! (`REFLEX_*_OK`/`REFLEX_CHECK`/`REFLEX_DOCTOR_CHECK`) get a JSON form.
//! `check.rs`'s `REFLEX_CHECK_FAIL` diagnostics (stderr, varying fields per
//! failure reason) are folded into `CheckJson`'s `pass`/`fail_reason`
//! instead of mirrored line-for-line -- a nonzero exit code plus stderr text
//! already covers the failure case for a caller that isn't parsing JSON, and
//! the two `--reference` failure reasons don't share a field shape worth
//! preserving 1:1 in a typed struct.
//!
//! Plain-text output (no `--json`) is unchanged byte-for-byte by this
//! module's existence -- every call site branches on the flag and calls
//! either its existing `println!` or [`print_json_line`], never both.

use serde::Serialize;

/// Version of the `--json` output contract, emitted as the `schema_version`
/// field on the versioned result objects ([`GenerateResultJson`],
/// [`System1ResultJson`], [`SmokeResultJson`], and [`PhaseTimingJson`]).
///
/// Additive/forward-compatible: `serde` readers that ignore unknown fields
/// keep working across a minor bump, so a *major* bump is the signal that an
/// existing field changed meaning or was removed. Keep in sync with the
/// README's "`--json` output contract" section.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Serializes `value` as one line of JSON to stdout, matching `src/ipc.rs`'s
/// `write_json_line` discipline (flushed immediately) but deliberately not
/// sharing that function directly -- `ipc.rs`'s version is private and lives
/// behind the `ipc` feature, and duplicating these few lines here keeps
/// `json-output` decoupled from the IPC transport feature.
pub fn print_json_line<T: Serialize>(value: &T) {
    let json = serde_json::to_string(value).expect("serializing --json output");
    println!("{json}");
}

/// `REFLEX_LORA_OK` -- identical shape across `generate`/`system1`/`bench`,
/// the three subcommands that support `--lora`.
#[derive(Serialize)]
pub struct LoraAppliedJson {
    pub path: String,
    pub tensors_applied: usize,
}

/// `REFLEX_GENERATE_KV_EXPORT_OK`.
#[derive(Serialize)]
pub struct GenerateKvExportJson {
    pub path: String,
    pub kind: &'static str,
    pub seq_len: usize,
    pub num_layers: usize,
}

/// `REFLEX_GENERATE_OK`. `process_start_to_last_token_ms`/`num_generated`/
/// `token_ids` are `None` on the `--export-kv` early-exit path, which prints
/// a narrower line than the ordinary decode path -- see `generate.rs`'s two
/// `REFLEX_GENERATE_OK` call sites.
#[derive(Serialize)]
pub struct GenerateResultJson {
    pub schema_version: &'static str,
    pub process_start_to_first_token_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_start_to_last_token_ms: Option<f64>,
    pub gguf_open_ms: f64,
    pub cuda_init_ms: f64,
    pub model_load_ms: f64,
    pub prompt_eval_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_generated: Option<usize>,
    pub token_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
    pub token_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_method: Option<&'static str>,
}

/// `REFLEX_SYSTEM1_CANDIDATE_OK`.
#[derive(Serialize)]
pub struct System1CandidateJson {
    pub idx: usize,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub score: f32,
    pub probability: f32,
}

/// `REFLEX_SYSTEM1_OK`.
#[derive(Serialize)]
pub struct System1ResultJson {
    pub schema_version: &'static str,
    pub process_start_to_result_ms: f64,
    pub gguf_open_ms: f64,
    pub cuda_init_ms: f64,
    pub model_load_ms: f64,
    pub prompt_eval_ms: f64,
    pub num_candidates: usize,
    pub best_idx: usize,
    pub best_text: String,
    pub entropy: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_method: Option<&'static str>,
}

/// `REFLEX_SMOKE_OK`.
#[derive(Serialize)]
pub struct SmokeResultJson {
    pub schema_version: &'static str,
    pub process_start_to_first_result_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_method: Option<&'static str>,
}

/// `REFLEX_PHASE_OK` -- one per named execution phase (`gguf_open`,
/// `cuda_init`, `model_load`, `prompt_eval`), printed *before* the aggregate
/// `REFLEX_GENERATE_OK`/`REFLEX_SYSTEM1_OK` line. `energy_joules` is the
/// delta across just this phase (not cumulative) and is `None` when no
/// energy measurement is available (no `nvml` feature, or NVML unavailable on
/// this machine). See `src/energy.rs`'s doc comment for the
/// `total_energy_counter`-vs-`polled_power` granularity caveat that makes
/// short-phase deltas coarse in counter mode.
#[derive(Serialize)]
pub struct PhaseTimingJson {
    pub schema_version: &'static str,
    pub phase: &'static str,
    pub duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_joules: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_method: Option<&'static str>,
}

/// `REFLEX_BENCH_VRAM_OK`.
#[derive(Serialize)]
pub struct BenchVramJson {
    pub model_resident_mib: usize,
    pub free_before_load_mib: usize,
    pub free_after_load_mib: usize,
}

/// `REFLEX_BENCH_WARM_OK`/`REFLEX_BENCH_SYSTEM1_OK` (identical shape, `kind`
/// distinguishes them since both otherwise serialize the same fields).
#[derive(Serialize)]
pub struct BenchStatsJson {
    pub kind: &'static str,
    pub prompt_tokens: usize,
    pub warmup: usize,
    pub iters: usize,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// `REFLEX_BENCH_THROUGHPUT_OK`.
#[derive(Serialize)]
pub struct BenchThroughputJson {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub warmup: usize,
    pub iters: usize,
    pub tokens_per_sec: f64,
    pub ms_per_token: f64,
}

/// `REFLEX_BENCH_ENERGY_OK`.
#[derive(Serialize)]
pub struct BenchEnergyJson {
    pub prompt_tokens: usize,
    pub iters: usize,
    pub total_joules: f64,
    pub joules_per_forward_pass: f64,
    pub energy_method: &'static str,
}

/// `REFLEX_CHECK` plus its outcome -- see this module's doc comment's
/// "Scope boundary" section for why `REFLEX_CHECK_FAIL`'s varying
/// per-reason fields aren't mirrored 1:1.
#[derive(Serialize)]
pub struct CheckJson {
    pub token_ids: Vec<u32>,
    pub token_texts: Vec<String>,
    pub logit_checksum: f64,
    pub top1_logit: f32,
    pub vocab_size: usize,
    /// `None` when `--reference` wasn't passed (nothing to compare against).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_reason: Option<String>,
}

/// `REFLEX_DOCTOR_CHECK`.
#[derive(Serialize)]
pub struct DoctorCheckJson {
    pub name: &'static str,
    pub status: &'static str,
    pub detail: String,
}

/// `REFLEX_DOCTOR_OK`/`REFLEX_DOCTOR_FAIL` -- `ok` distinguishes them
/// (`checks_failed == 0`).
#[derive(Serialize)]
pub struct DoctorSummaryJson {
    pub ok: bool,
    pub checks_passed: usize,
    pub checks_warned: usize,
    pub checks_failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // These lock the `--json` contract's one non-plain-text field: the version
    // is emitted, and (being declared first) serializes first, so a consumer
    // can read it before the rest of the object. They only run under
    // `cargo test --features json-output` (the module is feature-gated), not
    // in CI's default-feature test job -- see ci.yml.

    #[test]
    fn smoke_result_json_emits_schema_version_first() {
        let json = serde_json::to_string(&SmokeResultJson {
            schema_version: SCHEMA_VERSION,
            process_start_to_first_result_ms: 1.0,
            joules: None,
            energy_method: None,
        })
        .unwrap();
        assert_eq!(
            json,
            r#"{"schema_version":"1.0.0","process_start_to_first_result_ms":1.0}"#
        );
    }

    #[test]
    fn phase_timing_json_emits_schema_version_first() {
        let json = serde_json::to_string(&PhaseTimingJson {
            schema_version: SCHEMA_VERSION,
            phase: "prompt_eval",
            duration_ms: 1.0,
            energy_joules: None,
            energy_method: None,
        })
        .unwrap();
        assert_eq!(
            json,
            r#"{"schema_version":"1.0.0","phase":"prompt_eval","duration_ms":1.0}"#
        );
    }
}
