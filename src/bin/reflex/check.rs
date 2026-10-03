//! Formalizes this project's own byte-exact-vs-llama.cpp verification methodology
//! (see docs/benchmarks.md) as a user-facing correctness check instead of an
//! ad hoc development-only comparison. Runs a forward pass on `<gguf>` for
//! `<prompt>` and either prints a comparable summary (no `--reference`), or
//! compares against a hand-written reference file: expected token ids (exact
//! match -- the primary, always-available check, mirroring this project's own
//! established convention) and, if the reference supplies one, a logit checksum
//! compared within `--tolerance`. Cross-engine exact bit-match is **not**
//! expected even when both implementations are correct (cuBLAS/naive-GEMV/
//! llama.cpp's own kernels sum in a different order) -- see README.md's
//! System1 section for this project's own documented ~1e-4 gather-vs-full-vocab
//! delta precedent -- so `--tolerance` defaults to a real, nonzero relative
//! tolerance, not `0.0`.
//!
//! Reference file format (deliberately plain text, not JSON): line 1 is the
//! expected, comma-separated token ids; an optional line 2 is the expected
//! `logit_checksum` as a single `f64`.
//!
//! Usage: `reflex check <path-to-gguf> <prompt> [--max-tokens N] [--reference <file>] [--tolerance F] [--weights f16|f32] [--json]`
//!
//! Weights default to `f32` here, not the engine-wide `f16`: the byte-exact
//! comparison against llama.cpp is defined on the exact dequantized f32
//! weights. `--weights f16` (or `REFLEX_WEIGHTS=f16`) checks the f16 mode
//! instead, where token agreement is expected but logit checksums move by
//! roughly f16's rounding of each weight.
//!
//! Exit codes: `0` = pass, `1` = mismatch, `2` = internal error (bad args, load/
//! generate failure, malformed reference file) -- scriptable for CI. Uses
//! `std::process::exit` directly rather than `reflex_engine::fast_exit`,
//! deliberately: this exit-code contract is the whole point of the subcommand.
//!
//! `--json` (needs `cargo build --features json-output`) prints the result
//! as one line of JSON instead of the plain `REFLEX_CHECK key=value` text
//! plus a separate pass/fail line -- see `reflex_engine::cli_output`'s doc
//! comment for the exact shape and its "Scope boundary" note on why
//! `REFLEX_CHECK_FAIL`'s per-reason stderr diagnostics aren't mirrored 1:1.

use reflex_engine::diagnostics;
use reflex_engine::gguf::GgufFile;
use reflex_engine::model::WeightsDtype;

struct Reference {
    token_ids: Vec<u32>,
    logit_checksum: Option<f64>,
}

fn parse_reference(path: &str) -> Result<Reference, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read reference file {path:?}: {e}"))?;
    let mut lines = content.lines();
    let ids_line = lines
        .next()
        .ok_or_else(|| format!("reference file {path:?} is empty"))?;
    let token_ids: Vec<u32> = ids_line
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .map_err(|e| format!("reference file {path:?}: invalid token id {s:?}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    if token_ids.is_empty() {
        return Err(format!(
            "reference file {path:?}: line 1 must list at least one comma-separated token id"
        ));
    }
    let logit_checksum = match lines.next() {
        Some(line) if !line.trim().is_empty() => Some(line.trim().parse::<f64>().map_err(|e| {
            format!("reference file {path:?}: invalid logit_checksum {line:?}: {e}")
        })?),
        _ => None,
    };
    Ok(Reference {
        token_ids,
        logit_checksum,
    })
}

fn exit_usage_error(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

pub fn run(args: Vec<String>) {
    let mut gguf_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut max_tokens: usize = 1;
    let mut reference_path: Option<String> = None;
    let mut tolerance: f64 = 1e-2;
    let mut weights_flag: Option<String> = None;
    let mut json = false;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--max-tokens" => {
                let raw = args
                    .next()
                    .unwrap_or_else(|| exit_usage_error("--max-tokens requires a number"));
                max_tokens = raw.parse().unwrap_or_else(|_| {
                    exit_usage_error(&format!(
                        "--max-tokens must be a positive integer, got {raw:?}"
                    ))
                });
            }
            "--reference" => {
                reference_path = Some(
                    args.next()
                        .unwrap_or_else(|| exit_usage_error("--reference requires a file path")),
                )
            }
            "--weights" => {
                weights_flag = Some(
                    args.next()
                        .unwrap_or_else(|| exit_usage_error("--weights requires `f16` or `f32`")),
                )
            }
            "--tolerance" => {
                let raw = args
                    .next()
                    .unwrap_or_else(|| exit_usage_error("--tolerance requires a number"));
                tolerance = raw.parse().unwrap_or_else(|_| {
                    exit_usage_error(&format!("--tolerance must be a number, got {raw:?}"))
                });
            }
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            _ if prompt.is_none() => prompt = Some(arg),
            other => exit_usage_error(&format!("unexpected argument: {other}")),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| {
        exit_usage_error("usage: reflex check <path-to-gguf> <prompt> [--max-tokens N] [--reference <file>] [--tolerance F] [--weights f16|f32]")
    });
    let prompt = prompt.unwrap_or_else(|| exit_usage_error("a prompt is required"));
    if max_tokens == 0 {
        exit_usage_error("--max-tokens must be at least 1");
    }

    let file = GgufFile::open(&gguf_path)
        .unwrap_or_else(|e| exit_usage_error(&format!("failed to open {gguf_path}: {e}")));
    let device = diagnostics::init_device_with_diagnostics(0)
        .unwrap_or_else(|e| exit_usage_error(&e.to_string()));
    if let Ok(diag) = diagnostics::probe(&device) {
        eprintln!("{diag}");
    }
    // The reference methodology is defined on exact f32 weights, so `check`
    // falls back to f32 rather than the engine-wide f16 default.
    let model = crate::load_model(
        device,
        &file,
        weights_flag.as_deref(),
        WeightsDtype::F32,
        None,
    )
    .unwrap_or_else(|e| exit_usage_error(&format!("failed to load model: {e}")));
    let weights_dtype = model.weights_dtype().as_str();

    let mut first_logits: Vec<f32> = Vec::new();
    let (token_ids, _text) = model
        .generate(
            &prompt,
            max_tokens,
            None,
            &reflex_engine::sampling::SamplingParams::default(),
            |logits| first_logits = logits.to_vec(),
            |_id, _text| {},
        )
        .unwrap_or_else(|e| exit_usage_error(&format!("generate failed: {e}")));
    let token_texts = model.decode_tokens(&token_ids);
    crate::report_f16_activation_stats(&model);
    let vocab_size = first_logits.len();
    let logit_checksum: f64 = first_logits.iter().map(|&x| x as f64).sum();
    let top1_logit = first_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    if !json {
        let token_ids_str: Vec<String> = token_ids.iter().map(|t| t.to_string()).collect();
        println!(
            "REFLEX_CHECK token_ids=[{}] token_texts={:?} logit_checksum={logit_checksum:.6} top1_logit={top1_logit:.6} vocab_size={vocab_size} weights_dtype={weights_dtype}",
            token_ids_str.join(","),
            token_texts,
        );
    }

    let Some(reference_path) = reference_path else {
        if json {
            #[cfg(feature = "json-output")]
            reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::CheckJson {
                token_ids,
                token_texts,
                logit_checksum,
                top1_logit,
                vocab_size,
                weights_dtype,
                pass: None,
                fail_reason: None,
            });
            #[cfg(not(feature = "json-output"))]
            exit_usage_error(
                "--json requires this binary to be built with `cargo build --features json-output`",
            );
        }
        std::process::exit(0);
    };
    let reference = parse_reference(&reference_path).unwrap_or_else(|e| exit_usage_error(&e));

    let mut ok = true;
    // Only read back from the `#[cfg(feature = "json-output")]` branch at
    // the end of this function -- a build without that feature never reads
    // it, hence the `#[allow]` rather than a false "never read" warning.
    #[allow(unused_assignments)]
    let mut fail_reason: Option<String> = None;
    if reference.token_ids != token_ids {
        eprintln!(
            "REFLEX_CHECK_FAIL reason=token_id_mismatch expected={:?} got={token_ids:?}",
            reference.token_ids
        );
        fail_reason = Some(format!(
            "token_id_mismatch expected={:?} got={token_ids:?}",
            reference.token_ids
        ));
        ok = false;
    }
    if let Some(expected_checksum) = reference.logit_checksum {
        let denom = expected_checksum.abs().max(1.0);
        let relative_delta = (logit_checksum - expected_checksum).abs() / denom;
        if relative_delta > tolerance {
            eprintln!(
                "REFLEX_CHECK_FAIL reason=logit_checksum_mismatch expected={expected_checksum:.6} got={logit_checksum:.6} \
                 relative_delta={relative_delta:.6} tolerance={tolerance:.6}"
            );
            let msg = format!(
                "logit_checksum_mismatch expected={expected_checksum:.6} got={logit_checksum:.6} relative_delta={relative_delta:.6} tolerance={tolerance:.6}"
            );
            #[allow(unused_assignments)]
            {
                fail_reason = Some(match fail_reason {
                    Some(prev) => format!("{prev}; {msg}"),
                    None => msg,
                });
            }
            ok = false;
        }
    }

    if json {
        #[cfg(feature = "json-output")]
        reflex_engine::cli_output::print_json_line(&reflex_engine::cli_output::CheckJson {
            token_ids,
            token_texts,
            logit_checksum,
            top1_logit,
            vocab_size,
            weights_dtype,
            pass: Some(ok),
            fail_reason,
        });
        #[cfg(not(feature = "json-output"))]
        exit_usage_error(
            "--json requires this binary to be built with `cargo build --features json-output`",
        );
    } else if ok {
        println!("REFLEX_CHECK_PASS");
    }

    if ok {
        std::process::exit(0);
    }
    std::process::exit(1);
}
