//! Reflex: a GGUF-native, AOT-compiled CUDA inference engine optimized for
//! cold-start latency (process launch -> first token), not sustained server
//! throughput. Single binary, subcommand-dispatched -- see each subcommand
//! module's doc comment for its own usage and scope.
//!
//! Usage: `reflex <subcommand> [args...]`
//!
//! Subcommands:
//!   generate  - load a GGUF and generate tokens (the main entry point)
//!   system1   - single-pass, non-autoregressive candidate scoring
//!   smoke     - minimal AOT-kernel-launch smoke test, no GGUF needed
//!   bench     - warm-latency microbenchmark (forward pass, decode, system1)
//!   check     - byte-exact-vs-reference correctness check, CI-scriptable
//!   doctor    - one-shot GPU/CUDA/NVML health report, no GGUF needed
//!   stdio     - local JSON-line IPC over stdio (needs --features ipc)
//!   uds       - local JSON-line IPC over a Unix Domain Socket (needs
//!               --features ipc; Unix-only)

mod bench;
mod check;
mod doctor;
mod generate;
mod phase;
mod smoke;
mod system1;

#[cfg(feature = "ipc")]
mod stdio;
#[cfg(feature = "ipc")]
mod uds;

const USAGE: &str = "usage: reflex <subcommand> [args...]\n\
     \n\
     subcommands:\n\
     \x20 generate   load a GGUF and generate tokens\n\
     \x20 system1    single-pass candidate scoring\n\
     \x20 smoke      minimal AOT-kernel-launch smoke test\n\
     \x20 bench      warm-latency microbenchmark\n\
     \x20 check      byte-exact-vs-reference correctness check\n\
     \x20 doctor     one-shot GPU/CUDA/NVML health report\n\
     \x20 stdio      local JSON-line IPC over stdio (needs --features ipc)\n\
     \x20 uds        local JSON-line IPC over a Unix Domain Socket (needs --features ipc, Unix-only)\n\
     \n\
     run `reflex <subcommand>` with no further arguments to see that subcommand's own usage.";

/// Reports a runtime error as a single `error: <context>: <message>` line on
/// stderr and exits with status 1 -- for engine errors a user can act on (e.g.
/// the attention context-length limit, `reflex_engine::limits`), which
/// `.expect()` would otherwise dump as a panic with a Debug-quoted string.
pub(crate) fn fail(context: &str, err: impl std::fmt::Display) -> ! {
    eprintln!("error: {context}: {err}");
    reflex_engine::fast_exit(1)
}

/// Loads `file` the way every model-loading subcommand does: matrix weights in
/// the dtype `--weights` names (`weights_flag`), else `REFLEX_WEIGHTS`, else
/// `default_weights`; and, when `--lora` was given, with that adapter's target
/// weights kept `f32` until the caller's `apply_lora` merges them (see
/// `LoadOptions::lora_adapter`).
pub(crate) fn load_model(
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    file: &reflex_engine::gguf::GgufFile,
    weights_flag: Option<&str>,
    default_weights: reflex_engine::model::WeightsDtype,
    lora_path: Option<&str>,
) -> Result<reflex_engine::model::Model, reflex_engine::error::ReflexError> {
    let weights = reflex_engine::model::WeightsDtype::resolve_or(weights_flag, default_weights)?;
    let opts = reflex_engine::model::LoadOptions {
        weights,
        lora_adapter: lora_path.map(std::path::PathBuf::from),
    };
    reflex_engine::model::Model::load_with_options(device, file, &opts)
}

fn main() {
    let mut args = std::env::args();
    let _program = args.next();
    let subcommand = args.next().unwrap_or_else(|| {
        eprintln!("{USAGE}");
        std::process::exit(2);
    });
    let rest: Vec<String> = args.collect();

    match subcommand.as_str() {
        "generate" => generate::run(rest),
        "system1" => system1::run(rest),
        "smoke" => smoke::run(rest),
        "bench" => bench::run(rest),
        "check" => check::run(rest),
        "doctor" => doctor::run(rest),
        #[cfg(feature = "ipc")]
        "stdio" => stdio::run(rest),
        #[cfg(feature = "ipc")]
        "uds" => uds::run(rest),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
        }
        other => {
            eprintln!("unknown subcommand: {other:?}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}
