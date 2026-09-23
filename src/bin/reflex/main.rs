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
//!   stdio     - local JSON-line IPC over stdio (needs --features ipc)
//!   uds       - local JSON-line IPC over a Unix Domain Socket (needs
//!               --features ipc; Unix-only)

mod bench;
mod check;
mod generate;
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
     \x20 stdio      local JSON-line IPC over stdio (needs --features ipc)\n\
     \x20 uds        local JSON-line IPC over a Unix Domain Socket (needs --features ipc, Unix-only)\n\
     \n\
     run `reflex <subcommand>` with no further arguments to see that subcommand's own usage.";

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
