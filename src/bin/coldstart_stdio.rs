//! Local, non-network IPC over stdio: reads one JSON request per stdin line,
//! processes it fully (via `coldstart_infer::ipc::handle_request`), and writes one
//! JSON response line to stdout before reading the next -- see `src/ipc.rs`'s
//! module doc comment for the protocol and CLAUDE.md/README.md's Non-goals for why
//! this exists instead of an HTTP server. Ideal for local subprocess orchestration
//! (MCP tool integrations, shell agents, other-language callers that don't want a
//! network socket).
//!
//! Usage: `coldstart_stdio <path-to-gguf> [--lora <adapter.gguf>]`
//! (requires `cargo build --features ipc`)

use coldstart_infer::gguf::GgufFile;
use coldstart_infer::model::Model;
use coldstart_infer::{diagnostics, ipc};

fn main() {
    let mut gguf_path: Option<String> = None;
    let mut lora_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
            _ if gguf_path.is_none() => gguf_path = Some(arg),
            other => panic!("unexpected argument: {other}"),
        }
    }
    let gguf_path = gguf_path.unwrap_or_else(|| panic!("usage: coldstart_stdio <path-to-gguf> [--lora <adapter.gguf>]"));

    let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
    let device = diagnostics::init_device_with_diagnostics(0).unwrap_or_else(|e| panic!("{e}"));
    if let Ok(diag) = diagnostics::probe(&device) {
        eprintln!("{diag}");
    }
    let mut model = Model::load(device, &file).expect("failed to load model");

    if let Some(lora_path) = &lora_path {
        let applied = model.apply_lora(std::path::Path::new(lora_path)).expect("failed to apply LoRA adapter");
        eprintln!("COLDSTART_STDIO_LORA_OK path={lora_path:?} tensors_applied={applied}");
    }

    eprintln!("COLDSTART_STDIO_READY path={gguf_path:?}");
    ipc::run_stdio_loop(&model).unwrap_or_else(|e| panic!("{e}"));
}
