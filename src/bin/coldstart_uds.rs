//! Local, non-network IPC over a Unix Domain Socket: binds `socket-path`, then
//! accepts and fully drains one connection (`coldstart_infer::ipc::run_request_loop`,
//! the same line-delimited JSON protocol `coldstart_stdio` uses) before accepting
//! the next -- never a thread pool, matching this engine's permanent `batch_size ==
//! 1`/no-concurrent-server constraint (see CLAUDE.md/README.md's Non-goals). A
//! second client connecting while the first is still being processed simply waits
//! in the OS accept queue.
//!
//! Usage: `coldstart_uds <path-to-gguf> <socket-path> [--lora <adapter.gguf>]`
//! (requires `cargo build --features ipc`)
//!
//! **Unix-only.** `std::os::unix::net::UnixListener` has no Windows counterpart in
//! the Rust standard library (Windows 10 1803+ does support `AF_UNIX` at the OS
//! level, but std never wired up a path to it there -- that would need the
//! third-party `uds_windows` crate, not attempted here). Use `coldstart_stdio` on
//! Windows instead.

#[cfg(unix)]
mod imp {
    use coldstart_infer::gguf::GgufFile;
    use coldstart_infer::model::Model;
    use coldstart_infer::{diagnostics, ipc};
    use std::io::{BufReader, BufWriter};
    use std::os::unix::net::UnixListener;

    pub fn run() {
        let mut gguf_path: Option<String> = None;
        let mut socket_path: Option<String> = None;
        let mut lora_path: Option<String> = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--lora" => lora_path = Some(args.next().expect("--lora requires a file path")),
                _ if gguf_path.is_none() => gguf_path = Some(arg),
                _ if socket_path.is_none() => socket_path = Some(arg),
                other => panic!("unexpected argument: {other}"),
            }
        }
        let gguf_path =
            gguf_path.unwrap_or_else(|| panic!("usage: coldstart_uds <path-to-gguf> <socket-path> [--lora <adapter.gguf>]"));
        let socket_path =
            socket_path.unwrap_or_else(|| panic!("usage: coldstart_uds <path-to-gguf> <socket-path> [--lora <adapter.gguf>]"));

        let file = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("failed to open {gguf_path}: {e}"));
        let device = diagnostics::init_device_with_diagnostics(0).unwrap_or_else(|e| panic!("{e}"));
        if let Ok(diag) = diagnostics::probe(&device) {
            eprintln!("{diag}");
        }
        let mut model = Model::load(device, &file).expect("failed to load model");

        if let Some(lora_path) = &lora_path {
            let applied = model.apply_lora(std::path::Path::new(lora_path)).expect("failed to apply LoRA adapter");
            eprintln!("COLDSTART_UDS_LORA_OK path={lora_path:?} tensors_applied={applied}");
        }

        // A prior run that crashed/was killed can leave a stale socket file behind
        // -- bind() fails with AddrInUse against an existing path even if nothing
        // is listening on it, so clear it first (matches common Unix daemon
        // convention). Only removes a file that's actually there; ignores
        // "not found".
        if let Err(e) = std::fs::remove_file(&socket_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                panic!("failed to remove stale socket at {socket_path:?}: {e}");
            }
        }
        let listener = UnixListener::bind(&socket_path).unwrap_or_else(|e| panic!("failed to bind UDS at {socket_path:?}: {e}"));
        eprintln!("COLDSTART_UDS_READY path={gguf_path:?} socket={socket_path:?}");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let reader = match stream.try_clone() {
                        Ok(s) => BufReader::new(s),
                        Err(e) => {
                            eprintln!("COLDSTART_UDS_CONNECTION_ERROR error=\"failed to clone stream: {e}\"");
                            continue;
                        }
                    };
                    let writer = BufWriter::new(stream);
                    if let Err(e) = ipc::run_request_loop(&model, reader, writer) {
                        eprintln!("COLDSTART_UDS_CONNECTION_ERROR error={e:?}");
                    }
                }
                Err(e) => eprintln!("COLDSTART_UDS_ACCEPT_ERROR error={e}"),
            }
        }
    }
}

#[cfg(unix)]
fn main() {
    imp::run();
}

#[cfg(not(unix))]
fn main() {
    eprintln!(
        "coldstart_uds is Unix-only (std::os::unix::net::UnixListener has no Windows counterpart in this build) -- \
         use coldstart_stdio instead."
    );
    std::process::exit(1);
}
