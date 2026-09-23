//! coldstart-infer: a GGUF-native inference engine optimized for cold-start
//! energy/latency (process launch -> first token), not sustained server throughput.
//! See README.md for the niche rationale and MVP ordering.

pub mod aot;
pub mod calibration;
pub mod dequant;
pub mod dequant_iq;
pub mod dequant_iq_tables;
pub mod diagnostics;
pub mod ffi;
pub mod gated_deltanet;
pub mod gguf;
#[cfg(feature = "download")]
pub mod hf;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod kv_io;
pub mod lora;
pub mod model;
pub mod moe;
#[cfg(feature = "python")]
pub mod python;
pub mod tokenizer;

/// Terminates the process immediately via the raw `_exit` syscall, skipping
/// libc's `atexit` handlers -- notably the CUDA driver's own registered
/// context-teardown hook. On GPU-virtualized rented instances (confirmed on
/// a ThunderCompute A6000: `smoke_coldstart`, which does essentially no GPU
/// work beyond `CudaDevice::new`, still took ~5.6s of pure post-result
/// teardown before this existed), that hook's handshake with the host's
/// virtualization proxy costs multiple seconds of wall-clock time that has
/// nothing to do with the process's actual work and does not scale with how
/// much was allocated -- consolidating allocations into fewer, larger
/// buffers was tried first and made no measurable difference, which is what
/// pointed at the `atexit` hook itself rather than allocation count (see
/// DECISIONS.md's "fast-exit after printing the benchmark result" entry for
/// the full investigation).
///
/// Every `*_coldstart` binary's job is finished by the time it calls this --
/// the OS reclaims all process resources (GPU context, file descriptors,
/// memory) on exit regardless of whether userspace tears them down first.
/// Flushes stdout/stderr first, since `_exit` skips the buffered-writer
/// flushes that would otherwise happen via `Drop` on a graceful return.
pub fn fast_exit(code: i32) -> ! {
    use std::io::Write;
    std::io::stdout().flush().ok();
    std::io::stderr().flush().ok();
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    unsafe { _exit(code) }
}
