//! The engine's error type. Every fallible engine function returns
//! `Result<_, ReflexError>`, so callers (the CLI, the C FFI, the IPC protocol and
//! through it the OpenAI sidecar, the Python bindings) can branch on *what kind* of
//! failure happened instead of matching on message text.
//!
//! The message (`Display`) is exactly the text the engine produced before errors were
//! typed, so users and scripts that grep for it keep working. `Debug` prints that same
//! message the way a `String` would, so `.expect()` panics read exactly as before too.
//!
//! Implemented by hand rather than with `thiserror`: the core build's dependency list
//! is a documented contract (see README's "Build features"), and `thiserror` would add
//! a proc-macro toolchain to every build for a dozen trivial `Display` arms.

use std::fmt;

use crate::limits::CONTEXT_LENGTH_EXCEEDED_PREFIX;

/// Every error the engine returns. See the module docs.
#[derive(Clone, PartialEq, Eq)]
pub enum ReflexError {
    /// Malformed or truncated GGUF file: bad header, missing or mistyped metadata,
    /// tensor data that doesn't match its declared type or shape.
    Gguf(String),
    /// A well-formed model this engine doesn't support: an unknown architecture, or a
    /// known one with an unsupported configuration.
    UnsupportedArchitecture(String),
    /// A CUDA driver call (module load, kernel launch, copy) failed, other than running
    /// out of memory. Also covers a GPU this build's kernels can't run on.
    Cuda(String),
    /// A cuBLAS call failed, other than running out of memory.
    Cublas(String),
    /// A GPU (or pinned host) allocation failed.
    OutOfMemory(String),
    /// Prompt encoding or token decoding failed.
    Tokenizer(String),
    /// The request needs more sequence positions than the attention kernels support
    /// (see [`crate::limits`]). `requested` is `None` if the sum overflowed `usize`.
    ContextOverflow {
        requested: Option<usize>,
        imported: usize,
        prompt: usize,
        max_new: usize,
        max: usize,
    },
    /// An exported KV-cache file is malformed or doesn't match the model it's
    /// imported into.
    KvCache(String),
    /// A LoRA adapter is malformed or doesn't match the model it's applied to.
    Lora(String),
    /// A file couldn't be opened, created, read or written.
    Io(String),
    /// The caller passed an invalid argument (e.g. `max_new_tokens == 0`, no System1
    /// candidates, an out-of-range sampling parameter).
    InvalidInput(String),
    /// Anything not classified above, mostly internal invariant violations.
    Other(String),
}

/// Stable machine-readable error categories, shared by the C FFI
/// (`reflex_last_error_code`) and the IPC protocol (`error_kind`). Values never change
/// meaning; new categories are only ever appended.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflexErrorCode {
    /// No error.
    Ok = 0,
    Gguf = 1,
    UnsupportedArchitecture = 2,
    Cuda = 3,
    Cublas = 4,
    OutOfMemory = 5,
    Tokenizer = 6,
    ContextOverflow = 7,
    KvCache = 8,
    Lora = 9,
    Io = 10,
    InvalidInput = 11,
    Other = 12,
    /// The engine panicked; only the C FFI reports this (it catches the panic).
    Panic = 13,
}

impl ReflexErrorCode {
    /// The `snake_case` name used as IPC's `error_kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            ReflexErrorCode::Ok => "ok",
            ReflexErrorCode::Gguf => "gguf",
            ReflexErrorCode::UnsupportedArchitecture => "unsupported_architecture",
            ReflexErrorCode::Cuda => "cuda",
            ReflexErrorCode::Cublas => "cublas",
            ReflexErrorCode::OutOfMemory => "out_of_memory",
            ReflexErrorCode::Tokenizer => "tokenizer",
            ReflexErrorCode::ContextOverflow => "context_overflow",
            ReflexErrorCode::KvCache => "kv_cache",
            ReflexErrorCode::Lora => "lora",
            ReflexErrorCode::Io => "io",
            ReflexErrorCode::InvalidInput => "invalid_input",
            ReflexErrorCode::Other => "other",
            ReflexErrorCode::Panic => "panic",
        }
    }
}

impl ReflexError {
    pub fn code(&self) -> ReflexErrorCode {
        match self {
            ReflexError::Gguf(_) => ReflexErrorCode::Gguf,
            ReflexError::UnsupportedArchitecture(_) => ReflexErrorCode::UnsupportedArchitecture,
            ReflexError::Cuda(_) => ReflexErrorCode::Cuda,
            ReflexError::Cublas(_) => ReflexErrorCode::Cublas,
            ReflexError::OutOfMemory(_) => ReflexErrorCode::OutOfMemory,
            ReflexError::Tokenizer(_) => ReflexErrorCode::Tokenizer,
            ReflexError::ContextOverflow { .. } => ReflexErrorCode::ContextOverflow,
            ReflexError::KvCache(_) => ReflexErrorCode::KvCache,
            ReflexError::Lora(_) => ReflexErrorCode::Lora,
            ReflexError::Io(_) => ReflexErrorCode::Io,
            ReflexError::InvalidInput(_) => ReflexErrorCode::InvalidInput,
            ReflexError::Other(_) => ReflexErrorCode::Other,
        }
    }

    /// Shorthand for `self.code().as_str()`.
    pub fn kind(&self) -> &'static str {
        self.code().as_str()
    }

    /// The same kind of error with a new message, for adding context to an error from a
    /// call that already classified it, e.g.
    /// `.map_err(|e| e.rewrap(format!("load weight '{name}': {e}")))`. A
    /// `ContextOverflow` has a fixed, structured message and is returned unchanged.
    pub fn rewrap(&self, message: String) -> ReflexError {
        match self {
            ReflexError::Gguf(_) => ReflexError::Gguf(message),
            ReflexError::UnsupportedArchitecture(_) => {
                ReflexError::UnsupportedArchitecture(message)
            }
            ReflexError::Cuda(_) => ReflexError::Cuda(message),
            ReflexError::Cublas(_) => ReflexError::Cublas(message),
            ReflexError::OutOfMemory(_) => ReflexError::OutOfMemory(message),
            ReflexError::Tokenizer(_) => ReflexError::Tokenizer(message),
            ReflexError::ContextOverflow { .. } => self.clone(),
            ReflexError::KvCache(_) => ReflexError::KvCache(message),
            ReflexError::Lora(_) => ReflexError::Lora(message),
            ReflexError::Io(_) => ReflexError::Io(message),
            ReflexError::InvalidInput(_) => ReflexError::InvalidInput(message),
            ReflexError::Other(_) => ReflexError::Other(message),
        }
    }

    /// Classifies a failed CUDA driver or cuBLAS call: [`ReflexError::OutOfMemory`]
    /// when the call ran out of memory, otherwise [`ReflexError::Cuda`] or
    /// [`ReflexError::Cublas`]. `message` is the full, already-formatted text. Prefer
    /// the [`gpu_err!`](crate::gpu_err) macro, which formats it in place.
    pub fn gpu(failure: &impl GpuFailure, message: String) -> ReflexError {
        if failure.is_out_of_memory() {
            ReflexError::OutOfMemory(message)
        } else if failure.is_cublas() {
            ReflexError::Cublas(message)
        } else {
            ReflexError::Cuda(message)
        }
    }
}

impl fmt::Display for ReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReflexError::ContextOverflow {
                requested,
                imported,
                prompt,
                max_new,
                max,
            } => {
                let requested =
                    requested.map_or_else(|| "more than usize::MAX".to_string(), |t| t.to_string());
                write!(
                    f,
                    "{CONTEXT_LENGTH_EXCEEDED_PREFIX}: request needs {requested} positions \
                     (imported KV {imported} + prompt {prompt} + generation headroom {max_new}), \
                     but this engine's attention kernels support at most {max} \
                     positions per sequence. This is a Reflex engine limit, not the model's \
                     context length; shorten the prompt or lower max_tokens"
                )
            }
            ReflexError::Gguf(m)
            | ReflexError::UnsupportedArchitecture(m)
            | ReflexError::Cuda(m)
            | ReflexError::Cublas(m)
            | ReflexError::OutOfMemory(m)
            | ReflexError::Tokenizer(m)
            | ReflexError::KvCache(m)
            | ReflexError::Lora(m)
            | ReflexError::Io(m)
            | ReflexError::InvalidInput(m)
            | ReflexError::Other(m) => f.write_str(m),
        }
    }
}

/// Prints the message exactly as `Debug` on the old `String` error did (quoted and
/// escaped), so `.expect("...")` panic output is unchanged.
impl fmt::Debug for ReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_string(), f)
    }
}

impl std::error::Error for ReflexError {}

/// Unclassified string errors (e.g. from a helper that still builds its message with
/// `format!`) become [`ReflexError::Other`].
impl From<String> for ReflexError {
    fn from(message: String) -> Self {
        ReflexError::Other(message)
    }
}

impl From<&str> for ReflexError {
    fn from(message: &str) -> Self {
        ReflexError::Other(message.to_string())
    }
}

/// Lets code that still reports errors as `String` (the CLI binaries, IPC framing)
/// use `?` on engine calls.
impl From<ReflexError> for String {
    fn from(e: ReflexError) -> Self {
        e.to_string()
    }
}

/// A failed CUDA driver or cuBLAS call that [`ReflexError::gpu`] can classify.
pub trait GpuFailure: fmt::Display {
    fn is_out_of_memory(&self) -> bool;
    fn is_cublas(&self) -> bool {
        false
    }
}

impl GpuFailure for cudarc::driver::DriverError {
    fn is_out_of_memory(&self) -> bool {
        self.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
    }
}

impl GpuFailure for cudarc::cublas::result::CublasError {
    fn is_out_of_memory(&self) -> bool {
        self.0 == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED
    }
    fn is_cublas(&self) -> bool {
        true
    }
}

/// `ReflexError::$kind(format!(...))`: classify an error and format its message in
/// one step, with the same argument syntax as `format!`, e.g.
/// `Err(reflex_err!(Gguf, "tensor {name} not found"))`.
#[macro_export]
macro_rules! reflex_err {
    ($kind:ident, $($fmt:tt)*) => {
        $crate::error::ReflexError::$kind(format!($($fmt)*))
    };
}

/// [`ReflexError::gpu`] with the message formatted in place, e.g.
/// `.map_err(|e| gpu_err!(e, "attn launch: {e}"))`. `$err` must be a cudarc
/// `DriverError` or `CublasError`.
#[macro_export]
macro_rules! gpu_err {
    ($err:expr, $($fmt:tt)*) => {
        $crate::error::ReflexError::gpu(&$err, format!($($fmt)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::ATTN_MAX_POSITIONS;

    #[test]
    fn display_and_debug_match_the_old_string_error() {
        let e = ReflexError::Gguf("bad magic \"GGUX\"".to_string());
        assert_eq!(e.to_string(), "bad magic \"GGUX\"");
        // `.expect()` prints Debug; it must look like the old `String`'s Debug.
        assert_eq!(
            format!("{e:?}"),
            format!("{:?}", "bad magic \"GGUX\"".to_string())
        );
    }

    #[test]
    fn codes_are_stable() {
        // These values are part of the C ABI (reflex_last_error_code): never renumber.
        assert_eq!(ReflexErrorCode::Ok as i32, 0);
        assert_eq!(ReflexErrorCode::Gguf as i32, 1);
        assert_eq!(ReflexErrorCode::ContextOverflow as i32, 7);
        assert_eq!(ReflexErrorCode::OutOfMemory as i32, 5);
        assert_eq!(ReflexErrorCode::Other as i32, 12);
        assert_eq!(ReflexErrorCode::Panic as i32, 13);
    }

    #[test]
    fn kinds_round_trip_through_codes() {
        let e = ReflexError::InvalidInput("max_new_tokens must be at least 1".into());
        assert_eq!(e.code(), ReflexErrorCode::InvalidInput);
        assert_eq!(e.kind(), "invalid_input");
        assert_eq!(ReflexError::from("x".to_string()).kind(), "other");
    }

    #[test]
    fn context_overflow_message_matches_limits_contract() {
        let e = ReflexError::ContextOverflow {
            requested: Some(ATTN_MAX_POSITIONS + 1),
            imported: 0,
            prompt: ATTN_MAX_POSITIONS + 1,
            max_new: 0,
            max: ATTN_MAX_POSITIONS,
        };
        let m = e.to_string();
        assert!(m.starts_with(CONTEXT_LENGTH_EXCEEDED_PREFIX), "{m}");
        assert!(m.contains("not the model's context length"), "{m}");
        assert_eq!(e.kind(), "context_overflow");
    }
}
