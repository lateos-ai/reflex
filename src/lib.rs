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
