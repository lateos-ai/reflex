//! coldstart-infer: a GGUF-native inference engine optimized for cold-start
//! energy/latency (process launch -> first token), not sustained server throughput.
//! See README.md for the niche rationale and MVP ordering.

pub mod aot;
pub mod dequant;
pub mod dequant_iq;
pub mod dequant_iq_tables;
pub mod gguf;
pub mod model;
pub mod tokenizer;
