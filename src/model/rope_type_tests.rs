use super::*;

/// Locks in the per-architecture RoPE-convention mapping so a future
/// architecture addition can't silently inherit the wrong convention.
/// `llama`/`mistral`/`mixtral` are the consecutive-pair NORM case
/// (`llama.cpp`'s `llama_model_rope_type`); Qwen3 and the pre-existing
/// MoE fixtures use the half-split NEOX case.
#[test]
fn rope_type_mapping_matches_llama_cpp() {
    for arch in ["llama", "mistral", "mixtral"] {
        assert_eq!(rope_type_for(arch), RopeType::Norm, "{arch}");
    }
    // `kolibri1`: added to llama.cpp's NEOX list by the community patch, and
    // vLLM's default in Aleph Alpha's official plugin (docs/design/kolibri.md).
    for arch in ["qwen3", "qwen3moe", "kolibri1", "some_other_moe"] {
        assert_eq!(rope_type_for(arch), RopeType::Neox, "{arch}");
    }
}
