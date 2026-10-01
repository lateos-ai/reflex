use super::*;
use crate::gguf::GgufFile;

/// `test-data/tiny-qwen3moe.gguf`: a synthetic `qwen3moe` GGUF, hand-built the
/// same way as `test-data/deepseek-tiny-mla.gguf` (random-weight HF checkpoint
/// run through llama.cpp's own real, unmodified `convert_hf_to_gguf.py` --
/// source archived as `test-data/tiny-qwen3moe-src.tar.gz`). Unlike
/// `Tiny-Moe.Q4_K_M.gguf` (the only other local MoE fixture, a Mixtral-style
/// file with `expert_used_count == expert_count`, which can't prove top-k
/// routing excludes anything -- see docs/DEVELOPMENT.md's "Known test-fixture
/// limitations"), this fixture sets `num_experts_per_tok=2 < num_experts=8` and
/// includes real Qwen3 QK-Norm tensors (`attn_q_norm`/`attn_k_norm`), closing
/// both gaps that entry names. Also has a `gpt2`-style tokenizer (reused
/// verbatim from `deepseek-tiny-mla`'s), enabling text-level byte-exact resume
/// verification the way `Tiny-Moe`'s SentencePiece tokenizer could not.
const QWEN3MOE_FIXTURE: &str = "test-data/tiny-qwen3moe.gguf";

/// Host-only (no GPU/CUDA device needed -- `GgufFile::open`/`parse_model_config`
/// are pure mmap/metadata parsing): confirms the fixture actually has the
/// properties it was built for before any GPU-hardware test relies on them.
/// Still `#[ignore]`d like every other local-fixture test in this file, since
/// `test-data/*.gguf` is gitignored and won't exist on a fresh checkout/CI
/// runner -- run with `cargo test -- --ignored qwen3moe_fixture_has_excluding_topk_and_qk_norm`.
#[test]
#[ignore]
fn qwen3moe_fixture_has_excluding_topk_and_qk_norm() {
    let file = GgufFile::open(QWEN3MOE_FIXTURE).expect("failed to open qwen3moe fixture");

    let architecture = file
        .metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("");
    assert_eq!(
        architecture, "qwen3moe",
        "fixture should report the real qwen3moe architecture string"
    );

    let (_, block_count, moe) =
        parse_model_config(&file).expect("parse_model_config failed on qwen3moe fixture");
    let moe = moe.expect("fixture should be detected as an MoE architecture");
    assert_eq!(moe.expert_count, 8);
    assert_eq!(moe.expert_used_count, 2);
    assert!(
        moe.expert_used_count < moe.expert_count,
        "this fixture's whole purpose is expert_used_count < expert_count, unlike Tiny-Moe.Q4_K_M.gguf"
    );
    assert_eq!(block_count, 2);

    for i in 0..block_count {
        assert!(
            file.tensor_info(&format!("blk.{i}.attn_q_norm.weight"))
                .is_some(),
            "layer {i} missing attn_q_norm.weight (QK-Norm) -- the other gap this fixture closes"
        );
        assert!(
            file.tensor_info(&format!("blk.{i}.attn_k_norm.weight"))
                .is_some(),
            "layer {i} missing attn_k_norm.weight (QK-Norm) -- the other gap this fixture closes"
        );
    }
}

/// Real-hardware follow-up to the host-only test above: loads the fixture on a
/// real CUDA device and runs actual generation, proving the fixture isn't just
/// metadata-valid but usable end-to-end (MoE routing/QK-Norm exercised for
/// real, not just declared in metadata). `#[ignore]`d for both reasons every
/// other local-fixture test is (file availability, real GPU needed) -- run
/// with `cargo test --release -- --ignored qwen3moe_fixture_generates_without_error`.
#[test]
#[ignore]
fn qwen3moe_fixture_generates_without_error() {
    let file = GgufFile::open(QWEN3MOE_FIXTURE).expect("failed to open qwen3moe fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load qwen3moe fixture");
    let (tokens, _text) = model
        .generate(
            "Once upon a time",
            5,
            None,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )
        .expect("generate failed");
    assert!(!tokens.is_empty(), "expected at least one generated token");
}

/// `test-data/tiny-qwen35moe.gguf`: the public random-weight
/// `yujiepan/qwen3.5-moe-tiny-random` HF checkpoint (authentic
/// `Qwen3_5MoeForConditionalGeneration` tensor names/shapes and the real
/// Qwen3.5 tokenizer) run through llama.cpp's own real, unmodified
/// `convert_hf_to_gguf.py --outtype f32 --no-mtp`. Source + provenance
/// archived as `test-data/tiny-qwen35moe-src.tar.gz`.
const QWEN35MOE_FIXTURE: &str = "test-data/tiny-qwen35moe.gguf";

/// Host-only, like `qwen3moe_fixture_has_excluding_topk_and_qk_norm`:
/// confirms the fixture really exercises what `qwen35moe` support needs
/// -- top-k that excludes experts, both hybrid mixer kinds, the full
/// routed + gated-shared-expert tensor set on every layer, and no MTP
/// block -- before any GPU test relies on it. Run with
/// `cargo test -- --ignored qwen35moe_fixture_has_routed_and_shared_experts`.
#[test]
#[ignore]
fn qwen35moe_fixture_has_routed_and_shared_experts() {
    let file = GgufFile::open(QWEN35MOE_FIXTURE).expect("failed to open qwen35moe fixture");
    let architecture = file
        .metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("");
    assert_eq!(architecture, "qwen35moe");

    let block_count = u64_meta(&file, "qwen35moe.block_count").expect("block_count") as usize;
    assert_eq!(
        u64_meta(&file, "qwen35moe.nextn_predict_layers").unwrap_or(0),
        0,
        "fixture must be converted with --no-mtp"
    );
    let expert_count = u64_meta(&file, "qwen35moe.expert_count").expect("expert_count");
    let moe = parse_hybrid_moe_config(&file, architecture)
        .expect("parse_hybrid_moe_config failed on qwen35moe fixture");
    assert!(
        (moe.expert_used_count as u64) < expert_count,
        "top-k must exclude some experts"
    );

    let is_gdn = parse_hybrid_layer_kinds(&file, architecture, block_count)
        .expect("parse_hybrid_layer_kinds failed");
    assert!(is_gdn.iter().any(|&g| g), "no Gated DeltaNet layer");
    assert!(is_gdn.iter().any(|&g| !g), "no Gated Attention layer");

    for i in 0..block_count {
        for t in [
            "ffn_gate_inp",
            "ffn_gate_exps",
            "ffn_up_exps",
            "ffn_down_exps",
            "ffn_gate_inp_shexp",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            assert!(
                file.tensor_info(&format!("blk.{i}.{t}.weight")).is_some(),
                "layer {i} missing {t}.weight"
            );
        }
        assert!(
            file.tensor_info(&format!("blk.{i}.ffn_gate.weight"))
                .is_none(),
            "layer {i} unexpectedly has a dense ffn_gate"
        );
    }
}

/// Real-GPU end-to-end load + generation of the `qwen35moe` fixture. Run
/// with `cargo test --release -- --ignored qwen35moe_fixture_generates_without_error`.
#[test]
#[ignore]
fn qwen35moe_fixture_generates_without_error() {
    let file = GgufFile::open(QWEN35MOE_FIXTURE).expect("failed to open qwen35moe fixture");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let model = Model::load(device, &file).expect("failed to load qwen35moe fixture");
    assert!(
        model.hybrid.as_ref().is_some_and(|h| h.moe.is_some()),
        "qwen35moe fixture should load as a hybrid MoE model"
    );
    let (tokens, _text) = model
        .generate(
            "Once upon a time",
            5,
            None,
            &SamplingParams::default(),
            |_logits| {},
            |_id, _text| {},
        )
        .expect("generate failed");
    assert!(!tokens.is_empty(), "expected at least one generated token");
}
