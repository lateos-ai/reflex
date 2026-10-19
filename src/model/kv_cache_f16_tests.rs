//! The f16 working K/V cache (`REFLEX_KV_CACHE=f16`, `kv_cache.rs`) on the GPU:
//!
//! - **Accuracy**, measured like the f16 cache files of
//!   docs/design/agent-relay.md's phase F: after a ~700-token prompt, the next
//!   token's logits with an f16 cache against an f32 one (relative L2 and max
//!   |difference|), and how many of 32 greedy ids agree. f32 weights, so the cache
//!   rounding is the only difference. Each test prints an `F16_KV_CACHE` line and
//!   asserts the logits stay within f16 storage's expected size (rel L2 < 1e-2).
//! - **Pinned prefixes and cache files**: with an f16 cache, a pinned prefix
//!   downloads (by default) as an f16 file whose import resumes bit for bit like the
//!   pinned prefix itself, and an f32 file resumes into an f16 cache too.
//!
//! `#[ignore]`d: they need a GPU and a model. Dense: `REFLEX_TEST_GGUF` (the
//! nightly's Qwen3-0.6B; `SPM_GGUF` TinyLlama for the `spm_model` test); hybrid:
//! `REFLEX_TEST_GGUF`, the nightly's `HYBRID_GGUF` (Qwen3.5-0.8B); MoE and MLA: the
//! tiny fixtures.

use super::*;
use crate::kv_io::{self, HybridLayerCacheData, ImportedKv, KvDtype};
use crate::sampling::SamplingParams;
use cudarc::driver::CudaDevice;

const QWEN3MOE_FIXTURE: &str = "test-data/tiny-qwen3moe.gguf";
const MLA_FIXTURE: &str = "test-data/deepseek-tiny-mla.gguf";
const CONTINUATION: &str = " Summarize the rules above in one sentence:";
const N: usize = 32;

fn load_f32(path: &str) -> Model {
    let file = crate::gguf::GgufFile::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let opts = LoadOptions {
        weights: WeightsDtype::F32,
        lora_adapter: None,
    };
    Model::load_with_options(device, &file, &opts).expect("load model")
}

fn test_gguf(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a GGUF path"))
}

/// A ~700-token prompt: long enough that decode attention splits and the tiled
/// prefill kernel runs.
fn prompt() -> String {
    (1..=24)
        .map(|i| {
            format!(
                "Rule {i}: when you receive a note about step {i}, check its inputs, state your assumptions and keep the output format stable.\n"
            )
        })
        .collect()
}

/// `(max |a - b|, ||a - b|| / ||b||)`.
fn diff(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max = 0f32;
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        max = max.max((x - y).abs());
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (max, (num / den).sqrt())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The first token's logits and the greedy ids of an `N`-token generate with
/// cache type `dtype`.
fn generate_with(model: &Model, dtype: KvDtype, prompt: &str) -> (Vec<f32>, Vec<u32>) {
    model.set_kv_cache_dtype(dtype);
    model.check_kv_dtype_supported().expect("supported");
    let mut first = Vec::new();
    let (ids, _) = model
        .generate(
            prompt,
            N,
            None,
            &SamplingParams::default(),
            |l| first = l.to_vec(),
            |_, _| {},
        )
        .expect("generate");
    (first, ids)
}

/// Generates with an f32 cache, then an f16 one, and reports and bounds the
/// difference. The f32 run must not load the f16 module.
fn measure(model: &Model, tag: &str) {
    let prompt = prompt();
    let (base, base_ids) = generate_with(model, KvDtype::F32, &prompt);
    assert!(
        !model.kv_f16.loaded(),
        "{tag}: an f32 cache loaded the f16 module"
    );
    let (logits, ids) = generate_with(model, KvDtype::F16, &prompt);
    model.set_kv_cache_dtype(KvDtype::F32);
    assert!(model.kv_f16.loaded());
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "{tag}: non-finite logits"
    );
    let (max, rel) = diff(&logits, &base);
    let same = ids
        .iter()
        .zip(&base_ids)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "F16_KV_CACHE {tag}: next-token logits rel L2 {rel:.3e}, max |d| {max:.3e}; greedy ids identical for the first {same}/{} (f32 run {} tokens)",
        ids.len(),
        base_ids.len()
    );
    assert!(
        rel < 1e-2,
        "{tag}: an f16 cache moved logits by rel L2 {rel}"
    );
}

/// The element type of an imported cache's K/V values.
fn kv_values_dtype(kv: &ImportedKv) -> KvDtype {
    match kv {
        ImportedKv::Dense(c) => c.k_caches[0].dtype(),
        ImportedKv::Mla(c) => c.kv_caches[0].dtype(),
        ImportedKv::Hybrid(c) => c
            .layers
            .iter()
            .find_map(|l| match l {
                HybridLayerCacheData::Attn { k_cache, .. } => Some(k_cache.dtype()),
                HybridLayerCacheData::Gdn { .. } => None,
            })
            .expect("an attention layer"),
    }
}

/// Writes `kv` to a temporary cache file and reads it back. The import reads the
/// payload lazily, so the caller removes the file (the returned path) after use.
fn through_file(kv: &ImportedKv, tag: &str) -> (ImportedKv, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!(
        "reflex-f16-kv-{}-{}.cskv",
        tag.replace(' ', "-"),
        std::process::id()
    ));
    let p = path.to_str().expect("utf-8 temp path");
    kv_io::export_kv(p, kv).expect("export");
    (kv_io::import_kv(p).expect("import"), path)
}

fn resume(model: &Model, from: Resume<'_>) -> (Vec<f32>, Vec<u32>) {
    let mut logits = Vec::new();
    let (ids, _, _) = model
        .generate_resumed(
            CONTINUATION,
            N,
            from,
            &SamplingParams::default(),
            |l| logits = l.to_vec(),
            |_, _| {},
        )
        .expect("resume");
    (logits, ids)
}

/// With an f16 cache: the pinned prefix downloads as f16 values by default, and
/// resuming from that file is bit-identical to resuming from the pinned prefix (a
/// plain copy each way). An f32 file (from an f32-cache run) resumes into the f16
/// cache, rounded on the way in, close to the f16 run.
fn check_pinned_and_files(model: &Model, tag: &str) {
    let greedy = SamplingParams::default();
    let prompt = prompt();

    model.set_kv_cache_dtype(KvDtype::F32);
    let (_, _, pinned32, _) = model
        .generate_pin_prefix(&prompt, Resume::Fresh, &greedy, |_| {}, |_, _| {})
        .expect("pin prefix, f32 cache");
    let file32 = model.download_prefix(&pinned32).expect("download f32");
    assert_eq!(kv_values_dtype(&file32), KvDtype::F32);
    let (file32, path32) = through_file(&file32, &format!("{tag}-f32"));
    drop(pinned32);

    model.set_kv_cache_dtype(KvDtype::F16);
    let (_, _, pinned, _) = model
        .generate_pin_prefix(&prompt, Resume::Fresh, &greedy, |_| {}, |_, _| {})
        .expect("pin prefix, f16 cache");
    let file = model.download_prefix(&pinned).expect("download f16");
    assert_eq!(
        kv_values_dtype(&file),
        KvDtype::F16,
        "{tag}: an f16 cache's export defaults to f16"
    );
    let (file, path16) = through_file(&file, &format!("{tag}-f16"));

    let (pinned_logits, pinned_ids) = resume(model, Resume::Pinned(&pinned));
    let (file_logits, file_ids) = resume(model, Resume::File(&file));
    assert!(
        bits(&file_logits) == bits(&pinned_logits),
        "{tag}: resuming from the f16 file differs from the pinned prefix"
    );
    assert_eq!(file_ids, pinned_ids, "{tag}: greedy ids after the f16 file");

    let (f32_logits, f32_ids) = resume(model, Resume::File(&file32));
    model.set_kv_cache_dtype(KvDtype::F32);
    for p in [path32, path16] {
        let _ = std::fs::remove_file(p);
    }
    assert!(f32_logits.iter().all(|x| x.is_finite()));
    let (max, rel) = diff(&f32_logits, &pinned_logits);
    let same = f32_ids
        .iter()
        .zip(&pinned_ids)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "F16_KV_CACHE {tag}: f32 file into an f16 cache vs the f16 pinned prefix: rel L2 {rel:.3e}, max |d| {max:.3e}; greedy ids identical for the first {same}/{N}"
    );
    assert!(
        rel < 1e-2,
        "{tag}: f32 file into an f16 cache, rel L2 {rel}"
    );
}

#[test]
#[ignore]
fn f16_kv_cache_accuracy_dense() {
    let model = load_f32(&test_gguf("REFLEX_TEST_GGUF"));
    measure(&model, "dense");
    check_pinned_and_files(&model, "dense");
}

/// TinyLlama (SentencePiece, d64, GQA group 8): the nightly's `SPM_GGUF`.
#[test]
#[ignore]
fn f16_kv_cache_spm_model_accuracy() {
    let model = load_f32(&test_gguf("REFLEX_TEST_GGUF"));
    measure(&model, "spm");
}

/// Qwen3.5 (d256 gated attention next to GatedDeltaNet, whose state stays f32).
#[test]
#[ignore]
fn hybrid_f16_kv_cache_accuracy() {
    let model = load_f32(&test_gguf("REFLEX_TEST_GGUF"));
    measure(&model, "hybrid");
    check_pinned_and_files(&model, "hybrid");
}

#[test]
#[ignore]
fn qwen3moe_fixture_f16_kv_cache_accuracy() {
    let model = load_f32(QWEN3MOE_FIXTURE);
    measure(&model, "qwen3moe fixture");
}

/// MLA's compressed latent cache, through online attention with K and V as views
/// of one f16 buffer.
#[test]
#[ignore]
fn mla_fixture_f16_kv_cache_accuracy() {
    let model = load_f32(MLA_FIXTURE);
    measure(&model, "mla fixture");
    check_pinned_and_files(&model, "mla fixture");
}

/// Gemma 4 and Granite 4.0-H have no f16 cache path: the setting is refused.
#[test]
#[ignore]
fn f16_kv_cache_is_rejected_where_unsupported() {
    for path in [
        "test-data/tiny-gemma4.gguf",
        "test-data/tiny-granitehybrid.gguf",
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping {path}: not present");
            continue;
        }
        let model = load_f32(path);
        model.set_kv_cache_dtype(KvDtype::F16);
        let err = model.check_kv_dtype_supported().unwrap_err();
        assert!(err.to_string().contains("REFLEX_KV_CACHE"), "{path}: {err}");
    }
}
