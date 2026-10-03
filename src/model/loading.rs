//! Weight loading: the pipelined host-to-device upload, on-device dequant, and the
//! lazy token embedding.

use super::*;
use crate::error::ReflexError;

/// Element type matrix weights are stored in on the device (`--weights`,
/// `REFLEX_WEIGHTS`). Only matrix weights (see [`is_matrix_weight`]) follow it;
/// norms, biases, routers and every activation and KV-cache buffer are `f32`
/// in both modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsDtype {
    /// Half the VRAM and half the per-token weight traffic of `F32`. Matmul
    /// kernels read f16 weights and accumulate in f32.
    F16,
    /// The exact reference mode: every weight is the f32 the dequant produced,
    /// bit for bit what `reflex check`'s llama.cpp comparison was built on.
    F32,
}

impl WeightsDtype {
    /// What a model loads with when neither `--weights` nor `REFLEX_WEIGHTS`
    /// says otherwise.
    pub const DEFAULT: WeightsDtype = WeightsDtype::F32;

    pub fn as_str(self) -> &'static str {
        match self {
            WeightsDtype::F16 => "f16",
            WeightsDtype::F32 => "f32",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ReflexError> {
        match s {
            "f16" => Ok(WeightsDtype::F16),
            "f32" => Ok(WeightsDtype::F32),
            other => Err(crate::reflex_err!(
                InvalidInput,
                "weights dtype must be `f16` or `f32`, got {other:?}"
            )),
        }
    }

    /// `flag` (a subcommand's `--weights` value) if given, else
    /// `REFLEX_WEIGHTS` if set and non-empty, else [`Self::DEFAULT`].
    pub fn resolve(flag: Option<&str>) -> Result<Self, ReflexError> {
        Self::resolve_or(flag, Self::DEFAULT)
    }

    /// [`Self::resolve`] with a caller-chosen fallback in place of
    /// [`Self::DEFAULT`] (`reflex check` falls back to `F32`).
    pub fn resolve_or(flag: Option<&str>, default: Self) -> Result<Self, ReflexError> {
        if let Some(f) = flag {
            return Self::parse(f);
        }
        match std::env::var("REFLEX_WEIGHTS") {
            Ok(v) if !v.is_empty() => {
                Self::parse(&v).map_err(|e| e.rewrap(format!("REFLEX_WEIGHTS: {e}")))
            }
            _ => Ok(default),
        }
    }
}

impl Default for WeightsDtype {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl std::fmt::Display for WeightsDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options for [`Model::load_with_options`].
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub weights: WeightsDtype,
    /// The LoRA adapter the caller will pass to [`Model::apply_lora`] right
    /// after loading, if any. Only its tensor names are read here: the base
    /// weights it targets are loaded as `f32`, so `apply_lora` adds the delta
    /// to the exact dequantized values and rounds to f16 once, afterwards.
    pub lora_adapter: Option<std::path::PathBuf>,
}

/// Which element type each loaded tensor gets: [`LoadOptions`] resolved
/// against the adapter's target names, threaded through the `load_*` paths.
pub(super) struct WeightPolicy {
    pub(super) matrix_dtype: WeightsDtype,
    /// Matrix weights a LoRA adapter will be merged into: loaded `f32`
    /// regardless of `matrix_dtype` (see [`LoadOptions::lora_adapter`]).
    pub(super) keep_f32: std::collections::HashSet<String>,
}

impl WeightPolicy {
    pub(super) fn from_options(opts: &LoadOptions) -> Result<Self, ReflexError> {
        let keep_f32 = match &opts.lora_adapter {
            Some(path) => lora::target_names(path)?.into_iter().collect(),
            None => std::collections::HashSet::new(),
        };
        Ok(Self {
            matrix_dtype: opts.weights,
            keep_f32,
        })
    }

    /// `Some(dtype)` for a matrix weight, `None` for a tensor that always
    /// stays `f32` (see [`is_matrix_weight`]).
    fn matrix_dtype_for(&self, name: &str, shape: &[u64]) -> Option<WeightsDtype> {
        if !is_matrix_weight(name, shape) {
            return None;
        }
        if self.keep_f32.contains(name) {
            return Some(WeightsDtype::F32);
        }
        Some(self.matrix_dtype)
    }
}

/// A weight tensor's device buffer. `F32` is every tensor in `--weights f32`
/// mode and the non-matrix tensors in both modes; `F16` is a matrix weight in
/// `--weights f16` mode. A future block-quantized variant (raw GGUF blocks
/// read by quantized matmul kernels) slots in here as a third arm; every
/// matmul wrapper in `kernels.rs` already dispatches on this enum.
pub(super) enum WeightData {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
}

/// A zero-copy view of part of a [`Weight`] (one MoE expert's or one MLA
/// head's slice of a stacked 3-D tensor), with the same element type.
pub(super) enum WeightView<'a> {
    F32(CudaView<'a, f32>),
    F16(CudaView<'a, half::f16>),
}

/// A weight tensor, dequantized once at load time and uploaded to device
/// memory immediately after (see `Model::load`'s `load_weight`) so no
/// forward-pass call re-uploads it -- kernels below read `self.data` (or a
/// zero-copy [`WeightView`] slice of it, for per-expert MoE tensors)
/// directly. Shape is the original GGUF shape (`[in_features,
/// out_features]` for a 2-D `nn.Linear`-style weight, `[hidden_size]` for a
/// norm weight).
pub(super) struct Weight {
    pub(super) data: WeightData,
    pub(super) shape: Vec<u64>,
}

impl Weight {
    /// The `f32` buffer of a tensor that is always `f32` (a norm, bias or
    /// state-space vector, see [`is_matrix_weight`]). Errors on an f16 matrix
    /// weight instead of handing an f32 kernel the wrong element type.
    pub(super) fn f32(&self) -> Result<&CudaSlice<f32>, ReflexError> {
        match &self.data {
            WeightData::F32(s) => Ok(s),
            WeightData::F16(_) => Err(crate::reflex_err!(
                Other,
                "weight of shape {:?} is stored as f16, but this op reads it as f32",
                self.shape
            )),
        }
    }

    /// Elements `range` of this weight's flat buffer, without copying.
    pub(super) fn view(&self, range: std::ops::Range<usize>) -> WeightView<'_> {
        match &self.data {
            WeightData::F32(s) => WeightView::F32(s.slice(range)),
            WeightData::F16(s) => WeightView::F16(s.slice(range)),
        }
    }
}

/// `token_embd`, dequantized lazily one row at a time instead of eagerly
/// decoding the whole `[vocab_size, hidden_size]` matrix at load time.
/// `forward_prompt`'s embedding lookup is a host-side gather that reads only
/// a handful of rows per token in the prompt/generation loop (`batch_size`
/// is always 1, see docs/DEVELOPMENT.md's Non-goals), yet the eager version decoded
/// every row unconditionally -- ~548ms, ~63% of `model_load_ms` for
/// Qwen3-0.6B-Q4_K_M on a T4, the single largest piece of cold-start time in
/// the whole engine (found while profiling a smaller optimization). Same
/// lazy-materialization pattern [`LmHead::TiedLazy`] already proved out for
/// `lm_head` -- not new math, no numerics change: every row this decodes is
/// byte-identical to what eagerly dequantizing the whole tensor would have
/// produced at that row's offset, since GGUF block dequantization has no
/// cross-block state (decoding a row's blocks in isolation is the same
/// computation as decoding them as part of the full tensor).
///
/// `raw` points straight into the GGUF mmap, which it keeps alive
/// ([`crate::gguf::SharedBytes`]). It used to be an owned copy, and that copy
/// was the largest single part of model load: ~103 ms of ~231 ms for
/// Qwen3-0.6B's 127.6 MB Q6_K `token_embd` on a T4, mostly faulting in the
/// freshly allocated `Vec` (reading the same file pages later, on a tied LM
/// head's first full-vocab use, costs only ~7 ms). Pages are now read only
/// when a row (or the full table) is actually used. The model keeps the file
/// mapped for its lifetime, as llama.cpp does by default.
pub(super) struct LazyTokenEmbedding {
    pub(super) raw: crate::gguf::SharedBytes,
    pub(super) ggml_type: GgmlType,
    pub(super) hidden_size: usize,
    pub(super) vocab_size: usize,
    /// Bytes one row (`hidden_size` elements) occupies in `raw` -- an exact
    /// number of on-disk blocks, by ggml's own invariant that a quantized
    /// tensor's row width is always a multiple of its block size.
    pub(super) row_bytes: usize,
    /// Rows dequantized so far, keyed by token id -- a repeated token (a
    /// longer prompt, or the same token recurring across a generation loop)
    /// reuses the cached decode instead of redoing it. `RefCell`, not a
    /// `Mutex`: this project never runs more than one request at a time
    /// (`batch_size` is a permanent constraint, see docs/DEVELOPMENT.md's Non-goals),
    /// so there is never a concurrent borrower -- the same reasoning
    /// `LmHead::TiedLazy`'s `OnceLock::get`/`set` already relies on.
    pub(super) cache: RefCell<HashMap<u32, Vec<f32>>>,
}

impl LazyTokenEmbedding {
    /// `shape` is `token_embd.weight`'s GGUF shape, `[hidden_size,
    /// vocab_size]` (row-major `(vocab_size, hidden_size)` flat data).
    pub(super) fn new(
        ggml_type: GgmlType,
        raw: crate::gguf::SharedBytes,
        shape: &[u64],
    ) -> Result<Self, ReflexError> {
        let hidden_size = shape[0] as usize;
        let vocab_size = shape[1] as usize;
        let (block_size, block_bytes) = crate::gguf::ggml_type_block_dims(ggml_type)?;
        if !(hidden_size as u64).is_multiple_of(block_size) {
            return Err(crate::reflex_err!(Gguf,
                "token_embd row width {hidden_size} is not a multiple of {ggml_type:?}'s block size {block_size}"
            ));
        }
        let row_bytes = (hidden_size as u64 / block_size * block_bytes) as usize;
        Ok(Self {
            raw,
            ggml_type,
            hidden_size,
            vocab_size,
            row_bytes,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Test-only accessor (mirrors `Self::lm_head_argmax`'s `#[cfg(test)]`
    /// convention) -- no non-test caller needs the vocab size on its own,
    /// only `Self::row` internally.
    #[cfg(test)]
    pub(super) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Dequantizes (or returns the already-cached decode of) row `token_id`
    /// -- `hidden_size` contiguous `f32`s.
    pub(super) fn row(&self, token_id: u32) -> Result<std::cell::Ref<'_, [f32]>, ReflexError> {
        if token_id as usize >= self.vocab_size {
            return Err(crate::reflex_err!(
                Other,
                "token id {token_id} out of range (vocab_size={})",
                self.vocab_size
            ));
        }
        if !self.cache.borrow().contains_key(&token_id) {
            let start = token_id as usize * self.row_bytes;
            let end = start + self.row_bytes;
            let block = self.raw.get(start..end).ok_or_else(|| {
                crate::reflex_err!(Gguf,
                    "token_embd row {token_id} byte range {start}..{end} exceeds raw buffer length {}",
                    self.raw.len()
                )
            })?;
            let decoded = dequant::dequantize(self.ggml_type, block, self.hidden_size as u64)?;
            self.cache.borrow_mut().insert(token_id, decoded);
        }
        Ok(std::cell::Ref::map(self.cache.borrow(), |m| {
            m.get(&token_id).expect("just inserted above").as_slice()
        }))
    }
}

/// The LM head, dense/MoE `Model::load` only (`load_hybrid`/`load_mla` still
/// always use `Resident` -- this laziness is scoped to the one path
/// `system1_evaluate` actually needs it for, not a general Model change).
///
/// A tied-embedding model (no separate `output.weight` tensor) used to
/// eagerly re-upload `token_embd`'s already-dequantized host bytes as a
/// second, full `[hidden_size, vocab_size]` device copy at load time --
/// unconditionally, even for a `reflex system1` run that only ever gathers a
/// handful of candidate rows via `Self::gemv_gather`. For `Qwen3-0.6B`
/// (vocab 151936 x hidden 1024 x 4 bytes), that's ~594 MiB uploaded and
/// resident for nothing. `TiedLazy` defers that upload until something
/// actually needs the *full* matrix (`Self::lm_head_resident`, used by
/// `Self::lm_head_logits`'s full-vocab GEMV) -- `system1_evaluate`'s gather
/// path (`Self::gemv_gather_lm_head`) instead uploads only the requested
/// rows straight from the host-resident `token_embd`, and never forces the
/// full upload at all.
pub(super) enum LmHead {
    /// A real separate `output.weight` tensor, or a tied model whose full
    /// matrix some earlier full-vocab call already forced resident.
    Resident(Weight),
    /// Tied to `token_embd`, not yet forced fully resident. `shape` is
    /// `output.weight`'s GGUF shape (`[hidden_size, vocab_size]`), needed
    /// before the upload happens; `token_embd`'s host bytes (row-major
    /// `(vocab_size, hidden_size)`, same layout `output.weight` would have)
    /// are the source of truth until `Self::lm_head_resident` is called.
    TiedLazy {
        shape: Vec<u64>,
        cell: std::sync::OnceLock<Weight>,
    },
}

/// GGUF super-block sizes for the block types dequantized on-device
/// (`src/kernels_cuda/dequant.cu`, Phase 2 round 3 for Q4_K/Q6_K, extended to
/// Q5_K post-MVP) -- must match `dequant.rs`'s `QK_K` and the block-byte-size
/// table `gguf.rs::ggml_type_size_bytes` computes independently for the same
/// types.
/// CUDA warp width -- `gemv_kernel`/`gemv_gather_kernel`'s warp-per-row
/// launch geometry (`Model::gemv_raw`/`Model::gemv_gather`) assigns exactly
/// one warp to each output row, so the block size passed to `LaunchConfig`
/// must always be a multiple of this.
pub(super) const WARP_SIZE: u32 = 32;

pub(super) const QK_K: usize = 256;

pub(super) const QK_LEGACY: usize = 32;

pub(super) const Q4K_BLOCK_BYTES: usize = 144;

pub(super) const Q5K_BLOCK_BYTES: usize = 176;

pub(super) const Q6K_BLOCK_BYTES: usize = 210;

pub(super) const Q4_0_BLOCK_BYTES: usize = 18;

pub(super) const Q4_1_BLOCK_BYTES: usize = 20;

pub(super) const Q5_0_BLOCK_BYTES: usize = 22;

pub(super) const Q5_1_BLOCK_BYTES: usize = 24;

pub(super) const Q8_0_BLOCK_BYTES: usize = 34;

pub(super) const Q8_1_BLOCK_BYTES: usize = 36;

pub(super) const Q2K_BLOCK_BYTES: usize = 84;

pub(super) const Q3K_BLOCK_BYTES: usize = 110;

pub(super) const Q8K_BLOCK_BYTES: usize = 292;

pub(super) const IQ2XXS_BLOCK_BYTES: usize = 66;

pub(super) const IQ2XS_BLOCK_BYTES: usize = 74;

pub(super) const IQ2S_BLOCK_BYTES: usize = 82;

pub(super) const IQ3XXS_BLOCK_BYTES: usize = 98;

pub(super) const IQ3S_BLOCK_BYTES: usize = 110;

pub(super) const IQ1S_BLOCK_BYTES: usize = 50;

pub(super) const IQ1M_BLOCK_BYTES: usize = 56;

pub(super) const IQ4XS_BLOCK_BYTES: usize = 136;

/// One output type's set of on-device dequant kernels
/// (`src/kernels_cuda/dequant.cu`), one per block-quantized GGUF format.
pub(super) struct FormatKernels {
    pub(super) q4k: AotKernel,
    pub(super) q5k: AotKernel,
    pub(super) q6k: AotKernel,
    pub(super) q4_0: AotKernel,
    pub(super) q4_1: AotKernel,
    pub(super) q5_0: AotKernel,
    pub(super) q5_1: AotKernel,
    pub(super) q8_0: AotKernel,
    pub(super) q8_1: AotKernel,
    pub(super) q2k: AotKernel,
    pub(super) q3k: AotKernel,
    pub(super) q8k: AotKernel,
    pub(super) iq2xxs: AotKernel,
    pub(super) iq2xs: AotKernel,
    pub(super) iq2s: AotKernel,
    pub(super) iq3xxs: AotKernel,
    pub(super) iq3s: AotKernel,
    pub(super) iq1s: AotKernel,
    pub(super) iq1m: AotKernel,
    pub(super) iq4xs: AotKernel,
}

/// The formats in [`FormatKernels`]' field order, as their `dequant.cu`
/// kernel-name stems.
const DEQUANT_FORMATS: [&str; 20] = [
    "q4k", "q5k", "q6k", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "q2k", "q3k", "q8k",
    "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq1s", "iq1m", "iq4xs",
];

impl FormatKernels {
    fn from_iter(fns: &mut impl Iterator<Item = AotKernel>) -> Result<Self, ReflexError> {
        let mut next = || {
            fns.next()
                .ok_or_else(|| ReflexError::Other("missing dequant kernel".to_string()))
        };
        Ok(Self {
            q4k: next()?,
            q5k: next()?,
            q6k: next()?,
            q4_0: next()?,
            q4_1: next()?,
            q5_0: next()?,
            q5_1: next()?,
            q8_0: next()?,
            q8_1: next()?,
            q2k: next()?,
            q3k: next()?,
            q8k: next()?,
            iq2xxs: next()?,
            iq2xs: next()?,
            iq2s: next()?,
            iq3xxs: next()?,
            iq3s: next()?,
            iq1s: next()?,
            iq1m: next()?,
            iq4xs: next()?,
        })
    }

    /// The kernel for `ggml_type` with its block size in bytes and elements,
    /// or `None` for a type with no on-device kernel (F32/F16/BF16, which go
    /// through the host fallback).
    pub(super) fn for_type(&self, ggml_type: GgmlType) -> Option<(&AotKernel, usize, usize)> {
        Some(match ggml_type {
            GgmlType::Q4K => (&self.q4k, Q4K_BLOCK_BYTES, QK_K),
            GgmlType::Q5K => (&self.q5k, Q5K_BLOCK_BYTES, QK_K),
            GgmlType::Q6K => (&self.q6k, Q6K_BLOCK_BYTES, QK_K),
            GgmlType::Q4_0 => (&self.q4_0, Q4_0_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q4_1 => (&self.q4_1, Q4_1_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q5_0 => (&self.q5_0, Q5_0_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q5_1 => (&self.q5_1, Q5_1_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q8_0 => (&self.q8_0, Q8_0_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q8_1 => (&self.q8_1, Q8_1_BLOCK_BYTES, QK_LEGACY),
            GgmlType::Q2K => (&self.q2k, Q2K_BLOCK_BYTES, QK_K),
            GgmlType::Q3K => (&self.q3k, Q3K_BLOCK_BYTES, QK_K),
            GgmlType::Q8K => (&self.q8k, Q8K_BLOCK_BYTES, QK_K),
            GgmlType::IQ2XXS => (&self.iq2xxs, IQ2XXS_BLOCK_BYTES, QK_K),
            GgmlType::IQ2XS => (&self.iq2xs, IQ2XS_BLOCK_BYTES, QK_K),
            GgmlType::IQ2S => (&self.iq2s, IQ2S_BLOCK_BYTES, QK_K),
            GgmlType::IQ3XXS => (&self.iq3xxs, IQ3XXS_BLOCK_BYTES, QK_K),
            GgmlType::IQ3S => (&self.iq3s, IQ3S_BLOCK_BYTES, QK_K),
            GgmlType::IQ1S => (&self.iq1s, IQ1S_BLOCK_BYTES, QK_K),
            GgmlType::IQ1M => (&self.iq1m, IQ1M_BLOCK_BYTES, QK_K),
            GgmlType::IQ4XS => (&self.iq4xs, IQ4XS_BLOCK_BYTES, QK_K),
            _ => return None,
        })
    }
}

/// Every on-device dequant kernel (`src/kernels_cuda/dequant.cu`) in both
/// output types, plus the f32/f16 conversion kernels
/// (`src/kernels_cuda/convert.cu`), loaded once at model-load time and
/// threaded through `Model::load`/`load_hybrid`/`load_mla`'s `load_weight`
/// closures. Bundled into one struct rather than growing
/// `dequantize_tensor_to_device`/`load_weight_device`'s parameter list by one
/// `&AotKernel` per format.
pub(super) struct DequantKernels {
    pub(super) f32: FormatKernels,
    pub(super) f16: FormatKernels,
    /// `f16_roundtrip_kernel`, see [`f16_roundtrip_enabled`].
    pub(super) f16_roundtrip: AotKernel,
    pub(super) f32_to_f16: AotKernel,
    pub(super) f16_to_f32: AotKernel,
}

/// Loads every on-device dequant and conversion kernel, one module load per
/// `.cu` file -- shared by `Model::load`/`load_hybrid`/`load_mla`, which each
/// used to repeat this loading boilerplate individually. Both output types
/// come from the one `dequant` module, so the f16 variants cost function
/// lookups, not a second module load.
pub(super) fn load_dequant_kernels(
    device: &Arc<CudaDevice>,
) -> Result<DequantKernels, ReflexError> {
    // `load_kernel_module` wants `&'static str`s; the 40 names are built once
    // per process and leaked (a few hundred bytes).
    static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let names = NAMES.get_or_init(|| {
        let f32_names = DEQUANT_FORMATS
            .iter()
            .map(|f| format!("dequantize_{f}_kernel"));
        let f16_names = DEQUANT_FORMATS
            .iter()
            .map(|f| format!("dequantize_{f}_f16_kernel"));
        f32_names
            .chain(f16_names)
            .map(|n| &*Box::leak(n.into_boxed_str()))
            .collect()
    });
    let mut fns = aot::load_kernel_module(
        device,
        include_bytes!(env!("REFLEX_KERNEL_DEQUANT")),
        "dequant",
        names,
    )?
    .into_iter();
    let f32 = FormatKernels::from_iter(&mut fns)?;
    let f16 = FormatKernels::from_iter(&mut fns)?;
    let mut convert = aot::load_kernel_module(
        device,
        include_bytes!(env!("REFLEX_KERNEL_CONVERT")),
        "convert",
        &[
            "f16_roundtrip_kernel",
            "f32_to_f16_kernel",
            "f16_to_f32_kernel",
        ],
    )?
    .into_iter();
    let mut next = || {
        convert
            .next()
            .ok_or_else(|| ReflexError::Other("missing convert kernel".to_string()))
    };
    Ok(DequantKernels {
        f32,
        f16,
        f16_roundtrip: next()?,
        f32_to_f16: next()?,
        f16_to_f32: next()?,
    })
}

/// Whether weight `name` (GGUF shape `shape`) is a matrix that only the
/// matmul kernels (`gemv`/`gemm` and their per-expert/per-head/gather
/// variants) read: the tensors `--weights f16` stores as f16 (and
/// `REFLEX_F16_ROUNDTRIP` rounds in f32 mode). Everything else stays `f32`:
/// 1-D norms, biases and decay vectors, the Gated DeltaNet `ssm_conv1d`
/// kernel (2-D, but read by the conv kernel), and
/// the MoE routers (`ffn_gate_inp`, `ffn_gate_inp_shexp`), which are tiny and
/// feed a discrete top-k choice where a rounding-induced near-tie flip would
/// change which experts run.
pub(super) fn is_matrix_weight(name: &str, shape: &[u64]) -> bool {
    if shape.len() < 2 {
        return false;
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let suffix = base.rsplit('.').next().unwrap_or(base);
    !matches!(
        suffix,
        "ssm_conv1d" | "ffn_gate_inp" | "ffn_gate_inp_shexp" | "token_embd"
    )
}

/// `REFLEX_F16_ROUNDTRIP=1`: a numerics probe that rounds every matrix weight
/// (see [`is_matrix_weight`]) to f16 and back right after its f32 dequant, so
/// a run measures what f16 weight storage alone does to the output.
pub(super) fn f16_roundtrip_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REFLEX_F16_ROUNDTRIP").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// Launches `f16_roundtrip_kernel` over all of `data`, on the default stream
/// (the same one the dequant kernel that produced `data` ran on).
pub(super) fn f16_roundtrip_in_place(
    kernels: &DequantKernels,
    data: &mut CudaSlice<f32>,
) -> Result<(), ReflexError> {
    let n = data.len();
    unsafe {
        kernels
            .f16_roundtrip
            .function
            .clone()
            .launch(elementwise_launch_cfg(n), (data, n as u64))
            .map_err(|e| crate::gpu_err!(e, "f16_roundtrip launch: {e}"))
    }
}

/// Dequantizes a matrix weight (see [`is_matrix_weight`]) to `dtype`: straight
/// to f16 through the `_f16` dequant kernels, or to f32 (with the
/// `REFLEX_F16_ROUNDTRIP` probe applied when it's on).
pub(super) fn dequantize_matrix_to_device(
    pipeline: &mut WeightLoadPipeline,
    kernels: &DequantKernels,
    dtype: WeightsDtype,
    ggml_type: GgmlType,
    bytes: &[u8],
    element_count: u64,
) -> Result<WeightData, ReflexError> {
    match dtype {
        WeightsDtype::F16 => Ok(WeightData::F16(dequantize_tensor_to_device_f16(
            pipeline,
            kernels,
            ggml_type,
            bytes,
            element_count,
        )?)),
        WeightsDtype::F32 => {
            let mut data =
                dequantize_tensor_to_device(pipeline, kernels, ggml_type, bytes, element_count)?;
            if f16_roundtrip_enabled() {
                f16_roundtrip_in_place(kernels, &mut data)?;
            }
            Ok(WeightData::F32(data))
        }
    }
}

/// Narrows an f32 buffer to a new f16 one on the device (round to nearest
/// even). Used after a LoRA merge, which happens in f32.
pub(super) fn f32_to_f16_on_device(
    device: &Arc<CudaDevice>,
    kernel: &AotKernel,
    src: &CudaSlice<f32>,
) -> Result<CudaSlice<half::f16>, ReflexError> {
    let n = src.len();
    let mut out = unsafe { device.alloc::<half::f16>(n) }
        .map_err(|e| crate::gpu_err!(e, "alloc f16 weight: {e}"))?;
    unsafe {
        kernel
            .function
            .clone()
            .launch(elementwise_launch_cfg(n), (src, &mut out, n as u64))
            .map_err(|e| crate::gpu_err!(e, "f32_to_f16 launch: {e}"))?;
    }
    Ok(out)
}

/// Widens an f16 buffer to a new f32 one on the device (exact).
pub(super) fn f16_to_f32_on_device(
    device: &Arc<CudaDevice>,
    kernel: &AotKernel,
    src: &CudaSlice<half::f16>,
) -> Result<CudaSlice<f32>, ReflexError> {
    let n = src.len();
    let mut out = unsafe { device.alloc::<f32>(n) }
        .map_err(|e| crate::gpu_err!(e, "alloc f32 weight: {e}"))?;
    unsafe {
        kernel
            .function
            .clone()
            .launch(elementwise_launch_cfg(n), (src, &mut out, n as u64))
            .map_err(|e| crate::gpu_err!(e, "f16_to_f32 launch: {e}"))?;
    }
    Ok(out)
}

/// One thread per element, 256 per block, for the `convert.cu` kernels.
fn elementwise_launch_cfg(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = ((n as u64).div_ceil(threads as u64) as u32).max(1);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One pinned (page-locked) host staging buffer for [`WeightLoadPipeline`],
/// grown lazily (never shrunk) to the largest tensor byte length seen so
/// far. Plain pageable host memory (e.g. straight from the mmap'd GGUF)
/// forces the CUDA driver to silently stage `cuMemcpyHtoDAsync` through its
/// own temporary pinned buffer instead of actually running it concurrently
/// with other work -- pinning it ourselves is what makes the H2D transfer
/// below genuinely overlap a dequant kernel on another stream. cudarc's
/// safe wrapper doesn't expose `cuMemHostAlloc`/`cuMemFreeHost` at all, so
/// this drops to the raw driver FFI (`cudarc::driver::sys`), the same
/// precedent `diagnostics.rs` already set for driver calls the safe layer
/// doesn't cover.
pub(super) struct PinnedHostBuffer {
    pub(super) ptr: *mut u8,
    pub(super) cap: usize,
}

impl PinnedHostBuffer {
    pub(super) fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Grows the buffer to hold at least `len` bytes if it doesn't already
    /// (real models reuse only a handful of distinct tensor byte lengths
    /// per slot, so this stops reallocating after the first few calls).
    ///
    /// # Safety
    /// The caller must guarantee no async transfer still reads this slot's
    /// current allocation -- [`WeightLoadPipeline::dequantize`] only grows
    /// a slot right after waiting for that slot's prior kernel (which also
    /// covers the copy that fed it) to finish.
    unsafe fn ensure_capacity(&mut self, len: usize) -> Result<(), ReflexError> {
        if len <= self.cap {
            return Ok(());
        }
        self.free();
        let mut raw_ptr: *mut core::ffi::c_void = std::ptr::null_mut();
        sys::lib()
            .cuMemHostAlloc(&mut raw_ptr, len, 0)
            .result()
            .map_err(|e| crate::gpu_err!(e, "cuMemHostAlloc({len} bytes): {e}"))?;
        self.ptr = raw_ptr as *mut u8;
        self.cap = len;
        Ok(())
    }

    /// Copies `src` into this buffer's first `src.len()` bytes.
    ///
    /// # Safety
    /// `ensure_capacity(src.len())` must have already succeeded, and no
    /// async transfer may still be reading this slot's previous contents.
    unsafe fn write(&mut self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len());
    }

    /// Borrows this buffer's first `len` bytes (`len <= self.cap`).
    pub(super) fn as_slice(&self, len: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    pub(super) fn free(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = sys::lib().cuMemFreeHost(self.ptr as *mut core::ffi::c_void);
            }
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        self.free();
    }
}

/// Double-buffered pipeline for the per-tensor on-device dequant load path
/// (`Model::load`/`load_hybrid`/`load_mla`'s `load_weight` closures, ~310
/// calls for a Qwen3-0.6B GGUF). The pre-pipeline code called
/// `CudaDevice::htod_sync_copy` (a *blocking* H2D copy of each tensor's raw
/// quantized bytes) before launching that tensor's dequant kernel,
/// sequentially -- nothing overlapped tensor N+1's transfer with tensor N's
/// kernel still running. This pipelines the two: raw bytes are staged into
/// one of two pinned host buffers and uploaded asynchronously on a forked
/// `copy_stream`, while the dequant kernel for a previous tensor is still
/// executing on the device's own default stream (`compute_stream` below --
/// every kernel launch in this codebase already runs there). Every dequant
/// kernel still reads byte-identical input and produces byte-identical
/// output to the pre-pipeline sequential path: this is a timing/memory-
/// movement change only.
///
/// Slot lifecycle for tensor `i` (`slot = i % 2`):
/// 1. If slot `slot` has been used before (tensor `i - 2`), block the
///    *host* on `copy_done[slot]` via `cuEventSynchronize`. This one is a
///    CPU-side wait on purpose, and it is the pipeline's only correctness-
///    critical synchronization: steps 2-5 are all asynchronous, so the CPU
///    runs arbitrarily far ahead of the GPU, and the pinned buffer is
///    written by the *CPU*. A GPU-side `cuStreamWaitEvent` would order the
///    copy stream but would not stop the host from overwriting (or, on
///    growth, freeing) a staging buffer whose previous H2D transfer is
///    still in flight -- a silent wrong-weight-bytes race. In steady state
///    this blocks for ~0: a whole other tensor's copy and kernel were
///    enqueued since this slot's last use.
/// 2. Make `copy_stream` wait on `kernel_done[slot]` (GPU-side): the
///    device-side staging buffer for this slot is reused every other
///    tensor, so the dequant kernel that last read it must finish before
///    this tensor's transfer overwrites it. Unlike step 1 this one is
///    correctly a *stream* wait -- the writer here is the GPU's copy
///    engine, not the host.
/// 3. Host-side `memcpy` this tensor's raw bytes into the pinned buffer,
///    and grow this slot's device staging buffer if this tensor is bigger
///    than anything seen on that slot so far.
/// 4. Async H2D copy on `copy_stream`, then record `copy_done[slot]`.
/// 5. `compute_stream` waits on `copy_done[slot]`, the dequant kernel
///    launches exactly as before, and `kernel_done[slot]` is recorded right
///    after it for step 2's use two tensors from now.
///
/// Both staging buffers (pinned host and device) are *reused* across
/// tensors rather than allocated per tensor. That is deliberate: allocating
/// the device buffer per tensor via `CudaDevice::alloc` would stream-order
/// it on `compute_stream`, and the cross-stream event that then has to make
/// `copy_stream` wait for that allocation also drags in every kernel
/// already queued on `compute_stream` -- which serializes copy N+1 behind
/// kernel N and destroys exactly the overlap this type exists to create
/// (measured: ~2% instead of ~20%+ on a real Qwen3-0.6B load).
pub(super) struct WeightLoadPipeline {
    pub(super) device: Arc<CudaDevice>,
    pub(super) copy_stream: sys::CUstream,
    pub(super) pinned: [PinnedHostBuffer; 2],
    /// Device-side raw quantized-byte staging buffers, one per slot, grown
    /// lazily (never shrunk) like their pinned host counterparts.
    pub(super) raw_dev: [Option<CudaSlice<u8>>; 2],
    pub(super) raw_cap: [usize; 2],
    pub(super) copy_done: [sys::CUevent; 2],
    /// Recorded on `compute_stream` right after the dequant kernel reading
    /// a slot is launched; `None` until that slot has been used once.
    pub(super) kernel_done: [Option<sys::CUevent>; 2],
    /// Whether `copy_done[slot]` has ever been recorded -- synchronizing on
    /// a never-recorded event returns immediately, but this keeps the
    /// intent explicit rather than relying on that.
    pub(super) slot_used: [bool; 2],
    pub(super) next: usize,
}

impl WeightLoadPipeline {
    pub(super) fn new(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let copy_stream = result::stream::create(result::stream::StreamKind::NonBlocking)
            .map_err(|e| crate::gpu_err!(e, "create pipeline copy stream: {e}"))?;
        let mk_event = || {
            result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|e| crate::gpu_err!(e, "create pipeline event: {e}"))
        };
        Ok(Self {
            device: device.clone(),
            copy_stream,
            pinned: [PinnedHostBuffer::new(), PinnedHostBuffer::new()],
            raw_dev: [None, None],
            raw_cap: [0, 0],
            copy_done: [mk_event()?, mk_event()?],
            kernel_done: [None, None],
            slot_used: [false, false],
            next: 0,
        })
    }

    /// Grows slot `slot`'s device-side staging buffer to hold at least
    /// `len` bytes. Rare after the first few tensors (a real model reuses a
    /// handful of distinct tensor byte lengths), so the full
    /// `device.synchronize()` here is cheap: it drains `compute_stream`,
    /// which both retires the old buffer's readers and guarantees the new
    /// stream-ordered allocation is materialized before `copy_stream`
    /// writes into it.
    pub(super) fn ensure_raw_capacity(
        &mut self,
        slot: usize,
        len: usize,
    ) -> Result<(), ReflexError> {
        if len <= self.raw_cap[slot] {
            return Ok(());
        }
        let buf = unsafe { self.device.alloc::<u8>(len) }
            .map_err(|e| crate::gpu_err!(e, "alloc device staging buffer: {e}"))?;
        self.raw_dev[slot] = Some(buf);
        self.raw_cap[slot] = len;
        self.device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "pipeline: sync after staging-buffer growth: {e}"))?;
        Ok(())
    }

    /// Pipelined replacement for the old sequential `htod_sync_copy` +
    /// kernel-launch (see the struct doc comment for the full slot
    /// lifecycle) -- same truncation behavior as before if the last block
    /// is only partially used. `T` is the dequant kernel's output element
    /// type (`f32` or `half::f16`); `kernel` must be the matching variant.
    pub(super) fn dequantize<T: DeviceRepr>(
        &mut self,
        kernel: &AotKernel,
        block_bytes: usize,
        block_elems: usize,
        bytes: &[u8],
        element_count: u64,
    ) -> Result<CudaSlice<T>, ReflexError> {
        let slot = self.next % 2;
        self.next += 1;
        let compute_stream = *self.device.cu_stream();

        // Host-side wait -- see the struct doc comment's step 1. Without
        // this, the CPU (which never blocks anywhere else in this method)
        // would overwrite or free a pinned buffer whose previous async H2D
        // transfer is still reading it.
        if self.slot_used[slot] {
            unsafe { sys::lib().cuEventSynchronize(self.copy_done[slot]) }
                .result()
                .map_err(|e| {
                    crate::gpu_err!(e, "pipeline: await slot {slot}'s prior H2D copy: {e}")
                })?;
        }

        // GPU-side wait -- step 2: this slot's device staging buffer is
        // about to be overwritten by the transfer below, so the kernel that
        // last read it has to be done first.
        if let Some(ev) = self.kernel_done[slot] {
            unsafe {
                result::stream::wait_event(
                    self.copy_stream,
                    ev,
                    sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
            }
            .map_err(|e| {
                crate::gpu_err!(e, "pipeline: wait for slot {slot}'s prior kernel: {e}")
            })?;
        }

        unsafe {
            self.pinned[slot].ensure_capacity(bytes.len())?;
            self.pinned[slot].write(bytes);
        }
        self.ensure_raw_capacity(slot, bytes.len())?;

        let raw_ptr = *self.raw_dev[slot]
            .as_ref()
            .expect("staging buffer set by ensure_raw_capacity")
            .device_ptr();
        unsafe {
            result::memcpy_htod_async(
                raw_ptr,
                self.pinned[slot].as_slice(bytes.len()),
                self.copy_stream,
            )
        }
        .map_err(|e| crate::gpu_err!(e, "pipeline: async H2D copy: {e}"))?;
        unsafe { result::event::record(self.copy_done[slot], self.copy_stream) }
            .map_err(|e| crate::gpu_err!(e, "pipeline: record copy_done: {e}"))?;
        self.slot_used[slot] = true;
        unsafe {
            result::stream::wait_event(
                compute_stream,
                self.copy_done[slot],
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .map_err(|e| crate::gpu_err!(e, "pipeline: wait for copy_done: {e}"))?;

        let num_blocks = bytes.len() / block_bytes;
        let out_len = num_blocks * block_elems;
        let mut dev_out = unsafe { self.device.alloc::<T>(out_len) }
            .map_err(|e| crate::gpu_err!(e, "alloc dequant output: {e}"))?;

        let threads = 256u32;
        let blocks = (num_blocks as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let raw = self.raw_dev[slot]
            .as_ref()
            .expect("staging buffer set by ensure_raw_capacity");
        unsafe {
            kernel
                .function
                .clone()
                .launch(launch_cfg, (raw, &mut dev_out, num_blocks as u32))
                .map_err(|e| crate::gpu_err!(e, "dequant kernel launch: {e}"))?;
        }

        if self.kernel_done[slot].is_none() {
            let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|e| crate::gpu_err!(e, "create pipeline kernel_done event: {e}"))?;
            self.kernel_done[slot] = Some(ev);
        }
        let kdone = self.kernel_done[slot].expect("kernel_done[slot] was just set");
        unsafe { result::event::record(kdone, compute_stream) }
            .map_err(|e| crate::gpu_err!(e, "pipeline: record kernel_done: {e}"))?;

        if out_len as u64 == element_count {
            return Ok(dev_out);
        }
        let n = element_count as usize;
        let mut truncated = unsafe { self.device.alloc::<T>(n) }
            .map_err(|e| crate::gpu_err!(e, "alloc truncated dequant output: {e}"))?;
        let src = dev_out.slice(0..n);
        self.device
            .dtod_copy(&src, &mut truncated)
            .map_err(|e| crate::gpu_err!(e, "truncate dequant output: {e}"))?;
        Ok(truncated)
    }
}

impl Drop for WeightLoadPipeline {
    fn drop(&mut self) {
        // Every slot's pinned buffer is about to be freed -- make sure no
        // async H2D copy on `copy_stream` is still reading it first
        // (freeing pinned host memory out from under an in-flight transfer
        // is a real, silently-corrupting driver bug, not just a leak).
        unsafe {
            let _ = result::stream::synchronize(self.copy_stream);
            let _ = result::stream::destroy(self.copy_stream);
            for ev in self.copy_done {
                let _ = result::event::destroy(ev);
            }
            for ev in self.kernel_done.into_iter().flatten() {
                let _ = result::event::destroy(ev);
            }
        }
    }
}

/// Dequantizes one tensor's raw quantized bytes straight to a device-resident
/// `f32` buffer. Every block-quantized format, including the IQ
/// (codebook/non-uniform) family, dequantizes on-device via
/// `src/kernels_cuda/dequant.cu` -- no host `f32` copy is ever materialized
/// for any of them, closing the gap with llama.cpp's CUDA backend, which
/// never materializes one either. Types with no on-device kernel (F32, F16,
/// BF16) fall back to the host `dequant::dequantize` path
/// (`src/dequant.rs`/`dequant_iq.rs`), which needs no `match` update for a
/// future format.
pub(super) fn dequantize_tensor_to_device(
    pipeline: &mut WeightLoadPipeline,
    kernels: &DequantKernels,
    ggml_type: GgmlType,
    bytes: &[u8],
    element_count: u64,
) -> Result<CudaSlice<f32>, ReflexError> {
    if let Some((kernel, block_bytes, block_elems)) = kernels.f32.for_type(ggml_type) {
        return pipeline.dequantize(kernel, block_bytes, block_elems, bytes, element_count);
    }
    let host = dequant::dequantize(ggml_type, bytes, element_count)?;
    pipeline
        .device
        .htod_sync_copy(&host)
        .map_err(|e| crate::gpu_err!(e, "upload weight to device: {e}"))
}

/// [`dequantize_tensor_to_device`] with f16 output: the same pipelined upload
/// feeding the `_f16` dequant kernel, which rounds each value once on store.
/// The host fallback narrows on the host (`half::f16::from_f32`, also round
/// to nearest even) and uploads half the bytes.
pub(super) fn dequantize_tensor_to_device_f16(
    pipeline: &mut WeightLoadPipeline,
    kernels: &DequantKernels,
    ggml_type: GgmlType,
    bytes: &[u8],
    element_count: u64,
) -> Result<CudaSlice<half::f16>, ReflexError> {
    if let Some((kernel, block_bytes, block_elems)) = kernels.f16.for_type(ggml_type) {
        return pipeline.dequantize(kernel, block_bytes, block_elems, bytes, element_count);
    }
    let host: Vec<half::f16> = dequant::dequantize(ggml_type, bytes, element_count)?
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    pipeline
        .device
        .htod_sync_copy(&host)
        .map_err(|e| crate::gpu_err!(e, "upload weight to device: {e}"))
}

/// Loads and dequantizes weight `name` straight to device memory -- shared
/// by `Model::load`/`load_hybrid`/`load_mla`'s own `load_weight` closures.
/// `policy` picks the element type: a matrix weight gets the requested
/// `--weights` dtype, everything else `f32` (see [`is_matrix_weight`]).
pub(super) fn load_weight_device(
    pipeline: &mut WeightLoadPipeline,
    kernels: &DequantKernels,
    policy: &WeightPolicy,
    file: &GgufFile,
    name: &str,
) -> Result<Weight, ReflexError> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing weight '{name}'"))?;
    let bytes = file.tensor_bytes(info)?;
    let data = match policy.matrix_dtype_for(name, &info.shape) {
        Some(dtype) => dequantize_matrix_to_device(
            pipeline,
            kernels,
            dtype,
            info.ggml_type,
            bytes,
            info.element_count(),
        ),
        None => dequantize_tensor_to_device(
            pipeline,
            kernels,
            info.ggml_type,
            bytes,
            info.element_count(),
        )
        .map(WeightData::F32),
    }
    .map_err(|e| e.rewrap(format!("load weight '{name}': {e}")))?;
    Ok(Weight {
        data,
        shape: info.shape.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_dtype_parses_and_flag_wins() {
        assert_eq!(WeightsDtype::parse("f16").unwrap(), WeightsDtype::F16);
        assert_eq!(WeightsDtype::parse("f32").unwrap(), WeightsDtype::F32);
        assert!(WeightsDtype::parse("bf16").is_err());
        assert!(WeightsDtype::parse("F16").is_err());
        // An explicit flag never consults REFLEX_WEIGHTS or the fallback.
        assert_eq!(
            WeightsDtype::resolve_or(Some("f16"), WeightsDtype::F32).unwrap(),
            WeightsDtype::F16
        );
        assert_eq!(
            WeightsDtype::resolve_or(Some("f32"), WeightsDtype::F16).unwrap(),
            WeightsDtype::F32
        );
        assert_eq!(WeightsDtype::F16.to_string(), "f16");
    }

    #[test]
    fn matrix_weights_are_the_matmul_operands_only() {
        assert!(is_matrix_weight("blk.0.attn_q.weight", &[1024, 2048]));
        assert!(is_matrix_weight(
            "blk.0.ffn_down_exps.weight",
            &[768, 2048, 128]
        ));
        assert!(is_matrix_weight("blk.3.attn_k_b.weight", &[128, 512, 16]));
        assert!(is_matrix_weight("output.weight", &[1024, 151936]));
        assert!(is_matrix_weight("blk.0.ssm_out.weight", &[2048, 1024]));
        // 1-D tensors.
        assert!(!is_matrix_weight("blk.0.attn_norm.weight", &[1024]));
        assert!(!is_matrix_weight("blk.0.ssm_dt.bias", &[32]));
        assert!(!is_matrix_weight("blk.0.ssm_a", &[32]));
        // 2-D, but not read by a matmul kernel, or a router.
        assert!(!is_matrix_weight("blk.0.ssm_conv1d.weight", &[4, 6144]));
        assert!(!is_matrix_weight("blk.0.ffn_gate_inp.weight", &[2048, 128]));
        assert!(!is_matrix_weight(
            "blk.0.ffn_gate_inp_shexp.weight",
            &[2048, 1]
        ));
        assert!(!is_matrix_weight("token_embd.weight", &[1024, 151936]));
    }
}
