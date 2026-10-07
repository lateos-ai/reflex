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
    pub const DEFAULT: WeightsDtype = WeightsDtype::F16;

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

/// What `--weights f16` prefill fed its f16 GEMMs so far
/// ([`Model::f16_activation_stats`]). Activations are cast to f16 for
/// `cublasGemmEx`; f16's largest finite value is 65504, and the cast saturates
/// there instead of producing inf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F16ActivationStats {
    /// Largest |activation| cast so far (NaN if a NaN was cast).
    pub max_abs: f32,
    /// How many activation values exceeded 65504 and were clamped.
    pub saturated: u32,
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
/// `--weights f16` mode; `Quant` is a matrix weight kept as its raw GGUF blocks
/// (`REFLEX_QUANT_RESIDENT=1`, dense path) and read by the quantized matmul
/// kernels. Every matmul wrapper in `kernels.rs` dispatches on this enum.
pub(super) enum WeightData {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
    /// Raw GGUF blocks of type `ty` (Q4_K for layer weights, Q6_K for the LM
    /// head) at `offset..offset + len` in a shared device buffer,
    /// byte-identical to the file.
    Quant {
        ty: GgmlType,
        arena: Arc<CudaSlice<u8>>,
        offset: usize,
        len: usize,
    },
}

/// A zero-copy view of part of a [`Weight`] (one MoE expert's or one MLA
/// head's slice of a stacked 3-D tensor), with the same element type.
pub(super) enum WeightView<'a> {
    F32(CudaView<'a, f32>),
    F16(CudaView<'a, half::f16>),
}

/// A weight tensor, uploaded to device memory once at load time (see
/// `Model::load`'s `load_weight`) so no forward-pass call re-uploads it --
/// kernels below read `self.data` (or a zero-copy [`WeightView`] slice of it,
/// for per-expert MoE tensors) directly. Shape is the original GGUF shape
/// (`[in_features, out_features]` for a 2-D `nn.Linear`-style weight,
/// `[hidden_size]` for a norm weight).
///
/// Matrix weights are dequantized at load to the `--weights` dtype. With
/// `REFLEX_QUANT_RESIDENT=1`, the dense path keeps Q4_K matmul weights (and a
/// Q6_K LM head) as their raw GGUF blocks instead ([`WeightData::Quant`]), and
/// the GEMV/GEMM wrappers dequantize as they compute -- see
/// docs/design/quantized-resident-weights.md. Paths that only handle one
/// element type go through [`Weight::f32`] or [`Weight::view`], which error
/// (never panic) on anything else.
pub(super) struct Weight {
    pub(super) data: WeightData,
    pub(super) shape: Vec<u64>,
}

impl Weight {
    /// The `f32` buffer of a tensor that is always `f32` (a norm, bias or
    /// state-space vector, see [`is_matrix_weight`]). Errors on an f16 or
    /// quantized-resident matrix weight instead of handing an f32 kernel the
    /// wrong element type.
    pub(super) fn f32(&self) -> Result<&CudaSlice<f32>, ReflexError> {
        match &self.data {
            WeightData::F32(s) => Ok(s),
            WeightData::F16(_) => Err(crate::reflex_err!(
                Other,
                "weight of shape {:?} is stored as f16, but this op reads it as f32",
                self.shape
            )),
            WeightData::Quant { ty, .. } => Err(crate::reflex_err!(
                Other,
                "internal: an f32-only code path got a quantized-resident ({ty:?}) weight of shape {:?}",
                self.shape
            )),
        }
    }

    #[cfg(test)]
    pub(super) fn dtype(&self) -> WeightsDtype {
        match &self.data {
            WeightData::F32(_) => WeightsDtype::F32,
            WeightData::F16(_) => WeightsDtype::F16,
            WeightData::Quant { ty, .. } => panic!("dtype() on a quantized-resident {ty:?} weight"),
        }
    }

    /// Device address of a quantized weight's first block, for kernel launches.
    pub(super) fn quant_ptr(&self) -> Option<u64> {
        match &self.data {
            WeightData::F32(_) | WeightData::F16(_) => None,
            WeightData::Quant { arena, offset, .. } => Some(*arena.device_ptr() + *offset as u64),
        }
    }

    /// The whole weight as a [`WeightView`]. Errors on a quantized-resident
    /// weight, which has no element view (`Model::gemm` routes those to the
    /// quantized kernels before asking for one).
    pub(super) fn full_view(&self) -> Result<WeightView<'_>, ReflexError> {
        self.view(0..self.shape.iter().product::<u64>() as usize)
    }

    /// Elements `range` of this weight's flat buffer, without copying. Errors
    /// on a quantized-resident weight (only the dense path keeps those, and it
    /// never slices a weight).
    pub(super) fn view(
        &self,
        range: std::ops::Range<usize>,
    ) -> Result<WeightView<'_>, ReflexError> {
        match &self.data {
            WeightData::F32(s) => Ok(WeightView::F32(s.slice(range))),
            WeightData::F16(s) => Ok(WeightView::F16(s.slice(range))),
            WeightData::Quant { ty, .. } => Err(crate::reflex_err!(
                Other,
                "internal: element view of a quantized-resident ({ty:?}) weight of shape {:?}",
                self.shape
            )),
        }
    }
}

/// One device allocation holding every quantized-resident tensor's raw
/// blocks (`REFLEX_QUANT_RESIDENT=1`), sized up front from the GGUF header so
/// the load makes one `cuMemAlloc` instead of one per tensor. Offsets are
/// 256-byte aligned.
pub(super) struct QuantArena {
    pub(super) buf: Arc<CudaSlice<u8>>,
    pub(super) used: usize,
}

pub(super) const QUANT_ARENA_ALIGN: usize = 256;

impl QuantArena {
    /// Sized for the tensors in `names` that `keep(name, type)` accepts;
    /// `Ok(None)` when it accepts none.
    pub(super) fn for_tensors(
        device: &Arc<CudaDevice>,
        file: &GgufFile,
        names: &[String],
        keep: impl Fn(&str, GgmlType) -> bool,
    ) -> Result<Option<Self>, ReflexError> {
        let mut total = 0usize;
        for name in names {
            if let Some(info) = file.tensor_info(name) {
                if keep(name, info.ggml_type) {
                    total += file
                        .tensor_bytes(info)?
                        .len()
                        .next_multiple_of(QUANT_ARENA_ALIGN);
                }
            }
        }
        if total == 0 {
            return Ok(None);
        }
        let buf = unsafe { device.alloc::<u8>(total) }
            .map_err(|e| crate::gpu_err!(e, "alloc quantized weight arena ({total} bytes): {e}"))?;
        Ok(Some(Self {
            buf: Arc::new(buf),
            used: 0,
        }))
    }

    fn reserve(&mut self, len: usize) -> Result<usize, ReflexError> {
        let offset = self.used;
        let end = offset + len.next_multiple_of(QUANT_ARENA_ALIGN);
        if end > self.buf.len() {
            return Err(crate::reflex_err!(
                Other,
                "quantized weight arena overflow: need {end} bytes, have {}",
                self.buf.len()
            ));
        }
        self.used = end;
        Ok(offset)
    }
}

/// `REFLEX_EXPERT_TRACE=1`: Kolibri-1 layers print each forward call's
/// routed experts to stderr, one `REFLEX_EXPERT_TRACE rows=N experts=...` line
/// per layer in layer order (all rows of a batched prefill on one line), for
/// measuring how many distinct experts a prompt touches
/// (docs/design/kolibri.md, Phase 4). Off by default.
pub(super) fn expert_trace_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("REFLEX_EXPERT_TRACE").is_ok_and(|v| v == "1"))
}

/// `REFLEX_QUANT_RESIDENT=1` turns on quantized-resident weights for the
/// dense path and Kolibri-1. Read once per load.
pub(super) fn quant_resident_enabled() -> bool {
    std::env::var("REFLEX_QUANT_RESIDENT").is_ok_and(|v| v == "1")
}

/// `REFLEX_LOAD_PROFILE=1` prints a `REFLEX_LOAD_PROFILE` line splitting
/// `model_load_ms` into its parts (dense path). Off by default: it adds GPU
/// timing events per tensor and a device sync at the end of the load.
pub(super) fn load_profile_enabled() -> bool {
    std::env::var("REFLEX_LOAD_PROFILE").is_ok_and(|v| v == "1")
}

/// Accumulators for `REFLEX_LOAD_PROFILE`. Host-side parts are wall time on
/// the loading thread; `h2d`/`dequant` pairs are GPU timing events read back
/// once at the end, so measuring them doesn't serialize the pipeline.
#[derive(Default)]
pub(super) struct LoadProfile {
    pub(super) pinned_wait: std::time::Duration,
    pub(super) pinned_fill: std::time::Duration,
    /// Time spent issuing `MADV_WILLNEED` readahead hints.
    pub(super) readahead: std::time::Duration,
    /// Bytes copied into the pinned slots (`pinned_fill`'s throughput).
    pub(super) fill_bytes: usize,
    pub(super) staging_grow: std::time::Duration,
    pub(super) out_alloc: std::time::Duration,
    pub(super) launch: std::time::Duration,
    pub(super) h2d_events: Vec<(sys::CUevent, sys::CUevent)>,
    pub(super) dequant_events: Vec<(sys::CUevent, sys::CUevent)>,
    pub(super) tensors_f32: usize,
    pub(super) tensors_quant: usize,
    /// Tensors uploaded without a dequant kernel (`F32` norms, host fallback).
    pub(super) tensors_host: usize,
    pub(super) h2d_bytes: usize,
    pub(super) f32_bytes: usize,
}

impl LoadProfile {
    fn timed_event() -> Result<sys::CUevent, ReflexError> {
        result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| crate::gpu_err!(e, "create profile event: {e}"))
    }

    /// Sum of GPU time over event pairs, in ms. Call only after the device
    /// has drained (every pair recorded and complete); destroys the events.
    fn drain_ms(pairs: &mut Vec<(sys::CUevent, sys::CUevent)>) -> f64 {
        let mut ms = 0f64;
        for (a, b) in pairs.drain(..) {
            unsafe {
                ms += result::event::elapsed(a, b).unwrap_or(0.0) as f64;
                let _ = result::event::destroy(a);
                let _ = result::event::destroy(b);
            }
        }
        ms
    }

    /// GPU sums `(h2d_ms, dequant_ms)`. The caller must have synchronized the
    /// device and the pipeline's copy stream first.
    pub(super) fn gpu_ms(&mut self) -> (f64, f64) {
        (
            Self::drain_ms(&mut self.h2d_events),
            Self::drain_ms(&mut self.dequant_events),
        )
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
    /// `cast_act_f16_kernel`: prefill activations into an f16 GEMM operand
    /// (`Model::gemm_view`'s f16 path).
    pub(super) cast_act_f16: AotKernel,
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
            "cast_act_f16_kernel",
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
        cast_act_f16: next()?,
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
/// grown lazily (never shrunk) up to [`STAGE_CHUNK_BYTES`]. Plain pageable
/// host memory (e.g. straight from the mmap'd GGUF)
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
    /// (chunks are capped at [`STAGE_CHUNK_BYTES`], so this stops
    /// reallocating after the first few calls).
    ///
    /// # Safety
    /// The caller must guarantee no async transfer still reads this slot's
    /// current allocation -- [`WeightLoadPipeline::stage_h2d`] only grows a
    /// slot right after waiting for that slot's prior copy to finish.
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

/// Largest piece of a tensor staged through one pinned slot. A bigger tensor
/// (a Kolibri-1 expert stack is hundreds of MB) is staged as several chunks,
/// so the H2D of chunk i overlaps the fill of chunk i+1 within one tensor and
/// pinned memory stays at 2 x this instead of 2 x the largest tensor.
const STAGE_CHUNK_BYTES: usize = 64 << 20;

/// Chunks smaller than this are filled on the loading thread alone: small
/// models (Qwen3-0.6B's matrix weights are all under ~2 MB) never wake the
/// fill workers, so their load is unchanged.
const PARALLEL_FILL_MIN_BYTES: usize = 8 << 20;

/// Smallest piece one fill thread copies.
const FILL_PIECE_MIN_BYTES: usize = 2 << 20;

/// How far past the current chunk [`WeightLoadPipeline::stage_h2d`] asks the
/// kernel to read ahead (`madvise(MADV_WILLNEED)`), so a cold load keeps the
/// disk busy instead of waiting on one page fault's readahead at a time.
const READAHEAD_BYTES: usize = 4 * STAGE_CHUNK_BYTES;

/// Fill threads (the loading thread included): `REFLEX_LOAD_THREADS`, else
/// the core count capped at 8.
fn fill_thread_count() -> usize {
    if let Some(n) = std::env::var("REFLEX_LOAD_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
}

/// `REFLEX_LOAD_READAHEAD=0` turns off the `MADV_WILLNEED` hint (for A/B
/// measurements); on by default.
fn readahead_enabled() -> bool {
    std::env::var("REFLEX_LOAD_READAHEAD").map_or(true, |v| v != "0")
}

/// Asks the kernel to start reading `range`'s pages in the background. A
/// hint only: errors are ignored, and it is a no-op off Linux.
#[cfg(target_os = "linux")]
fn advise_willneed(range: &[u8]) {
    extern "C" {
        fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: core::ffi::c_int) -> i32;
    }
    const MADV_WILLNEED: core::ffi::c_int = 3;
    if range.is_empty() {
        return;
    }
    // madvise wants a page-aligned start; the page holding `range`'s first
    // byte is part of the same mapping.
    let start = range.as_ptr() as usize & !4095;
    let end = range.as_ptr() as usize + range.len();
    unsafe {
        madvise(start as *mut core::ffi::c_void, end - start, MADV_WILLNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_willneed(_range: &[u8]) {}

/// One piece of a pinned-buffer fill, sent to a [`FillWorkers`] thread.
struct FillJob {
    src: *const u8,
    dst: *mut u8,
    len: usize,
}

// SAFETY: `FillWorkers::copy` keeps both ranges alive and untouched by
// anything else until the worker reports the piece done.
unsafe impl Send for FillJob {}

/// Helper threads that copy pieces of one chunk from the mmap into a pinned
/// buffer alongside the loading thread. Cold, each one takes its own page
/// faults, so several disk reads are in flight instead of one; warm, the copy
/// runs at several threads' memory bandwidth. They live only while a model
/// loads ([`WeightLoadPipeline::end_load`]) and only copy bytes -- not a
/// request pool.
struct FillWorkers {
    jobs: Vec<std::sync::mpsc::Sender<FillJob>>,
    done: std::sync::mpsc::Receiver<()>,
}

impl FillWorkers {
    /// Spawns up to `n` workers; `None` if not even one could start.
    fn spawn(n: usize) -> Option<Self> {
        let (done_tx, done) = std::sync::mpsc::channel();
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let (tx, rx) = std::sync::mpsc::channel::<FillJob>();
            let done_tx = done_tx.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("reflex-fill-{i}"))
                .spawn(move || {
                    for job in rx {
                        unsafe { std::ptr::copy_nonoverlapping(job.src, job.dst, job.len) };
                        if done_tx.send(()).is_err() {
                            break;
                        }
                    }
                });
            if spawned.is_err() {
                break;
            }
            jobs.push(tx);
        }
        (!jobs.is_empty()).then_some(Self { jobs, done })
    }

    /// Copies `src` to `dst` in `pieces` page-aligned parts: the first on
    /// the calling thread, the rest on workers. Returns only once every
    /// piece has landed. A piece whose worker is gone is copied here instead.
    ///
    /// # Safety
    /// `dst` must be valid for `src.len()` bytes and not overlap `src`.
    unsafe fn copy(&self, src: &[u8], dst: *mut u8, pieces: usize) {
        let piece = src
            .len()
            .div_ceil(pieces.max(1))
            .next_multiple_of(4096)
            .max(4096);
        let mut sent = 0usize;
        for (i, part) in src.chunks(piece).enumerate().skip(1) {
            let job = FillJob {
                src: part.as_ptr(),
                dst: dst.add(i * piece),
                len: part.len(),
            };
            match self.jobs.get(i - 1).map(|tx| tx.send(job)) {
                Some(Ok(())) => sent += 1,
                _ => std::ptr::copy_nonoverlapping(part.as_ptr(), dst.add(i * piece), part.len()),
            }
        }
        let first = &src[..piece.min(src.len())];
        std::ptr::copy_nonoverlapping(first.as_ptr(), dst, first.len());
        // Every sent job holds a live `done` sender, so this only fails if a
        // worker died mid-copy -- then no other job is still in flight
        // either (each worker runs one at a time and `sent` counts them all).
        for _ in 0..sent {
            if self.done.recv().is_err() {
                break;
            }
        }
    }
}

/// Pipelined host-to-device upload for the per-tensor load path
/// (`Model::load`/`load_hybrid`/`load_mla`'s `load_weight` closures, ~310
/// calls for a Qwen3-0.6B GGUF). The pre-pipeline code called
/// `CudaDevice::htod_sync_copy` (a *blocking* H2D copy of each tensor's raw
/// quantized bytes) before launching that tensor's dequant kernel,
/// sequentially -- nothing overlapped tensor N+1's transfer with tensor N's
/// kernel. This pipelines the two: raw bytes are staged through two pinned
/// host buffers and uploaded asynchronously on a forked `copy_stream`, while
/// the dequant kernel for a previous tensor is still executing on the
/// device's own default stream (`compute_stream` below -- every kernel
/// launch in this codebase already runs there). Every dequant kernel still
/// reads byte-identical input and produces byte-identical output to the
/// pre-pipeline sequential path: this is a timing/memory-movement change only.
///
/// Two kinds of slot, each double-buffered:
/// - **Pinned slots**, one per staged *chunk*: a tensor is staged in pieces
///   of at most [`STAGE_CHUNK_BYTES`], alternating between the two pinned
///   buffers, so a big tensor's chunk i+1 is filled while chunk i is on the
///   wire. A large chunk is filled by several threads ([`FillWorkers`]):
///   one host thread copying from the mmap (~4 GB/s cold, ~7.7 GB/s warm on
///   an A6000 host) was the whole load's bottleneck for a 47.5 GB model,
///   well under both the disk and PCIe.
/// - **Device staging slots**, one per *tensor* (`dequantize` only): the raw
///   bytes the dequant kernel reads.
///
/// Lifecycle for each chunk (pinned slot `p`):
/// 1. If slot `p` has been used before, block the *host* on `pinned_done[p]`
///    via `cuEventSynchronize`. This one is a CPU-side wait on purpose, and
///    it is the pipeline's only correctness-critical synchronization: the
///    rest is asynchronous, so the CPU runs arbitrarily far ahead of the
///    GPU, and the pinned buffer is written by the *CPU*. A GPU-side
///    `cuStreamWaitEvent` would order the copy stream but would not stop the
///    host from overwriting (or, on growth, freeing) a staging buffer whose
///    previous H2D transfer is still in flight -- a silent wrong-weight-bytes
///    race. It blocks only while the other slot's fill outruns the wire.
/// 2. Fill the pinned buffer (in parallel if the chunk is large), async H2D
///    on `copy_stream` to the chunk's place in the destination, and record
///    `pinned_done[p]`.
///
/// And for each tensor (`dequantize`/`upload_host_bytes`, device slot `d`):
/// 1. Make `copy_stream` wait on `kernel_done[d]` (GPU-side): the device
///    staging buffer for this slot is reused every other tensor, so the
///    dequant kernel that last read it must finish before this tensor's
///    transfer overwrites it. Unlike the pinned wait this one is correctly
///    a *stream* wait -- the writer here is the GPU's copy engine.
/// 2. Stage the chunks as above, then `compute_stream` waits on the last
///    chunk's `pinned_done` (the copy stream runs the chunks in order).
/// 3. Launch the dequant kernel (for a tensor with no kernel,
///    `upload_host_bytes`: a device-to-device copy into a fresh buffer) and
///    record `kernel_done[d]` after it. Nothing in the load loop blocks the
///    host on the compute stream: a `htod_sync_copy` per norm tensor used to
///    drain the whole pipeline twice per layer.
///
/// Both kinds of staging buffer are *reused* rather than allocated per
/// tensor. That is deliberate: allocating the device buffer per tensor via
/// `CudaDevice::alloc` would stream-order it on `compute_stream`, and the
/// cross-stream event that then has to make `copy_stream` wait for that
/// allocation also drags in every kernel already queued on `compute_stream`
/// -- which serializes copy N+1 behind kernel N and destroys exactly the
/// overlap this type exists to create (measured: ~2% instead of ~20%+ on a
/// real Qwen3-0.6B load).
pub(super) struct WeightLoadPipeline {
    pub(super) device: Arc<CudaDevice>,
    pub(super) copy_stream: sys::CUstream,
    pub(super) pinned: [PinnedHostBuffer; 2],
    /// Recorded on `copy_stream` after each pinned slot's last H2D copy.
    pub(super) pinned_done: [sys::CUevent; 2],
    /// Whether `pinned_done[slot]` has ever been recorded -- synchronizing on
    /// a never-recorded event returns immediately, but this keeps the
    /// intent explicit rather than relying on that.
    pub(super) pinned_used: [bool; 2],
    pub(super) next_pinned: usize,
    /// Device-side raw quantized-byte staging buffers, one per slot, grown
    /// lazily (never shrunk) like their pinned host counterparts.
    pub(super) raw_dev: [Option<CudaSlice<u8>>; 2],
    pub(super) raw_cap: [usize; 2],
    /// Recorded on `compute_stream` right after the dequant kernel reading
    /// a slot is launched; `None` until that slot has been used once.
    pub(super) kernel_done: [Option<sys::CUevent>; 2],
    pub(super) next: usize,
    /// Fill threads, the loading thread included (`REFLEX_LOAD_THREADS`).
    fill_threads: usize,
    /// Spawned on the first chunk big enough to split, dropped by `end_load`.
    fill_workers: Option<FillWorkers>,
    readahead: bool,
    /// `REFLEX_LOAD_PROFILE` accumulators; `None` (the default) adds no work.
    pub(super) profile: Option<LoadProfile>,
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
            pinned_done: [mk_event()?, mk_event()?],
            pinned_used: [false, false],
            next_pinned: 0,
            raw_dev: [None, None],
            raw_cap: [0, 0],
            kernel_done: [None, None],
            next: 0,
            fill_threads: fill_thread_count(),
            fill_workers: None,
            readahead: readahead_enabled(),
            profile: None,
        })
    }

    /// Lets the fill threads exit once the model's bulk load is done (the
    /// pipeline itself stays on `Model` for `lm_head_resident`, which
    /// respawns them if it needs them). Doesn't wait: no fill is in flight
    /// between calls.
    pub(super) fn end_load(&mut self) {
        self.fill_workers = None;
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
        let t = std::time::Instant::now();
        let buf = unsafe { self.device.alloc::<u8>(len) }
            .map_err(|e| crate::gpu_err!(e, "alloc device staging buffer: {e}"))?;
        self.raw_dev[slot] = Some(buf);
        self.raw_cap[slot] = len;
        self.device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "pipeline: sync after staging-buffer growth: {e}"))?;
        if let Some(p) = &mut self.profile {
            p.staging_grow += t.elapsed();
        }
        Ok(())
    }

    /// Per-tensor steps 1-2 (see the struct doc comment): takes the next
    /// device staging slot, orders the copy stream after the kernel that last
    /// read it, grows it if needed and stages `bytes` into it. The compute
    /// stream waits for the bytes; the caller enqueues the slot's reader on
    /// it and then calls [`Self::device_slot_read`].
    fn stage_to_device_slot(&mut self, bytes: &[u8]) -> Result<usize, ReflexError> {
        let slot = self.next % 2;
        self.next += 1;
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
        self.ensure_raw_capacity(slot, bytes.len())?;
        let raw_ptr = *self.raw_dev[slot]
            .as_ref()
            .expect("staging buffer set by ensure_raw_capacity")
            .device_ptr();
        self.stage_h2d(raw_ptr, bytes)?;
        Ok(slot)
    }

    /// Per-tensor step 3's event: records `kernel_done[slot]` on the compute
    /// stream after the slot's reader was enqueued there.
    fn device_slot_read(&mut self, slot: usize) -> Result<(), ReflexError> {
        if self.kernel_done[slot].is_none() {
            let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|e| crate::gpu_err!(e, "create pipeline kernel_done event: {e}"))?;
            self.kernel_done[slot] = Some(ev);
        }
        let kdone = self.kernel_done[slot].expect("kernel_done[slot] was just set");
        unsafe { result::event::record(kdone, *self.device.cu_stream()) }
            .map_err(|e| crate::gpu_err!(e, "pipeline: record kernel_done: {e}"))
    }

    /// Uploads host values that need no dequant kernel (an `F32` tensor's
    /// bytes as they are, or the host fallback's output) through the same
    /// device staging slots, then copies them device to device into a fresh
    /// buffer on the compute stream. This used to be `htod_sync_copy`, which
    /// synchronizes the compute stream: every norm tensor (two per layer)
    /// waited for all queued uploads and dequant kernels, ~0.9 s of a 2.3 s
    /// Mistral 7B load on a T4. `bytes.len()` must be a multiple of `T`'s size.
    pub(super) fn upload_host_bytes<T: DeviceRepr>(
        &mut self,
        bytes: &[u8],
    ) -> Result<CudaSlice<T>, ReflexError> {
        let n = bytes.len() / std::mem::size_of::<T>();
        let slot = self.stage_to_device_slot(bytes)?;
        let out = unsafe { self.device.alloc::<T>(n) }
            .map_err(|e| crate::gpu_err!(e, "alloc host-path weight: {e}"))?;
        let src = *self.raw_dev[slot]
            .as_ref()
            .expect("staging buffer set by ensure_raw_capacity")
            .device_ptr();
        unsafe {
            result::memcpy_dtod_async(
                *out.device_ptr(),
                src,
                bytes.len(),
                *self.device.cu_stream(),
            )
        }
        .map_err(|e| crate::gpu_err!(e, "pipeline: copy host-path weight: {e}"))?;
        self.device_slot_read(slot)?;
        if let Some(p) = &mut self.profile {
            p.tensors_host += 1;
        }
        Ok(out)
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
        let compute_stream = *self.device.cu_stream();
        let slot = self.stage_to_device_slot(bytes)?;

        let num_blocks = bytes.len() / block_bytes;
        let out_len = num_blocks * block_elems;
        let t_alloc = std::time::Instant::now();
        let mut dev_out = unsafe { self.device.alloc::<T>(out_len) }
            .map_err(|e| crate::gpu_err!(e, "alloc dequant output: {e}"))?;
        let t_launch = std::time::Instant::now();
        let dq_start = match &self.profile {
            Some(_) => {
                let ev = LoadProfile::timed_event()?;
                unsafe { result::event::record(ev, compute_stream) }
                    .map_err(|e| crate::gpu_err!(e, "profile: record dequant start: {e}"))?;
                Some(ev)
            }
            None => None,
        };

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
        if let (Some(p), Some(start)) = (&mut self.profile, dq_start) {
            let end = LoadProfile::timed_event()?;
            unsafe { result::event::record(end, compute_stream) }
                .map_err(|e| crate::gpu_err!(e, "profile: record dequant end: {e}"))?;
            p.dequant_events.push((start, end));
            p.out_alloc += t_launch - t_alloc;
            p.launch += t_launch.elapsed();
            p.tensors_f32 += 1;
            p.f32_bytes += out_len * 4;
        }

        self.device_slot_read(slot)?;

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

impl WeightLoadPipeline {
    /// Stages `bytes` through the pinned slots, chunk by chunk (see the
    /// struct doc comment's per-chunk lifecycle), into device memory at
    /// `dst`, then makes the compute stream wait for the last chunk. Shared
    /// by [`Self::dequantize`] and [`Self::upload_raw`].
    fn stage_h2d(&mut self, dst: sys::CUdeviceptr, bytes: &[u8]) -> Result<(), ReflexError> {
        let mut last_slot = None;
        let mut advised_end = 0usize;
        for (i, chunk) in bytes.chunks(STAGE_CHUNK_BYTES).enumerate() {
            let off = i * STAGE_CHUNK_BYTES;
            if self.readahead && bytes.len() > STAGE_CHUNK_BYTES {
                let t_ra = std::time::Instant::now();
                let want = (off + READAHEAD_BYTES).min(bytes.len());
                if want > advised_end {
                    advise_willneed(&bytes[advised_end.max(off)..want]);
                    advised_end = want;
                }
                if let Some(p) = &mut self.profile {
                    p.readahead += t_ra.elapsed();
                }
            }

            let slot = self.next_pinned % 2;
            self.next_pinned += 1;
            // Host-side wait -- per-chunk step 1. Without this, the CPU
            // (which never blocks anywhere else here) would overwrite or
            // free a pinned buffer whose previous async H2D transfer is
            // still reading it.
            let t_wait = std::time::Instant::now();
            if self.pinned_used[slot] {
                unsafe { sys::lib().cuEventSynchronize(self.pinned_done[slot]) }
                    .result()
                    .map_err(|e| {
                        crate::gpu_err!(e, "pipeline: await slot {slot}'s prior H2D copy: {e}")
                    })?;
            }
            let t_fill = std::time::Instant::now();
            unsafe { self.pinned[slot].ensure_capacity(chunk.len())? };
            self.fill(slot, chunk);
            if let Some(p) = &mut self.profile {
                p.pinned_wait += t_fill - t_wait;
                p.pinned_fill += t_fill.elapsed();
                p.fill_bytes += chunk.len();
            }
            self.h2d_async(dst + off as u64, slot, chunk.len())?;
            last_slot = Some(slot);
        }
        let Some(slot) = last_slot else {
            return Ok(());
        };
        unsafe {
            result::stream::wait_event(
                *self.device.cu_stream(),
                self.pinned_done[slot],
                sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }
        .map_err(|e| crate::gpu_err!(e, "pipeline: wait for copy: {e}"))
    }

    /// Copies `chunk` into pinned slot `slot`, split across the fill
    /// threads when it is big enough to be worth it. The slot must already
    /// hold `chunk.len()` bytes and have no copy reading it.
    fn fill(&mut self, slot: usize, chunk: &[u8]) {
        let dst = self.pinned[slot].ptr;
        let pieces = if chunk.len() >= PARALLEL_FILL_MIN_BYTES {
            self.fill_threads.min(chunk.len() / FILL_PIECE_MIN_BYTES)
        } else {
            1
        };
        if pieces > 1 && self.fill_workers.is_none() {
            self.fill_workers = FillWorkers::spawn(self.fill_threads - 1);
            if self.fill_workers.is_none() {
                self.fill_threads = 1;
            }
        }
        match &self.fill_workers {
            Some(w) if pieces > 1 => unsafe { w.copy(chunk, dst, pieces.min(w.jobs.len() + 1)) },
            _ => unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk.len()) },
        }
    }

    /// Per-chunk step 2's transfer: async copy of the first `len` bytes of
    /// pinned slot `slot` to `dst` on `copy_stream`, then record
    /// `pinned_done[slot]`.
    fn h2d_async(
        &mut self,
        dst: sys::CUdeviceptr,
        slot: usize,
        len: usize,
    ) -> Result<(), ReflexError> {
        let h2d_start = match &self.profile {
            Some(_) => {
                let ev = LoadProfile::timed_event()?;
                unsafe { result::event::record(ev, self.copy_stream) }
                    .map_err(|e| crate::gpu_err!(e, "profile: record h2d start: {e}"))?;
                Some(ev)
            }
            None => None,
        };
        unsafe {
            result::memcpy_htod_async(dst, self.pinned[slot].as_slice(len), self.copy_stream)
        }
        .map_err(|e| crate::gpu_err!(e, "pipeline: async H2D copy: {e}"))?;
        if let (Some(p), Some(start)) = (&mut self.profile, h2d_start) {
            let end = LoadProfile::timed_event()?;
            unsafe { result::event::record(end, self.copy_stream) }
                .map_err(|e| crate::gpu_err!(e, "profile: record h2d end: {e}"))?;
            p.h2d_events.push((start, end));
            p.h2d_bytes += len;
        }
        unsafe { result::event::record(self.pinned_done[slot], self.copy_stream) }
            .map_err(|e| crate::gpu_err!(e, "pipeline: record pinned_done: {e}"))?;
        self.pinned_used[slot] = true;
        Ok(())
    }

    /// Fill threads this pipeline uses (for `REFLEX_LOAD_PROFILE`).
    pub(super) fn fill_threads(&self) -> usize {
        self.fill_threads
    }

    /// Quantized-resident counterpart of [`Self::dequantize`]: stages `bytes`
    /// through the same pinned slots and copies them asynchronously to their
    /// final place in `arena` -- no device staging buffer, no dequant kernel,
    /// so only the per-chunk steps apply. The compute stream waits on the
    /// copy, so any later kernel reading the arena sees the bytes. Returns
    /// the tensor's `(offset, len)` in the arena.
    pub(super) fn upload_raw(
        &mut self,
        arena: &mut QuantArena,
        bytes: &[u8],
    ) -> Result<(usize, usize), ReflexError> {
        let offset = arena.reserve(bytes.len())?;
        let dst = *arena.buf.device_ptr() + offset as u64;
        self.stage_h2d(dst, bytes)?;
        if let Some(p) = &mut self.profile {
            p.tensors_quant += 1;
        }
        Ok((offset, bytes.len()))
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
            for ev in self.pinned_done {
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
    if ggml_type == GgmlType::F32 {
        // Already the f32 values: upload the file's bytes as they are.
        let len = (element_count as usize) * 4;
        let raw = bytes.get(..len).ok_or_else(|| {
            crate::reflex_err!(
                Gguf,
                "F32 tensor: {len} bytes needed, {} present",
                bytes.len()
            )
        })?;
        return pipeline.upload_host_bytes(raw);
    }
    let host = dequant::dequantize(ggml_type, bytes, element_count)?;
    pipeline.upload_host_bytes(as_bytes(&host))
}

/// A slice of plain numbers as its bytes (for [`WeightLoadPipeline::upload_host_bytes`]).
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
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
    pipeline.upload_host_bytes(as_bytes(&host))
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

/// [`load_weight_device`], but a 2-D tensor (or a 3-D per-expert stack, read
/// one expert at a time through `Model::quant_expert_weight`) whose type is in
/// `allowed` is kept as raw blocks in `arena` instead of being dequantized
/// (`REFLEX_QUANT_RESIDENT=1`: the dense path, and Kolibri-1's layers). Every
/// other tensor, and a LoRA target (`policy.keep_f32`, merged in f32), takes
/// [`load_weight_device`]'s path unchanged.
pub(super) fn load_weight_device_quant(
    pipeline: &mut WeightLoadPipeline,
    kernels: &DequantKernels,
    policy: &WeightPolicy,
    file: &GgufFile,
    name: &str,
    arena: &mut QuantArena,
    allowed: &[GgmlType],
) -> Result<Weight, ReflexError> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| crate::reflex_err!(Gguf, "missing weight '{name}'"))?;
    if !allowed.contains(&info.ggml_type)
        || !matches!(info.shape.len(), 2 | 3)
        || policy.keep_f32.contains(name)
    {
        return load_weight_device(pipeline, kernels, policy, file, name);
    }
    let bytes = file.tensor_bytes(info)?;
    let (offset, len) = pipeline
        .upload_raw(arena, bytes)
        .map_err(|e| e.rewrap(format!("load weight '{name}': {e}")))?;
    Ok(Weight {
        data: WeightData::Quant {
            ty: info.ggml_type,
            arena: arena.buf.clone(),
            offset,
            len,
        },
        shape: info.shape.clone(),
    })
}

/// Dequantizes a quantized-resident weight into a fresh `f32` device buffer
/// (reading the arena, no host traffic) and turns `w` into an `f32` weight --
/// used by `Model::apply_lora`, whose in-place delta add needs `f32`. A no-op
/// for an `f32` or `f16` weight (`apply_lora` widens an `f16` one itself).
pub(super) fn materialize_f32(
    device: &Arc<CudaDevice>,
    q4k_dequant: &cudarc::driver::CudaFunction,
    q6k_dequant: &cudarc::driver::CudaFunction,
    w: &mut Weight,
) -> Result<(), ReflexError> {
    let (dequant, num_blocks) = match &w.data {
        WeightData::F32(_) | WeightData::F16(_) => return Ok(()),
        WeightData::Quant {
            ty: GgmlType::Q4K,
            len,
            ..
        } => (q4k_dequant, *len / Q4K_BLOCK_BYTES),
        WeightData::Quant {
            ty: GgmlType::Q6K,
            len,
            ..
        } => (q6k_dequant, *len / Q6K_BLOCK_BYTES),
        WeightData::Quant { ty, .. } => {
            return Err(crate::reflex_err!(
                Other,
                "materializing a quantized-resident {ty:?} weight to f32 is not supported"
            ))
        }
    };
    let ptr = w
        .quant_ptr()
        .expect("quantized weight has a device pointer");
    let mut out = unsafe { device.alloc::<f32>(num_blocks * QK_K) }
        .map_err(|e| crate::gpu_err!(e, "alloc materialized weight: {e}"))?;
    let threads = 256u32;
    let launch_cfg = LaunchConfig {
        grid_dim: ((num_blocks as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dequant
            .clone()
            .launch(launch_cfg, (ptr, &mut out, num_blocks as u32))
            .map_err(|e| crate::gpu_err!(e, "materialize dequant launch: {e}"))?;
    }
    w.data = WeightData::F32(out);
    Ok(())
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

    #[test]
    fn parallel_fill_copies_every_byte() {
        let workers = FillWorkers::spawn(7).expect("spawn fill workers");
        for len in [0usize, 1, 4095, 4096 * 3 + 7, (9 << 20) + 13] {
            let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 4096) as u8).collect();
            for pieces in 1..=8 {
                let mut dst = vec![0u8; len];
                unsafe { workers.copy(&src, dst.as_mut_ptr(), pieces) };
                assert!(dst == src, "len={len} pieces={pieces}");
            }
        }
    }
}
