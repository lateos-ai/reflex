//! Resuming a generate call from a cached prefix: either a cache file's contents
//! (`crate::kv_io`, phase A of docs/design/agent-relay.md) or a prefix kept in VRAM
//! ([`PinnedPrefix`], phase B). Both seed the same freshly allocated working cache
//! and then share every later step: the file path uploads host-to-device straight
//! into it, the pinned path copies device-to-device into it. A pinned prefix is
//! never written by a request; each use gets its own copy.

use super::*;
use crate::kv_io::{DenseKvCache, HybridKvCache, ImportedKv, KvDtype, KvValues, MlaKvCache};
use std::cell::Cell;

/// What a generate call resumes from. Anything but `Fresh` makes `prompt` the
/// continuation after the cached positions, with no BOS added.
#[derive(Clone, Copy)]
pub enum Resume<'a> {
    /// No prefix: `prompt` starts a new sequence.
    Fresh,
    /// A cache file's contents (`crate::kv_io::import_kv`).
    File(&'a ImportedKv),
    /// A prefix kept in VRAM by [`Model::generate_pin_prefix`].
    Pinned(&'a PinnedPrefix),
}

impl Resume<'_> {
    fn format_name(&self) -> &'static str {
        match self {
            Resume::Fresh => "none",
            Resume::File(kv) => kv.format_name(),
            Resume::Pinned(p) => p.format_name(),
        }
    }
}

/// A prompt's model state kept in VRAM: what `pin_prefix` holds in the IPC
/// request loop's one slot (`crate::ipc::PrefixSlot`). It owns the device
/// buffers the pinning request wrote, so pinning copies nothing; each use
/// copies them ([`Resume::Pinned`]). Freed when dropped.
pub struct PinnedPrefix {
    pub(super) state: DeviceState,
}

/// A model's state after `seq_len` positions, still on the device. The buffers
/// may be longer than `seq_len` positions (the generate call's headroom); only
/// the first `seq_len` are meaningful.
pub(super) enum DeviceState {
    Dense(DenseDeviceKv),
    Hybrid(HybridDeviceState),
    Mla(MlaDeviceKv),
}

pub(super) struct DenseDeviceKv {
    pub(super) seq_len: usize,
    pub(super) k_caches: Vec<KvBuf>,
    pub(super) v_caches: Vec<KvBuf>,
    /// Whether the cached prompt's encoding ended on a special token: the
    /// `after_special` a continuation is encoded with
    /// (`Tokenizer::encode_continuation`). Only the SentencePiece scheme, which
    /// only the dense path runs, depends on it.
    pub(super) space_prefix_next: bool,
}

pub(super) struct HybridDeviceState {
    pub(super) seq_len: usize,
    pub(super) layers: Vec<HybridLayerState>,
}

pub(super) struct MlaDeviceKv {
    pub(super) seq_len: usize,
    pub(super) kv_caches: Vec<KvBuf>,
}

impl PinnedPrefix {
    /// Positions held (the prompt that was pinned, plus any prefix it resumed from).
    pub fn seq_len(&self) -> usize {
        match &self.state {
            DeviceState::Dense(d) => d.seq_len,
            DeviceState::Hybrid(d) => d.seq_len,
            DeviceState::Mla(d) => d.seq_len,
        }
    }

    /// VRAM held, in bytes.
    pub fn device_bytes(&self) -> usize {
        let kv = |c: &KvBuf| c.len() * c.dtype().size();
        let f32s = std::mem::size_of::<f32>();
        match &self.state {
            DeviceState::Dense(d) => d.k_caches.iter().chain(&d.v_caches).map(kv).sum(),
            DeviceState::Hybrid(d) => d
                .layers
                .iter()
                .map(|l| match l {
                    HybridLayerState::Attn { k_cache, v_cache } => kv(k_cache) + kv(v_cache),
                    HybridLayerState::Gdn {
                        conv_state,
                        recurrent,
                    } => (conv_state.len() + recurrent.len()) * f32s,
                })
                .sum(),
            DeviceState::Mla(d) => d.kv_caches.iter().map(kv).sum(),
        }
    }

    fn format_name(&self) -> &'static str {
        match &self.state {
            DeviceState::Dense(_) => "dense/MoE",
            DeviceState::Hybrid(_) => "hybrid",
            DeviceState::Mla(_) => "MLA",
        }
    }
}

/// One architecture's view of a [`Resume`]: where the working cache's first
/// `seq_len` positions come from. `copy_ms` receives the time it took to fill
/// them: the device-to-device copy of a pinned prefix, or a cache file's read
/// and upload (`WeightLoadPipeline::upload_state`).
pub(super) enum PrefixSource<'a, H, D> {
    Host {
        cache: &'a H,
        copy_ms: &'a Cell<Option<f64>>,
    },
    Device {
        state: &'a D,
        copy_ms: &'a Cell<Option<f64>>,
    },
}

pub(super) type DensePrefix<'a> = PrefixSource<'a, DenseKvCache, DenseDeviceKv>;
pub(super) type HybridPrefix<'a> = PrefixSource<'a, HybridKvCache, HybridDeviceState>;
pub(super) type MlaPrefix<'a> = PrefixSource<'a, MlaKvCache, MlaDeviceKv>;

impl<H, D> Clone for PrefixSource<'_, H, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<H, D> Copy for PrefixSource<'_, H, D> {}

pub(super) trait SeqLen {
    fn seq_len(&self) -> usize;
}
macro_rules! impl_seq_len {
    ($($t:ty),*) => {$(impl SeqLen for $t { fn seq_len(&self) -> usize { self.seq_len } })*};
}
impl_seq_len!(
    DenseKvCache,
    HybridKvCache,
    MlaKvCache,
    DenseDeviceKv,
    HybridDeviceState,
    MlaDeviceKv
);

impl<H: SeqLen, D: SeqLen> PrefixSource<'_, H, D> {
    pub(super) fn seq_len(&self) -> usize {
        match self {
            PrefixSource::Host { cache, .. } => cache.seq_len(),
            PrefixSource::Device { state, .. } => state.seq_len(),
        }
    }
}

impl DensePrefix<'_> {
    /// The `after_special` state to encode a continuation of this prefix with. A
    /// version-1 cache file doesn't record it; resuming from one adds the
    /// SentencePiece space, as before version 4.
    pub(super) fn space_prefix_next(&self) -> bool {
        match self {
            PrefixSource::Host { cache, .. } => cache.space_prefix_next.unwrap_or(true),
            PrefixSource::Device { state, .. } => state.space_prefix_next,
        }
    }
}

/// The `after_special` state a dense prompt resumed from `src` (fresh: `None`)
/// is encoded with.
pub(super) fn dense_after_special(src: Option<DensePrefix<'_>>) -> bool {
    src.is_none_or(|s| s.space_prefix_next())
}

/// Copies a device state's buffers to the host for a cache file. f32 values
/// are copied as they are. f16 values are rounded on the device first
/// (`cast_act_f16_kernel`: round to nearest even, saturating at +-65504 and
/// counting what saturated) and downloaded as f16 bits, half the bytes, with no
/// host conversion loop. [`Self::finish`] fails the export if anything
/// saturated: K/V values that large would not survive f16 storage.
pub(super) struct StateDownload<'m> {
    model: &'m Model,
    /// `cast_act_f16_kernel`'s `[max |x| bits, saturated count]`, f16 only.
    stats: Option<CudaSlice<u32>>,
}

impl<'m> StateDownload<'m> {
    pub(super) fn new(model: &'m Model, dtype: KvDtype) -> Result<Self, ReflexError> {
        let stats = match dtype {
            KvDtype::F32 => None,
            KvDtype::F16 => Some(
                model
                    .device
                    .alloc_zeros::<u32>(2)
                    .map_err(|e| crate::gpu_err!(e, "alloc f16 export stats: {e}"))?,
            ),
        };
        Ok(Self { model, stats })
    }

    /// `src[..n]` as `dtype` (f32 for a buffer that always stays f32, such as
    /// the GatedDeltaNet state).
    pub(super) fn get(
        &mut self,
        src: &CudaSlice<f32>,
        n: usize,
        dtype: KvDtype,
        what: &str,
    ) -> Result<KvValues, ReflexError> {
        let device = &self.model.device;
        match (dtype, self.stats.as_mut()) {
            (KvDtype::F16, Some(stats)) => {
                let mut out = unsafe { device.alloc::<u16>(n) }
                    .map_err(|e| crate::gpu_err!(e, "{what}: alloc f16 export buffer: {e}"))?;
                unsafe {
                    self.model
                        .dequant_kernels
                        .cast_act_f16
                        .function
                        .clone()
                        .launch(
                            elementwise_launch_cfg(n),
                            (&src.slice(0..n), &mut out, n as u64, &mut *stats),
                        )
                }
                .map_err(|e| crate::gpu_err!(e, "{what}: f16 export cast: {e}"))?;
                device
                    .dtoh_sync_copy(&out)
                    .map(KvValues::HostF16)
                    .map_err(|e| crate::gpu_err!(e, "{what} dtoh: {e}"))
            }
            _ => device
                .dtoh_sync_copy(&src.slice(0..n))
                .map(KvValues::Host)
                .map_err(|e| crate::gpu_err!(e, "{what} dtoh: {e}")),
        }
    }

    /// [`Self::get`] for a K/V cache buffer, f32 or f16 (`REFLEX_KV_CACHE`): an f16
    /// cache downloads its values as they are into an f16 file and widens them
    /// exactly (`f16_to_f32_kernel`, on the device) into an f32 one.
    pub(super) fn get_kv(
        &mut self,
        src: &KvBuf,
        n: usize,
        dtype: KvDtype,
        what: &str,
    ) -> Result<KvValues, ReflexError> {
        let s = match src {
            KvBuf::F32(s) => return self.get(s, n, dtype, what),
            KvBuf::F16(s) => s,
        };
        let device = &self.model.device;
        match dtype {
            KvDtype::F16 => device
                .dtoh_sync_copy(&s.slice(0..n))
                .map(|v| KvValues::HostF16(v.into_iter().map(half::f16::to_bits).collect()))
                .map_err(|e| crate::gpu_err!(e, "{what} dtoh: {e}")),
            KvDtype::F32 => {
                let mut out = unsafe { device.alloc::<f32>(n) }
                    .map_err(|e| crate::gpu_err!(e, "{what}: alloc f32 export buffer: {e}"))?;
                unsafe {
                    self.model
                        .dequant_kernels
                        .f16_to_f32
                        .function
                        .clone()
                        .launch(
                            elementwise_launch_cfg(n),
                            (&s.slice(0..n), &mut out, n as u64),
                        )
                }
                .map_err(|e| crate::gpu_err!(e, "{what}: f16 widen: {e}"))?;
                device
                    .dtoh_sync_copy(&out)
                    .map(KvValues::Host)
                    .map_err(|e| crate::gpu_err!(e, "{what} dtoh: {e}"))
            }
        }
    }

    /// Errors if any value rounded to f16 was outside its range.
    pub(super) fn finish(self) -> Result<(), ReflexError> {
        let Some(stats) = &self.stats else {
            return Ok(());
        };
        let s = self
            .model
            .device
            .dtoh_sync_copy(stats)
            .map_err(|e| crate::gpu_err!(e, "f16 export stats dtoh: {e}"))?;
        if s[1] > 0 {
            return Err(crate::reflex_err!(
                KvCache,
                "an f16 cache file can't hold this state: {} values exceed f16's range (largest |x| = {}); export it as f32",
                s[1],
                f32::from_bits(s[0])
            ));
        }
        Ok(())
    }
}

/// Errors unless an imported buffer holds exactly the `expected` values the
/// working cache's prefix region does.
pub(super) fn check_state_len(
    values: &crate::kv_io::KvValues,
    expected: usize,
    what: &str,
    layer_idx: usize,
) -> Result<(), ReflexError> {
    if values.len() != expected {
        return Err(crate::reflex_err!(
            KvCache,
            "imported {what} for layer {layer_idx} has {} values, expected {expected}",
            values.len()
        ));
    }
    Ok(())
}

/// `Option<PrefixSource>`'s start position: 0 for a fresh prompt.
pub(super) fn start_pos_of<H: SeqLen, D: SeqLen>(src: Option<PrefixSource<'_, H, D>>) -> usize {
    src.map_or(0, |s| s.seq_len())
}

/// What [`Model::run_resumed`] hands back: the generated tokens, the final
/// device state (none for architectures without a cache format), and the
/// pinned-prefix copy time.
struct Resumed {
    token_ids: Vec<u32>,
    text: String,
    state: Option<DeviceState>,
    copy_ms: Option<f64>,
}

impl Model {
    /// Runs `copies` (copies into the working cache on the default stream),
    /// waits for them, and records how long that took in `copy_ms`. Work
    /// already queued is waited for first, outside the timing: in a cold
    /// process the weight load's last kernels can still be running.
    pub(super) fn timed_prefix_copy(
        &self,
        copy_ms: &Cell<Option<f64>>,
        copies: impl FnOnce() -> Result<(), ReflexError>,
    ) -> Result<(), ReflexError> {
        self.device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "sync before prefix copy: {e}"))?;
        let start = std::time::Instant::now();
        copies()?;
        self.device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "prefix copy sync: {e}"))?;
        copy_ms.set(Some(start.elapsed().as_secs_f64() * 1000.0));
        Ok(())
    }

    /// Errors unless this model has a prefix-state format (cache files and
    /// pinned prefixes): Gemma 4 and granitehybrid don't yet.
    pub fn check_prefix_state_supported(&self) -> Result<(), ReflexError> {
        if self.gemma4.is_some() {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "KV cache export/import and pinned prefixes are not supported for Gemma 4 models yet"
            ));
        }
        if self.granite.is_some() {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "KV cache export/import and pinned prefixes are not supported for granitehybrid (Granite 4.0-H) models yet"
            ));
        }
        Ok(())
    }

    /// Like [`Model::generate`], resuming from any [`Resume`] source. Also
    /// returns the time spent copying a pinned prefix (`None` otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_resumed(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        resume: Resume<'_>,
        sampling: &SamplingParams,
        on_first_token: impl FnMut(&[f32]),
        on_token: impl FnMut(u32, &str),
    ) -> Result<(Vec<u32>, String, Option<f64>), ReflexError> {
        let r = self.run_resumed(
            prompt,
            max_new_tokens,
            resume,
            sampling,
            on_first_token,
            on_token,
        )?;
        Ok((r.token_ids, r.text, r.copy_ms))
    }

    /// Generates one token after `prompt` (resuming from `resume`) and keeps the
    /// model's state right after the prompt in VRAM. One token for the same
    /// reason as [`Model::generate_capture_kv`]: the Qwen3.5 hybrid's recurrent
    /// state can't be rolled back past later tokens. Also returns the time spent
    /// copying a pinned `resume` (`None` otherwise). Gemma 4 and granitehybrid
    /// return `UnsupportedArchitecture`.
    pub fn generate_pin_prefix(
        &self,
        prompt: &str,
        resume: Resume<'_>,
        sampling: &SamplingParams,
        on_first_token: impl FnMut(&[f32]),
        on_token: impl FnMut(u32, &str),
    ) -> Result<(Vec<u32>, String, PinnedPrefix, Option<f64>), ReflexError> {
        self.check_prefix_state_supported()?;
        if sampling.json_schema.is_some() {
            return Err(ReflexError::InvalidInput(
                "json_schema can't be combined with pinning or exporting a prefix: that keeps the state after the prompt, before any value is generated".to_string(),
            ));
        }
        let r = self.run_resumed(prompt, 1, resume, sampling, on_first_token, on_token)?;
        let state = r.state.ok_or_else(|| {
            ReflexError::Other("internal error: no device state to pin".to_string())
        })?;
        Ok((r.token_ids, r.text, PinnedPrefix { state }, r.copy_ms))
    }

    /// Downloads a pinned prefix into the host form `crate::kv_io::export_kv`
    /// writes, in the working cache's own type ([`Model::kv_cache_dtype`]: an f16
    /// cache gives an f16 file, as is). This is how [`Model::generate_capture_kv`]
    /// builds its result.
    pub fn download_prefix(&self, prefix: &PinnedPrefix) -> Result<ImportedKv, ReflexError> {
        self.download_prefix_as(prefix, self.kv_cache_dtype())
    }

    /// [`Self::download_prefix`] for a cache file of element type `dtype`: with
    /// f16 the values are rounded on the device (see [`StateDownload`]), and
    /// `export_kv` then writes the f16 format version.
    pub fn download_prefix_as(
        &self,
        prefix: &PinnedPrefix,
        dtype: KvDtype,
    ) -> Result<ImportedKv, ReflexError> {
        Ok(match &prefix.state {
            DeviceState::Dense(d) => ImportedKv::Dense(self.download_dense_kv(d, dtype)?),
            DeviceState::Hybrid(d) => {
                let h = self.hybrid.as_ref().ok_or_else(|| {
                    ReflexError::Other("hybrid prefix on a non-hybrid model".to_string())
                })?;
                ImportedKv::Hybrid(self.download_hybrid_state(h, d, dtype)?)
            }
            DeviceState::Mla(d) => {
                let m = self.mla.as_ref().ok_or_else(|| {
                    ReflexError::Other("MLA prefix on a non-MLA model".to_string())
                })?;
                ImportedKv::Mla(self.download_mla_kv(m, d, dtype)?)
            }
        })
    }

    /// The dispatcher behind [`Model::generate`], [`Model::generate_resumed`],
    /// [`Model::generate_pin_prefix`] and [`Model::generate_capture_kv`]: picks
    /// the architecture's generate function, maps `resume` onto its
    /// [`PrefixSource`], and wraps the final caches as a [`DeviceState`] (a move,
    /// not a copy).
    fn run_resumed(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        resume: Resume<'_>,
        sampling: &SamplingParams,
        on_first_token: impl FnMut(&[f32]),
        on_token: impl FnMut(u32, &str),
    ) -> Result<Resumed, ReflexError> {
        let copy_ms = Cell::new(None);
        if self.gemma4.is_some() || self.granite.is_some() {
            if !matches!(resume, Resume::Fresh) {
                self.check_prefix_state_supported()?;
            }
            let (token_ids, text) = if let Some(g) = &self.gemma4 {
                self.generate_gemma4_impl(
                    g,
                    prompt,
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?
            } else {
                let g = self.granite.as_ref().expect("checked above");
                self.generate_granite_impl(
                    g,
                    prompt,
                    max_new_tokens,
                    sampling,
                    on_first_token,
                    on_token,
                )?
            };
            return Ok(Resumed {
                token_ids,
                text,
                state: None,
                copy_ms: None,
            });
        }

        if let Some(h) = &self.hybrid {
            let src = self.hybrid_prefix(resume, &copy_ms)?;
            let (token_ids, text, layers, seq_len) = self.generate_hybrid_impl(
                h,
                prompt,
                src,
                max_new_tokens,
                sampling,
                on_first_token,
                on_token,
            )?;
            return Ok(Resumed {
                token_ids,
                text,
                state: Some(DeviceState::Hybrid(HybridDeviceState { seq_len, layers })),
                copy_ms: copy_ms.get(),
            });
        }
        if let Some(m) = &self.mla {
            let src = self.mla_prefix(resume, &copy_ms)?;
            let (token_ids, text, kv_caches, seq_len) = self.generate_mla_impl(
                m,
                prompt,
                src,
                max_new_tokens,
                sampling,
                on_first_token,
                on_token,
            )?;
            return Ok(Resumed {
                token_ids,
                text,
                state: Some(DeviceState::Mla(MlaDeviceKv { seq_len, kv_caches })),
                copy_ms: copy_ms.get(),
            });
        }
        let src = self.dense_prefix(resume, &copy_ms)?;
        let space_prefix_next = self
            .tokenizer
            .ends_after_special(prompt, dense_after_special(src));
        let (token_ids, text, k_caches, v_caches, seq_len) = self.generate_dense_impl(
            prompt,
            src,
            max_new_tokens,
            sampling,
            on_first_token,
            on_token,
        )?;
        Ok(Resumed {
            token_ids,
            text,
            state: Some(DeviceState::Dense(DenseDeviceKv {
                seq_len,
                k_caches,
                v_caches,
                space_prefix_next,
            })),
            copy_ms: copy_ms.get(),
        })
    }

    /// Like [`Model::system1_evaluate`], scoring the candidates after `prompt`
    /// resumed from `resume` (`prompt` is then the continuation, with no BOS
    /// added). Pinned prefixes and cache files share one path, as for
    /// [`Model::generate_resumed`]. Also returns the time spent copying a pinned
    /// prefix (`None` otherwise). A resumed `prompt` must still add at least one
    /// token: the scores come from its last position. On the Qwen3.5 hybrid each
    /// candidate must be a single token, as for a fresh prompt; Gemma 4 and
    /// granitehybrid return `UnsupportedArchitecture` for anything but
    /// [`Resume::Fresh`].
    pub fn system1_evaluate_resumed(
        &self,
        prompt: &str,
        resume: Resume<'_>,
        candidates: &[System1Candidate],
        temperature: f32,
    ) -> Result<(System1Response, Option<f64>), ReflexError> {
        let copy_ms = Cell::new(None);
        let response = if let Some(h) = &self.hybrid {
            let src = self.hybrid_prefix(resume, &copy_ms)?;
            self.system1_evaluate_hybrid(h, prompt, src, candidates, temperature)?
        } else if let Some(m) = &self.mla {
            let src = self.mla_prefix(resume, &copy_ms)?;
            self.system1_evaluate_mla(m, prompt, src, candidates, temperature)?
        } else if self.gemma4.is_some() || self.granite.is_some() {
            if !matches!(resume, Resume::Fresh) {
                self.check_prefix_state_supported()?;
            }
            if let Some(g) = &self.gemma4 {
                self.system1_evaluate_gemma4(g, prompt, candidates, temperature)?
            } else {
                let g = self.granite.as_ref().expect("checked above");
                self.system1_evaluate_granite(g, prompt, candidates, temperature)?
            }
        } else {
            let src = self.dense_prefix(resume, &copy_ms)?;
            self.system1_evaluate_dense(prompt, src, candidates, temperature)?
        };
        Ok((response, copy_ms.get()))
    }

    /// The error for a `resume` whose state is another architecture's format.
    fn prefix_format_mismatch(&self, resume: Resume<'_>) -> ReflexError {
        let model_format = if self.hybrid.is_some() {
            "hybrid"
        } else if self.mla.is_some() {
            "MLA"
        } else {
            "dense/MoE"
        };
        let what = match resume {
            Resume::Pinned(_) => "pinned prefix",
            _ => "imported KV cache file",
        };
        ReflexError::KvCache(format!(
            "{what} is {} format, but this model uses the {model_format} format",
            resume.format_name()
        ))
    }

    /// `resume` as the dense/MoE path's [`PrefixSource`].
    fn dense_prefix<'a>(
        &self,
        resume: Resume<'a>,
        copy_ms: &'a Cell<Option<f64>>,
    ) -> Result<Option<DensePrefix<'a>>, ReflexError> {
        Ok(match resume {
            Resume::Fresh => None,
            Resume::File(ImportedKv::Dense(cache)) => Some(PrefixSource::Host { cache, copy_ms }),
            Resume::Pinned(PinnedPrefix {
                state: DeviceState::Dense(d),
            }) => Some(PrefixSource::Device { state: d, copy_ms }),
            _ => return Err(self.prefix_format_mismatch(resume)),
        })
    }

    /// `resume` as the Qwen3.5 hybrid path's [`PrefixSource`].
    fn hybrid_prefix<'a>(
        &self,
        resume: Resume<'a>,
        copy_ms: &'a Cell<Option<f64>>,
    ) -> Result<Option<HybridPrefix<'a>>, ReflexError> {
        Ok(match resume {
            Resume::Fresh => None,
            Resume::File(ImportedKv::Hybrid(cache)) => Some(PrefixSource::Host { cache, copy_ms }),
            Resume::Pinned(PinnedPrefix {
                state: DeviceState::Hybrid(d),
            }) => Some(PrefixSource::Device { state: d, copy_ms }),
            _ => return Err(self.prefix_format_mismatch(resume)),
        })
    }

    /// `resume` as the MLA path's [`PrefixSource`].
    fn mla_prefix<'a>(
        &self,
        resume: Resume<'a>,
        copy_ms: &'a Cell<Option<f64>>,
    ) -> Result<Option<MlaPrefix<'a>>, ReflexError> {
        Ok(match resume {
            Resume::Fresh => None,
            Resume::File(ImportedKv::Mla(cache)) => Some(PrefixSource::Host { cache, copy_ms }),
            Resume::Pinned(PinnedPrefix {
                state: DeviceState::Mla(d),
            }) => Some(PrefixSource::Device { state: d, copy_ms }),
            _ => return Err(self.prefix_format_mismatch(resume)),
        })
    }
}
