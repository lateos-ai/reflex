//! The working K/V cache's element type. f32 by default; `REFLEX_KV_CACHE=f16` stores
//! every attention layer's K and V (and MLA's compressed latent) as f16, halving the
//! bytes decode attention reads and the cache's VRAM, at the cost of rounding each
//! cached value to 11 significant bits. It is chosen once per process, at load: a
//! pinned prefix (`crate::ipc::PrefixSlot`) holds its state in the same type and every
//! request uses it as is. Activations, attention scores and accumulators stay f32; the
//! kernels widen each K/V element as they load it (`kernels_cuda/kv_load.cuh`), and
//! every cache write rounds to nearest even, saturating at +-65504 like f16 cache-file
//! export. Qwen3.5's GatedDeltaNet state stays f32, as in f16 cache files.
//!
//! The f16 kernels live in modules of their own (`kv_f16.cu`, `attention_decode_f16.cu`,
//! `attention_tiled*_f16.cu`), loaded the first time an f16 cache needs them, so the
//! default f32 build loads and JIT-compiles exactly what it did before. Gemma 4,
//! Granite 4.0-H and `REFLEX_ATTN_KERNEL=legacy` reject `REFLEX_KV_CACHE=f16` at load.
//! Design and accuracy: docs/design/decode-speed.md, "f16 KV cache".

use super::*;
use crate::kv_io::KvDtype;

/// `REFLEX_KV_CACHE`: `f32` (the default) or `f16`.
pub(super) fn kv_dtype_from_env() -> Result<KvDtype, ReflexError> {
    match std::env::var("REFLEX_KV_CACHE") {
        Ok(v) if !v.is_empty() => {
            KvDtype::parse(&v).map_err(|e| e.rewrap("REFLEX_KV_CACHE".to_string()))
        }
        _ => Ok(KvDtype::F32),
    }
}

/// One layer's K or V cache (or MLA latent cache) on the device, in the model's
/// [`KvDtype`].
pub(super) enum KvBuf {
    F32(CudaSlice<f32>),
    F16(CudaSlice<half::f16>),
}

/// A range of a [`KvBuf`], what the attention kernels read.
pub(super) enum KvView<'a> {
    F32(CudaView<'a, f32>),
    F16(CudaView<'a, half::f16>),
}

impl KvBuf {
    pub(super) fn len(&self) -> usize {
        match self {
            KvBuf::F32(s) => s.len(),
            KvBuf::F16(s) => s.len(),
        }
    }

    pub(super) fn dtype(&self) -> KvDtype {
        match self {
            KvBuf::F32(_) => KvDtype::F32,
            KvBuf::F16(_) => KvDtype::F16,
        }
    }

    /// The device address of element `i`.
    pub(super) fn ptr_at(&self, i: usize) -> u64 {
        let base = match self {
            KvBuf::F32(s) => *s.device_ptr(),
            KvBuf::F16(s) => *s.device_ptr(),
        };
        base + (i * self.dtype().size()) as u64
    }

    pub(super) fn view(&self, range: std::ops::Range<usize>) -> KvView<'_> {
        match self {
            KvBuf::F32(s) => KvView::F32(s.slice(range)),
            KvBuf::F16(s) => KvView::F16(s.slice(range)),
        }
    }

    /// The f32 buffer (tests that round an f32 cache in place).
    #[cfg(test)]
    pub(super) fn f32_mut(&mut self) -> Result<&mut CudaSlice<f32>, ReflexError> {
        match self {
            KvBuf::F32(s) => Ok(s),
            KvBuf::F16(_) => Err(crate::reflex_err!(
                Other,
                "internal: an f32-only path got an f16 KV cache"
            )),
        }
    }
}

impl<'a> KvView<'a> {
    pub(super) fn dtype(&self) -> KvDtype {
        match self {
            KvView::F32(_) => KvDtype::F32,
            KvView::F16(_) => KvDtype::F16,
        }
    }

    /// The f32 view, for the legacy attention kernels (f32 caches only; an f16
    /// cache is rejected at load with `REFLEX_ATTN_KERNEL=legacy`).
    pub(super) fn f32(&self) -> Result<&CudaView<'a, f32>, ReflexError> {
        match self {
            KvView::F32(v) => Ok(v),
            KvView::F16(_) => Err(crate::reflex_err!(
                Other,
                "internal: the legacy attention kernels got an f16 KV cache"
            )),
        }
    }

    /// The elements from `start` on.
    pub(super) fn tail(&self, start: usize) -> KvView<'_> {
        match self {
            KvView::F32(v) => KvView::F32(v.slice(start..)),
            KvView::F16(v) => KvView::F16(v.slice(start..)),
        }
    }
}

/// `kv_f16.cu`, loaded the first time an f16 cache is written or read by online
/// attention (see the module docs).
pub(super) struct KvF16Kernels {
    kernels: std::sync::OnceLock<Vec<AotKernel>>,
}

const KV_F16_NAMES: [&str; 4] = [
    "attention_online_f16_kernel",
    "kv_store_f16_kernel",
    "mla_write_kv_cache_batch_f16_kernel",
    "kv_store_kv_f16_kernel",
];
pub(super) const KV_F16_ATTN_ONLINE: usize = 0;
pub(super) const KV_F16_STORE: usize = 1;
pub(super) const KV_F16_MLA_WRITE: usize = 2;
const KV_F16_STORE_KV: usize = 3;

impl KvF16Kernels {
    pub(super) fn new() -> Self {
        Self {
            kernels: std::sync::OnceLock::new(),
        }
    }

    pub(super) fn kernel(
        &self,
        device: &Arc<CudaDevice>,
        index: usize,
    ) -> Result<&AotKernel, ReflexError> {
        if let Some(k) = self.kernels.get() {
            return Ok(&k[index]);
        }
        let loaded = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_KV_F16")),
            "kv_f16",
            &KV_F16_NAMES,
        )?;
        Ok(&self.kernels.get_or_init(|| loaded)[index])
    }

    /// Whether the module has been loaded (tests).
    #[cfg(test)]
    pub(super) fn loaded(&self) -> bool {
        self.kernels.get().is_some()
    }
}

impl Model {
    /// The working K/V cache's element type (`REFLEX_KV_CACHE`, read at load).
    pub fn kv_cache_dtype(&self) -> KvDtype {
        self.kv_dtype.get()
    }

    /// Switches the cache type of a loaded model, for tests that compare the two on
    /// one model. Only between calls: nothing may hold a cache of the old type.
    #[cfg(test)]
    pub(super) fn set_kv_cache_dtype(&self, dtype: KvDtype) {
        self.kv_dtype.set(dtype);
    }

    /// Errors unless this model's architecture and attention kernels support an f16
    /// cache (see the module docs). Called at load and by tests that switch types.
    pub(super) fn check_kv_dtype_supported(&self) -> Result<(), ReflexError> {
        if self.kv_dtype.get() == KvDtype::F32 {
            return Ok(());
        }
        let unsupported = if self.gemma4.is_some() {
            Some("Gemma 4")
        } else if self.granite.is_some() {
            Some("Granite 4.0-H")
        } else if self.attn_impl == AttnImpl::Legacy {
            Some("REFLEX_ATTN_KERNEL=legacy")
        } else if self.encoder.is_some() {
            Some("embedding models")
        } else {
            None
        };
        match unsupported {
            Some(what) => Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "REFLEX_KV_CACHE=f16 is not supported with {what}; use the default f32 cache"
            )),
            None => Ok(()),
        }
    }

    /// A zeroed K/V cache buffer of `len` elements in the model's cache type.
    pub(super) fn alloc_kv(&self, len: usize, what: &str) -> Result<KvBuf, ReflexError> {
        match self.kv_dtype.get() {
            KvDtype::F32 => self.device.alloc_zeros::<f32>(len).map(KvBuf::F32),
            KvDtype::F16 => self.device.alloc_zeros::<half::f16>(len).map(KvBuf::F16),
        }
        .map_err(|e| crate::gpu_err!(e, "alloc {what}: {e}"))
    }

    /// Writes the `n` f32 values at device address `src` into `dst` from element
    /// `at` on: a device copy into an f32 cache, a rounding store
    /// (`kv_store_f16_kernel`) into an f16 one.
    pub(super) fn kv_store(
        &self,
        src: u64,
        dst: &KvBuf,
        at: usize,
        n: usize,
    ) -> Result<(), ReflexError> {
        if at + n > dst.len() {
            return Err(crate::reflex_err!(
                Other,
                "internal: KV cache write of {n} values at {at} past its {} elements",
                dst.len()
            ));
        }
        let to = dst.ptr_at(at);
        match dst {
            KvBuf::F32(_) => unsafe {
                result::memcpy_dtod_async(to, src, n * 4, *self.device.cu_stream())
                    .map_err(|e| crate::gpu_err!(e, "KV cache write: {e}"))
            },
            KvBuf::F16(_) => {
                let kernel = self.kv_f16.kernel(&self.device, KV_F16_STORE)?;
                unsafe {
                    kernel
                        .function
                        .clone()
                        .launch(elementwise_launch_cfg(n), (src, to, n as u64))
                        .map_err(|e| crate::gpu_err!(e, "kv_store_f16 launch: {e}"))
                }
            }
        }
    }

    /// [`Self::kv_store`] for one layer's K and V, `n` values each at element `at`
    /// of `k_dst`/`v_dst` (same type): two device copies into an f32 cache, one
    /// `kv_store_kv_f16_kernel` launch into an f16 one.
    pub(super) fn kv_store_kv(
        &self,
        k_src: u64,
        v_src: u64,
        k_dst: &KvBuf,
        v_dst: &KvBuf,
        at: usize,
        n: usize,
    ) -> Result<(), ReflexError> {
        if k_dst.dtype() == KvDtype::F32 || k_dst.dtype() != v_dst.dtype() {
            self.kv_store(k_src, k_dst, at, n)?;
            return self.kv_store(v_src, v_dst, at, n);
        }
        if at + n > k_dst.len().min(v_dst.len()) {
            return Err(crate::reflex_err!(
                Other,
                "internal: KV cache write of {n} values at {at} past its {} elements",
                k_dst.len().min(v_dst.len())
            ));
        }
        let kernel = self.kv_f16.kernel(&self.device, KV_F16_STORE_KV)?;
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    elementwise_launch_cfg(2 * n),
                    (k_src, v_src, k_dst.ptr_at(at), v_dst.ptr_at(at), n as u64),
                )
                .map_err(|e| crate::gpu_err!(e, "kv_store_kv_f16 launch: {e}"))
        }
    }

    /// Copies `dst`'s first `n` elements from `src` (same type): how a pinned prefix
    /// seeds a request's working cache.
    pub(super) fn kv_copy_prefix(
        &self,
        src: &KvBuf,
        dst: &mut KvBuf,
        n: usize,
        what: &str,
    ) -> Result<(), ReflexError> {
        if src.dtype() != dst.dtype() || n > src.len() || n > dst.len() {
            return Err(crate::reflex_err!(
                Other,
                "internal: {what}: copying {n} {} values into an {} cache of {}",
                src.dtype().as_str(),
                dst.dtype().as_str(),
                dst.len()
            ));
        }
        unsafe {
            result::memcpy_dtod_async(
                dst.ptr_at(0),
                src.ptr_at(0),
                n * src.dtype().size(),
                *self.device.cu_stream(),
            )
        }
        .map_err(|e| crate::gpu_err!(e, "{what}: {e}"))
    }
}

impl Model {
    /// `kv_store_f16_kernel` when the cache is f16 (what `upload_state` rounds an
    /// f32 cache file's values with), else `None`.
    pub(super) fn kv_narrow_kernel(&self) -> Result<Option<&AotKernel>, ReflexError> {
        match self.kv_dtype.get() {
            KvDtype::F32 => Ok(None),
            KvDtype::F16 => self.kv_f16.kernel(&self.device, KV_F16_STORE).map(Some),
        }
    }
}
