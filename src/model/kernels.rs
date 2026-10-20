//! Thin launch wrappers around the AOT-compiled CUDA kernels and cuBLAS GEMMs.

use super::*;
use crate::error::ReflexError;

/// Which attention kernels a model launches, read once at load time from the
/// `REFLEX_ATTN_KERNEL` environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttnImpl {
    /// `attention_online.cu` (the default): online softmax, warp-parallel over
    /// positions, split-K for long decode contexts.
    Online,
    /// The original per-shape kernels (`attention.cu`, `attention_prefill.cu`,
    /// `mla_attention*.cu`), kept for A/B comparison. Limited to
    /// [`crate::limits::LEGACY_ATTN_MAX_POSITIONS`].
    Legacy,
}

impl AttnImpl {
    pub(super) fn from_env() -> Result<Self, ReflexError> {
        match std::env::var("REFLEX_ATTN_KERNEL").as_deref() {
            Err(_) | Ok("") | Ok("online") => Ok(AttnImpl::Online),
            Ok("legacy") => Ok(AttnImpl::Legacy),
            Ok(other) => Err(crate::reflex_err!(
                InvalidInput,
                "REFLEX_ATTN_KERNEL must be `online` (the default) or `legacy`, got {other:?}"
            )),
        }
    }

    /// The most sequence positions these kernels support.
    pub(super) fn max_positions(self) -> usize {
        match self {
            AttnImpl::Online => crate::limits::ATTN_MAX_POSITIONS,
            AttnImpl::Legacy => crate::limits::LEGACY_ATTN_MAX_POSITIONS,
        }
    }
}

/// `attention_online_kernel`'s scalar arguments. Must match `struct AttnParams` in
/// `kernels_cuda/attention_online.cu` (and `attention_tiled.cu`) field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct AttnParams {
    k_pos_stride: u64,
    k_head_stride: u64,
    v_pos_stride: u64,
    v_head_stride: u64,
    num_q_heads: u32,
    group_size: u32,
    qk_dim: u32,
    v_dim: u32,
    start_pos: u32,
    num_splits: u32,
    split_len: u32,
    window: u32,
    scale: f32,
}

// SAFETY: a plain-old-data #[repr(C)] struct, passed to the kernel by value.
unsafe impl DeviceRepr for AttnParams {}

/// Shape and memory layout of one [`Model::attention_online`] call; see
/// `kernels_cuda/attention_online.cu` for what each field means.
pub(super) struct OnlineAttnShape {
    pub(super) rows: usize,
    pub(super) start_pos: usize,
    pub(super) num_q_heads: usize,
    pub(super) group_size: usize,
    pub(super) qk_dim: usize,
    pub(super) v_dim: usize,
    pub(super) k_pos_stride: usize,
    pub(super) k_head_stride: usize,
    pub(super) v_pos_stride: usize,
    pub(super) v_head_stride: usize,
    pub(super) scale: f32,
    /// Sliding window: query position `i` sees key `j` iff `i - j < window`.
    /// `0` = plain causal attention.
    pub(super) window: usize,
}

impl OnlineAttnShape {
    /// The kernel arguments for this shape, unsplit (`num_splits` 1). For
    /// `attention_tiled.cu`, which ignores the split fields.
    pub(super) fn unsplit_params(&self) -> AttnParams {
        AttnParams {
            k_pos_stride: self.k_pos_stride as u64,
            k_head_stride: self.k_head_stride as u64,
            v_pos_stride: self.v_pos_stride as u64,
            v_head_stride: self.v_head_stride as u64,
            num_q_heads: self.num_q_heads as u32,
            group_size: self.group_size as u32,
            qk_dim: self.qk_dim as u32,
            v_dim: self.v_dim as u32,
            start_pos: self.start_pos as u32,
            num_splits: 1,
            split_len: (self.start_pos + self.rows) as u32,
            window: self.window as u32,
            scale: self.scale,
        }
    }
}

/// `attention_tiled*.cu`'s block size.
const ATTN_TILED_THREADS: u32 = 256;

/// Batched prefill runs [`Model::attention_prefill`] through the tiled kernels from
/// this many rows up, and through [`Model::attention_online`] below it (unless a
/// cached prefix makes the tiled kernels win earlier, [`ATTN_TILED_MIN_START`]). Short prompts
/// (and `system1`) then never load a tiled module, which a PTX build without a warm
/// driver cache JIT-compiles in ~200 ms. See docs/design/tiled-prefill-attention.md.
pub(super) const ATTN_TILED_MIN_ROWS: usize = 64;

/// A resumed prefill (rows after `start_pos` cached positions: a pinned prefix or an
/// imported cache) runs the tiled kernels below [`ATTN_TILED_MIN_ROWS`] too, once
/// at least this many positions are cached and `rows * group_size` fills one block
/// of 64 query slots ([`ATTN_TILED_MIN_QUERIES`]). The online kernel reads every
/// cached K/V row once per (row, query head), so its cost grows with the prefix:
/// on a T4, 41 rows after ~1,490 cached positions spent 90 ms in it on
/// Qwen3-0.6B, against 1.4 ms with nothing cached. See
/// `attention_tiled_crossover_timings_resumed` and
/// docs/design/tiled-prefill-attention.md.
pub(super) const ATTN_TILED_MIN_START: usize = 64;

/// See [`ATTN_TILED_MIN_START`].
pub(super) const ATTN_TILED_MIN_QUERIES: usize = 64;

/// Whether [`Model::attention_prefill`] uses the tiled kernels for `rows` rows after
/// `start_pos` cached positions with GQA group `group_size` (shape permitting).
pub(super) fn attn_tiled_preferred(rows: usize, start_pos: usize, group_size: usize) -> bool {
    rows >= ATTN_TILED_MIN_ROWS
        || (start_pos >= ATTN_TILED_MIN_START && rows * group_size >= ATTN_TILED_MIN_QUERIES)
}

/// Query vectors per block of the tiled kernel for `head_dim`, if there is one
/// (`kernels_cuda/attention_tiled.cuh`).
fn attn_tiled_queries_per_block(head_dim: usize) -> Option<usize> {
    match head_dim {
        64 | 128 | 256 => Some(64),
        512 => Some(32),
        _ => None,
    }
}

/// Whether the tiled kernels cover `s`: at least two rows, a head dim with a
/// kernel, `qk_dim == v_dim`, and a GQA group that is at most the queries per
/// block or a multiple of it.
pub(super) fn attn_tiled_fits(s: &OnlineAttnShape) -> bool {
    let Some(m) = attn_tiled_queries_per_block(s.qk_dim) else {
        return false;
    };
    let g = s.group_size;
    s.rows >= 2
        && s.v_dim == s.qk_dim
        && g != 0
        && s.num_q_heads.is_multiple_of(g)
        && (g <= m || g.is_multiple_of(m))
}

/// Launches `kernel`, a tiled prefill attention kernel for `s.qk_dim`, on a shape
/// [`attn_tiled_fits`] accepts. Same layout, masking and output as
/// [`Model::attention_online`]; only the f32 summation order differs.
pub(super) fn launch_attention_tiled(
    device: &Arc<CudaDevice>,
    kernel: &AotKernel,
    q: &CudaSlice<f32>,
    k: &KvView,
    v: &KvView,
    s: &OnlineAttnShape,
) -> Result<CudaSlice<f32>, ReflexError> {
    let m = attn_tiled_queries_per_block(s.qk_dim)
        .ok_or_else(|| crate::reflex_err!(Other, "no tiled attention kernel for d {}", s.qk_dim))?;
    let g = s.group_size;
    let heads_per_block = g.min(m);
    let rows_per_block = m / heads_per_block;
    let num_kv_heads = s.num_q_heads / g;
    let cfg = LaunchConfig {
        grid_dim: (
            s.rows.div_ceil(rows_per_block) as u32,
            (num_kv_heads * (g / heads_per_block)) as u32,
            1,
        ),
        block_dim: (ATTN_TILED_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out = device
        .alloc_zeros::<f32>(s.rows * s.num_q_heads * s.v_dim)
        .map_err(|e| crate::gpu_err!(e, "attn_tiled alloc out: {e}"))?;
    let params = s.unsplit_params();
    let rows = s.rows as u32;
    let f = kernel.function.clone();
    unsafe {
        match (k, v) {
            (KvView::F32(k), KvView::F32(v)) => f.launch(cfg, (q, k, v, &mut out, params, rows)),
            (KvView::F16(k), KvView::F16(v)) => f.launch(cfg, (q, k, v, &mut out, params, rows)),
            _ => return Err(kv_mixed_err()),
        }
        .map_err(|e| crate::gpu_err!(e, "attn_tiled launch: {e}"))?;
    }
    Ok(out)
}

fn kv_mixed_err() -> ReflexError {
    crate::reflex_err!(Other, "internal: attention got K and V of different types")
}

/// Launches an attention kernel built for the cache type of `k`/`v` (f32 or f16
/// views; the caller picked `kernel` by that type), with the same arguments either
/// way: `(q, k, v, out, params)`.
#[allow(clippy::too_many_arguments)]
fn launch_kv(
    kernel: &AotKernel,
    cfg: LaunchConfig,
    q: &CudaSlice<f32>,
    k: &KvView,
    v: &KvView,
    out: u64,
    params: AttnParams,
    what: &str,
) -> Result<(), ReflexError> {
    let f = kernel.function.clone();
    unsafe {
        match (k, v) {
            (KvView::F32(k), KvView::F32(v)) => f.launch(cfg, (q, k, v, out, params)),
            (KvView::F16(k), KvView::F16(v)) => f.launch(cfg, (q, k, v, out, params)),
            _ => return Err(kv_mixed_err()),
        }
    }
    .map_err(|e| crate::gpu_err!(e, "{what} launch: {e}"))
}

/// The tiled prefill attention kernels for [`Model::attention_prefill`], each
/// module loaded on the first prefill that needs it. `REFLEX_ATTN_TILED=0` turns
/// them off (every prefill uses [`Model::attention_online`], as before they existed)
/// for A/B comparison. Gemma 4 has its own copy of the d256/d512 kernels
/// (`Gemma4Kernels`) and ignores this switch.
pub(super) struct TiledAttnKernels {
    enabled: std::cell::Cell<bool>,
    /// `[f32, f16 cache][d64, d128, d256]`.
    cells: [[std::sync::OnceLock<AotKernel>; 3]; 2],
}

impl TiledAttnKernels {
    pub(super) fn from_env() -> Result<Self, ReflexError> {
        let enabled = match std::env::var("REFLEX_ATTN_TILED").as_deref() {
            Err(_) | Ok("") | Ok("1") | Ok("on") => true,
            Ok("0") | Ok("off") => false,
            Ok(other) => {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "REFLEX_ATTN_TILED must be `1`/`on` (the default) or `0`/`off`, got {other:?}"
                ))
            }
        };
        Ok(Self {
            enabled: std::cell::Cell::new(enabled),
            cells: Default::default(),
        })
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled.get()
    }

    /// For tests that compare the two paths on one loaded model.
    #[cfg(test)]
    pub(super) fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    /// Whether the module for `head_dim` has been loaded (tests).
    #[cfg(test)]
    pub(super) fn loaded(&self, head_dim: usize) -> bool {
        let i = match head_dim {
            64 => 0,
            128 => 1,
            256 => 2,
            _ => return false,
        };
        self.cells.iter().any(|c| c[i].get().is_some())
    }

    /// The kernel for `head_dim` and the K/V cache type, loading its module on
    /// first use; `None` when there is no tiled kernel for that head dim. The f16
    /// cache's kernels are separate modules (`attention_tiled*_f16.cu`).
    pub(super) fn kernel(
        &self,
        device: &Arc<CudaDevice>,
        head_dim: usize,
        dtype: crate::kv_io::KvDtype,
    ) -> Result<Option<&AotKernel>, ReflexError> {
        use crate::kv_io::KvDtype::{F16, F32};
        let (i, bytes, module, name): (_, &'static [u8], _, _) = match (dtype, head_dim) {
            (F32, 64) => (
                0,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_D64")),
                "attention_tiled_d64",
                "attention_tiled_d64_kernel",
            ),
            (F32, 128) => (
                1,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_D128")),
                "attention_tiled_d128",
                "attention_tiled_d128_kernel",
            ),
            (F32, 256) => (
                2,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED")),
                "attention_tiled",
                "attention_tiled_d256_kernel",
            ),
            (F16, 64) => (
                0,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_D64_F16")),
                "attention_tiled_d64_f16",
                "attention_tiled_d64_f16_kernel",
            ),
            (F16, 128) => (
                1,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_D128_F16")),
                "attention_tiled_d128_f16",
                "attention_tiled_d128_f16_kernel",
            ),
            (F16, 256) => (
                2,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_TILED_F16")),
                "attention_tiled_f16",
                "attention_tiled_d256_f16_kernel",
            ),
            _ => return Ok(None),
        };
        let cell = &self.cells[usize::from(dtype == F16)][i];
        if let Some(k) = cell.get() {
            return Ok(Some(k));
        }
        let kernel = match device.get_func(module, name) {
            Some(function) => AotKernel { function },
            None => aot::load_kernel_module(device, bytes, module, &[name])?
                .pop()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module {module}"))?,
        };
        Ok(Some(cell.get_or_init(|| kernel)))
    }
}

/// `attention_decode.cu`'s block size, in threads.
const ATTN_DECODE_THREADS: u32 = 128;
/// The GQA decode kernels, in `attention_decode.cu`, as (head dim, query heads
/// per block). [`DecodeAttnKernels`] keeps them in this order.
const ATTN_DECODE_SHAPES: [(usize, usize); 11] = [
    (64, 1),
    (64, 2),
    (64, 4),
    (64, 8),
    (128, 1),
    (128, 2),
    (128, 4),
    (128, 8),
    (256, 1),
    (256, 2),
    (256, 4),
];
const ATTN_DECODE_NAMES: [&str; 11] = [
    "attention_decode_d64_h1_kernel",
    "attention_decode_d64_h2_kernel",
    "attention_decode_d64_h4_kernel",
    "attention_decode_d64_h8_kernel",
    "attention_decode_d128_h1_kernel",
    "attention_decode_d128_h2_kernel",
    "attention_decode_d128_h4_kernel",
    "attention_decode_d128_h8_kernel",
    "attention_decode_d256_h1_kernel",
    "attention_decode_d256_h2_kernel",
    "attention_decode_d256_h4_kernel",
];
const ATTN_DECODE_F16_NAMES: [&str; 11] = [
    "attention_decode_d64_h1_f16_kernel",
    "attention_decode_d64_h2_f16_kernel",
    "attention_decode_d64_h4_f16_kernel",
    "attention_decode_d64_h8_f16_kernel",
    "attention_decode_d128_h1_f16_kernel",
    "attention_decode_d128_h2_f16_kernel",
    "attention_decode_d128_h4_f16_kernel",
    "attention_decode_d128_h8_f16_kernel",
    "attention_decode_d256_h1_f16_kernel",
    "attention_decode_d256_h2_f16_kernel",
    "attention_decode_d256_h4_f16_kernel",
];
/// Fewest cached positions per decode split, so the combine step stays small
/// next to the work it merges.
const ATTN_DECODE_MIN_SPLIT: usize = 32;

/// Which `attention_decode.cu` kernel serves a decode step with this head dim and
/// GQA group: the most query heads per block (8, 4, 2 or 1) that divides the group
/// and has a kernel. `None` for head dims without one.
pub(super) fn attn_decode_kernel_index(head_dim: usize, group_size: usize) -> Option<usize> {
    ATTN_DECODE_SHAPES
        .iter()
        .enumerate()
        .filter(|&(_, &(d, hb))| d == head_dim && group_size.is_multiple_of(hb))
        .max_by_key(|&(_, &(_, hb))| hb)
        .map(|(i, _)| i)
}

/// The GQA decode kernels behind [`Model::attention`] (`attention_decode.cu`):
/// one module, loaded on the first decode step so cold start (which ends at the
/// first token) never pays for it. `REFLEX_ATTN_DECODE=online` sends decode back
/// to `attention_online_kernel` for A/B runs.
pub(super) struct DecodeAttnKernels {
    enabled: std::cell::Cell<bool>,
    kernels: std::sync::OnceLock<Vec<AotKernel>>,
    /// The same kernels for an f16 K/V cache (`attention_decode_f16.cu`).
    kernels_f16: std::sync::OnceLock<Vec<AotKernel>>,
    /// Streaming multiprocessors, read once, for sizing the splits.
    sm_count: std::sync::OnceLock<usize>,
}

impl DecodeAttnKernels {
    pub(super) fn from_env() -> Result<Self, ReflexError> {
        let enabled = match std::env::var("REFLEX_ATTN_DECODE").as_deref() {
            Err(_) | Ok("") | Ok("gqa") => true,
            Ok("online") => false,
            Ok(other) => {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "REFLEX_ATTN_DECODE must be `gqa` (the default) or `online`, got {other:?}"
                ))
            }
        };
        Ok(Self {
            enabled: std::cell::Cell::new(enabled),
            kernels: std::sync::OnceLock::new(),
            kernels_f16: std::sync::OnceLock::new(),
            sm_count: std::sync::OnceLock::new(),
        })
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled.get()
    }

    /// For tests that compare the two paths on one loaded model.
    #[cfg(test)]
    pub(super) fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    /// Whether the module has been loaded (tests).
    #[cfg(test)]
    pub(super) fn loaded(&self) -> bool {
        self.kernels.get().is_some()
    }

    fn kernels(
        &self,
        device: &Arc<CudaDevice>,
        dtype: crate::kv_io::KvDtype,
    ) -> Result<&[AotKernel], ReflexError> {
        let (cell, bytes, module, names): (_, &'static [u8], _, &[&str]) = match dtype {
            crate::kv_io::KvDtype::F32 => (
                &self.kernels,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_DECODE")),
                "attention_decode",
                &ATTN_DECODE_NAMES,
            ),
            crate::kv_io::KvDtype::F16 => (
                &self.kernels_f16,
                include_bytes!(env!("REFLEX_KERNEL_ATTENTION_DECODE_F16")),
                "attention_decode_f16",
                &ATTN_DECODE_F16_NAMES,
            ),
        };
        if let Some(k) = cell.get() {
            return Ok(k);
        }
        let loaded = aot::load_kernel_module(device, bytes, module, names)?;
        Ok(cell.get_or_init(|| loaded))
    }

    fn sm_count(&self, device: &Arc<CudaDevice>) -> usize {
        *self.sm_count.get_or_init(|| {
            device
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map(|n| n.max(1) as usize)
                .unwrap_or(40)
        })
    }
}

/// The fused decode ops behind [`Model::residual_rmsnorm`] and
/// [`Model::rmsnorm_pair`] (`decode_ops.cu`): one module, loaded on the first
/// decode step like [`DecodeAttnKernels`]. `REFLEX_DECODE_FUSE=0` runs the unfused
/// kernels instead, for A/B runs and the bit-identity tests.
pub(super) struct DecodeOpKernels {
    enabled: std::cell::Cell<bool>,
    kernels: std::sync::OnceLock<Vec<AotKernel>>,
}

const DECODE_OP_NAMES: [&str; 2] = ["add_rmsnorm_kernel", "rmsnorm_pair_kernel"];
const DECODE_OP_ADD_RMSNORM: usize = 0;
const DECODE_OP_RMSNORM_PAIR: usize = 1;
/// Widest row the fused norms stage in shared memory (rmsnorm's 47 KB limit).
const DECODE_OP_MAX_ROW_BYTES: usize = 47 * 1024;

impl DecodeOpKernels {
    pub(super) fn from_env() -> Result<Self, ReflexError> {
        let enabled = match std::env::var("REFLEX_DECODE_FUSE").as_deref() {
            Err(_) | Ok("") | Ok("1") => true,
            Ok("0") => false,
            Ok(other) => {
                return Err(crate::reflex_err!(
                    InvalidInput,
                    "REFLEX_DECODE_FUSE must be 1 (the default) or 0, got {other:?}"
                ))
            }
        };
        Ok(Self {
            enabled: std::cell::Cell::new(enabled),
            kernels: std::sync::OnceLock::new(),
        })
    }

    /// For tests that compare the two paths on one loaded model.
    #[cfg(test)]
    pub(super) fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    /// Whether the module has been loaded (tests).
    #[cfg(test)]
    pub(super) fn loaded(&self) -> bool {
        self.kernels.get().is_some()
    }

    /// The fused kernel `index`, or `None` when fusion is off or a row of
    /// `row_len` floats doesn't fit in shared memory.
    fn kernel(
        &self,
        device: &Arc<CudaDevice>,
        index: usize,
        row_len: usize,
    ) -> Result<Option<&AotKernel>, ReflexError> {
        if !self.enabled.get() || row_len * std::mem::size_of::<f32>() > DECODE_OP_MAX_ROW_BYTES {
            return Ok(None);
        }
        if let Some(k) = self.kernels.get() {
            return Ok(Some(&k[index]));
        }
        let loaded = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_DECODE_OPS")),
            "decode_ops",
            &DECODE_OP_NAMES,
        )?;
        Ok(Some(&self.kernels.get_or_init(|| loaded)[index]))
    }
}

/// rmsnorm's block size for a row of `hidden_size` floats.
fn rmsnorm_block_dim(hidden_size: usize) -> u32 {
    hidden_size.next_multiple_of(32).clamp(32, 256) as u32
}

/// `attention_online.cu`'s block size, in warps, and widest supported head.
const ATTN_ONLINE_WARPS: usize = 8;
const ATTN_ONLINE_MAX_DIM: usize = 1024;
/// A decode step splits its positions across blocks once there are more than this
/// many per block, up to `ATTN_ONLINE_MAX_SPLITS` blocks per head.
const ATTN_ONLINE_SPLIT_POSITIONS: usize = 256;
pub(super) const ATTN_ONLINE_MAX_SPLITS: usize = 64;

/// Loads a kernel and its f16-weight variant from one AOT module (one module
/// load, two function lookups).
pub(super) fn load_kernel_pair(
    device: &Arc<CudaDevice>,
    kernel_bytes: &'static [u8],
    module_name: &'static str,
    names: [&'static str; 2],
) -> Result<[AotKernel; 2], ReflexError> {
    let mut fns = aot::load_kernel_module(device, kernel_bytes, module_name, &names)?;
    let f16 = fns.pop();
    let f32 = fns.pop();
    match (f32, f16) {
        (Some(a), Some(b)) => Ok([a, b]),
        _ => Err(crate::reflex_err!(
            Cuda,
            "missing kernel in module {module_name}"
        )),
    }
}

impl Model {
    /// An f32 output buffer for a kernel that writes every element, so it skips
    /// `alloc_zeros`'s memset. A decode step allocates a few hundred of these;
    /// the memsets were ~5% of its GPU time and a third of its host-side driver
    /// calls on a T4 (docs/design/decode-speed.md). Only for kernels whose launch
    /// covers the whole buffer: anything they skip would hold stale data.
    pub(super) fn alloc_out(&self, len: usize, what: &str) -> Result<CudaSlice<f32>, ReflexError> {
        // SAFETY: f32 has no invalid bit patterns, and every caller's kernel
        // writes all `len` elements before anything reads them.
        unsafe { self.device.alloc::<f32>(len) }
            .map_err(|e| crate::gpu_err!(e, "{what} alloc out: {e}"))
    }

    /// One decode step's attention through `attention_decode.cu`: a block per
    /// (KV head, group chunk, split) reads each K/V row once for all its query
    /// heads. `s` must be a single row with `qk_dim == v_dim` and a kernel from
    /// [`attn_decode_kernel_index`]; the caller checks. Same output as
    /// [`Self::attention_online`]; only the f32 summation order differs.
    #[cfg(test)]
    pub(super) fn attention_decode_gqa(
        &self,
        q: &CudaSlice<f32>,
        k: &KvView,
        v: &KvView,
        s: &OnlineAttnShape,
        kernel_index: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let out = self.alloc_out(s.num_q_heads * s.v_dim, "attn_decode")?;
        self.attention_decode_gqa_to(q, k, v, s, kernel_index, *out.device_ptr(), None)?;
        Ok(out)
    }

    /// [`Self::attention_decode_gqa`] into the `num_q_heads * v_dim` floats at
    /// `out`. A split decode writes its partials to `partial` when given (at least
    /// `num_q_heads * ATTN_ONLINE_MAX_SPLITS * (v_dim + 2)` floats), else to a
    /// buffer allocated here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_decode_gqa_to(
        &self,
        q: &CudaSlice<f32>,
        k: &KvView,
        v: &KvView,
        s: &OnlineAttnShape,
        kernel_index: usize,
        out: u64,
        partial: Option<u64>,
    ) -> Result<(), ReflexError> {
        let kernel = &self.attn_decode.kernels(&self.device, k.dtype())?[kernel_index];
        let heads_per_block = ATTN_DECODE_SHAPES[kernel_index].1;
        let seq_len = s.start_pos + 1;
        let head_blocks = s.num_q_heads / heads_per_block;
        // About four blocks per SM, but no split shorter than ATTN_DECODE_MIN_SPLIT.
        let wanted = (4 * self.attn_decode.sm_count(&self.device)).div_ceil(head_blocks);
        let max_splits = seq_len.div_ceil(ATTN_DECODE_MIN_SPLIT);
        let splits = wanted.clamp(1, max_splits.clamp(1, ATTN_ONLINE_MAX_SPLITS));
        let split_len = seq_len.div_ceil(splits);
        let num_splits = seq_len.div_ceil(split_len);
        let mut params = s.unsplit_params();
        params.num_splits = num_splits as u32;
        params.split_len = split_len as u32;
        let cfg = LaunchConfig {
            grid_dim: (head_blocks as u32, num_splits as u32, 1),
            block_dim: (ATTN_DECODE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        if num_splits == 1 {
            return launch_kv(kernel, cfg, q, k, v, out, params, "attn_decode");
        }
        let owned;
        let partial = match partial {
            Some(p) => p,
            None => {
                owned = self.alloc_out(
                    s.num_q_heads * num_splits * (s.v_dim + 2),
                    "attn_decode partial",
                )?;
                *owned.device_ptr()
            }
        };
        unsafe {
            launch_kv(kernel, cfg, q, k, v, partial, params, "attn_decode")?;
            self.attn_online_combine_k
                .function
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, s.num_q_heads as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        partial,
                        out,
                        s.num_q_heads as u32,
                        s.v_dim as u32,
                        num_splits as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn_decode combine launch: {e}"))?;
        }
        Ok(())
    }

    /// Runs `attention_online_kernel` (and, for a split decode, its combine kernel):
    /// the default implementation behind [`Self::attention`],
    /// [`Self::attention_prefill`], [`Self::mla_attention`] and
    /// [`Self::mla_attention_prefill`]. Returns `[rows, num_q_heads, v_dim]`.
    pub(super) fn attention_online(
        &self,
        q: &CudaSlice<f32>,
        k: &KvView,
        v: &KvView,
        s: OnlineAttnShape,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let out = self.alloc_out(s.rows * s.num_q_heads * s.v_dim, "attn_online")?;
        self.attention_online_to(q, k, v, s, *out.device_ptr(), None)?;
        Ok(out)
    }

    /// [`Self::attention_online`] into the `rows * num_q_heads * v_dim` floats at
    /// `out`; `partial` as in [`Self::attention_decode_gqa_to`] (`rows` times as
    /// many floats).
    pub(super) fn attention_online_to(
        &self,
        q: &CudaSlice<f32>,
        k: &KvView,
        v: &KvView,
        s: OnlineAttnShape,
        out: u64,
        partial: Option<u64>,
    ) -> Result<(), ReflexError> {
        let kernel = match k.dtype() {
            crate::kv_io::KvDtype::F32 => &self.attn_online_k,
            crate::kv_io::KvDtype::F16 => self.kv_f16.kernel(&self.device, KV_F16_ATTN_ONLINE)?,
        };
        if s.qk_dim > ATTN_ONLINE_MAX_DIM || s.v_dim > ATTN_ONLINE_MAX_DIM {
            return Err(crate::reflex_err!(
                UnsupportedArchitecture,
                "attention head dims qk={} v={} exceed the online attention kernel's limit of {ATTN_ONLINE_MAX_DIM}",
                s.qk_dim,
                s.v_dim
            ));
        }
        // Prefill already has rows x heads blocks; only decode needs split-K.
        let max_seq_len = s.start_pos + s.rows;
        let num_splits = if s.rows == 1 {
            (max_seq_len / ATTN_ONLINE_SPLIT_POSITIONS).clamp(1, ATTN_ONLINE_MAX_SPLITS)
        } else {
            1
        };
        let split_len = max_seq_len.div_ceil(num_splits);
        let params = AttnParams {
            k_pos_stride: s.k_pos_stride as u64,
            k_head_stride: s.k_head_stride as u64,
            v_pos_stride: s.v_pos_stride as u64,
            v_head_stride: s.v_head_stride as u64,
            num_q_heads: s.num_q_heads as u32,
            group_size: s.group_size as u32,
            qk_dim: s.qk_dim as u32,
            v_dim: s.v_dim as u32,
            start_pos: s.start_pos as u32,
            num_splits: num_splits as u32,
            split_len: split_len as u32,
            window: s.window as u32,
            scale: s.scale,
        };
        let cfg = LaunchConfig {
            grid_dim: (s.rows as u32, s.num_q_heads as u32, num_splits as u32),
            block_dim: ((ATTN_ONLINE_WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: ((s.qk_dim + ATTN_ONLINE_WARPS * (s.v_dim + 2))
                * std::mem::size_of::<f32>()) as u32,
        };
        if num_splits == 1 {
            return launch_kv(kernel, cfg, q, k, v, out, params, "attn_online");
        }
        let owned;
        let partial = match partial {
            Some(p) => p,
            None => {
                owned = self.alloc_out(
                    s.rows * s.num_q_heads * num_splits * (s.v_dim + 2),
                    "attn_online partial",
                )?;
                *owned.device_ptr()
            }
        };
        unsafe {
            launch_kv(kernel, cfg, q, k, v, partial, params, "attn_online")?;
            self.attn_online_combine_k
                .function
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (s.rows as u32, s.num_q_heads as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        partial,
                        out,
                        s.num_q_heads as u32,
                        s.v_dim as u32,
                        num_splits as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn_online_combine launch: {e}"))?;
        }
        Ok(())
    }

    /// `x` is already device-resident (Phase 2 round 2) -- unlike the
    /// pre-round-2 version, no `htod`/`dtoh` happens here; the caller chains
    /// this op's `CudaSlice` output straight into the next op.
    pub(super) fn rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let dev_out = self.alloc_out(x.len(), "rmsnorm")?;
        self.rmsnorm_to(x, weight, rows, hidden_size, eps, *dev_out.device_ptr())?;
        Ok(dev_out)
    }

    /// [`Self::rmsnorm`] into the `rows * hidden_size` floats at `out`.
    pub(super) fn rmsnorm_to(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
        out: u64,
    ) -> Result<(), ReflexError> {
        // One block per row (see rmsnorm.cu), staging the row in shared memory
        // when it fits the default 48 KB (less the kernel's static shared float).
        let row_bytes = hidden_size * std::mem::size_of::<f32>();
        let launch_cfg = LaunchConfig {
            grid_dim: (rows.clamp(1, 1 << 20) as u32, 1, 1),
            block_dim: (rmsnorm_block_dim(hidden_size), 1, 1),
            shared_mem_bytes: if row_bytes <= 47 * 1024 {
                row_bytes as u32
            } else {
                0
            },
        };
        unsafe {
            self.rmsnorm_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (x, weight, out, rows as u32, hidden_size as u32, eps),
                )
                .map_err(|e| crate::gpu_err!(e, "rmsnorm launch: {e}"))?;
        }
        Ok(())
    }

    /// One row's residual add then RMSNorm: `hidden += delta`, returning
    /// `rmsnorm(hidden) * weight`. One `add_rmsnorm_kernel` launch on the decode
    /// path (see [`DecodeOpKernels`]), else [`Self::add_inplace`] then
    /// [`Self::rmsnorm`]; the two give the same bits.
    pub(super) fn residual_rmsnorm(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let dev_out = self.alloc_out(hidden_size, "add_rmsnorm")?;
        self.residual_rmsnorm_to(
            hidden,
            delta,
            weight,
            hidden_size,
            eps,
            *dev_out.device_ptr(),
        )?;
        Ok(dev_out)
    }

    /// [`Self::residual_rmsnorm`] into the `hidden_size` floats at `out`.
    pub(super) fn residual_rmsnorm_to(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        hidden_size: usize,
        eps: f32,
        out: u64,
    ) -> Result<(), ReflexError> {
        let fused = self
            .decode_ops
            .kernel(&self.device, DECODE_OP_ADD_RMSNORM, hidden_size)?;
        let Some(kernel) =
            fused.filter(|_| hidden.len() == hidden_size && delta.len() == hidden_size)
        else {
            self.add_inplace(hidden, delta)?;
            return self.rmsnorm_to(hidden, weight, 1, hidden_size, eps, out);
        };
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (rmsnorm_block_dim(hidden_size), 1, 1),
            shared_mem_bytes: (hidden_size * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(cfg, (hidden, delta, weight, out, hidden_size as u32, eps))
                .map_err(|e| crate::gpu_err!(e, "add_rmsnorm launch: {e}"))?;
        }
        Ok(())
    }

    /// Two per-head RMSNorms with the same width (Qwen3's Q-Norm and K-Norm):
    /// one `rmsnorm_pair_kernel` launch on the decode path, else two
    /// [`Self::rmsnorm`] calls; the two give the same bits.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rmsnorm_pair(
        &self,
        x1: &CudaSlice<f32>,
        w1: &CudaSlice<f32>,
        rows1: usize,
        x2: &CudaSlice<f32>,
        w2: &CudaSlice<f32>,
        rows2: usize,
        width: usize,
        eps: f32,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), ReflexError> {
        let o1 = self.alloc_out(x1.len(), "rmsnorm_pair")?;
        let o2 = self.alloc_out(x2.len(), "rmsnorm_pair")?;
        self.rmsnorm_pair_to(
            x1,
            w1,
            rows1,
            x2,
            w2,
            rows2,
            width,
            eps,
            (*o1.device_ptr(), *o2.device_ptr()),
        )?;
        Ok((o1, o2))
    }

    /// [`Self::rmsnorm_pair`] into the buffers at `out.0` (`rows1 * width`
    /// floats) and `out.1` (`rows2 * width`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rmsnorm_pair_to(
        &self,
        x1: &CudaSlice<f32>,
        w1: &CudaSlice<f32>,
        rows1: usize,
        x2: &CudaSlice<f32>,
        w2: &CudaSlice<f32>,
        rows2: usize,
        width: usize,
        eps: f32,
        out: (u64, u64),
    ) -> Result<(), ReflexError> {
        let fused = self
            .decode_ops
            .kernel(&self.device, DECODE_OP_RMSNORM_PAIR, width)?;
        let Some(kernel) = fused.filter(|_| x1.len() == rows1 * width && x2.len() == rows2 * width)
        else {
            self.rmsnorm_to(x1, w1, rows1, width, eps, out.0)?;
            return self.rmsnorm_to(x2, w2, rows2, width, eps, out.1);
        };
        let cfg = LaunchConfig {
            grid_dim: ((rows1 + rows2) as u32, 1, 1),
            block_dim: (rmsnorm_block_dim(width), 1, 1),
            shared_mem_bytes: (width * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    cfg,
                    (
                        x1,
                        w1,
                        out.0,
                        rows1 as u32,
                        x2,
                        w2,
                        out.1,
                        width as u32,
                        eps,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rmsnorm_pair launch: {e}"))?;
        }
        Ok(())
    }

    /// `x` and `w_dev` are both already device-resident (Phase 2 round 2) --
    /// `w_dev` is either a whole f32 [`Weight`]'s buffer (a `&CudaSlice<f32>`)
    /// or a zero-copy `CudaView` slice of one (see `Self::gemv_view`).
    pub(super) fn gemv_raw<W: DeviceRepr>(
        &self,
        x: &CudaSlice<f32>,
        w_dev: W,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }

        let dev_y = self.alloc_out(out_features, "gemv")?;
        self.launch_gemv_dense(
            &self.gemv_k,
            x,
            w_dev,
            *dev_y.device_ptr(),
            in_features,
            out_features,
        )?;
        Ok(dev_y)
    }

    /// `gemv_kernel` / `gemv_f16_kernel` (whichever `kernel` is) into the
    /// `out_features` floats at `y`. One warp per output row (gemv_kernel's doc
    /// comment has the coalescing rationale) -- 256 threads/block = 8 rows/block.
    fn launch_gemv_dense<X: DeviceRepr, W: DeviceRepr>(
        &self,
        kernel: &AotKernel,
        x: X,
        w: W,
        y: u64,
        in_features: usize,
        out_features: usize,
    ) -> Result<(), ReflexError> {
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let blocks = (out_features as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (x, w, y, in_features as u32, out_features as u32),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv launch: {e}"))?;
        }
        Ok(())
    }

    /// [`Self::gemv`] into the `w.shape[1]` floats at `y` instead of a new
    /// buffer, for the decode workspace (`decode_step.rs`). Same kernels and
    /// launch geometry as `gemv`, so the same bits. A weight type without a
    /// pointer-output path here goes through `gemv` and a device copy.
    pub(super) fn gemv_to(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        y: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_to: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        match &w.data {
            WeightData::F32(data) => {
                self.launch_gemv_dense(&self.gemv_k, x, data, y, in_features, out_features)
            }
            WeightData::F16(data) => self.launch_gemv_dense(
                &self.gemv_f16_k,
                x,
                &data.slice(..),
                y,
                in_features,
                out_features,
            ),
            WeightData::Quant {
                ty: GgmlType::Q4K, ..
            } if self.gemv_q4k_r1_k.is_some() => self.launch_gemv_q4k_r1(x, w, y),
            WeightData::Quant {
                ty: GgmlType::Q6K, ..
            } if self.gemv_q6k_r1_k.is_some() => {
                let out = self.launch_gemv_q6k_r1(x, w)?;
                // SAFETY: `y` holds `out_features` floats (the caller's buffer).
                let mut dst = unsafe { self.device.upgrade_device_ptr::<f32>(y, out_features) };
                let copied = self.device.dtod_copy(&out, &mut dst);
                out.leak();
                std::mem::forget(dst);
                copied.map_err(|e| crate::gpu_err!(e, "gemv_q6k_r1 copy: {e}"))
            }
            WeightData::Quant {
                ty: GgmlType::Q6K, ..
            } => self.launch_gemv_q6k(x, w, y),
            _ => {
                let out = self.gemv(x, w)?;
                // SAFETY: `y` holds `out_features` floats (the caller's buffer).
                let mut dst = unsafe { self.device.upgrade_device_ptr::<f32>(y, out_features) };
                let copied = self.device.dtod_copy(&out, &mut dst);
                dst.leak();
                copied.map_err(|e| crate::gpu_err!(e, "gemv_to copy: {e}"))
            }
        }
    }

    /// Fused Q/K/V GEMV: one launch reads `x` once and writes q, k, v outputs
    /// with separate weight pointers (`gemv_qkv_kernel`). One warp per output
    /// row across all three projections; total warps = q_out + k_out + v_out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_qkv(
        &self,
        x: &CudaSlice<f32>,
        w_q: &Weight,
        w_k: &Weight,
        w_v: &Weight,
        y_q: u64,
        y_k: u64,
        y_v: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w_q.shape[0] as usize;
        let q_out = w_q.shape[1] as usize;
        let k_out = w_k.shape[1] as usize;
        let v_out = w_v.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_qkv: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        self.launch_gemv_qkv(
            &self.gemv_qkv_k,
            x,
            w_q.f32()?,
            w_k.f32()?,
            w_v.f32()?,
            y_q,
            y_k,
            y_v,
            in_features,
            q_out,
            k_out,
            v_out,
        )
    }

    /// Fused Q/K/V GEMV for an f16 weight matrix.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_qkv_f16(
        &self,
        x: &CudaSlice<f32>,
        w_q: &Weight,
        w_k: &Weight,
        w_v: &Weight,
        y_q: u64,
        y_k: u64,
        y_v: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w_q.shape[0] as usize;
        let q_out = w_q.shape[1] as usize;
        let k_out = w_k.shape[1] as usize;
        let v_out = w_v.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_qkv_f16: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        self.launch_gemv_qkv(
            &self.gemv_qkv_f16_k,
            x,
            w_q.f16()?,
            w_k.f16()?,
            w_v.f16()?,
            y_q,
            y_k,
            y_v,
            in_features,
            q_out,
            k_out,
            v_out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gemv_qkv<W: DeviceRepr>(
        &self,
        kernel: &AotKernel,
        x: &CudaSlice<f32>,
        w_q: W,
        w_k: W,
        w_v: W,
        y_q: u64,
        y_k: u64,
        y_v: u64,
        in_features: usize,
        q_out: usize,
        k_out: usize,
        v_out: usize,
    ) -> Result<(), ReflexError> {
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let total_out = q_out + k_out + v_out;
        let blocks = (total_out as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w_q,
                        w_k,
                        w_v,
                        y_q,
                        y_k,
                        y_v,
                        in_features as u32,
                        q_out as u32,
                        k_out as u32,
                        v_out as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_qkv launch: {e}"))?;
        }
        Ok(())
    }

    /// Fused gate + up GEMV: one launch reads `x` once and writes gate, up
    /// outputs with separate weight pointers (`gemv_gate_up_kernel`).
    pub(super) fn gemv_gate_up(
        &self,
        x: &CudaSlice<f32>,
        w_gate: &Weight,
        w_up: &Weight,
        y_gate: u64,
        y_up: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w_gate.shape[0] as usize;
        let gate_out = w_gate.shape[1] as usize;
        let up_out = w_up.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_gate_up: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        self.launch_gemv_gate_up(
            &self.gemv_gate_up_k,
            x,
            w_gate.f32()?,
            w_up.f32()?,
            y_gate,
            y_up,
            in_features,
            gate_out,
            up_out,
        )
    }

    /// Fused gate + up GEMV for an f16 weight matrix.
    pub(super) fn gemv_gate_up_f16(
        &self,
        x: &CudaSlice<f32>,
        w_gate: &Weight,
        w_up: &Weight,
        y_gate: u64,
        y_up: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w_gate.shape[0] as usize;
        let gate_out = w_gate.shape[1] as usize;
        let up_out = w_up.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_gate_up_f16: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        self.launch_gemv_gate_up(
            &self.gemv_gate_up_f16_k,
            x,
            w_gate.f16()?,
            w_up.f16()?,
            y_gate,
            y_up,
            in_features,
            gate_out,
            up_out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gemv_gate_up<W: DeviceRepr>(
        &self,
        kernel: &AotKernel,
        x: &CudaSlice<f32>,
        w_gate: W,
        w_up: W,
        y_gate: u64,
        y_up: u64,
        in_features: usize,
        gate_out: usize,
        up_out: usize,
    ) -> Result<(), ReflexError> {
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let total_out = gate_out + up_out;
        let blocks = (total_out as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w_gate,
                        w_up,
                        y_gate,
                        y_up,
                        in_features as u32,
                        gate_out as u32,
                        up_out as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_gate_up launch: {e}"))?;
        }
        Ok(())
    }

    /// Dispatches Q/K/V GEMVs: uses fused `gemv_qkv` when fusion is enabled
    /// and all three weights are f32 or f16; falls back to separate `gemv_to`
    /// calls otherwise (quantized weights, or fusion disabled).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_qkv_to(
        &self,
        x: &CudaSlice<f32>,
        w_q: &Weight,
        w_k: &Weight,
        w_v: &Weight,
        y_q: u64,
        y_k: u64,
        y_v: u64,
    ) -> Result<(), ReflexError> {
        if !self.decode_ops.enabled.get() {
            self.gemv_to(x, w_q, y_q)?;
            self.gemv_to(x, w_k, y_k)?;
            self.gemv_to(x, w_v, y_v)?;
            return Ok(());
        }
        // Check if all weights are f32 or f16 (not quantized).
        let q_is_f32 = matches!(&w_q.data, WeightData::F32(_));
        let k_is_f32 = matches!(&w_k.data, WeightData::F32(_));
        let v_is_f32 = matches!(&w_v.data, WeightData::F32(_));
        if q_is_f32 && k_is_f32 && v_is_f32 {
            return self.gemv_qkv(x, w_q, w_k, w_v, y_q, y_k, y_v);
        }
        let q_is_f16 = matches!(&w_q.data, WeightData::F16(_));
        let k_is_f16 = matches!(&w_k.data, WeightData::F16(_));
        let v_is_f16 = matches!(&w_v.data, WeightData::F16(_));
        if q_is_f16 && k_is_f16 && v_is_f16 {
            return self.gemv_qkv_f16(x, w_q, w_k, w_v, y_q, y_k, y_v);
        }
        // Quantized weights: fall back to separate launches.
        self.gemv_to(x, w_q, y_q)?;
        self.gemv_to(x, w_k, y_k)?;
        self.gemv_to(x, w_v, y_v)?;
        Ok(())
    }

    /// Dispatches gate + up GEMVs: same pattern as `gemv_qkv_to`.
    pub(super) fn gemv_gate_up_to(
        &self,
        x: &CudaSlice<f32>,
        w_gate: &Weight,
        w_up: &Weight,
        y_gate: u64,
        y_up: u64,
    ) -> Result<(), ReflexError> {
        if !self.decode_ops.enabled.get() {
            self.gemv_to(x, w_gate, y_gate)?;
            self.gemv_to(x, w_up, y_up)?;
            return Ok(());
        }
        let gate_is_f32 = matches!(&w_gate.data, WeightData::F32(_));
        let up_is_f32 = matches!(&w_up.data, WeightData::F32(_));
        if gate_is_f32 && up_is_f32 {
            return self.gemv_gate_up(x, w_gate, w_up, y_gate, y_up);
        }
        let gate_is_f16 = matches!(&w_gate.data, WeightData::F16(_));
        let up_is_f16 = matches!(&w_up.data, WeightData::F16(_));
        if gate_is_f16 && up_is_f16 {
            return self.gemv_gate_up_f16(x, w_gate, w_up, y_gate, y_up);
        }
        self.gemv_to(x, w_gate, y_gate)?;
        self.gemv_to(x, w_up, y_up)?;
        Ok(())
    }

    pub(super) fn gemv(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        match &w.data {
            WeightData::F32(data) => self.gemv_raw(x, data, in_features, out_features),
            WeightData::F16(data) => {
                if x.len() != in_features {
                    return Err(crate::reflex_err!(
                        Other,
                        "gemv: x.len()={} != in_features={in_features}",
                        x.len()
                    ));
                }
                self.gemv_view(
                    x,
                    WeightView::F16(data.slice(..)),
                    in_features,
                    out_features,
                )
            }
            WeightData::Quant {
                ty: GgmlType::Q4K, ..
            } => self.gemv_q4k(x, w, 1),
            WeightData::Quant {
                ty: GgmlType::Q6K, ..
            } if self.gemv_q6k_r1_k.is_some() => self.launch_gemv_q6k_r1(x, w),
            WeightData::Quant {
                ty: GgmlType::Q6K, ..
            } => self.gemv_q6k(x, w),
            WeightData::Quant {
                ty: GgmlType::Q8_0, ..
            } => self.gemv_q8_0(x, w, 1),
            WeightData::Quant { ty, .. } => Err(crate::reflex_err!(
                Other,
                "internal: no GEMV for quantized-resident {ty:?}"
            )),
        }
    }

    /// `y[rows, out_features] = x[rows, in_features] @ w^T` straight from a
    /// quantized-resident Q8_0 weight (`gemv_q8_0_kernel`, the Gemma 4 path),
    /// launched in chunks of [`GEMV_Q8_0_MAX_ROWS`] rows -- each chunk reads
    /// the weight's bytes once for all its rows. `rows == 1` is the decode
    /// GEMV, which takes `gemv_q8_0_direct_kernel` (no shared-memory staging).
    pub(super) fn gemv_q8_0(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        let kernels = self.q8_0_k.as_ref().ok_or_else(|| {
            ReflexError::Other("internal: Q8_0 weight but gemv_q8_0_kernel not loaded".to_string())
        })?;
        let w_ptr = w.quant_ptr().ok_or_else(|| {
            ReflexError::Other("internal: gemv_q8_0 on a dequantized weight".to_string())
        })?;
        if x.len() != rows * in_features || !in_features.is_multiple_of(QK_LEGACY) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_q8_0: x.len()={} rows={rows} in_features={in_features} (must be rows*in_features, a multiple of {QK_LEGACY})",
                x.len()
            ));
        }
        let dev_y = self
            .device
            .alloc_zeros::<f32>(rows * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv_q8_0 alloc y: {e}"))?;
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let grid = ((out_features as u32).div_ceil(warps_per_block).max(1), 1, 1);
        let staged_grid = (
            (out_features as u32)
                .div_ceil(warps_per_block * GEMV_Q8_0_ROWS_PER_WARP)
                .max(1),
            1,
            1,
        );
        if rows == 1 {
            let mut dev_y = dev_y;
            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                kernels
                    .gemv_direct
                    .function
                    .clone()
                    .launch(
                        cfg,
                        (
                            x,
                            w_ptr,
                            &mut dev_y,
                            in_features as u32,
                            out_features as u32,
                        ),
                    )
                    .map_err(|e| crate::gpu_err!(e, "gemv_q8_0_direct launch: {e}"))?;
            }
            return Ok(dev_y);
        }
        let x_base = *x.device_ptr();
        let y_base = *dev_y.device_ptr();
        let mut r0 = 0usize;
        while r0 < rows {
            let r = (rows - r0).min(GEMV_Q8_0_MAX_ROWS);
            let launch_cfg = LaunchConfig {
                grid_dim: staged_grid,
                block_dim: (threads, 1, 1),
                shared_mem_bytes: (r * GEMV_Q8_0_TILE * 4) as u32,
            };
            let x_ptr = x_base + (r0 * in_features * 4) as u64;
            let y_ptr = y_base + (r0 * out_features * 4) as u64;
            unsafe {
                kernels
                    .gemv
                    .function
                    .clone()
                    .launch(
                        launch_cfg,
                        (
                            x_ptr,
                            w_ptr,
                            y_ptr,
                            in_features as u32,
                            out_features as u32,
                            r as u32,
                        ),
                    )
                    .map_err(|e| crate::gpu_err!(e, "gemv_q8_0 launch: {e}"))?;
            }
            r0 += r;
        }
        Ok(dev_y)
    }

    /// One-row GEMV straight from a quantized-resident Q6_K weight
    /// (`gemv_q6k_kernel`) -- the LM head in QR mode.
    pub(super) fn gemv_q6k(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        // Every output row is written (one warp per row).
        let dev_y = self.alloc_out(w.shape[1] as usize, "gemv_q6k")?;
        self.launch_gemv_q6k(x, w, *dev_y.device_ptr())?;
        Ok(dev_y)
    }

    /// `gemv_q6k_kernel` into the `w.shape[1]` floats at `y`.
    fn launch_gemv_q6k(&self, x: &CudaSlice<f32>, w: &Weight, y: u64) -> Result<(), ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        let kernel = self.gemv_q6k_k.as_ref().ok_or_else(|| {
            ReflexError::Other("internal: Q6_K weight but gemv_q6k_kernel not loaded".to_string())
        })?;
        let w_ptr = w
            .quant_ptr()
            .ok_or_else(|| ReflexError::Other("internal: gemv_q6k on an f32 weight".to_string()))?;
        if x.len() != in_features || !in_features.is_multiple_of(QK_K) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_q6k: x.len()={} in_features={in_features} (must match, a multiple of {QK_K})",
                x.len()
            ));
        }
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let launch_cfg = LaunchConfig {
            grid_dim: ((out_features as u32).div_ceil(warps_per_block).max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (x, w_ptr, y, in_features as u32, out_features as u32),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_q6k launch: {e}"))?;
        }
        Ok(())
    }

    /// `gemv_q6k_r1_kernel` (4 output rows per warp, decode path for Q6_K):
    /// into the `w.shape[1]` floats at `y`. Requires `gemv_q6k_r1_k`.
    fn launch_gemv_q6k_r1(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        let r1 = self.gemv_q6k_r1_k.as_ref().ok_or_else(|| {
            ReflexError::Other("internal: gemv_q6k_r1_kernel not loaded".to_string())
        })?;
        let w_ptr = w
            .quant_ptr()
            .ok_or_else(|| ReflexError::Other("internal: gemv_q6k on an f32 weight".to_string()))?;
        if x.len() != in_features || !in_features.is_multiple_of(QK_K) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_q6k_r1: x.len()={} in_features={in_features} (must match, a multiple of {QK_K})",
                x.len()
            ));
        }
        let dev_y = self.alloc_out(out_features, "gemv_q6k_r1")?;
        let per_block = (GEMV_Q6K_R1_WARPS * GEMV_Q6K_R1_ROWS) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((out_features as u32).div_ceil(per_block).max(1), 1, 1),
            block_dim: (GEMV_Q6K_R1_WARPS as u32 * WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            r1.function
                .clone()
                .launch(
                    cfg,
                    (
                        x,
                        w_ptr,
                        *dev_y.device_ptr(),
                        in_features as u32,
                        out_features as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_q6k_r1 launch: {e}"))?;
        }
        Ok(dev_y)
    }

    /// `gemv_q4k_r1_kernel` (the one-row decode GEMV, 4 output rows per warp)
    /// into the `w.shape[1]` floats at `y`. Requires `gemv_q4k_r1_k`.
    fn launch_gemv_q4k_r1(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        y: u64,
    ) -> Result<(), ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        let r1 = self.gemv_q4k_r1_k.as_ref().ok_or_else(|| {
            ReflexError::Other("internal: gemv_q4k_r1_kernel not loaded".to_string())
        })?;
        let w_ptr = w
            .quant_ptr()
            .ok_or_else(|| ReflexError::Other("internal: gemv_q4k on an f32 weight".to_string()))?;
        if x.len() != in_features || !in_features.is_multiple_of(QK_K) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_q4k: x.len()={} in_features={in_features} (must match, a multiple of {QK_K})",
                x.len()
            ));
        }
        let per_block = (GEMV_Q4K_R1_WARPS * GEMV_Q4K_R1_ROWS) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((out_features as u32).div_ceil(per_block).max(1), 1, 1),
            block_dim: (GEMV_Q4K_R1_WARPS as u32 * WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            r1.function
                .clone()
                .launch(cfg, (x, w_ptr, y, in_features as u32, out_features as u32))
                .map_err(|e| crate::gpu_err!(e, "gemv_q4k_r1 launch: {e}"))?;
        }
        Ok(())
    }

    /// `y[rows, out_features] = x[rows, in_features] @ w^T` straight from a
    /// quantized-resident Q4_K weight (`gemv_q4k_kernel`), launched in chunks
    /// of [`GEMV_Q4K_MAX_ROWS`] rows -- each chunk reads the weight's bytes
    /// once for all its rows. `rows == 1` is the decode GEMV, which takes
    /// `gemv_q4k_r1_kernel` (several output rows per warp; bit-identical).
    pub(super) fn gemv_q4k(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        let kernel = self.gemv_q4k_k.as_ref().ok_or_else(|| {
            ReflexError::Other("internal: Q4_K weight but gemv_q4k_kernel not loaded".to_string())
        })?;
        let w_ptr = w
            .quant_ptr()
            .ok_or_else(|| ReflexError::Other("internal: gemv_q4k on an f32 weight".to_string()))?;
        if x.len() != rows * in_features || !in_features.is_multiple_of(QK_K) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_q4k: x.len()={} rows={rows} in_features={in_features} (must be rows*in_features, a multiple of {QK_K})",
                x.len()
            ));
        }
        // Both kernels write every output element.
        let dev_y = self.alloc_out(rows * out_features, "gemv_q4k")?;
        if rows == 1 && self.gemv_q4k_r1_k.is_some() {
            self.launch_gemv_q4k_r1(x, w, *dev_y.device_ptr())?;
            return Ok(dev_y);
        }
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let launch_cfg = LaunchConfig {
            grid_dim: ((out_features as u32).div_ceil(warps_per_block).max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let x_base = *x.device_ptr();
        let y_base = *dev_y.device_ptr();
        let mut r0 = 0usize;
        while r0 < rows {
            let r = (rows - r0).min(GEMV_Q4K_MAX_ROWS);
            let x_ptr = x_base + (r0 * in_features * 4) as u64;
            let y_ptr = y_base + (r0 * out_features * 4) as u64;
            unsafe {
                kernel
                    .function
                    .clone()
                    .launch(
                        launch_cfg,
                        (
                            x_ptr,
                            w_ptr,
                            y_ptr,
                            in_features as u32,
                            out_features as u32,
                            r as u32,
                        ),
                    )
                    .map_err(|e| crate::gpu_err!(e, "gemv_q4k launch: {e}"))?;
            }
            r0 += r;
        }
        Ok(dev_y)
    }

    /// Batched linear projection: `y[rows, out_features] = x[rows, in_features]
    /// @ w^T`, via cuBLAS Sgemm instead of `gemv_k`'s naive per-row dot
    /// product loop -- the actual win for prefill (`rows` = prompt length):
    /// cuBLAS reuses each weight byte across all `rows` output columns in one
    /// pass instead of re-reading the whole weight matrix from global memory
    /// once per row, turning a bandwidth-bound op into a compute-bound one.
    /// Not used for the `rows == 1` decode step (`Self::forward_one_token_dense`
    /// still uses `Self::gemv`) -- a GEMM with n=1 is just a slower GEMV.
    ///
    /// `w.data` is row-major `[out_features, in_features]` and `x` is
    /// row-major `[rows, in_features]`; cuBLAS is column-major. Reinterpreting
    /// each row-major buffer as column-major gives its transpose for free, so
    /// `w.data`'s raw buffer read as column-major is already `[in_features,
    /// out_features]` == w^T -- computing `y^T = w^T applied via CUBLAS_OP_T
    /// on w, CUBLAS_OP_N on x` (both operands untouched, no transpose copy)
    /// writes `y`'s raw buffer such that reading it back as row-major gives
    /// exactly `[rows, out_features]`. This is the single easiest place in
    /// the batched-prefill path to get a silently-wrong, non-crashing result,
    /// so verify it against `Self::gemv_raw` row-by-row before trusting it
    /// (`prefill_batching_tests::prefill_dense_batched_matches_sequential_prefill` does this).
    ///
    /// An f16 weight (`--weights f16`) goes through `cublasGemmEx` instead, with
    /// `x` cast to f16 first (see [`Self::gemm_ex_f16`]); the layout trick is
    /// the same.
    pub(super) fn gemm(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != rows * in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemm: x.len()={} != rows*in_features={}",
                x.len(),
                rows * in_features
            ));
        }
        if let WeightData::Quant { ty, len, .. } = &w.data {
            return match ty {
                // Up to the threshold, the fused kernel reads the Q4_K bytes
                // once per 8 rows; above it, dequantize into scratch and use
                // cuBLAS.
                GgmlType::Q4K if rows <= quant_fused_max_rows() => self.gemv_q4k(x, w, rows),
                // Same for Q8_0 (Gemma 4), 16 rows per pass.
                GgmlType::Q8_0 if rows <= quant_fused_max_rows() => self.gemv_q8_0(x, w, rows),
                // No multi-row Q6_K kernel yet: one row is the decode GEMV,
                // which uses `gemv_q6k_r1_kernel` (4 output rows per warp);
                // anything more goes through the scratch path.
                GgmlType::Q6K if rows == 1 => {
                    let dev_y = self.launch_gemv_q6k_r1(x, w)?;
                    Ok(dev_y)
                }
                GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 => {
                    self.gemm_quant_scratch(x, w, *ty, *len, in_features, out_features, rows)
                }
                _ => Err(crate::reflex_err!(
                    Other,
                    "internal: batched GEMM on a quantized-resident {ty:?} weight is not supported"
                )),
            };
        }
        self.gemm_view(x, &w.full_view()?, in_features, out_features, rows)
    }

    /// [`Self::gemm`] for a quantized-resident weight that the fused kernel
    /// doesn't cover (Q4_K above its row threshold, Q6_K above one row):
    /// dequantize it into the reused scratch buffer (device to device, no host
    /// traffic) in the model's `--weights` dtype, then take the same cuBLAS
    /// path a weight stored in that dtype takes (`Sgemm` for `f32`; activation
    /// cast + `cublasGemmEx` for `f16`). Q4_K uses the coalesced dequant
    /// kernel (one thread block per Q4_K block); Q6_K its coalesced kernel
    /// when the module that has it is loaded, else the load-time dequant kernel
    /// (one thread per block), both writing the same values the load would
    /// have; Q8_0 its coalesced kernel (one thread per element).
    #[allow(clippy::too_many_arguments)]
    fn gemm_quant_scratch(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        ty: GgmlType,
        len: usize,
        in_features: usize,
        out_features: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let n = in_features * out_features;
        let w_ptr = w.quant_ptr().ok_or_else(|| {
            ReflexError::Other("internal: quantized weight without a device pointer".to_string())
        })?;
        let dtype = self.weights_dtype;
        let mut block_elems = QK_K;
        let (dq, num_blocks, launch_cfg) = match ty {
            GgmlType::Q4K => {
                let (kernel, kernel_name) = match dtype {
                    WeightsDtype::F32 => (
                        &self.dequant_q4k_coalesced_k,
                        "dequantize_q4k_coalesced_kernel",
                    ),
                    WeightsDtype::F16 => (
                        &self.dequant_q4k_coalesced_f16_k,
                        "dequantize_q4k_coalesced_f16_kernel",
                    ),
                };
                let dq = kernel.as_ref().ok_or_else(|| {
                    ReflexError::Other(format!(
                        "internal: Q4_K weight but {kernel_name} not loaded"
                    ))
                })?;
                let num_blocks = len / Q4K_BLOCK_BYTES;
                let cfg = LaunchConfig {
                    grid_dim: (num_blocks as u32, 1, 1),
                    block_dim: (QK_K as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                (dq, num_blocks, cfg)
            }
            GgmlType::Q6K => {
                let num_blocks = len / Q6K_BLOCK_BYTES;
                let coalesced = match dtype {
                    WeightsDtype::F32 => &self.dequant_q6k_coalesced_k,
                    WeightsDtype::F16 => &self.dequant_q6k_coalesced_f16_k,
                };
                if let Some(dq) = coalesced {
                    let cfg = LaunchConfig {
                        grid_dim: (num_blocks as u32, 1, 1),
                        block_dim: (QK_K as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    (dq, num_blocks, cfg)
                } else {
                    let dq = match dtype {
                        WeightsDtype::F32 => &self.dequant_kernels.f32.q6k,
                        WeightsDtype::F16 => &self.dequant_kernels.f16.q6k,
                    };
                    let threads = 256u32;
                    let cfg = LaunchConfig {
                        grid_dim: ((num_blocks as u32).div_ceil(threads).max(1), 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    (dq, num_blocks, cfg)
                }
            }
            GgmlType::Q8_0 => {
                let k = self.q8_0_k.as_ref().ok_or_else(|| {
                    ReflexError::Other(
                        "internal: Q8_0 weight but the gemv_q8_0 module is not loaded".to_string(),
                    )
                })?;
                let dq = match dtype {
                    WeightsDtype::F32 => &k.dequant_f32,
                    WeightsDtype::F16 => &k.dequant_f16,
                };
                block_elems = QK_LEGACY;
                let num_blocks = len / Q8_0_BLOCK_BYTES;
                let threads = 256u32;
                let cfg = LaunchConfig {
                    grid_dim: (
                        ((num_blocks * QK_LEGACY) as u32).div_ceil(threads).max(1),
                        1,
                        1,
                    ),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                (dq, num_blocks, cfg)
            }
            other => {
                return Err(crate::reflex_err!(
                    Other,
                    "internal: no scratch dequant for quantized-resident {other:?}"
                ))
            }
        };
        if num_blocks * block_elems != n {
            return Err(crate::reflex_err!(
                Other,
                "internal: {ty:?} weight holds {} elements, expected {in_features}x{out_features}",
                num_blocks * block_elems
            ));
        }

        let mut scratch = self.quant_scratch.borrow_mut();
        let fits = match (&*scratch, dtype) {
            (Some(WeightData::F32(s)), WeightsDtype::F32) => s.len() >= n,
            (Some(WeightData::F16(s)), WeightsDtype::F16) => s.len() >= n,
            _ => false,
        };
        if !fits {
            let alloc_err = |e| crate::gpu_err!(e, "quant scratch alloc: {e}");
            *scratch = Some(match dtype {
                WeightsDtype::F32 => {
                    WeightData::F32(unsafe { self.device.alloc::<f32>(n) }.map_err(alloc_err)?)
                }
                WeightsDtype::F16 => WeightData::F16(
                    unsafe { self.device.alloc::<half::f16>(n) }.map_err(alloc_err)?,
                ),
            });
        }
        let launch_err = |e| crate::gpu_err!(e, "quant scratch dequant launch: {e}");
        let view = match scratch.as_mut() {
            Some(WeightData::F32(buf)) => {
                unsafe {
                    dq.function
                        .clone()
                        .launch(launch_cfg, (w_ptr, &mut *buf, num_blocks as u32))
                        .map_err(launch_err)?;
                }
                WeightView::F32(buf.slice(0..n))
            }
            Some(WeightData::F16(buf)) => {
                unsafe {
                    dq.function
                        .clone()
                        .launch(launch_cfg, (w_ptr, &mut *buf, num_blocks as u32))
                        .map_err(launch_err)?;
                }
                WeightView::F16(buf.slice(0..n))
            }
            _ => {
                return Err(ReflexError::Other(
                    "internal: quantized-resident scratch buffer has the wrong type".to_string(),
                ))
            }
        };
        self.gemm_view(x, &view, in_features, out_features, rows)
    }

    /// Like [`Self::gemm`], but generic over both operands being any device-resident
    /// reference (`&CudaSlice<f32>` or `&CudaView<f32>`) and taking explicit
    /// `in_features`/`out_features` instead of reading them off a `Weight` -- the
    /// same relationship [`Self::gemv_view`] has to [`Self::gemv_raw`]/[`Self::gemv`],
    /// kept as a separate sibling method rather than folded into `gemm` for the same
    /// reason that trio stays separate. Needed for grouped-GEMM MoE batching
    /// (`Self::forward_layer_moe_batched`, `Self::forward_mla_moe_ffn_batched`), where
    /// `w` is one expert's `CudaView` slice (see [`Self::expert_weight_view`]) and `x`
    /// is a freshly gathered, variable-`rows`-sized group buffer, neither of which fit
    /// `gemm`'s `&Weight` signature. No length assertion on `x` (unlike `gemm`) -- a
    /// generic view type isn't cheaply length-checked here, so correctness relies on
    /// the caller passing consistent `rows`/`in_features`.
    pub(super) fn gemm_view<X: DevicePtr<f32>>(
        &self,
        x: &X,
        w: &WeightView,
        in_features: usize,
        out_features: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(rows * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemm_view alloc y: {e}"))?;
        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: out_features as i32,
            n: rows as i32,
            k: in_features as i32,
            alpha: 1.0f32,
            lda: in_features as i32,
            ldb: in_features as i32,
            beta: 0.0f32,
            ldc: out_features as i32,
        };
        match w {
            WeightView::F32(w) => unsafe {
                self.cublas
                    .gemm(cfg, w, x, &mut dev_y)
                    .map_err(|e| crate::gpu_err!(e, "gemm_view launch: {e:?}"))?;
            },
            WeightView::F16(w) => {
                let x16 = self.cast_activations_f16(*x.device_ptr(), rows * in_features)?;
                self.gemm_ex_f16(
                    *w.device_ptr(),
                    &x16,
                    &mut dev_y,
                    out_features,
                    rows,
                    in_features,
                )?;
            }
        }
        Ok(dev_y)
    }

    /// Casts `n` f32 activations at `x` into a new f16 buffer for
    /// [`Self::gemm_ex_f16`] (`cast_act_f16_kernel`: round to nearest even,
    /// saturating at +-65504, range folded into `self.f16_act_stats`).
    fn cast_activations_f16(
        &self,
        x: sys::CUdeviceptr,
        n: usize,
    ) -> Result<CudaSlice<half::f16>, ReflexError> {
        let mut x16 = unsafe { self.device.alloc::<half::f16>(n) }
            .map_err(|e| crate::gpu_err!(e, "alloc f16 activations: {e}"))?;
        let threads = 256u32;
        let blocks = ((n as u64).div_ceil(threads as u64) as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.dequant_kernels
                .cast_act_f16
                .function
                .clone()
                .launch(cfg, (x, &mut x16, n as u64, &self.f16_act_stats))
                .map_err(|e| crate::gpu_err!(e, "cast_act_f16 launch: {e}"))?;
        }
        Ok(x16)
    }

    /// `y[rows, out_features] = x16[rows, in_features] @ w^T` for an f16 weight
    /// at `w` (row-major `[out_features, in_features]`), through
    /// `cublasGemmEx` with f16 A and B, an f32 C and f32 compute
    /// (`CUBLAS_COMPUTE_32F`): products of f16 values are exact in f32 and
    /// accumulate in f32. Same row-major-as-column-major transpose trick and
    /// leading dimensions as the Sgemm path in [`Self::gemm_view`].
    ///
    /// The handle is pinned to `CUBLAS_PEDANTIC_MATH` for the f32 path (see
    /// `Model::cublas`); this call switches it to `CUBLAS_DEFAULT_MATH` so the
    /// f16 inputs can use tensor cores, and back afterwards. With an explicit
    /// `CUBLAS_COMPUTE_32F` the default mode still accumulates in f32 (no
    /// f16 or TF32 down-conversion); it only frees the summation order.
    fn gemm_ex_f16(
        &self,
        w: sys::CUdeviceptr,
        x16: &CudaSlice<half::f16>,
        y: &mut CudaSlice<f32>,
        out_features: usize,
        rows: usize,
        in_features: usize,
    ) -> Result<(), ReflexError> {
        gemm_ex_f16_raw(&self.cublas, w, x16, y, out_features, rows, in_features)
    }

    /// Shared shape-validation/slicing logic behind [`Self::gemv_expert`] and grouped-
    /// GEMM MoE batching's per-expert-group GEMMs (`Self::forward_layer_moe_batched`,
    /// `Self::forward_mla_moe_ffn_batched`): resolves expert `expert_idx`'s slice of a
    /// per-expert-stacked 3-D MoE tensor (shape `[in_features, out_features,
    /// expert_count]`). Expert `e`'s `in_features * out_features` elements are a
    /// contiguous chunk already in the same row-major `(out_features, in_features)`
    /// layout as a standalone 2-D weight (see this module's doc comment), so
    /// `CudaSlice::slice` gives a zero-copy device-side view -- no device-to-device
    /// copy, let alone a host round-trip.
    pub(super) fn expert_weight_view<'a>(
        w: &'a Weight,
        expert_idx: usize,
    ) -> Result<(WeightView<'a>, usize, usize), ReflexError> {
        let (in_features, out_features, expert_count) = match w.shape.as_slice() {
            [i, o, e] => (*i as usize, *o as usize, *e as usize),
            other => {
                return Err(crate::reflex_err!(
                    Other,
                    "expert_weight_view: expected 3-D per-expert tensor shape, got {other:?}"
                ))
            }
        };
        if expert_idx >= expert_count {
            return Err(crate::reflex_err!(Other, "expert_weight_view: expert_idx {expert_idx} out of range (expert_count={expert_count})"));
        }
        let expert_len = in_features * out_features;
        let start = expert_idx * expert_len;
        Ok((
            w.view(start..start + expert_len)?,
            in_features,
            out_features,
        ))
    }

    /// Expert `expert_idx` of a quantized-resident per-expert-stacked tensor as
    /// its own 2-D [`Weight`] (`[in_features, out_features]`), sharing the
    /// arena: an expert's chunk is whole rows of whole blocks, so it is a
    /// contiguous byte range at `expert_idx * len / expert_count`. `None` for an
    /// f32/f16 weight, which [`Self::expert_weight_view`] slices instead. Lets
    /// the quantized GEMV/GEMM dispatch in [`Self::gemv`]/[`Self::gemm`] serve
    /// one expert unchanged.
    pub(super) fn quant_expert_weight(
        w: &Weight,
        expert_idx: usize,
    ) -> Result<Option<Weight>, ReflexError> {
        let WeightData::Quant {
            ty,
            arena,
            offset,
            len,
        } = &w.data
        else {
            return Ok(None);
        };
        let (in_features, out_features, expert_count) = match w.shape.as_slice() {
            [i, o, e] => (*i, *o, *e as usize),
            other => {
                return Err(crate::reflex_err!(
                    Other,
                    "quant_expert_weight: expected 3-D per-expert tensor shape, got {other:?}"
                ))
            }
        };
        if expert_idx >= expert_count || len % expert_count != 0 {
            return Err(crate::reflex_err!(
                Other,
                "quant_expert_weight: expert {expert_idx} of {expert_count} in a {len}-byte {ty:?} tensor"
            ));
        }
        let expert_len = len / expert_count;
        Ok(Some(Weight {
            data: WeightData::Quant {
                ty: *ty,
                arena: arena.clone(),
                offset: offset + expert_idx * expert_len,
                len: expert_len,
            },
            shape: vec![in_features, out_features],
        }))
    }

    /// GEMV against expert `expert_idx`'s slice of a per-expert-stacked 3-D MoE tensor
    /// (see [`Self::expert_weight_view`]). `x` is generic (like [`Self::gemv_view`],
    /// which this delegates to) so callers can pass either an owned `&CudaSlice<f32>`
    /// (the ordinary per-token MoE path) or a `&CudaView<f32>` row-slice of a larger
    /// buffer. Used by the single-token decode path (`Self::forward_layer_moe`,
    /// `Self::forward_mla_moe_ffn`) -- the batched-prefill MoE FFN
    /// (`Self::forward_layer_moe_batched`, `Self::forward_mla_moe_ffn_batched`) instead
    /// groups every row routed to the same expert and runs one [`Self::gemm_view`]
    /// call per expert, so it calls [`Self::expert_weight_view`] directly rather than
    /// through this single-row wrapper.
    pub(super) fn gemv_expert<X: DeviceRepr>(
        &self,
        x: X,
        w: &Weight,
        expert_idx: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let (view, in_features, out_features) = Self::expert_weight_view(w, expert_idx)?;
        self.gemv_view(x, view, in_features, out_features)
    }

    /// Grouped-GEMM MoE batching's gather step (`moe_gather_kernel`,
    /// `kernels_cuda/elementwise.cu`): copies one expert group's selected rows out of
    /// `src` (a batched-prefill `[rows, hidden_size]` buffer, e.g. `ffn_normed`) into a
    /// fresh contiguous `[perm_row.len(), hidden_size]` buffer, so the group can be run
    /// through that expert's weights as one [`Self::gemm_view`] call. `perm_row` is
    /// already device-resident (uploaded once per expert group by the caller).
    pub(super) fn moe_gather(
        &self,
        src: &CudaSlice<f32>,
        perm_row: &CudaSlice<u32>,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let num_assignments = perm_row.len();
        let mut dst = self
            .device
            .alloc_zeros::<f32>(num_assignments * hidden_size)
            .map_err(|e| crate::gpu_err!(e, "moe_gather alloc: {e}"))?;
        let n = (num_assignments * hidden_size) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.moe_gather_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        src,
                        perm_row,
                        &mut dst,
                        num_assignments as u32,
                        hidden_size as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "moe_gather launch: {e}"))?;
        }
        Ok(dst)
    }

    /// Inverse of [`Self::moe_gather`]: weighted scatter-add of one expert group's
    /// down-projected FFN output (`src`, `[dest_row.len(), hidden_size]`) back into
    /// `dst` (`[rows, hidden_size]`, already allocated/seeded by the caller --
    /// zeroed for dense/MoE, or pre-seeded with MLA's shared-expert output).
    /// `moe_scatter_add_kernel` uses a plain `+=`, not an atomic add -- safe only
    /// because `dest_row` (this expert's group of selected rows) has no duplicate
    /// entries, which top-k routing guarantees (a row never selects the same expert
    /// twice) and because CUDA kernel launches on the default stream (the only stream
    /// this project uses) run sequentially, so distinct expert groups' launches never
    /// overlap either.
    pub(super) fn moe_scatter_add(
        &self,
        src: &CudaSlice<f32>,
        dest_row: &CudaSlice<u32>,
        weight: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        hidden_size: usize,
    ) -> Result<(), ReflexError> {
        let num_assignments = dest_row.len();
        let n = (num_assignments * hidden_size) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.moe_scatter_add_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        src,
                        dest_row,
                        weight,
                        dst,
                        num_assignments as u32,
                        hidden_size as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "moe_scatter_add launch: {e}"))?;
        }
        Ok(())
    }

    /// Like `gemv`, but computes only the output rows named by
    /// `row_indices` instead of every row `0..out_features` -- System1's
    /// candidate-subset LM-head scoring (`Self::system1_evaluate`), so a
    /// full-vocab GEMV and a vocab-sized D2H transfer are never paid when
    /// only a handful of candidate token ids' logits are needed. Returns
    /// the gathered logits already downloaded to host (unlike `gemv_raw`/
    /// `gemv_expert`, which return a device-resident `CudaSlice` for further
    /// on-device chaining) since every call site here is a leaf op wanting
    /// host floats.
    pub(super) fn gemv_gather(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        row_indices: &[u32],
    ) -> Result<Vec<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        if x.len() != in_features {
            return Err(crate::reflex_err!(
                Other,
                "gemv_gather: x.len()={} != in_features={in_features}",
                x.len()
            ));
        }
        if row_indices.is_empty() {
            return Err(ReflexError::Other(
                "gemv_gather: row_indices must not be empty".to_string(),
            ));
        }
        if let Some(&bad) = row_indices.iter().find(|&&r| r as usize >= out_features) {
            return Err(crate::reflex_err!(
                Other,
                "gemv_gather: row index {bad} out of range (out_features={out_features})"
            ));
        }
        if let WeightData::Quant { .. } = &w.data {
            // A quantized-resident LM head has no row-gather kernel: run the
            // full-vocab GEMV (it reads ~1/5 of the f32 bytes) and pick the
            // requested logits on the host.
            let all = self
                .device
                .dtoh_sync_copy(&self.gemv(x, w)?)
                .map_err(|e| crate::gpu_err!(e, "gemv_gather (quantized) dtoh: {e}"))?;
            return Ok(row_indices.iter().map(|&r| all[r as usize]).collect());
        }
        let num_rows = row_indices.len();
        let dev_indices = self
            .device
            .htod_sync_copy(row_indices)
            .map_err(|e| crate::gpu_err!(e, "gemv_gather upload row_indices: {e}"))?;
        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(num_rows)
            .map_err(|e| crate::gpu_err!(e, "gemv_gather alloc y: {e}"))?;
        // One warp per gathered row -- see gemv_raw's identical comment.
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let blocks = (num_rows as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (in_u32, rows_u32) = (in_features as u32, num_rows as u32);
        let launched = match &w.data {
            WeightData::F32(data) => unsafe {
                self.gemv_gather_k.function.clone().launch(
                    launch_cfg,
                    (x, data, &dev_indices, &mut dev_y, in_u32, rows_u32),
                )
            },
            WeightData::F16(data) => unsafe {
                self.gemv_gather_f16_k.function.clone().launch(
                    launch_cfg,
                    (x, data, &dev_indices, &mut dev_y, in_u32, rows_u32),
                )
            },
            WeightData::Quant { ty, .. } => {
                return Err(crate::reflex_err!(
                    Other,
                    "internal: gemv_gather reached the row-gather kernel with a quantized-resident {ty:?} weight"
                ))
            }
        };
        launched.map_err(|e| crate::gpu_err!(e, "gemv_gather launch: {e}"))?;
        self.device
            .dtoh_sync_copy(&dev_y)
            .map_err(|e| crate::gpu_err!(e, "gemv_gather dtoh: {e}"))
    }

    /// In-place: `t` is already device-resident. `position` is a plain
    /// scalar kernel argument rather than an uploaded device array --
    /// `batch_size` is a permanent project constraint (docs/DEVELOPMENT.md's
    /// Non-goals), so there is never more than one token's position to pass,
    /// and the previous per-call device allocation+upload for it was pure
    /// overhead (Phase 2 round 2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope(
        &self,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
        rope_type: RopeType,
    ) -> Result<(), ReflexError> {
        let t = *t.device_ptr();
        self.rope_at(
            t, num_heads, head_dim, rotary_dim, position, base, rope_type,
        )
    }

    /// [`Self::rope`] on the `num_heads * head_dim` floats at `t` (a decode
    /// workspace buffer or a K cache row).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_at(
        &self,
        t: u64,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
        rope_type: RopeType,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = match rope_type {
            RopeType::Neox => &self.rope_k,
            RopeType::Norm => self.rope_norm_k.as_ref().ok_or_else(|| {
                ReflexError::Other("rope_norm_kernel not loaded for this model".to_string())
            })?,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        position as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope launch: {e}"))?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::rope`]: rotates `rows` rows of `t`
    /// in one launch, row `r` at absolute position `start_pos + r` (unlike
    /// `rope`'s single scalar `position`, which only serves the `rows == 1`
    /// decode step). `t` is row-major `[rows, num_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_batch(
        &self,
        t: &mut CudaSlice<f32>,
        start_pos: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rows: usize,
        base: f32,
        rope_type: RopeType,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = match rope_type {
            RopeType::Neox => &self.rope_batch_k,
            RopeType::Norm => self.rope_norm_batch_k.as_ref().ok_or_else(|| {
                ReflexError::Other("rope_norm_batch_kernel not loaded for this model".to_string())
            })?,
        };
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        start_pos as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        rows as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_batch launch: {e}"))?;
        }
        Ok(())
    }

    /// Like [`Self::rope`], but launches `m.rope_norm_k` (`rope_norm_kernel` --
    /// consecutive-pair rotation) instead of the shared `self.rope_k`
    /// (`rope_kernel` -- half-split rotation). Only DeepSeek-V2/V3 MLA needs this
    /// (see `MlaModel::rope_norm_k`'s doc comment); every other architecture uses
    /// `Self::rope` unchanged.
    // Each parameter maps 1:1 to a distinct `rope_norm_k` launch argument;
    // bundling them into a struct would just relocate the count, not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_norm(
        &self,
        m: &MlaModel,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            m.rope_norm_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        position as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_norm launch: {e}"))?;
        }
        Ok(())
    }

    /// Like [`Self::rope_norm`], but launches `m.rope_norm_yarn_k`
    /// (`rope_norm_yarn_kernel`) with the extra YaRN parameters from
    /// `cfg.yarn` (see `MlaYarnConfig`'s doc comment). Used instead of
    /// `Self::rope_norm` whenever `cfg.yarn.is_some()`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_norm_yarn(
        &self,
        m: &MlaModel,
        yarn: &MlaYarnConfig,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            m.rope_norm_yarn_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        position as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        base,
                        yarn.freq_scale,
                        yarn.ext_factor,
                        yarn.attn_factor,
                        yarn.corr_dim_start,
                        yarn.corr_dim_end,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_norm_yarn launch: {e}"))?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::rope_norm`]: rotates `rows` rows of `t` in
    /// one launch, row `r` at absolute position `start_pos + r` (`rope_norm_batch_kernel`,
    /// same batching idea as [`Self::rope_batch`] applied to the consecutive-pair
    /// rotation). `t` is row-major `[rows, num_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_norm_batch(
        &self,
        m: &MlaModel,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        start_pos: usize,
        rows: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            m.rope_norm_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        start_pos as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        rows as u32,
                        base,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_norm_batch launch: {e}"))?;
        }
        Ok(())
    }

    /// Batched-prefill variant of [`Self::rope_norm_yarn`]: like [`Self::rope_norm_batch`],
    /// plus the extra YaRN parameters from `cfg.yarn` (`rope_norm_yarn_batch_kernel`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_norm_yarn_batch(
        &self,
        m: &MlaModel,
        yarn: &MlaYarnConfig,
        t: &mut CudaSlice<f32>,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        start_pos: usize,
        rows: usize,
        base: f32,
    ) -> Result<(), ReflexError> {
        let half_rotary = rotary_dim / 2;
        let total_pairs = (rows * num_heads * half_rotary) as u32;
        let threads = 256u32;
        let blocks = total_pairs.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            m.rope_norm_yarn_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        t,
                        start_pos as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                        rows as u32,
                        base,
                        yarn.freq_scale,
                        yarn.ext_factor,
                        yarn.attn_factor,
                        yarn.corr_dim_start,
                        yarn.corr_dim_end,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rope_norm_yarn_batch launch: {e}"))?;
        }
        Ok(())
    }

    /// `gate`/`up` are already device-resident, separate (not concatenated)
    /// buffers -- `silu_and_mul_kernel` takes them as two pointers, so no
    /// device-side concatenation step is needed either (Phase 2 round 2).
    pub(super) fn silu_and_mul(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        hidden_size: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let dev_out = self.alloc_out(hidden_size, "silu")?;
        self.silu_and_mul_to(gate, up, hidden_size, *dev_out.device_ptr())?;
        Ok(dev_out)
    }

    /// [`Self::silu_and_mul`] into the `hidden_size` floats at `out`.
    pub(super) fn silu_and_mul_to(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        hidden_size: usize,
        out: u64,
    ) -> Result<(), ReflexError> {
        let threads = 256u32;
        let blocks = (hidden_size as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.silu_k
                .function
                .clone()
                .launch(launch_cfg, (gate, up, out, 1u32, hidden_size as u32))
                .map_err(|e| crate::gpu_err!(e, "silu launch: {e}"))?;
        }
        Ok(())
    }

    /// Causal single-new-query attention against the full K/V cache so far
    /// (`k_cache`/`v_cache` views already include this position's own K/V --
    /// `seq_len = position + 1`). GQA-grouped: query head `h` reads KV head
    /// `h / (num_q_heads / num_kv_heads)`. `q`/`k_cache`/`v_cache` are all
    /// already device-resident (Phase 2 round 2) -- `k_cache`/`v_cache` are
    /// `CudaView`s into a preallocated per-layer device buffer, not a fresh
    /// upload of the whole cache history on every call. `window > 0` restricts
    /// the query to the last `window` positions (a sliding-window layer); it is
    /// applied by narrowing the K/V views, so neither kernel needs to know.
    // Each parameter maps 1:1 to a distinct attention-kernel launch argument;
    // bundling them into a struct would just relocate the count, not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &KvView,
        v_cache: &KvView,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        window: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            let out = self.alloc_out(num_q_heads * head_dim, "attention")?;
            self.attention_to(
                q,
                k_cache,
                v_cache,
                num_q_heads,
                num_kv_heads,
                head_dim,
                seq_len,
                window,
                *out.device_ptr(),
                None,
            )?;
            return Ok(out);
        }
        let first = if window > 0 {
            seq_len.saturating_sub(window)
        } else {
            0
        };
        let kv_stride = num_kv_heads * head_dim;
        let k_cache = &k_cache.tail(first * kv_stride);
        let v_cache = &v_cache.tail(first * kv_stride);
        let seq_len = seq_len - first;
        let (k_cache, v_cache) = (k_cache.f32()?, v_cache.f32()?);
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(num_q_heads * head_dim)
            .map_err(|e| crate::gpu_err!(e, "attn alloc out: {e}"))?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            self.attn_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        k_cache,
                        v_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        seq_len as u32,
                        scale,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// [`Self::attention`] into the `num_q_heads * head_dim` floats at `out`, with
    /// split partials in `partial` when given (see
    /// [`Self::attention_decode_gqa_to`]). Online attention only (the default
    /// `AttnImpl`); the legacy kernels have no pointer-output form.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_to(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &KvView,
        v_cache: &KvView,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        window: usize,
        out: u64,
        partial: Option<u64>,
    ) -> Result<(), ReflexError> {
        if self.attn_impl != AttnImpl::Online {
            return Err(crate::reflex_err!(
                Other,
                "internal: attention_to needs the online attention kernels"
            ));
        }
        let first = if window > 0 {
            seq_len.saturating_sub(window)
        } else {
            0
        };
        let kv_stride = num_kv_heads * head_dim;
        let k_cache = &k_cache.tail(first * kv_stride);
        let v_cache = &v_cache.tail(first * kv_stride);
        let seq_len = seq_len - first;
        let shape = OnlineAttnShape {
            rows: 1,
            start_pos: seq_len - 1,
            num_q_heads,
            group_size: num_q_heads / num_kv_heads,
            qk_dim: head_dim,
            v_dim: head_dim,
            k_pos_stride: num_kv_heads * head_dim,
            k_head_stride: head_dim,
            v_pos_stride: num_kv_heads * head_dim,
            v_head_stride: head_dim,
            scale: 1.0f32 / (head_dim as f32).sqrt(),
            window: 0,
        };
        if self.attn_decode.enabled() {
            if let Some(i) = attn_decode_kernel_index(head_dim, shape.group_size) {
                return self.attention_decode_gqa_to(q, k_cache, v_cache, &shape, i, out, partial);
            }
        }
        self.attention_online_to(q, k_cache, v_cache, shape, out, partial)
    }

    /// Batched-prefill variant of [`Self::attention`]: scores `rows` new
    /// query rows against the shared K/V cache in one launch (grid gains a
    /// query-row dimension), each row causally masked to its own
    /// `start_pos + row + 1` positions instead of one shared `seq_len`.
    /// `k_cache`/`v_cache` must already cover `0..start_pos+rows` positions
    /// (this batch's own K/V, written by the caller before this call --
    /// `Self::forward_attn_block_batched`). `q` is row-major `[rows,
    /// num_q_heads, head_dim]`; returns row-major `[rows, num_q_heads,
    /// head_dim]`. `window` is [`Self::attention`]'s sliding window (`0` =
    /// none), masked per row inside the kernel since each row's window starts
    /// at a different position.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &KvView,
        v_cache: &KvView,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        rows: usize,
        window: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            let shape = OnlineAttnShape {
                rows,
                start_pos,
                num_q_heads,
                group_size: num_q_heads / num_kv_heads,
                qk_dim: head_dim,
                v_dim: head_dim,
                k_pos_stride: num_kv_heads * head_dim,
                k_head_stride: head_dim,
                v_pos_stride: num_kv_heads * head_dim,
                v_head_stride: head_dim,
                scale: 1.0f32 / (head_dim as f32).sqrt(),
                window,
            };
            if attn_tiled_preferred(rows, start_pos, shape.group_size)
                && self.attn_tiled.enabled()
                && attn_tiled_fits(&shape)
            {
                if let Some(kernel) =
                    self.attn_tiled
                        .kernel(&self.device, head_dim, k_cache.dtype())?
                {
                    return launch_attention_tiled(
                        &self.device,
                        kernel,
                        q,
                        k_cache,
                        v_cache,
                        &shape,
                    );
                }
            }
            return self.attention_online(q, k_cache, v_cache, shape);
        }
        let (k_cache, v_cache) = (k_cache.f32()?, v_cache.f32()?);
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(rows * num_q_heads * head_dim)
            .map_err(|e| crate::gpu_err!(e, "attn_prefill alloc out: {e}"))?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let max_seq_len = start_pos + rows;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, rows as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (max_seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            self.attn_prefill_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        k_cache,
                        v_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        start_pos as u32,
                        rows as u32,
                        window as u32,
                        scale,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn_prefill launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// In-place residual add: `a[i] += b[i]`, both already device-resident
    /// (Phase 2 round 2) -- replaces the host-side
    /// `a.iter().zip(b.iter()).map(|(&x,&y)| x+y)` loops every forward
    /// function used to do, which required both operands on the host.
    pub(super) fn add_inplace(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> Result<(), ReflexError> {
        self.launch_add(a, b, a.len())
    }

    /// Row-broadcast add: `a` is `[a.len() / b.len(), b.len()]` row-major and
    /// every row gets `b` added (`add_kernel` with `m = b.len()`). Used for
    /// Qwen2's Q/K/V biases in batched prefill.
    pub(super) fn add_rows_inplace(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> Result<(), ReflexError> {
        if b.is_empty() || !a.len().is_multiple_of(b.len()) {
            return Err(crate::reflex_err!(
                Other,
                "add_rows_inplace: {} elements is not a whole number of {}-element rows",
                a.len(),
                b.len()
            ));
        }
        self.launch_add(a, b, b.len())
    }

    /// `add_kernel`: `a[i] += b[i % m]` over all of `a`.
    fn launch_add(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: usize,
    ) -> Result<(), ReflexError> {
        let n = a.len();
        self.add_at(*a.device_ptr(), n, b, m)
    }

    /// `add_kernel` on the `n` floats at `a`: `a[i] += b[i % m]` (a decode
    /// workspace buffer or a K/V cache row, for Qwen2's Q/K/V biases).
    pub(super) fn add_at(
        &self,
        a: u64,
        n: usize,
        b: &CudaSlice<f32>,
        m: usize,
    ) -> Result<(), ReflexError> {
        let n = n as u32;
        let m = m as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.add_k
                .function
                .clone()
                .launch(launch_cfg, (a, b, n, m))
                .map_err(|e| crate::gpu_err!(e, "add launch: {e}"))?;
        }
        Ok(())
    }

    /// Like [`Self::gemv_raw`], but generic over both operands being any
    /// device-resident reference (`&CudaSlice<f32>` or `&CudaView<f32>`) instead of
    /// requiring `x` to be a whole owned `CudaSlice`. Needed for MLA's per-head
    /// absorption/decompression steps (see `Self::forward_mla_attn_block`,
    /// `Self::gemv_per_head`), where `x` is a strided per-head slice of a larger
    /// buffer. No length assertion (unlike `gemv_raw`) -- a generic view type isn't
    /// cheaply length-checked here, so correctness relies on the caller passing
    /// consistent `in_features`/`out_features`.
    pub(super) fn gemv_view<X: DeviceRepr>(
        &self,
        x: X,
        w: WeightView,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let dev_y = self.alloc_out(out_features, "gemv_view")?;
        // Same launch geometry as gemv_raw (Self::launch_gemv_dense).
        let y = *dev_y.device_ptr();
        match &w {
            WeightView::F32(w_dev) => {
                self.launch_gemv_dense(&self.gemv_k, x, w_dev, y, in_features, out_features)?
            }
            WeightView::F16(w_dev) => {
                self.launch_gemv_dense(&self.gemv_f16_k, x, w_dev, y, in_features, out_features)?
            }
        }
        Ok(dev_y)
    }

    /// Applies a per-head-stacked weight tensor (`w`, shape `[in_features,
    /// out_features, n_head]` -- same layout convention as MoE's per-expert
    /// tensors, see `Self::gemv_expert`'s doc comment, just "expert" -> "head") to
    /// every head of a per-head-stacked input `x` (`[n_head, in_features]`
    /// row-major, contiguous per head), producing a per-head-stacked output
    /// (`[n_head, out_features]`). Unlike `gemv_expert` (which selects one of many
    /// experts per token), MLA's decompression step (`wv_b`, see
    /// `Self::forward_mla_attn_block`) always uses every head, so this loops over
    /// all of them, reusing the same `gemv_kernel` per head via zero-copy
    /// `CudaSlice::slice` views on both operands (via `Self::gemv_view`).
    pub(super) fn gemv_per_head(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
        n_head: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let (in_features, out_features, head_count) = match w.shape.as_slice() {
            [i, o, h] => (*i as usize, *o as usize, *h as usize),
            other => {
                return Err(crate::reflex_err!(
                    Other,
                    "gemv_per_head: expected 3-D per-head tensor shape, got {other:?}"
                ))
            }
        };
        if head_count != n_head {
            return Err(crate::reflex_err!(
                Other,
                "gemv_per_head: tensor's head dim {head_count} != n_head {n_head}"
            ));
        }
        let mut out = self
            .device
            .alloc_zeros::<f32>(n_head * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv_per_head alloc: {e}"))?;
        for h in 0..n_head {
            let w_view =
                w.view(h * in_features * out_features..(h + 1) * in_features * out_features)?;
            let x_view = x.slice(h * in_features..(h + 1) * in_features);
            let y = self.gemv_view(&x_view, w_view, in_features, out_features)?;
            let mut dst = out.slice_mut(h * out_features..(h + 1) * out_features);
            self.device
                .dtod_copy(&y, &mut dst)
                .map_err(|e| crate::gpu_err!(e, "gemv_per_head dtod head {h}: {e}"))?;
        }
        Ok(out)
    }

    /// Batched-prefill variant of [`Self::gemv_per_head`]: applies a per-head-stacked
    /// weight tensor `w` to every head of every row of a batched `rows`-row input in
    /// one launch (`gemv_per_head_batch_kernel`) instead of `rows` separate
    /// `Self::gemv_per_head` calls (each of which itself loops `n_head` times --
    /// looping this per row would mean `rows * n_head` launches, the exact
    /// launch-count blowup this kernel exists to avoid; see
    /// `Self::forward_mla_attn_block_batched`'s doc comment for the math). Unlike
    /// `gemv_per_head`, `x` need not be a contiguous `[rows, n_head, in_features]`
    /// buffer -- `x_row_stride`/`x_head_stride`/`x_head_offset` let the caller read
    /// directly out of a wider strided buffer (MLA's absorption step reads q_nope
    /// straight out of the `wq` projection output, which interleaves q_nope/q_pe per
    /// head, with no separate gather step). `out` is always a freshly allocated,
    /// contiguous `[rows, n_head, out_features]` buffer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_per_head_batch(
        &self,
        m: &MlaModel,
        x: &CudaSlice<f32>,
        w: &Weight,
        rows: usize,
        n_head: usize,
        x_row_stride: usize,
        x_head_stride: usize,
        x_head_offset: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let (in_features, out_features, head_count) = match w.shape.as_slice() {
            [i, o, h] => (*i as usize, *o as usize, *h as usize),
            other => {
                return Err(crate::reflex_err!(
                    Other,
                    "gemv_per_head_batch: expected 3-D per-head tensor shape, got {other:?}"
                ))
            }
        };
        if head_count != n_head {
            return Err(crate::reflex_err!(
                Other,
                "gemv_per_head_batch: tensor's head dim {head_count} != n_head {n_head}"
            ));
        }
        let mut out = self
            .device
            .alloc_zeros::<f32>(rows * n_head * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv_per_head_batch alloc: {e}"))?;

        let threads = 256u32;
        let out_blocks = (out_features as u32).div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (out_blocks, n_head as u32, rows as u32),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // Every launch argument after the three buffers, shared by both arms.
        let (d0, d1, d2, d3, d4, d5, d6) = (
            rows as u32,
            n_head as u32,
            in_features as u32,
            out_features as u32,
            x_row_stride as u32,
            x_head_stride as u32,
            x_head_offset as u32,
        );
        let launched = match &w.data {
            WeightData::F32(data) => unsafe {
                m.gemv_per_head_batch_k
                    .function
                    .clone()
                    .launch(launch_cfg, (x, data, &mut out, d0, d1, d2, d3, d4, d5, d6))
            },
            WeightData::F16(data) => unsafe {
                m.gemv_per_head_batch_f16_k
                    .function
                    .clone()
                    .launch(launch_cfg, (x, data, &mut out, d0, d1, d2, d3, d4, d5, d6))
            },
            WeightData::Quant { ty, .. } => {
                return Err(crate::reflex_err!(
                    Other,
                    "internal: gemv_per_head_batch on a quantized-resident {ty:?} weight"
                ))
            }
        };
        launched.map_err(|e| crate::gpu_err!(e, "gemv_per_head_batch launch: {e}"))?;
        Ok(out)
    }

    /// DeepSeek-V2/V3 MLA's MQA-style attention: `num_q_heads` query heads (each
    /// `qk_dim` wide) attend against a single shared compressed KV "head"
    /// (`kv_cache` view, `[seq_len, qk_dim]` row-major, one row per cached
    /// position -- the *same* row also serves as the value vector, using only its
    /// first `v_dim` elements, since MLA's whole point is that K and V share one
    /// compressed representation, unlike GQA's separate caches). Output is
    /// `[num_q_heads, v_dim]`, still in compressed latent space --
    /// `Self::gemv_per_head` (with `wv_b`) decompresses it afterward. `scale` must
    /// be `1/sqrt(qk_nope_head_dim + qk_rope_head_dim)` -- the *uncompressed*
    /// per-head dim, confirmed against llama.cpp's `deepseek2.cpp` (`kq_scale`) --
    /// **not** `1/sqrt(qk_dim)` (the compressed dot-product width), an easy
    /// mistake since every other op in this codebase scales by its own dot-product
    /// dimension.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_attention(
        &self,
        m: &MlaModel,
        q: &CudaSlice<f32>,
        kv_cache: &KvView,
        num_q_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            // MQA: every query head reads the one shared compressed row, which is
            // both K (all qk_dim) and V (its first v_dim).
            return self.attention_online(
                q,
                kv_cache,
                kv_cache,
                OnlineAttnShape {
                    rows: 1,
                    start_pos: seq_len - 1,
                    num_q_heads,
                    group_size: num_q_heads,
                    qk_dim,
                    v_dim,
                    k_pos_stride: qk_dim,
                    k_head_stride: 0,
                    v_pos_stride: qk_dim,
                    v_head_stride: 0,
                    scale,
                    window: 0,
                },
            );
        }
        let kv_cache = kv_cache.f32()?;
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(num_q_heads * v_dim)
            .map_err(|e| crate::gpu_err!(e, "mla_attn alloc out: {e}"))?;
        let block_dim = (qk_dim as u32).next_power_of_two();
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            m.mla_attn_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        kv_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        qk_dim as u32,
                        v_dim as u32,
                        seq_len as u32,
                        scale,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "mla_attn launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// Batched-prefill variant of [`Self::mla_attention`]: scores `rows` new query
    /// rows against the shared compressed MQA KV cache in one launch (grid gains a
    /// query-row dimension), each row causally masked to its own `start_pos + row +
    /// 1` positions -- same relationship [`Self::attention_prefill`] has to
    /// [`Self::attention`], applied to MLA's MQA/compressed-KV shape. `q` is
    /// row-major `[rows, num_q_heads, qk_dim]`; returns row-major `[rows,
    /// num_q_heads, v_dim]`. `kv_cache` must already cover `0..start_pos+rows`
    /// positions (this batch's own compressed KV, written by the caller before this
    /// call -- `Self::forward_mla_attn_block_batched`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_attention_prefill(
        &self,
        m: &MlaModel,
        q: &CudaSlice<f32>,
        kv_cache: &KvView,
        num_q_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        start_pos: usize,
        rows: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            return self.attention_online(
                q,
                kv_cache,
                kv_cache,
                OnlineAttnShape {
                    rows,
                    start_pos,
                    num_q_heads,
                    group_size: num_q_heads,
                    qk_dim,
                    v_dim,
                    k_pos_stride: qk_dim,
                    k_head_stride: 0,
                    v_pos_stride: qk_dim,
                    v_head_stride: 0,
                    scale,
                    window: 0,
                },
            );
        }
        let kv_cache = kv_cache.f32()?;
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(rows * num_q_heads * v_dim)
            .map_err(|e| crate::gpu_err!(e, "mla_attn_prefill alloc out: {e}"))?;
        let block_dim = (qk_dim as u32).next_power_of_two();
        let max_seq_len = start_pos + rows;
        let launch_cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, rows as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (max_seq_len * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            m.mla_attn_prefill_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        q,
                        kv_cache,
                        &mut dev_out,
                        num_q_heads as u32,
                        qk_dim as u32,
                        v_dim as u32,
                        start_pos as u32,
                        rows as u32,
                        scale,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "mla_attn_prefill launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// Batched-prefill helper: extracts a fixed-width, fixed-offset sub-slice of
    /// every (row, head) entry of `src` into its own contiguous output
    /// (`mla_extract_batch_kernel`) -- see that kernel's doc comment
    /// (`kernels_cuda/elementwise.cu`) for the exact layout contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_extract_batch(
        &self,
        m: &MlaModel,
        src: &CudaSlice<f32>,
        rows: usize,
        num_heads: usize,
        src_head_width: usize,
        dst_width: usize,
        src_head_offset: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut dst = self
            .device
            .alloc_zeros::<f32>(rows * num_heads * dst_width)
            .map_err(|e| crate::gpu_err!(e, "mla_extract_batch alloc: {e}"))?;
        let n = (rows * num_heads * dst_width) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            m.mla_extract_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        src,
                        &mut dst,
                        rows as u32,
                        num_heads as u32,
                        src_head_width as u32,
                        dst_width as u32,
                        src_head_offset as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "mla_extract_batch launch: {e}"))?;
        }
        Ok(dst)
    }

    /// Batched-prefill helper: merges per-head `absorbed` (kv_lora-wide) and
    /// already-RoPE'd `q_pe` (qk_rope-wide) into Qcur's per-head row
    /// (`mla_concat_qcur_batch_kernel`).
    // Each parameter maps 1:1 to a distinct `mla_concat_qcur_batch_kernel`
    // launch argument; bundling them into a struct would just relocate the
    // count, not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_concat_qcur_batch(
        &self,
        m: &MlaModel,
        absorbed: &CudaSlice<f32>,
        q_pe: &CudaSlice<f32>,
        rows: usize,
        n_head: usize,
        kv_lora: usize,
        qk_rope: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let qk_dim = kv_lora + qk_rope;
        let mut out = self
            .device
            .alloc_zeros::<f32>(rows * n_head * qk_dim)
            .map_err(|e| crate::gpu_err!(e, "mla_concat_qcur_batch alloc: {e}"))?;
        let n = (rows * n_head * qk_dim) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            m.mla_concat_qcur_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        absorbed,
                        q_pe,
                        &mut out,
                        rows as u32,
                        n_head as u32,
                        kv_lora as u32,
                        qk_rope as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "mla_concat_qcur_batch launch: {e}"))?;
        }
        Ok(out)
    }

    /// Batched-prefill helper: writes this batch's compressed Kcur (`kv_cmpr` concat
    /// `k_pe`) into `kv_cache` at rows `start_pos..start_pos+rows`
    /// (`mla_write_kv_cache_batch_kernel`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_write_kv_cache_batch(
        &self,
        m: &MlaModel,
        kv_cache: &mut KvBuf,
        kv_cmpr: &CudaSlice<f32>,
        k_pe: &CudaSlice<f32>,
        start_pos: usize,
        rows: usize,
        kv_lora: usize,
        qk_rope: usize,
    ) -> Result<(), ReflexError> {
        let qk_dim = kv_lora + qk_rope;
        let n = (rows * qk_dim) as u32;
        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let kernel = match kv_cache.dtype() {
            crate::kv_io::KvDtype::F32 => &m.mla_write_kv_cache_batch_k,
            crate::kv_io::KvDtype::F16 => self.kv_f16.kernel(&self.device, KV_F16_MLA_WRITE)?,
        };
        let kv_cache = kv_cache.ptr_at(0);
        unsafe {
            kernel
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        kv_cache,
                        kv_cmpr,
                        k_pe,
                        start_pos as u32,
                        rows as u32,
                        kv_lora as u32,
                        qk_rope as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "mla_write_kv_cache_batch launch: {e}"))?;
        }
        Ok(())
    }
}

/// The body of `Model::gemm_ex_f16`, callable without a `Model` (the load-time
/// cuBLAS warm-up in `Model::warm_cublas` runs exactly this call). See that method
/// for the layout and math-mode contract.
pub(super) fn gemm_ex_f16_raw(
    cublas: &CudaBlas,
    w: sys::CUdeviceptr,
    x16: &CudaSlice<half::f16>,
    y: &mut CudaSlice<f32>,
    out_features: usize,
    rows: usize,
    in_features: usize,
) -> Result<(), ReflexError> {
    let handle = *cublas.handle();
    let set_math = |mode| unsafe {
        cublas_sys::lib()
            .cublasSetMathMode(handle, mode)
            .result()
            .map_err(|e| crate::gpu_err!(e, "cublasSetMathMode: {e:?}"))
    };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    set_math(cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH)?;
    let launched = unsafe {
        cudarc::cublas::result::gemm_ex(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            out_features as i32,
            rows as i32,
            in_features as i32,
            (&alpha) as *const f32 as *const core::ffi::c_void,
            w as *const core::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_16F,
            in_features as i32,
            *x16.device_ptr() as *const core::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_16F,
            in_features as i32,
            (&beta) as *const f32 as *const core::ffi::c_void,
            *y.device_ptr() as *mut core::ffi::c_void,
            cublas_sys::cudaDataType::CUDA_R_32F,
            out_features as i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    set_math(cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)?;
    launched.map_err(|e| crate::gpu_err!(e, "gemm_ex_f16 launch: {e:?}"))
}

/// Rows per `gemv_q4k_kernel` launch -- must equal `GEMV_Q4K_MAX_ROWS` in
/// `kernels_cuda/gemv_q4k.cu`.
pub(super) const GEMV_Q4K_MAX_ROWS: usize = 8;

/// `gemv_q4k_r1_kernel`'s output rows per warp and warps per block -- must equal
/// `GEMV_Q4K_R1_ROWS`/`GEMV_Q4K_R1_WARPS` in `kernels_cuda/gemv_q4k.cu`.
pub(super) const GEMV_Q4K_R1_ROWS: usize = 4;
pub(super) const GEMV_Q4K_R1_WARPS: usize = 4;

/// `gemv_q6k_r1_kernel`'s output rows per warp and warps per block -- must equal
/// `GEMV_Q6K_R1_ROWS`/`GEMV_Q6K_R1_WARPS` in `kernels_cuda/gemv_q4k.cu`.
pub(super) const GEMV_Q6K_R1_ROWS: usize = 4;
pub(super) const GEMV_Q6K_R1_WARPS: usize = 4;

/// Rows per `gemv_q8_0_kernel` launch -- must equal `GEMV_Q8_0_MAX_ROWS` in
/// `kernels_cuda/gemv_q8_0.cu`.
pub(super) const GEMV_Q8_0_MAX_ROWS: usize = 16;

/// Output rows per warp in `gemv_q8_0_kernel` -- must equal
/// `GEMV_Q8_0_ROWS_PER_WARP` in `kernels_cuda/gemv_q8_0.cu`.
pub(super) const GEMV_Q8_0_ROWS_PER_WARP: u32 = 4;

/// Elements of `x` per row that `gemv_q8_0_kernel` stages in shared memory at
/// a time -- must equal `Q8_0_TILE` in `kernels_cuda/gemv_q8_0.cu`; the
/// launch passes `rows * GEMV_Q8_0_TILE` floats of dynamic shared memory.
pub(super) const GEMV_Q8_0_TILE: usize = 512;

/// The `kernels_cuda/gemv_q8_0.cu` kernels, loaded only when the Gemma 4 path
/// keeps Q8_0 weights quantized (QR is the default).
pub(super) struct Q8_0Kernels {
    /// `gemv_q8_0_kernel`: the fused 2-16 row prefill path (`x` staged in
    /// shared memory, 4 output rows per warp).
    pub(super) gemv: AotKernel,
    /// `gemv_q8_0_direct_kernel`: the one-row decode GEMV.
    pub(super) gemv_direct: AotKernel,
    /// `dequantize_q8_0_coalesced_kernel`: the `--weights f32` prefill scratch.
    pub(super) dequant_f32: AotKernel,
    /// `dequantize_q8_0_coalesced_f16_kernel`: the `--weights f16` scratch.
    pub(super) dequant_f16: AotKernel,
}

impl Q8_0Kernels {
    pub(super) fn load(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let mut fns = aot::load_kernel_module(
            device,
            include_bytes!(env!("REFLEX_KERNEL_GEMV_Q8_0")),
            "gemv_q8_0",
            &[
                "gemv_q8_0_kernel",
                "gemv_q8_0_direct_kernel",
                "dequantize_q8_0_coalesced_kernel",
                "dequantize_q8_0_coalesced_f16_kernel",
            ],
        )?
        .into_iter();
        let mut next = || {
            fns.next()
                .ok_or_else(|| crate::reflex_err!(Cuda, "missing kernel in module gemv_q8_0"))
        };
        Ok(Self {
            gemv: next()?,
            gemv_direct: next()?,
            dequant_f32: next()?,
            dequant_f16: next()?,
        })
    }
}

/// Largest prefill row count the quantized-resident path runs through the
/// fused `gemv_q4k_kernel` before switching to dequantize-to-scratch + cuBLAS
/// (`Model::gemm`). `REFLEX_QUANT_FUSED_MAX_ROWS` overrides it. Default 16: on
/// a T4 (Qwen3-0.6B) the fused path won at 12 rows (system1 prompt eval 49.4
/// vs 51.2 ms) and lost at 29 (warm prefill 63.9 vs 35.9 ms); 16 sits between
/// those two measured points and has not been measured itself.
pub(super) fn quant_fused_max_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("REFLEX_QUANT_FUSED_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}
