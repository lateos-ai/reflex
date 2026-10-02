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
/// `kernels_cuda/attention_online.cu` field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AttnParams {
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
}

/// `attention_online.cu`'s block size, in warps, and widest supported head.
const ATTN_ONLINE_WARPS: usize = 8;
const ATTN_ONLINE_MAX_DIM: usize = 1024;
/// A decode step splits its positions across blocks once there are more than this
/// many per block, up to `ATTN_ONLINE_MAX_SPLITS` blocks per head.
const ATTN_ONLINE_SPLIT_POSITIONS: usize = 256;
const ATTN_ONLINE_MAX_SPLITS: usize = 64;

impl Model {
    /// Runs `attention_online_kernel` (and, for a split decode, its combine kernel):
    /// the default implementation behind [`Self::attention`],
    /// [`Self::attention_prefill`], [`Self::mla_attention`] and
    /// [`Self::mla_attention_prefill`]. Returns `[rows, num_q_heads, v_dim]`.
    pub(super) fn attention_online(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaView<f32>,
        v: &CudaView<f32>,
        s: OnlineAttnShape,
    ) -> Result<CudaSlice<f32>, ReflexError> {
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
            scale: s.scale,
        };
        let cfg = LaunchConfig {
            grid_dim: (s.rows as u32, s.num_q_heads as u32, num_splits as u32),
            block_dim: ((ATTN_ONLINE_WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: ((s.qk_dim + ATTN_ONLINE_WARPS * (s.v_dim + 2))
                * std::mem::size_of::<f32>()) as u32,
        };
        let mut out = self
            .device
            .alloc_zeros::<f32>(s.rows * s.num_q_heads * s.v_dim)
            .map_err(|e| crate::gpu_err!(e, "attn_online alloc out: {e}"))?;
        if num_splits == 1 {
            unsafe {
                self.attn_online_k
                    .function
                    .clone()
                    .launch(cfg, (q, k, v, &mut out, params))
                    .map_err(|e| crate::gpu_err!(e, "attn_online launch: {e}"))?;
            }
            return Ok(out);
        }
        let mut partial = self
            .device
            .alloc_zeros::<f32>(s.rows * s.num_q_heads * num_splits * (s.v_dim + 2))
            .map_err(|e| crate::gpu_err!(e, "attn_online alloc partial: {e}"))?;
        unsafe {
            self.attn_online_k
                .function
                .clone()
                .launch(cfg, (q, k, v, &mut partial, params))
                .map_err(|e| crate::gpu_err!(e, "attn_online launch: {e}"))?;
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
                        &partial,
                        &mut out,
                        s.num_q_heads as u32,
                        s.v_dim as u32,
                        num_splits as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "attn_online_combine launch: {e}"))?;
        }
        Ok(out)
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
        let n = x.len() as u32;
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(x.len())
            .map_err(|e| crate::gpu_err!(e, "rmsnorm alloc out: {e}"))?;

        let threads = 256u32;
        let blocks = n.div_ceil(threads).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.rmsnorm_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        weight,
                        &mut dev_out,
                        rows as u32,
                        hidden_size as u32,
                        eps,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "rmsnorm launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// `x` and `w_dev` are both already device-resident (Phase 2 round 2) --
    /// `w_dev` is either `&self.data` on a whole [`Weight`] (a
    /// `&CudaSlice<f32>`) or a zero-copy `CudaView` slice of one (see
    /// `Self::gemv_expert`).
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

        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv alloc y: {e}"))?;

        // One warp per output row (gemv_kernel's doc comment has the
        // coalescing rationale) -- 256 threads/block = 8 warps/block, same
        // total thread count per block as before this rewrite, just
        // reinterpreted as 8 rows/block instead of 256 threads each doing
        // one full row.
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let blocks = (out_features as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w_dev,
                        &mut dev_y,
                        in_features as u32,
                        out_features as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv launch: {e}"))?;
        }
        Ok(dev_y)
    }

    pub(super) fn gemv(
        &self,
        x: &CudaSlice<f32>,
        w: &Weight,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let in_features = w.shape[0] as usize;
        let out_features = w.shape[1] as usize;
        match &w.data {
            WeightData::F32(d) => self.gemv_raw(x, d, in_features, out_features),
            WeightData::Q4K { .. } => self.gemv_q4k(x, w, 1),
        }
    }

    /// `y[rows, out_features] = x[rows, in_features] @ w^T` straight from a
    /// quantized-resident Q4_K weight (`gemv_q4k_kernel`), launched in chunks
    /// of [`GEMV_Q4K_MAX_ROWS`] rows -- each chunk reads the weight's bytes
    /// once for all its rows. `rows == 1` is the decode GEMV.
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
        let dev_y = self
            .device
            .alloc_zeros::<f32>(rows * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv_q4k alloc y: {e}"))?;
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
        let w_data = match &w.data {
            WeightData::F32(d) => d,
            WeightData::Q4K { len, .. } => {
                // Quantized-resident weight: up to the threshold, the fused
                // kernel reads the Q4_K bytes once per 8 rows; above it,
                // dequantize the weight into a reused scratch buffer on the
                // device (no host traffic) and use cuBLAS as for f32.
                if rows <= quant_fused_max_rows() {
                    return self.gemv_q4k(x, w, rows);
                }
                let n = in_features * out_features;
                let mut scratch = self.quant_scratch.borrow_mut();
                if scratch.as_ref().is_none_or(|s| s.len() < n) {
                    *scratch = Some(
                        unsafe { self.device.alloc::<f32>(n) }
                            .map_err(|e| crate::gpu_err!(e, "quant scratch alloc: {e}"))?,
                    );
                }
                let buf = scratch.as_mut().expect("just ensured");
                let num_blocks = *len / Q4K_BLOCK_BYTES;
                let threads = 256u32;
                let launch_cfg = LaunchConfig {
                    grid_dim: ((num_blocks as u32).div_ceil(threads).max(1), 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let w_ptr = w.quant_ptr().expect("Q4K weight has a device pointer");
                unsafe {
                    self.dequant_kernels
                        .q4k
                        .function
                        .clone()
                        .launch(launch_cfg, (w_ptr, &mut *buf, num_blocks as u32))
                        .map_err(|e| crate::gpu_err!(e, "quant scratch dequant launch: {e}"))?;
                }
                let view = buf.slice(0..n);
                return self.gemm_view(x, &view, in_features, out_features, rows);
            }
        };

        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(rows * out_features)
            .map_err(|e| crate::gpu_err!(e, "gemm alloc y: {e}"))?;

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
        unsafe {
            self.cublas
                .gemm(cfg, w_data, x, &mut dev_y)
                .map_err(|e| crate::gpu_err!(e, "gemm launch: {e:?}"))?;
        }
        Ok(dev_y)
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
    pub(super) fn gemm_view<X: DevicePtr<f32>, W: DevicePtr<f32>>(
        &self,
        x: &X,
        w: &W,
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
        unsafe {
            self.cublas
                .gemm(cfg, w, x, &mut dev_y)
                .map_err(|e| crate::gpu_err!(e, "gemm_view launch: {e:?}"))?;
        }
        Ok(dev_y)
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
    ) -> Result<(CudaView<'a, f32>, usize, usize), ReflexError> {
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
            w.f32()?.slice(start..start + expert_len),
            in_features,
            out_features,
        ))
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
        self.gemv_view(x, &view, in_features, out_features)
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
        unsafe {
            self.gemv_gather_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w.f32()?,
                        &dev_indices,
                        &mut dev_y,
                        in_features as u32,
                        num_rows as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_gather launch: {e}"))?;
        }
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
        let mut dev_out = self
            .device
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| crate::gpu_err!(e, "silu alloc out: {e}"))?;

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
                .launch(
                    launch_cfg,
                    (gate, up, &mut dev_out, 1u32, hidden_size as u32),
                )
                .map_err(|e| crate::gpu_err!(e, "silu launch: {e}"))?;
        }
        Ok(dev_out)
    }

    /// Causal single-new-query attention against the full K/V cache so far
    /// (`k_cache`/`v_cache` views already include this position's own K/V --
    /// `seq_len = position + 1`). GQA-grouped: query head `h` reads KV head
    /// `h / (num_q_heads / num_kv_heads)`. `q`/`k_cache`/`v_cache` are all
    /// already device-resident (Phase 2 round 2) -- `k_cache`/`v_cache` are
    /// `CudaView`s into a preallocated per-layer device buffer, not a fresh
    /// upload of the whole cache history on every call.
    // Each parameter maps 1:1 to a distinct attention-kernel launch argument;
    // bundling them into a struct would just relocate the count, not reduce it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaView<f32>,
        v_cache: &CudaView<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            return self.attention_online(
                q,
                k_cache,
                v_cache,
                OnlineAttnShape {
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
                },
            );
        }
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

    /// Batched-prefill variant of [`Self::attention`]: scores `rows` new
    /// query rows against the shared K/V cache in one launch (grid gains a
    /// query-row dimension), each row causally masked to its own
    /// `start_pos + row + 1` positions instead of one shared `seq_len`.
    /// `k_cache`/`v_cache` must already cover `0..start_pos+rows` positions
    /// (this batch's own K/V, written by the caller before this call --
    /// `Self::forward_attn_block_batched`). `q` is row-major `[rows,
    /// num_q_heads, head_dim]`; returns row-major `[rows, num_q_heads,
    /// head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill(
        &self,
        q: &CudaSlice<f32>,
        k_cache: &CudaView<f32>,
        v_cache: &CudaView<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        if self.attn_impl == AttnImpl::Online {
            return self.attention_online(
                q,
                k_cache,
                v_cache,
                OnlineAttnShape {
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
                },
            );
        }
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
        let n = a.len() as u32;
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
                .launch(launch_cfg, (a, b, n))
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
    pub(super) fn gemv_view<X: DeviceRepr, W: DeviceRepr>(
        &self,
        x: X,
        w_dev: W,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, ReflexError> {
        let mut dev_y = self
            .device
            .alloc_zeros::<f32>(out_features)
            .map_err(|e| crate::gpu_err!(e, "gemv_view alloc y: {e}"))?;
        // One warp per output row -- see gemv_raw's identical comment. Same
        // gemv_kernel, so this must stay in lockstep with gemv_raw's launch
        // geometry.
        let threads = 256u32;
        let warps_per_block = threads / WARP_SIZE;
        let blocks = (out_features as u32).div_ceil(warps_per_block).max(1);
        let launch_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.gemv_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w_dev,
                        &mut dev_y,
                        in_features as u32,
                        out_features as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_view launch: {e}"))?;
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
            let w_view = w
                .f32()?
                .slice(h * in_features * out_features..(h + 1) * in_features * out_features);
            let x_view = x.slice(h * in_features..(h + 1) * in_features);
            let y = self.gemv_view(&x_view, &w_view, in_features, out_features)?;
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
        unsafe {
            m.gemv_per_head_batch_k
                .function
                .clone()
                .launch(
                    launch_cfg,
                    (
                        x,
                        w.f32()?,
                        &mut out,
                        rows as u32,
                        n_head as u32,
                        in_features as u32,
                        out_features as u32,
                        x_row_stride as u32,
                        x_head_stride as u32,
                        x_head_offset as u32,
                    ),
                )
                .map_err(|e| crate::gpu_err!(e, "gemv_per_head_batch launch: {e}"))?;
        }
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
        kv_cache: &CudaView<f32>,
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
                },
            );
        }
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
        kv_cache: &CudaView<f32>,
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
                },
            );
        }
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
        kv_cache: &mut CudaSlice<f32>,
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
        unsafe {
            m.mla_write_kv_cache_batch_k
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

/// Rows per `gemv_q4k_kernel` launch -- must equal `GEMV_Q4K_MAX_ROWS` in
/// `kernels_cuda/gemv_q4k.cu`.
pub(super) const GEMV_Q4K_MAX_ROWS: usize = 8;

/// Largest prefill row count the quantized-resident path runs through the
/// fused `gemv_q4k_kernel` before switching to dequantize-to-scratch + cuBLAS
/// (`Model::gemm`). `REFLEX_QUANT_FUSED_MAX_ROWS` overrides it, for measuring
/// the crossover; the default is a placeholder until that is measured.
pub(super) fn quant_fused_max_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("REFLEX_QUANT_FUSED_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    })
}
