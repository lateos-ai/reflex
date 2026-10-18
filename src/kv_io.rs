//! Phase 3 (State I/O). Round 1 added raw KV-cache export/import for the
//! dense/MoE Qwen3 forward path only, and stopped at proving the file
//! round-trips byte-identical through a device upload/download -- there was
//! no per-token generation loop or `start_pos` anywhere in `model.rs` for
//! `--import-kv` to resume into. Round 2 adds that generation loop (see
//! `Model::generate` in `model.rs`) and extends export/import to the
//! Qwen3.5 hybrid path's per-layer state too (`HybridKvCache` -- the
//! `GatedAttention` sublayers' `k_cache`/`v_cache` need the same
//! `start_pos`-offset treatment as dense/MoE's; the `GatedDeltaNet`
//! sublayers' `conv_state`/`recurrent` are already streaming-friendly --
//! fixed-size, not indexed by position -- so they round-trip as-is). Round 3
//! adds MLA's single compressed latent-KV cache (`MlaKvCache` -- one buffer
//! per layer, shape `[seq_len, kv_lora_rank + qk_rope_head_dim]`, unlike
//! dense/MoE's separate K/V pair or hybrid's per-layer Attn/Gdn split), the
//! last of the three cache shapes this project's architectures produce. The
//! engine stays ignorant of where the file lives (NVMe, S3-backed FUSE,
//! tmpfs) -- see docs/DEVELOPMENT.md's Non-goals.
//!
//! File format is a flat, home-grown binary layout, not a stable public
//! spec: magic + version-gated, so the version-1 dense/MoE-only layout
//! round-1 shipped stays readable unchanged while version 2 adds the hybrid
//! layout and version 3 the MLA layout alongside it -- [`import_kv`]
//! dispatches on the version field to pick between them.
//!
//! **Importing reads only the header.** [`import_kv`] parses it, checks that
//! the file is exactly as long as the header says, and returns
//! [`KvValues::File`] ranges. The payload is read later, chunk by chunk,
//! straight into the model's pinned staging buffers on its way to the device
//! (`WeightLoadPipeline::upload_state`): reading it into fresh host vectors
//! first faulted in every page of two new buffers, which made a cold process's
//! read of a 342 MB file 2.6x slower than a warm one's (see
//! docs/design/agent-relay.md, "Faster cache files (phase F)"). The caller
//! owns the file and must not change it while a request resumes from it; a
//! file that shrinks in between fails that request with `KvCache`.
//!
//! Version 1 (dense/MoE) layout: `b"CSKV"` | version:u32 LE (=1) |
//! num_layers:u32 LE | seq_len:u32 LE | num_kv_heads:u32 LE | head_dim:u32
//! LE | per layer, k_cache then v_cache, each `seq_len * num_kv_heads *
//! head_dim` f32 LE values.
//!
//! Version 4 (dense/MoE, what [`export_dense_kv`] writes now): version 1's
//! layout with version = 4 and one byte after `head_dim`: `space_prefix_next`
//! (0 or 1), whether the cached prompt's encoding ended on a special token, so a
//! SentencePiece continuation gets its dummy-prefix space exactly when the
//! joined text would (`Tokenizer::encode_continuation`). A version-1 file has
//! no such byte; resuming from one adds the space, as before.
//!
//! Version 2 (hybrid) layout: `b"CSKV"` | version:u32 LE (=2) |
//! num_layers:u32 LE | seq_len:u32 LE (attn positions cached so far) |
//! attn_num_kv_heads:u32 LE | attn_head_dim:u32 LE | gdn_conv_state_len:u32
//! LE | gdn_recurrent_len:u32 LE | num_layers layer-kind bytes (0 =
//! GatedAttention, 1 = GatedDeltaNet) | per layer, in layer order: if
//! GatedAttention, k_cache then v_cache (each `seq_len * attn_num_kv_heads *
//! attn_head_dim` f32 LE values); if GatedDeltaNet, conv_state
//! (`gdn_conv_state_len` f32 LE values) then recurrent (`gdn_recurrent_len`
//! f32 LE values).
//!
//! Version 3 (MLA) layout: `b"CSKV"` | version:u32 LE (=3) | num_layers:u32
//! LE | seq_len:u32 LE | qk_dim:u32 LE (`kv_lora_rank + qk_rope_head_dim`) |
//! per layer, one `seq_len * qk_dim` f32 LE `kv_cache` (the compressed
//! `Kcur == kv_cmpr_normed ++ k_pe` latent, no separate V -- see
//! `Model::forward_mla_attn_block`).
//!
//! **f16 files** (opt-in: IPC `export_kv_dtype: "f16"`, CLI `--export-kv-dtype
//! f16`) halve the payload. Each is the f32 layout above with only the version
//! and the element type changed:
//! - version 5: version 4 (dense/MoE), every K/V value f16 LE;
//! - version 6: version 2 (hybrid), attention K/V f16 LE, while the
//!   GatedDeltaNet `conv_state`/`recurrent` stay f32 LE, as llama.cpp keeps
//!   recurrent state f32 next to an f16 KV cache. Rounding that state too
//!   measured ~15% more logit error than rounding the K/V alone (Qwen3.5-0.8B,
//!   docs/design/agent-relay.md, "Faster cache files (phase F)"), small either
//!   way, but it is a fixed size per model (~20 MB there), so keeping it f32
//!   is what makes a short hybrid prefix's f16 file barely smaller;
//! - version 7: version 3 (MLA), the latent f16 LE.
//!
//! The values are rounded to f16 on the device when exported (round to
//! nearest even; a value outside f16's range fails the export instead of
//! becoming infinite) and widened back to f32 on the device when imported, so
//! no host loop touches the payload. The working cache stays f32 either way.

use crate::error::ReflexError;
use std::io::Write;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"CSKV";
const VERSION_DENSE: u32 = 1;
const VERSION_HYBRID: u32 = 2;
const VERSION_MLA: u32 = 3;
const VERSION_DENSE_V4: u32 = 4;
const VERSION_DENSE_F16: u32 = 5;
const VERSION_HYBRID_F16: u32 = 6;
const VERSION_MLA_F16: u32 = 7;

/// More layers than any model has: a corrupt header's count fails here
/// instead of laying out billions of ranges.
const MAX_LAYERS: usize = 1 << 16;

/// Element type of a cache file's values (see the module doc's f16 files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvDtype {
    #[default]
    F32,
    F16,
}

impl KvDtype {
    pub fn size(self) -> usize {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KvDtype::F32 => "f32",
            KvDtype::F16 => "f16",
        }
    }

    /// Parses `"f32"`/`"f16"` (IPC `export_kv_dtype`, CLI `--export-kv-dtype`).
    pub fn parse(s: &str) -> Result<Self, ReflexError> {
        match s {
            "f32" => Ok(KvDtype::F32),
            "f16" => Ok(KvDtype::F16),
            other => Err(ReflexError::InvalidInput(format!(
                "unknown KV cache file dtype {other:?} (expected \"f32\" or \"f16\")"
            ))),
        }
    }
}

/// One cached buffer's values: in host memory (a download from the device,
/// or a test), or a range of an imported cache file that is read only when
/// it is uploaded.
#[derive(Debug, Clone)]
pub enum KvValues {
    Host(Vec<f32>),
    /// f16 bit patterns, as an f16 export downloads them.
    HostF16(Vec<u16>),
    File(FileRange),
}

/// `len` LE values of type `dtype` at byte `offset` of an imported cache file.
#[derive(Debug, Clone)]
pub struct FileRange {
    file: Arc<std::fs::File>,
    offset: u64,
    len: usize,
    dtype: KvDtype,
}

impl KvValues {
    /// Number of values.
    pub fn len(&self) -> usize {
        match self {
            KvValues::Host(v) => v.len(),
            KvValues::HostF16(v) => v.len(),
            KvValues::File(r) => r.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> KvDtype {
        match self {
            KvValues::Host(_) => KvDtype::F32,
            KvValues::HostF16(_) => KvDtype::F16,
            KvValues::File(r) => r.dtype,
        }
    }

    /// Size of the values as stored in a file (and as uploaded).
    pub fn byte_len(&self) -> usize {
        self.len() * self.dtype().size()
    }

    /// Copies bytes `[start, start + buf.len())` of the values' little-endian
    /// encoding into `buf`. For a file range this is one positioned read, so
    /// the bytes land in `buf` (a pinned staging buffer during an upload)
    /// without any intermediate host buffer.
    pub fn read_bytes(&self, start: usize, buf: &mut [u8]) -> Result<(), ReflexError> {
        match self {
            KvValues::Host(v) => {
                buf.copy_from_slice(&f32_le_bytes(v)[start..start + buf.len()]);
                Ok(())
            }
            KvValues::HostF16(v) => {
                buf.copy_from_slice(&u16_le_bytes(v)[start..start + buf.len()]);
                Ok(())
            }
            KvValues::File(r) => {
                read_exact_at(&r.file, buf, r.offset + start as u64).map_err(|e| {
                    crate::reflex_err!(
                        KvCache,
                        "read KV cache payload: {e} (was the file changed after import?)"
                    )
                })
            }
        }
    }

    /// The values as host f32s (f16 widened exactly), reading a file range.
    /// For tests and host-side checks; uploads convert on the device and never
    /// go through this.
    pub fn to_f32(&self) -> Result<Vec<f32>, ReflexError> {
        if let KvValues::Host(v) = self {
            return Ok(v.clone());
        }
        let mut bytes = vec![0u8; self.byte_len()];
        self.read_bytes(0, &mut bytes)?;
        Ok(match self.dtype() {
            KvDtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            KvDtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
        })
    }
}

impl From<Vec<f32>> for KvValues {
    fn from(v: Vec<f32>) -> Self {
        KvValues::Host(v)
    }
}

/// `v`'s little-endian bytes: a view on little-endian hosts, a copy otherwise.
fn f32_le_bytes(v: &[f32]) -> std::borrow::Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        // SAFETY: f32 has no padding and any byte pattern is a valid u8.
        std::borrow::Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        })
    } else {
        std::borrow::Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
}

/// `v`'s little-endian bytes, as [`f32_le_bytes`].
fn u16_le_bytes(v: &[u16]) -> std::borrow::Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        // SAFETY: u16 has no padding and any byte pattern is a valid u8.
        std::borrow::Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        })
    } else {
        std::borrow::Cow::Owned(v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
}

/// The element type all of `values` share, or an error naming `what`.
fn common_dtype<'a>(
    mut values: impl Iterator<Item = &'a KvValues>,
    what: &str,
) -> Result<KvDtype, ReflexError> {
    let first = values.next().map_or(KvDtype::F32, KvValues::dtype);
    if values.any(|v| v.dtype() != first) {
        return Err(crate::reflex_err!(
            KvCache,
            "{what} mix f32 and f16 values; a cache file holds one type"
        ));
    }
    Ok(first)
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(file, buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut std::mem::take(&mut buf)[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct DenseKvCache {
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub k_caches: Vec<KvValues>,
    pub v_caches: Vec<KvValues>,
    /// Whether the cached prompt's encoding ended on a special token (see the
    /// module doc's version 4). `None` for a version-1 file, which predates it.
    pub space_prefix_next: Option<bool>,
}

/// One hybrid layer's cached state, matching `model.rs`'s `HybridLayerState`
/// variant for that layer index one-to-one.
#[derive(Debug)]
pub enum HybridLayerCacheData {
    Attn {
        k_cache: KvValues,
        v_cache: KvValues,
    },
    Gdn {
        conv_state: KvValues,
        recurrent: KvValues,
    },
}

#[derive(Debug)]
pub struct HybridKvCache {
    pub seq_len: usize,
    pub attn_num_kv_heads: usize,
    pub attn_head_dim: usize,
    pub gdn_conv_state_len: usize,
    pub gdn_recurrent_len: usize,
    pub layers: Vec<HybridLayerCacheData>,
}

/// MLA's per-layer compressed latent-KV cache, matching
/// `Model::forward_mla_attn_block`'s `kv_cache` shape one-to-one: a single
/// `[seq_len, qk_dim]` buffer per layer (`qk_dim == kv_lora_rank +
/// qk_rope_head_dim`), not a separate K/V pair -- MLA's whole point is that
/// the compressed latent is shared and decompressed on the fly by `wk_b`/
/// `wv_b`, so there's nothing else to cache.
#[derive(Debug)]
pub struct MlaKvCache {
    pub seq_len: usize,
    pub qk_dim: usize,
    pub kv_caches: Vec<KvValues>,
}

/// What `import_kv` found in the file -- dispatched on by `Model::generate`
/// to pick the matching resume path (and to reject a format/architecture
/// mismatch with a clear error instead of silently misreading bytes). Also
/// what `Model::generate_capture_kv` hands back for [`export_kv`].
#[derive(Debug)]
pub enum ImportedKv {
    Dense(DenseKvCache),
    Hybrid(HybridKvCache),
    Mla(MlaKvCache),
}

impl ImportedKv {
    /// The layout's name, for error messages.
    pub fn format_name(&self) -> &'static str {
        match self {
            ImportedKv::Dense(_) => "dense/MoE",
            ImportedKv::Hybrid(_) => "hybrid",
            ImportedKv::Mla(_) => "MLA",
        }
    }

    /// Cached positions and layers.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            ImportedKv::Dense(c) => (c.seq_len, c.k_caches.len()),
            ImportedKv::Hybrid(c) => (c.seq_len, c.layers.len()),
            ImportedKv::Mla(c) => (c.seq_len, c.kv_caches.len()),
        }
    }
}

/// Writes any of the three layouts, the counterpart to [`import_kv`].
pub fn export_kv(path: &str, cache: &ImportedKv) -> Result<(), ReflexError> {
    match cache {
        ImportedKv::Dense(c) => export_dense_kv(path, c),
        ImportedKv::Hybrid(c) => export_hybrid_kv(path, c),
        ImportedKv::Mla(c) => export_mla_kv(path, c),
    }
}

fn write_u32s(f: &mut std::fs::File, fields: &[(&str, usize)]) -> Result<(), ReflexError> {
    for (name, v) in fields {
        f.write_all(&(*v as u32).to_le_bytes())
            .map_err(|e| crate::reflex_err!(Io, "write {name}: {e}"))?;
    }
    Ok(())
}

fn create_with_header(path: &str, version: u32) -> Result<std::fs::File, ReflexError> {
    let mut f =
        std::fs::File::create(path).map_err(|e| crate::reflex_err!(Io, "create {path}: {e}"))?;
    f.write_all(MAGIC)
        .map_err(|e| crate::reflex_err!(Io, "write magic: {e}"))?;
    f.write_all(&version.to_le_bytes())
        .map_err(|e| crate::reflex_err!(Io, "write version: {e}"))?;
    Ok(f)
}

pub fn export_dense_kv(path: &str, cache: &DenseKvCache) -> Result<(), ReflexError> {
    let kv = cache.k_caches.iter().chain(&cache.v_caches);
    let version = match common_dtype(kv, "K/V caches")? {
        KvDtype::F32 => VERSION_DENSE_V4,
        KvDtype::F16 => VERSION_DENSE_F16,
    };
    let mut f = create_with_header(path, version)?;
    write_u32s(
        &mut f,
        &[
            ("num_layers", cache.k_caches.len()),
            ("seq_len", cache.seq_len),
            ("num_kv_heads", cache.num_kv_heads),
            ("head_dim", cache.head_dim),
        ],
    )?;
    // A cache built by this engine always knows; a re-export of a version-1
    // file keeps that file's behavior (a space).
    f.write_all(&[u8::from(cache.space_prefix_next.unwrap_or(true))])
        .map_err(|e| crate::reflex_err!(Io, "write space_prefix_next: {e}"))?;

    for (k, v) in cache.k_caches.iter().zip(&cache.v_caches) {
        write_values(&mut f, k)?;
        write_values(&mut f, v)?;
    }
    Ok(())
}

pub fn import_dense_kv(path: &str) -> Result<DenseKvCache, ReflexError> {
    match import_kv(path)? {
        ImportedKv::Dense(c) => Ok(c),
        other => Err(crate::reflex_err!(
            KvCache,
            "{path}: expected a dense/MoE KV cache file, found {}",
            other.format_name()
        )),
    }
}

pub fn export_hybrid_kv(path: &str, cache: &HybridKvCache) -> Result<(), ReflexError> {
    let attn = cache.layers.iter().flat_map(|l| match l {
        HybridLayerCacheData::Attn { k_cache, v_cache } => vec![k_cache, v_cache],
        HybridLayerCacheData::Gdn { .. } => vec![],
    });
    let gdn = cache.layers.iter().flat_map(|l| match l {
        HybridLayerCacheData::Gdn {
            conv_state,
            recurrent,
        } => vec![conv_state, recurrent],
        HybridLayerCacheData::Attn { .. } => vec![],
    });
    if common_dtype(gdn, "GatedDeltaNet states")? != KvDtype::F32 {
        return Err(crate::reflex_err!(
            KvCache,
            "hybrid cache files keep the GatedDeltaNet state in f32"
        ));
    }
    let version = match common_dtype(attn, "attention K/V caches")? {
        KvDtype::F32 => VERSION_HYBRID,
        KvDtype::F16 => VERSION_HYBRID_F16,
    };
    let mut f = create_with_header(path, version)?;
    write_u32s(
        &mut f,
        &[
            ("num_layers", cache.layers.len()),
            ("seq_len", cache.seq_len),
            ("attn_num_kv_heads", cache.attn_num_kv_heads),
            ("attn_head_dim", cache.attn_head_dim),
            ("gdn_conv_state_len", cache.gdn_conv_state_len),
            ("gdn_recurrent_len", cache.gdn_recurrent_len),
        ],
    )?;

    let kinds: Vec<u8> = cache
        .layers
        .iter()
        .map(|l| match l {
            HybridLayerCacheData::Attn { .. } => 0u8,
            HybridLayerCacheData::Gdn { .. } => 1u8,
        })
        .collect();
    f.write_all(&kinds)
        .map_err(|e| crate::reflex_err!(Io, "write layer kinds: {e}"))?;

    for layer in &cache.layers {
        match layer {
            HybridLayerCacheData::Attn { k_cache, v_cache } => {
                write_values(&mut f, k_cache)?;
                write_values(&mut f, v_cache)?;
            }
            HybridLayerCacheData::Gdn {
                conv_state,
                recurrent,
            } => {
                write_values(&mut f, conv_state)?;
                write_values(&mut f, recurrent)?;
            }
        }
    }
    Ok(())
}

pub fn import_hybrid_kv(path: &str) -> Result<HybridKvCache, ReflexError> {
    match import_kv(path)? {
        ImportedKv::Hybrid(c) => Ok(c),
        other => Err(crate::reflex_err!(
            KvCache,
            "{path}: expected a hybrid KV cache file, found {}",
            other.format_name()
        )),
    }
}

pub fn export_mla_kv(path: &str, cache: &MlaKvCache) -> Result<(), ReflexError> {
    let version = match common_dtype(cache.kv_caches.iter(), "latent caches")? {
        KvDtype::F32 => VERSION_MLA,
        KvDtype::F16 => VERSION_MLA_F16,
    };
    let mut f = create_with_header(path, version)?;
    write_u32s(
        &mut f,
        &[
            ("num_layers", cache.kv_caches.len()),
            ("seq_len", cache.seq_len),
            ("qk_dim", cache.qk_dim),
        ],
    )?;
    for kv in &cache.kv_caches {
        write_values(&mut f, kv)?;
    }
    Ok(())
}

pub fn import_mla_kv(path: &str) -> Result<MlaKvCache, ReflexError> {
    match import_kv(path)? {
        ImportedKv::Mla(c) => Ok(c),
        other => Err(crate::reflex_err!(
            KvCache,
            "{path}: expected an MLA KV cache file, found {}",
            other.format_name()
        )),
    }
}

/// Reads the header, dispatches on the version to the matching layout, and
/// checks the file holds exactly the payload the header describes. The
/// payload itself is not read here (see the module doc).
pub fn import_kv(path: &str) -> Result<ImportedKv, ReflexError> {
    let file = std::fs::File::open(path).map_err(|e| crate::reflex_err!(Io, "open {path}: {e}"))?;
    let file_len = file
        .metadata()
        .map_err(|e| crate::reflex_err!(Io, "stat {path}: {e}"))?
        .len();
    let mut h = Header {
        file: Arc::new(file),
        pos: 0,
    };
    let mut magic = [0u8; 4];
    h.read(&mut magic, "magic")?;
    if &magic != MAGIC {
        return Err(crate::reflex_err!(
            KvCache,
            "{path}: not a reflex-engine KV cache file (bad magic)"
        ));
    }
    let version = h.u32("version")? as u32;
    let mut payload = Payload {
        file: h.file.clone(),
        next: 0,
    };
    use KvDtype::{F16, F32};
    let p = &mut payload;
    let kv = match version {
        VERSION_DENSE => ImportedKv::Dense(read_dense(&mut h, p, false, F32)?),
        VERSION_DENSE_V4 => ImportedKv::Dense(read_dense(&mut h, p, true, F32)?),
        VERSION_DENSE_F16 => ImportedKv::Dense(read_dense(&mut h, p, true, F16)?),
        VERSION_HYBRID => ImportedKv::Hybrid(read_hybrid(&mut h, p, F32)?),
        VERSION_HYBRID_F16 => ImportedKv::Hybrid(read_hybrid(&mut h, p, F16)?),
        VERSION_MLA => ImportedKv::Mla(read_mla(&mut h, p, F32)?),
        VERSION_MLA_F16 => ImportedKv::Mla(read_mla(&mut h, p, F16)?),
        other => {
            return Err(crate::reflex_err!(
                KvCache,
                "{path}: unsupported KV cache format version {other} (expected 1 to 7)"
            ))
        }
    };
    // `payload.next` counts from the end of the header; overflow while
    // summing ranges was already rejected.
    let expected = h.pos.checked_add(payload.next);
    if expected != Some(file_len) {
        return Err(match expected {
            Some(n) if n > file_len => crate::reflex_err!(
                KvCache,
                "{path}: truncated KV cache file: the header describes {n} bytes, the file has {file_len}"
            ),
            Some(n) => crate::reflex_err!(
                KvCache,
                "{path}: KV cache file has {} bytes after the payload the header describes",
                file_len - n
            ),
            None => crate::reflex_err!(KvCache, "{path}: KV cache header sizes overflow"),
        });
    }
    Ok(fix_offsets(kv, h.pos))
}

/// Sequential reader for the header fields.
struct Header {
    file: Arc<std::fs::File>,
    pos: u64,
}

impl Header {
    fn read(&mut self, buf: &mut [u8], what: &str) -> Result<(), ReflexError> {
        read_exact_at(&self.file, buf, self.pos)
            .map_err(|e| crate::reflex_err!(KvCache, "read {what}: {e}"))?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn u32(&mut self, what: &str) -> Result<usize, ReflexError> {
        let mut b = [0u8; 4];
        self.read(&mut b, what)?;
        Ok(u32::from_le_bytes(b) as usize)
    }

    fn num_layers(&mut self) -> Result<usize, ReflexError> {
        let n = self.u32("num_layers")?;
        if n > MAX_LAYERS {
            return Err(crate::reflex_err!(KvCache, "implausible layer count {n}"));
        }
        Ok(n)
    }
}

/// Lays the payload's ranges out one after another, relative to the end of
/// the header (whose length is known only once it has been read; see
/// [`fix_offsets`]).
struct Payload {
    file: Arc<std::fs::File>,
    next: u64,
}

impl Payload {
    fn take(&mut self, len: usize, dtype: KvDtype) -> Result<KvValues, ReflexError> {
        let bytes = (len as u64)
            .checked_mul(dtype.size() as u64)
            .and_then(|b| self.next.checked_add(b))
            .ok_or_else(|| crate::reflex_err!(KvCache, "KV cache header sizes overflow"))?;
        let r = FileRange {
            file: self.file.clone(),
            offset: self.next,
            len,
            dtype,
        };
        self.next = bytes;
        Ok(KvValues::File(r))
    }
}

fn checked_len(dims: &[usize]) -> Result<usize, ReflexError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| crate::reflex_err!(KvCache, "KV cache header sizes overflow"))
}

/// Shifts every range by the header's length `base`.
fn fix_offsets(mut kv: ImportedKv, base: u64) -> ImportedKv {
    let shift = |v: &mut KvValues| {
        if let KvValues::File(r) = v {
            r.offset += base;
        }
    };
    match &mut kv {
        ImportedKv::Dense(c) => c.k_caches.iter_mut().chain(&mut c.v_caches).for_each(shift),
        ImportedKv::Hybrid(c) => {
            for l in &mut c.layers {
                match l {
                    HybridLayerCacheData::Attn { k_cache, v_cache } => {
                        shift(k_cache);
                        shift(v_cache);
                    }
                    HybridLayerCacheData::Gdn {
                        conv_state,
                        recurrent,
                    } => {
                        shift(conv_state);
                        shift(recurrent);
                    }
                }
            }
        }
        ImportedKv::Mla(c) => c.kv_caches.iter_mut().for_each(shift),
    }
    kv
}

/// The dense header after the version field; `with_flag` for versions 4 and
/// 5, whose K/V values are `dtype`.
fn read_dense(
    h: &mut Header,
    p: &mut Payload,
    with_flag: bool,
    dtype: KvDtype,
) -> Result<DenseKvCache, ReflexError> {
    let num_layers = h.num_layers()?;
    let seq_len = h.u32("seq_len")?;
    crate::limits::check_positions(seq_len, 0, 0)?;
    let num_kv_heads = h.u32("num_kv_heads")?;
    let head_dim = h.u32("head_dim")?;
    let space_prefix_next = if with_flag {
        let mut b = [0u8; 1];
        h.read(&mut b, "space_prefix_next")?;
        match b[0] {
            0 => Some(false),
            1 => Some(true),
            other => {
                return Err(crate::reflex_err!(
                    KvCache,
                    "invalid space_prefix_next byte {other}"
                ))
            }
        }
    } else {
        None
    };
    let per_layer_len = checked_len(&[seq_len, num_kv_heads, head_dim])?;

    let mut k_caches = Vec::new();
    let mut v_caches = Vec::new();
    for _ in 0..num_layers {
        k_caches.push(p.take(per_layer_len, dtype)?);
        v_caches.push(p.take(per_layer_len, dtype)?);
    }

    Ok(DenseKvCache {
        seq_len,
        num_kv_heads,
        head_dim,
        k_caches,
        v_caches,
        space_prefix_next,
    })
}

/// The hybrid header after the version field; attention K/V values are
/// `attn_dtype`, the GatedDeltaNet state always f32.
fn read_hybrid(
    h: &mut Header,
    p: &mut Payload,
    attn_dtype: KvDtype,
) -> Result<HybridKvCache, ReflexError> {
    let num_layers = h.num_layers()?;
    let seq_len = h.u32("seq_len")?;
    crate::limits::check_positions(seq_len, 0, 0)?;
    let attn_num_kv_heads = h.u32("attn_num_kv_heads")?;
    let attn_head_dim = h.u32("attn_head_dim")?;
    let gdn_conv_state_len = h.u32("gdn_conv_state_len")?;
    let gdn_recurrent_len = h.u32("gdn_recurrent_len")?;

    let mut kinds = vec![0u8; num_layers];
    h.read(&mut kinds, "layer kinds")?;

    let attn_len = checked_len(&[seq_len, attn_num_kv_heads, attn_head_dim])?;
    let mut layers = Vec::with_capacity(num_layers);
    for (layer_idx, kind) in kinds.into_iter().enumerate() {
        match kind {
            0 => {
                let k_cache = p.take(attn_len, attn_dtype)?;
                let v_cache = p.take(attn_len, attn_dtype)?;
                layers.push(HybridLayerCacheData::Attn { k_cache, v_cache });
            }
            1 => {
                let conv_state = p.take(gdn_conv_state_len, KvDtype::F32)?;
                let recurrent = p.take(gdn_recurrent_len, KvDtype::F32)?;
                layers.push(HybridLayerCacheData::Gdn {
                    conv_state,
                    recurrent,
                });
            }
            other => {
                return Err(crate::reflex_err!(
                    KvCache,
                    "layer {layer_idx}: unknown hybrid layer kind byte {other} (expected 0 or 1)"
                ))
            }
        }
    }

    Ok(HybridKvCache {
        seq_len,
        attn_num_kv_heads,
        attn_head_dim,
        gdn_conv_state_len,
        gdn_recurrent_len,
        layers,
    })
}

fn read_mla(h: &mut Header, p: &mut Payload, dtype: KvDtype) -> Result<MlaKvCache, ReflexError> {
    let num_layers = h.num_layers()?;
    let seq_len = h.u32("seq_len")?;
    crate::limits::check_positions(seq_len, 0, 0)?;
    let qk_dim = h.u32("qk_dim")?;
    let per_layer_len = checked_len(&[seq_len, qk_dim])?;

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        kv_caches.push(p.take(per_layer_len, dtype)?);
    }

    Ok(MlaKvCache {
        seq_len,
        qk_dim,
        kv_caches,
    })
}

/// Largest piece of a file range copied through memory at once when
/// re-exporting it.
const COPY_CHUNK_BYTES: usize = 8 << 20;

fn write_values(f: &mut std::fs::File, values: &KvValues) -> Result<(), ReflexError> {
    let err = |e: std::io::Error| crate::reflex_err!(Io, "write KV cache values: {e}");
    match values {
        KvValues::Host(v) => f.write_all(&f32_le_bytes(v)).map_err(err),
        KvValues::HostF16(v) => f.write_all(&u16_le_bytes(v)).map_err(err),
        KvValues::File(_) => {
            let total = values.byte_len();
            let mut buf = vec![0u8; total.min(COPY_CHUNK_BYTES)];
            let mut start = 0;
            while start < total {
                let n = (total - start).min(buf.len());
                values.read_bytes(start, &mut buf[..n])?;
                f.write_all(&buf[..n]).map_err(err)?;
                start += n;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("reflex_kv_io_{name}_{}.bin", std::process::id()))
    }

    fn f32s(values: &[KvValues]) -> Vec<Vec<f32>> {
        values.iter().map(|v| v.to_f32().unwrap()).collect()
    }

    fn sample_dense() -> DenseKvCache {
        DenseKvCache {
            seq_len: 3,
            num_kv_heads: 2,
            head_dim: 4,
            k_caches: vec![
                (0..24)
                    .map(|i| i as f32 * 0.5 - 3.0)
                    .collect::<Vec<_>>()
                    .into(),
                (0..24)
                    .map(|i| -(i as f32) * 1.25)
                    .collect::<Vec<_>>()
                    .into(),
            ],
            v_caches: vec![
                (0..24).map(|i| (i as f32).sin()).collect::<Vec<_>>().into(),
                (0..24)
                    .map(|i| f32::MIN_POSITIVE * i as f32)
                    .collect::<Vec<_>>()
                    .into(),
            ],
            space_prefix_next: Some(false),
        }
    }

    #[test]
    fn export_then_import_dense_is_byte_exact() {
        let cache = sample_dense();
        let path = tmp("dense");
        let path_str = path.to_str().unwrap();

        export_dense_kv(path_str, &cache).expect("export failed");
        let reimported = import_dense_kv(path_str).expect("import failed");
        let via_import_kv = match import_kv(path_str).expect("import_kv failed") {
            ImportedKv::Dense(c) => c,
            _ => panic!("import_kv misdetected a dense file"),
        };
        let bytes = std::fs::read(&path).unwrap();
        // Header (25 bytes) plus the payload, nothing else.
        assert_eq!(bytes.len(), 25 + 4 * 24 * 4);
        // Re-exporting an imported cache (file ranges) to another path, and
        // the generic `export_kv`, write the same bytes.
        let copy = tmp("dense_copy");
        let wrapped = ImportedKv::Dense(reimported);
        assert_eq!(wrapped.format_name(), "dense/MoE");
        export_kv(copy.to_str().unwrap(), &wrapped).expect("export_kv failed");
        assert_eq!(std::fs::read(&copy).unwrap(), bytes);
        std::fs::remove_file(&copy).ok();
        let ImportedKv::Dense(reimported) = wrapped else {
            unreachable!()
        };

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.num_kv_heads, cache.num_kv_heads);
        assert_eq!(reimported.head_dim, cache.head_dim);
        assert_eq!(f32s(&reimported.k_caches), f32s(&cache.k_caches));
        assert_eq!(f32s(&reimported.v_caches), f32s(&cache.v_caches));
        assert_eq!(f32s(&via_import_kv.k_caches), f32s(&cache.k_caches));
        assert_eq!(reimported.space_prefix_next, Some(false));
        assert_eq!(via_import_kv.space_prefix_next, Some(false));
        // Ranges read in pieces give the same bytes as one read.
        let mut whole = vec![0u8; 96];
        reimported.v_caches[1].read_bytes(0, &mut whole).unwrap();
        let mut parts = vec![0u8; 96];
        reimported.v_caches[1]
            .read_bytes(0, &mut parts[..40])
            .unwrap();
        reimported.v_caches[1]
            .read_bytes(40, &mut parts[40..])
            .unwrap();
        assert_eq!(whole, parts);
        assert_eq!(&whole[..], &bytes[25 + 3 * 96..25 + 4 * 96]);
        std::fs::remove_file(&path).ok();
    }

    /// A version-1 file (no `space_prefix_next` byte) still imports, as `None`.
    #[test]
    fn import_reads_version_1_dense_files() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        for v in [VERSION_DENSE, 1, 1, 1, 2] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for x in [1.5f32, -2.0, 0.25, 4.0] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let path = tmp("v1");
        std::fs::write(&path, &bytes).unwrap();
        let path_str = path.to_str().unwrap();
        let ImportedKv::Dense(c) = import_kv(path_str).expect("v1 import") else {
            panic!("not dense")
        };
        assert_eq!(c.space_prefix_next, None);
        assert_eq!(f32s(&c.k_caches), vec![vec![1.5, -2.0]]);
        assert_eq!(f32s(&c.v_caches), vec![vec![0.25, 4.0]]);
        assert_eq!(import_dense_kv(path_str).unwrap().space_prefix_next, None);
        // Re-exporting writes version 4 and keeps the version-1 behavior (a space).
        let v4 = tmp("v1_to_v4");
        let v4_str = v4.to_str().unwrap();
        export_dense_kv(v4_str, &c).unwrap();
        assert_eq!(
            &std::fs::read(&v4).unwrap()[4..8],
            &VERSION_DENSE_V4.to_le_bytes()
        );
        assert_eq!(
            import_dense_kv(v4_str).unwrap().space_prefix_next,
            Some(true)
        );
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&v4).ok();
    }

    #[test]
    fn import_rejects_cache_longer_than_attention_limit() {
        let seq_len = crate::limits::ATTN_MAX_POSITIONS + 1;
        let cache = DenseKvCache {
            seq_len,
            num_kv_heads: 1,
            head_dim: 1,
            k_caches: vec![vec![0.0; seq_len].into()],
            v_caches: vec![vec![0.0; seq_len].into()],
            space_prefix_next: None,
        };

        let path = tmp("too_long");
        let path_str = path.to_str().unwrap();

        export_dense_kv(path_str, &cache).expect("export failed");
        let result = import_kv(path_str);
        std::fs::remove_file(&path).ok();
        let Err(err) = result else {
            panic!("oversized import must fail");
        };

        assert_eq!(err.code(), crate::error::ReflexErrorCode::ContextOverflow);
        assert!(
            err.to_string()
                .starts_with(crate::limits::CONTEXT_LENGTH_EXCEEDED_PREFIX),
            "{err}"
        );
    }

    #[test]
    fn export_then_import_hybrid_is_byte_exact() {
        let cache = HybridKvCache {
            seq_len: 2,
            attn_num_kv_heads: 2,
            attn_head_dim: 3,
            gdn_conv_state_len: 5,
            gdn_recurrent_len: 7,
            layers: vec![
                HybridLayerCacheData::Gdn {
                    conv_state: (0..5).map(|i| i as f32 * 0.1).collect::<Vec<_>>().into(),
                    recurrent: (0..7).map(|i| i as f32 * -0.2).collect::<Vec<_>>().into(),
                },
                HybridLayerCacheData::Attn {
                    k_cache: (0..12).map(|i| i as f32).collect::<Vec<_>>().into(),
                    v_cache: (0..12).map(|i| -(i as f32)).collect::<Vec<_>>().into(),
                },
            ],
        };

        let path = tmp("hybrid");
        let path_str = path.to_str().unwrap();

        export_hybrid_kv(path_str, &cache).expect("export failed");
        let reimported = import_hybrid_kv(path_str).expect("import failed");
        let via_import_kv = match import_kv(path_str).expect("import_kv failed") {
            ImportedKv::Hybrid(c) => c,
            _ => panic!("import_kv misdetected a hybrid file"),
        };
        let bytes = std::fs::read(&path).unwrap();
        let copy = tmp("hybrid_copy");
        export_hybrid_kv(copy.to_str().unwrap(), &reimported).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), bytes);
        std::fs::remove_file(&copy).ok();
        std::fs::remove_file(&path).ok();

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.layers.len(), cache.layers.len());
        match (&reimported.layers[0], &cache.layers[0]) {
            (
                HybridLayerCacheData::Gdn {
                    conv_state,
                    recurrent,
                },
                HybridLayerCacheData::Gdn {
                    conv_state: c2,
                    recurrent: r2,
                },
            ) => {
                assert_eq!(conv_state.to_f32().unwrap(), c2.to_f32().unwrap());
                assert_eq!(recurrent.to_f32().unwrap(), r2.to_f32().unwrap());
            }
            _ => panic!("layer 0 kind mismatch after round trip"),
        }
        match (&reimported.layers[1], &cache.layers[1]) {
            (
                HybridLayerCacheData::Attn { k_cache, v_cache },
                HybridLayerCacheData::Attn {
                    k_cache: k2,
                    v_cache: v2,
                },
            ) => {
                assert_eq!(k_cache.to_f32().unwrap(), k2.to_f32().unwrap());
                assert_eq!(v_cache.to_f32().unwrap(), v2.to_f32().unwrap());
            }
            _ => panic!("layer 1 kind mismatch after round trip"),
        }
        match via_import_kv.layers.len() {
            2 => {}
            other => panic!("import_kv returned {other} layers, expected 2"),
        }
    }

    #[test]
    fn export_then_import_mla_is_byte_exact() {
        let cache = MlaKvCache {
            seq_len: 4,
            qk_dim: 6,
            kv_caches: vec![
                (0..24)
                    .map(|i| i as f32 * 0.3 - 1.0)
                    .collect::<Vec<_>>()
                    .into(),
                (0..24)
                    .map(|i| -(i as f32) * 0.7)
                    .collect::<Vec<_>>()
                    .into(),
                (0..24).map(|i| (i as f32).cos()).collect::<Vec<_>>().into(),
            ],
        };

        let path = tmp("mla");
        let path_str = path.to_str().unwrap();

        export_mla_kv(path_str, &cache).expect("export failed");
        let reimported = import_mla_kv(path_str).expect("import failed");
        let via_import_kv = match import_kv(path_str).expect("import_kv failed") {
            ImportedKv::Mla(c) => c,
            _ => panic!("import_kv misdetected an MLA file"),
        };
        std::fs::remove_file(&path).ok();

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.qk_dim, cache.qk_dim);
        assert_eq!(f32s(&reimported.kv_caches), f32s(&cache.kv_caches));
        assert_eq!(f32s(&via_import_kv.kv_caches), f32s(&cache.kv_caches));
    }

    #[test]
    fn import_rejects_bad_magic() {
        let path = tmp("badmagic");
        std::fs::write(&path, b"NOPE\x01\x00\x00\x00").unwrap();
        let err = import_dense_kv(path.to_str().unwrap())
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("bad magic"));
    }

    /// Every way a file can disagree with its header fails at import, before
    /// any payload is read, with `KvCache`.
    #[test]
    fn import_rejects_short_long_and_corrupt_files() {
        let path = tmp("corrupt");
        let p = path.to_str().unwrap();
        export_dense_kv(p, &sample_dense()).unwrap();
        let good = std::fs::read(&path).unwrap();
        let reject = |bytes: &[u8], needle: &str| {
            std::fs::write(&path, bytes).unwrap();
            let err = import_kv(p).expect_err("corrupt file imported");
            assert_eq!(err.code(), crate::error::ReflexErrorCode::KvCache, "{err}");
            assert!(err.to_string().contains(needle), "{needle:?} not in {err}");
        };
        reject(&good[..good.len() - 1], "truncated");
        reject(&good[..30], "truncated");
        reject(&good[..10], "read");
        reject(&good[..2], "read magic");
        let mut long = good.clone();
        long.push(0);
        reject(&long, "1 bytes after the payload");
        let mut flag = good.clone();
        flag[24] = 7;
        reject(&flag, "space_prefix_next");
        let mut version = good.clone();
        version[4] = 9;
        reject(
            &version,
            "unsupported KV cache format version 9 (expected 1 to 7)",
        );
        // A layer count that would describe far more than the file holds.
        let mut layers = good.clone();
        layers[8..12].copy_from_slice(&1000u32.to_le_bytes());
        reject(&layers, "truncated");
        layers[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        reject(&layers, "implausible layer count");
        // Dimensions whose product overflows.
        let mut dims = good.clone();
        dims[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        dims[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        dims[12..16].copy_from_slice(&60_000u32.to_le_bytes());
        reject(&dims, "overflow");
        // A hybrid file with a bad layer kind byte.
        let hybrid = HybridKvCache {
            seq_len: 1,
            attn_num_kv_heads: 1,
            attn_head_dim: 1,
            gdn_conv_state_len: 1,
            gdn_recurrent_len: 1,
            layers: vec![HybridLayerCacheData::Gdn {
                conv_state: vec![1.0].into(),
                recurrent: vec![2.0].into(),
            }],
        };
        export_hybrid_kv(p, &hybrid).unwrap();
        let mut kind = std::fs::read(&path).unwrap();
        kind[32] = 5;
        reject(&kind, "unknown hybrid layer kind byte 5");
        // A file of another layout, through a layout-specific importer.
        export_hybrid_kv(p, &hybrid).unwrap();
        let err = import_mla_kv(p).unwrap_err();
        assert!(err.to_string().contains("expected an MLA"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// f16 bits for `xs`, as the device's round-to-nearest-even cast gives them.
    fn f16_bits(xs: &[f32]) -> KvValues {
        KvValues::HostF16(
            xs.iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect(),
        )
    }

    fn version_of(path: &std::path::Path) -> u32 {
        u32::from_le_bytes(std::fs::read(path).unwrap()[4..8].try_into().unwrap())
    }

    /// Dense version 5: f16 K/V, same header as version 4, half the payload;
    /// imports as f16 file ranges that widen back exactly.
    #[test]
    fn export_then_import_dense_f16_round_trips() {
        let f32_cache = sample_dense();
        let k: Vec<Vec<f32>> = f32s(&f32_cache.k_caches);
        let v: Vec<Vec<f32>> = f32s(&f32_cache.v_caches);
        let cache = DenseKvCache {
            k_caches: k.iter().map(|x| f16_bits(x)).collect(),
            v_caches: v.iter().map(|x| f16_bits(x)).collect(),
            ..f32_cache
        };
        let path = tmp("dense_f16");
        let p = path.to_str().unwrap();
        export_dense_kv(p, &cache).unwrap();
        assert_eq!(version_of(&path), VERSION_DENSE_F16);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 25 + 4 * 24 * 2);
        let back = import_dense_kv(p).unwrap();
        assert_eq!(back.space_prefix_next, Some(false));
        assert!(back.k_caches.iter().all(|x| x.dtype() == KvDtype::F16));
        let rounded = |xs: &[f32]| -> Vec<f32> {
            xs.iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect()
        };
        for (got, want) in f32s(&back.k_caches).iter().zip(&k) {
            assert_eq!(got, &rounded(want));
        }
        for (got, want) in f32s(&back.v_caches).iter().zip(&v) {
            assert_eq!(got, &rounded(want));
        }
        // Re-exporting the imported ranges keeps version 5 and the bytes.
        let copy = tmp("dense_f16_copy");
        export_kv(copy.to_str().unwrap(), &ImportedKv::Dense(back)).unwrap();
        assert_eq!(std::fs::read(&copy).unwrap(), bytes);
        std::fs::remove_file(&copy).ok();
        std::fs::remove_file(&path).ok();
    }

    /// Hybrid version 6: attention K/V f16, GatedDeltaNet state f32.
    #[test]
    fn export_then_import_hybrid_f16_keeps_gdn_state_f32() {
        let conv: Vec<f32> = (0..5).map(|i| i as f32 * 0.1 + 1e-4).collect();
        let rec: Vec<f32> = (0..7).map(|i| i as f32 * -0.2 - 1e-5).collect();
        let k: Vec<f32> = (0..12).map(|i| i as f32 * 0.33).collect();
        let cache = HybridKvCache {
            seq_len: 2,
            attn_num_kv_heads: 2,
            attn_head_dim: 3,
            gdn_conv_state_len: 5,
            gdn_recurrent_len: 7,
            layers: vec![
                HybridLayerCacheData::Gdn {
                    conv_state: conv.clone().into(),
                    recurrent: rec.clone().into(),
                },
                HybridLayerCacheData::Attn {
                    k_cache: f16_bits(&k),
                    v_cache: f16_bits(&k),
                },
            ],
        };
        let path = tmp("hybrid_f16");
        let p = path.to_str().unwrap();
        export_hybrid_kv(p, &cache).unwrap();
        assert_eq!(version_of(&path), VERSION_HYBRID_F16);
        // 34-byte header incl. 2 kind bytes, then f32 GDN (12 values), f16 K/V (24).
        assert_eq!(std::fs::read(&path).unwrap().len(), 34 + 12 * 4 + 24 * 2);
        let back = import_hybrid_kv(p).unwrap();
        let HybridLayerCacheData::Gdn {
            conv_state,
            recurrent,
        } = &back.layers[0]
        else {
            panic!("layer 0 kind")
        };
        assert_eq!(conv_state.dtype(), KvDtype::F32);
        assert_eq!(conv_state.to_f32().unwrap(), conv, "GDN state stays exact");
        assert_eq!(recurrent.to_f32().unwrap(), rec);
        let HybridLayerCacheData::Attn { k_cache, .. } = &back.layers[1] else {
            panic!("layer 1 kind")
        };
        assert_eq!(k_cache.dtype(), KvDtype::F16);
        let want: Vec<f32> = k.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
        assert_eq!(k_cache.to_f32().unwrap(), want);
        std::fs::remove_file(&path).ok();
    }

    /// MLA version 7: the latent in f16.
    #[test]
    fn export_then_import_mla_f16_round_trips() {
        let lat: Vec<Vec<f32>> = (0..3)
            .map(|l| (0..24).map(|i| (i * (l + 1)) as f32 * 0.07 - 0.5).collect())
            .collect();
        let cache = MlaKvCache {
            seq_len: 4,
            qk_dim: 6,
            kv_caches: lat.iter().map(|x| f16_bits(x)).collect(),
        };
        let path = tmp("mla_f16");
        let p = path.to_str().unwrap();
        export_mla_kv(p, &cache).unwrap();
        assert_eq!(version_of(&path), VERSION_MLA_F16);
        assert_eq!(std::fs::read(&path).unwrap().len(), 20 + 3 * 24 * 2);
        let back = import_mla_kv(p).unwrap();
        for (got, want) in f32s(&back.kv_caches).iter().zip(&lat) {
            let want: Vec<f32> = want
                .iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect();
            assert_eq!(got, &want);
        }
        std::fs::remove_file(&path).ok();
    }

    /// The writers refuse what no version describes: mixed f32/f16 K/V, and an
    /// f16 GatedDeltaNet state. A short f16 file is rejected by its f16 size.
    #[test]
    fn f16_files_reject_mixed_types_and_short_payloads() {
        let path = tmp("f16_bad");
        let p = path.to_str().unwrap();
        let mut mixed = sample_dense();
        mixed.v_caches[1] = f16_bits(&[0.0; 24]);
        let err = export_dense_kv(p, &mixed).unwrap_err();
        assert!(err.to_string().contains("mix f32 and f16"), "{err}");
        let hybrid = HybridKvCache {
            seq_len: 1,
            attn_num_kv_heads: 1,
            attn_head_dim: 1,
            gdn_conv_state_len: 1,
            gdn_recurrent_len: 1,
            layers: vec![HybridLayerCacheData::Gdn {
                conv_state: f16_bits(&[1.0]),
                recurrent: f16_bits(&[2.0]),
            }],
        };
        let err = export_hybrid_kv(p, &hybrid).unwrap_err();
        assert!(
            err.to_string().contains("GatedDeltaNet state in f32"),
            "{err}"
        );
        let mut f16 = sample_dense();
        f16.k_caches = f16.k_caches.iter().map(|_| f16_bits(&[0.5; 24])).collect();
        f16.v_caches = f16.v_caches.iter().map(|_| f16_bits(&[0.5; 24])).collect();
        export_dense_kv(p, &f16).unwrap();
        let good = std::fs::read(&path).unwrap();
        std::fs::write(&path, &good[..good.len() - 2]).unwrap();
        let err = import_kv(p).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");
        // An f32-sized payload under the f16 version has trailing bytes.
        let src = tmp_f32_dense();
        let mut v5 = std::fs::read(&src).unwrap();
        std::fs::remove_file(&src).ok();
        v5[4..8].copy_from_slice(&VERSION_DENSE_F16.to_le_bytes());
        std::fs::write(&path, &v5).unwrap();
        let err = import_kv(p).unwrap_err();
        assert!(err.to_string().contains("bytes after the payload"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// Writes `sample_dense()` as an f32 file and returns its path.
    fn tmp_f32_dense() -> std::path::PathBuf {
        let path = tmp("f32_dense_src");
        export_dense_kv(path.to_str().unwrap(), &sample_dense()).unwrap();
        path
    }

    #[test]
    fn kv_dtype_parses_only_f32_and_f16() {
        assert_eq!(KvDtype::parse("f32").unwrap(), KvDtype::F32);
        assert_eq!(KvDtype::parse("f16").unwrap(), KvDtype::F16);
        assert_eq!(KvDtype::default(), KvDtype::F32);
        for bad in ["", "F16", "bf16", "fp16", "f8"] {
            assert_eq!(KvDtype::parse(bad).unwrap_err().kind(), "invalid_input");
        }
        assert_eq!(KvDtype::F16.as_str(), "f16");
    }

    /// A file that shrinks after import fails the read with `KvCache`.
    #[test]
    fn read_after_truncation_fails_cleanly() {
        let path = tmp("shrunk");
        let p = path.to_str().unwrap();
        export_dense_kv(p, &sample_dense()).unwrap();
        let c = import_dense_kv(p).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..40]).unwrap();
        let err = c.v_caches[1].to_f32().unwrap_err();
        assert_eq!(err.code(), crate::error::ReflexErrorCode::KvCache, "{err}");
        drop(c);
        std::fs::remove_file(&path).ok();
    }
}
