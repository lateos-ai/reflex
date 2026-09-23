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
//! tmpfs) -- see CLAUDE.md's Non-goals.
//!
//! File format is a flat, home-grown binary layout, not a stable public
//! spec: magic + version-gated, so the version-1 dense/MoE-only layout
//! round-1 shipped stays readable unchanged (`import_dense_kv`) while
//! version 2 adds the hybrid layout (`import_hybrid_kv`) and version 3 adds
//! the MLA layout (`import_mla_kv`) alongside it -- `import_kv` dispatches
//! on the version field to pick between them.
//!
//! Version 1 (dense/MoE) layout: `b"CSKV"` | version:u32 LE (=1) |
//! num_layers:u32 LE | seq_len:u32 LE | num_kv_heads:u32 LE | head_dim:u32
//! LE | per layer, k_cache then v_cache, each `seq_len * num_kv_heads *
//! head_dim` f32 LE values.
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

use std::io::{Read, Write};

const MAGIC: &[u8; 4] = b"CSKV";
const VERSION_DENSE: u32 = 1;
const VERSION_HYBRID: u32 = 2;
const VERSION_MLA: u32 = 3;

#[derive(Debug)]
pub struct DenseKvCache {
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub k_caches: Vec<Vec<f32>>,
    pub v_caches: Vec<Vec<f32>>,
}

/// One hybrid layer's cached state, matching `model.rs`'s `HybridLayerState`
/// variant for that layer index one-to-one.
#[derive(Debug)]
pub enum HybridLayerCacheData {
    Attn { k_cache: Vec<f32>, v_cache: Vec<f32> },
    Gdn { conv_state: Vec<f32>, recurrent: Vec<f32> },
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
    pub kv_caches: Vec<Vec<f32>>,
}

/// What `import_kv` found in the file -- dispatched on by `Model::generate`
/// to pick the matching resume path (and to reject a format/architecture
/// mismatch with a clear error instead of silently misreading bytes).
#[derive(Debug)]
pub enum ImportedKv {
    Dense(DenseKvCache),
    Hybrid(HybridKvCache),
    Mla(MlaKvCache),
}

pub fn export_dense_kv(path: &str, cache: &DenseKvCache) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(MAGIC).map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&VERSION_DENSE.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;
    f.write_all(&(cache.k_caches.len() as u32).to_le_bytes()).map_err(|e| format!("write num_layers: {e}"))?;
    f.write_all(&(cache.seq_len as u32).to_le_bytes()).map_err(|e| format!("write seq_len: {e}"))?;
    f.write_all(&(cache.num_kv_heads as u32).to_le_bytes()).map_err(|e| format!("write num_kv_heads: {e}"))?;
    f.write_all(&(cache.head_dim as u32).to_le_bytes()).map_err(|e| format!("write head_dim: {e}"))?;

    for (k, v) in cache.k_caches.iter().zip(&cache.v_caches) {
        write_f32_slice(&mut f, k)?;
        write_f32_slice(&mut f, v)?;
    }
    Ok(())
}

pub fn import_dense_kv(path: &str) -> Result<DenseKvCache, String> {
    let mut f = open_and_check_magic(path)?;
    let version = read_u32(&mut f)?;
    if version != VERSION_DENSE {
        return Err(format!("{path}: unsupported dense KV cache format version {version} (expected {VERSION_DENSE})"));
    }
    read_dense_body(&mut f)
}

pub fn export_hybrid_kv(path: &str, cache: &HybridKvCache) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(MAGIC).map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&VERSION_HYBRID.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;
    f.write_all(&(cache.layers.len() as u32).to_le_bytes()).map_err(|e| format!("write num_layers: {e}"))?;
    f.write_all(&(cache.seq_len as u32).to_le_bytes()).map_err(|e| format!("write seq_len: {e}"))?;
    f.write_all(&(cache.attn_num_kv_heads as u32).to_le_bytes()).map_err(|e| format!("write attn_num_kv_heads: {e}"))?;
    f.write_all(&(cache.attn_head_dim as u32).to_le_bytes()).map_err(|e| format!("write attn_head_dim: {e}"))?;
    f.write_all(&(cache.gdn_conv_state_len as u32).to_le_bytes()).map_err(|e| format!("write gdn_conv_state_len: {e}"))?;
    f.write_all(&(cache.gdn_recurrent_len as u32).to_le_bytes()).map_err(|e| format!("write gdn_recurrent_len: {e}"))?;

    let kinds: Vec<u8> = cache
        .layers
        .iter()
        .map(|l| match l {
            HybridLayerCacheData::Attn { .. } => 0u8,
            HybridLayerCacheData::Gdn { .. } => 1u8,
        })
        .collect();
    f.write_all(&kinds).map_err(|e| format!("write layer kinds: {e}"))?;

    for layer in &cache.layers {
        match layer {
            HybridLayerCacheData::Attn { k_cache, v_cache } => {
                write_f32_slice(&mut f, k_cache)?;
                write_f32_slice(&mut f, v_cache)?;
            }
            HybridLayerCacheData::Gdn { conv_state, recurrent } => {
                write_f32_slice(&mut f, conv_state)?;
                write_f32_slice(&mut f, recurrent)?;
            }
        }
    }
    Ok(())
}

pub fn import_hybrid_kv(path: &str) -> Result<HybridKvCache, String> {
    let mut f = open_and_check_magic(path)?;
    let version = read_u32(&mut f)?;
    if version != VERSION_HYBRID {
        return Err(format!("{path}: unsupported hybrid KV cache format version {version} (expected {VERSION_HYBRID})"));
    }
    read_hybrid_body(&mut f)
}

pub fn export_mla_kv(path: &str, cache: &MlaKvCache) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(MAGIC).map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&VERSION_MLA.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;
    f.write_all(&(cache.kv_caches.len() as u32).to_le_bytes()).map_err(|e| format!("write num_layers: {e}"))?;
    f.write_all(&(cache.seq_len as u32).to_le_bytes()).map_err(|e| format!("write seq_len: {e}"))?;
    f.write_all(&(cache.qk_dim as u32).to_le_bytes()).map_err(|e| format!("write qk_dim: {e}"))?;

    for kv in &cache.kv_caches {
        write_f32_slice(&mut f, kv)?;
    }
    Ok(())
}

pub fn import_mla_kv(path: &str) -> Result<MlaKvCache, String> {
    let mut f = open_and_check_magic(path)?;
    let version = read_u32(&mut f)?;
    if version != VERSION_MLA {
        return Err(format!("{path}: unsupported MLA KV cache format version {version} (expected {VERSION_MLA})"));
    }
    read_mla_body(&mut f)
}

/// Reads the magic + version header once and dispatches on the version to
/// the matching body reader, so the CLI doesn't need to know in advance
/// whether a file holds a dense/MoE, hybrid, or MLA cache.
pub fn import_kv(path: &str) -> Result<ImportedKv, String> {
    let mut f = open_and_check_magic(path)?;
    let version = read_u32(&mut f)?;
    match version {
        VERSION_DENSE => Ok(ImportedKv::Dense(read_dense_body(&mut f)?)),
        VERSION_HYBRID => Ok(ImportedKv::Hybrid(read_hybrid_body(&mut f)?)),
        VERSION_MLA => Ok(ImportedKv::Mla(read_mla_body(&mut f)?)),
        other => Err(format!(
            "{path}: unsupported KV cache format version {other} (expected {VERSION_DENSE}, {VERSION_HYBRID}, or {VERSION_MLA})"
        )),
    }
}

fn open_and_check_magic(path: &str) -> Result<std::fs::File, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| format!("read magic: {e}"))?;
    if &magic != MAGIC {
        return Err(format!("{path}: not a reflex-engine KV cache file (bad magic)"));
    }
    Ok(f)
}

fn read_dense_body(f: &mut std::fs::File) -> Result<DenseKvCache, String> {
    let num_layers = read_u32(f)? as usize;
    let seq_len = read_u32(f)? as usize;
    let num_kv_heads = read_u32(f)? as usize;
    let head_dim = read_u32(f)? as usize;
    let per_layer_len = seq_len * num_kv_heads * head_dim;

    let mut k_caches = Vec::with_capacity(num_layers);
    let mut v_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        k_caches.push(read_f32_vec(f, per_layer_len)?);
        v_caches.push(read_f32_vec(f, per_layer_len)?);
    }

    Ok(DenseKvCache { seq_len, num_kv_heads, head_dim, k_caches, v_caches })
}

fn read_hybrid_body(f: &mut std::fs::File) -> Result<HybridKvCache, String> {
    let num_layers = read_u32(f)? as usize;
    let seq_len = read_u32(f)? as usize;
    let attn_num_kv_heads = read_u32(f)? as usize;
    let attn_head_dim = read_u32(f)? as usize;
    let gdn_conv_state_len = read_u32(f)? as usize;
    let gdn_recurrent_len = read_u32(f)? as usize;

    let mut kinds = vec![0u8; num_layers];
    f.read_exact(&mut kinds).map_err(|e| format!("read layer kinds: {e}"))?;

    let attn_len = seq_len * attn_num_kv_heads * attn_head_dim;
    let mut layers = Vec::with_capacity(num_layers);
    for (layer_idx, kind) in kinds.into_iter().enumerate() {
        match kind {
            0 => {
                let k_cache = read_f32_vec(f, attn_len)?;
                let v_cache = read_f32_vec(f, attn_len)?;
                layers.push(HybridLayerCacheData::Attn { k_cache, v_cache });
            }
            1 => {
                let conv_state = read_f32_vec(f, gdn_conv_state_len)?;
                let recurrent = read_f32_vec(f, gdn_recurrent_len)?;
                layers.push(HybridLayerCacheData::Gdn { conv_state, recurrent });
            }
            other => return Err(format!("layer {layer_idx}: unknown hybrid layer kind byte {other} (expected 0 or 1)")),
        }
    }

    Ok(HybridKvCache { seq_len, attn_num_kv_heads, attn_head_dim, gdn_conv_state_len, gdn_recurrent_len, layers })
}

fn read_mla_body(f: &mut std::fs::File) -> Result<MlaKvCache, String> {
    let num_layers = read_u32(f)? as usize;
    let seq_len = read_u32(f)? as usize;
    let qk_dim = read_u32(f)? as usize;
    let per_layer_len = seq_len * qk_dim;

    let mut kv_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        kv_caches.push(read_f32_vec(f, per_layer_len)?);
    }

    Ok(MlaKvCache { seq_len, qk_dim, kv_caches })
}

fn write_f32_slice(f: &mut std::fs::File, data: &[f32]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(data.len() * 4);
    for x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    f.write_all(&buf).map_err(|e| format!("write f32 slice: {e}"))
}

fn read_u32(f: &mut std::fs::File) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_vec(f: &mut std::fs::File, len: usize) -> Result<Vec<f32>, String> {
    let mut buf = vec![0u8; len * 4];
    f.read_exact(&mut buf).map_err(|e| format!("read f32 slice: {e}"))?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_then_import_dense_is_byte_exact() {
        let cache = DenseKvCache {
            seq_len: 3,
            num_kv_heads: 2,
            head_dim: 4,
            k_caches: vec![
                (0..24).map(|i| i as f32 * 0.5 - 3.0).collect(),
                (0..24).map(|i| -(i as f32) * 1.25).collect(),
            ],
            v_caches: vec![
                (0..24).map(|i| (i as f32).sin()).collect(),
                (0..24).map(|i| f32::MIN_POSITIVE * i as f32).collect(),
            ],
        };

        let path = std::env::temp_dir().join(format!("reflex_kv_io_test_dense_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();

        export_dense_kv(path_str, &cache).expect("export failed");
        let reimported = import_dense_kv(path_str).expect("import failed");
        let via_import_kv = match import_kv(path_str).expect("import_kv failed") {
            ImportedKv::Dense(c) => c,
            _ => panic!("import_kv misdetected a dense file"),
        };
        std::fs::remove_file(&path).ok();

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.num_kv_heads, cache.num_kv_heads);
        assert_eq!(reimported.head_dim, cache.head_dim);
        assert_eq!(reimported.k_caches, cache.k_caches);
        assert_eq!(reimported.v_caches, cache.v_caches);
        assert_eq!(via_import_kv.k_caches, cache.k_caches);
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
                    conv_state: (0..5).map(|i| i as f32 * 0.1).collect(),
                    recurrent: (0..7).map(|i| i as f32 * -0.2).collect(),
                },
                HybridLayerCacheData::Attn {
                    k_cache: (0..12).map(|i| i as f32).collect(),
                    v_cache: (0..12).map(|i| -(i as f32)).collect(),
                },
            ],
        };

        let path = std::env::temp_dir().join(format!("reflex_kv_io_test_hybrid_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();

        export_hybrid_kv(path_str, &cache).expect("export failed");
        let reimported = import_hybrid_kv(path_str).expect("import failed");
        let via_import_kv = match import_kv(path_str).expect("import_kv failed") {
            ImportedKv::Hybrid(c) => c,
            _ => panic!("import_kv misdetected a hybrid file"),
        };
        std::fs::remove_file(&path).ok();

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.layers.len(), cache.layers.len());
        match (&reimported.layers[0], &cache.layers[0]) {
            (HybridLayerCacheData::Gdn { conv_state, recurrent }, HybridLayerCacheData::Gdn { conv_state: c2, recurrent: r2 }) => {
                assert_eq!(conv_state, c2);
                assert_eq!(recurrent, r2);
            }
            _ => panic!("layer 0 kind mismatch after round trip"),
        }
        match (&reimported.layers[1], &cache.layers[1]) {
            (HybridLayerCacheData::Attn { k_cache, v_cache }, HybridLayerCacheData::Attn { k_cache: k2, v_cache: v2 }) => {
                assert_eq!(k_cache, k2);
                assert_eq!(v_cache, v2);
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
                (0..24).map(|i| i as f32 * 0.3 - 1.0).collect(),
                (0..24).map(|i| -(i as f32) * 0.7).collect(),
                (0..24).map(|i| (i as f32).cos()).collect(),
            ],
        };

        let path = std::env::temp_dir().join(format!("reflex_kv_io_test_mla_{}.bin", std::process::id()));
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
        assert_eq!(reimported.kv_caches, cache.kv_caches);
        assert_eq!(via_import_kv.kv_caches, cache.kv_caches);
    }

    #[test]
    fn import_rejects_bad_magic() {
        let path = std::env::temp_dir().join(format!("reflex_kv_io_badmagic_{}.bin", std::process::id()));
        std::fs::write(&path, b"NOPE\x01\x00\x00\x00").unwrap();
        let err = import_dense_kv(path.to_str().unwrap()).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("bad magic"));
    }
}
