//! Phase 3 (State I/O), round 1: raw KV-cache export/import for the dense/
//! MoE Qwen3 forward path only (see README.md's roadmap and STATUS.md for
//! why this round is scoped to dense and doesn't yet wire an imported cache
//! back into a forward pass -- there's no per-token generation loop or
//! `start_pos` anywhere in `model.rs` yet for it to resume into). The engine
//! stays ignorant of where the file lives (NVMe, S3-backed FUSE, tmpfs) --
//! see CLAUDE.md's Non-goals.
//!
//! File format is a flat, home-grown binary layout, not a stable public
//! spec: magic + version-gated so a later round can add per-architecture
//! variants (hybrid Gdn conv/recurrent state, MLA's single compressed
//! cache) without breaking round-1 files.
//!
//! Layout: `b"CSKV"` | version:u32 LE | num_layers:u32 LE | seq_len:u32 LE |
//! num_kv_heads:u32 LE | head_dim:u32 LE | per layer, k_cache then v_cache,
//! each `seq_len * num_kv_heads * head_dim` f32 LE values.

use cudarc::driver::CudaDevice;
use std::io::{Read, Write};
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"CSKV";
const VERSION: u32 = 1;

#[derive(Debug)]
pub struct DenseKvCache {
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub k_caches: Vec<Vec<f32>>,
    pub v_caches: Vec<Vec<f32>>,
}

pub fn export_dense_kv(path: &str, cache: &DenseKvCache) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(MAGIC).map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&VERSION.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;
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
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| format!("read magic: {e}"))?;
    if &magic != MAGIC {
        return Err(format!("{path}: not a coldstart-infer KV cache file (bad magic)"));
    }
    let version = read_u32(&mut f)?;
    if version != VERSION {
        return Err(format!("{path}: unsupported KV cache format version {version} (expected {VERSION})"));
    }
    let num_layers = read_u32(&mut f)? as usize;
    let seq_len = read_u32(&mut f)? as usize;
    let num_kv_heads = read_u32(&mut f)? as usize;
    let head_dim = read_u32(&mut f)? as usize;
    let per_layer_len = seq_len * num_kv_heads * head_dim;

    let mut k_caches = Vec::with_capacity(num_layers);
    let mut v_caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        k_caches.push(read_f32_vec(&mut f, per_layer_len)?);
        v_caches.push(read_f32_vec(&mut f, per_layer_len)?);
    }

    Ok(DenseKvCache { seq_len, num_kv_heads, head_dim, k_caches, v_caches })
}

/// Round-1 substitute for real resume: proves the exported/imported host
/// buffers survive an upload-to-device-and-back trip unchanged, i.e. the
/// bytes an orchestrator hands back to this engine later are exactly usable
/// as device-resident KV state once a generation loop exists to consume
/// them. Does not feed the cache into any forward pass.
pub fn verify_roundtrip_on_device(device: &Arc<CudaDevice>, cache: &DenseKvCache) -> Result<(), String> {
    for (i, (k, v)) in cache.k_caches.iter().zip(&cache.v_caches).enumerate() {
        let k_dev = device.htod_sync_copy(k).map_err(|e| format!("layer {i} k_cache htod: {e}"))?;
        let k_back = device.dtoh_sync_copy(&k_dev).map_err(|e| format!("layer {i} k_cache dtoh: {e}"))?;
        if k_back != *k {
            return Err(format!("layer {i} k_cache device round-trip mismatch"));
        }

        let v_dev = device.htod_sync_copy(v).map_err(|e| format!("layer {i} v_cache htod: {e}"))?;
        let v_back = device.dtoh_sync_copy(&v_dev).map_err(|e| format!("layer {i} v_cache dtoh: {e}"))?;
        if v_back != *v {
            return Err(format!("layer {i} v_cache device round-trip mismatch"));
        }
    }
    Ok(())
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
    fn export_then_import_is_byte_exact() {
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

        let path = std::env::temp_dir().join(format!("coldstart_kv_io_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();

        export_dense_kv(path_str, &cache).expect("export failed");
        let reimported = import_dense_kv(path_str).expect("import failed");
        std::fs::remove_file(&path).ok();

        assert_eq!(reimported.seq_len, cache.seq_len);
        assert_eq!(reimported.num_kv_heads, cache.num_kv_heads);
        assert_eq!(reimported.head_dim, cache.head_dim);
        assert_eq!(reimported.k_caches, cache.k_caches);
        assert_eq!(reimported.v_caches, cache.v_caches);
    }

    #[test]
    fn import_rejects_bad_magic() {
        let path = std::env::temp_dir().join(format!("coldstart_kv_io_badmagic_{}.bin", std::process::id()));
        std::fs::write(&path, b"NOPE\x01\x00\x00\x00").unwrap();
        let err = import_dense_kv(path.to_str().unwrap()).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("bad magic"));
    }
}
