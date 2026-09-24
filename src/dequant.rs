//! Dequantization math for GGUF/GGML block-quantized tensor formats
//! (Phase 21.1, second half).
//!
//! `gguf.rs` parses the container and hands back raw block bytes
//! (`GgufFile::tensor_bytes`); this module turns those bytes into `f32`.
//!
//! GGUF's own spec documents the *container* format, not the bit-packing
//! inside a `Q4_K`/`Q5_K`/etc. block — that packing is defined by whatever
//! `ggml-quants.c` happens to do. So instead of re-deriving the bit layout
//! from prose, every function here is a line-for-line port of the matching
//! `dequantize_row_*` / `get_scale_min_k4` function in the canonical
//! reference implementation (`ggml/src/ggml-quants.c` and
//! `ggml/src/ggml-common.h`, upstream `ggml-org/llama.cpp`, fetched
//! 2026-09-11) — variable names (`d`, `dmin`, `sc`, `m`, `ql`, `qh`, `is`,
//! `shift`, ...) are kept identical to the C so the port can be diffed
//! against the source by eye. Block byte-sizes here also match
//! `gguf.rs::ggml_type_size_bytes`, computed independently from the same
//! reference struct layouts.
//!
//! **Still not verified against a real on-disk GGUF file** — none is
//! available in this environment (see `gguf.rs`'s module doc for the same
//! caveat on structural parsing). What *is* verified here: the port matches
//! the reference algorithm, and every format's tests check hand-computed
//! expected values against known input bytes (not just round-tripped
//! through this module's own code).

use crate::gguf::GgmlType;
use half::{bf16, f16};

const QK_K: usize = 256;

fn le_f16(b: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

/// Port of `get_scale_min_k4` (ggml-quants.c): unpacks one of 8 (scale, min)
/// pairs from a Q4_K/Q5_K block's 12-byte `scales` field, each 6 bits wide.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one tensor's worth of raw block bytes to `f32`, truncated to
/// exactly `element_count` values (a quantized tensor's element count is
/// always a multiple of its block size by ggml's own invariant, but this
/// stays defensive rather than asserting it).
pub fn dequantize(
    ggml_type: GgmlType,
    bytes: &[u8],
    element_count: u64,
) -> Result<Vec<f32>, String> {
    let n = element_count as usize;
    let mut out = match ggml_type {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => bytes.as_chunks::<2>().0.iter().map(|c| le_f16(c)).collect(),
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
            .collect(),
        GgmlType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        GgmlType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
        GgmlType::I16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c) as f32)
            .collect(),
        GgmlType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        GgmlType::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        GgmlType::Q4_0 => dequantize_blocks(bytes, 18, 32, dequantize_block_q4_0),
        GgmlType::Q4_1 => dequantize_blocks(bytes, 20, 32, dequantize_block_q4_1),
        GgmlType::Q5_0 => dequantize_blocks(bytes, 22, 32, dequantize_block_q5_0),
        GgmlType::Q5_1 => dequantize_blocks(bytes, 24, 32, dequantize_block_q5_1),
        GgmlType::Q8_0 => dequantize_blocks(bytes, 34, 32, dequantize_block_q8_0),
        GgmlType::Q8_1 => dequantize_blocks(bytes, 36, 32, dequantize_block_q8_1),
        GgmlType::Q2K => dequantize_blocks(bytes, 84, QK_K, dequantize_block_q2_k),
        GgmlType::Q3K => dequantize_blocks(bytes, 110, QK_K, dequantize_block_q3_k),
        GgmlType::Q4K => dequantize_blocks(bytes, 144, QK_K, dequantize_block_q4_k),
        GgmlType::Q5K => dequantize_blocks(bytes, 176, QK_K, dequantize_block_q5_k),
        GgmlType::Q6K => dequantize_blocks(bytes, 210, QK_K, dequantize_block_q6_k),
        GgmlType::Q8K => dequantize_blocks(bytes, 292, QK_K, dequantize_block_q8_k),
        // Phase 21.13: IQ-family formats (ported in `dequant_iq.rs`).
        GgmlType::IQ2XXS => {
            dequantize_blocks(bytes, 66, QK_K, crate::dequant_iq::dequantize_block_iq2_xxs)
        }
        GgmlType::IQ2XS => {
            dequantize_blocks(bytes, 74, QK_K, crate::dequant_iq::dequantize_block_iq2_xs)
        }
        GgmlType::IQ2S => {
            dequantize_blocks(bytes, 82, QK_K, crate::dequant_iq::dequantize_block_iq2_s)
        }
        GgmlType::IQ3XXS => {
            dequantize_blocks(bytes, 98, QK_K, crate::dequant_iq::dequantize_block_iq3_xxs)
        }
        GgmlType::IQ3S => {
            dequantize_blocks(bytes, 110, QK_K, crate::dequant_iq::dequantize_block_iq3_s)
        }
        GgmlType::IQ1S => {
            dequantize_blocks(bytes, 50, QK_K, crate::dequant_iq::dequantize_block_iq1_s)
        }
        GgmlType::IQ1M => {
            dequantize_blocks(bytes, 56, QK_K, crate::dequant_iq::dequantize_block_iq1_m)
        }
        GgmlType::IQ4XS => {
            dequantize_blocks(bytes, 136, QK_K, crate::dequant_iq::dequantize_block_iq4_xs)
        }
        GgmlType::Unknown(t) => {
            return Err(format!(
                "unknown ggml_type={t}: no dequantizer known for it"
            ))
        }
    };
    out.truncate(n);
    Ok(out)
}

/// Chunk `bytes` into `block_bytes`-sized blocks, dequantize each into
/// `block_elems` output floats via `f`, and concatenate.
fn dequantize_blocks(
    bytes: &[u8],
    block_bytes: usize,
    block_elems: usize,
    f: fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    let nb = bytes.len() / block_bytes;
    let mut out = vec![0f32; nb * block_elems];
    for i in 0..nb {
        let block = &bytes[i * block_bytes..(i + 1) * block_bytes];
        f(block, &mut out[i * block_elems..(i + 1) * block_elems]);
    }
    out
}

/// Port of `dequantize_row_q4_0`.
fn dequantize_block_q4_0(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..18];
    for j in 0..16 {
        let x0 = (qs[j] & 0x0F) as i32 - 8;
        let x1 = (qs[j] >> 4) as i32 - 8;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

/// Port of `dequantize_row_q4_1`.
fn dequantize_block_q4_1(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let m = le_f16(&block[2..4]);
    let qs = &block[4..20];
    for j in 0..16 {
        let x0 = (qs[j] & 0x0F) as f32;
        let x1 = (qs[j] >> 4) as f32;
        y[j] = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

/// Port of `dequantize_row_q5_0`.
fn dequantize_block_q5_0(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
    let qs = &block[6..22];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = (((qs[j] & 0x0F) as u32 | xh_0) as i32) - 16;
        let x1 = (((qs[j] >> 4) as u32 | xh_1) as i32) - 16;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

/// Port of `dequantize_row_q5_1`.
fn dequantize_block_q5_1(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let m = le_f16(&block[2..4]);
    let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
    let qs = &block[8..24];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = (qs[j] & 0x0F) as u32 | xh_0;
        let x1 = (qs[j] >> 4) as u32 | xh_1;
        y[j] = x0 as f32 * d + m;
        y[j + 16] = x1 as f32 * d + m;
    }
}

/// Port of `dequantize_row_q8_0`.
fn dequantize_block_q8_0(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..34];
    for j in 0..32 {
        y[j] = qs[j] as i8 as f32 * d;
    }
}

/// Q8_1 has no `dequantize_row_q8_1` upstream (it's only ever a runtime
/// intermediate for quantized dot products, never a stored GGUF weight),
/// but its value semantics are the same `x = d*q` as Q8_0 — the extra `s`
/// field (`d * sum(qs)`) is a precomputed dot-product helper, not part of
/// the value reconstruction. Included so the dispatcher is total over every
/// named `GgmlType`.
fn dequantize_block_q8_1(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[4..36];
    for j in 0..32 {
        y[j] = qs[j] as i8 as f32 * d;
    }
}

/// Port of `dequantize_row_q2_K`. Block layout: `scales[16]`, `qs[64]`,
/// `d: f16`, `dmin: f16`.
fn dequantize_block_q2_k(block: &[u8], y: &mut [f32]) {
    let scales = &block[0..16];
    let q = &block[16..80];
    let d = le_f16(&block[80..82]);
    let min = le_f16(&block[82..84]);

    let mut is = 0usize;
    let mut y_off = 0usize;
    let mut q_off = 0usize;
    let mut n = 0;
    while n < QK_K {
        for shift in [0u8, 2, 4, 6] {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[y_off + l] = dl * (((q[q_off + l] >> shift) & 3) as i8) as f32 - ml;
            }

            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[y_off + 16 + l] = dl * (((q[q_off + l + 16] >> shift) & 3) as i8) as f32 - ml;
            }
            y_off += 32;
        }
        q_off += 32;
        n += 128;
    }
}

/// Port of `dequantize_row_q3_K`. Block layout: `hmask[32]`, `qs[64]`,
/// `scales[12]`, `d: f16`.
fn dequantize_block_q3_k(block: &[u8], y: &mut [f32]) {
    let hmask = &block[0..32];
    let q = &block[32..96];
    let raw_scales = &block[96..108];
    let d_all = le_f16(&block[108..110]);

    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes(raw_scales[4 * i..4 * i + 4].try_into().unwrap());
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for i in 0..4 {
        let b = aux[i].to_le_bytes();
        for k in 0..4 {
            scales[4 * i + k] = b[k] as i8;
        }
    }

    let mut m: u8 = 1;
    let mut is = 0usize;
    let mut y_off = 0usize;
    let mut q_off = 0usize;
    let mut n = 0;
    while n < QK_K {
        let mut shift = 0u8;
        for _ in 0..4 {
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let bit = if hmask[l] & m != 0 { 0 } else { 4 };
                y[y_off + l] = dl * (((q[q_off + l] >> shift) & 3) as i32 - bit) as f32;
            }
            y_off += 16;

            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let bit = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[y_off + l] = dl * (((q[q_off + l + 16] >> shift) & 3) as i32 - bit) as f32;
            }
            y_off += 16;

            shift += 2;
            m <<= 1;
        }
        q_off += 32;
        n += 128;
    }
}

/// Port of `dequantize_row_q4_K`. Block layout: `d: f16`, `dmin: f16`,
/// `scales[12]`, `qs[128]`.
fn dequantize_block_q4_k(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let min = le_f16(&block[2..4]);
    let scales = &block[4..16];
    let q = &block[16..144];

    let mut is = 0usize;
    let mut y_off = 0usize;
    let mut q_off = 0usize;
    let mut j = 0;
    while j < QK_K {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            y[y_off + l] = d1 * (q[q_off + l] & 0xF) as f32 - m1;
        }
        for l in 0..32 {
            y[y_off + 32 + l] = d2 * (q[q_off + l] >> 4) as f32 - m2;
        }
        q_off += 32;
        is += 2;
        y_off += 64;
        j += 64;
    }
}

/// Port of `dequantize_row_q5_K`. Block layout: `d: f16`, `dmin: f16`,
/// `scales[12]`, `qh[32]`, `qs[128]`.
fn dequantize_block_q5_k(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let min = le_f16(&block[2..4]);
    let scales = &block[4..16];
    let qh = &block[16..48];
    let ql = &block[48..176];

    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut j = 0;
    while j < QK_K {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            y[y_off + l] = d1 * ((ql[ql_off + l] & 0xF) + hi) as f32 - m1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[y_off + 32 + l] = d2 * ((ql[ql_off + l] >> 4) + hi) as f32 - m2;
        }
        ql_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        y_off += 64;
        j += 64;
    }
}

/// Port of `dequantize_row_q6_K`. Block layout: `ql[128]`, `qh[64]`,
/// `scales[16]` (i8), `d: f16`.
// `qh >> 0` below is a no-op shift, but it's kept for symmetry with the
// `>> 2`/`>> 4`/`>> 6` siblings it's ported alongside (matches the reference
// `ggml-quants.c` line-for-line, per this module's doc comment).
#[allow(clippy::identity_op)]
fn dequantize_block_q6_k(block: &[u8], y: &mut [f32]) {
    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let sc_all = &block[192..208];
    let d = le_f16(&block[208..210]);

    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut n = 0;
    while n < QK_K {
        for l in 0..32 {
            let is = l / 16;
            let qh = qh_all[qh_off + l];
            let q1 = (((ql_all[ql_off + l] & 0xF) | (((qh >> 0) & 3) << 4)) as i32) - 32;
            let q2 = (((ql_all[ql_off + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) as i32) - 32;
            let q3 = (((ql_all[ql_off + l] >> 4) | (((qh >> 4) & 3) << 4)) as i32) - 32;
            let q4 = (((ql_all[ql_off + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) as i32) - 32;
            y[y_off + l] = d * sc_all[sc_off + is] as i8 as f32 * q1 as f32;
            y[y_off + l + 32] = d * sc_all[sc_off + is + 2] as i8 as f32 * q2 as f32;
            y[y_off + l + 64] = d * sc_all[sc_off + is + 4] as i8 as f32 * q3 as f32;
            y[y_off + l + 96] = d * sc_all[sc_off + is + 6] as i8 as f32 * q4 as f32;
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
        n += 128;
    }
}

/// Port of `dequantize_row_q8_K`. Block layout: `d: f32`, `qs[256]` (i8),
/// `bsums[16]` (i16, unused for dequant — precomputed group sums for fast
/// dot products, same role as Q8_1's `s`).
fn dequantize_block_q8_k(block: &[u8], y: &mut [f32]) {
    let d = f32::from_le_bytes(block[0..4].try_into().unwrap());
    let qs = &block[4..260];
    for j in 0..QK_K {
        y[j] = qs[j] as i8 as f32 * d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    #[test]
    fn test_q4_0_matches_hand_computed_values() {
        // d = 2.0; nibbles chosen as 0, 8, 15 (plus padding zero nibbles)
        // so x = (nibble - 8) * d gives easily checked results:
        // nibble 0  -> (0-8)*2  = -16
        // nibble 8  -> (8-8)*2  = 0
        // nibble 15 -> (15-8)*2 = 14
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(2.0));
        // qs[0] holds elements 0 (low nibble) and 16 (high nibble).
        block[2] = 0x00; // low=0 -> elem0, high=0 -> elem16
        block[3] = 0x8F; // low=15 -> elem1, high=8 -> elem17
        let out = dequantize(GgmlType::Q4_0, &block, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], -16.0);
        assert_eq!(out[1], 14.0);
        assert_eq!(out[16], -16.0);
        assert_eq!(out[17], 0.0);
    }

    #[test]
    fn test_q4_1_matches_hand_computed_values() {
        // d=2.0, m=1.0; x = nibble*d + m
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&f16_bytes(2.0));
        block[2..4].copy_from_slice(&f16_bytes(1.0));
        block[4] = 0xF0; // low=0 -> elem0 = 0*2+1=1 ; high=15 -> elem16 = 15*2+1=31
        let out = dequantize(GgmlType::Q4_1, &block, 32).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[16], 31.0);
    }

    #[test]
    fn test_q5_0_high_bit_extends_range_past_nibble() {
        // d=1.0. qs[0] low nibble = 0xF (15), no high bit set for element 0
        // -> x0 = (15 | 0) - 16 = -1.
        // Set qh bit 0 (extends element 0's value to 16 | 15 = 31 -> 31-16=15).
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[6] = 0x0F; // qs[0] low nibble = 15, high nibble = 0
                         // qh (4 bytes at offset 2..6): bit 0 set -> affects element 0 (j=0).
        block[2] = 0x01;
        let out = dequantize(GgmlType::Q5_0, &block, 32).unwrap();
        assert_eq!(out[0], 15.0); // (0xF | 0x10) - 16 = 31-16 = 15
        assert_eq!(out[16], -16.0); // high nibble 0, no high bit -> (0|0)-16
    }

    #[test]
    fn test_q5_1_matches_hand_computed_values() {
        // d=1.0, m=0.5, qh=0 (no high-bit extension). qs[0] low=10,high=0.
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(0.5));
        block[8] = 0x0A;
        let out = dequantize(GgmlType::Q5_1, &block, 32).unwrap();
        assert_eq!(out[0], 10.5); // 10*1.0 + 0.5
        assert_eq!(out[16], 0.5); // 0*1.0 + 0.5
    }

    #[test]
    fn test_q8_1_ignores_sum_field_same_math_as_q8_0() {
        let mut block = vec![0u8; 36];
        block[0..2].copy_from_slice(&f16_bytes(0.25));
        block[4] = (-4i8) as u8; // qs[0] = -4
        let out = dequantize(GgmlType::Q8_1, &block, 32).unwrap();
        assert_eq!(out[0], -1.0); // -4 * 0.25
    }

    #[test]
    fn test_q3_k_all_zero_block_gives_hand_computed_value() {
        // An all-zero block (except d_all) drives scales[0] to 0 (both
        // packed-nibble sources are 0), so dl = d_all*(0-32) = -32; q bits
        // are 0 and hmask bit is unset (contributes -4), giving
        // y[0] = -32 * (0 - 4) = 128.
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&f16_bytes(1.0)); // d_all
        let out = dequantize(GgmlType::Q3K, &block, QK_K as u64).unwrap();
        assert_eq!(out[0], 128.0);
    }

    #[test]
    fn test_q5_k_high_bit_and_scale_min_hand_computed() {
        // scales[0]=5 (sc, j=0<4 branch), scales[4]=3 (m). d=1, dmin=1.
        // qh bit u1 set for l=0 -> +16; ql[0] low nibble = 2.
        // y[0] = d1*(2+16) - m1 = (1*5)*18 - (1*3) = 90-3 = 87.
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16_bytes(1.0)); // d
        block[2..4].copy_from_slice(&f16_bytes(1.0)); // dmin
        block[4] = 5; // scales[0] -> sc
        block[8] = 3; // scales[4] -> m
        block[16] = 0x01; // qh[0] bit 0 set
        block[48] = 0x02; // ql[0] low nibble = 2
        let out = dequantize(GgmlType::Q5K, &block, QK_K as u64).unwrap();
        assert_eq!(out[0], 87.0);
    }

    #[test]
    fn test_q8_0_signed_values_scaled_by_delta() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_bytes(0.5));
        block[2] = (-1i8) as u8; // qs[0] = -1
        block[3] = 100u8; // qs[1] = 100
        let out = dequantize(GgmlType::Q8_0, &block, 32).unwrap();
        assert_eq!(out[0], -0.5);
        assert_eq!(out[1], 50.0);
    }

    #[test]
    fn test_q2_k_all_zero_quants_gives_negative_min() {
        // scales[0] = sc with low nibble (scale factor for d) = 1, high
        // nibble (min factor) = 1; all qs bits zero -> for every element,
        // q=0 so y = dl*0 - ml = -ml = -(min*1).
        let mut block = vec![0u8; 84];
        block[0] = 0x11; // scales[0]: low nibble 1, high nibble 1
                         // d (offset 80) = 1.0, dmin (offset 82) = 3.0
        block[80..82].copy_from_slice(&f16_bytes(1.0));
        block[82..84].copy_from_slice(&f16_bytes(3.0));
        let out = dequantize(GgmlType::Q2K, &block, QK_K as u64).unwrap();
        // First 16 elements (l=0..16, shift=0, is=0 -> scales[0]) should
        // all be -3.0 since q bits are 0 and sc=0x11 -> dl=1*1=1, ml=3*1=3.
        for v in &out[0..16] {
            assert_eq!(*v, -3.0);
        }
    }

    #[test]
    fn test_q4_k_scale_zero_index_matches_get_scale_min_k4() {
        // For j=0 (<4): sc = scales[0] & 63, m = scales[4] & 63.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_bytes(2.0)); // d
        block[2..4].copy_from_slice(&f16_bytes(5.0)); // dmin
        block[4] = 10; // scales[0] -> sc=10
        block[8] = 4; // scales[4] -> m=4
                      // qs[0] low nibble = 3 -> element0 = d*sc*3 - dmin*m = 2*10*3 - 5*4 = 60-20=40
        block[16] = 0x03;
        let out = dequantize(GgmlType::Q4K, &block, QK_K as u64).unwrap();
        assert_eq!(out[0], 40.0);
    }

    #[test]
    fn test_q6_k_dequant_matches_hand_computed_value() {
        // scales[0] = 2 (i8), d = 1.0. For l=0, is=0: q1 uses
        // ql[0]&0xF combined with (qh[0]>>0)&3 shifted to bit4, minus 32.
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16_bytes(1.0)); // d
        block[192] = 2; // scales[0] (i8) = 2
                        // ql[0] low nibble = 0, qh[0] low 2 bits = 0 -> q1 = (0|0)-32 = -32
                        // y[0] = d * sc[0] * q1 = 1*2*(-32) = -64
        let out = dequantize(GgmlType::Q6K, &block, QK_K as u64).unwrap();
        assert_eq!(out[0], -64.0);
    }

    #[test]
    fn test_q8_k_uses_f32_delta_not_f16() {
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&2.5f32.to_le_bytes());
        block[4] = 4u8; // qs[0] = 4
        let out = dequantize(GgmlType::Q8K, &block, QK_K as u64).unwrap();
        assert_eq!(out[0], 10.0);
    }

    #[test]
    fn test_passthrough_f32_and_f16_and_bf16() {
        let f32_bytes = 3.5f32.to_le_bytes().to_vec();
        assert_eq!(dequantize(GgmlType::F32, &f32_bytes, 1).unwrap(), vec![3.5]);

        let f16_b = f16_bytes(3.5).to_vec();
        assert_eq!(dequantize(GgmlType::F16, &f16_b, 1).unwrap(), vec![3.5]);

        let bf16_b = bf16::from_f32(3.5).to_le_bytes().to_vec();
        assert_eq!(dequantize(GgmlType::Bf16, &bf16_b, 1).unwrap(), vec![3.5]);
    }

    #[test]
    fn test_unknown_type_errors_instead_of_panicking() {
        let result = dequantize(GgmlType::Unknown(999), &[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncates_to_requested_element_count() {
        // A single Q4_0 block always yields 32 elements; requesting fewer
        // (e.g. because the tensor's true length isn't block-aligned in a
        // pathological input) should truncate rather than over-return.
        let block = vec![0u8; 18];
        let out = dequantize(GgmlType::Q4_0, &block, 5).unwrap();
        assert_eq!(out.len(), 5);
    }
}
