//! Host-side dequantization math for the IQ-family GGML block formats
//! (Phase 21.13).
//!
//! `dequant.rs` covers the legacy and K-quant types; this module adds the
//! non-uniform / codebook ("i-quant") formats that IST-DASLab GSQ-RCO output
//! actually uses. Same porting standard as `dequant.rs`: every function is a
//! line-for-line port of the matching `dequantize_row_iq*` in the canonical
//! reference (`ggml/src/ggml-quants.c`, upstream `ggml-org/llama.cpp`,
//! fetched 2026-09-15) — variable names (`d`, `db`, `qs`, `qh`, `signs`,
//! `scales`, `aux32`, `aux8`, ...) kept identical so the port can be diffed
//! against the C by eye.
//!
//! The codebook/lattice lookup arrays these functions index are **not**
//! re-derived from prose; they are copied byte-for-byte from
//! `ggml/src/ggml-common.h` into [`crate::dequant_iq_tables`] by
//! `scripts/gen_iq_tables.py`, so a transcription typo can't silently change
//! a codebook entry.
//!
//! The exact type set implemented here is not the full speculative `IQ*`
//! family: it is exactly what a real published GSQ-RCO GGUF contains, from a
//! direct inspection of `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`'s own tensor
//! table (Phase 21.13.1): `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
//! `IQ3_XXS`, `IQ3_S`, `IQ4_XS`. See `PHASE21_13_PLAN.md`.

use crate::dequant_iq_tables as t;
use half::f16;

const QK_K: usize = 256;

/// `IQ1S_DELTA` / `IQ1M_DELTA` in the reference (`ggml-common.h`).
const IQ1S_DELTA: f32 = 0.125;

fn le_f16(b: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

fn le_u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn le_u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Byte `j` of a packed `uint64_t` codebook entry (`(const uint8_t *)&grid[j]`).
fn grid_u64_byte(entry: u64, j: usize) -> u8 {
    (entry >> (8 * j)) as u8
}

/// Byte `j` of a packed `uint64_t` entry interpreted as `int8_t` (iq1s grid).
fn grid_u64_i8(entry: u64, j: usize) -> i8 {
    (entry >> (8 * j)) as u8 as i8
}

/// Byte `j` of a packed `uint32_t` codebook entry (`(const uint8_t *)&grid[j]`).
fn grid_u32_byte(entry: u32, j: usize) -> u8 {
    (entry >> (8 * j)) as u8
}

fn sign_of(cond: bool) -> f32 {
    if cond {
        -1.0
    } else {
        1.0
    }
}

/// Port of `dequantize_row_iq2_xxs`. Block: `d: f16`, `qs: u16[32]`.
pub(crate) fn dequantize_block_iq2_xxs(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..66];

    let mut y_off = 0usize;
    for ib32 in 0..QK_K / 32 {
        // `x[i].qs` is `uint16_t[32]`, so the reference's `qs + 4*ib32`
        // pointer arithmetic advances 8 bytes per group, not 4.
        let aux32_0 = le_u32_at(qs, 8 * ib32);
        let aux32_1 = le_u32_at(qs, 8 * ib32 + 4);
        let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
        let aux8 = aux32_0.to_le_bytes();
        for l in 0..4 {
            let grid = t::IQ2XXS_GRID[aux8[l] as usize];
            let signs = t::KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[y_off + j] =
                    db * grid_u64_byte(grid, j) as f32 * sign_of(signs & t::KMASK_IQ2XS[j] != 0);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq2_xs`. Block: `d: f16`, `qs: u16[32]`, `scales: u8[8]`.
pub(crate) fn dequantize_block_iq2_xs(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..66];
    let scales = &block[66..74];

    let mut y_off = 0usize;
    for ib32 in 0..QK_K / 32 {
        let sc = scales[ib32];
        let db = [
            d * (0.5 + (sc & 0xf) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let qv = le_u16_at(qs, 2 * (4 * ib32 + l));
            let grid = t::IQ2XS_GRID[(qv & 511) as usize];
            let signs = t::KSIGNS_IQ2XS[(qv >> 9) as usize];
            for j in 0..8 {
                y[y_off + j] = db[l / 2]
                    * grid_u64_byte(grid, j) as f32
                    * sign_of(signs & t::KMASK_IQ2XS[j] != 0);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq2_s`. Block: `d: f16`, `qs: u8[64]`, `qh: u8[8]`, `scales: u8[8]`.
///
/// `signs` in the reference aliases `qs + QK_K/8` (the second 32 bytes of the
/// `qs` field), not a separate struct member.
pub(crate) fn dequantize_block_iq2_s(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    let signs = &qs[QK_K / 8..];

    let mut y_off = 0usize;
    for ib32 in 0..QK_K / 32 {
        let sc = scales[ib32];
        let db = [
            d * (0.5 + (sc & 0xf) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx = qs[4 * ib32 + l] as u32 | ((qh[ib32] as u32) << (8 - 2 * l)) & 0x300;
            let grid = t::IQ2S_GRID[idx as usize];
            let sg = signs[4 * ib32 + l];
            for j in 0..8 {
                y[y_off + j] =
                    dl * grid_u64_byte(grid, j) as f32 * sign_of(sg & t::KMASK_IQ2XS[j] != 0);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq3_xxs`. Block: `d: f16`, `qs: u8[96]`, where
/// `scales_and_signs` aliases `qs + QK_K/4`.
pub(crate) fn dequantize_block_iq3_xxs(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..98];
    let scales_and_signs = &qs[QK_K / 4..];

    let mut y_off = 0usize;
    for ib32 in 0..QK_K / 32 {
        let aux32 = le_u32_at(scales_and_signs, 4 * ib32);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = t::KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let grid1 = t::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
            let grid2 = t::IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
            for j in 0..4 {
                y[y_off + j] =
                    db * grid_u32_byte(grid1, j) as f32 * sign_of(signs & t::KMASK_IQ2XS[j] != 0);
                y[y_off + j + 4] = db
                    * grid_u32_byte(grid2, j) as f32
                    * sign_of(signs & t::KMASK_IQ2XS[j + 4] != 0);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq3_s`. Block: `d: f16`, `qs: u8[64]`, `qh: u8[8]`,
/// `signs: u8[32]`, `scales: u8[4]`.
pub(crate) fn dequantize_block_iq3_s(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];

    let mut y_off = 0usize;
    let mut ib32 = 0usize;
    while ib32 < QK_K / 32 {
        let k = ib32 / 2;
        let sc = scales[k];
        let db1 = d * (1.0 + 2.0 * (sc & 0xf) as f32);
        let db2 = d * (1.0 + 2.0 * (sc >> 4) as f32);

        // First half of the pair: `qh[0]`, `qs` window 16*k, `signs` window 8*k, db1.
        let qh0 = qh[2 * k] as u32;
        for l in 0..4 {
            let idx1 = qs[16 * k + 2 * l] as u32 | ((qh0 << (8 - 2 * l)) & 256);
            let idx2 = qs[16 * k + 2 * l + 1] as u32 | ((qh0 << (7 - 2 * l)) & 256);
            let grid1 = t::IQ3S_GRID[idx1 as usize];
            let grid2 = t::IQ3S_GRID[idx2 as usize];
            let sg = signs[8 * k + l];
            for j in 0..4 {
                y[y_off + j] =
                    db1 * grid_u32_byte(grid1, j) as f32 * sign_of(sg & t::KMASK_IQ2XS[j] != 0);
                y[y_off + j + 4] =
                    db1 * grid_u32_byte(grid2, j) as f32 * sign_of(sg & t::KMASK_IQ2XS[j + 4] != 0);
            }
            y_off += 8;
        }

        // Second half: `qh[1]`, `qs` window 16*k+8, `signs` window 8*k+4, db2.
        let qh1 = qh[2 * k + 1] as u32;
        for l in 0..4 {
            let idx1 = qs[16 * k + 8 + 2 * l] as u32 | ((qh1 << (8 - 2 * l)) & 256);
            let idx2 = qs[16 * k + 8 + 2 * l + 1] as u32 | ((qh1 << (7 - 2 * l)) & 256);
            let grid1 = t::IQ3S_GRID[idx1 as usize];
            let grid2 = t::IQ3S_GRID[idx2 as usize];
            let sg = signs[8 * k + 4 + l];
            for j in 0..4 {
                y[y_off + j] =
                    db2 * grid_u32_byte(grid1, j) as f32 * sign_of(sg & t::KMASK_IQ2XS[j] != 0);
                y[y_off + j + 4] =
                    db2 * grid_u32_byte(grid2, j) as f32 * sign_of(sg & t::KMASK_IQ2XS[j + 4] != 0);
            }
            y_off += 8;
        }

        ib32 += 2;
    }
}

/// Port of `dequantize_row_iq1_s`. Block: `d: f16`, `qs: u8[32]`, `qh: u16[8]`.
pub(crate) fn dequantize_block_iq1_s(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let qs = &block[2..34];
    let qh = &block[34..50];

    let mut y_off = 0usize;
    for ib in 0..QK_K / 32 {
        let qh_ib = le_u16_at(qh, 2 * ib);
        let dl = d * (2.0 * ((qh_ib >> 12) & 7) as f32 + 1.0);
        let delta = if qh_ib & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let idx = qs[4 * ib + l] as u32 | (((qh_ib >> (3 * l)) & 7) as u32) << 8;
            let grid = t::IQ1S_GRID[idx as usize];
            for j in 0..8 {
                y[y_off + j] = dl * (grid_u64_i8(grid, j) as f32 + delta);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq1_m`. Block: `qs: u8[32]`, `qh: u8[16]`,
/// `scales: u8[8]` (read as `u16[4]`; the super-scale is bit-packed across
/// the four, not a standalone `d` field).
pub(crate) fn dequantize_block_iq1_m(block: &[u8], y: &mut [f32]) {
    let qs = &block[0..32];
    let qh = &block[32..48];
    let sc_bytes = &block[48..56];

    let sc = [
        le_u16_at(sc_bytes, 0),
        le_u16_at(sc_bytes, 2),
        le_u16_at(sc_bytes, 4),
        le_u16_at(sc_bytes, 6),
    ];
    let scale_u16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16::from_bits(scale_u16).to_f32();

    let mut y_off = 0usize;
    for ib in 0..QK_K / 32 {
        let shift = 6 * (ib % 2);
        let dl1 = d * (2.0 * ((sc[ib / 2] >> shift) & 7) as f32 + 1.0);
        let dl2 = d * (2.0 * ((sc[ib / 2] >> (shift + 3)) & 7) as f32 + 1.0);

        let qs_off = 4 * ib;
        let qh_off = 2 * ib;
        let q0 = qh[qh_off] as u32;
        let q1 = qh[qh_off + 1] as u32;
        let idx = [
            qs[qs_off] as u32 | ((q0 << 8) & 0x700),
            qs[qs_off + 1] as u32 | ((q0 << 4) & 0x700),
            qs[qs_off + 2] as u32 | ((q1 << 8) & 0x700),
            qs[qs_off + 3] as u32 | ((q1 << 4) & 0x700),
        ];
        let delta = [
            if qh[qh_off] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[qh_off] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[qh_off + 1] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[qh_off + 1] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];

        for l in 0..4 {
            let dl = if l < 2 { dl1 } else { dl2 };
            let grid = t::IQ1S_GRID[idx[l] as usize];
            for j in 0..8 {
                y[y_off + j] = dl * (grid_u64_i8(grid, j) as f32 + delta[l]);
            }
            y_off += 8;
        }
    }
}

/// Port of `dequantize_row_iq4_xs`. Block: `d: f16`, `scales_h: u16`,
/// `scales_l: u8[4]`, `qs: u8[128]`.
pub(crate) fn dequantize_block_iq4_xs(block: &[u8], y: &mut [f32]) {
    let d = le_f16(&block[0..2]);
    let scales_h = le_u16_at(block, 2);
    let scales_l = &block[4..8];
    let qs = &block[8..136];

    let mut y_off = 0usize;
    for ib in 0..QK_K / 32 {
        let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as u32
            | ((((scales_h >> (2 * ib)) & 3) as u32) << 4)) as i32;
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            y[y_off + j] = dl * t::KVALUES_IQ4NL[(qs[16 * ib + j] & 0xf) as usize] as f32;
            y[y_off + j + 16] = dl * t::KVALUES_IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32;
        }
        y_off += 32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    /// All-zero payload with `d = 1.0` drives every codebook lookup to index 0
    /// (whose entry is known and simple) and every sign bit to `+`, so the
    /// whole block dequantizes to one hand-computable constant. Expected
    /// values below are written out as arithmetic, not read back from the
    /// implementation.
    #[test]
    fn test_iq2_xxs_all_zero_block_matches_hand_computed_value() {
        // db = 1 * (0.5 + 0) * 0.25 = 0.125; grid[0] bytes = 8 -> y = 0.125*8 = 1.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq2_xxs(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_iq2_xs_all_zero_block_matches_hand_computed_value() {
        // db = 1 * (0.5 + 0) * 0.25 = 0.125; grid[0] bytes = 8 -> y = 1.
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq2_xs(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_iq2_s_all_zero_block_matches_hand_computed_value() {
        // Same constant as iq2_xs: db = 0.125, grid[0] bytes = 8, signs[0] = 0.
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq2_s(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_iq3_xxs_all_zero_block_matches_hand_computed_value() {
        // db = 1 * (0.5 + 0) * 0.5 = 0.25; iq3xxs grid[0] = 0x04040404 -> byte 4 -> 1.0.
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq3_xxs(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_iq3_s_all_zero_block_matches_hand_computed_value() {
        // db1 = d*(1+2*0) = 1; iq3s grid[0] = 0x01010101 -> byte 1 -> 1.0.
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq3_s(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_iq1_s_all_zero_block_matches_hand_computed_value() {
        // dl = d*(2*0+1) = 1, delta = +0.125; iq1s grid[0] = all -1
        // -> y = 1 * (-1 + 0.125) = -0.875.
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq1_s(&block, &mut y);
        for v in &y {
            assert_eq!(*v, -0.875);
        }
    }

    #[test]
    fn test_iq1_m_super_scale_unpacking_and_hand_computed_values() {
        // sc[2] = 0x8000 puts 0x8 in scale_u16 bits 8..11, sc[3] = 0x3000
        // puts 0x3 in bits 12..15 -> scale_u16 = 0x3800 = f16 0.5.
        // Every per-block scale nibble is 0 -> dl = d*(2*0+1) = 0.5,
        // delta = +0.125, iq1s grid[0] = all -1 -> y = 0.5*(-0.875) = -0.4375.
        let mut block = vec![0u8; 56];
        block[52..54].copy_from_slice(&0x8000u16.to_le_bytes());
        block[54..56].copy_from_slice(&0x3000u16.to_le_bytes());
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq1_m(&block, &mut y);
        for v in &y {
            assert_eq!(*v, -0.4375);
        }
    }

    #[test]
    fn test_iq4_xs_all_zero_block_matches_hand_computed_value() {
        // ls = 0 -> dl = 1*(0 - 32) = -32; kvalues_iq4nl[0] = -127 -> y = 4064.
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq4_xs(&block, &mut y);
        for v in &y {
            assert_eq!(*v, 4064.0);
        }
    }

    #[test]
    fn test_iq_sign_bits_flip_sign() {
        // iq2_xxs: set the sign so that element 0 of every 8-group is negative.
        // ksigns_iq2xs[127] = 255 (all sign bits set), so the high nibble of
        // aux32_1 (bits 7..14 of aux32_1's low byte) must be 127 for l=0:
        // aux32_1 = 0x7f in its low byte. db = 1*(0.5 + 0)*0.25 = 0.125,
        // grid[0] byte = 8 -> y = -1.0.
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2 + 4] = 0x7f; // first byte of aux32_1 (offset 4 within the 8-byte pair)
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq2_xxs(&block, &mut y);
        assert_eq!(y[0], -1.0);
    }

    #[test]
    fn test_iq2_xs_scales_select_per_half_multiplier() {
        // scales[0] = 0x30 -> low nibble 0 (db0 = 0.125), high nibble 3
        // (db1 = (0.5+3)*0.25 = 0.875). l=0,1 use db0; l=2,3 use db1.
        // grid[0] byte = 8 -> y[0] = 1.0, y[16] = 7.0.
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[66] = 0x30;
        let mut y = vec![0f32; QK_K];
        dequantize_block_iq2_xs(&block, &mut y);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[16], 7.0);
    }
}
