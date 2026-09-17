//! GGUF file format parser (Phase 21.1).
//!
//! Parses the GGUF binary container format (used by llama.cpp-family model
//! releases) directly from a memory-mapped file: header, typed metadata
//! key/value store (architecture params like `llama.expert_count`, plus the
//! embedded tokenizer), and the tensor info table (name/shape/quant-type/
//! data-offset). Tensor *data* is read lazily as byte slices into the mmap
//! (see [`GgufFile::tensor_bytes`]) — the whole file is never copied into
//! process memory, since MoE checkpoints run tens of GB.
//!
//! Spec reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>.
//! Implemented from the spec text; **round-trip-tested here only against a
//! hand-built synthetic file that matches the documented byte layout, not
//! yet tried against a real HuggingFace/llama.cpp-exported GGUF file** —
//! same "unverified until tried on real data" posture this repo uses for
//! anything untested on real GPU hardware.
//!
//! Quantized tensor *dequantization* (turning `Q4_K`/`Q5_K`/etc. raw block
//! bytes into f32) is a separate concern, kept in a sibling module —
//! see [`crate::dequant::dequantize`].

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // bytes "GGUF" read little-endian as u32
const DEFAULT_ALIGNMENT: u64 = 32;

/// A parsed GGUF metadata value. Arrays can nest (an `Array` of `Array`),
/// matching the spec's recursive `GGUF_METADATA_VALUE_TYPE_ARRAY`.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u64),
            GgufValue::I16(v) if v >= 0 => Some(v as u64),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GgufValue::F32(v) => Some(v),
            GgufValue::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// GGML tensor element type (`ggml_type`). Only a subset used by common
/// GGUF releases is named explicitly; anything else round-trips through
/// `Unknown` rather than failing the whole file — a released model can ship
/// a newer quant type (e.g. `IQ*`) than this parser's dequantizers know
/// about, and structural parsing should still succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // Phase 21.13: the IQ-family ("i-quant") types a real GSQ-RCO GGUF uses.
    // Deliberately not the full speculative `IQ*` set — see `dequant_iq`'s
    // module doc for the real-file inspection this list came from.
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ3S,
    IQ2S,
    IQ4XS,
    IQ1M,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            29 => GgmlType::IQ1M,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::Bf16,
            other => GgmlType::Unknown(other),
        }
    }
}

/// One entry from the GGUF tensor info table.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Element shape, slowest-to-fastest-varying dimension order as stored
    /// in the file (GGUF stores dims in ggml's own order, not necessarily
    /// row-major-from-the-left — callers doing real math should consult the
    /// architecture's known layout, this parser passes dims through as-is).
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the data section (i.e. already
    /// excludes header+metadata+tensor-table+padding); always a multiple of
    /// the file's alignment.
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A parsed GGUF file, backed by a memory-mapped byte range so tensor data
/// is never copied wholesale into process memory.
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    data_section_start: usize,
}

impl GgufFile {
    /// Open and parse a GGUF file, mmap'ing its contents.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("opening GGUF file: {e}"))?;
        // Safety: standard mmap caveat — the file must not be mutated by
        // another process while mapped. Checkpoint files are read-only
        // model artifacts in every real usage this repo has.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap GGUF file: {e}"))?;
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, String> {
        let mut cur = Cursor::new(&mmap);

        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "not a GGUF file: magic={magic:#010x}, expected {GGUF_MAGIC:#010x}"
            ));
        }
        let version = cur.read_u32()?;
        let tensor_count = cur.read_u64()?;
        let metadata_kv_count = cur.read_u64()?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let value = cur.read_value()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            let ggml_type = GgmlType::from_u32(cur.read_u32()?);
            let offset = cur.read_u64()?;
            tensors.push(GgufTensorInfo { name, shape, ggml_type, offset });
        }

        // The data section starts at the next `alignment`-byte boundary
        // after the tensor info table.
        let unaligned_end = cur.pos as u64;
        let data_section_start = unaligned_end.div_ceil(alignment) * alignment;
        let data_section_start = data_section_start as usize;

        if data_section_start > mmap.len() {
            return Err(format!(
                "data section start {data_section_start} beyond file length {}",
                mmap.len()
            ));
        }

        Ok(GgufFile { mmap, version, metadata, tensors, data_section_start })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes for a tensor, as a slice directly into the mmap — no copy.
    pub fn tensor_bytes(&self, info: &GgufTensorInfo) -> Result<&[u8], String> {
        let start = self.data_section_start + info.offset as usize;
        let nbytes = ggml_type_size_bytes(info.ggml_type, info.element_count())?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| format!("tensor '{}' byte range overflows usize", info.name))?;
        self.mmap
            .get(start..end)
            .ok_or_else(|| format!("tensor '{}' range {start}..{end} exceeds file length {}", info.name, self.mmap.len()))
    }
}

/// Number of raw bytes an `element_count`-element tensor of `ggml_type`
/// occupies on disk. Block-quantized types round the element count up to a
/// full block per GGUF's own on-disk padding rule.
fn ggml_type_size_bytes(ggml_type: GgmlType, element_count: u64) -> Result<usize, String> {
    let (block_size, block_bytes): (u64, u64) = match ggml_type {
        GgmlType::F32 => (1, 4),
        GgmlType::F16 | GgmlType::Bf16 => (1, 2),
        GgmlType::F64 => (1, 8),
        GgmlType::I8 => (1, 1),
        GgmlType::I16 => (1, 2),
        GgmlType::I32 => (1, 4),
        GgmlType::I64 => (1, 8),
        GgmlType::Q8_0 => (32, 34),   // ggml_fp16_t d + 32 * int8
        GgmlType::Q8_1 => (32, 36),   // 2x ggml_fp16_t (d, s) + 32 * int8
        GgmlType::Q4_0 => (32, 18),   // ggml_fp16_t d + 16 bytes (32x nibble)
        GgmlType::Q4_1 => (32, 20),   // 2x ggml_fp16_t (d, m) + 16 bytes
        GgmlType::Q5_0 => (32, 22),   // ggml_fp16_t d + 4 bytes high bits + 16 bytes
        GgmlType::Q5_1 => (32, 24),   // 2x ggml_fp16_t (d, m) + 4 bytes high bits + 16 bytes
        GgmlType::Q2K => (256, 84),
        GgmlType::Q3K => (256, 110),
        GgmlType::Q4K => (256, 144),
        GgmlType::Q5K => (256, 176),
        GgmlType::Q6K => (256, 210),
        GgmlType::Q8K => (256, 292),
        // Phase 21.13: per-block byte sizes computed from the reference
        // `block_iq*` struct layouts in `ggml-common.h` (the same standard
        // `dequant.rs` uses for the K-quants), each verified by the
        // `static_assert`s in that header.
        GgmlType::IQ2XXS => (256, 66),  // f16 d + 32 * u16
        GgmlType::IQ2XS => (256, 74),   // f16 d + 32 * u16 + u8 scales[8]
        GgmlType::IQ2S => (256, 82),    // f16 d + u8 qs[64] + qh[8] + scales[8]
        GgmlType::IQ3XXS => (256, 98),  // f16 d + u8 qs[96]
        GgmlType::IQ3S => (256, 110),   // f16 d + qs[64] + qh[8] + signs[32] + scales[4]
        GgmlType::IQ1S => (256, 50),    // f16 d + u8 qs[32] + u16 qh[8]
        GgmlType::IQ1M => (256, 56),    // u8 qs[32] + qh[16] + scales[8] (packed f16 scale)
        GgmlType::IQ4XS => (256, 136),  // f16 d + u16 scales_h + scales_l[4] + qs[128]
        GgmlType::Unknown(t) => {
            return Err(format!("unknown ggml_type={t}: on-disk block size not known to this parser"))
        }
    };
    let num_blocks = element_count.div_ceil(block_size);
    Ok((num_blocks * block_bytes) as usize)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "cursor offset overflow".to_string())?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| format!("unexpected end of file at offset {} (need {n} more bytes)", self.pos))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.take(1)?[0] as i8)
    }
    fn read_u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn read_bool(&mut self) -> Result<bool, String> {
        Ok(self.read_u8()? != 0)
    }

    /// GGUF string: `u64` byte length, then that many UTF-8 bytes (not
    /// null-terminated).
    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| format!("string not valid UTF-8: {e}"))
    }

    fn read_value_type(&mut self) -> Result<u32, String> {
        self.read_u32()
    }

    fn read_scalar(&mut self, value_type: u32) -> Result<GgufValue, String> {
        Ok(match value_type {
            0 => GgufValue::U8(self.read_u8()?),
            1 => GgufValue::I8(self.read_i8()?),
            2 => GgufValue::U16(self.read_u16()?),
            3 => GgufValue::I16(self.read_i16()?),
            4 => GgufValue::U32(self.read_u32()?),
            5 => GgufValue::I32(self.read_i32()?),
            6 => GgufValue::F32(self.read_f32()?),
            7 => GgufValue::Bool(self.read_bool()?),
            8 => GgufValue::String(self.read_string()?),
            10 => GgufValue::U64(self.read_u64()?),
            11 => GgufValue::I64(self.read_i64()?),
            12 => GgufValue::F64(self.read_f64()?),
            other => return Err(format!("unsupported GGUF metadata value type {other}")),
        })
    }

    /// Read one metadata value, recursing into `ARRAY` (type 9).
    fn read_value(&mut self) -> Result<GgufValue, String> {
        let value_type = self.read_value_type()?;
        if value_type == 9 {
            let elem_type = self.read_value_type()?;
            let len = self.read_u64()?;
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                if elem_type == 9 {
                    items.push(self.read_value_as_array()?);
                } else {
                    items.push(self.read_scalar(elem_type)?);
                }
            }
            Ok(GgufValue::Array(items))
        } else {
            self.read_scalar(value_type)
        }
    }

    /// Helper for nested arrays: an array element that is itself declared
    /// as an array carries its own `elem_type`+`len` header (no outer
    /// value_type prefix, since the outer array already declared type 9).
    fn read_value_as_array(&mut self) -> Result<GgufValue, String> {
        let elem_type = self.read_value_type()?;
        let len = self.read_u64()?;
        let mut items = Vec::with_capacity(len as usize);
        for _ in 0..len {
            if elem_type == 9 {
                items.push(self.read_value_as_array()?);
            } else {
                items.push(self.read_scalar(elem_type)?);
            }
        }
        Ok(GgufValue::Array(items))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal, spec-correct synthetic GGUF byte buffer: 2
    /// metadata KVs (a string and a u32), one F32 tensor of 4 elements, and
    /// default alignment. Used because no real GGUF file is available in
    /// this repo/environment — this is round-trip self-consistency, not
    /// real-file verification (see module doc).
    fn build_synthetic_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count

        // metadata[0]: "general.architecture" = "llama" (string type = 8)
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut buf, "llama");

        // metadata[1]: "llama.attention.head_count" = 32 (u32 type = 4)
        write_string(&mut buf, "llama.attention.head_count");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        // tensor[0]: "tok.weight", shape [4], type F32 (0), offset 0
        write_string(&mut buf, "tok.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to 32-byte alignment for the data section.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Tensor data: 4 f32 values.
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_temp_file(bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        // pid alone collides across tests running in parallel in the same
        // process (they all share one pid) — add a per-call counter too.
        path.push(format!("rft_gguf_test_{}_{unique}.gguf", std::process::id()));
        let mut f = File::create(&path).expect("create temp gguf file");
        f.write_all(bytes).expect("write temp gguf file");
        path
    }

    #[test]
    fn test_parses_synthetic_gguf_header_and_metadata() {
        let bytes = build_synthetic_gguf();
        let path = write_temp_file(&bytes);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();

        assert_eq!(file.version, 3);
        assert_eq!(
            file.metadata.get("general.architecture").and_then(GgufValue::as_str),
            Some("llama")
        );
        assert_eq!(
            file.metadata.get("llama.attention.head_count").and_then(GgufValue::as_u64),
            Some(32)
        );
    }

    #[test]
    fn test_parses_tensor_info_and_reads_data_via_mmap() {
        let bytes = build_synthetic_gguf();
        let path = write_temp_file(&bytes);
        let file = GgufFile::open(&path).expect("parse synthetic GGUF");
        std::fs::remove_file(&path).ok();

        let info = file.tensor_info("tok.weight").expect("tensor present");
        assert_eq!(info.shape, vec![4]);
        assert_eq!(info.ggml_type, GgmlType::F32);

        let raw = file.tensor_bytes(info).expect("tensor bytes");
        assert_eq!(raw.len(), 16);
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rejects_bad_magic() {
        let mut bytes = build_synthetic_gguf();
        bytes[0] = 0; // corrupt magic
        let path = write_temp_file(&bytes);
        let result = GgufFile::open(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn test_array_metadata_value_parses() {
        // Build a file with an ARRAY-of-U32 metadata value:
        // "arr" -> [1, 2, 3] (array type = 9, elem type = 4 (u32))
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        write_string(&mut buf, "arr");
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem type u32
        buf.extend_from_slice(&3u64.to_le_bytes()); // len
        for v in [1u32, 2, 3] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        let path = write_temp_file(&buf);
        let file = GgufFile::open(&path).expect("parse array-metadata GGUF");
        std::fs::remove_file(&path).ok();

        match file.metadata.get("arr") {
            Some(GgufValue::Array(items)) => {
                let vals: Vec<u64> = items.iter().map(|v| v.as_u64().unwrap()).collect();
                assert_eq!(vals, vec![1, 2, 3]);
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn test_ggml_type_size_bytes_q8_0_rounds_up_to_full_block() {
        // 33 elements needs 2 Q8_0 blocks (32-element blocks) = 68 bytes.
        let size = ggml_type_size_bytes(GgmlType::Q8_0, 33).unwrap();
        assert_eq!(size, 2 * 34);
    }

    #[test]
    fn test_ggml_type_size_bytes_f32_exact() {
        let size = ggml_type_size_bytes(GgmlType::F32, 10).unwrap();
        assert_eq!(size, 40);
    }

    /// Phase 21.13: the IQ-family type ids a real GSQ-RCO GGUF uses must map
    /// to named variants (not `Unknown`) and to the exact reference block
    /// byte sizes from `ggml-common.h`.
    #[test]
    fn test_iq_family_type_id_and_block_size_table() {
        let cases = [
            (16u32, GgmlType::IQ2XXS, 66u64),
            (17, GgmlType::IQ2XS, 74),
            (18, GgmlType::IQ3XXS, 98),
            (19, GgmlType::IQ1S, 50),
            (21, GgmlType::IQ3S, 110),
            (22, GgmlType::IQ2S, 82),
            (23, GgmlType::IQ4XS, 136),
            (29, GgmlType::IQ1M, 56),
        ];
        for (id, ty, block_bytes) in cases {
            assert_eq!(GgmlType::from_u32(id), ty, "type id {id}");
            assert_eq!(ggml_type_size_bytes(ty, 256).unwrap() as u64, block_bytes, "{ty:?}");
            // 33 elements needs 2 super-blocks -> 2x the per-block size.
            assert_eq!(ggml_type_size_bytes(ty, 257).unwrap() as u64, 2 * block_bytes, "{ty:?}");
        }
    }
}
