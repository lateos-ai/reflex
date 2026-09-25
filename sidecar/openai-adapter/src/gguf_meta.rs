//! Minimal, standalone GGUF header + key/value-metadata reader.
//!
//! This crate deliberately does **not** depend on the root `reflex-engine` library
//! (see this crate's own `Cargo.toml`/README -- pulling that in would drag `build.rs`'s
//! `nvcc` requirement into the sidecar's build). The root `src/gguf.rs` parser is a
//! full tensor-info/mmap-based reader built for that constraint; this module hand-rolls
//! a much smaller subset of the same binary format (see
//! <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>) sufficient for one
//! job: pull specific string/scalar/array keys out of the metadata section, in
//! particular `tokenizer.chat_template`, `tokenizer.ggml.bos_token_id`/
//! `eos_token_id`, and `tokenizer.ggml.tokens` (so a template that references `{{
//! bos_token }}`/`{{ eos_token }}` can be given real values). Tensor info and tensor
//! data are never parsed -- reading stops as soon as the metadata key/value section
//! ends, well before the (potentially many-GB) tensor data.

use std::fs::File;
use std::io::{BufReader, Read};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" read little-endian as u32

// Several scalar variants are only ever constructed to keep the byte cursor advancing
// correctly past metadata keys this crate doesn't otherwise care about; only
// `String`/the integer `as_u64` conversions/`Array` are actually read back.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum GgufMetaValue {
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
    Array(Vec<GgufMetaValue>),
}

impl GgufMetaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufMetaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufMetaValue::U8(v) => Some(v as u64),
            GgufMetaValue::U16(v) => Some(v as u64),
            GgufMetaValue::U32(v) => Some(v as u64),
            GgufMetaValue::U64(v) => Some(v),
            GgufMetaValue::I8(v) if v >= 0 => Some(v as u64),
            GgufMetaValue::I16(v) if v >= 0 => Some(v as u64),
            GgufMetaValue::I32(v) if v >= 0 => Some(v as u64),
            GgufMetaValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufMetaValue::Array(items) => items.iter().map(GgufMetaValue::as_str).collect(),
            _ => None,
        }
    }
}

pub struct GgufMeta {
    pub metadata: std::collections::HashMap<String, GgufMetaValue>,
}

impl GgufMeta {
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufMetaValue::as_str)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufMetaValue::as_u64)
    }

    pub fn get_string_array(&self, key: &str) -> Option<Vec<&str>> {
        self.metadata
            .get(key)
            .and_then(GgufMetaValue::as_string_array)
    }

    /// Reads just the GGUF header + KV metadata section from `path` -- stops before
    /// the tensor info table / tensor data, so this is cheap regardless of the
    /// checkpoint's total on-disk size.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("opening GGUF file: {e}"))?;
        let mut r = Reader {
            inner: BufReader::new(file),
        };

        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "not a GGUF file: magic={magic:#010x}, expected {GGUF_MAGIC:#010x}"
            ));
        }
        let _version = r.read_u32()?;
        let _tensor_count = r.read_u64()?;
        let metadata_kv_count = r.read_u64()?;

        let mut metadata = std::collections::HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = r.read_string()?;
            let value = r.read_value()?;
            metadata.insert(key, value);
        }

        Ok(GgufMeta { metadata })
    }
}

struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn take(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; n];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| format!("unexpected end of file (need {n} more bytes): {e}"))?;
        Ok(buf)
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
        String::from_utf8(bytes).map_err(|e| format!("string not valid UTF-8: {e}"))
    }

    fn read_scalar(&mut self, value_type: u32) -> Result<GgufMetaValue, String> {
        Ok(match value_type {
            0 => GgufMetaValue::U8(self.read_u8()?),
            1 => GgufMetaValue::I8(self.read_i8()?),
            2 => GgufMetaValue::U16(self.read_u16()?),
            3 => GgufMetaValue::I16(self.read_i16()?),
            4 => GgufMetaValue::U32(self.read_u32()?),
            5 => GgufMetaValue::I32(self.read_i32()?),
            6 => GgufMetaValue::F32(self.read_f32()?),
            7 => GgufMetaValue::Bool(self.read_bool()?),
            8 => GgufMetaValue::String(self.read_string()?),
            10 => GgufMetaValue::U64(self.read_u64()?),
            11 => GgufMetaValue::I64(self.read_i64()?),
            12 => GgufMetaValue::F64(self.read_f64()?),
            other => return Err(format!("unsupported GGUF metadata value type {other}")),
        })
    }

    /// Reads one metadata value, recursing into `ARRAY` (type 9) -- covers every
    /// value type the format defines so an unrelated key never desyncs the cursor
    /// for keys parsed after it.
    fn read_value(&mut self) -> Result<GgufMetaValue, String> {
        let value_type = self.read_u32()?;
        if value_type == 9 {
            self.read_array_body()
        } else {
            self.read_scalar(value_type)
        }
    }

    fn read_array_body(&mut self) -> Result<GgufMetaValue, String> {
        let elem_type = self.read_u32()?;
        let len = self.read_u64()?;
        let mut items = Vec::with_capacity(len as usize);
        for _ in 0..len {
            if elem_type == 9 {
                items.push(self.read_array_body()?);
            } else {
                items.push(self.read_scalar(elem_type)?);
            }
        }
        Ok(GgufMetaValue::Array(items))
    }
}
