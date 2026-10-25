//! `reflex model-info`: what a GGUF says about itself, for provenance records
//! (model cards, deployment attestations). Reads the file's `general.*`
//! metadata (name, license, source URL, organization, quantizer, ...), its
//! architecture and `general.file_type`, and with `--sha256` the file's
//! SHA-256. No model load, no GPU: only the GGUF header is parsed (mmap), so
//! it runs on any machine, including a `REFLEX_SKIP_CUDA=1` build.
//!
//! Usage: `reflex model-info <path-to-gguf> [--sha256] [--json]`
//!
//! Plain text prints one `REFLEX_MODEL_INFO_GENERAL key=<key> value=<value>`
//! line per scalar (or array-of-scalars) `general.*` key, in key order, then
//! one `REFLEX_MODEL_INFO_OK` line. `--json` (needs `--features json-output`)
//! prints a single `ModelInfoJson` object instead (see
//! `reflex_engine::cli_output`). The hash is opt-in because it reads every
//! byte of the file: seconds for a multi-GB model.
//!
//! The metadata is whatever the file's converter wrote: a missing or wrong
//! `general.license` is reported as found, never filled in or corrected.

#[cfg(not(feature = "json-output"))]
use crate::phase::json_output_unavailable;
use reflex_engine::gguf::{GgufFile, GgufValue};
use std::collections::BTreeMap;
use std::io::Read;

const USAGE: &str = "usage: reflex model-info <path-to-gguf> [--sha256] [--json]";

pub fn run(args: Vec<String>) {
    let mut gguf_path: Option<String> = None;
    let mut sha256 = false;
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "--sha256" => sha256 = true,
            "--json" => json = true,
            _ if gguf_path.is_none() && !arg.starts_with("--") => gguf_path = Some(arg),
            other => {
                eprintln!("{USAGE}\nunexpected argument: {other}");
                std::process::exit(2);
            }
        }
    }
    let Some(gguf_path) = gguf_path else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };

    let info = ModelInfo::read(&gguf_path, sha256).unwrap_or_else(|e| crate::fail(&gguf_path, e));
    if json {
        #[cfg(feature = "json-output")]
        reflex_engine::cli_output::print_json_line(&info.to_json());
        #[cfg(not(feature = "json-output"))]
        json_output_unavailable();
    } else {
        for (key, value) in &info.general {
            println!("REFLEX_MODEL_INFO_GENERAL key=general.{key} value={value:?}");
        }
        let mut line = format!(
            "REFLEX_MODEL_INFO_OK path={:?} size_bytes={} gguf_version={} tensor_count={}",
            info.path, info.size_bytes, info.gguf_version, info.tensor_count
        );
        if let Some(arch) = &info.architecture {
            line.push_str(&format!(" architecture={arch}"));
        }
        if let Some(ft) = info.file_type {
            line.push_str(&format!(" file_type={ft}"));
        }
        if let Some(name) = info.file_type_name {
            line.push_str(&format!(" file_type_name={name}"));
        }
        if let Some(hash) = &info.sha256 {
            line.push_str(&format!(" sha256={hash}"));
        }
        println!("{line}");
    }
}

struct ModelInfo {
    path: String,
    size_bytes: u64,
    gguf_version: u32,
    tensor_count: usize,
    architecture: Option<String>,
    file_type: Option<u64>,
    file_type_name: Option<&'static str>,
    /// `general.*` keys other than `architecture`/`file_type`, without the
    /// `general.` prefix.
    general: BTreeMap<String, String>,
    sha256: Option<String>,
}

impl ModelInfo {
    fn read(path: &str, sha256: bool) -> Result<Self, String> {
        let size_bytes = std::fs::metadata(path)
            .map_err(|e| format!("reading file size: {e}"))?
            .len();
        let file = GgufFile::open(path).map_err(|e| e.to_string())?;
        let mut general = BTreeMap::new();
        for (key, value) in &file.metadata {
            let Some(key) = key.strip_prefix("general.") else {
                continue;
            };
            if matches!(key, "architecture" | "file_type") {
                continue;
            }
            if let Some(text) = value_text(value) {
                general.insert(key.to_string(), text);
            }
        }
        let file_type = file
            .metadata
            .get("general.file_type")
            .and_then(|v| v.as_u64());
        let info = Self {
            path: path.to_string(),
            size_bytes,
            gguf_version: file.version,
            tensor_count: file.tensors.len(),
            architecture: file
                .metadata
                .get("general.architecture")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            file_type,
            file_type_name: file_type.and_then(file_type_name),
            general,
            sha256: None,
        };
        drop(file);
        Ok(Self {
            sha256: if sha256 {
                Some(sha256_file(path).map_err(|e| format!("hashing: {e}"))?)
            } else {
                None
            },
            ..info
        })
    }

    #[cfg(feature = "json-output")]
    fn to_json(&self) -> reflex_engine::cli_output::ModelInfoJson {
        reflex_engine::cli_output::ModelInfoJson {
            schema_version: reflex_engine::cli_output::SCHEMA_VERSION,
            path: self.path.clone(),
            size_bytes: self.size_bytes,
            gguf_version: self.gguf_version,
            tensor_count: self.tensor_count,
            architecture: self.architecture.clone(),
            file_type: self.file_type,
            file_type_name: self.file_type_name,
            general: self.general.clone(),
            sha256: self.sha256.clone(),
        }
    }
}

/// A metadata value as text: scalars as written, an array of scalars joined
/// with `", "`. Nested arrays (none of the `general.*` keys the GGUF spec
/// defines) give `None` and are left out.
fn value_text(value: &GgufValue) -> Option<String> {
    Some(match value {
        GgufValue::String(s) => s.clone(),
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F32(v) => v.to_string(),
        GgufValue::F64(v) => v.to_string(),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::Array(items) => items
            .iter()
            .map(|item| match item {
                GgufValue::Array(_) => None,
                scalar => value_text(scalar),
            })
            .collect::<Option<Vec<_>>>()?
            .join(", "),
    })
}

/// llama.cpp's `llama_ftype` names (`general.file_type`), without the
/// `MOSTLY_` prefix. Unknown values return `None`; the number is still reported.
fn file_type_name(file_type: u64) -> Option<&'static str> {
    Some(match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

fn sha256_file(path: &str) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.hex_digest())
}

/// FIPS 180-4 SHA-256, so the core build keeps its four dependencies for one
/// opt-in flag.
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    total_len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len += data.len() as u64;
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len < 64 {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.block_len = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            self.compress(block);
        }
        self.block[..rest.len()].copy_from_slice(rest);
        self.block_len = rest.len();
    }

    fn hex_digest(mut self) -> String {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        let used = (self.block_len + 1) % 64;
        pad.resize(1 + if used <= 56 { 56 - used } else { 120 - used }, 0);
        pad.extend_from_slice(&bit_len.to_be_bytes());
        let total_len = self.total_len;
        self.update(&pad);
        self.total_len = total_len;
        debug_assert_eq!(self.block_len, 0);
        self.state.iter().map(|w| format!("{w:08x}")).collect()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.hex_digest()
    }

    #[test]
    fn sha256_matches_fips_180_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // One million 'a's, fed in uneven pieces to cover the partial-block path.
        let mut hasher = Sha256::new();
        let data = vec![b'a'; 1_000_000];
        for piece in data.chunks(997) {
            hasher.update(piece);
        }
        assert_eq!(
            hasher.hex_digest(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_padding_lengths_around_a_block_boundary() {
        // 55, 56 and 64 bytes: one-block, two-block and exact-block padding.
        for (len, expected) in [
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
        ] {
            assert_eq!(sha256_hex(&vec![b'a'; len]), expected, "{len} bytes");
        }
    }

    /// `test-data/tiny-qwen2.gguf` (see docs/DEVELOPMENT.md's test fixtures):
    /// its `general.*` keys as llama.cpp's converter wrote them, and its
    /// SHA-256 as `sha256sum` prints it. Skips if the fixture is absent.
    #[test]
    fn tiny_qwen2_model_info() {
        const FIXTURE: &str = "test-data/tiny-qwen2.gguf";
        if !std::path::Path::new(FIXTURE).exists() {
            eprintln!("skipping: {FIXTURE} not found");
            return;
        }
        let info = ModelInfo::read(FIXTURE, true).expect("model-info on the fixture");
        assert_eq!(info.architecture.as_deref(), Some("qwen2"));
        assert_eq!(info.gguf_version, 3);
        assert_eq!(
            info.size_bytes,
            std::fs::metadata(FIXTURE).unwrap().len(),
            "size"
        );
        assert_eq!(
            info.sha256.as_deref(),
            Some("0345b81cccca11e769286ef155f47826bfbeb522b965045b826b8981af6edcd4")
        );
        assert_eq!(
            (info.file_type, info.file_type_name),
            (Some(0), Some("F32"))
        );
        assert_eq!(info.tensor_count, 26);
        assert_eq!(info.general["name"], "Tiny Qwen2");
        assert_eq!(info.general["size_label"], "10M");
        assert_eq!(info.general["sampling.top_k"], "20");
        // The converter wrote no license, so none is reported.
        assert!(!info.general.contains_key("license"));
        assert!(!info.general.contains_key("architecture"));
        assert!(!info.general.contains_key("file_type"));
        let without_hash = ModelInfo::read(FIXTURE, false).unwrap();
        assert_eq!(without_hash.sha256, None, "hash only with --sha256");
    }

    /// A real Granite 4.0-H 350M header (the Hub dropdown's GGUF, tensor data
    /// stripped): the license and base-model keys IBM's file carries. Skips if
    /// absent.
    #[test]
    fn granite_header_reports_its_license() {
        const HEADER: &str = "test-data/granite-4.0-h-350m-header.gguf";
        if !std::path::Path::new(HEADER).exists() {
            eprintln!("skipping: {HEADER} not found");
            return;
        }
        let info = ModelInfo::read(HEADER, false).unwrap();
        assert_eq!(info.architecture.as_deref(), Some("granitehybrid"));
        assert_eq!(info.file_type_name, Some("Q8_0"));
        assert_eq!(info.general["license"], "apache-2.0");
        assert_eq!(info.general["base_model.0.organization"], "Ibm Granite");
    }

    #[test]
    fn arrays_of_scalars_are_joined_and_nested_arrays_dropped() {
        let tags = GgufValue::Array(vec![
            GgufValue::String("a".into()),
            GgufValue::String("b".into()),
        ]);
        assert_eq!(value_text(&tags).as_deref(), Some("a, b"));
        let nested = GgufValue::Array(vec![GgufValue::Array(vec![GgufValue::U8(1)])]);
        assert_eq!(value_text(&nested), None);
    }
}
