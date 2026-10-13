#!/usr/bin/env python3
"""Rewrites the head of a GGUF file as a tokenizer-only GGUF.

Input: the first N bytes of a real GGUF (an HTTP range request on the first
~40 MB is enough: the metadata section, vocab and merges included, comes before
the tensor table). Output: the same header and metadata key/value bytes,
copied verbatim, with `tensor_count = 0` and padding to `general.alignment`.
The `split.*` keys of a multi-file GGUF's first shard are dropped: they
describe how the tensors are spread over the shards, and with them llama.cpp
goes looking for the other files.
The result loads in Reflex (`Tokenizer::from_gguf`) and in llama.cpp's
`llama-tokenize`, which only read the vocab. This is how the
`test-data/*-tokenizer.gguf` fixtures are made (see docs/DEVELOPMENT.md's
test-fixture section).

Usage: python3 scripts/gguf_tokenizer_only.py <head.gguf> <out.gguf>
Fetch a head with e.g.
  curl -L -r 0-41943039 -o head.gguf https://huggingface.co/<repo>/resolve/main/<file>.gguf
"""

import struct
import sys

FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY, UINT32 = 8, 9, 4


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    if data[:4] != b"GGUF":
        sys.exit(f"{src}: not a GGUF file")
    (version,) = struct.unpack_from("<I", data, 4)
    if version < 2:
        sys.exit(f"{src}: GGUF v{version} unsupported (needs v2+, 64-bit counts)")
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24
    alignment = 32

    def skip_value(pos: int, ty: int) -> int:
        if ty in FIXED:
            return pos + FIXED[ty]
        if ty == STRING:
            (n,) = struct.unpack_from("<Q", data, pos)
            return pos + 8 + n
        if ty == ARRAY:
            item_ty, n = struct.unpack_from("<IQ", data, pos)
            pos += 12
            if item_ty in FIXED:
                return pos + FIXED[item_ty] * n
            for _ in range(n):
                pos = skip_value(pos, item_ty)
            return pos
        sys.exit(f"unknown GGUF value type {ty} at byte {pos}")

    kvs = bytearray()
    kept = 0
    for _ in range(n_kv):
        kv_start = pos
        (klen,) = struct.unpack_from("<Q", data, pos)
        key = data[pos + 8 : pos + 8 + klen].decode("utf-8")
        pos += 8 + klen
        (ty,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if key == "general.alignment" and ty == UINT32:
            (alignment,) = struct.unpack_from("<I", data, pos)
        pos = skip_value(pos, ty)
        if pos > len(data):
            sys.exit(f"{src}: metadata runs past the {len(data)} bytes fetched; fetch more")
        if not key.startswith("split."):
            kvs += data[kv_start:pos]
            kept += 1

    out = bytearray(data[:8])
    out += struct.pack("<QQ", 0, kept)
    out += kvs
    out += b"\0" * ((-len(out)) % alignment)
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out)} bytes, {kept} keys (dropped {n_tensors} tensor infos, {n_kv - kept} split keys)")


if __name__ == "__main__":
    main()
