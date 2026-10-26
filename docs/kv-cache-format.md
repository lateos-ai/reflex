# KV cache file format

A KV cache file holds a model's state after a prompt, so a later process can resume from
it instead of prefilling again. `reflex generate --export-kv <file>` (IPC `export_kv`)
writes one, and `--import-kv <file>` (IPC `import_kv`) resumes from it. This page is the
byte-level spec, plus what an orchestrator chaining processes through these files can
rely on. For the flags, the f16 accuracy numbers and timings, see
[reference.md](reference.md#kv-cache-files).

The engine doesn't manage these files. It reads and writes the path it is given. Where
files live, how long they're kept and which one belongs to which agent are the
orchestrator's job (see [DEVELOPMENT.md](DEVELOPMENT.md)'s Non-goals).

## Layout

Every integer is a `u32`, little endian. Every value is little endian, either `f32` or
IEEE `f16`. A file is a header followed by the payload, with no padding, alignment or
trailing bytes. The header always starts with the 4 ASCII bytes `CSKV` and a `u32`
version.

| version | layout | values | written by today's exporter |
|---:|---|---|---|
| 1 | dense/MoE | f32 | no (still imports) |
| 2 | hybrid | f32 | yes, f32 export |
| 3 | MLA | f32 | yes, f32 export |
| 4 | dense/MoE, with `space_prefix_next` | f32 | yes, f32 export |
| 5 | as version 4 | f16 | yes, f16 export |
| 6 | as version 2 | attention K/V f16, GatedDeltaNet state f32 | yes, f16 export |
| 7 | as version 3 | f16 | yes, f16 export |

### Dense/MoE (versions 1, 4, 5)

```
"CSKV" | version | num_layers | seq_len | num_kv_heads | head_dim
       | space_prefix_next: u8        (versions 4 and 5 only; 0 or 1)
payload, for each layer in order:
       k_cache: seq_len * num_kv_heads * head_dim values
       v_cache: seq_len * num_kv_heads * head_dim values
```

The header is 24 bytes in version 1 and 25 bytes in versions 4 and 5.
`space_prefix_next` records whether the cached prompt's encoding ended on a special
token. It tells a SentencePiece tokenizer whether the continuation gets its leading
space. A version-1 file has no such byte; resuming from one adds the space.

Used by Qwen3, Qwen3-MoE, Qwen2/2.5, Llama, Mistral and Kolibri-1.

### Hybrid (versions 2, 6)

```
"CSKV" | version | num_layers | seq_len | attn_num_kv_heads | attn_head_dim
       | gdn_conv_state_len | gdn_recurrent_len
       | num_layers kind bytes: 0 = attention, 1 = GatedDeltaNet
payload, for each layer in order:
       attention:     k_cache, v_cache  (seq_len * attn_num_kv_heads * attn_head_dim each)
       GatedDeltaNet: conv_state (gdn_conv_state_len), recurrent (gdn_recurrent_len), always f32
```

`seq_len` counts the positions in the attention layers' cache. The GatedDeltaNet state
has a fixed size, whatever the prompt length. In version 6 only the attention K/V values
are f16. The recurrent state stays f32 because rounding it costs more accuracy than
rounding K/V does.

Used by Qwen3.5 (`qwen35`, `qwen35moe`).

### MLA (versions 3, 7)

```
"CSKV" | version | num_layers | seq_len | qk_dim
payload, for each layer in order:
       kv_cache: seq_len * qk_dim values
```

`qk_dim` is `kv_lora_rank + qk_rope_head_dim`: the compressed latent, followed by the
RoPE part of K. There is no separate V. Used by DeepSeek-V2/V3-style models
(`deepseek2`).

### Not available

Gemma 4 and Granite 4.0-H have no cache format and return `unsupported_architecture`.
Embedding models refuse prefix state with `invalid_input`.

## What you can rely on

- **Old files keep importing.** Each version above keeps its meaning, and a new layout
  gets a new number. The byte layouts are pinned by hand-built files in
  `src/kv_io.rs`'s tests, so a change to the reader and writer together can't move
  the format silently.
- **Exports write the newest version for their layout**: 4, 2 or 3 for f32, and 5, 6
  or 7 for f16. f16 is opt-in (`--export-kv-dtype f16`, IPC `"export_kv_dtype":
  "f16"`) unless the process runs an f16 working cache (`REFLEX_KV_CACHE=f16`): exports
  default to the working cache's type. Any engine imports any of the seven versions,
  into an f32 or an f16 working cache.
- **Bad files fail before any model work**, with the error category `kv_cache`:
  - wrong magic or an unknown version;
  - a file length that doesn't match its header, whether short or with bytes left over;
  - an invalid flag or layer-kind byte;
  - a layer count or head shape that doesn't match the model;
  - another architecture's layout, such as an MLA file given to a dense model.

  A file holding more positions than the attention kernels support (65,535) fails with
  `context_overflow`.
- **Don't modify a file while a request resumes from it.** Import reads only the header
  up front. The payload is read while it's uploaded, and a file that shrinks in between
  fails that request with `kv_cache`.

## What the format does not check

The header records shapes, not which model wrote the file. A file from a different model
that happens to have the same shape (for example a fine-tune of the same base, the same
model with another LoRA adapter, or another weight storage mode) imports without an
error, and generation then continues from state that model never produced. **Key files
by model file, adapter and engine version on the orchestrator side**, and resume only
into the same combination that exported them. A file is also tied to its tokenizer: the
continuation's tokens are appended at position `seq_len`.
