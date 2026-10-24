# Text embeddings: `mistral3` encoders (Nemotron-3.5-Embed)

Reflex can run an embedding model: a GGUF whose attention is bidirectional
(`<arch>.attention.causal = false`) and whose output is the pooled final hidden state.
The model this was built for is NVIDIA's
[Nemotron-3.5-Embed-1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16), a
retrieval embedding model on the Ministral 3 text tower (`general.architecture =
mistral3`). Text only: its Pixtral vision tower (the separate `mmproj` GGUF) is not
supported.

Embedding endpoints are usually scale-to-zero and serve one request at a time with no
decode loop, so the cold-start cost is most of what a caller waits for. That makes
them a good fit for this engine.

## The model

| | Nemotron-3.5-Embed-1B |
|---|---|
| Layers, hidden, FFN | 16, 2048, 6144 (SwiGLU) |
| Attention | 24 query / 8 KV heads of dim 128, **bidirectional**, no QK-norm, no biases |
| RoPE | consecutive-pair (llama.cpp `NORM`), base 1e6, **YaRN** factor 16 over an original context of 16,384 |
| Query temperature | Llama-4 style: Q times `1 + 0.1 * ln(floor(pos / 16384) + 1)`, after RoPE |
| Output | final RMSNorm, **mean** over every token, L2-normalized, 2048 dims (truncatable) |
| Tokenizer | Tekken BPE (`pre = pixtral`), no BOS/EOS |
| Input format | the GGUF's chat template: `"query: " + text` or `"passage: " + text`, with U+200C after each `[` and `<` of the text |

## Design

**Detection.** `config::is_encoder` is true iff `<arch>.attention.causal` is `false`.
`parse_encoder_config` then requires `pooling_type = 1` (mean; other pooling types are
rejected) and reads YaRN and the query temperature. A causal model with a pooling type
(Qwen3-Embedding's last-token pooling) is not an encoder and loads as before.
`mistral3` is accepted only as an encoder: a Ministral 3 chat model would also need
YaRN and the temperature scale on the decode path, so it is rejected by name.

**Forward pass.** An encoder runs the dense path's batched prefill unchanged except
for three hooks, each a no-op for every other model:

- `Model::rope_batch` runs `rope_norm_yarn_batch_kernel` (the YaRN kernel MLA already
  uses) when the encoder has YaRN.
- `Model::encoder_temp_scale` multiplies Q after RoPE by the temperature factor. The
  factor is exactly 1 below the floor scale, so the kernel only launches once a row
  reaches position 16,384.
- `Model::attention_prefill` sends an encoder to `attention_prefill_bidir`.

Then `Model::embed` applies the final norm to every row, averages the rows
(`mean_pool_kernel`, one thread per column summing in row order, so the result doesn't
depend on the launch shape) and L2-normalizes on the host. There is no LM head and no
tied-embedding dequantization. `generate`, `system1` and prefix caching return
`invalid_input` on an encoder, and `embed` returns it on a language model.

**Bidirectional attention.** The online and tiled kernel bodies gained a
`bool CAUSAL = true` template parameter:

- **Online:** a query row's range is `[0, start_pos + rows)` instead of
  `[0, start_pos + row]`.
- **Tiled:** every row walks every K/V tile, and only padding positions are masked.

The bidirectional instances live in their own modules: `encoder.cu` (online, plus the
temperature and pooling kernels) and `attention_tiled_bidir.cu` (head dims 64 and 128).
Causal models never load them, and the causal cubins are byte-identical to before (see
Results). The f16 K/V cache and `REFLEX_ATTN_KERNEL=legacy` are refused for encoders.

**YaRN parameters.** These follow llama.cpp's `llama-context.cpp` setup and ggml's
`ggml_rope_yarn_corr_dims`, which agree with HF `transformers`'
`_compute_yarn_parameters`:

- **Correction range:** floor/ceil-truncated, [20, 37] for this model.
- **Ramp:** the same linear ramp.
- **cos/sin magnitude:** `get_mscale(f, 1) / get_mscale(f, yarn_log_multiplier)`, which
  is 1.0 here. The HF config has `mscale = mscale_all_dim = 1`, and the GGUF's
  `yarn_log_multiplier` is 1.

**Tokenizer.** The Tekken split and the bracket-marker normalization are described in
[tokenizer-pre-types.md](tokenizer-pre-types.md).

## Surfaces

| Surface | Call |
|---|---|
| CLI | `reflex embed <gguf> "query: What is NeMo?" [--json]` |
| IPC (`reflex stdio`/`uds`) | `{"prompt": "query: What is NeMo?", "embed": true}` → `embedding`, `prompt_tokens` |
| C FFI | `reflex_embed(handle, text, &out)` / `reflex_free_embed_result(&out)` |
| HTTP (sidecar) | `POST /v1/embeddings` (OpenAI shape, plus `input_type: query\|passage`) |
| Runpod Hub | a job whose input has `input` and no `messages` goes to `/v1/embeddings` |

The engine embeds the text it is given. The sidecar renders each input through the
GGUF's own chat template, with a system message holding the prefix and a user message
holding the text. For engine-level callers, the text must already carry the prefix.

A list input is embedded one text at a time, in order. The engine never batches,
by design (docs/DEVELOPMENT.md's Non-goals).

## Results

Measured 2026-10-23 on a Tesla T4 (AWS g4dn.xlarge), against HF `transformers` 5.20.2
in fp32 (SDPA, the model card's mean pooling), over 27 inputs:
- queries and passages in English, German, French, Chinese, Japanese, Russian, Spanish,
  Hindi and Arabic;
- code, and text with `[INST]`/`<s>`/`[CLS]`/`<mask>`;
- three passages of 1,244, 9,304 and 17,364 tokens (the last past the 16,384-position
  original context, so the query temperature applies).

The reference script and inputs are in `test-data/tiny-mistral3-embed-src.tar.gz`.

**Agreement with HF** (cosine, worst over the 27 inputs; token ids identical on all 27):

| GGUF | `--weights f32` | default (`f16`, Q4_K_M kept quantized) |
|---|---|---|
| BF16 (`maternion/Nemotron-3.5-Embed-1B-GGUF`) | **1.0000000** | 0.9999998 |
| Q8_0 (same repo) | 0.99986 | 0.99986 |
| Q4_K_M (`W4D/Nemotron-3.5-Embed-1B-GGUF`) | 0.9754 | 0.9754 |

Q4_K_M is the file's quantization, not the engine: its worst inputs are the two 3-token
queries (an average over three tokens), it gives ~0.99 elsewhere, and the cosine is the
same with quantized-resident weights (the default) and with `REFLEX_QUANT_RESIDENT=0`.
llama.cpp on the same file gives 0.978 and 0.977 on those two inputs (Reflex: 0.979 and
0.975).

**llama.cpp as a third implementation.** llama.cpp `1a24df1`'s `llama-embedding --pooling
mean` on the same BF16 file agrees with HF to 1.000000 on 22 of the 25 inputs it ran
(it skipped the two longest). It reaches only 0.968, 0.993 and 0.991 on the three
inputs containing `[` or `<`. For a `pre = pixtral` file it keeps the template's U+200C
markers, which HF's tokenizer (and Reflex's) removes. On Q8_0 it gives 0.9997-0.9999
on the other inputs, against Reflex's 0.9999: llama.cpp quantizes activations to 8
bits, Reflex keeps them in f32.

**The fixture** (`tiny-mistral3-embed.gguf`, 15 inputs up to 159 tokens, past its
original context of 64): cosine 1.0000000 with f32 weights, through both the tiled and
the online attention kernels, and 0.9999981 with f16 weights. HF's own output with one
feature removed scores well below the f32 test bar of 0.99999: causal attention 0.477,
no YaRN 0.831, no query temperature 0.99990.

**Causal models are unchanged.**
- **Cubins:** with `REFLEX_CUDA_ARCH=sm_75`, the `attention_online`,
  `attention_tiled*` (f32 and f16), `kv_f16`, `attention_decode*` and `rope` cubins are
  byte-identical to master's.
- **Greedy tokens:** 32-token generations are the same on Qwen3-0.6B, Mistral 7B,
  Qwen3.5-0.8B and TinyLlama, for a 45-character prompt and a 2,940-character one
  (the tiled path).
- **GPU suite:** 141 passed, 4 skipped (optional models), and 1 failed: the Kolibri-1
  QR test, which fails on master too.

**Cold start** (`reflex embed`, one 11-token query):

| GGUF | Process start to embedding (Reflex's own timer, p50 n=10) | Whole process, wall clock (p50 n=10) | llama.cpp `llama-embedding`, wall clock |
|---|---|---|---|
| Q4_K_M | 494 ms (model load 305 ms) | 604 ms | 2,646 ms |
| Q8_0 | 550 ms (model load 342 ms) | 672 ms | 2,744 ms |
| BF16 | 670 ms (model load 452 ms) | | |

BF16 loaded in 4.1 s before this work moved F16/BF16 tensor conversion onto the GPU
(the pipelined loader now runs a one-element dequant kernel for them, exactly as the
host conversion rounds). The 11-token forward pass is 8 ms on Q4_K_M (quantized
resident, the fused multi-row kernel) and 25 ms on Q8_0 (dequantized to f16, cuBLAS).


## Not supported

- **Images**: the `mmproj` Pixtral tower.
- **Pooling other than mean**: CLS, last-token, rank.
- **Causal embedding models**: Qwen3-Embedding loads as a language model.
- **Ministral 3 chat models.**
- **Q8_0 kept quantized on the dense path**: it is dequantized to the `--weights` dtype
  at load, as for other dense models.
