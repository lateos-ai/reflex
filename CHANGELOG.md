# Changelog

What has been built, in order. Measurements live in
[docs/benchmarks.md](docs/benchmarks.md).

## Unreleased

- **Run report.** `generate` and `system1` end with one `REFLEX_RUN_REPORT` line (and a
  `--json` object): the phase timings plus the GPU name and compute capability, driver
  versions, kernel image (PTX/cubin/fatbin, native or PTX JIT), weight mode
  (quantized-resident/f16/f32), KV cache type, model file name and size, and energy when
  measured, so a pasted run can be compared with published numbers. Collected after the
  timed work from what the run already has; existing `REFLEX_*_OK` and `REFLEX_PHASE_OK`
  lines are unchanged. `--json` schema 1.5.0. Fields:
  [docs/reference.md](docs/reference.md#run-report).

- **Energy in the Docker images.** Every image (`Dockerfile`'s `reflex`, `adapter` and
  `runpod-lb` targets, `.runpod/Dockerfile`, `.modal/Dockerfile`) is now built with the
  `nvml` feature, so `reflex generate`/`system1`/`smoke` in a container print `joules=`
  and `energy_method=` when run with `-e REFLEX_ENERGY=1`. The `reflex` image also
  contains `reflex-energy`.
  `REFLEX_FEATURES` now defaults to `nvml`; pass `ipc,nvml` where you passed `ipc`. NVML
  comes from the host driver through the CUDA base images' `utility` capability; without
  it the binary runs as before, without the energy fields, and `reflex doctor` warns.
  Served responses (sidecar, Runpod handler) are unchanged and carry no energy figures.

- **In-process energy is opt-in: `REFLEX_ENERGY=1`.** A binary built with `nvml` no
  longer loads NVML unless `REFLEX_ENERGY=1` is set. Starting NVML and reading its
  counter at every phase boundary happened inside the timed window and added about 30 ms
  to cold start on a T4 (`smoke` p50 144.2 vs 177.2 ms, `system1` 354.3 vs 384.6 ms,
  n=10, interleaved). Set `REFLEX_ENERGY=1` to get `joules=` and the per-phase energy
  fields back. `reflex doctor` and `reflex-energy` work as before. With the flag unset
  the `nvml` image is back at the image without `nvml` (T4, p50, n=10, interleaved:
  `smoke` 144.1 vs 145.5 ms, `system1` 350.9 vs 354.2 ms).

- **Faster first constrained request.** The JSON Schema vocabulary index, built on a
  process's first `json_schema` request, now takes fewer passes and allocations: token
  bytes go into one buffer, the sort is on integer keys, and one pass builds the trie.
  It gives the same index (checked against the old build on all 45 tokenizers in
  `test-data/`) and is still built lazily. T4, Qwen3-0.6B: build 65 → 31 ms, and a
  schema's extra first-token cost on a fresh process +57 → +23 ms. Outputs, `reflex check`,
  unconstrained cold start and the T4 GPU test suite are unchanged.
  `REFLEX_GRAMMAR_PROFILE=1` prints the build time on its own. Cold and warm numbers:
  [docs/design/agent-relay.md](docs/design/agent-relay.md), phase C.

- **Faster cold start: shorter load-init path.** The load's one background thread builds
  the tokenizer and then the cuBLAS handle. The load thread now builds cuBLAS itself if
  it finishes its weights first. The tokenizer builds its vocab and merge lookups as
  open-addressing indexes over one byte buffer: same token ids, checked on 63.46M ids.
  cuBLAS setup waits for the prefetch readers and for both pinned staging slots to be
  allocated and touched. A first touch of a fresh pinned slot during `CudaBlas::new`
  could stall one fill for ~78 ms. Still one extra thread per load.
  `REFLEX_LOAD_PROFILE=1` adds `REFLEX_LOAD_INIT`, `REFLEX_LOAD_CUBLAS` and
  `REFLEX_LOAD_SLOTS` lines. `system1` p50, n=10, interleaved with the previous master:
  T4: Qwen3-0.6B 413 → 342 ms, Qwen3-1.7B 459 → 414 ms, Mistral 7B 1285 → 1150 ms.
  A10G: 431 → 376, 488 → 406, 935 → 818 ms. `reflex check` output identical on both
  GPUs, and the T4 GPU test suite gives the same results as master.
- **Opt-in compliance records.** `reflex model-info <gguf>` prints a GGUF's
  provenance metadata (license, source, size, optional sha256), no GPU needed. The
  Runpod Hub worker gains three deploy fields, all off by default: `COMPLIANCE=on`
  adds an attestation (model sha256 checked against `.runpod/models.json`, license,
  engine version, GPU, the job's token counts and wall time) to every response and
  answers `{"compliance": "report"}` with the image's SBOMs, third-party licenses and a
  model card; `AUDIT_LOG=stdout|volume` writes one JSON line per job, with the prompt
  and answer optionally hashed or in full (`AUDIT_LOG_CONTENT`). The images carry the
  compliance bundle (CycloneDX SBOMs, license texts, package lists) under
  `/usr/share/reflex/compliance/`. With every option off, responses are unchanged:
  checked on an RTX A5000 against the previous handler, byte-identical for chat,
  streamed chat and classify. See [docs/compliance/README.md](docs/compliance/README.md).
- **Runpod Hub: optional RAG.** A new `RAG` deploy field (off by default) makes the Hub
  worker run an embedding model (Nemotron-3.5-Embed-1B) next to the chat model. A chat
  job with a `rag` object is answered from the passages closest to its question, taken
  from documents sent with the job, a folder on the network volume (`RAG_DIR`) or a
  corpus URL (`RAG_CORPUS_URL`). The passages are returned with the answer, and the
  corpus index is cached on the volume. Retrieval is host-side Python in the worker
  (`.runpod/rag.py`); the engine and the adapter are unchanged. See
  [.runpod/README.md](.runpod/README.md#optional-rag).
- **Batched Gated DeltaNet prefill (Qwen3.5, `qwen35moe`).** Prompts no longer run
  through the Gated DeltaNet layers one token at a time. The projections and the FFN
  are GEMMs over the whole prompt, and the conv and delta-rule recurrences run in
  kernels that walk the rows in order, keeping the recurrent state in registers. Each
  row is bit-identical to the single-token kernels; only the GEMM summation order
  changes (relative L2 1.2e-6 on the last hidden row with f32 weights). lev `Q8_0` on a
  T4: a 1,054-token prefill takes 0.67 s (was 26.3 s), and the `/v1/systemone`
  support-ticket example 330 ms warm (was 7.06 s) and 1.7 s from a cold process (was
  8.4 s; llama-server 3.5 s). `reflex check` token ids unchanged. See
  [docs/design/decision-models.md](docs/design/decision-models.md#batched-gated-deltanet-prefill).
- **Decision models: `POST /v1/systemone`.** The OpenAI adapter answers llama.cpp's
  `/v1/systemone` (TypeSafe's System One API) for the label-token decision models on
  the Qwen3.5 hybrid path: lev (ggml-org's `lev-GGUF`) and OpenJev, text only. A
  request is a `state` and typed questions (`choice`, `score`, `noul`); each answer is
  read from the logits of one label token per option, with the temperatures stored in
  the GGUF. New:
  - **Engine:** IPC `label_prompts`, the raw logits of single-token labels at the end of
    several prompts. Their common token prefix is evaluated once and each prompt
    continues from a copy of it; the session's pinned prefix is untouched.
  - **Sidecar:** the `systemone` template, rendered with llama.cpp's `tojson` rules and
    lev's sorted keys and reversed second variant, plus llama.cpp's validation, error
    messages, temperature buckets and confidence formulas. `kev`/`laya` models and
    images get `501`.

  Checked against llama.cpp `a4cb4c6` on a T4 with 53 requests (1 to 60 options, every
  question type, string/object/array/chat states, non-ASCII, floats, special-token
  text, 1K-8K-token states, 22 error cases). Every prompt matched llama.cpp's own
  renderer byte for byte and token for token. Against llama-server with f32 activations
  and an f32 KV cache, the largest probability difference was 1.3e-3 on lev `Q8_0`
  (f16 weights) and 2.9e-4 on the tiny test models, with the same choices and
  `input_tokens` everywhere. With the batched Gated DeltaNet prefill (above), the
  support-ticket example takes 330 ms warm on a T4 (llama-server: 300 ms). See
  [docs/design/decision-models.md](docs/design/decision-models.md#results).
- **Text embeddings: Nemotron-3.5-Embed-1B.** A GGUF with bidirectional attention
  (`<arch>.attention.causal = false`) now loads as an embedding model. NVIDIA's
  Nemotron-3.5-Embed-1B (`mistral3`) is the one supported, text only. New:
  - **Engine:** bidirectional attention kernels, YaRN RoPE and the Llama-4 query
    temperature on the dense path, mean pooling, and the Tekken tokenizer
    (`pre = pixtral`).
  - **Surfaces:** `reflex embed`, IPC `"embed": true`, C FFI `reflex_embed`, the OpenAI
    adapter's `POST /v1/embeddings` (with `input_type`, `dimensions` and base64), and
    Hub jobs with an `input` field. The Hub's Model dropdown offers
    Nemotron-3.5-Embed-1B Q8_0 (1.2 GB download).

  Against HF `transformers` in fp32 on a T4, over 27 inputs up to 17,364 tokens, the
  BF16 file with f32 weights matches to a cosine of 1.0000000; Q8_0 gives 0.99986, and
  Q4_K_M 0.975 (its quantization: llama.cpp gives the same on that file).
  From process start to the first embedding takes ~0.5 s (Q4_K_M) to 0.67 s (BF16) on
  the T4. Causal models are unaffected: their attention cubins are byte-identical and
  greedy tokens are unchanged. See [docs/design/embeddings.md](docs/design/embeddings.md).
- **F16 and BF16 GGUFs load on the GPU.** Their tensors used to be converted on the host,
  one thread, before upload; they now go through the pipelined loader like quantized
  ones. Nemotron-3.5-Embed BF16's load fell from 4.1 s to 0.45 s on a T4.
- **Runpod Hub: pick the model when you deploy.** The listing's deploy form has a Model
  dropdown with 12 GGUFs covering every architecture that fits its 16-24 GB cards:
  Qwen3 0.6B (the default, still baked into the image) / 1.7B / 4B / 8B, Qwen2.5 1.5B /
  7B Instruct, TinyLlama 1.1B Chat, Mistral 7B Instruct v0.2, Qwen3.5 0.8B, Granite 4.0-H
  350M / 1B and Gemma 4 12B-it. Any choice but the default is downloaded once when a
  worker starts (to the network volume when one is attached), and a Custom GGUF URL field
  (with an optional Hugging Face token) takes any other supported model. Container disk
  10 -> 25 GB. See [.runpod/README.md](.runpod/README.md), "Choosing the model".
- **IPC `stop_token_ids`.** A `generate` request can name extra stop tokens; the
  generation ends at EOS or at the first of those. Without the field nothing changes
  (EOS-only, what `reflex check` uses).
- **OpenAI adapter: chat works on non-Qwen3 models.** Testing every Hub model showed three
  bugs, now fixed:
  - Replies ended in the stop token's text (`<|im_end|>`, `</s>`,
    `<|end_of_text|>`). The adapter now strips it, in streams too.
  - Models whose turn ends on a token other than their EOS kept generating. The adapter
    now sends each model's end-of-turn tokens as `stop_token_ids` (Qwen3.5,
    Gemma 4 `<turn|>`).
  - The Qwen3.5, Gemma 4 and Mistral chat templates failed to load, so those models
    got a generic prompt. Templates now render with Hugging Face's settings
    (`trim_blocks`, `lstrip_blocks`, Python string/dict methods), and a template that
    rejects a system message gets it folded into the first user turn. All 12 models'
    templates render byte-identical to Python `jinja2`.

  Checked on a Runpod A40: all 12 models download, load and answer chat, streamed chat
  and classify jobs, and `reflex generate`'s greedy tokens are unchanged against the
  previous master (no stop tokens passed).
- **128-row tile for long quantized prefills** (sm_80+). Above 64 rows the tensor-core
  prefill kernel can now carry 128 input rows per block, so it dequantizes each weight half
  as often. It's used when its grid still covers half the SMs and its last block isn't
  mostly padding. It gives the same bits as the 64-row tile at the same split count.
  `REFLEX_QUANT_MMA_TILE=64|128` forces either tile. Both tiles also got cheaper
  dequantization (same f16 values) and `ldmatrix` loads.
  - **Warm prefill on an A10G**, p50 ms, new / before / QR off: Qwen3-0.6B at 29 / 113 /
    449 tokens 5.55 / 9.65 / 25.8 (was 5.63 / 9.88 / 27.1; QR off 7.38 / 9.89 / 28.2);
    Mistral 7B at 33 / 129 / 513 tokens 23.0 / 71.5 / 186.6 (was 23.8 / 73.5 / 214.3; QR
    off 39.5 / 75.0 / 195.4). QR is now faster than QR off at every A10G bucket.
  - **T4 unchanged.** On sm_75 the 128-row tile ran no faster than the 64-row one, so
    Turing keeps the 64-row tile and the row cap. The 64-row kernel's shared changes still help it there: T4
    Qwen3-0.6B 8.06 / 16.5 ms at 29 / 113 tokens (was 8.44 / 16.7), Mistral 7B 60.5 / 204 ms
    at 33 / 129 (was 63.4 / 210). QR off is still faster on the T4 from 129 tokens on Mistral
    and at 449 on Qwen3-0.6B.
  - Greedy tokens are identical across both tiles, forced 1 split, `REFLEX_QUANT_MMA=0`
    and QR off, on Qwen3-0.6B and Mistral 7B, on both GPUs. See "Step 5" in
    [docs/design/quantized-mma-prefill.md](docs/design/quantized-mma-prefill.md).
- **Tensor-core prefill straight from quantized weights** (#47, #48). With quantized-resident
  weights and `--weights f16` (both defaults), prefill GEMMs of 2+ rows on Q4_K and Q6_K
  weights now run a `mma.sync` kernel (`gemm_q_mma.cu`, sm_75+, loaded on first use). It
  dequantizes the raw GGUF blocks in registers to the same f16 values the old
  dequantize-to-scratch path wrote and accumulates in f32. That removes the scratch round
  trip that made QR prefill slower than `REFLEX_QUANT_RESIDENT=0`.
  - **Split-K** for grids too small to fill the GPU, with a fixed-order reduction (no
    atomics, so repeated runs give the same bits). `REFLEX_QUANT_MMA_SPLITS=<n>` forces the
    count.
  - **Row cap on sm_75.** Turing has no k16 MMA, so on a T4 Q4_K GEMMs above 384 rows and
    Q6_K above 256 go back to the scratch path. There's no cap from sm_80 up.
    `REFLEX_QUANT_MMA_MAX_ROWS=<n>` sets the cap on any device, and `REFLEX_QUANT_MMA=0`
    turns the kernel off.
  - **Warm prefill**, p50 ms, new / QR before / QR off. A10G: Qwen3-0.6B at 29 / 113 / 449
    tokens 5.65 / 9.92 / 27.3 (was 11.25 / 14.42 / 32.9; QR off 7.41 / 9.93 / 28.5); Mistral
    7B at 33 / 129 / 513 tokens 23.9 / 73.7 / 213.5 (was 105.1 / 140.3 / 261.8; QR off
    39.5 / 75.1 / 195.8). T4: Qwen3-0.6B 8.40 / 16.6 / 68.7 (was 23.3 / 30.1 / 69.2; QR off
    10.9 / 16.8 / 55.5); Mistral 7B 64.1 / 214 / 636 (was 298 / 359 / 645; QR off 85.5 /
    151 / 432). Every bucket is at or below its old QR number. QR off is still faster for
    Mistral at 513 tokens on the A10G, and on the T4 for Mistral from 129 tokens and
    Qwen3-0.6B at 449. Decode is unchanged.
  - **Correctness.** Greedy tokens are identical with the kernel, forced 1 split,
    `REFLEX_QUANT_MMA=0` and QR off, on both GPUs. `reflex check` against llama.cpp on the T4
    (Qwen3-0.6B and Mistral 7B, short and 500+-token prompts) gives the same 32 tokens in all
    modes. The one fork from stock llama.cpp is a 0.022 near-tie, which llama.cpp itself
    resolves the other way with flash attention off and an f32 KV cache.
  - See [docs/design/quantized-mma-prefill.md](docs/design/quantized-mma-prefill.md).
- **Quantized-resident weights are now the default.** `REFLEX_QUANT_RESIDENT` defaults to
  on; set it to `0` to opt out and load matrix weights as `f16`. This halves VRAM usage
  for quantized models (Q4_K/Q6_K on the dense path, Q8_0 on Gemma 4), speeds up cold
  start by removing per-tensor allocations, and improves decode speed by reading half the
  weight bytes per token. Short warm prompts and `system1` become slower (~15–40 ms extra)
  because the fused quantized kernels lose to cuBLAS tensor cores above ~4 rows. See
  [design/quantized-resident-weights.md](docs/design/quantized-resident-weights.md).
  **Verified 2026-10-20** on an AWS EC2 `g5.xlarge` (NVIDIA A10G, driver 595.91.07,
  CUDA 13.2, `sm_86`): GPU suite 128 passed / 0 failed / 2 skipped in both modes;
  47 `reflex check` comparisons with zero token-id divergences. Qwen3-0.6B
  `system1` 445–451 ms vs 442–444 (1–2% slower), `generate` 458–461 vs 620,
  decode 3.48 vs 3.81 ms/token, 270 vs 878 MiB; Mistral 7B `system1` 1.03 vs
  2.93 s and 4.1 vs 13.8 GB; Gemma 4 12B first token 1,994–1,999 ms at 11.1 GB
  (the `f16` opt-out OOMs on 24 GB). Also fixed two decode-path bugs this exposed:
  an f16 fused-GEMV double reduction and one-row Q6_K `gemv` not using the r1
  kernel that `gemv_to` uses.

- **Decode speed, round 2.** These change no outputs: greedy ids and logits are
  bit-identical, and the f32 attention kernels compile to the same cubins.
  - **Norms.** RMSNorm's single summing thread reads four floats at a time. The
    decode path fuses each residual add into the next norm and runs Q-Norm and K-Norm
    in one launch. `REFLEX_DECODE_FUSE=0` turns the fusion off.
  - **Workspace.** For dense models (Qwen3, Qwen2, Llama/Mistral), `generate`'s decode
    step reuses one workspace for all its intermediates and writes K/V straight into
    the cache, which removes ~390 allocations and frees per step.
    `REFLEX_DECODE_WS=0` turns it off.
  - **Results.** On a T4, Qwen3-0.6B decodes 6-9% faster and TinyLlama 3-4%. A
    32-token agent hop with a pinned ~1,550-token prefix takes 280 ms instead of 296
    ms, or 259 ms with the f16 KV cache below.
  - **Tried and dropped.** CUDA graphs per decode step gained under 2%, so they are
    not included.
  - See [docs/design/decode-speed.md](docs/design/decode-speed.md).
- **Opt-in f16 KV cache.** `REFLEX_KV_CACHE=f16` stores the working K/V cache in f16,
  at half the bytes. It covers every attention layer of dense, MoE, Kolibri-1 and
  Qwen3.5 models and MLA's latent; Qwen3.5's recurrent state stays f32.
  - **Speed.** On a T4, Qwen3-0.6B decodes 8-10% faster at ~1,550 positions.
  - **Accuracy.** It moves next-token logits by about 1e-4 to 7e-4 (relative L2)
    against an f32 cache, with the same 32 greedy tokens on every model measured.
  - **Per process.** The setting is read at load. The pinned prefix holds its state in
    the same type.
  - **Cache files.** With an f16 cache, export defaults to f16 files, and f32 files
    import with rounding.
  - **Unsupported.** Gemma 4, Granite 4.0-H and `REFLEX_ATTN_KERNEL=legacy` refuse it.
  - **Default.** Unchanged: f32, byte-identical.
- **Faster decode.** Decode attention for dense, MoE, Kolibri-1 and Qwen3.5 now runs a
  GQA-aware kernel (`attention_decode.cu`). It reads each cached K/V row once for every
  query head that shares it and keeps two loads in flight per warp; it loads on the
  first decode step. Decode outputs that kernels fully overwrite are no longer zeroed
  first. On a T4, Qwen3-0.6B decodes 15% faster at short context and 27% faster at
  ~1,550 positions; TinyLlama is 23% faster at ~1,550. `REFLEX_ATTN_DECODE=online`
  restores the previous kernel. See
  [docs/design/decode-speed.md](docs/design/decode-speed.md).
- **Agent handoff benchmark.** `scripts/agent_relay_bench.py` measures the per-hop
  round trip of a chain of sequential agent requests against `reflex stdio`,
  `reflex uds` or llama-server. The IPC protocol gains an opt-in `"timings": true`
  request field, which adds `parse_ms`/`first_token_ms`/`model_ms` to the final
  line; when the field is omitted, responses are unchanged. See
  [docs/design/agent-relay.md](docs/design/agent-relay.md).
- **Prefix reuse over IPC.** `reflex stdio`/`uds` requests can name an `export_kv`
  file, to save the state after the prompt, and an `import_kv` file, to resume from
  one. They use the same cache formats as `--export-kv`/`--import-kv`, and a request
  can set both to layer prefixes. The caller owns the files, so the engine still keeps
  nothing between requests. `agent_relay_bench.py --reuse-prefix` measures it.
- **One pinned prefix in VRAM over IPC.** A `reflex stdio`/`uds` request with
  `"pin_prefix": true` (and `"max_tokens": 1`) keeps the state after its prompt in the
  session's one slot. Later requests with `"use_pinned": true` resume from it through
  a device-to-device copy, with no file read or upload, and the slot itself never
  changes. Setting both fields layers a longer prefix on top. There is one slot, with
  no names and no eviction; anything more is left to the orchestrator. Each use needs
  a second copy of the prefix's KV in VRAM while it runs, and fails with
  `out_of_memory`, leaving the session working, if that can't be allocated. It covers
  dense/MoE (Qwen3, Qwen2, Llama/Mistral, Kolibri-1), Qwen3.5 and MLA; Gemma 4 and
  Granite 4.0-H return `unsupported_architecture`. On a T4 with Qwen3-0.6B and a
  ~1,490-token shared prefix, a 32-token hop takes 364 ms instead of 501 ms
  recomputed or 547 ms from a file. `agent_relay_bench.py --pin-prefix` measures it.
  `ipc::handle_request`, `handle_request_streaming` and `run_request_loop` now take
  the session's `&mut PrefixSlot`.
- **Faster quantized-resident decode.** With `REFLEX_QUANT_RESIDENT=1`, the decode-step
  Q4_K GEMV computes 4 output rows per warp and reuses each activation load across them.
  It is 1.6-1.8x faster per call and bit-identical to the previous kernel. Dense
  models' Q6_K layer tensors now stay quantized too (as Kolibri-1's already did), with a
  coalesced Q6_K dequant for prefill, and the Q4_K/Q6_K GEMVs no longer zero their
  outputs first. On a T4, decode is 16-31% faster: Qwen3-0.6B 6.09 -> 4.86 ms/token,
  Mistral 7B 45.8 -> 31.7 ms/token. Prefill costs 1-8% more, and Mistral's first token
  comes ~650 ms sooner. See
  [docs/design/quantized-resident-weights.md](docs/design/quantized-resident-weights.md).
- **Faster prefill after a cached prefix.** A prompt that resumes from a pinned prefix
  or an imported cache now runs the tiled attention kernel once `rows * group` fills
  one of its blocks, instead of waiting for 64 rows. Below that, the online kernel
  read the whole cached prefix once per row and head. On a T4 with Qwen3-0.6B, 41 rows
  after 1,489 cached positions reach the first token in 38 ms instead of 107 ms, and
  the agent relay's pinned long-prefix hop drops from 364 to 295 ms. Fresh prompts
  are unchanged. See
  [docs/design/tiled-prefill-attention.md](docs/design/tiled-prefill-attention.md).
- **Exact prefix reuse on SentencePiece models.** An `import_kv`/`use_pinned`
  continuation (and `reflex generate --import-kv`) on Llama-2, Mistral or TinyLlama
  used to be encoded with the tokenizer's leading `▁` space even when the prefix ended
  on raw text. So it wasn't token-identical to one request over the joined text: on
  TinyLlama, `<|user|>` after a cached `…</s>\n` became `▁<` instead of `<`. A
  continuation now gets that space only when the prefix ended on a special token, as
  llama.cpp's tokenizer does for the joined text.
  - Dense KV cache files gain one byte for this (format version 4). Version-1 files
    still import, with the old behavior.
  - The relay tests compare the split prompt's ids instead of its token count, and
    cover TinyLlama (`SPM_GGUF` in the GPU nightly).
  - gpt2-style tokenizers are unaffected.
- **JSON Schema constrained decoding.** A `reflex stdio`/`uds` request with
  `"json_schema": {...}`, or `reflex generate --json-schema <file>`, produces one
  JSON value that validates against the schema, so handoff payloads and tool
  calls parse on the first try. Tokens that can't continue a conforming value get a
  `-inf` logit before greedy or temperature/top-k/top-p sampling, and nothing else
  changes: a greedy output that already conforms comes out identical. It works on
  every architecture, with `use_pinned` and `import_kv`.
  - The supported subset is `type`, `properties`, `required`,
    `additionalProperties: false`, `items`, `enum`, `const`, `anyOf` and local `$ref`.
    Any other keyword is rejected with `invalid_input`, naming it.
  - Output is compact JSON with properties in schema order. The only whitespace is
    one optional space after each `:` and `,`, so a model keeps its usual
    `"key": "value"` tokens. If `max_tokens` runs out
    first, the request fails with the new `grammar_dead_end` error kind (C code 14),
    which carries the partial text.
  - The vocabulary trie that masking walks is built on the first constrained request,
    so cold start and unconstrained requests don't pay for it.
  - With `"timings": true`, `mask_ms_total` reports the masking time. Failed
    `generate` requests now carry `timings` too when asked.
  - `agent_relay_bench.py --json-schema` and `scripts/json_schema_compare.py` (against
    llama-server) measure it.
  - On a T4, masking costs 0.32 ms per token on Qwen3's 152K vocabulary. llama-server's
    grammar adds about 4.5 ms per token there. 20 of 20 comparison cases validate on
    both engines, and cold start is unchanged.
  - A constrained relay hop stops at the end of the value, so it is shorter than a
    free one: 447 against 576 ms with the default 64-token hop.

  See [docs/design/agent-relay.md](docs/design/agent-relay.md) and
  [docs/reference.md](docs/reference.md#json-schema-output).
- **Multi-process relay chains and energy.** `scripts/agent_relay_bench.py` gains
  cold targets, where every hop is a fresh process: `cold`, `cold-chain` (resuming a
  shared prefix through `--import-kv`), `cold-warm` (a cold export imported by a warm
  `reflex stdio`) and `llama-cli` (llama.cpp's completion tool, optionally with its
  `--prompt-cache`). With `--energy` it measures GPU joules per hop through
  `reflex-energy`. `reflex generate`'s `REFLEX_GENERATE_OK` line gains `kv_io_ms`, the
  cache file's read or write time; the `--json` schema is now 1.2.0.
  - On a T4 with Qwen3-0.6B, a cold hop takes 1,048 ms, against 1,466 ms for
    llama.cpp's, and 230 ms warm.
  - With a ~1,490-token prefix, resuming a cold hop from its 342 MB cache file
    (1,499 ms) is slower than recomputing the prefix (1,322 ms). That's mostly page
    faults in reading the file into fresh memory.

  See [docs/design/agent-relay.md](docs/design/agent-relay.md#multi-process-chains-phase-d).
- **System-1 scoring over a resumed prefix.** A `reflex stdio`/`uds` request with
  `candidates` can now set `use_pinned` or `import_kv`. It scores the candidates after
  the cached prefix plus `prompt`, and gives the token ids and `chosen` of one request
  over the joined text, with scores equal up to float summation order. This works on
  dense, MoE, MLA and the Qwen3.5 hybrid (single-token candidates); Gemma 4 and
  Granite 4.0-H return `unsupported_architecture` with a prefix. `pin_prefix` and
  `export_kv` stay generate-only.
  - Every `candidates` response gains `chosen`, the index of the most probable
    candidate.
  - With `"timings": true`, every response reports the pinned slot after the request:
    `pinned_seq_len` and `pinned_bytes`. Without it, responses are unchanged.
  - `agent_relay_bench.py --system1 <labels>` adds a routing request to every turn.
    On a T4 with Qwen3-0.6B and a ~1,490-token shared prompt, routing against the
    pinned prefix takes 45 ms, against 285 ms over the whole prompt.

  See [docs/design/agent-relay.md](docs/design/agent-relay.md#system-1-over-a-resumed-prefix-phase-e).
- **Faster cache-file imports, and opt-in f16 cache files.**
  - `--import-kv`/`import_kv` reads only the file's header up front, and checks the
    file's length against it. The values are then read straight into the engine's
    pinned staging buffers on their way to the GPU. Before, they went through two
    fresh host buffers, and faulting those in made a cold process's read of a 342 MB
    file take 284 ms.
  - On a T4 with Qwen3-0.6B and a ~1,490-token prefix, a cold hop resumed from the
    file takes 1,203 ms instead of 1,530 ms, now faster than recomputing the prefix
    (1,336 ms). A warm `reflex stdio` request reusing the file reaches its first
    token in 91 ms instead of 223 ms. File formats, existing files and imported state
    are unchanged.
  - `kv_io_ms` now includes the upload into the cache, since the read and the upload
    overlap.
  - `--export-kv-dtype f16` (CLI) and `"export_kv_dtype": "f16"` (IPC) write the file
    in f16, half the size: format versions 5, 6 and 7. The Qwen3.5 hybrid keeps its
    recurrent state f32. Resuming from an f16 file isn't exact. On the test prompts it
    moved the next token's logits by at most 7e-4 (relative L2) and gave the same
    tokens. Exports stay f32 unless asked, and every format still imports.
  - With f16 files the cold hop above takes 1,182 ms. `agent_relay_bench.py` gains
    `--kv-dtype`.

  See [docs/design/agent-relay.md](docs/design/agent-relay.md#faster-cache-files-phase-f)
  and [docs/reference.md](docs/reference.md#kv-cache-files).
- **`response_format` in the OpenAI sidecar.** `/v1/chat/completions` accepts
  `{"type": "json_schema", "json_schema": {"schema": ...}}` and maps it onto the
  engine's JSON Schema constrained decoding, streaming or not. The schema passes
  through as raw JSON, so properties come out in schema order. `{"type":
  "json_object"}` allows any JSON object, and `{"type": "text"}` is unchanged.
  - A schema outside the engine's subset is a `400` `invalid_request_error`
    (`invalid_input`, naming the keyword). `max_tokens` running out before the value
    is complete is a `400` with code `grammar_dead_end`.
  - An engine error after a stream has started now arrives as an OpenAI-style
    `{"error": ...}` event before `data: [DONE]`, as a timeout already did, instead of
    a `finish_reason: "stop"` chunk.
  - HTTP requests still can't reach the pinned slot or cache files: `pin_prefix`,
    `use_pinned`, `import_kv` and `export_kv` are ignored, like any unknown field.

  See [sidecar/openai-adapter/README.md](sidecar/openai-adapter/README.md#response_format-json-schema-output).
- **Composing agents: docs and examples.** No engine change. A "Composing agents"
  section in [docs/design/agent-relay.md](docs/design/agent-relay.md#composing-agents-phase-h)
  sets out what belongs in Reflex and what belongs in the orchestrator, and which hop
  shapes IPC and the CLI support.
  - `examples/relay_envelope.schema.json` (JSON Schema draft 2020-12): an
    orchestrator-side envelope for a hop's decision, carry text, cache file and
    timings, built from the engine's existing output.
  - `examples/relay_orchestrator.py` (Python 3.8+, stdlib only): a router and a worker
    as warm `reflex stdio` processes with pinned prefixes (a System-1 decision against
    the pin, a JSON Schema handoff and a cache-file export), then a cold
    `reflex generate --import-kv` sub-agent.
  - `examples/relay_chain.sh` (POSIX sh): the same chain with a fresh process per hop.

## 2026-10-15

- **Tiled prefill attention for dense, MoE, Kolibri-1 and Qwen3.5.** Batched prefills
  of 64 rows or more run the tiled kernel Gemma 4 introduced, now a template with
  modules for head dims 64 and 128 as well. GQA groups that don't divide its block
  (Qwen2.5's 6 and 7) are padded. Each module loads on the first long prefill, so short
  prompts and cold start don't pay for it. On a T4, Qwen3-0.6B prefills 2K tokens 5.4x
  faster and 8K tokens 10.5x faster; TinyLlama 5.3x at 2K; Mistral 7B 2.8x at 2K.
  `REFLEX_ATTN_TILED=0` turns it off. See
  [docs/design/tiled-prefill-attention.md](docs/design/tiled-prefill-attention.md).
- **Granite 4.0-H (`granitehybrid`), the first Mamba-2 hybrid.** Mamba-2 SSM layers
  plus a few attention layers (no RoPE), each followed by a SwiGLU FFN, with Granite's
  muP scales (embedding, residual, attention, logits).
  - **New kernels** in `kernels_cuda/mamba2.cu`: causal conv with bias, the selective
    scan (per-head `A`, softplus `dt` with bias, `B`/`C` over groups), and the D skip
    with the `z` gate and grouped RMSNorm. They are checked against a host reference,
    `reference/mamba2_reference.rs`. On SSM layers the projections stay batched over
    the prompt and the recurrence walks the rows inside each kernel; there is no
    chunked parallel scan.
  - **Tokenizer:** new pre type `dbrx` (Granite 4.0's): the Llama-3 split without
    `ignore_merges`, matching `llama-tokenize` 13/13.
  - **Rejected at load, with the key or tensor named:** MoE Granite 4.0-H (Tiny,
    Small), RoPE on the attention layers, attention/FFN biases, SSM layers without a
    norm or conv bias, and SSM shapes the kernels don't cover. `mamba`, `mamba2`,
    `jamba`, `falcon-h1`, `nemotron_h` and `nemotron_h_moe` are rejected by name;
    before, Jamba and Nemotron-H MoE reached the generic MoE path and failed on a
    missing tensor. `--lora` and `--export-kv`/`--import-kv` are rejected for this
    architecture. `system1` takes single-token candidates only.
  - **Verified** against llama.cpp `79e2e74` on an RTX 4000 Ada:
    - The synthetic fixture matches 7/7 prompts × 20 greedy tokens (f32 and f16).
    - With f32 activations, Granite 4.0-H 350M and 1B `Q8_0` match 5/6 each. The one
      divergence per model is a near-tie (margin 0.002 and 0.033) where Reflex picks
      llama.cpp's own runner-up.
    - `reflex check` is unchanged on all 48 existing rows. The GPU nightly passes 65,
      fails 0, skips 3.
  - **Performance:** cold start on the 350M is 0.76 s p50 vs `llama-simple`'s 0.83 s.
    Per-token decode is flat from 57 to 8,191 tokens of context.

  Design, survey and results: [docs/design/mamba2-hybrid.md](docs/design/mamba2-hybrid.md).

## 2026-10-14

- **Runpod Hub test job no longer depends on one data center's RTX A4500s**
  (`v0.2.8-runpod-hub`). The listing had stayed on `v0.2.4-runpod-hub`: v0.2.6's build
  failed because its release targeted a commit SHA, which the Hub tries to clone as a
  branch, and v0.2.7's test job found no RTX A4500, RTX 2000 Ada or RTX 4000 Ada capacity
  (all three in EU-RO-1 only). `tests.json` now pins an RTX A5000, and `hub.json` adds
  `AMPERE_24` after `AMPERE_16` and allows CUDA 13.2 hosts. Image, handler, tests and
  baked model are unchanged. Details: `.runpod/README.md`, "Update, 2026-10-14".
- **Runpod Hub test job pinned to an RTX 4090** (`v0.2.9-runpod-hub`). v0.2.8's test job
  found no RTX A5000 either, and showed that the test's fallbacks are the pinned GPU's own
  pool (RTX 3090, L4), not `hub.json`'s `gpuIds`. The RTX 4090 is what most of Runpod's own
  Hub workers pin; the test config now also lists `allowedCudaVersions`, and `hub.json`
  adds `ADA_24` as a last-resort pool.
- **GPU nightly: Kolibri-1 fixture tests skip when their fixtures are absent.**
  `scripts/gpu_nightly_tests.sh` had no rule for the nine Kolibri-1 tests, so they ran
  against the dense model and failed opening `test-data/tiny-kolibri1.gguf` /
  `tiny-kolibri1-f32.gguf`, failing every nightly since 2026-10-08. They now report SKIP
  with the missing path, like the other fixture tests.

## 2026-10-13

- **MLA (`deepseek2`) widened to small real checkpoints: DeepSeek-Coder-V2-Lite,
  GigaChat3.1-10B-A1.8B and GLM-4.7-Flash.**
  - Sigmoid expert gating with the `exp_probs_b` selection bias (DeepSeek-V3's rule:
    select on `sigmoid(logits) + bias`, weight by the unbiased sigmoid). The MLA path
    used to ignore `expert_gating_func`, so a sigmoid file would have routed with softmax
    silently; its tokenizer was rejected first, so no file actually did.
  - llama.cpp's rule for GLM-4.7 files with no gating key (47/48 layers, 154,880 vocab:
    sigmoid).
  - `expert_weights_scale` of 0 is a no-op, as in llama.cpp.
  - Q-LoRA queries (`wq_b · rmsnorm(wq_a · x)`), picked by which tensors the file has.
  - Pre types `gigachat` (GPT-2 split) and `glm4` (Llama-3 split, `ignore_merges`, no BOS).
  - Rejected at load, with the key named: group-limited routing (`expert_group_count > 1`),
    other gating functions, softmax with `exp_probs_b`, attention temperature scaling;
    MTP/NextN and all-MoE files as before. `--lora` stays rejected on MLA.
  - Tests: config parsing of seven real `deepseek2` headers (Coder-V2-Lite current and
    legacy, GigaChat3.1, GLM-4.7-Flash with and without the gating key, Moonlight,
    Kanana-2), rejections, routing vs a host reference, tokenizer goldens, and a new
    synthetic `tiny-mla-v3` fixture (Q-LoRA, sigmoid routing, bias, scale, norm) whose
    every new feature changes llama.cpp's greedy output when dropped. GPU tests: llama.cpp
    golden ids, batched == sequential prefill, f16 vs f32, KV export/import, `system1`.
  - Verified on an A100 80 GB against llama.cpp `79e2e74` with f32 activations:
    - `reflex check` is unchanged on all 45 existing rows, real DeepSeek-V2-Lite and
      Coder-V2-Lite included.
    - Greedy tokens agree up to a near-tie on every prompt. Prompts matching all 20
      tokens: fixture 6/7, V2-Lite 6/7, Coder-V2-Lite 6/7, GigaChat3.1 5/7,
      GLM-4.7-Flash 4/7.
    - At each of the 8 divergences, Reflex picks llama.cpp's own runner-up, 0.005 to
      0.19 logits behind.
  - Real DeepSeek-V2-Lite re-verified with its `deepseek-llm` tokenizer on digit runs,
    punctuation runs, Chinese and mixed scripts.
  - Cold start on GigaChat3.1 `q8_0`: 4.22 s p50 vs `llama-simple`'s 4.40 s (n=30).

  Details: [docs/design/mla-widening.md](docs/design/mla-widening.md).

- **Qwen2 / Qwen2.5 (`qwen2`), including Qwen2.5-Math and Qwen2.5-Coder.** These run on
  the dense path. The new piece is Qwen2's Q/K/V projection biases, added right after
  each projection and before RoPE, as llama.cpp's `build_qkv` does. Batched prefill
  adds them with the existing `add_kernel`, which now broadcasts one row over many.
  Residual adds don't change and give bit-identical results.
  - `--weights f16|f32`, `REFLEX_QUANT_RESIDENT=1`, `--lora`, `--export-kv`/`--import-kv`
    and `system1` all work, through the existing dense code.
  - ChatML, the Math prompt and Coder's FIM and repo tokens (`<|fim_prefix|>`,
    `<|file_sep|>`, ...) tokenize as single special tokens, matching `llama-tokenize`.
  - Rejected at load, with the tensor or key named: YaRN `rope.scaling`,
    `attn_output.bias`, `output.bias`, a fused `attn_qkv`, and a partial or misshapen
    bias set. The output-bias and fused-QKV checks cover every dense architecture.
    `qwen2moe` used to pass the generic MoE check and would have run without its shared
    expert. Any MoE file with shared-expert tensors is now rejected.
  - Tests: config parsing of five real Qwen2.5 headers (0.5B, 1.5B, Math, Coder ×2),
    rejections, and a synthetic `tiny-qwen2` fixture plus LoRA, both converted with
    llama.cpp's own converters, with greedy golden ids from llama.cpp.
  - Verified on an RTX 3090 against llama.cpp `79e2e74` with f32 activations: identical
    greedy tokens (7 prompts × 20 tokens) on the fixture with and without a LoRA, on
    Qwen2.5-0.5B-Instruct (`f32` and `f16`), Qwen2.5-1.5B-Instruct (also
    `REFLEX_QUANT_RESIDENT=1`), Coder-1.5B and Math-1.5B. Stock llama.cpp's 8-bit
    activations flip 3 of 7 on the 0.5B, as on Gemma 4.
  - Cold start on Qwen2.5-0.5B-Instruct: 0.650 s p50 vs `llama-simple`'s 0.700 s (n=30,
    same pod).
  - `reflex check` on every fixture plus Qwen3-0.6B is unchanged, 30/30 rows.

  Details: [docs/design/qwen2.md](docs/design/qwen2.md).

- **`qwen2` tokenizer split now uses llama.cpp's Unicode tables.** It was the last pre
  type still classifying characters with Rust's `is_alphabetic`/`is_numeric`, which
  count Arabic harakat as letters. It now runs llama.cpp's
  `unicode_regex_split_custom_qwen2` port, like every other pre type. Output changes only
  on text whose characters the two disagree on, such as combining marks.
  - Qwen2 golden test vs `llama-tokenize`: 12/13 -> 13/13.
  - Kolibri-1 (`kolibri1`, same split): unchanged against HF `tokenizers`, including two
    new Arabic and Devanagari strings.
  - The `reflex check` prompts tokenize identically on every fixture, and the GPU check
    shows all 30 rows unchanged.

  Details: [docs/design/tokenizer-pre-types.md](docs/design/tokenizer-pre-types.md#resolved-the-original-qwen2-port).

## 2026-10-12

- **Tokenizer pre-types: `tokenizer.ggml.pre` is now read and checked.** Before this,
  every `gpt2`-model GGUF got the Qwen2 split regex. A Llama-3 or DeepSeek file
  therefore loaded and tokenized wrongly, with no error.
  - New ports, each hand-ported from llama.cpp `79e2e74`: `llama-bpe` (Llama 3),
    `deepseek-llm` (DeepSeek LLM, DeepSeek-V2/V2-Lite), `deepseek-v3`, `qwen35`
    (Qwen3.5/3.8: letter runs also take combining marks) and `gpt-2`. They classify
    characters with llama.cpp's own Unicode tables, transcribed into
    `src/tokenizer_unicode.rs`. No new dependency.
  - Unchanged: `qwen2` and `kolibri1`.
  - Any other value fails to load, and the error names it. A missing key keeps the old
    Qwen2 behavior.
  - Each type matches `llama-tokenize` token for token on a 13-string tricky set. The
    tests use tokenizer-only fixtures, made by the new `scripts/gguf_tokenizer_only.py`.
  - `reflex check` on an RTX 3090 (every fixture plus Qwen3-0.6B) is unchanged on 29 of
    30 rows, logit checksums included. The 30th is `tiny-qwen3moe` (`gpt-2`) on the
    long prompt, whose prompt ids now match `llama-tokenize`.
  - Known gap, unchanged on purpose: the `qwen2` split still disagrees with llama.cpp on
    Arabic harakat.

  Details: [docs/design/tokenizer-pre-types.md](docs/design/tokenizer-pre-types.md).

- **Runpod Hub release `v0.2.6-runpod-hub`.** Rebuilds the Hub worker from current
  `master`, so the classifier gets the bit-identical `rmsnorm_kernel` fix below: faster
  prefill, same outputs. The image, the baked Qwen3-0.6B-Q8_0 model, `hub.json` and
  `handler.py` are unchanged. Gemma 4 support is in the engine but not reachable from
  the Hub worker, which serves only its baked model.

- **Long-prompt prefill 4-7x faster on Gemma 4, and faster prefill everywhere.** Two
  causes, measured with a new per-phase profiler (`REFLEX_PREFILL_PROFILE=1`):
  - `rmsnorm_kernel` recomputed each row's sum of squares in every thread (rows x
    hidden^2 work). It now runs one block per row with the same summation order, so
    outputs are bit-identical for every architecture (`reflex check` unchanged on 12
    configurations). This was 40-68% of Gemma 4 prefill. Qwen3-0.6B warm prefill at 449
    tokens: 67.5 -> 53.9 ms.
  - Gemma 4 prefill attention now uses a tiled f32 kernel that reuses each K/V tile
    across a block of query rows and the query heads of a GQA group. Decode and other
    architectures are unchanged.

  On an RTX A40, 12B prefill at 3,713 tokens: 13.4 s -> 1.87 s. 31B
  (`REFLEX_QUANT_RESIDENT=1`): 33.1 s -> 4.66 s. First token on a 2,641-token prompt:
  12B 10.1 s -> 3.4 s, 31B 23.0 s -> 6.1 s. That is faster than llama.cpp computing in
  f32 and 1.3-1.6x behind stock llama.cpp. Greedy tokens against the fully-f32 llama.cpp
  reference are unchanged: 12B 12/12, 31B 5/6 (the known 982-token near-tie). PTX builds
  without a warm driver cache JIT the new module once, ~340 ms; cubin and fatbin builds,
  including the Docker image, don't. Details:
  [Long-prompt prefill](docs/design/gemma4.md#long-prompt-prefill-2026-10-12).

- **Gemma 4 on 48 GB cards: quantized-resident `Q8_0`** (`REFLEX_QUANT_RESIDENT=1`,
  opt-in). Gemma 4's `Q8_0` layer weights and tied LM head stay quantized on the GPU,
  read by new `gemv_q8_0` kernels. Above 16 prompt rows they are dequantized into a
  reused f16 scratch for cuBLAS. Other architectures' outputs are unchanged (`reflex
  check` identical, checksums included). Verified on an RTX A40 against llama.cpp
  computing fully in f32:
  - **31B-it `Q8_0` now runs on a 48 GB card:** 37.9 GB peak at 2,641 tokens. Greedy
    tokens identical on 5/6 prompts up to 2,641 tokens. The sixth, a 982-token
    repeated-sentence prompt, starts with a near-tie that the llama.cpp reference resolves
    differently on an A40 and an A100; Reflex agrees with the A100 run. `system1`
    within 0.022.
  - **12B-it:** 11.1 GB resident instead of 20.9. 18/18 prompt and mode combinations
    identical to the reference, and `system1` scores closer to it than with f16 weights.
  - **Faster cold start, 12B:** model load 1.0 s instead of 1.5-2.3 s (the f16 path's 328
    per-tensor allocations cost 0.6-1.5 s). First token 1.99 s instead of 2.49 s wall
    (llama.cpp 3.85 s). Cold `system1` 1.38 s instead of 1.81 s. The 31B's first token
    is 3.6 s against llama.cpp's 8.4 s.
  - **Decode 22% faster** on the 12B (44.8 vs 57.7 ms/token).
  - **Slower:** warm prefill, by ~65 ms on long prompts, and warm 1-token `system1`
    (109 vs 73 ms). The fused kernel can't match f16 cuBLAS on tensor cores above ~4
    prompt rows.

  Details: [quantized-resident weights](docs/design/quantized-resident-weights.md#q8_0-on-gemma-4-results-2026-10-12).

## 2026-10-11

- **Gemma 4 31B and Qwen3.8-27B verified** on an A100 80 GB, both `Q8_0` with f16
  weights. No code changed.
  - **Gemma 4 31B-it:** greedy tokens identical to llama.cpp running fully in f32 on 5/5
    prompts (up to 1,262 tokens, past the 1,024-token window), and on a natural
    2,641-token prompt. `system1` scores within 0.03. For long prompts the reference
    needs an f32 KV cache with flash attention off on top of f32 activations: llama.cpp's
    f16 flash attention moves the 31B's logits by up to ~0.5 there.
  - **Qwen3.8-27B** (`qwen35`): 4/4 prompts identical to llama.cpp with f32 activations,
    including a 981-token prompt. The prompt ids match too, so the GGUF's
    `add_bos_token = false` is honored. Single-token `system1` scores within 0.007
    (the hybrid path still accepts only single-token labels).

  Gemma 4 details: [docs/design/gemma4.md](docs/design/gemma4.md#31b-results-2026-10-11).

## 2026-10-09

- **Gemma 4 support** (`gemma4`, dense, text only; verified on the 12B, the 31B is
  not run yet: its header parses, but it needs an 80 GB card in f16; verified
  2026-10-11, above). The new path is `src/model/gemma4.rs` with its own kernels (`kernels_cuda/gemma4.cu`), plus a third
  tokenizer mode (`tokenizer.ggml.model = "gemma4"`). Other architectures' code paths and
  kernels are unchanged; on an A40 their `reflex check` output is identical to before,
  checksums included. Verified on an A40:
  - **Synthetic fixture:** greedy tokens identical to llama.cpp, 20/20 on 4 prompts
    (one 982 tokens long against an 8-token window).
  - **Real Gemma 4 12B-it `Q8_0`:** 4/4 prompts identical to llama.cpp running with
    f32 activations, including a 1,122-token prompt past the 1,024-token window. Stock
    llama.cpp's 8-bit activation quantization moves some logits by up to ~0.8, so it
    disagrees on near-ties.
  - **`system1`:** multi-token labels supported, scores within 0.06 of llama.cpp's.
  - **Tokenizer:** identical ids on 25 strings to HF `tokenizers` and `llama-tokenize`.

  First token after 2.41 s vs llama.cpp's 3.49 s (`llama-simple`, file in page cache,
  n=5). As a classifier, about 74 ms per decision warm (`reflex stdio`) and about 2.1 s
  as a cold process. Details: [docs/design/gemma4.md](docs/design/gemma4.md).

- **Runpod Hub worker repackaged as a classifier listing** (`v0.2.5-runpod-hub`). The
  queue-based Hub worker now bakes **Qwen3-0.6B-Q8_0** (was Q4_K_M; `.runpod/Dockerfile`)
  and leads with `POST /v1/classify`: a job whose `input` has `labels` scores every label
  as a continuation of `prompt` in one prefill and returns per-label probabilities and
  entropy, generating nothing. Chat completions stay available on the same endpoint.
  `.runpod/README.md` gains a caller contract -- send `prompt` not `messages` for Qwen3's
  ` thinking` bias, use one-token labels with a leading space, and treat `probability` as
  relative to the label set so thresholds/escalation stay the caller's job -- and
  `tests.json` adds a 4-label routing smoke test. No engine change; `serverless/runpod/`
  (load-balancing) is untouched.

## 2026-10-07

- **Lazy expert upload for Kolibri-1** (opt-in `REFLEX_LAZY_EXPERTS=1`, with
  `REFLEX_QUANT_RESIDENT=1`): the load reserves the experts' GPU space but copies none
  of them; each layer uploads its routed experts the first time it needs them, so a
  short prompt reads a fraction of the 47.5 GB file before its first token. Output is
  identical to the eager load (tokens and System1 scores, verified on a T4 on the tiny
  Kolibri fixture, and on the real model). On an RTX A6000, a 5-token prompt's cold first
  token drops from 8.55 s to **2.33 s** (llama.cpp: 16.8 s on the same host) after
  uploading 4.9% of the experts; warm, 3.51 -> 1.21 s; a cold System1 decision 8.4 ->
  2.9 s. A 627-token prompt touches half the experts and gains only ~1 s, since each
  layer waits for its uploads. Details in
  [docs/design/kolibri.md](docs/design/kolibri.md), Phase 4 step 2.
- **`POST /v1/classify` in the OpenAI sidecar**: Reflex's System1 candidate scoring over
  HTTP. A request names a `prompt` (or chat `messages`) and up to 64 `labels`; the engine
  runs one prefill, scores each label as a continuation, and the response carries each
  label's probability within the set, the most probable label and the entropy. Nothing
  is generated. On a T4 with Qwen3-0.6B a warm request takes ~10 ms and returns exactly
  the probabilities `reflex system1` gives. A label's score sums raw token logits, so
  labels of different token lengths aren't comparable: the response carries a `warning`
  when they differ. On Kolibri-1 (RTX A6000), a warm decision takes ~0.2 s and a cold
  process reaches its result in ~8.6 s; bare zero-shot prompts got 2 of 3 test cases
  wrong there, so check a classifier on real data first
  ([docs/design/kolibri.md](docs/design/kolibri.md)). Not an OpenAI endpoint (OpenAI has no classification API). The Runpod
  Hub worker sends any job whose input has `labels` to it, and `tests.json` gains a
  classification smoke test.
- **Kolibri-1 support** (Aleph Alpha's 78B MoE, 3.46B active, 384 experts, top-6,
  released 2026-10-03 under Apache 2.0). A third layer variant on the dense/MoE path:
  sigmoid routing with a selection-only bias, sandwich norms, sliding-window RoPE layers
  alternating with full-attention NoPE layers. Its experts stay quantized on the GPU with
  `REFLEX_QUANT_RESIDENT=1` (new `Q6_K` kernels alongside `Q4_K`), so the 47.5 GB
  `Q4_K_M` runs in 45.4 GiB. On an A100 it matches llama.cpp (`836d571` plus the community
  `kolibri1` patch; mainline has no support yet) 20/20 tokens on five German and English
  prompts up to 627 tokens, once llama.cpp computes with f32 activations: its default Q8
  activations flip near-tie expert choices. Details in
  [docs/design/kolibri.md](docs/design/kolibri.md).
- **Parallel model load.** `WeightLoadPipeline` stages tensors in 64 MB chunks filled by
  several threads (`REFLEX_LOAD_THREADS`, default min(cores, 8)), with prefetch reader
  threads (`REFLEX_LOAD_READERS`, default 16) faulting pages in up to 1 GB ahead; `F32`
  norms no longer stall the pipeline with a synchronous copy. Kolibri-1 on an RTX A6000
  (4.3 GB/s disk), first token: cold 71.6 s -> 19.7 s (llama.cpp 41.3 s), warm 7.6 s ->
  3.7 s (llama.cpp 7.1 s), measured before the prefetch readers. On a second A6000 host
  (6.1 GB/s disk) with the readers: cold 8.5 s (9.25 s with readers off; llama.cpp
  17.2 s), warm 3.8-4.0 s (llama.cpp 7.6 s). T4 warm load: Mistral 7B
  quantized-resident 2306 -> 1738 ms, Qwen3-4B 1098 -> 805 ms; Qwen3-0.6B unchanged.
- **Runpod Hub worker: streaming and an `ADAPTER_ARGS` deploy field** (`v0.2.3-runpod-hub`).
  `.runpod/handler.py` is now a generator handler: a job with `"stream": true` yields each
  `chat.completion.chunk` (readable live from `/stream/{job_id}`), and any other job yields
  its one `chat.completion`. **Breaking for existing callers:** a non-streaming job's
  `output` is now a one-element list (the `worker-vllm` convention), not the bare object.
  Adapter errors fail the job with the adapter's error JSON. `hub.json` declares
  `ADAPTER_ARGS` as an advanced deploy field, allows the whole `AMPERE_16` pool (Ada cards
  included, now that the fatbin is verified on Ada), and names the baked-in model. The
  release also moves the Hub image to the fatbin kernels and slim runtime.

## 2026-10-06

- **Fixed: Mistral 7B produced garbled text.** The SentencePiece tokenizer merged
  characters only by `tokenizer.ggml.merges`, which many llama-family GGUFs (including
  TheBloke's Mistral-7B-v0.1) don't contain, so those prompts were encoded one character
  per token. It now merges by token score, as llama.cpp does. On a T4, Reflex's prompt
  token ids match llama.cpp's `llama-tokenize` on 12 of 12 test strings (spaces, digits,
  punctuation, code, accents, CJK, emoji) for both Mistral-7B and TinyLlama, and
  Mistral's generated text matches `llama-simple`. TinyLlama and Qwen3 output is
  byte-identical to before.

## 2026-10-02

- **Runpod comparison against llama.cpp, finished**: two more llama.cpp cold runs bring
  both engines to n=5 on the L4 (engine load 0.49 s vs. 0.95 s, ~1.9x; end-to-end still
  inconclusive), and the `--no-warmup` variant (n=3, 0.92 s) shows llama-server's warmup
  is not where its startup time goes. See
  [docs/runpod-llamacpp-comparison.md](docs/runpod-llamacpp-comparison.md#results).
- **Faster model load: the token embedding stays in the file mapping.** The table used to
  be copied out of the mmap at load (127.6 MB for Qwen3-0.6B Q4_K_M), which took ~103 ms of
  ~231 ms model load on a T4. Rows are now read straight from the mapping when used. On a
  T4, `system1` cold start went from ~452 to ~421 ms and `generate` from ~700 to ~674 ms
  (p50, n=10, two interleaved rounds); output is unchanged. The model keeps the GGUF file
  mapped for its lifetime.
- **f16 weight storage, the new default**: matrix weights are dequantized straight to
  `f16` on the GPU (every dequant kernel has an `_f16` twin), halving weight VRAM to about
  2 bytes per parameter. Decode uses f16-weight GEMV kernels with f32 accumulation;
  prefill uses `cublasGemmEx` (f16 inputs, f32 compute), with an activation cast that
  saturates at 65504 instead of overflowing and reports what it clamped. `--weights f32`
  / `REFLEX_WEIGHTS=f32` keeps the exact reference mode, and `reflex check` defaults to
  it. LoRA merges stay in f32. Result lines gain `weights_dtype` (`--json` schema
  `1.1.0`). With f16 weights, cuBLAS's first-call cost is paid on the load's worker
  thread. Verified on a T4 (2026-10-04): f16 and f32 give the same tokens on every tested
  model, VRAM halves, decode is 36-42% faster, and cold start is 7% (`system1`) and 8%
  (`generate`) faster than f32. Details in
  [docs/benchmarks.md](docs/benchmarks.md#f16-weight-storage).

## 2026-10-01

- **External energy measurement**: `reflex-energy` (built with `--features nvml`) measures
  a command's whole-process GPU energy from outside it, with the GPU's idle draw
  subtracted. A cold `reflex system1` on a T4 costs ~5.4 J net of ~27.7 J gross; most of
  the gross figure is the GPU idling. `scripts/bench_cold_energy.sh` runs it n times, and
  the GPU nightly workflow reports it.
- **Runpod comparison against llama.cpp, prepared**: an image for llama.cpp's official
  CUDA server on the same Runpod load-balancing setup, and a procedure in
  [docs/runpod-llamacpp-comparison.md](docs/runpod-llamacpp-comparison.md). First run on
  an L4: engine load 0.49 s (Reflex) vs. 0.95 s (llama.cpp); end-to-end wall clock is
  dominated by the platform and inconclusive (Reflex n=5, llama.cpp n=3).
- **`.gitattributes` forces LF for `*.sh`**: a Windows checkout (`core.autocrlf=true`)
  put CRLF into `serverless/runpod-llamacpp/start.sh`, and the built image failed with
  `exec /start.sh failed: No such file or directory`.

## 2026-09-30

- **Online-softmax attention kernels** (`src/kernels_cuda/attention_online.cu`) replace the
  four original attention kernels: 8x faster prefill and decode at 8K context on a T4,
  context limit raised from 11,264 to 65,535 positions. The old kernels stay selectable
  with `REFLEX_ATTN_KERNEL=legacy`.
- **Typed errors**: every engine error has a stable category (`src/error.rs`), exposed as
  IPC `error_kind`, the C FFI's `reflex_last_error_code()`, and HTTP status codes in the
  sidecar. Message text unchanged.
- **`src/model.rs` split** into `src/model/` modules (config, loading, kernels, dense,
  hybrid, mla), move-only.
- **Slim, consolidated Docker images**: one multi-target root `Dockerfile`; runtime is
  CUDA `base` + cuBLAS (images 51-66% smaller); kernels default to a multi-arch fatbin,
  which removed ~0.8 s of driver JIT from every fresh-container start.
- **Multi-arch fatbin** verified on a T4; diagnostics now fail clearly on a GPU older than
  every compiled architecture.
- **Public contributor doc** `docs/DEVELOPMENT.md`.
- **GPU nightly CI** workflow with a cold-start regression gate (off until its secrets
  are configured; see [docs/gpu-ci.md](docs/gpu-ci.md)).
- **Request limits in the OpenAI sidecar** (`max_tokens` cap, prompt size, queue depth,
  timeout), and a clean `context length exceeded` error instead of a CUDA launch failure.
- **Llama / Mistral** (dense GQA) and **`qwen35moe`** architectures.

## Model architectures

All four steps below are **done** and verified on real hardware:

1. **Dense Qwen3** — the best-understood, most well-documented architecture to build
   against first; proves the AOT-compilation + cold-start-benchmark harness works at all.
2. **Qwen3-MoE**
3. **Qwen3.5 hybrid Gated DeltaNet mixer**
4. **DeepSeek-V2/V3 MLA** — deliberately last; a genuinely different (compressed
   latent-KV) caching strategy, not an incremental GQA extension.

### Later additions (breadth on demand, not a roadmap axis)

The four families above cover the architectural *mechanisms* — dense attention, sparse
MoE, hybrid linear attention, latent-KV attention. Further coverage is added only when a
concrete target model pulls it, never chased for its own sake: broad model support is
llama.cpp's axis, not this engine's (see [Non-goals](README.md#non-goals) — the same reason
throughput and serving features are out of scope). The two additions so far, in order:

- **Llama / Mistral (dense GQA)** — *implemented and verified byte-exact against llama.cpp
  on TinyLlama-1.1B*. The cheapest addition, because GQA, RMSNorm,
  SwiGLU, the SentencePiece tokenizer, and the device-resident dense/MoE forward path
  already exist generically; the real delta is the RoPE convention
  (`LLAMA_ROPE_TYPE_NORM`'s consecutive-pair rotation, reusing the `rope_norm_kernel` the
  MLA path already ships) plus an explicit architecture whitelist. Mistral-7B GGUFs report
  `general.architecture = "llama"` (llama.cpp has no bare `mistral` arch), so this path
  covers Mistral-7B too — though running a 7B model needs >15GB because every weight is
  held `f32`.
- **`qwen35moe`** — *implemented and verified byte-exact against llama.cpp* on a
  converted `qwen3.5-moe-tiny-random` checkpoint (128 experts / top-10, sigmoid-gated
  shared expert). The Qwen3.5 hybrid trunk is reused unchanged; every
  layer's FFN becomes routed MoE (renormalized top-k, reusing the existing per-expert and
  grouped-GEMM dispatch) plus a shared expert scaled per token by
  `sigmoid(ffn_gate_inp_shexp · x)` — the one convention that differs from MLA's
  always-on shared expert. No new kernels. Real Qwen3.5-35B-A3B needs far more VRAM than
  the verification GPU because every weight is held `f32`.

Each addition counts as done only after an independent-implementation comparison
(llama.cpp, or a host CPU reference) on real generated tokens.

## Productization phases

Once the model-architecture MVP proves the engine handles the target model
families at all, the next axis is making the *cold-start path itself* faster and
adoptable — without ever crossing into building a serving platform. The framing: let
vLLM win the warm-throughput race; Reflex wins by being the fastest way to turn
cold compute into one output token, then getting out of the way. All four phases below
are **done**:

- **Phase 1** — Single-shot CLI: process launch -> one forward pass -> exit.
- **Phase 2 — Fast IO**: weights upload to the GPU once (not re-uploaded per kernel
  call), device-resident activations through a whole layer, and on-GPU dequant kernels
  for the block types this project's fixtures use for the bulk of weight bytes. This is
  the work behind the llama.cpp benchmark result above.
- **Phase 3 — State I/O**: `--export-kv <file>`/`--import-kv <file>` for raw K/V-cache
  dump/load/resume, across all four architectures. Reflex stays ignorant of *where*
  that file lives (NVMe, an S3-backed FUSE mount, tmpfs) — that's the orchestrator's
  job, not this engine's.
- **Phase 4 — Embeddability**: `--lora <path>` (load-time adapter application, no
  runtime hot-swap multiplexer) and a Rust C-FFI surface (`src/ffi.rs`,
  `include/reflex_engine.h`) so an external orchestrator can embed Reflex directly
  instead of `exec`-ing a binary.
