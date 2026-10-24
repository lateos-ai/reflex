# Reflex on the Runpod Hub (queue-based worker)

This directory exists for exactly one reason: **Runpod's Hub publishing pipeline** — the
public, browsable catalog other people one-click-deploy from (e.g.
`runpod-workers/worker-vllm`) — only supports **queue-based** workers (a Python handler using
`runpod.serverless.start()`). It does not currently support publishing a **load-balancing**
endpoint to the Hub at all.

[`serverless/runpod/`](../serverless/runpod/README.md) is the actual recommended deployment
path for Reflex on Runpod: it's a real, measured, load-balancing-endpoint deployment with
~3.4x faster cold start than Runpod's own official `worker-vllm` (see
[`docs/serverless-cost-comparison.md`](../docs/serverless-cost-comparison.md)), and it needs
zero Python glue code. **If you're deploying manually rather than discovering Reflex via the
Hub, use `serverless/runpod/` instead — it's the leaner path.**

This directory is a second, parallel, purely additive build that exists solely so Reflex can
appear in the Hub's catalog. It does not replace or modify `serverless/runpod/` in any way.

## What `handler.py` actually is

A translation shim, not a reimplementation. On container start it spawns the exact same,
unmodified `reflex-openai-adapter` binary `serverless/runpod/` uses (over an internal-only
loopback port — queue-based endpoints don't proxy external HTTP to a configured port the way
load-balancing endpoints do), waits for its existing three-state `/healthz` check to report
ready, and then forwards each Runpod job's `input` straight to that process's existing
`POST /v1/chat/completions`, unmodified. No IPC/HTTP protocol logic is reimplemented in
Python — see [`docs/DEVELOPMENT.md`](../docs/DEVELOPMENT.md)'s non-goals: the core engine never grows a network socket or
an internal queue/scheduler, and this handler doesn't change that. The queue here is Runpod's
own platform queue, external to the engine, exactly like `serverless/runpod/`'s load balancer
is also external to the engine.

## Job input and output

The job `input` is an OpenAI chat-completions request body (`messages`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `stream`), passed to the adapter as is.

**Classification**: a job whose `input` has `labels` goes to the adapter's
`POST /v1/classify` instead (see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md#post-v1classify)):
one engine pass scores each label as a continuation of `prompt` (or of `messages`,
chat-templated) and returns their probabilities. Its output is a one-element list
holding that `classification` object, like a non-streaming chat job:

```json
{"input": {"prompt": "Review: The battery died after two days. Sentiment:",
           "labels": [" positive", " negative"]}}
```

**Embeddings**: a job whose `input` has `input` and no `messages` goes to the adapter's
`POST /v1/embeddings` (see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md#post-v1embeddings)).
It needs an embedding model: pick Nemotron-3.5-Embed-1B in the Model dropdown. That
model serves only embedding jobs; chat and classify jobs on it return an error, and
embedding jobs on the other models do too. The output is a one-element list holding the
OpenAI `list` object:

```json
{"input": {"input": ["What is NVIDIA NeMo Framework?"], "input_type": "query"}}
```

**Retrieval (RAG)**: with the deploy form's RAG field set to `on`, a chat job with a `rag`
object is answered from your documents, and its output carries the passages used. See
[Optional RAG](#optional-rag).

The worker bakes **Qwen3-0.6B-Q8_0** (not the Q4_K_M the `runpod-lb` image uses): the
listing leads with classification, and on a sub-3B model Q8_0 preserves the fine-grained
logits the label softmax reads where a 4-bit quant can move a near-tie. No generation
happens, so the extra bandwidth is paid once at load, not per decoded token.

## Choosing the model

The deploy form's **Model** dropdown (`MODEL`) picks the GGUF. The default is the baked
Qwen3-0.6B Q8_0, which needs no download. Every other choice is downloaded from Hugging
Face when a worker starts, before it takes jobs:

| Model | Architecture | Download | GPU |
|---|---|---|---|
| Qwen3-0.6B Q8_0 (default, baked in) | `qwen3` | none | 16 GB |
| Qwen3-1.7B / 4B / 8B Q4_K_M | `qwen3` | 1.1 / 2.5 / 5.0 GB | 16 GB |
| Qwen2.5-1.5B / 7B Instruct Q4_K_M | `qwen2` | 1.1 / 4.7 GB | 16 GB |
| TinyLlama-1.1B-Chat Q4_K_M | `llama` | 0.7 GB | 16 GB |
| Mistral-7B-Instruct-v0.2 Q4_K_M | `llama` | 4.4 GB | 16 GB |
| Qwen3.5-0.8B Q4_K_M | `qwen35` (hybrid Gated DeltaNet) | 0.5 GB | 16 GB |
| Granite 4.0-H 350M / 1B Q8_0 | `granitehybrid` (Mamba-2 hybrid) | 0.4 / 1.6 GB | 16 GB |
| Gemma 4 12B-it Q8_0 | `gemma4` | 12.7 GB | 24 GB |
| Nemotron-3.5-Embed-1B Q8_0 (embeddings only) | `mistral3` (encoder) | 1.2 GB | 16 GB |

Checked 2026-10-23 on a Runpod A40 with this branch's engine, adapter and `handler.py`:
every model downloads, loads and answers a chat, a streamed chat and a classify job.
Seconds from worker start to ready (download plus load; download speed varies run to
run, Gemma 4 12B took 49 s and 79 s in two runs):

| Model | Ready (s) | Model | Ready (s) |
|---|---|---|---|
| Qwen3-0.6B (baked) | 0.5 | TinyLlama-1.1B | 9.1 |
| Qwen3-1.7B | 3.4 | Mistral-7B | 28.3 |
| Qwen3-4B | 31.8 | Qwen3.5-0.8B | 2.9 |
| Qwen3-8B | 17.0 | Granite 4.0-H 350M | 4.5 |
| Qwen2.5-1.5B | 3.4 | Granite 4.0-H 1B | 10.0 |
| Qwen2.5-7B | 13.2 | Gemma 4 12B | 80.5 |

Nemotron-3.5-Embed-1B was checked the same way on an AWS T4: it downloads (9.1 s), is
ready in 11.4 s, answers embedding jobs (including `dimensions` and base64), and returns
`invalid_input` for chat and classify jobs.

- **Where downloads go.** With a network volume attached, to
  `/runpod-volume/reflex-models`, so only the first worker ever downloads a model. Without
  one, to the container disk (25 GB by default), so every new worker downloads again; the
  log prints progress, and the time adds to that worker's cold start.
- **Gemma 4 12B needs a 24 GB card.** Pick the `AMPERE_24` or `ADA_24` pool when
  deploying it; the default `AMPERE_16` cards don't fit it.
- **Custom GGUF URL** (`GGUF_URL`, advanced) overrides the dropdown with any direct GGUF
  URL, and **Hugging Face Token** (`HF_TOKEN`, advanced) authenticates gated or private
  Hugging Face repos (it's sent only to `huggingface.co`). The model must be one of the
  architectures in the main README's
  [supported-models table](../README.md#supported-models-and-limits) and fit the GPU and
  the disk. Several supported models don't fit this listing's 16-24 GB cards: the
  DeepSeek-V2/V3 MLA family (about 80 GB), Kolibri-1 78B, Qwen3.8-27B and Gemma 4 31B
  (48-80 GB), and real Qwen3-MoE checkpoints (MoE weights aren't kept quantized). Run
  those with the `reflex` CLI or `serverless/runpod/` on a bigger GPU.
- **Classification runs on every model but Nemotron** and the caller contract below applies to all
  of them, but accuracy is the model's: on the sentiment smoke test the default
  Qwen3-0.6B Q8_0 and most picks rate the review negative, while TinyLlama, Granite
  350M and Gemma 4 12B (whose instruction tuning expects its chat format) rate it
  positive. Check a model on your own labels before relying on it. Chat uses each GGUF's own chat template (see
  [`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md)).

### Classifier caller contract

`POST /v1/classify` scores labels, it does not generate one. Three things decide whether
its numbers mean anything, all of them the caller's to get right:

- **Send `prompt`, not `messages`.** With `messages`, labels are scored at the start of
  the assistant's reply, and Qwen3 opens every reply with a ` thinking` block that a label
  is not expected at. On the sentiment fixture, `messages` rated it `positive` at 0.89
  while a plain `prompt` rated the correct `" negative"` at 0.91. End the `prompt` where
  the label naturally follows (`... Sentiment:`).
- **Use one-token labels, and mind the leading space.** A label is the literal text that
  follows the prompt, so it usually wants a leading space (`" negative"`, not
  `"negative"`). `score` sums the label's raw token logits, so a longer label wins on
  length alone; the response carries a `warning` and each label's `tokens` when they
  differ. `A`/`B`/`C` or verified single-token words are safest.
- **`probability` is over the label set, not the vocabulary.** It sums to 1 across the
  labels you sent and says nothing about how likely the true class is outside them --
  the softmax is a fixed, deterministic transform of the scores, so it will happily
  return a confident winner for a prompt that fits none of the labels. The engine
  applies no cutoff; threshold on `probability` and `entropy` (bits; 0 is certain,
  `log2(labels.len())` is uniform) in the caller, and route the uncertain band to
  human review or a larger model. Label-order bias is also uncorrected: calibrate on
  held-out data and do not trust the listed order.

Strategy B from the general classifier playbook -- grammar-constrained JSON output for
multi-label extraction -- is **not** implemented here: `system1_evaluate` is scoring
only, and Reflex has no grammar-constrained sampling yet (it is a queued backlog item).
Use chat completions with a prompt that demands JSON if you need it, and parse
defensively.

The Hub smoke tests (`tests.json`) only assert the job completes; they do not assert the
predicted label. Check accuracy locally with `reflex system1` on real data before
trusting a deployment.

`handler` is a Runpod **generator** handler started with `return_aggregate_stream`, the
same convention `runpod-workers/worker-vllm` uses:

- **`"stream": true`**: each Server-Sent Event the adapter sends is yielded as one
  `chat.completion.chunk` object. Read them live from `/stream/{job_id}`; `/run` +
  `/status` and `/runsync` return the whole list once the job finishes.
- **otherwise**: the single complete `chat.completion` object is yielded once, so the job's
  `output` is a **one-element list** `[{...}]`. Up to `v0.2.2-runpod-hub` this handler
  returned that object bare and forced `stream` to `false`.

An adapter error (`400`/`429`/`503`/`504` before the first token, or a mid-stream error
event such as the request timeout) fails the job, with the adapter's OpenAI-shaped error
JSON as the job's `error`.

`ADAPTER_ARGS` (an advanced field in the Hub's deploy form, empty by default) appends
`reflex-openai-adapter` flags, e.g. `--max-tokens-cap 1024 --default-max-tokens 128`; see
[`sidecar/openai-adapter/README.md`](../sidecar/openai-adapter/README.md)'s flag list and
request limits.

## Optional RAG

Off by default; with `RAG` off, the worker behaves exactly as described above. With
`RAG` set to `on`, the worker starts a second `reflex-openai-adapter` that serves an
embedding model (`RAG_EMBED_MODEL`, Nemotron-3.5-Embed-1B Q8_0 by default) on another
loopback port, next to the chat model. A chat job with a `rag` object then goes through
these steps in `handler.py` and `rag.py`:

1. Embeds the question: the last user message, or `rag.query`.
2. Embeds the job's own `rag.documents`, if any.
3. Scores them and the corpus index by cosine similarity, and keeps the best `top_k`
   passages.
4. Writes those passages, numbered, into the system message, with an instruction to answer
   from them and cite `[n]`.
5. Sends the request to the chat model as usual.

The engine and the adapter are unchanged; retrieval is host-side Python in the worker.

```json
{"input": {
  "messages": [{"role": "user", "content": "Where is the spare key?"}],
  "max_tokens": 64,
  "rag": {"documents": ["The spare key is under the blue pot.", "Lunch is at noon."]}
}}
```

The output is the usual one-element list, with the chat completion plus a `rag` object:

```json
[{"object": "chat.completion", "choices": [...],
  "rag": {"query": "Where is the spare key?",
          "passages": [{"n": 1, "id": "doc0#0", "doc": "doc0", "origin": "request",
                        "score": 0.61, "text": "The spare key is under the blue pot."}]}}]
```

`n` is the number the prompt cites. `origin` is `request` or `corpus`. `score` is the
cosine similarity. With `"stream": true`, the first streamed item is
`{"object": "rag.context", "query": ..., "passages": [...]}`, followed by the usual
`chat.completion.chunk`s.

**`rag` fields** (all optional, `{}` searches the corpus with the last user message):

| Field | Default | Meaning |
|---|---|---|
| `documents` | `[]` | Up to 64 strings or `{"id", "text"}` objects, 1 MB of text in all, chunked and searched for this job only (nothing is stored) |
| `query` | last user message | The text to search with |
| `top_k` | `RAG_TOP_K` (4) | Passages to add, 1-32 |
| `use_corpus` | `true` | `false` searches only `documents` |
| `min_score` | none | Drop passages scoring below this |

`rag` can't be combined with `labels`. A `rag` job on a worker with RAG off fails with an
error saying so. On a RAG worker, plain embedding jobs (`input` and no `messages`) go to
the embedding model, so one endpoint can also serve embeddings for client-side retrieval.

**The corpus.** There are two optional sources, embedded when a worker starts, before it
takes jobs:

- **`RAG_DIR`**, by default `/runpod-volume/rag` on the network volume: every `.txt` and
  `.md` file is one document (its id is its path), and every line of a `.jsonl` file is
  one (`{"text": ..., "id": optional}`). Other files and dot-directories are skipped,
  and the log lists the skipped files. A missing default folder is fine; a folder you
  set yourself must exist.
- **`RAG_CORPUS_URL`**: a `.jsonl`, `.txt` or `.md` file, or a `.tar.gz`/`.tgz`/`.tar`/`.zip`
  of them. It's downloaded once (the same resuming downloader as the models, and
  `HF_TOKEN` applies to huggingface.co). An archive member with an absolute path or `..`
  fails the worker, and links are skipped.

Documents are split into passages of `RAG_CHUNK_CHARS` characters (1200), each repeating
the last `RAG_CHUNK_OVERLAP` characters (200) of the one before, at paragraph, line,
sentence or word breaks. The index (vectors plus passage text) is cached under
`/runpod-volume/reflex-rag/index/<key>/`, or `/models/reflex-rag/` without a volume. The
key covers the embedding model, the chunk settings and every source file's path, size and
modification time, so editing a file or a setting rebuilds the index on the next worker
start, and unchanged corpora load from the cache. The retrieved context is capped at
`RAG_MAX_CONTEXT_CHARS` (24000; set it as an endpoint environment variable): lower-scored
passages are dropped first.

**Costs to plan for:**

- **Start-up.** Indexing is one engine forward pass per passage, so time a large corpus
  on the first start. The cache makes later starts quick only with a network volume; on
  container disk, every new worker re-embeds.
- **Memory.** The embedding model adds a 1.2 GB download and about 2.4 GB of VRAM (f16
  weights) to the chat model's.
- **Context.** Passages go into the prompt, so a small model's context window and the
  adapter's 256 KiB prompt limit bound `top_k` × chunk size.

Answer quality is the chat model's: a 0.6B model follows the passages less reliably than
a 7B one.

## GPU pin: portable PTX, not sm_86 (unlike `serverless/runpod/`)

The same mixed-architecture hazard documented in
[`serverless/runpod/README.md`'s GPU-selection section](../serverless/runpod/README.md)
applies to this pool (`AMPERE_16`: Ampere `sm_86` RTX A4000/A4500 alongside Ada `sm_89` RTX
2000/4000 Ada), but the *fix* is different here. `serverless/runpod/`'s endpoint is a real,
already-deployed resource whose GPU can be pinned directly via `set-endpoint-gpus`. A Hub
listing has no equivalent lever: Runpod's own Hub build/test pipeline schedules the automated
test job wherever it wants, and **real evidence (2026-09-27, below) shows it does not reliably
honor either `hub.json`'s `gpuIds` exclusion list or `tests.json`'s `gpuTypeId` pin.** Since an
`sm_86`-pinned cubin crashes outright on an `sm_89` card (not just slower — a hard panic), this
build first switched to **portable PTX**, JIT-compiled by the driver to whichever GPU is present.
That works anywhere but was later measured to cost about 0.8 s of JIT on every fresh worker (a
new container has no driver JIT cache). It now builds a **multi-arch fatbin**
(`REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90` in `.runpod/Dockerfile`): a native image for
every card in the pool, Ada included, plus PTX that newer GPUs JIT. So it runs whichever card the
Hub picks, with no JIT on the listed ones. Confirmed on real Runpod workers on 2026-10-02 (see
"Fatbin on Ada" below), so `hub.json` now allows the whole `AMPERE_16` pool, Ada cards included.

Since `v0.2.9-runpod-hub`, `hub.json` allows `AMPERE_16`, then `AMPERE_24` (RTX A5000, RTX 3090,
L4), then `ADA_24` (RTX 4090), and `tests.json` pins the test job to an RTX 4090 (see "Update,
2026-10-14" below). The pool order is a preference: deployed endpoints use the cheaper
`AMPERE_16` ($0.58/hr serverless) when it has capacity and the next pool otherwise. Every card
in these pools is `sm_86` or `sm_89`, which the fatbin carries natively.

## Non-goals for this addition

- No changes to `serverless/runpod/`, `sidecar/openai-adapter/`, or any core engine file. Both
  deployment paths remain independently valid.
- No new engine capability, no batching, no protocol redesign.
- Not a promise the Hub submission will be approved on any particular timeline — the
  automated build/test gate (`tests.json`) and Runpod's manual review are their process, not
  ours to control.

## Real deployment findings (2026-09-27, real RTX A4500 on Runpod)

This build has been tested end-to-end against a real, temporary queue-based Runpod endpoint
(created directly via the API for pre-submission verification, not via the Hub — the endpoint
was deleted immediately after this test).

**GHCR image visibility.** Runpod had no container registry credential configured for this
account, so a private image couldn't be pulled. The image (`ghcr.io/lateos-ai/reflex-runpod-hub`)
had to be made public before the test endpoint could start. If a private image is ever needed
here instead, a registry credential must be created in Runpod first (a GitHub PAT with
`read:packages` scope) — there is no way around one of these two options.

**The GPU-arch pin worked exactly as designed.** Both workers that spun up during this test
landed on the pinned RTX A4500 (`sm_86`), confirmed via `list-endpoint-workers`.

**The same autoscaler-overrun bug already documented in
[`serverless/runpod/README.md`](../serverless/runpod/README.md) reproduced here too.** With
`workers.max` explicitly set to `1`, `list-endpoint-workers` showed **2** workers (one
`RUNNING`, one `INITIALIZING`) shortly after the first job completed. This is the same platform
behavior already seen on two prior test endpoints for the load-balancing deployment — a live
Runpod bug, not a configuration mistake on this project's side. Anyone deploying this worker for
real should not assume `workers.max` is a hard ceiling and should monitor actual worker count
independently if cost bounding matters.

### Real measured numbers (queue-based endpoint)

A real cold invocation (`min=0` workers, genuine scale-from-zero) via `runsync`/`get-job-status`:

| Phase | Time | Source |
|---|---|---|
| `delayTime` (Runpod's own queueing + cold-start provisioning) | **~71.3s** | Real job status |
| `executionTime` (the actual chat-completion call, warm) | **~1.2s** | Real job status |

**This is a different profile from the load-balancing deployment's ~42-44s** (see
`serverless/runpod/README.md`) — expected, since a queue-based endpoint's delay/execution split
is a different mechanism from an LB endpoint's direct HTTP proxy, and this hasn't been tuned or
re-measured across multiple runs the way the LB deployment's numbers were. Treat this as a first
real data point, not a final benchmark.

## Real Hub build/test findings (2026-09-27)

After linking the `lateos-ai/reflex` repo in Runpod's Hub console (under the `lateos-ai` org,
not a personal account — the account/org selector in the "Add Repo" dialog defaults to the
personal account, which is easy to miss if the target repo lives in an org), Runpod's own
pipeline built `.runpod/Dockerfile` against the `v0.2.0-runpod-hub` release. **The Docker build
itself succeeded** — the build-time model fetch (see `.runpod/Dockerfile`'s comment) worked
correctly against Runpod's own build infrastructure, not just locally.

**The automated test job failed**, and the failure is the reason this build no longer pins
`sm_86` (see the GPU pin section above): the test worker's own startup log reported `GPU: NVIDIA
RTX 2000 Ada Generation (sm_89)` — one of the exact two SKUs `hub.json`'s `gpuIds` and
`tests.json`'s `gpuTypeId: "NVIDIA RTX A4500"` both tried to exclude/pin away from. The
`sm_86`-compiled `reflex` binary panicked on load with a clear, correct error (`"this binary's
CUDA kernels were compiled for compute capability 8.6 (sm_86), but the detected GPU ... has
compute capability 8.9"`), and the test timed out. This is strong evidence that at least one of
Runpod's Hub-specific GPU-selection fields is not honored by the Hub's own test-scheduling
infrastructure, independent of whatever `set-endpoint-gpus`/`gpu.excludedTypes` behavior a real
deployed endpoint has (which `serverless/runpod/`'s real deployment *did* confirm works
correctly — see that README). Switching to portable PTX resolves this by making the GPU
architecture irrelevant to correctness.

Also fixed from this same first attempt: `hub.json`'s `category` field was set to
`"language-models"`, a value not in the Hub UI's actual set (`Image`/`Video`/`Audio`/
`Language`/`Embedding`, confirmed by inspecting the live "Add Repo" form) — corrected to
`"language"`.

## Fatbin on Ada (2026-10-02)

The Ada exclusions in `hub.json` were removed after these checks on real Runpod serverless
workers, all from the fatbin images:

- This directory's queue image (`ghcr.io/lateos-ai/reflex-runpod-hub:latest`, rebuilt from
  `.runpod/Dockerfile`, 2.37 GB, down from 4.86 GB before the slim runtime) on an endpoint
  allowing the whole `AMPERE_16` pool, with no exclusions: the `tests.json` smoke input completed
  on an RTX A4500 (`sm_86`), both from a cold worker and warm. The scheduler also started a
  second worker on an **RTX 2000 Ada** (`sm_89`) in EUR-IS-1, but it stayed `THROTTLED` (no
  capacity at that host) and never served a job.
- The load-balancing image (root `Dockerfile`, `--target runpod-lb`, same kernel build args)
  served `/v1/chat/completions` on an **NVIDIA L4** (Ada, `sm_89`, the same compute capability
  as the RTX 2000/4000 Ada), engine ready ~0.4 s after container start, same output as on the
  A4500. Neither small `AMPERE_16` Ada SKU had serverless capacity on CUDA 12.x/13.0 hosts
  during the run, which is why the L4 stood in.

So the `sm_89` code path that used to panic on the RTX 2000 Ada is verified on Runpod, and the
pool's Ampere cards keep working. The listing picks up the new `hub.json` with the next Hub
release.

## Status

**Listed in Runpod's Hub catalog** (checked 2026-10-07): the Hub's public catalog API
returns `lateos-ai/reflex` (listing id `cmuj7rnqr000007jwfsbb4ipc`) both for an owner filter
and for a plain `reflex` search, which it did not on 2026-09-30. The listing page is
<https://console.runpod.io/hub/lateos-ai/reflex>; `https://www.runpod.io/hub/lateos-ai/reflex`
still 404s. The listed release at that check was **`v0.2.2-runpod-hub`** (build image
`registry.runpod.net/lateos-ai-reflex-master-runpod-dockerfile:16370973d`), which predates the
fatbin kernels and the slim runtime, so every fresh worker still pays the PTX JIT.
**`v0.2.3-runpod-hub`** brings the fatbin build, the Ada-inclusive `gpuIds`, streaming, and
the `ADAPTER_ARGS` deploy field.

**Update, checked 2026-10-09:** the listing has moved to **`v0.2.3-runpod-hub`**. The Hub
console (listing page <https://console.runpod.io/hub/lateos-ai/reflex>) now shows the
v0.2.3 release promoted from `master` at tag `v0.2.3-runpod-hub`, completed 2026-10-08,
with the fatbin/slim-runtime description and its test results passing. The build-image
digest is not exposed in the listing UI; the release name is what the Hub builds from, so
the PTX JIT the v0.2.2 image paid is no longer in the catalog path. The GHCR image the Hub
publishes is `ghcr.io/lateos-ai/reflex-runpod-hub`, digest (verified 2026-10-09)
`sha256:6ecb8e1f0200c2736141014eadc9ad38bee7560fad52b48e02dbacd31d510f33`. The v0.2.2 note
above is kept for history.

**Update, 2026-10-09 (classifier release, `v0.2.5-runpod-hub`):** the worker now bakes
Qwen3-0.6B-**Q8_0** (was Q4_K_M) and the listing leads with classification; `hub.json` is
retitled, `.runpod/README.md`'s "Classifier caller contract" documents how to use it, and
`tests.json` gains a 4-label routing smoke test. `POST /v1/classify` and `handler.py`'s
`labels` branch are the same code that has been on `master` since `v0.2.3`; this release is
packaging, doc and model-quant only. `v0.2.4-runpod-hub` is the adoption/Hub-refresh release
(#16), not this one.

**Update, 2026-10-12 (`v0.2.6-runpod-hub`):** an engine-only rebuild. `rmsnorm_kernel` now
runs one block per row instead of recomputing each row's sum of squares in every thread,
with the same summation order, so outputs are bit-identical. Classification scores and
generated tokens can't change; only prefill gets faster. Measured on an RTX A40 with
Qwen3-0.6B (f16 weights), warm prefill went 8.0 -> 7.2 ms at 29 tokens, 14.2 -> 10.6 ms at
113 and 67.5 -> 53.9 ms at 449. The fix was GPU-verified on sm_86 only; the Hub's
build-time `tests.json` run is the first check on the other fatbin architectures.
`.runpod/Dockerfile`, `hub.json`, `handler.py`, `tests.json` and the baked model are
unchanged.

`iconUrl` in `hub.json` points at a real hosted asset (`.runpod/icon.jpg` on `master`), not
the old `TODO:` placeholder. The GPU-selection fields in `hub.json`/`tests.json` are kept as
advisory intent only (real evidence above shows the Hub test scheduler does not honor them).

**Update, 2026-10-14 (`v0.2.8-runpod-hub`): why v0.2.5 to v0.2.7 never listed.** The listing
stayed on `v0.2.4-runpod-hub` through three releases. The Hub console showed two causes:

- `v0.2.6-runpod-hub` failed at its first build step ("Could not find remote branch 5a6a9c44…
  to clone"). The Hub clones a release's target as a **branch**, and v0.2.6 had been created
  with a commit SHA as its target. Always create Hub releases with `--target master`.
- `v0.2.7-runpod-hub` (target `master`, same image) built, but its test job failed: "There were
  no instances available for your specified GPU NVIDIA RTX A4500 nor on any of the fallback GPUs
  NVIDIA RTX 2000 Ada Generation, NVIDIA RTX 4000 Ada Generation." The test fallbacks come from
  `hub.json`'s `gpuIds`, and in serverless stock all three cards sit in one data center
  (EU-RO-1), so they run out together. `v0.2.5-runpod-hub` has the same pin and pool, which
  fits it failing the same way.

So this release pins the test job to an RTX A5000 (`sm_86` like the A4500, serverless stock HIGH
across four data centers on 2026-10-14), adds `AMPERE_24` as a fallback pool, and allows CUDA
13.2 hosts, which most A4500/A5000 serverless hosts report (the image's CUDA 12.4 runtime runs on
any newer driver). The image, `handler.py`, the tests themselves and the baked model are
unchanged.

**Follow-up, same day (`v0.2.9-runpod-hub`).** v0.2.8's test job failed the same way: "no
instances available for your specified GPU NVIDIA RTX A5000 nor on any of the fallback GPUs
NVIDIA GeForce RTX 3090, NVIDIA L4". Two things it showed:

- The test job's fallbacks are the other cards in the **pinned GPU's own pool** (`AMPERE_24`
  here), not `hub.json`'s `gpuIds`. Widening `gpuIds` doesn't help the test; only the pin does.
- Stock moves within the hour: the A5000 went from HIGH to out on every CUDA version, with the
  A4500, A4000 and RTX 3090.

So `tests.json` now pins an **RTX 4090**, as most of Runpod's own Hub workers do
(`worker-sglang`, `worker-comfyui`, `worker-faster_whisper`). It is the largest serverless fleet,
and the only 24 GB card with stock at that check. The test config also lists
`allowedCudaVersions` explicitly, like theirs. `hub.json` adds `ADA_24` as a last-resort pool for
deployed endpoints. No pin guarantees capacity: if a test job still finds none, retrying it from
the Hub console when stock returns needs no new release.

**Update, 2026-10-23 (`v0.2.12-runpod-hub`): pick the model at deploy time.** `hub.json`
gains the Model dropdown (12 GGUFs, Qwen3-0.6B Q8_0 still baked in and the default), a
Custom GGUF URL field and a Hugging Face token field, and the container disk goes from 10
to 25 GB for the largest pick (see "Choosing the model"). Running every pick through
`handler.py` on a Runpod A40 first showed broken chat on most non-Qwen3 models (stop-token
text in replies, no stop at Qwen3.5's and Gemma 4's end-of-turn tokens, three chat
templates failing to load). The same release fixes those in the engine (IPC
`stop_token_ids`) and in `reflex-openai-adapter`. `tests.json` is unchanged: the Hub's
test job runs the default model.
