# Decision models and `POST /v1/systemone`

A **decision model** answers typed questions about a piece of text (the *state*) in one
forward pass per question. Nothing is generated: the answer is a probability
distribution over the options the caller gives, read off one position of the prompt.
llama.cpp added these models and a `POST /v1/systemone` endpoint, in the shape of
TypeSafe's System One API, in PR #29818 (merge commit `a4cb4c6`, 2026-10-02). ggml-org
publishes converted GGUFs of five of them.

This is the workload Reflex was built around (`system1`, the sidecar's
`/v1/classify`): one prompt, a handful of candidate continuations, no decode loop.
The cold-start cost is most of what a caller waits for, so a decision endpoint is a
good fit for this engine.

This note covers Phase 1: the **label-token** models (`lev`, and OpenJev's text input)
on the existing `qwen35` path. Kev-4B (`kev`: a projection head and dot products of
hidden states) and the ModernBERT models (`laya`) need new engine outputs and a new
architecture, and get their own notes.

Everything below is read from llama.cpp at `a4cb4c6`
(`tools/server/server-decision.{h,cpp}`, the `server-context.cpp` and `common.cpp`
parts of the PR, `conversion/lev.py`, `conversion/qwen.py`, `gguf-py`) and from the real
GGUF headers of `ggml-org/lev-GGUF`, `ggml-org/OpenJev-GGUF` and
`ggml-org/tinyopenjev-for-testing-gguf`, fetched with HTTP range requests.

## The models

| GGUF | `decision.type` | arch | size | licence | how it scores | Phase |
|---|---|---|---|---|---|---|
| lev | `lev` | `qwen35`, 32 layers, hidden 2560, **tied** LM head | 4.2B | Apache-2.0 | logits of one label token per option; `noul` read from a 9-step rating scale | 1 |
| OpenJev | `openjev` | `qwen35`, 64 layers, hidden 5120 (Qwen3.8-27B) + vision `mmproj` | 27B | CC BY-NC 4.0 | logits of one label token per option | 1 (text only) |
| Kev-4B | `kev` | `qwen35` + `cls.output` head | 4B | | scaled dot product of projected hidden states | 2 |
| Laya, Julia-1 | `laya` | `modern-bert` | 421M, 144M | | head blocks + scorer on one marker token per option | 3 |

lev and OpenJev are plain `qwen35` files: their tensors are exactly the ones a Qwen3.5
GGUF has (no extra head, no MTP blocks), and both use the Qwen3.5 tokenizer
(`gpt2` model, `pre = qwen35`, 248,320 tokens, no BOS added). What makes them decision
models is metadata only.

### Metadata

| Key | lev | OpenJev |
|---|---|---|
| `qwen35.decision.type` | `"lev"` | `"openjev"` |
| `qwen35.decision.temperature.choice` | 1.4606453 | 0.85 |
| `qwen35.decision.temperature.choice.small` | 1.4660316 | |
| `qwen35.decision.temperature.choice.mid` | 1.3228487 | |
| `qwen35.decision.temperature.choice.large` | 1.4983294 | |
| `qwen35.decision.temperature.score` | 1.8687167 | 0.85 |
| `qwen35.decision.temperature.noul` | 1.2821622 | 1.5547129 (0.85 x 1.829074) |
| `tokenizer.chat_template.systemone` | the lev prompt | the OpenJev prompt |
| `tokenizer.chat_templates` | `["systemone"]` | `["systemone"]` |

All temperatures are `f32`. Neither file has a default `tokenizer.chat_template`, so
`/v1/chat/completions` has no template to render with. The other `decision.*` keys
(`block_count`, `max_head_tokens`) belong to `laya`.

`ggml-org/tinyopenjev-for-testing-gguf` is llama.cpp's own test model: the first 4 of
OpenJev's 64 layers cut to hidden size 64, with the same decision metadata and template
(Q8_0, 45 MB, CC BY-NC like OpenJev, output not meaningful).

### Label tokens

The score of an option is the raw logit of its **label token** at the last prompt
position. llama.cpp builds the label list once, at load, by tokenizing each label code
alone (no special tokens) and keeping it only if it is one token:

- **OpenJev**: the 52 letters `A`-`Z`, `a`-`z`. All 52 must be single tokens, or the
  load fails. At most 52 options.
- **lev**: the codes `A`-`Z`, then `AA`-`ZZ`, keeping only the single-token ones, up to
  255. On the Qwen3.5 vocabulary that is the 26 letters plus the first 229 two-letter
  codes that are one token; the 255th is `JT`. At most 255 options.

`A` is token 32 and `Z` token 57 in both files.

**lev's `noul` is read at labels, not digits.** The lev prompt asks for "a digit from 0
to 8", but llama.cpp reads the 9 ratings at the first 9 *labels* (`A`-`I`), not at the
tokens `0`-`8` (the code comments say so explicitly). Reflex follows llama.cpp, since
llama.cpp is the reference this work is verified against.

## Requests and answers

The API is TypeSafe's System One shape:

```json
{
  "state": "Customer message: I was charged twice for my order last week.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "payments and refunds", "shipping": null, "technical": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]},
    "angry":   {"type": "noul", "instructions": "Is the customer angry?"}
  }
}
```

**Validation** (each failure is a `400` naming the question):

- `state` must be present and not null. It can be any JSON value.
- `questions` must be a non-empty object. Its order is kept: answers come back in
  request order.
- Every question needs non-null `instructions` (any JSON value) and a `type`:
  - `choice`: `criteria` is a non-empty object, option key to description (any value,
    `null` allowed). Options keep the object's order.
  - `score`: `criteria` is an array of 2 to 10 level descriptions, lowest first. The
    option keys are `"0"`, `"1"`, ...
  - `noul`: `criteria` is absent, null, or an object with optional `true`/`false`
    descriptions. The options are `[false, true]`, or `[true, false]` for OpenJev.
- More options than the model has labels is a `400`.

**Images** come from an `images` array of data URLs, or from `image_url` parts inside a
`state` made of chat messages. Phase 1 is text only. llama.cpp answers `501` when the
model or server can't take images, and so does Reflex.

**Answers:**

| type | fields |
|---|---|
| `choice` | `type`, `choice` (most probable key), `probabilities` (key to p, in option order), `confidence` |
| `score` | `type`, `score` (expected level index), `legend` (key to description), `probabilities`, `confidence` |
| `noul` | `type`, `noul` (probability of true) |

plus `model` and `usage: {input_tokens, output_tokens: 0}`. `input_tokens` is the sum of
every evaluated prompt's full length, including tokens shared with another prompt and
both of lev's variants. A model that is not a decision model gets `501`.

## Prompt rendering

The prompt is the GGUF's `systemone` template, rendered once per question **variant**
with these variables:

| variable | value |
|---|---|
| `id` | the question id |
| `type` | `"choice"`, `"score"` or `"noul"` |
| `instructions` | the request's value, as given |
| `state` | the request's value, as given |
| `options` | one object per option: `key`, `description` (null if absent), and for lev `label` (the option's label code by **position**) |
| `images` | empty in Phase 1 |

The templates print a string as-is and anything else through `tojson`
(`{{ x if x is string else x | tojson }}`). For lev, every object in this input,
including the state, instructions and descriptions, is first re-ordered with its keys
sorted (lev was trained on sorted keys). OpenJev keeps the caller's key order.

The rendered text is tokenized with special tokens parsed and nothing added (no BOS).
Both templates end with the assistant turn opened and an empty think block:
`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, so the label is the first
token of the answer.

**lev's prompt**: a fixed system message, then `# Evidence` (the state), `# Criterion`
(the instructions, or the question id if they are empty), and either `# Options` (one
line `{label}. {key}: {description}` per option, or `{label}. (level {key} of {n-1})
{description}` for a score) or, for `noul`, a `# Scale` block (0 = certainly no ... 8 =
certainly yes, plus the `yes:`/`no:` descriptions when given).

**OpenJev's prompt**: one user turn with `State:`, `Question:` (a score adds "Rate along
the ordered levels below (lowest first)."), and `Options:`, one line `[{letter}]
{key}: {description}` per option, or `[A] yes: ...`/`[B] no: ...` for `noul`, with
default descriptions "The statement is true." / "The statement is false.".

### `tojson` must match llama.cpp's

llama.cpp renders with its own Jinja engine (`common/jinja`), and its `tojson` is not
Python's or minijinja's:

- separators `", "` and `": "`, no indentation;
- object keys in insertion order (never sorted);
- strings escape only `"`, `\`, `\b`, `\f`, `\n`, `\r`, `\t` and other bytes below
  0x20 (as `\u00XX`); non-ASCII is written as UTF-8 and `<`, `>`, `&`, `'` are not
  escaped;
- integers as integers; floats through C++'s default stream format, which is `%g` with
  6 significant digits (`19.99`, `0.333333`, `1e-07`, `1e+20`);
- `null`, `true`, `false`.

minijinja's built-in `tojson` is compact (no spaces) and HTML-escapes, so the sidecar
registers its own filter with the rules above. A test checks the rendered prompts
against strings rendered by llama.cpp.

### Variants

lev shows a `choice` question with 2 or more options **twice**: once in request order,
once reversed. The labels stay in position order (`A` is always the first line), so the
reversal moves the keys, not the labels. This cancels lev's preference for the first
label. Every other case (lev `score` and `noul`, all of OpenJev) has one variant.

## From logits to answers

For each variant `v`, the scores `s_v` are the raw logits of the variant's labels: one
per option, or for lev `noul` the 9 rating labels.

**Temperature.** `T` is looked up as `"{type}.{bucket}"`, then `"{type}"`, then 1.0,
where `n` is the number of options (2 for a `noul`, even on lev's rating scale):

- lev buckets: `small` for n <= 8, `mid` for n <= 26, `large` above;
- every other type: `2` for n <= 2, `3_5`, `6_10`, `11` above (no shipped file uses
  these yet).

llama.cpp reads each temperature as text: the `f32` goes through `std::to_string`
(6 decimals) and back through `strtof`, so lev's `choice` temperature is 1.460645, not
1.4606453. The sidecar does the same round trip.

**Probabilities.** A softmax per variant, then the mean of the variants, in double
precision:

```
p_v[i] = exp((s_v[i] - max(s_v)) / T) / sum_j exp((s_v[j] - max(s_v)) / T)
probs[i] = mean over v of p_v[i], with the reversed variant mapped back (index n-1-i)
```

`s_v[i] - max` is computed in `f32`, the rest in `f64`.

**Answers.**

- `choice`: `choice` is the first option with the highest probability.
  `confidence = max(0, (p_max - 1/n) / (1 - 1/n))`, and 1 when n < 2.
- `score`: `score = sum_i i * p_i`. With `m` the most probable level,
  `confidence = max(0, 1 - D / D_u)`, where `D = sum_i p_i * |i - m|` and
  `D_u = sum_i |i - (n-1)/2| / n` (the uniform distribution's spread around its center).
- `noul`, OpenJev: `noul` is the probability of the `true` option.
- `noul`, lev: `noul = sum_r p_r * r / 8` over the 9 ratings `r = 0..8`.

The temperatures are calibration constants stored in the file. llama.cpp's docs say the
probabilities are not guaranteed to be calibrated for a given dataset, and neither does
Reflex.

## Engine and sidecar

The split follows `/v1/classify`: the sidecar owns everything specific to the API,
the engine scores token positions.

**Sidecar** (`POST /v1/systemone`):

- reads `{arch}.decision.*` and the `systemone` template from the GGUF (it already
  parses GGUF metadata for the chat template), and answers `501` when there is no
  `decision.type`, or for `kev`/`laya` until those phases land;
- validates requests, renders every variant, and builds the label table (below);
- sends **one** engine request per HTTP request, holding every variant of every
  question;
- computes the probabilities, confidences and the response.

**Engine:** `Model::label_logits`, reached by the IPC request
`{"label_prompts": [{"prompt": ..., "labels": ["A", "B", ...]}, ...]}`. For each prompt
it returns `{"tokens", "token_ids", "logits"}`: the prompt's token count and the raw
logit of each label at its last position. The response also carries `shared_tokens`.
The request combines with `id` and `timings` only. That is a generalization of hybrid `system1`, which
already gathers raw LM-head logits for single-token candidates (`gemv_gather_lm_head`)
but resolves them as continuations of one prompt and returns only one prompt's scores.

- Each label string is tokenized alone, without special tokens, and must be exactly one
  token (llama.cpp's rule). Anything else fails the request.
- **Shared prefix.** The questions of a request all start with the same tokens (the
  system message and the state), and only the end differs. The engine tokenizes every
  prompt, takes the longest token prefix common to all of them (leaving at least one
  token of each prompt to evaluate), prefills that prefix once, and runs each prompt's
  remaining tokens from a copy of that state, in order. This is llama.cpp's grouping
  (`server_decision_group_tasks`), with all of a request's prompts in one group because
  there is a single sequence. The prefix is taken on tokens, not text, so no prompt is
  ever split inside a token. On the hybrid model the copy is the same state copy a
  pinned prefix uses (`HybridPrefix`), since the recurrent state of the Gated DeltaNet
  layers can't be rewound.
- The transient prefix lives only for the request. It does not touch the session's one
  pinned prefix (`pin_prefix`/`use_pinned`), and the request does not accept those
  fields.
- One job, run sequentially: no batching across prompts and no queue. The Non-goals
  stand unchanged.

The label table is built by the sidecar, since lev's labels are also template input:
the codes are `A`-`Z` then `AA`-`ZZ` for lev, the 52 letters for OpenJev. Which
two-letter codes are one token depends on the vocabulary. For a byte-level BPE
vocabulary, a two-letter ASCII code is one token exactly when the merge of its two
letters exists, so the sidecar reads it from the GGUF's tokens and merges; the engine
still tokenizes each label and rejects any that isn't one token, so a disagreement fails
loudly.

**Cold start.** lev's LM head is tied to its token embedding. The hybrid loader today
dequantizes the whole tied table to the device at load (248,320 x 2560); a decision
request reads at most 255 of its rows. The gather path already works on the lazy table
(`gemv_gather_lm_head_matches_full_vocab_gemv_while_still_lazy`), so keeping it lazy
for the hybrid tied case is a measured follow-up, not a precondition.

## Verification

**Reference.** llama.cpp at `a4cb4c6` (`llama-server`, `POST /v1/systemone`) on the same
GGUF, the same requests, on the same GPU. Two configurations:

1. **Matched**: f32 activations (`LLAMA_NO_Q8=1`, the patch already used for the
   Gemma 4 and Kolibri comparisons), f32 KV cache with flash attention off
   (`-ctk f32 -ctv f32 -fa off`). This isolates Reflex's own error. **Pass**: every
   probability, `score` and `noul` within 2e-3, and the same `choice` and the same
   `input_tokens`.
2. **Stock** `llama-server` defaults, reported as-is, for what users of llama.cpp see.

Reflex runs f16 weights (lev's f32 weights don't fit a 16 GB T4). The `-np` setting
changes how llama.cpp groups prompts and so its own results slightly (its test allows
0.01 between shared and unshared prefixes on the tiny model), so the matched run is done
with `-np 1` and with the default, and both differences are reported.

**Prompts.** A small C++ harness on llama.cpp's own `server_decision_context` (the
code `llama-server` runs: its `parse_questions`, `parse_state`, `render` and
`fill_task`'s tokenization, on `common/jinja`) renders and tokenizes the request set;
the sidecar's rendering must match it byte for byte, and the engine's tokenization
token for token. This covers the `tojson` rules, lev's sorted keys, both variants, and
non-ASCII and float values in the state.

**Requests.** The PR's support-ticket request (three questions over a string state),
plus 20 or more others: 2 to 52 options (above 26 on lev reaches the two-letter labels
and the `mid`/`large` buckets), every question type, `noul` with and without
descriptions, string, object, array and chat-message states, non-ASCII text, a state
that contains special-token text (both sides parse it as special tokens), and long
states (1K to 8K tokens).

**Models.**

- **lev Q8_0** on a T4: the real-model check.
- **tinyopenjev** (f32 weights, since it is tiny): the OpenJev template and rules.
- **A tiny lev fixture**: tinyopenjev's weights with lev's metadata and template written
  in by `gguf-py`, to run lev's rules (variants, buckets, the rating scale) in the
  nightly suite without the 4B model. Both tiny files stay local (CC BY-NC) and get a
  GPU nightly `requirement()` rule.
- **OpenJev 27B** itself does not fit a T4. It is the Qwen3.8-27B architecture that has
  already been verified against llama.cpp on an A100 80 GB, so the text path is covered
  by tinyopenjev plus that run; a full OpenJev check needs a rented 80 GB GPU.

**Regression.** Existing `system1`, `/v1/classify` and `reflex check` outputs unchanged;
GPU nightly suite.

**Benchmarks** (T4): process start to first answer, Reflex vs `llama-server` start plus
one request, with the support-ticket request; warm per-request latency on the same
request, Reflex vs `llama-server`.

## Results

AWS `g4dn.xlarge` (Tesla T4 16 GB, 4 vCPUs, gp3 root volume), 2026-10-24. llama.cpp
`a4cb4c6` built twice for `sm_75`: stock, and with `LLAMA_NO_Q8` (quantized weights
through cuBLAS instead of 8-bit activation kernels). The matched configuration also
sets `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, without which that cuBLAS path computes in
f16 on a T4. Every llama-server run used `-c 32768`. Reflex ran the sidecar in front of
`reflex stdio`, with f16 weights for lev and f32 for the tiny models.

**Request set.** 53 requests, sent unchanged to both servers: the support-ticket request;
an object state and the same state as a JSON string; `choice` with 1, 2, 5, 8, 9, 26, 27,
52 and (lev only) 60 options; `score` with 2 and 10 levels, including non-string level
descriptions; `noul` without criteria, with only `true`, and with `null`; array,
chat-message and `{"messages": ...}` states; French, Chinese, Greek, Hebrew, Russian and
emoji text, also in keys and question ids; floats (`19.99`, `1e-7`, `1e20`, `1.5e300`,
`0.0`, a 13-digit fraction) and JSON escapes; special-token text inside the state;
empty and non-string instructions; unsorted nested keys; ten mixed questions in one
request; long states of about 1K, 4K and 8K tokens per prompt; and 22 error cases.

**Prompts and tokens.** Every rendered prompt matched llama.cpp's renderer byte for byte
(54 prompts on tinyopenjev, 79 each on tiny-lev and lev), and the engine's tokenization,
label tokens and temperatures matched too. So did every `400` message and both image
refusals. One difference was found and fixed: a body that is not a JSON object
(`[1, 2]`) gets llama.cpp's `"state" must be provided` (nlohmann's `contains` is false
on a non-object), not a message of the sidecar's own. A body that is not valid JSON is
a `400` from both, with different parser messages.

**Answers.** Reflex against each llama-server configuration, over every answered request
(30 on tinyopenjev, 31 on the lev models). "Worst" is the largest difference of any
probability, `score` or `noul`:

| model | vs matched, `-np 1` | vs matched, default (4 slots) | vs stock |
|---|---|---|---|
| tinyopenjev (f32) | 4.6e-7 | 2.9e-4 | 1.3e-3, one `choice` differs |
| tiny-lev (f32) | 2.4e-7 | 2.5e-7 | 9.0e-4, four `choice`s differ |
| lev `Q8_0` (Reflex f16) | 1.29e-3 | 1.29e-3 | 2.0e-2, 31 values over 2e-3, one `choice` differs |

Both matched configurations **pass** on all three models: every value within 2e-3, the
same `choice` for every question and the same `input_tokens` for every request. The
other columns are llama.cpp's own spread:

- **`-np` on the tiny OpenJev model.** llama.cpp's matched runs with `-np 1` and with
  the default differ from each other by the same 2.9e-4 (on tiny-lev by 2e-8, on lev by
  1.9e-6). With 4 slots it evaluates a request's prompts as one group from a shared
  prefix; Reflex shares a prefix too, but the summation order of its kernels lines up
  with llama.cpp's unshared run there.
- **Stock llama.cpp** is far from its own matched build: up to 1.85e-2 on lev (lev
  `array_state`'s `noul` 0.848 stock, 0.829 matched, 0.828 Reflex). Stock 8-bit
  activations move lev's label logits by a few hundredths. The `choice` differences in
  that column are near-ties on the random tiny models (top two within 4e-4) and one on
  lev (`ten_questions.q9`, 12 options, top two 6e-6 apart); stock llama.cpp
  disagrees with its matched build on exactly the same questions.
- **lev's 1.29e-3** is Reflex's f16 weights against f32 arithmetic on the `Q8_0` blocks
  (an f16 product of an int8 and an f16 scale is rounded; lev's f32 weights don't fit
  the T4). It is under the 2e-3 bar, but it is the margin to watch if a later change
  touches the hybrid path.

A model without `decision.type` gets `501` `This model is not a decision model` from
both.

**Regression.** The GPU nightly suite on this branch: 148 passed, 0 failed, 4 skipped
(real Granite, IQ, DeepSeek-V2-Lite and Qwen2.5 GGUFs, not on this machine), including
the five `label_logits_tests` and the two `ipc_label_logits_*` tests, and the existing
`system1`, prefill and llama.cpp fixture tests. `reflex check` itself was not re-run.

**Benchmarks**, support-ticket request on lev `Q8_0` (four prompts, 484 tokens in all,
67 of them shared):

| | Reflex (sidecar + `reflex stdio`, f16) | llama-server (stock) |
|---|---|---|
| cold, page cache warm (n=5): process start to first answer | 8.40 s (8.38-8.43) | 3.54 s (3.50-3.64) |
| cold, page cache dropped (n=3) | 41.2 s | 37.2 s |
| warm, per request (n=100) | 7.06 s | 295 ms |

Reflex loads faster (about 1.3 s from process start to ready, against about 3.2 s for
llama-server), but **its first request costs 7 s, and so does every later one.** The
cost is not in `label_logits`: `REFLEX_PREFILL_PROFILE=1` puts 99.5% of a 149-token lev
prefill in the Gated DeltaNet mixer (3,659 of 3,680 ms; 821 ms on Qwen3.5-0.8B), with the
host enqueue time equal to the GPU time. The Qwen3.5 hybrid path runs those layers one
token at a time, so each evaluated token costs about 25 ms on lev (283 evaluated tokens
here: the shared 67 once, then 46, 46, 82 and 42). llama.cpp's chunked prefill evaluates
the whole request in ~0.3 s. Prefix sharing helps (484 tokens are counted, 283
evaluated), but it can't close a ~25x gap.

With the page cache dropped, both are bound by the volume (135 MB/s for the 4.5 GB file,
33 s for a plain `dd`), and Reflex's 41 s is that read plus the 7 s request. An earlier
disk-cold run showed llama-server at 4.1 s; it was measured while the previous Reflex
process was still exiting and holding the file mapped, so the cache drop missed those
pages. The benchmark now waits for every server process to exit before dropping the
cache.

**What it means.** The decision endpoint is correct: Reflex gives llama.cpp's answers on
lev and the tiny models. It is not yet the cold-start win this work was meant to be. On
the Qwen3.5 hybrid models that needs a batched (chunked) Gated DeltaNet prefill; Granite
4.0-H already walks the rows inside each kernel for its Mamba-2 layers, which is the
smaller step.

## Not supported in Phase 1

- Images (`501`), and so OpenJev's screenshot lane.
- `kev` and `laya` models (`501` from the sidecar until Phases 2 and 3).
- lev's large-option-set head (`mode_b_head.pt`); llama.cpp's converter does not
  convert it either.
- Streaming.
