# Reflex adoption plan — serverless / scale-to-zero

An ordered backlog for getting Reflex in front of platform operators who bill per
cold second (Runpod, then AWS scale-to-zero). Not a GitHub-stars plan, and not a
feature plan. It is written to be executed **one task at a time**.

Audience lock: a person who creates a `workersMin=0` GPU endpoint. Steady-state
chat, PyPI, Hacker News, and r/LocalLLaMA are out of scope until this path exists
and the claims below are the only ones in public.

## How to use this file

When asked to "do the next adoption task", pick the first unchecked task in the
status table whose dependencies are done, and follow these rules:

1. **Read `docs/DEVELOPMENT.md` first.** Its Non-goals win over this plan. Nothing
   here adds an in-core HTTP server, a queue, a scheduler, or `batch_size` > 1.
   HTTP stays in the sidecar. If a step seems to conflict, stop and ask.
2. **Work on `master`, not `public`.** The public branch is synced from master.
3. **One task per branch and per PR**: `adopt/A<nn>-<slug>`. No drive-by refactors.
4. **Never publish a number you did not measure.** Copy figures from
   `docs/serverless-cost-comparison.md`, `docs/runpod-llamacpp-comparison.md`,
   `docs/benchmarks.md`, or `CHANGELOG.md`. If a figure is not there, write `TBD`.
5. **Tags:**
   - `[cpu]` is docs and config only. Finish it here.
   - `[human]` needs a Runpod/Modal/GitHub account or money. Prepare the diff,
     set status to `needs-human`, and stop. Do not mark it done.
6. When a task is done, tick it in the status table and add a one-line note.

## What not to do

- Do not lead with 1.13× versus llama.cpp `llama-simple`, or with 48–150× versus
  vLLM on a dedicated T4. Those are engine-bench rows. This audience is billed for
  platform wall clock.
- Do not claim an end-to-end serverless win versus llama.cpp. Engine load was
  ~0.49 s vs ~0.95 s on an L4 (n=5). End-to-end was inconclusive (platform variance
  24–73 s). See `docs/runpod-llamacpp-comparison.md`.
- Do not quote the Runpod vLLM comparison without the caveat in the same sentence:
  ~42–44 s vs ~150 s, same GPU tier, same day; Reflex baked in `Q4_K_M`, vLLM
  downloaded `bf16` at cold start, and the endpoint types differed (load-balancing
  vs queue). See `docs/serverless-cost-comparison.md`.
- Do not send anyone `docs/pitch.md` until A04 lands. It still says f16 residency
  is planned and VRAM is ~4 bytes per parameter. Neither is true.
- Do not list Modal next to Runpod as an equal deploy path until A06 is measured.
- Do not rename the project, open a Discord, or publish to PyPI in this plan.

## Allowed public sentences

Use these verbatim, or do not use a number:

- "On Runpod serverless, a cold Reflex invocation was ~42–44 s wall clock versus
  ~150 s for Runpod's official vLLM worker, same GPU tier, same day. Most of both
  numbers is the platform provisioning a GPU. vLLM also downloaded bf16 weights;
  Reflex served a Q4_K_M GGUF baked into the image. Endpoint types differed
  (load-balancing vs queue)."
- "Reflex's own engine load on that platform was ~1.6 s of the ~42–44 s. The rest
  no engine avoids."
- "Against an official llama.cpp server image on the same platform, engine load was
  ~0.5 s vs ~0.95 s on an L4. End-to-end, platform variance (24–73 s) was larger
  than that gap, so that comparison is inconclusive."
- "The job this is for is a single cold decision (`POST /v1/classify`), not a warm
  chat pool. `batch_size` is always 1. Steady traffic wants vLLM."

## Status

| ID | Task | Tags | Depends on | Status |
|---|---|---|---|---|
| A01 | Hub catalog on the fatbin release | `[human]` | — | done 2026-10-09 — listing is on `v0.2.3-runpod-hub` (promoted from `master`, completed 2026-10-08). Recorded in `.runpod/README.md` Status. |
| A02 | One operator page | `[cpu]` | — | done 2026-10-09 — `docs/deploy-serverless.md` added (both Runpod paths, classify example, four sourced claims, Modal hold). Links to long-form docs, does not rewrite them. |
| A03 | Versioned image contract | `[cpu]` | A02 | done 2026-10-09 — GHCR digests verified for both images (LB `sha256:219edd36…`, Hub `sha256:6ecb8e1f…`, both unchanged since the T06 push). Pin rule + digests added to `docs/deploy-serverless.md` and `docs/reference.md`; release-note contract added. |
| A04 | Retire stale pitch claims | `[cpu]` | — | done 2026-10-09 — `docs/pitch.md` gained a "partly superseded" banner; weight-storage (f16 default), architecture count (six, matching README), and all PyO3 mentions (do not build) corrected. Talk tracks/benchmark tables otherwise untouched. |
| A05 | README deploy block points at the operator page | `[cpu]` | A02 | done 2026-10-09 — README Deploying section opens with `docs/deploy-serverless.md` as the first row + classify note; Modal marked experimental. `llms.txt` gained a "Serverless / scale-to-zero" section and the Modal hold. No numbers added. |
| A06 | Modal image parity, then measure | `[cpu]` `[human]` | A02 | done 2026-10-09 — kept the pinned `sm_89` cubin (documented in `.modal/README.md`) and aligned `.modal/Dockerfile` to the root slim `base`+`libcublas-12-4` runtime stage. Step 2 run on a real Modal L4, `n=3`: 8.1s median local submit → first token (6.5/8.8/8.1s; a first post-build run at 28.4s is excluded as a fresh image pull). `docs/deploy-serverless.md` Modal hold replaced with the measurement. |
| A07 | One outbound note to platform operators | `[human]` | A01, A02, A04, A05 | draft ready 2026-10-09 — post text written under "Outbound draft"; `.runpod/hub.json` description updated to match (needs a new `v0.2.x-runpod-hub` release for the listing to pick it up). Remaining `[human]`: cut the Hub release and post the note, then paste the post URL here. |

---

## A01 — Hub catalog on the fatbin release `[human]`

**Why.** As of the 2026-10-07 catalog check in `.runpod/README.md`, the public
listing still served `v0.2.2-runpod-hub`. That image predates the fatbin kernels
and the slim runtime, so every fresh Hub worker still pays the PTX JIT (~0.8 s).
The local git tag `v0.2.3-runpod-hub` already exists and is what `CHANGELOG.md`
describes (fatbin, Ada-inclusive `gpuIds`, streaming, `ADAPTER_ARGS`). The listing
only moves when that release is what the Hub build pipeline consumes.

**Steps.**

1. Confirm `v0.2.3-runpod-hub` is on `origin` (`git ls-remote --tags origin`).
   If it is only local, push the tag. Do not cut a new tag if this one already
   matches the fatbin Dockerfile.
2. Open the listing
   <https://console.runpod.io/hub/lateos-ai/reflex> and the catalog API the
   `.runpod/README.md` status section used. Record the release name and image
   digest the catalog actually builds.
3. If it is still `v0.2.2-runpod-hub`, trigger the Hub rebuild against
   `v0.2.3-runpod-hub`. Do not change handler behavior in this task.
4. Update the Status section of `.runpod/README.md` with the date, the release
   the catalog returned, and the image digest. If the catalog did not move,
   leave this task `needs-human` and say what blocked it.

**Acceptance.** The catalog's listed release is `v0.2.3-runpod-hub` (or a later
tag that is the same fatbin image), and `.runpod/README.md` says so with a date.
No new engine code.

### A01 runbook (human) — DONE 2026-10-09

Resolved from the Runpod Hub console: the listing now serves
**`v0.2.3-runpod-hub`** (promoted from `master` at tag `v0.2.3-runpod-hub`,
completed 2026-10-08, fatbin/slim-runtime description, test results passing). No
rebuild was needed — the catalog had already moved off `v0.2.2-runpod-hub`. The
build-image digest is not shown in the listing UI; the release name is what the
Hub builds from. Recorded in `.runpod/README.md`'s Status section. The
verification notes below are kept for history.

Verified 2026-10-09:

- `v0.2.3-runpod-hub` is on the `github` remote as an **annotated** tag:
  tag object `1f364f71ebe1e8d05278cf3560329327a4cdb93c`, commit
  `9941d283f34e4f0bad16c1bc096192746b2e0652`. (`v0.2.2-runpod-hub` is a
  lightweight tag at commit `16370973d229339f3848784e366e57643b0dfc73`, which is
  exactly the build-image suffix the catalog still shows.)
- `v0.2.3-runpod-hub`'s `.runpod/Dockerfile` defaults
  `REFLEX_CUDA_ARCHS="sm_75,sm_80,sm_86,sm_89,sm_90"` (fatbin) on the slim
  `nvidia/cuda:12.4.1-base` runtime, so it is the release that removes the PTX JIT.
- Remote note: this clone's `origin` is a dead local path (`E:/coldstart-infer.git`).
  The real remote is named `github`; use `git ls-remote --tags github` here.

To finish:

1. Open <https://console.runpod.io/hub/lateos-ai/reflex> in a browser and note
   the release name and the built image reference/digest. The Hub build-image
   tag follows the tag commit (v0.2.2 → `...dockerfile:16370973d`), so a v0.2.3
   build should end in the `v0.2.3` tag object or commit above; if it still ends
   in `16370973d`, the catalog is on the old image.
2. If the listing is still `v0.2.2-runpod-hub`, trigger a Hub rebuild against
   `v0.2.3-runpod-hub`. Change no handler behavior.
3. Record date, release name, and image digest in the Status section of
   `.runpod/README.md`, then tick A01 here.

---

## A02 — One operator page `[cpu]`

**Why.** A buyer currently crosses the README deploy table, `serverless/runpod/`,
`.runpod/`, two AWS guides, a Modal README, and two comparison docs. The
recommended path (load-balancing) and the discovery path (Hub queue worker)
contradict each other unless you already know why both exist.

**Steps.**

1. Add `docs/deploy-serverless.md`. It is the only page a new operator should
   need. Structure:
   - One sentence: Reflex is the engine you run at `workersMin=0` for a single
     cold decision. Steady chat wants vLLM. Link Non-goals in the README.
   - **Path 1 (recommended): Runpod load-balancing.** Image
     `ghcr.io/lateos-ai/reflex-runpod` (digest pin comes from A03; until A03,
     say `:latest` moves and point at `scripts/deploy_runpod.sh` and
     `serverless/runpod/README.md`). `workersMin=0`. Health check `/ping`.
     Show `POST /v1/classify` with the sentiment JSON already in
     `sidecar/openai-adapter/README.md`. Do not invent a new example.
   - **Path 2 (discovery only): Runpod Hub.** Link the listing. Say it is a
     queue worker because the Hub cannot publish a load-balancing endpoint.
     Show the same decision as a job `input` with `labels`, copied from
     `.runpod/README.md`. Say a non-streaming job's `output` is a one-element
     list as of `v0.2.3-runpod-hub`.
   - **AWS scale-to-zero:** one paragraph and a link to `docs/aws-deployment.md`.
     Do not duplicate that guide.
   - **Modal:** one sentence that `.modal/` is an older image recipe and is not
     a supported deploy path until A06 records a measurement. Link
     `.modal/README.md` for the experiment, not as a quickstart.
   - **Claims:** paste the four allowed sentences from the top of this file,
     each linked to its source doc. No other numbers.
2. Do not rewrite `serverless/runpod/README.md` or `.runpod/README.md`. Link them
   as the long form.

**Acceptance.** `docs/deploy-serverless.md` exists, contains both Runpod paths,
the classify example, the four allowed sentences, and an explicit Modal hold.
A reader never has to open `docs/DEVELOPMENT.md` to create an endpoint.

---

## A03 — Versioned image contract `[cpu]`

**Why.** Both public images are documented as `:latest`
(`ghcr.io/lateos-ai/reflex-runpod`, `ghcr.io/lateos-ai/reflex-runpod-hub`). A
buyer who pins `:latest` gets surprise rebuilds. Known-good digests from the
2026-10-02 push (see the T06 note in `docs/IMPROVEMENT_PLAN-9-30.md`):
load-balancing `219edd36` (2.06 GB), Hub `6ecb8e1f` (2.37 GB). Do not treat
those as current if a later push exists — check the registry, and if you cannot,
write the date and "confirm before pinning."

**Steps.**

1. In `docs/deploy-serverless.md`, replace the bare `:latest` recommendation
   with: pin a digest; `:latest` is a moving alias. Record the digest you
   verified, or `TBD — confirm ghcr digest before pinning` if this environment
   cannot query the registry.
2. In `docs/reference.md`'s Docker section, add the two image names and the same
   pin rule. Do not remove the local `docker build` instructions.
3. Add a three-line release note to `docs/deploy-serverless.md`: a Hub or
   load-balancing release records the image digest, the fatbin arch list, and
   whether cold-start numbers changed. Do not invent a GHCR version tag that
   was never pushed.

**Acceptance.** The operator page tells a buyer to pin a digest, and says what
to do when the digest is unverified. No image is pushed from this task.

---

## A04 — Retire stale pitch claims `[cpu]`

**Why.** `docs/pitch.md` is in the tree a buyer clones. Its objection table still
says weights are held `f32` (~4 bytes per parameter) and that f16 residency is
planned. The default is `f16`. It also says four architectures and that PyO3
bindings exist; the README says the Python bindings do not currently build, and
Kolibri-1 and Llama are supported.

**Steps.**

1. At the top of `docs/pitch.md`, under the title, add a banner: this file is an
   internal cheat-sheet; every number and limit in the README and
   `docs/benchmarks.md` wins if they disagree. Do not use this file in a listing,
   a post, or a Hub description.
2. Fix only the claims that are now false, so a skimming reader is not misled:
   - VRAM row: default is `f16` (~2 bytes per parameter for matrix weights);
     `f32` is opt-in. Point at the README's weight-storage paragraph. Do not
     restate a param-count ceiling you have not remeasured.
   - Architecture count: match the README table. Do not add architectures the
     README does not list.
   - PyO3: say the bindings do not currently build, matching the README. Do not
     claim an embed path that fails.
3. Do not rewrite the talk tracks or the benchmark tables in this task. If a
   talk track still quotes a stale VRAM line, delete that sentence rather than
   inventing a replacement number.

**Acceptance.** A search of `docs/pitch.md` no longer says f16 is planned, no
longer claims PyO3 works, and the banner is the first thing under the title.

---

## A05 — README deploy block points at the operator page `[cpu]`

**Why.** The README is what the listing and `llms.txt` send people to. Its
Deploying table is accurate and too flat: Hub, load-balancing, Modal, and AWS
look like equal choices, and the first screen is a `cargo build` quickstart.

**Steps.**

1. In the README Deploying section, make `docs/deploy-serverless.md` the first
   row, labeled as the operator start. Keep the other rows as the long form.
2. Add one sentence above the table: for a scale-to-zero endpoint, start there;
   the job shape is `POST /v1/classify`, not chat. Link the sidecar README's
   classify section.
3. In `llms.txt`, add the operator page next to the platform sections, and the
   same one-sentence hold on Modal. Do not delete the local `cargo` commands.
4. Do not move the cold-start headline or the comparison table. Do not add a
   number that is not already in the README.

**Acceptance.** From the README Deploying section, the next click is
`docs/deploy-serverless.md`. `llms.txt` names that file. Modal is not described
as a supported quickstart.

---

## A06 — Modal image parity, then measure `[cpu]` `[human]`

**Why.** `.modal/Dockerfile` still pins `REFLEX_CUDA_ARCH=sm_89` and was not
part of the slim-runtime change. Promoting it would ship a different image than
the one the Runpod numbers were measured on.

**Steps.**

1. `[cpu]` Align `.modal/Dockerfile` with the root `Dockerfile`'s fatbin default
   (`REFLEX_CUDA_ARCHS=sm_75,sm_80,sm_86,sm_89,sm_90`) and the slim runtime base,
   or document in `.modal/README.md` why a pinned `sm_89` cubin is still the
   right call for Modal's single-SKU selection. Do not change engine code.
   Update `.modal/app.py` build args to match. Leave the buyer page's "not
   supported" sentence in place.
2. `[human]` Deploy with `python -m modal run .modal/app.py` and record cold
   wall clock the same way `.modal/README.md` already describes. Write the
   number into `.modal/README.md` with n, GPU, and date. Then, and only then,
   replace the hold sentence in `docs/deploy-serverless.md` with a link and
   that measurement. If you cannot run Modal, stop at `needs-human` after step 1.

**Acceptance.** Either Modal stays marked unsupported, or the operator page
links a measurement you just recorded. No Modal number is copied from the
Runpod comparison.

---

## A07 — One outbound note to platform operators `[human]`

**Why.** Promotion before A01–A05 sends people to a stale Hub image and a README
that does not start at the endpoint. One note, after the path exists. Not a
launch sequence.

**Steps.**

1. Draft the note in this file under "Outbound draft" (replace the placeholder
   below) using only the allowed sentences. Title it around billed cold time,
   not "faster than llama.cpp." Include the classify JSON and the link to
   `docs/deploy-serverless.md`. Name the audience: Runpod and scale-to-zero
   operators.
2. Publish it where those operators already are (the Runpod Hub listing
   description, and one post). Do not post it to Hacker News or r/LocalLLaMA
   as part of this task.
3. Paste the URL into the status table note.

**Acceptance.** The published text contains the vLLM caveat and the llama.cpp
inconclusive sentence, links the operator page, and shows classify rather than
a story prompt. The Hub listing description matches that text.

### Outbound draft

**Status:** drafted 2026-10-09, ready to publish. Publishing (the Hub listing
description via a new `v0.2.x-runpod-hub` release, and one post) is the remaining
`[human]` step. Record the post URL in the A07 status note when done. Do not post
to Hacker News or r/LocalLLaMA.

**Title:** What you actually pay for on a scale-to-zero GPU: billed cold time

**Audience:** Runpod Serverless operators and anyone running scale-to-zero GPU
endpoints.

**Body:**

> If you run a Runpod Serverless endpoint at `workersMin=0`, your invoice is
> dominated by the seconds a worker spends becoming useful — not tokens/sec. Most
> serving stacks are optimized for the other axis.
>
> Reflex is a GGUF-native inference engine whose CUDA kernels are compiled ahead of
> time by `nvcc` at build time and loaded by the driver at process start — no
> runtime NVRTC, no CUDA-graph capture. On Runpod serverless, a cold Reflex
> invocation was ~42–44 s wall clock versus ~150 s for Runpod's official vLLM
> worker, same GPU tier, same day. Most of both numbers is the platform
> provisioning a GPU. vLLM also downloaded bf16 weights; Reflex served a Q4_K_M
> GGUF baked into the image. Endpoint types differed (load-balancing vs queue).
>
> Reflex's own engine load on that platform was ~1.6 s of the ~42–44 s. The rest no
> engine avoids.
>
> Against an official llama.cpp server image on the same platform, engine load was
> ~0.5 s vs ~0.95 s on an L4. End-to-end, platform variance (24–73 s) was larger
> than that gap, so that comparison is inconclusive.
>
> The job this is for is a single cold decision, not a warm chat pool. One prompt,
> one pass, no generation:
>
> ```bash
> curl "<endpoint>/v1/classify" -H "Content-Type: application/json" -d '{
>   "prompt": "Review: The battery died after two days. Sentiment:",
>   "labels": [" positive", " negative"]
> }'
> ```
>
> ```json
> {
>   "label": " negative",
>   "labels": [
>     {"label": " positive", "probability": 0.08692727, "tokens": 1},
>     {"label": " negative", "probability": 0.9130727, "tokens": 1}
>   ],
>   "entropy": 0.42612946
> }
> ```
>
> `probability` is relative to this label set only (it sums to 1 over `labels`), not
> a vocabulary-wide probability. Use labels of equal token length.
>
> `batch_size` is always 1. Steady traffic wants vLLM. If your traffic is bursty,
> single-shot, or scale-to-zero, this is the axis that matters.
>
> Start here: https://github.com/lateos-ai/reflex/blob/master/docs/deploy-serverless.md
> — Runpod load-balancing first, then the Hub, AWS scale-to-zero, and Modal.

---

## Done when

- A new operator can create a `workersMin=0` endpoint from `docs/deploy-serverless.md`
  without opening `docs/DEVELOPMENT.md`.
- The Hub catalog digest is the fatbin image, and the README says to pin a digest.
- Every public sentence used in the listing is one of the allowed sentences above.
- Modal is either unpromoted or has its own measurement.
- No engine feature landed as part of this plan.
