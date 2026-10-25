# Compliance records

> **Not legal advice.** These pages describe what Reflex does and what records it can
> produce. They do not say what any law requires of you or whether a deployment
> complies. Check your obligations with a qualified advisor.

Reflex can produce four kinds of records. Each is opt-in, and with every option off
nothing changes: no file is hashed, nothing is logged, and every response is the same
as without them. Nothing here runs on the engine's cold-start path.

| Record | What it is | How to get it |
| --- | --- | --- |
| Supply chain | CycloneDX SBOMs of the engine and the OpenAI adapter, the license text of every Rust dependency, and (in an image) the pip and apt package lists and base image names | [Build the bundle](#the-compliance-bundle) |
| Model provenance | A GGUF's `general.*` metadata (name, license, source, quantizer, ...), architecture, quantization and sha256 | `reflex model-info <gguf> [--sha256] [--json]` ([reference](../reference.md#reflex-model-info)) |
| Per-job attestation | Which model file (sha256, license) and engine served a job, on which GPU, with its token counts and wall time | Runpod Hub worker, `COMPLIANCE=on` |
| Audit log | One JSON line per job: id, time, endpoint, model sha256, status, token counts, latency, and optionally the prompt and answer (hashed or in full) | Runpod Hub worker, `AUDIT_LOG=stdout` or `volume` |

The Hub options are described in [.runpod/README.md](../../.runpod/README.md#compliance-records-and-audit-log).

Templates and statements for deployers:

- [model-card-template.md](model-card-template.md): a model card to fill in. The Hub
  worker's report job fills in the parts it can know.
- [data-handling.md](data-handling.md): what Reflex does with prompts, outputs and
  files.
- [eu-ai-act-deployer-notes.md](eu-ai-act-deployer-notes.md): where these records may
  help with the EU AI Act's transparency and record-keeping provisions.

## The compliance bundle

The bundle is built from the lockfiles, so it needs no GPU or CUDA toolkit:

- **CI**: the `Compliance bundle` workflow (`.github/workflows/compliance.yml`) runs on
  every `v*` tag and on demand, and uploads the bundle as an artifact.
- **Locally**: install the tools at the versions the workflow uses, then run the
  script from the repo root:

  ```sh
  cargo install --locked cargo-cyclonedx@0.5.9
  cargo install --locked --features cli cargo-about@0.9.2
  sh scripts/compliance-bundle.sh compliance-bundle
  ```

- **In an image**: build with `--build-arg COMPLIANCE_BUNDLE=1` (root `Dockerfile`, any
  target). The files go to `/usr/share/reflex/compliance/`. The Runpod Hub image
  (`.runpod/Dockerfile`) includes it by default. It also records the pip packages and
  the baked model's sha256 there.

| File | Contents |
| --- | --- |
| `reflex.cdx.json` | CycloneDX 1.5 SBOM of the engine crate with the `ipc` feature (what the adapter images ship) |
| `reflex-openai-adapter.cdx.json` | CycloneDX 1.5 SBOM of `sidecar/openai-adapter` |
| `THIRD_PARTY_LICENSES.html`, `THIRD_PARTY_LICENSES-adapter.html` | License text of every crate in each lockfile, all optional features included |
| `build.json` (image only) | Engine crate version, `rustc` version, base images; on the Hub image, the baked model and its sha256 |
| `dpkg.txt` (image only) | `dpkg-query -W` of the runtime image |
| `pip-freeze.txt` (Hub image only) | `pip3 freeze --all` of the runtime image |
| `data-handling.md` (image only) | The statement in this folder |

The license policy is `deny.toml`, checked by `cargo deny check licenses bans sources`
in CI for both Cargo projects. It allows MIT, Apache-2.0, BSD, ISC, Unicode-3.0 and Zlib,
crates.io sources only. Two exceptions, both pulled in only by the optional `download`
feature (`hf-hub`) and not in any image: `webpki-roots` (CDLA-Permissive-2.0, CA
certificate data) and `option-ext` (MPL-2.0).

The Rust toolchain is pinned in `rust-toolchain.toml`, and the Hub image pins its pip
packages, so a bundle describes a build that can be repeated.

## What these records don't do

- **No retention or rotation.** The audit log is written and never read, rotated,
  shipped or deleted by Reflex. Keeping it for as long as you need, protecting it and
  deleting it are your job. The same goes for Runpod's own log retention when you use
  `AUDIT_LOG=stdout`.
- **No verification of upstream claims.** A model's license comes from its GGUF
  metadata and from the Hub worker's `models.json`, which records each listed model's
  Hugging Face model-card license as checked on the date in that file. Neither
  replaces reading the license yourself.
- **No in-engine state.** The engine keeps no log or store; all of the above lives in
  the CLI's output and the Hub worker (see
  [DEVELOPMENT.md's non-goals](../DEVELOPMENT.md)).
