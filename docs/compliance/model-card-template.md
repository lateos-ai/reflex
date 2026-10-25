# Model card: _model name_

> **Not legal advice.** This is a template for recording facts about a model you
> deploy. Filling it in does not by itself meet any legal obligation. Check your
> obligations with a qualified advisor.

Fill in one card per model file you serve. The **Identity** and **License** sections
can come straight from Reflex:

- `reflex model-info <file.gguf> --sha256 --json` prints the file's sha256, size,
  architecture, quantization and every `general.*` key (name, license, source URL,
  organization, quantizer, base models, ...).
- On the Runpod Hub worker with `COMPLIANCE=on`, a `{"compliance": "report"}` job
  returns this card with those sections filled in from the model file and the
  worker's `models.json`.

Everything under **For the deployer** depends on how you use the model, so only you
can write it.

## Identity

| Field | Value |
| --- | --- |
| Model name | `general.name` |
| File | |
| sha256 | |
| Size (bytes) | |
| Architecture | `general.architecture` |
| Quantization | `general.file_type` (e.g. Q8_0, Q4_K_M) |
| Source | where you downloaded the file, with the revision or commit |
| Base model | `general.base_model.*`, or the upstream model page |
| Quantized or converted by | `general.quantized_by`, or the repository owner |

## License

| Field | Value |
| --- | --- |
| License | SPDX id or name |
| License URL | |
| License as stated in the file | `general.license` / `general.license.name` (may be missing) |
| Use restrictions you must pass on | e.g. an acceptable-use policy |
| Attribution required | |

## Serving setup

| Field | Value |
| --- | --- |
| Engine | Reflex, image or release used |
| Image digest or build id | |
| Weights dtype | `f16` (default) or `f32` (`--weights`) |
| GPU | |
| Request limits | adapter flags (`--max-tokens-cap`, ...) |
| Records kept | attestation / audit log mode, and where they are stored |

## For the deployer

- **Intended use and users:**
- **Uses that are out of scope:**
- **Evaluation:** what you tested, on what data, with what results.
- **Known limitations and risks for this use:** including the upstream model card's
  own list.
- **Human oversight:** who reviews outputs, and when.
- **How people are told they are dealing with an AI system:**
- **Incident contact:**
- **Date of this card and its next review:**
