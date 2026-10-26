# EU AI Act: notes for deployers

> **Not legal advice.** These notes point at provisions of Regulation (EU) 2024/1689
> (the AI Act) that deployers often ask about, and say which Reflex records may help
> with them. Whether and how the Act applies to you depends on your use case, your
> role and the dates in force when you deploy. Check with a qualified advisor and the
> current text of the Act, including any amendments to its timeline.

## Roles

The Act puts different duties on a **provider** (who places an AI system or model on
the market under their name) and a **deployer** (who uses one in their own
activities). Reflex is open-source serving software: running it to answer your users
usually makes you the deployer of the system you build with it, and the model's
publisher (Qwen, IBM, Google, ...) the provider of the general-purpose AI model.
Building a product on top and offering it to others can make you a provider too.

General-purpose AI model duties (technical documentation, a copyright policy, a
training-data summary) fall on the model's provider. The model card template asks for
the upstream model card and license for that reason: you pass on what the provider
published. Reflex does not produce those documents.

## Provisions and the records that may help

| Topic | Where in the Act | Records that may help |
| --- | --- | --- |
| AI literacy of staff who operate the system | Art. 4 | none (organizational) |
| Telling people they are interacting with an AI system, and marking AI-generated content | Art. 50 | none directly: this is your application's job. The attestation shows which model produced a response, for your own records. |
| Using a high-risk system according to its instructions, with human oversight, and monitoring it | Art. 26(1), (2), (5) | [model card](model-card-template.md) (intended use, oversight), [data handling](data-handling.md) |
| Keeping the logs a high-risk system generates, where they are under your control, for at least six months unless other law says otherwise | Art. 26(6) | audit log (`AUDIT_LOG`): one line per job. Reflex does not retain or rotate it; how long you keep it is up to you. |
| Automatic logging capability of high-risk systems (a provider duty) | Art. 12 | audit log and attestation: when, which model file (sha256), which engine build, the outcome |
| Information for a data protection impact assessment, where one is needed | Art. 26(9), GDPR Art. 35 | [data handling](data-handling.md), the audit log's content mode |
| Knowing what software is deployed (supply-chain transparency) | not an AI Act duty as such; see also the Cyber Resilience Act for products with digital elements | SBOMs and third-party licenses in the [compliance bundle](README.md#the-compliance-bundle) |

Most of these duties apply only to **high-risk** systems (Annex III uses such as
employment, credit scoring, education or access to essential services, or a safety
component under Annex I). Whether your use is high-risk is a legal question about the
use, not about the software. Classification, chat and embedding are all general
techniques, and the same model can be high-risk in one product and not in another.

## Choosing the audit log's content mode

- `off` (default): proves that a job ran, with which model and how it ended, without
  keeping any content.
- `hash`: also lets you show later that a given prompt and answer are the ones that
  were processed, if you keep them elsewhere. A sha256 of short or guessable text
  (a label, a yes/no) can be reversed by trying candidates, so treat hashes of
  personal data as personal data.
- `full`: keeps the job input and the answer. Weigh it against data minimization, and
  decide retention, access control and deletion before turning it on.

## What Reflex does not do

It does not classify your use case, generate transparency notices, run conformity
assessments, register systems, report incidents, keep logs for a period, or check
that a model's license allows your use. The records above are inputs to that work.
