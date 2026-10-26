# Data handling

> **Not legal advice.** This page states what the Reflex software does with data. It
> does not cover the platform you run it on (Runpod, AWS, your own servers), your
> application, or what any law requires. Check your obligations with a qualified
> advisor.

This applies to the `reflex` engine, the `reflex-openai-adapter` sidecar and the
Runpod Hub worker (`.runpod/`) as shipped in this repository.

## Prompts and outputs

- **One job at a time, nothing kept between jobs.** The engine runs requests one
  after another (`batch_size` is always 1). The adapter queues up to 16 waiting
  requests (`--max-queue-depth`), but each one gets a fresh context: the HTTP adapter
  never pins a prefix or reuses one request's KV cache for another. When a job ends,
  its prompt and output are no longer held, apart from GPU and process memory that the
  next job overwrites.
- **Not logged by default.** None of the worker's, adapter's or engine's log lines
  contain prompt or output text. They record start-up, model loading, errors and
  timings.
- **Logged only if you turn it on.** The Hub worker's `AUDIT_LOG` writes one line per
  job. With `AUDIT_LOG_CONTENT=off` (the default) the line has no prompt or output.
  With `hash` it has their sha256. With `full` it has the job input and the answer text
  as sent, which may include personal data. Embedding vectors are never logged. Where
  the lines go is up to you: Runpod's log stream (`stdout`) or a file per worker on
  your network volume (`volume`).
- **Attestations hold no content.** With `COMPLIANCE=on`, each response carries model
  and engine identifiers, token counts and timings. With RAG, it also names the
  documents the passages came from, never the job's text.

## Network

- **The engine has no network socket.** It is reached only through local IPC (`reflex
  stdio`, `reflex uds`), its C/Python bindings or its CLI.
- **The adapter listens on HTTP.** On the Hub worker it binds `127.0.0.1` only, and
  the worker's handler is its only client.
- **Downloads happen only at worker start.** The Hub worker fetches the selected model
  (`MODEL` or `GGUF_URL`), and with RAG on the embedding model and `RAG_CORPUS_URL`.
  It sends `HF_TOKEN` only to `huggingface.co`. No job causes a download, and no job
  data is sent anywhere except back to the caller through Runpod's job API.

## Files

| What | Where | When |
| --- | --- | --- |
| Downloaded models, and their `.sha256` when a compliance option is on | `/runpod-volume/reflex-models/` (network volume) or the container disk | worker start |
| RAG corpus index (chunks and vectors of your corpus) | `/runpod-volume/reflex-rag/` or the container disk | worker start, rebuilt when the corpus changes |
| Audit log | `/runpod-volume/reflex-audit/<worker-id>.jsonl` | per job, only with `AUDIT_LOG=volume` |

Documents sent in a job's `rag.documents` are chunked and embedded in memory for that
job only, and are not written to disk. Nothing on the volume is deleted or rotated by
Reflex.

## Third parties

Reflex itself has no telemetry: it sends nothing to its maintainers or to any
analytics service.
The platform you deploy on, Hugging Face (for downloads) and the model's license
terms are separate, and their own policies apply.
