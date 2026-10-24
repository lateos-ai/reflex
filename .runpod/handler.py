"""Runpod Hub queue-worker entrypoint for Reflex.

Translation shim only, no engine logic: spawns the same, unmodified
reflex-openai-adapter binary that serverless/runpod/ uses as a load-balancing
endpoint, waits for it to report ready, and forwards each Runpod job to its
existing POST /v1/chat/completions over loopback -- or, for a job whose input
has "labels", to POST /v1/classify (one engine pass scoring each label; the
one response is yielded once, like a non-streaming chat job), or, for a job
whose input has "input" and no "messages", to POST /v1/embeddings (an embedding
model such as Nemotron-3.5-Embed; yielded once the same way). See
../.runpod/README.md and ../serverless/runpod/README.md for why two deployment
paths exist.

Model choice: the image bakes Qwen3-0.6B Q8_0 (BAKED_MODEL, at GGUF_PATH).
MODEL (the Hub deploy form's dropdown) names a Hugging Face "<owner>/<repo>/<file>"
GGUF; anything other than the baked one is downloaded once at worker start, before
the adapter launches. GGUF_URL, if set, overrides MODEL with any direct GGUF URL.
Downloads go to the network volume (/runpod-volume/reflex-models) when one is
attached, so later workers reuse them, else to the container disk. HF_TOKEN, if
set, is sent to huggingface.co for gated repos.

Optional RAG (RAG=on, see rag.py): a second adapter serves an embedding GGUF
(RAG_EMBED_MODEL, Nemotron-3.5-Embed-1B by default) on another loopback port.
At worker start the corpus (a RAG_DIR folder and/or RAG_CORPUS_URL) is chunked,
embedded and cached as an index; a job whose input has "rag" gets the passages
closest to its question written into its system message, and the answer comes
back with those passages. Embedding jobs on such a worker go to the second
adapter. With RAG off (the default) nothing here changes.

The handler is a generator, so Runpod's /stream/{job_id} works. With
"stream": true in the job input, each Server-Sent Event the adapter sends is
yielded as one chat.completion.chunk dict; otherwise the single complete
chat.completion is yielded once. return_aggregate_stream makes /run and
/runsync return those yields as a list, so a non-streaming job's output is a
one-element list (the same shape runpod-workers/worker-vllm returns).
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import runpod

import rag

ADAPTER_HOST = "127.0.0.1"
ADAPTER_PORT = 8000
ADAPTER_BASE = f"http://{ADAPTER_HOST}:{ADAPTER_PORT}"
EMBED_PORT = 8001
EMBED_BASE = f"http://{ADAPTER_HOST}:{EMBED_PORT}"
READY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 0.25

VOLUME_DIR = "/runpod-volume"
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_CHUNK = 8 << 20

DEFAULT_EMBED_MODEL = "maternion/Nemotron-3.5-Embed-1B-GGUF/Nemotron-3.5-Embed-1B-Q8_0.gguf"

_rag_cfg = None
_rag_index = None


def _fail(message):
    print(f"reflex worker: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def _log(message):
    print(f"reflex worker: {message}", flush=True)


def _model_url(model, url="", what="MODEL"):
    """The GGUF to download for a model id ("<owner>/<repo>/<file>.gguf") or a
    direct URL, or None to use the baked model."""
    if url:
        if not url.startswith(("https://", "http://")):
            _fail(f"GGUF_URL must be an http(s) URL, got {url!r}")
        return url
    if not model or model == os.environ.get("BAKED_MODEL"):
        return None
    owner, _, rest = model.partition("/")
    repo, _, filename = rest.partition("/")
    if not (owner and repo and filename.endswith(".gguf")):
        _fail(f"{what} must be <owner>/<repo>/<file>.gguf, got {model!r}")
    return f"https://huggingface.co/{owner}/{repo}/resolve/main/{urllib.parse.quote(filename)}"


def _download(url, dest):
    """Fetch url to dest through a per-process '.part' file (workers sharing a
    network volume may download the same model at once), resuming after a
    dropped connection, then rename it into place."""
    part = f"{dest}.{os.getpid()}.part"
    headers = {"User-Agent": "reflex-runpod-worker"}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"
    total = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        request = urllib.request.Request(
            url, headers=dict(headers, Range=f"bytes={have}-") if have else headers
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # the server ignored the range: start over
                length = resp.headers.get("Content-Length")
                if length is not None:
                    total = have + int(length)
                    free = shutil.disk_usage(os.path.dirname(dest)).free
                    if total - have > free:
                        _fail(
                            f"{url} needs {(total - have) / 1e9:.1f} GB but only {free / 1e9:.1f} GB "
                            f"is free in {os.path.dirname(dest)}: raise the container disk "
                            "or attach a network volume"
                        )
                started = last_report = time.monotonic()
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            last_report = now
                            of = f" of {total / 1e9:.2f}" if total else ""
                            print(f"reflex worker: {have / 1e9:.2f}{of} GB downloaded", flush=True)
            if total is None or have == total:
                os.replace(part, dest)
                return
            print(f"reflex worker: download stopped at {have} of {total} bytes", flush=True)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                hint = " (gated or private repo: set HF_TOKEN)" if e.code != 404 else ""
                _fail(f"{url}: HTTP {e.code}{hint}")
            print(f"reflex worker: {url}: HTTP {e.code} (attempt {attempt})", flush=True)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"reflex worker: {url}: {e} (attempt {attempt})", flush=True)
        time.sleep(min(2**attempt, 30))
    _fail(f"{url}: download failed after {DOWNLOAD_ATTEMPTS} attempts")


def _cache_root(name):
    """Where downloads and indexes go: the network volume when one is attached, so
    later workers reuse them, else the container disk."""
    if os.path.isdir(VOLUME_DIR):
        return os.path.join(VOLUME_DIR, name)
    return os.path.join("/models", name)


def _resolve_model(model, url="", what="MODEL"):
    """Path of the GGUF to serve: the baked one, or the selected one, downloaded once."""
    url = _model_url(model, url, what)
    if url is None:
        return os.environ["GGUF_PATH"]
    # One directory per URL, so the file keeps its own name (the API's `model`
    # field is its stem) and two URLs with the same file name don't collide.
    cache = os.path.join(_cache_root("reflex-models"), hashlib.sha256(url.encode()).hexdigest()[:12])
    os.makedirs(cache, exist_ok=True)
    name = os.path.basename(urllib.parse.urlparse(url).path) or "model.gguf"
    dest = os.path.join(cache, name)
    if os.path.exists(dest):
        print(f"reflex worker: using cached {dest}", flush=True)
        return dest
    print(f"reflex worker: downloading {url} to {dest}", flush=True)
    started = time.monotonic()
    _download(url, dest)
    print(f"reflex worker: downloaded in {time.monotonic() - started:.1f} s", flush=True)
    return dest


def _wait_for_ready(process, base):
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(
                f"reflex-openai-adapter exited early with code {process.returncode}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2) as resp:
                if resp.status == 200:
                    return
                # 204: still loading the model, keep polling.
        except urllib.error.HTTPError as e:
            if e.code == 503:
                print("reflex-openai-adapter reports its managed process died", file=sys.stderr)
                sys.exit(1)
        except (urllib.error.URLError, ConnectionError):
            pass  # adapter not accepting connections yet
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"reflex-openai-adapter did not become ready within {READY_TIMEOUT_SECONDS}s", file=sys.stderr)
    sys.exit(1)


def _watchdog(process):
    process.wait()
    print(
        f"reflex-openai-adapter exited unexpectedly with code {process.returncode}",
        file=sys.stderr,
    )
    os._exit(1)


def _spawn_adapter(gguf_path, port, extra_args):
    """Start one reflex-openai-adapter on a loopback port and wait until it is
    ready; if it ever exits, the watchdog takes the whole worker down."""
    process = subprocess.Popen(
        ["reflex-openai-adapter", gguf_path, "--host", ADAPTER_HOST, "--port", str(port)]
        + extra_args,
    )
    _wait_for_ready(process, f"http://{ADAPTER_HOST}:{port}")
    threading.Thread(target=_watchdog, args=(process,), daemon=True).start()


def _post_embeddings(body):
    """POST body to the embedding adapter's /v1/embeddings and return its JSON."""
    request = urllib.request.Request(
        f"{EMBED_BASE}/v1/embeddings",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise rag.RagError(f"embedding server: HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise rag.RagError(f"embedding server: {e}") from None


def _start_rag():
    """Start the embedding adapter, then load or build the corpus index."""
    global _rag_cfg, _rag_index
    try:
        cfg = rag.config_from_env(os.environ, DEFAULT_EMBED_MODEL)
    except rag.RagError as e:
        _fail(str(e))
    if not os.environ.get("GGUF_URL", "").strip() and os.environ.get("MODEL", "").strip() == cfg["embed_model"]:
        _fail("RAG needs a chat model in Model; the embedding model is RAG_EMBED_MODEL")
    embed_path = _resolve_model(cfg["embed_model"], what="RAG_EMBED_MODEL")
    _spawn_adapter(embed_path, EMBED_PORT, [])

    cache_root = _cache_root("reflex-rag")
    sources = []
    if os.path.isdir(cfg["dir"]):
        sources.append((cfg["dir"], "dir"))
    elif cfg["dir"] != rag.DEFAULT_DIR:
        # The default folder is optional (the deploy form always sends it); one the
        # user named must exist.
        _fail(f"RAG_DIR {cfg['dir']} does not exist (is the network volume attached?)")
    try:
        if cfg["corpus_url"]:
            sources.append((rag.fetch_corpus(cfg["corpus_url"], cache_root, _download, _log), "url"))
        if sources:
            started = time.monotonic()
            _rag_index = rag.build_or_load_index(
                sources, cfg, cache_root, _post_embeddings, _log, exclude=(_cache_root("reflex-models"),)
            )
            if _rag_index is not None:
                _log(f"corpus index ready in {time.monotonic() - started:.1f} s ({len(_rag_index)} chunks)")
        else:
            _log("no corpus configured; only per-request documents will be searched")
    except rag.RagError as e:
        _fail(f"RAG corpus: {e}")
    _rag_cfg = cfg


def _start_adapters():
    if rag.env_flag(os.environ.get("RAG", "off")):
        _start_rag()
    gguf_path = _resolve_model(os.environ.get("MODEL", "").strip(), os.environ.get("GGUF_URL", "").strip())
    # Optional extra adapter flags (e.g. "--max-tokens-cap 1024"); unset = the
    # adapter's own request-limit defaults. Its default --request-timeout-secs (300)
    # matches this shim's own urlopen timeout below.
    extra_args = shlex.split(os.environ.get("ADAPTER_ARGS", ""))
    _spawn_adapter(gguf_path, ADAPTER_PORT, extra_args)


def _sse_events(resp):
    """Yield each SSE event's data payload until the adapter's [DONE] line."""
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line.startswith("data:"):
            continue  # blank separators; the adapter sends no other SSE fields
        data = line[len("data:"):].lstrip(" ")
        if data == "[DONE]":
            return
        yield json.loads(data)


def handler(event):
    payload = dict(event.get("input", {}))
    base, rag_info = ADAPTER_BASE, None
    if "rag" in payload:
        if _rag_cfg is None:
            yield {"error": 'this worker has RAG off: redeploy with RAG set to "on" to send "rag" jobs'}
            return
        try:
            payload, rag_info = rag.prepare(payload, _rag_cfg, _rag_index, _post_embeddings)
        except rag.RagError as e:
            yield {"error": str(e)}
            return
    if "labels" in payload:
        path, stream = "/v1/classify", False
    elif "input" in payload and "messages" not in payload:
        path, stream = "/v1/embeddings", False
        if _rag_cfg is not None:
            base = EMBED_BASE  # a RAG worker's embedding model
    else:
        path, stream = "/v1/chat/completions", bool(payload.get("stream", False))
        payload["stream"] = stream

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # The timeout is per socket read, so a long stream is fine as long as
        # tokens keep arriving; the adapter's own --request-timeout-secs bounds
        # the whole job.
        with urllib.request.urlopen(request, timeout=300) as resp:
            if not stream:
                result = json.loads(resp.read())
                if rag_info is not None:
                    result["rag"] = rag_info
                yield result
                return
            if rag_info is not None:
                # The passages first, so a streaming client can show its sources
                # before the answer.
                yield dict(object="rag.context", **rag_info)
            for chunk in _sse_events(resp):
                if "error" in chunk:
                    # A mid-stream error event (e.g. the adapter's timeout).
                    yield {"error": json.dumps(chunk["error"])}
                    return
                yield chunk
    except urllib.error.HTTPError as e:
        yield {"error": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        yield {"error": str(e)}


if __name__ == "__main__":
    _start_adapters()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
