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

ADAPTER_HOST = "127.0.0.1"
ADAPTER_PORT = 8000
ADAPTER_BASE = f"http://{ADAPTER_HOST}:{ADAPTER_PORT}"
READY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 0.25

VOLUME_DIR = "/runpod-volume"
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_CHUNK = 8 << 20

_adapter_process = None


def _fail(message):
    print(f"reflex worker: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def _model_url():
    """The GGUF to download, or None to use the baked model."""
    url = os.environ.get("GGUF_URL", "").strip()
    if url:
        if not url.startswith(("https://", "http://")):
            _fail(f"GGUF_URL must be an http(s) URL, got {url!r}")
        return url
    model = os.environ.get("MODEL", "").strip()
    if not model or model == os.environ.get("BAKED_MODEL"):
        return None
    owner, _, rest = model.partition("/")
    repo, _, filename = rest.partition("/")
    if not (owner and repo and filename.endswith(".gguf")):
        _fail(f"MODEL must be <owner>/<repo>/<file>.gguf, got {model!r}")
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


def _resolve_model():
    """Path of the GGUF to serve: the baked one, or the selected one, downloaded once."""
    url = _model_url()
    if url is None:
        return os.environ["GGUF_PATH"]
    root = os.path.join(VOLUME_DIR, "reflex-models") if os.path.isdir(VOLUME_DIR) else "/models"
    # One directory per URL, so the file keeps its own name (the API's `model`
    # field is its stem) and two URLs with the same file name don't collide.
    cache = os.path.join(root, hashlib.sha256(url.encode()).hexdigest()[:12])
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


def _wait_for_ready():
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _adapter_process.poll() is not None:
            print(
                f"reflex-openai-adapter exited early with code {_adapter_process.returncode}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            with urllib.request.urlopen(f"{ADAPTER_BASE}/healthz", timeout=2) as resp:
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


def _watchdog():
    _adapter_process.wait()
    print(
        f"reflex-openai-adapter exited unexpectedly with code {_adapter_process.returncode}",
        file=sys.stderr,
    )
    os._exit(1)


def _start_adapter():
    global _adapter_process
    gguf_path = _resolve_model()
    # Optional extra adapter flags (e.g. "--max-tokens-cap 1024"); unset = the
    # adapter's own request-limit defaults. Its default --request-timeout-secs (300)
    # matches this shim's own urlopen timeout below.
    extra_args = shlex.split(os.environ.get("ADAPTER_ARGS", ""))
    _adapter_process = subprocess.Popen(
        ["reflex-openai-adapter", gguf_path, "--host", ADAPTER_HOST, "--port", str(ADAPTER_PORT)]
        + extra_args,
    )
    _wait_for_ready()
    threading.Thread(target=_watchdog, daemon=True).start()


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
    if "labels" in payload:
        path, stream = "/v1/classify", False
    elif "input" in payload and "messages" not in payload:
        path, stream = "/v1/embeddings", False
    else:
        path, stream = "/v1/chat/completions", bool(payload.get("stream", False))
        payload["stream"] = stream

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{ADAPTER_BASE}{path}",
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
                yield json.loads(resp.read())
                return
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
    _start_adapter()
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
