"""Runpod Hub queue-worker entrypoint for Reflex.

Translation shim only, no engine logic: spawns the same, unmodified
reflex-openai-adapter binary that serverless/runpod/ uses as a load-balancing
endpoint, waits for it to report ready, and forwards each Runpod job to its
existing POST /v1/chat/completions over loopback -- or, for a job whose input
has "labels", to POST /v1/classify (one engine pass scoring each label; the
one response is yielded once, like a non-streaming chat job). See
../.runpod/README.md and ../serverless/runpod/README.md for why two deployment
paths exist.

The handler is a generator, so Runpod's /stream/{job_id} works. With
"stream": true in the job input, each Server-Sent Event the adapter sends is
yielded as one chat.completion.chunk dict; otherwise the single complete
chat.completion is yielded once. return_aggregate_stream makes /run and
/runsync return those yields as a list, so a non-streaming job's output is a
one-element list (the same shape runpod-workers/worker-vllm returns).
"""

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import runpod

ADAPTER_HOST = "127.0.0.1"
ADAPTER_PORT = 8000
ADAPTER_BASE = f"http://{ADAPTER_HOST}:{ADAPTER_PORT}"
READY_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 0.25

_adapter_process = None


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
    gguf_path = os.environ["GGUF_PATH"]
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
