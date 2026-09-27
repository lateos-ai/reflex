"""Runpod Hub queue-worker entrypoint for Reflex.

Translation shim only, no engine logic: spawns the same, unmodified
reflex-openai-adapter binary that serverless/runpod/ uses as a load-balancing
endpoint, waits for it to report ready, and forwards each Runpod job to its
existing POST /v1/chat/completions over loopback. See ../.runpod/README.md and
../serverless/runpod/README.md for why two deployment paths exist.

v1 limitation: non-streaming only. Streaming would mean translating Server-Sent
Events into a Runpod generator handler -- real new logic, out of scope for a
shim whose only job is speaking Runpod's queue job-envelope format.
"""

import json
import os
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
    _adapter_process = subprocess.Popen(
        ["reflex-openai-adapter", gguf_path, "--host", ADAPTER_HOST, "--port", str(ADAPTER_PORT)],
    )
    _wait_for_ready()
    threading.Thread(target=_watchdog, daemon=True).start()


def handler(event):
    payload = dict(event.get("input", {}))
    payload["stream"] = False  # v1: non-streaming only, see module docstring

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{ADAPTER_BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        return {"error": str(e)}


_start_adapter()
runpod.serverless.start({"handler": handler})
