"""Modal Phase 3 comparison baseline: plain vLLM serving Qwen3-0.6B on Modal.

This is the *repeatable* counterpart to Phase 0's one-off check
(docs/modal-cold-start-phase0.md) -- same model, same GPU, same "deliberately
plain" config (no FAST_BOOT/--enforce-eager, no HF-weights caching volume, no
pre-warming, no min_containers), but committed as a script so Phase 3 can run
it multiple times for a real p50/p95 sample instead of citing Phase 0's n=1
run. Mirrors Modal's own official llm_inference example pattern
(Image.from_registry(...).entrypoint([]).uv_pip_install(...), `vllm serve` via
subprocess.Popen inside @app.server) -- the natural default a user deploying
vLLM on Modal would follow, not a strawman.

Deliberately no modal.Volume for the Hugging Face cache: caching would make
later runs artificially faster than Phase 0's genuinely-cold number, and
Phase 0's own setup explicitly excluded pre-warming/caching. Each `modal run`
here pays a real weights download, matching Phase 0's methodology exactly so
this is a fair repeat, not a friendlier rerun.

Run from anywhere (no build-context requirement -- from_registry, not
from_dockerfile):

    PYTHONUTF8=1 python -m modal run .modal/app_vllm.py
"""

import json
import time
import urllib.request

import modal

MODEL_NAME = "Qwen/Qwen3-0.6B"
PORT = 8000

app = modal.App("reflex-modal-phase3-vllm")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0")
)


@app.server(
    image=image,
    gpu="L4",
    port=PORT,
    unauthenticated=True,
    startup_timeout=300,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL_NAME,
                "--served-model-name",
                "reflex",
                "--host",
                "0.0.0.0",
                "--port",
                str(PORT),
            ]
        )

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def main(test_timeout: int = 280):
    """Same client-side methodology as .modal/app.py's Phase 1 entrypoint and
    Phase 0's check: `modal run` against a fresh ephemeral App guarantees no
    warm-container reuse on the first call, so this is a genuine cold start
    measured end to end from the calling machine.
    """
    url = Server.get_url()
    print(f"Server URL: {url}")

    t0 = time.monotonic()
    deadline = t0 + test_timeout
    healthy = False
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if not healthy:
        raise RuntimeError(f"server never became healthy within {test_timeout}s")
    t1 = time.monotonic()
    print(f"cold start (local submit -> /health 200): {t1 - t0:.1f}s")

    payload = json.dumps(
        {
            "model": "reflex",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 8,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t2 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    t3 = time.monotonic()
    print(f"first completion latency (post-healthy): {t3 - t2:.2f}s")
    print(f"total (local submit -> first token): {t3 - t0:.1f}s")
    print(body.decode())
