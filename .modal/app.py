"""Modal Phase 1 deployment for Reflex.

Packages the existing, unmodified reflex + reflex-openai-adapter binaries (built by
Dockerfile in this directory) as a Modal Server -- the same "no new engine code, just
packaging" pattern serverless/runpod/ and sidecar/openai-adapter/ already use. See
docs/modal-cold-start-phase0.md for why Modal specifically, and this directory's
README.md for real measured numbers once they exist.

Model: the same Qwen3-0.6B-Q4_K_M baked into serverless/runpod/model.gguf, for the same
reason it's baked into every other deployment in this project -- a runtime download
would add HuggingFace-fetch latency straight into the cold-start number this deployment
exists to measure.

GPU: L4 (Ada Lovelace, sm_89) -- matches Phase 0's own check (docs/modal-cold-start-
phase0.md) exactly, so the plain-vLLM-vs-Reflex comparison is apples-to-apples on the
same GPU type, and matches Dockerfile's REFLEX_CUDA_ARCH default.

Run from the repo root (build context requirement -- see Dockerfile's own comment):

    python -m modal run .modal/app.py     # one-shot: deploy ephemerally, measure, tear down
    python -m modal deploy .modal/app.py  # persistent deployment
"""

import json
import time
import urllib.request
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = Path(__file__).resolve().parent / "Dockerfile"

PORT = 8000
GGUF_PATH = "/models/model.gguf"

app = modal.App("reflex-modal-phase1")

image = modal.Image.from_dockerfile(
    DOCKERFILE,
    context_dir=REPO_ROOT,
    build_args={"REFLEX_CUDA_ARCH": "sm_89"},
    # Dockerfile's runtime stage is a plain nvidia/cuda:*-runtime-* base with no
    # Python -- Modal's own container init needs an interpreter to run this file's
    # @modal.enter()/@modal.exit() lifecycle methods, so it must be added here since
    # the Dockerfile itself has no reason to install one otherwise.
    add_python="3.12",
)


@app.server(
    image=image,
    gpu="L4",
    port=PORT,
    unauthenticated=True,
    startup_timeout=120,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        self.process = subprocess.Popen(
            [
                "/usr/local/bin/reflex-openai-adapter",
                GGUF_PATH,
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
def main(test_timeout: int = 180):
    """One-shot cold-start measurement, mirroring Phase 0's methodology: `modal run`
    against a fresh ephemeral App guarantees no warm-container reuse is possible on the
    first call, so this measures a genuine cold start end to end from the calling
    machine, not just container-internal timing.
    """
    url = Server.get_url()
    print(f"Server URL: {url}")

    t0 = time.monotonic()
    deadline = t0 + test_timeout
    healthy = False
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=5) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if not healthy:
        raise RuntimeError(f"server never became healthy within {test_timeout}s")
    t1 = time.monotonic()
    print(f"cold start (local submit -> /healthz 200): {t1 - t0:.1f}s")

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
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    t3 = time.monotonic()
    print(f"first completion latency (post-healthy): {t3 - t2:.2f}s")
    print(f"total (local submit -> first token): {t3 - t0:.1f}s")
    print(body.decode())
