"""Host-only tests for compliance.py and its wiring in handler.py (no GPU, no
adapter): python3 -m unittest discover .runpod

handler.py is imported with a stub `runpod` module (only its __main__ block uses the
SDK), and the adapter's HTTP API and Hugging Face downloads are faked by patching
urllib.request.urlopen.
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from unittest import mock

try:
    import runpod  # noqa: F401
except ImportError:
    sys.modules["runpod"] = types.ModuleType("runpod")

import compliance
import handler

HERE = os.path.dirname(os.path.abspath(__file__))


def _log(_):
    pass


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None, lines=None, fail_after=None):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}
        self.lines = lines
        self.fail_after = fail_after
        self.sent = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise ConnectionError("connection reset")
        if self.fail_after is not None and n > 0:
            n = min(n, self.fail_after - self.sent)
        data = self.body.read(n)
        self.sent += len(data)
        return data

    def __iter__(self):
        return iter(self.lines or [])


class FakeServer:
    """Serves one file with Range support, dropping the first connection after
    `drop_at` bytes; with ignore_range it answers every request with the whole file."""

    def __init__(self, data, drop_at=None, ignore_range=False):
        self.data = data
        self.drop_at = drop_at
        self.ignore_range = ignore_range
        self.ranges = []

    def __call__(self, request, timeout=None):
        rng = request.get_header("Range")
        self.ranges.append(rng)
        start = int(rng[len("bytes=") : -1]) if rng and not self.ignore_range else 0
        body = self.data[start:]
        fail_after = self.drop_at if len(self.ranges) == 1 else None
        return FakeResponse(
            body,
            status=206 if start else 200,
            headers={"Content-Length": str(len(body))},
            fail_after=fail_after,
        )


@contextlib.contextmanager
def handler_state(**values):
    """Set handler's module globals for one test, restoring them afterwards."""
    names = ("_attestation", "_audit", "_audit_content", "_model_sha256", "_rag_cfg", "_rag_index", "_hash_downloads")
    saved = {n: getattr(handler, n) for n in names}
    try:
        for n, v in values.items():
            assert n in names, n
            setattr(handler, n, v)
        yield
    finally:
        for n, v in saved.items():
            setattr(handler, n, v)


CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there friend"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
CHUNKS = [
    {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
    for t in ("Hel", "lo")
]
# The adapter's real /v1/classify shape (checked against the real engine): no usage.
CLASSIFY = {
    "id": "classify-reflex-0",
    "object": "classification",
    "created": 1,
    "model": "m",
    "label": " negative",
    "label_index": 1,
    "labels": [
        {"label": " positive", "probability": 0.1, "score": 13.2, "tokens": 1},
        {"label": " negative", "probability": 0.9, "score": 14.6, "tokens": 1},
    ],
    "entropy": 0.47,
}
EMBED = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.5]}], "model": "m", "usage": {"prompt_tokens": 3, "total_tokens": 3}}


def fake_adapter(request, timeout=None):
    """The adapter's three endpoints; a chat job whose last message is "fail"
    gets an HTTP 400."""
    body = json.loads(request.data)
    url = request.full_url
    if url.endswith("/v1/classify"):
        return FakeResponse(json.dumps(CLASSIFY).encode())
    if url.endswith("/v1/embeddings"):
        return FakeResponse(json.dumps(EMBED).encode())
    if body["messages"][-1]["content"] == "fail":
        raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"error":"bad request"}'))
    if body["stream"]:
        lines = [f"data: {json.dumps(c)}\n".encode() for c in CHUNKS] + [b"\n", b"data: [DONE]\n"]
        return FakeResponse(lines=lines)
    return FakeResponse(json.dumps(CHAT).encode())


JOBS = {
    "chat": {"id": "j1", "input": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}},
    "stream": {"id": "j2", "input": {"messages": [{"role": "user", "content": "hi"}], "stream": True}},
    "classify": {"id": "j3", "input": {"prompt": "p", "labels": [" positive", " negative"]}},
    "embed": {"id": "j4", "input": {"input": ["text"]}},
    "error": {"id": "j5", "input": {"messages": [{"role": "user", "content": "fail"}]}},
}

# What every job returned before compliance.py existed (and must still return with
# every option off).
EXPECTED_OFF = {
    "chat": [CHAT],
    "stream": CHUNKS,
    "classify": [CLASSIFY],
    "embed": [EMBED],
    "error": [{"error": '{"error":"bad request"}'}],
}


def run_job(job):
    with mock.patch("urllib.request.urlopen", fake_adapter):
        return list(handler.handler(job))


STATIC = compliance.static_block(
    {"id": "o/r/m.gguf", "file": "m.gguf", "sha256": "ab" * 32, "license": "apache-2.0"},
    {"REFLEX_IMAGE_DIGEST": "sha256:feed"},
    "NVIDIA GeForce RTX 4090",
    {"reflex_engine_crate_version": "0.2.0"},
)


class ConfigTests(unittest.TestCase):
    def test_everything_defaults_to_off(self):
        cfg = compliance.config_from_env({})
        self.assertEqual(cfg, {"compliance": False, "audit_log": "off", "audit_content": "off"})
        self.assertFalse(compliance.enabled(cfg))
        # The deploy form sends every key, with its default.
        form = {"COMPLIANCE": "off", "AUDIT_LOG": "off", "AUDIT_LOG_CONTENT": "off"}
        self.assertFalse(compliance.enabled(compliance.config_from_env(form)))

    def test_values_are_checked(self):
        cfg = compliance.config_from_env({"COMPLIANCE": "on", "AUDIT_LOG": "Volume", "AUDIT_LOG_CONTENT": "hash"})
        self.assertEqual(cfg, {"compliance": True, "audit_log": "volume", "audit_content": "hash"})
        with self.assertRaises(compliance.ComplianceError):
            compliance.config_from_env({"AUDIT_LOG": "file"})
        with self.assertRaises(compliance.ComplianceError):
            compliance.config_from_env({"AUDIT_LOG": "stdout", "AUDIT_LOG_CONTENT": "everything"})
        with self.assertRaisesRegex(compliance.ComplianceError, "needs AUDIT_LOG"):
            compliance.config_from_env({"AUDIT_LOG_CONTENT": "full"})


class DownloadHashTests(unittest.TestCase):
    DATA = bytes(range(256)) * 40_000  # ~10 MB: more than one DOWNLOAD_CHUNK

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "model.gguf")
        self.expected = hashlib.sha256(self.DATA).hexdigest()
        patcher = mock.patch("time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def download(self, server, sha256=True):
        with mock.patch("urllib.request.urlopen", server):
            handler._download("https://example.com/model.gguf", self.dest, sha256=sha256)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.DATA)

    def test_fresh_download_saves_its_hash(self):
        self.download(FakeServer(self.DATA))
        self.assertEqual(compliance.read_sha256(self.dest), self.expected)
        with open(compliance.sidecar(self.dest)) as f:
            self.assertEqual(f.read(), f"{self.expected}  model.gguf\n")  # sha256sum -c format

    def test_resumed_download_hashes_the_whole_file(self):
        server = FakeServer(self.DATA, drop_at=3_000_000)
        self.download(server)
        self.assertEqual(server.ranges, [None, "bytes=3000000-"])
        self.assertEqual(compliance.read_sha256(self.dest), self.expected)

    def test_a_part_file_already_on_disk_is_rehashed_once(self):
        with open(f"{self.dest}.{os.getpid()}.part", "wb") as f:
            f.write(self.DATA[:1_234_567])
        server = FakeServer(self.DATA)
        self.download(server)
        self.assertEqual(server.ranges, ["bytes=1234567-"])
        self.assertEqual(compliance.read_sha256(self.dest), self.expected)

    def test_a_server_that_ignores_the_range_restarts_the_hash(self):
        server = FakeServer(self.DATA, drop_at=2_000_000, ignore_range=True)
        self.download(server)
        self.assertEqual(compliance.read_sha256(self.dest), self.expected)

    def test_no_hash_without_a_compliance_option(self):
        self.download(FakeServer(self.DATA), sha256=False)
        self.assertFalse(os.path.exists(compliance.sidecar(self.dest)))

    def test_cached_file_is_hashed_once_then_read_back(self):
        with open(self.dest, "wb") as f:
            f.write(self.DATA)
        self.assertEqual(compliance.file_sha256(self.dest, _log), self.expected)
        with mock.patch("hashlib.sha256", side_effect=AssertionError("hashed again")):
            self.assertEqual(compliance.file_sha256(self.dest, _log), self.expected)

    def test_stale_or_malformed_sidecar_is_ignored(self):
        with open(self.dest, "wb") as f:
            f.write(self.DATA)
        compliance.write_sha256(self.dest, "0" * 64)
        st = os.stat(self.dest)
        os.utime(compliance.sidecar(self.dest), ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
        self.assertIsNone(compliance.read_sha256(self.dest), "older than the file")
        with open(compliance.sidecar(self.dest), "w") as f:
            f.write("not a hash\n")
        self.assertIsNone(compliance.read_sha256(self.dest))
        self.assertEqual(compliance.file_sha256(self.dest, _log), self.expected)


def fake_model_info(path):
    info = {
        "schema_version": "1.4.0",
        "path": path,
        "size_bytes": os.path.getsize(path),
        "architecture": "qwen3",
        "file_type": 7,
        "file_type_name": "Q8_0",
        "general": {"name": "Qwen3 0.6B", "license": "apache-2.0", "sampling.top_k": "20", "quantization_version": "2"},
    }

    def run(argv, **_):
        assert argv == ["reflex", "model-info", path, "--json"], argv
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr="")

    return run


class ModelTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "m.gguf")
        with open(self.path, "wb") as f:
            f.write(b"GGUF fake model")
        self.digest = hashlib.sha256(b"GGUF fake model").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_models_json_covers_the_dropdown(self):
        with open(os.path.join(HERE, "hub.json"), encoding="utf-8") as f:
            env = {e["key"]: e["input"] for e in json.load(f)["config"]["env"]}
        options = {o["value"] for o in env["MODEL"]["options"]}
        models = compliance.load_models()
        self.assertEqual(set(models), options)
        for model_id, entry in models.items():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", model_id)
            self.assertTrue(entry["license"] and entry["license_url"].startswith("https://"), model_id)
            self.assertTrue(entry["source"].startswith("https://huggingface.co/"), model_id)
        self.assertIn(env["RAG_EMBED_MODEL"]["default"], models)

    def test_matching_hash_is_described(self):
        models = {"o/r/m.gguf": {"license": "apache-2.0", "license_url": "https://x", "source": "https://hf/o/r", "sha256": self.digest}}
        out = compliance.describe_model(self.path, "o/r/m.gguf", "", models, _log, run=fake_model_info(self.path))
        self.assertEqual(out["sha256"], self.digest)
        self.assertTrue(out["sha256_matches_models_json"])
        self.assertEqual(out["license"], "apache-2.0")
        self.assertEqual(out["file_type"], "Q8_0")
        # Provenance keys only: sampling defaults and the like are left out.
        self.assertEqual(out["gguf_general"], {"name": "Qwen3 0.6B", "license": "apache-2.0"})

    def test_mismatched_hash_fails(self):
        models = {"o/r/m.gguf": {"license": "apache-2.0", "sha256": "0" * 64, "revision": "abc"}}
        with self.assertRaisesRegex(compliance.ComplianceError, "does not match"):
            compliance.describe_model(self.path, "o/r/m.gguf", "", models, _log, run=fake_model_info(self.path))

    def test_mismatch_stops_the_worker(self):
        env = {"MODEL": "o/r/m.gguf", "GGUF_URL": ""}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            compliance, "load_models", return_value={"o/r/m.gguf": {"license": "x", "sha256": "0" * 64}}
        ), handler_state(), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                handler._start_compliance(compliance.config_from_env({"COMPLIANCE": "on"}), self.path, None)

    def test_user_supplied_and_unlisted_models(self):
        url_model = compliance.describe_model(
            self.path, "", "https://example.com/m.gguf", {}, _log, run=fake_model_info(self.path)
        )
        self.assertEqual(url_model["license"], compliance.USER_SUPPLIED)
        self.assertIsNone(url_model["sha256_matches_models_json"])
        self.assertEqual(url_model["url"], "https://example.com/m.gguf")
        unlisted = compliance.describe_model(self.path, "a/b/c.gguf", "", {}, _log, run=fake_model_info(self.path))
        self.assertEqual(unlisted["license"], compliance.NOT_LISTED)

    def test_gpu_name(self):
        def ok(*a, **k):
            return types.SimpleNamespace(returncode=0, stdout="NVIDIA L4\n", stderr="")

        self.assertEqual(compliance.gpu_name(ok), "NVIDIA L4")

        def missing(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        self.assertIsNone(compliance.gpu_name(missing))


class AllOffTests(unittest.TestCase):
    def test_responses_are_unchanged(self):
        with handler_state(_attestation=None, _audit=None):
            for name, job in JOBS.items():
                out = run_job(job)
                self.assertEqual(json.dumps(out), json.dumps(EXPECTED_OFF[name]), name)

    def test_compliance_key_on_an_ordinary_job_is_forwarded_as_before(self):
        job = {"id": "j", "input": dict(JOBS["chat"]["input"], compliance="report")}
        with handler_state(_attestation=None, _audit=None):
            self.assertEqual(run_job(job), [CHAT])

    def test_report_request_is_refused(self):
        with handler_state(_attestation=None, _audit=None):
            out = run_job({"id": "j", "input": {"compliance": "report"}})
        self.assertIn("COMPLIANCE off", out[0]["error"])


class AttestationTests(unittest.TestCase):
    def test_non_streaming_jobs_carry_the_block(self):
        with handler_state(_attestation=STATIC, _audit=None):
            for name in ("chat", "classify", "embed"):
                (out,) = run_job(JOBS[name])
                block = out.pop("compliance")
                self.assertEqual(out, EXPECTED_OFF[name][0], "the answer itself is unchanged")
                self.assertEqual(block["schema"], compliance.SCHEMA)
                self.assertEqual(block["model"]["sha256"], "ab" * 32)
                self.assertEqual(block["engine"], {"reflex_engine_crate_version": "0.2.0", "image_digest": "sha256:feed", "compliance_bundle": True})
                self.assertEqual(block["gpu"], "NVIDIA GeForce RTX 4090")
                self.assertEqual(block["job"]["endpoint"], {"chat": "chat", "classify": "classify", "embed": "embed"}[name])
                if name == "classify":
                    self.assertNotIn("usage", block["job"], "a classify response has no usage to copy")
                else:
                    self.assertEqual(block["job"]["usage"], EXPECTED_OFF[name][0]["usage"])
                self.assertGreaterEqual(block["job"]["wall_time_ms"], 0)

    def test_stream_ends_with_an_attestation_event(self):
        with handler_state(_attestation=STATIC, _audit=None):
            out = run_job(JOBS["stream"])
        self.assertEqual(out[:-1], CHUNKS)
        last = out[-1]
        self.assertEqual(last["object"], "compliance.attestation")
        self.assertEqual(last["job"]["stream_chunks"], 2)
        self.assertEqual(last["model"], STATIC["model"])

    def test_errors_are_passed_through(self):
        with handler_state(_attestation=STATIC, _audit=None):
            self.assertEqual(run_job(JOBS["error"]), EXPECTED_OFF["error"])

    def test_rag_sources(self):
        info = {"passages": [{"doc": "a.md", "origin": "dir"}, {"doc": "a.md", "origin": "dir"}, {"doc": "q", "origin": "request"}]}
        block = compliance.job_attestation(STATIC, {"endpoint": "chat", "rag_info": info}, 1.0)
        self.assertEqual(block["job"]["rag_sources"], [{"origin": "dir", "doc": "a.md"}, {"origin": "request", "doc": "q"}])

    def test_report_job(self):
        with tempfile.TemporaryDirectory() as bundle:
            with open(os.path.join(bundle, "reflex.cdx.json"), "w") as f:
                json.dump({"bomFormat": "CycloneDX"}, f)
            with open(os.path.join(bundle, "pip-freeze.txt"), "w") as f:
                f.write("runpod==1.7.9\n")
            with handler_state(_attestation=STATIC, _audit=None), mock.patch.object(compliance, "BUNDLE_DIR", bundle):
                (out,) = run_job({"id": "r", "input": {"compliance": "report"}})
        self.assertEqual(out["object"], "compliance.report")
        self.assertEqual(out["files"]["reflex.cdx.json"], {"bomFormat": "CycloneDX"})
        self.assertEqual(out["files"]["pip-freeze.txt"], "runpod==1.7.9\n")
        self.assertIn("THIRD_PARTY_LICENSES.html", out["missing"])
        self.assertNotIn("reflex.cdx.json", out["missing"])
        self.assertIn("ab" * 32, out["model_card"])
        self.assertIn("not legal advice", out["model_card"])
        self.assertEqual(out["attestation"], STATIC)

    def test_bad_report_value(self):
        with handler_state(_attestation=STATIC, _audit=None):
            (out,) = run_job({"id": "r", "input": {"compliance": "everything"}})
        self.assertIn('must be "report"', out["error"])


class AuditTests(unittest.TestCase):
    def jobs_with_audit(self, content, attestation=None):
        buf = io.StringIO()
        audit = compliance.AuditLog("stdout", "/nonexistent", {}, out=buf)
        outputs = {}
        with handler_state(_attestation=attestation, _audit=audit, _audit_content=content, _model_sha256="cd" * 32):
            for name, job in JOBS.items():
                outputs[name] = run_job(job)
        lines = [json.loads(line) for line in buf.getvalue().splitlines()]
        return outputs, lines

    def test_one_line_per_job_and_responses_unchanged(self):
        outputs, lines = self.jobs_with_audit("off")
        for name in JOBS:
            self.assertEqual(json.dumps(outputs[name]), json.dumps(EXPECTED_OFF[name]), name)
        self.assertEqual([line["job_id"] for line in lines], ["j1", "j2", "j3", "j4", "j5"])
        self.assertEqual([line["endpoint"] for line in lines], ["chat", "chat", "classify", "embed", "chat"])
        self.assertEqual([line["status"] for line in lines], ["ok"] * 4 + ["error"])
        self.assertEqual(lines[4]["error_class"], "http_400")
        self.assertEqual(lines[0]["usage"], CHAT["usage"])
        self.assertEqual(lines[1]["stream_chunks"], 2)
        self.assertNotIn("usage", lines[2])
        self.assertEqual(lines[3]["usage"], EMBED["usage"])
        for line in lines:
            self.assertEqual(line["schema"], compliance.AUDIT_SCHEMA)
            self.assertEqual(line["model_sha256"], "cd" * 32)
            self.assertRegex(line["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
            self.assertGreaterEqual(line["latency_ms"], 0)
            for key in ("prompt", "output", "prompt_sha256", "output_sha256"):
                self.assertNotIn(key, line)

    def test_hash_mode(self):
        _, lines = self.jobs_with_audit("hash")
        chat, stream, classify, embed, _ = lines
        self.assertEqual(chat["prompt_sha256"], compliance.sha256_of(JOBS["chat"]["input"]))
        self.assertEqual(chat["output_sha256"], hashlib.sha256(b"Hello there friend").hexdigest())
        self.assertEqual(stream["output_sha256"], hashlib.sha256(b"Hello").hexdigest())
        self.assertRegex(classify["output_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(embed["output_sha256"], "vectors are never logged")
        self.assertNotIn("prompt", chat)

    def test_full_mode(self):
        _, lines = self.jobs_with_audit("full")
        chat, stream, classify, embed, _ = lines
        self.assertEqual(chat["prompt"], JOBS["chat"]["input"])
        self.assertEqual(chat["output"], "Hello there friend")
        self.assertEqual(stream["output"], "Hello")
        self.assertEqual(classify["output"]["label"], " negative")
        self.assertIsNone(embed["output"])

    def test_audit_with_attestation(self):
        outputs, lines = self.jobs_with_audit("off", attestation=STATIC)
        self.assertEqual(len(lines), 5)
        self.assertIn("compliance", outputs["chat"][0])

    def test_volume_mode_writes_one_file_per_worker(self):
        with tempfile.TemporaryDirectory() as volume:
            audit = compliance.AuditLog("volume", volume, {"RUNPOD_POD_ID": "abc/123"})
            audit.write({"n": 1})
            audit.write({"n": 2})
            path = os.path.join(volume, "reflex-audit", "abc_123.jsonl")
            with open(path, encoding="utf-8") as f:
                self.assertEqual([json.loads(line) for line in f], [{"n": 1}, {"n": 2}])

    def test_volume_mode_needs_a_volume(self):
        with self.assertRaisesRegex(compliance.ComplianceError, "network volume"):
            compliance.AuditLog("volume", os.path.join(tempfile.gettempdir(), "no-such-volume-xyz"), {})


if __name__ == "__main__":
    unittest.main()
