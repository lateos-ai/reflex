"""Host-only checks on hub.json and tests.json, and on how a failed constrained
(response_format) chat job reaches Runpod: python3 -m unittest discover .runpod

The Hub's test job passes when a job completes, so a schema job that came back as a
completed job carrying an error would prove nothing. runpod's SDK (1.12.0, pinned in
.runpod/Dockerfile) ends a generator handler's job as failed at the first yielded item
that has an "error" key, so these tests check that handler.py yields exactly such an
item for the adapter's grammar_dead_end, in both modes.
"""

import io
import json
import os
import sys
import types
import unittest
import urllib.error
from unittest import mock

try:
    import runpod  # noqa: F401
except ImportError:
    sys.modules["runpod"] = types.ModuleType("runpod")

import handler
from test_compliance import CHUNKS, FakeResponse

HERE = os.path.dirname(os.path.abspath(__file__))

# The engine's supported json_schema keywords (src/grammar.rs: ANNOTATIONS, VALIDATION,
# and $defs for local $ref targets).
SCHEMA_KEYWORDS = {
    "title", "description", "$comment", "$schema", "default", "examples",
    "type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "anyOf", "$ref", "$defs",
}  # fmt: skip


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def _schema_keywords(schema):
    """Every keyword used anywhere in a schema."""
    found = set()
    if isinstance(schema, dict):
        found |= set(schema)
        for key, value in schema.items():
            if key in ("properties", "$defs"):
                for sub in value.values():
                    found |= _schema_keywords(sub)
            elif key in ("items", "additionalProperties"):
                found |= _schema_keywords(value)
            elif key == "anyOf":
                for sub in value:
                    found |= _schema_keywords(sub)
    return found


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.config = _load("hub.json")["config"]
        self.inputs = {e["key"]: e["input"] for e in self.config["env"] if "input" in e}

    def test_presets_use_existing_keys_and_option_values(self):
        presets = self.config["presets"]
        self.assertTrue(presets)
        for preset in presets:
            self.assertTrue(preset["name"])
            for key, value in preset["defaults"].items():
                self.assertIn(key, self.inputs, f"{preset['name']}: unknown env key {key}")
                options = self.inputs[key].get("options")
                if options is not None:
                    values = [o["value"] for o in options]
                    self.assertIn(value, values, f"{preset['name']}: {key}={value!r}")

    def test_every_preset_sets_the_same_toggles(self):
        # So picking one preset after another leaves nothing of the first behind.
        keys = [set(p["defaults"]) for p in self.config["presets"]]
        for k in keys:
            self.assertEqual(k, keys[0])

    def test_rag_preset_does_not_serve_the_embedding_model(self):
        embed = self.inputs["RAG_EMBED_MODEL"]["default"]
        for preset in self.config["presets"]:
            d = preset["defaults"]
            if d.get("RAG") == "on":
                self.assertNotEqual(d["MODEL"], embed)

    def test_presets_avoid_models_that_misrate_the_sentiment_test(self):
        # .runpod/README.md: TinyLlama, Granite 350M and Gemma 4 12B rate the
        # sentiment smoke test positive.
        for preset in self.config["presets"]:
            model = preset["defaults"]["MODEL"].lower()
            for bad in ("tinyllama", "granite-4.0-h-350m", "gemma-4"):
                self.assertNotIn(bad, model, preset["name"])


class SchemaTestJobTests(unittest.TestCase):
    def test_json_schema_job_is_well_formed(self):
        jobs = [t for t in _load("tests.json")["tests"] if "response_format" in t["input"]]
        self.assertEqual(len(jobs), 1)
        job = jobs[0]["input"]
        fmt = job["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        schema = fmt["json_schema"]["schema"]
        self.assertLessEqual(_schema_keywords(schema), SCHEMA_KEYWORDS)
        self.assertNotIn("labels", job)  # would go to /v1/classify instead
        self.assertFalse(job.get("stream", False))
        self.assertGreaterEqual(job["max_tokens"], 128)

    def test_existing_tests_are_kept(self):
        names = {t["name"] for t in _load("tests.json")["tests"]}
        for name in ("smoke-chat-completion", "smoke-chat-completion-stream", "smoke-classify", "smoke-classify-routing"):
            self.assertIn(name, names)


DEAD_END = {
    "error": {
        "message": "json_schema: max_tokens ran out after 8 tokens, before the value was complete",
        "type": "invalid_request_error",
        "param": None,
        "code": "grammar_dead_end",
    }
}


def dead_end_adapter(request, timeout=None):
    """The adapter's two grammar_dead_end shapes: a 400 before any token, or (streaming)
    the tokens already sent followed by an error event and [DONE]."""
    body = json.loads(request.data)
    if not body["stream"]:
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(json.dumps(DEAD_END).encode()))
    lines = [f"data: {json.dumps(c)}\n".encode() for c in CHUNKS]
    lines += [f"data: {json.dumps(DEAD_END)}\n".encode(), b"data: [DONE]\n"]
    return FakeResponse(lines=lines)


def run_sdk_rule(items):
    """runpod 1.12.0's handle_job for a generator handler: the first item with an
    "error" key becomes the job result (status FAILED); otherwise the job completes."""
    for item in items:
        if item.get("error"):
            return "FAILED", item["error"]
    return "COMPLETED", items


class GrammarErrorTests(unittest.TestCase):
    def _run(self, stream):
        job = {
            "id": "j",
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": stream,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
            },
        }
        with mock.patch("urllib.request.urlopen", dead_end_adapter):
            return list(handler.handler(job))

    def test_non_streaming_dead_end_fails_the_job(self):
        items = self._run(stream=False)
        self.assertEqual(len(items), 1)
        status, error = run_sdk_rule(items)
        self.assertEqual(status, "FAILED")
        self.assertEqual(json.loads(error)["error"]["code"], "grammar_dead_end")

    def test_streaming_dead_end_fails_the_job_after_the_sent_tokens(self):
        items = self._run(stream=True)
        self.assertEqual(items[:-1], CHUNKS)
        status, error = run_sdk_rule(items)
        self.assertEqual(status, "FAILED")
        self.assertEqual(json.loads(error)["code"], "grammar_dead_end")

    def test_response_format_is_forwarded_unchanged(self):
        seen = {}

        def capture(request, timeout=None):
            seen.update(json.loads(request.data))
            return FakeResponse(json.dumps({"object": "chat.completion", "choices": []}).encode())

        fmt = next(t["input"]["response_format"] for t in _load("tests.json")["tests"] if "response_format" in t["input"])
        job = {"id": "j", "input": {"messages": [{"role": "user", "content": "hi"}], "response_format": fmt}}
        with mock.patch("urllib.request.urlopen", capture):
            items = list(handler.handler(job))
        self.assertEqual(run_sdk_rule(items)[0], "COMPLETED")
        self.assertEqual(seen["response_format"], fmt)


if __name__ == "__main__":
    unittest.main()
