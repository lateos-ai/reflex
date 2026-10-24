"""Host-only tests for rag.py (no GPU, no adapter): python3 -m unittest discover .runpod

The embedding server is faked with a bag-of-words hash embedder that speaks the
adapter's /v1/embeddings response shape (base64 little-endian f32, unit length).
"""

import base64
import io
import json
import os
import re
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np

import rag

DIM = 64


def _write(path, data):
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as f:
        f.write(data)


def _vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[sum(map(ord, word)) % DIM] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def __call__(self, body):
        self.calls.append(body)
        assert body["encoding_format"] == "base64"
        assert body["input_type"] in ("query", "passage")
        assert 1 <= len(body["input"]) <= rag.EMBED_BATCH
        data = [
            {"object": "embedding", "index": i, "embedding": base64.b64encode(_vec(t).astype("<f4").tobytes()).decode()}
            for i, t in enumerate(body["input"])
        ]
        return {"object": "list", "data": list(reversed(data))}  # order must not matter


CFG = {
    "embed_model": "o/r/e.gguf",
    "top_k": 2,
    "chunk_chars": 200,
    "chunk_overlap": 40,
    "max_context_chars": 4000,
}


def _log(_):
    pass


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(rag.chunk_text("  hello world  ", 100, 20), ["hello world"])
        self.assertEqual(rag.chunk_text("   ", 100, 20), [])

    def test_windows_are_bounded_and_cover_the_text(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = rag.chunk_text(text, 120, 30)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        words = set()
        for c in chunks:
            words.update(c.split())
        self.assertEqual(words, set(text.split()))  # nothing lost, no word cut in half

    def test_overlap_repeats_the_previous_tail(self):
        text = " ".join(f"w{i}" for i in range(200))
        chunks = rag.chunk_text(text, 100, 30)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.split()[-1] in b.split(), True)

    def test_prefers_paragraph_breaks(self):
        text = "a" * 70 + "\n\n" + "b" * 70
        self.assertEqual(rag.chunk_text(text, 100, 0), ["a" * 70, "b" * 70])

    def test_unbreakable_text_still_progresses(self):
        chunks = rag.chunk_text("x" * 1000, 100, 30)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertGreaterEqual(sum(len(c) for c in chunks), 1000)


class SourceTests(unittest.TestCase):
    def test_load_sources(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sub"))
            os.makedirs(os.path.join(d, ".hidden"))
            _write(os.path.join(d, "a.txt"), "alpha")
            _write(os.path.join(d, "sub", "b.md"), "beta")
            _write(os.path.join(d, ".hidden", "c.txt"), "hidden")
            _write(os.path.join(d, "img.png"), b"\x89PNG")
            with open(os.path.join(d, "c.jsonl"), "w") as f:
                f.write(json.dumps({"id": "faq-1", "text": "gamma"}) + "\n\n")
                f.write(json.dumps({"text": "delta"}) + "\n")
            docs, fp, skipped = rag.load_sources(d, "dir")
        self.assertEqual(
            docs,
            [
                {"id": "a.txt", "text": "alpha"},
                {"id": "faq-1", "text": "gamma"},
                {"id": "c.jsonl:3", "text": "delta"},
                {"id": "sub/b.md", "text": "beta"},
            ],
        )
        self.assertEqual(skipped, ["img.png"])
        self.assertEqual([p for p, _, _ in fp], ["dir/a.txt", "dir/c.jsonl", "dir/sub/b.md"])

    def test_bad_jsonl_line_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "c.jsonl"), '{"title": "no text"}\n')
            with self.assertRaisesRegex(rag.RagError, "c.jsonl:1"):
                rag.load_sources(d, "dir")


def _tar(path, members):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


class ExtractTests(unittest.TestCase):
    def test_tar_and_zip(self):
        with tempfile.TemporaryDirectory() as d:
            _tar(os.path.join(d, "c.tar.gz"), [("docs/a.txt", b"alpha"), ("./b.md", b"beta")])
            self.assertEqual(rag.safe_extract(os.path.join(d, "c.tar.gz"), os.path.join(d, "t")), 2)
            with open(os.path.join(d, "t", "docs", "a.txt")) as f:
                self.assertEqual(f.read(), "alpha")
            with zipfile.ZipFile(os.path.join(d, "c.zip"), "w") as z:
                z.writestr("x/y.txt", "why")
            self.assertEqual(rag.safe_extract(os.path.join(d, "c.zip"), os.path.join(d, "z")), 1)
            with open(os.path.join(d, "z", "x", "y.txt")) as f:
                self.assertEqual(f.read(), "why")

    def test_rejects_traversal_and_absolute_members(self):
        for bad in ("../evil.txt", "a/../../evil.txt", "/etc/evil.txt", "C:/evil.txt"):
            with tempfile.TemporaryDirectory() as d:
                _tar(os.path.join(d, "c.tar.gz"), [(bad, b"x")])
                with self.assertRaisesRegex(rag.RagError, "unsafe member"):
                    rag.safe_extract(os.path.join(d, "c.tar.gz"), os.path.join(d, "out"))
                with zipfile.ZipFile(os.path.join(d, "c.zip"), "w") as z:
                    z.writestr(bad, "x")
                with self.assertRaisesRegex(rag.RagError, "unsafe member"):
                    rag.safe_extract(os.path.join(d, "c.zip"), os.path.join(d, "out2"))
                self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "evil.txt")))

    def test_skips_links(self):
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(os.path.join(d, "c.tar"), "w") as t:
                link = tarfile.TarInfo("link.txt")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                t.addfile(link)
            self.assertEqual(rag.safe_extract(os.path.join(d, "c.tar"), os.path.join(d, "out")), 0)

    def test_fetch_corpus_downloads_once(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.jsonl")
            _write(src, json.dumps({"text": "alpha"}) + "\n")
            fetched = []

            def download(url, dest):
                fetched.append(url)
                with open(src, "rb") as f, open(dest, "wb") as out:
                    out.write(f.read())

            url = "https://example.com/data/corpus.jsonl"
            root = rag.fetch_corpus(url, os.path.join(d, "cache"), download, _log)
            again = rag.fetch_corpus(url, os.path.join(d, "cache"), download, _log)
            self.assertEqual(root, again)
            self.assertEqual(fetched, [url])
            self.assertEqual(os.listdir(root), ["corpus.jsonl"])
            with self.assertRaisesRegex(rag.RagError, "must end in"):
                rag.fetch_corpus("https://example.com/x.pdf", d, download, _log)


class IndexTests(unittest.TestCase):
    def _corpus(self, d):
        docs = os.path.join(d, "docs")
        os.makedirs(docs)
        _write(os.path.join(docs, "cats.txt"), "Cats purr and sleep all day.")
        _write(os.path.join(docs, "rockets.txt"), "Rockets burn kerosene and oxygen.")
        return docs

    def test_build_then_load_cached(self):
        with tempfile.TemporaryDirectory() as d:
            docs = self._corpus(d)
            emb = FakeEmbedder()
            first = rag.build_or_load_index([(docs, "dir")], CFG, os.path.join(d, "cache"), emb, _log)
            self.assertEqual(len(first), 2)
            self.assertEqual(len(emb.calls), 1)
            second = rag.build_or_load_index([(docs, "dir")], CFG, os.path.join(d, "cache"), emb, _log)
            self.assertEqual(len(emb.calls), 1)  # loaded, not re-embedded
            self.assertEqual(second.chunks, first.chunks)
            np.testing.assert_array_equal(second.vectors, first.vectors)

    def test_changed_file_or_settings_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            docs = self._corpus(d)
            cache = os.path.join(d, "cache")
            emb = FakeEmbedder()
            rag.build_or_load_index([(docs, "dir")], CFG, cache, emb, _log)
            with open(os.path.join(docs, "cats.txt"), "a") as f:
                f.write(" And they hunt mice.")
            rag.build_or_load_index([(docs, "dir")], CFG, cache, emb, _log)
            self.assertEqual(len(emb.calls), 2)
            rag.build_or_load_index([(docs, "dir")], dict(CFG, chunk_chars=300), cache, emb, _log)
            self.assertEqual(len(emb.calls), 3)

    def test_batches_at_the_adapter_limit(self):
        texts = [f"text {i}" for i in range(rag.EMBED_BATCH * 2 + 3)]
        emb = FakeEmbedder()
        vectors = rag.embed_texts(texts, "passage", emb)
        self.assertEqual(vectors.shape, (len(texts), DIM))
        self.assertEqual([len(c["input"]) for c in emb.calls], [rag.EMBED_BATCH, rag.EMBED_BATCH, 3])
        np.testing.assert_allclose(vectors[5], _vec("text 5"))

    def test_volume_root_skips_the_worker_caches(self):
        # RAG_DIR set to the whole volume must not index the index's own chunks.jsonl.
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "notes.txt"), "real notes")
            models = os.path.join(d, "reflex-models")
            os.makedirs(models)
            _write(os.path.join(models, "readme.txt"), "not a document")
            cache = os.path.join(d, "reflex-rag")
            emb = FakeEmbedder()
            rag.build_or_load_index([(d, "dir")], CFG, cache, emb, _log, exclude=(models,))
            _write(os.path.join(d, "more.txt"), "more notes")  # rebuild, walking past the cache
            index = rag.build_or_load_index([(d, "dir")], CFG, cache, emb, _log, exclude=(models,))
        self.assertEqual(sorted(c["doc"] for c in index.chunks), ["more.txt", "notes.txt"])

    def test_empty_corpus_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(rag.build_or_load_index([(d, "dir")], CFG, os.path.join(d, "c"), FakeEmbedder(), _log))


def _chunk(cid, doc, origin, text):
    return {"id": cid, "doc": doc, "origin": origin, "text": text}


class RetrieveTests(unittest.TestCase):
    def test_merges_groups_by_score(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        a = ([_chunk("a#0", "a", "corpus", "A0"), _chunk("a#1", "a", "corpus", "A1")], np.array([[0.2, 0], [0.9, 0]], dtype=np.float32))
        b = ([_chunk("b#0", "b", "request", "B0")], np.array([[0.5, 0]], dtype=np.float32))
        picked = rag.retrieve(q, [a, b], 2)
        self.assertEqual([(c["id"], round(s, 2)) for c, s in picked], [("a#1", 0.9), ("b#0", 0.5)])
        self.assertEqual([c["id"] for c, _ in rag.retrieve(q, [a, b], 3, min_score=0.4)], ["a#1", "b#0"])


class GroundTests(unittest.TestCase):
    PICKED = [(_chunk("x#0", "x.txt", "corpus", "first"), 0.9), (_chunk("y#0", "y", "request", "second"), 0.5)]

    def test_adds_a_system_message(self):
        msgs = [{"role": "user", "content": "q?"}]
        out, passages = rag.ground_messages(msgs, self.PICKED, 4000)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("[1] (x.txt) first", out[0]["content"])
        self.assertIn("[2] (y) second", out[0]["content"])
        self.assertEqual(out[1:], msgs)
        self.assertEqual(msgs, [{"role": "user", "content": "q?"}])  # input untouched
        self.assertEqual([(p["n"], p["id"], p["origin"], p["score"]) for p in passages], [(1, "x#0", "corpus", 0.9), (2, "y#0", "request", 0.5)])

    def test_merges_into_an_existing_system_message(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q?"}]
        out, _ = rag.ground_messages(msgs, self.PICKED, 4000)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["content"].startswith("Be brief.\n\n" + rag.GROUNDING))

    def test_context_cap_drops_lower_scored(self):
        picked = [(_chunk("x#0", "x", "corpus", "a" * 60), 0.9), (_chunk("y#0", "y", "corpus", "b" * 60), 0.5)]
        _, passages = rag.ground_messages([{"role": "user", "content": "q"}], picked, 100)
        self.assertEqual([p["id"] for p in passages], ["x#0"])
        out, passages = rag.ground_messages([{"role": "user", "content": "q"}], picked, 20)
        self.assertEqual(len(passages), 1)  # the best one always goes in, truncated
        self.assertIn("[1] (x) " + "a" * 12, out[0]["content"])

    def test_no_match_says_so(self):
        out, passages = rag.ground_messages([{"role": "user", "content": "q"}], [], 4000)
        self.assertEqual(passages, [])
        self.assertIn("No passages matched", out[0]["content"])


class PrepareTests(unittest.TestCase):
    def _index(self, d):
        docs = os.path.join(d, "docs")
        os.makedirs(docs)
        _write(os.path.join(docs, "rockets.txt"), "Rockets burn kerosene and liquid oxygen.")
        _write(os.path.join(docs, "cats.txt"), "Cats purr and sleep all day.")
        return rag.build_or_load_index([(docs, "dir")], CFG, os.path.join(d, "cache"), FakeEmbedder(), _log)

    def test_corpus_and_request_documents(self):
        with tempfile.TemporaryDirectory() as d:
            index = self._index(d)
            payload = {
                "messages": [{"role": "user", "content": "What do rockets burn?"}],
                "max_tokens": 32,
                "rag": {"documents": [{"id": "memo", "text": "The office plant needs water."}], "top_k": 1},
            }
            out, info = rag.prepare(payload, CFG, index, FakeEmbedder())
        self.assertNotIn("rag", out)
        self.assertIn("rag", payload)  # caller's payload untouched
        self.assertEqual(out["max_tokens"], 32)
        self.assertEqual(info["query"], "What do rockets burn?")
        self.assertEqual([p["doc"] for p in info["passages"]], ["rockets.txt"])
        self.assertIn("kerosene", out["messages"][0]["content"])

    def test_request_documents_only_and_content_parts(self):
        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Where is the spare key?"}]}],
            "rag": {"documents": ["The spare key is under the blue pot.", "Lunch is at noon."], "top_k": 1},
        }
        out, info = rag.prepare(payload, CFG, None, FakeEmbedder())
        self.assertEqual(info["passages"][0]["id"], "doc0#0")
        self.assertEqual(info["passages"][0]["origin"], "request")

    def test_use_corpus_false_and_explicit_query(self):
        with tempfile.TemporaryDirectory() as d:
            index = self._index(d)
            payload = {"messages": [{"role": "user", "content": "hi"}], "rag": {"use_corpus": False, "query": "cats", "documents": ["dogs bark"]}}
            _, info = rag.prepare(payload, CFG, index, FakeEmbedder())
        self.assertEqual({p["origin"] for p in info["passages"]}, {"request"})
        self.assertEqual(info["query"], "cats")

    def test_errors(self):
        msgs = [{"role": "user", "content": "q"}]
        cases = [
            ({"messages": msgs, "rag": []}, "must be an object"),
            ({"messages": msgs, "rag": {"bogus": 1}}, "unknown"),
            ({"messages": msgs, "rag": {}}, "nothing to search"),
            ({"messages": msgs, "rag": {"documents": "x"}}, "must be a list"),
            ({"messages": msgs, "rag": {"documents": [1]}}, r"documents\[0\]"),
            ({"messages": msgs, "rag": {"documents": ["x"] * 65}}, "over the limit"),
            ({"messages": msgs, "rag": {"documents": ["x"], "top_k": 0}}, "top_k"),
            ({"messages": msgs, "rag": {"documents": ["x"], "top_k": True}}, "top_k"),
            ({"messages": [{"role": "assistant", "content": "a"}], "rag": {"documents": ["x"]}}, "no query"),
            ({"prompt": "p", "labels": [" a"], "rag": {}}, "labels"),
            ({"prompt": "p", "rag": {}}, "needs \"messages\""),
        ]
        for payload, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(rag.RagError, pattern):
                    rag.prepare(payload, CFG, None, FakeEmbedder())


class ConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        cfg = rag.config_from_env({}, "o/r/e.gguf")
        self.assertEqual(cfg["embed_model"], "o/r/e.gguf")
        self.assertEqual(cfg["dir"], "/runpod-volume/rag")
        self.assertEqual((cfg["top_k"], cfg["chunk_chars"], cfg["chunk_overlap"]), (4, 1200, 200))
        cfg = rag.config_from_env({"RAG_TOP_K": "8", "RAG_EMBED_MODEL": " a/b/c.gguf "}, "o/r/e.gguf")
        self.assertEqual((cfg["top_k"], cfg["embed_model"]), (8, "a/b/c.gguf"))

    def test_bad_values(self):
        for env, pattern in [
            ({"RAG_TOP_K": "x"}, "integer"),
            ({"RAG_TOP_K": "0"}, "at least"),
            ({"RAG_TOP_K": "99"}, "at most"),
            ({"RAG_CORPUS_URL": "file:///etc"}, "http"),
        ]:
            with self.subTest(env=env), self.assertRaisesRegex(rag.RagError, pattern):
                rag.config_from_env(env, "o/r/e.gguf")

    def test_env_flag(self):
        self.assertTrue(all(rag.env_flag(v) for v in ("on", "ON", "1", "true", " yes ")))
        self.assertFalse(any(rag.env_flag(v) for v in ("off", "", "0", "no")))


if __name__ == "__main__":
    unittest.main()
