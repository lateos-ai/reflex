"""Optional retrieval-augmented generation for the Runpod Hub worker (RAG=on).

Host-side only, like the rest of handler.py: the engine and reflex-openai-adapter are
unchanged. The worker runs a second adapter serving an embedding GGUF; this module
chunks the user's documents, embeds them through that adapter's /v1/embeddings,
picks the passages closest to each question and writes them into the chat request's
system message. Nothing here runs inside the engine, and the only state it keeps is
the corpus index: plain files under the cache directory, rebuilt whenever the corpus
or the chunking settings change.

Everything that talks to the network is passed in (embed_fn, download_fn), so the
module can be tested without a GPU (see test_rag.py).
"""

import base64
import hashlib
import json
import os
import shutil
import tarfile
import urllib.parse
import zipfile

import numpy as np

# The adapter's per-request input limit (MAX_INPUTS in
# sidecar/openai-adapter/src/embeddings.rs).
EMBED_BATCH = 256
TEXT_SUFFIXES = (".txt", ".md")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")
MAX_REQUEST_DOCUMENTS = 64
MAX_REQUEST_CHARS = 1 << 20
MAX_TOP_K = 32
DEFAULT_DIR = "/runpod-volume/rag"

GROUNDING = (
    "Answer the user's question using the passages below. Cite each passage you use "
    "as [n]. If the passages do not contain the answer, say so.\n\n"
)


class RagError(Exception):
    """A problem with a job's rag request: the job fails with this message."""


def env_flag(value):
    return str(value).strip().lower() in ("1", "on", "true", "yes")


def chunk_text(text, size, overlap):
    """Split text into windows of at most `size` characters, each starting `overlap`
    characters before the previous one ended. A window ends at a paragraph break,
    else a line or sentence end, else a space, when one falls in its second half."""
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            window = text[pos:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep, size // 2)
                if cut != -1:
                    end = pos + cut + len(sep)
                    break
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        nxt = end - overlap
        if overlap:
            # Start the overlap at a word boundary, not mid-word.
            space = text.find(" ", nxt, end)
            if space != -1:
                nxt = space + 1
        pos = nxt if nxt > pos else end
    return chunks


def load_sources(root, label, exclude=()):
    """Documents under root: each .txt/.md file is one document (id: its relative
    path), each line of a .jsonl file one document ({"text": ..., "id": optional}).
    Dot-directories and the directories in exclude (the worker's own model and index
    caches, in case root is the whole volume) are not walked.
    Returns (docs, fingerprint, skipped): docs as {"id", "text"} dicts in path order,
    fingerprint as (path, size, mtime_ns) tuples for the index cache key."""
    excluded = {os.path.realpath(p) for p in exclude}
    docs, fingerprint, skipped = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".") and os.path.realpath(os.path.join(dirpath, d)) not in excluded
        )
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            lower = name.lower()
            if name.startswith(".") or not lower.endswith(TEXT_SUFFIXES + (".jsonl",)):
                skipped.append(rel)
                continue
            st = os.stat(path)
            fingerprint.append((f"{label}/{rel}", st.st_size, st.st_mtime_ns))
            with open(path, encoding="utf-8", errors="replace") as f:
                if lower.endswith(".jsonl"):
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise RagError(f"{rel}:{lineno}: not JSON ({e})") from None
                        doc = _request_doc(record, f"{rel}:{lineno}")
                        if doc is None:
                            raise RagError(f'{rel}:{lineno}: needs a "text" string')
                        docs.append(doc)
                else:
                    docs.append({"id": rel, "text": f.read()})
    return docs, fingerprint, skipped


def _safe_member_path(dest, name):
    """dest/name, or None when the member name is absolute or leaves dest."""
    name = name.replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return os.path.join(dest, *parts)


def safe_extract(archive, dest):
    """Extract the regular files of a .tar(.gz)/.tgz/.zip into dest. A member with an
    absolute path or a '..' component is an error; links and device files are
    skipped. Returns the number of files written."""
    written = 0
    os.makedirs(dest, exist_ok=True)
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = _safe_member_path(dest, info.filename)
                if target is None:
                    raise RagError(f"{os.path.basename(archive)}: unsafe member {info.filename!r}")
                if info.is_dir():
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
        return written
    with tarfile.open(archive) as t:
        for member in t:
            target = _safe_member_path(dest, member.name)
            if target is None:
                raise RagError(f"{os.path.basename(archive)}: unsafe member {member.name!r}")
            if not member.isfile():
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = t.extractfile(member)
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
    return written


def fetch_corpus(url, cache_root, download_fn, log):
    """Download the corpus at url once into cache_root/corpus/<hash>/ and return the
    directory to load documents from. An archive is extracted there; a single
    .txt/.md/.jsonl file is kept under its own name."""
    name = os.path.basename(urllib.parse.urlparse(url).path) or "corpus.jsonl"
    lower = name.lower()
    if not lower.endswith(ARCHIVE_SUFFIXES + TEXT_SUFFIXES + (".jsonl",)):
        raise RagError(
            f"RAG_CORPUS_URL must end in .jsonl, .txt, .md, .tar.gz, .tgz, .tar or .zip, got {name!r}"
        )
    base = os.path.join(cache_root, "corpus", hashlib.sha256(url.encode()).hexdigest()[:12])
    docs_dir = os.path.join(base, "docs")
    done = os.path.join(base, ".complete")
    if os.path.exists(done):
        log(f"using cached corpus {docs_dir}")
        return docs_dir
    os.makedirs(base, exist_ok=True)
    fetched = os.path.join(base, name)
    if not os.path.exists(fetched):
        log(f"downloading corpus {url}")
        download_fn(url, fetched)
    # Extract into a per-process directory, then rename it into place, so two workers
    # sharing a volume never read a half-written tree.
    staging = f"{docs_dir}.{os.getpid()}.part"
    shutil.rmtree(staging, ignore_errors=True)
    if lower.endswith(ARCHIVE_SUFFIXES):
        log(f"extracted {safe_extract(fetched, staging)} files from {name}")
    else:
        os.makedirs(staging)
        shutil.copyfile(fetched, os.path.join(staging, name))
    try:
        os.replace(staging, docs_dir)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)  # another worker got there first
        if not os.path.isdir(docs_dir):
            raise
    open(done, "w").close()
    return docs_dir


def index_key(embed_model, size, overlap, fingerprint):
    blob = json.dumps(
        {"embed_model": embed_model, "size": size, "overlap": overlap, "files": sorted(fingerprint)}
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def chunk_docs(docs, size, overlap, origin):
    chunks = []
    for doc in docs:
        for n, text in enumerate(chunk_text(doc["text"], size, overlap)):
            chunks.append({"id": f"{doc['id']}#{n}", "doc": doc["id"], "origin": origin, "text": text})
    return chunks


def embed_texts(texts, input_type, post_embeddings):
    """Embed texts through the adapter, EMBED_BATCH at a time. post_embeddings(body)
    returns the adapter's parsed /v1/embeddings response. Returns an (n, d) float32
    array of unit vectors (the engine L2-normalizes them)."""
    rows = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        resp = post_embeddings(
            {"input": batch, "input_type": input_type, "encoding_format": "base64"}
        )
        data = sorted(resp["data"], key=lambda d: d["index"])
        if len(data) != len(batch):
            raise RagError(f"embedding server returned {len(data)} vectors for {len(batch)} texts")
        for item in data:
            rows.append(np.frombuffer(base64.b64decode(item["embedding"]), dtype="<f4"))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


class Index:
    def __init__(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors

    def __len__(self):
        return len(self.chunks)


def build_or_load_index(sources, cfg, cache_root, post_embeddings, log, exclude=()):
    """The corpus index for sources (a list of (directory, label)), loaded from
    cache_root/index/<key>/ when an index with the same key exists, else built and
    saved there. None when the sources hold no documents. cache_root and the
    directories in exclude are never read as sources."""
    docs, fingerprint = [], []
    for root, label in sources:
        d, fp, skipped = load_sources(root, label, (cache_root,) + tuple(exclude))
        if skipped:
            # One line, not one per file: a repository archive skips hundreds.
            shown = ", ".join(skipped[:5]) + (f", ... ({len(skipped) - 5} more)" if len(skipped) > 5 else "")
            log(f"skipping {len(skipped)} {label} file(s) that are not .txt, .md or .jsonl: {shown}")
        docs += [dict(doc, id=f"{label}/{doc['id']}" if len(sources) > 1 else doc["id"]) for doc in d]
        fingerprint += fp
    if not docs:
        log("no corpus documents found; only per-request documents will be searched")
        return None
    key = index_key(cfg["embed_model"], cfg["chunk_chars"], cfg["chunk_overlap"], fingerprint)
    out = os.path.join(cache_root, "index", key)
    vectors_path = os.path.join(out, "vectors.npy")
    chunks_path = os.path.join(out, "chunks.jsonl")
    if os.path.exists(vectors_path) and os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        vectors = np.load(vectors_path)
        log(f"using cached index {out} ({len(chunks)} chunks)")
        return Index(chunks, vectors)
    chunks = chunk_docs(docs, cfg["chunk_chars"], cfg["chunk_overlap"], "corpus")
    log(f"embedding {len(chunks)} chunks from {len(docs)} documents")
    vectors = embed_texts([c["text"] for c in chunks], "passage", post_embeddings)
    os.makedirs(out, exist_ok=True)
    pid = os.getpid()
    with open(f"{vectors_path}.{pid}.part", "wb") as f:
        np.save(f, vectors)
    with open(f"{chunks_path}.{pid}.part", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # chunks.jsonl last: the cache counts as present only once both files are.
    os.replace(f"{vectors_path}.{pid}.part", vectors_path)
    os.replace(f"{chunks_path}.{pid}.part", chunks_path)
    log(f"saved index {out}")
    return Index(chunks, vectors)


def _text_of(content):
    """A message's text: a string, or the text parts of a content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _request_doc(item, default_id):
    if isinstance(item, str):
        return {"id": default_id, "text": item}
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        return {"id": str(item.get("id", default_id)), "text": item["text"]}
    return None


def parse_rag(rag, messages, default_top_k):
    """Validate a job's "rag" object. Returns (query, request docs, top_k, use_corpus,
    min_score)."""
    if not isinstance(rag, dict):
        raise RagError('"rag" must be an object, e.g. {} or {"documents": [...]}')
    unknown = set(rag) - {"documents", "top_k", "query", "use_corpus", "min_score"}
    if unknown:
        raise RagError(f'unknown "rag" field(s): {", ".join(sorted(unknown))}')
    query = rag.get("query")
    if query is None:
        for message in reversed(messages or []):
            if isinstance(message, dict) and message.get("role") == "user":
                query = _text_of(message.get("content"))
                break
    if not isinstance(query, str) or not query.strip():
        raise RagError('no query: send a user message or "rag": {"query": "..."}')
    raw_docs = rag.get("documents", [])
    if not isinstance(raw_docs, list):
        raise RagError('"rag.documents" must be a list')
    if len(raw_docs) > MAX_REQUEST_DOCUMENTS:
        raise RagError(f'"rag.documents" has {len(raw_docs)} entries, over the limit of {MAX_REQUEST_DOCUMENTS}')
    docs = []
    for i, item in enumerate(raw_docs):
        doc = _request_doc(item, f"doc{i}")
        if doc is None:
            raise RagError(f'rag.documents[{i}] must be a string or {{"id", "text"}}')
        docs.append(doc)
    if sum(len(d["text"]) for d in docs) > MAX_REQUEST_CHARS:
        raise RagError(f'"rag.documents" is over {MAX_REQUEST_CHARS} characters in all')
    top_k = rag.get("top_k", default_top_k)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise RagError(f'"rag.top_k" must be an integer from 1 to {MAX_TOP_K}')
    use_corpus = rag.get("use_corpus", True)
    if not isinstance(use_corpus, bool):
        raise RagError('"rag.use_corpus" must be true or false')
    min_score = rag.get("min_score")
    if min_score is not None and not isinstance(min_score, (int, float)):
        raise RagError('"rag.min_score" must be a number')
    return query, docs, top_k, use_corpus, min_score


def retrieve(query_vec, candidates, top_k, min_score=None):
    """The top_k (chunk, score) pairs by dot product over candidates, a list of
    (chunks, vectors) groups; ties keep candidate order."""
    scored = []
    for chunks, vectors in candidates:
        if not chunks:
            continue
        scores = vectors @ query_vec
        scored += [(float(s), c) for c, s in zip(chunks, scores)]
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    picked = []
    for i in order[:top_k]:
        score, chunk = scored[i]
        if min_score is not None and score < min_score:
            break
        picked.append((chunk, score))
    return picked


def ground_messages(messages, picked, max_chars):
    """messages with the passages written into the system message (merged after an
    existing leading system message, else added first), and the passages used, in
    score order, numbered as the prompt cites them. Lower-scored passages are dropped
    once the context would exceed max_chars; the first one is truncated to fit."""
    passages, blocks, used = [], [], 0
    for chunk, score in picked:
        n = len(passages) + 1
        block = f"[{n}] ({chunk['doc']}) {chunk['text']}"
        if used + len(block) > max_chars:
            if passages:
                break
            block = block[:max_chars]
        blocks.append(block)
        used += len(block) + 2
        passages.append(
            {
                "n": n,
                "id": chunk["id"],
                "doc": chunk["doc"],
                "origin": chunk["origin"],
                "score": round(score, 4),
                "text": chunk["text"],
            }
        )
    context = GROUNDING + "\n\n".join(blocks) if blocks else (
        "No passages matched the user's question. Say that the documents do not cover it."
    )
    messages = list(messages or [])
    first = messages[0] if messages else None
    if isinstance(first, dict) and first.get("role") == "system" and isinstance(first.get("content"), str):
        messages[0] = dict(first, content=f"{first['content']}\n\n{context}")
    else:
        messages.insert(0, {"role": "system", "content": context})
    return messages, passages


def prepare(payload, cfg, index, post_embeddings):
    """A rag job's chat payload ("rag" removed, passages in the system message) and
    the "rag" object returned with the answer."""
    payload = dict(payload)
    rag = payload.pop("rag")
    if "labels" in payload:
        raise RagError('"rag" is for chat jobs; it cannot be combined with "labels"')
    if "messages" not in payload:
        raise RagError('a "rag" job needs "messages"')
    query, docs, top_k, use_corpus, min_score = parse_rag(rag, payload["messages"], cfg["top_k"])
    request_chunks = chunk_docs(docs, cfg["chunk_chars"], cfg["chunk_overlap"], "request")
    texts = [c["text"] for c in request_chunks]
    vectors = embed_texts(texts, "passage", post_embeddings) if texts else None
    query_vec = embed_texts([query], "query", post_embeddings)[0]
    candidates = []
    if use_corpus and index is not None:
        candidates.append((index.chunks, index.vectors))
    if request_chunks:
        candidates.append((request_chunks, vectors))
    if not candidates:
        raise RagError(
            'nothing to search: send "rag.documents", or deploy with a corpus '
            "(RAG_DIR on a network volume, or RAG_CORPUS_URL)"
        )
    picked = retrieve(query_vec, candidates, top_k, min_score)
    payload["messages"], passages = ground_messages(payload["messages"], picked, cfg["max_context_chars"])
    return payload, {"query": query, "passages": passages}


def config_from_env(env, default_embed_model):
    def number(key, default, minimum):
        raw = env.get(key, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise RagError(f"{key} must be an integer, got {raw!r}") from None
        if value < minimum:
            raise RagError(f"{key} must be at least {minimum}, got {value}")
        return value

    cfg = {
        "embed_model": env.get("RAG_EMBED_MODEL", "").strip() or default_embed_model,
        "dir": env.get("RAG_DIR", "").strip() or DEFAULT_DIR,
        "corpus_url": env.get("RAG_CORPUS_URL", "").strip(),
        "top_k": number("RAG_TOP_K", 4, 1),
        "chunk_chars": number("RAG_CHUNK_CHARS", 1200, 100),
        "chunk_overlap": number("RAG_CHUNK_OVERLAP", 200, 0),
        "max_context_chars": number("RAG_MAX_CONTEXT_CHARS", 24000, 1000),
    }
    if cfg["top_k"] > MAX_TOP_K:
        raise RagError(f"RAG_TOP_K must be at most {MAX_TOP_K}")
    if cfg["corpus_url"] and not cfg["corpus_url"].startswith(("https://", "http://")):
        raise RagError(f"RAG_CORPUS_URL must be an http(s) URL, got {cfg['corpus_url']!r}")
    return cfg
