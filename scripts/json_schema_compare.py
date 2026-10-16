#!/usr/bin/env python3
"""Compare JSON Schema constrained decoding: Reflex `json_schema` vs llama-server.

Runs a fixed set of (schema, prompt) cases, greedy, on a warm `reflex stdio`
process and on an already-running llama-server (`POST /completion` with its
`json_schema` field), each with and without the schema. Reports:

  valid          outputs that parse as JSON and validate against the schema
                 (`jsonschema` if installed, else this script's own validator for
                 the subset Reflex supports)
  decode/token   (engine total - first token) / (tokens - 1), with and without
                 the schema; the difference is what constraining costs per token
  mask/token     Reflex only: `timings.mask_ms_total` / tokens

Byte-identical output across engines isn't expected: llama.cpp's grammar allows
optional whitespace and Reflex's output is compact (see docs/design/agent-relay.md).

Usage:
  scripts/json_schema_compare.py --reflex target/release/reflex --model m.gguf \\
      --url http://127.0.0.1:8080 --template qwen3 [--max-tokens 192] [--json-out r.json]
  --template: qwen3 (ChatML with an empty think block), zephyr (TinyLlama-Chat), raw
  --hint: append a compact-JSON instruction with an example to every prompt
  Either side can be skipped: --no-llama, or omit --model.

Standard library only (plus `jsonschema` if available).
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HANDOFF = {
    "type": "object",
    "properties": {
        "agent": {"type": "string"},
        "status": {"enum": ["done", "blocked", "needs_review"]},
        "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
        "confidence": {"type": "number"},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["agent", "status", "steps"],
    "additionalProperties": False,
    "$defs": {
        "step": {
            "type": "object",
            "properties": {
                "tool": {"enum": ["search", "read_file", "run_tests"]},
                "args": {"type": "object", "properties": {"query": {"type": "string"}}},
                "ok": {"type": "boolean"},
            },
            "required": ["tool", "ok"],
            "additionalProperties": False,
        }
    },
}
TOOL_CALL = {
    "type": "object",
    "properties": {
        "name": {"enum": ["get_weather", "search_web", "send_email"]},
        "arguments": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "body"],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}
SENTIMENT = {"enum": ["positive", "negative", "neutral"]}
EXTRACT = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "city": {"type": ["string", "null"]},
    },
    "required": ["people", "city"],
    "additionalProperties": False,
}
VERDICT = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "score": {"type": "integer"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approve", "score", "reasons"],
    "additionalProperties": False,
}

CASES = [
    ("handoff", HANDOFF, "Write the hand-off record, as JSON, for a planner agent that finished planning a CLI tool which deduplicates lines in a file. It ran one search step for \"dedupe lines\" and it worked."),
    ("handoff", HANDOFF, "Give the hand-off record, as JSON, for a reviewer agent that is blocked waiting for test results."),
    ("handoff", HANDOFF, "As JSON: a coder agent finished, ran the tests (they passed) and read the file main.rs. Confidence 0.8."),
    ("handoff", HANDOFF, "JSON hand-off for a tester agent whose test run failed; it needs review."),
    ("tool_call", TOOL_CALL, "Call the right tool, as JSON: what's the weather in Lisbon in celsius?"),
    ("tool_call", TOOL_CALL, "Call the right tool, as JSON: find recent papers on speculative decoding."),
    ("tool_call", TOOL_CALL, "Call the right tool, as JSON: email bob@example.com that the build is green."),
    ("tool_call", TOOL_CALL, "Call the right tool, as JSON: is it going to rain in Tokyo tomorrow, in fahrenheit?"),
    ("sentiment", SENTIMENT, "Classify the sentiment of this review as a JSON string: \"The battery died after two days and support never answered.\""),
    ("sentiment", SENTIMENT, "Classify the sentiment as a JSON string: \"Absolutely loved it, would buy again!\""),
    ("sentiment", SENTIMENT, "Classify the sentiment as a JSON string: \"The package arrived on Tuesday.\""),
    ("sentiment", SENTIMENT, "Classify the sentiment as a JSON string: \"It works, I guess, but the manual is useless.\""),
    ("extract", EXTRACT, "Extract the people (with ages when given) and the city, as JSON: \"Maria (34) and her brother Tom met Ana in Porto.\""),
    ("extract", EXTRACT, "Extract the people and the city as JSON: \"Dr. Chen, 51, flew from Toronto to meet Priya.\""),
    ("extract", EXTRACT, "Extract the people and the city as JSON: \"Nobody was home.\""),
    ("extract", EXTRACT, "Extract the people (with ages) and the city as JSON: \"Kenji, 8, and Yuki, 10, live in Osaka with their grandmother Hana.\""),
    ("verdict", VERDICT, "Review this change and answer as JSON (approve, a 1-10 score, reasons): \"Removed all error handling to make the code shorter.\""),
    ("verdict", VERDICT, "Review this change and answer as JSON (approve, a 1-10 score, reasons): \"Added unit tests for the parser and fixed an off-by-one bug.\""),
    ("verdict", VERDICT, "Review as JSON (approve, 1-10 score, reasons): \"Renamed variables for clarity, no behavior change.\""),
    ("verdict", VERDICT, "Review as JSON (approve, 1-10 score, reasons): \"Hard-coded the production database password in the config file.\""),
]

TEMPLATES = {
    "qwen3": "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "zephyr": "<|user|>\n{u}</s>\n<|assistant|>\n",
    "raw": "{u}\n",
}


# --- validation --------------------------------------------------------------


def make_validator(schema):
    """`jsonschema` when available; else a validator for Reflex's subset."""
    try:
        import jsonschema  # noqa: F401

        cls = jsonschema.validators.validator_for(schema)
        v = cls(schema)
        return lambda value: v.is_valid(value), "jsonschema " + jsonschema.__version__
    except ImportError:
        return (lambda value: subset_valid(schema, schema, value)), "built-in subset validator"


def subset_valid(root, s, v):
    if s is True:
        return True
    if "$ref" in s:
        ref = s["$ref"]
        target = root if ref == "#" else root["$defs"][ref[len("#/$defs/"):]]
        return subset_valid(root, target, v)
    if "anyOf" in s:
        return any(subset_valid(root, b, v) for b in s["anyOf"])
    if "enum" in s and not any(eq(v, e) for e in s["enum"]):
        return False
    if "const" in s and not eq(v, s["const"]):
        return False
    if "type" in s:
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(is_type(v, t) for t in types):
            return False
    if isinstance(v, dict):
        props = s.get("properties", {})
        for k, val in v.items():
            if k in props:
                if not subset_valid(root, props[k], val):
                    return False
            elif s.get("additionalProperties") is False:
                return False
        if any(r not in v for r in s.get("required", [])):
            return False
    if isinstance(v, list) and "items" in s:
        return all(subset_valid(root, s["items"], x) for x in v)
    return True


def is_type(v, t):
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool) or isinstance(v, float) and v.is_integer()
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def check(text, schema):
    try:
        value = json.loads(text)
    except ValueError:
        return False, False
    return True, make_validator(schema)[0](value)


# --- engines -----------------------------------------------------------------


class Reflex:
    def __init__(self, binary, model, extra):
        self.proc = subprocess.Popen(
            [binary, "stdio", model] + extra,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            line = self.proc.stderr.readline().decode("utf-8", "replace")
            if not line:
                raise RuntimeError("reflex stdio exited before it was ready")
            if "REFLEX_STDIO_READY" in line:
                break
        threading.Thread(target=lambda: [None for _ in self.proc.stderr], daemon=True).start()

    def run(self, prompt, schema, max_tokens):
        req = {"prompt": prompt, "max_tokens": max_tokens, "timings": True}
        if schema is not None:
            req["json_schema"] = schema
        self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        msg = json.loads(self.proc.stdout.readline())
        t = msg.get("timings") or {}
        ids = msg.get("token_ids") or []
        n = len(ids)
        text = msg.get("text", "")
        if schema is None:
            # Unconstrained text includes EOS's text; drop it before judging JSON.
            for eos_text in ("<|im_end|>", "</s>", "<|endoftext|>"):
                if text.endswith(eos_text):
                    text = text[: -len(eos_text)]
        decode = None
        if n > 1 and t.get("first_token_ms") is not None:
            decode = (t["model_ms"] - t["first_token_ms"]) / (n - 1)
        mask = t.get("mask_ms_total")
        return {
            "ok": bool(msg.get("ok")),
            "error_kind": msg.get("error_kind"),
            "text": text,
            "tokens": n,
            "decode_ms_per_token": decode,
            "mask_ms_per_token": mask / n if mask is not None and n else None,
            "mask_ms_total": mask,
        }

    def close(self):
        self.proc.kill()
        self.proc.wait()


def llama(url, prompt, schema, max_tokens):
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
    }
    if schema is not None:
        body["json_schema"] = schema
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        msg = json.loads(resp.read())
    t = msg.get("timings") or {}
    return {
        "ok": True,
        "error_kind": None,
        "text": msg.get("content", ""),
        "tokens": t.get("predicted_n"),
        "decode_ms_per_token": t.get("predicted_per_token_ms"),
        "mask_ms_per_token": None,
        "mask_ms_total": None,
        "stop_type": msg.get("stop_type"),
    }


# --- report ------------------------------------------------------------------


def summarize(rows, engine):
    out = {}
    for constrained in (False, True):
        rs = [r for r in rows if r["engine"] == engine and r["constrained"] == constrained]
        if not rs:
            continue
        key = "constrained" if constrained else "unconstrained"
        dec = [r["decode_ms_per_token"] for r in rs if r["decode_ms_per_token"] is not None]
        mask = [r["mask_ms_per_token"] for r in rs if r["mask_ms_per_token"] is not None]
        out[key] = {
            "n": len(rs),
            "parsed": sum(r["parsed"] for r in rs),
            "valid": sum(r["valid"] for r in rs),
            "tokens_mean": statistics.fmean(r["tokens"] or 0 for r in rs),
            "decode_ms_per_token_mean": statistics.fmean(dec) if dec else None,
            "mask_ms_per_token_mean": statistics.fmean(mask) if mask else None,
            "mask_ms_per_token_max": max(mask) if mask else None,
        }
    return out


def fmt(x, d=2):
    return "" if x is None else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reflex", default="target/release/reflex")
    ap.add_argument("--model", help="GGUF for `reflex stdio`; omit to skip Reflex")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--no-llama", action="store_true")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="qwen3")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument(
        "--hint",
        action="store_true",
        help="append a compact-JSON instruction with an example to every prompt (both engines)",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    print(f"validator: {make_validator(HANDOFF)[1]}", file=sys.stderr)
    rows = []
    reflex = Reflex(args.reflex, args.model, args.engine_arg) if args.model else None
    try:
        for name, schema, user in CASES:
            if args.hint:
                user += ' Answer with compact JSON with no spaces, for example {"a":"x","b":[1,2]}.'
            prompt = TEMPLATES[args.template].format(u=user)
            for constrained in (False, True):
                s = schema if constrained else None
                runs = []
                if reflex:
                    runs.append(("reflex", reflex.run(prompt, s, args.max_tokens)))
                if not args.no_llama:
                    runs.append(("llama-server", llama(args.url, prompt, s, args.max_tokens)))
                for engine, r in runs:
                    parsed, valid = check(r["text"], schema) if r["ok"] else (False, False)
                    r.update(engine=engine, case=name, constrained=constrained, parsed=parsed, valid=valid)
                    rows.append(r)
                    print(
                        f"{engine:12} {name:9} {'schema' if constrained else 'free  '} valid={valid!s:5} "
                        f"tokens={r['tokens']} {r['text'][:100]!r}",
                        file=sys.stderr,
                    )
    finally:
        if reflex:
            reflex.close()

    summary = {e: summarize(rows, e) for e in ("reflex", "llama-server")}
    hint = ", compact-JSON hint" if args.hint else ""
    print(f"\n## json_schema: {len(CASES)} cases, greedy, max_tokens {args.max_tokens}{hint}\n")
    print("| engine | schema | valid | parsed as JSON | tokens (mean) | decode/token (ms) | mask/token mean (ms) | mask/token max (ms) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for engine, by in summary.items():
        for key, s in by.items():
            print(
                f"| {engine} | {'yes' if key == 'constrained' else 'no'} | {s['valid']}/{s['n']} | "
                f"{s['parsed']}/{s['n']} | {fmt(s['tokens_mean'], 1)} | {fmt(s['decode_ms_per_token_mean'])} | "
                f"{fmt(s['mask_ms_per_token_mean'], 3)} | {fmt(s['mask_ms_per_token_max'], 3)} |"
            )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "summary": summary, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
