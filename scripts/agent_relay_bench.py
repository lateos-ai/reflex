#!/usr/bin/env python3
"""Measure agent-to-agent handoff latency: per-hop round trip over a warm engine.

Runs an N-hop chain against one warm process. Each hop's prompt is a shared
system prompt, the next agent role, and the previous hop's output, which stands
in for one agent handing work to the next. Every hop is one request, sent only
after the previous hop's final response arrives, which is the pattern a
multi-agent loop produces. Reports p50/p95 per hop of:

  rtt           client-measured round trip (send request -> final line read)
  first_token   engine: model call start -> first generated token
  model         engine: whole model call
  overhead      rtt - engine time (transport, JSON, process boundary)

Targets:
  stdio         spawns `reflex stdio <model>` and talks over its pipes
  uds           spawns `reflex uds <model> <socket>` and connects (Unix only)
  llama-server  an already-running llama.cpp server (`--url`), POST /completion;
                engine time comes from its `timings` block

Reflex requests set `"timings": true` (src/ipc.rs, IpcTimings). Greedy decoding
everywhere, so runs are comparable.

Usage:
  scripts/agent_relay_bench.py stdio --reflex target/release/reflex --model m.gguf
  scripts/agent_relay_bench.py uds --reflex target/release/reflex --model m.gguf
  scripts/agent_relay_bench.py llama-server --url http://127.0.0.1:8080
  common: [--hops 20] [--warmup 3] [--max-tokens 32] [--carry-chars 400]
          [--json-out results.json]

Standard library only, so it runs anywhere Python 3.8+ does.
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

SYSTEM_PROMPT = (
    "You are one agent in a team of cooperating agents. Each agent receives the "
    "previous agent's note, does its own part of the task, and writes a short note "
    "for the next agent. Keep notes brief and concrete.\n"
)
ROLES = ["planner", "researcher", "coder", "reviewer", "tester", "summarizer"]
TASK = "Task: design a small command-line tool that deduplicates lines in a file."


def hop_prompt(hop, previous, carry_chars):
    role = ROLES[hop % len(ROLES)]
    note = previous[-carry_chars:] if previous else TASK
    return f"{SYSTEM_PROMPT}Previous note: {note}\nYou are the {role}. Your note:"


class ReflexLines:
    """Line-delimited JSON over a pair of binary streams (pipe or socket)."""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def request(self, payload):
        line = (json.dumps(payload) + "\n").encode("utf-8")
        start = time.perf_counter()
        self.writer.write(line)
        self.writer.flush()
        while True:
            raw = self.reader.readline()
            if not raw:
                raise RuntimeError("engine closed the connection mid-request")
            msg = json.loads(raw)
            if msg.get("event") == "final":
                return msg, (time.perf_counter() - start) * 1000.0


def wait_for_ready(proc, marker, timeout_s):
    """Reads the engine's stderr until its READY line; returns the lines seen."""
    seen = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stderr.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").rstrip()
        seen.append(line)
        if marker in line:
            return seen
    raise RuntimeError(
        f"engine never printed {marker} (exit={proc.poll()}):\n" + "\n".join(seen[-20:])
    )


def drain_stderr(proc):
    """Keeps a pipe-full of stderr from blocking the engine during a long run."""
    import threading

    def pump():
        for _ in iter(proc.stderr.readline, b""):
            pass

    threading.Thread(target=pump, daemon=True).start()


def reflex_hop(client, prompt, max_tokens):
    msg, rtt = client.request(
        {"prompt": prompt, "max_tokens": max_tokens, "timings": True}
    )
    if not msg.get("ok"):
        raise RuntimeError(f"engine error: {msg.get('error_kind')}: {msg.get('error')}")
    t = msg.get("timings") or {}
    engine = t.get("parse_ms", 0.0) + t.get("model_ms", 0.0)
    return {
        "rtt_ms": rtt,
        "first_token_ms": t.get("first_token_ms"),
        "model_ms": t.get("model_ms"),
        "overhead_ms": rtt - engine if t else None,
        "tokens": len(msg.get("token_ids") or []),
    }, msg.get("text", "")


def llama_hop(url, prompt, max_tokens, cache_prompt):
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": cache_prompt,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        msg = json.loads(resp.read())
    rtt = (time.perf_counter() - start) * 1000.0
    t = msg.get("timings") or {}
    prompt_ms = t.get("prompt_ms")
    model = (prompt_ms or 0.0) + (t.get("predicted_ms") or 0.0)
    return {
        "rtt_ms": rtt,
        # llama.cpp reports prompt processing separately from decode; prompt_ms
        # plus one decode step is its closest analogue of first_token_ms.
        "first_token_ms": (prompt_ms + t.get("predicted_per_token_ms", 0.0))
        if prompt_ms is not None
        else None,
        "model_ms": model if t else None,
        "overhead_ms": rtt - model if t else None,
        "tokens": t.get("predicted_n"),
    }, msg.get("content", "")


def run_chain(do_hop, hops, warmup, max_tokens, carry_chars):
    samples = []
    previous = ""
    for hop in range(warmup + hops):
        sample, text = do_hop(hop_prompt(hop, previous, carry_chars), max_tokens)
        previous = text.strip()
        if hop >= warmup:
            samples.append(sample)
    return samples


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(samples):
    out = {}
    for key in ("rtt_ms", "first_token_ms", "model_ms", "overhead_ms", "tokens"):
        vals = [s[key] for s in samples if s.get(key) is not None]
        if vals:
            out[key] = {
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "mean": statistics.fmean(vals),
            }
    return out


def print_table(target, summary, n):
    print(f"\n## agent relay: {target}, {n} hops\n")
    print("| metric | p50 | p95 | mean |")
    print("|---|---:|---:|---:|")
    for key, label in (
        ("rtt_ms", "round trip (ms)"),
        ("first_token_ms", "first token (ms)"),
        ("model_ms", "engine total (ms)"),
        ("overhead_ms", "overhead (ms)"),
        ("tokens", "tokens generated"),
    ):
        if key in summary:
            s = summary[key]
            print(f"| {label} | {s['p50']:.2f} | {s['p95']:.2f} | {s['mean']:.2f} |")


def spawn_reflex(args, extra):
    cmd = [args.reflex, args.target, args.model] + extra + args.engine_arg
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", choices=["stdio", "uds", "llama-server"])
    ap.add_argument("--reflex", default="target/release/reflex")
    ap.add_argument("--model", help="GGUF path (stdio/uds)")
    ap.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        help="extra argument passed to the reflex subcommand (repeatable)",
    )
    ap.add_argument("--socket", help="UDS path (default: a temp file)")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL")
    ap.add_argument(
        "--llama-cache-prompt",
        action="store_true",
        help="let llama-server reuse the shared prefix (default off; Reflex has no prefix reuse yet)",
    )
    ap.add_argument("--hops", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--carry-chars", type=int, default=400)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    proc = None
    try:
        if args.target == "llama-server":

            def do_hop(prompt, max_tokens):
                return llama_hop(args.url, prompt, max_tokens, args.llama_cache_prompt)

        else:
            if not args.model:
                ap.error("--model is required for stdio/uds")
            if args.target == "stdio":
                proc = spawn_reflex(args, [])
                wait_for_ready(proc, "REFLEX_STDIO_READY", args.ready_timeout)
                drain_stderr(proc)
                client = ReflexLines(proc.stdout, proc.stdin)
            else:
                if not hasattr(socket, "AF_UNIX"):
                    ap.error("uds needs a platform with AF_UNIX sockets; use stdio")
                sock_path = args.socket or os.path.join(
                    tempfile.mkdtemp(prefix="reflex-relay-"), "reflex.sock"
                )
                proc = spawn_reflex(args, [sock_path])
                wait_for_ready(proc, "REFLEX_UDS_READY", args.ready_timeout)
                drain_stderr(proc)
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.connect(sock_path)
                client = ReflexLines(conn.makefile("rb"), conn.makefile("wb"))

            def do_hop(prompt, max_tokens):
                return reflex_hop(client, prompt, max_tokens)

        samples = run_chain(do_hop, args.hops, args.warmup, args.max_tokens, args.carry_chars)
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()

    summary = summarize(samples)
    print_table(args.target, summary, len(samples))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target": args.target,
                    "model": args.model,
                    "hops": args.hops,
                    "warmup": args.warmup,
                    "max_tokens": args.max_tokens,
                    "carry_chars": args.carry_chars,
                    "llama_cache_prompt": args.llama_cache_prompt,
                    "summary": summary,
                    "samples": samples,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
