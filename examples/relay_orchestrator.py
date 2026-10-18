#!/usr/bin/env python3
"""Three agents composed by an orchestrator, on top of Reflex's local interfaces.

1. A router: a warm `reflex stdio` process with a pinned role prefix. It makes a
   System-1 decision (`candidates`) against the pin (`use_pinned`).
2. A worker: a second warm process with its own pinned prefix. It writes a JSON
   Schema-constrained handoff, then exports its context to a cache file.
3. A cold sub-agent: a fresh `reflex generate --import-kv` process that resumes from
   that file.

Each hop's result goes into an envelope (examples/relay_envelope.schema.json),
printed as one JSON line on stdout. The envelope, the routing and the files belong
to this script, not to the engine: see docs/design/agent-relay.md, "Composing
agents".

Needs a release build with IPC (`cargo build --release --features ipc --bin reflex`),
a GPU and a dense/MoE GGUF, e.g. Qwen3-0.6B. Python 3.8+, standard library only.

    examples/relay_orchestrator.py --reflex target/release/reflex --model Qwen3-0.6B-Q4_K_M.gguf
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

ENVELOPE_VERSION = 1

ROUTER_ROLE = (
    "You are the router of a small software team. Read each task and name the team "
    "member who should handle it: the worker writes code, the reviewer checks code, "
    "the planner breaks large goals into steps.\n"
)
ROUTER_LABELS = ["worker", "reviewer", "planner"]
ROUTER_QUESTION = "The team member who should handle this task is the"

WORKER_ROLE = (
    "You are the worker of a small software team. For each task, write a handoff for "
    "the sub-agent who will do the work: a one-sentence summary of the task and the "
    "single next step.\n"
)
HANDOFF_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["summary", "next_step"],
}
SUBAGENT_INSTRUCTION = "Sub-agent: I will now carry out the next step. First,"

DEFAULT_TASK = "Add a --dry-run flag to the deploy script so it prints the commands instead of running them."


class RelayError(RuntimeError):
    pass


def task_line(task):
    return "Task: " + task + "\n"


def handoff_prompt(task):
    """The worker's continuation after its pinned role."""
    return task_line(task) + "Handoff JSON:\n"


def export_prompt(task, handoff):
    """The text whose state the worker exports: its role, the task and the handoff.

    It ends with a newline so the sub-agent's continuation starts on a token
    boundary: a resumed request matches the joined text only if the split falls
    between tokens.
    """
    return WORKER_ROLE + handoff_prompt(task) + handoff + "\n"


class WarmAgent:
    """One `reflex stdio` process: one JSON line in, JSON lines out."""

    def __init__(self, name, reflex, model, extra_args, ready_timeout_s=300.0):
        self.name = name
        self.stderr_tail = deque(maxlen=40)
        self.proc = subprocess.Popen(
            [reflex, "stdio", model] + extra_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready = threading.Event()

        def pump():
            for raw in iter(self.proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                self.stderr_tail.append(line)
                if line.startswith("REFLEX_STDIO_READY"):
                    ready.set()
            ready.set()  # EOF: the process died; request() reports why

        threading.Thread(target=pump, daemon=True).start()
        if not ready.wait(ready_timeout_s) or self.proc.poll() is not None:
            self.kill()
            raise RelayError(self._died("never became ready"))

    def _died(self, what):
        return "%s: reflex stdio %s (exit=%s):\n%s" % (
            self.name, what, self.proc.poll(), "\n".join(self.stderr_tail))

    def request(self, payload):
        """Sends one request; returns (final response, round trip in ms).

        A response with "ok": false raises RelayError with the engine's
        error_kind and message.
        """
        line = (json.dumps(payload) + "\n").encode("utf-8")
        start = time.perf_counter()
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RelayError(self._died("closed its input"))
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                self.proc.wait()
                raise RelayError(self._died("exited mid-request"))
            msg = json.loads(raw)
            if msg.get("event") == "final":
                break
        rtt = (time.perf_counter() - start) * 1000.0
        if not msg.get("ok"):
            raise RelayError("%s: engine error %s: %s" % (
                self.name, msg.get("error_kind"), msg.get("error")))
        return msg, rtt

    def close(self, timeout_s=60.0):
        """Closes stdin (the loop exits at EOF) and checks the exit status."""
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.kill()
                raise RelayError(self._died("didn't exit after EOF"))
        if self.proc.returncode != 0:
            raise RelayError(self._died("exited with an error"))

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def ipc_timings(msg, wall_ms):
    """The envelope's timings from an IPC response's `timings` object."""
    t = msg.get("timings") or {}
    out = {"wall_ms": round(wall_ms, 3)}
    for key in ("first_token_ms", "model_ms", "kv_io_ms", "mask_ms_total",
                "pinned_seq_len", "pinned_bytes"):
        if t.get(key) is not None:
            out[key] = t[key]
    return out


def envelope(agent, hop, **fields):
    env = {"version": ENVELOPE_VERSION, "agent": agent, "hop": hop}
    env.update(fields)
    return env


def emit(env):
    print(json.dumps(env), flush=True)


def run_cold_subagent(args, kv_path):
    """A fresh `reflex generate --import-kv` process. Returns (CLI --json result, wall ms)."""
    cmd = [args.reflex, "generate", args.model, SUBAGENT_INSTRUCTION,
           "--import-kv", kv_path, "--max-tokens", str(args.max_tokens), "--json"]
    cmd += weights_args(args)
    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = (time.perf_counter() - start) * 1000.0
    stdout = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RelayError("sub-agent: reflex generate exited with %d:\n%s\n%s" % (
            proc.returncode, proc.stderr.decode("utf-8", "replace")[-2000:], stdout[-2000:]))
    result = None
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and "token_text" in obj:
            result = obj
    if result is None:
        raise RelayError("sub-agent: no result line in reflex generate --json output "
                         "(was the binary built with --features ipc or json-output?):\n"
                         + stdout[-2000:])
    return result, wall


def weights_args(args):
    return ["--weights", args.weights] if args.weights else []


def default_kv_dir():
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Run a router, a worker and a cold sub-agent over Reflex, "
                    "printing one relay envelope per hop as a JSON line.")
    p.add_argument("--reflex", required=True,
                   help="the reflex binary, built with --features ipc")
    p.add_argument("--model", required=True, help="a dense/MoE GGUF, e.g. Qwen3-0.6B")
    p.add_argument("--task", default=DEFAULT_TASK, help="the task to route and hand off")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="tokens for the handoff and the sub-agent (default 64)")
    p.add_argument("--weights", choices=["f16", "f32"],
                   help="passed to every reflex process (default: the engine's)")
    p.add_argument("--kv-dtype", choices=["f32", "f16"], default="f32",
                   help="the handoff file's element type (default f32, exact resume)")
    p.add_argument("--kv-dir", default=None,
                   help="where the handoff file goes (default /dev/shm, else the temp dir)")
    p.add_argument("--keep-kv", action="store_true",
                   help="keep the handoff file instead of deleting it at the end")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kv_path = os.path.join(args.kv_dir or default_kv_dir(),
                           "relay-handoff-%d.cskv" % os.getpid())
    agents = []
    try:
        router = WarmAgent("router", args.reflex, args.model, weights_args(args))
        agents.append(router)
        worker = WarmAgent("worker", args.reflex, args.model, weights_args(args))
        agents.append(worker)

        # Each warm process pins its own role once; every later request resumes from it.
        router.request({"prompt": ROUTER_ROLE, "pin_prefix": True, "max_tokens": 1})
        worker.request({"prompt": WORKER_ROLE, "pin_prefix": True, "max_tokens": 1})

        # Hop 0: the router decides who handles the task, against its pin.
        msg, rtt = router.request({
            "prompt": task_line(args.task) + ROUTER_QUESTION,
            "use_pinned": True,
            "candidates": [" " + label for label in ROUTER_LABELS],
            "timings": True,
        })
        chosen = msg["chosen"]
        emit(envelope("router", 0, decision={
            "label": ROUTER_LABELS[chosen],
            "chosen": chosen,
            "probability": msg["candidates"][chosen]["probability"],
            "entropy": msg["entropy"],
        }, timings=ipc_timings(msg, rtt)))
        # A real orchestrator dispatches on `chosen` here. This example has one
        # worker process, which takes the task whichever label won.

        # Hop 1: the worker writes a constrained handoff over its pin, then exports
        # its role, the task and the handoff to a file. json_schema can't be combined
        # with export_kv, nor use_pinned with export_kv, so these are two requests,
        # and the export prefills the (short) role again.
        msg, rtt = worker.request({
            "prompt": handoff_prompt(args.task),
            "use_pinned": True,
            "json_schema": HANDOFF_SCHEMA,
            "max_tokens": args.max_tokens,
            "timings": True,
        })
        handoff = msg["text"]
        try:
            json.loads(handoff)
        except ValueError as e:
            raise RelayError("worker: handoff is not JSON (%s): %r" % (e, handoff))
        export = {"prompt": export_prompt(args.task, handoff), "export_kv": kv_path,
                  "max_tokens": 1, "timings": True}
        if args.kv_dtype != "f32":
            export["export_kv_dtype"] = args.kv_dtype
        exp_msg, exp_rtt = worker.request(export)
        timings = ipc_timings(msg, rtt + exp_rtt)
        exp_t = exp_msg.get("timings") or {}
        # kv_io_ms is the export's file write; the slot facts are as after it.
        for key in ("kv_io_ms", "pinned_seq_len", "pinned_bytes"):
            if exp_t.get(key) is not None:
                timings[key] = exp_t[key]
        emit(envelope("worker", 1, carry=handoff,
                      kv={"path": kv_path, "dtype": args.kv_dtype}, timings=timings))

        # The warm processes are done; shut them down before the cold hop.
        for agent in agents:
            agent.close()
        agents = []

        # Hop 2: a cold sub-agent resumes from the worker's file.
        result, wall = run_cold_subagent(args, kv_path)
        timings = {"wall_ms": round(wall, 3),
                   "first_token_ms": result["process_start_to_first_token_ms"]}
        if result.get("kv_io_ms") is not None:
            timings["kv_io_ms"] = result["kv_io_ms"]
        emit(envelope("subagent", 2, carry=result["token_text"], timings=timings))
    except RelayError as e:
        print("relay_orchestrator: %s" % e, file=sys.stderr)
        return 1
    finally:
        for agent in agents:
            agent.kill()
        if not args.keep_kv and os.path.exists(kv_path):
            os.remove(kv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
