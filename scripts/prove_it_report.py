#!/usr/bin/env python3
"""Summarize scripts/prove_it.sh's runs and compare them with a published reference.

Reads the run directory prove_it.sh writes (stdout_<i>.log and wall_ms_<i> for each
timed `reflex system1` run), takes each run's REFLEX_RUN_REPORT line, prints p50 per
phase next to the closest reference row in bench/baseline-*.json, and says what
differs between your setup and the reference's (GPU, kernel image, weight mode, model).

Being slower or faster than the reference is never an error. Exit codes:
0 = report printed, 1 = unusable runs (a missing or malformed report line, runs that
disagree about the setup, no reference files), 2 = usage error.

Usage:
  scripts/prove_it_report.py <run-dir> [--baselines 'bench/baseline-*.json']

Standard library only, so it runs anywhere Python 3.8+ does.
"""

import argparse
import glob
import json
import os
import re
import sys

REPORT_PREFIX = "REFLEX_RUN_REPORT "

# (row label, key in a run's phases, key in a baseline's "phases")
PHASES = [
    ("process launch (external - internal)", "process_launch", "process_launch"),
    ("gguf open", "gguf_open_ms", "gguf_open"),
    ("cuda init", "cuda_init_ms", "cuda_init"),
    ("model load", "model_load_ms", "model_load"),
    ("prompt eval (scoring pass)", "prompt_eval_ms", "prompt_eval"),
    ("total (process start to result)", "process_start_to_first_result_ms", "total"),
]

# The setup fields every run must agree on; a mixed set of runs can't be summarized.
SETUP_KEYS = ("command", "gpu_name", "compute_capability", "kernel_format",
              "kernel_image", "kernel_image_arch", "weights", "weights_dtype",
              "kv_cache", "model_file")

_TOKEN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\S*)')


class ReportError(Exception):
    pass


def _unquote(value):
    """Undoes Rust's `{:?}` string escaping (what the report uses for free text)."""
    body = value[1:-1]
    out = []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "u" and body[i + 2:i + 3] == "{":
            end = body.index("}", i)
            out.append(chr(int(body[i + 3:end], 16)))
            i = end + 1
        else:
            raise ReportError("bad escape in %r" % value)
    return "".join(out)


def parse_report_line(line):
    """`REFLEX_RUN_REPORT k=v ...` -> {key: str}. Quoted values are unescaped."""
    if not line.startswith(REPORT_PREFIX):
        raise ReportError("not a REFLEX_RUN_REPORT line: %r" % line[:80])
    rest = line[len(REPORT_PREFIX):].strip()
    fields = {}
    pos = 0
    while pos < len(rest):
        if rest[pos] == " ":
            pos += 1
            continue
        m = _TOKEN.match(rest, pos)
        if m is None:
            raise ReportError("unparsable REFLEX_RUN_REPORT near %r" % rest[pos:pos + 40])
        key, value = m.group(1), m.group(2)
        fields[key] = _unquote(value) if value.startswith('"') else value
        pos = m.end()
    for key in ("schema_version", "command", "process_start_to_first_result_ms"):
        if key not in fields:
            raise ReportError("REFLEX_RUN_REPORT has no %s" % key)
    return fields


def report_from_output(text):
    """The last REFLEX_RUN_REPORT line in one run's stdout, parsed."""
    lines = [l for l in text.splitlines() if l.startswith(REPORT_PREFIX)]
    if not lines:
        raise ReportError("no REFLEX_RUN_REPORT line (is the binary older than schema 1.5.0?)")
    return parse_report_line(lines[-1]), lines[-1]


def _float(fields, key, where):
    try:
        return float(fields[key])
    except (KeyError, ValueError):
        raise ReportError("%s: %s missing or not a number" % (where, key))


def load_runs(run_dir):
    """[(fields, raw_line, phases)] for stdout_1.log, stdout_2.log, ... in order."""
    runs = []
    i = 1
    while True:
        stdout = os.path.join(run_dir, "stdout_%d.log" % i)
        if not os.path.exists(stdout):
            break
        with open(stdout, encoding="utf-8", errors="replace") as f:
            fields, raw = report_from_output(f.read())
        where = "run %d" % i
        phases = {key: _float(fields, key, where) for _, key, _ in PHASES[1:]}
        wall_path = os.path.join(run_dir, "wall_ms_%d" % i)
        if os.path.exists(wall_path):
            with open(wall_path, encoding="utf-8") as f:
                wall = float(f.read().strip())
            phases["process_launch"] = wall - phases["process_start_to_first_result_ms"]
        runs.append((fields, raw, phases))
        i += 1
    if not runs:
        raise ReportError("no stdout_1.log in %s" % run_dir)
    first = runs[0][0]
    for n, (fields, _, _) in enumerate(runs[1:], start=2):
        for key in SETUP_KEYS:
            if fields.get(key) != first.get(key):
                raise ReportError("run %d reports %s=%r, run 1 %r" % (n, key, fields.get(key), first.get(key)))
    return runs


def percentile(values, p):
    """Nearest rank, the same rule as the bench scripts' awk `percentile`."""
    values = sorted(values)
    idx = int(p / 100.0 * len(values) + 0.9999)
    return values[min(max(idx, 1), len(values)) - 1]


def load_baselines(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = path
        out.append(data)
    return out


def _cc(value):
    try:
        major, minor = str(value).split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def pick_baseline(baselines, gpu_name, compute_capability):
    """(baseline, why): same GPU name, else the nearest compute capability."""
    if not baselines:
        return None, "no reference files"
    for b in baselines:
        if gpu_name and b.get("gpu", "").lower() == gpu_name.lower():
            return b, "same GPU"
    mine = _cc(compute_capability)
    scored = []
    for b in baselines:
        theirs = _cc(b.get("compute_capability"))
        if mine is None or theirs is None:
            continue
        distance = abs((mine[0] * 10 + mine[1]) - (theirs[0] * 10 + theirs[1]))
        scored.append((distance, b.get("gpu", ""), b))
    if not scored:
        return baselines[0], "no compute capability to match on"
    scored.sort(key=lambda t: (t[0], t[1]))
    distance, _, b = scored[0]
    if distance == 0:
        return b, "same compute capability (%s)" % compute_capability
    return b, "nearest compute capability (yours %s, reference %s)" % (
        compute_capability, b.get("compute_capability"))


def kernel_image(fields):
    parts = [fields.get("kernel_format"), fields.get("kernel_image"), fields.get("kernel_image_arch")]
    return " ".join(p for p in parts if p)


def differences(fields, n_runs, baseline):
    """What differs between this setup and the reference's, one phrase each."""
    out = []
    pairs = (
        ("GPU", fields.get("gpu_name"), baseline.get("gpu")),
        ("kernel image", kernel_image(fields), baseline.get("kernel_image")),
        ("weights", fields.get("weights"), baseline.get("weights")),
        ("KV cache", fields.get("kv_cache"), baseline.get("kv_cache")),
        ("model", fields.get("model_file"), baseline.get("model")),
        ("command", fields.get("command"), baseline.get("subcommand")),
    )
    for label, mine, theirs in pairs:
        if mine and theirs and mine != theirs:
            out.append("%s %s (reference %s)" % (label, mine, theirs))
    if baseline.get("n_runs") and baseline["n_runs"] != n_runs:
        out.append("n=%d (reference n=%d)" % (n_runs, baseline["n_runs"]))
    if fields.get("kernel_image") == "ptx_jit":
        out.append("PTX is JIT-compiled by the driver at load; a warm JIT cache hides most of it")
    return out


def render(runs, baseline, why, baselines):
    fields = runs[0][0]
    n = len(runs)
    lines = []
    gpu = fields.get("gpu_name", "unknown GPU")
    lines.append("Your GPU: %s (compute capability %s), kernels %s, weights %s, KV cache %s"
                 % (gpu, fields.get("compute_capability", "?"), kernel_image(fields) or "?",
                    fields.get("weights", "?"), fields.get("kv_cache", "?")))
    lines.append("Model: %s, reflex %s, n=%d cold runs" % (fields.get("model_file", "?"), fields["command"], n))
    lines.append("")
    if baseline is None:
        lines.append("No reference row to compare with.")
        ref_phases = {}
    else:
        ref_phases = baseline.get("phases", {})
        lines.append("Reference: %s, %s (%s), %s" % (
            baseline.get("gpu", "?"), baseline.get("recorded", "?"), why,
            os.path.basename(baseline["_path"])))
        if baseline.get("published"):
            lines.append("  published in %s" % baseline["published"])
        if baseline.get("note"):
            lines.append("  note: %s" % baseline["note"])
        diff = differences(fields, n, baseline)
        lines.append("  differs: " + ("; ".join(diff) if diff else "nothing the report shows"))
    lines.append("")
    lines.append("| phase | yours p50 ms | reference p50 ms | delta |")
    lines.append("|---|---|---|---|")
    for label, key, ref_key in PHASES:
        values = [p[key] for _, _, p in runs if key in p]
        mine = percentile(values, 50) if values else None
        ref = ref_phases.get(ref_key, {}).get("p50_ms")
        mine_s = "%.1f" % mine if mine is not None else "n/a"
        ref_s = "%.1f" % ref if ref is not None else "-"
        if mine is not None and ref:
            delta = "%+.1f%%" % ((mine - ref) / ref * 100.0)
        else:
            delta = "-"
        lines.append("| %s | %s | %s | %s |" % (label, mine_s, ref_s, delta))
    others = [b for b in baselines if b is not baseline]
    if others:
        lines.append("")
        lines.append("Other references: " + ", ".join(
            "%s (%s)" % (b.get("gpu", "?"), os.path.basename(b["_path"])) for b in others))
    lines.append("")
    lines.append("Paste this with a benchmark or bug report:")
    lines.append(runs[len(runs) // 2][1])
    return "\n".join(lines) + "\n"


def main(argv=None):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="directory with stdout_<i>.log and wall_ms_<i> files")
    ap.add_argument("--baselines", default=os.path.join(root, "bench", "baseline-*.json"),
                    help="glob of reference JSON files (default bench/baseline-*.json)")
    args = ap.parse_args(argv)
    try:
        runs = load_runs(args.run_dir)
        baselines = load_baselines(args.baselines)
        if not baselines:
            raise ReportError("no reference files match %s" % args.baselines)
        fields = runs[0][0]
        baseline, why = pick_baseline(baselines, fields.get("gpu_name"), fields.get("compute_capability"))
    except (OSError, ValueError, ReportError) as e:
        print("prove_it_report: %s" % e, file=sys.stderr)
        return 1
    sys.stdout.write(render(runs, baseline, why, baselines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
