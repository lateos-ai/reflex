#!/usr/bin/env python3
"""Compare a cold-start phase benchmark against a committed baseline.

Parses the markdown table that scripts/bench_cold_start_phases_system1.sh (or
bench_cold_start_phases.sh) prints, compares each phase's p50 against a baseline
JSON such as bench/baseline-t4.json, prints a markdown report (suitable for
$GITHUB_STEP_SUMMARY), and exits non-zero if the *total* p50 regressed by more
than the threshold. Per-phase deltas are reported for diagnosis only, since
single phases (process launch, gguf open) are noisy at the tens-of-ms scale.

Exit codes: 0 = within threshold (or regression against a provisional baseline,
reported as a warning), 1 = regression, 2 = unusable input (missing phases, bad
files) -- a broken bench must never read as a pass.

Usage:
  scripts/bench_compare.py <bench-output.txt> --baseline bench/baseline-t4.json
      [--threshold-pct 15] [--strict] [--summary-file $GITHUB_STEP_SUMMARY]
  scripts/bench_compare.py <bench-output.txt> --write-baseline <new.json>
      [--source "nightly run <url>"]    # intentional baseline update, see docs/gpu-ci.md

Standard library only, so it runs anywhere Python 3.8+ does.
"""

import argparse
import datetime
import json
import re
import sys

# Row-label prefix in the bench table -> stable key used in the baseline JSON.
# Prefix match, because labels carry explanatory suffixes that may be reworded.
PHASES = [
    ("process launch", "process_launch"),
    ("gguf open", "gguf_open"),
    ("cuda init", "cuda_init"),
    ("model load", "model_load"),
    ("prompt eval", "prompt_eval"),
    ("total", "total"),
]
REQUIRED = [key for _, key in PHASES]

_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+|n/a)[^|]*\|\s*([0-9.]+|n/a)[^|]*\|")
_HEADER = re.compile(r"Cold-start phase breakdown \(reflex (\w+)\), n=(\d+) runs, (\S+?):?\s*$")


class BenchParseError(Exception):
    pass


def parse_bench_output(text):
    """Returns {"meta": {...}, "phases": {key: {"p50_ms", "p95_ms"}}}."""
    meta = {}
    phases = {}
    for line in text.splitlines():
        header = _HEADER.search(line)
        if header:
            meta = {
                "subcommand": header.group(1),
                "n_runs": int(header.group(2)),
                "model": header.group(3),
            }
            continue
        row = _ROW.match(line.strip())
        if not row:
            continue
        label = row.group(1).lower()
        key = next((k for prefix, k in PHASES if label.startswith(prefix)), None)
        if key is None:
            continue
        p50, p95 = row.group(2), row.group(3)
        if p50 == "n/a":
            raise BenchParseError(f"phase {key!r} has no measurements (row: {line.strip()})")
        phases[key] = {"p50_ms": float(p50), "p95_ms": None if p95 == "n/a" else float(p95)}
    missing = [k for k in REQUIRED if k not in phases]
    if missing:
        raise BenchParseError(f"bench output is missing phases: {', '.join(missing)}")
    return {"meta": meta, "phases": phases}


def compare(current, baseline, threshold_pct):
    """Returns (rows, total_delta_pct, regressed). rows: (key, base_p50, cur_p50, delta_pct)."""
    base_phases = baseline.get("phases", {})
    missing = [k for k in REQUIRED if k not in base_phases]
    if missing:
        raise BenchParseError(f"baseline is missing phases: {', '.join(missing)}")
    rows = []
    for key in REQUIRED:
        base = base_phases[key]["p50_ms"]
        cur = current["phases"][key]["p50_ms"]
        delta = (cur - base) / base * 100.0 if base > 0 else 0.0
        rows.append((key, base, cur, delta))
    total_delta = rows[-1][3]
    return rows, total_delta, total_delta > threshold_pct


def render_report(rows, total_delta, regressed, threshold_pct, baseline, current):
    provisional = baseline.get("provisional", False)
    lines = []
    if regressed:
        verdict = "REGRESSION" + (" (provisional baseline, warning only)" if provisional else "")
    else:
        verdict = "OK"
    lines.append(f"### Cold-start bench vs baseline: {verdict}")
    lines.append("")
    lines.append(
        f"Total p50 changed by **{total_delta:+.1f}%** (fails above +{threshold_pct:g}%). "
        f"Baseline: {baseline.get('source', 'unknown source')}"
        + (" -- **provisional**, replace it after the first real nightly run." if provisional else "")
    )
    meta = current.get("meta") or {}
    if meta:
        lines.append(
            f"Current run: `reflex {meta.get('subcommand')}`, n={meta.get('n_runs')}, {meta.get('model')}."
        )
    base_meta = {k: baseline.get(k) for k in ("subcommand", "model")}
    mismatched = [k for k, v in base_meta.items() if v and meta.get(k) and v != meta.get(k)]
    if mismatched:
        lines.append(f"**Warning:** baseline and run differ in {', '.join(mismatched)}; the comparison is not like-for-like.")
    lines.append("")
    lines.append("| phase | baseline p50 ms | current p50 ms | delta |")
    lines.append("|---|---|---|---|")
    for key, base, cur, delta in rows:
        lines.append(f"| {key} | {base:.1f} | {cur:.1f} | {delta:+.1f}% |")
    return "\n".join(lines) + "\n"


def build_baseline(current, source):
    meta = current.get("meta") or {}
    return {
        "provisional": False,
        "source": source,
        "recorded": datetime.date.today().isoformat(),
        "gpu": "Tesla T4",
        "subcommand": meta.get("subcommand"),
        "model": meta.get("model"),
        "n_runs": meta.get("n_runs"),
        "phases": current["phases"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench_output", help="file holding the bench script's stdout")
    ap.add_argument("--baseline", help="baseline JSON to compare against")
    ap.add_argument("--threshold-pct", type=float, default=15.0,
                    help="max allowed total-p50 regression, percent (default 15)")
    ap.add_argument("--strict", action="store_true",
                    help="fail on regression even when the baseline is provisional")
    ap.add_argument("--summary-file", help="also append the markdown report here")
    ap.add_argument("--write-baseline", metavar="PATH",
                    help="write a new (non-provisional) baseline from this run instead of comparing")
    ap.add_argument("--source", default="manual update",
                    help="provenance note stored in a written baseline")
    args = ap.parse_args(argv)

    try:
        with open(args.bench_output, encoding="utf-8") as f:
            current = parse_bench_output(f.read())
        if args.write_baseline:
            with open(args.write_baseline, "w", encoding="utf-8") as f:
                json.dump(build_baseline(current, args.source), f, indent=2)
                f.write("\n")
            print(f"wrote baseline {args.write_baseline}")
            return 0
        if not args.baseline:
            ap.error("--baseline is required unless --write-baseline is given")
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
        rows, total_delta, regressed = compare(current, baseline, args.threshold_pct)
    except (OSError, ValueError, BenchParseError) as e:
        print(f"bench_compare: {e}", file=sys.stderr)
        return 2

    report = render_report(rows, total_delta, regressed, args.threshold_pct, baseline, current)
    print(report)
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(report)
    if regressed and (args.strict or not baseline.get("provisional", False)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
