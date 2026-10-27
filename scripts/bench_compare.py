#!/usr/bin/env python3
"""Compare a cold-start phase benchmark against a committed baseline.

Parses the markdown table that scripts/bench_cold_start_phases_system1.sh (or
bench_cold_start_phases.sh) prints, compares each phase's p50 against a baseline
JSON such as bench/baseline-t4.json, prints a markdown report (suitable for
$GITHUB_STEP_SUMMARY), and exits non-zero if the *total* p50 regressed by more
than the threshold. Per-phase deltas are reported for diagnosis only, since
single phases (process launch, gguf open) are noisy at the tens-of-ms scale.

With --relay it reads scripts/bench_relay_nightly.sh's output instead: three
scripts/agent_relay_bench.py tables over one warm `reflex stdio` (pinned-prefix,
`candidates` and `json_schema` hops), compared against a baseline such as
bench/baseline-relay-t4.json. Each hop kind has one gated p50 (RELAY_KINDS); it
regresses when it grows by more than the threshold *and* by at least
--min-delta-ms, since a warm hop is small enough for a few ms of jitter to be a
large percentage. Fewer valid JSON outputs on the constrained hop also count as a
regression. A baseline value of "TBD" means none was measured yet: the report says
so and the run exits 0.

Exit codes: 0 = within threshold (or regression against a provisional baseline,
reported as a warning), 1 = regression, 2 = unusable input (missing phases, bad
files) -- a broken bench must never read as a pass.

Usage:
  scripts/bench_compare.py <bench-output.txt> --baseline bench/baseline-t4.json
      [--threshold-pct 15] [--strict] [--summary-file $GITHUB_STEP_SUMMARY]
  scripts/bench_compare.py <bench-output.txt> --write-baseline <new.json>
      [--source "nightly run <url>"]    # intentional baseline update, see docs/gpu-ci.md
  scripts/bench_compare.py <relay-output.txt> --relay --baseline bench/baseline-relay-t4.json
      [--threshold-pct 15] [--min-delta-ms 3] [--strict] [--summary-file ...]
  scripts/bench_compare.py <relay-output.txt> --relay --write-baseline <new.json>
      [--provisional] [--source ...]

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


# Hop kind -> (gated metric, every metric kept from its table). Keys are
# scripts/agent_relay_bench.py's METRICS keys; its table labels map back to them.
RELAY_KINDS = {
    "pinned": ("rtt_ms", ["rtt_ms", "first_token_ms", "kv_io_ms", "decode_ms_per_token", "tokens"]),
    "candidates": ("s1_rtt_ms", ["s1_rtt_ms", "s1_model_ms", "s1_kv_io_ms"]),
    "constrained": ("rtt_ms", ["rtt_ms", "first_token_ms", "mask_ms", "tokens"]),
}
# Config fields a relay baseline must share with a run to be like-for-like.
RELAY_CONFIG = ("model", "target", "hops", "warmup", "max_tokens", "fixed_notes", "sampling")

_RELAY_CONFIG = re.compile(r"^Agent relay bench config: (.*)$")
_RELAY_HEADER = re.compile(r"^## agent relay: (.+), (\d+) hops\s*$")
_RELAY_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(-?[0-9.]+)\s*\|\s*(-?[0-9.]+)\s*\|\s*(-?[0-9.]+)\s*\|$")
_RELAY_VALID = re.compile(r"^valid JSON outputs: (\d+)/(\d+)\s*$")


def _relay_labels():
    # Imported lazily so the system1 path never depends on the relay script.
    from agent_relay_bench import METRICS

    return {label: key for key, label in METRICS}


def _relay_kind(label):
    if "prefix pinned" not in label:
        return None
    if "system1" in label:
        return "candidates"
    if "json_schema" in label:
        return "constrained"
    return "pinned"


def _config_value(v):
    return int(v) if v.isdigit() else v


def parse_relay_output(text):
    """Returns {"config": {...}, "kinds": {kind: {"label", "n_hops", "metrics", "valid"}}}."""
    labels = _relay_labels()
    config = {}
    kinds = {}
    section = None
    for line in text.splitlines():
        line = line.strip()
        m = _RELAY_CONFIG.match(line)
        if m:
            config = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
            config = {k: _config_value(v) for k, v in config.items()}
            continue
        m = _RELAY_HEADER.match(line)
        if m:
            kind = _relay_kind(m.group(1))
            section = None
            if kind is not None:
                if kind in kinds:
                    raise BenchParseError(f"relay output has two {kind!r} tables")
                section = kinds[kind] = {
                    "label": m.group(1),
                    "n_hops": int(m.group(2)),
                    "metrics": {},
                }
            continue
        if section is None:
            continue
        m = _RELAY_ROW.match(line)
        if m and m.group(1) in labels:
            section["metrics"][labels[m.group(1)]] = {
                "p50": float(m.group(2)),
                "p95": float(m.group(3)),
            }
            continue
        m = _RELAY_VALID.match(line)
        if m:
            section["valid"] = {"count": int(m.group(1)), "of": int(m.group(2))}
    missing = [k for k in RELAY_KINDS if k not in kinds]
    if missing:
        raise BenchParseError(f"relay output is missing hop kinds: {', '.join(missing)}")
    for kind, (gate, keep) in RELAY_KINDS.items():
        found = kinds[kind]["metrics"]
        if gate not in found:
            raise BenchParseError(f"relay {kind!r} table has no {gate} row")
        kinds[kind]["metrics"] = {k: found[k] for k in keep if k in found}
    if "valid" not in kinds["constrained"]:
        raise BenchParseError("relay 'constrained' table has no valid-JSON count")
    return {"config": config, "kinds": kinds}


def _measured(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_relay(current, baseline, threshold_pct, min_delta_ms):
    """Returns rows: (kind, metric, base_p50, cur_p50, delta_pct, status), one per gate
    plus one for the constrained hop's valid-JSON count. status: ok, regressed,
    no baseline. base_p50 is None when the baseline has no number yet."""
    base_kinds = baseline.get("hop_kinds", {})
    missing = [k for k in RELAY_KINDS if k not in base_kinds]
    if missing:
        raise BenchParseError(f"baseline is missing hop kinds: {', '.join(missing)}")
    rows = []
    for kind, (gate, _) in RELAY_KINDS.items():
        base = base_kinds[kind].get("metrics", {}).get(gate, {}).get("p50")
        cur = current["kinds"][kind]["metrics"][gate]["p50"]
        if not _measured(base):
            rows.append((kind, gate, None, cur, None, "no baseline"))
            continue
        delta = (cur - base) / base * 100.0 if base > 0 else 0.0
        regressed = delta > threshold_pct and cur - base >= min_delta_ms
        rows.append((kind, gate, base, cur, delta, "regressed" if regressed else "ok"))
    base_valid = base_kinds["constrained"].get("valid", {}).get("count")
    cur_valid = current["kinds"]["constrained"]["valid"]["count"]
    if _measured(base_valid):
        status = "regressed" if cur_valid < base_valid else "ok"
        rows.append(("constrained", "valid", base_valid, cur_valid, None, status))
    else:
        rows.append(("constrained", "valid", None, cur_valid, None, "no baseline"))
    return rows


def _fmt(v, d=1):
    return f"{v:.{d}f}" if _measured(v) else "TBD"


def render_relay_report(rows, threshold_pct, min_delta_ms, baseline, current):
    provisional = baseline.get("provisional", False)
    statuses = {r[5] for r in rows}
    if "regressed" in statuses:
        verdict = "REGRESSION" + (" (provisional baseline, warning only)" if provisional else "")
    elif statuses == {"no baseline"}:
        verdict = "NO BASELINE YET"
    else:
        verdict = "OK"
    lines = [f"### Agent relay bench vs baseline: {verdict}", ""]
    note = baseline.get("provisional_reason") or "replace it after the first real nightly run"
    lines.append(
        f"A hop kind fails when its gated p50 grows by more than +{threshold_pct:g}% "
        f"and by at least {min_delta_ms:g} ms, or the constrained hop loses valid JSON "
        f"outputs. Baseline: {baseline.get('source', 'unknown source')}"
        + (f" -- **provisional** ({note})." if provisional else "")
    )
    cfg = current.get("config") or {}
    if cfg:
        lines.append("Current run: " + ", ".join(f"{k}={v}" for k, v in cfg.items()) + ".")
    mismatched = [
        k
        for k in RELAY_CONFIG
        if _measured_or_str(baseline.get(k)) and cfg.get(k) is not None and baseline.get(k) != cfg.get(k)
    ]
    for kind in RELAY_KINDS:
        base_tokens = baseline["hop_kinds"][kind].get("metrics", {}).get("tokens", {}).get("p50")
        cur_tokens = current["kinds"][kind]["metrics"].get("tokens", {}).get("p50")
        if _measured(base_tokens) and cur_tokens is not None and base_tokens != cur_tokens:
            mismatched.append(f"{kind} tokens")
    if mismatched:
        lines.append(
            f"**Warning:** baseline and run differ in {', '.join(mismatched)}; "
            "the comparison is not like-for-like."
        )
    lines += [
        "",
        "| hop kind | gated | baseline p50 | current p50 | delta | status |",
        "|---|---|---|---|---|---|",
    ]
    for kind, metric, base, cur, delta, status in rows:
        d = 0 if metric == "valid" else 1
        shown = "" if delta is None else f"{delta:+.1f}%"
        lines.append(f"| {kind} | {metric} | {_fmt(base, d)} | {_fmt(cur, d)} | {shown} | {status} |")
    lines += ["", "Other p50s, for diagnosis only:", ""]
    lines += ["| hop kind | metric | baseline p50 | current p50 |", "|---|---|---|---|"]
    for kind, (gate, keep) in RELAY_KINDS.items():
        base_metrics = baseline["hop_kinds"][kind].get("metrics", {})
        for key in keep:
            if key == gate or key not in current["kinds"][kind]["metrics"]:
                continue
            base = base_metrics.get(key, {}).get("p50")
            cur = current["kinds"][kind]["metrics"][key]["p50"]
            d = 3 if key.startswith("mask") else 2
            lines.append(f"| {kind} | {key} | {_fmt(base, d)} | {_fmt(cur, d)} |")
    return "\n".join(lines) + "\n"


def _measured_or_str(v):
    return _measured(v) or (isinstance(v, str) and v != "TBD")


def build_relay_baseline(current, source, provisional):
    out = {"provisional": provisional}
    if provisional:
        out["provisional_reason"] = "until two scheduled nightly runs exist"
    out.update({"source": source, "recorded": datetime.date.today().isoformat(), "gpu": "Tesla T4"})
    config = current.get("config") or {}
    for k in RELAY_CONFIG:
        out[k] = config.get(k)
    out["hop_kinds"] = {}
    for kind, (gate, _) in RELAY_KINDS.items():
        entry = {"gate": gate, "metrics": current["kinds"][kind]["metrics"]}
        if "valid" in current["kinds"][kind]:
            entry["valid"] = current["kinds"][kind]["valid"]
        out["hop_kinds"][kind] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench_output", help="file holding the bench script's stdout")
    ap.add_argument("--baseline", help="baseline JSON to compare against")
    ap.add_argument("--threshold-pct", type=float, default=15.0,
                    help="max allowed total-p50 regression (with --relay: per gated hop p50), "
                         "percent (default 15)")
    ap.add_argument("--relay", action="store_true",
                    help="the input is scripts/bench_relay_nightly.sh's output")
    ap.add_argument("--min-delta-ms", type=float, default=3.0,
                    help="--relay: a gated p50 must also grow by at least this many ms (default 3)")
    ap.add_argument("--strict", action="store_true",
                    help="fail on regression even when the baseline is provisional")
    ap.add_argument("--summary-file", help="also append the markdown report here")
    ap.add_argument("--write-baseline", metavar="PATH",
                    help="write a new (non-provisional) baseline from this run instead of comparing")
    ap.add_argument("--provisional", action="store_true",
                    help="--relay --write-baseline: mark the written baseline provisional")
    ap.add_argument("--source", default="manual update",
                    help="provenance note stored in a written baseline")
    args = ap.parse_args(argv)
    if args.provisional and not (args.relay and args.write_baseline):
        ap.error("--provisional is for --relay --write-baseline")

    try:
        with open(args.bench_output, encoding="utf-8") as f:
            text = f.read()
        current = parse_relay_output(text) if args.relay else parse_bench_output(text)
        if args.write_baseline:
            if args.relay:
                written = build_relay_baseline(current, args.source, args.provisional)
            else:
                written = build_baseline(current, args.source)
            with open(args.write_baseline, "w", encoding="utf-8") as f:
                json.dump(written, f, indent=2)
                f.write("\n")
            print(f"wrote baseline {args.write_baseline}")
            return 0
        if not args.baseline:
            ap.error("--baseline is required unless --write-baseline is given")
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
        if args.relay:
            rows = compare_relay(current, baseline, args.threshold_pct, args.min_delta_ms)
            regressed = any(r[5] == "regressed" for r in rows)
        else:
            rows, total_delta, regressed = compare(current, baseline, args.threshold_pct)
    except (OSError, ValueError, BenchParseError) as e:
        print(f"bench_compare: {e}", file=sys.stderr)
        return 2

    if args.relay:
        report = render_relay_report(rows, args.threshold_pct, args.min_delta_ms, baseline, current)
    else:
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
