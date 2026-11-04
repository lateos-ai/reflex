#!/usr/bin/env python3
"""One-row results table for a GPU nightly run.

Reads the bench outputs the nightly writes under bench-results/ and prints one
markdown table row: date, commit, GPU, cold `system1` and `generate` p50, whole-process
gross energy p50, and the warm relay hops' gated p50s. The nightly appends it to the job
summary and keeps it as bench-results/nightly-summary.md in the run's artifact; the
same header is docs/gpu-ci.md's "Latest results" table, so refreshing that table at
release time is copying the row.

Inputs (each optional):
  --system1   scripts/bench_cold_start_phases_system1.sh output
  --generate  scripts/bench_cold_start_phases.sh output
  --energy    scripts/bench_cold_energy.sh output
  --relay     scripts/bench_relay_nightly.sh output

An input that is missing or doesn't parse becomes an "n/a" cell plus a note saying
why; a cell is never filled from anywhere but its input. The table is informational
(the compare steps do the gating), so this exits 0 unless its arguments are wrong.

Usage:
  scripts/nightly_summary.py --commit <sha> --gpu "Tesla T4" [--date YYYY-MM-DD]
      [--system1 F] [--generate F] [--energy F] [--relay F]
      [--out bench-results/nightly-summary.md] [--summary-file $GITHUB_STEP_SUMMARY]

Standard library only, so it runs anywhere Python 3.8+ does.
"""

import argparse
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_compare  # noqa: E402

COLUMNS = [
    "date",
    "commit",
    "GPU",
    "system1 p50",
    "generate p50",
    "gross energy p50",
    "relay pinned p50",
    "relay candidates p50",
    "relay constrained p50",
]
NA = "n/a"

_PHASES_HEADER = re.compile(
    r"Cold-start phase breakdown(?: \(reflex \w+\))?, n=(\d+) runs, (\S+?):?\s*$")
_ENERGY_HEADER = re.compile(r"Whole-process energy \(reflex (\w+)\), n=(\d+) runs, ([^,]+),")
_ENERGY_GROSS = re.compile(r"^\|\s*gross energy[^|]*\|\s*([0-9.]+|n/a)\s*\|")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def cold_total(text):
    """(total p50 ms, n_runs or None, model or None) from a cold-start phase table."""
    total = bench_compare.parse_bench_output(text)["phases"]["total"]["p50_ms"]
    n, model = None, None
    for line in text.splitlines():
        m = _PHASES_HEADER.search(line)
        if m:
            n, model = int(m.group(1)), m.group(2)
    return total, n, model


def gross_energy(text):
    """(gross energy p50 in J, n_runs or None, model or None) from bench_cold_energy.sh.

    Gross, not net of idle: with the counter read on its updates, gross is stable run to
    run, while the ~1 J net of a cold start moves with the idle window and read method.
    """
    n, model, gross = None, None, None
    for line in text.splitlines():
        line = line.strip()
        m = _ENERGY_HEADER.search(line)
        if m:
            n, model = int(m.group(2)), m.group(3).strip()
            continue
        m = _ENERGY_GROSS.match(line)
        if m:
            if m.group(1) == NA:
                raise bench_compare.BenchParseError("gross energy row has no measurements")
            gross = float(m.group(1))
    if gross is None:
        raise bench_compare.BenchParseError("no gross energy row")
    return gross, n, model


def relay_p50s(text):
    """({kind: gated p50 ms}, hops or None, valid "k/n" or None) from the relay bench."""
    parsed = bench_compare.parse_relay_output(text)
    p50s = {}
    for kind, (gate, _) in bench_compare.RELAY_KINDS.items():
        p50s[kind] = parsed["kinds"][kind]["metrics"][gate]["p50"]
    valid = parsed["kinds"]["constrained"]["valid"]
    hops = parsed["config"].get("hops")
    return p50s, hops, f"{valid['count']}/{valid['of']}"


def _load(name, path, parse, notes):
    """parse(read(path)), or None with a note when the input is absent or unusable."""
    if not path:
        notes.append(f"{name}: not given")
        return None
    if not os.path.exists(path):
        notes.append(f"{name}: {path} was not produced")
        return None
    try:
        return parse(_read(path))
    except (OSError, ValueError, KeyError, bench_compare.BenchParseError) as e:
        notes.append(f"{name}: {path} did not parse ({e})")
        return None


def build(args):
    """Returns (row cells, detail lines, notes)."""
    notes = []
    details = []
    system1 = _load("system1", args.system1, cold_total, notes)
    generate = _load("generate", args.generate, cold_total, notes)
    energy = _load("energy", args.energy, gross_energy, notes)
    relay = _load("relay", args.relay, relay_p50s, notes)

    def ms(v):
        return NA if v is None else f"{v:.1f} ms"

    cells = [
        args.date,
        args.commit[:7] if args.commit else NA,
        args.gpu or NA,
        ms(system1 and system1[0]),
        ms(generate and generate[0]),
        NA if energy is None else f"{energy[0]:.2f} J",
    ]
    if relay is None:
        cells += [NA] * 3
    else:
        cells += [ms(relay[0][kind]) for kind in ("pinned", "candidates", "constrained")]

    for name, got, what in (
        ("system1", system1, "cold processes, process start to result"),
        ("generate", generate, "cold processes, process start to first token"),
        ("gross energy", energy, "cold `system1` processes, whole process, not net of idle"),
    ):
        if got is not None:
            n = f"n={got[1]} " if got[1] is not None else ""
            model = f", {got[2]}" if got[2] else ""
            details.append(f"{name}: {n}{what}{model}")
    if relay is not None:
        hops = f"{relay[1]} hops" if relay[1] is not None else "hops"
        details.append(
            f"relay: {hops} per kind on one warm `reflex stdio`, gated p50 "
            f"(pinned and constrained round trip, candidates System-1 round trip); "
            f"constrained valid JSON {relay[2]}")
    return cells, details, notes


def render(cells, details, notes):
    lines = ["### Nightly results", ""]
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("|" + "---|" * len(COLUMNS))
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for d in details:
        lines.append(f"- {d}")
    for n in notes:
        lines.append(f"- missing: {n}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    help="run date (default: today, UTC)")
    ap.add_argument("--commit", help="commit the run built")
    ap.add_argument("--gpu", help="GPU name, e.g. from nvidia-smi")
    ap.add_argument("--system1", help="bench_cold_start_phases_system1.sh output")
    ap.add_argument("--generate", help="bench_cold_start_phases.sh output")
    ap.add_argument("--energy", help="bench_cold_energy.sh output")
    ap.add_argument("--relay", help="bench_relay_nightly.sh output")
    ap.add_argument("--out", help="also write the table to this file")
    ap.add_argument("--summary-file", help="also append the table here")
    args = ap.parse_args(argv)

    report = render(*build(args))
    print(report, end="")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
