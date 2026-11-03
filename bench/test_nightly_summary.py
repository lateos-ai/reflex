"""Unit tests for scripts/nightly_summary.py, against sample bench logs in bench/testdata/.

Run from the repo root: python -m unittest discover -s bench -v
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nightly_summary  # noqa: E402

TESTDATA = os.path.join(HERE, "testdata")


def sample(name):
    return os.path.join(TESTDATA, name)


ALL_INPUTS = [
    "--system1", sample("system1_ok.txt"),
    "--generate", sample("generate_ok.txt"),
    "--energy", sample("energy_ok.txt"),
    "--relay", sample("relay_ok.txt"),
]
META = ["--date", "2026-01-02", "--commit", "0123456789abcdef", "--gpu", "Tesla T4"]


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = nightly_summary.main(argv)
    return code, out.getvalue()


def row_cells(report):
    """The cells of the table's one data row."""
    rows = [l for l in report.splitlines() if l.startswith("| ") and not l.startswith("| date")]
    assert len(rows) == 1, rows
    return [c.strip() for c in rows[0].strip("|").split("|")]


class ParserTest(unittest.TestCase):
    def test_system1_total(self):
        with open(sample("system1_ok.txt"), encoding="utf-8") as f:
            self.assertEqual(nightly_summary.cold_total(f.read()),
                             (489.0, 10, "Qwen3-0.6B-Q4_K_M.gguf"))

    def test_generate_header_has_no_subcommand(self):
        # bench_cold_start_phases.sh's header omits "(reflex generate)".
        with open(sample("generate_ok.txt"), encoding="utf-8") as f:
            self.assertEqual(nightly_summary.cold_total(f.read()),
                             (400.0, 10, "Qwen3-0.6B-Q4_K_M.gguf"))

    def test_net_energy(self):
        with open(sample("energy_ok.txt"), encoding="utf-8") as f:
            self.assertEqual(nightly_summary.net_energy(f.read()),
                             (12.0, 10, "Qwen3-0.6B-Q4_K_M.gguf"))

    def test_net_energy_without_values_is_an_error(self):
        with open(sample("energy_no_counter.txt"), encoding="utf-8") as f:
            text = f.read()
        with self.assertRaises(nightly_summary.bench_compare.BenchParseError):
            nightly_summary.net_energy(text)

    def test_relay_gated_p50s(self):
        with open(sample("relay_ok.txt"), encoding="utf-8") as f:
            p50s, hops, valid = nightly_summary.relay_p50s(f.read())
        self.assertEqual(p50s, {"pinned": 230.0, "candidates": 10.0, "constrained": 260.0})
        self.assertEqual(hops, 20)
        self.assertEqual(valid, "20/20")


class TableTest(unittest.TestCase):
    def test_full_row(self):
        code, report = run(META + ALL_INPUTS)
        self.assertEqual(code, 0)
        self.assertEqual(row_cells(report), [
            "2026-01-02", "0123456", "Tesla T4", "489.0 ms", "400.0 ms", "12.00 J",
            "230.0 ms", "10.0 ms", "260.0 ms",
        ])
        self.assertNotIn("missing:", report)
        self.assertIn("relay: 20 hops per kind", report)
        self.assertIn("constrained valid JSON 20/20", report)

    def test_header_matches_docs_latest_results(self):
        # Refreshing docs/gpu-ci.md is copying the row, so the headers must agree.
        code, report = run(META + ALL_INPUTS)
        header = next(l for l in report.splitlines() if l.startswith("| date"))
        with open(os.path.join(ROOT, "docs", "gpu-ci.md"), encoding="utf-8") as f:
            docs = f.read()
        self.assertIn("## Latest results", docs)
        self.assertIn(header, docs.split("## Latest results", 1)[1])

    def test_missing_file_is_na_with_a_note(self):
        argv = META + ALL_INPUTS[:4] + ["--energy", sample("does_not_exist.txt"),
                                        "--relay", sample("relay_ok.txt")]
        code, report = run(argv)
        self.assertEqual(code, 0)
        cells = row_cells(report)
        self.assertEqual(cells[5], "n/a")
        self.assertEqual(cells[3], "489.0 ms")
        self.assertRegex(report, r"missing: energy: .*does_not_exist\.txt was not produced")

    def test_unparsable_input_is_na_with_a_note(self):
        argv = META + ["--system1", sample("system1_missing_phase.txt"),
                       "--relay", sample("relay_missing_kind.txt"),
                       "--energy", sample("energy_no_counter.txt")]
        code, report = run(argv)
        self.assertEqual(code, 0)
        cells = row_cells(report)
        self.assertEqual(cells[3:], ["n/a"] * 6)
        self.assertRegex(report, r"missing: system1: .* did not parse")
        self.assertRegex(report, r"missing: relay: .* did not parse")
        self.assertRegex(report, r"missing: energy: .* did not parse")
        self.assertIn("missing: generate: not given", report)

    def test_no_inputs_only_meta(self):
        code, report = run(["--date", "2026-01-02"])
        self.assertEqual(code, 0)
        self.assertEqual(row_cells(report), ["2026-01-02"] + ["n/a"] * 8)

    def test_no_number_without_an_input(self):
        # Every numeric cell must come from a given input.
        code, report = run(META + ["--system1", sample("system1_ok.txt")])
        numeric = [c for c in row_cells(report)[3:] if re.search(r"\d", c)]
        self.assertEqual(numeric, ["489.0 ms"])

    def test_out_and_summary_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "nightly-summary.md")
            summary = os.path.join(d, "summary.md")
            with open(summary, "w", encoding="utf-8") as f:
                f.write("earlier step\n")
            code, report = run(META + ALL_INPUTS + ["--out", out, "--summary-file", summary])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), report)
            with open(summary, encoding="utf-8") as f:
                self.assertEqual(f.read(), "earlier step\n" + report)


if __name__ == "__main__":
    unittest.main()
