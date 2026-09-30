"""Unit tests for scripts/bench_compare.py, against sample bench logs in bench/testdata/.

Run from the repo root: python -m unittest discover -s bench -v
"""

import copy
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import bench_compare  # noqa: E402

TESTDATA = os.path.join(HERE, "testdata")
BASELINE = os.path.join(HERE, "baseline-t4.json")


def sample(name):
    return os.path.join(TESTDATA, name)


def load_baseline(provisional):
    with open(BASELINE, encoding="utf-8") as f:
        b = json.load(f)
    b["provisional"] = provisional
    return b


class TempBaseline:
    """Writes a baseline JSON to a temp file for CLI-level tests."""

    def __init__(self, baseline):
        self.baseline = baseline

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.baseline, f)
        return self.path

    def __exit__(self, *exc):
        os.remove(self.path)


def run_quiet(argv):
    """Runs main() with stdout/stderr silenced; returns the exit code."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            return bench_compare.main(argv)
        finally:
            sys.stdout, sys.stderr = old


class ParseTests(unittest.TestCase):
    def test_parses_every_phase_and_the_header(self):
        with open(sample("system1_ok.txt"), encoding="utf-8") as f:
            parsed = bench_compare.parse_bench_output(f.read())
        self.assertEqual(set(parsed["phases"]), set(bench_compare.REQUIRED))
        self.assertEqual(parsed["phases"]["total"], {"p50_ms": 489.0, "p95_ms": 497.3})
        self.assertEqual(parsed["phases"]["process_launch"]["p50_ms"], 110.4)
        self.assertEqual(
            parsed["meta"],
            {"subcommand": "system1", "n_runs": 10, "model": "Qwen3-0.6B-Q4_K_M.gguf"},
        )

    def test_per_run_table_rows_are_ignored(self):
        # bench_cold_common.sh's "| 1 | 0:00.60 | ..." rows must not be read as phases.
        with open(sample("system1_ok.txt"), encoding="utf-8") as f:
            parsed = bench_compare.parse_bench_output(f.read())
        self.assertEqual(len(parsed["phases"]), len(bench_compare.REQUIRED))

    def test_missing_measurements_are_an_error_not_a_pass(self):
        with open(sample("system1_missing_phase.txt"), encoding="utf-8") as f:
            text = f.read()
        with self.assertRaisesRegex(bench_compare.BenchParseError, "prompt_eval"):
            bench_compare.parse_bench_output(text)

    def test_empty_output_is_an_error(self):
        with self.assertRaisesRegex(bench_compare.BenchParseError, "missing phases"):
            bench_compare.parse_bench_output("bench crashed before printing anything\n")


class CompareTests(unittest.TestCase):
    def setUp(self):
        with open(sample("system1_ok.txt"), encoding="utf-8") as f:
            self.ok = bench_compare.parse_bench_output(f.read())
        with open(sample("system1_regressed.txt"), encoding="utf-8") as f:
            self.regressed = bench_compare.parse_bench_output(f.read())

    def test_within_noise_passes(self):
        _, delta, regressed = bench_compare.compare(self.ok, load_baseline(False), 15.0)
        self.assertAlmostEqual(delta, (489.0 - 485.4) / 485.4 * 100, places=6)
        self.assertFalse(regressed)

    def test_twenty_percent_total_regression_fails_at_default_threshold(self):
        rows, delta, regressed = bench_compare.compare(self.regressed, load_baseline(False), 15.0)
        self.assertGreater(delta, 15.0)
        self.assertTrue(regressed)
        model_load = next(r for r in rows if r[0] == "model_load")
        self.assertGreater(model_load[3], 40.0)

    def test_threshold_is_configurable(self):
        _, _, regressed = bench_compare.compare(self.regressed, load_baseline(False), 25.0)
        self.assertFalse(regressed)

    def test_baseline_missing_a_phase_is_an_error(self):
        broken = load_baseline(False)
        del broken["phases"]["cuda_init"]
        with self.assertRaisesRegex(bench_compare.BenchParseError, "cuda_init"):
            bench_compare.compare(self.ok, broken, 15.0)

    def test_report_names_the_verdict_and_every_phase(self):
        baseline = load_baseline(True)
        rows, delta, regressed = bench_compare.compare(self.regressed, baseline, 15.0)
        report = bench_compare.render_report(rows, delta, regressed, 15.0, baseline, self.regressed)
        self.assertIn("REGRESSION (provisional baseline, warning only)", report)
        for key in bench_compare.REQUIRED:
            self.assertIn(f"| {key} |", report)

    def test_report_flags_a_model_mismatch(self):
        baseline = load_baseline(False)
        baseline["model"] = "Qwen3.5-0.8B-Q4_K_M.gguf"
        rows, delta, regressed = bench_compare.compare(self.ok, baseline, 15.0)
        report = bench_compare.render_report(rows, delta, regressed, 15.0, baseline, self.ok)
        self.assertIn("not like-for-like", report)


class CliTests(unittest.TestCase):
    def test_exit_codes(self):
        with TempBaseline(load_baseline(False)) as real:
            self.assertEqual(run_quiet([sample("system1_ok.txt"), "--baseline", real]), 0)
            self.assertEqual(run_quiet([sample("system1_regressed.txt"), "--baseline", real]), 1)
            self.assertEqual(run_quiet([sample("system1_missing_phase.txt"), "--baseline", real]), 2)
            self.assertEqual(
                run_quiet([sample("system1_regressed.txt"), "--baseline", real, "--threshold-pct", "25"]), 0
            )

    def test_provisional_baseline_warns_unless_strict(self):
        with TempBaseline(load_baseline(True)) as prov:
            self.assertEqual(run_quiet([sample("system1_regressed.txt"), "--baseline", prov]), 0)
            self.assertEqual(run_quiet([sample("system1_regressed.txt"), "--baseline", prov, "--strict"]), 1)

    def test_committed_baseline_is_provisional_and_complete(self):
        with open(BASELINE, encoding="utf-8") as f:
            b = json.load(f)
        self.assertTrue(b["provisional"])
        self.assertEqual(set(b["phases"]), set(bench_compare.REQUIRED))

    def test_write_baseline_round_trips(self):
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            code = run_quiet([sample("system1_ok.txt"), "--write-baseline", out, "--source", "test"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as f:
                written = json.load(f)
            self.assertFalse(written["provisional"])
            self.assertEqual(written["source"], "test")
            self.assertEqual(written["phases"]["total"]["p50_ms"], 489.0)
            # A baseline written from a run compares clean against that same run.
            self.assertEqual(run_quiet([sample("system1_ok.txt"), "--baseline", out]), 0)
        finally:
            os.remove(out)

    def test_summary_file_is_appended(self):
        fd, summary = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            with TempBaseline(copy.deepcopy(load_baseline(False))) as real:
                run_quiet([sample("system1_ok.txt"), "--baseline", real, "--summary-file", summary])
                run_quiet([sample("system1_ok.txt"), "--baseline", real, "--summary-file", summary])
            with open(summary, encoding="utf-8") as f:
                self.assertEqual(f.read().count("### Cold-start bench vs baseline: OK"), 2)
        finally:
            os.remove(summary)


if __name__ == "__main__":
    unittest.main()
