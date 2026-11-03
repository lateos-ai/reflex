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


RELAY_BASELINE = os.path.join(HERE, "relay-baseline-t4.json")


def parse_relay(name):
    with open(sample(name), encoding="utf-8") as f:
        return bench_compare.parse_relay_output(f.read())


def measured_relay_baseline(provisional):
    """A baseline with numbers, built from relay_ok.txt (the committed one may be TBD)."""
    return bench_compare.build_relay_baseline(parse_relay("relay_ok.txt"), "test", provisional)


def statuses(rows):
    return {(kind, metric): status for kind, metric, _, _, _, status in rows}


class RelayParseTests(unittest.TestCase):
    def test_parses_every_hop_kind_and_the_config(self):
        parsed = parse_relay("relay_ok.txt")
        self.assertEqual(set(parsed["kinds"]), set(bench_compare.RELAY_KINDS))
        self.assertEqual(parsed["kinds"]["pinned"]["metrics"]["rtt_ms"], {"p50": 230.0, "p95": 241.5})
        self.assertEqual(parsed["kinds"]["candidates"]["metrics"]["s1_rtt_ms"]["p50"], 10.0)
        self.assertEqual(parsed["kinds"]["constrained"]["metrics"]["mask_ms"]["p50"], 9.5)
        self.assertEqual(parsed["kinds"]["constrained"]["valid"], {"count": 20, "of": 20})
        self.assertEqual(parsed["kinds"]["pinned"]["n_hops"], 20)
        self.assertEqual(parsed["config"]["model"], "Qwen3-0.6B-Q4_K_M.gguf")
        self.assertEqual(parsed["config"]["hops"], 20)

    def test_candidates_table_keeps_only_its_own_metrics(self):
        # The System-1 run's table also has the generate hop's rows; they belong to `pinned`.
        metrics = parse_relay("relay_ok.txt")["kinds"]["candidates"]["metrics"]
        self.assertEqual(set(metrics), {"s1_rtt_ms", "s1_model_ms", "s1_kv_io_ms"})

    def test_missing_hop_kind_is_an_error_not_a_pass(self):
        with self.assertRaisesRegex(bench_compare.BenchParseError, "constrained"):
            parse_relay("relay_missing_kind.txt")

    def test_system1_output_is_not_relay_output(self):
        with self.assertRaisesRegex(bench_compare.BenchParseError, "missing hop kinds"):
            parse_relay("system1_ok.txt")

    def test_constrained_table_without_valid_count_is_an_error(self):
        with open(sample("relay_ok.txt"), encoding="utf-8") as f:
            text = f.read().replace("valid JSON outputs: 20/20", "")
        with self.assertRaisesRegex(bench_compare.BenchParseError, "valid-JSON"):
            bench_compare.parse_relay_output(text)


class RelayCompareTests(unittest.TestCase):
    def test_same_run_passes(self):
        rows = bench_compare.compare_relay(parse_relay("relay_ok.txt"), measured_relay_baseline(False), 15.0, 3.0)
        self.assertEqual(set(statuses(rows).values()), {"ok"})

    def test_regression_needs_both_percent_and_ms(self):
        rows = bench_compare.compare_relay(
            parse_relay("relay_regressed.txt"), measured_relay_baseline(False), 15.0, 3.0
        )
        st = statuses(rows)
        self.assertEqual(st[("pinned", "rtt_ms")], "regressed")
        # +20% on a 10 ms hop is only +2 ms: under the absolute floor.
        self.assertEqual(st[("candidates", "s1_rtt_ms")], "ok")
        self.assertEqual(st[("constrained", "rtt_ms")], "ok")
        rows = bench_compare.compare_relay(
            parse_relay("relay_regressed.txt"), measured_relay_baseline(False), 15.0, 1.0
        )
        self.assertEqual(statuses(rows)[("candidates", "s1_rtt_ms")], "regressed")

    def test_threshold_is_configurable(self):
        rows = bench_compare.compare_relay(
            parse_relay("relay_regressed.txt"), measured_relay_baseline(False), 30.0, 3.0
        )
        self.assertEqual(set(statuses(rows).values()), {"ok"})

    def test_fewer_valid_json_outputs_is_a_regression(self):
        rows = bench_compare.compare_relay(
            parse_relay("relay_invalid_json.txt"), measured_relay_baseline(False), 15.0, 3.0
        )
        st = statuses(rows)
        self.assertEqual(st[("constrained", "valid")], "regressed")
        self.assertEqual(st[("constrained", "rtt_ms")], "ok")

    def test_tbd_baseline_reports_no_baseline(self):
        with open(RELAY_BASELINE, encoding="utf-8") as f:
            tbd = json.load(f)
        for entry in tbd["hop_kinds"].values():
            for m in entry["metrics"].values():
                m["p50"] = "TBD"
            if "valid" in entry:
                entry["valid"]["count"] = "TBD"
        current = parse_relay("relay_regressed.txt")
        rows = bench_compare.compare_relay(current, tbd, 15.0, 3.0)
        self.assertEqual(set(statuses(rows).values()), {"no baseline"})
        report = bench_compare.render_relay_report(rows, 15.0, 3.0, tbd, current)
        self.assertIn("NO BASELINE YET", report)
        self.assertIn("| pinned | rtt_ms | TBD | 290.0 |", report)

    def test_baseline_missing_a_hop_kind_is_an_error(self):
        broken = measured_relay_baseline(False)
        del broken["hop_kinds"]["candidates"]
        with self.assertRaisesRegex(bench_compare.BenchParseError, "candidates"):
            bench_compare.compare_relay(parse_relay("relay_ok.txt"), broken, 15.0, 3.0)

    def test_report_flags_config_and_token_mismatches(self):
        baseline = measured_relay_baseline(False)
        baseline["max_tokens"] = 64
        baseline["hop_kinds"]["pinned"]["metrics"]["tokens"]["p50"] = 64.0
        current = parse_relay("relay_ok.txt")
        rows = bench_compare.compare_relay(current, baseline, 15.0, 3.0)
        report = bench_compare.render_relay_report(rows, 15.0, 3.0, baseline, current)
        self.assertIn("max_tokens, pinned tokens", report)
        self.assertIn("not like-for-like", report)

    def test_report_names_the_verdict(self):
        baseline = measured_relay_baseline(True)
        current = parse_relay("relay_regressed.txt")
        rows = bench_compare.compare_relay(current, baseline, 15.0, 3.0)
        report = bench_compare.render_relay_report(rows, 15.0, 3.0, baseline, current)
        self.assertIn("REGRESSION (provisional baseline, warning only)", report)
        self.assertIn("until two scheduled nightly runs exist", report)
        self.assertIn("| pinned | rtt_ms | 230.0 | 290.0 | +26.1% | regressed |", report)


class RelayCliTests(unittest.TestCase):
    def test_exit_codes(self):
        with TempBaseline(measured_relay_baseline(False)) as real:
            relay = ["--relay", "--baseline", real]
            self.assertEqual(run_quiet([sample("relay_ok.txt")] + relay), 0)
            self.assertEqual(run_quiet([sample("relay_regressed.txt")] + relay), 1)
            self.assertEqual(run_quiet([sample("relay_invalid_json.txt")] + relay), 1)
            self.assertEqual(run_quiet([sample("relay_missing_kind.txt")] + relay), 2)
            self.assertEqual(run_quiet([sample("relay_regressed.txt")] + relay + ["--threshold-pct", "30"]), 0)

    def test_provisional_baseline_warns_unless_strict(self):
        with TempBaseline(measured_relay_baseline(True)) as prov:
            relay = ["--relay", "--baseline", prov]
            self.assertEqual(run_quiet([sample("relay_regressed.txt")] + relay), 0)
            self.assertEqual(run_quiet([sample("relay_regressed.txt")] + relay + ["--strict"]), 1)

    def test_committed_relay_baseline_is_provisional_and_complete(self):
        with open(RELAY_BASELINE, encoding="utf-8") as f:
            b = json.load(f)
        self.assertTrue(b["provisional"])
        self.assertEqual(b["provisional_reason"], "until two scheduled nightly runs exist")
        self.assertEqual(set(b["hop_kinds"]), set(bench_compare.RELAY_KINDS))
        for kind, (gate, keep) in bench_compare.RELAY_KINDS.items():
            self.assertEqual(b["hop_kinds"][kind]["gate"], gate)
            self.assertIn(gate, b["hop_kinds"][kind]["metrics"])
        self.assertIn("valid", b["hop_kinds"]["constrained"])
        # Whatever it holds, the committed baseline must load and compare without error.
        self.assertIn(run_quiet([sample("relay_ok.txt"), "--relay", "--baseline", RELAY_BASELINE]), (0, 1))

    def test_write_baseline_round_trips(self):
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            argv = [sample("relay_ok.txt"), "--relay", "--write-baseline", out, "--source", "test", "--provisional"]
            self.assertEqual(run_quiet(argv), 0)
            with open(out, encoding="utf-8") as f:
                written = json.load(f)
            self.assertTrue(written["provisional"])
            self.assertEqual(written["source"], "test")
            self.assertEqual(written["hops"], 20)
            self.assertEqual(written["hop_kinds"]["constrained"]["valid"], {"count": 20, "of": 20})
            self.assertEqual(run_quiet([sample("relay_ok.txt"), "--relay", "--baseline", out, "--strict"]), 0)
        finally:
            os.remove(out)

    def test_provisional_flag_needs_relay_write_baseline(self):
        with self.assertRaises(SystemExit):
            run_quiet([sample("system1_ok.txt"), "--write-baseline", "x.json", "--provisional"])


if __name__ == "__main__":
    unittest.main()
