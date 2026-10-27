"""The output parsers read the same values with and without the REFLEX_RUN_REPORT line.

Each test runs a parser's own code (shell functions cut out of the script and run
with bash, or the Python function imported) on a sample `system1`/`generate`
output in bench/testdata/run_report_*, once as printed and once with the report
line removed, and checks both give the same answer.

Run from the repo root: python -m unittest discover -s bench -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TESTDATA = os.path.join(HERE, "testdata")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import agent_relay_bench  # noqa: E402
import relay_orchestrator  # noqa: E402

BASH = shutil.which("bash")


def read(name):
    with open(os.path.join(TESTDATA, name), encoding="utf-8", newline="") as f:
        return f.read()


def is_report(line):
    if line.startswith("REFLEX_RUN_REPORT "):
        return True
    try:
        obj = json.loads(line)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("report") == "run"


def without_report(text):
    return "".join(line for line in text.splitlines(True) if not is_report(line))


def shell_functions(script, *names):
    """The definitions of `names` in `script`, verbatim."""
    with open(os.path.join(ROOT, script), encoding="utf-8") as f:
        src = f.read()
    out = []
    for name in names:
        m = re.search(r"^%s\(\)\s*\{[^\n]*\n.*?^\}\n" % re.escape(name), src, re.M | re.S)
        if m is None:
            raise AssertionError("no %s() in %s" % (name, script))
        out.append(m.group(0))
    return "".join(out)


class OutputPair:
    """A sample output written twice to temp files: as printed, and without the report."""

    def __init__(self, name):
        self.text = read(name)
        self.dir = tempfile.mkdtemp()
        self.new = os.path.join(self.dir, "new.log")
        self.old = os.path.join(self.dir, "old.log")
        for path, text in ((self.new, self.text), (self.old, without_report(self.text))):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def bash(script, *args, env=None):
    proc = subprocess.run(
        [BASH, "-c", script, "bash", *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


class FixturesTest(unittest.TestCase):
    def test_fixtures_carry_one_report(self):
        for name in ("run_report_system1.txt", "run_report_generate.txt",
                     "run_report_system1.jsonl", "run_report_generate.jsonl"):
            lines = read(name).splitlines()
            self.assertEqual(sum(map(is_report, lines)), 1, name)
            self.assertTrue(is_report(lines[-1]), name + ": report is not the last line")


@unittest.skipIf(BASH is None, "bash not found")
class PhaseScriptsTest(unittest.TestCase):
    """scripts/bench_cold_start_phases.sh and bench_cold_start_phases_system1.sh."""

    CASES = (
        ("scripts/bench_cold_start_phases_system1.sh", "run_report_system1.txt",
         "process_start_to_result_ms"),
        ("scripts/bench_cold_start_phases.sh", "run_report_generate.txt",
         "process_start_to_first_token_ms"),
    )

    def test_fields_unchanged(self):
        for script, sample, total in self.CASES:
            funcs = shell_functions(script, "extract_field", "extract_phase_field",
                                    "extract_total_joules")
            calls = [funcs]
            for field in (total, "gguf_open_ms", "cuda_init_ms", "model_load_ms", "prompt_eval_ms"):
                calls.append('echo "%s=$(extract_field "$1" %s)"' % (field, field))
            for phase in ("gguf_open", "cuda_init", "model_load", "prompt_eval"):
                calls.append('echo "%s=$(extract_phase_field "$1" %s energy_joules)"' % (phase, phase))
            calls.append('echo "total_joules=$(extract_total_joules "$1")"')
            pair = OutputPair(sample)
            try:
                new = bash("\n".join(calls), pair.new)
                old = bash("\n".join(calls), pair.old)
            finally:
                pair.cleanup()
            with self.subTest(script=script):
                self.assertEqual(new, old)
                self.assertNotIn("=\n", new, "a field came back empty:\n" + new)


@unittest.skipIf(BASH is None, "bash not found")
class EnergyScriptTest(unittest.TestCase):
    """scripts/bench_cold_energy.sh's `field`, on the REFLEX_SYSTEM1_OK joules."""

    def test_joules_unchanged(self):
        funcs = shell_functions("scripts/bench_cold_energy.sh", "field")
        pair = OutputPair("run_report_system1.txt")
        try:
            results = {}
            for which, path in (("new", pair.new), ("old", pair.old)):
                os.replace(path, os.path.join(pair.dir, "run_1.log"))
                script = funcs + 'results_dir="$1"; n_runs=1; field joules REFLEX_SYSTEM1_OK'
                results[which] = bash(script, pair.dir)
        finally:
            pair.cleanup()
        self.assertEqual(results["new"], results["old"])
        self.assertNotEqual(results["new"].strip(), "")


@unittest.skipIf(BASH is None, "bash not found")
class GuardrailTest(unittest.TestCase):
    """examples/system1_guardrail.sh reads the anchored REFLEX_SYSTEM1_OK line."""

    def test_best_text_and_entropy_unchanged(self):
        funcs = shell_functions("examples/system1_guardrail.sh", "extract", "extract_quoted")
        script = funcs + (
            "line=\"$(grep '^REFLEX_SYSTEM1_OK' \"$1\")\"\n"
            'echo "$(extract_quoted best_text "$line")|$(extract entropy "$line")|$(extract best_idx "$line")"'
        )
        pair = OutputPair("run_report_system1.txt")
        try:
            self.assertEqual(bash(script, pair.new), bash(script, pair.old))
        finally:
            pair.cleanup()


@unittest.skipIf(BASH is None, "bash not found")
class RelayChainTest(unittest.TestCase):
    """examples/relay_chain.sh picks the last --json line with best_idx/token_text."""

    def run_chain(self, sample, key, fields):
        funcs = shell_functions("examples/relay_chain.sh", "die", "result_line", "json_str", "json_num")
        lines = [funcs, 'res=$(result_line "$1" %s)' % key]
        for kind, field in fields:
            lines.append('echo "%s=$(%s "$res" %s)"' % (field, kind, field))
        pair = OutputPair(sample)
        try:
            return bash("\n".join(lines), pair.new), bash("\n".join(lines), pair.old)
        finally:
            pair.cleanup()

    def test_router_hop(self):
        new, old = self.run_chain("run_report_system1.jsonl", "best_idx", (
            ("json_num", "best_idx"), ("json_str", "best_text"), ("json_num", "entropy")))
        self.assertEqual(new, old)

    def test_worker_hop(self):
        new, old = self.run_chain("run_report_generate.jsonl", "token_text", (
            ("json_str", "token_text"), ("json_num", "process_start_to_first_token_ms"),
            ("json_num", "kv_io_ms")))
        self.assertEqual(new, old)


class RelayOrchestratorTest(unittest.TestCase):
    """examples/relay_orchestrator.py's cold sub-agent result selection."""

    def test_result_unchanged(self):
        args = types.SimpleNamespace(reflex="reflex", model="m.gguf", max_tokens=2, weights=None)
        results = []
        for text in (read("run_report_generate.jsonl"), without_report(read("run_report_generate.jsonl"))):
            proc = types.SimpleNamespace(returncode=0, stdout=text.encode(), stderr=b"")
            with mock.patch.object(relay_orchestrator.subprocess, "run", return_value=proc):
                result, _wall = relay_orchestrator.run_cold_subagent(args, "x.cskv")
            results.append(result)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["token_text"], " a b")


class AgentRelayBenchTest(unittest.TestCase):
    """scripts/agent_relay_bench.py's cold-generate parser."""

    def test_generate_fields_unchanged(self):
        text = read("run_report_generate.txt")
        self.assertEqual(agent_relay_bench.parse_reflex_generate(text),
                         agent_relay_bench.parse_reflex_generate(without_report(text)))


if __name__ == "__main__":
    unittest.main()
