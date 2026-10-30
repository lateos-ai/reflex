"""scripts/prove_it.sh and the report it prints (scripts/prove_it_report.py).

The parser and comparison run on real `reflex` output captured on a T4, Qwen3-0.6B-Q4_K_M
(bench/testdata: run_report_system1.txt from an sm_75 cubin build; prove_it_system1_ptx.txt
from a portable-PTX build and prove_it_doctor.txt/prove_it_smoke.txt from a cubin build,
all three default features, as prove_it.sh builds them). The whole script runs against a stub
`reflex` that prints those captures, to check its exit codes.

Run from the repo root: python -m unittest discover -s bench -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TESTDATA = os.path.join(HERE, "testdata")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prove_it_report as pr  # noqa: E402

BASH = shutil.which("bash")
BASELINES = os.path.join(ROOT, "bench", "baseline-*.json")


def read(name):
    with open(os.path.join(TESTDATA, name), encoding="utf-8", newline="") as f:
        return f.read()


def report_line(text):
    return [l for l in text.splitlines() if l.startswith(pr.REPORT_PREFIX)][-1]


def write_runs(directory, outputs, walls=None):
    for i, text in enumerate(outputs, start=1):
        with open(os.path.join(directory, "stdout_%d.log" % i), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        if walls is not None:
            with open(os.path.join(directory, "wall_ms_%d" % i), "w", encoding="utf-8") as f:
                f.write("%.3f\n" % walls[i - 1])


def scale_total(text, factor):
    """The capture with every phase and total multiplied by `factor` (a slower GPU)."""
    def repl(m):
        return "%s=%.3f" % (m.group(1), float(m.group(2)) * factor)
    return re.sub(r"\b(process_start_to_(?:first_)?result_ms|gguf_open_ms|cuda_init_ms|"
                  r"model_load_ms|prompt_eval_ms)=([0-9.]+)", repl, text)


class ParseTest(unittest.TestCase):
    def test_cubin_capture(self):
        f, _ = pr.report_from_output(read("run_report_system1.txt"))
        self.assertEqual(f["command"], "system1")
        self.assertEqual(f["gpu_name"], "Tesla T4")
        self.assertEqual(f["compute_capability"], "7.5")
        self.assertEqual(pr.kernel_image(f), "cubin native sm_75")
        self.assertEqual(f["weights"], "qr")
        self.assertEqual(f["model_file"], "Qwen3-0.6B-Q4_K_M.gguf")
        self.assertEqual(float(f["process_start_to_first_result_ms"]), 379.854)

    def test_ptx_capture(self):
        f, _ = pr.report_from_output(read("prove_it_system1_ptx.txt"))
        self.assertEqual(f["kernel_format"], "ptx")
        self.assertEqual(f["kernel_image"], "ptx_jit")
        self.assertNotIn("kernel_image_arch", f)
        self.assertEqual(pr.kernel_image(f), "ptx ptx_jit")

    def test_quoted_values(self):
        f = pr.parse_report_line(
            'REFLEX_RUN_REPORT schema_version=1.5.0 command=system1 '
            'process_start_to_first_result_ms=1.000 gpu_name="GPU \\"x\\" y\\\\z" '
            'model_file="a b=c.gguf" arch="\\u{e9}t\\u{e9}"')
        self.assertEqual(f["gpu_name"], 'GPU "x" y\\z')
        self.assertEqual(f["model_file"], "a b=c.gguf")
        self.assertEqual(f["arch"], "été")

    def test_rejects(self):
        for bad in (
            "REFLEX_SYSTEM1_OK process_start_to_result_ms=1",
            'REFLEX_RUN_REPORT schema_version=1.5.0 command=system1 gpu_name="unterminated',
            "REFLEX_RUN_REPORT schema_version=1.5.0 command=system1",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(pr.ReportError):
                    pr.parse_report_line(bad)
        text = "".join(l for l in read("run_report_system1.txt").splitlines(True)
                       if not l.startswith(pr.REPORT_PREFIX))
        with self.assertRaises(pr.ReportError):
            pr.report_from_output(text)

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(pr.percentile([5, 1, 4, 2, 3], 50), 3)
        self.assertEqual(pr.percentile([1, 2, 3, 4], 50), 2)
        self.assertEqual(pr.percentile([7], 50), 7)


class PickTest(unittest.TestCase):
    def setUp(self):
        self.baselines = pr.load_baselines(BASELINES)
        self.gpus = {b["gpu"] for b in self.baselines}

    def pick(self, name, cc):
        b, why = pr.pick_baseline(self.baselines, name, cc)
        return b["gpu"], why

    def test_every_baseline_has_what_the_report_compares(self):
        self.assertIn("Tesla T4", self.gpus)
        for b in self.baselines:
            with self.subTest(path=b["_path"]):
                for key in ("gpu", "compute_capability", "kernel_image", "weights", "subcommand",
                            "model", "recorded", "phases"):
                    self.assertIn(key, b)
                self.assertIn("total", b["phases"])
                self.assertIsNotNone(pr._cc(b["compute_capability"]))

    def test_same_gpu_wins(self):
        self.assertEqual(self.pick("Tesla T4", "7.5"), ("Tesla T4", "same GPU"))
        self.assertEqual(self.pick("tesla t4", None), ("Tesla T4", "same GPU"))

    def test_nearest_compute_capability(self):
        gpu, why = self.pick("Tesla V100-SXM2-16GB", "7.0")
        self.assertEqual(gpu, "Tesla T4")
        self.assertIn("nearest", why)
        if "NVIDIA A10G" in self.gpus:
            self.assertEqual(self.pick("NVIDIA RTX A6000", "8.6")[0], "NVIDIA A10G")
            self.assertEqual(self.pick("NVIDIA L4", "8.9")[0], "NVIDIA A10G")


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def main(self):
        from io import StringIO
        from unittest import mock
        out, err = StringIO(), StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            code = pr.main([self.dir, "--baselines", BASELINES])
        return code, out.getvalue(), err.getvalue()

    def test_t4_comparison(self):
        text = read("run_report_system1.txt")
        write_runs(self.dir, [text] * 5, walls=[480.0, 481.0, 482.0, 483.0, 484.0])
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertIn("Reference: Tesla T4, 2026-09-30 (same GPU), baseline-t4.json", out)
        self.assertIn("weights qr (reference f32)", out)
        self.assertIn("n=5 (reference n=10)", out)
        self.assertNotIn("kernel image", out.split("differs:")[1].split("\n")[0])
        self.assertIn("| total (process start to result) | 379.9 | 485.4 | -21.7% |", out)
        self.assertIn("| process launch (external - internal) | 102.1 | 112.1 | -8.9% |", out)
        self.assertIn(report_line(text), out)

    def test_slower_is_not_a_failure(self):
        write_runs(self.dir, [scale_total(read("run_report_system1.txt"), 2.0)] * 3)
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertIn("| total (process start to result) | 759.7 | 485.4 | +56.5% |", out)
        self.assertIn("| process launch (external - internal) | n/a | 112.1 | - |", out)

    def test_ptx_note(self):
        write_runs(self.dir, [read("prove_it_system1_ptx.txt")])
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertIn("kernel image ptx ptx_jit (reference cubin native sm_75)", out)
        self.assertIn("JIT-compiled", out)

    def test_unparsable_run_fails(self):
        good = read("run_report_system1.txt")
        write_runs(self.dir, [good, good.replace("REFLEX_RUN_REPORT", "REFLEX_RUN_REPROT")])
        code, _, err = self.main()
        self.assertEqual(code, 1)
        self.assertIn("no REFLEX_RUN_REPORT line", err)

    def test_mixed_setups_fail(self):
        good = read("run_report_system1.txt")
        write_runs(self.dir, [good, good.replace("weights=qr", "weights=f16")])
        code, _, err = self.main()
        self.assertEqual(code, 1)
        self.assertIn("weights", err)

    def test_empty_dir_fails(self):
        self.assertEqual(self.main()[0], 1)


STUB = r"""#!/usr/bin/env bash
case "$1" in
  doctor) cat "$FAKE_DIR/prove_it_doctor.txt"; exit "${FAKE_DOCTOR_EXIT:-0}" ;;
  smoke) cat "$FAKE_DIR/prove_it_smoke.txt"; exit "${FAKE_SMOKE_EXIT:-0}" ;;
  system1)
    if [ -n "${FAKE_SYSTEM1_EXIT:-}" ]; then echo "error: boom" >&2; exit "$FAKE_SYSTEM1_EXIT"; fi
    if [ -n "${FAKE_NO_REPORT:-}" ]; then grep -v '^REFLEX_RUN_REPORT' "$FAKE_DIR/run_report_system1.txt"; exit 0; fi
    cat "$FAKE_DIR/run_report_system1.txt" ;;
  *) exit 2 ;;
esac
"""


@unittest.skipIf(BASH is None, "bash not found")
class ScriptTest(unittest.TestCase):
    """prove_it.sh end to end, against a stub reflex binary."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.stub = os.path.join(self.dir, "reflex")
        with open(self.stub, "w", encoding="utf-8", newline="\n") as f:
            f.write(STUB)
        os.chmod(self.stub, 0o755)
        self.gguf = os.path.join(self.dir, "model.gguf")
        open(self.gguf, "wb").close()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_script(self, *args, **env):
        e = dict(os.environ, REFLEX_BIN=self.stub, FAKE_DIR=TESTDATA, PYTHON=sys.executable,
                 PROVE_IT_RUNS="3", PROVE_IT_OUT=os.path.join(self.dir, "out"))
        e.update(env)
        return subprocess.run([BASH, os.path.join(ROOT, "scripts", "prove_it.sh"), *args],
                              capture_output=True, text=True, env=e)

    def test_ok(self):
        proc = self.run_script(self.gguf)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("REFLEX_DOCTOR_OK", proc.stdout)
        self.assertIn("REFLEX_SMOKE_OK", proc.stdout)
        self.assertIn("run 3: process_start_to_result_ms=379.854", proc.stdout)
        self.assertIn("Reference: Tesla T4", proc.stdout)
        self.assertIn("| total (process start to result) | 379.9 | 485.4 | -21.7% |", proc.stdout)

    def test_missing_model_prints_download(self):
        proc = self.run_script(os.path.join(self.dir, "nope.gguf"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("model not found", proc.stderr)
        self.assertIn("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/"
                      "Qwen3-0.6B-Q4_K_M.gguf", proc.stderr)
        self.assertNotIn("reflex doctor", proc.stdout)

    def test_real_failures_exit_1(self):
        cases = (
            ({"FAKE_DOCTOR_EXIT": "1"}, "reflex doctor failed"),
            ({"FAKE_SMOKE_EXIT": "1"}, "reflex smoke failed"),
            ({"FAKE_SYSTEM1_EXIT": "1"}, "reflex system1 failed"),
            ({"FAKE_NO_REPORT": "1"}, "could not read the runs"),
        )
        for env, message in cases:
            with self.subTest(env=env):
                shutil.rmtree(os.path.join(self.dir, "out"), ignore_errors=True)
                proc = self.run_script(self.gguf, **env)
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertIn(message, proc.stderr)

    def test_bad_run_count(self):
        self.assertEqual(self.run_script(self.gguf, PROVE_IT_RUNS="0").returncode, 2)


if __name__ == "__main__":
    unittest.main()
