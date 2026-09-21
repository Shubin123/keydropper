"""End-to-end: drive the CLI as a real subprocess (fast commands only)."""

import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "keydropper.cli", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


class TestCLI(unittest.TestCase):
    def test_info(self):
        r = run_cli("info")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("feature vector", r.stdout)

    def test_segment_reports_f1(self):
        r = run_cli("segment", "hello world")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("onset detection", r.stdout)
        self.assertIn("F1=", r.stdout)

    def test_synth_writes_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "synth")
            r = run_cli("synth", "--out", out, "--per-key", "2")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "dataset.json")))

    def test_help_exits_cleanly(self):
        r = run_cli("--help")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
