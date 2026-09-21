#!/usr/bin/env python3
"""End-to-end synthetic demo: prints segmentation quality, recognizer accuracy, and
the interference model's leakage reduction.

Run from the repo root:
    PYTHONPATH=src python3 scripts/demo.py
or simply:
    make demo
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from keydropper.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
