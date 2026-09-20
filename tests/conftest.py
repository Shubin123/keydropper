"""pytest bootstrap: make ``src/`` and this test dir importable."""
import os
import sys

_ROOT = os.path.dirname(__file__)
_SRC = os.path.abspath(os.path.join(_ROOT, "..", "src"))
for p in (_SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
