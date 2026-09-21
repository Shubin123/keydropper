# keydropper dev-loop shortcuts.
# The core runs on the stdlib; `make venv` adds numpy+pytest for a faster loop.

PY ?= python3
VENV = .venv
VPY = $(VENV)/bin/python

.PHONY: help venv test test-fast test-stdlib smoke demo eval-defense clean

help:
	@echo "make venv         - create .venv with numpy + pytest (fast dev loop)"
	@echo "make test         - run the full suite (uses .venv if present)"
	@echo "make test-fast    - run only smoke + unit tests"
	@echo "make test-stdlib  - run tests with system python, no third-party deps"
	@echo "make demo         - end-to-end synthetic pipeline + defense report"
	@echo "make eval-defense - attacker keystroke-recovery with vs without masking"

venv:
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install -U pip
	$(VPY) -m pip install -r requirements-dev.txt

RUN = $(if $(wildcard $(VPY)),$(VPY),$(PY))

test:
	$(RUN) -m pytest tests/

test-fast:
	$(RUN) -m pytest tests/test_smoke.py tests/test_dsp.py tests/test_synth.py \
		tests/test_models.py tests/test_features.py tests/test_segment.py

test-stdlib:
	cd tests && $(PY) -m unittest discover -p 'test_*.py'

smoke:
	$(RUN) -m pytest tests/test_smoke.py

demo:
	PYTHONPATH=src $(RUN) -m keydropper.cli demo

eval-defense:
	PYTHONPATH=src $(RUN) -m keydropper.cli eval-defense

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist
