"""Interference models: active acoustic countermeasures against the keyboard side channel.

Two tiers (Phase 5 of the plan):
  * ``masker`` — spectrally-matched decoy masking that runs today, pure Python.
  * ``adversarial`` — a perturbation generator trained against a recognizer (PyTorch).

``evaluate_defense`` is the red/blue harness that measures how far the defense drives
an attacker's keystroke-recovery accuracy toward chance.
"""

from .masker import DecoyBank, apply_masking, make_decoy
from .evaluate_defense import keystroke_recovery, DefenseReport, run_defense_benchmark

__all__ = [
    "DecoyBank",
    "apply_masking",
    "make_decoy",
    "keystroke_recovery",
    "DefenseReport",
    "run_defense_benchmark",
]
