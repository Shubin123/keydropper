"""Integration: the interference model must suppress the attacker (blue-team goal).

The whole point of the project: active masking should drive a working eavesdropper's
keystroke-recovery accuracy down toward chance. These tests assert the direction and a
meaningful magnitude, deterministically (fixed seeds), so a defense regression fails
CI rather than silently weakening protection.
"""

import unittest

import _pathfix  # noqa: F401
from helpers import fast_config, FAST_KEYS

from keydropper.interference.evaluate_defense import run_defense_benchmark
from keydropper.interference.masker import DecoyBank, apply_masking


class TestDefenseSuppressesAttacker(unittest.TestCase):
    def setUp(self):
        self.cfg = fast_config()
        self.keys = FAST_KEYS

    def test_clean_attacker_is_strong_then_defense_cuts_it(self):
        report = run_defense_benchmark(self.keys, "x", self.cfg, per_key_train=25, eval_len=120)
        # 1. Without a defense the attacker recovers most keystrokes (leakage exists).
        self.assertGreater(report.recovery_clean, 0.5,
                           f"clean attacker too weak to be a fair test: {report.summary()}")
        # 2. The masking defense meaningfully reduces recovery.
        self.assertLess(report.recovery_defended, report.recovery_clean - 0.15,
                        f"defense did not meaningfully reduce leakage: {report.summary()}")
        # 3. Reported chance level is sane.
        self.assertAlmostEqual(report.chance, 1.0 / len(set(self.keys)), places=6)

    def test_masking_changes_the_signal(self):
        from keydropper import synth
        stream, truth = synth.render_stream("abcde fghij", self.cfg.audio, seed=3,
                                            window_ms=self.cfg.segment.window_ms)
        bank = DecoyBank.synthetic(self.keys, self.cfg.audio, self.cfg.segment.window_ms)
        defended = apply_masking(stream, truth, self.cfg.audio, self.cfg.model, bank, seed=5)
        self.assertEqual(len(defended), len(stream))
        self.assertNotEqual(defended, list(stream))  # decoys were actually injected

    def test_report_leakage_reduction_is_positive(self):
        report = run_defense_benchmark(self.keys, "x", self.cfg, per_key_train=25, eval_len=120)
        self.assertGreaterEqual(report.leakage_reduction, 0.0)


if __name__ == "__main__":
    unittest.main()
