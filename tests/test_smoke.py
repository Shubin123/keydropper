"""Smoke: everything imports and a tiny end-to-end run completes fast.

The first thing to run in the dev loop — if this fails, something is badly broken.
"""

import importlib
import unittest

import _pathfix  # noqa: F401
from helpers import fast_config, FAST_KEYS


class TestImports(unittest.TestCase):
    def test_all_modules_import(self):
        for mod in [
            "keydropper",
            "keydropper.config",
            "keydropper.dsp",
            "keydropper.synth",
            "keydropper.segment",
            "keydropper.features",
            "keydropper.models",
            "keydropper.data",
            "keydropper.capture",
            "keydropper.cli",
            "keydropper.interference",
            "keydropper.interference.masker",
            "keydropper.interference.adversarial",
            "keydropper.interference.evaluate_defense",
        ]:
            importlib.import_module(mod)


class TestTinyEndToEnd(unittest.TestCase):
    def test_pipeline_runs(self):
        from keydropper import synth, segment as seg_mod, models as M
        cfg = fast_config()
        stream, truth = synth.render_stream("abc", cfg.audio, seed=1,
                                            window_ms=cfg.segment.window_ms)
        onsets, clips = seg_mod.segment(stream, cfg.audio, cfg.segment)
        self.assertGreater(len(onsets), 0)
        nf = M.infer_n_frames(cfg.segment.window_ms, cfg.feature, cfg.audio)
        feats = M.featurize_clips(clips, cfg.audio, cfg.feature, nf)
        self.assertEqual(len(feats), len(clips))
        self.assertTrue(all(len(f) == nf * cfg.feature.n_mels for f in feats))


if __name__ == "__main__":
    unittest.main()
