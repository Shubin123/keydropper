import unittest

import _pathfix  # noqa: F401

from keydropper.config import AudioConfig, FeatureConfig
from keydropper import synth, features as F


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.audio = AudioConfig()
        self.feat = FeatureConfig()

    def test_logmel_shape_and_determinism(self):
        clip = synth.key_signature("a", self.audio, window_ms=120.0)
        m1 = F.logmel(clip, self.audio, self.feat)
        m2 = F.logmel(clip, self.audio, self.feat)
        self.assertEqual(m1, m2)  # deterministic
        self.assertTrue(all(len(row) == self.feat.n_mels for row in m1))
        self.assertGreater(len(m1), 1)

    def test_fixed_length_pads_and_truncates(self):
        clip = synth.key_signature("b", self.audio, window_ms=120.0)
        mat = F.logmel(clip, self.audio, self.feat)
        padded = F.fixed_length(mat, len(mat) + 5, self.feat.n_mels)
        self.assertEqual(len(padded), len(mat) + 5)
        trunc = F.fixed_length(mat, 2, self.feat.n_mels)
        self.assertEqual(len(trunc), 2)

    def test_numpy_path_matches_pure_python(self):
        try:
            import numpy as np  # noqa: F401
        except Exception:
            self.skipTest("numpy not installed")
        clip = synth.key_signature("c", self.audio, window_ms=120.0)
        pure = F.logmel(clip, self.audio, self.feat)
        fast = F.logmel_np(clip, self.audio, self.feat)
        self.assertEqual(len(pure), fast.shape[0])
        for i in range(len(pure)):
            for j in range(self.feat.n_mels):
                self.assertAlmostEqual(pure[i][j], float(fast[i][j]), places=4)


if __name__ == "__main__":
    unittest.main()
