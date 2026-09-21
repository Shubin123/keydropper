import unittest

import _pathfix  # noqa: F401

from keydropper.config import AudioConfig, SegmentConfig
from keydropper import synth, segment as seg_mod


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.audio = AudioConfig()
        self.seg = SegmentConfig()

    def test_detects_synthetic_onsets(self):
        text = "hello world typing test"
        stream, truth = synth.render_stream(text, self.audio, seed=7, window_ms=self.seg.window_ms)
        onsets = seg_mod.detect_onsets(stream, self.audio, self.seg)
        tol = int(self.seg.refractory_ms * 0.5 * self.audio.sample_rate / 1000.0)
        tp, fp, fn = seg_mod.match_onsets(onsets, [o for o, _ in truth], tol)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # Clean synthetic audio: the detector should be strong.
        self.assertGreaterEqual(recall, 0.85, f"recall too low: {recall}")
        self.assertGreaterEqual(f1, 0.85, f"F1 too low: {f1}")

    def test_silence_yields_no_onsets(self):
        silence = [0.0] * self.audio.sample_rate
        onsets = seg_mod.detect_onsets(silence, self.audio, self.seg)
        self.assertEqual(onsets, [])

    def test_windows_have_fixed_length(self):
        text = "abc def"
        stream, _ = synth.render_stream(text, self.audio, seed=3, window_ms=self.seg.window_ms)
        onsets, clips = seg_mod.segment(stream, self.audio, self.seg)
        expected = int(round(self.seg.window_ms * self.audio.sample_rate / 1000.0))
        self.assertTrue(all(len(c) == expected for c in clips))
        self.assertEqual(len(onsets), len(clips))


if __name__ == "__main__":
    unittest.main()
