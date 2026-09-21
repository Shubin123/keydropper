"""Integration: the data-assembly layer wires segmentation to labels correctly."""

import unittest

import _pathfix  # noqa: F401
from helpers import fast_config, FAST_KEYS

from keydropper import synth, segment as seg_mod
from keydropper.data import label_detected_clips


class TestDataAssembly(unittest.TestCase):
    def setUp(self):
        self.cfg = fast_config()

    def test_detected_clips_get_correct_labels(self):
        text = "abcdefghij"
        stream, truth = synth.render_stream(text, self.cfg.audio, seed=1,
                                            window_ms=self.cfg.segment.window_ms)
        clips, labels = label_detected_clips(stream, truth, self.cfg)
        # On clean synthetic audio segmentation is near-perfect, so most true keys are
        # recovered and labeled in typing order.
        self.assertGreaterEqual(len(labels), int(0.8 * len(truth)))
        true_seq = [k for _, k in truth]
        # The labeled sequence should be an in-order subsequence of the true keys.
        it = iter(true_seq)
        self.assertTrue(all(any(lbl == t for t in it) for lbl in labels),
                        f"labels not an ordered subsequence: {labels} vs {true_seq}")

    def test_clip_length_is_fixed(self):
        stream, truth = synth.render_stream("abc def", self.cfg.audio, seed=2,
                                            window_ms=self.cfg.segment.window_ms)
        clips, _ = label_detected_clips(stream, truth, self.cfg)
        expected = int(round(self.cfg.segment.window_ms * self.cfg.audio.sample_rate / 1000.0))
        self.assertTrue(all(len(c) == expected for c in clips))

    def test_onset_matching_counts(self):
        detected = [100, 200, 305]
        truth = [102, 198, 500]
        tp, fp, fn = seg_mod.match_onsets(detected, truth, tolerance_samples=10)
        self.assertEqual((tp, fp, fn), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
