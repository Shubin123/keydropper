"""Integration: the attacker (recognizer) must generalize ACROSS recordings.

These are the regression tests for the two bugs that made the pipeline silently
useless during development:
  1. keyboard identity leaking from the stream seed (every recording a different
     keyboard) -> zero cross-stream transfer;
  2. absolute-position features + noise-frame domination collapsing under
     standardization.
If cross-stream recognition ever drops back to chance, one of those has regressed.
"""

import random
import unittest

import _pathfix  # noqa: F401
from helpers import fast_config, FAST_KEYS

from keysoundmap import synth, models as M
from keysoundmap.data import label_detected_clips, build_training_set


def _stream_features(cfg, keys, seed, n=120):
    rng = random.Random(seed)
    chars = [(" " if k == "<space>" else k) for k in keys]
    text = "".join(rng.choice(chars) for _ in range(n))
    stream, truth = synth.render_stream(text, cfg.audio, seed=seed, window_ms=cfg.segment.window_ms)
    clips, labels = label_detected_clips(stream, truth, cfg)
    nf = M.infer_n_frames(cfg.segment.window_ms, cfg.feature, cfg.audio)
    return M.featurize_clips(clips, cfg.audio, cfg.feature, nf), labels


class TestCrossStreamRecognition(unittest.TestCase):
    def setUp(self):
        self.cfg = fast_config()
        self.keys = FAST_KEYS
        self.chance = 1.0 / len(self.keys)

    def test_recognizer_generalizes_to_a_new_recording(self):
        Xtr, ytr = _stream_features(self.cfg, self.keys, seed=1)
        Xte, yte = _stream_features(self.cfg, self.keys, seed=2)  # different recording
        clf = M.KNNClassifier(k=3).fit(Xtr, ytr)
        acc = M.accuracy(clf.predict(Xte), yte)
        # A working attacker must be WELL above chance on an unseen recording.
        self.assertGreater(acc, 0.5, f"cross-stream acc {acc:.3f} ~ chance {self.chance:.3f}: "
                                     "keyboard identity or feature alignment has regressed")

    def test_training_set_is_segmentation_matched(self):
        # Clips built for training come through the same onset detector as inference.
        clips, labels = build_training_set(self.keys, 12, self.cfg)
        self.assertGreater(len(clips), 0)
        self.assertEqual(len(clips), len(labels))
        self.assertTrue(set(labels).issubset(set(self.keys)))

    def test_end_to_end_train_then_recognize_stream(self):
        clips, labels = build_training_set(self.keys, 15, self.cfg)
        nf = M.infer_n_frames(self.cfg.segment.window_ms, self.cfg.feature, self.cfg.audio)
        X = M.featurize_clips(clips, self.cfg.audio, self.cfg.feature, nf)
        clf = M.KNNClassifier(k=3).fit(X, labels)
        Xte, yte = _stream_features(self.cfg, self.keys, seed=99)
        acc = M.accuracy(clf.predict(Xte), yte)
        self.assertGreater(acc, 0.5, f"end-to-end recognition too low: {acc:.3f}")


if __name__ == "__main__":
    unittest.main()
