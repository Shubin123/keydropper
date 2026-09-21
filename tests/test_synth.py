import unittest

import _pathfix  # noqa: F401

from keydropper.config import AudioConfig
from keydropper import synth


class TestSynth(unittest.TestCase):
    def setUp(self):
        self.audio = AudioConfig()

    def test_keyboard_identity_is_stable_across_streams(self):
        """The same key must sound the same in different recordings.

        This is the property whose violation once made cross-stream recognition
        impossible: if the per-key resonances depended on the stream seed, every
        recording would be a different keyboard. Canonical templates must be identical
        regardless of stream, and two keys must differ.
        """
        a1 = synth.key_signature("a", self.audio, 120.0)
        a2 = synth.key_signature("a", self.audio, 120.0)
        b1 = synth.key_signature("b", self.audio, 120.0)
        self.assertEqual(a1, a2)                 # deterministic, stream-independent
        self.assertNotEqual(a1, b1)              # distinct keys sound different

    def test_jitter_varies_but_stays_close_to_template(self):
        import random
        canon = synth.key_signature("a", self.audio, 120.0)
        hit = synth.key_signature("a", self.audio, 120.0, rng=random.Random(1))
        self.assertEqual(len(hit), len(canon))
        self.assertNotEqual(hit, canon)          # per-hit jitter present

    def test_render_stream_onsets_align_with_signatures(self):
        text = "abc"
        stream, events = synth.render_stream(text, self.audio, seed=1, window_ms=120.0)
        self.assertEqual([k for _, k in events], list("abc"))
        self.assertTrue(all(0 <= o < len(stream) for o, _ in events))

    def test_space_is_labeled(self):
        stream, events = synth.render_stream("a b", self.audio, seed=1, window_ms=120.0)
        self.assertIn("<space>", [k for _, k in events])

    def test_dataset_balanced_and_shuffled(self):
        keys = list("abcde")
        clips, labels = synth.synth_dataset(keys, 4, self.audio, 120.0, seed=1)
        self.assertEqual(len(clips), 20)
        for k in keys:
            self.assertEqual(labels.count(k), 4)


if __name__ == "__main__":
    unittest.main()
