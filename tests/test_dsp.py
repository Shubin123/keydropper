import cmath
import math
import unittest

import _pathfix  # noqa: F401

from keysoundmap import dsp


def naive_dft_power(frame, n_fft):
    buf = list(frame) + [0.0] * (n_fft - len(frame))
    half = n_fft // 2 + 1
    out = []
    for k in range(half):
        acc = 0j
        for t in range(n_fft):
            acc += buf[t] * cmath.exp(-2j * math.pi * k * t / n_fft)
        out.append(abs(acc) ** 2)
    return out


class TestDSP(unittest.TestCase):
    def test_fft_matches_naive_dft(self):
        frame = [math.sin(2 * math.pi * 5 * t / 64) + 0.3 * math.cos(2 * math.pi * 11 * t / 64)
                 for t in range(50)]
        fast = dsp.rfft_power(frame, 64)
        slow = naive_dft_power(frame, 64)
        self.assertEqual(len(fast), len(slow))
        for a, b in zip(fast, slow):
            self.assertAlmostEqual(a, b, places=6)

    def test_sine_peaks_at_expected_bin(self):
        n_fft = 128
        k0 = 10
        frame = [math.sin(2 * math.pi * k0 * t / n_fft) for t in range(n_fft)]
        power = dsp.rfft_power(frame, n_fft)
        peak_bin = max(range(len(power)), key=lambda i: power[i])
        self.assertEqual(peak_bin, k0)

    def test_hann_window_bounds(self):
        w = dsp.hann_window(16)
        self.assertEqual(len(w), 16)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in w))
        self.assertAlmostEqual(w[0], 0.0, places=9)

    def test_framing_counts(self):
        x = list(range(100))
        frames = dsp.frame_signal(x, frame_len=10, hop=5)
        self.assertEqual(len(frames), 1 + (100 - 10) // 5)
        self.assertEqual(frames[0], list(range(10)))

    def test_next_power_of_two(self):
        self.assertEqual(dsp.next_power_of_two(1), 1)
        self.assertEqual(dsp.next_power_of_two(17), 32)
        self.assertTrue(dsp.is_power_of_two(64))
        self.assertFalse(dsp.is_power_of_two(63))


if __name__ == "__main__":
    unittest.main()
