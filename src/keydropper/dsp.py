"""Minimal DSP primitives with no third-party dependencies.

Everything here works on plain Python lists of floats so the pipeline runs anywhere,
including CI with only the standard library. When numpy is importable we still keep
these as the reference implementation; production speed comes from the numpy path in
``features.py`` (guarded import), validated against these functions in the tests.

The FFT is an iterative radix-2 Cooley-Tukey transform, so ``n_fft`` must be a power
of two (enforced by ``FeatureConfig``).
"""

from __future__ import annotations

import cmath
import math
from typing import List, Sequence, Tuple


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hann_window(n: int) -> List[float]:
    """Periodic Hann window (matches numpy/librosa ``sym=False`` convention)."""
    if n == 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / n) for i in range(n)]


def frame_signal(
    x: Sequence[float], frame_len: int, hop: int
) -> List[List[float]]:
    """Split ``x`` into overlapping frames of ``frame_len`` advanced by ``hop``.

    Trailing samples that do not fill a whole frame are dropped (standard STFT
    behaviour). Returns an empty list if the signal is shorter than one frame.
    """
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    n = len(x)
    frames: List[List[float]] = []
    if n < frame_len:
        return frames
    start = 0
    while start + frame_len <= n:
        frames.append(list(x[start : start + frame_len]))
        start += hop
    return frames


def _fft(a: List[complex]) -> List[complex]:
    """In-place-style iterative radix-2 FFT. ``len(a)`` must be a power of two."""
    n = len(a)
    if n <= 1:
        return a
    if not is_power_of_two(n):
        raise ValueError(f"FFT length must be a power of two, got {n}")

    # Bit-reversal permutation.
    a = list(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Butterflies.
    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wlen = cmath.rect(1.0, ang)
        half = length >> 1
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def rfft_power(frame: Sequence[float], n_fft: int) -> List[float]:
    """Power spectrum (|X|^2) of one real frame, zero-padded/truncated to ``n_fft``.

    Returns ``n_fft // 2 + 1`` non-negative bins (the non-redundant half of the
    spectrum), matching ``numpy.fft.rfft`` bin layout.
    """
    if not is_power_of_two(n_fft):
        raise ValueError(f"n_fft must be a power of two, got {n_fft}")
    buf: List[complex] = [0j] * n_fft
    m = min(len(frame), n_fft)
    for i in range(m):
        buf[i] = complex(frame[i], 0.0)
    spec = _fft(buf)
    half = n_fft // 2 + 1
    return [(spec[i].real * spec[i].real + spec[i].imag * spec[i].imag) for i in range(half)]


def short_time_energy(
    x: Sequence[float], frame_len: int, hop: int
) -> Tuple[List[float], List[int]]:
    """Per-frame RMS energy envelope and the sample index at each frame's centre.

    Used by the onset segmenter. Cheap and allocation-light so it is fine to run on
    long streams.
    """
    env: List[float] = []
    centers: List[int] = []
    n = len(x)
    if n == 0:
        return env, centers
    start = 0
    while start < n:
        end = min(start + frame_len, n)
        s = 0.0
        for i in range(start, end):
            v = x[i]
            s += v * v
        count = end - start
        env.append(math.sqrt(s / count) if count else 0.0)
        centers.append(start + count // 2)
        start += hop
    return env, centers


def moving_average(x: Sequence[float], k: int) -> List[float]:
    """Centered moving average with window ``k`` (odd recommended); edge-clamped."""
    if k <= 1:
        return list(x)
    n = len(x)
    half = k // 2
    out: List[float] = [0.0] * n
    # Prefix sums for O(n).
    pref = [0.0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + x[i]
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (pref[hi] - pref[lo]) / (hi - lo)
    return out


def percentile(sorted_or_not: Sequence[float], q: float) -> float:
    """Simple percentile (q in [0, 100]); used for a robust noise-floor estimate."""
    if not sorted_or_not:
        return 0.0
    xs = sorted(sorted_or_not)
    if q <= 0:
        return xs[0]
    if q >= 100:
        return xs[-1]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac
