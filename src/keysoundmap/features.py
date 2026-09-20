"""Log-mel spectrogram features for a single keystroke clip.

Log-mel is the standard front end for both the keyboard-acoustic attack literature
and the keyword-spotting small models we borrow from, so we use it for the recognizer
and reuse the same representation when evaluating the defense.

A clip becomes a fixed ``(n_frames, n_mels)`` matrix. The pure-Python path uses the
radix-2 FFT in ``dsp``; if numpy is available a vectorized path is used and is checked
against the reference in the tests.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Sequence, Tuple

from .config import AudioConfig, FeatureConfig
from . import dsp

try:  # optional acceleration only
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    _np = None


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> Tuple[Tuple[float, ...], ...]:
    """Triangular mel filterbank as a tuple of ``n_mels`` rows over rfft bins.

    Cached because the bank depends only on config, not on the audio.
    """
    n_bins = n_fft // 2 + 1
    fmax = min(fmax, sr / 2.0)
    mel_lo, mel_hi = hz_to_mel(fmin), hz_to_mel(fmax)
    # n_mels + 2 edge points -> n_mels triangles.
    mel_points = [mel_lo + (mel_hi - mel_lo) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    bin_freqs = [i * sr / n_fft for i in range(n_bins)]

    bank: List[Tuple[float, ...]] = []
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        row = [0.0] * n_bins
        for b, f in enumerate(bin_freqs):
            if left <= f <= center and center > left:
                row[b] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                row[b] = (right - f) / (right - center)
        bank.append(tuple(row))
    return tuple(bank)


def logmel(
    clip: Sequence[float], audio: AudioConfig, feat: FeatureConfig
) -> List[List[float]]:
    """Return an ``(n_frames, n_mels)`` log-mel matrix for one clip."""
    sr = audio.sample_rate
    win = max(1, int(round(feat.win_ms * sr / 1000.0)))
    hop = max(1, int(round(feat.hop_ms * sr / 1000.0)))
    n_fft = feat.n_fft
    if win > n_fft:
        win = n_fft
    window = dsp.hann_window(win)
    bank = mel_filterbank(sr, n_fft, feat.n_mels, feat.fmin, feat.fmax)

    frames = dsp.frame_signal(clip, win, hop)
    out: List[List[float]] = []
    for fr in frames:
        wf = [fr[i] * window[i] for i in range(win)]
        power = dsp.rfft_power(wf, n_fft)
        row: List[float] = []
        for filt in bank:
            acc = 0.0
            for b in range(len(power)):
                w = filt[b]
                if w:
                    acc += w * power[b]
            row.append(math.log(acc + feat.log_floor))
        out.append(row)
    if not out:
        # Degenerate very-short clip: emit a single zeroed frame so shapes are stable.
        out.append([math.log(feat.log_floor)] * feat.n_mels)
    return out


def logmel_np(clip, audio: AudioConfig, feat: FeatureConfig):
    """numpy fast path returning an ``(n_frames, n_mels)`` array (if numpy present)."""
    if _np is None:  # pragma: no cover
        raise RuntimeError("numpy not available")
    sr = audio.sample_rate
    win = min(max(1, int(round(feat.win_ms * sr / 1000.0))), feat.n_fft)
    hop = max(1, int(round(feat.hop_ms * sr / 1000.0)))
    x = _np.asarray(clip, dtype=_np.float64)
    if len(x) < win:
        x = _np.pad(x, (0, win - len(x)))
    n_frames = 1 + (len(x) - win) // hop
    idx = _np.arange(win)[None, :] + hop * _np.arange(n_frames)[:, None]
    frames = x[idx] * _np.hanning(win + 1)[:-1][None, :]  # periodic Hann
    spec = _np.fft.rfft(frames, n=feat.n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    bank = _np.asarray(mel_filterbank(sr, feat.n_fft, feat.n_mels, feat.fmin, feat.fmax))
    mel = power @ bank.T
    return _np.log(mel + feat.log_floor)


def align_peak(mat: List[List[float]], target_frame: int, n_frames: int, n_mels: int) -> List[List[float]]:
    """Shift a log-mel matrix so its highest-energy frame sits at ``target_frame``.

    The onset detector cannot place a key's transient at exactly the same offset in
    every recording (jitter, noise, and key-specific envelopes move the picked peak),
    so absolute-position features do not transfer across streams. Aligning each clip to
    its own transient peak — the "push-peak alignment" used in the keyboard-acoustics
    literature — makes features comparable regardless of where segmentation cut. This
    is what lets a recognizer generalize from one recording to another (and it is the
    single change that turns cross-stream accuracy from chance into real recognition).
    """
    if not mat:
        return [[0.0] * n_mels for _ in range(n_frames)]
    # Per-frame energy proxy: mean log-mel (transient frame is broadband -> highest).
    energies = [sum(row) / len(row) for row in mat]
    peak = max(range(len(energies)), key=lambda i: energies[i])
    shift = target_frame - peak
    floor_row = [min(min(r) for r in mat)] * n_mels
    out: List[List[float]] = []
    for f in range(n_frames):
        src = f - shift
        if 0 <= src < len(mat):
            out.append(list(mat[src]))
        else:
            out.append(list(floor_row))
    return out


def flatten(mat: Sequence[Sequence[float]]) -> List[float]:
    """Flatten a feature matrix to a 1-D vector (row-major)."""
    out: List[float] = []
    for row in mat:
        out.extend(row)
    return out


def fixed_length(mat: List[List[float]], n_frames: int, n_mels: int) -> List[List[float]]:
    """Pad/truncate a feature matrix to exactly ``n_frames`` rows for batching."""
    out = [list(r) for r in mat[:n_frames]]
    while len(out) < n_frames:
        out.append([0.0] * n_mels)
    return out
