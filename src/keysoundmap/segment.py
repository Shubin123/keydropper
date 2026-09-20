"""Keystroke onset detection and window extraction.

A continuous recording is turned into discrete keystroke clips by:
  1. computing a short-time energy envelope,
  2. smoothing it,
  3. estimating a robust noise floor and an adaptive threshold from it,
  4. picking onsets where the envelope rises through the threshold, subject to a
     refractory period (you cannot physically type faster than ~15 keys/s), and
  5. cutting a fixed-length window around each onset for the feature extractor.

Pure Python; scored against synthetic ground-truth onsets in the tests.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import AudioConfig, SegmentConfig
from . import dsp


def _ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round(ms * sr / 1000.0)))


def detect_onsets(
    x: Sequence[float], audio: AudioConfig, seg: SegmentConfig
) -> List[int]:
    """Return sample indices of detected keystroke onsets, in order."""
    sr = audio.sample_rate
    frame_len = _ms_to_samples(seg.frame_ms, sr)
    hop = frame_len  # non-overlapping energy frames keep the envelope cheap
    env, centers = dsp.short_time_energy(x, frame_len, hop)
    if not env:
        return []

    smooth_frames = max(1, int(round(seg.smooth_ms / seg.frame_ms)))
    if smooth_frames % 2 == 0:
        smooth_frames += 1
    sm = dsp.moving_average(env, smooth_frames)

    noise_floor = dsp.percentile(sm, 20.0)
    # Guard: near-silent input still needs a positive threshold.
    peak = max(sm)
    floor = max(noise_floor, 1e-6 * (peak if peak > 0 else 1.0))
    threshold = seg.threshold_ratio * floor

    refractory = _ms_to_samples(seg.refractory_ms, sr)

    onsets: List[int] = []
    last_onset = -refractory - 1
    armed = True  # require the envelope to fall back below threshold before re-firing
    for i in range(1, len(sm)):
        rising = sm[i] > sm[i - 1]
        if armed and sm[i] >= threshold and rising:
            center = centers[i]
            if center - last_onset >= refractory:
                # Refine to the local energy maximum within the smoothing window.
                onsets.append(_refine_onset(sm, centers, i))
                last_onset = onsets[-1]
                armed = False
        elif sm[i] < threshold:
            armed = True
    return onsets


def _refine_onset(sm: List[float], centers: List[int], i: int) -> int:
    """Snap an onset to the nearby envelope peak for a stable window center."""
    j = i
    n = len(sm)
    # Walk uphill a few frames to the local peak.
    steps = 0
    while j + 1 < n and sm[j + 1] >= sm[j] and steps < 8:
        j += 1
        steps += 1
    return centers[j]


def extract_windows(
    x: Sequence[float],
    onsets: Sequence[int],
    audio: AudioConfig,
    seg: SegmentConfig,
) -> List[List[float]]:
    """Cut a fixed-length, zero-padded clip around each onset."""
    sr = audio.sample_rate
    win = _ms_to_samples(seg.window_ms, sr)
    pre = _ms_to_samples(seg.pre_onset_ms, sr)
    n = len(x)
    clips: List[List[float]] = []
    for onset in onsets:
        start = onset - pre
        clip = [0.0] * win
        for k in range(win):
            src = start + k
            if 0 <= src < n:
                clip[k] = x[src]
        clips.append(clip)
    return clips


def segment(
    x: Sequence[float], audio: AudioConfig, seg: SegmentConfig
) -> Tuple[List[int], List[List[float]]]:
    """Convenience: return ``(onsets, clips)`` in one call."""
    onsets = detect_onsets(x, audio, seg)
    clips = extract_windows(x, onsets, audio, seg)
    return onsets, clips


def match_onsets(
    detected: Sequence[int],
    truth: Sequence[int],
    tolerance_samples: int,
) -> Tuple[int, int, int]:
    """Greedy one-to-one match of detected vs true onsets within a tolerance.

    Returns ``(true_positives, false_positives, false_negatives)`` for scoring the
    segmenter (precision/recall/F1) against synthetic ground truth.
    """
    truth_used = [False] * len(truth)
    tp = 0
    det = sorted(detected)
    tru = sorted(truth)
    for d in det:
        best = -1
        best_dist = tolerance_samples + 1
        for idx, t in enumerate(tru):
            if truth_used[idx]:
                continue
            dist = abs(d - t)
            if dist <= tolerance_samples and dist < best_dist:
                best_dist = dist
                best = idx
        if best >= 0:
            truth_used[best] = True
            tp += 1
    fp = len(det) - tp
    fn = len(tru) - tp
    return tp, fp, fn
