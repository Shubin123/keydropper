"""Dataset assembly that keeps train and test distributions matched.

The subtle, important rule the integration tests enforce: **the recognizer must be
trained on clips produced by the very same segmentation it will see at inference
time**. If you train on idealized, perfectly-centered keystroke windows but test on
windows cut by the onset detector, alignment differs and accuracy collapses — a
classic acoustic-side-channel pipeline pitfall.

So training data is built by *rendering typing streams, segmenting them, and labeling
each detected clip by the ground-truth key nearest its onset*. Real captures
(``capture.py``) feed the same function via their ``(onset_sample, key)`` events.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from .config import Config
from . import synth, segment as seg_mod


def label_detected_clips(
    stream: Sequence[float],
    truth_events: Sequence[Tuple[int, str]],
    cfg: Config,
) -> Tuple[List[List[float]], List[str]]:
    """Segment ``stream`` and label each detected clip by the nearest true onset.

    Detected onsets with no true onset within tolerance are dropped (they would be
    insertions/decoys, which carry no reliable label for supervised training).
    """
    onsets, clips = seg_mod.segment(stream, cfg.audio, cfg.segment)
    sr = cfg.audio.sample_rate
    tol = int(round(cfg.segment.refractory_ms * 0.5 * sr / 1000.0))
    truth = sorted(truth_events)
    used = [False] * len(truth)

    out_clips: List[List[float]] = []
    out_labels: List[str] = []
    for onset, clip in zip(onsets, clips):
        best = -1
        best_dist = tol + 1
        for i, (t_onset, _k) in enumerate(truth):
            if used[i]:
                continue
            d = abs(onset - t_onset)
            if d <= tol and d < best_dist:
                best_dist = d
                best = i
        if best >= 0:
            used[best] = True
            out_clips.append(clip)
            out_labels.append(truth[best][1])
    return out_clips, out_labels


def build_training_set(
    keys: Sequence[str],
    per_key: int,
    cfg: Config,
    seed: int = 0,
    transform=None,
) -> Tuple[List[List[float]], List[str]]:
    """Build a segmentation-matched training set.

    Renders shuffled streams containing ``per_key`` presses of each key, segments them,
    and returns labeled clips. Because these clips pass through the onset detector, they
    match the distribution the recognizer sees on real streams.

    ``transform(stream, truth_events) -> stream`` optionally rewrites each rendered
    stream before segmentation. That is how an **adaptive attacker** collects training
    data: with the countermeasure switched on, so it learns the masked distribution
    rather than the clean one. Phase 6 of the plan requires the defense to survive it.
    """
    rng = random.Random(cfg.seed + seed)
    sequence: List[str] = []
    for k in keys:
        sequence.extend([k] * per_key)
    rng.shuffle(sequence)

    # Render in chunks so streams stay a manageable length.
    chunk = 40
    clips: List[List[float]] = []
    labels: List[str] = []
    for start in range(0, len(sequence), chunk):
        part = sequence[start : start + chunk]
        text = "".join(" " if k == "<space>" else k for k in part)
        # render_stream maps ' ' -> '<space>', other chars to themselves.
        stream, truth = synth.render_stream(
            text, cfg.audio, seed=cfg.seed + seed + start, window_ms=cfg.segment.window_ms
        )
        if transform is not None:
            stream = transform(stream, truth)
        c, y = label_detected_clips(stream, truth, cfg)
        clips.extend(c)
        labels.extend(y)
    return clips, labels
