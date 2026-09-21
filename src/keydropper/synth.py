"""Synthetic keyboard-audio generator.

Real keystroke recordings are the eventual input (see ``capture.py`` +
``docs/DATA_COLLECTION.md``), but for development, CI, and — importantly — for
*evaluating the defense* we need a controllable acoustic model with known ground
truth. Each key is given a distinct but stable signature so a recognizer can learn to
separate keys, while per-hit jitter (amplitude, micro-detuning, noise, timing) keeps
the task non-trivial and forces the models to generalize.

Physical model (deliberately simple, matched to the literature's description of
keyboard emanations): a keystroke is a sharp transient — a "hit" peak with a handful
of key-specific damped resonances — followed by a weaker "release" peak, over a floor
of broadband noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import AudioConfig


@dataclass(frozen=True)
class Resonance:
    freq: float      # Hz
    decay: float     # amplitude e-folding time in seconds
    amp: float


# A single fixed "keyboard identity": the per-key resonances must NOT depend on which
# recording (stream) they appear in, otherwise every recording would be a different
# physical keyboard and no recognizer could ever generalize across recordings. Only
# per-hit jitter and background noise vary between recordings.
KEYBOARD_SEED = 20240517


def _key_resonances(key: str, keyboard_seed: int = KEYBOARD_SEED) -> List[Resonance]:
    """Deterministic per-key set of damped resonances for a fixed keyboard.

    Seeded on ``(keyboard_seed, key)`` so a key sounds the same in every recording,
    which is what lets a recognizer trained on one session transfer to another.
    """
    rng = random.Random(f"{keyboard_seed}:{key}")
    n = 5  # more resonances -> richer, more separable per-key spectral signatures
    out: List[Resonance] = []
    for _ in range(n):
        out.append(
            Resonance(
                freq=rng.uniform(400.0, 7200.0),
                decay=rng.uniform(0.005, 0.030),
                amp=rng.uniform(0.4, 1.0),
            )
        )
    return out


def key_signature(
    key: str,
    audio: AudioConfig,
    window_ms: float,
    keyboard_seed: int = KEYBOARD_SEED,
    rng: random.Random | None = None,
) -> List[float]:
    """Render one keystroke waveform (length ``window_ms`` at the configured rate).

    The key's spectral identity comes from ``keyboard_seed`` (fixed per keyboard);
    ``rng`` (when given) injects per-hit variation (amplitude, micro-detuning, noise).
    Without ``rng`` the canonical, noise-free template is returned — used as a masking
    decoy source and in tests.
    """
    sr = audio.sample_rate
    n = max(1, int(round(window_ms * sr / 1000.0)))
    res = _key_resonances(key, keyboard_seed)

    # Per-hit variation.
    amp_scale = 1.0
    detune = 1.0
    noise_amp = 0.0
    if rng is not None:
        # Real keyboards have a *consistent* per-key signature (why the attack works);
        # keep hit-to-hit variation realistic but modest so key identity dominates.
        amp_scale = rng.uniform(0.9, 1.1)
        detune = rng.uniform(0.995, 1.005)
        noise_amp = rng.uniform(0.002, 0.008)

    out = [0.0] * n
    hit_at = int(0.10 * n)          # transient starts a little into the window
    release_at = int(0.55 * n)      # weaker second peak (key returning)
    for i in range(n):
        s = 0.0
        # Hit transient.
        t = (i - hit_at) / sr
        if t >= 0:
            for r in res:
                s += r.amp * math.exp(-t / r.decay) * math.sin(2 * math.pi * r.freq * detune * t)
        # Release transient (softer, faster decay).
        t2 = (i - release_at) / sr
        if t2 >= 0:
            for r in res:
                s += 0.4 * r.amp * math.exp(-t2 / (0.6 * r.decay)) * math.sin(
                    2 * math.pi * r.freq * detune * t2
                )
        out[i] = amp_scale * s

    if noise_amp and rng is not None:
        for i in range(n):
            out[i] += rng.gauss(0.0, noise_amp)

    # Normalize to unit peak so downstream gains are interpretable.
    peak = max((abs(v) for v in out), default=1.0) or 1.0
    return [v / peak for v in out]


def synth_dataset(
    keys: Sequence[str],
    per_key: int,
    audio: AudioConfig,
    window_ms: float,
    seed: int = 1234,
    keyboard_seed: int = KEYBOARD_SEED,
) -> Tuple[List[List[float]], List[str]]:
    """Return ``(clips, labels)``: ``per_key`` jittered windows for each key.

    ``seed`` varies per-hit jitter/noise (the recording); ``keyboard_seed`` fixes the
    keyboard identity so every dataset describes the same physical keyboard.
    """
    rng = random.Random(seed)
    clips: List[List[float]] = []
    labels: List[str] = []
    for key in keys:
        for _ in range(per_key):
            clips.append(key_signature(key, audio, window_ms, keyboard_seed=keyboard_seed, rng=rng))
            labels.append(key)
    # Shuffle so train/val splits are not grouped by key.
    idx = list(range(len(clips)))
    rng.shuffle(idx)
    return [clips[i] for i in idx], [labels[i] for i in idx]


def render_stream(
    text: str,
    audio: AudioConfig,
    seed: int = 1234,
    keys_per_second: float = 6.0,
    window_ms: float = 120.0,
    background_noise: float = 0.01,
    keyboard_seed: int = KEYBOARD_SEED,
) -> Tuple[List[float], List[Tuple[int, str]]]:
    """Render a continuous typing stream for ``text``.

    Returns ``(samples, events)`` where each event is ``(onset_sample, key)`` — the
    ground truth the onset segmenter is scored against. Spaces are treated as the
    ``"<space>"`` key; other characters are rendered as themselves.
    """
    sr = audio.sample_rate
    rng = random.Random(seed)
    inter = 1.0 / keys_per_second
    onsets: List[Tuple[int, str]] = []
    # Reserve room; we place each keystroke signature at a jittered inter-key interval.
    positions: List[int] = []
    t = 0.2  # start after 200 ms of silence
    for ch in text:
        gap = inter * rng.uniform(0.6, 1.4)
        t += gap
        positions.append(int(t * sr))
    total = (positions[-1] if positions else int(0.4 * sr)) + int(window_ms * sr / 1000.0) + int(0.2 * sr)
    samples = [rng.gauss(0.0, background_noise) for _ in range(total)]

    for pos, ch in zip(positions, text):
        key = "<space>" if ch == " " else ch
        sig = key_signature(key, audio, window_ms, keyboard_seed=keyboard_seed, rng=rng)
        # The transient "onset" is ~10% into the signature window (see key_signature).
        onset = pos + int(0.10 * len(sig))
        onsets.append((onset, key))
        for i, v in enumerate(sig):
            j = pos + i
            if 0 <= j < total:
                samples[j] += v
    return samples, onsets
