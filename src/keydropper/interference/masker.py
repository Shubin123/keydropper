"""Tier-1 countermeasure: spectrally-matched decoy masking.

Rationale (Anand et al., *A Sound for a Sound*, FC'16): masking a keystroke with
**other keystroke sounds** beats white noise, because the decoys share the real
signal's spectral and temporal statistics, so an attacker cannot separate real from
fake by a fixed filter or by spectral shape. We exploit the defender's structural
advantage: the user's own machine *knows* when a key is pressed (an OS key event), so
it can emit decoys proactively, tightly aligned to the real transient, something the
remote eavesdropper cannot do.

The defense emits, for each real keypress, several decoy keystroke bursts drawn from a
bank of keystroke templates, at jittered timings and gains. Real and decoy onsets then
overlap and interleave, corrupting both the attacker's segmentation (extra/î merged
onsets) and its per-key classification (spectra summed with foreign keys).

The decoy bank here is generated from ``synth`` for a dependency-free demo. In a real
deployment it would be filled with short recordings of the user's own keyboard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config import AudioConfig, ModelConfig
from .. import synth


@dataclass
class DecoyBank:
    """A pool of keystroke templates the masker samples decoys from."""

    templates: List[List[float]]
    keys: List[str]

    @staticmethod
    def synthetic(keys: Sequence[str], audio: AudioConfig, window_ms: float, seed: int = 999) -> "DecoyBank":
        # Decoys must sound like the SAME keyboard (that is what makes them
        # indistinguishable), so they use the default keyboard identity. ``seed`` is
        # kept for API symmetry but does not change the keyboard here.
        tmpl = [synth.key_signature(k, audio, window_ms) for k in keys]
        return DecoyBank(templates=tmpl, keys=list(keys))

    def sample(self, rng: random.Random) -> List[float]:
        i = rng.randrange(len(self.templates))
        return self.templates[i]


def make_decoy(
    bank: DecoyBank, gain: float, rng: random.Random
) -> List[float]:
    """Draw one decoy burst and apply gain + small amplitude jitter."""
    tmpl = bank.sample(rng)
    g = gain * rng.uniform(0.85, 1.15)
    return [v * g for v in tmpl]


def apply_masking(
    samples: Sequence[float],
    key_events: Sequence[Tuple[int, str]],
    audio: AudioConfig,
    model: ModelConfig,
    bank: DecoyBank,
    seed: int = 4242,
) -> List[float]:
    """Return a defended copy of ``samples`` with decoy bursts mixed in.

    ``key_events`` are ``(onset_sample, key)`` pairs the defender observes from the OS.
    For each event we emit ``model.decoys_per_key`` decoys at timings jittered around
    the true onset by up to ``model.timing_jitter_ms``, so decoys cannot be removed by
    a fixed offset. The real key identity is *not* used — decoys are drawn blindly from
    the bank, which is what makes them indistinguishable in aggregate.
    """
    sr = audio.sample_rate
    rng = random.Random(seed)
    out = list(samples)
    n = len(out)
    jitter = max(1, int(round(model.timing_jitter_ms * sr / 1000.0)))

    for onset, _key in key_events:
        for _ in range(model.decoys_per_key):
            decoy = make_decoy(bank, model.mask_gain, rng)
            offset = onset + rng.randint(-jitter, jitter)
            # Decoy transient sits ~10% into its window (see synth.key_signature).
            start = offset - int(0.10 * len(decoy))
            for i, v in enumerate(decoy):
                j = start + i
                if 0 <= j < n:
                    out[j] += v
    return out


def synthesize_decoy_events(
    key_events: Sequence[Tuple[int, str]],
    audio: AudioConfig,
    model: ModelConfig,
    seed: int = 4242,
) -> List[int]:
    """Onset samples the masker *would* add — handy for analysis/plots (not required)."""
    sr = audio.sample_rate
    rng = random.Random(seed)
    jitter = max(1, int(round(model.timing_jitter_ms * sr / 1000.0)))
    out: List[int] = []
    for onset, _ in key_events:
        for _ in range(model.decoys_per_key):
            out.append(onset + rng.randint(-jitter, jitter))
    return out


def realtime_playback_available() -> bool:
    """True if ``sounddevice`` is importable for live masking (Phase 7)."""
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:
        return False
