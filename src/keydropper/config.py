"""Configuration dataclasses shared across the pipeline.

Keeping every knob in one place makes experiments reproducible: a run is fully
described by a ``Config`` instance, which can be serialized to JSON and stored next
to results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class AudioConfig:
    """Raw-audio conventions. Everything downstream assumes mono float samples."""

    sample_rate: int = 16_000  # 16 kHz captures keystroke transients well and is cheap
    # Keystroke transients have energy up to a few kHz; 16 kHz (Nyquist 8 kHz) is ample
    # and matches the keyword-spotting model lineage we borrow from.


@dataclass(frozen=True)
class SegmentConfig:
    """Onset-based keystroke segmentation."""

    frame_ms: float = 4.0            # energy-envelope hop
    smooth_ms: float = 8.0           # moving-average smoothing of the envelope
    threshold_ratio: float = 3.0     # onset when smoothed energy > ratio * noise floor
    refractory_ms: float = 70.0      # min spacing between onsets (typing is < ~15 keys/s)
    window_ms: float = 120.0         # clip length extracted around each onset
    pre_onset_ms: float = 10.0       # include a little context before the transient


@dataclass(frozen=True)
class FeatureConfig:
    """Log-mel spectrogram per keystroke window."""

    n_fft: int = 512                 # power of two -> radix-2 FFT in the pure-Python core
    hop_ms: float = 4.0
    win_ms: float = 16.0
    n_mels: int = 40
    fmin: float = 40.0
    fmax: float = 8_000.0
    log_floor: float = 1e-10         # avoid log(0)

    def n_frames(self, window_ms: float) -> int:
        """Number of STFT frames produced for a clip of ``window_ms`` (informational)."""
        # frames are computed by features.py; this mirror is handy for shape checks.
        from math import floor
        hop = self.hop_ms
        return max(1, floor((window_ms - self.win_ms) / hop) + 1)


@dataclass(frozen=True)
class ModelConfig:
    """Recognizer + interference model hyperparameters."""

    # --- recognizer (small, efficient) ---
    arch: str = "tiny_key_cnn"       # {"tiny_key_cnn", "bcresnet_lite", "prototype"}
    width: int = 32                  # base channel width for the CNN
    dropout: float = 0.1
    # --- interference / defense ---
    # Protection is a monotonic trade-off against audible noise. Measured against the
    # strongest attacker we test (see PLAN.md Phase 6), starting from 93% recovery:
    #   decoys=2 gain=0.9 -> 37% (9.9x chance)    decoys=4 gain=1.2 -> 13% (3.4x)
    #   decoys=4 gain=1.5 -> 11% (3.1x)           decoys=6 gain=1.5 ->  7% (1.8x)
    # The default is the balanced setting; raise both for stronger protection.
    decoys_per_key: int = 4          # masking bursts emitted per real keystroke
    mask_gain: float = 1.2           # loudness of masking relative to a real keystroke
    timing_jitter_ms: float = 12.0   # decoy timing jitter to defeat alignment attacks


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 1234

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @staticmethod
    def default() -> "Config":
        return Config()
