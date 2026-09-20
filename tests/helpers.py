"""Shared test helpers: a small, fast configuration and key set.

The full pipeline (40 mels, 512-pt FFT, 27 keys) is realistic but slowish in pure
Python. Tests use a trimmed config so the dev loop stays snappy while still exercising
every stage. Correctness properties (cross-stream generalization, defense suppression)
hold under both configs.
"""

from __future__ import annotations

import dataclasses

from keysoundmap.config import Config, FeatureConfig

# A small keyboard for fast tests.
FAST_KEYS = list("abcdefghij") + ["<space>"]


def fast_config() -> Config:
    feat = FeatureConfig(n_fft=256, n_mels=24)
    return dataclasses.replace(Config(), feature=feat)
