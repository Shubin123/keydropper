"""keydropper — a blue-team study of the keyboard acoustic side channel.

We reproduce a compact keystroke recognizer (the threat "measuring stick") only so
that we can build and prove an *interference model*: an active acoustic countermeasure
that collapses an eavesdropper's transcription accuracy toward chance.

The core pipeline (DSP, segmentation, features, prototype classifier, masking) is
implemented in pure Python so it runs with no third-party dependencies. numpy/torch
fast paths and the deep models are optional and used for the training phases.
"""

from .config import AudioConfig, FeatureConfig, SegmentConfig, ModelConfig, Config

__all__ = [
    "AudioConfig",
    "FeatureConfig",
    "SegmentConfig",
    "ModelConfig",
    "Config",
]

__version__ = "0.1.0"
