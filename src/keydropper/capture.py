"""Phase 1 real-data capture: record mic audio time-aligned to key events.

Produces the labeled ``(audio window -> key)`` pairs the recognizer trains on. Needs
``sounddevice`` (audio) and ``pynput`` (key events); both are imported lazily so the
rest of the package works without them. See ``docs/DATA_COLLECTION.md`` for the
protocol that keeps the dataset honest (session-disjoint splits, random strings, fixed
rig).

Only use this on your own machine and keyboard. It logs which keys *you* press while
recording *your* microphone; it is the instrument for measuring your own leakage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import AudioConfig


@dataclass
class CaptureSession:
    """Holds a recording and the key events captured alongside it."""

    sample_rate: int
    audio: List[float] = field(default_factory=list)
    events: List[Tuple[float, str]] = field(default_factory=list)  # (t_seconds, key)
    started_at: float = 0.0

    def save(self, wav_path: str, labels_path: str) -> None:
        """Write audio to WAV (stdlib ``wave``) and events to JSON."""
        import wave
        import struct

        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16
            w.setframerate(self.sample_rate)
            frames = bytearray()
            for s in self.audio:
                v = max(-1.0, min(1.0, s))
                frames += struct.pack("<h", int(v * 32767))
            w.writeframes(bytes(frames))
        with open(labels_path, "w") as f:
            json.dump(
                {"sample_rate": self.sample_rate, "events": self.events},
                f,
                indent=2,
            )

    def to_onset_samples(self) -> List[Tuple[int, str]]:
        """Convert timestamped events to ``(onset_sample, key)`` for the pipeline."""
        return [(int(t * self.sample_rate), key) for t, key in self.events]


def record(audio: AudioConfig, duration_s: float) -> CaptureSession:  # pragma: no cover
    """Record ``duration_s`` of mic audio while logging key events.

    Blocks for the duration. Requires ``sounddevice`` and ``pynput``.
    """
    try:
        import sounddevice as sd
        from pynput import keyboard
    except Exception as exc:
        raise ImportError(
            "record() needs sounddevice + pynput (pip install -r requirements-capture.txt)"
        ) from exc

    session = CaptureSession(sample_rate=audio.sample_rate, started_at=time.time())

    def on_press(key):
        try:
            name = key.char if hasattr(key, "char") and key.char else str(key)
        except Exception:
            name = str(key)
        session.events.append((time.time() - session.started_at, name))

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    frames = int(duration_s * audio.sample_rate)
    rec = sd.rec(frames, samplerate=audio.sample_rate, channels=1, dtype="float32")
    sd.wait()
    listener.stop()
    session.audio = [float(x[0]) for x in rec]
    return session
