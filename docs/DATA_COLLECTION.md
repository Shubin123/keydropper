# Data collection protocol (Phase 1)

Real recordings replace the synthetic generator once the pipeline is validated. The
protocol below keeps the dataset honest so measured accuracy reflects the acoustic
channel, not leakage between train and test.

## Rig
- One keyboard, one microphone, fixed distance and angle. Note the model of each.
- Quiet room first (upper-bound leakage), then realistic noise (fan, room tone).
- Sample rate 16 kHz mono is enough; higher is fine (downsample in the pipeline).

## Capture
`keydropper.capture.record(...)` records mic audio while `pynput` logs keypress
timestamps, then `CaptureSession.save(wav, labels_json)` writes a WAV plus a JSON of
`(t_seconds, key)` events. `to_onset_samples()` converts events to the
`(onset_sample, key)` form the pipeline consumes.

```python
from keydropper.config import AudioConfig
from keydropper.capture import record
s = record(AudioConfig(), duration_s=120)
s.save("data/raw/session01.wav", "data/raw/session01.labels.json")
```

## What to type
- **Random strings** for the core train/test sets, so a language model cannot inflate
  scores — we want the raw acoustic accuracy, which the defense must lower.
- Cover every key you care about with enough repetitions (aim for ≥100 per key total).
- A separate natural-text session is fine for the *decoding* phase (Phase 4), kept out
  of the acoustic train/test split.

## Splitting (critical)
- **Split by session, never by keystroke.** Put whole recordings entirely in train or
  entirely in test. Keystrokes from the same recording are correlated (same mic
  placement, same room tone); mixing them across splits leaks and overstates accuracy —
  the same class of bug the cross-stream regression test guards against synthetically.
- Report accuracy on a session the recognizer never saw.

## Feeding the pipeline
Segment each recording with `segment.segment`, label detected clips against the logged
events with `data.label_detected_clips`, then featurize with `models.featurize_clips`.
The synthetic path in `data.build_training_set` mirrors exactly this flow, so swapping in
real data is a drop-in change.
