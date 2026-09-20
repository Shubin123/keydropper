# keysoundmap

**Keyboards make noise. That noise can be mapped to keystrokes.** This repository
studies that acoustic side channel **as a blue team**: it reproduces a compact
keystroke recognizer only so that it can build and *prove* an **interference model** —
an active countermeasure that drives an eavesdropper's transcription accuracy back
toward random guessing.

> Defensive use only, on your own hardware. See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Two models, one repo
1. **Recognizer** (the threat yardstick) — a small, efficient, on-device keystroke
   classifier. Used to *measure* leakage and to give the defense a concrete adversary.
2. **Interference model** (the product) — emits sound from the user's own device that
   corrupts the eavesdropper's channel.

## What runs today (no dependencies required)
The whole core pipeline — DSP, onset segmentation, log-mel features, a baseline
recognizer, and the Tier-1 masking defense — is pure Python and runs on the standard
library. A synthetic keyboard generator provides labeled audio so it all works without
a microphone or a dataset.

```bash
# optional but faster dev loop (numpy fast path + pytest):
make venv

make demo          # segment -> recognize -> defend, end to end (synthetic)
make eval-defense  # attacker keystroke-recovery WITH vs WITHOUT masking
make test          # full test suite
make test-stdlib   # prove it runs with zero third-party deps
```

Example `make demo` output:

```
[Phase 2] onset detection: precision=1.000 recall=1.000 F1=1.000
[Phase 3] small recognizer on an unseen recording (cross-stream)
          cross-stream per-key accuracy = 0.975 (chance = 0.037)
[Phase 5/6] interference model vs the attacker
          attacker recovery  no-defense=0.935  defended=0.390  (leakage reduction=+0.545)
```

i.e. a 97.5%-accurate eavesdropper is cut to ~39% by the masking defense.

## Layout
```
src/keysoundmap/
  config.py        # all hyperparameters in one reproducible place
  dsp.py           # pure-Python framing, Hann window, radix-2 FFT, energy envelope
  synth.py         # synthetic keyboard: fixed per-key identity + per-hit jitter
  segment.py       # energy-based onset detection + window extraction
  features.py      # log-mel spectrogram (pure path + numpy fast path) with peak alignment
  data.py          # segmentation-matched dataset assembly (train == inference front end)
  models.py        # PrototypeClassifier, KNNClassifier, TinyKeyCNN/BCResNetLite (torch)
  capture.py       # Phase 1: record mic audio time-aligned to key events
  interference/
    masker.py            # Tier-1: spectrally-matched decoy masking (runs today)
    adversarial.py       # Tier-2: learned adversarial masker (torch)
    evaluate_defense.py  # red/blue harness: leakage reduction metric
  cli.py           # `python -m keysoundmap.cli {demo,segment,eval-defense,synth,info}`
tests/             # smoke, unit, integration (cross-stream + defense), e2e (CLI)
docs/              # THREAT_MODEL.md, DATA_COLLECTION.md
PLAN.md            # the multi-phase plan and state-of-the-art references
```

## Efficiency / small-model note
The deep recognizer (`TinyKeyCNN`) is a depthwise-separable CNN in the keyword-spotting
family (DS-CNN / MobileNet lineage), kept under a few hundred K parameters and
quantization-ready, so it fits real-time on-device use — the defender must run in real
time, and a lean recognizer is a fair, reproducible yardstick. See `PLAN.md`.

## Status
Phases 0–3 and the Tier-1 defense (Phases 5–6 core) are built and tested. Real-data
capture, sequence/LM decoding, the learned adversarial masker, and the real-time
defender daemon are designed and scaffolded — see `PLAN.md`.
