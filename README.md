# keydropper

**Keyboards make noise. That noise can be mapped to keystrokes.** This repository
studies that acoustic side channel **as a blue team**: it reproduces a compact
keystroke recognizer only so that it can build and *prove* an **interference model** —
an active countermeasure that drives an eavesdropper's transcription accuracy back
toward random guessing.

> Defensive use only, on your own hardware. See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Web app (runs in your browser)
The whole pipeline is also ported to a **self-contained, dependency-free web app** in
[`web/`](web/) — open `web/index.html` directly, or serve it over GitHub Pages.

- **Synthetic lab** — compare a lightweight recognizer against generated keystroke audio
  with and without modeled masking. Results are illustrative and do not represent real
  keyboards. Works anywhere, even from `file://`.
- **Live capture** — grant mic access, type to teach it your own keyboard, then let it
  read you; toggle masking to defend in real time. Needs HTTPS (Pages) or localhost.
- **Labeled WAV examples** — play or analyze three generated clips (`demo`, `hello world`,
  `blue team`) using the same browser recognizer as the lab. They are toy synthesized
  audio, not physical keyboard recordings. Drag modules to reorder them, or use the move
  buttons for keyboard and touch access. Regenerate/check them with
  `node scripts/generate_example_wav.js` and `node scripts/check_example_wav.js`.
- **Local WAV analysis** — select a mono 16 kHz PCM16 WAV file for in-browser analysis.
  The waveform and timestamped expected/predicted key markers share one seekable timeline;
  the playhead follows playback. Supply expected text to show ordered label matches;
  without it, results are shown as unlabeled predictions.

**Public demo:** [shubin123.github.io/keydropper](https://shubin123.github.io/keydropper/).
GitHub Pages publishes `web/` through GitHub Actions. Updates to `web/` deploy
automatically from `main`; see [the workflow](.github/workflows/deploy-pages.yml).

The browser app is a defensive teaching demo, not a port of Keytap3. It has no analytics
or upload endpoint; live audio is processed in the current tab.

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
          cross-stream per-key accuracy = 0.958 (chance = 0.037)
[Phase 5/6] interference model vs the attacker
          attacker recovery  no-defense=0.945  defended=0.130
          ADAPTIVE attacker (trained on masked audio)=0.125
          => strongest attacker under defense=0.130 (guaranteed reduction=+0.815)
```

i.e. a ~95%-accurate eavesdropper is cut to ~13% — and that number holds against an
**adaptive** attacker that knows the defense exists and trains on masked audio.
Interestingly, adapting makes the attacker *worse*: the decoys corrupt its per-key
prototypes, so it does better training on clean audio. The reported guarantee always
takes the **strongest** attacker, never the flattering one.

Protection is a dial traded against audible noise (`decoys_per_key`, `mask_gain`) —
from 9.9× chance at the gentlest setting down to 1.8× at the loudest. See
[`PLAN.md`](PLAN.md) Phase 6 for the full table.

## Layout
```
src/keydropper/
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
  cli.py           # `python -m keydropper.cli {demo,segment,eval-defense,synth,info}`
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
Phases 0–3, the Tier-1 masking defense (Phase 5), and the **red/blue evaluation loop
including the adaptive attacker (Phase 6)** are built, measured, and tested. Real-data
capture, sequence/LM decoding, the learned adversarial masker (Tier-2), and the
real-time defender daemon are designed and scaffolded — see `PLAN.md`.
