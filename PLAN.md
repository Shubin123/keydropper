# keydropper — Multi-Phase Plan

**Keyboards make noise. That noise can be mapped to keystrokes.**
This project studies the acoustic side channel of keyboards **as a blue team**: we
reproduce the state-of-the-art attack in a controlled way *in order to defeat it*.
The headline deliverable is the **interference (countermeasure) model** — an active
defense that collapses an eavesdropper's transcription accuracy toward random
guessing while staying tolerable for the user.

> **Scope & ethics.** All work is defensive. The recognizer we build is a *threat
> model / measuring stick* used to quantify leakage and to prove the countermeasure
> works. Everything runs on hardware and recordings we own, with no deployment
> against third parties. See `docs/THREAT_MODEL.md`.

---

## The problem in one picture

```
                 ┌────────────── ATTACKER (we simulate this to measure risk) ─────────────┐
  key press  →   │  mic → onset segmentation → log-mel features → small CNN → per-key      │
  (ground truth) │  posterior → (optional) language-model decoding → recovered text        │
                 └───────────────────────────────────────────────────────────────────────┘
                                         ▲
                                         │ acoustic emanation (the leak)
  ┌──────────── DEFENDER (our real product) ─────────────┐
  │  keypress event → interference model → masking /      │  ← raises attacker error
  │  adversarial audio emitted from the user's own device │     toward chance
  └───────────────────────────────────────────────────────┘
```

Two models, one repo:
1. **Recognizer** (the "red" measuring stick): small, efficient, on-device.
2. **Interference model** (the "blue" product): active acoustic countermeasure.

---

## State of the art we build on

**Attack side (what we must defeat):**
- Asonov & Agrawal 2004; Zhuang, Zhou & Tygar 2005 — keyboards leak because each key
  has a distinct press/release acoustic signature; unsupervised + HMM/language priors.
- Harrison, Toreini & Mehrnezhad 2023, *A Practical Deep Learning-Based Acoustic Side
  Channel Attack on Keyboards* — phone/Zoom recordings, log-mel + **CoAtNet**, ~95%.
- 2025 transformer + LLM decoding, and LLM "typo-correction" over noisy spectrograms,
  pushing noisy/VoIP viability up sharply.
- **ASCA-Net** 2026 — dynamic spectral gating + ResNet-34 + semantic post-correction,
  reported ~99.9% clean / ~97% VoIP, ~13 ms latency. This is the bar our defense is
  measured against.

**Defense side (what we extend):**
- Anand et al. 2016, *A Sound for a Sound* — masking with **real keystroke sounds**
  beats white noise, because it matches the signal's own statistics.
- *Keyboard Emanations in Remote Voice Calls: Noise(less) Masking Defenses*.
- SoK: Acoustic Side Channels (2023) — taxonomy of channels and mitigations.
- Emerging: **GAN / adversarial** masking that is spectrally matched and, ideally,
  optimized directly against a recognizer.

**Small-model design (our efficiency requirement):** keyword-spotting lineage —
**DS-CNN** (depthwise-separable CNN, MLPerf Tiny), **BC-ResNet** (broadcasted
residual), MobileNetV3-Small — all operate on log-mel spectrograms, run < a few
hundred K params, and quantize to INT8 for real-time on-device inference. We target
a **TinyKeyCNN** in that class rather than a heavy CoAtNet/ResNet-34, because the
defender must run in real time on a laptop *and* because a lean, honest recognizer is
a fair, reproducible yardstick.

Sources are listed at the bottom of this file.

---

## Phase 0 — Foundation & scaffolding  ✅ *built in this pass*
**Goal:** a runnable skeleton with zero mandatory heavy dependencies.
- Package layout `src/keydropper/*`, config dataclasses, CLI, tests, docs.
- Pure-Python DSP (framing, Hann window, radix-2 FFT, power spectrum) so the pipeline
  runs anywhere; optional numpy/torch fast paths documented for production.
- **Synthetic keystroke generator** (`synth.py`): physically-motivated per-key click
  signatures (touch peak + hit peak + key-specific resonances + noise) → a labeled
  toy dataset, so segmentation/features/recognizer/defense can be exercised in CI
  without a microphone or real data.
**Exit criteria:** `python -m keydropper.cli demo` runs end-to-end; `tests/` pass on
stdlib alone.

## Phase 1 — Data acquisition  ✅ *core built (synthetic) + real-capture design*
**Goal:** labeled `(audio window → key)` pairs.
- **Synthetic** dataset for development/CI — done.
- **Real capture harness** (`capture.py`): record mic audio while logging keypress
  events with timestamps (via `pynput`), producing time-aligned labels. Protocol in
  `docs/DATA_COLLECTION.md`: fixed keyboard/mic/distance, prompted text + random
  strings (so language priors don't inflate scores), multiple sessions for
  train/val/test splits *by session* (never leak the same recording across splits).
**Exit criteria:** ≥ a few thousand labeled keystrokes per keyboard, held-out session.

## Phase 2 — Signal processing & segmentation  ✅ *built*
**Goal:** turn a continuous stream into per-keystroke, fixed-size feature tensors.
- **Onset segmentation** (`segment.py`): short-time energy envelope → adaptive
  threshold → onset picking with refractory period; extract a fixed window around
  each hit (captures both press transients). Pure Python, tested against synthetic
  ground-truth onsets.
- **Features** (`features.py`): log-mel spectrogram per keystroke window → fixed
  `(n_mels × n_frames)` tensor. Mel filterbank + framing implemented on the pure DSP
  core, numpy fast-path optional.
**Exit criteria:** onset detector recovers synthetic onsets at high F1; feature shapes
are deterministic and model-ready.

## Phase 3 — Recognizer: the small, efficient model  ◐ *baseline built, deep model coded*
**Goal:** a compact keystroke classifier — our threat yardstick.
- **Runnable-now baseline** (`models.py::PrototypeClassifier`): nearest-centroid /
  k-NN over log-mel features, pure Python. Establishes the pipeline and a first
  accuracy number with no torch.
- **Deep model** (`models.py::TinyKeyCNN`, PyTorch, guarded import): depthwise-
  separable CNN (~<150K params), log-mel input, INT8-quantization-ready. Optional
  `BCResNetLite` variant. Training loop, early stopping, per-key confusion export.
- Efficiency budget documented (params, MACs, latency target < ~10 ms/keystroke).
**Exit criteria:** deep model beats baseline on held-out session; profiled to fit the
real-time budget; exported to ONNX/TFLite.

## Phase 4 — Sequence / language decoding (attacker realism)  ○ *designed*
**Goal:** measure *worst-case* leakage the way a real attacker would.
- Beam search over per-key posteriors + n-gram / small LM prior; optional LLM
  typo-correction pass (as in 2025 work) to model the strongest adversary.
- Report both **raw per-key accuracy** (defense target) and **decoded text accuracy**.
**Exit criteria:** an attacker score we can drive down in Phase 5–6.

## Phase 5 — Interference model: active countermeasure  ◐ *masking built, adversarial coded*
**Goal:** the blue-team product — emit sound from the user's own device that destroys
the attacker's channel.
- **Tier 1 — spectrally-matched masking** (`interference/masker.py`, built): on each
  detected/known keypress, emit a *decoy keystroke* burst (random key signature,
  jittered timing/gain). Matches signal statistics (*A Sound for a Sound*), so the
  attacker cannot separate real from decoy by spectrum alone. Runs pure-Python;
  real-time playback via `sounddevice` when present.
- **Tier 2 — adversarial masking** (`interference/adversarial.py`, coded): a small
  generator trained to add minimally-intrusive perturbations that **maximize the
  recognizer's loss** (white-box against our own TinyKeyCNN; transfer-tested to held-
  out recognizers to avoid overfitting one attacker). Perceptual/energy budget
  constrains audibility.
- Practicality knobs: latency, loudness, user-tolerability, and a "known-keypress"
  mode (defender sees its own key events, so it can mask *proactively*, an advantage
  the attacker lacks).
**Exit criteria:** attacker per-key accuracy under defense drops toward chance
(1/#keys) at an acceptable loudness.

## Phase 6 — Defense evaluation & red/blue loop  ◐ *harness built*
**Goal:** prove it, adaptively.
- `interference/evaluate_defense.py`: accuracy **with vs without** masking →
  *leakage-reduction* metric; ROC of attacker confidence.
- **Adaptive attacker:** retrain the recognizer *on masked audio* (attacker adapts) →
  re-measure. Iterate defense vs adaptive attacker until the defense holds against an
  attacker that knows the defense exists (Kerckhoffs for countermeasures).
**Exit criteria:** defense still suppresses accuracy against an attacker trained on
defended audio.

## Phase 7 — Real-time defender app & packaging  ○ *designed*
**Goal:** a background daemon: hook key events → emit masking with < a few-ms latency;
config for loudness/aggressiveness; on-device, no audio leaves the machine.
**Exit criteria:** runs continuously on the target laptop within CPU/loudness budget.

## Phase 8 — Reporting  ○ *designed*
Reproducible benchmark: leakage-reduction table across masking tiers and adaptive
attackers, efficiency profile of the small model, and defender-usability notes.

Legend: ✅ built · ◐ partially built (runnable core + coded deep parts) · ○ designed.

---

## What runs today (Phases 0–3 + 5/6 cores)
```
python -m keydropper.cli demo        # synth → segment → features → classify → defend
python -m keydropper.cli synth ...    # write a synthetic labeled dataset
python -m keydropper.cli eval-defense # attacker accuracy with vs without masking
python -m pytest -q                    # (stdlib unittest also works: python -m unittest)
```

## Sources
- Harrison, Toreini, Mehrnezhad 2023 — arXiv:2308.01074
- SoK: Acoustic Side Channels 2023 — arXiv:2308.03806
- Transformers + LLMs for ASCA 2025 — arXiv:2502.09782
- LLM typo-correction for noisy ASCA 2025 — arXiv:2504.11622
- ASCA-Net 2026 — Springer, J. Computer Virology & Hacking Techniques
- Anand et al., *A Sound for a Sound* 2016 — FC'16
- Zhuang, Zhou, Tygar 2005 — *Keyboard Acoustic Emanations Revisited*
- Keyword-spotting small models: DS-CNN (MLPerf Tiny), BC-ResNet, MobileNetV3-Small
