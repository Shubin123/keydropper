# Threat model & ethics

## What the threat is
A microphone near a keyboard captures each keystroke's sound. Because every key has a
slightly different acoustic signature, a machine-learning model can map that sound back
to the key that was pressed and reconstruct what someone typed — including passwords,
messages, and other secrets. Recent work shows this is practical even over a laptop mic
or a video call (see `PLAN.md` sources), with per-key accuracy well above 90%.

## Who this project is for (blue team)
The goal is **defense**: to measure how much a given setup leaks and to build an active
countermeasure that stops it. Concretely:

- The **recognizer** in this repo is a *measuring instrument*. We build a small,
  efficient eavesdropper so we can quantify leakage and have a concrete adversary to
  test the defense against. It is deliberately honest and reproducible, not a weapon.
- The **interference model** is the product: it emits sound from the user's own device
  that drives the eavesdropper's accuracy down toward random guessing.

## Rules we hold ourselves to
- **Own hardware only.** Capture (`capture.py`) records *your* microphone and logs
  *your* keypresses, on a machine you control, to measure *your* leakage.
- **No targeting others.** Nothing here is for recording or transcribing other people's
  typing. The recognizer exists to be defeated by the defense.
- **Kerckhoffs for defenses.** A countermeasure that only works because the attacker
  doesn't know about it is not a real defense. Phase 6 explicitly re-tests the defense
  against an attacker retrained on defended audio.
- **Report honestly.** Metrics are reported with and without the defense, at a fixed
  loudness/energy budget, against a fair (clean-trained) attacker.

## Why publishing the attack side is acceptable here
The attack techniques are already public in the peer-reviewed literature. Re-implementing
a compact version in the open, purely to build and validate a countermeasure, follows
standard defensive-security practice: you cannot prove a defense works without a concrete
adversary to test it against.
