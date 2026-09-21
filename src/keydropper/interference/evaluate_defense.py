"""Red/blue evaluation: how much leakage does the interference model remove?

The metric is **keystroke-recovery accuracy**: of the keys the user actually typed,
what fraction does the attacker both (a) segment out and (b) label correctly. We
report it with and without the defense; the drop is the leakage reduction. Chance
level is ``1 / n_classes`` (plus the fact that decoys add insertions the attacker must
also survive).

This harness deliberately gives the attacker a fair shot: the recognizer is trained on
clean data, and the defense does not touch the recognizer. Phase 6 extends it with an
*adaptive* attacker retrained on defended audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config import Config
from .. import segment as seg_mod
from .. import models as M
from .. import synth
from ..data import build_training_set
from .masker import DecoyBank, apply_masking


@dataclass
class DefenseReport:
    n_true_keys: int
    n_classes: int
    recovery_clean: float       # attacker accuracy with no defense
    recovery_defended: float    # attacker accuracy under masking
    chance: float               # 1 / n_classes
    insertions_clean: int
    insertions_defended: int

    @property
    def leakage_reduction(self) -> float:
        """Absolute drop in attacker recovery accuracy caused by the defense."""
        return self.recovery_clean - self.recovery_defended

    def summary(self) -> str:
        return (
            f"true keys={self.n_true_keys}  classes={self.n_classes}  "
            f"chance={self.chance:.3f}\n"
            f"attacker recovery  no-defense={self.recovery_clean:.3f}  "
            f"defended={self.recovery_defended:.3f}  "
            f"(leakage reduction={self.leakage_reduction:+.3f})\n"
            f"decoy-driven insertions  no-defense={self.insertions_clean}  "
            f"defended={self.insertions_defended}"
        )


def keystroke_recovery(
    stream: Sequence[float],
    truth_events: Sequence[Tuple[int, str]],
    recognizer,
    cfg: Config,
    n_frames: int,
) -> Tuple[float, int]:
    """Run the full attacker pipeline on ``stream`` and score it against ground truth.

    Returns ``(recovery_accuracy, insertions)`` where a true key counts as recovered
    only if a detected onset lands within tolerance *and* its predicted label matches.
    """
    onsets, clips = seg_mod.segment(stream, cfg.audio, cfg.segment)
    if not onsets:
        return 0.0, 0
    feats = M.featurize_clips(clips, cfg.audio, cfg.feature, n_frames)
    preds = recognizer.predict(feats)

    sr = cfg.audio.sample_rate
    tol = int(round(cfg.segment.refractory_ms * 0.5 * sr / 1000.0))

    detected = sorted(zip(onsets, preds), key=lambda p: p[0])
    used = [False] * len(detected)
    correct = 0
    for t_onset, t_key in truth_events:
        best = -1
        best_dist = tol + 1
        for i, (d_onset, _pred) in enumerate(detected):
            if used[i]:
                continue
            dist = abs(d_onset - t_onset)
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best = i
        if best >= 0:
            used[best] = True
            if detected[best][1] == t_key:
                correct += 1
    insertions = sum(1 for u in used if not u)
    recovery = correct / len(truth_events) if truth_events else 0.0
    return recovery, insertions


def _random_corpus(keys: Sequence[str], length: int, seed: int) -> str:
    """Deterministic in-vocabulary eval text (random keys, so no language prior helps)."""
    import random

    rng = random.Random(seed)
    chars = [(" " if k == "<space>" else k) for k in keys]
    return "".join(rng.choice(chars) for _ in range(length))


def run_defense_benchmark(
    keys: Sequence[str],
    text: str,
    cfg: Config,
    per_key_train: int = 40,
    eval_len: int = 200,
) -> DefenseReport:
    """End-to-end, dependency-free benchmark of the Tier-1 masking defense.

    1. Build a segmentation-matched clean training set and fit the attacker.
    2. Render a clean eval stream (a long random in-vocab corpus for a stable estimate;
       ``text`` seeds it when it is already long enough).
    3. Score the attacker on the clean stream (leakage upper bound).
    4. Apply masking using the defender's own key events, score again.
    """
    window_ms = cfg.segment.window_ms
    n_frames = M.infer_n_frames(window_ms, cfg.feature, cfg.audio)

    # 1. Train the attacker on segmentation-matched clean data (same front end it will
    #    see at inference: streams -> onset detector -> clips). This is what makes the
    #    "no defense" recovery meaningful instead of collapsing on an alignment mismatch.
    clips, labels = build_training_set(keys, per_key_train, cfg)
    X = M.featurize_clips(clips, cfg.audio, cfg.feature, n_frames)
    recognizer = M.KNNClassifier(k=3).fit(X, labels)

    # 2. Clean eval stream + ground truth over a long random corpus.
    eval_text = text if len(text) >= eval_len else _random_corpus(keys, eval_len, cfg.seed + 11)
    stream, truth = synth.render_stream(eval_text, cfg.audio, seed=cfg.seed + 7, window_ms=window_ms)

    # 3. Attacker vs no defense.
    rec_clean, ins_clean = keystroke_recovery(stream, truth, recognizer, cfg, n_frames)

    # 4. Attacker vs masking defense (defender observes its own key events == truth).
    bank = DecoyBank.synthetic(keys, cfg.audio, window_ms, seed=cfg.seed + 1)
    defended = apply_masking(stream, truth, cfg.audio, cfg.model, bank, seed=cfg.seed + 3)
    rec_def, ins_def = keystroke_recovery(defended, truth, recognizer, cfg, n_frames)

    return DefenseReport(
        n_true_keys=len(truth),
        n_classes=len(set(labels)),
        recovery_clean=rec_clean,
        recovery_defended=rec_def,
        chance=1.0 / max(1, len(set(labels))),
        insertions_clean=ins_clean,
        insertions_defended=ins_def,
    )
