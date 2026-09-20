"""Command-line entry point.

Runs the parts that work with zero third-party dependencies, so the plan's early
phases are demonstrable immediately:

    python -m keysoundmap.cli demo            # full toy pipeline + defense in one go
    python -m keysoundmap.cli synth --out D   # write a synthetic labeled dataset
    python -m keysoundmap.cli segment "text"  # show onset detection on a synth stream
    python -m keysoundmap.cli eval-defense     # attacker accuracy with/without masking
    python -m keysoundmap.cli info             # config + environment summary
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from .config import Config
from . import synth, segment as seg_mod, models as M
from .data import build_training_set, label_detected_clips
from .interference.evaluate_defense import run_defense_benchmark, _random_corpus


DEFAULT_KEYS: List[str] = list("abcdefghijklmnopqrstuvwxyz") + ["<space>"]
DEMO_TEXT = "the quick brown fox"


def _cfg_from_args(args) -> Config:
    return Config()


def cmd_info(args) -> int:
    cfg = _cfg_from_args(args)
    have = {}
    for mod in ("numpy", "torch", "sounddevice", "pynput", "soundfile"):
        try:
            __import__(mod)
            have[mod] = True
        except Exception:
            have[mod] = False
    n_frames = M.infer_n_frames(cfg.segment.window_ms, cfg.feature, cfg.audio)
    print("keysoundmap — configuration")
    print(cfg.to_json())
    print(f"\nfeature vector: {n_frames} frames x {cfg.feature.n_mels} mels "
          f"= {n_frames * cfg.feature.n_mels} dims")
    print("optional deps present:", json.dumps(have))
    return 0


def cmd_synth(args) -> int:
    cfg = _cfg_from_args(args)
    keys = DEFAULT_KEYS
    clips, labels = synth.synth_dataset(keys, args.per_key, cfg.audio, cfg.segment.window_ms, seed=cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    # Store as JSON so it works without numpy; small toy sets only.
    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump({"labels": labels, "clips": clips}, f)
    print(f"wrote {len(clips)} clips ({len(set(labels))} classes) to {args.out}/dataset.json")
    return 0


def cmd_segment(args) -> int:
    cfg = _cfg_from_args(args)
    stream, truth = synth.render_stream(args.text, cfg.audio, seed=cfg.seed, window_ms=cfg.segment.window_ms)
    onsets = seg_mod.detect_onsets(stream, cfg.audio, cfg.segment)
    sr = cfg.audio.sample_rate
    tol = int(round(cfg.segment.refractory_ms * 0.5 * sr / 1000.0))
    tp, fp, fn = seg_mod.match_onsets(onsets, [o for o, _ in truth], tol)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"text={args.text!r}  true onsets={len(truth)}  detected={len(onsets)}")
    print(f"onset detection: precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
    return 0


def cmd_eval_defense(args) -> int:
    cfg = _cfg_from_args(args)
    report = run_defense_benchmark(DEFAULT_KEYS, args.text, cfg, per_key_train=args.per_key)
    print(report.summary())
    return 0


def cmd_demo(args) -> int:
    cfg = _cfg_from_args(args)
    print("=== keysoundmap demo (synthetic, dependency-free) ===\n")

    # Phase 2: segmentation quality.
    print("[Phase 2] onset segmentation on a typed stream")
    cmd_segment(argparse.Namespace(text=DEMO_TEXT))

    # Phase 3: recognizer accuracy on a *different* recording (the honest number:
    # train and eval streams pass through the same segmenter, keyboard identity fixed).
    print("\n[Phase 3] small recognizer on an unseen recording (cross-stream)")
    keys = DEFAULT_KEYS
    win = cfg.segment.window_ms
    n_frames = M.infer_n_frames(win, cfg.feature, cfg.audio)
    clips, labels = build_training_set(keys, 40, cfg)
    Xtr = M.featurize_clips(clips, cfg.audio, cfg.feature, n_frames)
    clf = M.KNNClassifier(k=3).fit(Xtr, labels)
    eval_stream, eval_truth = synth.render_stream(
        _random_corpus(keys, 120, cfg.seed + 321), cfg.audio, seed=cfg.seed + 654, window_ms=win
    )
    ec, ey = label_detected_clips(eval_stream, eval_truth, cfg)
    Xe = M.featurize_clips(ec, cfg.audio, cfg.feature, n_frames)
    acc = M.accuracy(clf.predict(Xe), ey)
    print(f"cross-stream per-key accuracy = {acc:.3f} (chance = {1/len(set(labels)):.3f})")

    # Phase 5/6: does the interference model suppress the attacker?
    print("\n[Phase 5/6] interference model vs the attacker")
    report = run_defense_benchmark(keys, DEMO_TEXT, cfg, per_key_train=50)
    print(report.summary())
    print("\nBlue-team read: masking should pull attacker recovery down toward chance.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keysoundmap", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("info", help="print config + environment")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("synth", help="write a synthetic labeled dataset")
    sp.add_argument("--out", default="data/processed/synth")
    sp.add_argument("--per-key", type=int, default=40)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("segment", help="run onset detection on a synth stream")
    sp.add_argument("text", nargs="?", default=DEMO_TEXT)
    sp.set_defaults(func=cmd_segment)

    sp = sub.add_parser("eval-defense", help="attacker accuracy with/without masking")
    sp.add_argument("text", nargs="?", default=DEMO_TEXT)
    sp.add_argument("--per-key", type=int, default=40)
    sp.set_defaults(func=cmd_eval_defense)

    sp = sub.add_parser("demo", help="end-to-end toy pipeline + defense")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
