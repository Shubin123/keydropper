"""Tier-2 countermeasure: a learned adversarial masker (PyTorch).

Where Tier-1 masking hides real keystrokes behind statistically-matched decoys, Tier-2
learns an additive audio perturbation that directly **maximizes the recognizer's
loss** under a loudness/energy budget, so the leaked spectrogram is pushed across the
model's decision boundaries while staying tolerable to the user.

Threat-model discipline (blue team):
  * White-box against *our own* recognizer, but the perturbation generator is trained
    with transfer in mind — validate it against a held-out recognizer (e.g. train on
    ``TinyKeyCNN``, test the drop on ``BCResNetLite``) so we do not merely overfit one
    attacker. A defense that only fools the model it was trained on is not a defense.
  * A budget constraint (L2 / perceptual) keeps the countermeasure usable; we report
    the accuracy drop *at a fixed budget*, not an unbounded one.

This module is a real training skeleton; it needs torch and a trained recognizer. The
Tier-1 masker in ``masker.py`` is what runs with no dependencies today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as Fnn
        return torch, nn, Fnn
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The adversarial masker needs PyTorch (pip install -r requirements-train.txt)."
        ) from exc


@dataclass
class AdversaryConfig:
    epochs: int = 30
    lr: float = 1e-3
    budget: float = 0.05        # max per-sample perturbation amplitude (loudness cap)
    hidden: int = 64
    transfer_check: bool = True


def build_perturbation_generator(clip_len: int, cfg: AdversaryConfig):
    """A tiny 1-D conv generator: clip -> bounded additive perturbation of same length.

    Kept small on purpose — it must run in real time alongside the Tier-1 masker.
    """
    torch, nn, _ = _require_torch()
    h = cfg.hidden

    class PerturbGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, h, 9, padding=4), nn.ReLU(True),
                nn.Conv1d(h, h, 9, padding=4), nn.ReLU(True),
                nn.Conv1d(h, 1, 9, padding=4),
            )
            self.budget = cfg.budget

        def forward(self, clip):  # clip: (B, clip_len)
            x = clip.unsqueeze(1)
            delta = torch.tanh(self.net(x)).squeeze(1)  # (-1, 1)
            return self.budget * delta                  # bounded perturbation

    return PerturbGen()


def train_adversarial_masker(
    recognizer,                       # nn.Module mapping (B, n_mels, n_frames) -> logits
    featurizer: Callable,             # differentiable clip -> log-mel (torch)
    train_clips,                      # (B, clip_len) tensor of raw keystroke clips
    labels,                           # (B,) long tensor of true key ids
    cfg: Optional[AdversaryConfig] = None,
    transfer_recognizer=None,         # held-out model for transfer validation
):
    """Train the perturbation generator to *minimize* recognizer accuracy at a budget.

    Returns the trained generator. The loss is the negative cross-entropy of the
    recognizer on perturbed clips (we want the attacker to be wrong), with the budget
    enforced structurally by the ``tanh`` head.
    """
    torch, nn, Fnn = _require_torch()
    cfg = cfg or AdversaryConfig()
    clip_len = train_clips.shape[-1]
    gen = build_perturbation_generator(clip_len, cfg)
    opt = torch.optim.Adam(gen.parameters(), lr=cfg.lr)

    recognizer.eval()
    for p in recognizer.parameters():
        p.requires_grad_(False)

    for epoch in range(cfg.epochs):
        gen.train()
        opt.zero_grad()
        delta = gen(train_clips)
        adv = train_clips + delta
        feats = featurizer(adv)                     # differentiable front end
        logits = recognizer(feats)
        # Maximize recognizer loss == minimize negative CE.
        loss = -Fnn.cross_entropy(logits, labels)
        loss.backward()
        opt.step()

    if cfg.transfer_check and transfer_recognizer is not None:
        gen.eval()
        with torch.no_grad():
            adv = train_clips + gen(train_clips)
            feats = featurizer(adv)
            acc = (transfer_recognizer(feats).argmax(1) == labels).float().mean().item()
        print(f"[adversarial] transfer attacker accuracy under defense: {acc:.3f}")
    return gen
