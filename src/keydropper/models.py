"""Keystroke recognizers.

Two things live here, matching Phase 3 of the plan:

* ``PrototypeClassifier`` — a dependency-free nearest-centroid baseline. It lets the
  whole pipeline (segment -> features -> classify -> evaluate defense) run and report
  a real accuracy number with only the standard library. It is our smoke test and the
  first attacker the defense is measured against.

* ``TinyKeyCNN`` / ``BCResNetLite`` — the small, efficient deep models (PyTorch,
  imported lazily). These are the "state of the art, but small" recognizers: a
  depthwise-separable CNN in the DS-CNN / MobileNet / keyword-spotting family, sized
  for real-time on-device inference and INT8 quantization, rather than a heavy
  CoAtNet/ResNet-34. They are the honest yardstick the countermeasure must beat.

The deep classes raise a clear error if torch is missing, but importing this module
never requires torch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .config import AudioConfig, FeatureConfig, ModelConfig
from . import features as F

try:  # optional acceleration for the k-NN attacker
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    _np = None


# --------------------------------------------------------------------------------------
# Featurization shared by all recognizers
# --------------------------------------------------------------------------------------

def featurize_clips(
    clips: Sequence[Sequence[float]],
    audio: AudioConfig,
    feat: FeatureConfig,
    n_frames: int,
) -> List[List[float]]:
    """Turn raw clips into fixed-length flattened log-mel vectors.

    Uses the numpy fast path automatically when numpy is importable (much faster for
    the dev loop) and the reference pure-Python path otherwise; the two are checked
    against each other in the tests.
    """
    use_np = F._np is not None
    target = max(1, n_frames // 4)  # place each transient a quarter into the window
    out: List[List[float]] = []
    for clip in clips:
        if use_np:
            mat = F.logmel_np(clip, audio, feat).tolist()  # (frames, n_mels)
        else:
            mat = F.logmel(clip, audio, feat)
        mat = F.align_peak(mat, target, n_frames, feat.n_mels)
        out.append(F.flatten(mat))
    return out


def infer_n_frames(window_ms: float, feat: FeatureConfig, audio: AudioConfig) -> int:
    """Number of STFT frames a ``window_ms`` clip yields — used to size feature vectors."""
    sr = audio.sample_rate
    win = min(max(1, int(round(feat.win_ms * sr / 1000.0))), feat.n_fft)
    hop = max(1, int(round(feat.hop_ms * sr / 1000.0)))
    n = max(1, int(round(window_ms * sr / 1000.0)))
    if n < win:
        return 1
    return 1 + (n - win) // hop


# --------------------------------------------------------------------------------------
# Baseline: nearest-centroid (pure Python)
# --------------------------------------------------------------------------------------

class PrototypeClassifier:
    """Standardize features, then classify by nearest class centroid.

    Simple, fast, deterministic, and dependency-free. Good enough to prove the
    pipeline and to serve as a first (weak) attacker in the defense evaluation.
    """

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self._centroids: Dict[str, List[float]] = {}
        self._mean: List[float] = []
        self._std: List[float] = []
        self._dim = 0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[str]) -> "PrototypeClassifier":
        if not X:
            raise ValueError("empty training set")
        self._dim = len(X[0])
        n = len(X)
        # Feature standardization.
        self._mean = [0.0] * self._dim
        for row in X:
            for j in range(self._dim):
                self._mean[j] += row[j]
        self._mean = [m / n for m in self._mean]
        var = [0.0] * self._dim
        for row in X:
            for j in range(self._dim):
                d = row[j] - self._mean[j]
                var[j] += d * d
        self._std = [math.sqrt(v / n) or 1.0 for v in var]

        sums: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for row, label in zip(X, y):
            z = self._standardize(row)
            if label not in sums:
                sums[label] = [0.0] * self._dim
                counts[label] = 0
            acc = sums[label]
            for j in range(self._dim):
                acc[j] += z[j]
            counts[label] += 1
        self._centroids = {
            k: [s / counts[k] for s in vec] for k, vec in sums.items()
        }
        self.classes_ = sorted(self._centroids)
        return self

    def _standardize(self, row: Sequence[float]) -> List[float]:
        return [(row[j] - self._mean[j]) / self._std[j] for j in range(self._dim)]

    def _distances(self, row: Sequence[float]) -> Dict[str, float]:
        z = self._standardize(row)
        out: Dict[str, float] = {}
        for k, c in self._centroids.items():
            s = 0.0
            for j in range(self._dim):
                d = z[j] - c[j]
                s += d * d
            out[k] = s
        return out

    def predict(self, X: Sequence[Sequence[float]]) -> List[str]:
        preds: List[str] = []
        for row in X:
            dists = self._distances(row)
            preds.append(min(dists, key=dists.get))
        return preds

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[Dict[str, float]]:
        """Softmax over negative distances — a confidence proxy for ROC/decoding."""
        out: List[Dict[str, float]] = []
        for row in X:
            dists = self._distances(row)
            m = min(dists.values())
            exps = {k: math.exp(-(v - m)) for k, v in dists.items()}
            z = sum(exps.values()) or 1.0
            out.append({k: e / z for k, e in exps.items()})
        return out


def accuracy(pred: Sequence[str], true: Sequence[str]) -> float:
    if not true:
        return 0.0
    return sum(1 for p, t in zip(pred, true) if p == t) / len(true)


class KNNClassifier:
    """Standardized k-nearest-neighbours — a stronger attacker than the centroid.

    Uses numpy (vectorized, memory-safe ``||a-b||^2 = a^2 + b^2 - 2ab`` in test
    batches) when available, and a pure-Python fallback otherwise. This is the default
    attacker in the defense benchmark because a *strong* clean attacker is what makes a
    drop under the countermeasure meaningful.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.classes_: List[str] = []
        self._Xtr = None            # numpy array or list-of-lists
        self._ytr: List[str] = []
        self._mean = None
        self._std = None
        self._dim = 0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[str]) -> "KNNClassifier":
        if not X:
            raise ValueError("empty training set")
        self._dim = len(X[0])
        self._ytr = list(y)
        self.classes_ = sorted(set(y))
        if _np is not None:
            arr = _np.asarray(X, dtype=_np.float64)
            self._mean = arr.mean(axis=0)
            self._std = arr.std(axis=0) + 1e-8
            self._Xtr = (arr - self._mean) / self._std
        else:  # pragma: no cover - exercised only without numpy
            n = len(X)
            self._mean = [sum(col) / n for col in zip(*X)]
            var = [sum((row[j] - self._mean[j]) ** 2 for row in X) / n for j in range(self._dim)]
            self._std = [math.sqrt(v) + 1e-8 for v in var]
            self._Xtr = [[(row[j] - self._mean[j]) / self._std[j] for j in range(self._dim)] for row in X]
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[str]:
        if _np is not None:
            Xa = (_np.asarray(X, dtype=_np.float64) - self._mean) / self._std
            tr = self._Xtr
            tr_sq = (tr * tr).sum(axis=1)                # (n_train,)
            out: List[str] = []
            batch = 128
            ytr = self._ytr
            for s in range(0, len(Xa), batch):
                chunk = Xa[s : s + batch]
                d = (chunk * chunk).sum(axis=1)[:, None] + tr_sq[None, :] - 2.0 * chunk @ tr.T
                nn = _np.argpartition(d, min(self.k, d.shape[1] - 1), axis=1)[:, : self.k]
                for row in nn:
                    votes = [ytr[i] for i in row]
                    out.append(max(set(votes), key=votes.count))
            return out
        # pure-Python fallback
        out = []
        for row in X:
            z = [(row[j] - self._mean[j]) / self._std[j] for j in range(self._dim)]
            dists = []
            for i, tr in enumerate(self._Xtr):
                s = 0.0
                for j in range(self._dim):
                    dd = z[j] - tr[j]
                    s += dd * dd
                dists.append((s, self._ytr[i]))
            dists.sort(key=lambda p: p[0])
            votes = [lbl for _, lbl in dists[: self.k]]
            out.append(max(set(votes), key=votes.count))
        return out


# --------------------------------------------------------------------------------------
# Deep, small, efficient models (PyTorch, lazily imported)
# --------------------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        return torch, nn
    except Exception as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "TinyKeyCNN/BCResNetLite need PyTorch. Install the training extras:\n"
            "    python -m venv .venv && . .venv/bin/activate\n"
            "    pip install -r requirements-train.txt"
        ) from exc


def build_tiny_key_cnn(n_classes: int, n_mels: int, cfg: ModelConfig):
    """Depthwise-separable CNN over a (1, n_mels, n_frames) log-mel image.

    ~100-150K params depending on ``width``; designed to quantize to INT8 and run in
    well under ~10 ms/keystroke on a laptop CPU. Returns an ``nn.Module``.
    """
    torch, nn = _require_torch()
    w = cfg.width

    class DSConv(nn.Module):
        """Depthwise separable conv block = depthwise + pointwise (MobileNet/DS-CNN)."""

        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
            self.bn1 = nn.BatchNorm2d(cin)
            self.pw = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.act(self.bn1(self.dw(x)))
            x = self.act(self.bn2(self.pw(x)))
            return x

    class TinyKeyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, w, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                DSConv(w, w, stride=2),
                DSConv(w, w * 2, stride=2),
                DSConv(w * 2, w * 2, stride=1),
                DSConv(w * 2, w * 4, stride=2),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(cfg.dropout)
            self.fc = nn.Linear(w * 4, n_classes)

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)  # (B, n_mels, n_frames) -> (B, 1, n_mels, n_frames)
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x).flatten(1)
            x = self.drop(x)
            return self.fc(x)

    return TinyKeyCNN()


def build_bcresnet_lite(n_classes: int, n_mels: int, cfg: ModelConfig):
    """A lightweight broadcasted-residual (BC-ResNet-style) alternative recognizer.

    BC-ResNet factorizes 2-D convolutions into a 1-D frequency conv plus a broadcasted
    1-D temporal conv, which is even more parameter-efficient than DS-CNN for audio.
    Provided as a second attacker so the defense is not tuned to one architecture.
    """
    torch, nn = _require_torch()
    w = cfg.width

    class BCBlock(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.freq = nn.Conv2d(cin, cout, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
            self.bnf = nn.BatchNorm2d(cout)
            self.temp = nn.Conv2d(cout, cout, (1, 3), padding=(0, 1), groups=cout, bias=False)
            self.bnt = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)
            self.proj = None
            if cin != cout or stride != 1:
                self.proj = nn.Conv2d(cin, cout, 1, stride=(stride, 1), bias=False)

        def forward(self, x):
            idn = x if self.proj is None else self.proj(x)
            f = self.act(self.bnf(self.freq(x)))
            # Broadcast temporal features across the frequency axis (mean over freq).
            t = self.bnt(self.temp(f))
            t = t + f.mean(dim=2, keepdim=True)
            return self.act(t + idn)

    class BCResNetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True))
            self.blocks = nn.Sequential(
                BCBlock(w, w, stride=2),
                BCBlock(w, w * 2, stride=2),
                BCBlock(w * 2, w * 2, stride=1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(w * 2, n_classes)

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self.stem(x)
            x = self.blocks(x)
            return self.fc(self.pool(x).flatten(1))

    return BCResNetLite()


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())
