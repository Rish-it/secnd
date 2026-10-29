"""Evaluation metrics. `y` holds class indices, `p` an (n, k) array of probabilities."""

import numpy as np
from scipy.stats import beta
from sklearn.metrics import f1_score

EPS = 1e-12


def accuracy(y, p) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def macro_f1(y, p) -> float:
    return float(f1_score(y, np.argmax(p, axis=1), average="macro", zero_division=0))


def nll(y, p) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def brier(y, p) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _ece(confidence, hit, bins: int) -> float:
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    gap = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            gap += mask.mean() * abs(hit[mask].mean() - confidence[mask].mean())
    return float(gap)


def ece(y, p, bins: int = 10) -> float:
    """Top-label expected calibration error over equal-width bins."""
    return _ece(p.max(axis=1), np.argmax(p, axis=1) == y, bins)


def classwise_ece(y, p, bins: int = 10) -> float:
    """Mean over classes of the ECE of p[:, c] as a forecast of y == c."""
    return float(np.mean([_ece(p[:, c], y == c, bins) for c in range(p.shape[1])]))


def aurc(y, p) -> float:
    """Area under the risk-coverage curve: mean error rate when keeping the k most confident."""
    order = np.argsort(-p.max(axis=1), kind="stable")
    errors = (np.argmax(p, axis=1) != y)[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(y) + 1)))


def coverage_at(p, tau: float) -> float:
    return float(np.mean(p.max(axis=1) >= tau))


def precision_above(y, p, tau: float) -> float:
    """Accuracy on the items answered automatically at threshold tau; nan when none are."""
    kept = p.max(axis=1) >= tau
    return float(np.mean(np.argmax(p[kept], axis=1) == y[kept])) if kept.any() else float("nan")


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact lower bound on a success rate after k successes in n trials."""
    return float(beta.ppf(alpha, k, n - k + 1)) if k > 0 else 0.0
