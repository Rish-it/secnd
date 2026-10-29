"""Turn raw model probabilities into a calibrated claim: one temperature, then a threshold whose
automatic answers are at least 99% precise with 95% confidence (one-sided Clopper-Pearson).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

TARGET = 0.99
ALPHA = 0.05
LOG_T_BOUNDS = (np.log(0.05), np.log(20))


def scale(p: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log p / T). Classes with p = 0 (never seen in training) stay at 0."""
    with np.errstate(divide="ignore"):
        logits = np.log(p) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    """The T minimising calib NLL, searched on log T in [0.05, 20].

    Rows whose true class has p = 0 are left out: no temperature can move them.
    """
    keep = p[np.arange(len(y)), y] > 0
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return 1.0

    def nll(log_t):
        return -np.log(scale(p, np.exp(log_t))[np.arange(len(y)), y]).mean()

    return float(np.exp(minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded").x))


def choose_threshold(y: np.ndarray, p: np.ndarray, target: float = TARGET, alpha: float = ALPHA):
    """The lowest tau reached by walking down from the most confident answers while the calib
    answers at confidence >= tau keep a CP lower bound >= target; the walk stops at the first tau
    that fails (fixed-sequence testing, so picking among many taus keeps the 95% guarantee).

    The walk starts at the first tau with ceil(log(alpha) / log(target)) answers (299 at 0.99 /
    0.05): above it no tau can pass, so those are not tested. Returns (tau, n_auto, correct,
    lower_bound); tau is None when the first tested tau fails, so nothing is answered.
    """
    confidence = p.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    confidence = confidence[order]
    hits = np.cumsum(p.argmax(axis=1)[order] == y[order])
    n = np.arange(1, len(y) + 1)
    # A threshold must take every row tied at its confidence, so only the last of a tie counts.
    last = np.append(confidence[1:] != confidence[:-1], True)
    lower = np.where(hits > 0, beta.ppf(alpha, hits, n - hits + 1), 0.0)
    tested = np.flatnonzero(last & (n >= np.ceil(np.log(alpha) / np.log(target))))
    failed = lower[tested] < target
    passed = tested[: np.argmax(failed)] if failed.any() else tested
    if len(passed) == 0:
        return None, 0, 0, 0.0
    i = passed[-1]
    return float(confidence[i]), int(n[i]), int(hits[i]), float(lower[i])
