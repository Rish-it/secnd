import numpy as np

from backend import gate_study as gs


def _calib(correct: bool, n=1200):
    rng = np.random.default_rng(0)
    y = rng.integers(3, size=n)
    p = np.full((n, 3), 0.01)
    guess = y if correct else (y + 1) % 3
    p[np.arange(n), guess] = rng.uniform(0.5, 1.0, n)
    p /= p.sum(1, keepdims=True)
    return y[: n // 2], p[: n // 2], y[n // 2 :], p[n // 2 :]


def test_every_rule_accepts_a_perfect_model_and_refuses_a_wrong_one():
    for rule in gs.RULES.values():
        assert rule(*_calib(True)) is not None
        assert rule(*_calib(False)) is None
