import numpy as np

from backend import calibrate, metrics


def _overconfident(n=4000, k=5, seed=0):
    """True probabilities are softmax(z); the model reports softmax(3 z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, k))
    true = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = (true.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)
    reported = np.exp(3 * z) / np.exp(3 * z).sum(axis=1, keepdims=True)
    return y, reported


def test_temperature_fixes_overconfidence():
    y, p = _overconfident()
    t = calibrate.fit_temperature(y, p)
    assert 2.5 < t < 3.5
    assert metrics.ece(y, calibrate.scale(p, t)) < metrics.ece(y, p) / 3


def test_scale_keeps_unseen_classes_at_zero_and_rows_summing_to_one():
    p = np.array([[0.7, 0.3, 0.0], [0.2, 0.2, 0.6]])
    q = calibrate.scale(p, 2.0)
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert q[0, 2] == 0
    np.testing.assert_allclose(q[0, :2], np.sqrt([0.7, 0.3]) / np.sqrt([0.7, 0.3]).sum())


def _rows(confidences, correct):
    p = np.zeros((len(confidences), 2))
    p[:, 0] = confidences
    p[:, 1] = 1 - np.asarray(confidences)
    y = np.where(correct, 0, 1)
    return y, p


def test_threshold_needs_299_clean_answers():
    y, p = _rows([0.999] * 299 + [0.6] * 50, [True] * 299 + [False] * 50)
    tau, n, correct, lower = calibrate.choose_threshold(y, p)
    assert (tau, n, correct) == (0.999, 299, 299) and lower >= 0.99

    y, p = _rows([0.999] * 298, [True] * 298)
    assert calibrate.choose_threshold(y, p) == (None, 0, 0, 0.0)


def test_threshold_stops_at_the_first_tau_that_fails():
    # Scanning every tau and keeping the lowest that passes tests many hypotheses at once; walking
    # down from the top and stopping at the first failure keeps the 95% guarantee.
    conf = [0.99] * 400 + [0.97] * 20 + [0.95] * 2000
    right = [True] * 400 + [True] * 10 + [False] * 10 + [True] * 2000
    y, p = _rows(conf, right)
    tau, n, correct, _ = calibrate.choose_threshold(y, p)
    assert (tau, n, correct) == (0.99, 400, 400)


def test_threshold_takes_the_lowest_qualifying_tau_and_whole_ties():
    conf = [0.99] * 600 + [0.95] * 100 + [0.9] * 10
    right = [True] * 600 + [True] * 99 + [False] + [False] * 10
    y, p = _rows(conf, right)
    tau, n, correct, _ = calibrate.choose_threshold(y, p)
    # 699/700 still clears 0.99; adding the ten wrong 0.9 rows does not.
    assert (tau, n, correct) == (0.95, 700, 699)
