import math

import numpy as np
import pytest

from backend import metrics

Y = np.array([0, 1, 2, 3])
PERFECT = np.eye(4)
UNIFORM = np.full((4, 4), 0.25)


def test_perfect_model():
    assert metrics.accuracy(Y, PERFECT) == 1
    assert metrics.macro_f1(Y, PERFECT) == 1
    assert metrics.ece(Y, PERFECT) == 0
    assert metrics.classwise_ece(Y, PERFECT) == 0
    assert metrics.brier(Y, PERFECT) == 0
    assert metrics.nll(Y, PERFECT) == pytest.approx(0, abs=1e-9)
    assert metrics.aurc(Y, PERFECT) == 0


def test_uniform_model():
    # Brier: (1 - 1/4)^2 + 3 * (1/4)^2 = 0.75. Argmax picks class 0, right 1 time in 4 = conf.
    assert metrics.brier(Y, UNIFORM) == pytest.approx(0.75)
    assert metrics.nll(Y, UNIFORM) == pytest.approx(math.log(4))
    assert metrics.accuracy(Y, UNIFORM) == 0.25
    assert metrics.ece(Y, UNIFORM) == pytest.approx(0)


def test_overconfident_model_ece():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1]] * 4)
    assert metrics.ece(y, p) == pytest.approx(0.4)
    # Class 0: forecast 0.9, happens half the time; class 1: forecast 0.1, same: both gaps 0.4.
    assert metrics.classwise_ece(y, p) == pytest.approx(0.4)


def test_selective_metrics():
    y = np.array([0, 0, 0, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    # Most confident first: right, wrong, right, wrong -> risks 0, 1/2, 1/3, 2/4.
    assert metrics.aurc(y, p) == pytest.approx((0 + 1 / 2 + 1 / 3 + 2 / 4) / 4)
    assert metrics.coverage_at(p, 0.75) == 0.5
    assert metrics.precision_above(y, p, 0.75) == 0.5
    assert metrics.precision_above(y, p, 0.7) == pytest.approx(2 / 3)
    assert math.isnan(metrics.precision_above(y, p, 0.95))


def test_clopper_pearson_lower():
    # With k = n the one-sided bound is alpha ** (1 / n): 300 of 300 clears 0.99.
    assert metrics.clopper_pearson_lower(300, 300) == pytest.approx(0.05 ** (1 / 300))
    assert metrics.clopper_pearson_lower(300, 300) > 0.99
    assert metrics.clopper_pearson_lower(299, 300) < 0.99
    assert metrics.clopper_pearson_lower(0, 10) == 0
