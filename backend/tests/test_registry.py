import numpy as np
import pytest

from backend import registry


def _p(predicted, confidence=None, k=3):
    """One-hot-ish rows; confidence varies across rows unless it is given."""
    if confidence is None:
        confidence = np.linspace(0.8, 0.99, len(predicted))
    confidence = np.broadcast_to(confidence, len(predicted))
    p = np.repeat(((1 - confidence) / (k - 1))[:, None], k, axis=1)
    p[np.arange(len(predicted)), predicted] = confidence
    return p


def test_gate_passes_a_sound_candidate_and_names_every_failure():
    y = np.array([0, 1, 2, 0, 1, 2] * 20)
    good = _p(y)
    assert registry.gate(y, good, good, tau=0.95, calib_lower=0.991) == []

    worse = y.copy()
    worse[:6] = (worse[:6] + 1) % 3  # 5% fewer right
    reasons = registry.gate(y, _p(np.zeros_like(y)), _p(worse), tau=0.95, calib_lower=0.98)
    assert len(reasons) == 3 and "below" in reasons[0] and "collapsed" in reasons[1]
    assert "regression" in reasons[2]
    assert "regression" in registry.gate(y, _p(worse), good)[0]


def test_gate_refuses_a_candidate_that_certifies_less():
    y = np.array([0, 1, 2, 0, 1, 2] * 20)
    good = _p(y)
    assert registry.gate(y, good, good, tau=0.95, calib_lower=0.991, current_tau=0.95) == []
    reasons = registry.gate(y, good, good, tau=None, current_tau=0.95)
    assert len(reasons) == 1 and reasons[0].startswith("certified coverage drops 0.2")


def test_gate_flags_flat_confidence_as_collapse():
    assert registry.collapsed(_p(np.array([0, 1, 2]), confidence=0.5))
    p = _p(np.array([0, 1, 2]))
    p[0] = [0.6, 0.2, 0.2]
    assert not registry.collapsed(p)


def test_registry_saves_numbered_versions_and_rolls_back(tmp_path):
    models = registry.Registry(tmp_path, "default")
    assert models.versions() == [] and models.current() is None

    first = models.save({"weights": 1}, {"coverage": 0.1})
    second = models.save({"weights": 2}, {"coverage": 0.2})
    assert (first, second) == ("ren-1.0", "ren-1.1")
    assert models.card(second) == {"coverage": 0.2, "model_version": "ren-1.1"}

    models.promote(second)
    assert models.current() == second and models.load(second) == {"weights": 2}
    models.promote(first)
    assert models.current() == first
    assert not (models.root / "current.tmp").exists()
    with pytest.raises(KeyError):
        models.promote("ren-1.9")
