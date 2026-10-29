import numpy as np

from backend import robustness


def _calibrated(n=6000, k=4, seed=0):
    """Half the rows are near-certain, half are coin flips; labels follow the reported p."""
    rng = np.random.default_rng(seed)
    sure = rng.random(n) < 0.5
    p = np.full((n, k), 0.0005 / (k - 1))
    p[:, 0] = 0.9995
    p[~sure] = 1 / k
    p = np.take_along_axis(p, rng.permuted(np.tile(np.arange(k), (n, 1)), axis=1), axis=1)
    y = (p.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)
    return y, p


def test_a_calibrated_model_passes_and_covers_its_sure_half():
    y, p = _calibrated()
    runs, summary = robustness.robustness(y, p, splits=40, n_jobs=1)
    assert len(runs) == 40
    assert summary["pass_rate"] >= 0.9
    assert 0.4 < summary["mean_coverage"] < 0.55


def test_noise_answers_nothing_and_still_passes():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), 2000)
    y = rng.integers(0, 4, 2000)
    runs, summary = robustness.robustness(y, p, splits=10, n_jobs=1)
    assert summary["pass_rate"] == 1 and summary["mean_coverage"] == 0
    assert runs["tau"].isna().all()
