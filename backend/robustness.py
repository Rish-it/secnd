"""The headline check: does the 99% claim survive re-splitting calib and test 2,000 times?

Each split shuffles the held-out pool (calib + test rows) into two halves, refits the temperature
and threshold on one half and measures precision above the threshold on the other. A split
passes when that precision is >= 0.99 or nothing is answered automatically.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backend import calibrate


def one_split(y: np.ndarray, p: np.ndarray, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(len(y))
    calib, test = order[: len(y) // 2], order[len(y) // 2 :]
    temperature = calibrate.fit_temperature(y[calib], p[calib])
    tau, n_auto, _, lower = calibrate.choose_threshold(
        y[calib], calibrate.scale(p[calib], temperature)
    )
    q = calibrate.scale(p[test], temperature)
    auto = q.max(axis=1) >= tau if tau is not None else np.zeros(len(test), bool)
    precision = (q[auto].argmax(axis=1) == y[test][auto]).mean() if auto.any() else np.nan
    return {
        "seed": seed,
        "temperature": temperature,
        "tau": tau,
        "calib_auto": n_auto,
        "calib_lower": lower,
        "coverage": auto.mean(),
        "precision": precision,
        "passed": bool(not auto.any() or precision >= calibrate.TARGET),
    }


def robustness(y: np.ndarray, p: np.ndarray, splits: int = 2000, n_jobs: int = -1):
    """Per-split results and the summary: pass rate, mean coverage, worst 1% precision."""
    runs = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(delayed(one_split)(y, p, seed) for seed in range(splits))
    )
    summary = {
        "splits": splits,
        "pass_rate": runs["passed"].mean(),
        "mean_coverage": runs["coverage"].mean(),
        "worst_1pct_precision": runs["precision"].quantile(0.01),
    }
    return runs, summary
