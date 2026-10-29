"""Compare gate rules on saved reader predictions, on CPU. The plan, fixed before any result was
seen, is docs/results/s2/gate-study-plan.md.

    python -m backend.gate_study ic-ettin-150m-s6000-near ic-ettin-32m-s6000-near
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from backend.calibrate import ALPHA, SELECT, choose_threshold, fit_temperature, scale

MIN_N = int(np.ceil(np.log(ALPHA) / np.log(SELECT)))  # 427
PILOT = 0.997
GRID = np.arange(1, 10) / 10
HALVINGS = 200
TRIALS = 2000
SYNTHETIC_ROWS = 1200  # about MTOP's clean calib


def cp_lower(k: int, n: int) -> float:
    return float(beta.ppf(ALPHA, k, n - k + 1)) if k > 0 else 0.0


def ranked(y, p):
    """Confidence, sorted high to low, cumulative hits, and which positions end a tie."""
    conf = p.max(1)
    order = np.argsort(-conf, kind="stable")
    conf = conf[order]
    hits = np.cumsum(p.argmax(1)[order] == y[order])
    last = np.append(conf[1:] != conf[:-1], True)
    return conf, hits, last


def passes(tau, y, p) -> bool:
    accepted = p.max(1) >= tau
    n = int(accepted.sum())
    return n > 0 and cp_lower(int((p.argmax(1) == y)[accepted].sum()), n) >= SELECT


def rule_a(cert_y, cert_p, pilot_y, pilot_p):
    """The current rule: walk every confidence on the certify half."""
    return choose_threshold(cert_y, cert_p)[0]


def rule_b(cert_y, cert_p, pilot_y, pilot_p):
    """One threshold chosen on the pilot half, tested once on the certify half."""
    conf, hits, last = ranked(pilot_y, pilot_p)
    ok = np.flatnonzero(last & (hits / np.arange(1, len(conf) + 1) >= PILOT))
    if len(ok) == 0:
        return None
    tau = float(conf[ok[-1]])
    return tau if passes(tau, cert_y, cert_p) else None


def rule_c(cert_y, cert_p, pilot_y, pilot_p):
    """Nine pilot thresholds, walked on the certify half; too-small cohorts are skipped."""
    conf = np.sort(pilot_p.max(1))[::-1]
    best = None
    for tau in conf[(GRID * len(conf)).astype(int) - 1]:
        if (cert_p.max(1) >= tau).sum() < MIN_N:
            continue
        if not passes(tau, cert_y, cert_p):
            break
        best = float(tau)
    return best


RULES = {"A": rule_a, "B": rule_b, "C": rule_c}


def headroom(y, p) -> float:
    """Largest share of test, taken by confidence, that is at least SELECT precise (min 1%)."""
    conf, hits, last = ranked(y, p)
    n = np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(last & (hits / n >= SELECT) & (n >= 0.01 * len(conf)))
    return float(n[ok[-1]] / len(conf)) if len(ok) else 0.0


def study(path: Path) -> dict:
    d = np.load(path)
    pc, pt = (d[k].astype(np.float64) for k in ("p_calib", "p_test"))
    pc, pt = pc / pc.sum(1, keepdims=True), pt / pt.sum(1, keepdims=True)
    yc, yt = d["y_calib"], d["y_test"]
    hits = pt.argmax(1) == yt
    runs = {r: [] for r in RULES}
    for seed in range(HALVINGS):
        half = np.random.default_rng(seed).permutation(len(yc))
        fit, cert = half[: len(yc) // 2], half[len(yc) // 2 :]
        t = fit_temperature(yc[fit], pc[fit])
        pilot, certify, test = scale(pc[fit], t), scale(pc[cert], t), scale(pt, t)
        for name, rule in RULES.items():
            tau = rule(yc[cert], certify, yc[fit], pilot)
            accepted = test.max(1) >= tau if tau is not None else np.zeros(len(yt), bool)
            precision = float(hits[accepted].mean()) if accepted.any() else None
            runs[name].append((float(accepted.mean()), precision))
        if seed == 0:
            room = headroom(yt, test)
    out = {"accuracy": round(float(hits.mean()), 4), "calib": len(yc), "test": len(yt)}
    for name, rows in runs.items():
        cov = np.array([c for c, _ in rows])
        unsafe = np.mean([p is not None and p < 0.99 for _, p in rows])
        out[name] = {
            "median_coverage": round(float(np.median(cov)), 4),
            "certifies": round(float((cov > 0).mean()), 3),
            "unsafe_on_test": round(float(unsafe), 3),
        }
    scaled = room * (len(yc) - len(yc) // 2)
    out["headroom"] = round(room, 4)
    if out["A"]["median_coverage"] == 0:
        out["zero_cause"] = (
            "sample size" if scaled >= MIN_N else "residual errors" if room == 0 else "weak ranking"
        )
    return out


CURVES = {
    # P(correct | score s) = 1 - 0.1 (1 - s)^3: precision above tau is 1 - 0.025 (1 - tau)^3,
    # 99% from tau 0.263, so a safe rule accepts at most ~74% of rows, near the 427 floor.
    "monotone": (lambda s: 1 - 0.1 * (1 - s) ** 3, None),
    "confident dip": (
        lambda s: np.where((s > 0.9) & (s < 0.95), 0.9, 1 - 0.1 * (1 - s) ** 3),
        None,
    ),
    "heavy ties": (lambda s: 1 - 0.1 * (1 - s) ** 3, 20),
}


def synthetic(seed: int = 0) -> dict:
    """How often each rule accepts anything, and accepts a set whose true precision is below 99%."""
    rng = np.random.default_rng(seed)
    out = {}
    for curve, (correct, levels) in CURVES.items():
        grid = (np.arange(100_000) + 0.5) / 100_000
        if levels:
            grid = np.ceil(grid * levels) / levels
        c = correct(grid)
        unsafe, accepts = dict.fromkeys(RULES, 0), dict.fromkeys(RULES, 0)
        for _ in range(TRIALS):
            s = rng.random(SYNTHETIC_ROWS)
            if levels:
                s = np.ceil(s * levels) / levels
            y = (rng.random(SYNTHETIC_ROWS) >= correct(s)).astype(int)  # 0 = correct
            p = np.stack([0.5 + 0.5 * s, 0.5 - 0.5 * s], 1)
            pilot, cert = slice(0, SYNTHETIC_ROWS // 2), slice(SYNTHETIC_ROWS // 2, None)
            for name, rule in RULES.items():
                tau = rule(y[cert], p[cert], y[pilot], p[pilot])
                accepts[name] += tau is not None
                unsafe[name] += tau is not None and c[0.5 + 0.5 * grid >= tau].mean() < 0.99
        out[curve] = {
            name: {"accepts": accepts[name] / TRIALS, "unsafe": unsafe[name] / TRIALS}
            for name in RULES
        }
    return out


def main():
    from backend.config import Settings

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    folder = Settings().data_dir / "predictions"
    report = {
        "plan": "docs/results/s2/gate-study-plan.md",
        "models": {
            m: {p.stem: study(p) for p in sorted((folder / m).glob("*.npz"))} for m in args.models
        },
        "synthetic": synthetic(),
    }
    path = Path("docs/results/s2/gate-study.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["synthetic"]))


if __name__ == "__main__":
    main()
