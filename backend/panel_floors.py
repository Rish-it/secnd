"""The refit gate with its registered floors on all MTOP test rows, from saved probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from backend import incontext as ic
from backend.calibrate import scale

RUNS = {
    "ic-ettin-150m-s6000-near": "attacks-floor-audited",
    "ic-ettin-400m-s6000-near": "attacks-floor-audited-400m",
}
SAVED = {"0": "reader-0", "8": "all-reader-8", "32": "all-reader-32"}


def supports(task, answers):
    """Highest similarity between each test message and history rows carrying its answer."""
    history = torch.tensor([r[1] for r in task.train], device=task.train_vecs.device)
    sims = task.test_vecs.float() @ task.train_vecs.float().T
    out = np.full(len(answers), -1.0)
    for label in np.unique(answers):
        rows = np.flatnonzero(answers == label)
        mask = history == int(label)
        if mask.any():
            out[rows] = sims[rows][:, mask].max(1).values.cpu().numpy()
    return out


def score(data, model, report):
    """Every test row of MTOP through one committed gate. `report` holds the gate per shot count."""
    task = ic.held_out(data)["mtop"]
    ic.index([task])
    y, k, results = ic.labels_of(task.test), len(task.labels), {}
    for shots, name in SAVED.items():
        gate = report["results"][shots]
        if gate["threshold"] is None:  # certified nothing on calibration, so it accepts nothing
            results[shots] = dict(
                test_rows=len(y),
                accepted_without_floor=0,
                accepted=0,
                errors=0,
                coverage=0.0,
                precision=None,
                low95=0.0,
            )
            continue
        saved = np.load(data / "predictions" / model / f"mtop-{name}.npz")
        assert (saved["y_test"] == y).all()
        p = scale(saved["p_test"].astype(np.float64), gate["temperature"])[:, :k]
        answers = p.argmax(1)
        without = p.max(1) >= gate["threshold"]
        kept = without & (supports(task, answers) >= gate["floor"]["floor"])
        n, hits = int(kept.sum()), int((answers[kept] == y[kept]).sum())
        results[shots] = dict(
            test_rows=len(y),
            accepted_without_floor=int(without.sum()),
            accepted=n,
            errors=n - hits,
            coverage=round(n / len(y), 4),
            precision=round(hits / n, 4) if n else None,
            low95=round(float(beta.ppf(0.025, hits, n - hits + 1)), 4) if hits else 0.0,
        )
        print(model, shots, results[shots], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".data"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    results = {}
    for model, run in RUNS.items():
        report = json.loads(Path(f"docs/results/s3r/{run}.json").read_text())
        results[model] = score(args.data, model, report)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
