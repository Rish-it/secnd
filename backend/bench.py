"""Bake-off grid: `uv run python -m backend.bench [--dataset ipsa] [--quick]`.

Every featurizer x classifier x label budget x seed is trained on `training_rows` and scored on
the fixed test split. Writes docs/results/s2/bakeoff[-quick].csv and a mean +/- std table.
"""

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from backend import metrics
from backend.config import Settings
from backend.data import load_dataset, split_dataset
from backend.eval import TAU, training_rows
from backend.features import FEATURIZERS
from backend.model import CLASSIFIERS, Model

BUDGETS = (50, 100, 200, 500, "all")
SEEDS = range(5)
QUICK = {"budgets": (100, 500), "seeds": range(2)}
METRICS = ["accuracy", "nll", "ece", "coverage"]
OUT = Path("docs/results/s2")


def run(dataset: str, budgets, seeds) -> pd.DataFrame:
    settings = Settings()
    frame = load_dataset(dataset, settings.data_dir)
    labels = sorted(frame["label"].unique())
    cache = settings.data_dir / "cache"
    rows = []
    for seed in seeds:
        split = split_dataset(frame, seed)
        test = frame[split == "test"]
        for budget in budgets:
            # Every seed trains "all" on the same rows, so one seed is enough there.
            if budget == "all" and seed != seeds[0]:
                continue
            train = training_rows(frame, split, len(frame) if budget == "all" else budget, seed)
            for featurizer in FEATURIZERS:
                for classifier in CLASSIFIERS:
                    started = time.perf_counter()
                    model = Model(labels, featurizer, classifier, seed, cache).fit(train)
                    fit = time.perf_counter() - started
                    y, p = model.encode(test), model.predict_proba(test)
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model.id,
                            "budget": budget,
                            "seed": seed,
                            "accuracy": metrics.accuracy(y, p),
                            "nll": metrics.nll(y, p),
                            "ece": metrics.ece(y, p),
                            "coverage": metrics.coverage_at(p, TAU),
                            "precision": metrics.precision_above(y, p, TAU),
                            "fit_seconds": round(fit, 2),
                        }
                    )
                    print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def table(results: pd.DataFrame) -> str:
    """Mean +/- std per model and budget, best NLL first within each budget."""
    stats = results.groupby(["budget", "model"], sort=False)[METRICS].agg(["mean", "std"])
    lines = ["| budget | model | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    for budget in results["budget"].unique():
        for model, row in stats.loc[budget].sort_values(("nll", "mean")).iterrows():
            cells = [f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}" for m in METRICS]
            lines.append(f"| {budget} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines).replace("± nan", "(1 seed)") + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ipsa")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    grid = QUICK if args.quick else {"budgets": BUDGETS, "seeds": SEEDS}
    warnings.filterwarnings("ignore")  # lbfgs stops at max_iter by design (see model.LR)
    results = run(args.dataset, grid["budgets"], list(grid["seeds"]))
    name = f"bakeoff-{args.dataset}" + ("-quick" if args.quick else "")
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(table(results))
    print(table(results))


if __name__ == "__main__":
    main()
