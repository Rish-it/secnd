"""Bake-off grid: `uv run python -m backend.bench [--dataset ipsa] [--quick]`.

Every featurizer x classifier x label budget x seed is trained on `training_rows` and scored on
the fixed test split. Writes docs/results/s2/bakeoff[-quick].csv and a mean +/- std table.

Speed: `uv run python -m backend.bench --speed` measures each finalist in its own process and
writes docs/results/s2/speed.csv.
"""

import argparse
import json
import pickle
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
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
FINALISTS = ("char_tfidf+lr", "potion+lr", "stack+lr", "minilm+lr", "potion+knn")
# model.safetensors on the HF Hub at the pinned revisions (backend.features.EMBEDDERS)
WEIGHTS_BYTES = {"potion": 30_236_760, "minilm": 90_868_376}


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


def _ms(timings) -> tuple[float, float]:
    return tuple(round(float(np.percentile(timings, q)) * 1000, 3) for q in (50, 99))


def speed_one(model_id: str) -> dict:
    """Latency with cold texts (no embedding cache), retrain time, memory and artifact size."""
    featurizer, classifier = model_id.split("+")
    frame = load_dataset("ipsa", Settings().data_dir)
    split = split_dataset(frame, 0)
    labels = sorted(frame["label"].unique())
    test = frame[split == "test"].sample(200 + 20 * 256, random_state=0)
    model = Model(labels, featurizer, classifier).fit(training_rows(frame, split, 500, 0))
    model.predict_proba(test.head(1))  # load the embedder once, as a running server would

    single = []
    for _, row in test.head(200).iterrows():
        started = time.perf_counter()
        model.predict_proba(row.to_frame().T)
        single.append(time.perf_counter() - started)
    batches = []
    for start in range(200, len(test), 256):
        started = time.perf_counter()
        model.predict_proba(test.iloc[start : start + 256])
        batches.append(time.perf_counter() - started)
    retrain = {}
    for rows in (1000, 10000):
        train = training_rows(frame, split, rows, 0)
        started = time.perf_counter()
        Model(labels, featurizer, classifier, c=getattr(model, "c", None)).fit(train)
        retrain[rows] = round(time.perf_counter() - started, 3)

    weights = WEIGHTS_BYTES.get("potion" if featurizer == "stack" else featurizer, 0)
    p50, p99 = _ms(single)
    b50, b99 = _ms(batches)
    return {
        "model": model_id,
        "predict_ms_p50_batch1": p50,
        "predict_ms_p99_batch1": p99,
        "predict_ms_p50_batch256": b50,
        "predict_ms_p99_batch256": b99,
        "retrain_s_1k": retrain[1000],
        "retrain_s_10k": retrain[10000],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "artifact_bytes": len(pickle.dumps(model)) + weights,
        "needs_torch": featurizer != "char_tfidf",
    }


def speed() -> pd.DataFrame:
    rows = []
    for model_id in FINALISTS:
        out = subprocess.run(
            [sys.executable, "-m", "backend.bench", "--speed-one", model_id],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ipsa")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--speed-one")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # lbfgs stops at max_iter by design (see model.LR)
    if args.speed_one:
        print(json.dumps(speed_one(args.speed_one)))
        return
    if args.speed:
        OUT.mkdir(parents=True, exist_ok=True)
        speed().to_csv(OUT / "speed.csv", index=False)
        return
    grid = QUICK if args.quick else {"budgets": BUDGETS, "seeds": SEEDS}
    results = run(args.dataset, grid["budgets"], list(grid["seeds"]))
    name = f"bakeoff-{args.dataset}" + ("-quick" if args.quick else "")
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(table(results))
    print(table(results))


if __name__ == "__main__":
    main()
