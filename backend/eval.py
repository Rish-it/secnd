"""Eval card: `uv run python -m backend.eval --model tfidf_lr --budget 200 [--dataset ipsa]`."""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend import metrics
from backend.baselines import BASELINES
from backend.config import Settings
from backend.data import load_dataset, split_dataset, split_manifest

TAU = 0.99


def _round(value: float) -> float | None:
    return None if np.isnan(value) else round(value, 4)


QUALITY = ["accuracy", "macro_f1", "nll", "brier", "ece", "classwise_ece", "aurc"]


def training_rows(frame: pd.DataFrame, split: pd.Series, budget: int, seed: int) -> pd.DataFrame:
    """The seed rows, then the earliest stream rows (a seeded shuffle if undated) up to budget."""
    stream = frame[split == "stream"]
    if stream["date"].notna().all():
        stream = stream.sort_values(["date", "id"])
    else:
        stream = stream.sample(frac=1, random_state=seed)
    return pd.concat(
        [frame[split == "seed"], stream.head(max(budget - len(frame[split == "seed"]), 0))]
    )


def quality(y, p) -> dict:
    return {name: round(getattr(metrics, name)(y, p), 4) for name in QUALITY}


def card(dataset: str, model_name: str, budget: int, seed: int = 0) -> dict:
    frame = load_dataset(dataset, Settings().data_dir)
    split = split_dataset(frame, seed)
    labels = sorted(frame["label"].unique())
    train = training_rows(frame, split, budget, seed)

    started = time.perf_counter()
    model = BASELINES[model_name](labels).fit(train)
    fit_seconds = time.perf_counter() - started

    results = {}
    for part in ("calib", "test"):
        rows = frame[split == part]
        results[part] = (rows, model.encode(rows), model.predict_proba(rows))
    test, y, p = results["test"]
    calib, y_calib, p_calib = results["calib"]

    kept = p_calib.max(axis=1) >= TAU
    correct = int((p_calib[kept].argmax(axis=1) == y_calib[kept]).sum())
    seen = test["merchant_key"].isin(set(train["merchant_key"]) - {""}).to_numpy()
    recall = (
        pd.Series(p.argmax(axis=1) == y).groupby(test["label"].to_numpy()).agg(["mean", "size"])
    )
    worst = recall[recall["size"] >= 20].sort_values("mean").head(5)

    timings = []
    for _, row in test.sample(min(200, len(test)), random_state=seed).iterrows():
        started = time.perf_counter()
        model.predict_proba(row.to_frame().T)
        timings.append(time.perf_counter() - started)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return {
        "provenance": {
            "dataset": dataset,
            "model": model_name,
            "budget": budget,
            "train_rows": len(train),
            "seed": seed,
            "split_hash": split_manifest(dataset, frame, split)["hash"],
            "git_sha": sha.stdout.strip(),
        },
        "quality": quality(y, p),
        "claim": {
            "tau": TAU,
            "coverage": round(metrics.coverage_at(p, TAU), 4),
            "precision_above": _round(metrics.precision_above(y, p, TAU)),
            "calib_auto": {"n": int(kept.sum()), "correct": correct},
            "calib_cp_lower": round(metrics.clopper_pearson_lower(correct, int(kept.sum())), 4),
        },
        "slices": {
            name: {
                "rows": int(mask.sum()),
                "accuracy": round(metrics.accuracy(y[mask], p[mask]), 4),
            }
            for name, mask in (("seen_merchant", seen), ("unseen_merchant", ~seen))
            if mask.any()
        },
        "worst_classes": {label: round(row["mean"], 4) for label, row in worst.iterrows()},
        "speed": {
            "fit_seconds": round(fit_seconds, 3),
            "predict_ms_p50": round(float(np.percentile(timings, 50)) * 1000, 3),
            "predict_ms_p99": round(float(np.percentile(timings, 99)) * 1000, 3),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
            if sys.platform == "darwin"
            else round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--dataset", default="ipsa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = card(args.dataset, args.model, args.budget, args.seed)
    out = Path("docs/results/s1") / f"card-{args.dataset}-{args.model}-b{args.budget}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("quality", "claim", "slices")}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
