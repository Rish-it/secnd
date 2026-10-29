"""Eval card: `uv run python -m backend.eval --model tfidf_lr --budget 200 [--dataset ipsa]`.

A featurizer+classifier model (`--model minilm+lr`) gets the calibrated card: temperature and
threshold fitted on calib, the claim measured on test, plus its curves in docs/results/s3/.

Leakage audit: `uv run python -m backend.eval --leakage [--dataset ipsa]`.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend import calibrate, metrics
from backend.baselines import BASELINES
from backend.config import Settings
from backend.data import DATASETS, download_dataset, load_dataset, split_dataset, split_manifest

TAU = 0.99


def _round(value: float) -> float | None:
    return None if np.isnan(value) else round(value, 4)


QUALITY = ["accuracy", "macro_f1", "nll", "brier", "ece", "aurc"]


def training_rows(frame: pd.DataFrame, split: pd.Series, budget: int, seed: int) -> pd.DataFrame:
    """The seed rows, then the earliest stream rows (a seeded shuffle if undated) up to budget.

    A budget below the seed size takes a seeded sample of the seed rows.
    """
    seed_rows = frame[split == "seed"]
    if budget < len(seed_rows):
        return seed_rows.sample(budget, random_state=seed)
    stream = frame[split == "stream"]
    if stream["date"].notna().all():
        stream = stream.sort_values(["date", "id"])
    else:
        stream = stream.sample(frac=1, random_state=seed)
    return pd.concat([seed_rows, stream.head(budget - len(seed_rows))])


def quality(y, p, labels: list[str]) -> dict:
    return {
        **{name: round(getattr(metrics, name)(y, p), 4) for name in QUALITY},
        "classwise_ece": {
            label: round(metrics._ece(p[:, index], y == index, bins=10), 4)
            for index, label in enumerate(labels)
        },
    }


def curves(y, p, bins: int = 10) -> dict:
    """Reliability bins (top-label) and the coverage/precision trade-off as the threshold moves."""
    confidence, hit = p.max(axis=1), p.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    reliability = [
        {
            "confidence": round(float(confidence[which == b].mean()), 4),
            "accuracy": round(float(hit[which == b].mean()), 4),
            "count": int((which == b).sum()),
        }
        for b in range(bins)
        if (which == b).any()
    ]
    order = np.argsort(-confidence, kind="stable")
    precision = np.cumsum(hit[order]) / np.arange(1, len(y) + 1)
    steps = np.unique(np.linspace(0, len(y) - 1, 101).astype(int))
    trade_off = [
        {
            "threshold": round(float(confidence[order][i]), 4),
            "coverage": round((i + 1) / len(y), 4),
            "precision": round(float(precision[i]), 4),
        }
        for i in steps
    ]
    return {"reliability": reliability, "coverage_precision": trade_off}


def card(dataset: str, model_name: str, budget: int, seed: int = 0, curves_out=None) -> dict:
    """An EvalCard dict; with `curves_out`, the test-set curves are written there too."""
    frame = load_dataset(dataset, Settings().data_dir)
    split = split_dataset(frame, seed)
    labels = sorted(frame["label"].unique())
    train = training_rows(frame, split, budget, seed)

    calibrated = model_name not in BASELINES
    if calibrated:
        from backend.model import Model

        featurizer, classifier = model_name.split("+")
        cache = Settings().data_dir / "cache"
        model = Model(labels, featurizer, classifier, seed, cache).fit(train)
    else:
        model = BASELINES[model_name](labels).fit(train)

    results = {}
    for part in ("calib", "test"):
        rows = frame[split == part]
        results[part] = (rows, model.encode(rows), model.predict_proba(rows))
    test, y, p = results["test"]
    _, y_calib, p_calib = results["calib"]

    if calibrated:
        temperature = calibrate.fit_temperature(y_calib, p_calib)
        p, p_calib = calibrate.scale(p, temperature), calibrate.scale(p_calib, temperature)
        tau, _, _, lower = calibrate.choose_threshold(y_calib, p_calib)
        robustness = Path("docs/results/s3") / f"robustness-{dataset}-{model_name}-b{budget}.json"
        pass_rate = json.loads(robustness.read_text())["pass_rate"] if robustness.exists() else None
    else:
        tau, pass_rate = TAU, None
    kept = p_calib.max(axis=1) >= (tau if tau is not None else np.inf)
    if curves_out:
        curves_out.write_text(json.dumps(curves(y, p), indent=2) + "\n")
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
    manifest = split_manifest(dataset, frame, split)
    slices = {
        name: {
            "count": int(mask.sum()),
            "accuracy": round(metrics.accuracy(y[mask], p[mask]), 4) if mask.any() else None,
        }
        for name, mask in (("seen_merchant", seen), ("unseen_merchant", ~seen))
    }
    return {
        "model_version": f"{dataset}-{model_name}-b{budget}-s{seed}",
        **quality(y, p, labels),
        "threshold": None if tau is None else round(tau, 6),
        "coverage": 0.0 if tau is None else round(metrics.coverage_at(p, tau), 4),
        "precision": None if tau is None else _round(metrics.precision_above(y, p, tau)),
        "cp_lower_bound": round(metrics.clopper_pearson_lower(correct, int(kept.sum())), 4)
        if kept.any()
        else None,
        "robustness_pass_rate": pass_rate,
        **slices,
        "worst_classes": {
            label: {"count": int(row["size"]), "accuracy": round(row["mean"], 4)}
            for label, row in worst.iterrows()
        },
        "featurize_ms": None,
        "predict_ms": {
            "p50": round(float(np.percentile(timings, 50)) * 1000, 3),
            "p99": round(float(np.percentile(timings, 99)) * 1000, 3),
        },
        "retrain_ms": None,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "artifact_bytes": None,
        "image_bytes": None,
        "cost_per_1000_usd": None,
        "data_hash": hashlib.sha256(
            json.dumps(manifest["sources"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "split_hash": manifest["hash"],
        "featurizer_id": model.id if calibrated else None,
        "seed": seed,
        "git_sha": sha.stdout.strip(),
    }


def _lookup_accuracy(column: pd.Series, label: pd.Series, train: pd.Series) -> float:
    """Accuracy of answering with the commonest training label for each non-empty column value."""
    filled = column != ""
    best = label[train & filled].groupby(column[train & filled]).agg(lambda v: v.mode()[0])
    test = ~train & filled & column.isin(best.index)
    return round(float((column[test].map(best) == label[test]).mean()), 4)


def leakage(dataset: str, seed: int = 0) -> dict:
    """How much test shares with training, and whether any side column gives the label away."""
    frame = load_dataset(dataset, Settings().data_dir)
    split = split_dataset(frame, seed)
    labels = sorted(frame["label"].unique())
    test = frame[split == "test"]
    report = {"dataset": dataset, "test_rows": len(test), "budgets": {}}
    for budget in (200, 5000, int((split.isin(["seed", "stream"])).sum())):
        train = training_rows(frame, split, budget, seed)
        keys = set(train["merchant_key"]) - {""}
        seen = test["merchant_key"].isin(keys)
        memory = BASELINES["merchant_memory"](labels).fit(train)
        p = memory.predict_proba(test[seen])
        report["budgets"][budget] = {
            "test_rows_with_seen_merchant": round(float(seen.mean()), 4),
            "test_keys_seen": round(
                len(set(test["merchant_key"]) & keys) / test["merchant_key"].nunique(), 4
            ),
            "test_text_exact_in_train": round(
                float(test["text"].isin(set(train["text"])).mean()), 4
            ),
            "memory_accuracy_all_test": round(
                metrics.accuracy(memory.encode(test), memory.predict_proba(test)), 4
            ),
            "memory_accuracy_seen": round(metrics.accuracy(memory.encode(test[seen]), p), 4)
            if seen.any()
            else None,
        }
    report["empty_text_test"] = round(float((test["text"] == "").mean()), 4)
    if dataset == "ipsa":
        folder = download_dataset(dataset, Settings().data_dir)
        raw = pd.concat(
            [
                pd.read_csv(folder / f, encoding="utf-8-sig", dtype=str)
                for f, _, _ in DATASETS[dataset]
            ],
            ignore_index=True,
        )
        train = split.isin(["seed", "stream"])
        report["side_columns"] = {
            column: {
                "filled": round(float(raw[column].notna().mean()), 4),
                "values": int(raw[column].nunique()),
                "lookup_accuracy_on_test": _lookup_accuracy(
                    raw[column].fillna(""), frame["label"], train
                ),
            }
            for column in ("Short Description", "Category")
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help=f"{', '.join(BASELINES)} or featurizer+classifier")
    parser.add_argument("--leakage", action="store_true")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--dataset", default="ipsa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.leakage:
        report = leakage(args.dataset, args.seed)
        out = Path("docs/results/s1") / f"leakage-{args.dataset}.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if not args.model:
        parser.error("--model is required unless --leakage is given")
    folder = Path("docs/results") / ("s1" if args.model in BASELINES else "s3")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}-{args.model}-b{args.budget}"
    extra = None if args.model in BASELINES else folder / f"curves-{name}.json"
    result = card(args.dataset, args.model, args.budget, args.seed, extra)
    out = folder / f"card-{name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("accuracy", "ece", "coverage", "precision", "cp_lower_bound")},
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
