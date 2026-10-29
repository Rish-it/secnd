"""Fixed-test comparison: `uv run python -m backend.muse_baseline [--limit 5]`."""

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from backend import metrics
from backend.baselines import BASELINES
from backend.config import Settings
from backend.data import load_dataset, split_dataset, split_manifest
from backend.eval import training_rows
from backend.muse import SPEND_PATH, BudgetRun, LabelResult, MuseClient, MuseError, MuseItem, Usage

RESULTS = Path("docs/results/s5")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sample(frame: pd.DataFrame, seed: int = 0, rows: int = 500):
    split = split_dataset(frame, seed)
    test = frame[split == "test"].sample(n=rows, random_state=seed)
    return split, test


def item(row: pd.Series) -> MuseItem:
    return MuseItem(
        item_id=row["id"],
        text=row["text"],
        amount=None if pd.isna(row["amount"]) else float(row["amount"]),
        date=None if pd.isna(row["date"]) else row["date"].date(),
    )


def read_predictions(path: Path, plan: dict) -> dict[str, LabelResult]:
    results = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        record = json.loads(line)
        result = LabelResult.model_validate(record["result"])
        if (
            record["experiment_hash"] != digest(plan)
            or result.item_id not in plan["row_ids"]
            or result.item_id in results
            or set(result.probabilities) != set(plan["labels"])
            or result.label not in result.probabilities
            or abs(sum(result.probabilities.values()) - 1) > 1e-6
            or result.confidence != result.probabilities[result.label]
            or result.confidence != max(result.probabilities.values())
        ):
            raise ValueError("Prediction checkpoint does not match the fixed experiment.")
        results[result.item_id] = result
    return results


def collect(settings, plan, test, directory, *, limit, git_sha, transport=None):
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "muse-sample.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Sample or labeling protocol changed; refusing to reuse predictions.")
    write_json(plan_path, plan)
    predictions_path = directory / "muse-predictions.jsonl"
    results = read_predictions(predictions_path, plan)
    pending = test[~test["id"].isin(results)].head(limit)
    if pending.empty:
        return results
    if results:
        measured = Usage(
            **{
                field: math.ceil(np.mean([getattr(r.usage, field) for r in results.values()]))
                for field in Usage.model_fields
            }
        )
    else:
        measured = Usage.model_validate(json.loads((directory / "smoke.json").read_text())["usage"])
    spent = sum(
        (Decimal(row["cost_usd"]) for row in baseline_spend(directory / SPEND_PATH.name)),
        Decimal(0),
    )
    settings = settings.model_copy(
        update={"muse_budget_usd": max(Decimal(0), settings.muse_budget_usd - spent)}
    )
    with (
        BudgetRun(
            settings,
            task="muse-baseline",
            rows=len(pending),
            measured_tokens=measured,
            ledger_path=directory / SPEND_PATH.name,
        ) as run,
        MuseClient(settings, run=run, transport=transport) as client,
        predictions_path.open("a") as checkpoint,
    ):
        for _, row in pending.iterrows():
            result = client.zero_shot([item(row)], plan["labels"])[0]
            checkpoint.write(
                json.dumps(
                    {
                        "experiment_hash": digest(plan),
                        "git_sha": git_sha,
                        "result": result.model_dump(mode="json"),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            checkpoint.flush()
            results[result.item_id] = result
            print(
                json.dumps({"completed": len(results), "run_cost_usd": str(run.cost_usd)}),
                flush=True,
            )
    return results


def baseline_spend(ledger):
    if not ledger.exists():
        return []
    with ledger.open() as source:
        return [row for row in csv.DictReader(source) if row["task"] == "muse-baseline"]


def summarize(test, results, ledger):
    ordered = [results[row_id] for row_id in test["id"]]
    spend = baseline_spend(ledger)
    cost = sum((Decimal(row["cost_usd"]) for row in spend), Decimal(0))
    return {
        "rows": len(ordered),
        "accuracy": float(
            np.mean([r.label == y for r, y in zip(ordered, test["label"], strict=True)])
        ),
        "nll": None,
        "ece": None,
        "confidence_source": "self_reported",
        "predict_ms": {
            "p50": float(np.percentile([r.latency_ms for r in ordered], 50)),
            "p99": float(np.percentile([r.latency_ms for r in ordered], 99)),
        },
        "usage": {field: sum(int(row[field]) for row in spend) for field in Usage.model_fields},
        "cost_usd": str(cost),
        "cost_per_1000_usd": str(cost * 1000 / len(ordered)),
        "usage_includes_retry_estimates": any("estimated" in row["status"] for row in spend),
    }


def compare(frame, split, test, seed=0, budgets=(200, 5000)):
    rows = []
    labels = sorted(frame["label"].unique())
    for budget in budgets:
        train = training_rows(frame, split, budget, seed)
        model = BASELINES["tfidf_lr"](labels).fit(train)
        p = model.predict_proba(test)
        y = model.encode(test)
        timings = []
        for _, row in test.iterrows():
            started = time.perf_counter()
            model.predict_proba(row.to_frame().T)
            timings.append((time.perf_counter() - started) * 1000)
        rows.append(
            {
                "model": "tfidf_lr",
                "training_rows": len(train),
                **{name: getattr(metrics, name)(y, p) for name in ("accuracy", "nll", "ece")},
                "predict_ms": {
                    "p50": float(np.percentile(timings, 50)),
                    "p99": float(np.percentile(timings, 99)),
                },
                "cost_per_1000_usd": None,
            }
        )
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=500, help="Maximum new paid rows this invocation."
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("--limit must be between 1 and 500")
    settings = Settings()
    frame = load_dataset("ipsa", settings.data_dir)
    split, test = sample(frame)
    manifest = split_manifest("ipsa", frame, split)
    plan = {
        "dataset": "ipsa",
        "seed": 0,
        "rows": 500,
        "row_ids": test["id"].tolist(),
        "labels": sorted(frame["label"].unique()),
        "data_hash": digest(manifest["sources"]),
        "split_hash": manifest["hash"],
        "sample_hash": digest(test["id"].tolist()),
        "input_hash": digest([item(row).model_dump(mode="json") for _, row in test.iterrows()]),
        "protocol_hash": digest(inspect.getsource(MuseClient.zero_shot)),
        "model": settings.muse_model,
        "base_url": settings.muse_base_url,
    }
    git_sha = subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], text=True).strip()
    try:
        results = collect(settings, plan, test, RESULTS, limit=args.limit, git_sha=git_sha)
    except (MuseError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    if len(results) != 500:
        print(f"Saved {len(results)}/500 rows; rerun to resume the same sample.")
        return
    provenance = {
        key: value for key, value in plan.items() if key not in ("row_ids", "labels", "base_url")
    }
    baseline = {
        **provenance,
        "git_sha": git_sha,
        "measured_at": datetime.now(UTC).isoformat(),
        **summarize(test, results, RESULTS / SPEND_PATH.name),
    }
    write_json(RESULTS / "muse-baseline.json", baseline)
    comparison = {
        **provenance,
        "git_sha": git_sha,
        "muse": baseline,
        "baselines": compare(frame, split, test),
    }
    write_json(RESULTS / "comparison.json", comparison)
    print(json.dumps({"accuracy": baseline["accuracy"], "cost_usd": baseline["cost_usd"]}))


if __name__ == "__main__":
    main()
