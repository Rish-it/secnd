"""Replay a stream through the learning loop: `uv run python -m backend.sim --dataset mtop`.

Ren starts from the 200 seed labels. Each round it routes the next window of stream items; the
ones it cannot answer automatically wait for a person, who answers `--batch` of them in the order
the query strategy picks (true labels stand in for the person), plus every spot-check. Then it
retrains once, and the round is scored on the fixed test split. Writes
docs/results/s4/curve-<dataset>-<strategy>-s<seed>.csv.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backend import metrics
from backend.config import Settings
from backend.data import load_dataset, split_dataset
from backend.loop import Loop

STRATEGIES = ("least", "margin", "random")
WINDOW = 1000
OUT = Path("docs/results/s4")


def pick(decisions: list[dict], k: int, strategy: str, rng) -> list[int]:
    """Indices of the queued decisions a person answers first."""
    queued = [i for i, d in enumerate(decisions) if not d["auto"] and not d["spot_check"]]
    if strategy == "random":
        return list(rng.permutation(queued)[:k])
    if strategy == "margin":
        top = [sorted(decisions[i]["probabilities"].values(), reverse=True) for i in queued]
        score = [t[0] - (t[1] if len(t) > 1 else 0) for t in top]
    else:
        score = [decisions[i]["prob"] for i in queued]
    return [queued[i] for i in np.argsort(score, kind="stable")[:k]]


def score(ren: Loop, test: pd.DataFrame) -> dict:
    """Coverage, precision above the threshold and accuracy of the current version on test."""
    p = ren._proba(ren.current["model"], ren._x(test), ren.current["temperature"])
    y = test["label"].map(ren.index).to_numpy()
    tau = ren.current["tau"]
    return {
        "coverage": 0.0 if tau is None else metrics.coverage_at(p, tau),
        "precision": None if tau is None else metrics.precision_above(y, p, tau),
        "accuracy": metrics.accuracy(y, p),
    }


def simulate(
    dataset: str, rounds=20, batch=100, seed=0, strategy="least", on_step=None, featurizer="minilm"
):
    """One row per round; `on_step(event)` is called after every routed window and answer."""
    settings = Settings()
    frame = load_dataset(dataset, settings.data_dir)
    split = split_dataset(frame, seed)
    labels = sorted(frame["label"].unique())
    calib = frame[split == "calib"]
    ren = Loop(labels, calib, featurizer, seed=seed, cache_dir=settings.data_dir / "cache")
    test = frame[split == "test"]
    seed_rows = frame[split == "seed"]
    ren.learn(seed_rows, seed_rows["label"], "seed")
    result = ren.retrain()
    stream = frame[split == "stream"]
    rng = np.random.default_rng(seed)
    history, position = [], 0
    for round_ in range(rounds + 1):
        row = {"round": round_, "labels": len(ren.y), **result, **score(ren, test)}
        row.pop("reasons")
        history.append(row)
        if on_step:
            on_step({"kind": "round", "row": row, "loop": ren})
        if round_ == rounds:
            break
        size = min(WINDOW, len(stream) // rounds)
        window = stream.iloc[position : position + size]
        position += size
        decisions = ren.route(window)
        chosen = pick(decisions, batch, strategy, rng)
        checks = [i for i, d in enumerate(decisions) if d["spot_check"]]
        for i in chosen + checks:
            item = window.iloc[[i]]
            source = "spot_check" if decisions[i]["spot_check"] else "answer"
            ren.learn(item, item["label"], source)
            if on_step:
                on_step({"kind": "answer", "item": item, "decision": decisions[i], "loop": ren})
        if on_step:
            on_step({"kind": "auto", "decisions": [d for d in decisions if d["auto"]], "loop": ren})
        result = ren.retrain()
    return pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mtop")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=STRATEGIES, default="random")
    parser.add_argument("--featurizer", default="minilm", help="e.g. pt-minilm (TOPv2-tuned)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    curve = simulate(
        args.dataset, args.rounds, args.batch, args.seed, args.strategy, featurizer=args.featurizer
    )
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = "" if args.featurizer == "minilm" else f"-{args.featurizer}"
    out = OUT / f"curve-{args.dataset}{encoder}-{args.strategy}-s{args.seed}.csv"
    curve.to_csv(out, index=False)
    print(curve.round(4).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
