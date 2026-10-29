"""Fixed-split cold-start handoff; paid labels require --label and a $10 task cap."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import numpy as np
from filelock import FileLock

from backend.config import Settings
from backend.data import load_dataset, split_dataset, split_manifest
from backend.encoders import fetch
from backend.eval import training_rows
from backend.loop import Loop
from backend.muse import BudgetRun, MuseClient, MuseError, MuseItem, MuseOutputError, Usage
from backend.providers import provider_settings
from backend.sim import WINDOW, pick, score

TASK = "s5-mtop-cold-start"
CAP = Decimal("10")


def task_spend(ledger: Path) -> Decimal:
    if not ledger.exists():
        return Decimal(0)
    with ledger.open() as source:
        return sum(
            (Decimal(r["cost_usd"]) for r in csv.DictReader(source) if r["task"] == TASK),
            Decimal(0),
        )


def select(frame, split, rows=500, seed=0):
    if not 1 <= rows <= 500:
        raise ValueError("Select between 1 and 500 stream rows.")
    chosen = frame[split == "stream"].sample(rows, random_state=seed)
    if chosen.id.duplicated().any():
        raise ValueError("Stream item IDs must be unique.")
    return chosen


def collect(
    frame,
    split,
    settings,
    path,
    ledger,
    *,
    rows=500,
    seed=0,
    allow_paid=False,
    client_factory=MuseClient,
):
    chosen = select(frame, split, rows, seed)
    labels = sorted(frame.label.unique())
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        cache = (
            json.loads(path.read_text())
            if path.exists()
            else {"model": settings.muse_model, "items": {}}
        )
        if cache["model"] != settings.muse_model:
            raise ValueError("Cold-start cache model differs from the selected teacher.")
        missing = []
        for row in chosen.itertuples():
            key = hashlib.sha256(row.text.encode()).hexdigest()
            old = cache["items"].get(str(row.id))
            if old:
                if old["text_sha256"] != key or old["label"] not in labels:
                    raise ValueError("Cold-start cache does not match the selected training text.")
            else:
                missing.append(row)
        if missing and not allow_paid:
            raise ValueError("Cold-start cache incomplete; --label enables budgeted teacher calls.")
        if missing:
            with BudgetRun(
                settings,
                task=TASK,
                rows=len(missing),
                provider="meta",
                measured_tokens=Usage(prompt_tokens=6000, completion_tokens=2048),
                ledger_path=ledger,
            ) as run:
                # The ledger lock is held: the cap applies to this task across every resume.
                run.budget = min(run.budget, CAP - task_spend(ledger))
                if run.estimate > run.budget:
                    raise MuseError("Cold-start estimate exceeds the remaining $10 task cap.")
                with client_factory(settings, run=run) as client:

                    def label(row):
                        for attempt in range(2):
                            try:
                                return client.cold_start(
                                    [MuseItem(item_id=str(row.id), text=row.text)], labels
                                )[0]
                            except MuseOutputError:
                                if attempt:
                                    raise

                    failures = 0
                    with ThreadPoolExecutor(max_workers=8) as workers:
                        pending = {workers.submit(label, row): row for row in missing}
                        for future in as_completed(pending):
                            row = pending[future]
                            try:
                                result = future.result()
                            except MuseError:
                                failures += 1
                                continue
                            cache["items"][str(row.id)] = {
                                **result.model_dump(mode="json"),
                                "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                            }
                            staged = path.with_suffix(".tmp")
                            staged.write_text(json.dumps(cache, sort_keys=True))
                            os.replace(staged, path)
                            count = sum(str(r.id) in cache["items"] for r in chosen.itertuples())
                            if count % 50 == 0:
                                print(f"Cold-start labels cached: {count}/{rows}", flush=True)
                    if failures:
                        raise MuseError("Some cold-start labels failed; completed rows are cached.")
        return chosen, [cache["items"][str(item)]["label"] for item in chosen.id]


def stats(loop, test, result):
    measured = (
        score(loop, test)
        if loop.current
        else {"coverage": 0.0, "precision": None, "accuracy": None}
    )
    if loop.guarded:
        measured.update(coverage=0.0, precision=None)
    return {**result, **measured, "labels": len(loop.y), "guarded": loop.guarded}


def evaluate(
    frame,
    split,
    pseudo,
    pseudo_labels,
    *,
    featurizer="pt-minilm",
    seed=0,
    rounds=20,
    batch=100,
    cache_dir=None,
):
    labels = sorted(frame.label.unique())
    calib, test = frame[split == "calib"], frame[split == "test"]
    if not split.loc[pseudo.index].eq("stream").all():
        raise ValueError("Teacher labels must come only from stream rows.")
    if len(pseudo_labels) != len(pseudo) or not set(pseudo_labels).issubset(labels):
        raise ValueError("Teacher labels must match the selected stream rows and label set.")
    if rounds < 1 or batch < 1:
        raise ValueError("Rounds and batch must be positive.")

    def fresh():
        return Loop(labels, calib, featurizer, seed=seed, cache_dir=cache_dir)

    comparisons = []
    for budget in (50, 200):
        ren = fresh()
        human = training_rows(frame, split, budget, seed)
        ren.learn(human, human.label, "seed")
        before = stats(ren, test, ren.retrain())
        ren.learn(pseudo, pseudo_labels, "muse_cold_start")
        after = stats(ren, test, ren.retrain())
        comparisons.append(
            {"human_budget": budget, "without_teacher": before, "with_teacher": after}
        )

    ren = fresh()
    ren.learn(pseudo, pseudo_labels, "muse_cold_start")
    result = ren.retrain()
    remaining = frame[(split == "stream") & ~frame.id.isin(pseudo.id)]
    if len(remaining) < rounds:
        raise ValueError("Not enough unused stream rows for the requested rounds.")
    size = min(WINDOW, len(remaining) // rounds)
    rng = np.random.default_rng(seed)
    history, position, automatic, answers = [], 0, 0, 0
    for round_ in range(rounds + 1):
        completed = len(pseudo) + automatic + answers
        history.append(
            {
                "round": round_,
                "human_labels": answers,
                **stats(ren, test, result),
                "muse_decisions": len(pseudo),
                "ren_decisions": automatic,
                "human_decisions": answers,
                "pending": position - automatic - answers,
                "shares": {
                    "muse": len(pseudo) / completed,
                    "ren": automatic / completed,
                    "human": answers / completed,
                },
            }
        )
        if round_ == rounds:
            break
        window = remaining.iloc[position : position + size]
        position += len(window)
        if ren.current:
            decisions = ren.route(window)
            chosen = pick(decisions, batch, "random", rng)
            checks = [i for i, d in enumerate(decisions) if d["spot_check"]]
            automatic += sum(d["auto"] for d in decisions)
        else:
            chosen = list(rng.permutation(len(window))[:batch])
            checks = []
        for i in chosen + checks:
            item = window.iloc[[i]]
            ren.learn(item, item.label, "spot_check" if i in checks else "answer")
        answers += len(chosen) + len(checks)
        result = ren.retrain()
    return {"comparisons": comparisons, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="store_true", help="Allow paid Meta teacher calls")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    settings = Settings()
    frame = load_dataset("mtop", settings.data_dir)
    split = split_dataset(frame, args.seed)
    teacher = provider_settings(settings, "meta")
    path = settings.data_dir / "cache" / f"cold-start-mtop-s{args.seed}.json"
    ledger = Path("docs/results/s5/spend-meta.csv")
    pseudo, labels = collect(
        frame, split, teacher, path, ledger, rows=args.rows, seed=args.seed, allow_paid=args.label
    )
    if settings.featurizer == "pt-minilm":
        fetch(settings.featurizer, settings.data_dir)
    result = evaluate(
        frame,
        split,
        pseudo,
        labels,
        featurizer=settings.featurizer,
        seed=args.seed,
        rounds=args.rounds,
        batch=args.batch,
        cache_dir=settings.data_dir / "cache",
    )
    result.update(
        dataset="mtop",
        seed=args.seed,
        featurizer=settings.featurizer,
        teacher=teacher.muse_model,
        teacher_rows=len(pseudo),
        teacher_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        task_spend_usd=str(task_spend(ledger)),
        task_cap_usd=str(CAP),
        split=split_manifest("mtop", frame, split),
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        scope="Offline simulation: true stream labels stand in for human answers. "
        "Shares count completed decisions cumulatively; pending questions are separate.",
    )
    out = Path("docs/results/s5/handoff.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {out}; task spend ${result['task_spend_usd']} / $10", flush=True)


if __name__ == "__main__":
    main()
