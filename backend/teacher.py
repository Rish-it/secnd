"""Meta's models as Ren's teachers: they label a business's unlabelled history, so Ren starts with
no human label. Only history rows (official train minus calib) are ever sent; calib and test
keep their gold labels for the gate and the score.

Pilot: each teacher labels the same history rows whose gold labels are known, to measure
accuracy and cost per row before any larger run.
Cascade: Muse Spark labels the history; Muse Wonder relabels the rows Spark was unsure of.
Examples: Muse Wonder writes messages for each label from the label names alone, so Ren starts
with history before a single real record is labelled (or sent anywhere).

Each provider stays within a $20 cap for this round's tasks; answers are cached by text hash
(no text stored), so a rerun never pays twice.

    python -m backend.teacher --dataset mtop --pilot 100 --teacher wonder --paid
    python -m backend.teacher --dataset mtop --rows 4000 --teacher spark --paid
    python -m backend.teacher --dataset mtop --rows 4000 --teacher wonder --below 0.9 --paid
    python -m backend.teacher --dataset mtop --rows 4000 --below 0.9 --eval
    python -m backend.teacher --dataset mtop --examples 10 --paid    # then add --eval
"""

import argparse
import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

from backend import incontext as ic
from backend.config import Settings
from backend.escalate import ask, key, load_cache
from backend.muse import MuseItem

TEACHERS = {"spark": "baseten", "spark13": "meta", "wonder": "meta-wonder"}
CAP = Decimal("20")
# Baseten rate-limits past a few parallel calls; Meta's API allows 3,000 requests a minute.
WORKERS = {"baseten": 2, "meta": 8, "meta-wonder": 8}
MODEL = "ic-ettin-150m-s6000-near"


def task_name(teacher: str) -> str:
    return f"s2-teach-{teacher}"


def billed_with(provider: str) -> list[str]:
    """This round's tasks on the same ledger as `provider`: they share its cap."""
    meta = provider.startswith("meta")
    tasks = [task_name(t) for t, p in TEACHERS.items() if p.startswith("meta") == meta]
    return tasks + ([task_name("examples")] if meta else [])


def history(task: ic.Task, rows: int) -> list[tuple[str, int]]:
    """The same nested draw `ic.capped` makes, without needing the vectors."""
    keep = np.random.default_rng(0).permutation(len(task.train))[:rows]
    return [task.train[i] for i in keep]


def cache_path(data_dir: Path, teacher: str, dataset: str) -> Path:
    folder = data_dir / "muse"
    folder.mkdir(exist_ok=True)
    return folder / f"teach-{teacher}-{dataset}.json"


def label(settings, task, rows, teacher, *, paid):
    jobs = [(key(t), MuseItem(item_id=key(t)[:32], text=t)) for t, _ in rows]
    return ask(
        settings,
        cache_path(settings.data_dir, teacher, task.name),
        jobs,
        task_name=task_name(teacher),
        label=lambda client, item: client.teach(item, task.labels),
        allow_paid=paid,
        provider=TEACHERS[teacher],
        cap=CAP,
        capped=billed_with(TEACHERS[teacher]),
        workers=WORKERS[TEACHERS[teacher]],
    )


def write_examples(settings, task, n, *, paid):
    return ask(
        settings,
        cache_path(settings.data_dir, f"examples{n}", task.name),
        [(key(label), label) for label in task.labels],
        task_name=task_name("examples"),
        label=lambda client, label: client.examples(label, task.labels, n),
        allow_paid=paid,
        provider="meta-wonder",
        cap=CAP,
        capped=billed_with("meta-wonder"),
        workers=WORKERS["meta-wonder"],
        fields=("label", "messages", "usage", "latency_ms"),
    )


def invented(task, cache) -> list[tuple[str, int]]:
    """Every invented message once, under the label it was written for."""
    rows = {}
    for i, label in enumerate(task.labels):
        for message in cache[key(label)]["messages"]:
            rows.setdefault(message, i)
    return list(rows.items())


def accuracy(task, rows, cache) -> dict:
    index = {name: i for i, name in enumerate(task.labels)}
    seen = [(y, cache[key(t)]) for t, y in rows if key(t) in cache]
    hits = np.array([index[a["label"]] == y for y, a in seen])
    usage = [a["usage"] for _, a in seen]
    return {
        "rows": len(seen),
        "accuracy": round(float(hits.mean()), 4),
        "mean_usage": {f: round(float(np.mean([u[f] for u in usage])), 1) for f in usage[0]},
    }


def first_labels(data_dir, dataset) -> dict:
    """Muse Spark's answers: open weights on Baseten where it answered, Spark 1.3 elsewhere."""
    return load_cache(cache_path(data_dir, "spark13", dataset)) | load_cache(
        cache_path(data_dir, "spark", dataset)
    )


def taught(task, rows, spark, wonder, below):
    """History relabelled by the teachers: Wonder's label where Spark was below `below`."""
    index = {name: i for i, name in enumerate(task.labels)}
    out = []
    for text, _ in rows:
        answer = spark[key(text)]
        if below is not None and answer["confidence"] < below:
            answer = wonder[key(text)]
        out.append((text, index[answer["label"]]))
    return out


def scored(model, tok, variants) -> dict:
    report = {}
    for name, t in variants:
        for shots in (8, 32):
            report[f"{name}-reader-{shots}"] = ic.score(
                t, ic.predict(model, tok, t, shots, near=True)
            )
        report[f"{name}-head"] = ic.score(t, ic.head(t))
    return report


def evaluate(settings, task, rows, below, model_name) -> dict:
    spark = first_labels(settings.data_dir, task.name)
    wonder = load_cache(cache_path(settings.data_dir, "wonder", task.name)) if below else {}
    ic.index([task])
    task = ic.capped(task, len(rows))
    assert task.train == rows
    teaching = taught(task, rows, spark, wonder, below)
    report = {
        "history": len(rows),
        "below": below,
        "relabelled": sum(spark[key(t)]["confidence"] < below for t, _ in rows) if below else 0,
        "teacher_accuracy_on_history": round(
            float(np.mean([a[1] == b[1] for a, b in zip(rows, teaching, strict=True)])), 4
        ),
    }
    model, tok = ic.load(settings.data_dir / "encoders" / model_name)
    taught_task = dataclasses.replace(task, train=teaching)
    return report | scored(model, tok, (("gold", task), ("taught", taught_task)))


def evaluate_examples(settings, task, n, model_name) -> dict:
    """Ren with only invented history: no real record labelled."""
    rows = invented(task, load_cache(cache_path(settings.data_dir, f"examples{n}", task.name)))
    made = dataclasses.replace(task, train=rows)
    ic.index([made])
    model, tok = ic.load(settings.data_dir / "encoders" / model_name)
    return {
        "per_label": n,
        "history": len(rows),
        "reader-0": ic.score(made, ic.predict(model, tok, made, 0)),
        **scored(model, tok, (("invented", made),)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mtop", "banking77"), default="mtop")
    parser.add_argument("--teacher", choices=tuple(TEACHERS), default="spark")
    parser.add_argument("--pilot", type=int, help="label this many history rows and score them")
    parser.add_argument("--rows", type=int, help="history rows to label or evaluate")
    parser.add_argument("--below", type=float, help="Wonder relabels Spark answers below this")
    parser.add_argument("--examples", type=int, help="invented messages per label")
    parser.add_argument("--model", default=MODEL, help="reader to evaluate")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--paid", action="store_true", help="allow paid calls")
    args = parser.parse_args()
    settings = Settings()
    task = ic.held_out(settings.data_dir)[args.dataset]
    out = Path("docs/results/s2")
    suffix = "" if args.model == MODEL else "-" + args.model
    if args.examples:
        if args.eval:
            report = {"dataset": args.dataset, "model": args.model}
            report |= evaluate_examples(settings, task, args.examples, args.model)
            path = out / f"teacher-{args.dataset}-examples{args.examples}{suffix}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        else:
            cache = write_examples(settings, task, args.examples, paid=args.paid)
            print(len(invented(task, cache)), "distinct messages")
        return
    rows = history(task, args.pilot or args.rows)
    if args.eval:
        report = {
            "dataset": args.dataset,
            "model": args.model,
            **evaluate(settings, task, rows, args.below, args.model),
        }
        name = f"teacher-{args.dataset}-{len(rows)}" + (f"-w{args.below}" if args.below else "")
        name += suffix
        (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.teacher == "spark13":
        done = load_cache(cache_path(settings.data_dir, "spark", task.name))
        rows = [r for r in rows if key(r[0]) not in done]
    if args.below is not None:
        spark = first_labels(settings.data_dir, task.name)
        rows = [
            r for r in rows if key(r[0]) in spark and spark[key(r[0])]["confidence"] < args.below
        ]
    cache = label(settings, task, rows, args.teacher, paid=args.paid)
    result = {"dataset": args.dataset, "teacher": args.teacher, **accuracy(task, rows, cache)}
    print(json.dumps(result))
    if args.pilot:
        path = out / "teacher-pilot.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        saved[f"{args.dataset}-{args.teacher}"] = result
        path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
