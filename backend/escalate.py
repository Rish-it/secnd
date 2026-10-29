"""Muse Spark as Ren's second opinion and as its teacher, on MTOP with 200 labels of history.

Escalation: for each record, Muse Spark sees Ren's top options and the 8 nearest past decisions
and picks one (or none). Where it agrees with Ren, that agreement ranks the record above every
disagreement, and the unchanged 99% gate certifies on calib alone.

Teacher: Muse Spark labels the 200 history messages with no human label, and the reader reads
those as its history.

Paid calls need --label and stay inside a $12 task cap; answers are cached by text hash (no text
stored), so a rerun never pays twice.

    python -m backend.escalate --label --pilot 5        # measure cost per call first
    python -m backend.escalate --label                  # cert half + 1,000 test rows
    python -m backend.escalate --label --teacher        # Muse labels the 200 history rows
"""

import argparse
import dataclasses
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import numpy as np

from backend import incontext as ic
from backend.calibrate import choose_threshold
from backend.config import Settings
from backend.handoff import task_spend
from backend.muse import (
    SPEND_PATH,
    BudgetRun,
    LabelledItem,
    MuseClient,
    MuseError,
    MuseItem,
    MuseOutputError,
    Usage,
)

TASK = {"escalate": "s2-mtop-escalate", "teacher": "s2-mtop-teacher"}
CAP = Decimal("12")
HISTORY = 200
OPTIONS = 5
EXAMPLES = 8
TEST_ROWS = 1000
# Eight parallel calls hit the provider's rate limit after ~50 requests; two do not.
WORKERS = 2
MODEL = "ic-ettin-150m-s6000-near"
# Until a pilot has measured real usage, estimate each call conservatively.
GUESS = Usage(prompt_tokens=2000, completion_tokens=2048)


def key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def setup(data_dir: Path):
    """MTOP with 200 history rows, MiniLM vectors, and the rows escalation asks about."""
    task = ic.held_out(data_dir)["mtop"]
    ic.index([task])
    task = ic.capped(task, HISTORY)
    half = np.random.default_rng(0).permutation(len(task.calib))
    cert = np.sort(half[len(task.calib) // 2 :])  # the certify half that `ic.score` uses
    test = np.sort(np.random.default_rng(1).permutation(len(task.test))[:TEST_ROWS])
    return task, cert, test


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def ask(settings, cache_path, jobs, *, task_name, label, allow_paid):
    """Run `label(job)` for every job not yet cached; jobs are (key, payload)."""
    cache = load_cache(cache_path)
    missing = [(k, job) for k, job in jobs if k not in cache]
    print(f"{task_name}: {len(jobs) - len(missing)} cached, {len(missing)} to ask", flush=True)
    if not missing:
        return cache
    if not allow_paid:
        raise SystemExit("Cache incomplete; --label enables paid calls.")
    seen = [Usage(**v["usage"]) for v in cache.values()]
    measured = (
        Usage(
            prompt_tokens=int(1.5 * np.mean([u.prompt_tokens for u in seen])),
            completion_tokens=int(1.5 * np.mean([u.completion_tokens for u in seen])),
        )
        if seen
        else GUESS
    )
    with BudgetRun(
        settings,
        task=task_name,
        rows=len(missing),
        provider="baseten",
        measured_tokens=measured,
        ledger_path=SPEND_PATH,
    ) as run:
        run.budget = min(
            run.budget,
            CAP
            - task_spend(SPEND_PATH, TASK["escalate"])
            - task_spend(SPEND_PATH, TASK["teacher"]),
        )
        print(f"estimate ${run.estimate:.2f}, allowed ${run.budget:.2f}", flush=True)
        if run.estimate > run.budget:
            raise MuseError("Estimate exceeds the remaining task cap.")
        with MuseClient(settings, run=run) as client:

            def one(job):
                for attempt in range(2):
                    try:
                        return label(client, job)
                    except MuseOutputError:
                        if attempt:
                            raise

            failures = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as workers:
                pending = {workers.submit(one, job): k for k, job in missing}
                for n, future in enumerate(as_completed(pending), 1):
                    try:
                        result = future.result()
                    except MuseError:
                        failures += 1
                        continue
                    cache[pending[future]] = result.model_dump(
                        mode="json", include={"label", "confidence", "usage", "latency_ms"}
                    )
                    if n % 50 == 0 or n == len(missing):
                        staged = cache_path.with_suffix(".tmp")
                        staged.write_text(json.dumps(cache, sort_keys=True))
                        os.replace(staged, cache_path)
                        print(f"  {n}/{len(missing)} ${run.cost_usd:.2f}", flush=True)
        staged = cache_path.with_suffix(".tmp")
        staged.write_text(json.dumps(cache, sort_keys=True))
        os.replace(staged, cache_path)
        print(f"{task_name}: spent ${run.cost_usd:.4f}, {failures} failed", flush=True)
    return cache


def escalation_jobs(task, p, rows, part):
    """One job per record: Ren's top options and the nearest history decisions."""
    texts, vecs = getattr(task, part), getattr(task, f"{part}_vecs")
    jobs = []
    for j in rows:
        text = texts[j][0]
        top = np.argsort(-p[j])[:OPTIONS]
        shots = ic.nearest(task, vecs[j], EXAMPLES, skip=text)
        jobs.append(
            (
                key(text),
                (
                    MuseItem(item_id=key(text)[:32], text=text),
                    [task.labels[o] for o in top],
                    [
                        LabelledItem(item_id=str(i), text=t, label=task.labels[y])
                        for i, (t, y) in enumerate(shots)
                    ],
                ),
            )
        )
    return jobs


def cascade(task, p, answers, rows, part):
    """Ren's answer and a score: Ren's confidence, plus 1 where Muse picked the same label."""
    texts = getattr(task, part)
    index = {label: i for i, label in enumerate(task.labels)}
    pred, score, muse = p[rows].argmax(1), p[rows].max(1), []
    for j in rows:
        label = answers.get(key(texts[j][0]), {}).get("label", "none")
        muse.append(index.get(label, -1))
    muse = np.array(muse)
    return pred, score + (muse == pred), muse


def gate(pred, score, y):
    """`choose_threshold` reads argmax and max, so encode each row as its prediction and score."""
    p = np.zeros((len(pred), int(max(pred.max(), y.max())) + 1))
    p[np.arange(len(pred)), pred] = score
    return choose_threshold(y, p)


def escalation_report(task, p_calib, p_test, cert, test, answers) -> dict:
    yc, yt = ic.labels_of(task.calib)[cert], ic.labels_of(task.test)[test]
    out = {}
    for name, use_muse in (("ren", False), ("ren+muse", True)):
        pc, sc, _ = cascade(task, p_calib, answers if use_muse else {}, cert, "calib")
        pt, st, muse = cascade(task, p_test, answers if use_muse else {}, test, "test")
        tau, n_auto, _, lower = gate(pc, sc, yc)
        auto = st >= tau if tau is not None else np.zeros(len(yt), bool)
        out[name] = {
            "certified_coverage_test": round(float(auto.mean()), 4),
            "precision_auto_test": round(float((pt == yt)[auto].mean()), 4) if auto.any() else None,
            "cert_accepted": n_auto,
            "cert_cp_lower": round(lower, 4),
        }
    agree = muse == pt
    out["test"] = {
        "rows": len(test),
        "ren_accuracy": round(float((pt == yt).mean()), 4),
        "muse_accuracy": round(float((muse == yt).mean()), 4),
        "muse_none": round(float((muse == -1).mean()), 4),
        "gold_in_ren_top_options": round(
            float(
                np.mean(
                    [y in np.argsort(-p_test[j])[:OPTIONS] for j, y in zip(test, yt, strict=True)]
                )
            ),
            4,
        ),
        "agree": round(float(agree.mean()), 4),
        "precision_when_agree": round(float((pt == yt)[agree].mean()), 4),
        "muse_accuracy_when_disagree": round(float((muse == yt)[~agree].mean()), 4),
        "ren_accuracy_when_disagree": round(float((pt == yt)[~agree].mean()), 4),
    }
    return out


def teacher_report(data_dir: Path, model_name: str, task, cache: dict) -> dict:
    """The reader and the logistic head with Muse Spark's labels as history, no human label."""
    import torch
    from transformers import AutoTokenizer

    from backend.features import device

    folder = data_dir / "encoders" / model_name
    model = ic.Reader(str(folder / "encoder"))
    model.scorer.load_state_dict(torch.load(folder / "scorer.pt", map_location="cpu"))
    model = model.to(device()).eval()
    tok = AutoTokenizer.from_pretrained(folder / "encoder")
    index = {label: i for i, label in enumerate(task.labels)}
    taught = dataclasses.replace(
        task, train=[(t, index[cache[key(t)]["label"]]) for t, _ in task.train]
    )
    return {
        "teacher_accuracy_on_history": round(
            float(np.mean([a[1] == b[1] for a, b in zip(task.train, taught.train, strict=True)])),
            4,
        ),
        "reader-8": ic.score(taught, ic.predict(model, tok, taught, 8, near=True)),
        "head": ic.score(taught, ic.head(taught)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--label", action="store_true", help="allow paid calls")
    parser.add_argument("--pilot", type=int, help="ask only this many test rows")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--teacher-eval", action="store_true", help="score Muse-labelled history")
    args = parser.parse_args()
    settings = Settings()
    data_dir = settings.data_dir
    task, cert, test = setup(data_dir)
    cache_dir = data_dir / "muse"
    cache_dir.mkdir(exist_ok=True)
    if args.teacher_eval:
        cache = load_cache(cache_dir / "teacher-mtop-200.json")
        report = {"model": args.model, "history": HISTORY, "muse_model": settings.muse_model}
        report |= teacher_report(data_dir, args.model, task, cache)
        path = Path("docs/results/s2/muse-teacher-mtop.json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.teacher:
        jobs = [(key(t), MuseItem(item_id=key(t)[:32], text=t)) for t, _ in task.train]
        ask(
            settings,
            cache_dir / "teacher-mtop-200.json",
            jobs,
            task_name=TASK["teacher"],
            label=lambda client, item: client.zero_shot([item], task.labels)[0],
            allow_paid=args.label,
        )
        return
    saved = np.load(data_dir / "predictions" / args.model / f"mtop-{HISTORY}-reader-8.npz")
    p_calib, p_test = (saved[k].astype(np.float64) for k in ("p_calib", "p_test"))
    jobs = (
        escalation_jobs(task, p_test, test[: args.pilot], "test")
        if args.pilot
        else (
            escalation_jobs(task, p_calib, cert, "calib")
            + escalation_jobs(task, p_test, test, "test")
        )
    )
    answers = ask(
        settings,
        cache_dir / f"escalate-{args.model}.json",
        jobs,
        task_name=TASK["escalate"],
        label=lambda client, job: client.escalate(*job),
        allow_paid=args.label,
    )
    if args.pilot:
        usage = [answers[k]["usage"] for k, _ in jobs if k in answers]
        print(
            json.dumps(
                {
                    "calls": len(usage),
                    "mean_usage": {f: float(np.mean([u[f] for u in usage])) for f in usage[0]},
                }
            )
        )
        return
    report = {
        "model": args.model,
        "history": HISTORY,
        "muse_model": settings.muse_model,
        **escalation_report(task, p_calib, p_test, cert, test, answers),
    }
    path = Path("docs/results/s2/muse-escalate-mtop.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
