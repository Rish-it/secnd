"""Amendment 4: apply the panel's MTOP votes to calibration and refit the gate from saved
probabilities. Test labels are never changed. BANKING77 failed its control rule, so it stays."""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from backend import incontext as ic
from backend.panel_votes import PANEL, tally

MODELS = ("ic-ettin-150m-s6000-near", "ic-ettin-400m-s6000-near")
CONDITIONS = (
    *(f"{b}-reader-{s}" for b in ("200", "1000", "all") for s in (8, 32)),
    "reader-0",
)


def audited(rows, changed, key, labels):
    """`rows` with each changed label moved to the panel's option; only `changed` ids differ."""
    out = list(rows)
    for change in changed:
        entry = key[change["id"]]
        i = entry["calib_index"]
        assert labels[rows[i][1]] == entry["gold"], change["id"]
        out[i] = (rows[i][0], labels.index(entry["candidate"]))
    return out


def load_audit(csv_path, key_path, votes_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8", newline="")))
    key = json.loads(Path(key_path).read_text())
    votes = {(r["model"], r["id"]): r["answer"] for r in map(json.loads, open(votes_path))}
    return rows, votes, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".data"))
    parser.add_argument("--audit", required=True, type=Path, help="folder with CSVs, keys, votes")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows, votes, key = load_audit(
        args.audit / "calib-audit.csv",
        args.audit / "calib-audit-key.json",
        args.audit / "votes-mtop.jsonl",
    )
    mtop = tally(rows, votes, key)
    assert not mtop["biased"]
    banking = tally(
        *load_audit(
            args.audit / "calib-audit-banking77.csv",
            args.audit / "calib-audit-banking77-key.json",
            args.audit / "votes-banking77.jsonl",
        )
    )

    task = ic.held_out(args.data)["mtop"]
    fixed = dataclasses.replace(
        task, calib=audited(task.calib, mtop["changed"], key, list(task.labels))
    )
    assert fixed.test == task.test
    for change in mtop["changed"]:
        row = next(r for r in rows if r["id"] == change["id"])
        change.update(message=row["message"], new=key[change["id"]]["candidate"])

    results = {}
    for model in MODELS:
        results[model] = {}
        for condition in CONDITIONS:
            saved = np.load(args.data / "predictions" / model / f"mtop-{condition}.npz")
            assert (saved["y_calib"] == ic.labels_of(task.calib)).all()
            assert (saved["y_test"] == ic.labels_of(task.test)).all()
            p = {part: saved[f"p_{part}"].astype(np.float64) for part in ic.PARTS}
            results[model][condition] = dict(
                dataset_labels=ic.score(task, p), audited_calib_labels=ic.score(fixed, p)
            )
            print(model, condition, flush=True)

    args.out.write_text(
        json.dumps(
            dict(
                kind="model audit, not a human audit",
                panel=PANEL,
                test_labels="unchanged; precision is measured against them",
                mtop=mtop,
                banking77=dict(
                    used=False,
                    reason="controls overturned above the 2% rule",
                    controls=banking["controls"],
                    controls_overturned=banking["controls_overturned"],
                ),
                results=results,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
