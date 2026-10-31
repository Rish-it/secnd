"""Amendment 4's written Meta-app tasks: how often Ren agrees with the model panel's label.

These tasks are written, not collected, and the labels are model labels, so this is agreement with
a panel and never accuracy on real data.
"""

from collections import Counter

import numpy as np

from backend.typed import Decision, predict, train

UNLABELLED = ("", "insufficient information")


def decision(row: dict) -> Decision:
    keys = row["options"].split(" | ")
    gold = np.full(len(keys), 0.1 / len(keys))
    gold[keys.index(row["panel_label"])] += 0.9
    return Decision(
        case=row["id"],
        kind="choice",
        text=f"choice {row['decision_type']} decision: Choose the best action.",
        keys=keys,
        meanings=keys,
        state=row["text"],
        gold=gold,
        label=keys.index(row["panel_label"]),
    )


def agreement(rows: list[dict], picks: list[str]) -> dict:
    """Agreement per decision type, next to always picking that type's most common panel label."""
    out = {"kind": "agreement with a model panel on written tasks, not accuracy on real data"}
    for kind in (*sorted({r["decision_type"] for r in rows}), "all"):
        part = [i for i, r in enumerate(rows) if kind in ("all", r["decision_type"])]
        labels = [rows[i]["panel_label"] for i in part]
        hits = sum(picks[i] == rows[i]["panel_label"] for i in part)
        common = sum(
            Counter(r["panel_label"] for r in rows if r["decision_type"] == t).most_common(1)[0][1]
            for t in {rows[i]["decision_type"] for i in part}
        )
        out[kind] = dict(
            rows=len(labels),
            agreed=hits,
            agreement=round(hits / len(labels), 4),
            most_common_baseline=round(common / len(labels), 4),
        )
    return out


def score(model, tok, rows: list[dict], init: str) -> dict:
    labelled = [r for r in rows if r["panel_label"] not in UNLABELLED]
    ds = [decision(r) for r in labelled]
    picks = [d.keys[int(p.argmax())] for d, p in zip(ds, predict(model, tok, ds), strict=True)]
    return dict(
        model=init,
        rows_total=len(rows),
        rows_without_panel_label=len(rows) - len(labelled),
        **agreement(labelled, picks),
    )


def tune(init: str, data_dir, history: list[dict], rows: list[dict], epochs=6, seed=0) -> dict:
    """Fine-tune on panel-labelled history; validation is a slice of history, `rows` only score."""
    labelled = [decision(r) for r in history if r["panel_label"] not in UNLABELLED]
    picked = np.random.default_rng(seed).permutation(len(labelled))
    cut = len(labelled) // 6
    val = [labelled[i] for i in picked[:cut]]
    fit = [labelled[i] for i in picked[cut:]]
    test = [decision(r) for r in rows if r["panel_label"] not in UNLABELLED]
    probs, report, model, tok = train(
        init, data_dir, epochs, seed, batch_size=8, splits=(fit, val, test), keep=True
    )
    picks = [d.keys[int(p.argmax())] for d, p in zip(test, predict(model, tok, test), strict=True)]
    return dict(
        model=f"typed-{init}-e{epochs}-s{seed}",
        fit=len(fit),
        val=len(val),
        val_history=report["history"],
        **agreement([r for r in rows if r["panel_label"] not in UNLABELLED], picks),
    )


if __name__ == "__main__":
    rows = [
        {"decision_type": "ad", "panel_label": "pause"},
        {"decision_type": "ad", "panel_label": "pause"},
        {"decision_type": "ad", "panel_label": "keep"},
        {"decision_type": "inbox", "panel_label": "reply now"},
    ]
    result = agreement(rows, ["pause", "keep", "keep", "reply now"])
    assert result["ad"] == dict(rows=3, agreed=2, agreement=0.6667, most_common_baseline=0.6667)
    assert result["all"]["agreed"] == 3 and result["all"]["most_common_baseline"] == 0.75
    assert (
        decision(
            {
                "id": "x",
                "decision_type": "ad",
                "options": "pause | keep",
                "text": "t",
                "panel_label": "keep",
            }
        ).label
        == 1
    )
