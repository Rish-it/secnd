import csv
import json

import pandas as pd
import pytest

from backend.config import Settings
from backend.handoff import TASK, collect, evaluate, select
from backend.muse import SPEND_FIELDS, LabelResult, MuseError, Usage


def data():
    rows = []
    for i in range(600):
        label = "a" if i % 2 else "b"
        rows.append(
            dict(
                id=str(i),
                text=("apple fruit " if label == "a" else "train rail ") + str(i),
                label=label,
                amount=None,
                date=None,
                merchant_key="",
            )
        )
    frame = pd.DataFrame(rows)
    split = pd.Series(["seed"] * 200 + ["stream"] * 200 + ["calib"] * 100 + ["test"] * 100)
    return frame, split


class Teacher:
    calls = []

    def __init__(self, settings, run):
        self.run = run

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cold_start(self, items, labels):
        self.calls.append(items[0].item_id)
        usage = Usage(prompt_tokens=100, completion_tokens=20)
        self.run.record(usage)
        return [
            LabelResult(
                item_id=items[0].item_id,
                label="a" if "apple" in items[0].text else "b",
                confidence=0.9,
                usage=usage,
                latency_ms=1,
            )
        ]


def test_stream_only_resumable_cache_and_cap(tmp_path):
    frame, split = data()
    settings = Settings(_env_file=None)
    path, ledger = tmp_path / "cache.json", tmp_path / "spend.csv"
    Teacher.calls = []
    selected, labels = collect(
        frame, split, settings, path, ledger, rows=6, allow_paid=True, client_factory=Teacher
    )
    assert len(labels) == 6 and split.loc[selected.index].eq("stream").all()
    assert len(Teacher.calls) == 6
    collect(frame, split, settings, path, ledger, rows=6, client_factory=Teacher)
    assert len(Teacher.calls) == 6
    assert not set(Teacher.calls) & set(frame.loc[split.isin(["calib", "test"]), "id"])
    cached = json.loads(path.read_text())
    cached["items"][str(selected.id.iloc[0])]["text_sha256"] = "wrong"
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="does not match"):
        collect(frame, split, settings, path, ledger, rows=6)
    path.unlink()
    with ledger.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=SPEND_FIELDS)
        writer.writeheader()
        writer.writerow({**dict.fromkeys(SPEND_FIELDS, "0"), "task": TASK, "cost_usd": "9.99"})
    with pytest.raises(MuseError, match=r"remaining \$10 task cap"):
        collect(
            frame, split, settings, path, ledger, rows=6, allow_paid=True, client_factory=Teacher
        )
    assert len(Teacher.calls) == 6


def test_handoff_shares_and_frozen_holdouts(tmp_path):
    frame, split = data()
    pseudo = select(frame, split, 20)
    result = evaluate(
        frame,
        split,
        pseudo,
        pseudo.label.tolist(),
        featurizer="char_tfidf",
        rounds=2,
        batch=5,
        cache_dir=tmp_path,
    )
    assert len(result["history"]) == 3
    first, last = result["history"][0], result["history"][-1]
    assert first["human_labels"] == 0 and first["shares"]["muse"] == 1
    assert last["labels"] == 20 + last["human_labels"]
    for row in result["history"]:
        assert sum(row["shares"].values()) == pytest.approx(1)
        assert row["pending"] >= 0
    assert [c["human_budget"] for c in result["comparisons"]] == [50, 200]
    assert result["comparisons"][0]["without_teacher"]["labels"] == 50
    assert result["comparisons"][0]["with_teacher"]["labels"] == 70
    with pytest.raises(ValueError, match="only from stream"):
        evaluate(frame, split, frame[split == "test"].head(20), pseudo.label.tolist())
