import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from backend import data, eval


def test_training_rows_take_seed_then_earliest_stream():
    frame = data._frame(
        pd.Series([f"row {i}" for i in range(6)]),
        pd.Series(["a", "b"] * 3),
        date=pd.date_range("2024-01-01", periods=6, freq="D").to_series().iloc[::-1].to_numpy(),
    )
    split = pd.Series(["stream", "stream", "seed", "stream", "calib", "test"], index=frame.index)

    rows = eval.training_rows(frame, split, budget=3, seed=0)

    # Dates run backwards, so the earliest stream rows are ids 3 then 1, not 0 then 1.
    assert rows["id"].tolist() == [2, 3, 1]


def test_lookup_accuracy_scores_a_label_revealing_column():
    column = pd.Series(["gas", "gas", "water", "gas", "water", ""])
    label = pd.Series(["utilities", "utilities", "utilities", "utilities", "rent", "rent"])
    train = pd.Series([True, True, True, False, False, False])
    # Test rows: gas -> utilities (right), water -> utilities (wrong), empty is skipped.
    assert eval._lookup_accuracy(column, label, train) == 0.5


@pytest.fixture
def local_dataset(tmp_path, monkeypatch):
    def write(*, wrong_confident=False):
        rows = 400
        frame = pd.DataFrame(
            {
                "Details": ["Train" if i < 360 else "New vendor" for i in range(rows)],
                "Cost Type": [
                    ("Travel" if i < 289 else "Software")
                    if wrong_confident
                    else ("Travel" if i % 2 else "Software")
                    for i in range(rows)
                ],
                "Amount Claimed": [25] * rows,
                "Date": pd.date_range("2024-01-01", periods=rows).strftime("%d/%m/%Y"),
                "Parliamentary ID": ["office-1"] * rows,
            }
        )
        folder = tmp_path / "datasets/ipsa"
        folder.mkdir(parents=True, exist_ok=True)
        sources = []
        for name, part in (("a.csv", frame.iloc[:200]), ("b.csv", frame.iloc[200:])):
            path = folder / name
            part.to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sources.append((name, "https://unused.example/" + name, digest))
        monkeypatch.setitem(data.DATASETS, "ipsa", sources)
        monkeypatch.setenv("SECND_DATA_DIR", str(tmp_path))
        return {name: digest for name, _, digest in sources}

    return write


def test_card_has_flat_api_fields_and_hashes_all_sources(local_dataset):
    sources = local_dataset()
    result = eval.card("ipsa", "majority", budget=200)
    assert result["model_version"] == "ipsa-majority-b200-s0"
    assert (
        result["data_hash"]
        == hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert len(result["split_hash"]) == 64
    assert result["threshold"] == 0.99
    assert result["coverage"] == 0
    assert result["cp_lower_bound"] is None
    assert result["precision"] is None
    assert set(result["classwise_ece"]) == {"Travel", "Software"}
    assert result["seen_merchant"]["count"] + result["unseen_merchant"]["count"] == 60
    assert all(value["count"] >= 20 for value in result["worst_classes"].values())
    assert isinstance(result["peak_rss_bytes"], int)
    assert result["peak_rss_bytes"] > 0
    assert result["predict_ms"]["p99"] >= result["predict_ms"]["p50"] >= 0
    assert not {"provenance", "quality", "claim", "slices", "speed"} & result.keys()
    json.dumps(result, allow_nan=False)


def test_card_distinguishes_no_observations_from_zero_successes(local_dataset):
    local_dataset(wrong_confident=True)
    result = eval.card("ipsa", "majority", budget=200)
    assert result["coverage"] == 1
    assert result["precision"] == 0
    assert result["cp_lower_bound"] == 0


def test_quality_reports_each_class_ece():
    result = eval.quality(
        np.array([0, 1, 0, 1]), np.array([[0.9, 0.1]] * 4), ["Travel", "Software"]
    )
    assert result["classwise_ece"] == {"Travel": 0.4, "Software": 0.4}
