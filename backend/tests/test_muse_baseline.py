import json
import time
from decimal import Decimal

import httpx
import numpy as np
import pandas as pd
import pytest

from backend.config import Settings
from backend.data import split_manifest
from backend.muse import MuseError
from backend.muse_baseline import collect, compare, digest, read_predictions, sample, summarize


def frame():
    return pd.DataFrame(
        {
            "id": list(range(2000)),
            "text": ["Rail fare" if i % 2 else "Stationery paper" for i in range(2000)],
            "amount": 10.0,
            "date": pd.date_range("2020-01-01", periods=2000),
            "label": ["Travel" if i % 2 else "Office" for i in range(2000)],
            "account": "office",
            "merchant_key": [f"key-{i}" for i in range(2000)],
        }
    )


def test_comparison_uses_identical_fixed_test_items_without_training_leakage():
    data = frame()
    split, test = sample(data, rows=12)
    _, again = sample(data, rows=12)
    assert test["id"].tolist() == again["id"].tolist()
    assert set(split.loc[test.index]) == {"test"}
    assert all(isinstance(row_id, str) for row_id in test["id"])
    manifest = split_manifest("ipsa", data, split)
    assert len(manifest["hash"]) == 64
    compared = compare(data, split, test, budgets=(200, 500))
    assert [r["training_rows"] for r in compared] == [200, 500]
    assert all(r["accuracy"] == 1 for r in compared)
    assert all(0 <= r["ece"] <= 1 and np.isfinite(r["nll"]) for r in compared)
    assert all(r["cost_per_1000_usd"] is None for r in compared)


@pytest.mark.parametrize("invalid_id", [None, "3"])
def test_checkpoint_resumes_without_repeat_calls_and_accounts_for_all_paid_rows(
    tmp_path, invalid_id
):
    settings = Settings(muse_api_key="test-only-key", _env_file=None)
    test = frame().iloc[[1, 3, 5]].assign(id=["1", "3", "5"])
    plan = {"row_ids": test["id"].tolist(), "labels": ["Office", "Travel"]}
    (tmp_path / "smoke.json").write_text(
        json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    )
    requested = []
    called_at = []

    def handler(request):
        data = json.loads(request.content)
        supplied = json.loads(data["messages"][1]["content"])
        assert set(supplied["item"]) == {"item_id", "text", "amount", "date"}
        requested.append(supplied["item"]["item_id"])
        called_at.append(time.monotonic())
        content = json.dumps({"label": "Travel", "confidence": 0.9})
        if supplied["item"]["item_id"] == invalid_id:
            content = "invalid distribution"
        return httpx.Response(
            200,
            json={
                "model": data["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [{"message": {"content": content}}],
            },
        )

    for limit in (1, 2, 2):
        result = collect(
            settings,
            plan,
            test,
            tmp_path,
            limit=limit,
            git_sha="test-revision",
            transport=httpx.MockTransport(handler),
        )
        if limit == 1:
            with pytest.raises(MuseError, match="Estimated run cost"):
                collect(
                    settings.model_copy(update={"muse_budget_usd": Decimal("0.0005")}),
                    plan,
                    test,
                    tmp_path,
                    limit=1,
                    git_sha="test-revision",
                    transport=httpx.MockTransport(handler),
                )
            assert requested == ["1"]
    assert requested[0] == "1" and set(requested[1:]) == {"3", "5"}
    assert called_at[-1] - called_at[-2] >= 0.9
    assert set(result) == {"1", "3", "5"}
    measured = summarize(test, result, tmp_path / "spend.csv")
    assert measured["rows"] == 3
    assert measured["accuracy"] == (2 / 3 if invalid_id else 1)
    assert measured["invalid_outputs"] == (1 if invalid_id else 0)
    assert measured["nll"] is measured["ece"] is None
    assert measured["usage"]["prompt_tokens"] == 300
    assert Decimal(measured["cost_per_1000_usd"]) == Decimal("0.3375")
    assert "test-only-key" not in (tmp_path / "muse-predictions.jsonl").read_text()
    changed = {**plan, "labels": ["Different"]}
    with pytest.raises(ValueError, match="protocol changed"):
        collect(settings, changed, test, tmp_path, limit=2, git_sha="test-revision")
    record = json.loads((tmp_path / "muse-predictions.jsonl").read_text().splitlines()[0])
    record["experiment_hash"] = digest(changed)
    (tmp_path / "muse-predictions.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="fixed experiment"):
        read_predictions(tmp_path / "muse-predictions.jsonl", plan)
