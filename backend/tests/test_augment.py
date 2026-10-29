import json
from decimal import Decimal

import httpx
import pandas as pd
import pytest

from backend.augment import generate, read_cache, text_key, training_texts
from backend.config import Settings
from backend.data import split_dataset
from backend.eval import training_rows
from backend.muse import MuseOutputError


def test_selection_uses_only_fixed_training_rows_for_each_budget_and_seed():
    frame = pd.DataFrame(
        {
            "id": range(2000),
            "text": [f"text {i}" for i in range(2000)],
            "date": pd.date_range("2020-01-01", periods=2000),
            "merchant_key": [f"key{i}" for i in range(2000)],
            "label": "NEVER SEND",
        }
    )
    texts, manifests, blanks = training_texts(frame)
    expected = set()
    for seed in range(5):
        split = split_dataset(frame, seed)
        for budget in (100, 500):
            train = training_rows(frame, split, budget, seed)
            assert set(split.loc[train.index]) <= {"seed", "stream"}
            expected.update(train["text"])
    assert set(texts) == expected
    assert len(manifests) == 10 and blanks == 0


def test_cache_resumes_atomically_without_sending_labels_or_repeating_texts(tmp_path):
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        supplied = json.loads(payload["messages"][1]["content"])
        assert all(set(row) == {"key", "text"} for row in supplied)
        assert "NEVER SEND" not in request.content.decode()
        calls.extend(row["text"] for row in supplied)
        return httpx.Response(
            200,
            json={
                "model": "muse-spark-1.3",
                "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "key": row["key"],
                                            "paraphrases": [
                                                row["text"] + " first",
                                                row["text"] + " second",
                                            ],
                                        }
                                        for row in supplied
                                    ]
                                }
                            )
                        }
                    }
                ],
            },
        )

    config = Settings(meta_api_key="meta-test", _env_file=None)
    for limit in (1, None, None):
        result = generate(
            ["train fare", "office rent", "train fare", ""],
            config,
            directory=tmp_path / "cache",
            ledger_directory=tmp_path / "ledgers",
            limit=limit,
            transport=httpx.MockTransport(handler),
        )
    assert calls == ["train fare", "office rent"]
    assert result["complete"] and result["paraphrases"] == 4
    cache = read_cache(tmp_path / "cache")
    assert set(cache) == {text_key("train fare"), text_key("office rent")}
    assert not list((tmp_path / "cache").glob("tmp*"))
    before = len(calls)
    with pytest.raises(MuseOutputError):
        generate(
            ["new text"],
            config,
            directory=tmp_path / "cache",
            ledger_directory=tmp_path / "ledgers",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "model": "muse-spark-1.3",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                        "choices": [{"message": {"content": '{"items":[]}'}}],
                    },
                )
            ),
        )
    assert len(read_cache(tmp_path / "cache")) == 2 and len(calls) == before
    assert Decimal("0") < sum(
        Decimal(row.split(",")[-2])
        for row in (tmp_path / "ledgers/spend-meta.csv").read_text().splitlines()[1:]
    )


def test_malformed_json_is_retried_within_same_provider_and_billed(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        supplied = json.loads(json.loads(request.content)["messages"][1]["content"])
        content = (
            '{"items":[] channel_subscribe}'
            if len(calls) == 1
            else json.dumps(
                {
                    "items": [
                        {
                            "key": row["key"],
                            "paraphrases": ["fare for the train", "rail ticket cost"],
                        }
                        for row in supplied
                    ]
                }
            )
        )
        return httpx.Response(
            200,
            json={
                "model": "muse-spark-1.3",
                "usage": {"prompt_tokens": 100, "completion_tokens": 100},
                "choices": [{"message": {"content": content}}],
            },
        )

    result = generate(
        ["train fare"],
        Settings(meta_api_key="meta-test", _env_file=None),
        directory=tmp_path / "cache",
        ledger_directory=tmp_path / "ledger",
        transport=httpx.MockTransport(handler),
    )
    assert result["complete"] and calls == ["api.meta.ai", "api.meta.ai"]
    assert result["new_spend_usd"]["meta"] == "0.00110"
    assert len(read_cache(tmp_path / "cache")) == 1
