import csv
import json
from decimal import Decimal

import httpx
import pytest

from backend.config import Settings
from backend.muse import BudgetRun, MuseError, Usage, smoke, usage_cost


def settings():
    return Settings(muse_api_key="test-only-key", _env_file=None)


def test_smoke_sends_one_bounded_request_and_returns_only_metadata(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-only-key"
        assert body["model"] == "meta-models/muse-spark-open-1026"
        assert body["max_tokens"] == 64
        assert "response_format" not in body
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "ready"}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            },
        )

    result = smoke(
        settings(), transport=httpx.MockTransport(handler), ledger_path=tmp_path / "spend.csv"
    )
    assert len(requests) == 1
    assert result.usage.reasoning_tokens == 8
    assert "ready" not in result.model_dump_json()
    assert "test-only-key" not in result.model_dump_json()


def test_smoke_sanitizes_provider_errors_and_requires_a_key(monkeypatch, tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="sensitive response"))
    with pytest.raises(MuseError, match="HTTP 401") as caught:
        smoke(settings(), transport=transport, ledger_path=tmp_path / "spend.csv")
    assert "sensitive" not in str(caught.value)
    monkeypatch.delenv("SECND_MUSE_API_KEY", raising=False)
    with pytest.raises(MuseError, match="SECND_MUSE_API_KEY"):
        smoke(
            Settings(muse_api_key="", _env_file=None),
            transport=transport,
            ledger_path=tmp_path / "spend.csv",
        )


def test_budget_hand_checked_cost_and_failed_run_are_recorded(tmp_path):
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, reasoning_tokens=500_000)
    assert usage_cost(usage) == Decimal("5.50")
    ledger = tmp_path / "spend.csv"
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        BudgetRun(
            settings(), task="baseline", rows=1, measured_tokens=usage, ledger_path=ledger
        ) as run,
    ):
        run.check(usage)
        run.record(usage)
        raise RuntimeError("interrupted")
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert Decimal(rows[0]["cost_usd"]) == Decimal("5.50")
    assert rows[0]["reasoning_tokens"] == "500000"
    assert "test-only-key" not in ledger.read_text()


def test_budget_rejects_overestimate_before_http_and_caps_runtime_spend(tmp_path):
    ledger = tmp_path / "spend.csv"
    settings_ = Settings(muse_api_key="test-only-key", muse_budget_usd="0", _env_file=None)
    calls = []
    with pytest.raises(MuseError, match="Estimated run cost"):
        smoke(
            settings_,
            transport=httpx.MockTransport(lambda request: calls.append(request)),
            ledger_path=ledger,
        )
    assert calls == []
    assert not ledger.exists()
    unit = Usage(prompt_tokens=1_000_000, completion_tokens=0)
    with BudgetRun(
        settings(), task="baseline", rows=1, measured_tokens=unit, ledger_path=ledger
    ) as run:
        run.record(Usage(prompt_tokens=8_000_000, completion_tokens=0))
        with pytest.raises(MuseError, match="next request"):
            run.check(unit)


def test_budget_total_limit_and_corrupt_ledger_fail_closed(tmp_path):
    ledger = tmp_path / "spend.csv"
    usage = Usage(prompt_tokens=8_000_000, completion_tokens=0)
    for _ in range(3):
        with BudgetRun(
            settings(), task="baseline", rows=1, measured_tokens=usage, ledger_path=ledger
        ) as run:
            run.record(usage)
    with (
        pytest.raises(MuseError, match="total spending limit"),
        BudgetRun(settings(), task="baseline", rows=1, measured_tokens=usage, ledger_path=ledger),
    ):
        pytest.fail("Run must be rejected")
    ledger.write_text("invalid,header\n")
    with (
        pytest.raises(MuseError, match="invalid header"),
        BudgetRun(settings(), task="baseline", rows=1, measured_tokens=usage, ledger_path=ledger),
    ):
        pytest.fail("Run must be rejected")
