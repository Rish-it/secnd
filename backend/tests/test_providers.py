import csv
import json
from decimal import Decimal

import httpx
import pytest

from backend.config import Settings
from backend.muse import BudgetRun, MuseError, MuseItem, MuseOutputError, Usage, usage_cost
from backend.providers import MuseProviders, ledger_path, provider_settings


def settings():
    return Settings(
        muse_api_key="baseten-test",
        meta_api_key="meta-test",
        nebius_api_key="nebius-test",
        _env_file=None,
    )


@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_chain_fails_over_only_on_unavailability_and_keeps_separate_ledgers(tmp_path, failure):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body["model"]))
        if request.url.host != "api.studio.nebius.com":
            if failure == "timeout":
                raise httpx.ReadTimeout("private details", request=request)
            return httpx.Response(failure)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                "choices": [{"message": {"content": '{"label":"Travel","confidence":0.8}'}}],
            },
        )

    with MuseProviders(
        settings(),
        task="failover",
        rows=1,
        measured_tokens=Usage(prompt_tokens=100, completion_tokens=40),
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.zero_shot([MuseItem(item_id="a", text="Train")], ["Travel"])
        assert result[0].label == "Travel"
        assert client.last_provider == "nebius"
    assert [host for host, _ in calls] == ["api.meta.ai"] * 3 + ["inference.baseten.co"] * 3 + [
        "api.studio.nebius.com"
    ]
    for provider in ("meta", "baseten", "nebius"):
        row = list(csv.DictReader(ledger_path(tmp_path, provider).open()))[0]
        assert Decimal(row["cost_usd"]) > 0
        assert "test" not in ledger_path(tmp_path, provider).read_text()


@pytest.mark.parametrize(
    "status,content",
    [(200, '{"label":"Travel","confidence":0.8}'), (200, "invalid"), (401, "private")],
)
def test_valid_or_semantic_or_auth_result_never_falls_back(tmp_path, status, content):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(
            status,
            json={
                "model": "muse-spark-1.3",
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                "choices": [{"message": {"content": content}}],
            },
        )

    with MuseProviders(
        settings(),
        task="no-fallback",
        rows=1,
        measured_tokens=Usage(prompt_tokens=100, completion_tokens=40),
        directory=tmp_path,
        transport=httpx.MockTransport(handler),
    ) as client:
        if status == 401 or content == "invalid":
            with pytest.raises(MuseOutputError if status == 200 else MuseError):
                client.zero_shot([MuseItem(item_id="a", text="Train")], ["Travel"])
        else:
            assert (
                client.zero_shot([MuseItem(item_id="a", text="Train")], ["Travel"])[0].label
                == "Travel"
            )
    assert calls == ["api.meta.ai"]
    assert not ledger_path(tmp_path, "baseten").exists()


def test_prices_caps_and_provider_config(tmp_path):
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    for provider, cap in (("meta", 30), ("baseten", 30), ("nebius", 25)):
        assert usage_cost(usage, provider) == Decimal("5.50")
        configured = provider_settings(settings(), provider)
        assert "test" not in repr(configured)
        path = ledger_path(tmp_path, provider)
        with BudgetRun(
            configured,
            task="cap",
            rows=1,
            measured_tokens=usage,
            provider=provider,
            ledger_path=path,
        ) as run:
            run.record(Usage(prompt_tokens=cap * 800_000, completion_tokens=0))
        with pytest.raises(MuseError, match="total spending limit"):
            with BudgetRun(
                configured,
                task="cap",
                rows=1,
                measured_tokens=usage,
                provider=provider,
                ledger_path=path,
            ):
                pytest.fail("Provider total cap must survive restarts")


def test_budget_and_connection_errors_do_not_trigger_failover(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        raise httpx.ConnectError("private", request=request)

    for config in (settings(), settings().model_copy(update={"meta_budget_usd": Decimal(0)})):
        with MuseProviders(
            config,
            task="closed",
            rows=1,
            measured_tokens=Usage(prompt_tokens=100, completion_tokens=40),
            directory=tmp_path,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(MuseError):
                client.zero_shot([MuseItem(item_id="a", text="Train")], ["Travel"])
    assert calls == ["api.meta.ai"]
    assert not ledger_path(tmp_path, "baseten").exists()
