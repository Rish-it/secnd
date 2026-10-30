import json
from decimal import Decimal

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.cold_start import ColdStartService
from backend.config import Settings
from backend.reader_serving import ReaderUnavailable
from backend.store import Store
from backend.workspace import WorkspaceService

API = "/api/v1/workspaces/default"


class FakeReader:
    def __init__(self, _):
        self.calls = []

    def validate_labels(self, labels):
        pass

    def predict(self, labels, history, frame):
        self.calls.append(history)
        return [
            dict(
                item_id=str(row.id),
                label=labels[0],
                probabilities={labels[0]: 0.7, labels[1]: 0.3},
                prob=0.7,
                auto=False,
                threshold=None,
                model_version="ren-reader-test",
                reasons={"neighbours": [], "merchant_memory": False},
                item={"item_id": str(row.id), "text": row.text},
                spot_check=False,
            )
            for row in frame.itertuples()
        ]


def provider(calls, *, tokens=100, fail=False):
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["model"] == "muse-wonder-1.0"
        if fail:
            return httpx.Response(401, text="private provider response")
        label = json.loads(payload["messages"][1]["content"])["label"]
        return httpx.Response(
            200,
            json={
                "model": "muse-wonder-1.0",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"messages": [f"{label} example {i}" for i in range(10)]}
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": tokens},
            },
        )

    return httpx.MockTransport(handle)


@pytest.fixture
def live(tmp_path, monkeypatch):
    calls = []
    settings = Settings(
        data_dir=tmp_path,
        frontend_dist=tmp_path / "none",
        meta_api_key="fake-test-key",
        _env_file=None,
    )
    store = Store(settings.database_path)
    cold = ColdStartService(store, settings, reader_factory=FakeReader, transport=provider(calls))
    cold_ledger = tmp_path / "spend-meta.csv"
    # Preserve the production ledger path; tests always use their own file.
    monkeypatch.setattr(ColdStartService, "ledger", property(lambda _: cold_ledger))
    service = WorkspaceService(store, tmp_path)
    with TestClient(
        create_app(settings, workspace_service=service, cold_start_service=cold)
    ) as client:
        yield client, cold, calls


def finish(cold, workspace="default"):
    cold._pending[workspace].result(timeout=5)
    return cold.state(workspace)


def begin(client, labels=None, workspace="default"):
    return client.post(
        f"/api/v1/workspaces/{workspace}/cold-start",
        json={"labels": labels or ["Travel", "Food"], "paid": True},
    )


def test_label_only_history_edits_predict_and_answer_never_call_teacher(live):
    client, cold, calls = live
    response = begin(client)
    assert response.status_code == 200 and response.headers["X-Secnd-Mode"] == "live"
    state = finish(cold)
    assert state["status"] == "ready" and len(state["history"]) == 20
    assert len(calls) == 2
    assert all("Travel" in x["messages"][0]["content"] for x in calls)
    config = cold.store.workspace("default")
    assert config["calib"] == [] and config["mode"] == "reader"
    first = state["history"][0]
    edited = client.patch(
        API + "/cold-start/history/" + first["id"],
        json={"text": "an edited train journey", "label": "Travel"},
    )
    assert edited.status_code == 200
    removed = state["history"][1]
    assert client.delete(API + "/cold-start/history/" + removed["id"]).status_code == 200
    predicted = client.post(
        API + "/predict", json={"items": [{"item_id": "q", "text": "a new train ticket"}]}
    )
    assert predicted.status_code == 200
    assert predicted.json()[0]["auto"] is False
    history = cold.reader().calls[-1]
    assert any(h["text"] == "an edited train journey" and h["label"] == "Travel" for h in history)
    assert removed["id"] not in {h["id"] for h in history}
    answer = client.post(
        API + "/answers",
        json={"item_id": "q", "label": "Travel"},
        headers={"Idempotency-Key": "answer"},
    )
    assert answer.status_code == 200 and answer.json()["retrain"] == "history_updated"
    assert (
        client.post(
            API + "/answers",
            json={"item_id": "q", "label": "Travel"},
            headers={"Idempotency-Key": "answer"},
        ).json()
        == answer.json()
    )
    assert len(cold.store.labels("default")) == 1
    client.post(API + "/predict", json={"items": [{"item_id": "q2", "text": "another message"}]})
    assert any(h["id"] == "q" and h["label"] == "Travel" for h in cold.reader().calls[-1])
    metrics = client.get(API + "/metrics").json()
    assert metrics["card"]["coverage"] == 0 and metrics["card"]["accuracy"] is None
    assert metrics["card"]["threshold"] is None and len(calls) == 2
    # Resume uses cache and never undoes edits or deletion tombstones.
    assert begin(client).status_code == 200
    again = finish(cold)
    assert len(again["history"]) == 19 and len(calls) == 2
    assert any(h["text"] == "an edited train journey" for h in again["history"])


def test_cache_canonicalizes_label_order_and_is_shared_across_workspaces(live):
    client, cold, calls = live
    begin(client)
    original = finish(cold)
    assert begin(client, ["Food", "Travel"], workspace="another").status_code == 200
    shared = finish(cold, "another")
    assert shared["history"] == original["history"] and shared["spent_usd"] == "0"
    assert len(calls) == 2
    cold._reader = None
    cold.start("default", ["Food", "Travel"])
    assert finish(cold)["status"] == "ready" and len(calls) == 2
    assert begin(client, ["A", "B"]).status_code == 409
    assert client.get("/api/v1/workspaces/another/cold-start").status_code == 200


def test_import_and_validation_are_bounded_before_paid_calls(live):
    client, cold, calls = live
    for body in (
        {"labels": ["a", "a"], "paid": True},
        {"labels": ["a", "b"], "paid": False},
        {"labels": ["  ", "b"], "paid": True},
        {"labels": ["a", "b"]},
    ):
        assert client.post(API + "/cold-start", json=body).status_code == 422
    assert calls == []
    bad = client.post(
        API + "/cold-start:import",
        files={"file": ("labels.csv", b"label,description\na,b")},
        data={"paid": "true"},
    )
    assert bad.status_code == 422 and calls == []
    good = client.post(
        API + "/cold-start:import",
        files={"file": ("labels.csv", "\ufefflabel\nFood\nTravel".encode())},
        data={"paid": "true"},
    )
    assert good.status_code == 200
    assert finish(cold)["labels"] == ["Food", "Travel"]
    assert len(calls) == 2
    first = cold.state("default")["history"][0]
    assert (
        client.patch(
            API + "/cold-start/history/" + first["id"], json={"text": "changed", "label": "Unknown"}
        ).status_code
        == 422
    )
    assert (
        client.delete("/api/v1/workspaces/missing/cold-start/history/" + first["id"]).status_code
        == 404
    )


def test_missing_reader_refuses_before_generation(live):
    client, cold, calls = live

    def unavailable(_):
        raise ReaderUnavailable("Install the reader.")

    cold.reader_factory = unavailable
    assert begin(client).status_code == 503
    assert calls == [] and not cold.owns("default")


def test_workspace_budget_survives_resumes_and_is_separate_from_other_workspaces(live):
    client, cold, calls = live
    # Each failed provider call books the conservative reservation; no retry on 401.
    cold.transport = provider(calls, fail=True)
    begin(client)
    state = finish(cold)
    assert state["status"] == "stopped" and len(calls) == 1
    assert 0 < Decimal(state["spent_usd"]) <= 1
    assert begin(client).status_code == 200
    assert finish(cold)["status"] == "stopped" and len(calls) == 1
    assert "private provider response" not in json.dumps(state)
    # A different label set does not share failed cache entries.
    cold.settings = cold.settings.model_copy(update={"meta_budget_usd": Decimal("0.05")})
    assert begin(client, ["A", "B"], workspace="small").status_code == 200
    assert finish(cold, "small")["status"] == "stopped"
    assert len(calls) == 1 and cold.spent("small") == 0


def test_interrupted_request_is_never_automatically_repaid(live):
    client, cold, calls = live
    with cold.cache(["Food", "Travel"]) as (cache, save):
        cache["rows"]["Food"] = {"status": "attempted"}
        save()
    begin(client)
    assert finish(cold)["status"] == "stopped"
    assert calls == []


def test_new_service_reopens_history_without_reinventing(live):
    client, cold, calls = live
    begin(client)
    finish(cold)
    replacement = ColdStartService(
        cold.store, cold.settings, reader_factory=FakeReader, transport=provider(calls)
    )
    try:
        assert len(replacement.state("default")["history"]) == 20
        assert (
            replacement.route(
                "default",
                pd.DataFrame([{"id": "restart", "text": "a message", "merchant_key": "a message"}]),
            )[0]["auto"]
            is False
        )
        assert len(calls) == 2
    finally:
        replacement.close()


def test_one_dollar_cap_stops_a_partial_history_and_resume_never_rebuys(live):
    client, cold, calls = live
    cold.transport = provider(calls, tokens=3000)
    labels = [f"label {i:02d}" for i in range(24)]
    assert begin(client, labels).status_code == 200
    first = finish(cold)
    assert first["status"] == "stopped"
    assert 0 < len(calls) < len(labels)
    assert len(first["history"]) == 10 * len(calls)
    assert Decimal(first["spent_usd"]) <= 1
    previous_calls = len(calls)
    assert begin(client, labels).status_code == 200
    assert finish(cold)["spent_usd"] == first["spent_usd"]
    assert len(calls) == previous_calls
