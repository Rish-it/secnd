"""Inbox state, safe answer retraction, and historical evidence."""

import json
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.config import Settings
from backend.store import SCHEMA_VERSION, IdempotencyConflict, Store
from backend.tests.test_workspace import RecordingLoop
from backend.workspace import WorkspaceService

API = "/api/v1/workspaces/demo"


@pytest.fixture
def inbox(tmp_path):
    settings = Settings(data_dir=tmp_path, frontend_dist=tmp_path / "none", _env_file=None)
    service = WorkspaceService(Store(settings.database_path), tmp_path, RecordingLoop)
    calib = pd.DataFrame([dict(id="hold", text="private holdout", merchant_key="", label="a")])
    service.create("demo", ["a", "b", "c", "d"], calib, featurizer="char_tfidf")
    service.store.put_items(
        "demo", [dict(id=str(i), text=f"history {i}", merchant_key="") for i in range(2)]
    )
    for i, label in enumerate(["a", "b"]):
        service.store.add_label("demo", str(i), label)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        yield client, service


def predict(client, identifier="incoming"):
    response = client.post(
        API + "/predict",
        json={"items": [dict(item_id=identifier, text="New question", source_app="instagram")]},
    )
    assert response.status_code == 200, response.text
    return response.json()[0]


def answer(client, key="first", label="b", identifier="incoming"):
    return client.post(
        API + "/answers",
        json=dict(item_id=identifier, label=label),
        headers={"Idempotency-Key": key},
    )


def undo(client, key="first"):
    return client.post(API + "/answers:undo", json={"idempotency_key": key})


def test_undo_preserves_spot_check_guard_and_pending_checks(inbox):
    client, service = inbox
    predict(client)
    assert answer(client).status_code == 200
    service.drain()
    loop = service._loops["demo"]
    loop.checks = [0, 20]
    loop.pending = {"another-spot-check": "a"}
    assert undo(client).status_code == 200
    rebuilt = service._loops["demo"]
    assert rebuilt.checks == [0, 20]
    assert rebuilt.pending == {"another-spot-check": "a"}


def test_all_labels_source_and_correction_survive_restart(inbox):
    client, service = inbox
    assert client.get(API + "/labels").json() == ["a", "b", "c", "d"]
    predict(client)
    queue = client.get(API + "/queue").json()["items"]
    assert queue[0]["source_app"] == queue[0]["item"]["source_app"] == "instagram"
    assert answer(client, label="d").status_code == 200
    service.drain()
    assert client.get(API + "/queue").json()["items"] == []
    overridden = client.get(API + "/overridden").json()["items"]
    assert overridden[0]["answer_label"] == "d"
    assert overridden[0]["label"] == "a"
    reopened = Store(service.store.path)
    assert reopened.decision_page("demo", "overridden")["items"][0]["answer_label"] == "d"
    assert reopened.labels("demo")[-1]["label"] == "d"


def test_undo_confirm_reopens_question_and_removes_training_label(inbox):
    client, service = inbox
    predict(client)
    assert answer(client, label="a").status_code == 200
    service.drain()
    assert client.get(API + "/queue").json()["items"] == []
    assert undo(client).status_code == 200
    service.drain()
    assert len(service.store.labels("demo")) == 2
    assert all(row[0] != "incoming" for row in service._loops["demo"].learned)
    assert service.store.workspace("demo")["calib"][0]["id"] == "hold"
    assert client.get(API + "/queue").json()["items"][0]["item_id"] == "incoming"
    assert undo(client).status_code == 200
    assert answer(client, label="a").status_code == 409
    assert answer(client, key="new", label="a").status_code == 200


def test_undo_requires_reverse_order_and_is_scoped(inbox):
    client, service = inbox
    predict(client)
    predict(client, "second")
    assert answer(client).status_code == 200
    assert answer(client, key="second", identifier="second").status_code == 200
    service.drain()
    assert undo(client).status_code == 409
    assert undo(client, "second").status_code == 200
    assert undo(client).status_code == 200
    assert (
        client.post(
            "/api/v1/workspaces/another/answers:undo", json={"idempotency_key": "first"}
        ).status_code
        == 404
    )
    assert undo(client, "unknown").status_code == 404
    assert len(service.store.labels("demo")) == 2


def test_override_auto_then_undo_restores_original_proof(inbox):
    client, service = inbox
    decision = predict(client)
    service.store.record_predictions(
        "demo",
        [service.store.item("demo", "incoming")],
        [{**decision, "auto": True, "threshold": 0.8, "cp_lower_bound": 0.99123}],
    )
    assert answer(client).status_code == 200
    service.drain()
    assert client.get(API + "/feed").json()["items"] == []
    assert client.get(API + "/overridden").json()["items"][0]["answer_label"] == "b"
    assert undo(client).status_code == 200
    saved = client.get(API + "/feed").json()["items"][0]
    assert saved["cp_lower_bound"] == 0.99123 and saved["threshold"] == 0.8
    assert saved["answer_label"] is None
    assert client.get(API + "/overridden").json()["items"] == []


def test_migrate_v2_keeps_existing_answers_and_idempotency(tmp_path):
    from backend.store import TABLES

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        for statement in TABLES:
            db.execute(statement)
        db.execute("PRAGMA user_version=2")
    store = Store(path)
    store.create_workspace("a", {"labels": ["a", "b"]})
    store.put_items("a", [dict(id="x", text="x", merchant_key="")])
    store.answer("a", "x", "b", "k", {"model_version": "old"})
    Store(path)
    row, created = store.undo_answer("a", "k")
    assert created and json.loads(row["result_json"])["model_version"] == "old"
    assert store.labels("a") == []
    assert not store.undo_answer("a", "k")[1]
    with pytest.raises(IdempotencyConflict, match="undone"):
        store.answer("a", "x", "b", "k", {})
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_overridden_pagination_and_cursors(inbox):
    client, service = inbox
    for i in range(4):
        predict(client, f"item-{i}")
        assert answer(client, key=f"key-{i}", identifier=f"item-{i}").status_code == 200
    service.drain()
    first = client.get(API + "/overridden?limit=2").json()
    second = client.get(
        API + "/overridden", params={"limit": 2, "cursor": first["next_cursor"]}
    ).json()
    assert len({x["item_id"] for x in first["items"] + second["items"]}) == 4
    assert second["next_cursor"] is None
    assert client.get(API + "/queue", params={"cursor": first["next_cursor"]}).status_code == 422


def test_refused_rebuild_keeps_pre_answer_model(inbox):
    client, service = inbox
    initial = predict(client)["model_version"]
    assert answer(client).status_code == 200
    service.drain()
    assert service.card("demo").model_version != initial

    class RefusingLoop(RecordingLoop):
        def retrain(self):
            return {"promoted": False}

    service.factory = RefusingLoop
    assert undo(client).status_code == 200
    assert service.card("demo").model_version == initial
    assert all(row[0] != "incoming" for row in service._loops["demo"].learned)
