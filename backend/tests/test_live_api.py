import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.config import Settings
from backend.loop import Loop
from backend.store import Store
from backend.workspace import WorkspaceService

API = "/api/v1/workspaces/default"


def cpu_loop(*args, **kwargs):
    kwargs["featurizer"] = "char_tfidf"
    return Loop(*args, **kwargs)


def csv_rows(n=200):
    rows = ["item_id,text,amount,category"]
    for i in range(n):
        label = "Travel" if i % 2 else "Food"
        # Unique merchant groups keep the calibration partition nontrivial.
        text = "flight ticket journey" if i % 2 else "pizza dinner restaurant"
        tag = chr(97 + i // 26) + chr(97 + i % 26)
        if i % 17 == 0:
            label = "Food" if label == "Travel" else "Travel"
        rows.append(f"past-{i},{text} {tag} number {i},{i % 7},{label}")
    return ("\n".join(rows) + "\n").encode()


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("SECND_FEATURIZER", "char_tfidf")
    service = WorkspaceService(Store(tmp_path / "db.sqlite3"), tmp_path, cpu_loop)
    settings = Settings(data_dir=tmp_path, frontend_dist=tmp_path / "none", _env_file=None)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        yield client, service


def upload(client, content=None, workspace="default"):
    return client.post(
        f"/api/v1/workspaces/{workspace}/decisions:import",
        files={"file": ("data.csv", content or csv_rows(), "text/csv")},
        data={"target_column": "category"},
    )


def test_import_prediction_queue_answer_and_restart_use_real_loop(live):
    client, service = live
    assert client.get(API + "/queue").headers["X-Secnd-Mode"] == "fixture"
    response = upload(client)
    assert response.status_code == 200, response.text
    config = service.store.workspace("default")
    training = service.store.labels("default")
    heldout = {row["id"] for row in config["calib"]}
    assert len(training) + len(heldout) == 200
    assert not heldout & {row["item_id"] for row in training}
    assert all(row["source"] == "seed" and row["weight"] == 1 for row in training)
    assert response.headers["X-Secnd-Mode"] == "live"
    assert response.json()["card"]["model_version"] == "ren-1.0"
    assert upload(client).status_code == 409
    assert len(service.store.labels("default")) == len(training)
    predicted = client.post(
        API + "/predict",
        json={"items": [{"item_id": "question", "text": "something I have never seen"}]},
    )
    assert predicted.status_code == 200
    assert predicted.headers["X-Secnd-Mode"] == "live"
    assert predicted.json()[0]["auto"] is False
    queued = client.get(API + "/queue")
    assert queued.headers["X-Secnd-Mode"] == "live"
    assert queued.json()["items"][0]["item"]["text"] == "something I have never seen"
    body = {"item_id": "question", "label": "Food"}
    headers = {"Idempotency-Key": "answer-1"}
    first = client.post(API + "/answers", json=body, headers=headers)
    assert first.status_code == 200 and first.json()["retrain"] == "queued"
    assert client.get(API + "/queue").json()["items"] == []
    service.drain()
    assert client.post(API + "/answers", json=body, headers=headers).json() == first.json()
    assert len(service.store.labels("default")) == len(training) + 1
    assert client.get(API + "/models").status_code == 200
    assert client.get("/api/v1/workspaces/other/feed").headers["X-Secnd-Mode"] == "fixture"
    restarted = WorkspaceService(service.store, service.data_dir, cpu_loop)
    try:
        assert restarted.page("default", "queue")["items"] == []
        assert restarted.has_model("default")
        frame = pd.DataFrame(
            [
                dict(
                    id="restart",
                    text="pizza dinner restaurant",
                    amount=None,
                    date=None,
                    merchant_key="new",
                )
            ]
        )
        assert restarted.route("default", frame)[0]["model_version"].startswith("ren-1.")
        assert len(service.store.predictions("default")) == 2
    finally:
        restarted.close()


def test_unpromoted_workspace_stays_fixture_and_does_not_invent_predictions(live):
    client, service = live
    calib = pd.DataFrame([dict(id="held", text="held", label="a", merchant_key="held")])
    service.create("empty", ["a", "b"], calib)
    for route in ("feed", "queue", "models"):
        response = client.get(f"/api/v1/workspaces/empty/{route}")
        assert response.status_code == 200
        assert response.headers["X-Secnd-Mode"] == "fixture"
        assert response.json()["items"] == []
    response = client.post(
        "/api/v1/workspaces/empty/predict",
        json={"items": [{"item_id": "no-model", "text": "example"}]},
    )
    assert response.status_code == 409
    assert service.store.predictions("empty") == []


def test_invalid_imports_leave_no_workspace(live):
    client, service = live
    for content in (
        b"text,category\nitem,one\n",
        b"text,category\nx,a\nx,b\n",
        b"id,text,category\nsame,first,a\nsame,second,b\n",
        b"text,amount,category\nfirst,NaN,a\nsecond,1,b\n",
    ):
        assert upload(client, content).status_code == 422
        with pytest.raises(KeyError):
            service.store.workspace("default")


def test_rollback_changes_the_next_real_prediction_without_retraining(live):
    client, service = live
    assert upload(client).status_code == 200
    original = service.card("default").model_version
    service.route(
        "default",
        pd.DataFrame(
            [
                dict(
                    id="change",
                    text="new dinner",
                    amount=None,
                    date=None,
                    merchant_key="new dinner",
                )
            ]
        ),
    )
    service.answer("default", "change", "Food", "change")
    service.drain()
    versions_before = client.get(API + "/models").json()["items"]
    assert len(versions_before) >= 2
    service._loops["default"].checks = [0, 20]
    response = client.post(API + f"/models/{original}:rollback")
    assert response.status_code == 200
    result = client.post(
        API + "/predict",
        json={"items": [{"item_id": "after-rollback", "text": "a different meal"}]},
    )
    assert result.status_code == 200
    assert result.json()[0]["model_version"] == original
    assert client.get(API + "/metrics").json()["guarded"] is False
    assert len(client.get(API + "/models").json()["items"]) == len(versions_before)


def test_registry_reads_share_the_retrain_lock(live):
    import threading

    client, service = live
    assert upload(client).status_code == 200
    started, completed = threading.Event(), threading.Event()

    def read():
        started.set()
        response = client.get(API + "/models")
        assert response.status_code == 200
        completed.set()

    with service._lock("default"):
        reader = threading.Thread(target=read)
        reader.start()
        assert started.wait(2)
        assert not completed.wait(0.05)
    reader.join(5)
    assert completed.is_set()


def test_workspace_metrics_track_versions_and_show_live_guard(live):
    client, service = live
    assert upload(client).status_code == 200
    before = client.get(API + "/metrics")
    assert before.status_code == 200
    assert before.headers["X-Secnd-Mode"] == "live"
    assert before.json()["dataset"] is None
    assert len(before.json()["history"]) == 1
    assert before.json()["guarded"] is False
    assert before.json()["labels"] == len(service.store.labels("default"))
    assert before.json()["history"][0]["model_version"] == service.card("default").model_version
    client.post(API + "/predict", json={"items": [{"item_id": "next", "text": "meal"}]})
    answered = client.post(
        API + "/answers",
        json={"item_id": "next", "label": "Food"},
        headers={"Idempotency-Key": "metrics-answer"},
    )
    assert answered.status_code == 200
    service.drain()
    after = client.get(API + "/metrics").json()
    assert len(after["history"]) == len(client.get(API + "/models").json()["items"])
    assert after["labels"] == before.json()["labels"] + 1
    assert after["history"][0] == before.json()["history"][0]
    service._loops["default"].checks = [0, 20]
    assert client.get(API + "/metrics").json()["guarded"] is True
    benchmark = client.get(API + "/metrics?dataset=mtop")
    assert benchmark.headers["X-Secnd-Mode"] == "evaluation"
    assert benchmark.json()["history"] == []


def test_undo_keeps_real_spot_check_guard_active(live):
    client, service = live
    assert upload(client).status_code == 200
    client.post(
        API + "/predict", json={"items": [{"item_id": "guard-undo", "text": "new question"}]}
    )
    assert (
        client.post(
            API + "/answers",
            json={"item_id": "guard-undo", "label": "Food"},
            headers={"Idempotency-Key": "guard-undo"},
        ).status_code
        == 200
    )
    service.drain()
    service._loops["default"].checks = [0, 20]
    assert service._loops["default"].guarded
    assert (
        client.post(API + "/answers:undo", json={"idempotency_key": "guard-undo"}).status_code
        == 200
    )
    assert service._loops["default"].guarded
    assert client.get(API + "/metrics").json()["guarded"]
