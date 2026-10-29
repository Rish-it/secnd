import threading

import pandas as pd
import pytest

from backend.store import Store
from backend.workspace import WorkspaceService


class RecordingLoop:
    def __init__(self, labels, calib, **kwargs):
        self.learned, self.answers, self.trained = [], [], 0
        self.calib = calib
        self.models = kwargs["models"]
        self.block = None

    def learn(self, frame, labels, source):
        self.learned.extend(zip(frame.id, labels, [source] * len(frame), strict=True))

    def retrain(self):
        self.trained += 1
        v = self.models.save({}, {"coverage": 0.2, "threshold": 0.99})
        self.models.promote(v)

    def on_answer(self, frame, label, source):
        self.answers.append((str(frame.id.iloc[0]), label, source, threading.get_ident()))
        if self.block:
            assert self.block.wait(5)
        self.learn(frame, [label], source)
        self.retrain()

    def route(self, frame):
        return [
            dict(
                item_id=str(row.id),
                label="a",
                prob=0.9,
                probabilities={"a": 0.9, "b": 0.1},
                auto=False,
                threshold=None,
                model_version=self.models.current(),
                reasons={"merchant_memory": False},
                spot_check=False,
            )
            for row in frame.itertuples()
        ]


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    svc = WorkspaceService(store, tmp_path, RecordingLoop)
    calib = pd.DataFrame([dict(id="heldout", text="hold", merchant_key="hold", label="a")])
    svc.create("a", ["a", "b"], calib, featurizer="char_tfidf")
    store.put_items("a", [dict(id=str(i), text="text", merchant_key="") for i in range(4)])
    store.add_label("a", "0", "a", "seed")
    store.add_label("a", "1", "b", "muse_cold_start")
    yield svc
    svc.close()


def test_rebuild_replays_sources_in_order_and_logs_every_decision(service):
    frame = pd.DataFrame([dict(id="2", text="text", merchant_key="")])
    first = service.route("a", frame)
    loop = service._loops["a"]
    assert loop.learned == [("0", "a", "seed"), ("1", "b", "muse_cold_start")]
    assert loop.trained == 1
    assert len(service.store.predictions("a")) == 1
    service.route("a", frame)
    assert len(service.store.predictions("a")) == 2
    assert service.store.models("a")[0]["is_current"]
    assert first[0]["item_id"] == "2"
    restarted = WorkspaceService(service.store, service.data_dir, RecordingLoop)
    try:
        restarted.route("a", frame)
        assert restarted._loops["a"].learned == loop.learned
    finally:
        restarted.close()


def test_answer_is_durable_before_background_work_and_replays_without_retraining(service):
    frame = pd.DataFrame([dict(id="2", text="text", merchant_key="")])
    service.route("a", frame)
    loop = service._loops["a"]
    release = threading.Event()
    loop.block = release
    try:
        result = service.answer("a", "2", "b", "key", source="spot_check")
        assert result["retrain"] == "queued"
        assert result["coverage"] == 0.2
        assert len(service.store.labels("a")) == 3
        replay = service.answer("a", "2", "b", "key")
        assert replay == result
    finally:
        release.set()
    service.drain()
    assert [(r[0], r[1], r[2]) for r in loop.answers] == [("2", "b", "spot_check")]
    assert loop.answers[0][3] != threading.get_ident()
    assert loop.trained == 2
    assert len(service.store.labels("a")) == 3


def test_first_answer_after_restart_is_learned_exactly_once(service):
    service.answer("a", "2", "b", "first")
    service.answer("a", "3", "a", "second")
    service.drain()
    loop = service._loops["a"]
    assert loop.learned == [
        ("0", "a", "seed"),
        ("1", "b", "muse_cold_start"),
        ("2", "b", "answer"),
        ("3", "a", "answer"),
    ]
    assert len(loop.answers) == 2


def test_no_calibration_or_unknown_label_can_become_an_answer(service):
    service.store.put_items("a", [dict(id="heldout", text="hold", merchant_key="hold")])
    for item, label in [("heldout", "a"), ("2", "unknown")]:
        with pytest.raises(ValueError):
            service.answer("a", item, label, item)
    assert len(service.store.labels("a")) == 2
    with pytest.raises(ValueError):
        service.route("../escape", pd.DataFrame())


def test_failed_retrain_keeps_label_for_rebuild(service):
    service.route("a", pd.DataFrame([dict(id="2", text="text", merchant_key="")]))
    loop = service._loops["a"]

    def fail(*args):
        raise RuntimeError("failed fit")

    loop.on_answer = fail
    service.answer("a", "2", "b", "durable")
    with pytest.raises(RuntimeError, match="retraining failed"):
        service.drain()
    assert len(service.store.labels("a")) == 3
    service.route("a", pd.DataFrame([dict(id="3", text="text", merchant_key="")]))
    assert service._loops["a"].learned[-1] == ("2", "b", "answer")


def test_one_class_answer_waits_for_second_class_without_losing_rows(service):
    calib = pd.DataFrame([dict(id="calib2", text="held", merchant_key="held", label="a")])
    service.create("cold", ["a", "b"], calib, featurizer="char_tfidf")
    service.store.put_items("cold", [dict(id=str(i), text="x", merchant_key="") for i in range(2)])
    service.answer("cold", "0", "a", "first")
    service.drain()
    loop = service._loops["cold"]
    assert loop.trained == 0
    service.answer("cold", "1", "b", "second")
    service.drain()
    assert loop.learned == [("0", "a", "answer"), ("1", "b", "answer")]
    assert loop.trained == 1


def test_http_predict_and_answer_use_persisted_workspace(service, tmp_path):
    from fastapi.testclient import TestClient

    from backend.api import create_app
    from backend.config import Settings

    settings = Settings(data_dir=tmp_path, frontend_dist=tmp_path / "none", _env_file=None)
    api = "/api/v1/workspaces/a"
    with TestClient(create_app(settings, workspace_service=service)) as client:
        predicted = client.post(
            api + "/predict", json={"items": [{"item_id": "routed", "text": "Rail"}]}
        )
        assert predicted.status_code == 200
        assert predicted.headers["X-Secnd-Mode"] == "live"
        assert len(service.store.predictions("a")) == 1
        answer = {"item_id": "routed", "label": "b"}
        headers = {"Idempotency-Key": "http-answer"}
        first = client.post(api + "/answers", json=answer, headers=headers)
        assert first.status_code == 200
        assert first.headers["X-Secnd-Mode"] == "live"
        assert first.json()["retrain"] == "queued"
        service.drain()
        assert client.post(api + "/answers", json=answer, headers=headers).json() == first.json()
        assert len(service.store.labels("a")) == 3
        conflict = client.post(api + "/answers", json={**answer, "label": "a"}, headers=headers)
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "http_409"
        assert (
            client.post(
                api + "/answers",
                json={**answer, "item_id": "absent"},
                headers={"Idempotency-Key": "other"},
            ).status_code
            == 404
        )
        assert (
            client.post(
                api + "/answers",
                json={**answer, "label": "unknown"},
                headers={"Idempotency-Key": "other"},
            ).status_code
            == 422
        )


def test_real_loop_promotions_are_readable_through_models_api(tmp_path):
    from fastapi.testclient import TestClient

    from backend.api import create_app
    from backend.config import Settings
    from backend.schemas import EvalCard

    def rows(n, start):
        return pd.DataFrame(
            [
                dict(
                    id=str(i),
                    text=("flight travel ticket" if i % 2 else "pizza meal restaurant")
                    + f" {i % 9}",
                    amount=float(i % 5),
                    date=None,
                    merchant_key="",
                    label="Travel" if i % 2 else "Food",
                )
                for i in range(start, start + n)
            ]
        )

    store = Store(tmp_path / "store.sqlite3")
    service = WorkspaceService(store, tmp_path)
    calib = rows(500, 10000)
    noisy = calib.index % 17 == 0
    calib.loc[noisy, "label"] = calib.loc[noisy, "label"].map({"Food": "Travel", "Travel": "Food"})
    service.create("default", ["Food", "Travel"], calib, featurizer="char_tfidf")
    train = rows(100, 0)
    store.put_items("default", train.to_dict("records"))
    for row in train.itertuples():
        store.add_label("default", row.id, row.label, "seed")
    settings = Settings(data_dir=tmp_path, frontend_dist=tmp_path / "none", _env_file=None)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        assert len(service.route("default", rows(2, 20000))) == 2
        response = client.get("/api/v1/workspaces/default/models")
        assert response.status_code == 200
        versions = response.json()["items"]
        assert len(versions) == 1 and versions[0]["current"]
        measured = EvalCard.model_validate(versions[0]["card"])
        assert measured.model_version == "ren-1.0"
        assert measured.accuracy is not None and measured.ece is not None
        service.answer("default", "20000", "Food", "real-answer")
        service.drain()
        response = client.get("/api/v1/workspaces/default/models")
        assert response.status_code == 200
        assert sum(v["current"] for v in response.json()["items"]) == 1
        assert len(store.labels("default")) == 101
        assert len(store.predictions("default")) == 2


def test_workspace_default_and_logged_fallback(tmp_path, caplog, monkeypatch):
    from backend import workspace

    monkeypatch.setenv("SECND_FEATURIZER", "pt-minilm")
    calls = []
    monkeypatch.setattr(workspace, "fetch", lambda name, data_dir: calls.append((name, data_dir)))
    service = WorkspaceService(Store(tmp_path / "db.sqlite3"), tmp_path)
    calib = pd.DataFrame([dict(id="held", text="held", label="a")])
    try:
        service.create("ren", ["a", "b"], calib)
        assert service.store.workspace("ren")["featurizer"] == "pt-minilm"
        assert calls == [("pt-minilm", tmp_path)]

        def fail(*args):
            raise ValueError("Missing snapshot")

        monkeypatch.setattr(workspace, "fetch", fail)
        service.create("fallback", ["a", "b"], calib)
        assert service.store.workspace("fallback")["featurizer"] == "minilm"
        assert "fetch/verification failed (ValueError); falling back to minilm" in caplog.text
        monkeypatch.setenv("SECND_FEATURIZER", "char_tfidf")
        service.create("explicit", ["a", "b"], calib)
        assert service.store.workspace("explicit")["featurizer"] == "char_tfidf"
    finally:
        service.close()
