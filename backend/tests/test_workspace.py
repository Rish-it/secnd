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
    service.create("cold", ["a", "b"], calib)
    service.store.put_items("cold", [dict(id=str(i), text="x", merchant_key="") for i in range(2)])
    service.answer("cold", "0", "a", "first")
    service.drain()
    loop = service._loops["cold"]
    assert loop.trained == 0
    service.answer("cold", "1", "b", "second")
    service.drain()
    assert loop.learned == [("0", "a", "answer"), ("1", "b", "answer")]
    assert loop.trained == 1
