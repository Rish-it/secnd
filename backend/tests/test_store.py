import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.store import IdempotencyConflict, Store, initialize


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "db.sqlite3")
    result.create_workspace("a", {"labels": ["Travel", "Food"]})
    result.create_workspace("b", {"labels": ["Travel", "Food"]})
    for ws in ("a", "b"):
        result.put_items(
            ws, [{"id": "1", "text": "", "amount": 12.5, "date": "2026-10-29", "merchant_key": ""}]
        )
    return result


@pytest.mark.parametrize("previous", [0, 1])
def test_migration_from_bootstrap_is_repeatable_and_keeps_data(tmp_path, previous):
    path = tmp_path / "db.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version={previous}")
    store = Store(path)
    store.create_workspace("saved", {"labels": ["a"]})
    initialize(path)
    assert store.workspace("saved") == {"labels": ["a"]}
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_newer_schema_is_never_downgraded(tmp_path):
    path = tmp_path / "db.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=3")
    with pytest.raises(RuntimeError, match="newer"):
        initialize(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_workspace_item_labels_predictions_and_models_round_trip(store):
    store.add_label("a", "1", "Travel", "seed")
    store.add_label("a", "1", "Food", "muse_cold_start")
    labels = store.labels("a")
    assert [(r["label"], r["source"], r["weight"]) for r in labels] == [
        ("Travel", "seed", 1),
        ("Food", "muse_cold_start", 0.3),
    ]
    assert store.labels("b") == []
    assert store.item("a", "1")["text"] == ""
    decision = dict(
        item_id="1",
        model_version="ren-1.0",
        label="Travel",
        prob=0.9,
        probabilities={"Travel": 0.9},
        auto=False,
        threshold=None,
        spot_check=True,
        reasons={"merchant_memory": False},
    )
    store.record_predictions("a", [store.item("a", "1")], [decision])
    assert store.predictions("a") == [decision]
    assert store.predictions("b") == []
    store.set_models("a", [{"version": "ren-1.0", "card": {"coverage": 0}, "is_current": True}])
    assert store.models("a")[0]["card"] == {"coverage": 0}
    assert store.models("a")[0]["is_current"] is True
    assert store.models("b") == []


def test_concurrent_replays_commit_one_answer_and_one_label(store):
    def answer(_):
        return store.answer("a", "1", "Travel", "key", {"version": "old"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(answer, range(16)))
    assert sum(inserted for _, inserted in results) == 1
    assert all(result == {"version": "old"} for result, _ in results)
    assert len(store.labels("a")) == 1
    assert store.answer("a", "1", "Travel", "key", {"version": "new"}) == (
        {"version": "old"},
        False,
    )
    assert store.answer("b", "1", "Food", "key", {})[1]
    with pytest.raises(IdempotencyConflict):
        store.answer("a", "1", "Food", "key", {})
    assert len(store.labels("a")) == 1


def test_failed_writes_leave_no_partial_training_rows(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.answer("a", "missing", "Travel", "key", {})
    assert store.labels("a") == []
    with pytest.raises(ValueError):
        store.answer("a", "1", "Travel", "key", {"bad": float("nan")})
    assert store.labels("a") == []
    with pytest.raises(ValueError):
        store.add_label("a", "1", "Travel", "test")
    with pytest.raises(ValueError):
        store.put_items("a", [{"id": "1", "text": "changed", "merchant_key": ""}])
    assert store.item("a", "1")["text"] == ""
    with pytest.raises(KeyError):
        store.item("missing", "1")


def test_import_is_atomic_and_decision_pages_are_stable_and_scoped(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        store.create_workspace(
            "broken",
            {},
            items=[dict(id="x", text="x", merchant_key="")],
            labels=[("missing", "a")],
        )
    with pytest.raises(KeyError):
        store.workspace("broken")
    store.create_workspace("a", {})
    store.create_workspace("b", {})
    items = [dict(id=str(i), text=f"item {i}", merchant_key="") for i in range(8)]
    decisions = [
        dict(
            item_id=str(i),
            model_version="ren-1.0",
            label="a",
            prob=0.5 + i / 20,
            probabilities={"a": 0.5 + i / 20},
            auto=i >= 5,
            threshold=0.7,
            reasons={},
            spot_check=i == 1,
        )
        for i in range(8)
    ]
    store.record_predictions("a", items, decisions)
    first = store.decision_page("a", "queue", limit=2)
    assert [d["item_id"] for d in first["items"]] == ["0", "1"]
    assert first["items"][1]["spot_check"]
    assert first["items"][0]["item"]["text"] == "item 0"
    # A concurrent answer removes a prior question without shifting the next cursor.
    store.answer("a", "0", "a", "answer", {})
    second = store.decision_page("a", "queue", first["next_cursor"], 2)
    assert [d["item_id"] for d in second["items"]] == ["2", "3"]
    third = store.decision_page("a", "queue", second["next_cursor"], 2)
    assert [d["item_id"] for d in third["items"]] == ["4"]
    assert third["next_cursor"] is None
    feed = store.decision_page("a", "feed", limit=1)
    assert feed["items"][0]["item_id"] == "7"
    assert store.decision_page("a", "feed", feed["next_cursor"], 1)["items"][0]["item_id"] == "6"
    # Every prediction is logged, but a page contains only the latest decision per item.
    store.record_predictions("a", [items[5]], [{**decisions[5], "auto": False, "prob": 0.6}])
    assert len(store.predictions("a")) == 9
    assert [d["item_id"] for d in store.decision_page("a", "feed")["items"]] == ["7", "6"]
    assert store.decision_page("b", "queue")["items"] == []
    for workspace, kind, cursor in (
        ("b", "queue", first["next_cursor"]),
        ("a", "feed", first["next_cursor"]),
        ("a", "queue", "not-a-cursor"),
    ):
        with pytest.raises(ValueError, match="cursor"):
            store.decision_page(workspace, kind, cursor)
