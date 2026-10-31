from types import SimpleNamespace

import pytest

from backend.autonomy import Autonomy, Policy, lower
from backend.store import Store


@pytest.fixture
def autonomy(tmp_path):
    import threading

    service = SimpleNamespace(store=Store(tmp_path / "db"), _lock=lambda ws: threading.RLock())
    return Autonomy(service)


def reviews(a, n, corrected=0, ws="one", source="ads", confidence=0.95):
    for i in range(n):
        a.review(
            ws,
            f"item-{i}",
            "keep" if i >= corrected else "pause",
            f"key-{i}",
            {"label": "keep", "prob": confidence, "source_app": source},
        )


@pytest.mark.parametrize("bar,minimum", [(90, 29), (95, 59), (98, 149), (99, 299)])
def test_exact_one_sided_unlock_counts(autonomy, bar, minimum):
    a = autonomy
    a.configure("one", "ad", Policy(level="act", bar=bar))
    reviews(a, minimum - 1)
    status = a.status("one", "ad")
    assert not status["unlocked"] and status["confirmations_left"] == 1
    reviews(a, minimum)
    assert a.status("one", "ad")["unlocked"]
    assert a.allowed("one", "ad", "pause")
    assert lower(minimum, minimum) >= bar / 100
    assert not a.allowed("two", "ad", "pause")
    assert not a.allowed("one", "inbox", "send message")


def test_correction_drops_act_immediately_and_deduplicates(autonomy):
    a = autonomy
    a.configure("one", "ad", Policy(level="act", bar=90))
    reviews(a, 29)
    assert a.allowed("one", "ad")
    a.review(
        "one", "item-1", "pause", "correction", {"label": "keep", "prob": 0.95, "source_app": "ads"}
    )
    state = a.status("one", "ad")
    assert state["reviewed"] == 29 and state["corrected"] == 1
    assert state["effective_level"] == "ask" and state["confirmations_left"] > 0


def test_confidence_cut_window_and_high_risk_actions(autonomy):
    a = autonomy
    a.configure("one", "ad", Policy(level="act", bar=99, allow_raise_budget=True))
    reviews(a, 299, confidence=0.89)
    assert a.status("one", "ad")["reviewed"] == 0
    reviews(a, 299, ws="other")
    assert not a.allowed("one", "ad", "raise budget")
    for i in range(299):
        a.review(
            "one",
            f"high-{i}",
            "keep",
            f"high-{i}",
            {"label": "keep", "prob": 0.99, "source_app": "ads"},
        )
    assert a.allowed("one", "ad", "raise budget")
    a.configure("one", "ad", Policy(level="act", bar=98, allow_raise_budget=True))
    assert not a.allowed("one", "ad", "raise budget")
    assert not a.allowed("one", "inbox", "send message")


def test_opt_in_aggregate_is_counts_only_and_cannot_unlock(autonomy):
    from backend.config import Settings
    from backend.harness import Harness
    from backend.store import encode

    a = autonomy
    h = Harness(Settings(_env_file=None), a.service, None, None)
    for ws in ("one", "two"):
        a.service.store.create_workspace(ws, {"labels": ["keep", "pause"]})
        a.service.store.record_predictions(
            ws,
            [{"id": "x", "text": "private-customer-text", "merchant_key": ""}],
            [
                {
                    "item_id": "x",
                    "label": "keep",
                    "prob": 0.99,
                    "auto": False,
                    "threshold": None,
                    "spot_check": False,
                    "model_version": "v1",
                    "source_app": "ads",
                }
            ],
        )
        a.review(ws, "x", "keep", "k", {"label": "keep", "prob": 0.99, "source_app": "ads"})
    assert all(r["proposals"] == 0 for r in a.aggregate()["counts"])
    h.configure("one", share_counts=True)
    result = a.aggregate()
    row = result["counts"][0]
    assert row == {
        "decision_type": "ad",
        "proposals": 1,
        "confirmed": 1,
        "corrected": 0,
        "workspaces": 1,
    }
    assert "private-customer-text" not in encode(result)
    assert not a.allowed("two", "ad")
    h.configure("one", share_counts=False)
    assert all(r["proposals"] == 0 for r in a.aggregate()["counts"])
