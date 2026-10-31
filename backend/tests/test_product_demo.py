"""The product demo is isolated, scripted and incapable of provider writes."""

import socket

import pytest
from fastapi.testclient import TestClient

from backend.demo_seed import create_demo

ROOT = "/api/v1/workspaces/default"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Synthetic demo attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    with TestClient(create_demo(tmp_path / "synthetic")) as client:
        yield client


def test_demo_story_counts_evidence_correction_and_outcomes(demo):
    queue = demo.get(ROOT + "/queue").json()["items"]
    assert len(queue) == 20
    assert {row["source_app"] for row in queue} == {"ads", "whatsapp", "messenger", "instagram"}
    assert all(row["item"]["text"].startswith("Synthetic ·") for row in queue)
    policies = {row["decision_type"]: row for row in demo.get(ROOT + "/autonomy").json()}
    assert policies["ad"]["confirmed"] == policies["ad"]["reviewed"] == 61
    assert policies["ad"]["effective_level"] == "act"
    assert policies["inbox"]["reviewed"] == 40
    assert policies["inbox"]["confirmations_left"] > 0
    assert policies["account"]["level"] == "act"
    assert policies["account"]["effective_level"] == "ask"
    assert policies["account"]["corrected"] == 1
    assert policies["account"]["lower_bound"] < 0.95
    acted = demo.get(ROOT + "/feed").json()["items"]
    assert len(acted) == 3
    assert all(row["automatic_basis"] == "workspace_bound" for row in acted)
    outcomes = demo.get(ROOT + "/outcomes").json()
    assert len(outcomes) == 3
    assert all(row["status"] == "synthetic_executed" for row in outcomes)
    assert all(set(["spend", "cpa", "ctr"]).issubset(row["after"]) for row in outcomes)
    activity = demo.get(ROOT + "/agent").json()["activity"]
    assert any("Lower bound" in row["event"] for row in activity)
    assert demo.get(ROOT + "/serving").json()["model_id"] == "synthetic-scripted-demo"


def test_demo_confirm_correct_undo_are_durable_and_idempotent(demo):
    row = demo.get(ROOT + "/queue").json()["items"][0]
    payload = {"item_id": row["item_id"], "label": row["label"]}
    headers = {"Idempotency-Key": "synthetic-test-review"}
    assert demo.post(ROOT + "/answers", json=payload, headers=headers).status_code == 200
    assert demo.post(ROOT + "/answers", json=payload, headers=headers).status_code == 200
    assert len(demo.get(ROOT + "/queue").json()["items"]) == 19
    assert (
        demo.post(
            ROOT + "/answers:undo", json={"idempotency_key": headers["Idempotency-Key"]}
        ).status_code
        == 200
    )
    assert len(demo.get(ROOT + "/queue").json()["items"]) == 20
    payload["label"] = next(label for label in row["probabilities"] if label != row["label"])
    headers["Idempotency-Key"] = "synthetic-test-correction"
    assert demo.post(ROOT + "/answers", json=payload, headers=headers).status_code == 200
    corrected = demo.get(ROOT + "/overridden").json()["items"]
    assert any(
        item["item_id"] == row["item_id"] and item["answer_label"] == payload["label"]
        for item in corrected
    )


def test_demo_never_opens_existing_data(tmp_path):
    existing = tmp_path / "ordinary-business"
    existing.mkdir()
    database = existing / "secnd.sqlite3"
    database.write_bytes(b"synthetic-test-sentinel-not-a-database")
    with pytest.raises(ValueError, match="new empty directory"):
        create_demo(existing)
    assert database.read_bytes() == b"synthetic-test-sentinel-not-a-database"
    alias = tmp_path / "alias"
    alias.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="new empty directory"):
        create_demo(alias)


def test_demo_blocks_live_setup_generation_and_writes_and_can_stop(demo):
    for path in [
        "/auth/register",
        "/workspaces/default/cold-start",
        "/workspaces/default/integrations/ads/insights:sync",
    ]:
        assert demo.post("/api/v1" + path, json={}).status_code == 409
    assert demo.get(ROOT + "/integrations/meta/oauth/start").status_code == 409
    connection = demo.post("/api/demo/connect").json()
    assert connection["selected"] == []
    selected = [connection["assets"][0]["key"]]
    assert (
        demo.put(ROOT + "/integrations/meta/accounts", json={"assets": selected}).json()["selected"]
        == selected
    )
    assert demo.post("/api/v1/agent/kill").status_code == 200
    assert demo.get(ROOT + "/agent").json()["global_killed"]
