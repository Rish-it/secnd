"""Regressions for the branch security and concurrency review."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.meta_ads import ActionBlocked, AdsService
from backend.tests.test_auth import register
from backend.tests.test_auth import secured as secured_fixture
from backend.tests.test_meta_ads import ads as ads_fixture
from backend.tests.test_meta_ads import enable, sync

secured, ads = secured_fixture, ads_fixture


def test_origin_scheme_is_checked(secured):
    assert register(secured).status_code == 200
    response = secured.post(
        "/api/v1/workspaces",
        json={"workspace_id": "cross-scheme"},
        headers={"Origin": "https://testserver"},
    )
    assert response.status_code == 403


def test_kill_blocks_ad_undo(ads):
    client, service, settings, state, _ = ads
    service._loops["ads"].action = "lower budget"
    enable(client, settings)
    proposal = sync(client).json()[0]
    assert proposal["status"] == "executed"
    state["budget"] = "9000"
    client.post("/api/v1/agent/kill")
    with pytest.raises(ActionBlocked):
        client.app.state.meta_ads.undo("ads", proposal["item_id"])
    assert len(state["posts"]) == 1


def test_kill_during_snapshot_blocks_send(ads, monkeypatch):
    client, service, settings, state, _ = ads
    proposal = sync(client).json()[0]
    enable(client, settings)
    manager = client.app.state.meta_ads
    original = manager.client_for

    def client_for(ws):
        provider = original(ws)
        snapshot = provider.snapshot

        def stopped(*args):
            result = snapshot(*args)
            with service.store.connect(write=True) as db:
                db.execute("UPDATE agent_control SET killed=1")
            return result

        provider.snapshot = stopped
        return provider

    monkeypatch.setattr(manager, "client_for", client_for)
    with pytest.raises(ActionBlocked):
        manager.execute("ads", proposal["item_id"])
    assert not state["posts"]


def test_undo_claim_is_atomic_across_service_instances(ads, monkeypatch):
    client, service, settings, state, transport = ads
    service._loops["ads"].action = "lower budget"
    enable(client, settings)
    proposal = sync(client).json()[0]
    state["budget"] = "9000"
    twin = copy.copy(service)
    twin._guard, twin._locks = threading.RLock(), {}
    managers = [client.app.state.meta_ads, AdsService(settings, twin, transport=transport)]
    barrier = threading.Barrier(2)
    for manager in managers:
        original = manager.client_for

        def client_for(ws, original=original):
            provider = original(ws)
            snapshot = provider.snapshot

            def together(*args):
                result = snapshot(*args)
                barrier.wait(timeout=5)
                return result

            provider.snapshot = together
            return provider

        monkeypatch.setattr(manager, "client_for", client_for)

    def undo(manager):
        try:
            manager.undo("ads", proposal["item_id"])
        except ActionBlocked:
            return "blocked"
        return "undone"

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(undo, managers))
    assert sorted(results) == ["blocked", "undone"]
    assert len(state["posts"]) == 2


def test_all_saved_decisions_keep_model_and_basis(ads):
    client, service, _, _, _ = ads
    proposal = sync(client).json()[0]
    saved = service.store.latest_prediction("ads", proposal["item_id"])
    assert saved["model_id"] == "workspace-head"
    assert saved["model_revision"] == proposal["decision"]["model_revision"]
    assert saved["automatic_basis"] == "model_certified"
    assert saved["automatic_evidence"]["cp_lower_bound"] >= 0.99


def test_stale_workspace_bound_message_cannot_send(tmp_path):
    from types import SimpleNamespace

    import httpx

    from backend.autonomy import Policy
    from backend.config import Settings
    from backend.meta_webhook import Message, MetaWebhook
    from backend.store import Store
    from backend.workspace import WorkspaceService

    config = Settings(_env_file=None, data_dir=tmp_path, whatsapp_send_enabled=True)
    service = WorkspaceService(Store(config.database_path), tmp_path)
    service.store.create_workspace("mine", {"labels": ["yes", "no"], "calib": []})
    service.meta_oauth = SimpleNamespace(owned=lambda ws: True, token=lambda *args: "test-token")
    service.harness = SimpleNamespace(
        killed=lambda: False, status=lambda ws: {"enabled": True}, log=lambda *args: None
    )
    identity = ["ren-v1", "revision-1"]
    service.decision_identity = lambda *args: tuple(identity)
    service.autonomy.owner = lambda ws: "person"
    service.autonomy.configure(
        "mine", "inbox", Policy(level="act", bar=99, allow_customer_messages=True)
    )
    for i in range(299):
        service.autonomy.review(
            "mine",
            f"past-{i}",
            "yes",
            f"key-{i}",
            dict(label="yes", prob=0.99, source_app="whatsapp"),
        )
    message = Message(
        platform="whatsapp", message_id="one", sender="10", account="20", text="question"
    )
    service.store.record_predictions(
        "mine",
        [dict(id=message.item_id, text=message.text, merchant_key="")],
        [
            dict(
                item_id=message.item_id,
                label="yes",
                prob=0.99,
                auto=True,
                threshold=0.9,
                model_version="old-history",
                spot_check=False,
                source_app="whatsapp",
                model_id="ren-v0",
                model_revision="old-revision",
                automatic_basis="workspace_bound",
            )
        ],
    )
    sent = []
    webhook = MetaWebhook(
        config,
        service,
        transport=httpx.MockTransport(
            lambda request: (
                sent.append(request) or httpx.Response(200, json={"messages": [{"id": "1"}]})
            )
        ),
    )
    result = webhook._deliver("mine", message, {"reply": "yes", "status": "dry_run"})
    assert not sent
    assert result["status"] == "queued"
    service.close()
