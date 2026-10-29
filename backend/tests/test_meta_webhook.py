import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend import meta_webhook
from backend.api import create_app
from backend.config import Settings
from backend.store import Store
from backend.tests.test_live_api import API, cpu_loop, upload
from backend.workspace import WorkspaceService

URL = API + "/integrations/meta/webhook"
FIXTURES = Path(__file__).parent / "fixtures/meta"


def payload(name="whatsapp"):
    return json.loads((FIXTURES / f"{name}-text.json").read_text())


def signed(client, body, signature=None):
    raw = json.dumps(body, ensure_ascii=True).encode()
    digest = hmac.new(b"test-app-secret", raw, hashlib.sha256).hexdigest()
    return client.post(
        URL,
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature or "sha256=" + digest,
        },
    )


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("SECND_FEATURIZER", "char_tfidf")
    settings = Settings(
        data_dir=tmp_path,
        frontend_dist=tmp_path / "none",
        _env_file=None,
        meta_app_secret="test-app-secret",
        meta_verify_token="test-verify",
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path, cpu_loop)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        assert upload(client).status_code == 200
        yield client, service, settings


def test_handshake_and_signatures_fail_closed(live):
    client, service, settings = live
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "test-verify",
        "hub.challenge": "1158201444",
    }
    response = client.get(URL, params=params)
    assert response.status_code == 200 and response.text == "1158201444"
    assert response.headers["Content-Type"].startswith("text/plain")
    assert client.get(URL, params={**params, "hub.verify_token": "wrong"}).status_code == 403
    assert client.get(URL, params={**params, "hub.mode": "other"}).status_code == 403
    for signature in ("invalid", "sha1=" + "0" * 40, "sha256=" + "0" * 64):
        assert signed(client, payload(), signature).status_code == 403
    assert client.post(URL, json=payload()).status_code == 403
    assert service.store.predictions("default") == []
    settings.meta_app_secret = SecretStr("")
    assert signed(client, payload()).status_code == 503
    assert "test-app-secret" not in repr(settings)
    assert "test-verify" not in settings.model_dump_json()


def test_both_platforms_queue_and_human_answers_retrain_once(live):
    client, service, _ = live
    initial = len(service.store.labels("default"))
    for platform in ("whatsapp", "messenger"):
        response = signed(client, payload(platform))
        assert response.status_code == 200, response.text
        result = response.json()["items"][0]
        assert result["platform"] == platform and result["status"] == "queued"
        assert result["reply"] is None
        assert client.get(API + "/queue").json()["items"]
        answer = client.post(
            API + "/answers",
            json={"item_id": result["item_id"], "label": "Food"},
            headers={"Idempotency-Key": platform + "-human"},
        )
        assert answer.status_code == 200 and answer.json()["retrain"] == "queued"
        service.drain()
        replay = signed(client, payload(platform)).json()["items"][0]
        assert replay["duplicate"] is True and replay["item_id"] == result["item_id"]
    assert len(service.store.labels("default")) == initial + 2
    assert len(service.store.predictions("default")) == 2
    assert client.get(API + "/queue").json()["items"] == []
    assert all(row["source"] == "answer" for row in service.store.labels("default")[-2:])


def test_message_id_deduplication_survives_restart_and_rejects_changed_content(live):
    client, service, settings = live
    original = signed(client, payload()).json()["items"][0]
    restarted = WorkspaceService(Store(service.store.path), service.data_dir, cpu_loop)
    with TestClient(create_app(settings, workspace_service=restarted)) as second:
        replay = signed(second, payload()).json()["items"][0]
        assert replay == {**original, "duplicate": True}
        assert len(restarted.store.predictions("default")) == 1
        changed = payload()
        changed["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "changed"
        assert signed(second, changed).status_code == 409


def test_concurrent_duplicate_delivery_routes_once(live):
    from concurrent.futures import ThreadPoolExecutor

    client, service, _ = live
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: signed(client, payload()), range(2)))
    assert all(response.status_code == 200 for response in responses)
    assert sorted(response.json()["items"][0]["duplicate"] for response in responses) == [
        False,
        True,
    ]
    assert len(service.store.predictions("default")) == 1


@pytest.mark.parametrize(
    "enabled,token", [(False, ""), (True, ""), (False, "test-token"), (True, "test-token")]
)
def test_automatic_reply_requires_both_explicit_flag_and_token(live, monkeypatch, enabled, token):
    client, service, settings = live
    loop = service._loops["default"]
    loop.current["tau"], loop.spot_check_rate = 0.0, 0.0
    settings.whatsapp_send_enabled = enabled
    settings.whatsapp_token = SecretStr(token)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200, json={"messages": [{"id": "outbound-test"}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(meta_webhook.httpx, "post", post)
    result = signed(client, payload()).json()["items"][0]
    assert result["status"] == ("sent" if enabled and token else "dry_run")
    assert bool(calls) == bool(enabled and token)
    if calls:
        assert calls[0][0] == "https://graph.facebook.com/v25.0/100000000000002/messages"
        assert calls[0][1]["json"]["text"]["body"] == result["reply"]
    assert signed(client, payload()).json()["items"][0]["duplicate"] is True
    assert len(calls) <= 1
    # Messenger uses a different token type; its outbound reply remains a preview.
    assert signed(client, payload("messenger")).json()["items"][0]["status"] == "dry_run"
    assert len(calls) <= 1


def test_send_timeout_is_never_retried_on_duplicate(live, monkeypatch):
    client, service, settings = live
    loop = service._loops["default"]
    loop.current["tau"], loop.spot_check_rate = 0.0, 0.0
    settings.whatsapp_send_enabled = True
    settings.whatsapp_token = SecretStr("test-token")
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise httpx.ReadTimeout("test timeout")

    monkeypatch.setattr(meta_webhook.httpx, "post", fail)
    assert signed(client, payload()).json()["items"][0]["status"] == "delivery_unknown"
    assert signed(client, payload()).json()["items"][0]["status"] == "delivery_unknown"
    assert len(calls) == 1


def test_unsupported_events_and_malformed_signed_payloads(live):
    client, service, _ = live
    wa = payload()
    wa["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "image"
    ms = payload("messenger")
    ms["entry"][0]["messaging"][0]["message"]["is_echo"] = True
    for body in (wa, ms):
        result = signed(client, body)
        assert result.status_code == 200
        assert result.json() == {"accepted": 0, "ignored": 1, "items": []}
    for body in ([], {"entry": ["bad"]}, {"object": "page", "entry": None}):
        assert signed(client, body).status_code == 422
    assert service.store.predictions("default") == []
    assert client.post(URL, content=b"x" * (meta_webhook.MAX_BODY + 1)).status_code == 413
