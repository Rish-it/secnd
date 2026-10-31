import base64
import hashlib
import hmac
import json
import logging
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.config import Settings

BASE = "/api/v1/workspaces/mine/integrations/meta"


@pytest.fixture
def oauth(tmp_path):
    calls = []

    def transport(request):
        calls.append(request)
        path = request.url.path
        if path.endswith("oauth/access_token"):
            return httpx.Response(
                200, json={"access_token": "private-long-token", "expires_in": 5184000}
            )
        if path.endswith("/me"):
            return httpx.Response(200, json={"id": "123"})
        if path.endswith("/me/adaccounts"):
            return httpx.Response(200, json={"data": [{"id": "act_10", "name": "Sandbox"}]})
        if path.endswith("/me/accounts"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "20",
                            "name": "Page",
                            "access_token": "private-page-token",
                            "instagram_business_account": {"id": "30"},
                        }
                    ]
                },
            )
        if path.endswith("/me/businesses"):
            return httpx.Response(200, json={"data": [{"id": "40"}]})
        if path.endswith("/40/owned_whatsapp_business_accounts"):
            return httpx.Response(200, json={"data": [{"id": "50"}]})
        if path.endswith("/50/phone_numbers"):
            return httpx.Response(200, json={"data": [{"id": "60"}]})
        raise AssertionError(str(request.url))

    config = Settings(
        data_dir=tmp_path,
        _env_file=None,
        demo_mode=False,
        session_cookie_secure=False,
        meta_app_id="1",
        meta_login_config_id="2",
        meta_app_secret="app-secret",
        meta_redirect_uri="https://app.example/api/v1/meta/oauth/callback",
        meta_public_url="https://app.example",
        meta_token_key=Fernet.generate_key().decode(),
        ads_sandbox_accounts=["act_10"],
    )
    with TestClient(create_app(config, meta_transport=httpx.MockTransport(transport))) as client:
        assert (
            client.post(
                "/api/v1/auth/register",
                json={"username": "person", "password": "long-test-password"},
            ).status_code
            == 200
        )
        assert client.post("/api/v1/workspaces", json={"workspace_id": "mine"}).status_code == 200
        yield client, calls


def start(client):
    response = client.get(BASE + "/oauth/start", follow_redirects=False)
    assert response.status_code == 303
    params = parse_qs(urlsplit(response.headers["location"]).query)
    assert params["redirect_uri"] == ["https://app.example/api/v1/meta/oauth/callback"]
    return params["state"][0]


def connect(client):
    return client.get(
        "/api/v1/meta/oauth/callback",
        params={"state": start(client), "code": "private-code"},
        follow_redirects=False,
    )


def test_state_mismatch_denial_expiry_and_single_use(oauth):
    client, calls = oauth
    state = start(client)
    assert (
        client.get("/api/v1/meta/oauth/callback", params={"state": "bad", "code": "x"}).status_code
        == 403
    )
    assert (
        client.get("/api/v1/meta/oauth/callback", params={"state": state, "code": "x"}).status_code
        == 403
    )
    state = start(client)
    response = client.get(
        "/api/v1/meta/oauth/callback", params={"state": state, "error": "access_denied"}
    )
    assert response.status_code == 403 and "role" in response.text
    state = start(client)
    with client.app.state.workspaces.store.connect(write=True) as db:
        db.execute("UPDATE meta_oauth_states SET expires=0")
    assert (
        client.get("/api/v1/meta/oauth/callback", params={"state": state, "code": "x"}).status_code
        == 403
    )
    assert calls == []


def test_encrypted_tokens_never_in_responses_or_logs_and_selected_assets(oauth, caplog):
    client, calls = oauth
    caplog.set_level(logging.DEBUG)
    response = connect(client)
    assert response.status_code == 303
    status = client.get(BASE + "/connection")
    assert {a["kind"] for a in status.json()["assets"]} == {
        "ads",
        "messenger",
        "instagram",
        "whatsapp",
    }
    assert "private-long-token" not in response.text + status.text + caplog.text
    assert "private-page-token" not in status.text + caplog.text
    with client.app.state.workspaces.store.connect() as db:
        row = db.execute("SELECT token_blob FROM meta_connections").fetchone()
    assert b"private-long-token" not in row[0]
    assert (
        client.put(BASE + "/accounts", json={"assets": ["ads:act_10", "messenger:20"]}).status_code
        == 200
    )
    assert client.put(BASE + "/accounts", json={"assets": ["ads:act_999"]}).status_code == 422
    assert client.get(BASE + "/status").json()["accounts"]["messenger"] == ["20"]
    assert all("private" not in str(r.url) for r in calls)
    assert client.post("/api/v1/meta/disconnect").json() == {"connected": False}
    with client.app.state.workspaces.store.connect() as db:
        assert db.execute("SELECT count(*) FROM meta_connections").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM meta_assets").fetchone()[0] == 0


def test_signed_data_deletion_removes_token_and_derived_data(oauth):
    client, _ = oauth
    assert connect(client).status_code == 303
    assert (
        client.post(
            "/api/v1/meta/data-deletion", data={"signed_request": "bad.payload"}
        ).status_code
        == 403
    )
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"algorithm": "HMAC-SHA256", "user_id": "123"}).encode()
        )
        .decode()
        .rstrip("=")
    )
    sig = (
        base64.urlsafe_b64encode(hmac.new(b"app-secret", payload.encode(), hashlib.sha256).digest())
        .decode()
        .rstrip("=")
    )
    response = client.post(
        "/api/v1/meta/data-deletion", data={"signed_request": sig + "." + payload}
    )
    assert response.status_code == 200
    assert client.get(urlsplit(response.json()["url"]).path).json()["completed"]
    assert client.get("/api/v1/workspaces").json()["items"] == []
    assert not client.get(BASE + "/connection").status_code == 200


def test_deletion_after_disconnect_purges_history_and_model_files(oauth):
    client, _ = oauth
    assert connect(client).status_code == 303
    service = client.app.state.workspaces
    service.store.create_workspace("mine", {"labels": ["yes", "no"]})
    service.store.put_items(
        "mine", [{"id": "private-message", "text": "private content", "merchant_key": ""}]
    )
    service.store.add_label("mine", "private-message", "yes")
    artifact = service.data_dir / "workspaces" / "mine" / "models" / "saved"
    artifact.mkdir(parents=True)
    (artifact / "card.json").write_text("{}")
    assert client.post("/api/v1/meta/disconnect").status_code == 200
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"algorithm": "HMAC-SHA256", "user_id": "123"}).encode()
        )
        .decode()
        .rstrip("=")
    )
    signature = (
        base64.urlsafe_b64encode(hmac.new(b"app-secret", payload.encode(), hashlib.sha256).digest())
        .decode()
        .rstrip("=")
    )
    response = client.post(
        "/api/v1/meta/data-deletion", data={"signed_request": signature + "." + payload}
    )
    assert response.status_code == 200
    assert not artifact.exists()
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM items WHERE workspace_id='mine'").fetchone()[0] == 0
        assert (
            db.execute("SELECT COUNT(*) FROM labels WHERE workspace_id='mine'").fetchone()[0] == 0
        )
        assert db.execute("SELECT COUNT(*) FROM meta_identities").fetchone()[0] == 0
    assert (
        client.post(
            "/api/v1/meta/data-deletion", data={"signed_request": signature + "." + payload}
        ).status_code
        == 200
    )


def test_browser_denial_returns_to_connection_screen_without_provider_details(oauth):
    client, _ = oauth
    state = start(client)
    response = client.get(
        "/api/v1/meta/oauth/callback",
        params={"state": state, "error": "access_denied"},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?workspace=mine&meta_error=refused"
