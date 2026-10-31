import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.config import Settings


@pytest.fixture
def secured(tmp_path):
    with TestClient(
        create_app(
            Settings(
                data_dir=tmp_path, _env_file=None, demo_mode=False, session_cookie_secure=False
            )
        )
    ) as client:
        yield client


def register(client, name="person-one"):
    return client.post(
        "/api/v1/auth/register", json={"username": name, "password": "long-test-password"}
    )


@pytest.mark.parametrize(
    "path",
    [
        "queue",
        "feed",
        "labels",
        "metrics",
        "models",
        "serving",
        "integrations/ads/status",
        "integrations/meta/status",
    ],
)
def test_every_workspace_surface_requires_session(secured, path):
    assert secured.get("/api/v1/workspaces/private/" + path).status_code == 401


def test_ownership_applies_to_reads_writes_and_imports(secured):
    assert register(secured).status_code == 200
    assert secured.post("/api/v1/workspaces", json={"workspace_id": "private"}).status_code == 200
    assert secured.get("/api/v1/workspaces/private/queue").status_code == 200
    assert secured.post("/api/v1/auth/logout").status_code == 200
    assert register(secured, "person-two").status_code == 200
    assert secured.get("/api/v1/workspaces").json()["items"] == []
    for method, path in [
        ("GET", "queue"),
        ("POST", "answers"),
        ("POST", "decisions:import"),
        ("PUT", "integrations/ads/preferences"),
    ]:
        assert secured.request(method, "/api/v1/workspaces/private/" + path).status_code == 404
    assert secured.post("/api/v1/workspaces", json={"workspace_id": "private"}).status_code == 409


def test_login_logout_and_csrf(secured):
    assert register(secured).status_code == 200
    assert "HttpOnly" in next(iter(secured.cookies.jar))._rest
    assert secured.post("/api/v1/auth/logout").status_code == 200
    assert not secured.get("/api/v1/auth/session").json()["signed_in"]
    assert (
        secured.post(
            "/api/v1/auth/login", json={"username": "person-one", "password": "wrong-password"}
        ).status_code
        == 401
    )
    assert (
        secured.post(
            "/api/v1/auth/login", json={"username": "person-one", "password": "long-test-password"}
        ).status_code
        == 200
    )
    assert (
        secured.post(
            "/api/v1/workspaces",
            json={"workspace_id": "evil"},
            headers={"Origin": "https://evil.example"},
        ).status_code
        == 403
    )


def test_demo_fallback_never_reaches_signed_in_workspaces(tmp_path):
    with TestClient(
        create_app(
            Settings(data_dir=tmp_path, _env_file=None, demo_mode=True, session_cookie_secure=False)
        )
    ) as client:
        assert register(client).status_code == 200
        assert (
            client.post("/api/v1/workspaces", json={"workspace_id": "private"}).status_code == 200
        )
        assert client.get("/api/v1/workspaces/private/queue").status_code == 200
        client.post("/api/v1/auth/logout")
        assert client.get("/api/v1/workspaces/private/queue").status_code == 404
        assert client.get("/api/v1/workspaces/default/queue").status_code == 200
        assert "private" not in client.get("/api/v1/workspaces").json()["items"]


def test_login_limits_name_and_address_without_account_disclosure(secured, monkeypatch):
    from backend.auth import Auth

    monkeypatch.setattr(Auth, "password", staticmethod(lambda value, salt: value))
    assert register(secured).status_code == 200
    for name in ("PERSON-ONE", "missing-name"):
        responses = [
            secured.post(
                "/api/v1/auth/login", json={"username": name, "password": "wrong-password"}
            )
            for _ in range(6)
        ]
        assert [r.status_code for r in responses] == [401] * 5 + [429]
        assert "retry" in responses[-1].json()["message"].lower()
        assert int(responses[-1].headers["Retry-After"]) > 0
    for i in range(10):
        assert (
            secured.post(
                "/api/v1/auth/login",
                json={"username": f"rotated-{i}", "password": "wrong-password"},
            ).status_code
            == 401
        )
    assert (
        secured.post(
            "/api/v1/auth/login", json={"username": "new-name", "password": "wrong-password"}
        ).status_code
        == 429
    )
    # Forwarded headers cannot choose a new rate-limit address.
    assert (
        secured.post(
            "/api/v1/auth/login",
            headers={"X-Forwarded-For": "192.0.2.4"},
            json={"username": "fresh-name", "password": "wrong-password"},
        ).status_code
        == 429
    )


def test_registration_limits_persist_in_sqlite_and_expire(secured, monkeypatch):
    import backend.auth as module
    from backend.auth import Auth

    monkeypatch.setattr(Auth, "password", staticmethod(lambda value, salt: value))
    clock = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    for i in range(5):
        assert register(secured, f"person-{i}").status_code == 200
    assert register(secured, "blocked").status_code == 429
    secured.app.state.auth = Auth(secured.app.state.auth.store, secured.app.state.auth.settings)
    assert register(secured, "still-blocked").status_code == 429
    clock[0] += 3601
    assert register(secured, "unblocked").status_code == 200
