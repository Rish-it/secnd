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
