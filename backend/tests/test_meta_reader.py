"""Reader requests use a mocked model, never downloaded weights."""

from types import SimpleNamespace

import pandas as pd

from backend.config import Settings
from backend.store import Store
from backend.tests.test_reader_serving import stand_in as reader_fixture
from backend.workspace import WorkspaceService

stand_in = reader_fixture


def test_meta_reader_uses_own_options_and_fresh_corrections(tmp_path, monkeypatch):
    from backend.meta_reader import MetaDecisionRuntime

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        meta_decision_model_folder=tmp_path / "saved",
        meta_decision_model_id="ren-meta",
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path)
    service.store.create_workspace("one", {"labels": ["pause", "keep"], "calib": []})
    captured = []

    class Reader:
        options = SimpleNamespace(model_id="ren-meta")
        artifact = SimpleNamespace(revision="sha256-model")

        def predict(self, labels, history, frame, **kwargs):
            captured.append((labels, history, kwargs["decision"]))
            return [
                dict(
                    item_id=str(frame.iloc[0].id),
                    label="keep",
                    probabilities={"keep": 0.96, "pause": 0.04},
                    prob=0.96,
                    auto=False,
                    threshold=None,
                    spot_check=False,
                    model_version="history-id",
                    reasons={},
                )
            ]

    service.meta_decisions = MetaDecisionRuntime(service, settings, factory=lambda *args: Reader())
    monkeypatch.setattr(service.autonomy, "sample", lambda *args: False)
    frame = pd.DataFrame([dict(id="next", text="Ad CPA rose", source_app="ads", merchant_key="")])
    result = service.route("one", frame)[0]
    assert result["model_id"] == "ren-meta" and result["model_revision"] == "sha256-model"
    assert captured[0][2].keys == ["pause", "keep"]
    assert all(captured[0][2].meanings)

    def retired_reader(*args):
        raise AssertionError("The configured Meta reader must own the correction path.")

    service.cold_start = SimpleNamespace(
        owns=lambda ws: True, answer=retired_reader, close=lambda: None
    )
    service.answer("one", "next", "pause", "correction")
    service.route("one", frame.assign(id="later"))
    assert captured[-1][1] == [{"id": "next", "text": "Ad CPA rose", "label": "pause"}]
    assert not service._pending
    service.close()


def test_remote_timeout_asks_even_when_workspace_is_unlocked(tmp_path, monkeypatch):
    import httpx

    from backend.autonomy import Policy
    from backend.meta_reader import MetaDecisionRuntime

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        meta_decision_model_id="ren-remote",
        reader_endpoint_url="https://reader.example/decide",
        reader_endpoint_token="test-token",
        reader_endpoint_revision="v1",
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path)
    service.store.create_workspace("one", {"labels": ["pause", "keep"], "calib": []})
    seen = []

    def timeout(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        raise httpx.ReadTimeout("secret provider detail", request=request)

    service.meta_decisions = MetaDecisionRuntime(
        service, settings, transport=httpx.MockTransport(timeout)
    )
    monkeypatch.setattr(service.autonomy, "owner", lambda ws: "person")
    monkeypatch.setattr(service.autonomy, "sample", lambda *args: False)
    service.autonomy.configure("one", "ad", Policy(level="act", bar=90, confidence_cut=0))
    for i in range(29):
        service.autonomy.review(
            "one", f"past-{i}", "keep", f"review-{i}", dict(label="keep", prob=1, source_app="ads")
        )
    result = service.route(
        "one", pd.DataFrame([dict(id="next", text="CPA rose", merchant_key="", source_app="ads")])
    )[0]
    assert not result["auto"]
    assert result["inference_error"] == "Reader unavailable. Please choose an answer."
    assert "secret" not in str(result)
    assert len(seen) == 1
    service.close()


def test_remote_reply_is_validated_and_each_row_keeps_its_revision(tmp_path):
    import httpx

    from backend.meta_reader import MetaDecisionRuntime

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        meta_decision_model_id="ren-remote",
        reader_endpoint_url="https://reader.example",
        reader_endpoint_token="test-token",
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path)
    service.store.create_workspace("one", {"labels": ["yes", "no"], "calib": []})
    responses = iter(
        [
            {"model_id": "ren-remote", "revision": "v1", "probabilities": {"yes": 0.9, "no": 0.1}},
            {"model_id": "ren-remote", "revision": "v2", "probabilities": {"yes": 0.8, "no": 0.2}},
            {"model_id": "wrong-model", "revision": "v2", "probabilities": {"yes": 1, "no": 0}},
        ]
    )
    runtime = MetaDecisionRuntime(
        service,
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(responses))),
    )
    service.meta_decisions = runtime
    rows = service.route(
        "one",
        pd.DataFrame(
            [
                dict(id=f"item-{i}", text="Question", merchant_key="", source_app="account")
                for i in range(3)
            ]
        ),
    )
    assert [row["model_revision"] for row in rows[:2]] == ["v1", "v2"]
    assert rows[-1]["inference_error"] and not rows[-1]["auto"]
    assert service.answer("one", "item-0", "no", "correction") == service.answer(
        "one", "item-0", "no", "correction"
    )
    service.close()


def test_endpoint_requires_bearer_and_returns_model_identity():
    from fastapi.testclient import TestClient

    from backend.reader_endpoint import create_reader_api

    class Reader:
        options = SimpleNamespace(model_id="ren-test")
        artifact = SimpleNamespace(revision="v1")

        def predict(self, labels, history, frame, **kwargs):
            assert not kwargs["allow_auto"]
            assert kwargs["decision"].meanings == ["Pause ad", "Keep ad"]
            return [{"probabilities": {"pause": 0.9, "keep": 0.1}}]

    body = dict(
        item_id="one",
        text="Ad state",
        decision_type="ad",
        keys=["pause", "keep"],
        meanings=["Pause ad", "Keep ad"],
        history=[],
    )
    with TestClient(create_reader_api(Reader(), "private-test-token")) as client:
        assert client.post("/", json=body).status_code == 401
        response = client.post(
            "/", json=body, headers={"Authorization": "Bearer private-test-token"}
        )
        assert response.json() == {
            "model_id": "ren-test",
            "revision": "v1",
            "probabilities": {"pause": 0.9, "keep": 0.1},
        }
        assert (
            client.post(
                "/",
                json={**body, "keys": ["pause", "pause"]},
                headers={"Authorization": "Bearer private-test-token"},
            ).status_code
            == 422
        )


def test_typed_rendering_reuses_eight_nearest_examples(stand_in, tmp_path, monkeypatch):
    import numpy as np

    from backend import typed
    from backend.reader_serving import CPUReader
    from backend.tests.test_reader_serving import history

    reader = CPUReader(tmp_path)
    calls = []
    original = typed.render

    def render(tok, decision, order):
        calls.append(decision)
        return original(tok, decision, order)

    monkeypatch.setattr(typed, "render", render)
    decision = typed.Decision(
        "one",
        "choice",
        "Choose an action",
        ["a", "b"],
        ["Pause the ad", "Keep the ad"],
        "Ad state",
        np.zeros(2),
        0,
    )
    result = reader.predict(
        ["a", "b"], history(), pd.DataFrame([dict(id="one", text="Ad state")]), decision=decision
    )[0]
    assert len(result["reasons"]["neighbours"]) == 8
    assert calls[0].meanings == ["Pause the ad", "Keep the ad"]
    assert calls[0].state.count("Example:") == 8


def test_signed_in_reader_earns_act_after_history_changes(stand_in, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.api import create_app
    from backend.tests.test_reader_serving import history
    from backend.tests.test_workspace import RecordingLoop

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        demo_mode=False,
        session_cookie_secure=False,
        workspace_backends={"mine": "reader"},
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path, RecordingLoop)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        assert (
            client.post(
                "/api/v1/auth/register",
                json={"username": "owner", "password": "long-test-password"},
            ).status_code
            == 200
        )
        assert client.post("/api/v1/workspaces", json={"workspace_id": "mine"}).status_code == 200
        calib = pd.DataFrame([dict(id="hold", text="held out", merchant_key="", label="a")])
        service.create("mine", ["a", "b"], calib, featurizer="char_tfidf")
        for row in history():
            service.store.put_items("mine", [{**row, "merchant_key": ""}])
            service.store.add_label("mine", row["id"], row["label"])
        monkeypatch.setattr(service.autonomy, "sample", lambda *args: False)
        base = "/api/v1/workspaces/mine"
        assert (
            client.put(base + "/autonomy/account", json={"level": "act", "bar": 90}).status_code
            == 200
        )
        for i in range(29):
            result = client.post(
                base + "/predict",
                json={
                    "items": [
                        {"item_id": f"item-{i}", "text": f"question {i}", "source_app": "account"}
                    ]
                },
            )
            assert result.status_code == 200, result.text
            assert not result.json()[0]["auto"]
            answer = client.post(
                base + "/answers",
                json={"item_id": f"item-{i}", "label": "a"},
                headers={"Idempotency-Key": f"answer-{i}"},
            )
            assert answer.status_code == 200, answer.text
        result = client.post(
            base + "/predict",
            json={
                "items": [{"item_id": "earned", "text": "new question", "source_app": "account"}]
            },
        )
        assert result.status_code == 200, result.text
        decision = result.json()[0]
        assert decision["auto"] and decision["automatic_basis"] == "workspace_bound"
        assert not decision["reader"]["scope_authorized"]
        assert decision["cp_lower_bound"] is None
        assert client.get(base + "/feed").json()["items"][0]["automatic_basis"] == "workspace_bound"
        service.readers.reader("mine").artifact.revision = "changed-model"
        assert (
            next(
                r for r in client.get(base + "/autonomy").json() if r["decision_type"] == "account"
            )["reviewed"]
            == 0
        )


def test_serving_status_reports_meta_model_instead_of_old_head(tmp_path):
    from fastapi.testclient import TestClient

    from backend.api import create_app
    from backend.meta_reader import MetaDecisionRuntime

    settings = Settings(_env_file=None, data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        service = client.app.state.workspaces
        service.store.create_workspace(
            "one", {"labels": ["yes", "no"], "calib": [], "featurizer": "char_tfidf"}
        )
        settings.meta_decision_model_folder = tmp_path / "saved"
        settings.meta_decision_model_id = "ren-meta"
        reader = SimpleNamespace(
            options=SimpleNamespace(model_id="ren-meta"), artifact=SimpleNamespace(revision="v1")
        )
        service.meta_decisions = MetaDecisionRuntime(
            service, settings, factory=lambda *args: reader
        )
        response = client.get("/api/v1/workspaces/one/serving")
        assert response.status_code == 200
        assert response.json()["model_id"] == "ren-meta"
        assert response.json()["revision"] == "v1"
        assert response.json()["policy_complete"] is False


def test_correction_is_recent_when_remote_history_is_bounded(tmp_path):
    from backend.meta_reader import MetaDecisionRuntime

    settings = Settings(_env_file=None, data_dir=tmp_path)
    service = WorkspaceService(Store(settings.database_path), tmp_path)
    service.store.create_workspace("one", {"labels": ["yes", "no"], "calib": []})
    for item in ["old", "recent"]:
        service.store.put_items("one", [dict(id=item, text=item, merchant_key="")])
        service.store.add_label("one", item, "yes")
    service.store.answer("one", "old", "no", "correction", {})
    rows = MetaDecisionRuntime(service, settings).history("one")
    assert [row["id"] for row in rows] == ["recent", "old"]
    assert rows[-1]["label"] == "no"
    service.close()
