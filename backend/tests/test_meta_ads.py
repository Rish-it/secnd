"""Every Ads request uses MockTransport. No live account or model calls."""

import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend.api import create_app
from backend.config import Settings
from backend.meta_ads import ACTIONS, AdsService
from backend.store import Store
from backend.tests.test_workspace import RecordingLoop
from backend.workspace import WorkspaceService

API = "/api/v1/workspaces/ads"
ADS = API + "/integrations/ads"
DATES = {"since": "2026-10-01", "until": "2026-10-02"}


class AdsLoop(RecordingLoop):
    action, automatic, spot_check, guarded = "pause", True, False, False

    def retrain(self):
        artifact = {"model": SimpleNamespace(C=1), "temperature": 1, "tau": 0.9, "lower": 0.995}
        version = self.models.save(
            artifact, {"coverage": 0.8, "threshold": 0.9, "cp_lower_bound": 0.995}
        )
        self.models.promote(version)
        self.current = {**artifact, "version": version}

    def route(self, frame):
        return [
            dict(
                item_id=str(row.id),
                label=self.action,
                probabilities={
                    label: 0.997 if label == self.action else 0.001 for label in ACTIONS
                },
                prob=0.997,
                auto=self.automatic,
                spot_check=self.spot_check,
                threshold=0.9,
                model_version=self.models.current(),
                reasons={},
            )
            for row in frame.itertuples()
        ]


@pytest.fixture
def ads(tmp_path, monkeypatch):
    # Any accidental unmocked HTTP request fails the test before network access.
    def no_network(*args, **kwargs):
        raise AssertionError("Live transports are forbidden in connector tests.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    settings = Settings(
        data_dir=tmp_path,
        frontend_dist=tmp_path / "none",
        _env_file=None,
        ads_token="ads-test-secret",
        ads_workspace_accounts={"ads": "act_900"},
        ads_sandbox_accounts=["act_900"],
    )
    service = WorkspaceService(Store(settings.database_path), tmp_path, AdsLoop)
    calib = pd.DataFrame([dict(id="hold", text="held out", label="keep", merchant_key="")])
    service.create("ads", ACTIONS, calib, featurizer="char_tfidf")
    service.store.put_items("ads", [dict(id="seed", text="past", merchant_key="")])
    service.store.add_label("ads", "seed", "keep")
    service._runtime("ads").retrain()
    service.store.create_workspace("other", {"labels": ACTIONS})
    state = dict(
        budget="10000",
        spend="42.00",
        posts=[],
        calls=[],
        failure=None,
        account="900",
        paging=None,
        cpa="7.00",
    )

    def respond(request):
        state["calls"].append((request.method, request.url.path, dict(request.url.params)))
        assert request.headers["Authorization"] == "Bearer ads-test-secret"
        assert "ads-test-secret" not in str(request.url)
        if state["failure"] == "read" and request.method == "GET":
            return httpx.Response(401, json={"error": {"message": "private ads-test-secret"}})
        if request.method == "POST":
            state["posts"].append(request)
            if state["failure"] == "timeout":
                raise httpx.ReadTimeout("private ads-test-secret", request=request)
            if state["failure"] == "false":
                return httpx.Response(200, json={"success": False})
            return httpx.Response(200, json={"success": True})
        if request.url.path.endswith("/insights"):
            level = request.url.params["level"]
            row = dict(
                account_id=state["account"],
                account_currency="USD",
                campaign_id="201",
                campaign_name="Test campaign",
                spend=state["spend"],
                ctr="1.75",
                cost_per_action_type=[{"action_type": "purchase", "value": state["cpa"]}],
                date_start=DATES["since"],
                date_stop=DATES["until"],
            )
            if level == "adset":
                row.update(adset_id="301", adset_name="Test ad set")
            return httpx.Response(
                200,
                json={"data": [row], **({"paging": state["paging"]} if state["paging"] else {})},
            )
        return httpx.Response(
            200,
            json=dict(
                id=request.url.path.rsplit("/", 1)[-1],
                account_id=state["account"],
                **({"daily_budget": state["budget"]} if state["budget"] is not None else {}),
            ),
        )

    transport = httpx.MockTransport(respond)
    with TestClient(create_app(settings, workspace_service=service)) as client:
        client.app.state.meta_ads = AdsService(settings, service, transport=transport)
        yield client, service, settings, state, transport


def sync(client, level="adset"):
    return client.post(ADS + "/insights:sync", json={**DATES, "level": level})


def enable(client, settings):
    settings.ads_send_enabled = True
    response = client.put(ADS + "/preferences", json={"auto_act": True})
    assert response.status_code == 200 and response.json()["auto_act"]


def test_readonly_proposal_is_typed_logged_and_enters_workspace(ads):
    client, service, settings, state, _ = ads
    status = client.get(ADS + "/status").json()
    assert not status["auto_act"] and not status["server_send_enabled"]
    response = sync(client)
    assert response.status_code == 200, response.text
    proposal = response.json()[0]
    assert proposal["action"] == "pause" and proposal["status"] == "proposed"
    assert proposal["insight"]["spend"] == "42.00"
    assert proposal["insight"]["cpa"] == "7.00" and proposal["insight"]["ctr"] == "1.75"
    saved = service.store.latest_prediction("ads", proposal["item_id"])
    assert saved["source_app"] == "ads"
    assert client.get(API + "/feed").json()["items"][0]["item_id"] == proposal["item_id"]
    assert client.get(ADS + "/log").json()["items"][0]["event"] == "proposed"
    assert not state["posts"]
    assert sync(client).json()[0]["duplicate"]
    assert len(service.store.predictions("ads")) == 1
    assert len(client.get(ADS + "/log").json()["items"]) == 1
    assert not settings.cold_start_paid


@pytest.mark.parametrize("server,workspace", [(False, False), (True, False), (False, True)])
def test_both_switches_required(ads, server, workspace):
    client, _, settings, state, _ = ads
    settings.ads_send_enabled = server
    client.put(ADS + "/preferences", json={"auto_act": workspace})
    item = sync(client).json()[0]
    response = client.post(ADS + f"/proposals/{item['item_id']}:execute")
    assert response.status_code == 409 and not state["posts"]
    assert client.get(ADS + "/log").json()["items"][-1]["event"] == "blocked"


@pytest.mark.parametrize(
    "action,payload",
    [
        ("pause", "status=PAUSED"),
        ("raise budget", "daily_budget=11000"),
        ("lower budget", "daily_budget=9000"),
        ("keep", None),
    ],
)
def test_certified_actions_use_mock_transport_once_and_record_execution(ads, action, payload):
    client, service, settings, state, transport = ads
    service._loops["ads"].action = action
    enable(client, settings)
    item = sync(client).json()[0]
    assert item["status"] == ("executed" if payload else "kept")
    assert len(state["posts"]) == (1 if payload else 0)
    if payload:
        assert state["posts"][0].content.decode() == payload
    assert client.post(ADS + f"/proposals/{item['item_id']}:execute").json()["duplicate"]
    client.app.state.meta_ads = AdsService(settings, service, transport=transport)
    assert sync(client).json()[0]["duplicate"]
    assert len(state["posts"]) == (1 if payload else 0)
    events = [row["event"] for row in client.get(ADS + "/log").json()["items"]]
    assert events[-3:] == ["proposed", "execution_started", "executed"]


@pytest.mark.parametrize("failure", ["timeout", "false"])
def test_unknown_delivery_never_retries(ads, failure):
    client, _, settings, state, _ = ads
    enable(client, settings)
    state["failure"] = failure
    item = sync(client).json()[0]
    assert item["status"] == "delivery_unknown"
    assert (
        client.post(ADS + f"/proposals/{item['item_id']}:execute").json()["status"]
        == "delivery_unknown"
    )
    assert len(state["posts"]) == 1
    assert "ads-test-secret" not in client.get(ADS + "/log").text


@pytest.mark.parametrize(
    "reason", ["not_auto", "spot_check", "guarded", "bound", "version", "answer"]
)
def test_current_certification_and_no_human_override_required(ads, reason):
    client, service, settings, state, _ = ads
    loop = service._loops["ads"]
    if reason == "not_auto":
        loop.automatic = False
    if reason == "spot_check":
        loop.automatic = False
        loop.spot_check = True
    item = sync(client).json()[0]
    if reason == "guarded":
        loop.guarded = True
    elif reason in ("bound", "version"):
        card = service.card("ads").model_dump()
        if reason == "bound":
            card["cp_lower_bound"] = 0.98
            (loop.models.root / loop.models.current() / "card.json").write_text(json.dumps(card))
        else:
            version = loop.models.save(
                {k: v for k, v in loop.current.items() if k != "version"}, card
            )
            loop.models.promote(version)
    elif reason == "answer":
        assert (
            client.post(
                API + "/answers",
                json={"item_id": item["item_id"], "label": "keep"},
                headers={"Idempotency-Key": "human"},
            ).status_code
            == 200
        )
        service.drain()
    enable(client, settings)
    response = client.post(ADS + f"/proposals/{item['item_id']}:execute")
    assert response.status_code == 409 and not state["posts"]


def test_changed_budget_and_second_snapshot_cannot_apply_twice(ads):
    client, _, settings, state, _ = ads
    item = sync(client).json()[0]
    enable(client, settings)
    state["budget"] = "15000"
    assert client.post(ADS + f"/proposals/{item['item_id']}:execute").status_code == 409
    state["budget"] = "10000"
    assert client.post(ADS + f"/proposals/{item['item_id']}:execute").json()["status"] == "executed"
    state["spend"] = "45.00"
    assert sync(client).status_code == 200
    assert len(state["posts"]) == 1
    assert client.get(ADS + "/log").json()["items"][-1]["event"] == "blocked"


def test_campaign_level_and_missing_cpa_do_not_invent_measurements(ads):
    client, _, _, state, _ = ads
    state["cpa"] = None
    # Missing CPA is absent rather than a fake zero.
    item = sync(client, "campaign").json()[0]
    assert item["insight"]["target_id"] == "201" and item["insight"]["cpa"] is None


def test_missing_adset_budget_never_changes_campaign_budget(ads):
    client, service, settings, state, _ = ads
    service._loops["ads"].action = "raise budget"
    state["budget"] = None
    enable(client, settings)
    item = sync(client).json()[0]
    assert item["status"] == "proposed" and not state["posts"]
    assert not any(path.endswith("/201") for _, path, _ in state["calls"])


@pytest.mark.parametrize("misconfiguration", ["mapping", "sandbox", "token", "labels"])
def test_unmapped_real_accounts_empty_token_and_wrong_labels_fail_closed(ads, misconfiguration):
    client, service, settings, state, _ = ads
    if misconfiguration == "mapping":
        settings.ads_workspace_accounts = {}
    elif misconfiguration == "sandbox":
        settings.ads_sandbox_accounts = []
    elif misconfiguration == "token":
        settings.ads_token = SecretStr("")
    elif misconfiguration == "labels":
        with service.store.connect(write=True) as db:
            db.execute(
                "UPDATE workspaces SET config_json=? WHERE workspace_id='ads'",
                ('{"labels":["a","b"]}',),
            )
    response = sync(client)
    assert response.status_code in (409, 503) and not state["calls"]
    assert "ads-test-secret" not in response.text


@pytest.mark.parametrize("invalid", ["NaN", "-1", "Infinity", "not money"])
def test_invalid_insight_numbers_rejected_without_prediction_or_write(ads, invalid):
    client, service, _, state, _ = ads
    state["spend"] = invalid
    assert sync(client).status_code == 503
    assert not service.store.predictions("ads") and not state["posts"]


def test_cross_workspace_proposal_and_logs_are_isolated(ads):
    client, _, _, _, _ = ads
    item = sync(client).json()[0]
    other = "/api/v1/workspaces/other/integrations/ads"
    assert client.post(other + f"/proposals/{item['item_id']}:execute").status_code == 404
    assert client.get(other + "/log").json()["items"] == []
    assert client.put(other + "/preferences", json={"auto_act": True}).status_code == 200
    assert not client.get(ADS + "/status").json()["auto_act"]


def test_errors_and_settings_never_expose_token(ads, caplog):
    client, _, settings, state, _ = ads
    state["failure"] = "read"
    response = sync(client)
    assert response.status_code == 503
    assert (
        "ads-test-secret"
        not in response.text + caplog.text + repr(settings) + settings.model_dump_json()
    )


def test_pagination_uses_fixed_graph_path_and_rejects_repeated_cursors(ads):
    client, service, _, state, _ = ads
    state["paging"] = {
        "next": "https://evil.invalid/?access_token=private",
        "cursors": {"after": "again"},
    }
    assert sync(client).status_code == 503
    assert len(state["calls"]) == 2
    assert all(path == "/v25.0/act_900/insights" for _, path, _ in state["calls"])
    assert state["calls"][1][2]["after"] == "again"
    assert not service.store.predictions("ads")


def test_concurrent_execution_replays_mutation_once(ads):
    from concurrent.futures import ThreadPoolExecutor

    client, _, settings, state, _ = ads
    item = sync(client).json()[0]
    enable(client, settings)
    url = ADS + f"/proposals/{item['item_id']}:execute"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(url), range(2)))
    assert all(response.status_code == 200 for response in responses)
    assert len(state["posts"]) == 1
    assert sorted(response.json()["duplicate"] for response in responses) == [False, True]


def test_interrupted_execution_is_durably_unknown_and_never_retried(ads):
    client, service, _, state, transport = ads
    item = sync(client).json()[0]
    old = client.app.state.meta_ads.get("ads", item["item_id"])
    old.status = "executing"
    with service.store.connect(write=True) as db:
        db.execute(
            "UPDATE ads_proposals SET proposal_json=? WHERE workspace_id=? AND item_id=?",
            (old.model_dump_json(), "ads", item["item_id"]),
        )
        db.execute(
            "INSERT INTO ads_execution_claims VALUES (?,?,?,?)",
            ("ads", old.insight.scope(), item["item_id"], "executing"),
        )
    client.app.state.meta_ads = AdsService(
        client.app.state.meta_ads.settings, service, transport=transport
    )
    url = ADS + f"/proposals/{item['item_id']}:execute"
    assert client.post(url).json()["status"] == "delivery_unknown"
    assert client.post(url).json()["duplicate"]
    assert not state["posts"]
    assert (
        client.get(ADS + "/log").json()["items"][-1]["details"]["reason"] == "interrupted_execution"
    )


def test_insight_id_cannot_reuse_a_conflicting_existing_prediction(ads):
    from backend.meta_ads import AdsSyncRequest

    client, _, _, state, _ = ads
    service = client.app.state.meta_ads
    body = AdsSyncRequest(**DATES)
    row = service.client.insights("act_900", body)[0]
    insight = service._insight("act_900", "adset", row, body)
    assert (
        client.post(
            API + "/predict",
            json={"items": [{"item_id": insight.item_id(), "text": "Different content"}]},
        ).status_code
        == 200
    )
    assert sync(client).status_code == 409
    assert not state["posts"]
