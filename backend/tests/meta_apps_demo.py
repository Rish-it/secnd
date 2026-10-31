"""Synthetic connector console, deterministic model fixture, mocked Graph HTTP only."""

import argparse
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pandas as pd
import uvicorn

from backend.api import create_app
from backend.config import Settings
from backend.meta_ads import ACTIONS, AdsService
from backend.store import Store
from backend.tests.test_meta_ads import AdsLoop
from backend.workspace import WorkspaceService


class DemoAds(AdsService):
    def status(self, workspace):
        return super().status(workspace).model_copy(update={"demo": True})


def create_demo(directory):
    settings = Settings(
        data_dir=directory,
        _env_file=None,
        demo_mode=True,
        meta_app_secret="demo-only",
        meta_verify_token="demo-only",
        ads_token="demo-only",
        whatsapp_send_enabled=False,
        ads_send_enabled=False,
        cold_start_paid=False,
        meta_workspace_accounts={
            "whatsapp:100000000000002": "default",
            "messenger:100000000000003": "default",
            "instagram:100000000000005": "default",
        },
        ads_workspace_accounts={"default": "act_900"},
        ads_sandbox_accounts=["act_900"],
    )
    service = WorkspaceService(Store(settings.database_path), directory, AdsLoop)
    service.create(
        "default",
        ACTIONS,
        pd.DataFrame([dict(id="hold", text="held-out fixture", label="keep", merchant_key="")]),
        featurizer="char_tfidf",
    )
    service._runtime("default").retrain()
    service._loops["default"].automatic = False

    def respond(request):
        if request.method != "GET":
            raise AssertionError("The browser demo cannot send an Ads mutation.")
        if request.url.path.endswith("/insights"):
            dates = json.loads(request.url.params["time_range"])
            return httpx.Response(
                200,
                json={
                    "data": [
                        dict(
                            account_id="900",
                            account_currency="USD",
                            campaign_id="201",
                            campaign_name="Synthetic launch campaign",
                            adset_id="301",
                            adset_name="Synthetic launch ad set",
                            spend="42.00",
                            ctr="1.75",
                            cost_per_action_type=[{"action_type": "purchase", "value": "7.00"}],
                            date_start=dates["since"],
                            date_stop=dates["until"],
                        )
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": request.url.path.rsplit("/", 1)[-1],
                "account_id": "900",
                "daily_budget": "10000",
            },
        )

    app = create_app(settings, workspace_service=service)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original(app):
            app.state.meta_ads = DemoAds(settings, service, transport=httpx.MockTransport(respond))
            service.harness.ads = app.state.meta_ads
            yield

    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8032)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="secnd-meta-") as folder:
        uvicorn.run(create_demo(Path(folder)), host="127.0.0.1", port=args.port, access_log=False)
