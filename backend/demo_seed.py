"""Disposable, explicitly synthetic product story. No checkpoints, secrets or network calls.

Run from the repository root: UV_NO_SYNC=true uv run python -m backend.demo_seed.
Every launch creates a NEW temporary directory. Existing directories with contents are refused.
This is a scripted product demonstration, not model evaluation or live Meta verification.
"""

import argparse
import tempfile
from pathlib import Path

import httpx
import uvicorn
from fastapi.responses import JSONResponse

from backend.autonomy import Policy, kind
from backend.config import Settings
from backend.harness import Harness
from backend.meta_ads import AdsService
from backend.schemas import EvalCard
from backend.store import Store, encode, timestamp
from backend.workspace import WorkspaceService

BUSINESS = "default"
MODEL = "synthetic-scripted-demo"
AD_OPTIONS = ["pause", "keep", "raise budget", "lower budget"]
MESSAGE_OPTIONS = [
    "Share opening hours",
    "Send delivery update",
    "Offer collection",
    "Ask a person",
]
ACCOUNT_OPTIONS = ["Keep access", "Remove access", "Ask a person"]
ASSETS = [
    {"key": "ads:9001", "id": "9001", "kind": "ads", "name": "Synthetic Fold & Field · Ads"},
    {
        "key": "whatsapp:9002",
        "id": "9002",
        "kind": "whatsapp",
        "name": "Synthetic Fold & Field · Orders",
    },
    {
        "key": "messenger:9003",
        "id": "9003",
        "kind": "messenger",
        "name": "Synthetic Fold & Field · Page",
    },
    {
        "key": "instagram:9004",
        "id": "9004",
        "kind": "instagram",
        "name": "Synthetic Fold & Field · Studio",
    },
]
# Fictional business, products and customers throughout; monetary values are USD.
PENDING = [
    (
        "ads",
        "Weekend collection ad spent $184 for 4 orders. Cost per order is $46; your target is $24.",
        "pause",
        0.94,
    ),
    (
        "ads",
        "Linen tote ad: $126 spend, 9 orders, $14 per order and 2.8% "
        "click-through. Daily budget is $30.",
        "keep",
        0.97,
    ),
    (
        "ads",
        "New colour launch: $90 spend, no orders and 0.6% click-through after three days.",
        "pause",
        0.92,
    ),
    (
        "ads",
        "Returning-customer ad: 12 orders at $11 each. The $25 daily budget runs out before noon.",
        "raise budget",
        0.96,
    ),
    (
        "ads",
        "Studio visit ad: cost per booking rose from $18 to $29. Daily budget is $40.",
        "lower budget",
        0.93,
    ),
    (
        "whatsapp",
        "“Is the studio open this Sunday? I would like to see the new totes.”",
        "Share opening hours",
        0.98,
    ),
    (
        "whatsapp",
        "“My order FF-204 is due today. Could you check when it will arrive?”",
        "Send delivery update",
        0.96,
    ),
    (
        "whatsapp",
        "“Can I collect my moss-green tote tomorrow instead of having it shipped?”",
        "Offer collection",
        0.95,
    ),
    (
        "whatsapp",
        "“The strap arrived damaged. Could someone help arrange a replacement?”",
        "Ask a person",
        0.99,
    ),
    (
        "whatsapp",
        "“Are you open late on Friday? I finish work at six.”",
        "Share opening hours",
        0.97,
    ),
    (
        "messenger",
        "“I ordered two canvas pouches on Monday. Has order FF-216 left the studio?”",
        "Send delivery update",
        0.94,
    ),
    (
        "messenger",
        "“Could I pick up a gift at the studio on Saturday morning?”",
        "Offer collection",
        0.98,
    ),
    (
        "messenger",
        "“I was charged twice for order FF-219. Can somebody look into it?”",
        "Ask a person",
        0.99,
    ),
    ("messenger", "“What time does the studio close today?”", "Share opening hours", 0.96),
    (
        "messenger",
        "“Tracking has not moved for four days. This is a birthday gift.”",
        "Ask a person",
        0.93,
    ),
    (
        "instagram",
        "“Love the oat tote. Can I come by the studio to see it this weekend?”",
        "Share opening hours",
        0.97,
    ),
    (
        "instagram",
        "“Can you hold a rust-colour pouch for me to collect tomorrow?”",
        "Offer collection",
        0.95,
    ),
    (
        "instagram",
        "“My parcel says delivered but it is not here. Order FF-225.”",
        "Ask a person",
        0.98,
    ),
    (
        "instagram",
        "“Is order FF-231 on its way? I leave town next week.”",
        "Send delivery update",
        0.94,
    ),
    (
        "instagram",
        "“Can our community group place an order for 40 custom totes?”",
        "Ask a person",
        0.92,
    ),
]


def item(identifier, text):
    return {
        "id": identifier,
        "text": f"Synthetic · {text}",
        "merchant_key": "",
        "amount": None,
        "date": None,
    }


def decision(identifier, source, label, confidence, *, auto=False):
    options = AD_OPTIONS if source == "ads" else MESSAGE_OPTIONS if source else ACCOUNT_OPTIONS
    rest = (1 - confidence) / (len(options) - 1)
    return {
        "item_id": identifier,
        "source_app": source,
        "label": label,
        "prob": confidence,
        "probabilities": {option: confidence if option == label else rest for option in options},
        "auto": auto,
        "threshold": 0.9,
        "spot_check": False,
        "model_version": MODEL,
        "cp_lower_bound": None,
        "reasons": {
            "merchant_memory": False,
            "neighbours": [
                {
                    "item_id": f"synthetic-past-{identifier}",
                    "text": "Synthetic · You paused an ad after cost per order stayed "
                    "above your target."
                    if source == "ads"
                    else "Synthetic · You chose this action for a similar customer question.",
                    "label": label,
                    "similarity": 0.94,
                },
                {
                    "item_id": f"synthetic-past-2-{identifier}",
                    "text": "Synthetic · A second similar situation from your saved decisions.",
                    "label": label,
                    "similarity": 0.88,
                },
            ],
        },
    }


class SyntheticService(WorkspaceService):
    """Persist real reviews/undo, with explicitly SCRIPTED proposals and no model fitting."""

    def has_model(self, workspace):
        return workspace == BUSINESS

    def card(self, workspace):
        return EvalCard(model_version=MODEL, coverage=0)

    def answer(self, workspace, item_id, label, idempotency_key, *, source=None):
        with self._lock(workspace):
            previous = self.store.latest_prediction(workspace, item_id)
            if not previous or label not in previous["probabilities"]:
                raise ValueError("Choose an action offered on this synthetic decision.")
            before = self.autonomy.status(workspace, kind(previous.get("source_app")))
            result, created = self.store.answer(
                workspace,
                item_id,
                label,
                idempotency_key,
                {
                    "item_id": item_id,
                    "coverage": 0,
                    "model_version": MODEL,
                    "retrain": "history_updated",
                },
                review=(
                    kind(previous.get("source_app")),
                    previous["label"] == label,
                    previous["prob"],
                ),
            )
            if created and getattr(self, "harness", None):
                after = self.autonomy.status(workspace, kind(previous.get("source_app")))
                event = (
                    "Synthetic · Confirmed a suggestion"
                    if previous["label"] == label
                    else "Synthetic · Corrected a suggestion"
                )
                self.harness.log(
                    workspace,
                    f"{event}: {previous['label']} → {label}. Lower bound "
                    f"{before['lower_bound']:.2%} → {after['lower_bound']:.2%}.",
                    item_id,
                )
            return result

    def undo_answer(self, workspace, key):
        with self._lock(workspace):
            row, changed = self.store.undo_answer(workspace, key)
            if changed:
                self.harness.log(
                    workspace, "Synthetic · Review undone; previous answer restored", row["item_id"]
                )
            return {
                "item_id": row["item_id"],
                "coverage": 0,
                "model_version": MODEL,
                "retrain": "history_updated",
            }


def create_demo(directory: Path):
    """Only a new or completely empty directory is accepted. No database path option exists."""
    directory = Path(directory).absolute()
    if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
        raise ValueError(
            "Synthetic demo requires a new empty directory; existing data is never opened."
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".secnd-synthetic-demo").write_text(
        "Synthetic, disposable, scripted product story.\n"
    )
    # Explicit values override operator settings. Never load .env, a checkpoint, or a provider key.
    settings = Settings(
        _env_file=None,
        data_dir=directory,
        demo_mode=True,
        serving_backend="head",
        workspace_backends={},
        workspace_readers={},
        featurizer="char_tfidf",
        session_cookie_secure=False,
        agent_enabled=False,
        ads_send_enabled=False,
        whatsapp_send_enabled=False,
        meta_message_send_enabled=False,
        cold_start_paid=False,
        meta_api_key="",
        muse_api_key="",
        nebius_api_key="",
        meta_app_secret="",
        meta_app_id="",
        meta_login_config_id="",
        meta_token_key="",
        meta_verify_token="",
        whatsapp_token="",
        ads_token="",
        ads_workspace_accounts={},
        meta_workspace_accounts={},
        ads_sandbox_accounts=[],
    )
    service = SyntheticService(Store(settings.database_path), directory)
    service.store.create_workspace(
        BUSINESS,
        {
            "labels": AD_OPTIONS + MESSAGE_OPTIONS + ACCOUNT_OPTIONS,
            "calib": [],
            "featurizer": MODEL,
            "seed": 0,
        },
    )
    # Tables are initialized using production storage, with a transport that always refuses I/O.
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503, json={"error": {"message": "Synthetic demo: external transport disabled."}}
        )
    )
    ads = AdsService(settings, service, transport=transport)
    harness = Harness(settings, service, ads, None)
    service.harness = harness
    for decision_type, level, bar in [
        ("ad", "act", 95),
        ("inbox", "ask", 95),
        ("account", "act", 95),
    ]:
        service.autonomy.configure(BUSINESS, decision_type, Policy(level=level, bar=bar))
    # Actual statistical bounds from synthetic review counts, not invented percentages.
    with service.store.connect(write=True) as db:
        for decision_type, count in [("ad", 61), ("inbox", 40), ("account", 61)]:
            for i in range(count):
                db.execute(
                    "INSERT INTO autonomy_reviews VALUES (?,?,?,?,?,?,?)",
                    (
                        BUSINESS,
                        f"synthetic-{decision_type}-{i}",
                        f"synthetic-{decision_type}-past-{i}",
                        decision_type,
                        True,
                        0.99,
                        timestamp(),
                    ),
                )
    for i, (source, text, label, confidence) in enumerate(PENDING):
        identifier = f"synthetic-pending-{i:02}"
        service.store.record_predictions(
            BUSINESS, [item(identifier, text)], [decision(identifier, source, label, confidence)]
        )
    # A real saved correction drops Account below its bar, so its effective level becomes Ask.
    corrected_id = "synthetic-account-correction"
    service.store.record_predictions(
        BUSINESS,
        [
            item(
                corrected_id,
                "A temporary studio helper still has account access after their assignment ended.",
            )
        ],
        [decision(corrected_id, None, "Keep access", 0.99)],
    )
    service.answer(BUSINESS, corrected_id, "Remove access", "synthetic-correction")
    for i, action in enumerate(["pause", "keep", "lower budget"]):
        identifier = f"synthetic-acted-{i}"
        proposal_decision = decision(identifier, "ads", action, 0.99, auto=True)
        status = service.autonomy.status(BUSINESS, "ad")
        proposal_decision["autonomy"] = {
            key: status[key] for key in ("reviewed", "confirmed", "lower_bound", "effective_level")
        }
        proposal_decision["basis"] = {
            "kind": "personal_reviews",
            "confirmed": status["confirmed"],
            "reviewed": status["reviewed"],
            "lower_bound": status["lower_bound"],
        }
        service.store.record_predictions(
            BUSINESS,
            [
                item(
                    identifier,
                    [
                        "Paused the weekend collection ad after cost per order exceeded $24.",
                        "Kept the linen tote ad at its $30 daily budget.",
                        "Lowered the studio visit ad’s daily budget from $40 to $36.",
                    ][i],
                )
            ],
            [proposal_decision],
        )
        before = {
            "spend": ["184", "126", "203"][i],
            "cpa": ["46", "14", "29"][i],
            "ctr": ["0.9", "2.8", "1.2"][i],
            "currency": "USD",
            "since": "2026-10-10",
            "until": "2026-10-16",
        }
        after = {
            "spend": ["0", "140", "180"][i],
            "cpa": [None, "14", "22.50"][i],
            "ctr": [None, "2.9", "1.5"][i],
            "currency": "USD",
            "since": "2026-10-17",
            "until": "2026-10-23",
        }
        proposal = {
            "item_id": identifier,
            "action": action,
            "status": "synthetic_executed",
            "decision": proposal_decision,
            "insight": before,
        }
        with service.store.connect(write=True) as db:
            db.execute(
                "INSERT INTO ads_proposals VALUES (?,?,?,?)",
                (BUSINESS, identifier, encode(proposal), "2026-10-17T00:00:00+00:00"),
            )
            db.execute(
                "INSERT INTO ad_outcomes VALUES (?,?,?,?,?,?)",
                (
                    BUSINESS,
                    identifier,
                    encode(before),
                    encode(after),
                    "2026-10-24",
                    "2026-10-24T00:00:00+00:00",
                ),
            )
        harness.log(
            BUSINESS,
            f"Synthetic · Ren acted: {action}. Evidence: 61 of 61 confirmed; no Meta change.",
            identifier,
        )
    harness.log(
        BUSINESS, "Synthetic · Seven-day ad observations recorded; these do not prove causation"
    )
    # backend.api creates a module-level app on import. Disable its dotenv source too.
    previous_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    try:
        from backend.api import create_app
    finally:
        Settings.model_config["env_file"] = previous_env_file
    app = create_app(settings, workspace_service=service, meta_transport=transport)
    connection = {
        "connected": True,
        "assets": ASSETS,
        "selected": [asset["key"] for asset in ASSETS],
        "access_note": "Synthetic Meta connection; no OAuth request.",
    }

    @app.middleware("http")
    async def synthetic_story(request, call_next):
        path = request.url.path
        root = f"/api/v1/workspaces/{BUSINESS}"
        if path == "/api/demo/connect" and request.method == "POST":
            connection["connected"] = True
            connection["selected"] = []
            response = JSONResponse(connection)
        elif path == f"{root}/integrations/meta/connection":
            response = JSONResponse(connection)
        elif path == f"{root}/integrations/meta/accounts" and request.method == "PUT":
            body = await request.json()
            selected = body.get("assets", [])
            if not isinstance(selected, list) or not set(selected).issubset(
                {asset["key"] for asset in ASSETS}
            ):
                response = JSONResponse(
                    {"message": "Choose synthetic accounts from the list."}, status_code=422
                )
            else:
                connection["selected"] = selected
                response = JSONResponse(connection)
        elif path == f"{root}/answers" and request.method == "POST":
            body = await request.json()
            try:
                response = JSONResponse(
                    service.answer(
                        BUSINESS,
                        body["item_id"],
                        body["label"],
                        request.headers.get("idempotency-key", ""),
                    )
                )
            except (KeyError, ValueError) as error:
                response = JSONResponse({"message": str(error)}, status_code=422)
        elif path == f"{root}/integrations/meta/status":
            response = JSONResponse(
                {
                    "accounts": {
                        source: [
                            asset["id"]
                            for asset in ASSETS
                            if asset["kind"] == source and asset["key"] in connection["selected"]
                        ]
                        for source in ("whatsapp", "messenger", "instagram")
                    },
                    "model_ready": False,
                    "signing_configured": False,
                    "whatsapp_send_enabled": False,
                    "message_send_enabled": False,
                }
            )
        elif path == f"{root}/integrations/ads/status":
            response = JSONResponse(
                {
                    "demo": True,
                    "auto_act": False,
                    "account": "act_9001" if "ads:9001" in connection["selected"] else None,
                    "sandbox_allowed": "ads:9001" in connection["selected"],
                    "token_configured": False,
                    "server_send_enabled": False,
                    "workspace_ready": False,
                    "budget_step_percent": 10,
                }
            )
        elif path == f"{root}/serving":
            response = JSONResponse(
                {
                    "backend": "head",
                    "model_id": MODEL,
                    "device": "cpu",
                    "synthetic": True,
                    "business_name": "Synthetic Fold & Field",
                    "scope_note": "Synthetic scripted decisions; no running Ren checkpoint.",
                }
            )
        elif path in [f"{root}/feed", f"{root}/queue", f"{root}/overridden"]:
            response = JSONResponse(
                service.page(
                    BUSINESS,
                    path.rsplit("/", 1)[1],
                    request.query_params.get("cursor"),
                    int(request.query_params.get("limit", "20")),
                )
            )
        elif path == f"{root}/integrations/meta/oauth/start":
            response = JSONResponse(
                {
                    "message": "Synthetic demo cannot open live OAuth. Use /?setup=1 for "
                    "the synthetic walkthrough."
                },
                status_code=409,
            )
        elif request.method not in ("GET", "HEAD") and not (
            path
            in [
                f"{root}/answers",
                f"{root}/answers:undo",
                f"{root}/agent",
                "/api/v1/agent/kill",
                "/api/preview",
            ]
            or path.startswith(f"{root}/autonomy/")
        ):
            response = JSONResponse(
                {
                    "message": "This synthetic demo permits reviews and settings only. "
                    "Imports, sign-in, generation and provider writes are disabled."
                },
                status_code=409,
            )
        else:
            response = await call_next(request)
        response.headers["X-Secnd-Demo"] = "synthetic"
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="secnd-synthetic-demo-") as folder:
        app = create_demo(Path(folder))
        print(
            f"Synthetic demo only · http://127.0.0.1:{args.port}/?setup=1 · fresh disposable data"
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
