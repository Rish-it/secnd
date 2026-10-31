"""Sandbox Marketing insights -> Ren decisions. Writes require both explicit switches."""

import hashlib
import json
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.config import Settings
from backend.connector_source import attach_source
from backend.schemas import Decision, Error, WorkspaceId
from backend.store import encode, timestamp
from backend.workspace import WorkspaceService

AdAction = Literal["pause", "keep", "raise budget", "lower budget"]
ACTIONS = ["pause", "keep", "raise budget", "lower budget"]
MetaId = Annotated[str, Field(pattern=r"^\d{1,32}$")]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False, max_digits=20, decimal_places=6)]


class AdsUnavailable(RuntimeError):
    pass


class ActionBlocked(ValueError):
    pass


class AdsSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["campaign", "adset"] = "adset"
    since: date = Field(default_factory=lambda: date.today() - timedelta(days=1))
    until: date = Field(default_factory=lambda: date.today() - timedelta(days=1))

    @model_validator(mode="after")
    def dates(self):
        if not 0 <= (self.until - self.since).days <= 30 or self.until >= date.today():
            raise ValueError("Choose a completed date range of at most 31 days.")
        return self


class Insight(BaseModel):
    account_id: MetaId
    level: Literal["campaign", "adset"]
    target_id: MetaId
    campaign_id: MetaId
    name: str = Field(min_length=1, max_length=500)
    since: date
    until: date
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    spend: Money
    cpa: Money | None
    cpa_action_type: str
    ctr: Money | None
    # Graph budgets are integer minor currency units; never infer a currency multiplier.
    daily_budget: Annotated[int, Field(gt=0, le=10**15)] | None

    def scope(self):
        return encode(
            [
                self.account_id,
                self.level,
                self.target_id,
                self.since.isoformat(),
                self.until.isoformat(),
            ]
        )

    def item_id(self):
        content = self.model_dump_json()
        return "ads-" + hashlib.sha256(content.encode()).hexdigest()[:48]

    def text(self):
        return (
            f"Meta Ads {self.level}: {self.name}. Date range {self.since} to {self.until}. "
            f"Spend {self.spend} {self.currency}; CPA ({self.cpa_action_type}) "
            f"{self.cpa if self.cpa is not None else 'unavailable'}; "
            f"CTR {self.ctr if self.ctr is not None else 'unavailable'} percent. "
            f"Daily budget {self.daily_budget if self.daily_budget is not None else 'unavailable'} "
            "minor currency units. Choose pause, keep, raise budget, or lower budget."
        )


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auto_act: bool = False


class AdsStatus(Preference):
    demo: bool = False
    account: str | None
    sandbox_allowed: bool
    token_configured: bool
    server_send_enabled: bool
    workspace_ready: bool
    budget_step_percent: int


class Proposal(BaseModel):
    item_id: str
    insight: Insight
    action: AdAction
    decision: Decision
    status: Literal[
        "pending", "proposed", "executing", "executed", "delivery_unknown", "kept", "undone"
    ]
    duplicate: bool = False


class MarketingClient:
    def __init__(self, settings: Settings, transport=None, *, token=None, anonymous=False):
        self.settings = settings
        self.transport = transport
        self.token, self.anonymous = token, anonymous

    def request(self, method, path, **kwargs):
        token = self.settings.ads_token.get_secret_value() if self.token is None else self.token
        if not token and not self.anonymous:
            raise AdsUnavailable("Configure the sandbox Ads token before fetching insights.")
        try:
            with httpx.Client(
                base_url=f"https://graph.facebook.com/{self.settings.meta_graph_version}/",
                headers={"Authorization": f"Bearer {token}"} if token else {},
                transport=self.transport,
                timeout=10,
                follow_redirects=False,
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or "error" in payload:
                    raise ValueError
                return payload
        except (httpx.HTTPError, ValueError):
            # Never expose Graph bodies, request headers, tokens, or provider exception text.
            raise AdsUnavailable(
                "Meta Ads request failed. Check sandbox access and permissions."
            ) from None

    def insights(self, account, body: AdsSyncRequest):
        fields = [
            "account_id",
            "account_currency",
            "campaign_id",
            "campaign_name",
            "spend",
            "cost_per_action_type",
            "ctr",
            "date_start",
            "date_stop",
        ]
        if body.level == "adset":
            fields += ["adset_id", "adset_name"]
        params = {
            "fields": ",".join(fields),
            "level": body.level,
            "time_range": encode(
                {"since": body.since.isoformat(), "until": body.until.isoformat()}
            ),
            "limit": "100",
        }
        rows, seen = [], set()
        for _ in range(10):
            page = self.request("GET", f"{account}/insights", params=params)
            if not isinstance(page.get("data"), list):
                raise AdsUnavailable("Meta Ads returned invalid insights.")
            rows.extend(page["data"])
            paging = page.get("paging", {})
            if not isinstance(paging, dict):
                raise AdsUnavailable("Meta Ads returned invalid pagination.")
            if not paging.get("next"):
                return rows
            cursors = paging.get("cursors", {})
            after = cursors.get("after") if isinstance(cursors, dict) else None
            if not isinstance(after, str) or not 1 <= len(after) <= 512 or after in seen:
                raise AdsUnavailable("Meta Ads returned invalid pagination.")
            # Do not follow Graph's next URL: it can contain credentials or an untrusted host.
            seen.add(after)
            params["after"] = after
        raise AdsUnavailable("Insights exceed the sandbox import limit. Choose a smaller range.")

    def snapshot(self, target_id, account):
        value = self.request(
            "GET", target_id, params={"fields": "id,account_id,daily_budget,status"}
        )
        if value.get("id") != target_id or str(value.get("account_id")) != account.removeprefix(
            "act_"
        ):
            raise AdsUnavailable("The ad target does not belong to the mapped sandbox account.")
        budget = value.get("daily_budget")
        if budget is not None and (not str(budget).isdigit() or not 0 < int(budget) <= 10**15):
            raise AdsUnavailable("Meta Ads returned an invalid daily budget.")
        return {
            "daily_budget": int(budget) if budget is not None else None,
            "status": value.get("status"),
        }

    def target(self, target_id, account):
        return self.snapshot(target_id, account)["daily_budget"]


class AdsService:
    def __init__(self, settings: Settings, service: WorkspaceService, *, transport=None):
        self.settings, self.service = settings, service
        self.client = MarketingClient(settings, transport)
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS ads_preferences (
                workspace_id TEXT PRIMARY KEY REFERENCES workspaces(workspace_id),
                auto_act BOOLEAN NOT NULL DEFAULT 0
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS ads_proposals (
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                item_id TEXT NOT NULL, proposal_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, item_id)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS ads_execution_claims (
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                scope TEXT NOT NULL, item_id TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(workspace_id, scope)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS ads_action_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                item_id TEXT, event TEXT NOT NULL, details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")

        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS ads_reversals (
                workspace_id TEXT NOT NULL, item_id TEXT NOT NULL, before_json TEXT NOT NULL,
                after_json TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(workspace_id,item_id))""")

    def log(self, workspace, item_id, event, details):
        with self.service.store.connect(write=True) as db:
            self._log(db, workspace, item_id, event, details)

    @staticmethod
    def _log(db, workspace, item_id, event, details):
        db.execute(
            "INSERT INTO ads_action_log(workspace_id,item_id,event,details_json,created_at) "
            "VALUES (?,?,?,?,?)",
            (workspace, item_id, event, encode(details), timestamp()),
        )

    def client_for(self, workspace):
        oauth = getattr(self.service, "meta_oauth", None)
        if oauth and oauth.owned(workspace):
            account = self.account(workspace)
            return MarketingClient(
                self.settings, self.client.transport, token=oauth.token(workspace, "ads", account)
            )
        return self.client

    def mapped_account(self, workspace):
        oauth = getattr(self.service, "meta_oauth", None)
        if oauth and oauth.owned(workspace):
            return next(
                (r["asset_id"] for r in oauth.mappings(workspace) if r["kind"] == "ads"), None
            )
        return (
            self.settings.ads_workspace_accounts.get(workspace) if self.settings.demo_mode else None
        )

    def account(self, workspace):
        account = self.mapped_account(workspace)
        if (
            account not in self.settings.ads_sandbox_accounts
            or not account
            or not (
                account.startswith("act_") and account[4:].isdigit() and 1 <= len(account[4:]) <= 32
            )
        ):
            raise ActionBlocked("Map an allowed sandbox Ads account to this workspace first.")
        return account

    def status(self, workspace):
        try:
            config = self.service.store.workspace(workspace)
        except KeyError:
            config = {}
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT auto_act FROM ads_preferences WHERE workspace_id=?", (workspace,)
            ).fetchone()
        account = self.mapped_account(workspace)
        return AdsStatus(
            auto_act=bool(row and row[0]),
            account=account,
            sandbox_allowed=bool(account and account in self.settings.ads_sandbox_accounts),
            token_configured=bool(self.mapped_account(workspace))
            if (
                getattr(self.service, "meta_oauth", None)
                and self.service.meta_oauth.owned(workspace)
            )
            else bool(self.settings.demo_mode and self.settings.ads_token.get_secret_value()),
            server_send_enabled=self.settings.ads_send_enabled,
            workspace_ready=set(config.get("labels", [])) == set(ACTIONS)
            and self.service.has_model(workspace),
            budget_step_percent=self.settings.ads_budget_step_percent,
        )

    def preferences(self, workspace, body: Preference):
        self.service.store.workspace(workspace)
        with self.service._lock(workspace), self.service.store.connect(write=True) as db:
            db.execute(
                "INSERT INTO ads_preferences VALUES (?,?) ON CONFLICT(workspace_id) "
                "DO UPDATE SET auto_act=excluded.auto_act",
                (workspace, body.auto_act),
            )
            self._log(db, workspace, None, "auto_act_changed", {"auto_act": body.auto_act})
        return self.status(workspace)

    def _insight(self, account, level, row, body, client=None, *, include_budget=True):
        try:
            target = row[f"{level}_id"]
            # Validate IDs before using them in a URL.
            if not isinstance(target, str) or not target.isdigit() or not 1 <= len(target) <= 32:
                raise ValueError
            if row["account_id"] != account.removeprefix("act_"):
                raise ValueError
            if (
                row["date_start"] != body.since.isoformat()
                or row["date_stop"] != body.until.isoformat()
            ):
                raise ValueError
            costs = row.get("cost_per_action_type", [])
            if not isinstance(costs, list):
                raise ValueError
            cpa = next(
                (
                    c["value"]
                    for c in costs
                    if c["action_type"] == self.settings.ads_cpa_action_type
                ),
                None,
            )
            return Insight(
                account_id=row["account_id"],
                level=level,
                target_id=target,
                campaign_id=row["campaign_id"],
                name=row[f"{level}_name"],
                since=row["date_start"],
                until=row["date_stop"],
                currency=row["account_currency"],
                spend=row["spend"],
                cpa=cpa,
                cpa_action_type=self.settings.ads_cpa_action_type,
                ctr=row.get("ctr"),
                daily_budget=(client or self.client).target(target, account)
                if include_budget
                else None,
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            raise AdsUnavailable("Meta Ads returned invalid insight fields.") from None

    def get(self, workspace, item_id):
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT proposal_json FROM ads_proposals WHERE workspace_id=? AND item_id=?",
                (workspace, item_id),
            ).fetchone()
        if row is None:
            raise KeyError(item_id)
        proposal = Proposal.model_validate_json(row[0])
        if proposal.status == "executing":
            # Process loss between claim and response: never repeat a potentially applied write.
            proposal.status = "delivery_unknown"
            with self.service.store.connect(write=True) as db:
                db.execute(
                    "UPDATE ads_proposals SET proposal_json=? WHERE workspace_id=? AND item_id=?",
                    (proposal.model_dump_json(), workspace, item_id),
                )
                db.execute(
                    "UPDATE ads_execution_claims SET status='delivery_unknown' "
                    "WHERE workspace_id=? AND scope=?",
                    (workspace, proposal.insight.scope()),
                )
                self._log(
                    db,
                    workspace,
                    item_id,
                    "delivery_unknown",
                    {"action": proposal.action, "reason": "interrupted_execution"},
                )
        return proposal

    def sync(self, workspace, body):
        with self.service._lock(workspace):
            self.service.store.workspace(workspace)
            account = self.account(workspace)
            if not self.status(workspace).workspace_ready:
                raise ActionBlocked(
                    "Train this workspace with pause, keep, raise budget "
                    "and lower budget labels first."
                )
            insights = [
                self._insight(account, body.level, row, body, self.client_for(workspace))
                for row in self.client_for(workspace).insights(account, body)
            ]
            proposals = []
            for insight in insights:
                item_id = insight.item_id()
                try:
                    old = self.get(workspace, item_id)
                except KeyError:
                    old = None
                if old:
                    old.duplicate = True
                    proposals.append(old)
                    continue
                decision = self.service.store.latest_prediction(workspace, item_id)
                if decision is not None:
                    saved = self.service.store.item(workspace, item_id)
                    if (saved["text"], saved["amount"], saved["date"], saved["merchant_key"]) != (
                        insight.text(),
                        None,
                        None,
                        "",
                    ):
                        raise ActionBlocked("An insight item ID already has different content.")
                if decision is None:
                    decision = self.service.route(
                        workspace,
                        pd.DataFrame(
                            [
                                dict(
                                    id=item_id,
                                    text=insight.text(),
                                    amount=None,
                                    date=None,
                                    merchant_key="",
                                    source_app="ads",
                                )
                            ]
                        ),
                    )[0]
                decision = attach_source(self.service, workspace, item_id, "ads")
                if decision["label"] not in ACTIONS:
                    raise ActionBlocked("Ren returned a label outside the Ads action schema.")
                proposal = Proposal(
                    item_id=item_id,
                    insight=insight,
                    action=decision["label"],
                    decision=decision,
                    status="proposed" if decision["auto"] else "pending",
                )
                with self.service.store.connect(write=True) as db:
                    db.execute(
                        "INSERT INTO ads_proposals VALUES (?,?,?,?)",
                        (workspace, item_id, proposal.model_dump_json(), timestamp()),
                    )
                    self._log(
                        db,
                        workspace,
                        item_id,
                        "proposed",
                        {
                            "action": proposal.action,
                            "model_version": decision["model_version"],
                            "auto": decision["auto"],
                        },
                    )
                harness = getattr(self.service, "harness", None)
                if harness:
                    harness.outcomes.track(workspace, proposal.model_dump(mode="json"), timestamp())
                if self.settings.ads_send_enabled and self.status(workspace).auto_act:
                    try:
                        proposal = self.execute(workspace, item_id)
                    except ActionBlocked as error:
                        self.log(workspace, item_id, "blocked", {"reason": str(error)})
                proposals.append(proposal)
            return proposals

    def controls(self, workspace, *, automatic=False):
        harness = getattr(self.service, "harness", None)
        if harness and harness.killed():
            raise ActionBlocked("The global kill switch is on.")
        if (
            getattr(self.service, "meta_oauth", None)
            and self.service.meta_oauth.owned(workspace)
            and not harness.status(workspace)["enabled"]
        ):
            raise ActionBlocked("This workspace is stopped.")
        if not self.settings.ads_send_enabled:
            raise ActionBlocked("Ads execution switch is off. Proposal only.")
        if automatic and not self.status(workspace).auto_act:
            raise ActionBlocked("Workspace auto-act is off. Proposal only.")

    def execute(self, workspace, item_id, *, confirmed_action=None):
        with self.service._lock(workspace):
            if getattr(self.service, "harness", None) and self.service.harness.killed():
                raise ActionBlocked("The global kill switch is on.")
            if (
                getattr(self.service, "meta_oauth", None)
                and self.service.meta_oauth.owned(workspace)
                and not self.service.harness.status(workspace)["enabled"]
            ):
                raise ActionBlocked("This workspace is stopped.")
            proposal = self.get(workspace, item_id)
            if proposal.status in ("executed", "delivery_unknown", "kept", "undone"):
                proposal.duplicate = True
                return proposal
            account = self.account(workspace)
            if account.removeprefix("act_") != proposal.insight.account_id:
                raise ActionBlocked("Workspace account mapping changed. Fetch new insights.")
            if not self.settings.ads_send_enabled:
                raise ActionBlocked("Ads execution switch is off. Proposal only.")
            if confirmed_action is None and not self.status(workspace).auto_act:
                raise ActionBlocked("Workspace auto-act is off. Proposal only.")
            if confirmed_action is not None:
                labels = [
                    r
                    for r in self.service.store.labels(workspace)
                    if r["item_id"] == item_id and r["source"] in ("answer", "spot_check")
                ]
                if (
                    not labels
                    or labels[-1]["label"] != confirmed_action
                    or confirmed_action not in ACTIONS
                ):
                    raise ActionBlocked("Confirm the action in your inbox first.")
                if self.service.autonomy.policy(workspace, "ad").level == "watch":
                    raise ActionBlocked("Watch mode saves proposals only.")
                proposal.action = confirmed_action
            autonomy = getattr(self.service, "autonomy", None)
            if (
                confirmed_action is None
                and autonomy
                and autonomy.active(workspace)
                and not autonomy.allowed(workspace, "ad", proposal.action)
            ):
                raise ActionBlocked(
                    "This action needs your confirmation under the workspace autonomy policy."
                )
            decision, card = proposal.decision, self.service.card(workspace)
            if confirmed_action is None and not (
                decision.auto
                and not decision.spot_check
                and decision.threshold is not None
                and decision.prob >= decision.threshold
                and (
                    (autonomy and autonomy.authorizes(workspace, decision))
                    or (
                        decision.automatic_basis != "workspace_bound"
                        and card.cp_lower_bound is not None
                        and card.cp_lower_bound >= 0.99
                        and card.threshold == decision.threshold
                        and card.model_version == decision.model_version
                    )
                )
                and not self.service.metrics(workspace)["guarded"]
            ):
                raise ActionBlocked(
                    "This Ren decision is not currently certified. Answer it in the inbox."
                )
            # A correction in the inbox always cancels the original automatic action.
            if confirmed_action is None and any(
                row["item_id"] == item_id and row["source"] in ("answer", "spot_check")
                for row in self.service.store.labels(workspace)
            ):
                raise ActionBlocked(
                    "This proposal has a human answer. Fetch new insights before acting."
                )
            payload = {}
            if proposal.action == "pause":
                payload = {"status": "PAUSED"}
            elif proposal.action in ("raise budget", "lower budget"):
                previous = proposal.insight.daily_budget
                if previous is None:
                    raise ActionBlocked(
                        "No ad-level daily budget. "
                        "Campaign-budget changes require a separate proposal."
                    )
                step = Decimal(self.settings.ads_budget_step_percent) / 100
                factor = 1 + step if proposal.action == "raise budget" else 1 - step
                budget = int((previous * factor).to_integral_value(rounding=ROUND_HALF_UP))
                if not 0 < budget <= 10**15 or budget == previous:
                    raise ActionBlocked("Budget adjustment is outside the supported range.")
                payload = {"daily_budget": str(budget)}
            snapshot = self.client_for(workspace).snapshot(proposal.insight.target_id, account)
            self.controls(workspace, automatic=confirmed_action is None)
            if snapshot["daily_budget"] != proposal.insight.daily_budget:
                raise ActionBlocked("Daily budget changed since this proposal. Fetch new insights.")
            with self.service.store.connect(write=True) as db:
                old = db.execute(
                    "SELECT item_id,status FROM ads_execution_claims "
                    "WHERE workspace_id=? AND scope=?",
                    (workspace, proposal.insight.scope()),
                ).fetchone()
                if old:
                    raise ActionBlocked(
                        "An action was already attempted for this target and date range. "
                        "Review its log."
                    )
                db.execute(
                    "INSERT INTO ads_execution_claims VALUES (?,?,?,?)",
                    (workspace, proposal.insight.scope(), item_id, "executing"),
                )
                before = {}
                if "daily_budget" in payload:
                    before = {"daily_budget": str(snapshot["daily_budget"])}
                elif "status" in payload and snapshot["status"] in ("ACTIVE", "PAUSED"):
                    before = {"status": snapshot["status"]}
                if before:
                    db.execute(
                        "INSERT OR REPLACE INTO ads_reversals VALUES (?,?,?,?,?)",
                        (workspace, item_id, encode(before), encode(payload), "available"),
                    )
                proposal.status = "executing"
                db.execute(
                    "UPDATE ads_proposals SET proposal_json=? WHERE workspace_id=? AND item_id=?",
                    (proposal.model_dump_json(), workspace, item_id),
                )
                self._log(
                    db,
                    workspace,
                    item_id,
                    "execution_started",
                    {"action": proposal.action, "payload": payload},
                )
            if not payload:
                proposal.status = "kept"
            else:
                try:
                    result = self.client_for(workspace).request(
                        "POST", proposal.insight.target_id, data=payload
                    )
                    proposal.status = (
                        "executed" if result.get("success") is True else "delivery_unknown"
                    )
                except AdsUnavailable:
                    proposal.status = "delivery_unknown"
            with self.service.store.connect(write=True) as db:
                db.execute(
                    "UPDATE ads_execution_claims SET status=? WHERE workspace_id=? AND scope=?",
                    (proposal.status, workspace, proposal.insight.scope()),
                )
                db.execute(
                    "UPDATE ads_proposals SET proposal_json=? WHERE workspace_id=? AND item_id=?",
                    (proposal.model_dump_json(), workspace, item_id),
                )
                self._log(
                    db,
                    workspace,
                    item_id,
                    "executed" if proposal.status in ("executed", "kept") else "delivery_unknown",
                    {"action": proposal.action, "status": proposal.status},
                )
            return proposal

    def undo(self, workspace, item_id):
        with self.service._lock(workspace):
            self.controls(workspace)
            proposal = self.get(workspace, item_id)
            if proposal.status == "undone":
                return proposal
            if not self.settings.ads_send_enabled or proposal.status != "executed":
                raise ActionBlocked(
                    "Only a completed action can be undone with the execution switch on."
                )
            account = self.account(workspace)
            if account.removeprefix("act_") != proposal.insight.account_id:
                raise ActionBlocked("Reconnect the original sandbox account before undoing.")
            with self.service.store.connect() as db:
                row = db.execute(
                    "SELECT * FROM ads_reversals WHERE workspace_id=? AND item_id=?",
                    (workspace, item_id),
                ).fetchone()
            if not row or row["status"] != "available":
                raise ActionBlocked("No safe undo is available. Review the action log.")
            client = self.client_for(workspace)
            current = client.snapshot(proposal.insight.target_id, account)
            if any(str(current.get(k)) != str(v) for k, v in json.loads(row["after_json"]).items()):
                raise ActionBlocked(
                    "The target changed after this action. Review it before restoring."
                )
            self.controls(workspace)
            with self.service.store.connect(write=True) as db:
                claimed = db.execute(
                    "UPDATE ads_reversals SET status='undoing' WHERE workspace_id=? "
                    "AND item_id=? AND status='available'",
                    (workspace, item_id),
                ).rowcount
                if claimed != 1:
                    raise ActionBlocked("Undo was already attempted. Review its action log.")
                self._log(db, workspace, item_id, "undo_started", {})
            try:
                result = client.request(
                    "POST", proposal.insight.target_id, data=json.loads(row["before_json"])
                )
                status = "undone" if result.get("success") is True else "delivery_unknown"
            except AdsUnavailable:
                status = "delivery_unknown"
            with self.service.store.connect(write=True) as db:
                db.execute(
                    "UPDATE ads_reversals SET status=? WHERE workspace_id=? AND item_id=?",
                    (status, workspace, item_id),
                )
                self._log(db, workspace, item_id, "undo_" + status, {})
                if status == "undone":
                    proposal.status = "undone"
                    db.execute(
                        "UPDATE ads_proposals SET proposal_json=? WHERE workspace_id=? AND "
                        "item_id=?",
                        (proposal.model_dump_json(), workspace, item_id),
                    )
            return proposal

    def history(self, workspace, *, after=0, limit=20):
        self.service.store.workspace(workspace)
        with self.service.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM ads_action_log WHERE workspace_id=? AND id>? ORDER BY id LIMIT ?",
                (workspace, after, limit + 1),
            ).fetchall()
        return {
            "items": [
                {
                    "id": row["id"],
                    "item_id": row["item_id"],
                    "event": row["event"],
                    "details": json.loads(row["details_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows[:limit]
            ],
            "next_after": rows[limit - 1]["id"] if len(rows) > limit else None,
        }


def router():
    routes = APIRouter(
        prefix="/api/v1/workspaces/{ws}/integrations/ads",
        tags=["Meta Ads"],
        responses={status: {"model": Error} for status in (404, 409, 422, 503)},
    )

    def ads(request):
        return request.app.state.meta_ads

    @routes.get("/status", response_model=AdsStatus)
    def status(ws: WorkspaceId, request: Request):
        return ads(request).status(ws)

    @routes.put("/preferences", response_model=AdsStatus)
    def preferences(ws: WorkspaceId, body: Preference, request: Request):
        try:
            return ads(request).preferences(ws, body)
        except KeyError:
            raise HTTPException(404, "Workspace was not found.") from None

    @routes.post("/insights:sync", response_model=list[Proposal])
    def sync(ws: WorkspaceId, body: AdsSyncRequest, request: Request):
        try:
            return ads(request).sync(ws, body)
        except KeyError:
            raise HTTPException(404, "Workspace was not found.") from None
        except ActionBlocked as error:
            raise HTTPException(409, str(error)) from None
        except AdsUnavailable as error:
            raise HTTPException(503, str(error)) from None

    @routes.post("/proposals/{item_id}:execute", response_model=Proposal)
    def execute(ws: WorkspaceId, item_id: str, request: Request):
        try:
            return ads(request).execute(ws, item_id)
        except KeyError:
            raise HTTPException(404, "Proposal was not found in this workspace.") from None
        except ActionBlocked as error:
            ads(request).log(ws, item_id, "blocked", {"reason": str(error)})
            raise HTTPException(409, str(error)) from None
        except AdsUnavailable as error:
            raise HTTPException(503, str(error)) from None

    @routes.post("/proposals/{item_id}:undo", response_model=Proposal)
    def undo(ws: WorkspaceId, item_id: str, request: Request):
        try:
            return ads(request).undo(ws, item_id)
        except KeyError:
            raise HTTPException(404, "Proposal was not found in this workspace.") from None
        except ActionBlocked as error:
            raise HTTPException(409, str(error)) from None
        except AdsUnavailable as error:
            raise HTTPException(503, str(error)) from None

    @routes.get("/log")
    def log(
        ws: WorkspaceId,
        request: Request,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        try:
            return ads(request).history(ws, after=after, limit=limit)
        except KeyError:
            raise HTTPException(404, "Workspace was not found.") from None

    return routes
