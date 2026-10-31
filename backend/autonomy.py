"""Workspace evidence for earned autonomy, independent of public model certification."""

import random
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from backend import metrics
from backend.loop import SPOT_CHECK_RATE
from backend.schemas import WorkspaceId
from backend.store import timestamp

KINDS = ("ad", "inbox", "account")
WINDOW = 1000


def kind(source):
    return (
        "ad"
        if source == "ads"
        else "inbox"
        if source in ("whatsapp", "messenger", "instagram")
        else "account"
    )


def lower(confirmed, reviewed):
    return float(beta.ppf(0.05, confirmed, reviewed - confirmed + 1)) if confirmed else 0.0


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["watch", "ask", "act"] = "watch"
    bar: Literal[90, 95, 98, 99] = 99
    confidence_cut: float = Field(default=0.9, ge=0, le=1)
    allow_raise_budget: bool = False
    allow_customer_messages: bool = False


class Autonomy:
    def __init__(self, service):
        self.service = service
        self._checks = {}
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS autonomy_policies (
                workspace_id TEXT NOT NULL, kind TEXT NOT NULL, policy_json TEXT NOT NULL,
                PRIMARY KEY(workspace_id,kind))""")
            db.execute("""CREATE TABLE IF NOT EXISTS autonomy_reviews (
                workspace_id TEXT NOT NULL, answer_key TEXT NOT NULL, item_id TEXT NOT NULL,
                kind TEXT NOT NULL, confirmed BOOLEAN NOT NULL, confidence REAL NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(workspace_id,answer_key))""")

            columns = {r[1] for r in db.execute("PRAGMA table_info(autonomy_reviews)")}
            for column in ("person_id", "model_id", "model_revision"):
                if column not in columns:
                    db.execute(
                        f"ALTER TABLE autonomy_reviews ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                    )

    def owner(self, ws):
        with self.service.store.connect() as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='workspace_owners'"
            ).fetchone():
                return ""
            row = db.execute(
                "SELECT user_id FROM workspace_owners WHERE workspace_id=?", (ws,)
            ).fetchone()
        return row[0] if row else ""

    def identity(self, ws, decision_type, previous=None):
        if previous and previous.get("model_id"):
            return previous["model_id"], previous.get("model_revision", "")
        if previous and previous.get("reader"):
            return previous["reader"]["model_id"], previous["reader"]["revision"]
        if previous and previous.get("model_version"):
            return "workspace-head", previous["model_version"]
        if hasattr(self.service, "decision_identity"):
            return self.service.decision_identity(ws, decision_type)
        return "", ""

    def observe(self, ws, item, label, previous):
        if previous and previous.get("spot_check"):
            identity = self.identity(ws, kind(previous.get("source_app")), previous)
            state = self._checks.setdefault((ws, *identity), {})
            state.setdefault(item, previous["label"] == label)

    def guarded(self, ws, identity):
        checked = list(self._checks.get((ws, *identity), {}).values())
        return metrics.clopper_pearson_lower(len(checked) - sum(checked), len(checked)) > 0.01

    def sample(self, ws, decision):
        return random.random() < SPOT_CHECK_RATE

    def active(self, ws):
        oauth = getattr(self.service, "meta_oauth", None)
        if oauth and oauth.owned(ws):
            return True
        with self.service.store.connect() as db:
            return (
                db.execute("SELECT 1 FROM autonomy_policies WHERE workspace_id=?", (ws,)).fetchone()
                is not None
            )

    def policy(self, ws, decision_type):
        if decision_type not in KINDS:
            raise HTTPException(422, "Choose ad, inbox or account decisions.")
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT policy_json FROM autonomy_policies WHERE workspace_id=? AND kind=?",
                (ws, decision_type),
            ).fetchone()
        return Policy.model_validate_json(row[0]) if row else Policy()

    def configure(self, ws, decision_type, body):
        self.policy(ws, decision_type)
        with self.service._lock(ws), self.service.store.connect(write=True) as db:
            db.execute(
                "INSERT OR REPLACE INTO autonomy_policies VALUES (?,?,?)",
                (ws, decision_type, body.model_dump_json()),
            )
        if getattr(self.service, "harness", None):
            self.service.harness.log(ws, "autonomy_preferences_saved")
        return self.status(ws, decision_type)

    def review(self, ws, item, label, key, previous):
        if not previous or item.startswith("outcome-"):
            return
        with self.service.store.connect(write=True) as db:
            db.execute(
                "INSERT OR IGNORE INTO autonomy_reviews VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    ws,
                    key,
                    item,
                    kind(previous.get("source_app")),
                    previous["label"] == label,
                    previous["prob"],
                    timestamp(),
                    self.owner(ws),
                    *self.identity(ws, kind(previous.get("source_app")), previous),
                ),
            )
        self.observe(ws, item, label, previous)

    def reviews(self, ws, decision_type, cut):
        with self.service.store.connect() as db:
            rows = db.execute(
                """SELECT r.* FROM autonomy_reviews r
                WHERE r.workspace_id=? AND r.kind=? AND NOT EXISTS (
                    SELECT 1 FROM answer_undos u WHERE u.workspace_id=r.workspace_id
                    AND u.idempotency_key=r.answer_key)
                ORDER BY r.created_at DESC, r.answer_key DESC""",
                (ws, decision_type),
            ).fetchall()
        owner = self.owner(ws)
        identity = self.identity(ws, decision_type)
        rows = [
            r
            for r in rows
            if not owner
            or (r["person_id"] == owner and (r["model_id"], r["model_revision"]) == identity)
        ]
        latest = {}
        for row in rows:
            latest.setdefault(row["item_id"], row)
        return [bool(r["confirmed"]) for r in latest.values() if r["confidence"] >= cut][:WINDOW]

    def status(self, ws, decision_type):
        policy = self.policy(ws, decision_type)
        recent = self.reviews(ws, decision_type, policy.confidence_cut)
        n, k = len(recent), sum(recent)
        bound = lower(k, n)
        unlocked = bound >= policy.bar / 100

        # Appending confirmations eventually ages older reviews out of the fixed window.
        def future(extra):
            retained = recent[: max(0, WINDOW - extra)]
            return lower(sum(retained) + extra, len(retained) + extra)

        lo, hi = 0, WINDOW
        while lo < hi:
            mid = (lo + hi) // 2
            if future(mid) >= policy.bar / 100:
                hi = mid
            else:
                lo = mid + 1
        return {
            **policy.model_dump(),
            "decision_type": decision_type,
            "model_id": self.identity(ws, decision_type)[0],
            "model_revision": self.identity(ws, decision_type)[1],
            "reviewed": n,
            "confirmed": k,
            "corrected": n - k,
            "lower_bound": bound,
            "unlocked": unlocked,
            "confirmations_left": lo,
            "window": WINDOW,
            "effective_level": "ask" if policy.level == "act" and not unlocked else policy.level,
        }

    def allowed(self, ws, decision_type, action=None):
        status = self.status(ws, decision_type)
        if status["effective_level"] != "act":
            return False
        if action in ("raise budget", "send message"):
            flag = "allow_raise_budget" if action == "raise budget" else "allow_customer_messages"
            return status["bar"] == 99 and status[flag]
        return True

    def authorizes(self, ws, decision):
        decision = decision if isinstance(decision, dict) else decision.model_dump()
        decision_type = kind(decision.get("source_app"))
        action = "send message" if decision_type == "inbox" else decision["label"]
        status = self.status(ws, decision_type)
        identity = self.identity(ws, decision_type)
        return bool(
            self.owner(ws)
            and decision.get("automatic_basis") == "workspace_bound"
            and decision.get("auto")
            and not decision.get("spot_check")
            and not decision.get("inference_error")
            and (decision.get("model_id"), decision.get("model_revision")) == identity
            and decision["prob"] >= status["confidence_cut"]
            and self.allowed(ws, decision_type, action)
            and not self.guarded(ws, identity)
            and not self.service.automatic_guarded(ws)
        )

    def apply(self, ws, decisions):
        for decision in decisions:
            if decision.get("auto"):
                decision["automatic_basis"] = "model_certified"
                decision["automatic_evidence"] = {
                    "prob": decision["prob"],
                    "model_id": decision.get("model_id"),
                    "model_revision": decision.get("model_revision"),
                    "threshold": decision.get("threshold"),
                    "cp_lower_bound": decision.get("cp_lower_bound"),
                    "reader": decision.get("reader"),
                }
        if not self.active(ws):
            return decisions
        for decision in decisions:
            decision_type = kind(decision.get("source_app"))
            status = self.status(ws, decision_type)
            decision["autonomy"] = {
                key: status[key]
                for key in (
                    "decision_type",
                    "reviewed",
                    "confirmed",
                    "lower_bound",
                    "bar",
                    "effective_level",
                    "confidence_cut",
                )
            }
            action = "send message" if decision_type == "inbox" else decision["label"]
            if decision["prob"] < status["confidence_cut"] or not self.allowed(
                ws, decision_type, action
            ):
                decision["auto"] = False
            identity = self.identity(ws, decision_type, decision)
            guarded = self.guarded(ws, identity) or (
                hasattr(self.service, "automatic_guarded") and self.service.automatic_guarded(ws)
            )
            if (
                self.owner(ws)
                and decision["prob"] >= status["confidence_cut"]
                and self.allowed(ws, decision_type, action)
                and not guarded
                and identity == self.identity(ws, decision_type)
                and not decision.get("inference_error")
                and not decision["item_id"].startswith("outcome-")
            ):
                if not decision.get("auto") and not decision.get("spot_check"):
                    decision["spot_check"] = self.sample(ws, decision)
                decision["auto"] = not decision.get("spot_check", False)
                decision["threshold"] = status["confidence_cut"]
                decision["cp_lower_bound"] = None
                decision["automatic_basis"] = "workspace_bound"
                decision["automatic_evidence"] = dict(
                    decision["autonomy"], model_id=identity[0], model_revision=identity[1]
                )
            if guarded or decision.get("spot_check"):
                decision["auto"] = False
            if not decision["auto"]:
                decision["automatic_basis"] = None
                decision["automatic_evidence"] = None
        return decisions

    def aggregate(self):
        # SQL emits counts only. No text, names or identifiers enter the aggregate result.
        result = []
        with self.service.store.connect() as db:
            for decision_type in KINDS:
                proposals = db.execute(
                    """SELECT COUNT(*), COUNT(DISTINCT workspace_id) FROM (
                    SELECT DISTINCT p.workspace_id,p.item_id FROM predictions p
                    JOIN agent_preferences a ON a.workspace_id=p.workspace_id AND a.share_counts=1
                    WHERE CASE json_extract(p.decision_json,'$.source_app')
                        WHEN 'ads' THEN 'ad' WHEN 'whatsapp' THEN 'inbox'
                        WHEN 'messenger' THEN 'inbox' WHEN 'instagram' THEN 'inbox'
                        ELSE 'account' END = ?)""",
                    (decision_type,),
                ).fetchone()
                reviews = db.execute(
                    """SELECT COALESCE(SUM(confirmed),0),COUNT(*) FROM (
                    SELECT r.confirmed, ROW_NUMBER() OVER (
                        PARTITION BY r.workspace_id,r.item_id
                        ORDER BY r.created_at DESC,r.answer_key DESC) AS rank
                    FROM autonomy_reviews r JOIN agent_preferences a
                    ON a.workspace_id=r.workspace_id AND a.share_counts=1
                    WHERE r.kind=? AND NOT EXISTS (SELECT 1 FROM answer_undos u
                    WHERE u.workspace_id=r.workspace_id AND u.idempotency_key=r.answer_key)
                    ) WHERE rank=1""",
                    (decision_type,),
                ).fetchone()
                result.append(
                    {
                        "decision_type": decision_type,
                        "proposals": proposals[0],
                        "confirmed": reviews[0],
                        "corrected": reviews[1] - reviews[0],
                        "workspaces": proposals[1],
                    }
                )
        return {"context_only": True, "counts": result}


def router():
    routes = APIRouter(tags=["Autonomy"])

    @routes.get("/api/v1/workspaces/{ws}/autonomy")
    def status(ws: WorkspaceId, request: Request):
        return [request.app.state.workspaces.autonomy.status(ws, k) for k in KINDS]

    @routes.put("/api/v1/workspaces/{ws}/autonomy/{decision_type}")
    def configure(ws: WorkspaceId, decision_type: str, body: Policy, request: Request):
        return request.app.state.workspaces.autonomy.configure(ws, decision_type, body)

    @routes.get("/api/v1/track-record")
    def aggregate(request: Request):
        request.app.state.auth.principal(request)
        return request.app.state.workspaces.autonomy.aggregate()

    return routes
