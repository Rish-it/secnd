"""Workspace evidence for earned autonomy, additional to model certification."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

from backend.schemas import WorkspaceId
from backend.store import encode, timestamp

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
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS autonomy_policies (
                workspace_id TEXT NOT NULL, kind TEXT NOT NULL, policy_json TEXT NOT NULL,
                PRIMARY KEY(workspace_id,kind))""")
            db.execute("""CREATE TABLE IF NOT EXISTS autonomy_reviews (
                workspace_id TEXT NOT NULL, answer_key TEXT NOT NULL, item_id TEXT NOT NULL,
                kind TEXT NOT NULL, confirmed BOOLEAN NOT NULL, confidence REAL NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(workspace_id,answer_key))""")

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
                "INSERT OR IGNORE INTO autonomy_reviews VALUES (?,?,?,?,?,?,?)",
                (
                    ws,
                    key,
                    item,
                    kind(previous.get("source_app")),
                    previous["label"] == label,
                    previous["prob"],
                    timestamp(),
                ),
            )

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

    def apply(self, ws, decisions):
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
            with self.service.store.connect(write=True) as db:
                db.execute(
                    "UPDATE predictions SET auto=?,decision_json=? WHERE workspace_id=? "
                    "AND id=(SELECT id FROM predictions WHERE workspace_id=? AND item_id=? "
                    "ORDER BY created_at DESC,id DESC LIMIT 1)",
                    (decision["auto"], encode(decision), ws, ws, decision["item_id"]),
                )
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
