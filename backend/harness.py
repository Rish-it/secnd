"""One-process scheduled workspace work with durable switches and an audit trail."""

import csv
import io
import json
import threading
import time
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from backend.meta_ads import AdsSyncRequest
from backend.meta_webhook import Message
from backend.schemas import WorkspaceId
from backend.store import encode, timestamp


class Harness:
    def __init__(self, settings, service, ads, webhook):
        self.settings, self.service, self.ads, self.webhook = settings, service, ads, webhook
        self.stop_event = threading.Event()
        self.thread = None
        with service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_preferences (
                workspace_id TEXT PRIMARY KEY, enabled BOOLEAN NOT NULL DEFAULT 0,
                last_ads REAL NOT NULL DEFAULT 0, share_counts BOOLEAN NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_control (
                id INTEGER PRIMARY KEY CHECK(id=1), killed BOOLEAN NOT NULL DEFAULT 0)""")
            db.execute("INSERT OR IGNORE INTO agent_control VALUES (1,0)")
            if settings.agent_reset_kill_on_start:
                db.execute("UPDATE agent_control SET killed=0 WHERE id=1")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL,
                event TEXT NOT NULL, item_id TEXT, created_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_messages (
                workspace_id TEXT NOT NULL, item_id TEXT NOT NULL, message_json TEXT NOT NULL,
                status TEXT NOT NULL, PRIMARY KEY(workspace_id,item_id))""")

        self.outcomes = Outcomes(self)

    def killed(self):
        with self.service.store.connect() as db:
            return bool(db.execute("SELECT killed FROM agent_control WHERE id=1").fetchone()[0])

    def status(self, ws):
        with self.service.store.connect() as db:
            row = db.execute(
                "SELECT * FROM agent_preferences WHERE workspace_id=?", (ws,)
            ).fetchone()
            activity = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM agent_activity WHERE workspace_id=? ORDER BY id DESC LIMIT 100",
                    (ws,),
                )
            ]
        return {
            "enabled": bool(row and row["enabled"]),
            "share_counts": bool(row and row["share_counts"]),
            "global_killed": self.killed(),
            "worker_enabled": self.settings.agent_enabled,
            "activity": activity,
        }

    def log(self, ws, event, item=None):
        if ws in getattr(self.service, "_deleted", set()):
            return
        with self.service.store.connect(write=True) as db:
            db.execute(
                "INSERT INTO agent_activity(workspace_id,event,item_id,created_at) VALUES "
                "(?,?,?,?)",
                (ws, event, item, timestamp()),
            )

    def configure(self, ws, enabled=None, share_counts=None):
        with self.service._lock(ws), self.service.store.connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO agent_preferences(workspace_id) VALUES (?)", (ws,))
            if enabled is not None:
                db.execute(
                    "UPDATE agent_preferences SET enabled=? WHERE workspace_id=?", (enabled, ws)
                )
            if share_counts is not None:
                db.execute(
                    "UPDATE agent_preferences SET share_counts=? WHERE workspace_id=?",
                    (share_counts, ws),
                )
        self.log(ws, "preferences_saved")
        return self.status(ws)

    def enqueue(self, ws, message):
        with self.service._lock(ws), self.service.store.connect(write=True) as db:
            previous = db.execute(
                "SELECT message_json FROM agent_messages WHERE workspace_id=? AND item_id=?",
                (ws, message.item_id),
            ).fetchone()
            if previous and previous[0] != message.model_dump_json():
                raise HTTPException(409, "A message ID was reused with different content.")
            db.execute(
                "INSERT OR IGNORE INTO agent_messages VALUES (?,?,?,'pending')",
                (ws, message.item_id, message.model_dump_json()),
            )
        self.log(ws, "message_received", message.item_id)
        return {"item_id": message.item_id, "status": "queued", "duplicate": bool(previous)}

    def tick(self):
        if self.killed():
            return
        with self.service.store.connect() as db:
            workspaces = db.execute(
                "SELECT workspace_id,last_ads FROM agent_preferences WHERE enabled=1"
            ).fetchall()
        for row in workspaces:
            ws = row["workspace_id"]
            try:
                with self.service._lock(ws):
                    if self.killed() or not self.status(ws)["enabled"]:
                        continue
                    self.log(ws, "cycle_started")
                    with self.service.store.connect() as db:
                        messages = db.execute(
                            "SELECT * FROM agent_messages WHERE workspace_id=? AND "
                            "status='pending' LIMIT 100",
                            (ws,),
                        ).fetchall()
                    for message in messages:
                        if self.killed() or not self.status(ws)["enabled"]:
                            break
                        result = self.webhook.handle(
                            ws, Message.model_validate_json(message["message_json"])
                        )
                        with self.service.store.connect(write=True) as db:
                            db.execute(
                                "UPDATE agent_messages SET status=? WHERE workspace_id=? "
                                "AND item_id=?",
                                (result["status"], ws, message["item_id"]),
                            )
                        self.log(ws, "message_" + result["status"], message["item_id"])
                    if self.killed() or not self.status(ws)["enabled"]:
                        continue
                    if time.time() - row["last_ads"] >= self.settings.agent_insights_seconds:
                        with self.service.store.connect(write=True) as db:
                            db.execute(
                                "UPDATE agent_preferences SET last_ads=? WHERE workspace_id=?",
                                (time.time(), ws),
                            )
                        if self.ads.status(ws).sandbox_allowed:
                            self.log(ws, "insights_requested")
                            proposals = self.ads.sync(ws, AdsSyncRequest())
                            for proposal in proposals:
                                self.log(ws, "decision_" + proposal.status, proposal.item_id)
                    self.collect_outcomes(ws)
                    self.log(ws, "cycle_completed")
            except Exception:
                # No provider exception text (which may carry credentials) enters the audit log.
                self.log(ws, "cycle_failed")

    def collect_outcomes(self, ws):
        if hasattr(self, "outcomes"):
            self.outcomes.collect(ws, today=date.today())

    def start(self):
        if not self.settings.agent_enabled:
            return

        def run():
            while not self.stop_event.is_set():
                try:
                    self.tick()
                except Exception:
                    # Storage can be temporarily unavailable; the next tick retries durable work.
                    pass
                self.stop_event.wait(self.settings.agent_poll_seconds)

        self.thread = threading.Thread(target=run, name="workspace-worker", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()


class Outcomes:
    def __init__(self, harness):
        self.harness = harness
        with harness.service.store.connect(write=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS ad_outcomes (
                workspace_id TEXT NOT NULL, item_id TEXT NOT NULL, before_json TEXT NOT NULL,
                after_json TEXT, due_date TEXT NOT NULL, collected_at TEXT,
                PRIMARY KEY(workspace_id,item_id))""")

    @staticmethod
    def metrics(insight):
        return {k: insight[k] for k in ("spend", "cpa", "ctr", "currency", "since", "until")}

    def track(self, ws, proposal, created_at):
        due = datetime.fromisoformat(created_at).date() + timedelta(days=7)
        before = self.metrics(proposal["insight"])
        with self.harness.service.store.connect(write=True) as db:
            db.execute(
                "INSERT OR IGNORE INTO ad_outcomes VALUES (?,?,?,NULL,?,NULL)",
                (ws, proposal["item_id"], encode(before), due.isoformat()),
            )

    def collect(self, ws, today):
        h = self.harness
        with h.service.store.connect() as db:
            proposals = db.execute(
                "SELECT * FROM ads_proposals WHERE workspace_id=?", (ws,)
            ).fetchall()
        for row in proposals:
            try:
                proposal = json.loads(row["proposal_json"])
                self.track(ws, proposal, row["created_at"])
                with h.service.store.connect() as db:
                    saved = db.execute(
                        "SELECT * FROM ad_outcomes WHERE workspace_id=? AND item_id=?",
                        (ws, row["item_id"]),
                    ).fetchone()
                due = date.fromisoformat(saved["due_date"])
                if saved["after_json"] is not None:
                    self.queue_review(
                        ws,
                        row["item_id"],
                        json.loads(saved["before_json"]),
                        json.loads(saved["after_json"]),
                        proposal,
                    )
                    continue
                if today < due or h.killed():
                    continue
                account = h.ads.account(ws)
                insight = proposal["insight"]
                if account.removeprefix("act_") != insight["account_id"]:
                    h.log(ws, "outcome_account_changed", row["item_id"])
                    continue
                window = AdsSyncRequest(
                    level=insight["level"],
                    since=due - timedelta(days=7),
                    until=due - timedelta(days=1),
                )
                client = h.ads.client_for(ws)
                rows = client.insights(account, window)
                match = next(
                    (r for r in rows if r.get(f"{window.level}_id") == insight["target_id"]), None
                )
                if match is None:
                    h.log(ws, "outcome_unavailable", row["item_id"])
                    continue
                after = h.ads._insight(
                    account, window.level, match, window, client, include_budget=False
                ).model_dump(mode="json")
                with h.service.store.connect(write=True) as db:
                    db.execute(
                        "UPDATE ad_outcomes SET after_json=?,collected_at=? WHERE "
                        "workspace_id=? AND item_id=? AND after_json IS NULL",
                        (encode(self.metrics(after)), timestamp(), ws, row["item_id"]),
                    )
                h.log(ws, "outcome_recorded", row["item_id"])
                self.queue_review(
                    ws,
                    row["item_id"],
                    json.loads(saved["before_json"]),
                    self.metrics(after),
                    proposal,
                )
            except Exception:
                h.log(ws, "outcome_fetch_failed", row["item_id"])

    def queue_review(self, ws, item_id, before, after, proposal=None):
        service = self.harness.service
        if hasattr(service, "queue_outcome_review"):
            return service.queue_outcome_review(
                ws,
                item_id,
                before,
                after,
                action=(proposal or {}).get("action"),
                status=(proposal or {}).get("status"),
            )

    def rows(self, ws):
        with self.harness.service.store.connect() as db:
            rows = db.execute(
                "SELECT o.*,p.proposal_json FROM ad_outcomes o JOIN ads_proposals p ON "
                "p.workspace_id=o.workspace_id AND p.item_id=o.item_id WHERE "
                "o.workspace_id=? ORDER BY o.due_date DESC",
                (ws,),
            ).fetchall()
        result = []
        for row in rows:
            proposal = json.loads(row["proposal_json"])
            labels = [
                r
                for r in self.harness.service.store.labels(ws)
                if r["item_id"] == row["item_id"] and r["source"] in ("answer", "spot_check")
            ]
            result.append(
                {
                    "item_id": row["item_id"],
                    "action": proposal["action"],
                    "review_label": labels[-1]["label"] if labels else None,
                    "before": json.loads(row["before_json"]),
                    "after": json.loads(row["after_json"]) if row["after_json"] else None,
                    "due_date": row["due_date"],
                    "collected_at": row["collected_at"],
                    "model_version": proposal["decision"]["model_version"],
                    "status": proposal["status"],
                }
            )
        return result

    def csv(self, ws):
        output = io.StringIO()
        columns = [
            "item_id",
            "action",
            "review_label",
            "status",
            "model_version",
            "due_date",
            "collected_at",
            "currency",
            "before_spend",
            "before_cpa",
            "before_ctr",
            "after_spend",
            "after_cpa",
            "after_ctr",
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in self.rows(ws):
            record = {key: row[key] for key in columns if key in row}
            record["currency"] = row["before"]["currency"]
            for phase in ("before", "after"):
                for metric in ("spend", "cpa", "ctr"):
                    record[f"{phase}_{metric}"] = (row[phase] or {}).get(metric)
            writer.writerow(
                {
                    k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                    for k, v in record.items()
                }
            )
        return output.getvalue()


class AgentPreference(BaseModel):
    enabled: bool
    share_counts: bool = False


def router():
    routes = APIRouter(tags=["Activity"])

    @routes.get("/api/v1/workspaces/{ws}/agent")
    def status(ws: WorkspaceId, request: Request):
        return request.app.state.harness.status(ws)

    @routes.put("/api/v1/workspaces/{ws}/agent")
    def configure(ws: WorkspaceId, body: AgentPreference, request: Request):
        return request.app.state.harness.configure(
            ws, enabled=body.enabled, share_counts=body.share_counts
        )

    @routes.get("/api/v1/workspaces/{ws}/outcomes")
    def outcomes(ws: WorkspaceId, request: Request):
        return request.app.state.harness.outcomes.rows(ws)

    @routes.get("/api/v1/workspaces/{ws}/outcomes.csv")
    def export(ws: WorkspaceId, request: Request):
        return Response(
            request.app.state.harness.outcomes.csv(ws),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="secnd-outcomes.csv"'},
        )

    @routes.post("/api/v1/agent/kill")
    def kill(request: Request):
        user = request.app.state.auth.principal(request)
        if user == "demo" and not request.app.state.harness.settings.demo_mode:
            raise HTTPException(401, "Sign in to stop automatic actions.")
        with request.app.state.workspaces.store.connect(write=True) as db:
            db.execute("UPDATE agent_control SET killed=1 WHERE id=1")
        return {"global_killed": True}

    return routes
