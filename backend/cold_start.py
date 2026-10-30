"""Budgeted invented history; teacher calls are confined to explicit onboarding jobs."""

import csv
import fcntl
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from threading import RLock

import pandas as pd

from backend.muse import BudgetRun, MuseClient, MuseError, Usage
from backend.providers import ledger_path, provider_settings
from backend.reader_serving import CPUReader, history_version
from backend.schemas import EvalCard

CAP = Decimal("1")


class ColdStartConflict(ValueError):
    pass


def label_hash(labels):
    return hashlib.sha256(json.dumps(sorted(labels), ensure_ascii=False).encode()).hexdigest()


def imported_labels(content):
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    except (UnicodeError, csv.Error):
        raise ValueError("Choose a UTF-8 label list with one label per row.") from None
    rows = [r for r in rows if r and any(x.strip() for x in r)]
    if any(len(r) != 1 for r in rows):
        raise ValueError("The label list needs one column.")
    labels = [r[0].strip() for r in rows]
    if labels and labels[0].lower() in ("label", "labels", "category"):
        labels = labels[1:]
    return labels


class ColdStartService:
    def __init__(self, store, settings, *, reader_factory=CPUReader, transport=None, demo=False):
        self.store, self.settings = store, settings
        self.reader_factory, self.transport, self.demo = reader_factory, transport, demo
        self._reader, self._guard = None, RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="invented-history")
        self._pending = {}
        self.cache_dir = settings.data_dir / "cache" / "invented-history"
        with store.connect(write=True) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS cold_starts (
                workspace_id TEXT PRIMARY KEY REFERENCES workspaces(workspace_id),
                label_hash TEXT NOT NULL, status TEXT NOT NULL, error TEXT)"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS invented_history (
                workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                id TEXT NOT NULL, label TEXT NOT NULL, text TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(workspace_id, id))"""
            )

            db.execute(
                "UPDATE cold_starts SET status='stopped', "
                "error='Server restarted; resume to use saved progress.' "
                "WHERE status IN ('queued','generating')"
            )

    def owns(self, workspace):
        with self.store.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM cold_starts WHERE workspace_id=?", (workspace,)
                ).fetchone()
                is not None
            )

    def reader(self):
        with self._guard:
            if self._reader is None:
                self._reader = self.reader_factory(self.settings.data_dir)
            return self._reader

    @staticmethod
    def task(workspace):
        return "cold-start-" + hashlib.sha256(workspace.encode()).hexdigest()[:24]

    @property
    def ledger(self):
        return ledger_path(Path("docs/results/s5"), "meta-wonder")

    def spent(self, workspace=None, ledger=None):
        """One workspace's invented-history spend, or every workspace's when `workspace` is None."""
        path = ledger or self.ledger
        if not path.exists():
            return Decimal(0)
        mine = self.task(workspace) if workspace else None
        with path.open(newline="") as stream:
            return sum(
                (
                    Decimal(row["cost_usd"])
                    for row in csv.DictReader(stream)
                    if row["task"] == mine
                    or (mine is None and row["task"].startswith("cold-start-"))
                ),
                Decimal(0),
            )

    def start(self, workspace, labels):
        labels = sorted(labels)
        key = label_hash(labels)
        # Fail before any paid call when the decision model is unavailable.
        self.reader().validate_labels(labels)
        with self._guard:
            try:
                existing = self.store.workspace(workspace)
            except KeyError:
                existing = None
            if existing and (not self.owns(workspace) or existing["labels"] != labels):
                raise ColdStartConflict("Choose a new workspace for a different label list.")
            if not existing:
                self.store.create_workspace(
                    workspace, {"labels": labels, "calib": [], "mode": "reader", "seed": 0}
                )
                with self.store.connect(write=True) as db:
                    db.execute(
                        "INSERT INTO cold_starts VALUES (?, ?, 'queued', NULL)", (workspace, key)
                    )
            future = self._pending.get(workspace)
            if future is None or future.done():
                self._status(workspace, "queued")
                self._pending[workspace] = self._executor.submit(self._generate, workspace, labels)
        return self.state(workspace)

    @contextmanager
    def cache(self, labels):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (label_hash(labels) + ".json")
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(path.read_text()) if path.exists() else {"labels": labels, "rows": {}}
            )
            try:
                yield state, lambda: self._save(path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _save(path, state):
        staged = path.with_suffix(".tmp")
        staged.write_text(json.dumps(state, allow_nan=False) + "\n")
        staged.replace(path)

    def _status(self, workspace, status, error=None):
        with self.store.connect(write=True) as db:
            db.execute(
                "UPDATE cold_starts SET status=?, error=? WHERE workspace_id=?",
                (status, error, workspace),
            )

    def _copy(self, workspace, label, messages):
        with self.store.connect(write=True) as db:
            for index, text in enumerate(messages):
                key = hashlib.sha256(f"{label}\0{index}".encode()).hexdigest()[:24]
                # Existing edits and deletion tombstones always win over the shared cache.
                db.execute(
                    "INSERT OR IGNORE INTO invented_history VALUES (?, ?, ?, ?, 0)",
                    (workspace, key, label, text),
                )

    def _generate(self, workspace, labels):
        self._status(workspace, "generating")
        try:
            with self.cache(labels) as (cache, save):
                for label in labels:
                    cached = cache["rows"].get(label)
                    if cached:
                        if cached["status"] == "complete":
                            self._copy(workspace, label, cached["messages"])
                            continue
                        self._status(
                            workspace,
                            "stopped",
                            "An earlier request needs review; it will not be paid again.",
                        )
                        return
                    if not self.settings.cold_start_paid:
                        self._status(
                            workspace,
                            "stopped",
                            "Writing new examples is turned off on this server; "
                            "label lists it has already seen still work.",
                        )
                        return
                    settings = provider_settings(self.settings, "meta-wonder")
                    settings = settings.model_copy(update={"muse_budget_usd": CAP})
                    # One ledger entry per label; BudgetRun serializes the shared Meta ledger.
                    with BudgetRun(
                        settings,
                        task=self.task(workspace),
                        rows=1,
                        measured_tokens=Usage(prompt_tokens=0, completion_tokens=0),
                        ledger_path=self.ledger,
                        provider="meta-wonder",
                    ) as run:
                        remaining = min(
                            min(CAP, self.settings.meta_budget_usd) - self.spent(workspace),
                            self.settings.cold_start_total_usd - self.spent(),
                        )
                        if remaining <= 0:
                            raise MuseError("The example budget is exhausted.")
                        run.budget = remaining
                        with MuseClient(settings, run=run, transport=self.transport) as client:
                            # Reserve a durable attempt marker before network I/O. An interrupted
                            # or failed label is never paid for again automatically.
                            cache["rows"][label] = {"status": "attempted"}
                            save()
                            try:
                                result = client.examples(label, labels, n=10)
                            except MuseError:
                                if run.cost_usd == 0:
                                    # A budget refusal before transport can be resumed safely.
                                    del cache["rows"][label]
                                    save()
                                raise
                            cache["rows"][label] = {
                                "status": "complete",
                                "messages": result.messages,
                            }
                            save()
                            self._copy(workspace, label, result.messages)
            self._status(workspace, "ready")
        except MuseError:
            self._status(
                workspace,
                "stopped",
                "Generation stopped. Cached history is kept; "
                "the $1 cap and provider limits still apply.",
            )
        except Exception:
            self._status(
                workspace,
                "stopped",
                "Generation was interrupted. Saved history is kept; "
                "attempted requests will not be paid again.",
            )

    def state(self, workspace):
        with self.store.connect() as db:
            row = db.execute(
                "SELECT * FROM cold_starts WHERE workspace_id=?", (workspace,)
            ).fetchone()
            history = db.execute(
                "SELECT id, label, text FROM invented_history "
                "WHERE workspace_id=? AND deleted=0 ORDER BY label,id",
                (workspace,),
            ).fetchall()
        if row is None:
            raise KeyError(workspace)
        return dict(
            workspace_id=workspace,
            labels=self.store.workspace(workspace)["labels"],
            status=row["status"],
            error=row["error"],
            history=[dict(h) for h in history],
            spent_usd=str(self.spent(workspace)),
            cap_usd=str(CAP),
            demo=self.demo,
        )

    def edit(self, workspace, item_id, text=None, label=None, *, delete=False):
        config = self.store.workspace(workspace)
        if label is not None and label not in config["labels"]:
            raise ValueError("Choose one of this workspace's labels.")
        with self.store.connect(write=True) as db:
            old = db.execute(
                "SELECT * FROM invented_history WHERE workspace_id=? AND id=? AND deleted=0",
                (workspace, item_id),
            ).fetchone()
            if old is None:
                raise KeyError(item_id)
            db.execute(
                "UPDATE invented_history SET text=?, label=?, deleted=? "
                "WHERE workspace_id=? AND id=?",
                (
                    text if text is not None else old["text"],
                    label or old["label"],
                    int(delete),
                    workspace,
                    item_id,
                ),
            )
        return self.state(workspace)

    def history(self, workspace):
        state = self.state(workspace)
        rows = {r["id"]: r for r in state["history"]}
        for r in self.store.labels(workspace):
            rows[r["item_id"]] = {"id": r["item_id"], "text": r["text"], "label": r["label"]}
        return list(rows.values())

    def card(self, workspace):
        labels = self.store.workspace(workspace)["labels"]
        return EvalCard(model_version=history_version(labels, self.history(workspace)), coverage=0)

    def route(self, workspace, frame: pd.DataFrame):
        rows = self.reader().predict(
            self.store.workspace(workspace)["labels"], self.history(workspace), frame
        )
        for row, item in zip(rows, frame.to_dict("records"), strict=True):
            row["source_app"] = item.get("source_app")
        self.store.record_predictions(workspace, json.loads(frame.to_json(orient="records")), rows)
        return rows

    def answer(self, workspace, item_id, label, key):
        self.store.item(workspace, item_id)
        if label not in self.store.workspace(workspace)["labels"]:
            raise ValueError("Choose one of this workspace's labels.")
        result, _ = self.store.answer(
            workspace,
            item_id,
            label,
            key,
            dict(
                item_id=item_id,
                coverage=0,
                model_version=self.card(workspace).model_version,
                retrain="history_updated",
            ),
        )
        return result

    def close(self):
        self._executor.shutdown(wait=True)
