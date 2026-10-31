"""Persistent workspace loops and a background answer dispatcher."""

import hashlib
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter
from sklearn.metrics.pairwise import cosine_similarity

from backend.autonomy import Autonomy, kind
from backend.config import Settings
from backend.encoders import fetch
from backend.features import featurize
from backend.importing import holdout, labelled_rows
from backend.registry import Registry
from backend.schemas import Decision, EvalCard, WorkspaceId
from backend.store import Store

logger = logging.getLogger(__name__)


def loop_factory(*args, **kwargs):
    # The core stays independent of persistence and HTTP.
    from backend.loop import Loop

    return Loop(*args, **kwargs)


class WorkspaceExists(ValueError):
    """Imports must not replace a workspace's fixed holdout or saved answers."""


class WorkspaceService:
    def __init__(self, store: Store, data_dir: Path, factory=loop_factory):
        self.store, self.data_dir, self.factory = store, data_dir, factory
        self._guard = threading.RLock()
        self._locks, self._loops, self._positions = {}, {}, {}
        self._pending, self._errors = set(), {}
        self.cold_start = None
        self.readers = None
        self.autonomy = Autonomy(self)
        # carmy: one process; use a durable worker queue before running multiple API processes.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="workspace-answer")

    def create(
        self, workspace, labels, calib: pd.DataFrame, *, featurizer=None, seed=0, training=None
    ):
        TypeAdapter(WorkspaceId).validate_python(workspace)
        if len(set(labels)) < 2 or len(labels) != len(set(labels)):
            raise ValueError("A workspace needs at least two distinct labels.")
        if calib.empty or not set(calib.label).issubset(labels):
            raise ValueError("Calibration rows must use the workspace labels.")
        if training is not None:
            if set(training.id.astype(str)) & set(calib.id.astype(str)) or not set(
                training.label
            ).issubset(labels):
                raise ValueError("Training labels must be disjoint from calibration.")
        calib = calib.assign(amount=calib.get("amount"), date=calib.get("date"))
        featurizer = featurizer or Settings().featurizer
        if featurizer == "pt-minilm":
            try:
                fetch(featurizer, self.data_dir)
            except Exception as error:
                logger.warning(
                    "Pinned pt-minilm fetch/verification failed (%s); falling back to minilm.",
                    type(error).__name__,
                )
                featurizer = "minilm"
        # pandas normalises timestamps and missing numeric values to portable JSON values.
        self.store.create_workspace(
            workspace,
            {
                "labels": labels,
                "calib": json.loads(calib.to_json(orient="records", date_format="iso")),
                "featurizer": featurizer,
                "seed": seed,
            },
            items=[]
            if training is None
            else json.loads(training.to_json(orient="records", date_format="iso")),
            labels=[] if training is None else list(zip(training.id, training.label, strict=True)),
        )

    def import_decisions(self, workspace, content, target_column):
        frame = labelled_rows(content, target_column)
        train, calib = holdout(frame)
        with self._lock(workspace):
            try:
                self.create(workspace, sorted(frame.label.unique().tolist()), calib, training=train)
            except sqlite3.IntegrityError:
                raise WorkspaceExists(
                    "This workspace already exists. Choose a new workspace."
                ) from None
            self._runtime(workspace)
            return {
                "workspace_id": workspace,
                "row_count": len(frame),
                "target_column": target_column,
                "card": self.card(workspace),
            }

    def has_model(self, workspace):
        if self.cold_start and self.cold_start.owns(workspace):
            return True
        if self.readers and self.readers.enabled(workspace):
            self.store.workspace(workspace)
            return True
        return Registry(self.data_dir, workspace).current() is not None

    def card(self, workspace):
        if self.cold_start and self.cold_start.owns(workspace):
            return self.cold_start.card(workspace)
        if self.readers and self.readers.enabled(workspace):
            return self.readers.card(workspace)
        registry = Registry(self.data_dir, workspace)
        current = registry.current()
        return (
            EvalCard.model_validate(registry.card(current))
            if current
            else EvalCard(model_version="untrained", coverage=0)
        )

    def page(self, workspace, kind, cursor=None, limit=20):
        return self.store.decision_page(workspace, kind, cursor, limit)

    def metrics(self, workspace):
        with self._lock(workspace):
            if self.cold_start and self.cold_start.owns(workspace):
                return dict(
                    card=self.card(workspace),
                    history=[],
                    guarded=False,
                    labels=len(self.cold_start.history(workspace)),
                )
            if self.readers and self.readers.enabled(workspace):
                return dict(
                    card=self.card(workspace),
                    history=[],
                    guarded=self.readers.guarded(workspace),
                    labels=len(self.readers.history(workspace)),
                )
            registry = Registry(self.data_dir, workspace)
            history = []
            for version in registry.versions():
                card = EvalCard.model_validate(registry.card(version))
                saved = (registry.root / version / "card.json").stat().st_mtime
                history.append(
                    {
                        "timestamp": datetime.fromtimestamp(saved, UTC),
                        "model_version": version,
                        "coverage": card.coverage,
                        "precision": card.precision,
                    }
                )
            return {
                "card": self.card(workspace),
                "history": history,
                "guarded": bool(getattr(self._loops.get(workspace), "guarded", False)),
                "labels": len(self.store.labels(workspace)),
            }

    def _lock(self, workspace):
        TypeAdapter(WorkspaceId).validate_python(workspace)
        with self._guard:
            return self._locks.setdefault(workspace, threading.RLock())

    def _validate_training(self, workspace, frame, labels):
        config = self.store.workspace(workspace)
        if not set(labels).issubset(config["labels"]):
            raise ValueError("An answer must use a workspace label.")
        heldout = {str(row["id"]) for row in config["calib"]}
        if set(frame.id.astype(str)) & heldout:
            raise ValueError("Calibration items cannot become training labels.")
        return config

    def _sync_models(self, workspace):
        registry = Registry(self.data_dir, workspace)
        current = registry.current()
        self.store.set_models(
            workspace,
            [
                {
                    "version": version,
                    "card": registry.card(version),
                    "is_current": version == current,
                }
                for version in registry.versions()
            ],
        )

    def _restore_current(self, workspace, loop):
        registry = Registry(self.data_dir, workspace)
        version = registry.current()
        if version and (loop.current is None or loop.current["version"] != version):
            candidate = registry.load(version)
            loop.current = {**candidate, "version": version}
            loop.c = candidate["model"].C
            loop.pending, loop.checks = {}, [0, 0]
            self._sync_models(workspace)

    def _runtime(self, workspace, *, before=None):
        if workspace not in self._loops:
            config = self.store.workspace(workspace)
            loop = self.factory(
                config["labels"],
                pd.DataFrame(config["calib"]),
                featurizer=config["featurizer"],
                seed=config["seed"],
                cache_dir=self.data_dir / "workspaces" / workspace / "cache",
                models=Registry(self.data_dir, workspace),
            )
            history = [
                r for r in self.store.labels(workspace) if before is None or r["position"] < before
            ]
            for row in history:
                frame = pd.DataFrame([{**row, "id": row["item_id"]}])
                self._validate_training(workspace, frame, [row["label"]])
                loop.learn(frame, [row["label"]], row["source"])
            self._restore_current(workspace, loop)
            if len({r["label"] for r in history}) >= 2:
                loop.retrain()
                self._sync_models(workspace)
            with self._guard:
                self._errors.pop(workspace, None)
            self._loops[workspace] = loop
            self._positions[workspace] = max((r["position"] for r in history), default=0)
        loop = self._loops[workspace]
        self._restore_current(workspace, loop)
        return loop

    def route(self, workspace, frame: pd.DataFrame) -> list[dict]:
        with self._lock(workspace):
            decisions = self._route(workspace, frame)
            sources = {str(r["id"]): r.get("source_app") for r in frame.to_dict("records")}
            for row in decisions:
                row["source_app"] = sources.get(row["item_id"])
            return self.autonomy.apply(workspace, decisions)

    def _route(self, workspace, frame: pd.DataFrame) -> list[dict]:
        with self._lock(workspace):
            if self.cold_start and self.cold_start.owns(workspace):
                return self.cold_start.route(workspace, frame)
            if self.readers and self.readers.enabled(workspace):
                return self.readers.route(workspace, frame)
            loop = self._runtime(workspace)
            card = self.card(workspace)
            decisions = [
                Decision.model_validate(row).model_dump(mode="json") for row in loop.route(frame)
            ]
            # Read-only evidence over training history; calibration rows never enter this list.
            history = {r["item_id"]: r for r in self.store.labels(workspace)}
            rows = list(history.values())
            similarities = None
            if rows and hasattr(loop, "featurizer"):
                past = pd.DataFrame(rows).assign(amount=None, date=None)
                query = frame.assign(amount=None, date=None)
                similarities = cosine_similarity(
                    featurize(query, loop.featurizer, loop.cache_dir),
                    featurize(past, loop.featurizer, loop.cache_dir),
                )
            for index, (decision, item) in enumerate(
                zip(decisions, frame.to_dict("records"), strict=True)
            ):
                decision["source_app"] = item.get("source_app")
                decision["cp_lower_bound"] = card.cp_lower_bound
                if similarities is not None:
                    nearest = sorted(range(len(rows)), key=lambda i: -similarities[index, i])
                    decision["reasons"]["neighbours"] = [
                        dict(
                            item_id=rows[i]["item_id"],
                            text=rows[i]["text"],
                            label=rows[i]["label"],
                            similarity=float(max(0, min(1, similarities[index, i]))),
                        )
                        for i in nearest
                        if rows[i]["item_id"] != decision["item_id"]
                    ][:3]
            self.store.record_predictions(
                workspace, json.loads(frame.to_json(orient="records", date_format="iso")), decisions
            )
            return decisions

    def queue_outcome_review(self, workspace, item_id, before, after, *, action=None, status=None):
        review_id = "outcome-" + hashlib.sha256(item_id.encode()).hexdigest()[:40]
        with self._lock(workspace):
            if self.store.latest_prediction(workspace, review_id):
                return review_id
            text = (
                "Review this observed ad result before adding it to history. "
                f"Proposal: {action or 'unavailable'}; execution: {status or 'unavailable'}. "
                f"Before: {json.dumps(before)}. Seven days after: {json.dumps(after)}. "
                "Which action would you choose in light of this result? "
                "These observations alone do not establish causation."
            )
            self.route(
                workspace,
                pd.DataFrame(
                    [
                        dict(
                            id=review_id,
                            text=text,
                            amount=None,
                            date=None,
                            merchant_key="",
                            source_app="ads",
                        )
                    ]
                ),
            )
            # Outcomes are observations, not automatically inferred action labels.
            with self.store.connect(write=True) as db:
                row = self.store.latest_prediction(workspace, review_id)
                row["auto"], row["spot_check"] = False, False
                db.execute(
                    "UPDATE predictions SET auto=0,spot_check=0,decision_json=? WHERE "
                    "workspace_id=? AND item_id=?",
                    (json.dumps(row), workspace, review_id),
                )
            return review_id

    def undo_answer(self, workspace, key):
        with self._lock(workspace):
            row, changed = self.store.undo_answer(workspace, key)
            cold = self.cold_start and self.cold_start.owns(workspace)
            if changed and not cold:
                old_loop = self._loops.get(workspace)
                checks = list(getattr(old_loop, "checks", [0, 0]))
                pending = dict(getattr(old_loop, "pending", {}))
                # Restore the pre-answer artifact before rebuilding. A refused candidate must
                # never leave a classifier trained on the retracted answer serving requests.
                previous = json.loads(row["result_json"])["model_version"]
                registry = Registry(self.data_dir, workspace)
                if previous in registry.versions():
                    registry.promote(previous)
                else:
                    (registry.root / "current").unlink(missing_ok=True)
                self._loops.pop(workspace, None)
                self._positions.pop(workspace, None)
                rebuilt = self._runtime(workspace)
                # Retraction changes history, not the evidence that triggered the live guard.
                rebuilt.checks, rebuilt.pending = checks, pending
            current = self.card(workspace)
            return dict(
                item_id=row["item_id"],
                coverage=current.coverage,
                model_version=current.model_version,
                retrain="history_updated" if cold else "queued",
            )

    def answer(self, workspace, item_id, label, idempotency_key, *, source=None) -> dict:
        if self.cold_start and self.cold_start.owns(workspace):
            with self._lock(workspace):
                previous = self.store.latest_prediction(workspace, item_id)
                result = self.cold_start.answer(workspace, item_id, label, idempotency_key)
                self.autonomy.review(workspace, item_id, label, idempotency_key, previous)
                return result
        item = self.store.item(workspace, item_id)
        previous = self.store.latest_prediction(workspace, item_id)
        if source is None:
            source = "spot_check" if previous and previous.get("spot_check") else "answer"
        self._validate_training(workspace, pd.DataFrame([item]), [label])
        current = self.card(workspace)
        result, created = self.store.answer(
            workspace,
            item_id,
            label,
            idempotency_key,
            {
                "item_id": item_id,
                "coverage": current.coverage,
                "model_version": current.model_version,
                "retrain": "queued",
            },
            source,
            review=None
            if not previous or item_id.startswith("outcome-")
            else (kind(previous.get("source_app")), previous["label"] == label, previous["prob"]),
        )
        if created:
            if self.readers and source == "spot_check":
                self.readers.on_answer(workspace, item_id, label, previous)
            with self.store.connect() as db:
                position = db.execute(
                    """SELECT l.position FROM answers a JOIN labels l
                    ON l.workspace_id=a.workspace_id AND l.id=a.label_id
                    WHERE a.workspace_id=? AND a.idempotency_key=?""",
                    (workspace, idempotency_key),
                ).fetchone()[0]
            with self._guard:
                future = self._executor.submit(self._apply, workspace, position)
                self._pending.add(future)
                future.add_done_callback(lambda done: self._finished(workspace, done))
        return result

    def _finished(self, workspace, future):
        with self._guard:
            self._pending.discard(future)
            if error := future.exception():
                self._errors[workspace] = error
                logger.error("Workspace retraining failed; durable labels will replay on rebuild.")

    def _apply(self, workspace, position):
        with self._lock(workspace):
            try:
                loop = self._runtime(workspace, before=position)
                history = self.store.labels(workspace)
                known = {r["label"] for r in history if r["position"] <= self._positions[workspace]}
                for row in history:
                    if row["position"] <= self._positions[workspace]:
                        continue
                    frame = pd.DataFrame([{**row, "id": row["item_id"]}])
                    self._validate_training(workspace, frame, [row["label"]])
                    known.add(row["label"])
                    if len(known) >= 2:
                        loop.on_answer(frame, row["label"], row["source"])
                    else:
                        loop.learn(frame, [row["label"]], row["source"])
                    self._positions[workspace] = row["position"]
                self._sync_models(workspace)
                with self._guard:
                    self._errors.pop(workspace, None)
            except Exception:
                self._loops.pop(workspace, None)
                self._positions.pop(workspace, None)
                raise

    def drain(self):
        with self._guard:
            pending = list(self._pending)
        wait(pending)
        for future in pending:
            try:
                future.result()
            except Exception as error:
                raise RuntimeError("Workspace retraining failed.") from error
        with self._guard:
            if self._errors:
                raise RuntimeError("Workspace retraining failed.") from next(
                    iter(self._errors.values())
                )

    def close(self):
        if self.cold_start:
            self.cold_start.close()
        self._executor.shutdown(wait=True)
