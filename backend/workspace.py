"""Persistent workspace loops and a background answer dispatcher."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from backend.config import Settings
from backend.encoders import fetch
from backend.registry import Registry
from backend.schemas import Decision, WorkspaceId
from backend.store import Store

logger = logging.getLogger(__name__)


def loop_factory(*args, **kwargs):
    # The core stays independent of persistence and HTTP.
    from backend.loop import Loop

    return Loop(*args, **kwargs)


class WorkspaceService:
    def __init__(self, store: Store, data_dir: Path, factory=loop_factory):
        self.store, self.data_dir, self.factory = store, data_dir, factory
        self._guard = threading.RLock()
        self._locks, self._loops, self._positions = {}, {}, {}
        self._pending, self._errors = set(), {}
        # carmy: one process; use a durable worker queue before running multiple API processes.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="workspace-answer")

    def create(self, workspace, labels, calib: pd.DataFrame, *, featurizer=None, seed=0):
        TypeAdapter(WorkspaceId).validate_python(workspace)
        if len(set(labels)) < 2 or len(labels) != len(set(labels)):
            raise ValueError("A workspace needs at least two distinct labels.")
        if calib.empty or not set(calib.label).issubset(labels):
            raise ValueError("Calibration rows must use the workspace labels.")
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
        )

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

    def _runtime(self, workspace, *, before=None):
        if workspace not in self._loops:
            config = self.store.workspace(workspace)
            loop = self.factory(
                config["labels"],
                pd.DataFrame(config["calib"]),
                featurizer=config["featurizer"],
                seed=config["seed"],
                cache_dir=self.data_dir / "cache",
                models=Registry(self.data_dir, workspace),
            )
            history = [
                r for r in self.store.labels(workspace) if before is None or r["position"] < before
            ]
            for row in history:
                frame = pd.DataFrame([{**row, "id": row["item_id"]}])
                self._validate_training(workspace, frame, [row["label"]])
                loop.learn(frame, [row["label"]], row["source"])
            if len({r["label"] for r in history}) >= 2:
                loop.retrain()
                self._sync_models(workspace)
            with self._guard:
                self._errors.pop(workspace, None)
            self._loops[workspace] = loop
            self._positions[workspace] = max((r["position"] for r in history), default=0)
        return self._loops[workspace]

    def route(self, workspace, frame: pd.DataFrame) -> list[dict]:
        with self._lock(workspace):
            loop = self._runtime(workspace)
            decisions = [
                Decision.model_validate(row).model_dump(mode="json") for row in loop.route(frame)
            ]
            self.store.record_predictions(
                workspace, json.loads(frame.to_json(orient="records", date_format="iso")), decisions
            )
            return decisions

    def answer(self, workspace, item_id, label, idempotency_key, *, source="answer") -> dict:
        item = self.store.item(workspace, item_id)
        self._validate_training(workspace, pd.DataFrame([item]), [label])
        current = next((row for row in self.store.models(workspace) if row["is_current"]), None)
        result, created = self.store.answer(
            workspace,
            item_id,
            label,
            idempotency_key,
            {
                "item_id": item_id,
                "coverage": current["card"].get("coverage", 0) if current else 0,
                "model_version": current["version"] if current else "untrained",
                "retrain": "queued",
            },
            source,
        )
        if created:
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
        self._executor.shutdown(wait=True)
