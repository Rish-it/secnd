"""Workspace selection and persisted inference for the configurable reader."""

import json
from threading import RLock

from backend.reader_policy import MANIFEST
from backend.reader_scope import scope_digest
from backend.reader_serving import CPUReader
from backend.schemas import EvalCard


class ReaderRuntime:
    def __init__(self, store, settings, factory=CPUReader):
        self.store, self.settings, self.factory = store, settings, factory
        self._readers, self._lock = {}, RLock()
        self.scope = json.loads(MANIFEST.read_text())["scope"]

    def enabled(self, workspace):
        return self.settings.backend_for(workspace) == "reader"

    def reader(self, workspace):
        options = self.settings.reader_for(workspace)
        key = options.model_dump_json()
        with self._lock:
            if key not in self._readers:
                self._readers[key] = self.factory(self.settings.data_dir, options)
            return self._readers[key]

    def startup(self):
        # Eager loading makes bad model/device configuration a startup error, never a fallback.
        if self.settings.serving_backend == "reader":
            self.reader("")
        for workspace in self.settings.workspace_readers | self.settings.workspace_backends:
            if self.enabled(workspace):
                self.reader(workspace)

    def history(self, workspace):
        config = self.store.workspace(workspace)
        excluded = {str(row["id"]) for row in config["calib"]}
        # Last effective human label wins; held-out calibration never enters reader context.
        rows = {}
        for row in self.store.labels(workspace):
            if row["item_id"] not in excluded:
                rows[row["item_id"]] = {
                    "id": row["item_id"],
                    "text": row["text"],
                    "label": row["label"],
                }
        return list(rows.values())

    def scope_matches(self, labels, history):
        return (
            len(history) == self.scope["history_rows"]
            and scope_digest(labels, history) == self.scope["history_sha256"]
        )

    def route(self, workspace, frame):
        labels = self.store.workspace(workspace)["labels"]
        history = self.history(workspace)
        rows = self.reader(workspace).predict(
            labels, history, frame, allow_auto=self.scope_matches(labels, history)
        )
        self.store.record_predictions(
            workspace, json.loads(frame.to_json(orient="records", date_format="iso")), rows
        )
        return rows

    def card(self, workspace):
        labels = self.store.workspace(workspace)["labels"]
        history = self.history(workspace)
        # Published benchmark metrics are exposed separately; never substitute them for a
        # workspace's own measured card. No new calibration or threshold fitting at serving.
        return EvalCard(model_version=self.reader(workspace).version(labels, history), coverage=0)

    def status(self, workspace, *, cold_history=None):
        config = self.store.workspace(workspace)
        history = self.history(workspace) if cold_history is None else cold_history
        reader = self.reader(workspace)
        policy = reader.policy
        return {
            "backend": "reader",
            "model_id": reader.options.model_id,
            "revision": reader.artifact.revision,
            "device": reader.device,
            "examples": reader.options.examples,
            "history_rows": len(history),
            "temperature": policy.temperature,
            "threshold": policy.threshold,
            "support_floor": policy.floor,
            "policy_complete": policy.complete,
            "scope_matches": cold_history is None and self.scope_matches(config["labels"], history),
            "scope_note": "Automatic reader answers require the audited full MTOP history. "
            "Smaller, edited or invented histories require human confirmation.",
            "benchmark": reader.artifact.benchmark(reader.options),
        }
