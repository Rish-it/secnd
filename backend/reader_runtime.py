"""Workspace selection and persisted inference for the configurable reader."""

import json
from threading import RLock

import numpy as np

from backend import metrics
from backend.calibrate import TARGET
from backend.loop import SPOT_CHECK_RATE
from backend.reader_policy import MANIFEST
from backend.reader_scope import scope_digest
from backend.reader_serving import CPUReader
from backend.registry import Registry
from backend.schemas import EvalCard


class ReaderRuntime:
    def __init__(self, store, settings, factory=CPUReader):
        self.store, self.settings, self.factory = store, settings, factory
        self._readers, self._lock = {}, RLock()
        self._checks = {}
        self.scope = json.loads(MANIFEST.read_text())["scope"]

    def enabled(self, workspace):
        if workspace in self.settings.workspace_backends:
            return self.settings.workspace_backends[workspace] == "reader"
        if workspace in self.settings.workspace_readers:
            return True
        # A registered workspace keeps its own calibrated head unless explicitly selected.
        if Registry(self.settings.data_dir, workspace).current() is not None:
            return False
        return self.settings.serving_backend == "reader"

    def _state(self, workspace):
        reader = self.reader(workspace)
        key = (workspace, reader.options.model_dump_json(), reader.artifact.policy_file)
        with self._lock:
            if key not in self._checks:
                self._checks[key] = {
                    "right": 0,
                    "checked": 0,
                    "answers": set(),
                    "rng": np.random.default_rng(self.store.workspace(workspace).get("seed", 0)),
                }
            return self._checks[key]

    def guarded(self, workspace):
        state = self._state(workspace)
        with self._lock:
            right, checked = state["right"], state["checked"]
        return metrics.clopper_pearson_lower(checked - right, checked) > 1 - TARGET

    def on_answer(self, workspace, item_id, label, previous):
        info = (previous or {}).get("reader")
        if not info or not previous.get("spot_check"):
            return
        reader = self.reader(workspace)
        if (info["model_id"], info["revision"], info["examples"]) != (
            reader.options.model_id,
            reader.artifact.revision,
            reader.options.examples,
        ):
            return
        state = self._state(workspace)
        key = (item_id, previous["model_version"])
        with self._lock:
            if key not in state["answers"]:
                state["answers"].add(key)
                state["right"] += previous["label"] == label
                state["checked"] += 1

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
            labels,
            history,
            frame,
            allow_auto=self.scope_matches(labels, history) and not self.guarded(workspace),
        )
        state = self._state(workspace)
        for row in rows:
            if row["auto"] and state["rng"].random() < SPOT_CHECK_RATE:
                row["auto"], row["spot_check"] = False, True
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
            "guarded": self.guarded(workspace),
            "scope_note": "Automatic reader answers require the audited full MTOP history. "
            "Smaller, edited or invented histories require human confirmation.",
            "benchmark": reader.artifact.benchmark(reader.options),
        }
