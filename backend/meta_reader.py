"""Meta tasks use their own option text and the workspace's current labelled history."""

import json
from threading import RLock

import numpy as np

from backend.autonomy import kind
from backend.reader_serving import SavedDecisionReader
from backend.schemas import EvalCard

MEANINGS = {
    "pause": "Pause this ad or ad set.",
    "keep": "Keep the current ad settings.",
    "raise budget": "Increase the current daily ad budget.",
    "lower budget": "Decrease the current daily ad budget.",
}


class MetaDecisionRuntime:
    def __init__(self, service, settings, factory=SavedDecisionReader, transport=None):
        self.service, self.settings, self.factory = service, settings, factory
        self._reader, self._lock = None, RLock()
        self.transport = transport

    def enabled(self, workspace):
        return bool(
            (self.settings.meta_decision_model_folder or self.settings.reader_endpoint_url)
            and self.settings.meta_decision_model_id
        )

    def reader(self):
        with self._lock:
            if self._reader is None:
                if self.settings.reader_endpoint_url:
                    from backend.reader_endpoint import RemoteReader

                    self._reader = RemoteReader(self.settings, self.transport)
                    return self._reader
                self._reader = self.factory(
                    self.settings.meta_decision_model_folder, self.settings.meta_decision_model_id
                )
            return self._reader

    def identity(self):
        reader = self.reader()
        return reader.options.model_id, reader.artifact.revision

    def history(self, workspace):
        config = self.service.store.workspace(workspace)
        excluded = {str(r["id"]) for r in config.get("calib", [])}
        rows = {}
        for r in self.service.store.labels(workspace):
            if r["item_id"] not in excluded:
                rows.pop(r["item_id"], None)
                rows[r["item_id"]] = dict(id=r["item_id"], text=r["text"], label=r["label"])
        return list(rows.values())

    def card(self, workspace):
        return EvalCard(model_version="meta-reader-" + self.identity()[1][:32], coverage=0)

    def status(self, workspace):
        model_id, revision = self.identity()
        return dict(
            backend="reader",
            model_id=model_id,
            revision=revision,
            device="remote" if self.settings.reader_endpoint_url else "cpu",
            examples=8,
            history_rows=len(self.history(workspace)),
            policy_complete=False,
            scope_matches=False,
            guarded=self.service.autonomy.guarded(workspace, (model_id, revision)),
            scope_note=(
                "Meta decision reader. Automatic permission uses owned workspace reviews "
                "only; no inherited public certificate."
            ),
        )

    def route(self, workspace, frame):
        from backend.typed import Decision

        labels = self.service.store.workspace(workspace)["labels"]
        history = self.history(workspace)
        results = []
        for record in frame.to_dict("records"):
            decision_type = kind(record.get("source_app"))
            decision = Decision(
                case=str(record["id"]),
                kind="choice",
                text=f"choice Meta {decision_type}: Choose the person's preferred action.",
                keys=labels,
                meanings=[MEANINGS.get(k, k) if decision_type == "ad" else k for k in labels],
                state=record["text"],
                gold=np.zeros(len(labels)),
                label=0,
            )
            rows = self.reader().predict(
                labels,
                history,
                frame[frame.id == record["id"]],
                allow_auto=False,
                decision=decision,
            )
            for result in rows:
                if not result.get("model_id"):
                    result["model_id"], result["model_revision"] = self.identity()
            results.extend(rows)
        self.service.store.record_predictions(
            workspace, json.loads(frame.to_json(orient="records", date_format="iso")), results
        )
        return results
