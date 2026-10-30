"""Persist connector provenance alongside the immutable prediction snapshot."""

import json

from backend.store import encode


def attach_source(service, workspace, item_id, source):
    with service.store.connect(write=True) as db:
        row = db.execute(
            "SELECT id, decision_json FROM predictions WHERE workspace_id=? AND item_id=? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (workspace, item_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("Connector prediction was not persisted.")
        decision = json.loads(row["decision_json"])
        decision["source_app"] = source
        db.execute(
            "UPDATE predictions SET decision_json=? WHERE workspace_id=? AND id=?",
            (encode(decision), workspace, row["id"]),
        )
    return decision
