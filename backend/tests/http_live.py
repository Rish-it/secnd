"""Run against a local server; creates one illustrative workspace, without provider calls."""

import argparse
import json
import subprocess
import time
import uuid

import httpx

from backend.tests.test_live_api import csv_rows

parser = argparse.ArgumentParser(description="Verify live CPU routes in an isolated workspace.")
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
args = parser.parse_args()
workspace = "http-" + uuid.uuid4().hex[:12]
api = f"{args.base_url.rstrip('/')}/api/v1/workspaces/{workspace}"
with httpx.Client(timeout=120) as client:
    assert client.get(api + "/queue").headers["X-Secnd-Mode"] == "fixture"
    imported = client.post(
        api + "/decisions:import",
        files={"file": ("history.csv", csv_rows(), "text/csv")},
        data={"target_column": "category"},
    )
    assert imported.status_code == 200, imported.text
    assert imported.headers["X-Secnd-Mode"] == "live", imported.text
    version = imported.json()["card"]["model_version"]
    first_metrics = client.get(api + "/metrics")
    assert first_metrics.headers["X-Secnd-Mode"] == "live"
    assert len(first_metrics.json()["history"]) == 1
    assert first_metrics.json()["guarded"] is False
    response = client.post(
        api + "/predict",
        json={"items": [{"item_id": "http-question", "text": "an unfamiliar everyday expense"}]},
    )
    assert response.status_code == 200, response.text
    prediction = response.json()[0]
    assert prediction["model_version"] == version
    queued = client.get(api + "/queue")
    assert queued.status_code == 200 and queued.headers["X-Secnd-Mode"] == "live"
    assert any(r["item_id"] == "http-question" for r in queued.json()["items"])
    body = {"item_id": "http-question", "label": "Food"}
    headers = {"Idempotency-Key": "http-answer"}
    first = client.post(api + "/answers", json=body, headers=headers)
    assert first.status_code == 200 and first.json()["retrain"] == "queued", first.text
    assert client.post(api + "/answers", json=body, headers=headers).json() == first.json()
    assert not client.get(api + "/queue").json()["items"]
    deadline = time.monotonic() + 30
    while True:
        versions = client.get(api + "/models")
        assert versions.status_code == 200, versions.text
        if len(versions.json()["items"]) >= 2:
            break
        assert time.monotonic() < deadline
        time.sleep(0.1)
    metrics = client.get(api + "/metrics").json()
    assert len(metrics["history"]) == len(versions.json()["items"])
    assert metrics["labels"] == first_metrics.json()["labels"] + 1
    rollback = client.post(api + "/models/" + version + ":rollback")
    assert rollback.status_code == 200, rollback.text
    restored = client.post(
        api + "/predict",
        json={"items": [{"item_id": "restored", "text": "another unfamiliar expense"}]},
    )
    assert restored.status_code == 200 and restored.json()[0]["model_version"] == version, (
        restored.text
    )
    report = {
        "source_revision": subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "featurizer": imported.json()["card"]["featurizer_id"],
        "history_versions": len(metrics["history"]),
        "guarded": metrics["guarded"],
        "device": "cpu",
        "illustrative_rows": 200,
        "initial_mode": "fixture",
        "import_mode": imported.headers["X-Secnd-Mode"],
        "predict_mode": response.headers["X-Secnd-Mode"],
        "models_status": versions.status_code,
        "answer_queued": True,
        "idempotency_replay_equal": True,
        "answered_question_removed": True,
        "rollback_prediction_version": restored.json()[0]["model_version"],
        "model_versions": len(versions.json()["items"]),
        "scope": "Illustrative HTTP integration; no benchmark coverage claim.",
    }
    print(json.dumps(report))
