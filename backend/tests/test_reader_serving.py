import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.calibrate import scale
from backend.config import ReaderOptions, Settings
from backend.reader_policy import ReaderArtifact, ReaderPolicy, ReaderUnavailable
from backend.reader_runtime import ReaderRuntime
from backend.reader_scope import scope_digest
from backend.reader_serving import CPUReader
from backend.store import Store
from backend.tests.test_workspace import RecordingLoop
from backend.workspace import WorkspaceService


class Tokenizer:
    mask_token, sep_token = "[MASK]", "[SEP]"
    mask_token_id, pad_token_id = 103, 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [101] + [103 if t == "[MASK]" else 10 for t in text.split()]}


class Embedder:
    def encode(self, texts, **kwargs):
        values = []
        for text in texts:
            cosine = float(text.rsplit(":", 1)[1]) if text.startswith("history:") else 1.0
            values.append([cosine, np.sqrt(max(0, 1 - cosine**2))])
        return np.asarray(values, dtype=np.float32)


class TinyModel:
    logit = 15.0

    def __call__(self, *args):
        return torch.tensor([[self.logit, 0.0]])


@pytest.fixture
def stand_in(monkeypatch):
    # Only model/tokenizer/embedding transport is replaced. Policy lookup, temperature,
    # retrieval, support calculation, authorization and Decision validation are production code.
    monkeypatch.setattr(ReaderArtifact, "verify_checkpoint", lambda *args: None)

    def load(reader):
        reader.model, reader.tok, reader.embedder = TinyModel(), Tokenizer(), Embedder()

    monkeypatch.setattr(CPUReader, "_load", load)


def history(cosine=0.81, *, carrying=True):
    return [
        {"id": str(i), "text": f"history:{i}:{0.95 - i * 0.001}", "label": "b"} for i in range(8)
    ] + ([{"id": "support", "text": f"history:support:{cosine}", "label": "a"}] if carrying else [])


def predict(reader, rows, **kwargs):
    return reader.predict(
        ["a", "b"], rows, pd.DataFrame([{"id": "next", "text": "message"}]), **kwargs
    )[0]


@pytest.mark.parametrize("cosine,expected", [(0.79, False), (0.81, True), (0.8, True)])
def test_support_floor_uses_all_rows_of_predicted_intent(stand_in, tmp_path, cosine, expected):
    reader = CPUReader(tmp_path)
    decision = predict(reader, history(cosine), allow_auto=True)
    assert decision["auto"] is expected
    assert decision["reader"]["support"] == pytest.approx(cosine)
    assert decision["reader"]["support_floor"] == 0.80
    assert decision["prob"] > decision["threshold"]
    assert all(n["label"] == "b" for n in decision["reasons"]["neighbours"])


def test_no_history_for_predicted_intent_fails_closed(stand_in, tmp_path):
    decision = predict(CPUReader(tmp_path), history(carrying=False), allow_auto=True)
    assert decision["reader"]["support"] is None and not decision["auto"]


def test_temperature_is_loaded_and_applied_before_threshold(stand_in, tmp_path):
    reader = CPUReader(tmp_path)
    reader.model.logit = 6
    decision = predict(reader, history(), allow_auto=True)
    raw = torch.softmax(torch.tensor([[6.0, 0.0]]), -1).numpy()
    assert decision["prob"] == pytest.approx(scale(raw, reader.policy.temperature)[0, 0])
    assert raw.max() >= reader.policy.threshold > decision["prob"]
    assert not decision["auto"]


@pytest.mark.parametrize("missing", ["temperature", "threshold", "floor"])
def test_missing_policy_value_fails_closed_in_prediction(stand_in, tmp_path, missing):
    reader = CPUReader(tmp_path)
    reader.policy = replace(reader.policy, **{missing: None})
    decision = predict(reader, history(), allow_auto=True)
    assert not decision["auto"] and not decision["reader"]["policy_complete"]
    assert decision["threshold"] is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"floor": None},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"threshold": 1.1},
    ],
)
def test_missing_or_invalid_results_fail_closed(value):
    assert not ReaderPolicy.from_result(value).accepts(1, 1)


@pytest.mark.parametrize("examples", [0, 32])
def test_real_400m_null_operating_points_fail_closed(stand_in, tmp_path, examples):
    reader = CPUReader(
        tmp_path, ReaderOptions(model_id="ic-ettin-400m-s6000-near", examples=examples)
    )
    decision = predict(reader, history(), allow_auto=True)
    assert not reader.policy.complete and not decision["auto"]
    assert decision["threshold"] is None


def test_no_inherited_scope_and_too_few_examples_block(stand_in, tmp_path):
    reader = CPUReader(tmp_path)
    assert not predict(reader, history())["auto"]
    assert not predict(reader, history()[:1], allow_auto=True)["auto"]
    assert not predict(reader, [], allow_auto=True)["auto"]


def test_reader_switch_loads_both_profiles_without_gpu_work(stand_in, tmp_path):
    small = CPUReader(tmp_path)
    large = CPUReader(tmp_path, ReaderOptions(model_id="ic-ettin-400m-s6000-near", device="cuda"))
    assert small.device == "cpu" and large.device == "cuda"
    assert small.policy.floor == 0.8 and large.policy.floor == 0.85
    assert small.policy.temperature != large.policy.temperature
    assert small.artifact.revision != large.artifact.revision
    assert small.version(["a", "b"], []) != large.version(["a", "b"], [])


def test_configuration_defaults_and_workspace_overrides(monkeypatch):
    monkeypatch.delenv("SECND_SERVING_BACKEND")
    defaults = Settings(_env_file=None)
    assert defaults.serving_backend == "reader"
    assert defaults.reader == ReaderOptions()
    monkeypatch.setenv(
        "SECND_READER", json.dumps({"model_id": "ic-ettin-400m-s6000-near", "device": "cuda"})
    )
    monkeypatch.setenv("SECND_WORKSPACE_BACKENDS", '{"legacy":"head"}')
    monkeypatch.setenv("SECND_WORKSPACE_READERS", '{"cpu":{"device":"cpu","examples":32}}')
    settings = Settings(_env_file=None)
    assert settings.reader_for("gpu").device == "cuda"
    assert settings.reader_for("cpu").examples == 32
    assert settings.backend_for("legacy") == "head"


def test_missing_checkpoint_is_clear_startup_error(tmp_path):
    settings = Settings(data_dir=tmp_path, serving_backend="reader", _env_file=None)
    with pytest.raises(ReaderUnavailable, match="checkpoint is missing"):
        with TestClient(create_app(settings)):
            pytest.fail("A missing reader must not start the app or fall back")


def test_unavailable_device_is_clear_startup_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ReaderArtifact, "verify_checkpoint", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = Settings(
        data_dir=tmp_path,
        serving_backend="reader",
        reader=ReaderOptions(model_id="ic-ettin-400m-s6000-near", device="cuda"),
        _env_file=None,
    )
    with pytest.raises(ReaderUnavailable, match="CUDA reader requires"):
        with TestClient(create_app(settings)):
            pytest.fail("An unavailable configured GPU must not fall back to CPU")


def test_bad_revision_and_corrupt_checkpoint_fail_closed(tmp_path):
    with pytest.raises(ReaderUnavailable, match="revision"):
        CPUReader(tmp_path, ReaderOptions(revision="unregistered"))
    artifact = ReaderArtifact(ReaderOptions())
    artifact.hashes = {"scorer.pt": "a" * 64}
    (tmp_path / "scorer.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ReaderUnavailable, match="unexpected revision"):
        artifact.verify_checkpoint(tmp_path)


def test_scope_checks_content_labels_duplicates_and_count():
    rows = history()
    runtime = object.__new__(ReaderRuntime)
    runtime.scope = {"history_rows": len(rows), "history_sha256": scope_digest(["a", "b"], rows)}
    assert runtime.scope_matches(["b", "a"], rows[::-1])
    assert not runtime.scope_matches(["a", "b"], rows[:-1])
    assert not runtime.scope_matches(["a", "b"], [{**rows[0], "label": "a"}] + rows[1:])
    assert not runtime.scope_matches(["a", "b"], [rows[1]] + rows[1:])


def test_http_routes_reader_persists_queue_and_learns_answers(stand_in, tmp_path):
    settings = Settings(data_dir=tmp_path, serving_backend="reader", _env_file=None)
    svc = WorkspaceService(Store(settings.database_path), tmp_path, RecordingLoop)
    calib = pd.DataFrame([dict(id="heldout", text="private holdout", merchant_key="", label="a")])
    svc.create("default", ["a", "b"], calib, featurizer="char_tfidf")
    rows = [{**r, "merchant_key": ""} for r in history()]
    svc.store.put_items("default", rows)
    for row in rows:
        svc.store.add_label("default", row["id"], row["label"])
    with TestClient(create_app(settings, workspace_service=svc)) as client:
        result = client.post(
            "/api/v1/workspaces/default/predict",
            json={"items": [{"item_id": "q", "text": "message"}]},
        ).json()[0]
        assert result["reader"]["support"] > 0.8 and not result["auto"]
        assert not result["reader"]["scope_authorized"]
        queue = client.get("/api/v1/workspaces/default/queue").json()
        assert queue["items"][0]["item_id"] == "q"
        status = client.get("/api/v1/workspaces/default/serving").json()
        assert status["model_id"] == "ic-ettin-150m-s6000-near" and not status["scope_matches"]
        assert status["benchmark"]["history_rows"] == 11667
        answer = client.post(
            "/api/v1/workspaces/default/answers",
            json={"item_id": "q", "label": "a"},
            headers={"Idempotency-Key": "confirm"},
        )
        assert answer.status_code == 200
        svc.drain()
        assert "heldout" not in {r["id"] for r in svc.readers.history("default")}
        assert any(r["id"] == "q" and r["label"] == "a" for r in svc.readers.history("default"))
        assert svc._loops["default"].answers[0][0] == "q"
        # The frozen threshold and floor never change as answers enter history.
        after = client.get("/api/v1/workspaces/default/serving").json()
        assert (
            after["threshold"] == status["threshold"]
            and after["support_floor"] == status["support_floor"]
        )


def test_http_auto_answer_path_is_wired_for_authorized_history(stand_in, tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        serving_backend="reader",
        workspace_backends={"legacy": "head"},
        _env_file=None,
    )
    svc = WorkspaceService(Store(settings.database_path), tmp_path, RecordingLoop)
    for workspace in ("default", "legacy"):
        svc.create(
            workspace,
            ["a", "b"],
            pd.DataFrame([dict(id="heldout", text="hold", label="a")]),
            featurizer="char_tfidf",
        )
        rows = [{**r, "merchant_key": ""} for r in history()]
        svc.store.put_items(workspace, rows)
        for row in rows:
            svc.store.add_label(workspace, row["id"], row["label"])
    with TestClient(create_app(settings, workspace_service=svc)) as client:
        # Synthetic scope for a unit/integration test, never a production certification claim.
        svc.readers.scope = {
            "history_rows": len(rows),
            "history_sha256": scope_digest(["a", "b"], rows),
        }
        response = client.post(
            "/api/v1/workspaces/default/predict",
            json={"items": [{"item_id": "q", "text": "message"}]},
        )
        assert response.status_code == 200 and response.json()[0]["auto"]
        assert len(client.get("/api/v1/workspaces/default/feed").json()["items"]) == 1
        legacy = client.post(
            "/api/v1/workspaces/legacy/predict",
            json={"items": [{"item_id": "q", "text": "message"}]},
        ).json()[0]
        assert legacy["reader"] is None
        assert client.get("/api/v1/workspaces/legacy/serving").json()["backend"] == "head"
