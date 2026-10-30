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


def history(cosine=0.93, *, carrying=True):
    return [
        {"id": str(i), "text": f"history:{i}:{0.95 - i * 0.001}", "label": "b"} for i in range(8)
    ] + ([{"id": "support", "text": f"history:support:{cosine}", "label": "a"}] if carrying else [])


def predict(reader, rows, **kwargs):
    return reader.predict(
        ["a", "b"], rows, pd.DataFrame([{"id": "next", "text": "message"}]), **kwargs
    )[0]


@pytest.mark.parametrize("cosine,expected", [(0.91, False), (0.93, True), (0.92, True)])
def test_support_floor_uses_all_rows_of_predicted_intent(stand_in, tmp_path, cosine, expected):
    reader = CPUReader(tmp_path)
    decision = predict(reader, history(cosine), allow_auto=True)
    assert decision["auto"] is expected
    assert decision["reader"]["support"] == pytest.approx(cosine)
    assert decision["reader"]["support_floor"] == 0.92
    assert decision["prob"] > decision["threshold"]
    assert all(n["label"] == "b" for n in decision["reasons"]["neighbours"])


def test_no_history_for_predicted_intent_fails_closed(stand_in, tmp_path):
    decision = predict(CPUReader(tmp_path), history(carrying=False), allow_auto=True)
    assert decision["reader"]["support"] is None and not decision["auto"]


def test_temperature_is_loaded_and_applied_before_threshold(stand_in, tmp_path):
    reader = CPUReader(tmp_path)
    reader.model.logit = 3.5
    decision = predict(reader, history(), allow_auto=True)
    raw = torch.softmax(torch.tensor([[3.5, 0.0]]), -1).numpy()
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


@pytest.mark.parametrize("examples", [0, 8, 32])
def test_real_400m_audited_operating_points_are_complete(stand_in, tmp_path, examples):
    reader = CPUReader(
        tmp_path, ReaderOptions(model_id="ic-ettin-400m-s6000-near", examples=examples)
    )
    decision = predict(reader, history(), allow_auto=True)
    assert reader.policy.complete
    assert decision["threshold"] == reader.policy.threshold
    assert decision["auto"] is (examples != 32)


def test_no_inherited_scope_and_too_few_examples_block(stand_in, tmp_path):
    reader = CPUReader(tmp_path)
    assert not predict(reader, history())["auto"]
    assert not predict(reader, history()[:1], allow_auto=True)["auto"]
    assert not predict(reader, [], allow_auto=True)["auto"]


def test_reader_switch_loads_both_profiles_without_gpu_work(stand_in, tmp_path):
    small = CPUReader(tmp_path)
    large = CPUReader(tmp_path, ReaderOptions(model_id="ic-ettin-400m-s6000-near", device="cuda"))
    assert small.device == "cpu" and large.device == "cuda"
    assert small.policy.floor == 0.92 and large.policy.floor == 0.90
    assert small.policy.temperature != large.policy.temperature
    assert small.artifact.revision != large.artifact.revision
    assert small.version(["a", "b"], []) != large.version(["a", "b"], [])


def test_configuration_defaults_and_workspace_overrides(monkeypatch):
    monkeypatch.delenv("SECND_SERVING_BACKEND", raising=False)
    defaults = Settings(_env_file=None)
    assert defaults.serving_backend == "head"
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
    settings = Settings(data_dir=tmp_path, workspace_backends={"default": "reader"}, _env_file=None)
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


@pytest.mark.parametrize("global_backend", ["head", "reader"])
def test_registered_workspace_keeps_its_calibrated_gate_by_default(
    stand_in, tmp_path, global_backend
):
    from types import SimpleNamespace

    from backend.registry import Registry

    class CertifiedLoop(RecordingLoop):
        def retrain(self):
            artifact = {"model": SimpleNamespace(C=1), "temperature": 1, "tau": 0.8, "lower": 0.995}
            version = self.models.save(
                artifact, {"coverage": 1, "threshold": 0.8, "precision": 1, "cp_lower_bound": 0.995}
            )
            self.models.promote(version)
            self.current = {**artifact, "version": version}

        def route(self, frame):
            return [
                {**row, "auto": True, "threshold": self.current["tau"]}
                for row in super().route(frame)
            ]

    settings = Settings(data_dir=tmp_path, serving_backend=global_backend, _env_file=None)
    service = WorkspaceService(Store(settings.database_path), tmp_path, CertifiedLoop)
    service.create(
        "default",
        ["a", "b"],
        pd.DataFrame([dict(id="calib", text="holdout", label="a", merchant_key="")]),
        featurizer="char_tfidf",
    )
    rows = [{**r, "merchant_key": ""} for r in history()]
    service.store.put_items("default", rows)
    for row in rows:
        service.store.add_label("default", row["id"], row["label"])
    service._runtime("default")
    with TestClient(create_app(settings, workspace_service=service)) as client:
        assert Registry(tmp_path, "default").current()
        assert not service.readers.enabled("default")
        result = client.post(
            "/api/v1/workspaces/default/predict",
            json={"items": [{"item_id": "next", "text": "message"}]},
        ).json()[0]
        assert result["auto"] and result["threshold"] == 0.8
        assert result["reader"] is None
        assert client.get("/api/v1/workspaces/default/serving").json()["backend"] == "head"
        card = client.get("/api/v1/workspaces/default/metrics").json()["card"]
        assert card["coverage"] == 1 and card["cp_lower_bound"] == 0.995
        # A per-workspace selection is explicit and can override the registered model.
        service.readers.settings.workspace_backends["default"] = "reader"
        assert service.readers.enabled("default")


def test_clean_clone_starts_in_head_mode_without_a_reader(tmp_path, monkeypatch):
    monkeypatch.delenv("SECND_SERVING_BACKEND", raising=False)

    def forbidden(*args):
        pytest.fail("Default startup must not load a reader checkpoint")

    settings = Settings(data_dir=tmp_path, _env_file=None)
    with TestClient(create_app(settings, reader_factory=forbidden)) as client:
        assert client.get("/api/health").status_code == 200


def test_reader_spot_checks_use_existing_rate_and_queue_then_guard(stand_in, tmp_path):
    from types import SimpleNamespace

    from backend.loop import SPOT_CHECK_RATE
    from backend.reader_runtime import SPOT_CHECK_RATE as reader_rate

    assert reader_rate == SPOT_CHECK_RATE
    settings = Settings(data_dir=tmp_path, workspace_backends={"default": "reader"}, _env_file=None)
    service = WorkspaceService(Store(settings.database_path), tmp_path, RecordingLoop)
    service.create(
        "default",
        ["a", "b"],
        pd.DataFrame([dict(id="calib", text="heldout", label="a", merchant_key="")]),
        featurizer="char_tfidf",
    )
    rows = [{**r, "merchant_key": ""} for r in history()]
    service.store.put_items("default", rows)
    for row in rows:
        service.store.add_label("default", row["id"], row["label"])
    with TestClient(create_app(settings, workspace_service=service)) as client:
        runtime = service.readers
        # Isolate spot-check guard behavior from the separate exact-history safety rule.
        runtime.scope_matches = lambda *args: True
        draws = iter([SPOT_CHECK_RATE - 0.001, SPOT_CHECK_RATE + 0.001])
        runtime._state("default")["rng"] = SimpleNamespace(random=lambda: next(draws))
        result = client.post(
            "/api/v1/workspaces/default/predict",
            json={
                "items": [
                    {"item_id": "checked", "text": "message"},
                    {"item_id": "automatic", "text": "another message"},
                ]
            },
        ).json()
        assert result[0]["spot_check"] and not result[0]["auto"]
        assert result[1]["auto"] and not result[1]["spot_check"]
        queue = client.get("/api/v1/workspaces/default/queue").json()["items"]
        assert [row["item_id"] for row in queue] == ["checked"]
        headers = {"Idempotency-Key": "wrong-check"}
        answer = client.post(
            "/api/v1/workspaces/default/answers",
            json={"item_id": "checked", "label": "b"},
            headers=headers,
        )
        assert answer.status_code == 200
        service.drain()
        assert runtime.guarded("default")
        assert service.store.labels("default")[-1]["source"] == "spot_check"
        assert client.get("/api/v1/workspaces/default/metrics").json()["guarded"]
        assert client.get("/api/v1/workspaces/default/serving").json()["guarded"]
        # Idempotent replay cannot count the same check twice.
        client.post(
            "/api/v1/workspaces/default/answers",
            json={"item_id": "checked", "label": "b"},
            headers=headers,
        )
        assert runtime._state("default")["checked"] == 1
        blocked = client.post(
            "/api/v1/workspaces/default/predict",
            json={"items": [{"item_id": "after-guard", "text": "message after guard"}]},
        ).json()[0]
        assert not blocked["auto"] and not blocked["spot_check"]


def test_audited_manifest_and_floored_full_test_benchmark():
    import hashlib
    from pathlib import Path

    from backend.reader_policy import MANIFEST

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["source_commit"] == "ada0a0d"
    for model, entry in manifest["models"].items():
        expected = (
            "attacks-floor-audited-400m.json" if "400m" in model else "attacks-floor-audited.json"
        )
        assert entry["policy_file"] == "docs/results/s3r/" + expected
        raw = Path(entry["policy_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["policy_sha256"]
        for examples in (0, 8, 32):
            options = ReaderOptions(model_id=model, examples=examples)
            artifact = ReaderArtifact(options)
            frozen = json.loads(raw)["results"][str(examples)]
            assert artifact.policy == ReaderPolicy.from_result(frozen)
            card = artifact.benchmark(options)
            full = json.loads(Path(entry["floor_benchmark_file"]).read_text())[model][str(examples)]
            assert card["with_floor_coverage"] == full["coverage"]
            assert card["with_floor_precision"] == full["precision"]
            assert card["with_floor_sample"] == full["test_rows"]
            assert card["with_floor_low95"] == full["low95"]
            assert card["floor_result_file"] == "docs/results/s3r/floor-full-test.json"
            assert card["calibration_audit"] == "model"
