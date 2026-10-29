import hashlib
import json
import shutil

import pytest

from backend import encoders
from backend.config import Settings


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    files = {
        "config.json": b"{}",
        "1_Pooling/config.json": b"pool",
        "model.safetensors": b"weights",
    }
    pin = {
        "repo": "Secnd-AI/ren",
        "revision": Settings(_env_file=None).encoder_revision,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(pin))
    monkeypatch.setattr(encoders, "MANIFEST", manifest)
    monkeypatch.setattr(encoders, "BUNDLED", tmp_path / "bundle")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        for name, data in files.items():
            path = kwargs["local_dir"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    monkeypatch.setattr(encoders, "snapshot_download", download)
    return files, calls


def test_pinned_snapshot_is_verified_and_reused(snapshot, tmp_path):
    files, calls = snapshot
    target = encoders.fetch("pt-minilm", tmp_path)
    assert target == tmp_path / "encoders/pt-minilm"
    assert encoders.fetch("pt-minilm", tmp_path) == target
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "Secnd-AI/ren"
    assert calls[0]["revision"] == "5560f25414195f1d1224a6ae7efd19c6b7646d3f"
    assert set(calls[0]["allow_patterns"]) == set(files)
    (target / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        encoders.fetch("pt-minilm", tmp_path)


def test_bad_download_never_publishes_partial_snapshot(snapshot, tmp_path):
    files, _ = snapshot
    files["model.safetensors"] = b"corrupt"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        encoders.fetch("pt-minilm", tmp_path)
    assert not (tmp_path / "encoders/pt-minilm").exists()
    assert not list((tmp_path / "encoders").glob(".pt-minilm-*"))


def test_image_bundle_works_without_network(snapshot, tmp_path, monkeypatch):
    _, calls = snapshot
    original = encoders.fetch("pt-minilm", tmp_path / "build")
    encoders.BUNDLED.mkdir()
    shutil.copytree(original, encoders.BUNDLED / "pt-minilm")
    metadata = encoders.BUNDLED / "pt-minilm/.cache"
    metadata.mkdir()
    (metadata / "private").write_bytes(b"hub metadata")
    calls.clear()
    assert encoders.fetch("pt-minilm", tmp_path / "volume").is_dir()
    assert not (tmp_path / "volume/encoders/pt-minilm/.cache").exists()
    assert calls == []


def test_unpinned_override_is_rejected(snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("SECND_ENCODER_REVISION", "main")
    with pytest.raises(ValueError, match="committed release manifest"):
        encoders.fetch("pt-minilm", tmp_path)
    assert snapshot[1] == []
