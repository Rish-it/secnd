import json

import pytest

from backend.release import plan, publish


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: apache-2.0\nlanguage: en\n---\n# Ren\n")
    (tmp_path / "model.pkl").write_bytes(b"illustrative test artifact, never unpickled")
    (tmp_path / "card.json").write_text(json.dumps({"model_version": "ren-1.0", "coverage": 0}))
    (tmp_path / "runtime.json").write_text(
        json.dumps(
            {
                "model_id": "secnd-ren-1",
                "model_version": "ren-1.0",
                "labels": ["a", "b"],
                "featurizer": "minilm",
                "merchant_memory": {"merchant": 0},
            }
        )
    )
    return tmp_path


class NoNetwork:
    def create_repo(self, *args, **kwargs):
        raise AssertionError("Preview must never call the Hub")


def test_default_only_previews_and_hashes_files(bundle):
    result = publish(bundle, "owner/secnd-ren-1", api=NoNetwork())
    assert result["mode"] == "preview" and result["model_version"] == "ren-1.0"
    assert set(result["sha256"]) == {"README.md", "model.pkl", "card.json", "runtime.json"}
    assert all(len(value) == 64 for value in result["sha256"].values())
    assert result == publish(bundle, "owner/secnd-ren-1", api=NoNetwork())


def test_explicit_upload_uses_only_allowlisted_files(bundle):
    class RecordingApi:
        calls = []

        def create_repo(self, *args, **kwargs):
            self.calls.append(("create", args, kwargs))

        def upload_folder(self, **kwargs):
            self.calls.append(("upload", kwargs))

    api = RecordingApi()
    assert publish(bundle, "owner/secnd-ren-1", upload=True, api=api)["mode"] == "uploaded"
    assert api.calls[0][2]["private"] is False
    assert api.calls[1][1]["allow_patterns"] == [
        "README.md",
        "model.pkl",
        "card.json",
        "runtime.json",
    ]


def test_draft_missing_license_extras_and_links_are_refused(bundle):
    original = (bundle / "README.md").read_text()
    for text in ("# DRAFT Ren", "# Ren without a release license"):
        (bundle / "README.md").write_text(text)
        with pytest.raises(ValueError):
            publish(bundle, "owner/secnd-ren-1", upload=True, api=NoNetwork())
    (bundle / "README.md").write_text(original)
    extra = bundle / ".env"
    extra.write_text("test only")
    with pytest.raises(ValueError, match="only"):
        plan(bundle)
    extra.unlink()
    model = bundle / "model.pkl"
    model.unlink()
    model.symlink_to(bundle / "README.md")
    with pytest.raises(ValueError, match="regular file"):
        plan(bundle)


def test_version_and_runtime_must_match(bundle):
    runtime = json.loads((bundle / "runtime.json").read_text())
    runtime["model_version"] = "ren-1.1"
    (bundle / "runtime.json").write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="versions must match"):
        plan(bundle)
    with pytest.raises(ValueError, match="intended"):
        publish(bundle, "owner/other")
