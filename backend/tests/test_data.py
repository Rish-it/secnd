import hashlib

import pytest

from backend import data


def test_download_dataset_checks_sha256(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_bytes(b"text,label\nhello,greeting\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(data.DATASETS, "good", [("rows.csv", source.as_uri(), digest)])
    monkeypatch.setitem(data.DATASETS, "bad", [("rows.csv", source.as_uri(), "0" * 64)])

    folder = data.download_dataset("good", tmp_path / "data")
    assert (folder / "rows.csv").read_bytes() == source.read_bytes()

    with pytest.raises(ValueError, match="does not match"):
        data.download_dataset("bad", tmp_path / "data")
    assert not (tmp_path / "data" / "datasets" / "bad" / "rows.csv").exists()
