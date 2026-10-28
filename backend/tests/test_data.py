import hashlib

import pandas as pd
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


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[debit] SAFEWAY #1197", "safeway"),
        ("FACEBK [***] [200011725-2299]", "facebk"),
        ("XMA December 2024", "xma"),
        ("PP*CHIPOTLE 0385291 OAKLAND CA", "pp chipotle oakland ca"),
        ("M&S Simply Food, Sept 24", "m&s simply food"),
        ("[***]", ""),
    ],
)
def test_merchant_key(text, key):
    assert data.merchant_key(text) == key


def test_load_dataset_maps_ipsa_columns(tmp_path, monkeypatch):
    header = '"Parliamentary ID","Date","Cost Type","Details","Amount Claimed"\n'
    source = tmp_path / "22_23.csv"
    source.write_text(
        header
        + '4837,"08/10/2024","Rent","Office rent [200013636-120]","350.5"\n'
        + '4837,"09/10/2024","Rent","Office rent","350.5"\n'
        + '4837,"10/10/2024","RENT","Office rent","350.5"\n'
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(data.DATASETS, "ipsa", [("22_23.csv", source.as_uri(), digest)])

    frame = data.load_dataset("ipsa", tmp_path / "data")

    assert list(frame.columns) == data.COLUMNS
    row = frame.iloc[0]
    assert (row.text, row.label, row.amount, row.account) == (
        "Office rent [200013636-120]",
        "Rent",
        350.5,
        "4837",
    )
    assert row.date == pd.Timestamp("2024-10-08")
    assert row.merchant_key == "office rent"
    assert set(frame.label) == {"Rent"}
