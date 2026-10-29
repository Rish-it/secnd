import hashlib
import zipfile

import numpy as np
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


def test_load_dataset_reads_mtop_english_from_the_zip(tmp_path, monkeypatch):
    source = tmp_path / "mtop.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for part in ("train", "eval", "test"):
            archive.writestr(
                f"mtop/en/{part}.txt",
                f"1\tIN:GET_WEATHER\t\tweather {part}?\tweather\ten_XX\t[]\t{{}}\n",
            )
        archive.writestr(
            "mtop/de/train.txt", "2\tIN:GET_WEATHER\t\twetter?\tweather\tde_DE\t[]\t{}\n"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(data.DATASETS, "mtop", [("mtop.zip", source.as_uri(), digest)])

    frame = data.load_dataset("mtop", tmp_path / "data")

    assert list(frame.columns) == data.COLUMNS
    assert list(frame.text) == ["weather train?", "weather eval?", "weather test?"]
    assert set(frame.label) == {"IN:GET_WEATHER"}


def test_load_dataset_reads_topv2_top_intents(tmp_path, monkeypatch):
    files = []
    for part, parse in (
        ("train", "[IN:CREATE_ALARM Set the alarm [SL:DATE_TIME for 6am ] ]"),
        ("eval", "[IN:GET_WEATHER weather ]"),
        ("test", "[SL:UNSUPPORTED not an intent ]"),
    ):
        path = tmp_path / f"{part}.parquet"
        pd.DataFrame({"domain": ["x"], "utterance": [part], "semantic_parse": [parse]}).to_parquet(
            path
        )
        files.append((path.name, path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setitem(data.DATASETS, "topv2", files)

    frame = data.load_dataset("topv2", tmp_path / "data")

    assert list(frame.text) == ["train", "eval"]
    assert list(frame.label) == ["IN:CREATE_ALARM", "IN:GET_WEATHER"]


def _frame(n, dated):
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 400, n)
    return data._frame(
        pd.Series(
            [f"payee {chr(97 + k % 26)}{chr(97 + k // 26)} #{i}" for i, k in enumerate(keys)]
        ),
        pd.Series([f"class {k % 12}" for k in keys]),
        date=pd.date_range("2024-01-01", periods=n, freq="h") if dated else None,
    )


@pytest.mark.parametrize("dated", [True, False])
def test_split_dataset_is_disjoint_sized_and_stable(dated):
    frame = _frame(30000, dated)
    split = data.split_dataset(frame)

    counts = split.value_counts()
    assert set(counts.index) == set(data.SPLITS)
    assert counts["seed"] == data.SEED_ROWS
    assert 0.12 < counts["test"] / len(frame) < 0.18
    assert data.split_manifest("x", frame, split) == data.split_manifest(
        "x", frame, data.split_dataset(frame.copy())
    )

    seed_keys = set(frame.merchant_key[split == "seed"])
    calib_keys = set(frame.merchant_key[split == "calib"])
    test_keys = set(frame.merchant_key[split == "test"])
    if dated:
        assert frame.date[split == "test"].min() >= frame.date[split != "test"].max()
        assert (
            frame.date[split == "calib"].min() >= frame.date[split.isin(["seed", "stream"])].max()
        )
        assert frame.date[split == "seed"].max() <= frame.date[split == "stream"].quantile(0.2)
    else:
        assert not calib_keys & test_keys
        assert not (seed_keys | set(frame.merchant_key[split == "stream"])) & (
            calib_keys | test_keys
        )


def test_split_manifest_hash_changes_with_assignment():
    frame = _frame(1000, True)
    split = data.split_dataset(frame)
    moved = split.copy()
    moved.iloc[0] = "test"
    assert (
        data.split_manifest("x", frame, split)["hash"]
        != data.split_manifest("x", frame, moved)["hash"]
    )


def test_seed_quota_ties_do_not_depend_on_row_order():
    # 7 equal labels share 200 rows: 28 each and 4 left over, so the tie-break decides who gets 29.
    labels = pd.Series([f"class {i % 7}" for i in range(700)])
    quota = data.seed_quota(labels)
    assert quota.sum() == 200
    assert quota.to_dict() == data.seed_quota(labels.iloc[::-1]).to_dict()
    assert quota.to_dict() == data.seed_quota(labels.sample(frac=1, random_state=3)).to_dict()
    assert list(quota[quota == 29].index) == ["class 0", "class 1", "class 2", "class 3"]
