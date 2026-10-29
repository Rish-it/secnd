import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from backend.config import Settings

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_COLUMNS = 100
PREVIEW_ROWS = 5


class CSVPreview(BaseModel):
    filename: str
    columns: list[str]
    row_count: int
    rows: list[dict[str, str]]


def preview_csv(content: bytes, filename: str) -> CSVPreview:
    """Inspect a UTF-8 CSV without storing its contents or claiming to train a model."""
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("Choose a CSV smaller than 5 MiB.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save your CSV with UTF-8 encoding and try again.") from error
    if "\x00" in decoded:
        raise ValueError("This file contains binary data. Choose a UTF-8 CSV.")

    try:
        reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        headers = next(reader, [])
        columns = [column.strip() for column in headers]
        if not columns or not all(columns):
            raise ValueError("Your CSV needs a header with a name for every column.")
        if len(columns) < 2:
            raise ValueError("Include at least one input column and one label column.")
        if len(columns) > MAX_COLUMNS:
            raise ValueError("Choose a CSV with at most 100 columns.")
        if len(set(columns)) != len(columns):
            raise ValueError("Give each CSV column a unique name.")

        rows: list[dict[str, str]] = []
        row_count = 0
        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != len(columns):
                raise ValueError(f"CSV line {reader.line_num} has a different number of columns.")
            row_count += 1
            if len(rows) < PREVIEW_ROWS:
                rows.append(dict(zip(columns, row, strict=True)))
    except csv.Error as error:
        raise ValueError(
            "This CSV could not be read. Check its delimiters and quoted values."
        ) from error

    if row_count == 0:
        raise ValueError("Your CSV has a header but no decisions. Add at least one row.")

    return CSVPreview(filename=filename, columns=columns, row_count=row_count, rows=rows)


_DODATA = "https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/3e8d052cb7b362fb1a04b6064d2331982553ea56"
_IPSA = "https://www.theipsa.org.uk/api/download?type=individualBusinessCosts&year="
_BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"

# Public datasets pinned to a revision: name -> [(file name, URL, sha256)].
DATASETS: dict[str, list[tuple[str, str, str]]] = {
    # UK MP business-cost claims, Open Government Licence v3.0. IPSA may republish a year with
    # corrections; a hash mismatch then means re-pinning, never silently accepting new rows.
    "ipsa": [
        (
            "22_23.csv",
            f"{_IPSA}22_23",
            "6ea99df7fc2b8caa6f16fdcf39631b598bc4112233f480fc76cf0cf4ccf34f0c",
        ),
        (
            "23_24.csv",
            f"{_IPSA}23_24",
            "4f4f5f52f7850996b48d4e118af7660216eafaaa3b2d93b368a023ee244413e9",
        ),
        (
            "24_25.csv",
            f"{_IPSA}24_25",
            "2996500d0e11696366240061ba021ac1a629c35927e6c2bf345677ce57de7cb2",
        ),
    ],
    "us_bank_v2": [
        (
            "transactions.csv",
            f"{_DODATA}/transactions-synthetic.csv",
            "0424aed6e76f74a5b3b1ff61ccec43bc321622e6806da353b910b3b2c8108f6e",
        ),
    ],
    "banking77": [
        (
            "train.csv",
            f"{_BANKING77}/train.csv",
            "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        ),
        (
            "test.csv",
            f"{_BANKING77}/test.csv",
            "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        ),
    ],
    # Meta's multilingual assistant intents (MTOP), CC BY-SA 4.0; only the English files are read.
    "mtop": [
        (
            "mtop.zip",
            "https://dl.fbaipublicfiles.com/mtop/mtop.zip",
            "0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b",
        ),
    ],
}


def download_dataset(name: str, data_dir: Path) -> Path:
    """Fetch a dataset into data_dir/datasets/<name>/ and refuse any file whose sha256 differs."""
    folder = data_dir / "datasets" / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in DATASETS[name]:
        path = folder / filename
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}/{filename}: sha256 {actual} does not match {expected}")
        path.write_bytes(content)
    return folder


COLUMNS = ["id", "text", "amount", "date", "label", "account", "merchant_key"]

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_MONTH_WORD = re.compile(
    rf"\b(?:{_MONTHS}|january|february|march|april|june|july|august|september|october|november|december)\b"
)
_NOISE = re.compile(r"\[(?:debit|credit|\*+|[\d-]+)\]")


def merchant_key(text: str) -> str:
    """Reduce a description to the payee: drop ids, digits, months and punctuation."""
    key = _NOISE.sub(" ", text.lower())
    key = re.sub(r"[^a-z&]+", " ", key)
    return " ".join(_MONTH_WORD.sub(" ", key).split())


def _frame(text, label, amount=None, date=None, account=None) -> pd.DataFrame:
    text = text.fillna("").astype(str).str.strip()
    # Sources rename labels by case alone ("Pooled Staffing Services"); keep the commonest spelling.
    label = label.astype(str).str.strip()
    label = label.groupby(label.str.lower()).transform(lambda spelling: spelling.mode()[0])
    frame = pd.DataFrame(
        {
            "text": text,
            "amount": pd.Series(amount, index=text.index, dtype="float64"),
            "date": pd.Series(date, index=text.index, dtype="datetime64[ns]"),
            "label": label,
            "account": pd.Series(account, index=text.index, dtype="object"),
            "merchant_key": text.map(merchant_key),
        }
    ).reset_index(drop=True)
    frame.insert(0, "id", range(len(frame)))
    return frame[COLUMNS]


def load_dataset(name: str, data_dir: Path) -> pd.DataFrame:
    """Return one frame per dataset with the columns in COLUMNS; missing fields stay null."""
    folder = download_dataset(name, data_dir)
    if name == "ipsa":
        raw = pd.concat(
            [
                pd.read_csv(folder / file, encoding="utf-8-sig", dtype=str)
                for file, _, _ in DATASETS[name]
            ],
            ignore_index=True,
        )
        return _frame(
            raw["Details"],
            raw["Cost Type"],
            amount=pd.to_numeric(raw["Amount Claimed"]),
            date=pd.to_datetime(raw["Date"], format="%d/%m/%Y"),
            account=raw["Parliamentary ID"],
        )
    if name == "us_bank_v2":
        raw = pd.read_csv(folder / "transactions.csv", dtype=str)
        return _frame(raw["description"], raw["category"])
    if name == "banking77":
        raw = pd.concat([pd.read_csv(folder / f, dtype=str) for f in ("train.csv", "test.csv")])
        return _frame(raw["text"], raw["category"])
    if name == "mtop":
        with zipfile.ZipFile(folder / "mtop.zip") as archive:
            raw = pd.concat(
                pd.read_csv(
                    archive.open(f"mtop/en/{part}.txt"),
                    sep="\t",
                    header=None,
                    usecols=[1, 3],
                    names=["intent", "text"],
                    dtype=str,
                    quoting=csv.QUOTE_NONE,
                )
                for part in ("train", "eval", "test")
            )
        return _frame(raw["text"], raw["intent"])
    raise KeyError(name)


SPLITS = ("seed", "stream", "calib", "test")
SEED_ROWS = 200


def _tail_by_group(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Index:
    """Pick whole merchant groups in a seeded random order until `fraction` of rows is reached."""
    groups = frame["merchant_key"].where(frame["merchant_key"] != "", "#" + frame["id"].astype(str))
    sizes = groups.value_counts().sort_index().sample(frac=1, random_state=seed)
    chosen = sizes.index[sizes.cumsum() <= fraction * len(frame)]
    return frame.index[groups.isin(chosen)]


def seed_quota(labels: pd.Series) -> pd.Series:
    """Seed rows per label, proportional to `labels`; leftover rows go to the largest remainders.

    Ties break by label name with a stable sort, so every platform gives the same quota.
    """
    share = labels.value_counts(normalize=True).sort_index() * SEED_ROWS
    quota = share.astype(int)
    remainder = (share - quota).sort_values(ascending=False, kind="stable")
    quota[remainder.index[: SEED_ROWS - quota.sum()]] += 1
    return quota


def split_dataset(frame: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Assign every row to seed, stream, calib or test; same frame and seed give the same split.

    Dated data splits by time: test is the last 15%, calib the 15% of the rest just before it,
    so calib looks like test. Undated data splits by merchant group, so no payee spans
    calib and test. Seed is 200 rows, stratified by label, from the earliest pool rows.
    """
    dated = frame["date"].notna().all()
    order = frame.sort_values(["date", "id"]) if dated else frame.sample(frac=1, random_state=seed)
    split = pd.Series("stream", index=frame.index)
    if dated:
        n_test = round(0.15 * len(order))
        n_calib = round(0.15 * (len(order) - n_test))
        split[order.index[len(order) - n_test :]] = "test"
        split[order.index[len(order) - n_test - n_calib : len(order) - n_test]] = "calib"
    else:
        split[_tail_by_group(frame, 0.15, seed)] = "test"
        rest = frame[split == "stream"]
        split[_tail_by_group(rest, 0.15, seed + 1)] = "calib"

    pool = order[split[order.index] == "stream"]
    early = pool.iloc[: max(10 * SEED_ROWS, len(pool) // 10)]
    quota = seed_quota(early["label"])
    picked = [
        rows.sample(quota[label], random_state=seed).index
        for label, rows in early.groupby("label")
        if quota[label]
    ]
    split[pd.Index([i for index in picked for i in index])] = "seed"
    return split


def split_manifest(name: str, frame: pd.DataFrame, split: pd.Series) -> dict:
    """Counts per split and class plus a hash of the row ids in each split."""
    lines = "\n".join(f"{i}:{s}" for i, s in sorted(zip(frame["id"], split, strict=True)))
    manifest = {
        "dataset": name,
        "rows": len(frame),
        "hash": hashlib.sha256(lines.encode()).hexdigest(),
        "sources": {file: digest for file, _, digest in DATASETS.get(name, [])},
        "splits": {},
    }
    for part in SPLITS:
        rows = frame[split == part]
        manifest["splits"][part] = {
            "rows": len(rows),
            "first_date": str(rows["date"].min().date()) if rows["date"].notna().any() else None,
            "last_date": str(rows["date"].max().date()) if rows["date"].notna().any() else None,
            "classes": rows["label"].value_counts().sort_index().to_dict(),
        }
    return manifest


if __name__ == "__main__":
    out = Path("docs/results/s1")
    out.mkdir(parents=True, exist_ok=True)
    for dataset in sys.argv[1:] or DATASETS:
        frame = load_dataset(dataset, Settings().data_dir)
        manifest = split_manifest(dataset, frame, split_dataset(frame))
        (out / f"split-{dataset}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(dataset, manifest["hash"], {k: v["rows"] for k, v in manifest["splits"].items()})
