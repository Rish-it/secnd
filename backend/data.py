import csv
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

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


if __name__ == "__main__":
    for dataset in sys.argv[1:] or DATASETS:
        print(download_dataset(dataset, Settings().data_dir))
