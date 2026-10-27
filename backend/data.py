import csv
import io

from pydantic import BaseModel

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
