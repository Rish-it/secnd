"""Validate historical labels and freeze a workspace's calibration holdout once."""

import csv
import io

import pandas as pd
from pydantic import TypeAdapter
from sklearn.model_selection import GroupShuffleSplit

from backend.data import merchant_key, preview_csv
from backend.schemas import Item, Label


def labelled_rows(content: bytes, target_column: str) -> pd.DataFrame:
    inspected = preview_csv(content, "decisions.csv")
    if target_column not in inspected.columns:
        raise ValueError("The target column is missing.")
    text_column = next((c for c in ("text", "description") if c in inspected.columns), None)
    if text_column is None or target_column in (text_column, "amount", "date", "id", "item_id"):
        raise ValueError("Include a text or description column and a separate label column.")
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    next(reader)
    rows = []
    for values in reader:
        if not values or not any(v.strip() for v in values):
            continue
        row = dict(zip(inspected.columns, values, strict=True))
        item = Item.model_validate(
            {
                "item_id": row.get("item_id", row.get("id", f"import-{len(rows)}")),
                "text": row[text_column],
                "amount": row.get("amount") or None,
                "date": row.get("date") or None,
            }
        )
        label = TypeAdapter(Label).validate_python(row[target_column].strip())
        rows.append(
            dict(
                id=item.item_id,
                text=item.text,
                amount=item.amount,
                date=item.date.isoformat() if item.date else None,
                merchant_key=merchant_key(item.text),
                label=label,
            )
        )
    frame = pd.DataFrame(rows)
    if frame.id.duplicated().any():
        raise ValueError("Each imported item ID must be unique.")
    if frame.label.nunique() < 2:
        raise ValueError("Include examples from at least two labels.")
    return frame


def holdout(frame: pd.DataFrame, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Workspace cards measure calibration, with no claim about an independent test set."""
    if frame.date.notna().all():
        order = frame.sort_values(["date", "id"], kind="stable")
        count = max(1, round(len(frame) * 0.2))
        train, calib = order.iloc[:-count], order.iloc[-count:]
    else:
        groups = frame.merchant_key.where(frame.merchant_key != "", "#" + frame.id)
        if groups.nunique() < 2:
            raise ValueError("Include at least two distinct descriptions for a held-out group.")
        fit, held = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                frame, groups=groups
            )
        )
        train, calib = frame.iloc[fit], frame.iloc[held]
    if train.label.nunique() < 2:
        raise ValueError("The training portion needs at least two labels. Add more examples.")
    return train, calib
