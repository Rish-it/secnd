import pandas as pd
import pytest

from backend.importing import holdout, labelled_rows


def test_import_columns_and_labels_are_validated_without_leaking_target():
    content = (
        b"description,amount,date,chosen\ntrain fare,2,2020-01-01,Travel\npizza,3,2020-01-02,Food\n"
    )
    frame = labelled_rows(content, "chosen")
    assert frame.text.tolist() == ["train fare", "pizza"]
    assert set(frame.columns) == {"id", "text", "amount", "date", "merchant_key", "label"}
    for target in ("description", "amount", "date", "missing"):
        with pytest.raises(ValueError):
            labelled_rows(content, target)


def test_holdout_is_fixed_grouped_and_time_ordered():
    frame = pd.DataFrame(
        [
            dict(
                id=str(i),
                text="x",
                amount=None,
                date=None,
                merchant_key=f"merchant-{i // 2}",
                label="a" if i % 2 else "b",
            )
            for i in range(100)
        ]
    )
    train, calib = holdout(frame)
    again, held_again = holdout(frame)
    assert train.id.tolist() == again.id.tolist()
    assert calib.id.tolist() == held_again.id.tolist()
    assert set(train.merchant_key).isdisjoint(calib.merchant_key)
    dated = frame.assign(date=pd.date_range("2020-01-01", periods=len(frame)).strftime("%Y-%m-%d"))
    fit, held = holdout(dated)
    assert fit.date.max() < held.date.min()
