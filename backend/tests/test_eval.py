import pandas as pd

from backend import data, eval


def test_training_rows_take_seed_then_earliest_stream():
    frame = data._frame(
        pd.Series([f"row {i}" for i in range(6)]),
        pd.Series(["a", "b"] * 3),
        date=pd.date_range("2024-01-01", periods=6, freq="D").to_series().iloc[::-1].to_numpy(),
    )
    split = pd.Series(["stream", "stream", "seed", "stream", "calib", "test"], index=frame.index)

    rows = eval.training_rows(frame, split, budget=3, seed=0)

    # Dates run backwards, so the earliest stream rows are ids 3 then 1, not 0 then 1.
    assert rows["id"].tolist() == [2, 3, 1]
