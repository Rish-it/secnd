import numpy as np
import pandas as pd

from backend import bench, data


def test_a_rerun_skips_rows_already_written(tmp_path, monkeypatch):
    keys = np.random.default_rng(0).integers(0, 400, 2000)
    frame = data._frame(
        pd.Series(
            [f"payee {chr(97 + k % 26)}{chr(97 + k // 26)} #{i}" for i, k in enumerate(keys)]
        ),
        pd.Series([f"class {k % 5}" for k in keys]),
    )
    monkeypatch.setattr(bench, "load_dataset", lambda name, data_dir: frame)
    out = tmp_path / "grid.csv"
    first = bench.run("toy", (100,), [0], ("char_tfidf",), ("lr", "knn"), out)
    assert len(pd.read_csv(out)) == 2

    def no_fit(*args, **kwargs):
        raise AssertionError("refit a finished row")

    monkeypatch.setattr(bench.Model, "fit", no_fit)
    again = bench.run("toy", (100,), [0], ("char_tfidf",), ("lr", "knn"), out)
    pd.testing.assert_frame_equal(again, pd.read_csv(out))
    assert list(again["accuracy"]) == list(first["accuracy"])
