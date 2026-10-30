from types import SimpleNamespace

from backend.reader_audit import apply_audit, choose_floor


def test_floor_is_lowest_grid_value_that_caps_missing_accepts():
    # 1,000 rows allow 2 accepts; the third-highest support (0.62) must fall below the floor
    assert choose_floor([0.9, 0.7, 0.62, 0.4], 1000) == 0.63
    assert choose_floor([], 1000) == 0.0
    assert choose_floor([1.0, 1.0, 1.0], 1000) == 1.01


def test_apply_audit_moves_only_named_calibration_rows():
    task = SimpleNamespace(labels=["x", "y"], calib=[("a", 0), ("b", 0)])
    apply_audit(task, [dict(id="c1", message="b", old="x", new="y")])
    assert task.calib == [("a", 0), ("b", 1)]
