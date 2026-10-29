from copy import deepcopy

import pytest

from backend.ci_eval import compare


def test_reproduction_detects_quality_provenance_and_schema_drift():
    expected = {
        "accuracy": 0.8,
        "split_hash": "fixed",
        "classwise_ece": {"Travel": 0.01},
        "predict_ms": {"p50": 1},
        "peak_rss_bytes": 100,
        "git_sha": "old",
    }
    actual = {**expected, "predict_ms": {"p50": 2}, "peak_rss_bytes": 200, "git_sha": "new"}
    compare(expected, actual)
    for field, value in (
        ("accuracy", 0.7),
        ("split_hash", "changed"),
        ("classwise_ece", {"Travel": 0.02}),
        ("new_field", 0),
    ):
        changed = deepcopy(actual)
        changed[field] = value
        with pytest.raises(ValueError, match="Fixed evaluation changed"):
            compare(expected, changed)
