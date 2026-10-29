import json
from copy import deepcopy
from pathlib import Path

import pytest

from backend.ci_eval import compare


def test_reproduction_detects_quality_provenance_and_schema_drift():
    expected = {
        "accuracy": 0.8,
        "split_hash": "fixed",
        "classwise_ece": {"Travel": 0.01},
    }
    actual = deepcopy(expected)
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


def test_result_comparison_does_not_silently_drop_added_fields():
    with pytest.raises(ValueError, match="Fixed evaluation changed"):
        compare({"accuracy": 0.8}, {"accuracy": 0.8, "git_sha": "unexpected"})


@pytest.mark.parametrize("dataset", ["ipsa", "banking77"])
def test_committed_cards_detect_real_quality_and_split_drift(dataset):
    reference = json.loads(
        Path(f"docs/results/s8/eval-{dataset}-merchant_memory-b200.json").read_text()
    )
    compare(reference, deepcopy(reference))
    for field in ("accuracy", "split_hash"):
        changed = {**reference, field: None}
        with pytest.raises(ValueError, match="Fixed evaluation changed"):
            compare(reference, changed)
