import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.evidence import Evidence
from backend.registry import Registry


def test_saved_cards_and_curves_validate_and_missing_files_fail(tmp_path):
    root = Path(__file__).resolve().parents[2] / "docs/results/s3"
    evidence = Evidence(tmp_path, root)
    for dataset in ("mtop", "banking77", "ipsa"):
        card, curves = evidence.evaluation(dataset)
        assert card.model_version.startswith(dataset)
        assert len(curves.reliability) and len(curves.coverage_precision)
    card, _ = evidence.evaluation("ipsa")
    assert card.coverage == 0 and card.threshold is None and card.precision is None
    with pytest.raises(FileNotFoundError):
        Evidence(tmp_path, tmp_path).evaluation("mtop")


def test_registry_versions_promote_rollback_and_incomplete_card(tmp_path):
    evidence = Evidence(tmp_path, tmp_path)
    registry = Registry(tmp_path, "a")
    assert evidence.models("a") == []
    v0 = registry.save({}, {"coverage": 0.2})
    v1 = registry.save({}, {"coverage": 0.3})
    registry.promote(v0)
    assert [v.current for v in evidence.models("a")] == [True, False]
    assert evidence.promote("a", v1).current
    assert evidence.promote("a", v0).card.coverage == 0.2
    with pytest.raises(KeyError):
        evidence.promote("a", "missing")
    incomplete = registry.save({}, {"threshold": None})
    with pytest.raises(ValidationError):
        evidence.promote("a", incomplete)
    assert registry.current() == v0
    assert evidence.models("b") == []


def test_concurrent_pointer_changes_are_serialised(tmp_path):
    evidence = Evidence(tmp_path, tmp_path)
    registry = Registry(tmp_path, "a")
    versions = [registry.save({}, {"coverage": 0}) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda v: evidence.promote("a", v), versions * 8))
    assert len(result) == 16
    assert registry.current() in versions
    assert sum(v.current for v in evidence.models("a")) == 1


def test_invalid_curve_numbers_are_refused(tmp_path):
    card = {"model_version": "test", "coverage": 0}
    (tmp_path / "card-mtop-minilm+lr-b2000.json").write_text(json.dumps(card))
    (tmp_path / "curves-mtop-minilm+lr-b2000.json").write_text(
        json.dumps({"reliability": [{"confidence": 1.2, "accuracy": 0.5, "count": 1}]})
    )
    with pytest.raises(ValidationError):
        Evidence(tmp_path, tmp_path).evaluation("mtop")
