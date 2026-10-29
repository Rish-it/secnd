import pytest

from backend.red_team import ATTACKS, summarize, variant


def record(source, correct, accepted, prediction=0):
    return {
        "source": source,
        "correct": correct,
        "accepted": accepted,
        "prediction": prediction,
        "unchanged": False,
        "untruncated_tokens": 10,
    }


def test_zero_acceptance_does_not_claim_precision():
    result = summarize([record(0, False, False)])
    assert result["accepted_precision"] is None
    assert result["precision_interval95"] is None
    assert result["precision_upper95_one_sided"] is None
    assert not result["evidence_below_99"]


def test_missing_gold_acceptances_are_errors():
    _, options = variant("missing_intent", 2, "call mom", 1, [0, 1, 2], {})
    assert options == [0, 2]
    result = summarize([record(i, False, True) for i in range(100)])
    assert result["accepted_precision"] == 0
    assert result["accepted_errors"] == 100
    assert result["evidence_below_99"]


def test_small_paraphrase_sample_uses_matched_baseline():
    baseline = [record(0, True, True), record(1, True, False), record(2, True, True)]
    result = summarize([record(1, False, True, prediction=1)], baseline)
    assert result["baseline_sample_size"] == 1
    assert result["baseline_coverage"] == 0
    assert result["baseline_precision"] is None
    assert result["coverage_change_pp"] == 100
    assert result["prediction_flips"] == result["acceptance_flips"] == 1


@pytest.mark.parametrize("attack", ATTACKS)
def test_variants_reproducible_without_rng_shared_state(attack):
    args = (attack, 7, "Show weather tomorrow", 1, list(range(113)), {"7": "Weather tomorrow?"})
    first = variant(*args)
    variant("noisy_input", 99, "unrelated", 0, [0, 1], {})
    assert variant(*args) == first
    if attack == "option_order":
        assert sorted(first[1]) == list(range(113))
        assert first[1] != list(range(113))
