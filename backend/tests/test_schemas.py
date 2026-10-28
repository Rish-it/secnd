import pytest
from pydantic import ValidationError

from backend.schemas import Decision, EvalCard, PredictRequest


def test_top_k_probabilities_are_not_renormalized():
    decision = Decision(
        item_id="invoice-1",
        label="Travel",
        probabilities={"Travel": 0.8, "Software": 0.15},
        prob=0.8,
        auto=False,
        threshold=None,
        model_version="v0",
        reasons={"neighbours": [], "merchant_memory": False},
    )
    assert sum(decision.probabilities.values()) == pytest.approx(0.95)
    for change in [
        {"auto": True},
        {"auto": True, "threshold": 0.99},
        {"prob": 0.7},
        {"label": "Missing"},
        {"probabilities": {"Travel": 0.8, "Software": 0.3}},
    ]:
        with pytest.raises(ValidationError):
            Decision.model_validate(decision.model_dump() | change)


def test_unmeasured_eval_card_does_not_claim_accuracy_or_precision():
    card = EvalCard(model_version="fixture-v0", coverage=0)
    assert card.accuracy is None
    assert card.precision is None
    assert card.threshold is None
    with pytest.raises(ValidationError):
        EvalCard(model_version="v0", coverage=1.01)


def test_predict_rejects_empty_oversized_or_duplicate_batches():
    item = {"item_id": "a", "text": "Train"}
    assert len(PredictRequest(items=[item]).items) == 1
    for items in [[], [item, item], [dict(item, item_id=str(i)) for i in range(257)]]:
        with pytest.raises(ValidationError):
            PredictRequest(items=items)
    with pytest.raises(ValidationError):
        PredictRequest(items=[dict(item, amount=float("nan"))])
