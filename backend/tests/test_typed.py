import json

import numpy as np
import pandas as pd

from backend import typed


def _frame():
    questions = {
        "action": {
            "type": "choice",
            "instructions": "What next?",
            "criteria": {"stop": "Halt.", "go": "Proceed."},
        },
        "flag": {"type": "noul", "instructions": "Needs review."},
        "risk": {"type": "score", "instructions": "How risky?", "criteria": ["Low", "Mid", "High"]},
    }
    gold = {
        "action": {"label": "go", "probabilities": {"stop": 0.3, "go": 0.7}},
        "flag": {"label": "true", "noul": 0.6, "probabilities": {"false": 0.4, "true": 0.6}},
        "risk": {"label": "2", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    }
    return pd.DataFrame(
        {
            "id": ["c1"],
            "state": ["{}"],
            "questions": [json.dumps(questions)],
            "gold": [json.dumps(gold)],
        }
    )


def test_every_type_becomes_one_decision_with_its_gold_index():
    ds = {d.kind: d for d in typed.decisions(_frame())}
    assert ds["choice"].keys == ["stop", "go"] and ds["choice"].label == 1
    assert ds["noul"].keys == ["false", "true"] and ds["noul"].meanings == ["No.", "Yes."]
    assert ds["score"].keys == ["0", "1", "2"] and ds["score"].label == 2
    assert all(np.isclose(d.gold.sum(), 1) for d in ds.values())


def test_noul_counts_as_true_from_one_half():
    noul = next(d for d in typed.decisions(_frame()) if d.kind == "noul")
    assert typed.correct(noul, np.array([0.5, 0.5]))
    assert not typed.correct(noul, np.array([0.51, 0.49]))
