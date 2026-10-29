import numpy as np
import pandas as pd
import pytest

from backend import baselines, data

LABELS = ["food", "rent", "travel"]


def frame(rows):
    text, label = zip(*rows, strict=True)
    return data._frame(
        pd.Series(text),
        pd.Series(label),
        date=pd.date_range("2024-01-01", periods=len(rows), freq="D"),
    )


TRAIN = frame(
    [("TESCO STORES 1234", "food"), ("TESCO STORES 99", "rent"), ("Office rent March", "rent")]
    + [("SAINSBURYS", "food")] * 3
    + [("Rail ticket London", "travel")] * 2
)


@pytest.mark.parametrize("name", list(baselines.BASELINES))
def test_baselines_return_probabilities_over_all_labels(name):
    p = baselines.BASELINES[name](LABELS).fit(TRAIN).predict_proba(TRAIN)
    assert p.shape == (len(TRAIN), len(LABELS))
    assert np.allclose(p.sum(axis=1), 1)
    assert (p >= 0).all()


def test_majority_forecasts_training_frequencies():
    p = baselines.Majority(LABELS).fit(TRAIN).predict_proba(TRAIN.head(1))
    assert p[0] == pytest.approx([4 / 8, 2 / 8, 2 / 8])


def test_merchant_memory_uses_most_recent_label_then_prior():
    model = baselines.MerchantMemory(LABELS).fit(TRAIN)
    test = frame([("TESCO STORES 5", "food"), ("Unknown shop", "food")])
    p = model.predict_proba(test)
    assert p[0] == pytest.approx([0, 1, 0])  # tesco stores was last labelled rent
    assert p[1] == pytest.approx(model.prior)


def test_tfidf_lr_learns_text_and_skips_unseen_labels():
    labels = [*LABELS, "never_seen"]
    model = baselines.TfidfLR(labels).fit(TRAIN)
    p = model.predict_proba(frame([("SAINSBURYS LOCAL", "food"), ("Rail ticket", "travel")]))
    assert p.argmax(axis=1).tolist() == [0, 2]
    assert (p[:, 3] == 0).all()
