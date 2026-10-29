import numpy as np
import pandas as pd
import pytest

from backend import model


def _frame(n=240, seed=0):
    """Payee decides the label; text alone is ambiguous between two labels."""
    rng = np.random.default_rng(seed)
    payee = rng.integers(0, 12, n)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "text": [f"invoice {'abc'[p % 3]} ref {i}" for i, p in enumerate(payee)],
            "amount": rng.normal(100, 20, n),
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "label": [f"class {p % 4}" for p in payee],
            "merchant_key": [f"payee {p}" for p in payee],
        }
    )


def test_prefix_memory_never_sees_the_rows_own_label():
    train = _frame(6).assign(merchant_key=["a", "b", "a", "", "a", "b"])
    index = {label: i for i, label in enumerate(sorted(train["label"].unique()))}
    block, memory = model.memory_block(train, index)
    block = block.toarray()
    assert block[[0, 1, 3]].sum() == 0  # first sightings and the empty key
    assert block[2, index[train.label[0]]] == 1
    assert block[4, index[train.label[2]]] == 1
    assert block[5, index[train.label[1]]] == 1
    assert memory == {"a": index[train.label[4]], "b": index[train.label[5]]}
    unseen = train.drop(columns="label").assign(merchant_key=["b", "c", "", "a", "a", "b"])
    block, _ = model.memory_block(unseen, index, memory)
    assert block.toarray().argmax(axis=1)[[0, 3]].tolist() == [memory["b"], memory["a"]]
    assert block[[1, 2]].sum() == 0


@pytest.mark.parametrize("classifier", model.CLASSIFIERS)
def test_models_return_probabilities_over_every_label(classifier):
    frame = _frame()
    labels = sorted(frame["label"].unique()) + ["never seen"]
    fitted = model.Model(labels, "char_tfidf", classifier).fit(frame.head(200))
    p = fitted.predict_proba(frame.tail(40))
    assert p.shape == (40, len(labels))
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert (p[:, -1] == 0).all()


def test_merchant_memory_lifts_an_ambiguous_text_model():
    frame = _frame()
    labels = sorted(frame["label"].unique())
    fitted = model.Model(labels, "char_tfidf", "lr").fit(frame.head(200))
    test = frame.tail(40)
    accuracy = (fitted.predict_proba(test).argmax(axis=1) == fitted.encode(test)).mean()
    assert accuracy > 0.9


def test_tune_c_is_deterministic_and_on_the_grid():
    frame = _frame()
    labels = sorted(frame["label"].unique())
    fitted = model.Model(labels, "char_tfidf", "lr", seed=3).fit(frame)
    again = model.Model(labels, "char_tfidf", "lr", seed=3).fit(frame)
    assert fitted.c == again.c and fitted.c in model.C_GRID
    assert model.Model(labels, "char_tfidf", "lr", c=0.5).fit(frame).c == 0.5
