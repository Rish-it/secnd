import numpy as np
import pandas as pd
import pytest

from backend import features


def _frame():
    return pd.DataFrame(
        {
            "text": ["Train fare London", "Office rent", None],
            "amount": [86.0, -350.5, None],
            "date": pd.to_datetime(["2024-10-07", "2024-10-31", None]),
        }
    )


def test_numeric_block_is_hand_checkable():
    block = features.numeric(_frame())
    assert block.shape == (3, 13)
    log86 = np.log1p(86)
    np.testing.assert_allclose(block[0, :6], [log86, log86, 1, 1, 1, 0.2], rtol=1e-6)
    np.testing.assert_allclose(block[1, :6], [-np.log1p(350.5), np.log1p(350.5), -1, 1, 1, 1])
    assert block[0, 6:].tolist() == [1, 0, 0, 0, 0, 0, 0]  # Monday
    assert block[1, 6:].tolist() == [0, 0, 0, 1, 0, 0, 0]  # Thursday
    assert block[2].tolist() == [0] * 13
    np.testing.assert_array_equal(block, features.numeric(_frame()))


def test_char_tfidf_rows_do_not_depend_on_the_batch():
    frame = _frame()
    alone = features.featurize(frame.iloc[[1]], "char_tfidf")
    together = features.featurize(frame, "char_tfidf")
    assert (alone != together[1]).nnz == 0


def test_embedding_cache_hit_returns_identical_vectors(tmp_path, monkeypatch):
    calls = []

    def fake_encode(name, texts):
        calls.append(list(texts))
        return np.random.default_rng(len(calls)).normal(size=(len(texts), 4))

    monkeypatch.setattr(features, "_encode", fake_encode)
    texts = pd.Series(["a", "b", "a"])
    first = features.embed("potion", texts, tmp_path)
    second = features.embed("potion", texts, tmp_path)

    assert calls == [["a", "b"]]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], first[2])
    features.embed("minilm", texts, tmp_path)
    assert len(calls) == 2


@pytest.mark.parametrize("featurizer", ["potion", "stack"])
def test_featurize_appends_numeric_block(featurizer, tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_encode", lambda name, texts: np.ones((len(texts), 4)))
    matrix = features.featurize(_frame(), featurizer, tmp_path)
    text_width = 4 if featurizer == "potion" else 2**18 + 4
    assert matrix.shape == (3, text_width + 13)
    np.testing.assert_array_equal(matrix[:, -13:].toarray(), features.numeric(_frame()))
