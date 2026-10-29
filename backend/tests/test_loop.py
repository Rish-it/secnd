import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from backend import loop, registry
from backend.data import _frame
from backend.schemas import Decision, EvalCard

K, DIM = 6, 32
CENTRES = np.random.default_rng(42).normal(0, 1, (K, DIM))


def fake_featurize(frame, featurizer, cache_dir=None, encoder=None):
    """Class centre plus noise, read from the text: rows are separable but not trivially."""
    rows = []
    for text in frame["text"]:
        k, i = (int(part) for part in text.split()[1::2])
        noise = np.random.default_rng(i).normal(0, 0.8, DIM)
        rows.append(CENTRES[k] + noise)
    return sparse.csr_matrix(np.array(rows, dtype=np.float32))


def rows(n, start=0, shuffle=False):
    k = np.arange(start, start + n) % K
    frame = _frame(
        pd.Series([f"class {c} item {i}" for i, c in zip(range(start, start + n), k, strict=True)]),
        pd.Series([f"label {c}" for c in k]),
    )
    if shuffle:
        frame["label"] = frame["label"].sample(frac=1, random_state=0).to_numpy()
    return frame


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(loop, "featurize", fake_featurize)


def _loop(**kwargs):
    labels = [f"label {c}" for c in range(K)]
    return loop.Loop(labels, rows(3000, start=100_000), featurizer="fake", **kwargs)


def test_first_retrain_promotes_and_routes_valid_decisions(fake):
    ren = _loop()
    train = rows(600)
    ren.learn(train, train["label"], "seed")
    result = ren.retrain()
    assert result["promoted"] and result["model_version"] == "ren-1.0"
    decisions = ren.route(rows(200, start=50_000))
    for d in decisions:
        Decision.model_validate(d)
    assert any(d["auto"] for d in decisions)


def test_a_bad_version_is_refused_and_rollback_restores_the_previous(fake, tmp_path):
    models = registry.Registry(tmp_path, "default")
    ren = _loop(models=models)
    good = rows(600)
    ren.learn(good, good["label"], "seed")
    assert ren.retrain()["model_version"] == "ren-1.0"
    card = EvalCard.model_validate(models.card("ren-1.0"))  # the API serves this card
    assert card.model_version == "ren-1.0" and card.coverage > 0 and card.precision >= 0.99

    bad = rows(6000, start=10_000, shuffle=True)  # swamps the good rows with shuffled labels
    ren.learn(bad, bad["label"], "answer")
    result = ren.retrain()
    assert not result["promoted"] and result["reasons"]
    assert ren.current["version"] == "ren-1.0" and models.current() == "ren-1.0"

    models.save({"weights": "manual"}, {})
    models.promote("ren-1.1")
    models.promote("ren-1.0")  # a rollback is a promote of an older version
    assert models.current() == "ren-1.0"


def test_spot_checks_take_automatic_answers_to_a_person(fake):
    ren = _loop(spot_check_rate=1.0)
    train = rows(600)
    ren.learn(train, train["label"], "seed")
    ren.retrain()
    decisions = ren.route(rows(200, start=50_000))
    assert not any(d["auto"] for d in decisions)
    assert any(d["spot_check"] for d in decisions)


def test_wrong_spot_checks_stop_automatic_answers_until_a_new_version(fake):
    ren = _loop(spot_check_rate=1.0)
    train = rows(600)
    ren.learn(train, train["label"], "seed")
    ren.retrain()
    stream = rows(400, start=50_000)
    ren.route(stream)
    items = stream[stream["id"].astype(str).isin(ren.pending)].iloc[:105]
    for item in items["id"].astype(str)[100:]:  # Ren predicted these wrong; the person corrects it
        ren.pending[item] = next(x for x in ren.labels if x != ren.pending[item])
    for item, label in zip(items["id"].astype(str)[:100], items["label"][:100], strict=True):
        ren.pending[item] = label  # and these right
    ren.learn(items.iloc[:101], items["label"][:101], "spot_check")  # 1 error in 101
    assert not ren.guarded
    ren.learn(items.iloc[101:], items["label"][101:], "spot_check")  # 5 errors in 105: below 99%
    assert ren.guarded and ren.checks == [100, 105]
    assert all(d["threshold"] is None and not d["auto"] for d in ren.route(stream))
    more = rows(30, start=200_000)  # (fake rows saturate to one confidence past ~300 extra)
    ren.learn(more, more["label"], "answer")
    result = ren.retrain()
    assert result["promoted"], result["reasons"]
    assert not ren.guarded  # a new version starts clean


def test_cold_start_labels_weigh_less_than_answers(fake):
    ren = _loop()
    first = rows(2)
    ren.learn(first.iloc[[0]], ["label 0"], "muse_cold_start")
    ren.learn(first.iloc[[1]], ["label 1"], "answer")
    assert ren.weights == [loop.COLD_START_WEIGHT, 1.0] and loop.COLD_START_WEIGHT < 1
    with pytest.raises(ValueError):
        ren.learn(first.iloc[[0]], ["label 0"], "guess")


def test_retrain_at_10k_rows_is_under_a_second(fake):
    ren = _loop()
    train = rows(10_000)
    ren.learn(train, train["label"], "seed")
    ren.retrain()
    times = []
    for i in range(20):
        item = rows(1, start=200_000 + i)
        started = time.perf_counter()
        ren.on_answer(item, item["label"].iloc[0])
        times.append(time.perf_counter() - started)
    assert np.percentile(times, 95) < 1.0
