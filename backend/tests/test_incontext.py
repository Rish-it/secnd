import numpy as np
import torch

from backend import incontext as ic


def _task():
    train = [(f"row {i}", i % 3) for i in range(30)]
    vecs = torch.nn.functional.normalize(
        torch.eye(3)[[y for _, y in train]] + 0.01 * torch.randn(30, 3), dim=1
    )
    return ic.Task("t", "q?", ["a", "b", "c"], train, [("x", 0), ("y", 2)], vecs, vecs[[0, 2]])


def test_labels_read_as_plain_words():
    assert ic.humanize("IN:GET_WEATHER") == "get weather"
    assert ic.humanize("atis_flight+atis_airfare") == "flight or airfare"


def test_nearest_examples_share_the_query_label_and_skip_the_query():
    task = _task()
    rows = ic.nearest(task, task.test_vecs[1], 4, skip="row 2")
    assert len(rows) == 4 and all(y == 2 for _, y in rows) and ("row 2", 2) not in rows


def test_episode_puts_the_gold_label_among_the_options():
    task = _task()
    rng = np.random.default_rng(0)
    for _ in range(20):
        text, options, slot, shots = ic.episode(task, rng, 4)
        assert options[slot] == dict(task.train)[text]
        assert all(y in options and t != text for t, y in shots)


def test_capped_history_keeps_rows_and_vectors_aligned():
    task = _task()
    small = ic.capped(task, 10)
    assert len(small.train) == len(small.train_vecs) == 10
    assert ic.capped(task, None) is task
