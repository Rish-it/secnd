import numpy as np

from backend import finetune


def _margin(model, texts, y):
    z = model.encode(texts, normalize_embeddings=True)
    sim = z @ z.T
    same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool)
    return sim[same].mean() - sim[y[:, None] != y[None, :]].mean()


def test_tuning_pulls_same_label_rows_together():
    # Labels cut across topic, so the pretrained encoder cannot already separate them.
    texts = [
        f"{topic} {tag}"
        for tag in ("ticket", "refund")
        for topic in ("train", "hotel", "taxi", "flight")
    ]
    y = np.array([0] * 4 + [1] * 4)
    before = _margin(finetune.tune("minilm", texts, y, steps=0), texts, y)
    after = _margin(finetune.tune("minilm", texts, y, steps=30), texts, y)
    assert after > before + 0.1
