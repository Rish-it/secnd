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


def test_tuned_encoder_is_returned_on_cpu():
    model = finetune.tune("minilm", ["a b", "a c", "d e", "d f"], np.array([0, 0, 1, 1]), steps=1)
    assert str(model.device) == "cpu"


def test_pretune_drops_every_mtop_text_before_tuning(tmp_path, monkeypatch):
    import pandas as pd

    from backend import data, finetune

    frames = {
        "topv2": pd.DataFrame(
            {
                "text": ["Set an alarm!", "weather today", "play jazz"],
                "label": ["IN:A", "IN:B", "IN:C"],
            }
        ),
        "mtop": pd.DataFrame({"text": ["set an ALARM"], "label": ["IN:A"]}),
    }
    seen = {}

    class Saved:
        def save(self, path):
            seen["path"] = path

    def tune(name, texts, y, seed, steps):
        seen["texts"] = texts
        return Saved()

    monkeypatch.setenv("SECND_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data, "load_dataset", lambda name, _: frames[name])
    monkeypatch.setattr(finetune, "tune", tune)
    report = finetune.pretune("minilm", steps=1)
    assert seen["texts"] == ["weather today", "play jazz"]
    assert report["dropped_mtop_overlap"] == 1 and report["rows"] == 2
    assert seen["path"].endswith("encoders/pt-minilm")
