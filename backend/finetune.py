"""Fine-tune a sentence encoder on labelled rows, on the GPU when there is one.

Supervised contrastive loss over class-balanced batches (16 classes x 4 rows): rows with the same
label are pulled together, every other row in the batch is pushed away. The LR head on top is
unchanged, so only the encoder learns here and serving stays a CPU encode plus a linear layer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.features import EMBEDDERS, device

STEPS = 1000
CLASSES_PER_BATCH, ROWS_PER_CLASS = 16, 4
TEMPERATURE = 0.05
LEARNING_RATE = 2e-5


def tune(name: str, texts: list[str], y: np.ndarray, seed: int = 0, steps: int = STEPS):
    """A copy of encoder `name` tuned on (texts, y); classes with one row only act as negatives."""
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    repo, revision = EMBEDDERS[name]
    model = SentenceTransformer(repo, revision=revision, device=device())
    texts = np.asarray(texts, dtype=object)
    by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    anchors = [c for c, rows in by_class.items() if len(rows) >= 2]
    if not anchors:
        return release(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for _ in range(steps):
        classes = rng.choice(anchors, size=min(CLASSES_PER_BATCH, len(anchors)), replace=False)
        batch = np.concatenate(
            [
                rng.choice(by_class[c], size=min(ROWS_PER_CLASS, len(by_class[c])), replace=False)
                for c in classes
            ]
        )
        features = model.tokenize(texts[batch].tolist())
        features = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in features.items()}
        z = torch.nn.functional.normalize(model(features)["sentence_embedding"], dim=1)
        labels = torch.as_tensor(y[batch], device=model.device)
        own = torch.eye(len(batch), device=model.device, dtype=torch.bool)
        logits = (z @ z.T / TEMPERATURE).masked_fill(own, -1e9)
        positive = (labels[:, None] == labels[None, :]) & ~own
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -((log_prob * positive).sum(1) / positive.sum(1).clamp(min=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    del optimizer  # its moment buffers live on the GPU too
    return release(model)


def release(model):
    """Keep the tuned encoder on its device to embed, and hand the optimizer's memory back."""
    model = model.eval()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


PRETUNE_STEPS = 3000


def _norm(texts: pd.Series) -> pd.Series:
    return texts.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def pretune(name: str, steps: int = PRETUNE_STEPS) -> dict:
    """Tune encoder `name` on Meta's TOPv2 and save it as `pt-<name>`.

    Every TOPv2 request whose text also appears anywhere in MTOP is dropped first, so the encoder
    has seen none of MTOP's rows: a workspace's MTOP labels still count only in its own budget.
    """
    from backend.config import Settings
    from backend.data import load_dataset

    data_dir = Settings().data_dir
    top = load_dataset("topv2", data_dir)
    seen = set(_norm(load_dataset("mtop", data_dir)["text"]))
    keep = ~_norm(top["text"]).isin(seen)
    top = top[keep]
    y = pd.factorize(top["label"])[0]
    model = tune(name, top["text"].fillna("").tolist(), y, seed=0, steps=steps)
    model.save(str(data_dir / "encoders" / f"pt-{name}"))
    return {
        "encoder": f"pt-{name}",
        "base": EMBEDDERS[name][0],
        "rows": int(keep.sum()),
        "dropped_mtop_overlap": int((~keep).sum()),
        "intents": int(y.max() + 1),
        "steps": steps,
        "device": device(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretune", required=True, help="encoder to tune on TOPv2, e.g. bge")
    parser.add_argument("--steps", type=int, default=PRETUNE_STEPS)
    args = parser.parse_args()
    report = pretune(args.pretune, args.steps)
    out = Path("docs/results/s2") / f"pretune-{report['encoder']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
