"""Fine-tune a sentence encoder on labelled rows, on the GPU when there is one.

Supervised contrastive loss over class-balanced batches (16 classes x 4 rows): rows with the same
label are pulled together, every other row in the batch is pushed away. The LR head on top is
unchanged, so only the encoder learns here and serving stays a CPU encode plus a linear layer.
"""

import numpy as np
import torch

from backend.features import EMBEDDERS

STEPS = 1000
CLASSES_PER_BATCH, ROWS_PER_CLASS = 16, 4
TEMPERATURE = 0.05
LEARNING_RATE = 2e-5


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
        return model.eval()
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
    return model.eval()
