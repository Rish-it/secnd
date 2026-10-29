"""Featurizer x classifier models with the same fit / predict_proba shape as the baselines.

Merchant memory enters as evidence, not as an override: on ipsa the latest label of a known payee
is right only 88.5% of the time, so the classifier learns how far to trust it per class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from backend.baselines import Baseline
from backend.features import featurize

CLASSIFIERS = ("lr", "sgd", "knn")
C_GRID = (0.1, 1.0, 10.0)
TUNE_ROWS = 5000
LR = {"tol": 1e-3, "max_iter": 300}  # same test accuracy as the defaults on ipsa, 20x faster


def memory_block(train: pd.DataFrame, frame: pd.DataFrame, index: dict, prefix: bool):
    """One-hot of each row's payee's latest training label.

    With `prefix`, `frame` is `train` itself and each row only sees rows before it in (date, id)
    order, so a training row never sees its own label.
    """
    rows, cols = [], []
    if prefix:
        memory = {}
        ordered = frame.reset_index(drop=True).sort_values(["date", "id"], na_position="first")
        for row, key, label in zip(
            ordered.index, ordered["merchant_key"], ordered["label"], strict=True
        ):
            if key in memory:
                rows.append(row)
                cols.append(memory[key])
            if key:
                memory[key] = index[label]
    else:
        known = train[train["merchant_key"] != ""].sort_values(["date", "id"])
        memory = known.groupby("merchant_key")["label"].last().map(index).to_dict()
        for row, key in enumerate(frame["merchant_key"]):
            if key in memory:
                rows.append(row)
                cols.append(memory[key])
    return sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(frame), len(index))
    )


def tune_c(x, y, seed: int) -> float:
    """C with the lowest 3-fold log loss on at most TUNE_ROWS training rows."""
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(y))[:TUNE_ROWS]
    x, y = x[pick], y[pick]
    if len(y) < 6:
        return 1.0
    labels = np.unique(y)
    losses = []
    for c in C_GRID:
        loss = 0.0
        for fit, held in KFold(3, shuffle=True, random_state=seed).split(y):
            model = LogisticRegression(C=c, **LR).fit(x[fit], y[fit])
            p = np.full((len(held), len(labels)), 1e-9)
            p[:, np.searchsorted(labels, model.classes_)] = model.predict_proba(x[held])
            loss += log_loss(y[held], p / p.sum(axis=1, keepdims=True), labels=labels)
        losses.append(loss)
    return C_GRID[int(np.argmin(losses))]


class Model(Baseline):
    def __init__(
        self,
        labels: list[str],
        featurizer: str = "potion",
        classifier: str = "lr",
        seed: int = 0,
        cache_dir: Path | None = None,
    ):
        super().__init__(labels)
        self.featurizer, self.classifier, self.seed, self.cache_dir = (
            featurizer,
            classifier,
            seed,
            cache_dir,
        )
        self.id = f"{featurizer}+{classifier}"

    def _x(self, frame, prefix=False):
        memory = memory_block(self.train, frame, self.index, prefix)
        x = sparse.hstack([featurize(frame, self.featurizer, self.cache_dir), memory], format="csr")
        return x if prefix else x[:, self.columns]

    def fit(self, train):
        self.train = train
        x, y = self._x(train, prefix=True), self.encode(train)
        # Hashed columns no training row uses carry no weight; dropping them makes LR fast.
        self.columns = np.unique(x.indices)
        x = x[:, self.columns]
        if self.classifier == "lr":
            self.c = tune_c(x, y, self.seed)
            self.model = LogisticRegression(C=self.c, **LR)
        elif self.classifier == "sgd":
            # ponytail: one-vs-rest probabilities, renormalised; LR is the one-softmax option (R5)
            self.model = SGDClassifier(loss="log_loss", random_state=self.seed)
        else:
            self.model = KNeighborsClassifier(
                n_neighbors=min(10, len(y)), weights="distance", metric="cosine"
            )
        self.model.fit(x, y)
        return self

    def predict_proba(self, frame):
        p = np.zeros((len(frame), len(self.labels)))
        p[:, self.model.classes_] = self.model.predict_proba(self._x(frame))
        return p
