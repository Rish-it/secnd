"""Reference models every real model must beat; each returns (n, len(labels)) probabilities."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Baseline:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.index = {label: i for i, label in enumerate(labels)}

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["label"].map(self.index).to_numpy()

    def fit(self, train: pd.DataFrame) -> "Baseline":
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full((len(frame), len(self.labels)), 1 / len(self.labels))


class Majority(Baseline):
    """Always forecasts the training label frequencies."""

    def fit(self, train):
        counts = np.bincount(self.encode(train), minlength=len(self.labels))
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, frame):
        return np.tile(self.prior, (len(frame), 1))


class MerchantMemory(Majority):
    """A known merchant key gets its most recent label; anything else gets the prior."""

    def fit(self, train):
        super().fit(train)
        known = train[train["merchant_key"] != ""].sort_values(["date", "id"])
        self.memory = known.groupby("merchant_key")["label"].last().map(self.index).to_dict()
        return self

    def predict_proba(self, frame):
        p = super().predict_proba(frame)
        for row, key in enumerate(frame["merchant_key"]):
            if key in self.memory:
                p[row] = 0
                p[row, self.memory[key]] = 1
        return p


class TfidfLR(Baseline):
    """Character n-gram TF-IDF and multinomial logistic regression."""

    def fit(self, train):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.model = LogisticRegression(max_iter=2000)
        self.model.fit(self.vectorizer.fit_transform(train["text"]), self.encode(train))
        return self

    def predict_proba(self, frame):
        p = np.zeros((len(frame), len(self.labels)))
        p[:, self.model.classes_] = self.model.predict_proba(
            self.vectorizer.transform(frame["text"])
        )
        return p


BASELINES = {
    "random": Baseline,
    "majority": Majority,
    "merchant_memory": MerchantMemory,
    "tfidf_lr": TfidfLR,
}
