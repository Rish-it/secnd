"""The learning loop: route each item, learn from each answer, change the live model only through
the gate.

One workspace's loop keeps its training rows as features, a payee memory, a fixed calibration set
and the current version. `learn` appends answers; `retrain` refits the logistic-regression head
(warm-started from the current one), recalibrates on the fixed calib set only, runs the gate and
promotes the candidate or keeps the current version. `route` answers automatically only above the
current threshold, and sends a random share of those answers to a person as spot-checks.
"""

import copy
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from backend import calibrate, registry
from backend.features import featurize
from backend.model import LR, memory_block, tune_c

SPOT_CHECK_RATE = 0.05
TOP_K = 3
SOURCES = ("seed", "answer", "spot_check", "muse_cold_start")
COLD_START_WEIGHT = 0.3  # a Muse label counts for less than any answer from a person


class Loop:
    def __init__(
        self,
        labels: list[str],
        calib: pd.DataFrame,
        featurizer: str = "minilm",
        seed: int = 0,
        cache_dir: Path | None = None,
        models: registry.Registry | None = None,
        spot_check_rate: float = SPOT_CHECK_RATE,
    ):
        self.labels = list(labels)
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.featurizer, self.seed, self.cache_dir, self.models = (
            featurizer,
            seed,
            cache_dir,
            models,
        )
        self.spot_check_rate = spot_check_rate
        self.rng = np.random.default_rng(seed)
        self.calib = calib
        self.calib_y = calib["label"].map(self.index).to_numpy()
        self.calib_text = featurize(calib, featurizer, cache_dir)
        self.rows, self.y, self.weights, self.memory = [], [], [], {}
        self.c = None
        self.current = None  # {"model", "temperature", "tau", "lower", "version"}
        self.versions = 0
        # ponytail: one lock per workspace serialises retrains; a queue per workspace if answers
        # arrive faster than a retrain (~0.3 s at 10k rows) finishes.
        self.lock = threading.Lock()

    def _x(self, frame: pd.DataFrame, text=None):
        block, _ = memory_block(frame, self.index, self.memory)
        if text is None:
            text = featurize(frame, self.featurizer, self.cache_dir)
        return sparse.hstack([text, block], format="csr")

    def _proba(self, model, x, temperature):
        p = np.zeros((x.shape[0], len(self.labels)))
        p[:, model.classes_] = model.predict_proba(x)
        return calibrate.scale(p, temperature)

    def learn(self, frame: pd.DataFrame, labels, source: str = "answer"):
        """Append answered rows in arrival order; each row sees the payee memory before itself."""
        if source not in SOURCES:
            raise ValueError(f"unknown label source {source!r}")
        weight = COLD_START_WEIGHT if source == "muse_cold_start" else 1.0
        frame = frame.assign(label=list(labels))
        text = featurize(frame, self.featurizer, self.cache_dir)
        for i, (key, label) in enumerate(zip(frame["merchant_key"], frame["label"], strict=True)):
            block = np.zeros(len(self.labels), np.float32)
            if key in self.memory:
                block[self.memory[key]] = 1
            self.rows.append(sparse.hstack([text[i], sparse.csr_matrix(block)], format="csr"))
            self.y.append(self.index[label])
            self.weights.append(weight)
            if key:
                self.memory[key] = self.index[label]

    def retrain(self) -> dict:
        """Refit, recalibrate on the fixed calib set, gate; promote or keep the current version."""
        with self.lock:
            started = time.perf_counter()
            x, y, w = (
                sparse.vstack(self.rows, format="csr"),
                np.array(self.y),
                np.array(self.weights),
            )
            if self.c is None:
                self.c = tune_c(x, y, self.seed)
            previous = self.current["model"] if self.current else None
            if previous is not None and set(previous.classes_) == set(np.unique(y)):
                model = copy.deepcopy(previous)  # warm start without touching the live version
            else:
                model = LogisticRegression(C=self.c, warm_start=True, **LR)
            model.fit(x, y, sample_weight=w)

            calib_x = self._x(self.calib, self.calib_text)
            raw = self._proba(model, calib_x, 1.0)
            temperature = calibrate.fit_temperature(self.calib_y, raw)
            p = calibrate.scale(raw, temperature)
            tau, _, _, lower = calibrate.choose_threshold(self.calib_y, p)
            current = None
            if self.current:
                current = self._proba(self.current["model"], calib_x, self.current["temperature"])
            reasons = registry.gate(self.calib_y, p, current, tau, lower)
            promoted = not reasons
            if promoted:
                candidate = {"model": model, "temperature": temperature, "tau": tau, "lower": lower}
                if self.models:
                    version = self.models.save(
                        candidate, {"threshold": tau, "cp_lower_bound": lower}
                    )
                    self.models.promote(version)
                else:
                    version = f"{registry.PREFIX}{self.versions}"
                self.versions += 1
                self.current = {**candidate, "version": version}
            return {
                "promoted": promoted,
                "reasons": reasons,
                "model_version": self.current["version"] if self.current else None,
                "threshold": self.current["tau"] if self.current else None,
                "labels": len(y),
                "retrain_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    def on_answer(self, item: pd.DataFrame, label: str, source: str = "answer") -> dict:
        self.learn(item, [label], source)
        return self.retrain()

    def route(self, frame: pd.DataFrame) -> list[dict]:
        """One Decision (API schema) per row; automatic only at or above the current threshold."""
        if self.current is None:
            raise RuntimeError("no model yet: learn from the seed labels and retrain first")
        p = self._proba(self.current["model"], self._x(frame), self.current["temperature"])
        tau = self.current["tau"]
        decisions = []
        for row, probs in zip(frame.itertuples(), p, strict=True):
            top = np.argsort(-probs, kind="stable")[:TOP_K]
            auto = tau is not None and probs[top[0]] >= tau
            spot_check = bool(auto and self.rng.random() < self.spot_check_rate)
            decisions.append(
                {
                    "item_id": str(row.id),
                    "label": self.labels[top[0]],
                    "probabilities": {self.labels[i]: float(probs[i]) for i in top},
                    "prob": float(probs[top[0]]),
                    "auto": bool(auto and not spot_check),
                    "threshold": tau,
                    "model_version": self.current["version"],
                    "reasons": {"merchant_memory": row.merchant_key in self.memory},
                    "spot_check": spot_check,
                }
            )
        return decisions
