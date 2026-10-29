"""Promotion gate and the on-disk version registry of a workspace's models.

Layout: <data_dir>/workspaces/<ws>/models/<version>/{model.pkl, card.json} plus a `current` file
naming the live version, replaced atomically.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

from backend import metrics
from backend.calibrate import TARGET

PREFIX = "ren-1."
MAX_REGRESSION = 0.01


def collapsed(p: np.ndarray) -> bool:
    """A model that always predicts one class, or gives every row the same confidence."""
    return len(np.unique(p.argmax(axis=1))) < 2 or np.ptp(p.max(axis=1)) < 1e-6


def gate(
    y_regression, p_candidate, p_current=None, tau=None, calib_lower=0.0, current_tau=None
) -> list[str]:
    """Reasons to refuse the candidate; an empty list means it may be promoted."""
    reasons = []
    if tau is not None and calib_lower < TARGET:
        reasons.append(f"calib lower bound {calib_lower:.4f} is below {TARGET}")
    if collapsed(p_candidate):
        reasons.append("predictions collapsed")
    if p_current is not None:
        drop = (p_current.argmax(axis=1) == y_regression).mean() - (
            p_candidate.argmax(axis=1) == y_regression
        ).mean()
        if drop > MAX_REGRESSION:
            reasons.append(f"regression set accuracy drops {drop:.3f}")
        if current_tau is not None:  # coverage ratchet: never trade certified coverage away
            lost = metrics.coverage_at(p_current, current_tau) - (
                0.0 if tau is None else metrics.coverage_at(p_candidate, tau)
            )
            if lost > MAX_REGRESSION:
                reasons.append(f"certified coverage drops {lost:.3f}")
    return reasons


class Registry:
    def __init__(self, data_dir: Path, workspace: str):
        self.root = Path(data_dir) / "workspaces" / workspace / "models"

    def versions(self) -> list[str]:
        if not self.root.exists():
            return []
        found = [d.name for d in self.root.iterdir() if d.name.startswith(PREFIX) and d.is_dir()]
        return sorted(found, key=lambda v: int(v.removeprefix(PREFIX)))

    def save(self, model, card: dict) -> str:
        versions = self.versions()
        version = PREFIX + str(int(versions[-1].removeprefix(PREFIX)) + 1 if versions else 0)
        folder = self.root / version
        folder.mkdir(parents=True)
        (folder / "model.pkl").write_bytes(pickle.dumps(model))
        (folder / "card.json").write_text(json.dumps({**card, "model_version": version}, indent=2))
        return version

    def load(self, version: str):
        return pickle.loads((self.root / version / "model.pkl").read_bytes())

    def card(self, version: str) -> dict:
        return json.loads((self.root / version / "card.json").read_text())

    def current(self) -> str | None:
        pointer = self.root / "current"
        return pointer.read_text().strip() if pointer.exists() else None

    def promote(self, version: str):
        """Point `current` at `version`; a rollback is a promote of an older version."""
        if version not in self.versions():
            raise KeyError(version)
        staged = self.root / "current.tmp"
        staged.write_text(version)
        os.replace(staged, self.root / "current")
