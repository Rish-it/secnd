"""Read published evaluations and validated workspace registry cards."""

import threading
from pathlib import Path

from backend.registry import Registry
from backend.schemas import EvalCard, EvalCurves, ModelVersion


class Evidence:
    def __init__(self, data_dir: Path, evaluation_dir: Path):
        self.data_dir, self.evaluation_dir = data_dir, evaluation_dir
        self._guard = threading.Lock()
        self._locks = {}

    def evaluation(self, dataset: str) -> tuple[EvalCard, EvalCurves]:
        stem = f"{dataset}-minilm+lr-b2000.json"
        card = EvalCard.model_validate_json((self.evaluation_dir / f"card-{stem}").read_text())
        curves = EvalCurves.model_validate_json(
            (self.evaluation_dir / f"curves-{stem}").read_text()
        )
        return card, curves

    def _lock(self, workspace):
        with self._guard:
            return self._locks.setdefault(workspace, threading.RLock())

    def models(self, workspace: str) -> list[ModelVersion]:
        with self._lock(workspace):
            registry = Registry(self.data_dir, workspace)
            current = registry.current()
            return [
                ModelVersion(
                    model_version=version,
                    current=version == current,
                    card=EvalCard.model_validate(registry.card(version)),
                )
                for version in registry.versions()
            ]

    def promote(self, workspace: str, version: str) -> ModelVersion:
        with self._lock(workspace):
            registry = Registry(self.data_dir, workspace)
            if version not in registry.versions():
                raise KeyError(version)
            # Validate before moving the live pointer; incomplete measurements stay unavailable.
            card = EvalCard.model_validate(registry.card(version))
            if card.model_version != version:
                raise ValueError("The registry card must identify its own version.")
            registry.promote(version)
            return ModelVersion(model_version=version, current=True, card=card)
