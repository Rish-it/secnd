"""Isolated browser fixture. Local stand-ins only; no weights, network, teachers or sends."""

import tempfile
from pathlib import Path
from threading import RLock

import uvicorn

from backend.api import create_app
from backend.config import ReaderOptions, Settings
from backend.reader_policy import ReaderArtifact
from backend.reader_serving import CPUReader
from backend.store import Store
from backend.tests.test_reader_serving import Embedder, TinyModel, Tokenizer, history
from backend.workspace import WorkspaceService


class DemoReader(CPUReader):
    def __init__(self, data_dir, options):
        self.options = options
        self.artifact = ReaderArtifact(options)
        self.policy, self.device = self.artifact.policy, options.device
        self._lock = RLock()
        self._history_key, self._vectors = None, None
        self.model, self.tok, self.embedder = TinyModel(), Tokenizer(), Embedder()


def main():
    with tempfile.TemporaryDirectory(prefix="secnd-reader-browser-") as directory:
        settings = Settings(
            data_dir=Path(directory),
            serving_backend="reader",
            workspace_readers={
                "gpu": ReaderOptions(model_id="ic-ettin-400m-s6000-near", device="cuda", examples=0)
            },
            cold_start_paid=False,
            whatsapp_send_enabled=False,
            _env_file=None,
        )
        store = Store(settings.database_path)
        for workspace in ("default", "gpu"):
            rows = [{**row, "merchant_key": ""} for row in history()]
            store.create_workspace(
                workspace,
                {"labels": ["a", "b"], "calib": [], "featurizer": "char_tfidf", "seed": 0},
                items=rows,
                labels=[(r["id"], r["label"]) for r in rows],
            )
        app = create_app(
            settings,
            workspace_service=WorkspaceService(store, settings.data_dir),
            reader_factory=DemoReader,
        )

        @app.middleware("http")
        async def mark_fixture(request, call_next):
            response = await call_next(request)
            response.headers["X-Secnd-Mode"] = "demo"
            return response

        uvicorn.run(app, host="127.0.0.1", port=8034, access_log=False)


if __name__ == "__main__":
    main()
