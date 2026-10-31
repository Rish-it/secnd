"""Isolated synthetic inbox for browser checks, with actual CPU inference and no providers."""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import uvicorn

from backend.api import create_app
from backend.config import Settings
from backend.loop import Loop
from backend.store import Store
from backend.workspace import WorkspaceService

TEXTS = [
    "Please reply with the price and appointment availability",
    "Urgent account access problem escalate to support specialist",
    "Newsletter weekly update archive no response needed",
]
LABELS = ["Reply", "Escalate", "Archive"]


def rows(start, count, *, calibration=False):
    return pd.DataFrame(
        [
            dict(
                id=f"past-{i}",
                text=("Hello, a question" if calibration and i % 10 == 0 else TEXTS[i % 3])
                + f" reference {i % 29}",
                label=LABELS[i % 3],
                amount=None,
                date=None,
                merchant_key="",
            )
            for i in range(start, start + count)
        ]
    )


def create_demo(directory):
    settings = Settings(data_dir=directory, _env_file=None, demo_mode=True, featurizer="char_tfidf")
    service = WorkspaceService(
        Store(settings.database_path),
        directory,
        factory=lambda *args, **kwargs: Loop(*args, **kwargs, spot_check_rate=0),
    )
    service.create(
        "default",
        LABELS,
        rows(10000, 900, calibration=True),
        training=rows(0, 180),
        featurizer="char_tfidf",
    )
    incoming = []
    for i, source in enumerate(["whatsapp", "messenger", "instagram", "ads"]):
        incoming.extend(
            [
                dict(
                    id=f"pending-{i}",
                    text="Hello, a question",
                    amount=None,
                    date=None,
                    merchant_key="",
                    source_app=source,
                ),
                dict(
                    id=f"auto-{i}",
                    text=TEXTS[i % 3] + " reference 1",
                    amount=None,
                    date=None,
                    merchant_key="",
                    source_app=source,
                ),
            ]
        )
    service.route("default", pd.DataFrame(incoming))
    # Second workspace proves browser navigation scopes both queues and answers.
    service.create(
        "other",
        LABELS,
        rows(12000, 900, calibration=True),
        training=rows(200, 180),
        featurizer="char_tfidf",
    )
    service.route(
        "other",
        pd.DataFrame(
            [
                dict(
                    id="other-question",
                    text="Hello, a question",
                    amount=None,
                    date=None,
                    merchant_key="",
                    source_app="instagram",
                )
            ]
        ),
    )
    app = create_app(settings, workspace_service=service)

    @app.middleware("http")
    async def mark_demo(request, call_next):
        response = await call_next(request)
        response.headers["X-Secnd-Demo"] = "synthetic"
        return response

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8031)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="secnd-inbox-") as folder:
        uvicorn.run(create_demo(Path(folder)), host="127.0.0.1", port=args.port, access_log=False)
