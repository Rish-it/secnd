"""Local browser rehearsal: simulated Meta responses and the real CPU reader; no paid calls."""

import argparse
import json
from pathlib import Path

import httpx
import uvicorn

from backend.api import create_app
from backend.cold_start import ColdStartService
from backend.config import Settings
from backend.store import Store
from backend.workspace import WorkspaceService


class DemoHistory(ColdStartService):
    @property
    def ledger(self):
        return self.settings.data_dir / "spend-demo.csv"


def create_demo():
    directory = Path(".data/cold-start-demo")
    directory.mkdir(parents=True, exist_ok=True)
    encoders = directory / "encoders"
    if not encoders.exists():
        encoders.symlink_to(Path(".data/encoders").resolve(), target_is_directory=True)
    settings = Settings(data_dir=directory, meta_api_key="demo-only", _env_file=None)
    messages = {
        "Travel": [
            "I need a train ticket to the client office.",
            "Can I book a flight for the meeting?",
            "The taxi fare was for our business trip.",
            "Please reimburse my train journey.",
            "I booked a hotel for the conference.",
        ],
        "Software": [
            "Please renew our design tool subscription.",
            "The team needs a cloud hosting account.",
            "Can we buy a licence for the editor?",
            "Our monthly software bill is due.",
            "I need access to the project management app.",
        ],
    }

    def respond(request):
        payload = json.loads(request.content)
        label = json.loads(payload["messages"][1]["content"])["label"]
        examples = messages.get(label, [f"An illustrative message for {label}."])
        return httpx.Response(
            200,
            json={
                "model": "muse-wonder-1.0",
                "choices": [
                    {"message": {"content": json.dumps({"messages": examples + examples})}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100},
            },
        )

    store = Store(settings.database_path)
    cold = DemoHistory(store, settings, transport=httpx.MockTransport(respond), demo=True)
    return create_app(
        settings, workspace_service=WorkspaceService(store, directory), cold_start_service=cold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8028)
    args = parser.parse_args()
    uvicorn.run(create_demo(), host="127.0.0.1", port=args.port, access_log=False)
