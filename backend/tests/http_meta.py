"""Exercise a local Meta callback without sending replies or calling a teacher."""

import argparse
import hashlib
import hmac
import json
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from backend.config import Settings
from backend.tests.test_live_api import csv_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8022")
    args = parser.parse_args()
    settings = Settings()
    secret = settings.meta_app_secret.get_secret_value()
    token = settings.meta_verify_token.get_secret_value()
    assert secret and token, "Configure the Meta app secret and verification token."
    api = args.base_url.rstrip("/") + "/api/v1/workspaces/meta-" + uuid.uuid4().hex[:12]
    with httpx.Client(timeout=120) as client:
        imported = client.post(
            api + "/decisions:import",
            files={"file": ("history.csv", csv_rows(), "text/csv")},
            data={"target_column": "category"},
        )
        assert imported.status_code == 200, "Import failed."
        response = client.get(
            api + "/integrations/meta/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": token,
                "hub.challenge": "1158201444",
            },
        )
        assert response.status_code == 200 and response.text == "1158201444"
        fixtures = Path(__file__).parent / "fixtures/meta"
        results = []
        for platform in ("whatsapp", "messenger"):
            raw = (fixtures / f"{platform}-text.json").read_bytes()
            signature = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
            headers = {"Content-Type": "application/json", "X-Hub-Signature-256": signature}
            first = client.post(api + "/integrations/meta/webhook", content=raw, headers=headers)
            assert first.status_code == 200
            result = first.json()["items"][0]
            assert result["status"] == "queued", "Demo requires an uncertified illustrative model."
            duplicate = client.post(
                api + "/integrations/meta/webhook", content=raw, headers=headers
            )
            assert duplicate.status_code == 200
            assert duplicate.json()["items"][0]["duplicate"] is True
            answered = client.post(
                api + "/answers",
                json={"item_id": result["item_id"], "label": "Food"},
                headers={"Idempotency-Key": "human-" + platform},
            )
            assert answered.status_code == 200 and answered.json()["retrain"] == "queued"
            results.append({"platform": platform, "status": result["status"], "duplicate": True})
        assert client.get(api + "/queue").json()["items"] == []
        deadline = time.monotonic() + 30
        while len(client.get(api + "/models").json()["items"]) < 2:
            assert time.monotonic() < deadline, "Background retrain did not finish."
            time.sleep(0.1)
        print(
            json.dumps(
                {
                    "source_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "scope": "Illustrative local integration, not benchmark quality.",
                    "featurizer": imported.json()["card"]["featurizer_id"],
                    "handshake": True,
                    "messages": results,
                    "answered_queue_empty": True,
                    "background_retrain": True,
                    "outbound_sends": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
