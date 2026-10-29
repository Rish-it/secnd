"""Execute the preregistered S3R study on Modal's existing benchmark volume."""

import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .run_commands(
        "uv pip install --python /.uv/.venv/bin/python --reinstall-package torch "
        "--index-url https://pypi.org/simple torch==2.15.0"
    )
    .env({"SECND_DATA_DIR": "/vol/data", "HF_HOME": "/vol/hf"})
    .add_local_python_source("backend", copy=True)
    .add_local_file(str(ROOT / "docs/results/s3r/paraphrases.json"), "/paraphrases.json", copy=True)
)
volume = modal.Volume.from_name("secnd-bench")
app = modal.App("secnd-s3r-fixed-history")


@app.function(image=image, gpu="A10G", cpu=4, memory=16384, volumes={"/vol": volume}, timeout=1800)
def audit():
    from backend.config import Settings
    from backend.red_team import run

    return json.dumps(run(Settings().data_dir, json.loads(Path("/paraphrases.json").read_text())))


@app.local_entrypoint()
def main():
    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Capture local provenance before dispatch; return only JSON primitive types.
    provenance = {
        "protocol_sha256": file_hash(ROOT / "docs/results/s3r/protocol.md"),
        "runner_sha256": file_hash(ROOT / "backend/red_team.py"),
        "modal_runner_sha256": file_hash(ROOT / "backend/modal_red_team.py"),
        "paraphrases_sha256": file_hash(ROOT / "docs/results/s3r/paraphrases.json"),
    }
    started = time.time()
    call = audit.spawn()
    print(f"Modal call: {call.object_id}", flush=True)
    result = json.loads(call.get())
    result.update(provenance)
    result["modal"] = {
        "app_id": app.app_id,
        "function_call_id": call.object_id,
        "client_elapsed_seconds": time.time() - started,
        "requested_cpu": 4,
        "requested_memory_mib": 16384,
        "gpu": "A10G",
        "metered_spend_usd": None,
        "billing_status": "pending",
    }
    path = ROOT / "docs/results/s3r/red-team.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved docs/results/s3r/red-team.json", flush=True)
