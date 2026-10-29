"""Run preregistered attacks on GPU and serving measurements on a CPU allocation."""

import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .env({"SECND_DATA_DIR": "/vol/data", "HF_HOME": "/vol/hf"})
    .add_local_python_source("backend", copy=True)
    .add_local_file(
        str(ROOT / "docs/results/s3r/paraphrases.json"), "/audit-paraphrases.json", copy=True
    )
)
volume = modal.Volume.from_name("secnd-bench")
app = modal.App("secnd-reader-audit")


def persist(report, name):
    # Save before RPC return so client transfer failures cannot discard measurements.
    payload = json.dumps(report)
    path = Path("/vol/data/reader-audits") / f"{name}-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
    volume.commit()
    print(f"Saved {path}", flush=True)
    return payload


@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=7200)
def cpu(model_name: str):
    from backend.reader_audit import run

    return persist(run(Path("/vol/data"), latency=True, model_name=model_name), "cpu-" + model_name)


gpu_image = image.run_commands(
    "uv pip install --python /.uv/.venv/bin/python --reinstall-package torch "
    "--index-url https://pypi.org/simple torch==2.15.0"
)


@app.function(image=gpu_image, gpu="A10G", volumes={"/vol": volume}, timeout=7200)
def attacks(natural: bool = False):
    from backend.reader_audit import run

    return persist(run(Path("/vol/data"), natural=natural), "paraphrases" if natural else "attacks")


@app.local_entrypoint()
def main(
    latency: bool = False, natural: bool = False, model_name: str = "ic-ettin-150m-s6000-near"
):
    # Snapshot provenance before dispatch; later doc edits cannot change a run's identity.
    protocol_hash = hashlib.sha256((ROOT / "docs/results/s3r/protocol.md").read_bytes()).hexdigest()
    runner_hash = hashlib.sha256((ROOT / "backend/reader_audit.py").read_bytes()).hexdigest()
    report = json.loads(cpu.remote(model_name) if latency else attacks.remote(natural))
    report["protocol_sha256"] = protocol_hash
    report["runner_sha256"] = runner_hash
    name = (
        ("cpu" if "150m" in model_name else "cpu-32m")
        if latency
        else ("paraphrases" if natural else "attacks")
    )
    path = ROOT / "docs/results/s3r" / f"{name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
