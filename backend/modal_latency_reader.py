"""Measure both readers on separate CPU allocations; no paid model API calls."""

import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .env({"SECND_DATA_DIR": "/vol/data", "HF_HOME": "/vol/hf"})
    .add_local_python_source("backend", copy=True)
)
volume = modal.Volume.from_name("secnd-bench")
app = modal.App("secnd-s8-cpu-latency")


@app.function(image=image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=1800)
def reader150():
    from backend.config import Settings
    from backend.latency_reader import run

    return json.dumps(run(Settings().data_dir, "ic-ettin-150m-s6000-near"))


@app.function(image=image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=1800)
def reader32():
    from backend.config import Settings
    from backend.latency_reader import run

    return json.dumps(run(Settings().data_dir, "ic-ettin-32m-s6000-near"))


@app.function(image=image, cpu=4, memory=8192, volumes={"/vol": volume}, timeout=1800)
def reader(model_name: str, int8: bool):
    from backend.config import Settings
    from backend.latency_reader import run

    return json.dumps(run(Settings().data_dir, model_name, int8))


gpu_image = image.run_commands(
    "uv pip install --python /.uv/.venv/bin/python --reinstall-package torch "
    "--index-url https://pypi.org/simple torch==2.15.0"
)


@app.function(image=gpu_image, gpu="A10G", volumes={"/vol": volume}, timeout=1800)
def reader_gpu(model_name: str):
    from backend.config import Settings
    from backend.latency_reader import run

    return json.dumps(run(Settings().data_dir, model_name, dev="cuda"))


@app.local_entrypoint()
def gpu(model_name: str = "ic-ettin-400m-s6000-near"):
    """One A10G; retrieval stays on CPU, the reader is BF16 as in the attack audit."""
    report = json.loads(reader_gpu.remote(model_name))
    path = ROOT / f"docs/results/s8/latency-gpu-{model_name}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


@app.local_entrypoint()
def ren11(model_name: str = "ic-ettin-400m-s6000-near"):
    """FP32 and int8 on separate, identical CPU allocations."""
    runner_hash = hashlib.sha256((ROOT / "backend/latency_reader.py").read_bytes()).hexdigest()
    calls = [reader.spawn(model_name, False), reader.spawn(model_name, True)]
    report = {
        "schema_version": 1,
        "sla_p95_ms": 1000,
        "runner_sha256": runner_hash,
        "readers": [json.loads(call.get()) for call in calls],
        "modal": {
            "app_id": app.app_id,
            "function_call_ids": [call.object_id for call in calls],
            "requested_cpu_per_process": 4,
            "requested_memory_mib_per_process": 8192,
        },
    }
    path = ROOT / f"docs/results/s8/latency-{model_name}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


@app.local_entrypoint()
def main():
    runner_hash = hashlib.sha256((ROOT / "backend/latency_reader.py").read_bytes()).hexdigest()
    calls = [reader150.spawn(), reader32.spawn()]
    print("Modal calls: " + ", ".join(call.object_id for call in calls), flush=True)
    report = {
        "schema_version": 1,
        "sla_p95_ms": 1000,
        "sla_scope": "Warm request including retrieval, excluding HTTP and cold load",
        "runner_sha256": runner_hash,
        "readers": [json.loads(call.get()) for call in calls],
        "muse_spend_usd": 0,
        "modal": {
            "app_id": app.app_id,
            "function_call_ids": [call.object_id for call in calls],
            "requested_cpu_per_process": 4,
            "requested_memory_mib_per_process": 8192,
            "metered_spend_usd": None,
            "billing_status": "pending",
        },
    }
    path = ROOT / "docs/results/s8/latency-reader.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Saved docs/results/s8/latency-reader.json", flush=True)
