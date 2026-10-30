"""Local CPU HTTP measurement. Uses installed weights and fixed MTOP rows; no downloads/calls.

uv run python -m backend.tests.http_reader --artifacts DATA --output RESULT.json
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch

from backend.incontext import _task
from backend.reader_policy import MANIFEST
from backend.reader_scope import scope_digest
from backend.store import Store


def full_history(artifacts):
    with zipfile.ZipFile(artifacts / "datasets/mtop/mtop.zip") as archive:
        frames = {
            part: pd.read_csv(
                archive.open(f"mtop/en/{part}.txt"),
                sep="\t",
                header=None,
                usecols=[1, 3],
                names=["label", "text"],
                dtype=str,
                quoting=csv.QUOTE_NONE,
            )
            for part in ("train", "test")
        }
    train = frames["train"].sample(frac=1, random_state=0)
    task = _task("mtop", train[4000:], frames["test"], set(), train[:4000])
    digest = hashlib.sha256(
        json.dumps([task.train, task.calib, task.test], sort_keys=True).encode()
    ).hexdigest()
    manifest = json.loads(MANIFEST.read_text())
    assert digest == manifest["scope"]["split_sha256"]
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--hold", action="store_true", help="Keep the app running for browser inspection."
    )
    args = parser.parse_args()
    task = full_history(args.artifacts)
    selected = np.random.default_rng(20261030).permutation(len(task.test))[
        : args.warmups + args.requests
    ]
    manifest = json.loads(MANIFEST.read_text())
    with tempfile.TemporaryDirectory(prefix="secnd-reader-http-") as directory:
        data = Path(directory)
        (data / "encoders").mkdir()
        (data / "encoders/ic-ettin-150m-s6000-near").symlink_to(
            (args.artifacts / "encoders/ic-ettin-150m-s6000-near").resolve(),
            target_is_directory=True,
        )
        rows = [
            {"id": f"history-{i}", "text": text, "label": task.labels[label], "merchant_key": ""}
            for i, (text, label) in enumerate(task.train)
        ]
        assert scope_digest(task.labels, rows) == manifest["scope"]["history_sha256"]
        store = Store(data / "secnd.sqlite3")
        store.create_workspace(
            "default",
            {
                "labels": task.labels,
                "calib": [
                    {
                        "id": f"calibration-{i}",
                        "text": text,
                        "label": task.labels[label],
                        "amount": None,
                        "date": None,
                        "merchant_key": "",
                    }
                    for i, (text, label) in enumerate(task.calib)
                ],
                "featurizer": "char_tfidf",
                "seed": 0,
            },
            items=rows,
            labels=[(r["id"], r["label"]) for r in rows],
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
            port = sock.getsockname()[1]
        env = {
            **os.environ,
            "SECND_DATA_DIR": str(data),
            "SECND_SERVING_BACKEND": "reader",
            "SECND_READER": json.dumps(
                {"model_id": "ic-ettin-150m-s6000-near", "device": "cpu", "examples": 8}
            ),
            "SECND_WORKSPACE_BACKENDS": json.dumps({"default": "reader"}),
            "SECND_WORKSPACE_READERS": "{}",
            "SECND_WHATSAPP_SEND_ENABLED": "false",
            "SECND_COLD_START_PAID": "false",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        log = data / "server.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--no-access-log",
                ],
                env=env,
                stdout=stream,
                stderr=stream,
            )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                for _ in range(180):
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Reader API startup failed. Check the local reader and retrieval files."
                        )
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Reader API startup timed out.")
                startup = time.perf_counter() - started
                status = client.get("/api/v1/workspaces/default/serving").json()
                assert status["scope_matches"] and status["policy_complete"]
                records = []
                for index, source in enumerate(selected):
                    begin = time.perf_counter()
                    response = client.post(
                        "/api/v1/workspaces/default/predict",
                        json={
                            "items": [
                                {"item_id": f"request-{index}", "text": task.test[int(source)][0]}
                            ]
                        },
                    )
                    elapsed = (time.perf_counter() - begin) * 1000
                    response.raise_for_status()
                    decision = response.json()[0]
                    assert decision["reader"]["scope_authorized"]
                    if decision["auto"]:
                        assert decision["prob"] >= decision["threshold"]
                        assert decision["reader"]["support"] >= decision["reader"]["support_floor"]
                    if index >= args.warmups:
                        records.append(
                            {
                                "source": int(source),
                                "http_ms": elapsed,
                                "auto": decision["auto"],
                                "spot_check": decision["spot_check"],
                                "examples": decision["reader"]["examples"],
                            }
                        )
                durations = [r["http_ms"] for r in records]
                result = {
                    "measured_at": datetime.now(UTC).isoformat(),
                    "model_id": status["model_id"],
                    "revision": status["revision"],
                    "policy_file": manifest["models"][status["model_id"]]["policy_file"],
                    "policy_sha256": manifest["models"][status["model_id"]]["policy_sha256"],
                    "threshold": status["threshold"],
                    "support_floor": status["support_floor"],
                    "temperature": status["temperature"],
                    "device": "cpu",
                    "examples": 8,
                    "platform": platform.platform(),
                    "cpu": (
                        subprocess.check_output(
                            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                        ).strip()
                        if sys.platform == "darwin"
                        else platform.processor() or platform.machine()
                    ),
                    "torch": str(torch.__version__),
                    "logical_cores": os.cpu_count(),
                    "python": platform.python_version(),
                    "threads": 4,
                    "warmups": args.warmups,
                    "requests": len(records),
                    "history_rows": len(rows),
                    "labels": len(task.labels),
                    "split_sha256": manifest["scope"]["split_sha256"],
                    "startup_seconds": startup,
                    "p50_ms": float(np.percentile(durations, 50)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "measurement": (
                        "Sequential loopback HTTP requests: full-history SQL read, scope check, "
                        "MiniLM retrieval, tokenization, CPU FP32 reader, "
                        "frozen temperature/support "
                        "checks, SQLite persistence and JSON response. Warmup excludes model/index "
                        "initialization. No concurrency/throughput or quality claim."
                    ),
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "checkpoint_sha256": manifest["models"][status["model_id"]][
                        "checkpoint_sha256"
                    ],
                    "records": records,
                    "live_sends": 0,
                    "ad_changes": 0,
                    "paid_calls": 0,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k in ["p50_ms", "p95_ms", "requests", "history_rows"]
                        }
                    ),
                    flush=True,
                )
                if args.hold:
                    print(f"Browser URL http://127.0.0.1:{port}", flush=True)
                    while True:
                        time.sleep(1)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
