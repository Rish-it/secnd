"""CPU-only reader latency gate with separately timed embedding and search."""

import csv
import hashlib
import json
import platform
import resource
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend import incontext as ic

MODELS = ("ic-ettin-150m-s6000-near", "ic-ettin-32m-s6000-near")
WARMUPS = 20
REQUESTS = 100
SEED = 20261030


def statistics(milliseconds):
    values = np.asarray(milliseconds, dtype=float)
    return {
        "requests": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_requests_per_second": float(1000 * len(values) / values.sum()),
        "durations_ms": values.tolist(),
    }


def task_and_history(data_dir):
    with zipfile.ZipFile(data_dir / "datasets/mtop/mtop.zip") as archive:
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
    task = ic._task("mtop", train[4000:], frames["test"], set(), train[:4000])
    keep = np.random.default_rng(0).permutation(len(task.train))[:200]
    task.train = [task.train[i] for i in keep]
    return task


@torch.inference_mode()
def run(data_dir: Path, model_name: str) -> dict:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    from backend.features import EMBEDDERS

    if model_name not in MODELS:
        raise ValueError("Use one of the two fixed reader checkpoints.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = datetime.now(UTC).isoformat()
    task = task_and_history(data_dir)
    assert len(task.labels) == 113
    repo, revision = EMBEDDERS["minilm"]
    embedder = SentenceTransformer(repo, revision=revision, device="cpu")
    task.train_vecs = embedder.encode(
        [text for text, _ in task.train], convert_to_tensor=True, normalize_embeddings=True
    ).half()
    folder = data_dir / "encoders" / model_name
    load = time.perf_counter()
    model = ic.Reader(str(folder / "encoder")).to("cpu").eval()
    model.scorer.load_state_dict(
        torch.load(folder / "scorer.pt", map_location="cpu", weights_only=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
    load_seconds = time.perf_counter() - load
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32
    options = list(range(113))
    selected = np.random.default_rng(SEED).permutation(len(task.test))[: WARMUPS + REQUESTS]
    conditions = {}
    for shots in (0, 8, 32):
        durations = {name: [] for name in ("reader", "embed", "search", "retrieval", "request")}
        token_lengths = []
        for i, source in enumerate(selected):
            text = task.test[int(source)][0]
            request_start = time.perf_counter()
            context = []
            embed_ms = search_ms = 0.0
            if shots:
                query = embedder.encode(text, convert_to_tensor=True, normalize_embeddings=True)
                embed_end = time.perf_counter()
                context = ic.nearest(task, query.half(), shots, skip=text)
                search_end = time.perf_counter()
                embed_ms = 1000 * (embed_end - request_start)
                search_ms = 1000 * (search_end - embed_end)
            reader_start = time.perf_counter()
            ids = ic.render(tokenizer, task, text, options, context)
            packed = ic.batch(tokenizer, [(ids, 0)], "cpu")
            probabilities = torch.softmax(model(*packed[:4]).float(), -1)
            _ = float(probabilities.max())
            finished = time.perf_counter()
            if i >= WARMUPS:
                durations["reader"].append(1000 * (finished - reader_start))
                durations["request"].append(1000 * (finished - request_start))
                if shots:
                    durations["embed"].append(embed_ms)
                    durations["search"].append(search_ms)
                    durations["retrieval"].append(embed_ms + search_ms)
                token_lengths.append(len(ids))
        request_stats = statistics(durations["request"])
        conditions[str(shots)] = {
            "examples": shots,
            "reader": statistics(durations["reader"]),
            "retrieval": statistics(durations["retrieval"]) if shots else None,
            "embedding": statistics(durations["embed"]) if shots else None,
            "search": statistics(durations["search"]) if shots else None,
            "request": request_stats,
            "passes_p95_1s": request_stats["p95_ms"] <= 1000,
            "process_peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "tokens": token_lengths,
            "source_rows": selected[WARMUPS:].tolist(),
        }
        print(model_name, shots, request_stats["p95_ms"], flush=True)
    cpuinfo = Path("/proc/cpuinfo").read_text()
    return {
        "model": model_name,
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "history_rows": len(task.train),
        "options": len(options),
        "warmup_requests_per_condition": WARMUPS,
        "measured_requests_per_condition": REQUESTS,
        "seed": SEED,
        "sample_sha256": hashlib.sha256(json.dumps(selected.tolist()).encode()).hexdigest(),
        "history_sha256": hashlib.sha256(json.dumps(task.train).encode()).hexdigest(),
        "checkpoint_sha256": {
            str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
        "runtime": {
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in cpuinfo.splitlines()
                if line.startswith("model name")
            ),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "model_load_seconds": load_seconds,
        },
        "memory_scope": "Linux process peak RSS; includes reader, MiniLM, index and setup; "
        "cumulative across shot conditions within this model's own process.",
        "request_scope": "Sequential batch-one warm requests: MiniLM embedding and search "
        "when examples are used, then tokenization, packing, FP32 forward and "
        "softmax. Excludes HTTP, queueing and cold model/index loading.",
        "retrieval_model": {"repo": repo, "revision": revision},
        "conditions": conditions,
    }
