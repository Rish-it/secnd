"""Preregistered fixed-history reader audit; no training or paid teacher calls."""

import csv
import hashlib
import json
import platform
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from backend import calibrate
from backend import incontext as ic

MODEL = "ic-ettin-150m-s6000-near"
SEED = 20261030
NOISE = ("log", "record", "status", "metadata", "trace", "entry", "value", "note")
ATTACKS = (
    "baseline",
    "paraphrase",
    "typo",
    "option_order",
    "near_duplicate",
    "exact_repeat",
    "missing_intent",
    "long_input",
    "noisy_input",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mtop_task(data_dir: Path) -> ic.Task:
    """Reproduce the supplied split without changing the shared reader implementation."""
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
    assert (len(task.train), len(task.calib), len(task.test), len(task.labels)) == (
        11667,
        4000,
        4386,
        113,
    )
    return task


def variant(attack, source, text, gold, options, paraphrases):
    """Each attack has its own per-row RNG, independent of execution order."""
    rng = np.random.default_rng(SEED + source)
    options = list(options)
    if attack == "paraphrase":
        text = paraphrases[str(source)]
    elif attack == "typo":
        eligible = []
        for match in re.finditer(r"[A-Za-z]{5,}", text):
            eligible.extend(
                match.start() + i
                for i in range(1, len(match[0]) - 2)
                if match[0][i] != match[0][i + 1]
            )
        if eligible:
            # Choose a word uniformly, then a swappable internal pair uniformly.
            words = [
                m
                for m in re.finditer(r"[A-Za-z]{5,}", text)
                if any(m.start() <= i < m.end() for i in eligible)
            ]
            word = words[int(rng.integers(len(words)))]
            places = [i for i in eligible if word.start() <= i < word.end()]
            i = places[int(rng.integers(len(places)))]
            text = text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    elif attack == "option_order":
        options = rng.permutation(options).tolist()
    elif attack == "near_duplicate":
        text = " ".join(text.lower().split()) + rng.choice((".", "!", "?"))
    elif attack == "missing_intent":
        options.remove(gold)
    elif attack == "long_input":
        text = " ".join(rng.choice(NOISE, 1024)) + " Request: " + text
    elif attack == "noisy_input":
        before, after = rng.choice(NOISE, (2, 100))
        text = " ".join(before) + " Request: " + text + " " + " ".join(after)
    elif attack not in ("baseline", "exact_repeat"):
        raise ValueError(f"Unknown attack: {attack}")
    return str(text), options


def summarize(records, baseline=None):
    accepted = [r for r in records if r["accepted"]]
    n, k = len(accepted), sum(r["correct"] for r in accepted)
    precision = k / n if n else None
    coverage = n / len(records)
    upper = float(beta.ppf(0.95, k + 1, n - k)) if k < n else (1.0 if n else None)
    result = {
        "sample_size": len(records),
        "accepted": n,
        "accepted_errors": n - k,
        "accepted_precision": precision,
        "precision_minus_99": None if precision is None else precision - 0.99,
        "precision_interval95": (
            [
                float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
                float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
            ]
            if n
            else None
        ),
        "precision_upper95_one_sided": upper,
        "evidence_below_99": upper is not None and upper < 0.99,
        "coverage": coverage,
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "unchanged_messages": sum(r["unchanged"] for r in records),
        "truncated_inputs": sum(r["untruncated_tokens"] > ic.MAX_LEN for r in records),
        "records": records,
    }
    if baseline is not None:
        originals = {r["source"]: r for r in baseline}
        paired = [originals[r["source"]] for r in records]
        base = summarize(paired)
        result.update(
            baseline_sample_size=len(paired),
            baseline_accepted=base["accepted"],
            baseline_precision=base["accepted_precision"],
            baseline_coverage=base["coverage"],
            coverage_change_pp=100 * (coverage - base["coverage"]),
            prediction_flips=sum(
                a["prediction"] != b["prediction"] for a, b in zip(records, paired, strict=True)
            ),
            acceptance_flips=sum(
                a["accepted"] != b["accepted"] for a, b in zip(records, paired, strict=True)
            ),
        )
    return result


class LengthTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.length = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        self.length = len(self.tokenizer(text, truncation=False)["input_ids"])
        return self.tokenizer(text, **kwargs)


@torch.inference_mode()
def run(data_dir: Path, paraphrases: dict) -> dict:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    from backend.features import EMBEDDERS

    assert torch.cuda.is_available(), "Audit requires the preregistered Modal CUDA allocation."
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    torch.set_num_threads(4)
    folder = data_dir / "encoders" / MODEL
    model = ic.Reader(str(folder / "encoder")).to("cuda").eval()
    model.scorer.load_state_dict(
        torch.load(folder / "scorer.pt", map_location="cuda", weights_only=True)
    )
    tok = AutoTokenizer.from_pretrained(folder / "encoder")
    task = mtop_task(data_dir)
    split_hash = digest([task.train, task.calib, task.test])
    ic.index([task])
    task = ic.capped(task, 200)
    repo, revision = EMBEDDERS["minilm"]
    embedder = SentenceTransformer(repo, revision=revision, device="cuda")
    options = list(range(len(task.labels)))
    selected = np.random.default_rng(SEED).permutation(len(task.test))[:500].tolist()
    assert set(paraphrases) == {str(j) for j in selected[:50]}

    calibration = ic._predict(model, tok, task, "calib", 8, True, 0, 32)
    y = ic.labels_of(task.calib)
    indices = np.random.default_rng(0).permutation(len(y))
    fit, cert = indices[:2000], indices[2000:]
    temperature = calibrate.fit_temperature(y[fit], calibration[fit])
    threshold, cert_accepted, cert_correct, lower = calibrate.choose_threshold(
        y[cert], calibrate.scale(calibration[cert], temperature)
    )
    calibration_seconds = time.perf_counter() - started
    conditions = {}
    tracker = LengthTokenizer(tok)
    for attack in ATTACKS:
        rows = selected[:50] if attack == "paraphrase" else selected
        records = []
        for start in range(0, len(rows), 32):
            items, metadata = [], []
            for source in rows[start : start + 32]:
                original, gold = task.test[source]
                text, opts = variant(attack, source, original, gold, options, paraphrases)
                query = task.test_vecs[source]
                if text != original:
                    query = embedder.encode(
                        text, convert_to_tensor=True, normalize_embeddings=True
                    ).half()
                context = ic.nearest(task, query, 8, allowed=set(opts), skip=text)
                ids = ic.render(tracker, task, text, opts, context)
                items.append((ids, 0))
                metadata.append((source, gold, opts, text == original, len(ids), tracker.length))
            packed = ic.batch(tok, items, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*packed[:4]).float()
            p = torch.softmax(logits, -1).cpu().numpy()
            for probabilities, meta in zip(p, metadata, strict=True):
                source, gold, opts, unchanged, tokens, full_tokens = meta
                probabilities = calibrate.scale(probabilities[None, : len(opts)], temperature)[0]
                prediction = opts[int(probabilities.argmax())]
                confidence = float(probabilities.max())
                records.append(
                    {
                        "source": source,
                        "gold": gold,
                        "prediction": prediction,
                        "confidence": confidence,
                        "correct": prediction == gold,
                        "accepted": bool(threshold is not None and confidence >= threshold),
                        "unchanged": unchanged,
                        "tokens": tokens,
                        "untruncated_tokens": full_tokens,
                    }
                )
        conditions[attack] = summarize(
            records, None if attack == "baseline" else conditions["baseline"]["records"]
        )
        if attack == "missing_intent":
            conditions[attack]["false_accept_rate"] = conditions[attack]["coverage"]
        print(
            attack,
            conditions[attack]["accepted"],
            conditions[attack]["accepted_precision"],
            flush=True,
        )
    return {
        "schema_version": 1,
        "model": MODEL,
        "history_rows": len(task.train),
        "examples": 8,
        "options": len(options),
        "split_contract_commit": "e2581eecca945c5abbf5e5158c71eb51eac49067",
        "split_sha256": split_hash,
        "history_sha256": digest(task.train),
        "sample_sha256": digest(selected),
        "checkpoint_sha256": {
            str(p.relative_to(folder)): file_hash(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
        "gate_sha256": file_hash(Path(calibrate.__file__)),
        "reader_sha256": file_hash(Path(ic.__file__)),
        "gate": {
            "temperature": temperature,
            "threshold": threshold,
            "selection_target": calibrate.SELECT,
            "alpha": calibrate.ALPHA,
            "fit_rows": len(fit),
            "certification_rows": len(cert),
            "certification_accepted": cert_accepted,
            "certification_correct": cert_correct,
            "certification_cp_lower": lower,
        },
        "runtime": {
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "setup_and_calibration_seconds": calibration_seconds,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(),
            "precision": "CUDA BF16 autocast; FP32 softmax and calibration",
            "batch_size": 32,
        },
        "muse_spend_usd": 0,
        "conditions": conditions,
    }
