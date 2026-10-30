"""Frozen S3R protocol runner; no training or paid teacher calls."""

import hashlib
import json
import platform
import re
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from transformers import AutoTokenizer

from backend import incontext as ic
from backend.calibrate import choose_threshold, fit_temperature, scale

MODEL = "ic-ettin-150m-s6000-near"
FLOOR_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
MISSING_RATE = 0.002


def choose_floor(supports, n):
    """Lowest grid floor that leaves at most MISSING_RATE * n missing-intent answers accepted."""
    supports = np.asarray(supports, dtype=float)
    # 1.01 accepts nothing: duplicate texts under another intent can sit at similarity 1.0
    return float(next((f for f in FLOOR_GRID if (supports >= f).sum() <= MISSING_RATE * n), 1.01))


def apply_audit(task, changed):
    """Move audited calibration labels in place; test labels are never touched."""
    for change in changed:
        i = int(change["id"][1:])
        text, label = task.calib[i]
        assert text == change["message"] and task.labels[label] == change["old"], change["id"]
        task.calib[i] = (text, task.labels.index(change["new"]))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def typo(text):
    return re.sub(
        r"\b([A-Za-z])([A-Za-z])([A-Za-z])([A-Za-z]{2,})\b",
        lambda m: m[1] + m[3] + m[2] + m[4],
        text,
        count=1,
    )


@torch.inference_mode()
def run(data_dir, latency=False, natural=False, model_name=MODEL, floor=False, audit=None):
    torch.set_num_threads(4)
    dev = "cpu" if latency else ic.device()
    folder = data_dir / "encoders" / model_name
    start = time.perf_counter()
    model = ic.Reader(str(folder / "encoder")).to(dev).eval()
    model.scorer.load_state_dict(
        torch.load(folder / "scorer.pt", map_location=dev, weights_only=True)
    )
    model.none = (folder / "none").exists()
    tok = AutoTokenizer.from_pretrained(folder / "encoder")
    load_seconds = time.perf_counter() - start
    task = ic.held_out(data_dir)["mtop"]
    if audit:
        apply_audit(task, audit)
    ic.index([task])
    options = list(range(len(task.labels)))
    assert len(options) == 113
    selected = np.random.default_rng(20261030).permutation(len(task.test))[:500]
    rewrites = {}
    if natural:
        rewrites = json.loads(Path("/audit-paraphrases.json").read_text())
        selected = selected[:50]
        assert set(map(str, selected)) == set(rewrites)
    report = dict(
        model=model_name,
        device=dev,
        load_seconds=load_seconds,
        split_sha256=digest([task.train, task.calib, task.test]),
        checkpoint_sha256={
            str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
        torch=str(torch.__version__),
        platform=platform.platform(),
        threads=4,
        audited_calib_labels=len(audit or ()),
        results={},
    )
    if Path("/proc/cpuinfo").exists():
        report["cpu"] = next(
            x for x in Path("/proc/cpuinfo").read_text().splitlines() if "model name" in x
        )

    def infer(text, opts, context):
        ids = ic.render(tok, task, text, opts, context, model.none)
        packed = ic.batch(tok, [(ids, 0)], dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(*packed[:4]).float()
        p = torch.softmax(logits, -1).cpu().numpy()
        return p, len(ids)

    if latency:
        assert next(model.parameters()).device.type == "cpu"
        for shots in (0, 8, 32):
            contexts = [
                ic.nearest(task, task.test_vecs[j], shots, skip=task.test[j][0]) if shots else []
                for j in selected[:120]
            ]
            times, lengths = [], []
            for i, (j, context) in enumerate(zip(selected[:120], contexts, strict=True)):
                start = time.perf_counter()
                _, length = infer(task.test[j][0], options, context)
                elapsed = (time.perf_counter() - start) * 1000
                if i >= 20:
                    times.append(elapsed)
                    lengths.append(length)
            report["results"][str(shots)] = dict(
                p50_ms=float(np.percentile(times, 50)),
                p95_ms=float(np.percentile(times, 95)),
                durations_ms=times,
                tokens=lengths,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            print(
                model_name,
                shots,
                {
                    k: v
                    for k, v in report["results"][str(shots)].items()
                    if k not in ("durations_ms", "tokens")
                },
                flush=True,
            )
        report["scope"] = (
            "Batch one FP32; tokenization through softmax; "
            "retrieval excluded; Linux process peak RSS cumulative, includes retrieval index."
        )
        return report

    from sentence_transformers import SentenceTransformer

    from backend.features import EMBEDDERS

    repo, revision = EMBEDDERS["minilm"]
    embedder = SentenceTransformer(repo, revision=revision, device=dev)
    history_labels = torch.tensor([r[1] for r in task.train], device=task.train_vecs.device)

    def support(query, label):
        # Highest similarity between the message and any history row carrying `label`.
        sims = (task.train_vecs @ query)[history_labels == label]
        return float(sims.max()) if len(sims) else -1.0

    for shots in (0, 8, 32):
        pc = ic._predict(model, tok, task, "calib", shots, bool(shots), 0, 32)
        yc = ic.labels_of(task.calib)
        half = np.random.default_rng(0).permutation(len(yc))
        fit, cert = half[: len(yc) // 2], half[len(yc) // 2 :]
        temperature = fit_temperature(yc[fit], pc[fit])
        tau, _, _, _ = choose_threshold(yc[cert], scale(pc[cert], temperature)[:, : len(options)])
        cut, fitted = 0.0, None
        if floor and tau is not None:
            missed = []
            for j in cert:
                text, gold = task.calib[j]
                opts = [o for o in options if o != gold]
                query = task.calib_vecs[j]
                context = (
                    ic.nearest(task, query, shots, allowed=set(opts), skip=text) if shots else []
                )
                p = scale(infer(text, opts, context)[0], temperature)[0][: len(opts)]
                if p.max() >= tau:
                    missed.append(support(query, opts[int(p.argmax())]))
            cut = choose_floor(missed, len(cert))
            clean = scale(pc[cert], temperature)[:, : len(options)]
            kept = clean.max(axis=1) >= tau
            clean_support = np.array(
                [
                    support(task.calib_vecs[j], int(a))
                    for j, a in zip(cert, clean.argmax(axis=1), strict=True)
                ]
            )
            fitted = dict(
                floor=cut,
                calib_missing_accepted=len(missed),
                calib_missing_accepted_with_floor=int((np.array(missed) >= cut).sum()),
                calib_clean_accepted=int(kept.sum()),
                calib_clean_accepted_with_floor=int((kept & (clean_support >= cut)).sum()),
            )
            print(shots, fitted, flush=True)
        conditions = {}
        attack_names = (
            ("baseline", "natural")
            if natural
            else ("baseline", "paraphrase", "typo", "shuffle", "duplicate", "normalized", "missing")
        )
        for attack in attack_names:
            records = []
            rng = np.random.default_rng(20261030)
            for j in selected:
                text, gold = task.test[j]
                original = text
                opts = options.copy()
                if attack == "natural":
                    text = rewrites[str(j)]
                elif attack == "paraphrase":
                    text = "Please help me with this request: " + text
                elif attack == "typo":
                    text = typo(text)
                elif attack == "normalized":
                    text = " ".join(text.lower().split())
                elif attack == "shuffle":
                    opts = rng.permutation(opts).tolist()
                elif attack == "missing":
                    opts.remove(gold)
                query = task.test_vecs[j]
                if text != original and shots:
                    query = embedder.encode(
                        text, convert_to_tensor=True, normalize_embeddings=True
                    ).half()
                context = (
                    ic.nearest(task, query, shots, allowed=set(opts), skip=text) if shots else []
                )
                p, length = infer(text, opts, context)
                # a NONE column shares the temperature but is never an answer
                p = scale(p, temperature)[0][: len(opts)]
                pred = opts[int(p.argmax())]
                accepted = bool(tau is not None and p.max() >= tau)
                if floor:
                    accepted = accepted and support(query, pred) >= cut
                records.append(
                    dict(
                        source=int(j),
                        prediction=pred,
                        correct=bool(pred == gold),
                        accepted=accepted,
                        unchanged=text == original,
                        tokens=length,
                    )
                )
            accepted = [r for r in records if r["accepted"]]
            n = len(accepted)
            k = sum(r["correct"] for r in accepted)
            precision = k / n if n else None
            interval = (
                [
                    float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
                    float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
                ]
                if n
                else None
            )
            conditions[attack] = dict(
                accepted=n,
                errors=n - k,
                coverage=n / len(records),
                precision=precision,
                precision_minus_99=None if n == 0 else precision - 0.99,
                interval95=interval,
                records=records,
            )
            if attack != "baseline":
                base = conditions["baseline"]["records"]
                conditions[attack]["prediction_flips"] = sum(
                    a["prediction"] != b["prediction"] for a, b in zip(base, records, strict=True)
                )
                conditions[attack]["acceptance_flips"] = sum(
                    a["accepted"] != b["accepted"] for a, b in zip(base, records, strict=True)
                )
            print(shots, attack, n, precision, flush=True)
        report["results"][str(shots)] = dict(
            temperature=temperature, threshold=tau, floor=fitted, attacks=conditions
        )
    return report
