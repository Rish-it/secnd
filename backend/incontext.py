"""Ren-IC: one model for every label set. The labels and a business's own labelled examples are
part of the input, so a new business needs no fitted head, only its history as context.

Sequence: [CLS] question [MASK] label0 [MASK] label1 ... [SEP] message [SEP] examples [SEP].
Each option is scored from its [MASK] hidden state. Training draws episodes from many intent
sets with 0 to 32 examples of the same set in context; evaluation runs on sets never trained on
(BANKING77, and MTOP, whose texts are removed from every training pool).

Each held-out set has three parts: history (examples and fitted heads read only this), calib
(half fits the temperature, the other half certifies the 99% gate) and test (scored once).
"""

import argparse
import csv
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from backend.calibrate import choose_threshold, fit_temperature, scale
from backend.features import device
from backend.finetune import _norm
from backend.metrics import ece

ENCODERS = {
    "ettin-32m": ("jhu-clsp/ettin-encoder-32m", "1b8ba06455dd44f80fc9c1ca9e22806157a57379"),
    "ettin-68m": ("jhu-clsp/ettin-encoder-68m", "ac19ae4bc51093b31c475665ac872a936d056cc2"),
    "ettin-150m": ("jhu-clsp/ettin-encoder-150m", "45d08642849e5c5701b162671ac811b7654bfd9f"),
    "ettin-400m": ("jhu-clsp/ettin-encoder-400m", "7662476d60abb071a5bd319c9f3074f3072c062d"),
}
# BANKING77 ships no dev split: this many training rows, drawn once with seed 0, calibrate instead.
BANKING77_CALIB = 3000
MAX_LEN = 1536
MAX_OPTIONS = 128
SHOTS = (0, 0, 1, 2, 4, 8, 16, 32)
# Half the training episodes take the query's nearest past examples instead of random ones, as a
# business's history would be searched at serving time; evaluation reports both.
NEAREST = 0.5
EVAL = (
    ("0-shot", 0, False),
    ("8-random", 8, False),
    ("8-nearest", 8, True),
    ("32-nearest", 32, True),
)
INTENT = "What does the customer want?"
QUESTIONS = {"go_emotions": "Which emotion does this message express?"}


@dataclass
class Task:
    name: str
    question: str
    labels: list[str]
    train: list[tuple[str, int]]
    test: list[tuple[str, int]]
    train_vecs: torch.Tensor | None = None
    test_vecs: torch.Tensor | None = None
    calib: list[tuple[str, int]] = field(default_factory=list)
    calib_vecs: torch.Tensor | None = None


def index(tasks: list[Task]) -> None:
    """Unit MiniLM vectors for every row, kept on the GPU, to find a question's nearest examples."""
    from sentence_transformers import SentenceTransformer

    from backend.features import EMBEDDERS

    repo, revision = EMBEDDERS["minilm"]
    model = SentenceTransformer(repo, revision=revision, device=device())

    def encode(rows):
        texts = [r[0] for r in rows]
        vecs = model.encode(
            texts, batch_size=512, convert_to_tensor=True, normalize_embeddings=True
        )
        return vecs.half()

    for task in tasks:
        task.train_vecs, task.test_vecs = encode(task.train), encode(task.test)
        if task.calib:
            task.calib_vecs = encode(task.calib)


def nearest(
    task: Task, query: torch.Tensor, shots: int, allowed=None, skip: str | None = None
) -> list[tuple[str, int]]:
    """The `shots` training rows most like `query` (labels in `allowed`), most similar last."""
    order = torch.topk(
        task.train_vecs @ query, min(len(task.train), 4 * shots + 8)
    ).indices.tolist()
    rows = [
        task.train[i]
        for i in order
        if task.train[i][0] != skip and (allowed is None or task.train[i][1] in allowed)
    ]
    return rows[:shots][::-1]


def humanize(label: str) -> str:
    """IN:GET_WEATHER, card_arrival and atis_flight all read as plain words."""
    label = label.removeprefix("IN:").replace("atis_", "")
    return label.replace("_", " ").replace("+", " or ").lower().strip()


def _task(name, train: pd.DataFrame, test: pd.DataFrame, seen: set[str], calib=None) -> Task:
    """Rows are (text, label); training rows whose text a held-out set also has are dropped."""
    calib = _frame([], []) if calib is None else calib
    labels = sorted(set(train["label"]) | set(test["label"]) | set(calib["label"]))
    index = {label: i for i, label in enumerate(labels)}

    def rows(frame):
        frame = frame[~_norm(frame["text"]).isin(seen)]
        return list(zip(frame["text"], frame["label"].map(index), strict=True))

    return Task(
        name,
        QUESTIONS.get(name, INTENT),
        [humanize(x) for x in labels],
        rows(train),
        rows(test),
        calib=rows(calib),
    )


def _frame(text, label) -> pd.DataFrame:
    return pd.DataFrame(
        {"text": pd.Series(text, dtype=str).values, "label": pd.Series(label, dtype=str).values}
    )


def held_out(data_dir: Path) -> dict[str, Task]:
    """The evaluation sets: official train (history), MTOP's official eval or a fixed slice of
    BANKING77's train (calib), and official test."""
    folder = data_dir / "datasets"
    bank = {p: pd.read_csv(folder / "banking77" / f"{p}.csv", dtype=str) for p in ("train", "test")}
    with zipfile.ZipFile(folder / "mtop" / "mtop.zip") as archive:
        mtop = {
            p: pd.read_csv(
                archive.open(f"mtop/en/{p}.txt"),
                sep="\t",
                header=None,
                usecols=[1, 3],
                names=["intent", "text"],
                dtype=str,
                quoting=csv.QUOTE_NONE,
            )
            for p in ("train", "eval", "test")
        }
    shuffled = bank["train"].sample(frac=1, random_state=0)
    bank["calib"], bank["train"] = shuffled[:BANKING77_CALIB], shuffled[BANKING77_CALIB:]
    bank = {p: _frame(f["text"], f["category"]) for p, f in bank.items()}
    mtop = {p: _frame(f["text"], f["intent"]) for p, f in mtop.items()}
    return {
        "banking77": _task("banking77", bank["train"], bank["test"], set(), bank["calib"]),
        "mtop": _task("mtop", mtop["train"], mtop["test"], set(), mtop["eval"]),
    }


def training_pool(data_dir: Path, evals: dict[str, Task]) -> list[Task]:
    from backend.data import load_dataset

    seen = {
        t
        for task in evals.values()
        for t in _norm(pd.Series([r[0] for r in task.train + task.calib + task.test]))
    }
    folder = data_dir / "datasets"
    read = lambda name, part: pd.read_parquet(folder / name / f"{part}.parquet")  # noqa: E731
    clinc_names = json.loads((folder / "clinc150" / "labels.json").read_text())
    emotion_names = json.loads((folder / "go_emotions" / "labels.json").read_text())

    def emotions(part):
        frame = read("go_emotions", part)
        frame = frame[frame["labels"].map(len) == 1]
        return _frame(frame["text"], frame["labels"].map(lambda x: emotion_names[x[0]]))

    support = read("bitext_support", "train").sample(frac=1, random_state=0)
    cut = int(0.9 * len(support))
    top = load_dataset("topv2", data_dir)
    top = top.sample(frac=1, random_state=0)
    cut_top = int(0.95 * len(top))
    frames = {
        "topv2": (top[:cut_top], top[cut_top:]),
        "clinc150": tuple(
            _frame(f["text"], f["intent"].map(lambda i: clinc_names[i]))
            for f in (read("clinc150", "train"), read("clinc150", "test"))
        ),
        "massive_en": tuple(
            _frame(read("massive_en", p)["text"], read("massive_en", p)["label_text"])
            for p in ("train", "test")
        ),
        "snips": tuple(
            _frame(read("snips", p)["text"], read("snips", p)["category"])
            for p in ("train", "test")
        ),
        "atis": tuple(
            _frame(read("atis", p)["text"], read("atis", p)["intent"]) for p in ("train", "test")
        ),
        "go_emotions": (emotions("train"), emotions("test")),
        "bitext_support": tuple(
            _frame(f["instruction"], f["intent"]) for f in (support[:cut], support[cut:])
        ),
    }
    return [
        _task(name, train[["text", "label"]], test[["text", "label"]], seen)
        for name, (train, test) in frames.items()
    ]


class Reader(nn.Module):
    """An encoder plus a scorer read at every [MASK] marker."""

    def __init__(self, repo: str, revision: str | None = None):
        from transformers import AutoModel

        super().__init__()
        self.encoder = AutoModel.from_pretrained(repo, revision=revision)
        d = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ids, attention, markers, marker_mask):
        h = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        h = torch.gather(h, 1, markers[..., None].expand(-1, -1, h.shape[-1]))
        return self.scorer(h).squeeze(-1).masked_fill(~marker_mask, -1e4)


def render(
    tok, task: Task, text: str, options: list[int], shots: list[tuple[str, int]]
) -> list[int]:
    """Token ids for one question; options come before the message, so truncation drops examples."""
    clean = lambda s: str(s).replace(tok.mask_token, " ").replace(tok.sep_token, " ")  # noqa: E731
    head = task.question + "".join(f" {tok.mask_token} {task.labels[o]}" for o in options)
    tail = " ".join(f"{clean(t)} => {task.labels[y]} ;" for t, y in shots)
    return tok(
        f"{head} {tok.sep_token} {clean(text)} {tok.sep_token} {tail}",
        truncation=True,
        max_length=MAX_LEN,
    )["input_ids"]


def episode(task: Task, rng: np.random.Generator, shots: int):
    """(text, options, gold slot, examples) for one training question."""
    q = int(rng.integers(len(task.train)))
    text, gold = task.train[q]
    n = len(task.labels)
    k = int(rng.integers(2, min(n, MAX_OPTIONS) + 1))
    others = rng.permutation([i for i in range(n) if i != gold])[: k - 1]
    options = list(rng.permutation(np.append(others, gold)))
    allowed = set(options)
    if shots and task.train_vecs is not None and rng.random() < NEAREST:
        return (
            text,
            options,
            options.index(gold),
            nearest(task, task.train_vecs[q], shots, allowed, skip=text),
        )
    pool = [
        r
        for r in (task.train[i] for i in rng.integers(len(task.train), size=8 * shots + 8))
        if r[1] in allowed and r[0] != text
    ]
    return text, options, options.index(gold), pool[:shots]


def batch(tok, items, dev):
    width = max(len(ids) for ids, _ in items)
    ids = torch.full((len(items), width), tok.pad_token_id)
    attention = torch.zeros_like(ids)
    for i, (row, _) in enumerate(items):
        ids[i, : len(row)] = torch.tensor(row)
        attention[i, : len(row)] = 1
    found = [(ids[i] == tok.mask_token_id).nonzero().squeeze(-1) for i in range(len(items))]
    n = max(len(f) for f in found)
    markers = torch.zeros((len(items), n), dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.bool)
    for i, f in enumerate(found):
        markers[i, : len(f)] = f
        mask[i, : len(f)] = True
    gold = torch.tensor([g for _, g in items])
    return ids.to(dev), attention.to(dev), markers.to(dev), mask.to(dev), gold.to(dev)


def train(name: str, tasks: list[Task], steps: int, batch_size: int = 16, seed: int = 0):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    repo, revision = ENCODERS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = Reader(repo, revision).to(dev)
    optim = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 5e-5},
            {"params": model.scorer.parameters(), "lr": 2e-4},
        ],
        weight_decay=0.01,
    )
    sched = get_linear_schedule_with_warmup(optim, steps // 20, steps)
    weights = np.sqrt([len(t.train) for t in tasks])
    weights /= weights.sum()
    model.train()
    start, losses = time.time(), []
    for step in range(steps):
        items = []
        for _ in range(batch_size):
            task = tasks[rng.choice(len(tasks), p=weights)]
            text, options, gold, shots = episode(task, rng, int(rng.choice(SHOTS)))
            items.append((render(tok, task, text, options, shots), gold))
        ids, attention, markers, mask, gold = batch(tok, items, dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(ids, attention, markers, mask)
        loss = nn.functional.cross_entropy(logits.float(), gold)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            print(
                f"step {step + 1}/{steps} loss {np.mean(losses[-200:]):.4f}"
                f" {time.time() - start:.0f}s",
                flush=True,
            )
    return (
        model.eval(),
        tok,
        {
            "steps": steps,
            "final_loss": float(np.mean(losses[-200:])),
            "train_seconds": round(time.time() - start),
        },
    )


PARTS = ("calib", "test")


@torch.no_grad()
def predict(
    model, tok, task: Task, shots: int, near: bool = False, seed: int = 0, batch_size: int = 32
) -> dict[str, np.ndarray]:
    """P(label) for every calib and test row, with `shots` history examples in context."""
    return {part: _predict(model, tok, task, part, shots, near, seed, batch_size) for part in PARTS}


def _predict(model, tok, task, part, shots, near, seed, batch_size) -> np.ndarray:
    dev = next(model.parameters()).device.type
    rng = np.random.default_rng(seed)
    options = list(range(len(task.labels)))
    rows, vecs = getattr(task, part), getattr(task, f"{part}_vecs")
    out = []
    for start in range(0, len(rows), batch_size):
        items = []
        for j, (text, gold) in enumerate(rows[start : start + batch_size], start):
            if near and shots:
                context = nearest(task, vecs[j], shots, skip=text)
            else:
                context = (
                    [task.train[i] for i in rng.choice(len(task.train), size=shots, replace=False)]
                    if shots
                    else []
                )
            items.append((render(tok, task, text, options, context), gold))
        ids, attention, markers, mask, _ = batch(tok, items, dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(ids, attention, markers, mask).float()
        out.append(torch.softmax(logits, -1)[:, : len(options)].cpu().numpy())
    return np.concatenate(out)


def vote(task: Task, shots: int) -> dict[str, np.ndarray]:
    """Baseline with no reader: P(label) from a vote of the nearest examples."""
    out = {}
    for part in PARTS:
        rows, vecs = getattr(task, part), getattr(task, f"{part}_vecs")
        p = np.full((len(rows), len(task.labels)), 1e-3)
        for j, (text, _) in enumerate(rows):
            for _, y in nearest(task, vecs[j], shots, skip=text):
                p[j, y] += 1.0
        out[part] = p / p.sum(1, keepdims=True)
    return out


def labels_of(rows) -> np.ndarray:
    return np.array([g for _, g in rows])


def score(task: Task, p: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Temperature from one half of calib, the 99% gate certified on the other half, then both
    applied to test. Test labels only score; they choose nothing."""
    yc, yt = labels_of(task.calib), labels_of(task.test)
    half = np.random.default_rng(seed).permutation(len(yc))
    fit, cert = half[: len(yc) // 2], half[len(yc) // 2 :]
    t = fit_temperature(yc[fit], p["calib"][fit])
    tau, n_auto, _, lower = choose_threshold(yc[cert], scale(p["calib"][cert], t))
    pt = scale(p["test"], t)
    auto = pt.max(1) >= tau if tau is not None else np.zeros(len(yt), bool)
    hits = pt.argmax(1) == yt
    return {
        "accuracy": round(float(hits.mean()), 4),
        "ece_test": round(float(ece(yt, pt)), 4),
        "temperature": round(t, 3),
        "certified_coverage_test": round(float(auto.mean()), 4),
        "precision_auto_test": round(float(hits[auto].mean()), 4) if auto.any() else None,
        "cert_rows": len(cert),
        "cert_accepted": n_auto,
        "cert_cp_lower": round(lower, 4),
    }


BUDGETS = (200, 1000, None)


def capped(task: Task, n: int | None, seed: int = 0) -> Task:
    """The same set with only `n` labelled history rows (all when None), drawn at random."""
    if n is None or n >= len(task.train):
        return task
    keep = np.random.default_rng(seed).permutation(len(task.train))[:n]
    train = [task.train[i] for i in keep]
    return Task(
        task.name,
        task.question,
        task.labels,
        train,
        task.test,
        task.train_vecs[keep],
        task.test_vecs,
        task.calib,
        task.calib_vecs,
    )


def head(task: Task) -> dict[str, np.ndarray]:
    """Ren 1.0.0's recipe on the same history: logistic regression on the MiniLM vectors."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=10, max_iter=3000)
    clf.fit(task.train_vecs.float().cpu().numpy(), labels_of(task.train))
    out = {}
    for part in PARTS:
        vecs = getattr(task, f"{part}_vecs").float().cpu().numpy()
        out[part] = np.zeros((len(vecs), len(task.labels)))
        out[part][:, clf.classes_] = clf.predict_proba(vecs)
    return out


def budgeted(model, tok, evals: dict[str, Task], save: Path) -> dict:
    """Reader, vote and head given the same 200, 1,000 or all labelled history rows per set.
    Every probability matrix is also saved under `save`, so gate rules can be compared on CPU."""
    save.mkdir(parents=True, exist_ok=True)

    def kept(task, key, p):
        np.savez_compressed(
            save / f"{key}.npz",
            **{f"p_{k}": v.astype(np.float16) for k, v in p.items()},
            y_calib=labels_of(task.calib),
            y_test=labels_of(task.test),
        )
        return score(task, p)

    out = {}
    for name, task in evals.items():
        out[name] = {"reader-0": kept(task, f"{name}-reader-0", predict(model, tok, task, 0))}
        for n in BUDGETS:
            t, b = capped(task, n), str(n or "all")
            out[name][b] = {
                "head": kept(t, f"{name}-{b}-head", head(t)),
                "vote-8": kept(t, f"{name}-{b}-vote-8", vote(t, 8)),
                "reader-8": kept(t, f"{name}-{b}-reader-8", predict(model, tok, t, 8, near=True)),
                "reader-32": kept(
                    t, f"{name}-{b}-reader-32", predict(model, tok, t, 32, near=True)
                ),
            }
    return out


def evaluate(out: str) -> None:
    """Re-score a saved reader on the held-out sets at each history budget."""
    from transformers import AutoTokenizer

    from backend.config import Settings

    data_dir = Settings().data_dir
    folder = data_dir / "encoders" / out
    model = Reader(str(folder / "encoder"))
    model.scorer.load_state_dict(torch.load(folder / "scorer.pt", map_location="cpu"))
    model = model.to(device()).eval()
    tok = AutoTokenizer.from_pretrained(folder / "encoder")
    evals = held_out(data_dir)
    # Readers trained before calib was held out saw some of its texts (MTOP's eval shares ~1,000
    # with TOPv2): calibrate on the rest only.
    texts = lambda rows: _norm(pd.Series([r[0] for r in rows]))  # noqa: E731
    trained = {t for task in training_pool(data_dir, {}) for t in texts(task.train)}
    for task in evals.values():
        fresh = ~texts(task.calib).isin(trained)
        task.calib = [r for r, ok in zip(task.calib, fresh, strict=True) if ok]
    index(list(evals.values()))
    split = {
        name: {p: len(getattr(t, p)) for p in ("train", "calib", "test")}
        for name, t in evals.items()
    }
    budgets = budgeted(model, tok, evals, data_dir / "predictions" / out)
    report = {"model": out, "device": device(), "split": split, "budgets": budgets}
    path = Path("docs/results/s2") / f"incontext-{out}-clean.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["budgets"]))


def main():
    from backend.config import Settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=sorted(ENCODERS))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--evaluate", help="re-score a saved reader, e.g. ic-ettin-32m-s6000-near")
    args = parser.parse_args()
    if args.evaluate:
        return evaluate(args.evaluate)
    data_dir = Settings().data_dir
    evals = held_out(data_dir)
    tasks = training_pool(data_dir, evals)
    index(tasks + list(evals.values()))
    model, tok, report = train(args.encoder, tasks, args.steps)
    out = f"ic-{args.encoder}-s{args.steps}-near"
    model.encoder.save_pretrained(data_dir / "encoders" / out / "encoder")
    tok.save_pretrained(data_dir / "encoders" / out / "encoder")
    torch.save(model.scorer.state_dict(), data_dir / "encoders" / out / "scorer.pt")
    report |= {
        "model": out,
        "base": "@".join(ENCODERS[args.encoder]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": device(),
        "train_sets": {t.name: len(t.train) for t in tasks},
        "results": {
            name: {key: score(task, predict(model, tok, task, k, near)) for key, k, near in EVAL}
            | {f"vote-{k}": score(task, vote(task, k)) for k in (8, 32)}
            for name, task in evals.items()
        },
    }
    path = Path("docs/results/s2") / f"incontext-{out}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
