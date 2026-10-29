"""Ren on the public typed-decisions benchmark (Apache-2.0): 1,200 training cases, 400 test cases,
2,000 test decisions of three types, each scored as its top option against the gold label.

Every question becomes one reader pass: [CLS] type id: instructions [MASK] key: meaning ...
[SEP] state [SEP]. Training matches the benchmark's teacher distributions (soft cross-entropy),
with the option order shuffled. A tenth of the training cases, held out by case, picks the epoch
and fits one temperature per type; test is scored once, at the end.

    python -m backend.typed --init ettin-400m --epochs 8
    python -m backend.typed --init ic-ettin-150m-s6000-near    # start from the in-context reader
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from backend import incontext as ic
from backend.calibrate import fit_temperature, scale
from backend.features import device

MAX_LEN = 512
NOUL = {"false": "No.", "true": "Yes."}
VAL_SHARE = 0.1


@dataclass
class Decision:
    case: str
    kind: str
    text: str  # question with its options, as rendered for the reader's head
    keys: list[str]
    meanings: list[str]
    state: str
    gold: np.ndarray  # teacher distribution over keys
    label: int  # index of the gold label


def decisions(frame: pd.DataFrame) -> list[Decision]:
    out = []
    for row in frame.itertuples():
        questions, gold = json.loads(row.questions), json.loads(row.gold)
        for qid, q in questions.items():
            if qid not in gold:
                continue
            g, kind = gold[qid], q["type"]
            if kind == "choice":
                keys = list(q["criteria"])
                meanings = [q["criteria"][k] for k in keys]
            elif kind == "noul":
                keys = ["false", "true"]
                meanings = [q.get("criteria", NOUL)[k] for k in keys]
            else:
                meanings = list(q["criteria"])
                keys = [str(i) for i in range(len(meanings))]
            probs = np.array([float(g["probabilities"].get(k, 0.0)) for k in keys])
            out.append(
                Decision(
                    case=row.id,
                    kind=kind,
                    text=f"{kind} {qid.replace('_', ' ')}: {q['instructions']}",
                    keys=keys,
                    meanings=meanings,
                    state=row.state,
                    gold=probs / probs.sum(),
                    label=keys.index(
                        str(g["label"]).lower() if kind == "noul" else str(g["label"])
                    ),
                )
            )
    return out


def load(data_dir: Path):
    folder = data_dir / "datasets" / "typed_decisions"
    train = pd.read_parquet(folder / "train.parquet")
    test = pd.read_parquet(folder / "test.parquet")
    cases = np.random.default_rng(0).permutation(train.id.to_numpy())
    val_cases = set(cases[: int(VAL_SHARE * len(cases))])
    return (
        decisions(train[~train.id.isin(val_cases)]),
        decisions(train[train.id.isin(val_cases)]),
        decisions(test),
    )


def render(tok, d: Decision, order: list[int]) -> list[int]:
    head = d.text + "".join(f" {tok.mask_token} {d.keys[i]}: {d.meanings[i]}" for i in order)
    return tok(f"{head} {tok.sep_token} {d.state}", truncation=True, max_length=MAX_LEN)[
        "input_ids"
    ]


def reader(init: str, data_dir: Path):
    from transformers import AutoTokenizer

    if init in ic.ENCODERS:
        repo, revision = ic.ENCODERS[init]
        return ic.Reader(repo, revision), AutoTokenizer.from_pretrained(repo, revision=revision)
    folder = data_dir / "encoders" / init
    model = ic.Reader(str(folder / "encoder"))
    model.scorer.load_state_dict(torch.load(folder / "scorer.pt", map_location="cpu"))
    return model, AutoTokenizer.from_pretrained(folder / "encoder")


def forward(model, tok, items, dev):
    """Logits per decision, in each decision's own key order; items are (decision, order)."""
    rows = [(render(tok, d, order), 0) for d, order in items]
    ids, attention, markers, mask, _ = ic.batch(tok, rows, dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        logits = model(ids, attention, markers, mask).float()
    out = []
    for i, (_, order) in enumerate(items):
        back = torch.empty(len(order), dtype=torch.long, device=logits.device)
        back[torch.tensor(order, device=logits.device)] = torch.arange(len(order), device=dev)
        out.append(logits[i, : len(order)][back])
    return out


@torch.no_grad()
def predict(model, tok, ds: list[Decision], batch_size: int = 64) -> list[np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device.type
    out = []
    for start in range(0, len(ds), batch_size):
        chunk = [(d, list(range(len(d.keys)))) for d in ds[start : start + batch_size]]
        out += [torch.softmax(x, -1).cpu().numpy() for x in forward(model, tok, chunk, dev)]
    return out


def correct(d: Decision, p: np.ndarray) -> bool:
    """The benchmark's rule: top option; a noul answer is true from p(true) >= 0.5."""
    pred = int(p[1] >= 0.5) if d.kind == "noul" else int(p.argmax())
    return pred == d.label


def temperatures(ds, probs) -> dict[str, float]:
    return {
        kind: fit_temperature(
            np.array([d.label for d in ds if d.kind == kind]),
            pad([p for d, p in zip(ds, probs, strict=True) if d.kind == kind]),
        )
        for kind in ("choice", "noul", "score")
    }


def pad(probs: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(probs), max(len(p) for p in probs)))
    for i, p in enumerate(probs):
        out[i, : len(p)] = p
    return out


def report(ds, probs, temps) -> dict:
    scaled = [scale(p[None], temps[d.kind])[0] for d, p in zip(ds, probs, strict=True)]
    hits = np.array([correct(d, p) for d, p in zip(ds, scaled, strict=True)])
    pairs = [(d, p) for d, p in zip(ds, scaled, strict=True) if d.kind != "score"]
    conf = np.array([p.max() for p in scaled])
    return {
        "decisions": len(ds),
        "accuracy": round(float(hits.mean()), 4),
        "per_type": {
            k: round(float(hits[[d.kind == k for d in ds]].mean()), 4)
            for k in ("choice", "noul", "score")
        },
        "brier": round(float(np.mean([((p - d.gold) ** 2).sum() for d, p in pairs])), 4),
        "soft_accuracy": round(float(np.mean([(p * d.gold).sum() for d, p in pairs])), 4),
        "ece": round(float(ece_binary(conf, hits)), 4),
    }


def ece_binary(conf: np.ndarray, hits: np.ndarray, bins: int = 15) -> float:
    """15 equal-width bins of top probability against correctness, as the benchmark reports."""
    which = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(
        abs(conf[which == b].mean() - hits[which == b].mean()) * (which == b).mean()
        for b in range(bins)
        if (which == b).any()
    )


def train(init, data_dir, epochs, seed, batch_size=16, lr=3e-5):
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    fit, val, test = load(data_dir)
    model, tok = reader(init, data_dir)
    model = model.to(dev)
    optim = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.scorer.parameters(), "lr": 4 * lr},
        ],
        weight_decay=0.01,
    )
    steps = epochs * (len(fit) // batch_size)
    sched = get_linear_schedule_with_warmup(optim, steps // 10, steps)
    best, history, start = None, [], time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(fit))
        for b in range(0, len(order) - batch_size + 1, batch_size):
            items = [
                (fit[i], list(rng.permutation(len(fit[i].keys)))) for i in order[b : b + batch_size]
            ]
            logits = forward(model, tok, items, dev)
            loss = torch.stack(
                [
                    -(
                        torch.tensor(d.gold, device=dev, dtype=torch.float32) * x.log_softmax(-1)
                    ).sum()
                    for (d, _), x in zip(items, logits, strict=True)
                ]
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
        probs = predict(model, tok, val)
        acc = float(np.mean([correct(d, p) for d, p in zip(val, probs, strict=True)]))
        history.append(
            {
                "epoch": epoch + 1,
                "val_accuracy": round(acc, 4),
                "seconds": round(time.time() - start),
            }
        )
        print(history[-1], flush=True)
        if best is None or acc > best[0]:
            best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    val_probs = predict(model, tok, val)
    temps = temperatures(val, val_probs)
    return {
        "init": init,
        "seed": seed,
        "epochs": epochs,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": dev,
        "split": {"fit": len(fit), "val": len(val), "test": len(test)},
        "history": history,
        "temperatures": {k: round(v, 3) for k, v in temps.items()},
        "val": report(val, val_probs, temps),
        "test": report(test, predict(model, tok, test), temps),
    }


def main():
    from backend.config import Settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ettin-150m")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = train(args.init, Settings().data_dir, args.epochs, args.seed)
    path = Path("docs/results/s2") / f"typed-{args.init}-e{args.epochs}-s{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test"]))


if __name__ == "__main__":
    main()
