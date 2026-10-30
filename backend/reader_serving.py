"""CPU reader suggestions over a workspace's own editable history."""

import hashlib
import json
from pathlib import Path
from threading import RLock

import numpy as np

from backend.schemas import Decision, Item, Neighbour, Reasons

MODEL = "ic-ettin-150m-s6000-near"


class ReaderUnavailable(RuntimeError):
    pass


def history_version(labels, history):
    payload = json.dumps([labels, history], sort_keys=True)
    return "ren-reader-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class CPUReader:
    def __init__(self, data_dir: Path):
        self.folder = data_dir / "encoders" / MODEL
        if not (self.folder / "scorer.pt").is_file():
            raise ReaderUnavailable(
                "Install the Ren reader checkpoint before starting with labels."
            )
        self._lock = RLock()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer

            from backend.features import EMBEDDERS
            from backend.incontext import Reader

            torch.set_num_threads(4)
            self.model = Reader(str(self.folder / "encoder")).to("cpu").eval()
            self.model.scorer.load_state_dict(
                torch.load(self.folder / "scorer.pt", map_location="cpu", weights_only=True)
            )
            self.tok = AutoTokenizer.from_pretrained(self.folder / "encoder")
            repo, revision = EMBEDDERS["minilm"]
            self.embedder = SentenceTransformer(repo, revision=revision, device="cpu")
        except Exception:
            raise ReaderUnavailable(
                "The CPU reader could not load. Check its model files."
            ) from None
        self._history_key, self._vectors = None, None

    def validate_labels(self, labels):
        from backend import incontext as ic

        if any(self.tok.mask_token in label or self.tok.sep_token in label for label in labels):
            raise ValueError("Label names cannot contain reserved reader tokens.")
        task = ic.Task("workspace", ic.INTENT, labels, [], [])
        ids = ic.render(self.tok, task, "", list(range(len(labels))), [])
        if ids.count(self.tok.mask_token_id) != len(labels) or len(ids) > ic.MAX_LEN - 256:
            raise ValueError("Shorten the label names so every option leaves room for a message.")

    def predict(self, labels, history, frame):
        import torch

        from backend import incontext as ic

        task = ic.Task("workspace", ic.INTENT, labels, [], [])
        version = history_version(labels, history)
        with self._lock, torch.inference_mode():
            if history and version != self._history_key:
                self._vectors = self.embedder.encode(
                    [h["text"] for h in history], normalize_embeddings=True
                )
                self._history_key = version
            decisions = []
            for row in frame.itertuples():
                neighbours = []
                if history:
                    query = self.embedder.encode([row.text], normalize_embeddings=True)[0]
                    similarity = self._vectors @ query
                    selected = [
                        int(i)
                        for i in np.argsort(-similarity, kind="stable")
                        if history[i]["text"] != row.text
                    ][:8]
                    neighbours = [
                        Neighbour(
                            item_id=history[i]["id"],
                            text=history[i]["text"],
                            label=history[i]["label"],
                            similarity=float(np.clip(similarity[i], 0, 1)),
                        )
                        for i in selected
                    ]
                context = [(n.text, labels.index(n.label)) for n in neighbours[::-1]]
                ids = ic.render(self.tok, task, row.text, list(range(len(labels))), context)
                packed = ic.batch(self.tok, [(ids, 0)], "cpu")
                p = torch.softmax(self.model(*packed[:4]).float(), -1)[0].double().numpy()
                p /= p.sum()
                index = int(p.argmax())
                decisions.append(
                    Decision(
                        item_id=str(row.id),
                        label=labels[index],
                        probabilities=dict(zip(labels, p.tolist(), strict=True)),
                        prob=float(p[index]),
                        # Invented history supplies no independent calibration labels.
                        auto=False,
                        threshold=None,
                        model_version=version,
                        reasons=Reasons(neighbours=neighbours),
                        item=Item(item_id=str(row.id), text=row.text),
                    ).model_dump(mode="json")
                )
            return decisions
