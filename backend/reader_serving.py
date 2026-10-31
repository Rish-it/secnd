"""Configurable Ren reader over a workspace's history, with frozen support gating."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from threading import RLock

import numpy as np

from backend.config import ReaderOptions
from backend.reader_policy import ReaderArtifact, ReaderUnavailable, finite
from backend.schemas import Decision, Item, Neighbour, ReaderDecision, Reasons

MODEL = "ic-ettin-150m-s6000-near"


def history_version(labels, history):
    payload = json.dumps([labels, history], sort_keys=True)
    return "ren-reader-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class CPUReader:
    # Keep the existing name/import for onboarding; the device is now explicit configuration.
    def __init__(self, data_dir: Path, options: ReaderOptions | None = None):
        self.options = options or ReaderOptions()
        self.artifact = ReaderArtifact(self.options)
        self.policy = self.artifact.policy
        self.folder = data_dir / "encoders" / self.options.model_id
        self.device = self.options.device
        self._lock = RLock()
        self.artifact.verify_checkpoint(self.folder)
        try:
            self._load()
        except ReaderUnavailable:
            raise
        except Exception:
            raise ReaderUnavailable(
                f"Ren reader {self.options.model_id} could not load on {self.device}. "
                "Check the configured checkpoint, device and local MiniLM retrieval files."
            ) from None
        self._history_key, self._vectors = None, None

    def _load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        from backend.features import EMBEDDERS
        from backend.incontext import Reader

        if self.device == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise ReaderUnavailable(
                "Configured CUDA reader requires an available BF16-capable GPU."
            )
        torch.set_num_threads(4)
        self.model = Reader(str(self.folder / "encoder")).to(self.device).eval()
        self.model.scorer.load_state_dict(
            torch.load(self.folder / "scorer.pt", map_location=self.device, weights_only=True)
        )
        self.tok = AutoTokenizer.from_pretrained(self.folder / "encoder", local_files_only=True)
        repo, revision = EMBEDDERS["minilm"]
        self.embedder = SentenceTransformer(
            repo, revision=revision, device="cpu", local_files_only=True
        )

    def validate_labels(self, labels):
        from backend import incontext as ic

        if not 2 <= len(labels) <= ic.MAX_OPTIONS or len(set(labels)) != len(labels):
            raise ValueError("The reader needs 2–128 distinct labels.")
        if any(self.tok.mask_token in label or self.tok.sep_token in label for label in labels):
            raise ValueError("Label names cannot contain reserved reader tokens.")
        task = ic.Task("workspace", ic.INTENT, labels, [], [])
        ids = ic.render(self.tok, task, "", list(range(len(labels))), [])
        if ids.count(self.tok.mask_token_id) != len(labels) or len(ids) > ic.MAX_LEN - 256:
            raise ValueError("Shorten the label names so every option leaves room for a message.")

    def version(self, labels, history):
        identity = [
            self.options.model_id,
            self.artifact.revision,
            self.artifact.policy_file,
            self.options.examples,
            history_version(labels, history),
        ]
        return "ren-reader-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]

    def predict(self, labels, history, frame, *, allow_auto=False, decision=None):
        import torch

        from backend import incontext as ic

        self.validate_labels(labels)
        if any(h["label"] not in labels for h in history):
            raise ValueError("History rows must carry a workspace label.")
        task = ic.Task("workspace", ic.INTENT, labels, [], [])
        history_key = history_version(labels, history)
        version = self.version(labels, history)
        with self._lock, torch.inference_mode():
            if history and history_key != self._history_key:
                self._vectors = np.asarray(
                    self.embedder.encode([h["text"] for h in history], normalize_embeddings=True)
                )
                self._history_key = history_key
            decisions = []
            for row in frame.itertuples():
                neighbours, similarity = [], None
                if history:
                    query = self.embedder.encode([row.text], normalize_embeddings=True)[0]
                    similarity = self._vectors @ query
                    selected = [
                        int(i)
                        for i in np.argsort(-similarity, kind="stable")
                        if history[i]["text"] != row.text
                    ][: self.options.examples]
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
                if decision is not None:
                    from dataclasses import replace

                    from backend.typed import render

                    examples = "\n".join(
                        f"Example: {n.text} => {n.label}" for n in neighbours[::-1]
                    )
                    typed_decision = replace(decision, state=f"{examples}\n{row.text}")
                    ids = render(self.tok, typed_decision, list(range(len(labels))))
                    if ids.count(self.tok.mask_token_id) != len(labels):
                        raise ValueError("Every decision option must fit in the reader input.")
                packed = ic.batch(self.tok, [(ids, 0)], self.device)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else nullcontext()
                )
                with autocast:
                    raw = torch.softmax(self.model(*packed[:4]).float(), -1)[0]
                p = raw.double().cpu().numpy()
                # Apply the same probability-temperature transform as the audit, without fitting.
                if finite(self.policy.temperature) and self.policy.temperature > 0:
                    logp = np.log(np.clip(p, 1e-12, 1)) / self.policy.temperature
                    p = np.exp(logp - logp.max())
                p /= p.sum()
                index = int(p.argmax())
                carrying = [i for i, h in enumerate(history) if h["label"] == labels[index]]
                # Amendment 2 uses ALL history rows with this intent, not only retrieved examples.
                support = float(max(similarity[i] for i in carrying)) if carrying else None
                if support is not None:
                    support = float(np.clip(support, -1, 1))
                # These cutoffs were measured with FULL MTOP history only. Smaller/invented
                # histories have no inherited certification; callers must authorize the scope.
                auto = bool(
                    allow_auto
                    and len(context) == self.options.examples
                    and len(ids) < ic.MAX_LEN
                    and self.policy.accepts(float(p[index]), support)
                )
                decisions.append(
                    Decision(
                        item_id=str(row.id),
                        label=labels[index],
                        probabilities=dict(zip(labels, p.tolist(), strict=True)),
                        prob=float(p[index]),
                        auto=auto,
                        threshold=self.policy.threshold if self.policy.complete else None,
                        model_version=version,
                        reasons=Reasons(neighbours=neighbours),
                        item=Item(item_id=str(row.id), text=row.text),
                        reader=ReaderDecision(
                            model_id=self.options.model_id,
                            revision=self.artifact.revision,
                            device=self.device,
                            examples=len(context),
                            support=support,
                            support_floor=self.policy.floor if self.policy.complete else None,
                            temperature=self.policy.temperature
                            if finite(self.policy.temperature)
                            else None,
                            policy_complete=self.policy.complete,
                            scope_authorized=allow_auto,
                        ),
                    ).model_dump(mode="json")
                )
            return decisions


class SavedDecisionReader(CPUReader):
    """Local saved reader, with no inherited benchmark certification."""

    def __init__(self, folder, model_id, device="cpu"):
        from types import SimpleNamespace

        from backend.reader_policy import ReaderPolicy

        self.folder = Path(folder)
        files = sorted(p for p in self.folder.rglob("*") if p.is_file())
        if not (self.folder / "scorer.pt").is_file() or not (self.folder / "encoder").is_dir():
            raise ReaderUnavailable("Meta decision reader needs encoder/ and scorer.pt.")
        fingerprint = hashlib.sha256()
        for path in files:
            fingerprint.update(str(path.relative_to(self.folder)).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    fingerprint.update(block)
        self.options = SimpleNamespace(model_id=model_id, examples=8)
        self.artifact = SimpleNamespace(
            revision=fingerprint.hexdigest(), policy_file="workspace-only"
        )
        self.policy, self.device, self._lock = ReaderPolicy(), device, RLock()
        try:
            self._load()
        except Exception:
            raise ReaderUnavailable(
                "Meta decision reader could not load its local files."
            ) from None
        self._history_key, self._vectors = None, None
