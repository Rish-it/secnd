"""Turn a loaded frame into a sparse feature matrix: text features plus a numeric block."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDERS = {
    "potion": ("minishlab/potion-base-8M", "bf8b056651a2c21b8d2565580b8569da283cab23"),
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "mpnet": (
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
    "bge": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
}
FEATURIZERS = ("char_tfidf", "potion", "minilm", "stack")
# Bigger encoders and their fine-tuned versions (`ft-<encoder>`); the GPU grid runs these.
GPU_FEATURIZERS = ("mpnet", "bge", "ft-minilm", "ft-mpnet", "ft-bge")

_hashing = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), n_features=2**18, alternate_sign=False
)
_models = {}
_loaded = {}  # cache file -> {sha256(text): vector}, read from disk once per process


def char_tfidf(texts: pd.Series) -> sparse.csr_matrix:
    """Stateless hashed character 3-5 grams, so rows never depend on the batch they arrive in."""
    return _hashing.transform(texts.fillna(""))


def _encode(name: str, texts: list[str]) -> np.ndarray:
    if name not in _models:
        from sentence_transformers import SentenceTransformer

        repo, revision = EMBEDDERS[name]
        _models[name] = SentenceTransformer(repo, revision=revision, device="cpu")
    return _models[name].encode(texts, batch_size=256, normalize_embeddings=True)


def embed(name: str, texts: pd.Series, cache_dir: Path | None = None) -> np.ndarray:
    """Sentence embeddings, cached on disk by sha256(text) + featurizer id."""
    texts = texts.fillna("").tolist()
    keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    path = cache_dir / f"{name}.npz" if cache_dir else None
    cache = _loaded.setdefault(path, {})
    if path and not cache and path.exists():
        stored = np.load(path)
        cache.update(zip(stored["keys"], stored["vectors"], strict=True))
    missing = list(dict.fromkeys(key for key in keys if key not in cache))
    if missing:
        first = dict(zip(keys, texts, strict=True))
        cache.update(zip(missing, _encode(name, [first[key] for key in missing]), strict=True))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_suffix(".tmp.npz")
            np.savez(staged, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
            os.replace(staged, path)  # readers in other processes never see a half-written file
    return np.stack([cache[key] for key in keys]).astype(np.float32)


def numeric(frame: pd.DataFrame) -> np.ndarray:
    """Signed log amount, log |amount|, sign, presence flags, one-hot weekday, day of month."""
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    date = pd.to_datetime(frame["date"], errors="coerce")
    has_amount, has_date = amount.notna().to_numpy(), date.notna().to_numpy()
    a = amount.fillna(0).to_numpy(float)
    log = np.log1p(np.abs(a))
    weekday = np.zeros((len(frame), 7))
    weekday[np.flatnonzero(has_date), date[has_date].dt.day_of_week.to_numpy()] = 1
    day = np.where(has_date, (date.dt.day.fillna(1).to_numpy() - 1) / 30, 0)
    columns = [np.sign(a) * log, log, np.sign(a), has_amount, has_date, day]
    return np.column_stack([*columns, weekday]).astype(np.float32)


def featurize(frame: pd.DataFrame, featurizer: str, cache_dir: Path | None = None, encoder=None):
    """Rows of `frame` as a CSR matrix: the featurizer's text block, then the numeric block.

    `encoder` is a model tuned for one training set; its vectors are never cached.
    """
    if encoder is not None:
        texts = frame["text"].fillna("").tolist()
        text = sparse.csr_matrix(encoder.encode(texts, batch_size=256, normalize_embeddings=True))
    elif featurizer == "char_tfidf":
        text = char_tfidf(frame["text"])
    elif featurizer == "stack":
        text = sparse.hstack([char_tfidf(frame["text"]), embed("potion", frame["text"], cache_dir)])
    else:
        text = sparse.csr_matrix(embed(featurizer, frame["text"], cache_dir))
    return sparse.hstack([text, numeric(frame)], format="csr")
