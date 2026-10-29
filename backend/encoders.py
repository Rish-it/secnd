"""Fetch and verify the pinned Ren encoder: `python -m backend.encoders`."""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
from huggingface_hub import snapshot_download

from backend.config import Settings

MANIFEST = Path(__file__).with_name("encoder-ren-1.0.0.json")
BUNDLED = Path("/opt/secnd/encoders")


def _verify(path: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        file = path / name
        if not file.is_file():
            raise ValueError(f"Encoder file missing: {name}")
        with file.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Encoder sha256 mismatch: {name}")


def fetch(name: str, data_dir: Path) -> Path:
    """Publish a complete, verified snapshot; reuse only verified local files."""
    if name != "pt-minilm":
        raise ValueError(f"No pinned encoder: {name}")
    pin = json.loads(MANIFEST.read_text())
    settings = Settings()
    if (settings.encoder_repo, settings.encoder_revision) != (pin["repo"], pin["revision"]):
        raise ValueError("Encoder repo/revision must match the committed release manifest.")
    root = Path(data_dir) / "encoders"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    with FileLock(str(root / f".{name}.lock")):
        if target.exists():
            _verify(target, pin["sha256"])
            return target
        with TemporaryDirectory(prefix=f".{name}-", dir=root) as temp:
            staged = Path(temp) / name
            if (BUNDLED / name).is_dir():
                for file in pin["sha256"]:
                    destination = staged / file
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(BUNDLED / name / file, destination)
            else:
                snapshot_download(
                    repo_id=pin["repo"],
                    revision=pin["revision"],
                    allow_patterns=list(pin["sha256"]),
                    local_dir=staged,
                    token=False,
                )
            _verify(staged, pin["sha256"])
            staged.rename(target)
    return target


if __name__ == "__main__":
    fetch("pt-minilm", Settings().data_dir)
