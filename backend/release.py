"""Preview or explicitly upload a reviewed model release; never train or export user data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from backend.schemas import EvalCard

FILES = ("README.md", "model.pkl", "card.json", "runtime.json")


def plan(folder: Path) -> dict:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Supply a regular release directory.")
    if {path.name for path in folder.iterdir()} != set(FILES):
        raise ValueError(
            "The release must contain only README.md, model.pkl, card.json, runtime.json."
        )
    hashes = {}
    for name in FILES:
        path = folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Release file {name} must be a nonempty regular file.")
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    card = EvalCard.model_validate_json((folder / "card.json").read_text())
    if not re.fullmatch(r"ren-1\.\d+", card.model_version):
        raise ValueError("A release must identify a ren-1.<n> version.")
    runtime = json.loads((folder / "runtime.json").read_text())
    labels = runtime.get("labels", [])
    if (
        runtime.get("model_id") != "secnd-ren-1"
        or not isinstance(labels, list)
        or len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("runtime.json must identify secnd-ren-1 and its ordered unique labels.")
    if runtime.get("model_version") != card.model_version:
        raise ValueError("Runtime and evaluation versions must match.")
    if not runtime.get("featurizer") or not isinstance(runtime.get("merchant_memory"), dict):
        raise ValueError("Runtime must include its featurizer and merchant-memory snapshot.")
    if any(
        type(index) is not int or not 0 <= index < len(labels)
        for index in runtime["merchant_memory"].values()
    ):
        raise ValueError("Merchant memory must reference the ordered label indices.")
    return {"model_id": "secnd-ren-1", "model_version": card.model_version, "sha256": hashes}


def publish(folder: Path, repo_id: str, *, upload=False, api=None) -> dict:
    if not re.fullmatch(r"[\w.-]+/secnd-ren-1", repo_id):
        raise ValueError("Use the intended namespace/secnd-ren-1 model repository.")
    result = {**plan(folder), "repo_id": repo_id, "mode": "preview"}
    if not upload:
        return result
    from huggingface_hub import HfApi, ModelCard

    readme = (folder / "README.md").read_text()
    if "DRAFT" in readme or "RELEASE_TODO" in readme:
        raise ValueError("Finish reviewing the draft model card before uploading.")
    if not ModelCard(readme).data.to_dict().get("license"):
        raise ValueError("The reviewed model card must declare its release license.")
    api = api or HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        allow_patterns=list(FILES),
        commit_message=f"Publish Ren {result['model_version']}",
    )
    return {**result, "mode": "uploaded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--upload", action="store_true", help="Publish the reviewed files to the Hub."
    )
    args = parser.parse_args()
    try:
        result = publish(args.folder, args.repo_id, upload=args.upload)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except Exception:
        parser.exit(1, "Release failed. No credentials are included in this error.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
