"""Upload the shipped reader and Meta decision checkpoints from the volume to the Hub."""

from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==2.3.0")
volume = modal.Volume.from_name("secnd-bench")
app = modal.App("secnd-hf-reader")
REPO = "Secnd-AI/ren"
PATTERNS = ["encoder/*", "scorer.pt"]


@app.function(image=image, volumes={"/vol": volume}, timeout=3600)
def upload(reader: str, decisions: str, card: str, tag: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    for folder, target in ((reader, "reader"), (decisions, "meta")):
        api.upload_folder(
            repo_id=REPO,
            folder_path=f"/vol/data/encoders/{folder}",
            path_in_repo=target,
            allow_patterns=PATTERNS,
            commit_message=f"Add {target} checkpoint {folder}",
        )
    done = api.upload_file(
        repo_id=REPO,
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        commit_message="Model card for the reader release",
    )
    api.create_tag(REPO, tag=tag, exist_ok=False)
    return str(done)


@app.local_entrypoint()
def main(reader: str, decisions: str, card: str, tag: str):
    from dotenv import dotenv_values

    token = dotenv_values(".env").get("HF_TOKEN")
    assert token, "HF_TOKEN missing from .env"
    with_token = upload.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": token})])
    print(with_token.remote(reader, decisions, Path(card).read_text(), tag))
