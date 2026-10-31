"""Optional GPU reader endpoint. Importing this module does not deploy or invoke it."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .run_commands(
        "uv pip install --python /.uv/.venv/bin/python --reinstall-package torch "
        "--index-url https://pypi.org/simple torch==2.15.0"
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_python_source("backend", copy=True)
)
app = modal.App("secnd-reader")
volume = modal.Volume.from_name("secnd-bench")


@app.cls(
    image=image,
    gpu="A10G",
    cpu=4,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("secnd-reader")],
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    timeout=120,
    startup_timeout=300,
)
@modal.concurrent(max_inputs=1)
class ReaderEndpoint:
    @modal.enter()
    def load(self):
        import os

        from backend.reader_serving import SavedDecisionReader

        folder = Path(os.environ["SECND_READER_MODEL_PATH"]).resolve()
        if not folder.is_relative_to(Path("/vol").resolve()):
            raise ValueError("Reader model must be on secnd-bench under /vol.")
        self.reader = SavedDecisionReader(
            folder, os.environ["SECND_READER_MODEL_ID"], device="cuda"
        )

    @modal.asgi_app()
    def web(self):
        import os

        from backend.reader_endpoint import create_reader_api

        return create_reader_api(self.reader, os.environ["SECND_READER_BEARER_TOKEN"])
