"""Public synthetic demo. No volume, secrets or model; every new container starts a fresh story."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    .add_local_python_source("backend", copy=True)
    .add_local_file(
        str(ROOT / "backend/reader-manifest.json"), "/root/backend/reader-manifest.json"
    )
    .add_local_dir(str(ROOT / "frontend/dist"), "/root/frontend/dist")
    .add_local_dir(str(ROOT / "docs/results/s3"), "/root/docs/results/s3")
    .add_local_dir(str(ROOT / "docs/results/s3r"), "/root/docs/results/s3r")
    .add_local_dir(str(ROOT / "docs/results/s2"), "/root/docs/results/s2")
)
app = modal.App("secnd-demo")


@app.function(
    image=image, cpu=1, memory=2048, min_containers=0, max_containers=1, scaledown_window=300
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os
    import tempfile

    from backend.demo_seed import create_demo

    os.chdir("/root")
    return create_demo(Path(tempfile.mkdtemp(prefix="secnd-synthetic-demo-")))
