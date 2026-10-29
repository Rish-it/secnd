"""Run the GPU bake-off on a rented CUDA GPU: `uvx modal run backend/modal_bench.py --dataset mtop`.

Same code and seeds as `python -m backend.bench --gpu`; only the device differs. Data and model
downloads are cached on a Modal volume, and the results CSV is copied back to docs/results/s2/.
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT))
    # the lock pins CPU torch for the serving image; the PyPI wheel carries CUDA
    .run_commands(
        "uv pip install --python /.uv/.venv/bin/python --reinstall-package torch "
        "--index-url https://pypi.org/simple torch==2.15.0"
    )
    .env({"SECND_DATA_DIR": "/vol/data", "HF_HOME": "/vol/hf"})
    .add_local_python_source("backend")
)
volume = modal.Volume.from_name("secnd-bench", create_if_missing=True)
app = modal.App("secnd-bench", image=image)


@app.function(gpu="A10G", volumes={"/vol": volume}, timeout=3 * 3600)
def bench(dataset: str) -> str:
    import torch

    assert torch.cuda.is_available(), "no CUDA device: refusing to run the GPU grid on CPU"
    run = Path("/vol/run")
    run.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "-m", "backend.bench", "--gpu", "--dataset", dataset], cwd=run, check=True
    )
    volume.commit()
    return (run / f"docs/results/s2/bakeoff-{dataset}-gpu.csv").read_text()


@app.function(gpu="A10G", volumes={"/vol": volume}, timeout=2 * 3600)
def pretune(encoder: str, budget: int = 2000) -> dict[str, str]:
    """Tune `encoder` on TOPv2 (MTOP texts removed), then write MTOP's calibrated card with it."""
    import torch

    assert torch.cuda.is_available(), "no CUDA device: refusing to tune on CPU"
    run = Path("/vol/run")
    run.mkdir(parents=True, exist_ok=True)
    model = f"pt-{encoder}+lr"
    for args in (
        ["backend.finetune", "--pretune", encoder],
        ["backend.eval", "--dataset", "mtop", "--model", model, "--budget", str(budget)],
    ):
        subprocess.run(["python", "-m", *args], cwd=run, check=True)
    volume.commit()
    paths = [
        f"docs/results/s2/pretune-pt-{encoder}.json",
        f"docs/results/s3/card-mtop-{model}-b{budget}.json",
        f"docs/results/s3/curves-mtop-{model}-b{budget}.json",
    ]
    return {path: (run / path).read_text() for path in paths}


@app.local_entrypoint()
def tuned(encoder: str = "bge"):
    for path, text in pretune.remote(encoder).items():
        (ROOT / path).write_text(text)
        print(f"saved {path}")


@app.local_entrypoint()
def main(dataset: str = "mtop"):
    out = ROOT / f"docs/results/s2/bakeoff-{dataset}-gpu.csv"
    out.write_text(bench.remote(dataset))
    print(f"saved {out}")
