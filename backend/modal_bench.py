"""Run the GPU bake-off on a rented CUDA GPU: `uvx modal run backend/modal_bench.py --dataset mtop`.

Same code and seeds as `python -m backend.bench --gpu`; only the device differs. Data and model
downloads are cached on a Modal volume, and the results CSV is copied back to docs/results/s2/.
"""

import subprocess
import sys
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
def pretune(encoder: str, steps: int = 3000) -> str:
    """Tune `encoder` on TOPv2 (MTOP texts removed) and keep it on the volume."""
    import torch

    assert torch.cuda.is_available(), "no CUDA device: refusing to tune on CPU"
    run = Path("/vol/run")
    run.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "-m", "backend.finetune", "--pretune", encoder, "--steps", str(steps)],
        cwd=run,
        check=True,
    )
    volume.commit()
    out = f"pt-{encoder}" if steps == 3000 else f"pt-{encoder}-s{steps}"
    return (run / f"docs/results/s2/pretune-{out}.json").read_text()


@app.function(gpu="A10G", volumes={"/vol": volume}, timeout=4 * 3600)
def incontext(encoder: str, steps: int) -> str:
    """Train the in-context reader on the volume's training sets; score it on the held-out ones."""
    import torch

    assert torch.cuda.is_available(), "no CUDA device: refusing to train on CPU"
    run = Path("/vol/run")
    run.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "-m", "backend.incontext", "--encoder", encoder, "--steps", str(steps)],
        cwd=run,
        check=True,
    )
    volume.commit()
    return (run / f"docs/results/s2/incontext-ic-{encoder}-s{steps}-near.json").read_text()


@app.function(gpu="A10G", volumes={"/vol": volume}, timeout=2 * 3600)
def reader_budgets(out: str) -> str:
    """Re-score a saved reader at each labelled-history budget."""
    import torch

    assert torch.cuda.is_available(), "no CUDA device: refusing to score on CPU"
    run = Path("/vol/run")
    subprocess.run(["python", "-m", "backend.incontext", "--evaluate", out], cwd=run, check=True)
    return (run / f"docs/results/s2/incontext-{out}-budgets.json").read_text()


@app.local_entrypoint()
def budgets(out: str = "ic-ettin-32m-s6000-near"):
    path = ROOT / f"docs/results/s2/incontext-{out}-budgets.json"
    path.write_text(reader_budgets.remote(out))
    print(f"saved {path}")


@app.local_entrypoint()
def reader(encoder: str = "ettin-32m", steps: int = 6000):
    out = ROOT / f"docs/results/s2/incontext-ic-{encoder}-s{steps}-near.json"
    out.write_text(incontext.remote(encoder, steps))
    print(f"saved {out}")


@app.local_entrypoint()
def tuned(encoder: str = "minilm", steps: int = 3000):
    """Tune remotely, then fetch the encoder: score it locally on MPS with `backend.eval`."""
    out = f"pt-{encoder}" if steps == 3000 else f"pt-{encoder}-s{steps}"
    (ROOT / f"docs/results/s2/pretune-{out}.json").write_text(pretune.remote(encoder, steps))
    folder = ROOT / ".data/encoders"
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            "secnd-bench",
            f"/data/encoders/{out}",
            str(folder),
        ],
        check=True,
    )


@app.local_entrypoint()
def main(dataset: str = "mtop"):
    out = ROOT / f"docs/results/s2/bakeoff-{dataset}-gpu.csv"
    out.write_text(bench.remote(dataset))
    print(f"saved {out}")
