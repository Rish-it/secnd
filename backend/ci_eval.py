"""Reproduce every deterministic field of cheap, committed evaluation cards."""

import json
from difflib import unified_diff
from pathlib import Path

from backend.eval import card

DATASETS = ("ipsa", "banking77")
RESULTS = Path("docs/results/s8")
UNSTABLE = {"predict_ms", "peak_rss_bytes", "git_sha", "worst_classes"}


def compare(expected, actual):
    if expected != actual:
        difference = "\n".join(
            unified_diff(
                json.dumps(expected, indent=2, sort_keys=True).splitlines(),
                json.dumps(actual, indent=2, sort_keys=True).splitlines(),
                fromfile="committed",
                tofile="reproduced",
                lineterm="",
            )
        )
        raise ValueError("Fixed evaluation changed:\n" + difference)


def main():
    try:
        for dataset in DATASETS:
            measured = card(dataset, "merchant_memory", 200, 0)
            compare(
                json.loads((RESULTS / f"eval-{dataset}-merchant_memory-b200.json").read_text()),
                {k: v for k, v in measured.items() if k not in UNSTABLE},
            )
            print(
                f"Fixed {dataset.upper()} evaluation matches deterministic fields "
                "(budget 200, seed 0)."
            )
    except ValueError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
