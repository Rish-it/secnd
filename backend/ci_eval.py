"""Reproduce every deterministic field of one cheap, committed evaluation card."""

import json
from difflib import unified_diff
from pathlib import Path

from backend.eval import card

EXPECTED = Path("docs/results/s8/eval-ipsa-merchant_memory-b200.json")
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
        measured = card("ipsa", "merchant_memory", 200, 0)
        compare(
            json.loads(EXPECTED.read_text()),
            {k: v for k, v in measured.items() if k not in UNSTABLE},
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print("Fixed IPSA evaluation matches every deterministic field (budget 200, seed 0).")


if __name__ == "__main__":
    main()
