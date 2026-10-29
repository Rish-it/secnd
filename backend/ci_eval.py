"""Reproduce every deterministic field of one cheap, committed evaluation card."""

import json
from difflib import unified_diff
from pathlib import Path

from backend.eval import card

EXPECTED = Path("docs/results/s1/card-ipsa-merchant_memory-b200.json")
VOLATILE = {"predict_ms", "peak_rss_bytes", "git_sha"}


def compare(expected, actual):
    expected = {k: v for k, v in expected.items() if k not in VOLATILE}
    actual = {k: v for k, v in actual.items() if k not in VOLATILE}
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
        compare(json.loads(EXPECTED.read_text()), card("ipsa", "merchant_memory", 200, 0))
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print("Fixed IPSA evaluation matches every deterministic field (budget 200, seed 0).")


if __name__ == "__main__":
    main()
