"""Fresh, provider-isolated 500-row comparison on the original fixed IPSA sample."""

import argparse
import inspect
import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal

from backend.config import Settings
from backend.data import load_dataset, split_manifest
from backend.muse import MuseClient, MuseError
from backend.muse_baseline import RESULTS, collect, digest, item, sample, summarize, write_json
from backend.providers import ledger_path, provider_settings

TASK = "muse-baseline-provider"


def write_comparison():
    paths = [
        RESULTS / "providers" / provider / "muse-baseline.json" for provider in ("meta", "baseten")
    ]
    if not all(path.exists() for path in paths):
        return
    measured = [json.loads(path.read_text()) for path in paths]
    fields = ("data_hash", "split_hash", "sample_hash", "input_hash", "protocol_hash")
    if any(measured[0][field] != measured[1][field] for field in fields):
        raise ValueError("Provider comparison provenance differs.")
    winner = min(measured, key=lambda row: (-row["accuracy"], Decimal(row["cost_usd"])))
    write_json(
        RESULTS / "provider-comparison.json",
        {
            **{field: measured[0][field] for field in fields},
            "rows": 500,
            "providers": measured,
            "default_provider": winner["provider"],
            "selection_rule": "Higher observed accuracy; lower measured cost breaks a tie.",
            "selection_limit": (
                "Default chosen on this test sample; no significance or independent "
                "winner estimate claimed."
            ),
            "combined_spend_usd": str(
                sum((Decimal(row["cost_usd"]) for row in measured), Decimal(0))
            ),
        },
    )


def run(provider, *, limit=500):
    configured = provider_settings(Settings(), provider)
    configured = configured.model_copy(update={"muse_budget_usd": Decimal(4)})
    frame = load_dataset("ipsa", configured.data_dir)
    split, test = sample(frame)
    original = json.loads((RESULTS / "muse-sample.json").read_text())
    if test["id"].tolist() != original["row_ids"]:
        raise ValueError("Comparison must use the original 500 test rows.")
    if (
        digest([item(row).model_dump(mode="json") for _, row in test.iterrows()])
        != original["input_hash"]
        or sorted(frame["label"].unique()) != original["labels"]
    ):
        raise ValueError("Comparison input or allowed labels changed.")
    manifest = split_manifest("ipsa", frame, split)
    if (
        digest(manifest["sources"]) != original["data_hash"]
        or manifest["hash"] != original["split_hash"]
    ):
        raise ValueError("Dataset or split changed.")
    if digest(inspect.getsource(MuseClient.zero_shot)) != original["protocol_hash"]:
        raise ValueError("Comparison prompt must match the original protocol.")
    plan = {
        **original,
        "provider": provider,
        "model": configured.muse_model,
        "base_url": configured.muse_base_url,
    }
    directory = RESULTS / "providers" / provider
    sha = subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], text=True).strip()
    ledger = ledger_path(RESULTS, provider)
    trials = collect(
        configured,
        plan,
        test,
        directory,
        limit=limit,
        git_sha=sha,
        ledger=ledger,
        provider=provider,
        task=TASK,
    )
    if len(trials) != 500:
        print(f"Saved {len(trials)}/500 {provider} rows; rerun to resume.")
        return
    result = {
        key: value for key, value in plan.items() if key not in ("row_ids", "labels", "base_url")
    }
    result.update(
        git_sha=sha,
        measured_at=datetime.now(UTC).isoformat(),
        **summarize(test, trials, ledger, TASK),
    )
    write_json(directory / "muse-baseline.json", result)
    write_comparison()
    print(
        json.dumps(
            {"provider": provider, "accuracy": result["accuracy"], "cost_usd": result["cost_usd"]}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("meta", "baseten"), required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("--limit must be 1..500")
    try:
        run(args.provider, limit=args.limit)
    except (MuseError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
