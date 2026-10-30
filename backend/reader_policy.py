"""Read frozen reader operating points; serving never fits or adjusts them."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("reader-manifest.json")


class ReaderUnavailable(RuntimeError):
    pass


def finite(value, *, lower=0, upper=None):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= lower
        and (upper is None or value <= upper)
    )


def verified_json(relative, checksum):
    raw = (ROOT / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ReaderUnavailable("Committed reader results failed their checksum check.")
    return json.loads(raw)


@dataclass(frozen=True)
class ReaderPolicy:
    temperature: float | None = None
    threshold: float | None = None
    floor: float | None = None

    @classmethod
    def from_result(cls, result):
        result = result if isinstance(result, dict) else {}
        floor = result.get("floor")
        return cls(
            result.get("temperature"),
            result.get("threshold"),
            floor.get("floor") if isinstance(floor, dict) else None,
        )

    @property
    def complete(self):
        return (
            finite(self.temperature)
            and self.temperature > 0
            and finite(self.threshold, upper=1)
            and finite(self.floor, upper=1)
        )

    def accepts(self, confidence, support):
        return bool(
            self.complete
            and finite(confidence, upper=1)
            and finite(support, lower=-1, upper=1)
            and confidence >= self.threshold
            and support >= self.floor
        )


class ReaderArtifact:
    def __init__(self, options):
        try:
            entry = json.loads(MANIFEST.read_text())["models"][options.model_id]
            self.revision = entry["revision"]
            if options.revision is not None and options.revision != self.revision:
                raise ReaderUnavailable("Configured reader revision has no audited serving policy.")
            self.hashes = entry["checkpoint_sha256"]
            self.result = verified_json(entry["policy_file"], entry["policy_sha256"])
            self.evaluation = verified_json(entry["evaluation_file"], entry["evaluation_sha256"])
            if self.result.get("model") != options.model_id:
                raise ReaderUnavailable("Reader result model does not match the configured model.")
            self.policy = ReaderPolicy.from_result(
                self.result.get("results", {}).get(str(options.examples))
            )
            self.policy_file = entry["policy_file"]
            self.evaluation_file = entry["evaluation_file"]
        except (OSError, KeyError, ValueError, TypeError):
            raise ReaderUnavailable(
                "Committed reader configuration or results are unavailable."
            ) from None

    def verify_checkpoint(self, folder):
        for relative, checksum in self.hashes.items():
            path = folder / relative
            if not path.is_file():
                raise ReaderUnavailable(
                    "Configured Ren reader checkpoint is missing. Install its files."
                )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != checksum:
                raise ReaderUnavailable(
                    "Configured Ren reader checkpoint has an unexpected revision."
                )

    def benchmark(self, options):
        budgets = self.evaluation.get("budgets", {}).get("mtop", {})
        key = f"reader-{options.examples}"
        measured = (
            budgets.get(key, {}) if options.examples == 0 else budgets.get("all", {}).get(key, {})
        )
        result = self.result.get("results", {}).get(str(options.examples), {})
        audit = result.get("attacks", {}).get("baseline", {})
        return {
            "dataset": "MTOP",
            "history_rows": self.evaluation.get("split", {}).get("mtop", {}).get("train"),
            "accuracy": measured.get("accuracy"),
            "confidence_only_coverage": measured.get("certified_coverage_test"),
            "confidence_only_precision": measured.get("precision_auto_test"),
            "with_floor_coverage": audit.get("coverage"),
            "with_floor_precision": audit.get("precision"),
            "with_floor_sample": len(audit.get("records", [])),
            "evaluation_file": self.evaluation_file,
            "floor_result_file": self.policy_file,
        }
