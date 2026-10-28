"""Optional language-model access, separate from the decision path."""

import argparse
import csv
import fcntl
import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import BaseModel, Field, model_validator

from backend.config import Settings

TokenCount = Annotated[int, Field(ge=0, strict=True)]
SPEND_PATH = Path("docs/results/s5/spend.csv")
SPEND_FIELDS = [
    "time",
    "task",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cost_usd",
    "status",
]


class Usage(BaseModel):
    prompt_tokens: TokenCount
    completion_tokens: TokenCount
    reasoning_tokens: TokenCount = 0
    cached_tokens: TokenCount = 0

    @model_validator(mode="after")
    def check_details(self):
        if (
            self.reasoning_tokens > self.completion_tokens
            or self.cached_tokens > self.prompt_tokens
        ):
            raise ValueError("Token details must be included in their parent totals.")
        return self


class SmokeResult(BaseModel):
    status: int
    model: str
    usage: Usage


class MuseError(RuntimeError):
    pass


def usage_cost(usage: Usage) -> Decimal:
    # Reasoning is already part of completion_tokens; cached input is charged conservatively.
    return (
        usage.prompt_tokens * Decimal("1.25") + usage.completion_tokens * Decimal("4.25")
    ) / Decimal(1_000_000)


class BudgetRun:
    """Serialize paid runs against an append-only spend ledger on one host."""

    def __init__(
        self,
        settings: Settings,
        *,
        task: str,
        rows: int,
        measured_tokens: Usage,
        ledger_path: Path = SPEND_PATH,
    ):
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", task) or rows < 1:
            raise ValueError("A run needs a safe task name and a positive row count.")
        self.task = task
        self.budget = settings.muse_budget_usd
        self.estimate = rows * usage_cost(measured_tokens)
        self.ledger_path = ledger_path
        self.usage = Usage(prompt_tokens=0, completion_tokens=0)
        self.cost_usd = Decimal(0)
        self.previous_spend = Decimal(0)
        self.estimated_usage = False
        self._file = None

    def __enter__(self):
        if self.estimate > self.budget:
            raise MuseError("Estimated run cost exceeds SECND_MUSE_BUDGET_USD.")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.ledger_path.open("a+", newline="")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX)
            self._file.seek(0)
            reader = csv.DictReader(self._file)
            if reader.fieldnames is not None and reader.fieldnames != SPEND_FIELDS:
                raise MuseError("Spend ledger has an invalid header.")
            for row in reader:
                cost = Decimal(row["cost_usd"])
                if not cost.is_finite() or cost < 0:
                    raise MuseError("Spend ledger has an invalid cost.")
                self.previous_spend += cost
            if self.previous_spend + self.estimate > Decimal(30):
                raise MuseError("Estimated run cost exceeds the $30 total spending limit.")
        except (InvalidOperation, KeyError, TypeError):
            self._file.close()
            self._file = None
            raise MuseError("Spend ledger is invalid; refusing a paid run.") from None
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def check(self, usage: Usage):
        if self._file is None:
            raise MuseError("A paid request needs an active budget run.")
        projected = self.cost_usd + usage_cost(usage)
        if projected > self.budget or self.previous_spend + projected > Decimal(30):
            raise MuseError("The next request would exceed the spending limit.")

    def record(self, usage: Usage, *, estimated: bool = False):
        if self._file is None:
            raise MuseError("Cannot record spend outside an active run.")
        self.usage = Usage(
            **{
                field: getattr(self.usage, field) + value
                for field, value in usage.model_dump().items()
            }
        )
        self.cost_usd += usage_cost(usage)
        self.estimated_usage |= estimated

    def __exit__(self, error_type, error, traceback):
        try:
            self._file.seek(0, 2)
            writer = csv.DictWriter(self._file, fieldnames=SPEND_FIELDS)
            if self._file.tell() == 0:
                writer.writeheader()
            status = "completed" if error_type is None else "failed"
            if self.estimated_usage:
                status += "_estimated"
            writer.writerow(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "task": self.task,
                    **self.usage.model_dump(),
                    "cost_usd": str(self.cost_usd),
                    "status": status,
                }
            )
            self._file.flush()
        finally:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def parse_usage(data: dict) -> Usage:
    usage = data["usage"]
    return Usage(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    )


def smoke(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    ledger_path: Path = SPEND_PATH,
) -> SmokeResult:
    if not settings.muse_api_key.get_secret_value():
        raise MuseError("Set SECND_MUSE_API_KEY in .env.")
    reservation = Usage(prompt_tokens=1024, completion_tokens=64)
    with BudgetRun(
        settings, task="smoke", rows=1, measured_tokens=reservation, ledger_path=ledger_path
    ) as run:
        return _smoke_request(settings, run, reservation, transport)


def _smoke_request(settings, run, reservation, transport):
    run.check(reservation)
    try:
        with httpx.Client(timeout=30, transport=transport) as client:
            response = client.post(
                settings.muse_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + settings.muse_api_key.get_secret_value()},
                json={
                    "model": settings.muse_model,
                    "messages": [
                        {"role": "user", "content": "Reply with exactly one word: ready."}
                    ],
                    "max_tokens": 64,
                    "reasoning_effort": "low",
                },
            )
    except httpx.HTTPError:
        run.record(reservation, estimated=True)
        raise MuseError("Provider request failed; response details are withheld.") from None
    if response.status_code != 200:
        run.record(reservation, estimated=True)
        raise MuseError(f"Provider returned HTTP {response.status_code}; details are withheld.")
    try:
        data = response.json()
        usage = parse_usage(data)
    except (KeyError, TypeError, ValueError):
        run.record(reservation, estimated=True)
        raise MuseError("Provider returned invalid usage metadata; details are withheld.") from None
    run.record(usage)
    return SmokeResult(status=200, model=data["model"], usage=usage)


def main():
    parser = argparse.ArgumentParser(description="Check the optional provider connection.")
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    try:
        print(json.dumps(smoke(Settings()).model_dump()))
    except MuseError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
