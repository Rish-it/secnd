"""Optional language-model access, separate from the decision path."""

import argparse
import json
from typing import Annotated

import httpx
from pydantic import BaseModel, Field, model_validator

from backend.config import Settings

TokenCount = Annotated[int, Field(ge=0, strict=True)]


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


def parse_usage(data: dict) -> Usage:
    usage = data["usage"]
    return Usage(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    )


def smoke(settings: Settings, *, transport: httpx.BaseTransport | None = None) -> SmokeResult:
    if not settings.muse_api_key.get_secret_value():
        raise MuseError("Set SECND_MUSE_API_KEY in .env.")
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
        raise MuseError("Provider request failed; response details are withheld.") from None
    if response.status_code != 200:
        raise MuseError(f"Provider returned HTTP {response.status_code}; details are withheld.")
    try:
        data = response.json()
        return SmokeResult(status=200, model=data["model"], usage=parse_usage(data))
    except (KeyError, TypeError, ValueError):
        raise MuseError("Provider returned invalid usage metadata; details are withheld.") from None


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
