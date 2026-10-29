"""Provider selection and independent, budgeted failover for optional model calls."""

from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import httpx

from backend.config import Settings
from backend.muse import BudgetRun, MuseClient, MuseError, MuseUnavailable, Usage

Provider = Literal["meta", "baseten", "nebius"]
PROVIDERS: tuple[Provider, ...] = ("meta", "baseten", "nebius")


def provider_settings(settings: Settings, provider: Provider) -> Settings:
    if provider == "meta":
        values = dict(
            muse_api_key=settings.meta_api_key,
            muse_base_url=settings.meta_base_url,
            muse_model=settings.meta_model,
            muse_budget_usd=settings.meta_budget_usd,
        )
    elif provider == "nebius":
        values = dict(
            muse_api_key=settings.nebius_api_key,
            muse_base_url="https://api.studio.nebius.com/v1",
            muse_model="meta-models/muse-spark-open",
            muse_budget_usd=settings.nebius_budget_usd,
        )
    else:
        values = {}
    return settings.model_copy(update=values)


def ledger_path(directory: Path, provider: Provider) -> Path:
    return directory / ("spend.csv" if provider == "baseten" else f"spend-{provider}.csv")


class MuseProviders(MuseClient):
    def __init__(
        self,
        settings: Settings,
        *,
        task: str,
        rows: int,
        measured_tokens: Usage,
        directory: Path = Path("docs/results/s5"),
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = settings
        self.task, self.rows, self.measured_tokens = task, rows, measured_tokens
        self.directory, self.transport = directory, transport
        self.last_provider: Provider | None = None
        self.clients: dict[Provider, MuseClient] = {}
        self.stack = None

    def __enter__(self):
        self.stack = ExitStack()
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def _request(self, messages, *, max_tokens, tools=None):
        if self.stack is None:
            raise MuseError("Provider calls need an active context.")
        for provider in PROVIDERS:
            settings = provider_settings(self.config, provider)
            if not settings.muse_api_key.get_secret_value():
                continue
            if provider not in self.clients:
                run = self.stack.enter_context(
                    BudgetRun(
                        settings,
                        task=self.task,
                        rows=self.rows,
                        measured_tokens=self.measured_tokens,
                        provider=provider,
                        ledger_path=ledger_path(self.directory, provider),
                    )
                )
                self.clients[provider] = self.stack.enter_context(
                    MuseClient(settings, run=run, transport=self.transport)
                )
            self.last_provider = provider
            try:
                return self.clients[provider]._request(messages, max_tokens=max_tokens, tools=tools)
            except MuseUnavailable:
                continue
        raise MuseUnavailable("No configured provider is available.")
