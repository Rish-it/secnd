from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.config import Settings


def test_provider_defaults_and_secret_redaction(monkeypatch):
    for name in ("SECND_MUSE_API_KEY", "SECND_MUSE_BASE_URL", "SECND_MUSE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.muse_base_url == "https://inference.baseten.co/v1"
    assert settings.muse_model == "meta-models/muse-spark-open-1026"
    assert not settings.muse_api_key.get_secret_value()
    monkeypatch.setenv("SECND_MUSE_API_KEY", "test-only-key")
    settings = Settings(_env_file=None)
    assert "test-only-key" not in repr(settings)
    assert "test-only-key" not in settings.model_dump_json()


def test_budget_reads_environment_and_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("SECND_MUSE_BUDGET_USD", "0.50")
    assert Settings(_env_file=None).muse_budget_usd == Decimal("0.50")
    for value in ("-1", "NaN", "Infinity", "101"):
        monkeypatch.setenv("SECND_MUSE_BUDGET_USD", value)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
