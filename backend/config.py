from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECND_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".data")
    frontend_dist: Path = Path("frontend/dist")
    muse_api_key: SecretStr = SecretStr("")
    muse_base_url: str = "https://inference.baseten.co/v1"
    muse_model: str = "meta-models/muse-spark-open-1026"
    muse_budget_usd: Decimal = Field(default=Decimal("10"), ge=0, le=100)
    meta_api_key: SecretStr = SecretStr("")
    meta_base_url: str = "https://api.meta.ai/v1"
    meta_model: str = "muse-spark-1.3"
    meta_budget_usd: Decimal = Field(default=Decimal("10"), ge=0, le=30)
    nebius_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="NEBIUS_API_KEY")
    nebius_budget_usd: Decimal = Field(default=Decimal("10"), ge=0, le=25)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "secnd.sqlite3"
