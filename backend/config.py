from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECND_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".data")
    frontend_dist: Path = Path("frontend/dist")
    evaluation_dir: Path = Path(__file__).resolve().parents[1] / "docs/results/s3"
    featurizer: str = "pt-minilm"
    encoder_repo: str = "Secnd-AI/ren"
    encoder_revision: str = "5560f25414195f1d1224a6ae7efd19c6b7646d3f"
    meta_app_secret: SecretStr = SecretStr("")
    meta_verify_token: SecretStr = SecretStr("")
    whatsapp_token: SecretStr = SecretStr("")
    whatsapp_send_enabled: bool = False
    # Keys are platform:receiving-account-id, values are workspace ids. Unmapped is rejected.
    meta_workspace_accounts: dict[str, str] = Field(default_factory=dict)
    ads_token: SecretStr = SecretStr("")
    ads_send_enabled: bool = False
    ads_workspace_accounts: dict[str, str] = Field(default_factory=dict)
    ads_sandbox_accounts: list[str] = Field(default_factory=list)
    ads_cpa_action_type: str = Field(default="purchase", pattern=r"^[a-zA-Z0-9_.]{1,100}$")
    ads_budget_step_percent: int = Field(default=10, ge=1, le=25)
    meta_graph_version: str = Field(default="v25.0", pattern=r"^v\d+\.\d+$")
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
    # Inventing starting history pays Muse Wonder from a public endpoint: off unless the operator
    # turns it on, and capped across every workspace together.
    cold_start_paid: bool = False
    cold_start_total_usd: Decimal = Field(default=Decimal("3"), ge=0, le=10)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "secnd.sqlite3"
