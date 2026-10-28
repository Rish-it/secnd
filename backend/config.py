from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECND_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".data")
    frontend_dist: Path = Path("frontend/dist")
    muse_api_key: SecretStr = SecretStr("")
    muse_base_url: str = "https://inference.baseten.co/v1"
    muse_model: str = "meta-models/muse-spark-open-1026"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "secnd.sqlite3"
