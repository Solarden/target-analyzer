"""Server settings. Production values come from the environment (TA_* vars, pushed
to LUKS by deploy.sh); the defaults here are the zero-setup local-dev values.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=".env", extra="ignore")

    # SQLite for local dev; production points at the shared Postgres on petel.
    database_url: str = "sqlite:///./target_analyzer.db"
    secret_key: SecretStr = SecretStr("dev-insecure-change-me")
    secure_cookies: bool = False
    # sha256 of the machine bearer token (never the token itself). Unset -> ingest 503.
    ingest_token_hash: str | None = None
    allowed_ranges: str = "127.0.0.1/32,192.168.0.0/16,100.64.0.0/10"
    data_path: str = "./data"
    attachment_max_bytes: int = 25 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
