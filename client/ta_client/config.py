"""Client settings. See implementation.md §9 and §12.1.

Read from the user's home, never from the checkout: this repo is public, so the bearer
token has nowhere to live in it and a default pointing at a real server would be a
deployment detail in a public file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path.home() / ".config" / "target-analyzer" / "env"


class Settings(BaseSettings):
    # extra="ignore" because the server's TA_* vars share this namespace: a checkout that
    # runs both sides must not have the client refuse to start over TA_DATABASE_URL.
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=ENV_FILE, extra="ignore")

    # Unset is the off switch, not a broken install: a fresh clone processes photos and
    # leaves them queued, which is exactly what the outbox is for.
    server_url: str = ""
    ingest_token: SecretStr = SecretStr("")
    ship_original: bool = True
    outbox_path: Path = Path.home() / ".target-analyzer" / "outbox"
    # Covers the upload: several megabytes over bad wifi exceeds httpx's 5 s default, and
    # the folder then re-queues forever while looking like a network fault.
    ship_timeout: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
