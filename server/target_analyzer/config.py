"""Server settings. Production values come from the environment (TA_* vars); the
defaults here are the zero-setup local-dev values.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sentinel default for secret_key. Once the dashboard's cookie sessions land (P5)
# the app refuses to start with this value outside debug. Not a real secret (it is
# the *rejected* placeholder), so silence bandit B105.
INSECURE_DEFAULT_SECRET = "dev-insecure-change-me"  # nosec B105


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=".env", extra="ignore")

    debug: bool = False
    database_url: str = "sqlite:///./target_analyzer.db"
    secret_key: SecretStr = SecretStr(INSECURE_DEFAULT_SECRET)
    secure_cookies: bool = False
    # sha256 of the machine bearer token (never the token itself). Unset -> ingest 503.
    ingest_token_hash: str | None = None
    data_path: Path = Path("data")
    attachment_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1)

    @field_validator("ingest_token_hash")
    @classmethod
    def _is_a_sha256_digest(cls, raw: str | None) -> str | None:
        """Fail at boot on a malformed hash, rather than 401ing forever.

        The failure this prevents is silent: a truncated or upper-cased digest in
        `.env` compares unequal to every token the Mac will ever send, and the only
        symptom is an ingest that rejects a *correct* token with no clue why.
        """
        if raw is None:
            return None

        digest = raw.strip().lower()

        if not digest:
            return None

        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(
                "TA_INGEST_TOKEN_HASH must be a 64-character sha256 hex digest — "
                "mint one with `python -m target_analyzer.create_token`"
            )

        return digest


@lru_cache
def get_settings() -> Settings:
    return Settings()
