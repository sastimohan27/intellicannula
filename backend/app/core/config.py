"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "IntelliCannula API"
    api_version: str = "0.1.0"

    # Postgres. Default matches the docker-compose `postgres` service.
    database_url: str = "postgresql+psycopg2://intellicannula:intellicannula@localhost:5432/intellicannula"

    # Object storage for patient images (S3 or MinIO).
    s3_bucket: str = "intellicannula"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Where trained checkpoints live. Read by the vein / infection modules.
    model_weights_dir: Path = Path("models")

    # Local image store, used when S3 is not configured so the stack runs with
    # no cloud account.
    local_storage_dir: Path = Path("data/uploads")

    # Repo root, so the API can reach the module checkpoints without each
    # endpoint recomputing a relative path.
    repo_root: Path = Path(__file__).resolve().parents[3]

    # Recorded alongside every prediction. Bump when a model is retrained.
    vein_model_version: str = "attention-unet-v1"
    infection_model_version: str = "lgbm-risk-v1"

    # Set true in tests to skip loading real checkpoints.
    disable_ml: bool = False

    # CORS origins for the Next.js dashboard.
    frontend_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
