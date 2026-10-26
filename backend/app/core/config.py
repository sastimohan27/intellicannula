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

    # Vein mask post-processing, tuned on 100 val images for the epoch-16
    # checkpoint (re-tune whenever the model is retrained): val Dice 0.611.
    # 100 px scored marginally higher (0.615) but deletes short real veins. A higher threshold
    # and dropping tiny components removes the speckle and halo the network
    # predicts around real veins, which otherwise skews site selection.
    vein_mask_threshold: float = 0.9
    ml_device: str = "auto"   # auto | cpu | mps | cuda (ML_DEVICE env var)
    vein_min_component_px: int = 60

    # A site check is only scored when the detector finds a cannula hub at or
    # above this confidence. Measured on images with no cannula: 2/40 cubital
    # forearm frames scored 0.40-0.47 (max 0.474), a landscape photo 0.24; the
    # synthetic positives score 0.57-0.86. 0.5 rejects every negative seen.
    cannula_conf_threshold: float = 0.5
    # Below this nothing hub-like was seen ("no cannula"); between this and
    # cannula_conf_threshold the detector saw something it cannot verify.
    cannula_possible_threshold: float = 0.25

    # A follow-up is compared with the baseline only if they show the same
    # scene: RANSAC-consistent ORB matches. Measured: same scene (incl. moved,
    # rotated 8 deg, re-saved JPEG) 881-1865 inliers; unrelated scenes <= 11.
    scene_min_inliers: int = 50

    # Registered NIR capture devices: {"device-id": "secret token"}, from the
    # NIR_DEVICES env var as JSON. Empty by default - no live NIR hardware is
    # attached to this prototype, so only verified CUBITAL frames are accepted.
    nir_devices: dict[str, str] = {}
    infection_model_version: str = "erythema-v1"

    # Set true in tests to skip loading real checkpoints.
    disable_ml: bool = False

    # CORS origins for the Next.js dashboard.
    frontend_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
