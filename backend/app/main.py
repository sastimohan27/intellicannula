"""IntelliCannula FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, history, infection, patients, vein
from app.core.config import get_settings
from app.core.db import Base, engine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_all is adequate for a prototype; a real deployment uses the Alembic
    # migrations so schema changes are versioned rather than implicit.
    try:
        Base.metadata.create_all(bind=engine)
        log.info("database schema ready")
    except Exception as exc:  # noqa: BLE001 - the API is still useful without a DB
        log.warning("could not initialise database: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=(
        "Clinical decision support for peripheral IV cannulation. "
        "Every output is advisory and requires clinician confirmation - the "
        "system recommends, it never acts."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(vein.router)
app.include_router(infection.router)
app.include_router(history.router)
app.include_router(alerts.router)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"service": settings.app_name, "version": settings.api_version}


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe used by docker-compose and the smoke test."""
    return {"status": "ok", "version": settings.api_version}


@app.get("/models", tags=["meta"])
def model_versions() -> dict[str, object]:
    """Which model versions are live, and whether their weights are present.

    Exposed so a clinician-facing UI can show the model version alongside a
    recommendation, and so an auditor can confirm what was deployed when.
    """
    vein_path = settings.repo_root / "vein_module/models/attention_unet_best.pt"
    risk_path = settings.repo_root / "infection_module/models/risk_lgbm.txt"
    cannula_path = settings.repo_root / "infection_module/models/cannula_yolo11.pt"
    return {
        "vein": {
            "version": settings.vein_model_version,
            "weights_present": vein_path.exists(),
        },
        "infection_risk": {
            "version": settings.infection_model_version,
            "weights_present": risk_path.exists(),
        },
        "cannula_detector": {
            "version": "not-trained",
            "weights_present": cannula_path.exists(),
        },
    }
