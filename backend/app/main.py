"""IntelliCannula FastAPI entrypoint.

Phase 1 exposes only /health and /. The vein, infection, history, and alert
routers are added in Phase 6.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "version": settings.api_version}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by docker-compose and the Phase 1 smoke test."""
    return {"status": "ok", "version": settings.api_version}
