"""SQLAlchemy engine, session factory, and declarative base."""

from collections.abc import Generator
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


def _create_resilient_engine():
    """Attempt connecting to PostgreSQL; fallback to local SQLite if unreachable."""
    try:
        eng = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        with eng.connect():
            pass
        log.info("connected to primary database: %s", settings.database_url)
        return eng
    except Exception as exc:
        sqlite_path = settings.repo_root / "data/intellicannula.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite:///{sqlite_path}"
        log.warning(
            "could not reach PostgreSQL (%s); automatically using local SQLite store: %s",
            exc,
            sqlite_url,
        )
        return create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False},
            future=True,
        )


engine = _create_resilient_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models (populated in Phase 6)."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
