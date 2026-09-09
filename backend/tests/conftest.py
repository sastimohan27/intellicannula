"""Test fixtures: in-memory database, mocked ML pipeline.

The ML pipeline is mocked on purpose. These tests verify the *API contract* -
status codes, persistence, alert logic, error paths - and running real
inference would make them slow, non-deterministic, and dependent on a trained
checkpoint being present. Model behaviour is tested in the module test suites.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db
from app.main import app


@pytest.fixture
def db_session(tmp_path):
    # StaticPool keeps one connection so an in-memory SQLite database survives
    # across the requests in a single test.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    from app.core import storage

    monkeypatch.setattr(storage, "_store", None)
    monkeypatch.setattr(
        storage.ImageStore, "__init__",
        lambda self: (
            setattr(self, "_s3", None),
            setattr(self, "local_root", tmp_path / "uploads"),
            (tmp_path / "uploads").mkdir(parents=True, exist_ok=True),
            None,
        )[-1],
    )

    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def patient(client):
    response = client.post("/patients", json={"name": "Test Patient", "mrn": "MRN-001"})
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def mock_vein(monkeypatch):
    """A vein pipeline that returns a confident site."""
    from app.api import vein

    def fake(_data):
        return {
            "insertion_x": 128, "insertion_y": 96, "angle_deg": 22.5,
            "vein_width_px": 7.5, "confidence": 0.72,
            "reason": "clear winner", "mask_png": b"\x89PNG-fake",
            "overlay_png_base64": "ZmFrZS1vdmVybGF5",
            "model_version": "attention-unet-test",
        }

    monkeypatch.setattr(vein, "run_vein_scan", fake)


@pytest.fixture
def mock_vein_refuses(monkeypatch):
    """A vein pipeline that legitimately declines to recommend a site."""
    from app.api import vein

    def fake(_data):
        return {
            "insertion_x": None, "insertion_y": None, "angle_deg": None,
            "vein_width_px": None, "confidence": 0.0,
            "reason": "no segment met the minimum bar",
            "mask_png": b"\x89PNG-fake", "overlay_png_base64": "ZmFrZQ==",
            "model_version": "attention-unet-test",
        }

    monkeypatch.setattr(vein, "run_vein_scan", fake)


@pytest.fixture
def mock_site_check(monkeypatch):
    """Risk scores driven by a caller-controlled queue."""
    from app.api import infection

    scores = []

    def fake(_data, _pid, _baseline):
        risk = scores.pop(0) if scores else 0.1
        return {
            "risk_score": risk,
            "roi_bbox": [16.0, 16.0, 48.0, 48.0],
            "color_distance": risk * 20,
            "registration_degraded": False,
            "model_version": "lgbm-test",
        }

    monkeypatch.setattr(infection, "run_site_check", fake)
    return scores
