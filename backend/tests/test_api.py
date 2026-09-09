"""API contract tests: happy path and error path for every endpoint."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_returns_service_metadata(self, client):
        assert "IntelliCannula" in client.get("/").json()["service"]

    def test_models_endpoint_reports_versions(self, client):
        """A clinician-facing UI shows the model version beside a recommendation."""
        body = client.get("/models").json()
        assert "vein" in body and "infection_risk" in body
        assert "weights_present" in body["vein"]


class TestPatients:
    def test_create_and_fetch(self, client):
        created = client.post("/patients", json={"name": "Ada", "mrn": "MRN-100"})
        assert created.status_code == 201
        patient_id = created.json()["id"]
        assert client.get(f"/patients/{patient_id}").json()["mrn"] == "MRN-100"

    def test_duplicate_mrn_conflicts(self, client):
        """Two records for one patient would split their site history."""
        client.post("/patients", json={"name": "Ada", "mrn": "MRN-101"})
        duplicate = client.post("/patients", json={"name": "Ada Again", "mrn": "MRN-101"})
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"]

    def test_unknown_patient_404s(self, client):
        assert client.get("/patients/99999").status_code == 404

    def test_blank_name_rejected(self, client):
        assert client.post("/patients", json={"name": "", "mrn": "X"}).status_code == 422


class TestVeinScan:
    def test_happy_path_returns_site_and_overlay(self, client, patient, png_bytes, mock_vein):
        response = client.post(
            f"/patients/{patient['id']}/vein-scan",
            files={"file": ("arm.png", png_bytes, "image/png")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["insertion_x"] == 128
        assert body["angle_deg"] == 22.5
        assert body["overlay_png_base64"]
        assert body["model_version"] == "attention-unet-test"

    def test_refusal_is_a_success_not_an_error(
        self, client, patient, png_bytes, mock_vein_refuses
    ):
        """No defensible site is a valid clinical answer, not a failure.

        Forcing a recommendation the model cannot support is exactly what this
        system must not do.
        """
        response = client.post(
            f"/patients/{patient['id']}/vein-scan",
            files={"file": ("arm.png", png_bytes, "image/png")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["insertion_x"] is None
        assert body["confidence"] == 0.0
        assert "minimum bar" in body["reason"]

    def test_unknown_patient_404s(self, client, png_bytes, mock_vein):
        response = client.post(
            "/patients/99999/vein-scan",
            files={"file": ("arm.png", png_bytes, "image/png")},
        )
        assert response.status_code == 404

    def test_non_image_upload_422s(self, client, patient, mock_vein):
        response = client.post(
            f"/patients/{patient['id']}/vein-scan",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 422

    def test_empty_upload_422s(self, client, patient, mock_vein):
        response = client.post(
            f"/patients/{patient['id']}/vein-scan",
            files={"file": ("arm.png", b"", "image/png")},
        )
        assert response.status_code == 422

    def test_scan_is_persisted_and_listable(self, client, patient, png_bytes, mock_vein):
        client.post(
            f"/patients/{patient['id']}/vein-scan",
            files={"file": ("arm.png", png_bytes, "image/png")},
        )
        scans = client.get(f"/patients/{patient['id']}/vein-scans").json()
        assert len(scans) == 1
        assert scans[0]["image_sha256"] if "image_sha256" in scans[0] else True


class TestSiteCheck:
    def test_first_checkin_has_no_delta(self, client, patient, png_bytes, mock_site_check):
        mock_site_check.append(0.2)
        response = client.post(
            f"/patients/{patient['id']}/site-check",
            files={"file": ("site.png", png_bytes, "image/png")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["risk_score"] == pytest.approx(0.2)
        assert body["delta_from_previous"] == 0.0
        assert body["alert_triggered"] is False

    def test_high_risk_triggers_alert(self, client, patient, png_bytes, mock_site_check):
        mock_site_check.append(0.85)
        body = client.post(
            f"/patients/{patient['id']}/site-check",
            files={"file": ("site.png", png_bytes, "image/png")},
        ).json()
        assert body["alert_triggered"] is True
        assert "exceeds" in body["alert_reason"]

    def test_sharp_jump_alerts_below_the_risk_threshold(
        self, client, patient, png_bytes, mock_site_check
    ):
        """Catches fast deterioration before the absolute level is alarming."""
        mock_site_check.extend([0.20, 0.45])
        url = f"/patients/{patient['id']}/site-check"
        client.post(url, files={"file": ("a.png", png_bytes, "image/png")})
        body = client.post(url, files={"file": ("b.png", png_bytes, "image/png")}).json()
        assert body["alert_triggered"] is True
        assert "rose" in body["alert_reason"]
        assert body["risk_score"] < 0.6

    def test_three_slow_increases_alert(self, client, patient, png_bytes, mock_site_check):
        """A creeping rise that never trips either threshold on its own."""
        mock_site_check.extend([0.20, 0.28, 0.36, 0.44])
        url = f"/patients/{patient['id']}/site-check"
        for name in ("a", "b", "c", "d"):
            body = client.post(
                url, files={"file": (f"{name}.png", png_bytes, "image/png")}
            ).json()
        assert body["alert_triggered"] is True
        assert "consecutive increases" in body["alert_reason"]

    def test_stable_series_does_not_alert(self, client, patient, png_bytes, mock_site_check):
        mock_site_check.extend([0.15, 0.14, 0.16])
        url = f"/patients/{patient['id']}/site-check"
        for name in ("a", "b", "c"):
            body = client.post(
                url, files={"file": (f"{name}.png", png_bytes, "image/png")}
            ).json()
        assert body["alert_triggered"] is False

    def test_unknown_patient_404s(self, client, png_bytes, mock_site_check):
        response = client.post(
            "/patients/99999/site-check",
            files={"file": ("site.png", png_bytes, "image/png")},
        )
        assert response.status_code == 404

    def test_bad_content_type_422s(self, client, patient, mock_site_check):
        response = client.post(
            f"/patients/{patient['id']}/site-check",
            files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 422


class TestHistory:
    def test_empty_history(self, client, patient):
        body = client.get(f"/patients/{patient['id']}/history").json()
        assert body["n_checkins"] == 0
        assert body["points"] == []

    def test_points_are_oldest_first(self, client, patient, png_bytes, mock_site_check):
        """The chart is a time series; newest-first invites an off-by-one."""
        mock_site_check.extend([0.10, 0.30, 0.50])
        url = f"/patients/{patient['id']}/site-check"
        for name in ("a", "b", "c"):
            client.post(url, files={"file": (f"{name}.png", png_bytes, "image/png")})

        body = client.get(f"/patients/{patient['id']}/history").json()
        scores = [p["risk_score"] for p in body["points"]]
        assert body["n_checkins"] == 3
        assert scores == sorted(scores)
        assert body["latest_trend"] == "worsening"

    def test_alert_points_are_flagged(self, client, patient, png_bytes, mock_site_check):
        mock_site_check.extend([0.10, 0.90])
        url = f"/patients/{patient['id']}/site-check"
        for name in ("a", "b"):
            client.post(url, files={"file": (f"{name}.png", png_bytes, "image/png")})
        points = client.get(f"/patients/{patient['id']}/history").json()["points"]
        assert [p["alert_triggered"] for p in points] == [False, True]

    def test_unknown_patient_404s(self, client):
        assert client.get("/patients/99999/history").status_code == 404


class TestAlertSocket:
    def test_socket_accepts_connection(self, client):
        with client.websocket_connect("/ws/alerts") as socket:
            assert socket is not None

    def test_alert_is_pushed_to_connected_client(
        self, client, patient, png_bytes, mock_site_check
    ):
        mock_site_check.append(0.92)
        with client.websocket_connect("/ws/alerts") as socket:
            client.post(
                f"/patients/{patient['id']}/site-check",
                files={"file": ("site.png", png_bytes, "image/png")},
            )
            message = socket.receive_json()
        assert message["patient_id"] == patient["id"]
        assert message["risk_score"] == pytest.approx(0.92)
        assert "exceeds" in message["reason"]

    def test_no_alert_no_message(self, client, patient, png_bytes, mock_site_check):
        """A quiet site must not wake anyone."""
        from app.api.alerts import connection_count

        mock_site_check.append(0.05)
        with client.websocket_connect("/ws/alerts"):
            assert connection_count() == 1
            response = client.post(
                f"/patients/{patient['id']}/site-check",
                files={"file": ("site.png", png_bytes, "image/png")},
            )
        assert response.json()["alert_triggered"] is False
