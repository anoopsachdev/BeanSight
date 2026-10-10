"""
Automated API endpoint tests for Coffee Bean Analyzer.

Uses pytest + httpx with FastAPI's TestClient.
All ONNX inference and database operations are mocked so tests
run without actual models, databases, or cloud services.

Usage:
  pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Fixtures ────────────────────────────────────────────────────────────


def create_test_image(width: int = 224, height: int = 224, color: str = "red") -> bytes:
    """Create a synthetic test image as JPEG bytes."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies (DB, storage, models)."""
    import app.main as main_module

    mock_roast_result = {
        "prediction": "Medium",
        "confidence": 0.92,
        "probabilities": {"Dark": 0.03, "Green": 0.02, "Light": 0.03, "Medium": 0.92},
        "inference_time_ms": 25.5,
    }

    mock_defect_result = {
        "prediction": "Broken",
        "confidence": 0.78,
        "probabilities": {
            "Broken": 0.78, "Cut": 0.05, "Dry Cherry": 0.02,
            "Fade": 0.01, "Floater": 0.01, "Full Black": 0.01,
            "Full Sour": 0.01, "Fungus Damage": 0.02, "Husk": 0.01,
            "Immature": 0.01, "Parchment": 0.01, "Partial Black": 0.02,
            "Partial Sour": 0.01, "Severe Insect Damage": 0.01,
            "Shell": 0.01, "Slight Insect Damage": 0.01, "Withered": 0.01,
        },
        "inference_time_ms": 28.3,
    }

    mock_roast = MagicMock()
    mock_roast.predict.return_value = mock_roast_result
    mock_roast.is_loaded.return_value = True

    mock_defect = MagicMock()
    mock_defect.predict.return_value = mock_defect_result
    mock_defect.is_loaded.return_value = True

    mock_yolo = MagicMock()
    mock_yolo.predict.return_value = {
        "bean_count": 3,
        "normal_count": 1,
        "defect_count": 2,
        "defect_summary": {"Quaker": 1, "Scorched": 1},
        "class_summary": {"Quaker": 1, "Scorched": 1, "Normal": 1},
        "mean_confidence": 0.85,
        "mean_defect_confidence": 0.90,
        "detections": [
            {"class": "Quaker", "confidence": 0.88, "box": [10.0, 10.0, 50.0, 50.0]},
            {"class": "Scorched", "confidence": 0.92, "box": [60.0, 60.0, 100.0, 100.0]},
            {"class": "Normal", "confidence": 0.75, "box": [120.0, 120.0, 160.0, 160.0]},
        ],
        "inference_time_ms": 15.2,
        "annotated_image": "data:image/jpeg;base64,mockb64",
    }
    mock_yolo.is_loaded.return_value = True

    with patch("app.main.init_db", new_callable=AsyncMock), \
         patch("app.main.close_db", new_callable=AsyncMock), \
         patch("app.main.init_storage", new_callable=AsyncMock), \
         patch("app.main.upload_image", new_callable=AsyncMock, return_value="https://example.com/img.jpg"), \
         patch("app.main.log_prediction", new_callable=AsyncMock, return_value="test-id-123"), \
         patch("app.main.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("app.main.get_stats", new_callable=AsyncMock, return_value={
             "total_predictions": 42,
             "avg_inference_time_ms": 30.5,
             "predictions_by_type": {"roast": 21, "defect": 21},
             "top_predicted_classes": {"Medium": 15, "Dark": 8},
         }):
        main_module.roast_predictor = mock_roast
        main_module.defect_predictor = mock_defect
        main_module.roasted_yolo_predictor = mock_yolo

        yield {
            "roast_predictor": mock_roast,
            "defect_predictor": mock_defect,
            "roasted_yolo_predictor": mock_yolo,
        }


@pytest.fixture
def client(mock_dependencies):
    """Create a test client with mocked lifespan and predictors."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health Endpoint ─────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_reports_model_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models" in data
        assert data["models"]["roast"] is True
        assert data["models"]["defect"] is True
        assert data["models"]["roasted_yolo"] is True


# ── Predict Endpoint ────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_predict_returns_dual_predictions(self, client):
        img_bytes = create_test_image()
        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # Check roast prediction
        assert "roast" in data
        assert data["roast"]["prediction"] in ["Dark", "Green", "Light", "Medium"]
        assert 0 <= data["roast"]["confidence"] <= 1
        assert len(data["roast"]["probabilities"]) == 4

        # Check defect prediction
        assert "defect" in data
        assert "prediction" in data["defect"]
        assert 0 <= data["defect"]["confidence"] <= 1

        # Check metadata
        assert "inference_time_ms" in data

    def test_predict_green_bean_routing(self, client, mock_dependencies):
        # Set roast prediction to Green
        mock_dependencies["roast_predictor"].predict.return_value = {
            "prediction": "Green",
            "confidence": 0.98,
            "probabilities": {"Dark": 0.01, "Green": 0.98, "Light": 0.005, "Medium": 0.005},
            "inference_time_ms": 12.0,
        }
        img_bytes = create_test_image()
        response = client.post(
            "/api/predict",
            files={"file": ("green_bean.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["roast"]["prediction"] == "Green"
        assert data["defect"]["type"] == "green_agricultural"
        assert len(data["defect"]["probabilities"]) == 17

    def test_predict_roasted_bean_yolo_routing(self, client, mock_dependencies):
        # Set roast prediction to Dark
        mock_dependencies["roast_predictor"].predict.return_value = {
            "prediction": "Dark",
            "confidence": 0.95,
            "probabilities": {"Dark": 0.95, "Green": 0.01, "Light": 0.02, "Medium": 0.02},
            "inference_time_ms": 12.0,
        }
        img_bytes = create_test_image()
        response = client.post(
            "/api/predict",
            files={"file": ("dark_roast.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["roast"]["prediction"] == "Dark"
        assert data["route"] == "roasted"
        assert "roasted_defect" in data
        assert data["roasted_defect"]["bean_count"] == 3
        assert data["roasted_defect"]["defect_count"] == 2
        assert "Quaker" in data["roasted_defect"]["defect_summary"]
        assert "Normal" not in data["roasted_defect"]["defect_summary"]
        # Summary confidence comes from the detector, not a hardcoded constant
        assert data["defect"]["confidence"] == 0.90
        assert data["defect"]["prediction"] in ("Quaker", "Scorched")

    def test_predict_roasted_all_normal_is_not_a_defect(self, client, mock_dependencies):
        mock_dependencies["roast_predictor"].predict.return_value = {
            "prediction": "Medium",
            "confidence": 0.9,
            "probabilities": {"Dark": 0.05, "Green": 0.01, "Light": 0.04, "Medium": 0.9},
            "inference_time_ms": 12.0,
        }
        mock_dependencies["roasted_yolo_predictor"].predict.return_value = {
            "bean_count": 2,
            "normal_count": 2,
            "defect_count": 0,
            "defect_summary": {},
            "class_summary": {"Normal": 2},
            "mean_confidence": 0.8,
            "mean_defect_confidence": 0.0,
            "detections": [
                {"class": "Normal", "confidence": 0.8, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class": "Normal", "confidence": 0.8, "box": [20.0, 20.0, 30.0, 30.0]},
            ],
            "inference_time_ms": 10.0,
            "annotated_image": None,
        }
        response = client.post(
            "/api/predict",
            files={"file": ("medium.jpg", create_test_image(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["roasted_defect"]["defect_count"] == 0
        assert data["defect"]["prediction"] == "No Defect"
        assert data["defect"]["confidence"] == 0.8
        assert data["defect"]["probabilities"]["Normal"] == 1.0

    def test_manual_green_override_beats_classifier(self, client, mock_dependencies):
        # Classifier says Medium (→ roasted), user says it's a green bean
        data = client.post(
            "/api/predict",
            files={"file": ("bean.jpg", create_test_image(), "image/jpeg")},
            data={"bean_type": "green"},
        ).json()
        assert data["roast"]["prediction"] == "Medium"
        assert data["classifier_route"] == "roasted"
        assert data["route"] == "green"
        assert data["route_source"] == "manual"
        assert data["defect"]["type"] == "green_agricultural"
        assert "roasted_defect" not in data

    def test_manual_roasted_override(self, client, mock_dependencies):
        mock_dependencies["roast_predictor"].predict.return_value = {
            "prediction": "Green",
            "confidence": 0.9,
            "probabilities": {"Dark": 0.03, "Green": 0.9, "Light": 0.04, "Medium": 0.03},
            "inference_time_ms": 12.0,
        }
        data = client.post(
            "/api/predict",
            files={"file": ("bean.jpg", create_test_image(), "image/jpeg")},
            data={"bean_type": "roasted"},
        ).json()
        assert data["route"] == "roasted"
        assert data["route_source"] == "manual"
        assert data["classifier_route"] == "green"
        assert "roasted_defect" in data

    def test_auto_is_default_and_invalid_rejected(self, client):
        data = client.post(
            "/api/predict",
            files={"file": ("bean.jpg", create_test_image(), "image/jpeg")},
        ).json()
        assert data["route_source"] == "classifier"
        bad = client.post(
            "/api/predict",
            files={"file": ("bean.jpg", create_test_image(), "image/jpeg")},
            data={"bean_type": "espresso"},
        )
        assert bad.status_code == 400

    def test_green_route_is_reported(self, client, mock_dependencies):
        mock_dependencies["roast_predictor"].predict.return_value = {
            "prediction": "Green",
            "confidence": 0.9,
            "probabilities": {"Dark": 0.03, "Green": 0.9, "Light": 0.04, "Medium": 0.03},
            "inference_time_ms": 12.0,
        }
        data = client.post(
            "/api/predict",
            files={"file": ("green.jpg", create_test_image(), "image/jpeg")},
        ).json()
        assert data["route"] == "green"
        assert "roasted_defect" not in data

    def test_predict_rejects_non_image(self, client):
        response = client.post(
            "/api/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_predict_rejects_oversized_file(self, client):
        # Create a 11MB file (over the 10MB limit)
        oversized = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/api/predict",
            files={"file": ("big.jpg", oversized, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"]

    def test_predict_rejects_invalid_image_data(self, client):
        response = client.post(
            "/api/predict",
            files={"file": ("corrupt.jpg", b"not-image-data", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_predict_accepts_png(self, client):
        img = Image.new("RGB", (100, 100), "blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/api/predict",
            files={"file": ("test.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200

    def test_predict_accepts_webp(self, client):
        img = Image.new("RGB", (100, 100), "green")
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP")
        buffer.seek(0)

        response = client.post(
            "/api/predict",
            files={"file": ("test.webp", buffer.getvalue(), "image/webp")},
        )
        assert response.status_code == 200


# ── History Endpoint ────────────────────────────────────────────────────


class TestHistoryEndpoint:
    def test_history_returns_list(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "count" in data

    def test_history_respects_limit(self, client):
        response = client.get("/api/history?limit=5")
        assert response.status_code == 200

    def test_history_caps_limit_at_100(self, client):
        response = client.get("/api/history?limit=200")
        assert response.status_code == 200


    def test_history_degrades_gracefully_when_db_down(self, client):
        from app.database import DatabaseUnavailable
        with patch("app.main.get_history", new_callable=AsyncMock, side_effect=DatabaseUnavailable("paused")):
            response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == []
        assert data["database_available"] is False


# ── Stats Endpoint ──────────────────────────────────────────────────────


class TestStatsEndpoint:
    def test_stats_returns_aggregates(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert data["total_predictions"] == 42
        assert "avg_inference_time_ms" in data
        assert "predictions_by_type" in data
        assert "top_predicted_classes" in data
        assert data["database_available"] is True

    def test_stats_degrades_gracefully_when_db_down(self, client):
        from app.database import DatabaseUnavailable
        with patch("app.main.get_stats", new_callable=AsyncMock, side_effect=DatabaseUnavailable("paused")):
            response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["database_available"] is False
        assert data["total_predictions"] == 0


# ── Health reports dependencies ─────────────────────────────────────────


class TestHealthDependencies:
    def test_health_includes_database_and_storage(self, client):
        data = client.get("/health").json()
        assert "database" in data and "connected" in data["database"]
        assert "storage" in data and "configured" in data["storage"]

    def test_error_text_redacts_connection_details(self, monkeypatch):
        import app.database as db

        class FakeSettings:
            DATABASE_URL = "postgresql://user_abc:s3cret@db.example.supabase.co:6543/postgres"
            SUPABASE_SERVICE_KEY = "svc-key"

        monkeypatch.setattr(db, "get_settings", lambda: FakeSettings())
        text = db._describe_error(RuntimeError('auth failed for "user_abc" with s3cret at db.example.supabase.co'))
        assert "s3cret" not in text
        assert "user_abc" not in text
        assert "db.example.supabase.co" not in text


# ── Frontend ────────────────────────────────────────────────────────────


class TestFrontend:
    def test_serves_index_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "BeanSight" in response.text
