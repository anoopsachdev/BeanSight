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

    with patch("app.main.init_db", new_callable=AsyncMock) as mock_init_db, \
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
        # Patch the global predictors after import
        import app.main as main_module
        original_roast = main_module.roast_predictor
        original_defect = main_module.defect_predictor
        main_module.roast_predictor = mock_roast
        main_module.defect_predictor = mock_defect

        yield {
            "roast_predictor": mock_roast,
            "defect_predictor": mock_defect,
        }

        # Restore originals
        main_module.roast_predictor = original_roast
        main_module.defect_predictor = original_defect


@pytest.fixture
def client():
    """Create a test client with mocked lifespan."""
    from app.main import app
    # Use TestClient without lifespan to avoid real init
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
        assert data["defect"]["prediction"] in [
            "Broken", "Cut", "Dry Cherry", "Fade", "Floater",
            "Full Black", "Full Sour", "Fungus Damage", "Husk",
            "Immature", "Parchment", "Partial Black", "Partial Sour",
            "Severe Insect Damage", "Shell", "Slight Insect Damage", "Withered",
        ]
        assert 0 <= data["defect"]["confidence"] <= 1
        assert len(data["defect"]["probabilities"]) == 17

        # Check metadata
        assert "inference_time_ms" in data

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


# ── Frontend ────────────────────────────────────────────────────────────


class TestFrontend:
    def test_serves_index_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "BeanSight" in response.text
