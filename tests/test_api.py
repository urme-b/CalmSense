import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_features():

    return list(np.random.randn(50))


@pytest.fixture
def sample_feature_names():

    return [f"feature_{i}" for i in range(50)]


@pytest.fixture
def mock_model_manager():

    manager = Mock()
    manager.models = {"test_model": Mock()}
    manager.model_info = {
        "test_model": {
            "type": "random_forest",
            "version": "1.0.0",
            "num_classes": 3,
            "is_loaded": True,
        }
    }
    manager.default_model = "test_model"
    manager.CLASS_NAMES = ["Baseline", "Stress", "Amusement"]

    manager.predict = Mock(
        return_value={
            "prediction": 1,
            "class_name": "Stress",
            "probabilities": {"Baseline": 0.2, "Stress": 0.7, "Amusement": 0.1},
            "confidence": 0.7,
            "model_used": "test_model",
            "inference_time_ms": 5.0,
        }
    )

    manager.load_model = Mock(return_value=(True, 100.0))
    manager.unload_model = Mock(return_value=True)
    manager.is_model_loaded = Mock(return_value=True)
    manager.set_default_model = Mock(return_value=True)
    manager.get_model_info = Mock(
        return_value={
            "type": "random_forest",
            "version": "1.0.0",
            "num_classes": 3,
            "is_loaded": True,
        }
    )
    manager.list_models = Mock(
        return_value=[
            {
                "name": "test_model",
                "model_type": "random_forest",
                "version": "1.0.0",
                "num_classes": 3,
                "is_loaded": True,
            }
        ]
    )
    manager.get_statistics = Mock(
        return_value={
            "total_inferences": 100,
            "average_inference_time_ms": 5.0,
            "models_loaded": 1,
            "models_available": 1,
        }
    )
    manager.get_feature_names = Mock(return_value=None)
    manager.predict_batch = Mock(
        return_value=[
            {
                "prediction": 1,
                "class_name": "Stress",
                "probabilities": {"Baseline": 0.2, "Stress": 0.7, "Amusement": 0.1},
                "confidence": 0.7,
                "model_used": "test_model",
                "inference_time_ms": 5.0,
            },
            {
                "prediction": 1,
                "class_name": "Stress",
                "probabilities": {"Baseline": 0.2, "Stress": 0.7, "Amusement": 0.1},
                "confidence": 0.7,
                "model_used": "test_model",
                "inference_time_ms": 5.0,
            },
        ]
    )

    return manager


@pytest.fixture
def test_client(mock_model_manager):

    from fastapi.testclient import TestClient
    from api.main import app, app_state

    app_state.model_manager = mock_model_manager

    return TestClient(app)


# ============================================================================
# Schema Tests
# ============================================================================


class TestSchemas:
    def test_feature_vector_validation(self):

        from api.schemas import FeatureVector

        # Valid vector
        fv = FeatureVector(values=[1.0, 2.0, 3.0])
        assert len(fv.values) == 3

        # Empty vector should fail
        with pytest.raises(ValueError):
            FeatureVector(values=[])

    def test_prediction_request(self, sample_features):

        from api.schemas import PredictionRequest, FeatureVector

        request = PredictionRequest(
            features=FeatureVector(values=sample_features),
            model_name="test_model",
            return_probabilities=True,
        )

        assert request.model_name == "test_model"
        assert request.return_probabilities is True

    def test_batch_prediction_request_limit(self, sample_features):

        from api.schemas import BatchPredictionRequest, FeatureVector

        # Should work with reasonable
        samples = [FeatureVector(values=sample_features) for _ in range(10)]
        request = BatchPredictionRequest(samples=samples)
        assert len(request.samples) == 10

        # Should fail with too
        with pytest.raises(ValueError):
            samples = [FeatureVector(values=sample_features) for _ in range(1001)]
            BatchPredictionRequest(samples=samples)

    def test_explanation_types(self):

        from api.schemas import ExplanationType

        assert ExplanationType.SHAP.value == "shap"
        assert ExplanationType.LIME.value == "lime"
        assert ExplanationType.CLINICAL.value == "clinical"

    def test_model_info_schema(self):

        from api.schemas import ModelInfo

        info = ModelInfo(
            name="test_model",
            model_type="random_forest",
            version="1.0.0",
            num_classes=3,
            is_loaded=True,
        )

        assert info.name == "test_model"
        assert info.is_loaded is True


# ============================================================================
# Model Manager Tests
# ============================================================================


class TestModelManager:
    def test_initialization(self):

        from api.model_manager import ModelManager

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                manager = ModelManager(models_dir="./test_models")
                assert manager.models_dir.name == "test_models"
                assert manager.models == {}

    def test_predict_sklearn(self, mock_model_manager, sample_features):

        result = mock_model_manager.predict(
            model_name="test_model",
            features=np.array(sample_features),
            return_probabilities=True,
        )

        assert "prediction" in result
        assert "class_name" in result
        assert "probabilities" in result

    def test_register_model(self):

        from api.model_manager import ModelManager

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                manager = ModelManager(models_dir="./test_models")

                mock_model = Mock()
                mock_model.predict = Mock(return_value=np.array([1]))
                mock_model.predict_proba = Mock(
                    return_value=np.array([[0.2, 0.7, 0.1]])
                )

                manager.register_model(
                    model_name="custom_model", model=mock_model, model_type="custom"
                )

                assert "custom_model" in manager.models
                assert manager.is_model_loaded("custom_model")


# ============================================================================
# Prediction Endpoint Tests
# ============================================================================


class TestPredictionEndpoints:
    def test_single_prediction(
        self, test_client, sample_features, sample_feature_names
    ):

        response = test_client.post(
            "/api/v1/predict",
            json={
                "features": {
                    "values": sample_features,
                    "feature_names": sample_feature_names,
                },
                "model_name": "test_model",
                "return_probabilities": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "class_name" in data
        assert "confidence" in data

    def test_batch_prediction(self, test_client, sample_features):

        response = test_client.post(
            "/api/v1/predict/batch",
            json={
                "samples": [{"values": sample_features}, {"values": sample_features}],
                "model_name": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 2
        assert len(data["predictions"]) == 2

    def test_get_classes(self, test_client):

        response = test_client.get("/api/v1/predict/classes")

        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 3


# ============================================================================
# Model Management Endpoint Tests
# ============================================================================


class TestModelEndpoints:
    def test_list_models(self, test_client):

        response = test_client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default_model" in data

    def test_get_model_info(self, test_client):

        response = test_client.get("/api/v1/models/test_model")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_model"

    def test_load_model(self, test_client):

        response = test_client.post(
            "/api/v1/models/load",
            json={"model_name": "test_model", "force_reload": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_unload_model(self, test_client):

        response = test_client.post("/api/v1/models/test_model/unload")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_set_default_model(self, test_client):

        response = test_client.post("/api/v1/models/default/test_model")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_statistics(self, test_client):

        response = test_client.get("/api/v1/models/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "total_inferences" in data


# ============================================================================
# Explanation Endpoint Tests
# ============================================================================


class TestExplanationEndpoints:
    def test_explain_shap(
        self, test_client, sample_features, sample_feature_names, mock_model_manager
    ):

        # Patch explainability imports
        with patch("api.routes.explanation._get_shap_explanation") as mock_shap:
            mock_shap.return_value = (
                [
                    {
                        "feature_name": "feature_0",
                        "importance": 0.5,
                        "direction": "positive",
                    }
                ],
                None,
            )

            response = test_client.post(
                "/api/v1/explain",
                json={
                    "features": {
                        "values": sample_features,
                        "feature_names": sample_feature_names,
                    },
                    "explanation_type": "shap",
                    "num_features": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "feature_importances" in data

    def test_explain_clinical(self, test_client, sample_features, sample_feature_names):

        with patch("api.routes.explanation._get_clinical_explanation") as mock_clinical:
            mock_clinical.return_value = (
                [],
                {
                    "stress_level": "elevated",
                    "stress_score": 0.6,
                    "findings": [],
                    "summary": "Test summary",
                    "recommendations": ["Test recommendation"],
                    "disclaimer": "Test disclaimer",
                },
            )

            response = test_client.post(
                "/api/v1/explain",
                json={
                    "features": {
                        "values": sample_features,
                        "feature_names": sample_feature_names,
                    },
                    "explanation_type": "clinical",
                    "num_features": 10,
                },
            )

            assert response.status_code == 200

    def test_list_explanation_types(self, test_client):

        response = test_client.get("/api/v1/explain/types")

        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert len(data["types"]) >= 5


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoints:
    def test_health_check(self, test_client):

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_root_endpoint(self, test_client):

        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "status" in data


# ============================================================================
# WebSocket Tests
# ============================================================================


class TestWebSocketEndpoints:
    def test_websocket_connection(self, test_client):

        with test_client.websocket_connect("/ws/predict") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"

    def test_websocket_prediction(self, test_client, sample_features):

        with test_client.websocket_connect("/ws/predict") as websocket:
            # Receive connection message
            _ = websocket.receive_json()

            # Send prediction request
            websocket.send_json(
                {"type": "predict", "data": {"features": sample_features}}
            )

            # Receive prediction
            response = websocket.receive_json()
            assert response["type"] == "prediction"
            assert "prediction" in response["data"]

    def test_websocket_ping(self, test_client):

        with test_client.websocket_connect("/ws/predict") as websocket:
            _ = websocket.receive_json()

            websocket.send_json({"type": "ping", "data": {}})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_batch_prediction(self, test_client, sample_features):

        with test_client.websocket_connect("/ws/predict") as websocket:
            _ = websocket.receive_json()

            websocket.send_json(
                {
                    "type": "batch_predict",
                    "data": {
                        "samples": [sample_features, sample_features],
                        "batch_id": "test_batch",
                    },
                }
            )

            # Receive batch start
            response = websocket.receive_json()
            assert response["type"] == "batch_start"

            # Receive predictions
            for _ in range(2):
                response = websocket.receive_json()
                assert response["type"] == "batch_prediction"

            # Receive batch complete
            response = websocket.receive_json()
            assert response["type"] == "batch_complete"


# ============================================================================
# Connection Manager Tests
# ============================================================================


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connection_tracking(self):

        from api.routes.websocket import ConnectionManager

        manager = ConnectionManager()

        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()

        await manager.connect(mock_websocket, "test_client")
        assert manager.get_connection_count() == 1

        await manager.disconnect("test_client")
        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_group_management(self):

        from api.routes.websocket import ConnectionManager

        manager = ConnectionManager()

        await manager.join_group("client1", "group1")
        assert "client1" in manager.connection_groups.get("group1", set())

        await manager.leave_group("client1", "group1")
        assert "client1" not in manager.connection_groups.get("group1", set())


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    def test_invalid_features(self, test_client):

        response = test_client.post(
            "/api/v1/predict", json={"features": {"values": []}}
        )

        assert response.status_code == 422  # Validation error

    def test_model_not_found(self, test_client, sample_features, mock_model_manager):

        mock_model_manager.get_model_info.return_value = None

        response = test_client.get("/api/v1/models/nonexistent_model")

        assert response.status_code == 404


# ============================================================================
# Security Boundary Tests
# ============================================================================


class TestSecurityBoundaries:
    def test_nan_feature_rejected(self, test_client):

        response = test_client.post(
            "/api/v1/predict",
            content='{"features": {"values": [1.0, NaN, 3.0]}}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 422)

    def test_inf_feature_rejected(self, test_client):

        response = test_client.post(
            "/api/v1/predict",
            content='{"features": {"values": [1.0, Infinity, 3.0]}}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 422)

    def test_extreme_feature_value_rejected(self, test_client):

        response = test_client.post(
            "/api/v1/predict",
            json={"features": {"values": [1e16]}},
        )
        assert response.status_code == 422

    def test_path_traversal_model_name(self, test_client):

        response = test_client.post(
            "/api/v1/models/load",
            json={"model_name": "../../etc/passwd"},
        )
        assert response.status_code == 422

    def test_path_traversal_model_path(self, test_client):

        response = test_client.post(
            "/api/v1/models/load",
            json={"model_name": "safe_name", "model_path": "../../../etc/passwd"},
        )
        assert response.status_code == 422

    def test_absolute_path_model_rejected(self, test_client):

        response = test_client.post(
            "/api/v1/models/load",
            json={"model_name": "safe_name", "model_path": "/etc/passwd"},
        )
        assert response.status_code == 422

    def test_batch_size_limit(self, test_client, sample_features):

        # Build a request with
        samples = [{"values": sample_features} for _ in range(1001)]
        response = test_client.post(
            "/api/v1/predict/batch",
            json={"samples": samples},
        )
        assert response.status_code == 422

    def test_websocket_unknown_message_type(self, test_client):

        with test_client.websocket_connect("/ws/predict") as ws:
            _ = ws.receive_json()
            ws.send_json({"type": "drop_database", "data": {}})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Unknown" in response["data"]["message"]

    def test_security_headers_present(self, test_client):

        response = test_client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_model_name_special_chars_rejected(self, test_client):

        response = test_client.post(
            "/api/v1/models/load",
            json={"model_name": "model; rm -rf /"},
        )
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================


class TestAPIIntegration:
    def test_prediction_workflow(self, test_client, sample_features):

        # List models
        response = test_client.get("/api/v1/models")
        assert response.status_code == 200

        # Make prediction
        response = test_client.post(
            "/api/v1/predict", json={"features": {"values": sample_features}}
        )
        assert response.status_code == 200

        # Get statistics
        response = test_client.get("/api/v1/models/statistics")
        assert response.status_code == 200

    def test_middleware_headers(self, test_client):

        response = test_client.get("/health")

        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
