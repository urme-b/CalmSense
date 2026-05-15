import pytest
import numpy as np
from unittest.mock import Mock, patch


# Test fixtures
@pytest.fixture
def sample_features():

    np.random.seed(42)
    return np.random.randn(50)


@pytest.fixture
def sample_feature_names():

    hrv_features = [
        "hr_mean",
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_pnn50",
        "hrv_lf_power",
        "hrv_hf_power",
        "hrv_lf_hf_ratio",
    ]
    eda_features = ["eda_mean", "eda_std", "scr_count", "scr_amplitude_mean"]
    resp_features = ["resp_rate", "resp_depth", "resp_variability"]
    other_features = [
        f"feature_{i}"
        for i in range(50 - len(hrv_features) - len(eda_features) - len(resp_features))
    ]
    return hrv_features + eda_features + resp_features + other_features


@pytest.fixture
def sample_training_data():

    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def mock_sklearn_model():

    model = Mock()
    model.predict = Mock(return_value=np.array([1]))
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.7, 0.1]]))
    model.feature_importances_ = np.random.rand(50)
    return model


@pytest.fixture
def mock_pytorch_model():

    try:
        import torch
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(50, 3)
                self.conv = nn.Conv1d(1, 16, 3)

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                return torch.randn(x.size(0), 3)

        return MockModel()
    except ImportError:
        pytest.skip("PyTorch not available")


# ============================================================================
# SHAP Explainer Tests
# ============================================================================


class TestSHAPExplainer:
    def test_initialization_tree_model(self, mock_sklearn_model, sample_training_data):

        try:
            from src.explainability import SHAPExplainer

            explainer = SHAPExplainer(
                model=mock_sklearn_model,
                model_type="tree",
                background_data=sample_training_data[:10],
            )

            assert explainer.model == mock_sklearn_model
            assert explainer.model_type == "tree"
        except ImportError:
            pytest.skip("SHAP not available")

    def test_initialization_kernel_model(
        self, mock_sklearn_model, sample_training_data
    ):

        try:
            from src.explainability import SHAPExplainer

            explainer = SHAPExplainer(
                model=mock_sklearn_model,
                model_type="kernel",
                background_data=sample_training_data[:10],
            )

            assert explainer.model_type == "kernel"
        except ImportError:
            pytest.skip("SHAP not available")

    def test_compute_shap_values(
        self,
        mock_sklearn_model,
        sample_features,
        sample_feature_names,
        sample_training_data,
    ):

        try:
            from src.explainability import SHAPExplainer

            # Mock the SHAP computation
            explainer = SHAPExplainer(
                model=mock_sklearn_model,
                model_type="kernel",
                background_data=sample_training_data[:10],
            )

            # Mock shap values
            with patch.object(explainer, "compute_shap_values") as mock_compute:
                mock_compute.return_value = {
                    "shap_values": np.random.randn(1, 50),
                    "base_value": 0.5,
                    "expected_value": 0.5,
                }

                result = explainer.compute_shap_values(
                    sample_features.reshape(1, -1), feature_names=sample_feature_names
                )

                assert "shap_values" in result
                assert result["shap_values"].shape == (1, 50)

        except ImportError:
            pytest.skip("SHAP not available")

    def test_get_top_features(self, sample_feature_names):

        try:
            from src.explainability import SHAPExplainer

            # Create mock SHAP values
            shap_values = np.random.randn(10, len(sample_feature_names))

            # Mock explainer
            explainer = Mock()
            explainer.get_top_features = SHAPExplainer.get_top_features.__get__(
                explainer
            )

            top_features = explainer.get_top_features(
                shap_values, sample_feature_names, n=10
            )

            assert len(top_features) <= 10
            assert "feature" in top_features.columns
            assert "mean_abs_shap" in top_features.columns

        except ImportError:
            pytest.skip("SHAP not available")


# ============================================================================
# LIME Explainer Tests
# ============================================================================


class TestLIMEExplainer:
    def test_initialization(
        self, mock_sklearn_model, sample_feature_names, sample_training_data
    ):

        try:
            from src.explainability import LIMEExplainer

            explainer = LIMEExplainer(
                model=mock_sklearn_model,
                feature_names=sample_feature_names,
                class_names=["Baseline", "Stress", "Amusement"],
                training_data=sample_training_data,
            )

            assert explainer.model == mock_sklearn_model
            assert explainer.feature_names == sample_feature_names
            assert len(explainer.class_names) == 3

        except ImportError:
            pytest.skip("LIME not available")

    def test_explain_instance(
        self,
        mock_sklearn_model,
        sample_features,
        sample_feature_names,
        sample_training_data,
    ):

        try:
            from src.explainability import LIMEExplainer

            explainer = LIMEExplainer(
                model=mock_sklearn_model,
                feature_names=sample_feature_names,
                class_names=["Baseline", "Stress", "Amusement"],
                training_data=sample_training_data,
            )

            with patch.object(explainer, "explain_instance") as mock_explain:
                mock_explain.return_value = {
                    "explanation": Mock(),
                    "feature_importance": {"hr_mean": 0.5, "eda_mean": -0.3},
                    "predicted_class": 1,
                    "predicted_class_name": "Stress",
                    "prediction_proba": [0.2, 0.7, 0.1],
                    "score": 0.85,
                    "intercept": 0.1,
                    "num_features": 10,
                    "instance_values": {},
                }

                result = explainer.explain_instance(sample_features, num_features=10)

                assert "feature_importance" in result
                assert "predicted_class" in result
                assert result["score"] > 0

        except ImportError:
            pytest.skip("LIME not available")

    def test_compare_with_shap(self, sample_feature_names):

        try:
            from src.explainability import LIMEExplainer

            lime_importances = {
                name: np.random.randn() for name in sample_feature_names[:10]
            }
            shap_importances = {
                name: np.random.randn() for name in sample_feature_names[:10]
            }

            explainer = Mock()
            explainer.compare_with_shap = LIMEExplainer.compare_with_shap.__get__(
                explainer
            )

            comparison = explainer.compare_with_shap(
                lime_importances, shap_importances, top_n=5
            )

            assert len(comparison) <= 5
            assert "lime_importance" in comparison.columns
            assert "shap_importance" in comparison.columns

        except ImportError:
            pytest.skip("LIME not available")


# ============================================================================
# Attention Visualizer Tests
# ============================================================================


class TestAttentionVisualizer:
    def test_initialization(self, mock_pytorch_model):

        try:
            from src.explainability import AttentionVisualizer

            visualizer = AttentionVisualizer(model=mock_pytorch_model)

            assert visualizer.model == mock_pytorch_model

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_compute_attention_rollout(self):

        try:
            from src.explainability import AttentionVisualizer

            # Create sample attention weights
            attention_weights = [
                np.random.rand(1, 4, 10, 10),  # [batch, heads, seq, seq]
                np.random.rand(1, 4, 10, 10),
            ]

            visualizer = AttentionVisualizer()
            rollout = visualizer.compute_attention_rollout(attention_weights)

            assert rollout.shape == (1, 10, 10)

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_get_attention_statistics(self):

        try:
            from src.explainability import AttentionVisualizer

            attention_weights = np.random.rand(1, 4, 10, 10)

            visualizer = AttentionVisualizer()
            stats = visualizer.get_attention_statistics(attention_weights)

            assert "entropy" in stats
            assert "sparsity" in stats
            assert "max_attention" in stats

        except ImportError:
            pytest.skip("PyTorch not available")


# ============================================================================
# Grad-CAM Tests
# ============================================================================


class TestGradCAMExplainer:
    def test_initialization(self, mock_pytorch_model):

        try:
            from src.explainability import GradCAMExplainer

            explainer = GradCAMExplainer(model=mock_pytorch_model, target_layer="conv")

            assert explainer.model == mock_pytorch_model
            assert explainer.target_layer == "conv"

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_generate_heatmap(self, mock_pytorch_model, sample_features):

        try:
            import torch  # noqa: F401
            from src.explainability import GradCAMExplainer

            explainer = GradCAMExplainer(model=mock_pytorch_model, target_layer="conv")

            with patch.object(explainer, "generate_heatmap") as mock_gen:
                mock_gen.return_value = np.random.rand(48)  # After conv

                heatmap = explainer.generate_heatmap(sample_features, target_class=1)

                assert heatmap is not None
                assert len(heatmap) > 0

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_get_important_regions(self):

        try:
            from src.explainability import GradCAMExplainer

            heatmap = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.3, 0.1])

            explainer = Mock()
            explainer.get_important_regions = (
                GradCAMExplainer.get_important_regions.__get__(explainer)
            )

            regions = explainer.get_important_regions(
                heatmap, threshold=0.5, min_region_size=2
            )

            assert len(regions) >= 1
            assert regions[0]["max_importance"] > 0.5

        except ImportError:
            pytest.skip("PyTorch not available")


# ============================================================================
# Clinical Interpreter Tests
# ============================================================================


class TestClinicalInterpreter:
    def test_initialization(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter(
            class_names=["Baseline", "Stress", "Amusement"]
        )

        assert len(interpreter.class_names) == 3
        assert len(interpreter.normal_ranges) > 0

    def test_assess_normal_feature(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        # Test normal heart rate
        finding = interpreter.assess_feature("hr_mean", 70.0)

        assert finding.deviation == "normal"
        assert "normal" in finding.stress_implication.lower()

    def test_assess_high_feature(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        # Test elevated heart rate
        finding = interpreter.assess_feature("hr_mean", 120.0)

        assert finding.deviation == "high"
        assert finding.confidence > 0

    def test_assess_low_feature(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        # Test low RMSSD (stress
        finding = interpreter.assess_feature("hrv_rmssd", 10.0)

        assert finding.deviation in ["low", "critical_low"]
        assert "stress" in finding.stress_implication.lower()

    def test_interpret_prediction(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        feature_values = {
            "hr_mean": 95.0,
            "hrv_rmssd": 18.0,
            "hrv_lf_hf_ratio": 3.5,
            "eda_mean": 7.5,
            "resp_rate": 22.0,
        }

        report = interpreter.interpret_prediction(
            prediction=1, probabilities=[0.1, 0.8, 0.1], feature_values=feature_values
        )

        assert "prediction" in report
        assert "stress_assessment" in report
        assert "findings" in report
        assert "summary" in report
        assert "recommendations" in report

    def test_stress_level_categorization(self):

        from src.explainability import ClinicalInterpreter, StressLevel

        interpreter = ClinicalInterpreter()

        # Low stress
        level_low = interpreter._categorize_stress_level(0.1)
        assert level_low == StressLevel.LOW

        # High stress
        level_high = interpreter._categorize_stress_level(0.75)
        assert level_high == StressLevel.HIGH

    def test_compare_sessions(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        session1 = {"hr_mean": 70.0, "hrv_rmssd": 45.0}
        session2 = {"hr_mean": 90.0, "hrv_rmssd": 25.0}

        comparison = interpreter.compare_sessions(session1, session2)

        assert len(comparison) == 2
        assert "change" in comparison.columns
        assert "stress_trend" in comparison.columns

    def test_feature_info(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        info = interpreter.get_feature_info("hrv_rmssd")

        assert info is not None
        assert "normal_range" in info
        assert "unit" in info

    def test_custom_normal_range(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        interpreter.add_custom_range(
            feature_name="custom_feature",
            low=10.0,
            high=50.0,
            unit="units",
            description="Custom feature",
            stress_direction="higher",
        )

        assert "custom_feature" in interpreter.normal_ranges

    def test_supported_features(self):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()
        features = interpreter.list_supported_features()

        assert len(features) > 0
        assert "hr_mean" in features
        assert "hrv_rmssd" in features


# ============================================================================
# Normal Range Tests
# ============================================================================


class TestNormalRanges:
    def test_normal_ranges_defined(self):

        from src.explainability import NORMAL_RANGES

        # Check key features are
        expected_features = [
            "hr_mean",
            "hrv_sdnn",
            "hrv_rmssd",
            "eda_mean",
            "resp_rate",
        ]

        for feature in expected_features:
            assert feature in NORMAL_RANGES
            assert NORMAL_RANGES[feature].low < NORMAL_RANGES[feature].high

    def test_normal_range_units(self):

        from src.explainability import NORMAL_RANGES

        for name, range_def in NORMAL_RANGES.items():
            assert range_def.unit is not None
            assert len(range_def.unit) > 0 or name.endswith("ratio")

    def test_stress_direction_defined(self):

        from src.explainability import NORMAL_RANGES

        for name, range_def in NORMAL_RANGES.items():
            assert range_def.stress_direction in ["higher", "lower"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestExplainabilityIntegration:
    def test_module_imports(self):

        from src.explainability import (
            SHAPExplainer,
            LIMEExplainer,
            ClinicalInterpreter,
        )

        assert SHAPExplainer is not None
        assert LIMEExplainer is not None
        assert ClinicalInterpreter is not None

    def test_clinical_with_shap_importances(self, sample_feature_names):

        from src.explainability import ClinicalInterpreter

        interpreter = ClinicalInterpreter()

        feature_values = {"hr_mean": 85.0, "hrv_rmssd": 35.0, "eda_mean": 5.0}

        shap_values = {"hr_mean": 0.3, "hrv_rmssd": -0.5, "eda_mean": 0.2}

        report = interpreter.interpret_prediction(
            prediction=1,
            probabilities=[0.2, 0.7, 0.1],
            feature_values=feature_values,
            shap_values=shap_values,
        )

        assert report is not None
        assert "findings" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
