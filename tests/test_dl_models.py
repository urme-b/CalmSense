import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_sequence_data():

    np.random.seed(42)
    batch_size = 16
    seq_len = 100
    n_features = 8
    n_classes = 3

    X = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, batch_size)

    return X, y, n_features, n_classes, seq_len


@pytest.fixture
def sample_1d_data():

    np.random.seed(42)
    batch_size = 16
    n_channels = 1
    seq_len = 256
    n_classes = 3

    X = np.random.randn(batch_size, n_channels, seq_len).astype(np.float32)
    y = np.random.randint(0, n_classes, batch_size)

    return X, y, n_channels, n_classes, seq_len


@pytest.fixture
def sample_2d_data():

    np.random.seed(42)
    batch_size = 8
    n_channels = 3
    height = 64
    width = 64
    n_classes = 3

    X = np.random.randn(batch_size, n_channels, height, width).astype(np.float32)
    y = np.random.randint(0, n_classes, batch_size)

    return X, y, n_channels, n_classes


@pytest.fixture
def sample_tabular_data():

    np.random.seed(42)
    n_samples = 200
    n_features = 50
    n_classes = 3

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    return X, y, n_features, n_classes


@pytest.fixture
def sample_data_with_groups():

    np.random.seed(42)
    n_subjects = 5
    samples_per_subject = 40
    n_features = 20
    n_classes = 3

    X_list, y_list, groups_list = [], [], []

    for subject_id in range(n_subjects):
        X_subject = np.random.randn(samples_per_subject, n_features).astype(np.float32)
        y_subject = np.random.randint(0, n_classes, samples_per_subject)
        groups_subject = np.full(samples_per_subject, subject_id)

        X_list.append(X_subject)
        y_list.append(y_subject)
        groups_list.append(groups_subject)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    return X, y, groups, n_features, n_classes


# ============================================================================
# BaseDLModel Tests
# ============================================================================


class TestBaseDLModel:
    def test_base_model_is_abstract(self):

        from src.models.dl.base_dl_model import BaseDLModel

        with pytest.raises(TypeError):
            BaseDLModel(input_dim=64, num_classes=3)

    def test_device_resolution_cpu(self):

        from src.models.dl import CNN1D

        with patch("torch.cuda.is_available", return_value=False):
            model = CNN1D(input_dim=1, num_classes=3, device="cpu")
            assert model.get_device() == torch.device("cpu")

    def test_parameter_counting(self):

        from src.models.dl import CNN1D

        model = CNN1D(input_dim=1, num_classes=3, device="cpu")
        n_params = model.count_parameters()

        # Should have some trainable
        assert n_params > 0

        # Count manually
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params == manual_count

    def test_model_summary(self):

        from src.models.dl import CNN1D

        model = CNN1D(input_dim=1, num_classes=3, device="cpu")
        summary = model.summary()

        assert isinstance(summary, str)
        assert "CNN1D" in summary
        assert "params" in summary.lower()


# ============================================================================
# CNN1D Tests
# ============================================================================


class TestCNN1D:
    def test_cnn1d_initialization(self):

        from src.models.dl import CNN1D

        model = CNN1D(input_dim=1, num_classes=3, device="cpu")
        assert model.input_dim == 1
        assert model.num_classes == 3

    def test_cnn1d_forward_pass(self, sample_1d_data):

        from src.models.dl import CNN1D

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        model = CNN1D(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    def test_cnn1d_residual(self, sample_1d_data):

        from src.models.dl import CNN1DResidual

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        model = CNN1DResidual(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    def test_multiscale_cnn1d(self, sample_1d_data):

        from src.models.dl import MultiScaleCNN1D

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        model = MultiScaleCNN1D(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)


# ============================================================================
# CNN2D Transfer Learning Tests
# ============================================================================


class TestCNN2DTransferLearning:
    @pytest.mark.skipif(
        not pytest.importorskip("torchvision", reason="torchvision not installed"),
        reason="torchvision required",
    )
    def test_cnn2d_initialization(self):

        from src.models.dl import CNN2DTransferLearning

        for backbone in ["efficientnet_b0", "resnet18", "mobilenet_v3_small"]:
            model = CNN2DTransferLearning(
                input_dim=3,
                num_classes=3,
                backbone=backbone,
                pretrained=False,  # Don't download weights in
                device="cpu",
            )
            assert model.backbone_name == backbone

    @pytest.mark.skipif(
        not pytest.importorskip("torchvision", reason="torchvision not installed"),
        reason="torchvision required",
    )
    def test_cnn2d_forward_pass(self, sample_2d_data):

        from src.models.dl import CNN2DTransferLearning

        X, y, n_channels, n_classes = sample_2d_data

        model = CNN2DTransferLearning(
            input_dim=n_channels,
            num_classes=n_classes,
            backbone="resnet18",
            pretrained=False,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    @pytest.mark.skipif(
        not pytest.importorskip("torchvision", reason="torchvision not installed"),
        reason="torchvision required",
    )
    def test_freeze_unfreeze_backbone(self):

        from src.models.dl import CNN2DTransferLearning

        model = CNN2DTransferLearning(
            input_dim=3, num_classes=3, pretrained=False, device="cpu"
        )

        # Freeze
        model.freeze_backbone()
        frozen_params = sum(
            1 for p in model._feature_extractor.parameters() if not p.requires_grad
        )
        total_backbone_params = sum(1 for p in model._feature_extractor.parameters())
        assert frozen_params == total_backbone_params

        # Unfreeze
        model.unfreeze_backbone()
        trainable_params = sum(
            1 for p in model._feature_extractor.parameters() if p.requires_grad
        )
        assert trainable_params == total_backbone_params


# ============================================================================
# LSTM/GRU Tests
# ============================================================================


class TestLSTMModels:
    def test_bilstm_initialization(self):

        from src.models.dl import BiLSTMClassifier

        model = BiLSTMClassifier(
            input_dim=64, num_classes=3, hidden_dim=128, num_layers=2, device="cpu"
        )

        assert model.hidden_dim == 128
        assert model.num_layers == 2

    def test_bilstm_forward_pass(self, sample_sequence_data):

        from src.models.dl import BiLSTMClassifier

        X, y, n_features, n_classes, seq_len = sample_sequence_data

        model = BiLSTMClassifier(
            input_dim=n_features, num_classes=n_classes, hidden_dim=64, device="cpu"
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    def test_bilstm_with_attention(self, sample_sequence_data):

        from src.models.dl import BiLSTMClassifier

        X, y, n_features, n_classes, seq_len = sample_sequence_data

        model = BiLSTMClassifier(
            input_dim=n_features,
            num_classes=n_classes,
            use_attention=True,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output, attn_weights = model(x_tensor, return_attention=True)

        assert output.shape == (X.shape[0], n_classes)
        assert attn_weights.shape == (X.shape[0], seq_len)

    def test_gru_forward_pass(self, sample_sequence_data):

        from src.models.dl import GRUClassifier

        X, y, n_features, n_classes, seq_len = sample_sequence_data

        model = GRUClassifier(
            input_dim=n_features, num_classes=n_classes, hidden_dim=64, device="cpu"
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)


# ============================================================================
# CNN-LSTM Hybrid Tests
# ============================================================================


class TestCNNLSTMHybrid:
    def test_cnn_lstm_initialization(self):

        from src.models.dl import CNNLSTMHybrid

        model = CNNLSTMHybrid(
            input_dim=1, num_classes=3, sequence_length=256, device="cpu"
        )

        assert model.input_dim == 1
        assert model.num_classes == 3

    def test_cnn_lstm_forward_pass(self, sample_1d_data):

        from src.models.dl import CNNLSTMHybrid

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        model = CNNLSTMHybrid(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    def test_cnn_lstm_attention_weights(self, sample_1d_data):

        from src.models.dl import CNNLSTMHybrid

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        model = CNNLSTMHybrid(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )

        model.eval()
        x_tensor = torch.tensor(X)
        with torch.no_grad():
            output, attn_weights = model(x_tensor, return_attention=True)

        assert output.shape == (X.shape[0], n_classes)
        assert attn_weights is not None
        # Attention weights should sum
        assert torch.allclose(
            attn_weights.sum(dim=-1), torch.ones(X.shape[0]), atol=0.01
        )


# ============================================================================
# Transformer Tests
# ============================================================================


class TestTransformer:
    def test_transformer_initialization(self):

        from src.models.dl import StressTransformer

        model = StressTransformer(
            input_dim=64,
            num_classes=3,
            d_model=128,
            nhead=4,
            num_layers=2,
            device="cpu",
        )

        assert model.d_model == 128
        assert model.nhead == 4

    def test_transformer_forward_pass(self, sample_sequence_data):

        from src.models.dl import StressTransformer

        X, y, n_features, n_classes, seq_len = sample_sequence_data

        model = StressTransformer(
            input_dim=n_features,
            num_classes=n_classes,
            d_model=64,
            nhead=4,
            num_layers=2,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output = model(x_tensor)

        assert output.shape == (X.shape[0], n_classes)

    def test_transformer_attention_weights(self, sample_sequence_data):

        from src.models.dl import StressTransformer

        X, y, n_features, n_classes, seq_len = sample_sequence_data

        model = StressTransformer(
            input_dim=n_features,
            num_classes=n_classes,
            d_model=64,
            nhead=4,
            num_layers=2,
            device="cpu",
        )

        x_tensor = torch.tensor(X)
        output, attn_weights = model(x_tensor, return_attention=True)

        assert output.shape == (X.shape[0], n_classes)
        assert len(attn_weights) == 2  # num_layers attention weight tensors

    def test_positional_encoding_types(self, sample_sequence_data):

        from src.models.dl import StressTransformer

        X, y, n_features, n_classes, seq_len = sample_sequence_data
        x_tensor = torch.tensor(X)

        for encoding_type in ["learnable", "sinusoidal"]:
            model = StressTransformer(
                input_dim=n_features,
                num_classes=n_classes,
                d_model=64,
                positional_encoding=encoding_type,
                device="cpu",
            )

            output = model(x_tensor)
            assert output.shape == (X.shape[0], n_classes)


# ============================================================================
# Cross-Modal Attention Tests
# ============================================================================


class TestCrossModalAttention:
    def test_cross_modal_initialization(self):

        from src.models.dl import CrossModalAttention

        modality_dims = {"ECG": 64, "EDA": 32, "TEMP": 16}

        model = CrossModalAttention(
            modality_dims=modality_dims, num_classes=3, device="cpu"
        )

        assert model.num_modalities == 3

    def test_cross_modal_forward_pass(self):

        from src.models.dl import CrossModalAttention

        batch_size = 8
        seq_len = 50
        modality_dims = {"ECG": 64, "EDA": 32, "TEMP": 16}

        model = CrossModalAttention(
            modality_dims=modality_dims, num_classes=3, device="cpu"
        )

        # Create multimodal input
        x = {
            "ECG": torch.randn(batch_size, seq_len, 64),
            "EDA": torch.randn(batch_size, seq_len, 32),
            "TEMP": torch.randn(batch_size, seq_len, 16),
        }

        output = model(x)
        assert output.shape == (batch_size, 3)

    def test_cross_modal_missing_modality(self):

        from src.models.dl import CrossModalAttention

        batch_size = 8
        seq_len = 50
        modality_dims = {"ECG": 64, "EDA": 32, "TEMP": 16}

        model = CrossModalAttention(
            modality_dims=modality_dims,
            num_classes=3,
            fusion_method="attention",  # Works with missing modalities
            device="cpu",
        )

        # Only provide ECG and
        x = {
            "ECG": torch.randn(batch_size, seq_len, 64),
            "EDA": torch.randn(batch_size, seq_len, 32),
        }

        output = model(x)
        assert output.shape == (batch_size, 3)

    def test_modality_importance(self):

        from src.models.dl import CrossModalAttention

        batch_size = 8
        seq_len = 50
        modality_dims = {"ECG": 64, "EDA": 32, "TEMP": 16}

        model = CrossModalAttention(
            modality_dims=modality_dims,
            num_classes=3,
            fusion_method="attention",
            device="cpu",
        )

        x = {
            "ECG": torch.randn(batch_size, seq_len, 64),
            "EDA": torch.randn(batch_size, seq_len, 32),
            "TEMP": torch.randn(batch_size, seq_len, 16),
        }

        output, importance = model(x, return_importance=True)
        assert output.shape == (batch_size, 3)
        assert len(importance) == 3  # One importance per modality
        assert abs(sum(importance.values()) - 1.0) < 0.1  # Should sum close to


# ============================================================================
# Data Loader Tests
# ============================================================================


class TestDataLoaders:
    def test_stress_dataset_creation(self, sample_tabular_data):

        from src.models.dl import StressDataset

        X, y, n_features, n_classes = sample_tabular_data

        dataset = StressDataset(X, y)

        assert len(dataset) == len(X)

        x_sample, y_sample = dataset[0]
        assert isinstance(x_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)

    def test_stress_dataset_class_weights(self, sample_tabular_data):

        from src.models.dl import StressDataset

        X, y, n_features, n_classes = sample_tabular_data

        dataset = StressDataset(X, y)
        weights = dataset.get_class_weights()

        assert len(weights) == n_classes
        assert all(w > 0 for w in weights)

    def test_create_data_loaders(self, sample_data_with_groups):

        from src.models.dl import create_data_loaders

        X, y, groups, n_features, n_classes = sample_data_with_groups

        loaders = create_data_loaders(X, y, groups, batch_size=16, val_split=0.2)

        assert "train" in loaders
        assert "val" in loaders

        # Check batch iteration
        for batch_x, batch_y in loaders["train"]:
            assert batch_x.shape[1] == n_features
            break

    def test_create_loso_loaders(self, sample_data_with_groups):

        from src.models.dl import create_loso_loaders

        X, y, groups, n_features, n_classes = sample_data_with_groups

        test_subject = 0
        train_loader, val_loader, test_loader = create_loso_loaders(
            X, y, groups, test_subject, batch_size=16
        )

        assert train_loader is not None
        assert test_loader is not None

        # Test data should only
        for batch_x, batch_y in test_loader:
            assert batch_x.shape[1] == n_features
            break


# ============================================================================
# Augmentation Tests
# ============================================================================


class TestAugmentation:
    def test_signal_augmenter_jittering(self):

        from src.models.dl import SignalAugmenter

        augmenter = SignalAugmenter(random_state=42)
        x = np.sin(np.linspace(0, 4 * np.pi, 100))

        x_aug = augmenter.jittering(x, sigma=0.03)

        assert x_aug.shape == x.shape
        assert not np.allclose(x, x_aug)  # Should be different
        assert np.corrcoef(x, x_aug)[0, 1] > 0.9  # But highly correlated

    def test_signal_augmenter_scaling(self):

        from src.models.dl import SignalAugmenter

        augmenter = SignalAugmenter(random_state=42)
        x = np.sin(np.linspace(0, 4 * np.pi, 100))

        x_aug = augmenter.scaling(x, sigma=0.1)

        assert x_aug.shape == x.shape
        # Scaling preserves shape but

    def test_signal_augmenter_time_warping(self):

        from src.models.dl import SignalAugmenter

        augmenter = SignalAugmenter(random_state=42)
        x = np.sin(np.linspace(0, 4 * np.pi, 100))

        x_aug = augmenter.time_warping(x, sigma=0.2)

        assert x_aug.shape == x.shape

    def test_signal_augmenter_random_crop(self):

        from src.models.dl import SignalAugmenter

        augmenter = SignalAugmenter(random_state=42)
        x = np.sin(np.linspace(0, 4 * np.pi, 100))

        x_aug = augmenter.random_crop(x, crop_ratio=0.9)

        # Should return same length
        assert x_aug.shape == x.shape

    def test_signal_augmenter_mixup(self):

        from src.models.dl import SignalAugmenter

        augmenter = SignalAugmenter(random_state=42)
        x1 = np.sin(np.linspace(0, 4 * np.pi, 100))
        x2 = np.cos(np.linspace(0, 4 * np.pi, 100))

        mixed_x, mixed_y = augmenter.mixup(x1, x2, 0, 1, alpha=0.2)

        assert mixed_x.shape == x1.shape
        assert len(mixed_y) == 2  # One-hot encoded


# ============================================================================
# Trainer Tests
# ============================================================================


class TestDLTrainer:
    def test_trainer_initialization(self):

        from src.models.dl import DLTrainer, CNN1D

        model = CNN1D(input_dim=1, num_classes=3, device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trainer = DLTrainer(model, optimizer, criterion)

        assert trainer.model is model
        assert trainer.device == torch.device("cpu")

    def test_trainer_train_epoch(self, sample_1d_data):

        from src.models.dl import DLTrainer, CNN1D, StressDataset
        from torch.utils.data import DataLoader

        X, y, n_channels, n_classes, seq_len = sample_1d_data

        # Transpose to (batch, features)
        X_transposed = X

        model = CNN1D(
            input_dim=n_channels,
            num_classes=n_classes,
            sequence_length=seq_len,
            device="cpu",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        DLTrainer(model, optimizer, criterion)

        # Create simple data loader
        dataset = StressDataset(X_transposed.reshape(X.shape[0], -1), y)
        DataLoader(dataset, batch_size=8)

        # This would need proper

    def test_early_stopping(self):

        from src.models.dl.trainer import EarlyStopping

        early_stopping = EarlyStopping(patience=3, mode="min")

        # Create a simple model
        model = nn.Linear(10, 1)

        # Simulate improving loss
        assert not early_stopping(1.0, model)
        assert not early_stopping(0.9, model)
        assert not early_stopping(0.8, model)

        # Simulate no improvement
        assert not early_stopping(0.85, model)
        assert not early_stopping(0.82, model)
        assert early_stopping(0.81, model)  # Should stop after patience


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    def test_model_registry(self):

        from src.models.dl import list_models

        available = list_models()
        assert len(available) > 0
        assert "cnn1d" in available
        assert "bilstm" in available

    def test_get_model_factory(self):

        from src.models.dl import get_model

        model = get_model("cnn1d", input_dim=1, num_classes=3, device="cpu")
        assert model is not None
        assert model.num_classes == 3

    def test_unknown_model_raises(self):

        from src.models.dl import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
