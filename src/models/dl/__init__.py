# Base class
from .base_dl_model import BaseDLModel

# 1D CNN architectures
from .cnn_1d import (
    CNN1D,
    CNN1DResidual,
    MultiScaleCNN1D,
    ConvBlock1D,
    GlobalAveragePooling1D,
)

# 2D CNN with transfer
from .cnn_2d import (
    CNN2DTransferLearning,
    MultiChannelCNN2D,
    BACKBONE_CONFIGS,
)

# LSTM/GRU models
from .lstm_model import (
    BiLSTMClassifier,
    GRUClassifier,
    StackedLSTMClassifier,
    Attention,
)

# CNN-LSTM hybrid
from .cnn_lstm import (
    CNNLSTMHybrid,
    DeepCNNLSTM,
    TemporalAttention,
)

# Transformer
from .transformer_model import (
    StressTransformer,
    TemporalConvTransformer,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    MultiHeadAttention,
    TransformerEncoderLayer,
)

# Cross-modal attention
from .cross_modal_attention import (
    CrossModalAttention,
    MultimodalTransformer,
    ModalityEncoder,
    CrossAttentionBlock,
    GatedFusion,
)

# TabNet
from .tabnet_model import (
    TabNetWrapper,
    TabNetRegressor,
)

# Data loading
from .data_loader import (
    StressDataset,
    SequenceDataset,
    ImageStressDataset,
    MultimodalDataset,
    create_data_loaders,
    create_loso_loaders,
    create_sequence_loaders,
    collate_variable_length,
)

# Augmentation
from .augmentation import (
    SignalAugmenter,
    ImageAugmenter,
)

# Training
from .trainer import (
    DLTrainer,
    EarlyStopping,
    loso_cross_validate,
)

__all__ = [
    # Base
    "BaseDLModel",
    # 1D CNN
    "CNN1D",
    "CNN1DResidual",
    "MultiScaleCNN1D",
    "ConvBlock1D",
    "GlobalAveragePooling1D",
    # 2D CNN
    "CNN2DTransferLearning",
    "MultiChannelCNN2D",
    "BACKBONE_CONFIGS",
    # LSTM/GRU
    "BiLSTMClassifier",
    "GRUClassifier",
    "StackedLSTMClassifier",
    "Attention",
    # CNN-LSTM
    "CNNLSTMHybrid",
    "DeepCNNLSTM",
    "TemporalAttention",
    # Transformer
    "StressTransformer",
    "TemporalConvTransformer",
    "SinusoidalPositionalEncoding",
    "LearnablePositionalEncoding",
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    # Cross-modal
    "CrossModalAttention",
    "MultimodalTransformer",
    "ModalityEncoder",
    "CrossAttentionBlock",
    "GatedFusion",
    # TabNet
    "TabNetWrapper",
    "TabNetRegressor",
    # Data
    "StressDataset",
    "SequenceDataset",
    "ImageStressDataset",
    "MultimodalDataset",
    "create_data_loaders",
    "create_loso_loaders",
    "create_sequence_loaders",
    "collate_variable_length",
    # Augmentation
    "SignalAugmenter",
    "ImageAugmenter",
    # Training
    "DLTrainer",
    "EarlyStopping",
    "loso_cross_validate",
]


# Model registry for easy
MODEL_REGISTRY = {
    # 1D CNN
    "cnn1d": CNN1D,
    "cnn1d_residual": CNN1DResidual,
    "multiscale_cnn1d": MultiScaleCNN1D,
    # 2D CNN
    "cnn2d": CNN2DTransferLearning,
    "multichannel_cnn2d": MultiChannelCNN2D,
    # Recurrent
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
    "stacked_lstm": StackedLSTMClassifier,
    # Hybrid
    "cnn_lstm": CNNLSTMHybrid,
    "deep_cnn_lstm": DeepCNNLSTM,
    # Transformer
    "transformer": StressTransformer,
    "conv_transformer": TemporalConvTransformer,
    # Multimodal
    "cross_modal": CrossModalAttention,
    "multimodal_transformer": MultimodalTransformer,
}


def get_model(name: str, **kwargs) -> BaseDLModel:

    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:

    return list(MODEL_REGISTRY.keys())
