from .ml import (
    BaseMLModel,
    LogisticRegressionClassifier,
    SVMClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
    LightGBMClassifier,
    CatBoostClassifier,
    CrossValidator,
    HyperparameterTuner,
    StackingEnsemble,
    VotingEnsemble,
    ModelEvaluator,
    ImbalanceHandler,
    MLTrainingPipeline,
)

try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from .dl import (
        # Base
        BaseDLModel,
        # CNN
        CNN1D,
        CNN1DResidual,
        MultiScaleCNN1D,
        CNN2DTransferLearning,
        MultiChannelCNN2D,
        # Recurrent
        BiLSTMClassifier,
        GRUClassifier,
        # Hybrid
        CNNLSTMHybrid,
        DeepCNNLSTM,
        # Transformer
        StressTransformer,
        TemporalConvTransformer,
        # Cross-modal
        CrossModalAttention,
        MultimodalTransformer,
        # TabNet
        TabNetWrapper,
        # Data
        StressDataset,
        create_data_loaders,
        create_loso_loaders,
        # Augmentation
        SignalAugmenter,
        # Training
        DLTrainer,
        loso_cross_validate,
        # Utils
        get_model,
        list_models,
    )

__all__ = [
    # ML Base
    "BaseMLModel",
    # ML Classifiers
    "LogisticRegressionClassifier",
    "SVMClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "LightGBMClassifier",
    "CatBoostClassifier",
    # ML Validation
    "CrossValidator",
    # ML Tuning
    "HyperparameterTuner",
    # ML Ensembles
    "StackingEnsemble",
    "VotingEnsemble",
    # ML Evaluation
    "ModelEvaluator",
    # ML Utilities
    "ImbalanceHandler",
    "MLTrainingPipeline",
    # DL Base
    "BaseDLModel",
    # DL CNN
    "CNN1D",
    "CNN1DResidual",
    "MultiScaleCNN1D",
    "CNN2DTransferLearning",
    "MultiChannelCNN2D",
    # DL Recurrent
    "BiLSTMClassifier",
    "GRUClassifier",
    # DL Hybrid
    "CNNLSTMHybrid",
    "DeepCNNLSTM",
    # DL Transformer
    "StressTransformer",
    "TemporalConvTransformer",
    # DL Cross-modal
    "CrossModalAttention",
    "MultimodalTransformer",
    # DL TabNet
    "TabNetWrapper",
    # DL Data
    "StressDataset",
    "create_data_loaders",
    "create_loso_loaders",
    # DL Augmentation
    "SignalAugmenter",
    # DL Training
    "DLTrainer",
    "loso_cross_validate",
    # DL Utils
    "get_model",
    "list_models",
]
