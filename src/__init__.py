__version__ = "0.1.0"
__author__ = "urme-b"

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    WESAD_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    VALID_SUBJECTS,
    LABEL_NAMES,
    FS,
    FILTER_PARAMS,
    FEATURE_PARAMS,
    TRAINING_PARAMS,
    CONDITION_COLORS,
    get_project_root,
    ensure_directories,
)

from .logging_config import (
    setup_logging,
    get_logger,
    LoggerMixin,
)

from .utils import (
    timer,
    timeit,
    safe_divide,
    normalize_array,
    sliding_window,
    save_pickle,
    load_pickle,
    validate_array,
    check_signal_quality,
    get_timestamp,
)

from .data import WESADLoader
from .preprocessing import SignalProcessor
from .features import FeatureExtractor
from .explainability import SHAPExplainer, LIMEExplainer, FeatureImportance

__all__ = [
    "__version__",
    "__author__",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "WESAD_DIR",
    "MODELS_DIR",
    "FIGURES_DIR",
    "VALID_SUBJECTS",
    "LABEL_NAMES",
    "FS",
    "FILTER_PARAMS",
    "FEATURE_PARAMS",
    "TRAINING_PARAMS",
    "CONDITION_COLORS",
    "get_project_root",
    "ensure_directories",
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "timer",
    "timeit",
    "safe_divide",
    "normalize_array",
    "sliding_window",
    "save_pickle",
    "load_pickle",
    "validate_array",
    "check_signal_quality",
    "get_timestamp",
    "WESADLoader",
    "SignalProcessor",
    "FeatureExtractor",
    "SHAPExplainer",
    "LIMEExplainer",
    "FeatureImportance",
]


def _initialize():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)


_initialize()
