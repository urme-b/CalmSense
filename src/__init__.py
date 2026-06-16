__version__ = "0.1.0"
__author__ = "urme-b"

from .config import (
    CONDITION_COLORS,
    DATA_DIR,
    FEATURE_PARAMS,
    FIGURES_DIR,
    FILTER_PARAMS,
    FS,
    LABEL_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    VALID_SUBJECTS,
    WESAD_DIR,
    ensure_directories,
    get_project_root,
)
from .data import WESADLoader
from .logging_config import (
    LoggerMixin,
    get_logger,
    setup_logging,
)
from .preprocessing import SignalProcessor
from .utils import ensure_directory, timer

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
    "CONDITION_COLORS",
    "get_project_root",
    "ensure_directories",
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "timer",
    "ensure_directory",
    "WESADLoader",
    "SignalProcessor",
]


def _initialize():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)


_initialize()
