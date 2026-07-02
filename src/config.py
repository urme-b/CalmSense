from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Global random seed for reproducible runs; imported by scripts and set_seed().
SEED: int = 42

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
WESAD_DIR = RAW_DATA_DIR / "WESAD"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

CONFIG_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

# S1, S12 excluded
VALID_SUBJECTS: List[str] = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]

LABEL_NAMES: Dict[int, str] = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}


@dataclass(frozen=True)
class SamplingRates:
    # Chest (RespiBAN) — all channels at 700 Hz
    CHEST: float = 700.0

    # Wrist (Empatica E4)
    WRIST_ACC: float = 32.0
    WRIST_BVP: float = 64.0
    WRIST_EDA: float = 4.0
    WRIST_TEMP: float = 4.0


FS = SamplingRates()


@dataclass(frozen=True)
class FilterParams:
    ECG_BANDPASS_LOW: float = 0.5
    ECG_BANDPASS_HIGH: float = 40.0
    ECG_NOTCH_FREQ: float = 50.0  # EU powerline
    ECG_FILTER_ORDER: int = 4

    EDA_LOWPASS: float = 5.0
    EDA_MEDIAN_SIZE: int = 5
    EDA_FILTER_ORDER: int = 4

    EMG_BANDPASS_LOW: float = 20.0
    EMG_BANDPASS_HIGH: float = 300.0
    EMG_NOTCH_FREQ: float = 50.0
    EMG_FILTER_ORDER: int = 4

    RESP_BANDPASS_LOW: float = 0.1
    RESP_BANDPASS_HIGH: float = 0.5
    RESP_FILTER_ORDER: int = 4

    TEMP_LOWPASS: float = 0.1
    TEMP_FILTER_ORDER: int = 2

    NOTCH_Q_FACTOR: float = 30.0


FILTER_PARAMS = FilterParams()


@dataclass(frozen=True)
class FeatureParams:
    WINDOW_SIZE_SEC: float = 60.0
    WINDOW_OVERLAP: float = 0.5
    EPSILON: float = 1e-10


FEATURE_PARAMS = FeatureParams()

EPSILON: float = FEATURE_PARAMS.EPSILON

CONDITION_COLORS: Dict[str, str] = {
    "baseline": "#2ecc71",
    "stress": "#e74c3c",
    "amusement": "#3498db",
    "meditation": "#9b59b6",
    "not_defined": "#95a5a6",
    "ignore": "#bdc3c7",
}

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"
LOG_FILE: Path = LOGS_DIR / "calmsense.log"


def get_project_root() -> Path:
    return PROJECT_ROOT


def ensure_directories() -> None:
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        EXTERNAL_DATA_DIR,
        OUTPUT_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        CONFIG_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
