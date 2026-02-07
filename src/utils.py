import hashlib
import json
import pickle
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TypeVar, Union

import numpy as np

from .config import EPSILON, PROCESSED_DATA_DIR

T = TypeVar("T")


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    from .logging_config import get_logger

    logger = get_logger(__name__)

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{name} completed in {elapsed:.2f} seconds")


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        from .logging_config import get_logger

        logger = get_logger(__name__)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, fill_value)
    return result


def normalize_array(arr: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        return (arr - np.mean(arr)) / (np.std(arr) + EPSILON)
    elif method == "minmax":
        min_val, max_val = np.min(arr), np.max(arr)
        return (arr - min_val) / (max_val - min_val + EPSILON)
    elif method == "robust":
        median = np.median(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        return (arr - median) / (iqr + EPSILON)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sliding_window(arr: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    arr = np.asarray(arr)
    n_windows = (len(arr) - window_size) // step_size + 1
    if n_windows <= 0:
        return np.empty((0, window_size))

    # Stride tricks
    shape = (n_windows, window_size)
    strides = (arr.strides[0] * step_size, arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()


def compute_hash(data: Union[np.ndarray, Dict, str]) -> str:
    if isinstance(data, np.ndarray):
        return hashlib.md5(data.tobytes()).hexdigest()
    elif isinstance(data, dict):
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    else:
        return hashlib.md5(str(data).encode()).hexdigest()


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_cache_path(
    name: str, params: Optional[Dict] = None, extension: str = ".pkl"
) -> Path:
    if params:
        param_hash = compute_hash(params)[:8]
        filename = f"{name}_{param_hash}{extension}"
    else:
        filename = f"{name}{extension}"

    return PROCESSED_DATA_DIR / filename


def cache_result(
    cache_name: str, params: Optional[Dict] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            from .logging_config import get_logger

            logger = get_logger(__name__)

            cache_path = get_cache_path(cache_name, params)

            if cache_path.exists():
                logger.info(f"Loading cached result from {cache_path}")
                return load_pickle(cache_path)

            result = func(*args, **kwargs)

            logger.info(f"Caching result to {cache_path}")
            save_pickle(result, cache_path)

            return result

        return wrapper

    return decorator


def validate_array(
    arr: np.ndarray,
    name: str = "array",
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_length: Optional[int] = None,
    expected_ndim: Optional[int] = None,
) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr)}")

    if not allow_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")

    if not allow_inf and np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values")

    if min_length is not None and len(arr) < min_length:
        raise ValueError(f"{name} length {len(arr)} is less than minimum {min_length}")

    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(f"{name} has {arr.ndim} dimensions, expected {expected_ndim}")


def check_signal_quality(
    signal: np.ndarray, fs: float, name: str = "signal"
) -> Dict[str, Any]:
    signal = np.asarray(signal).flatten()

    mean_val = float(np.nanmean(signal))
    std_val = float(np.nanstd(signal))

    return {
        "name": name,
        "n_samples": len(signal),
        "duration_sec": len(signal) / fs,
        "n_nan": int(np.sum(np.isnan(signal))),
        "n_inf": int(np.sum(np.isinf(signal))),
        "mean": mean_val,
        "std": std_val,
        "min": float(np.nanmin(signal)),
        "max": float(np.nanmax(signal)),
        "n_outliers_3std": int(np.sum(np.abs(signal - mean_val) > 3 * std_val)),
        "is_valid": not (np.any(np.isnan(signal)) or np.any(np.isinf(signal))),
    }


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_samples(n_samples: int, fs: float) -> str:
    duration = n_samples / fs
    return f"{n_samples:,} samples ({duration:.1f}s)"


def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
