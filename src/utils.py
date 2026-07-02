import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Call once at the top of every entry point. With ``deterministic`` it also pins
    cuDNN and requests deterministic torch algorithms (``warn_only`` so ops without a
    deterministic kernel warn instead of crashing), so repeated runs with the same
    seed produce identical metrics. ``PYTHONHASHSEED`` is exported for subprocesses.

    Args:
        seed: The seed applied to every RNG.
        deterministic: Enable deterministic algorithm/cuDNN settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    from .logging_config import get_logger

    logger = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"{name} completed in {time.perf_counter() - start:.2f} seconds")


def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance() -> dict:
    """Git commit + UTC timestamp for stamping result artifacts, so every committed
    JSON records exactly which code produced it."""
    import subprocess
    from datetime import datetime, timezone

    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = "unknown"
    return {"git_sha": sha, "generated_at": datetime.now(timezone.utc).isoformat()}


def paired_effect_size(a, b) -> dict:
    """Paired Cohen's d and (small-sample-corrected) Hedges' g for two per-subject vectors.

    Complements p-values: reports the standardized magnitude of a - b, which is what
    matters at N=15 where significance is low-powered.
    """
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    g = d * (1 - 3 / (4 * n - 1)) if n > 1 else d  # Hedges correction for small n
    return {"cohens_d": d, "hedges_g": float(g), "n": int(n)}
