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
