from abc import ABC, abstractmethod
from typing import Any

from ...logging_config import LoggerMixin


class BaseMLModel(ABC, LoggerMixin):
    """Factory base: subclasses build a configured estimator via `_create_model`."""

    def __init__(self, model_name: str, random_state: int = 42, **kwargs):
        self.model_name = model_name
        self.random_state = random_state
        self.kwargs = kwargs

    @abstractmethod
    def _create_model(self) -> Any: ...
