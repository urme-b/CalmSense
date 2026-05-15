from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from .model_manager import ModelManager
from .feature_schema import FeatureSchemaStore


class AppState:
    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.feature_store: FeatureSchemaStore = FeatureSchemaStore()
        self.start_time: datetime = datetime.now(timezone.utc)
        self._request_count: int = 0
        self._lock: Lock = Lock()

    @property
    def request_count(self) -> int:
        return self._request_count

    def increment_request_count(self) -> int:
        with self._lock:
            self._request_count += 1
            return self._request_count


app_state = AppState()


def get_model_manager() -> ModelManager:
    if app_state.model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return app_state.model_manager


def get_feature_store() -> FeatureSchemaStore:
    return app_state.feature_store
