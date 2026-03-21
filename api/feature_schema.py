from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FeatureMeta:
    name: str
    dtype: str = "float64"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True


@dataclass
class FeatureSchemaStore:
    schemas: Dict[str, List[FeatureMeta]] = field(default_factory=dict)

    def register(self, model_name: str, features: List[FeatureMeta]) -> None:
        self.schemas[model_name] = features

    def register_from_data(
        self, model_name: str, feature_names: List[str], data: np.ndarray
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(feature_names):
            raise ValueError(
                f"Shape mismatch: {data.shape[1]} columns vs {len(feature_names)} names"
            )
        metas = []
        for i, name in enumerate(feature_names):
            col = data[:, i]
            metas.append(
                FeatureMeta(
                    name=name,
                    min_value=float(np.nanmin(col)),
                    max_value=float(np.nanmax(col)),
                )
            )
        self.schemas[model_name] = metas

    def validate(self, model_name: str, features: np.ndarray) -> List[str]:
        schema = self.schemas.get(model_name)
        if schema is None:
            return []

        errors = []
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != len(schema):
            return [f"Expected {len(schema)} features, got {features.shape[1]}"]

        for i, meta in enumerate(schema):
            col = features[:, i]
            if meta.min_value is not None and np.any(col < meta.min_value):
                errors.append(f"{meta.name}: below min ({meta.min_value})")
            if meta.max_value is not None and np.any(col > meta.max_value):
                errors.append(f"{meta.name}: above max ({meta.max_value})")

        return errors

    def get_schema(self, model_name: str) -> Optional[List[FeatureMeta]]:
        return self.schemas.get(model_name)
