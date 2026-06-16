import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import FS
from ..logging_config import LoggerMixin
from ..utils import ensure_directory, timer
from .accelerometer_features import AccelerometerFeatureExtractor
from .eda_features import EDAFeatureExtractor
from .hrv_frequency_domain import HRVFrequencyDomainExtractor
from .hrv_nonlinear import HRVNonlinearExtractor
from .hrv_time_domain import HRVTimeDomainExtractor
from .respiration_features import RespirationFeatureExtractor
from .temperature_features import TemperatureFeatureExtractor


class FeatureExtractionPipeline(LoggerMixin):
    DEFAULT_CONFIG = {
        "hrv_time": True,
        "hrv_frequency": True,
        "hrv_nonlinear": True,
        "eda": True,
        "temperature": True,
        "respiration": True,
        "accelerometer": True,
    }

    def __init__(
        self,
        feature_config: Optional[Dict[str, bool]] = None,
        chest_fs: float = FS.CHEST,
        wrist_eda_fs: float = FS.WRIST_EDA,
        wrist_acc_fs: float = FS.WRIST_ACC,
    ):
        self.feature_config = {**self.DEFAULT_CONFIG, **(feature_config or {})}

        self.extractors: Dict[str, Any] = {
            "hrv_time": HRVTimeDomainExtractor(),
            "hrv_frequency": HRVFrequencyDomainExtractor(),
            "hrv_nonlinear": HRVNonlinearExtractor(),
            "eda": EDAFeatureExtractor(sampling_rate=wrist_eda_fs),
            "temperature": TemperatureFeatureExtractor(
                sampling_rate=wrist_eda_fs
            ),  # wrist TEMP and EDA share 4 Hz
            "respiration": RespirationFeatureExtractor(sampling_rate=chest_fs),
            "accelerometer": AccelerometerFeatureExtractor(sampling_rate=wrist_acc_fs),
        }

        self._feature_names: Optional[List[str]] = None
        self.logger.info(
            f"FeatureExtractionPipeline initialized with config: "
            f"{sum(self.feature_config.values())} feature groups enabled"
        )

    def extract_window_features(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        rr = window_data.get("rr_intervals")

        features: Dict[str, float] = {}
        self._add(features, "hrv_time", "HRV_", lambda: self._hrv(rr, "hrv_time"))
        self._add(features, "hrv_frequency", "HRV_", lambda: self._hrv(rr, "hrv_frequency"))
        self._add(features, "hrv_nonlinear", "HRV_", lambda: self._hrv(rr, "hrv_nonlinear"))
        self._add(features, "eda", "EDA_", lambda: self._eda(window_data))
        self._add(features, "temperature", "TEMP_", lambda: self._temperature(window_data))
        self._add(features, "respiration", "RESP_", lambda: self._respiration(window_data))
        self._add(features, "accelerometer", "ACC_", lambda: self._accelerometer(window_data))
        return features

    def _add(self, features, group, prefix, compute) -> None:
        if not self.feature_config.get(group, True):
            return
        result = compute()
        if result is None:
            result = dict.fromkeys(self.extractors[group].get_feature_descriptions(), np.nan)
        features.update({k if k.startswith(prefix) else prefix + k: v for k, v in result.items()})

    def _hrv(self, rr, group):
        return None if rr is None else self.extractors[group].extract_all(rr)

    def _eda(self, w):
        tonic, raw = w.get("eda_tonic"), w.get("eda_raw")
        if tonic is None and raw is None:
            return None
        decomposed = {"tonic": tonic, "phasic": w.get("eda_phasic")}
        return self.extractors["eda"].extract_all(decomposed, w.get("scr_peaks"), raw)

    def _temperature(self, w):
        temp = w.get("temperature")
        return None if temp is None else self.extractors["temperature"].extract_all(temp)

    def _respiration(self, w):
        resp = w.get("respiration")
        if resp is None:
            return None
        return self.extractors["respiration"].extract_all(
            resp,
            breath_peaks=w.get("breath_peaks"),
            breath_troughs=w.get("breath_troughs"),
            breath_intervals=w.get("breath_intervals"),
        )

    def _accelerometer(self, w):
        acc = w.get("accelerometer")
        if acc is None:
            return None
        extractor = self.extractors["accelerometer"]
        if not isinstance(acc, dict):
            return extractor.extract_from_magnitude(acc)
        if "magnitude" in acc:
            return extractor.extract_from_magnitude(acc["magnitude"])
        empty = np.array([])
        return extractor.extract_all(acc.get("x", empty), acc.get("y", empty), acc.get("z", empty))

    def extract_all_features(
        self,
        processed_data: Union[pd.DataFrame, List[Dict]],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if isinstance(processed_data, pd.DataFrame):
            windows = processed_data.to_dict("records")
        else:
            windows = processed_data

        n_windows = len(windows)
        self.logger.info(f"Extracting features from {n_windows} windows")

        all_features = []

        with timer("Feature extraction"):
            for i, window_data in enumerate(windows):
                if show_progress and (i + 1) % 100 == 0:
                    self.logger.info(f"Processing window {i + 1}/{n_windows}")

                features = self.extract_window_features(window_data)

                for key in ["subject_id", "window_id", "label"]:
                    if key in window_data:
                        features[key] = window_data[key]

                all_features.append(features)

        features_df = pd.DataFrame(all_features)

        metadata_cols = ["subject_id", "window_id", "label"]
        existing_meta = [c for c in metadata_cols if c in features_df.columns]
        feature_cols = [c for c in features_df.columns if c not in metadata_cols]
        features_df = features_df[existing_meta + feature_cols]

        self.logger.info(
            f"Feature extraction complete: {len(features_df)} windows, {len(feature_cols)} features"
        )

        return features_df

    def get_feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names

        dummy_data = {
            "rr_intervals": np.array([800, 810, 795] * 20),
            "eda_tonic": np.ones(100) * 5,
            "eda_phasic": np.zeros(100),
            "scr_peaks": [],
            "temperature": np.ones(100) * 35,
            "respiration": np.sin(np.linspace(0, 10, 1000)),
            "accelerometer": {"magnitude": np.ones(100)},
        }

        features = self.extract_window_features(dummy_data)
        self._feature_names = list(features.keys())

        return self._feature_names

    def get_feature_count(self) -> int:
        return len(self.get_feature_names())

    def get_feature_descriptions(self) -> Dict[str, str]:
        descriptions = {}

        if self.feature_config.get("hrv_time", True):
            for k, v in self.extractors["hrv_time"].get_feature_descriptions().items():
                descriptions[f"HRV_{k}"] = v

        if self.feature_config.get("hrv_frequency", True):
            for k, v in self.extractors["hrv_frequency"].get_feature_descriptions().items():
                descriptions[f"HRV_{k}"] = v

        if self.feature_config.get("hrv_nonlinear", True):
            for k, v in self.extractors["hrv_nonlinear"].get_feature_descriptions().items():
                descriptions[f"HRV_{k}"] = v

        if self.feature_config.get("eda", True):
            for k, v in self.extractors["eda"].get_feature_descriptions().items():
                descriptions[f"EDA_{k}"] = v

        if self.feature_config.get("temperature", True):
            descriptions.update(self.extractors["temperature"].get_feature_descriptions())

        if self.feature_config.get("respiration", True):
            descriptions.update(self.extractors["respiration"].get_feature_descriptions())

        if self.feature_config.get("accelerometer", True):
            descriptions.update(self.extractors["accelerometer"].get_feature_descriptions())

        return descriptions

    def save_features(
        self,
        features_df: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "csv",
    ) -> Path:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)

        if file_format == "csv":
            features_df.to_csv(output_path, index=False)
        elif file_format == "parquet":
            features_df.to_parquet(output_path, index=False)
        elif file_format == "pickle":
            with open(output_path, "wb") as f:
                pickle.dump(features_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unknown format: {file_format}")

        self.logger.info(f"Features saved to {output_path}")
        return output_path

    def load_features(
        self, input_path: Union[str, Path], file_format: Optional[str] = None
    ) -> pd.DataFrame:
        input_path = Path(input_path)

        if file_format is None:
            file_format = input_path.suffix.lstrip(".")

        if file_format == "csv":
            return pd.read_csv(input_path)
        elif file_format == "parquet":
            return pd.read_parquet(input_path)
        elif file_format in ("pickle", "pkl"):
            with open(input_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {file_format}")

    def get_feature_groups(self) -> Dict[str, List[str]]:
        all_features = self.get_feature_names()

        groups: Dict[str, List[str]] = {
            "HRV Time-Domain": [],
            "HRV Frequency-Domain": [],
            "HRV Nonlinear": [],
            "EDA": [],
            "Temperature": [],
            "Respiration": [],
            "Accelerometer": [],
        }

        time_features = set(self.extractors["hrv_time"].get_feature_descriptions().keys())
        freq_features = set(self.extractors["hrv_frequency"].get_feature_descriptions().keys())
        nl_features = set(self.extractors["hrv_nonlinear"].get_feature_descriptions().keys())

        for feat in all_features:
            if feat.startswith("HRV_"):
                base = feat[4:]
                if base in time_features:
                    groups["HRV Time-Domain"].append(feat)
                elif base in freq_features:
                    groups["HRV Frequency-Domain"].append(feat)
                elif base in nl_features:
                    groups["HRV Nonlinear"].append(feat)
            elif feat.startswith("EDA_"):
                groups["EDA"].append(feat)
            elif feat.startswith("TEMP_"):
                groups["Temperature"].append(feat)
            elif feat.startswith("RESP_"):
                groups["Respiration"].append(feat)
            elif feat.startswith("ACC_"):
                groups["Accelerometer"].append(feat)

        return groups
