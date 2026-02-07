import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import FS
from ..logging_config import LoggerMixin
from ..utils import timer, ensure_directory
from .hrv_time_domain import HRVTimeDomainExtractor
from .hrv_frequency_domain import HRVFrequencyDomainExtractor
from .hrv_nonlinear import HRVNonlinearExtractor
from .eda_features import EDAFeatureExtractor
from .temperature_features import TemperatureFeatureExtractor
from .respiration_features import RespirationFeatureExtractor
from .accelerometer_features import AccelerometerFeatureExtractor


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

        self.extractors = {
            "hrv_time": HRVTimeDomainExtractor(),
            "hrv_frequency": HRVFrequencyDomainExtractor(),
            "hrv_nonlinear": HRVNonlinearExtractor(),
            "eda": EDAFeatureExtractor(sampling_rate=wrist_eda_fs),
            "temperature": TemperatureFeatureExtractor(
                sampling_rate=wrist_eda_fs
            ),  # WESAD wrist TEMP =
            "respiration": RespirationFeatureExtractor(sampling_rate=chest_fs),
            "accelerometer": AccelerometerFeatureExtractor(sampling_rate=wrist_acc_fs),
        }

        self._feature_names = None
        self.logger.info(
            f"FeatureExtractionPipeline initialized with config: "
            f"{sum(self.feature_config.values())} feature groups enabled"
        )

    def extract_window_features(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        features = {}

        if self.feature_config.get("hrv_time", True):
            rr = window_data.get("rr_intervals")
            if rr is not None:
                hrv_time = self.extractors["hrv_time"].extract_all(rr)
                features.update({f"HRV_{k}": v for k, v in hrv_time.items()})
            else:
                for k in self.extractors["hrv_time"].get_feature_descriptions().keys():
                    features[f"HRV_{k}"] = np.nan

        if self.feature_config.get("hrv_frequency", True):
            rr = window_data.get("rr_intervals")
            if rr is not None:
                hrv_freq = self.extractors["hrv_frequency"].extract_all(rr)
                features.update({f"HRV_{k}": v for k, v in hrv_freq.items()})
            else:
                for k in (
                    self.extractors["hrv_frequency"].get_feature_descriptions().keys()
                ):
                    features[f"HRV_{k}"] = np.nan

        if self.feature_config.get("hrv_nonlinear", True):
            rr = window_data.get("rr_intervals")
            if rr is not None:
                hrv_nl = self.extractors["hrv_nonlinear"].extract_all(rr)
                features.update({f"HRV_{k}": v for k, v in hrv_nl.items()})
            else:
                for k in (
                    self.extractors["hrv_nonlinear"].get_feature_descriptions().keys()
                ):
                    features[f"HRV_{k}"] = np.nan

        if self.feature_config.get("eda", True):
            eda_decomposed = {
                "tonic": window_data.get("eda_tonic"),
                "phasic": window_data.get("eda_phasic"),
            }
            scr_peaks = window_data.get("scr_peaks")
            raw_eda = window_data.get("eda_raw")

            if eda_decomposed["tonic"] is not None or raw_eda is not None:
                eda_features = self.extractors["eda"].extract_all(
                    eda_decomposed, scr_peaks, raw_eda
                )
                features.update({f"EDA_{k}": v for k, v in eda_features.items()})
            else:
                for k in self.extractors["eda"].get_feature_descriptions().keys():
                    features[f"EDA_{k}"] = np.nan

        if self.feature_config.get("temperature", True):
            temp = window_data.get("temperature")
            if temp is not None:
                temp_features = self.extractors["temperature"].extract_all(temp)
                features.update(
                    {
                        f"TEMP_{k}" if not k.startswith("TEMP_") else k: v
                        for k, v in temp_features.items()
                    }
                )
            else:
                for k in (
                    self.extractors["temperature"].get_feature_descriptions().keys()
                ):
                    features[f"TEMP_{k}" if not k.startswith("TEMP_") else k] = np.nan

        if self.feature_config.get("respiration", True):
            resp = window_data.get("respiration")
            if resp is not None:
                resp_features = self.extractors["respiration"].extract_all(
                    resp,
                    breath_peaks=window_data.get("breath_peaks"),
                    breath_troughs=window_data.get("breath_troughs"),
                    breath_intervals=window_data.get("breath_intervals"),
                )
                features.update(
                    {
                        f"RESP_{k}" if not k.startswith("RESP_") else k: v
                        for k, v in resp_features.items()
                    }
                )
            else:
                for k in (
                    self.extractors["respiration"].get_feature_descriptions().keys()
                ):
                    features[f"RESP_{k}" if not k.startswith("RESP_") else k] = np.nan

        if self.feature_config.get("accelerometer", True):
            acc_data = window_data.get("accelerometer")
            if acc_data is not None:
                if isinstance(acc_data, dict):
                    if "magnitude" in acc_data:
                        acc_features = self.extractors[
                            "accelerometer"
                        ].extract_from_magnitude(acc_data["magnitude"])
                    else:
                        acc_features = self.extractors["accelerometer"].extract_all(
                            acc_data.get("x", np.array([])),
                            acc_data.get("y", np.array([])),
                            acc_data.get("z", np.array([])),
                        )
                else:
                    acc_features = self.extractors[
                        "accelerometer"
                    ].extract_from_magnitude(acc_data)
                features.update(
                    {
                        f"ACC_{k}" if not k.startswith("ACC_") else k: v
                        for k, v in acc_features.items()
                    }
                )
            else:
                for k in (
                    self.extractors["accelerometer"].get_feature_descriptions().keys()
                ):
                    features[f"ACC_{k}" if not k.startswith("ACC_") else k] = np.nan

        return features

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
            f"Feature extraction complete: {len(features_df)} windows, "
            f"{len(feature_cols)} features"
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
            for k, v in (
                self.extractors["hrv_frequency"].get_feature_descriptions().items()
            ):
                descriptions[f"HRV_{k}"] = v

        if self.feature_config.get("hrv_nonlinear", True):
            for k, v in (
                self.extractors["hrv_nonlinear"].get_feature_descriptions().items()
            ):
                descriptions[f"HRV_{k}"] = v

        if self.feature_config.get("eda", True):
            for k, v in self.extractors["eda"].get_feature_descriptions().items():
                descriptions[f"EDA_{k}"] = v

        if self.feature_config.get("temperature", True):
            descriptions.update(
                self.extractors["temperature"].get_feature_descriptions()
            )

        if self.feature_config.get("respiration", True):
            descriptions.update(
                self.extractors["respiration"].get_feature_descriptions()
            )

        if self.feature_config.get("accelerometer", True):
            descriptions.update(
                self.extractors["accelerometer"].get_feature_descriptions()
            )

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

        groups = {
            "HRV Time-Domain": [],
            "HRV Frequency-Domain": [],
            "HRV Nonlinear": [],
            "EDA": [],
            "Temperature": [],
            "Respiration": [],
            "Accelerometer": [],
        }

        time_features = set(
            self.extractors["hrv_time"].get_feature_descriptions().keys()
        )
        freq_features = set(
            self.extractors["hrv_frequency"].get_feature_descriptions().keys()
        )
        nl_features = set(
            self.extractors["hrv_nonlinear"].get_feature_descriptions().keys()
        )

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
