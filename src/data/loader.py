import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import (
    VALID_SUBJECTS,
    LABEL_NAMES,
    FS,
    WESAD_DIR,
)
from ..logging_config import LoggerMixin


class WESADLoader(LoggerMixin):
    VALID_SUBJECTS = VALID_SUBJECTS
    LABELS = LABEL_NAMES

    CHEST_FS = FS.CHEST
    WRIST_ACC_FS = FS.WRIST_ACC
    WRIST_BVP_FS = FS.WRIST_BVP
    WRIST_EDA_FS = FS.WRIST_EDA
    WRIST_TEMP_FS = FS.WRIST_TEMP

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else WESAD_DIR
        self._validate_path()
        self.subjects = self._discover_subjects()

        self.logger.info(
            f"WESADLoader initialized: {len(self.subjects)} subjects available"
        )

    def _validate_path(self) -> None:
        if not self.data_path.exists():
            self.logger.error(f"WESAD data path not found: {self.data_path}")
            raise FileNotFoundError(
                f"WESAD data path not found: {self.data_path}\n"
                "Please download the dataset from: "
                "https://archive.ics.uci.edu/ml/datasets/WESAD\n"
                "See data/raw/README.md for instructions."
            )

    def _discover_subjects(self) -> List[str]:
        subjects = []
        for subj_id in self.VALID_SUBJECTS:
            pkl_path = self.data_path / subj_id / f"{subj_id}.pkl"
            if pkl_path.exists():
                subjects.append(subj_id)

        if not subjects:
            self.logger.warning("No subjects found in dataset directory")

        return subjects

    def load_subject(
        self, subject_id: str, signals: Optional[List[str]] = None
    ) -> Dict:
        if subject_id not in self.VALID_SUBJECTS:
            raise ValueError(
                f"Invalid subject ID: {subject_id}. "
                f"Valid subjects: {self.VALID_SUBJECTS}"
            )

        pkl_path = self.data_path / subject_id / f"{subject_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Subject data not found: {pkl_path}")

        self.logger.info(f"Loading subject {subject_id} from {pkl_path}")

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception as e:
            self.logger.error(f"Failed to load {subject_id}: {e}")
            raise

        chest_signals = data["signal"]["chest"]
        wrist_signals = data["signal"]["wrist"]

        if signals is not None:
            signals_upper = {s.upper() for s in signals}
            chest_signals = {
                k: v for k, v in chest_signals.items() if k.upper() in signals_upper
            }
            wrist_signals = {
                k: v for k, v in wrist_signals.items() if k.upper() in signals_upper
            }
            self.logger.debug(f"Filtered to signals: {signals}")

        result = {
            "subject": subject_id,
            "chest": chest_signals,
            "wrist": wrist_signals,
            "label": data["label"],
        }

        self.logger.info(
            f"Loaded {subject_id}: {len(result['label'])} samples "
            f"({len(result['label']) / self.CHEST_FS:.1f}s)"
        )

        return result

    def load_all_subjects(self, signals: Optional[List[str]] = None) -> Dict[str, Dict]:
        self.logger.info(f"Loading all {len(self.subjects)} subjects")

        all_data = {}
        failed = {}
        for subj_id in self.subjects:
            try:
                all_data[subj_id] = self.load_subject(subj_id, signals=signals)
            except FileNotFoundError as e:
                failed[subj_id] = f"missing: {e}"
                self.logger.warning(f"Missing {subj_id}: {e}")
            except Exception as e:
                failed[subj_id] = f"error: {e}"
                self.logger.error(f"Failed to load {subj_id}: {e}")

        self.logger.info(
            f"Loaded {len(all_data)}/{len(self.subjects)} subjects"
            + (f", {len(failed)} failed: {list(failed.keys())}" if failed else "")
        )
        return all_data

    def get_labels_binary(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels).flatten()
        mask = (labels == 1) | (labels == 2)
        binary_labels = np.where(labels[mask] == 2, 1, 0)

        self.logger.debug(
            f"Binary labels: {np.sum(binary_labels == 0)} baseline, "
            f"{np.sum(binary_labels == 1)} stress"
        )

        return binary_labels, mask

    def get_labels_multiclass(
        self, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels).flatten()
        mask = (labels == 1) | (labels == 2) | (labels == 3)
        multiclass_labels = labels[mask] - 1  # shift to 0, 1,

        return multiclass_labels, mask

    def get_subject_info(self, subject_id: str) -> Dict:
        quest_path = self.data_path / subject_id / f"{subject_id}_quest.csv"
        readme_path = self.data_path / subject_id / f"{subject_id}_readme.txt"

        info = {"subject_id": subject_id}

        if quest_path.exists():
            try:
                info["questionnaire"] = pd.read_csv(quest_path)
            except Exception as e:
                self.logger.warning(f"Failed to read questionnaire: {e}")

        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    info["notes"] = f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read notes: {e}")

        return info

    def get_label_statistics(self, labels: np.ndarray) -> Dict[str, int]:
        labels = np.asarray(labels).flatten()
        stats = {}

        for label_val, label_name in self.LABELS.items():
            count = int(np.sum(labels == label_val))
            if count > 0:
                stats[label_name] = count

        return stats

    def __repr__(self) -> str:
        return f"WESADLoader(path='{self.data_path}', subjects={len(self.subjects)})"

    def __len__(self) -> int:
        return len(self.subjects)
