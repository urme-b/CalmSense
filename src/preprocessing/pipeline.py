import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config import FS, FEATURE_PARAMS, LABEL_NAMES, VALID_SUBJECTS
from ..logging_config import LoggerMixin
from ..utils import timer, ensure_directory
from .ecg_processor import ECGProcessor
from .eda_processor import EDAProcessor
from .respiratory_processor import RespiratoryProcessor
from .windowing import SignalWindower
from .filters import SignalProcessor


class PreprocessingPipeline(LoggerMixin):
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        window_size_sec: float = FEATURE_PARAMS.WINDOW_SIZE_SEC,
        window_overlap: float = FEATURE_PARAMS.WINDOW_OVERLAP,
        chest_fs: float = FS.CHEST,
        wrist_eda_fs: float = FS.WRIST_EDA,
        wrist_bvp_fs: float = FS.WRIST_BVP,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = Path(output_dir) if output_dir else None

        self.config = {
            "window_size_sec": window_size_sec,
            "window_overlap": window_overlap,
            "chest_fs": chest_fs,
            "wrist_eda_fs": wrist_eda_fs,
            "wrist_bvp_fs": wrist_bvp_fs,
        }

        self.ecg_processor = ECGProcessor(sampling_rate=chest_fs)
        self.eda_processor_chest = EDAProcessor(sampling_rate=chest_fs)
        self.eda_processor_wrist = EDAProcessor(sampling_rate=wrist_eda_fs)
        self.resp_processor = RespiratoryProcessor(sampling_rate=chest_fs)
        self.signal_processor = SignalProcessor(fs=chest_fs)
        self.windower = SignalWindower(
            window_size_sec=window_size_sec,
            overlap=window_overlap,
            sampling_rate=chest_fs,
        )

        self.logger.info(
            f"PreprocessingPipeline initialized: "
            f"{window_size_sec}s windows, {window_overlap * 100:.0f}% overlap"
        )

    def load_subject_data(self, subject_id: str) -> Dict[str, Any]:
        if self.data_dir is None:
            raise ValueError("data_dir not specified")

        subject_path = self.data_dir / subject_id / f"{subject_id}.pkl"

        if not subject_path.exists():
            raise FileNotFoundError(f"Subject data not found: {subject_path}")

        with open(subject_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.logger.info(f"Loaded data for {subject_id}")
        return data

    def process_chest_signals(
        self, chest_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        results = {}

        if "ECG" in chest_data:
            ecg = chest_data["ECG"].flatten()
            with timer("ECG processing"):
                results["ECG"] = self.ecg_processor.process(ecg)

        if "EDA" in chest_data:
            eda = chest_data["EDA"].flatten()
            with timer("EDA (chest) processing"):
                results["EDA_chest"] = self.eda_processor_chest.process(eda)

        if "Resp" in chest_data:
            resp = chest_data["Resp"].flatten()
            with timer("Respiratory processing"):
                results["Resp"] = self.resp_processor.process(resp)

        if "EMG" in chest_data:
            emg = chest_data["EMG"].flatten()
            results["EMG"] = {"filtered_emg": self.signal_processor.process_emg(emg)}

        if "Temp" in chest_data:
            temp = chest_data["Temp"].flatten()
            results["Temp"] = {
                "filtered_temp": self.signal_processor.process_temperature(temp)
            }

        if "ACC" in chest_data:
            acc = np.asarray(chest_data["ACC"])
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            results["ACC"] = {"acc_xyz": acc, "acc_magnitude": acc_mag}

        return results

    def process_wrist_signals(
        self, wrist_data: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        results = {}

        if "EDA" in wrist_data:
            eda = wrist_data["EDA"].flatten()
            with timer("EDA (wrist) processing"):
                results["EDA_wrist"] = self.eda_processor_wrist.process(eda)

        if "BVP" in wrist_data:
            bvp = wrist_data["BVP"].flatten()
            results["BVP"] = {"raw_bvp": bvp}

        if "TEMP" in wrist_data:
            temp = wrist_data["TEMP"].flatten()
            results["Temp_wrist"] = {"raw_temp": temp}

        if "ACC" in wrist_data:
            acc = np.asarray(wrist_data["ACC"])
            if acc.ndim == 1:
                acc_mag = np.abs(acc)
            else:
                acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            results["ACC_wrist"] = {"acc_xyz": acc, "acc_magnitude": acc_mag}

        return results

    def extract_quality_summary(
        self, processed_results: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        quality_summary = {}

        for modality, results in processed_results.items():
            if "quality" in results:
                quality_summary[modality] = results["quality"]

        return quality_summary

    def create_feature_windows(
        self,
        processed_results: Dict[str, Dict],
        labels: np.ndarray,
        include_modalities: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        ref_length = None
        ref_signal = None

        if "ECG" in processed_results:
            ref_signal = processed_results["ECG"]["filtered_ecg"]
            ref_length = len(ref_signal)
        elif "Resp" in processed_results:
            ref_signal = processed_results["Resp"]["filtered_resp"]
            ref_length = len(ref_signal)

        if ref_length is None:
            raise ValueError("No reference signal found for windowing")

        if len(labels) != ref_length:
            self.logger.warning(
                f"Label length mismatch: {len(labels)} vs signal {ref_length}"
            )
            min_len = min(len(labels), ref_length)
            labels = labels[:min_len]

        signals_to_window = {}

        modalities = include_modalities or list(processed_results.keys())

        for modality in modalities:
            if modality not in processed_results:
                continue

            results = processed_results[modality]

            if modality == "ECG":
                signals_to_window["ECG"] = results.get("filtered_ecg")
                if "rr_interpolated" in results and len(results["rr_interpolated"]) > 0:
                    rr = results["rr_interpolated"]
                    r_peaks = results["r_peaks"]
                    if len(r_peaks) > 0:
                        rr_resampled = self._resample_to_length(rr, r_peaks, ref_length)
                        signals_to_window["RR"] = rr_resampled

            elif modality == "EDA_chest":
                filtered_eda = results.get("filtered_eda")
                signals_to_window["EDA_chest"] = filtered_eda
                # Fallback if missing
                signals_to_window["EDA_tonic"] = results.get("tonic") or filtered_eda
                signals_to_window["EDA_phasic"] = results.get("phasic") or filtered_eda

            elif modality == "Resp":
                signals_to_window["Resp"] = results.get("filtered_resp")

            elif modality == "EMG":
                signals_to_window["EMG"] = results.get("filtered_emg")

            elif modality == "Temp":
                signals_to_window["Temp"] = results.get("filtered_temp")

            elif modality == "ACC":
                signals_to_window["ACC_mag"] = results.get("acc_magnitude")

        windows_dict, window_labels, valid_mask = (
            self.windower.create_windows_multimodal(
                signals_to_window,
                labels,
                reference_signal="ECG"
                if "ECG" in signals_to_window
                else list(signals_to_window.keys())[0],
            )
        )

        return windows_dict, window_labels, valid_mask

    def _resample_to_length(
        self, values: np.ndarray, indices: np.ndarray, target_length: int
    ) -> np.ndarray:
        # Resample to uniform length
        from scipy.interpolate import interp1d

        if len(values) == 0 or len(indices) == 0:
            return np.zeros(target_length)

        if len(values) == len(indices) - 1:
            midpoints = (indices[:-1] + indices[1:]) / 2
        else:
            midpoints = indices[: len(values)]

        interp_func = interp1d(
            midpoints,
            values,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )

        x_full = np.arange(target_length)
        resampled = interp_func(x_full)

        return resampled

    def process_subject(
        self, subject_id: str, save_results: bool = True
    ) -> Dict[str, Any]:
        self.logger.info(f"Processing subject {subject_id}")

        with timer(f"Subject {subject_id} total processing"):
            raw_data = self.load_subject_data(subject_id)

            chest_data = raw_data.get("signal", {}).get("chest", {})
            wrist_data = raw_data.get("signal", {}).get("wrist", {})
            labels = raw_data.get("label", np.array([]))

            with timer("Chest signal processing"):
                chest_results = self.process_chest_signals(chest_data)

            with timer("Wrist signal processing"):
                wrist_results = self.process_wrist_signals(wrist_data)

            all_results = {**chest_results, **wrist_results}

            quality = self.extract_quality_summary(all_results)

            with timer("Feature windowing"):
                windows, window_labels, valid_mask = self.create_feature_windows(
                    all_results, labels
                )

            valid_windows = {k: v[valid_mask] for k, v in windows.items()}
            valid_labels = (
                window_labels[valid_mask] if window_labels is not None else None
            )

            output = {
                "subject_id": subject_id,
                "processed_signals": all_results,
                "windows": valid_windows,
                "labels": valid_labels,
                "quality": quality,
                "config": self.config,
                "n_windows": len(valid_labels) if valid_labels is not None else 0,
                "n_invalid": int(np.sum(~valid_mask)),
            }

            if save_results and self.output_dir is not None:
                self.save_processed_data(output, subject_id)

            self.logger.info(
                f"Subject {subject_id} complete: "
                f"{output['n_windows']} valid windows, "
                f"{output['n_invalid']} invalid"
            )

            return output

    def process_all_subjects(
        self, subjects: Optional[List[str]] = None, save_combined: bool = True
    ) -> Dict[str, Any]:
        subjects = subjects or VALID_SUBJECTS

        self.logger.info(f"Processing {len(subjects)} subjects")

        all_windows = {
            key: []
            for key in [
                "ECG",
                "EDA_chest",
                "EDA_tonic",
                "EDA_phasic",
                "Resp",
                "EMG",
                "Temp",
                "ACC_mag",
                "RR",
            ]
        }
        all_labels = []
        all_subject_ids = []
        quality_records = []

        for subject_id in subjects:
            try:
                result = self.process_subject(subject_id, save_results=True)

                for key in all_windows:
                    if key in result["windows"]:
                        all_windows[key].append(result["windows"][key])

                if result["labels"] is not None:
                    all_labels.append(result["labels"])
                    all_subject_ids.extend([subject_id] * len(result["labels"]))

                quality_records.append({"subject_id": subject_id, **result["quality"]})

            except FileNotFoundError as e:
                self.logger.error(f"Data not found for {subject_id}: {e}")
                continue
            except ValueError as e:
                self.logger.error(f"Invalid data for {subject_id}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error processing {subject_id}: {e}")
                continue

        combined_windows = {}
        for key, window_list in all_windows.items():
            if len(window_list) > 0:
                combined_windows[key] = np.concatenate(window_list, axis=0)

        combined_labels = np.concatenate(all_labels) if all_labels else np.array([])
        subject_ids_array = np.array(all_subject_ids)

        combined = {
            "windows": combined_windows,
            "labels": combined_labels,
            "subject_ids": subject_ids_array,
            "quality": quality_records,
            "config": self.config,
            "n_subjects": len(subjects),
            "n_windows_total": len(combined_labels),
        }

        if save_combined and self.output_dir is not None:
            self._save_combined_dataset(combined)

        self.logger.info(
            f"All subjects processed: {combined['n_windows_total']} total windows "
            f"from {combined['n_subjects']} subjects"
        )

        return combined

    def save_processed_data(self, data: Dict[str, Any], subject_id: str) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir not specified")

        ensure_directory(self.output_dir)
        output_path = self.output_dir / f"{subject_id}_processed.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Saved processed data to {output_path}")
        return output_path

    def _save_combined_dataset(self, combined: Dict[str, Any]) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir not specified")

        ensure_directory(self.output_dir)
        output_path = self.output_dir / "wesad_processed_combined.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Saved combined dataset to {output_path}")
        return output_path

    def get_label_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}

        for label, count in zip(unique, counts):
            name = LABEL_NAMES.get(int(label), f"unknown_{label}")
            distribution[name] = int(count)

        return distribution
