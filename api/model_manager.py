import io
import re
import time
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from threading import RLock

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.logging_config import LoggerMixin

_SAFE_PICKLE_MODULES = frozenset({
    "sklearn", "numpy", "scipy", "xgboost", "lightgbm",
    "catboost", "collections", "builtins", "copyreg",
    "_pickle", "operator", "types", "functools", "copy",
    "joblib", "numbers", "io",
})


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module.split(".")[0] not in _SAFE_PICKLE_MODULES:
            raise pickle.UnpicklingError(f"Forbidden: {module}.{name}")
        return super().find_class(module, name)


@dataclass
class ModelVersion:
    name: str
    version: str = "1.0.0"
    schema_hash: str = ""
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def parse_filename(filename: str) -> Tuple[str, str]:
        match = re.search(r"_v?(\d+\.\d+\.\d+)", filename)
        if match:
            return filename[: match.start()], match.group(1)
        return filename, "1.0.0"


class ModelManager(LoggerMixin):
    CLASS_NAMES = ["Baseline", "Stress", "Amusement"]

    def __init__(
        self,
        models_dir: Optional[str] = None,
        default_model: Optional[str] = None,
        device: Optional[str] = None,
        max_models_in_memory: int = 5,
        lazy_load: bool = True,
    ):
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.default_model = default_model
        self.max_models_in_memory = max_models_in_memory
        self.lazy_load = lazy_load

        if TORCH_AVAILABLE:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = None

        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        self.model_access_times: Dict[str, datetime] = {}
        self._model_refs: Dict[str, int] = defaultdict(int)

        self._lock = RLock()

        self.inference_count = 0
        self.total_inference_time = 0.0

        self._discover_models()

    def _discover_models(self) -> None:
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return

        model_extensions = [".pkl", ".joblib", ".pt", ".pth", ".h5", ".json"]

        for ext in model_extensions:
            for model_path in self.models_dir.glob(f"**/*{ext}"):
                model_name = model_path.stem
                if model_name not in self.model_info:
                    _, version = ModelVersion.parse_filename(model_name)
                    self.model_versions[model_name] = ModelVersion(
                        name=model_name, version=version
                    )
                    self.model_info[model_name] = {
                        "path": str(model_path),
                        "type": self._infer_model_type(model_path),
                        "is_loaded": False,
                        "version": version,
                        "num_classes": len(self.CLASS_NAMES),
                    }

        if self.model_info and not self.default_model:
            self.default_model = list(self.model_info.keys())[0]

        self.logger.info(f"Discovered {len(self.model_info)} models")

    def _infer_model_type(self, path: Path) -> str:
        name = path.stem.lower()

        if "rf" in name or "random_forest" in name:
            return "random_forest"
        elif "xgb" in name or "xgboost" in name:
            return "xgboost"
        elif "lgbm" in name or "lightgbm" in name:
            return "lightgbm"
        elif "cat" in name or "catboost" in name:
            return "catboost"
        elif "svm" in name:
            return "svm"
        elif "lr" in name or "logistic" in name:
            return "logistic_regression"
        elif "cnn" in name and "lstm" in name:
            return "cnn_lstm"
        elif "cnn" in name:
            return "cnn1d"
        elif "lstm" in name or "bilstm" in name:
            return "bilstm"
        elif "transformer" in name:
            return "transformer"
        elif "cross_modal" in name or "multimodal" in name:
            return "cross_modal"
        else:
            return "unknown"

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> Tuple[bool, float]:
        with self._lock:
            if model_name in self.models and not force_reload:
                self.logger.info(f"Model {model_name} already loaded")
                return True, 0.0

            if model_path is None:
                if model_name not in self.model_info:
                    self.logger.error(f"Model {model_name} not found")
                    return False, 0.0
                model_path = self.model_info[model_name]["path"]

            if len(self.models) >= self.max_models_in_memory:
                if not self._evict_oldest_model():
                    self.logger.warning(
                        f"Cannot evict models, limit ({self.max_models_in_memory}) may be exceeded"
                    )

            start_time = time.time()

            try:
                model = self._load_model_file(model_path)
                load_time_ms = (time.time() - start_time) * 1000

                self.models[model_name] = model
                self.model_access_times[model_name] = datetime.now(timezone.utc)

                if model_name in self.model_info:
                    self.model_info[model_name]["is_loaded"] = True
                    self.model_info[model_name]["load_time_ms"] = load_time_ms
                else:
                    _, version = ModelVersion.parse_filename(model_name)
                    self.model_info[model_name] = {
                        "path": model_path,
                        "type": self._infer_model_type(Path(model_path)),
                        "is_loaded": True,
                        "load_time_ms": load_time_ms,
                        "version": version,
                        "num_classes": len(self.CLASS_NAMES),
                    }

                # Track version
                if model_name not in self.model_versions:
                    _, ver = ModelVersion.parse_filename(model_name)
                    self.model_versions[model_name] = ModelVersion(
                        name=model_name, version=ver
                    )
                self.model_versions[model_name].loaded_at = datetime.now(timezone.utc)

                if TORCH_AVAILABLE and isinstance(model, nn.Module):
                    num_params = sum(p.numel() for p in model.parameters())
                    self.model_info[model_name]["num_parameters"] = num_params

                self.logger.info(f"Loaded model {model_name} in {load_time_ms:.2f}ms")
                return True, load_time_ms

            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                return False, 0.0

    def _load_model_file(self, path: str) -> Any:

        path = Path(path)

        # Validate path
        try:
            resolved_path = path.resolve(strict=True)
            models_dir_resolved = self.models_dir.resolve(strict=True)
            if not resolved_path.is_relative_to(models_dir_resolved):
                raise ValueError(
                    f"Model path must be within models directory: {self.models_dir}"
                )
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid model path: {e}")

        if path.suffix in [".pkl", ".joblib"]:
            self.logger.warning(f"Loading pickle file {path} with restricted unpickler.")
            with open(path, "rb") as f:
                return _RestrictedUnpickler(f).load()

        elif path.suffix in [".pt", ".pth"]:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")

            try:
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=True
                )
            except Exception:
                self.logger.warning(
                    f"Loading PyTorch model {path} with weights_only=False. "
                    "Ensure this file is from a trusted source."
                )
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_class = checkpoint.get("model_class")
                model_config = checkpoint.get("model_config", {})

                if model_class:
                    from src.models.dl import get_model

                    model = get_model(model_class, **model_config)
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    raise ValueError("Model class not found in checkpoint")
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
            else:
                raise ValueError("Unknown PyTorch checkpoint format")

            model.to(self.device)
            model.eval()
            return model

        elif path.suffix == ".json":
            with open(path, "r") as f:
                config = json.load(f)
            return config

        else:
            raise ValueError(f"Unknown model file extension: {path.suffix}")

    def _evict_oldest_model(self) -> bool:
        # Skip pinned models
        candidates = {
            k: v for k, v in self.model_access_times.items()
            if k != self.default_model and self._model_refs.get(k, 0) == 0
        }
        if not candidates:
            return False

        oldest = min(candidates, key=candidates.get)
        self.unload_model(oldest)
        self.logger.info(f"Evicted model {oldest} from memory")
        return True

    def unload_model(self, model_name: str) -> bool:
        with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                if model_name in self.model_access_times:
                    del self.model_access_times[model_name]
                if model_name in self.model_info:
                    self.model_info[model_name]["is_loaded"] = False
                return True
            return False

    def predict(
        self,
        model_name: Optional[str],
        features: np.ndarray,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        model_name = model_name or self.default_model

        if not model_name:
            raise ValueError("No model specified and no default model set")

        # Pin model
        with self._lock:
            if model_name not in self.models:
                if self.lazy_load:
                    success, _ = self.load_model(model_name)
                    if not success:
                        raise ValueError(f"Failed to load model: {model_name}")
                else:
                    raise ValueError(f"Model not loaded: {model_name}")

            model = self.models[model_name]
            self._model_refs[model_name] += 1
            self.model_access_times[model_name] = datetime.now(timezone.utc)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        start_time = time.time()

        try:
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                result = self._predict_pytorch(model, features, return_probabilities)
            else:
                result = self._predict_sklearn(model, features, return_probabilities)
        except Exception as e:
            # Graceful degradation
            fallback = self._get_fallback_model(model_name)
            if fallback is None:
                raise
            self.logger.warning(
                f"Model {model_name} failed ({e}), degrading to {fallback}"
            )
            fb_model = self.models[fallback]
            if TORCH_AVAILABLE and isinstance(fb_model, nn.Module):
                result = self._predict_pytorch(fb_model, features, return_probabilities)
            else:
                result = self._predict_sklearn(fb_model, features, return_probabilities)
            result["degraded"] = True
            result["original_model"] = model_name
            model_name = fallback
        finally:
            with self._lock:
                self._model_refs[model_name] -= 1

        inference_time_ms = (time.time() - start_time) * 1000

        self.inference_count += 1
        self.total_inference_time += inference_time_ms

        result.setdefault("probabilities", None)
        result["model_used"] = model_name
        result["inference_time_ms"] = inference_time_ms

        return result

    def _predict_sklearn(
        self, model: Any, features: np.ndarray, return_probabilities: bool
    ) -> Dict[str, Any]:
        prediction = int(model.predict(features)[0])

        result = {
            "prediction": prediction,
            "class_name": self.CLASS_NAMES[prediction],
        }

        if return_probabilities and hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            result["probabilities"] = {
                name: float(p) for name, p in zip(self.CLASS_NAMES, proba)
            }
            result["confidence"] = float(max(proba))
        else:
            result["confidence"] = 1.0

        return result

    def _predict_pytorch(
        self, model: nn.Module, features: np.ndarray, return_probabilities: bool
    ) -> Dict[str, Any]:
        model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)

            if x.dim() == 2:
                if hasattr(model, "input_dim"):
                    x = x.unsqueeze(1)  # add channel dim

            output = model(x)
            proba = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(proba))

        result = {
            "prediction": prediction,
            "class_name": self.CLASS_NAMES[prediction],
            "confidence": float(max(proba)),
        }

        if return_probabilities:
            result["probabilities"] = {
                name: float(p) for name, p in zip(self.CLASS_NAMES, proba)
            }

        return result

    def predict_batch(
        self,
        model_name: Optional[str],
        features: np.ndarray,
        return_probabilities: bool = True,
    ) -> List[Dict[str, Any]]:

        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("No model specified and no default model set")

        with self._lock:
            if model_name not in self.models:
                if self.lazy_load:
                    success, _ = self.load_model(model_name)
                    if not success:
                        raise ValueError(f"Failed to load model: {model_name}")
                else:
                    raise ValueError(f"Model not loaded: {model_name}")
            model = self.models[model_name]
            self._model_refs[model_name] += 1
            self.model_access_times[model_name] = datetime.now(timezone.utc)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        start_time = time.time()

        try:
            if not (TORCH_AVAILABLE and isinstance(model, nn.Module)):
                predictions = model.predict(features)
                probas = (
                    model.predict_proba(features)
                    if return_probabilities and hasattr(model, "predict_proba")
                    else None
                )

                results = []
                for i, pred in enumerate(predictions):
                    pred_int = int(pred)
                    result = {
                        "prediction": pred_int,
                        "class_name": self.CLASS_NAMES[pred_int],
                        "model_used": model_name,
                    }
                    if probas is not None:
                        result["probabilities"] = {
                            name: float(p) for name, p in zip(self.CLASS_NAMES, probas[i])
                        }
                        result["confidence"] = float(max(probas[i]))
                    else:
                        result["confidence"] = 1.0
                    results.append(result)
            else:
                results = []
                for i in range(len(features)):
                    result = self._predict_pytorch(
                        model, features[i : i + 1], return_probabilities
                    )
                    result["model_used"] = model_name
                    results.append(result)

            inference_time_ms = (time.time() - start_time) * 1000
            per_sample_ms = inference_time_ms / len(features)

            for r in results:
                r["inference_time_ms"] = per_sample_ms

            self.inference_count += len(features)
            self.total_inference_time += inference_time_ms

            return results
        finally:
            with self._lock:
                self._model_refs[model_name] -= 1

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        info = self.model_info.get(model_name)
        if info:
            if model_name in self.model_access_times:
                info["last_used"] = self.model_access_times[model_name]
            return info
        return None

    def list_models(self) -> List[Dict]:
        models = []
        for name, info in self.model_info.items():
            model_info = {
                "name": name,
                "model_type": info.get("type", "unknown"),
                "version": info.get("version", "1.0.0"),
                "num_classes": info.get("num_classes", len(self.CLASS_NAMES)),
                "is_loaded": name in self.models,
                "input_dim": info.get("input_dim"),
                "num_parameters": info.get("num_parameters"),
                "load_time_ms": info.get("load_time_ms"),
                "last_used": self.model_access_times.get(name),
            }
            models.append(model_info)
        return models

    def get_statistics(self) -> Dict[str, Any]:
        avg_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else None
        )
        return {
            "total_inferences": self.inference_count,
            "average_inference_time_ms": avg_time,
            "models_loaded": len(self.models),
            "models_available": len(self.model_info),
            "default_model": self.default_model,
            "device": str(self.device) if self.device else "cpu",
        }

    def get_feature_names(
        self, model_name: Optional[str] = None
    ) -> Optional[List[str]]:
        model_name = model_name or self.default_model
        if model_name and model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, "feature_names_in_"):
                return list(model.feature_names_in_)
            elif hasattr(model, "feature_names"):
                return list(model.feature_names)
        return None

    def register_model(
        self,
        model_name: str,
        model: Any,
        model_type: str = "unknown",
        feature_names: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            self.models[model_name] = model
            self.model_access_times[model_name] = datetime.now(timezone.utc)

            self.model_info[model_name] = {
                "type": model_type,
                "is_loaded": True,
                "version": "1.0.0",
                "num_classes": len(self.CLASS_NAMES),
                "feature_names": feature_names,
            }

            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                num_params = sum(p.numel() for p in model.parameters())
                self.model_info[model_name]["num_parameters"] = num_params

            if not self.default_model:
                self.default_model = model_name

            self.logger.info(f"Registered model: {model_name}")

    def is_model_loaded(self, model_name: str) -> bool:
        return model_name in self.models

    def get_default_model(self) -> Optional[str]:
        return self.default_model

    def set_default_model(self, model_name: str) -> bool:
        if model_name in self.model_info:
            self.default_model = model_name
            return True
        return False

    def _get_fallback_model(self, failed_model: str) -> Optional[str]:
        for name in self.models:
            if name != failed_model:
                return name
        return None

    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        model_names: Optional[List[str]] = None,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        if model_names is None:
            model_names = list(self.models.keys())

        if len(model_names) < 2:
            raise ValueError("Need at least 2 loaded models for uncertainty")

        results = []
        for name in model_names:
            results.append(self.predict(name, features.copy(), return_probabilities))

        # Majority vote
        votes = [r["prediction"] for r in results]
        majority = max(set(votes), key=votes.count)
        agreement = votes.count(majority) / len(votes)

        confidences = [r["confidence"] for r in results]

        # Per-class intervals
        class_intervals = {}
        all_probs = [r["probabilities"] for r in results if r.get("probabilities")]
        if all_probs:
            for cls in all_probs[0]:
                vals = [p[cls] for p in all_probs]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                class_intervals[cls] = {
                    "mean": mean,
                    "std": std,
                    "ci_lower": max(0.0, mean - 1.96 * std),
                    "ci_upper": min(1.0, mean + 1.96 * std),
                }

        return {
            "prediction": majority,
            "class_name": self.CLASS_NAMES[majority],
            "confidence": float(np.mean(confidences)),
            "probabilities": (
                {cls: ci["mean"] for cls, ci in class_intervals.items()}
                if class_intervals
                else None
            ),
            "uncertainty": {
                "confidence_std": float(np.std(confidences)),
                "model_agreement": agreement,
                "num_models": len(model_names),
                "class_intervals": class_intervals or None,
            },
            "models_used": model_names,
            "inference_time_ms": sum(r["inference_time_ms"] for r in results),
        }

    def verify_model(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.models:
            return {"status": "not_loaded", "model": model_name}

        try:
            names = self.get_feature_names(model_name)
            dim = len(names) if names else 20
            dummy = np.zeros(dim)
            self.predict(model_name, dummy)
            return {"status": "healthy", "model": model_name}
        except Exception as e:
            return {"status": "unhealthy", "model": model_name, "error": str(e)}
