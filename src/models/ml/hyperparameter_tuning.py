from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...logging_config import LoggerMixin
from .base_model import BaseMLModel
from .classifiers import get_classifier
from .cross_validation import CrossValidator


class HyperparameterTuner(LoggerMixin):
    # Default search spaces for
    SEARCH_SPACES = {
        "lr": {
            "C": ("log_uniform", 1e-4, 100),
            "penalty": ("categorical", ["l1", "l2"]),
            "solver": ("categorical", ["liblinear", "saga"]),
        },
        "svm": {
            "C": ("log_uniform", 0.1, 100),
            "kernel": ("categorical", ["rbf", "linear", "poly"]),
            "gamma": ("categorical", ["scale", "auto"]),
        },
        "rf": {
            "n_estimators": ("int", 100, 500),
            "max_depth": ("int", 5, 20),
            "min_samples_leaf": ("int", 2, 10),
            "min_samples_split": ("int", 2, 10),
            "max_features": ("categorical", ["sqrt", "log2"]),
        },
        "xgb": {
            "n_estimators": ("int", 100, 500),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("log_uniform", 0.01, 0.3),
            "subsample": ("uniform", 0.6, 1.0),
            "colsample_bytree": ("uniform", 0.6, 1.0),
            "gamma": ("uniform", 0, 5),
            "reg_alpha": ("log_uniform", 1e-8, 10),
            "reg_lambda": ("log_uniform", 1e-8, 10),
        },
        "lgbm": {
            "num_leaves": ("int", 20, 100),
            "n_estimators": ("int", 100, 500),
            "learning_rate": ("log_uniform", 0.01, 0.3),
            "max_depth": ("int", 3, 15),
            "subsample": ("uniform", 0.6, 1.0),
            "colsample_bytree": ("uniform", 0.6, 1.0),
            "reg_alpha": ("log_uniform", 1e-8, 10),
            "reg_lambda": ("log_uniform", 1e-8, 10),
            "min_child_samples": ("int", 5, 50),
        },
        "catboost": {
            "iterations": ("int", 200, 1000),
            "depth": ("int", 4, 10),
            "learning_rate": ("log_uniform", 0.01, 0.3),
            "l2_leaf_reg": ("log_uniform", 1, 10),
            "subsample": ("uniform", 0.6, 1.0),
            "random_strength": ("uniform", 0, 3),
        },
    }

    def __init__(
        self,
        cv_strategy: str = "loso",
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        n_jobs: int = 1,
        random_state: int = 42,
        pruner: Optional[str] = "median",
        sampler: str = "tpe",
    ):

        self.cv_strategy = cv_strategy
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.pruner_type = pruner
        self.sampler_type = sampler

        self.logger.debug(
            f"HyperparameterTuner initialized: cv={cv_strategy}, "
            f"n_trials={n_trials}, sampler={sampler}"
        )

    def _create_sampler(self) -> Any:

        try:
            import optuna

            if self.sampler_type == "tpe":
                return optuna.samplers.TPESampler(seed=self.random_state)
            elif self.sampler_type == "random":
                return optuna.samplers.RandomSampler(seed=self.random_state)
            elif self.sampler_type == "cmaes":
                return optuna.samplers.CmaEsSampler(seed=self.random_state)
            else:
                return optuna.samplers.TPESampler(seed=self.random_state)
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter tuning")

    def _create_pruner(self) -> Any:

        try:
            import optuna

            if self.pruner_type == "median":
                return optuna.pruners.MedianPruner(
                    n_startup_trials=5, n_warmup_steps=0, interval_steps=1
                )
            elif self.pruner_type == "hyperband":
                return optuna.pruners.HyperbandPruner()
            else:
                return optuna.pruners.NopPruner()
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter tuning")

    def _sample_params(
        self, trial: Any, classifier_name: str, custom_space: Optional[Dict] = None
    ) -> Dict[str, Any]:

        search_space = custom_space or self.SEARCH_SPACES.get(classifier_name, {})
        params = {}

        for param_name, param_spec in search_space.items():
            param_type = param_spec[0]

            if param_type == "int":
                _, low, high = param_spec
                params[param_name] = trial.suggest_int(param_name, low, high)

            elif param_type == "uniform":
                _, low, high = param_spec
                params[param_name] = trial.suggest_float(param_name, low, high)

            elif param_type == "log_uniform":
                _, low, high = param_spec
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=True
                )

            elif param_type == "categorical":
                _, choices = param_spec
                params[param_name] = trial.suggest_categorical(param_name, choices)

        return params

    def _create_objective(
        self,
        classifier_name: str,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        metric: str,
        custom_space: Optional[Dict],
    ) -> Callable:

        cv = CrossValidator(
            cv_strategy=self.cv_strategy, random_state=self.random_state
        )

        def objective(trial):
            params = self._sample_params(trial, classifier_name, custom_space)
            model = get_classifier(classifier_name, **params)

            try:
                results = cv.cross_validate(model, X, y, groups)
                score = results.get(f"{metric}_mean", results.get("accuracy_mean", 0))

                if np.isnan(score):
                    return 0.0

                return score

            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0

        return objective

    def tune(
        self,
        classifier_name: str,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        metric: str = "f1_macro",
        custom_space: Optional[Dict] = None,
        show_progress_bar: bool = True,
    ) -> Tuple[Dict[str, Any], float, Any]:

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required: pip install optuna")

        X = np.asarray(X)
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)

        self.logger.info(
            f"Starting hyperparameter tuning for {classifier_name}: "
            f"{self.n_trials} trials, metric={metric}"
        )

        # Create study
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        # Create objective
        objective = self._create_objective(
            classifier_name, X, y, groups, metric, custom_space
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
        )

        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(
            f"Tuning complete: best {metric}={best_score:.4f}, params={best_params}"
        )

        return best_params, best_score, study

    def tune_multiple_classifiers(
        self,
        classifier_names: List[str],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        metric: str = "f1_macro",
    ) -> Dict[str, Tuple[Dict, float, Any]]:

        results = {}

        for name in classifier_names:
            self.logger.info(f"Tuning {name}...")
            try:
                best_params, best_score, study = self.tune(name, X, y, groups, metric)
                results[name] = (best_params, best_score, study)
            except Exception as e:
                self.logger.error(f"Failed to tune {name}: {e}")
                results[name] = ({}, 0.0, None)

        # Log comparison
        self.logger.info("Tuning comparison:")
        for name, (params, score, _) in sorted(
            results.items(), key=lambda x: x[1][1], reverse=True
        ):
            self.logger.info(f"  {name}: {metric}={score:.4f}")

        return results

    def get_best_model(
        self, classifier_name: str, best_params: Dict[str, Any]
    ) -> BaseMLModel:

        return get_classifier(classifier_name, **best_params)

    @staticmethod
    def plot_optimization_history(study: Any, title: str = "Optimization History"):

        try:
            import optuna

            return optuna.visualization.matplotlib.plot_optimization_history(study)
        except ImportError:
            raise ImportError("optuna with matplotlib is required for plotting")

    @staticmethod
    def plot_param_importances(study: Any, title: str = "Parameter Importances"):

        try:
            import optuna

            return optuna.visualization.matplotlib.plot_param_importances(study)
        except ImportError:
            raise ImportError("optuna with matplotlib is required for plotting")

    def get_search_space(self, classifier_name: str) -> Dict:

        name_lower = classifier_name.lower()
        name_map = {
            "logistic": "lr",
            "logisticregression": "lr",
            "svc": "svm",
            "randomforest": "rf",
            "xgboost": "xgb",
            "lightgbm": "lgbm",
        }
        name_lower = name_map.get(name_lower, name_lower)

        return self.SEARCH_SPACES.get(name_lower, {})
