from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...logging_config import LoggerMixin
from .base_model import BaseMLModel
from .classifiers import get_classifier
from .cross_validation import CrossValidator
from .ensemble import StackingEnsemble, VotingEnsemble
from .evaluation import ModelEvaluator
from .hyperparameter_tuning import HyperparameterTuner
from .imbalance_handler import ImbalanceHandler


class MLTrainingPipeline(LoggerMixin):
    # Default classifiers to evaluate
    DEFAULT_CLASSIFIERS = ["lr", "svm", "rf", "xgb", "lgbm"]

    def __init__(
        self,
        cv_strategy: str = "loso",
        n_tuning_trials: int = 30,
        imbalance_strategy: str = "class_weight",
        random_state: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
    ):

        self.cv_strategy = cv_strategy
        self.n_tuning_trials = n_tuning_trials
        self.imbalance_strategy = imbalance_strategy
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else None

        # Initialize components
        self.cv = CrossValidator(cv_strategy=cv_strategy, random_state=random_state)
        self.evaluator = ModelEvaluator()
        self.imbalance_handler = ImbalanceHandler(
            strategy=imbalance_strategy, random_state=random_state
        )

        # Results storage
        self.baseline_results = {}
        self.tuned_results = {}
        self.ensemble_results = {}
        self.best_model = None
        self.best_model_name = None

        self.logger.info(
            f"MLTrainingPipeline initialized: cv={cv_strategy}, "
            f"trials={n_tuning_trials}, imbalance={imbalance_strategy}"
        )

    def run_full_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        classifiers: Optional[List[str]] = None,
        tune_top_k: int = 3,
        create_ensemble: bool = True,
        metric: str = "f1_macro",
    ) -> Dict[str, Any]:

        X = np.asarray(X)
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)

        classifiers = classifiers or self.DEFAULT_CLASSIFIERS

        self.logger.info("=" * 60)
        self.logger.info("Starting ML Training Pipeline")
        self.logger.info("=" * 60)

        # Step 1: Analyze data
        self.logger.info("\n[1/5] Analyzing Data")
        data_analysis = self._analyze_data(X, y, groups)
        self.logger.info(f"  Samples: {data_analysis['n_samples']}")
        self.logger.info(f"  Features: {data_analysis['n_features']}")
        self.logger.info(f"  Classes: {data_analysis['n_classes']}")
        if groups is not None:
            self.logger.info(f"  Subjects: {data_analysis['n_subjects']}")

        # Step 2: Baseline evaluation
        self.logger.info("\n[2/5] Running Baseline Evaluation")
        self.baseline_results = self._run_baseline_evaluation(
            X, y, groups, classifiers, metric
        )

        # Rank classifiers
        ranked = sorted(
            self.baseline_results.items(),
            key=lambda x: x[1].get(f"{metric}_mean", 0),
            reverse=True,
        )

        self.logger.info("\nBaseline Results (ranked by {})".format(metric))
        for i, (name, results) in enumerate(ranked):
            self.logger.info(
                f"  {i + 1}. {name}: {metric}={results[f'{metric}_mean']:.4f} "
                f"± {results[f'{metric}_std']:.4f}"
            )

        # Step 3: Hyperparameter tuning
        self.logger.info(f"\n[3/5] Tuning Top-{tune_top_k} Classifiers")
        top_classifiers = [name for name, _ in ranked[:tune_top_k]]
        self.tuned_results = self._run_hyperparameter_tuning(
            X, y, groups, top_classifiers, metric
        )

        # Step 4: Create ensembles
        if create_ensemble and len(self.tuned_results) >= 2:
            self.logger.info("\n[4/5] Creating Ensemble Models")
            self.ensemble_results = self._create_and_evaluate_ensembles(
                X, y, groups, metric
            )
        else:
            self.logger.info("\n[4/5] Skipping Ensemble (not enough models)")

        # Step 5: Final comparison
        self.logger.info("\n[5/5] Final Model Comparison")
        comparison_df = self._create_comparison(metric)

        # Determine best model
        all_results = {}
        all_results.update(
            {f"baseline_{k}": v for k, v in self.baseline_results.items()}
        )
        all_results.update({f"tuned_{k}": v for k, v in self.tuned_results.items()})
        all_results.update(self.ensemble_results)

        best_name, best_results = max(
            all_results.items(), key=lambda x: x[1].get(f"{metric}_mean", 0)
        )

        self.best_model_name = best_name
        self.logger.info(f"\nBest Model: {best_name}")
        self.logger.info(f"  {metric}: {best_results[f'{metric}_mean']:.4f}")
        self.logger.info(f"  Accuracy: {best_results.get('accuracy_mean', 0):.4f}")

        # Compile final results
        pipeline_results = {
            "baseline_results": self.baseline_results,
            "tuned_results": self.tuned_results,
            "ensemble_results": self.ensemble_results,
            "best_model_name": best_name,
            "best_accuracy": best_results.get("accuracy_mean", 0),
            "best_f1": best_results.get("f1_macro_mean", 0),
            "comparison_df": comparison_df,
            "data_analysis": data_analysis,
        }

        # Save results if output
        if self.output_dir:
            self._save_results(pipeline_results)

        return pipeline_results

    def _analyze_data(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]
    ) -> Dict[str, Any]:

        analysis = {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        }

        if groups is not None:
            analysis["n_subjects"] = len(np.unique(groups))

        # Imbalance analysis
        imbalance = self.imbalance_handler.analyze_imbalance(y)
        analysis["imbalance_ratio"] = imbalance["imbalance_ratio"]
        analysis["is_highly_imbalanced"] = imbalance["is_highly_imbalanced"]

        return analysis

    def _run_baseline_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        classifiers: List[str],
        metric: str,
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        for clf_name in classifiers:
            self.logger.info(f"  Evaluating {clf_name}...")
            try:
                model = get_classifier(clf_name, random_state=self.random_state)

                # Get class weights if
                if self.imbalance_strategy == "class_weight":
                    class_weights = self.imbalance_handler.get_class_weights(y)
                    model.kwargs["class_weight"] = class_weights

                cv_results = self.cv.cross_validate(model, X, y, groups)
                results[clf_name] = cv_results

            except Exception as e:
                self.logger.warning(f"  {clf_name} failed: {e}")
                results[clf_name] = {
                    "error": str(e),
                    f"{metric}_mean": 0,
                    f"{metric}_std": 0,
                }

        return results

    def _run_hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        classifiers: List[str],
        metric: str,
    ) -> Dict[str, Dict[str, Any]]:

        tuner = HyperparameterTuner(
            cv_strategy=self.cv_strategy,
            n_trials=self.n_tuning_trials,
            random_state=self.random_state,
        )

        results = {}

        for clf_name in classifiers:
            self.logger.info(f"  Tuning {clf_name}...")
            try:
                best_params, best_score, study = tuner.tune(
                    clf_name, X, y, groups, metric, show_progress_bar=False
                )

                # Re-evaluate with best params
                tuned_model = get_classifier(clf_name, **best_params)
                cv_results = self.cv.cross_validate(tuned_model, X, y, groups)
                cv_results["best_params"] = best_params
                cv_results["tuning_score"] = best_score

                results[clf_name] = cv_results
                self.logger.info(
                    f"    Best: {metric}={cv_results[f'{metric}_mean']:.4f}"
                )

            except Exception as e:
                self.logger.warning(f"  Tuning {clf_name} failed: {e}")

        return results

    def _create_and_evaluate_ensembles(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray], metric: str
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        # Get tuned models
        tuned_models = []
        for clf_name, clf_results in self.tuned_results.items():
            params = clf_results.get("best_params", {})
            model = get_classifier(clf_name, **params, random_state=self.random_state)
            tuned_models.append((clf_name, model))

        if len(tuned_models) < 2:
            self.logger.warning("Not enough models for ensemble")
            return results

        # Voting Ensemble
        self.logger.info("  Creating Voting Ensemble...")
        try:
            voting = VotingEnsemble(
                models=tuned_models, voting="soft", random_state=self.random_state
            )

            # Manual CV for ensemble
            voting_results = self._cross_validate_ensemble(voting, X, y, groups)
            results["voting_ensemble"] = voting_results
            self.logger.info(
                f"    Voting: {metric}={voting_results[f'{metric}_mean']:.4f}"
            )
        except Exception as e:
            self.logger.warning(f"  Voting ensemble failed: {e}")

        # Stacking Ensemble
        self.logger.info("  Creating Stacking Ensemble...")
        try:
            stacking = StackingEnsemble(
                base_models=tuned_models,
                meta_learner="lr",
                use_probas=True,
                random_state=self.random_state,
            )

            stacking_results = self._cross_validate_ensemble(stacking, X, y, groups)
            results["stacking_ensemble"] = stacking_results
            self.logger.info(
                f"    Stacking: {metric}={stacking_results[f'{metric}_mean']:.4f}"
            )
        except Exception as e:
            self.logger.warning(f"  Stacking ensemble failed: {e}")

        return results

    def _cross_validate_ensemble(
        self,
        ensemble: Union[VotingEnsemble, StackingEnsemble],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Dict[str, Any]:

        from sklearn.metrics import accuracy_score, f1_score

        # Get splits based on
        if self.cv_strategy == "loso" and groups is not None:
            splits = list(self.cv.get_loso_splits(X, y, groups))
        else:
            splits = list(self.cv.get_kfold_splits(X, y))

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone ensemble (create new
            ensemble_clone = ensemble.__class__(**ensemble.get_params())
            ensemble_clone.fit(X_train, y_train)

            y_pred = ensemble_clone.predict(X_test)

            fold_results.append(
                {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                }
            )

        # Aggregate
        results = {
            "cv_strategy": self.cv_strategy,
            "n_folds": len(splits),
        }

        for metric in ["accuracy", "f1_macro", "f1_weighted"]:
            values = [f[metric] for f in fold_results]
            results[f"{metric}_mean"] = np.mean(values)
            results[f"{metric}_std"] = np.std(values)

        return results

    def _create_comparison(self, metric: str) -> pd.DataFrame:

        rows = []

        # Baseline models
        for name, results in self.baseline_results.items():
            if "error" not in results:
                rows.append(
                    {
                        "model": name,
                        "type": "baseline",
                        "accuracy_mean": results.get("accuracy_mean", np.nan),
                        "accuracy_std": results.get("accuracy_std", np.nan),
                        "f1_macro_mean": results.get("f1_macro_mean", np.nan),
                        "f1_macro_std": results.get("f1_macro_std", np.nan),
                    }
                )

        # Tuned models
        for name, results in self.tuned_results.items():
            rows.append(
                {
                    "model": f"{name}_tuned",
                    "type": "tuned",
                    "accuracy_mean": results.get("accuracy_mean", np.nan),
                    "accuracy_std": results.get("accuracy_std", np.nan),
                    "f1_macro_mean": results.get("f1_macro_mean", np.nan),
                    "f1_macro_std": results.get("f1_macro_std", np.nan),
                }
            )

        # Ensembles
        for name, results in self.ensemble_results.items():
            rows.append(
                {
                    "model": name,
                    "type": "ensemble",
                    "accuracy_mean": results.get("accuracy_mean", np.nan),
                    "accuracy_std": results.get("accuracy_std", np.nan),
                    "f1_macro_mean": results.get("f1_macro_mean", np.nan),
                    "f1_macro_std": results.get("f1_macro_std", np.nan),
                }
            )

        df = pd.DataFrame(rows)

        if len(df) > 0:
            df = df.sort_values(f"{metric}_mean", ascending=False)

        return df

    def _save_results(self, results: Dict[str, Any]):

        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison DataFrame
        if "comparison_df" in results:
            csv_path = self.output_dir / "model_comparison.csv"
            results["comparison_df"].to_csv(csv_path, index=False)
            self.logger.info(f"Saved comparison to {csv_path}")

    def run_baseline_only(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        classifiers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:

        classifiers = classifiers or self.DEFAULT_CLASSIFIERS
        return self._run_baseline_evaluation(X, y, groups, classifiers, "f1_macro")

    def get_best_model(
        self, X: np.ndarray, y: np.ndarray, model_name: Optional[str] = None
    ) -> BaseMLModel:

        if model_name is None:
            model_name = self.best_model_name

        # Parse model name to
        if model_name.startswith("tuned_"):
            clf_name = model_name.replace("tuned_", "")
            params = self.tuned_results.get(clf_name, {}).get("best_params", {})
        elif model_name.startswith("baseline_"):
            clf_name = model_name.replace("baseline_", "")
            params = {}
        else:
            # Assume it's a raw
            clf_name = model_name
            params = self.tuned_results.get(clf_name, {}).get("best_params", {})

        model = get_classifier(clf_name, **params, random_state=self.random_state)
        model.fit(X, y)

        self.logger.info(f"Trained final model: {clf_name}")
        return model
