from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...logging_config import LoggerMixin
from .base_model import BaseMLModel


class CrossValidator(LoggerMixin):
    def __init__(
        self, cv_strategy: str = "loso", n_splits: int = 5, random_state: int = 42
    ):

        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger.debug(f"CrossValidator initialized with strategy={cv_strategy}")

    def get_loso_splits(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        from sklearn.model_selection import LeaveOneGroupOut

        logo = LeaveOneGroupOut()

        for train_idx, test_idx in logo.split(X, y, groups):
            yield train_idx, test_idx

    def get_kfold_splits(
        self, X: np.ndarray, y: np.ndarray, n_splits: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        from sklearn.model_selection import KFold

        n_splits = n_splits or self.n_splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X, y):
            yield train_idx, test_idx

    def get_stratified_splits(
        self, X: np.ndarray, y: np.ndarray, n_splits: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        from sklearn.model_selection import StratifiedKFold

        n_splits = n_splits or self.n_splits
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        for train_idx, test_idx in skf.split(X, y):
            yield train_idx, test_idx

    def get_group_kfold_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        from sklearn.model_selection import GroupKFold

        n_splits = n_splits or self.n_splits
        gkf = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in gkf.split(X, y, groups):
            yield train_idx, test_idx

    def cross_validate(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        X = np.asarray(X)
        y = np.asarray(y)

        # Get appropriate splits
        if self.cv_strategy == "loso":
            if groups is None:
                raise ValueError("LOSO requires groups (subject IDs)")
            splits = list(self.get_loso_splits(X, y, groups))
        elif self.cv_strategy == "kfold":
            splits = list(self.get_kfold_splits(X, y))
        elif self.cv_strategy == "stratified":
            splits = list(self.get_stratified_splits(X, y))
        elif self.cv_strategy == "group_kfold":
            if groups is None:
                raise ValueError("Group K-Fold requires groups")
            splits = list(self.get_group_kfold_splits(X, y, groups))
        else:
            raise ValueError(f"Unknown cv_strategy: {self.cv_strategy}")

        # Metrics storage
        fold_results = []
        all_predictions = np.zeros_like(y)
        all_probabilities = np.zeros((len(y), len(np.unique(y))))

        n_classes = len(np.unique(y))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone and fit model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)

            # Predictions
            y_pred = model_clone.predict(X_test)
            y_proba = model_clone.predict_proba(X_test)

            # Store predictions
            all_predictions[test_idx] = y_pred
            all_probabilities[test_idx] = y_proba

            # Compute metrics
            fold_metrics = {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            }

            # AUC-ROC (multi-class)
            try:
                if n_classes == 2:
                    fold_metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    fold_metrics["auc_roc"] = roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="macro"
                    )
            except Exception:
                fold_metrics["auc_roc"] = np.nan

            # Per-class accuracy (if groups
            if groups is not None:
                test_subjects = np.unique(groups[test_idx])
                fold_metrics["test_subjects"] = test_subjects.tolist()

            fold_results.append(fold_metrics)

            self.logger.debug(
                f"Fold {fold_idx + 1}: acc={fold_metrics['accuracy']:.3f}, "
                f"f1={fold_metrics['f1_macro']:.3f}"
            )

        # Aggregate results
        results = {
            "cv_strategy": self.cv_strategy,
            "n_folds": len(splits),
            "n_samples": len(y),
            "n_classes": n_classes,
            "fold_results": fold_results,
        }

        # Compute mean and std
        for metric in ["accuracy", "f1_macro", "f1_weighted", "auc_roc"]:
            values = [
                f[metric] for f in fold_results if not np.isnan(f.get(metric, np.nan))
            ]
            if values:
                results[f"{metric}_mean"] = np.mean(values)
                results[f"{metric}_std"] = np.std(values)
                results[f"{metric}_ci95"] = 1.96 * np.std(values) / np.sqrt(len(values))
            else:
                results[f"{metric}_mean"] = np.nan
                results[f"{metric}_std"] = np.nan

        if return_predictions:
            results["predictions"] = all_predictions
            results["probabilities"] = all_probabilities

        self.logger.info(
            f"{self.cv_strategy.upper()} CV complete: "
            f"accuracy={results['accuracy_mean']:.3f}±{results['accuracy_std']:.3f}, "
            f"F1={results['f1_macro_mean']:.3f}±{results['f1_macro_std']:.3f}"
        )

        return results

    def nested_cross_validate(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        param_grid: Dict[str, List[Any]],
        inner_cv: int = 3,
        scoring: str = "f1_macro",
    ) -> Dict[str, Any]:

        from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)

        outer_cv = LeaveOneGroupOut()
        outer_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            # Inner CV for hyperparameter
            from sklearn.model_selection import GroupKFold

            inner_cv_obj = GroupKFold(
                n_splits=min(inner_cv, len(np.unique(groups_train)))
            )

            base_model = model_class()
            grid_search = GridSearchCV(
                base_model.model if base_model.model else base_model._create_model(),
                param_grid,
                cv=inner_cv_obj,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )

            try:
                grid_search.fit(X_train, y_train, groups=groups_train)
                best_params = grid_search.best_params_
                best_model = grid_search.best_estimator_

                # Evaluate on outer test
                y_pred = best_model.predict(X_test)

                from sklearn.metrics import accuracy_score, f1_score

                outer_results.append(
                    {
                        "fold": fold_idx,
                        "best_params": best_params,
                        "best_inner_score": grid_search.best_score_,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_macro": f1_score(y_test, y_pred, average="macro"),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Fold {fold_idx} failed: {e}")
                outer_results.append({"fold": fold_idx, "error": str(e)})

        # Aggregate
        valid_results = [r for r in outer_results if "accuracy" in r]

        results = {
            "n_outer_folds": len(outer_results),
            "n_valid_folds": len(valid_results),
            "fold_results": outer_results,
        }

        if valid_results:
            results["accuracy_mean"] = np.mean([r["accuracy"] for r in valid_results])
            results["accuracy_std"] = np.std([r["accuracy"] for r in valid_results])
            results["f1_macro_mean"] = np.mean([r["f1_macro"] for r in valid_results])
            results["f1_macro_std"] = np.std([r["f1_macro"] for r in valid_results])

        return results

    def compare_cv_strategies(
        self, model: BaseMLModel, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> pd.DataFrame:

        strategies = ["loso", "kfold", "stratified"]
        results = []

        for strategy in strategies:
            cv = CrossValidator(cv_strategy=strategy)
            try:
                cv_results = cv.cross_validate(model, X, y, groups)
                results.append(
                    {
                        "strategy": strategy.upper(),
                        "accuracy_mean": cv_results["accuracy_mean"],
                        "accuracy_std": cv_results["accuracy_std"],
                        "f1_macro_mean": cv_results["f1_macro_mean"],
                        "f1_macro_std": cv_results["f1_macro_std"],
                        "n_folds": cv_results["n_folds"],
                    }
                )
            except Exception as e:
                self.logger.warning(f"{strategy} failed: {e}")

        comparison_df = pd.DataFrame(results)

        # Calculate overfitting gap (K-Fold
        if len(comparison_df) > 1:
            loso_acc = comparison_df[comparison_df["strategy"] == "LOSO"][
                "accuracy_mean"
            ].values
            kfold_acc = comparison_df[comparison_df["strategy"] == "KFOLD"][
                "accuracy_mean"
            ].values

            if len(loso_acc) > 0 and len(kfold_acc) > 0:
                gap = (kfold_acc[0] - loso_acc[0]) * 100
                self.logger.info(f"Overfitting gap (K-Fold - LOSO): {gap:.1f}%")

        return comparison_df

    def get_per_subject_performance(
        self, model: BaseMLModel, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> pd.DataFrame:

        from sklearn.metrics import accuracy_score, f1_score

        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)

        subject_results = []

        for train_idx, test_idx in self.get_loso_splits(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = np.unique(groups[test_idx])[0]

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            subject_results.append(
                {
                    "subject": test_subject,
                    "n_samples": len(test_idx),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                }
            )

        return pd.DataFrame(subject_results)
