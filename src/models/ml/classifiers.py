"""Configured classifier factory.

Each builder returns a ready-to-fit scikit-learn-compatible estimator with the
hyperparameters used throughout the benchmark. A flat registry keeps the family
in one place; `get_classifier("rf")` is the only entry point callers need.
"""

from typing import Any, Callable, Dict

SEED = 42


def _logistic_regression(random_state: int = SEED, **kwargs: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def _random_forest(random_state: int = SEED, **kwargs: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def _xgboost(random_state: int = SEED, **kwargs: Any) -> Any:
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        **kwargs,
    )


def _lightgbm(random_state: int = SEED, **kwargs: Any) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        num_leaves=50,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=0,
        min_child_samples=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **kwargs,
    )


_REGISTRY: Dict[str, Callable[..., Any]] = {
    "lr": _logistic_regression,
    "logistic": _logistic_regression,
    "rf": _random_forest,
    "random_forest": _random_forest,
    "xgb": _xgboost,
    "xgboost": _xgboost,
    "lgbm": _lightgbm,
    "lightgbm": _lightgbm,
}


def get_classifier(name: str, **kwargs: Any) -> Any:
    """Build a configured estimator by short name (e.g. ``"rf"``, ``"xgb"``)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown classifier: {name}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
