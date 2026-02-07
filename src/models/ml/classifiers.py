from typing import Any, Optional


from .base_model import BaseMLModel


class LogisticRegressionClassifier(BaseMLModel):
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        **kwargs,
    ):

        super().__init__("LogisticRegression", random_state, **kwargs)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _create_model(self) -> Any:

        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            **self.kwargs,
        )


class SVMClassifier(BaseMLModel):
    def __init__(
        self,
        C: float = 10.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        probability: bool = True,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        **kwargs,
    ):

        super().__init__("SVM", random_state, **kwargs)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.class_weight = class_weight

    def _create_model(self) -> Any:

        from sklearn.svm import SVC

        return SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs,
        )


class RandomForestClassifier(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 5,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):

        super().__init__("RandomForest", random_state, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _create_model(self) -> Any:

        from sklearn.ensemble import RandomForestClassifier as SklearnRF

        return SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )


class XGBoostClassifier(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        scale_pos_weight: float = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):

        super().__init__("XGBoost", random_state, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs

    def _create_model(self) -> Any:

        try:
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric="mlogloss",
                **self.kwargs,
            )
        except ImportError:
            self.logger.warning("XGBoost not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier as SklearnRF

            return SklearnRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )


class LightGBMClassifier(BaseMLModel):
    def __init__(
        self,
        num_leaves: int = 50,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        min_child_samples: int = 20,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):

        super().__init__("LightGBM", random_state, **kwargs)
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _create_model(self) -> Any:

        try:
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_samples=self.min_child_samples,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                **self.kwargs,
            )
        except ImportError:
            self.logger.warning("LightGBM not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier as SklearnRF

            return SklearnRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth > 0 else None,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )


class CatBoostClassifier(BaseMLModel):
    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3,
        subsample: float = 0.8,
        random_strength: float = 1,
        bagging_temperature: float = 1,
        auto_class_weights: Optional[str] = "Balanced",
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__("CatBoost", random_state, **kwargs)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.auto_class_weights = auto_class_weights
        self.verbose = verbose

    def _create_model(self) -> Any:

        try:
            import catboost as cb

            return cb.CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                subsample=self.subsample,
                random_strength=self.random_strength,
                bagging_temperature=self.bagging_temperature,
                auto_class_weights=self.auto_class_weights,
                random_state=self.random_state,
                verbose=self.verbose,
                **self.kwargs,
            )
        except ImportError:
            self.logger.warning("CatBoost not installed, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier as SklearnRF

            return SklearnRF(
                n_estimators=self.iterations,
                max_depth=self.depth,
                random_state=self.random_state,
                class_weight="balanced",
            )


# Factory function to get
def get_classifier(name: str, **kwargs) -> BaseMLModel:

    classifiers = {
        "lr": LogisticRegressionClassifier,
        "logistic": LogisticRegressionClassifier,
        "svm": SVMClassifier,
        "rf": RandomForestClassifier,
        "random_forest": RandomForestClassifier,
        "xgb": XGBoostClassifier,
        "xgboost": XGBoostClassifier,
        "lgbm": LightGBMClassifier,
        "lightgbm": LightGBMClassifier,
        "catboost": CatBoostClassifier,
        "cb": CatBoostClassifier,
    }

    name_lower = name.lower()
    if name_lower not in classifiers:
        raise ValueError(
            f"Unknown classifier: {name}. Available: {list(classifiers.keys())}"
        )

    return classifiers[name_lower](**kwargs)
