from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ...logging_config import LoggerMixin


class ImbalanceHandler(LoggerMixin):
    STRATEGIES = ["smote", "adasyn", "smote_enn", "smote_tomek", "class_weight", "none"]

    def __init__(
        self,
        strategy: str = "smote",
        sampling_strategy: Union[str, Dict, float] = "auto",
        random_state: int = 42,
        n_neighbors: int = 5,
        n_jobs: int = -1,
    ):

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {self.STRATEGIES}"
            )

        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self._sampler = None
        self.logger.debug(f"ImbalanceHandler initialized with strategy={strategy}")

    def _create_sampler(self) -> Any:

        try:
            from imblearn.over_sampling import SMOTE, ADASYN
            from imblearn.combine import SMOTEENN, SMOTETomek
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for resampling: "
                "pip install imbalanced-learn"
            )

        if self.strategy == "smote":
            return SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.n_neighbors,
            )

        elif self.strategy == "adasyn":
            return ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
            )

        elif self.strategy == "smote_enn":
            return SMOTEENN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

        elif self.strategy == "smote_tomek":
            return SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

        elif self.strategy in ["class_weight", "none"]:
            return None

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        X = np.asarray(X)
        y = np.asarray(y)

        if self.strategy in ["class_weight", "none"]:
            self.logger.debug(f"No resampling applied (strategy={self.strategy})")
            return X, y

        # Get original distribution
        unique, counts = np.unique(y, return_counts=True)
        original_dist = dict(zip(unique, counts))
        self.logger.info(f"Original class distribution: {original_dist}")

        # Create and apply sampler
        sampler = self._create_sampler()

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)

            # Log new distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            new_dist = dict(zip(unique, counts))
            self.logger.info(f"Resampled class distribution: {new_dist}")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.warning(f"Resampling failed: {e}. Returning original data.")
            return X, y

    def get_class_weights(
        self, y: np.ndarray, weight_type: str = "balanced"
    ) -> Dict[int, float]:

        from sklearn.utils.class_weight import compute_class_weight

        y = np.asarray(y)
        classes = np.unique(y)

        if weight_type == "balanced":
            weights = compute_class_weight("balanced", classes=classes, y=y)
        elif weight_type == "sqrt_balanced":
            # Square root scaling for
            base_weights = compute_class_weight("balanced", classes=classes, y=y)
            weights = np.sqrt(base_weights)
        else:
            # Equal weights
            weights = np.ones(len(classes))

        class_weights = dict(zip(classes, weights))
        self.logger.debug(f"Computed class weights: {class_weights}")

        return class_weights

    def get_sample_weights(
        self, y: np.ndarray, weight_type: str = "balanced"
    ) -> np.ndarray:

        class_weights = self.get_class_weights(y, weight_type)
        return np.array([class_weights[label] for label in y])

    def analyze_imbalance(self, y: np.ndarray) -> Dict[str, Any]:

        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)

        total = len(y)
        proportions = counts / total

        # Imbalance ratio (max /
        imbalance_ratio = max(counts) / min(counts)

        # Shannon entropy (measure of
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(unique))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        analysis = {
            "n_classes": len(unique),
            "class_counts": dict(zip(unique.astype(int).tolist(), counts.tolist())),
            "class_proportions": dict(
                zip(unique.astype(int).tolist(), proportions.tolist())
            ),
            "imbalance_ratio": imbalance_ratio,
            "minority_class": int(unique[np.argmin(counts)]),
            "majority_class": int(unique[np.argmax(counts)]),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "is_highly_imbalanced": imbalance_ratio > 3.0,
        }

        self.logger.info(
            f"Class imbalance analysis: ratio={imbalance_ratio:.2f}, "
            f"entropy={normalized_entropy:.3f}"
        )

        return analysis

    def recommend_strategy(self, y: np.ndarray, n_samples: int) -> str:

        analysis = self.analyze_imbalance(y)

        imbalance_ratio = analysis["imbalance_ratio"]
        minority_count = min(analysis["class_counts"].values())

        # Decision logic
        if imbalance_ratio < 1.5:
            recommendation = "none"
            reason = "Low imbalance, no resampling needed"

        elif imbalance_ratio < 3.0:
            recommendation = "class_weight"
            reason = "Moderate imbalance, class weights sufficient"

        elif minority_count < 100:
            recommendation = "smote"
            reason = "Small minority class, SMOTE for synthetic samples"

        elif n_samples > 10000:
            recommendation = "smote_enn"
            reason = "Large dataset with high imbalance, SMOTE + cleaning"

        else:
            recommendation = "adasyn"
            reason = "Adaptive sampling for moderate-sized imbalanced data"

        self.logger.info(f"Recommended strategy: {recommendation} ({reason})")

        return recommendation

    def plot_class_distribution(
        self,
        y: np.ndarray,
        y_resampled: Optional[np.ndarray] = None,
        class_names: Optional[Dict[int, str]] = None,
        figsize: Tuple[int, int] = (10, 5),
    ) -> Any:

        import matplotlib.pyplot as plt

        y = np.asarray(y)

        fig, axes = plt.subplots(
            1, 2 if y_resampled is not None else 1, figsize=figsize
        )

        if y_resampled is None:
            axes = [axes]

        # Original distribution
        unique, counts = np.unique(y, return_counts=True)
        labels = [
            class_names.get(c, f"Class {c}") if class_names else f"Class {c}"
            for c in unique
        ]

        axes[0].bar(labels, counts, color="steelblue", edgecolor="black")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Original Distribution")

        for i, (label, count) in enumerate(zip(labels, counts)):
            axes[0].text(i, count + 0.5, str(count), ha="center", va="bottom")

        # Resampled distribution
        if y_resampled is not None:
            y_resampled = np.asarray(y_resampled)
            unique_r, counts_r = np.unique(y_resampled, return_counts=True)
            labels_r = [
                class_names.get(c, f"Class {c}") if class_names else f"Class {c}"
                for c in unique_r
            ]

            axes[1].bar(labels_r, counts_r, color="coral", edgecolor="black")
            axes[1].set_xlabel("Class")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"After {self.strategy.upper()}")

            for i, (label, count) in enumerate(zip(labels_r, counts_r)):
                axes[1].text(i, count + 0.5, str(count), ha="center", va="bottom")

        plt.tight_layout()
        return fig


def resample_with_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    strategy: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    handler = ImbalanceHandler(strategy=strategy, random_state=random_state)

    X_list, y_list, groups_list = [], [], []

    for group in np.unique(groups):
        mask = groups == group
        X_group = X[mask]
        y_group = y[mask]

        # Only resample if there's
        if len(np.unique(y_group)) > 1 and len(y_group) >= 10:
            try:
                X_resampled, y_resampled = handler.resample(X_group, y_group)
                groups_resampled = np.full(len(y_resampled), group)
            except Exception:
                X_resampled, y_resampled = X_group, y_group
                groups_resampled = np.full(len(y_group), group)
        else:
            X_resampled, y_resampled = X_group, y_group
            groups_resampled = np.full(len(y_group), group)

        X_list.append(X_resampled)
        y_list.append(y_resampled)
        groups_list.append(groups_resampled)

    return (np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list))
