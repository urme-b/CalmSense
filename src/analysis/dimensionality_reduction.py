from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..logging_config import LoggerMixin


class DimensionalityReducer(LoggerMixin):
    def __init__(self):

        self._pca_model = None
        self._pca_loadings = None
        self._pca_explained_variance = None
        self._feature_names = None
        self.logger.debug("DimensionalityReducer initialized")

    def fit_pca(
        self,
        X: np.ndarray,
        n_components: Union[int, float] = 0.95,
        feature_names: Optional[List[str]] = None,
        standardize: bool = True,
    ) -> Dict[str, Any]:

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(n_features)
        ]

        # Handle missing values
        if np.any(np.isnan(X)):
            self.logger.warning("NaN values detected, imputing with column means")
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X = X.copy()
            for j in range(n_features):
                X[nan_mask[:, j], j] = col_means[j]

        # Standardize
        if standardize:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std[self._std == 0] = 1  # Avoid division by zero
            X_scaled = (X - self._mean) / self._std
        else:
            X_scaled = X
            self._mean = np.zeros(n_features)
            self._std = np.ones(n_features)

        # Compute PCA via SVD
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

        # Explained variance
        denom = max(n_samples - 1, 1)
        total_var = np.sum(S**2) / denom
        explained_variance = (S**2) / denom
        explained_variance_ratio = explained_variance / total_var

        # Determine number of components
        if isinstance(n_components, float) and n_components < 1:
            cumsum = np.cumsum(explained_variance_ratio)
            n_comp = np.argmax(cumsum >= n_components) + 1
        else:
            n_comp = min(int(n_components), min(n_samples, n_features))

        # Store components
        components = Vt[:n_comp, :]
        loadings = components.T  # (n_features, n_components)

        # Transform data
        transformed = X_scaled @ loadings

        # Store for later use
        self._pca_model = {
            "components": components,
            "mean": self._mean,
            "std": self._std,
            "n_components": n_comp,
        }
        self._pca_loadings = loadings
        self._pca_explained_variance = explained_variance_ratio

        # Scree plot data
        scree_data = pd.DataFrame(
            {
                "component": np.arange(1, len(explained_variance_ratio) + 1),
                "eigenvalue": explained_variance,
                "variance_explained": explained_variance_ratio,
                "cumulative_variance": np.cumsum(explained_variance_ratio),
            }
        )

        result = {
            "n_components": n_comp,
            "n_features_original": n_features,
            "explained_variance_ratio": explained_variance_ratio[:n_comp],
            "cumulative_variance": np.cumsum(explained_variance_ratio)[:n_comp],
            "total_variance_explained": np.sum(explained_variance_ratio[:n_comp]),
            "loadings": loadings[:, :n_comp],
            "components": components,
            "transformed_data": transformed,
            "scree_plot_data": scree_data,
            "feature_names": self._feature_names,
        }

        self.logger.info(
            f"PCA complete: {n_comp} components explain "
            f"{result['total_variance_explained'] * 100:.1f}% of variance"
        )

        return result

    def fit_tsne(
        self,
        X: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
        random_state: int = 42,
    ) -> np.ndarray:

        try:
            from sklearn.manifold import TSNE

            X = np.asarray(X)

            # Handle NaN
            if np.any(np.isnan(X)):
                self.logger.warning("NaN values detected, imputing with column means")
                col_means = np.nanmean(X, axis=0)
                nan_mask = np.isnan(X)
                X = X.copy()
                for j in range(X.shape[1]):
                    X[nan_mask[:, j], j] = col_means[j]

            # Perplexity must be less
            perplexity = min(perplexity, X.shape[0] - 1)

            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                max_iter=n_iter,
                learning_rate=learning_rate,
                random_state=random_state,
                init="pca",
            )

            embedded = tsne.fit_transform(X)

            self.logger.info(
                f"t-SNE complete: {X.shape[0]} samples embedded to {n_components}D "
                f"(perplexity={perplexity}, iterations={n_iter})"
            )

            return embedded

        except ImportError:
            self.logger.error("sklearn not available for t-SNE")
            return self._simple_tsne(X, n_components, perplexity, n_iter, random_state)

    def _simple_tsne(
        self,
        X: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> np.ndarray:

        np.random.seed(random_state)
        n_samples = X.shape[0]

        # Initialize randomly
        Y = np.random.randn(n_samples, n_components) * 0.01

        # Compute pairwise distances
        sum_X = np.sum(X**2, axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
        D = np.maximum(D, 0)

        # Compute affinities (simplified)
        P = np.exp(-D / (2 * perplexity**2))
        np.fill_diagonal(P, 0)
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)

        # Gradient descent
        learning_rate = 200.0
        momentum = 0.5
        velocity = np.zeros_like(Y)

        for iteration in range(n_iter):
            # Compute Q (t-distribution)
            sum_Y = np.sum(Y**2, axis=1)
            num = 1 / (1 + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * Y @ Y.T)
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Gradient
            PQ_diff = P - Q
            dY = np.zeros_like(Y)
            for i in range(n_samples):
                dY[i] = 4 * np.sum(
                    (PQ_diff[:, i] * num[:, i])[:, np.newaxis] * (Y[i] - Y), axis=0
                )

            # Momentum update
            velocity = momentum * velocity - learning_rate * dY
            Y = Y + velocity

            if iteration == 250:
                momentum = 0.8

        self.logger.warning(
            "Using simplified t-SNE; install sklearn for better results"
        )
        return Y

    def fit_umap(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ) -> np.ndarray:

        try:
            import umap

            X = np.asarray(X)

            # Handle NaN
            if np.any(np.isnan(X)):
                self.logger.warning("NaN values detected, imputing with column means")
                col_means = np.nanmean(X, axis=0)
                nan_mask = np.isnan(X)
                X = X.copy()
                for j in range(X.shape[1]):
                    X[nan_mask[:, j], j] = col_means[j]

            # Ensure n_neighbors < n_samples
            n_neighbors = min(n_neighbors, X.shape[0] - 1)

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )

            embedded = reducer.fit_transform(X)

            self.logger.info(
                f"UMAP complete: {X.shape[0]} samples embedded to {n_components}D "
                f"(n_neighbors={n_neighbors}, min_dist={min_dist})"
            )

            return embedded

        except ImportError:
            self.logger.warning("umap-learn not available, falling back to PCA + t-SNE")
            # Fallback: PCA to 50
            if X.shape[1] > 50:
                pca_result = self.fit_pca(X, n_components=50)
                X_reduced = pca_result["transformed_data"]
            else:
                X_reduced = X
            return self.fit_tsne(X_reduced, n_components)

    def get_top_loadings(
        self,
        feature_names: Optional[List[str]] = None,
        n_top: int = 10,
        component: int = 1,
    ) -> pd.DataFrame:

        if self._pca_loadings is None:
            raise ValueError("Must fit PCA first before getting loadings")

        if feature_names is None:
            feature_names = self._feature_names

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self._pca_loadings.shape[0])]

        comp_idx = component - 1  # Convert to 0-indexed

        if comp_idx >= self._pca_loadings.shape[1]:
            raise ValueError(f"Component {component} not available")

        loadings = self._pca_loadings[:, comp_idx]
        abs_loadings = np.abs(loadings)
        total_loading = np.sum(abs_loadings)

        # Create DataFrame
        loading_df = pd.DataFrame(
            {
                "feature": feature_names,
                "loading": loadings,
                "abs_loading": abs_loadings,
                "contribution_pct": 100 * abs_loadings / total_loading
                if total_loading > 0
                else 0,
            }
        )

        # Sort by absolute loading
        loading_df = loading_df.sort_values("abs_loading", ascending=False)

        # Add explained variance info
        if self._pca_explained_variance is not None:
            explained_var = self._pca_explained_variance[comp_idx]
            loading_df["component_variance_explained"] = explained_var

        self.logger.debug(f"Top {n_top} loadings for PC{component}")

        return loading_df.head(n_top)

    def get_all_loadings(
        self,
        feature_names: Optional[List[str]] = None,
        n_components: Optional[int] = None,
    ) -> pd.DataFrame:

        if self._pca_loadings is None:
            raise ValueError("Must fit PCA first")

        if feature_names is None:
            feature_names = self._feature_names or [
                f"feature_{i}" for i in range(self._pca_loadings.shape[0])
            ]

        if n_components is None:
            n_components = self._pca_loadings.shape[1]

        loadings = self._pca_loadings[:, :n_components]

        df = pd.DataFrame(
            loadings,
            index=feature_names,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )

        return df

    def transform(self, X: np.ndarray) -> np.ndarray:

        if self._pca_model is None:
            raise ValueError("Must fit PCA first")

        X = np.asarray(X)
        X_scaled = (X - self._pca_model["mean"]) / self._pca_model["std"]
        loadings = self._pca_model["components"].T[:, : self._pca_model["n_components"]]

        return X_scaled @ loadings

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:

        if self._pca_model is None:
            raise ValueError("Must fit PCA first")

        components = self._pca_model["components"][: X_transformed.shape[1], :]
        X_reconstructed = X_transformed @ components

        # Unscale
        return X_reconstructed * self._pca_model["std"] + self._pca_model["mean"]

    def compute_reconstruction_error(
        self, X: np.ndarray, n_components: Optional[int] = None
    ) -> Dict[str, float]:

        if self._pca_model is None:
            raise ValueError("Must fit PCA first")

        if n_components is None:
            n_components = self._pca_model["n_components"]

        X = np.asarray(X)
        X_scaled = (X - self._pca_model["mean"]) / self._pca_model["std"]

        # Project and reconstruct
        loadings = self._pca_model["components"][:n_components, :].T
        X_projected = X_scaled @ loadings
        X_reconstructed = X_projected @ loadings.T

        # Unscale
        X_reconstructed_orig = (
            X_reconstructed * self._pca_model["std"] + self._pca_model["mean"]
        )

        # Compute errors
        mse = np.mean((X - X_reconstructed_orig) ** 2)
        rmse = np.sqrt(mse)
        normalized_error = mse / np.var(X)

        return {
            "n_components": n_components,
            "mse": mse,
            "rmse": rmse,
            "normalized_error": normalized_error,
            "variance_captured": 1 - normalized_error,
        }

    def generate_biplot_data(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_loadings: float = 1.0,
    ) -> Dict[str, Any]:

        if self._pca_loadings is None:
            raise ValueError("Must fit PCA first")

        if feature_names is None:
            feature_names = self._feature_names

        # Scores (projected data)
        X_scaled = (np.asarray(X) - self._pca_model["mean"]) / self._pca_model["std"]
        scores = X_scaled @ self._pca_loadings[:, :2]

        # Scale loadings for visualization
        loadings = self._pca_loadings[:, :2] * scale_loadings

        return {
            "scores": scores,
            "loadings": loadings,
            "feature_names": feature_names,
            "explained_variance": self._pca_explained_variance[:2]
            if self._pca_explained_variance is not None
            else None,
        }
