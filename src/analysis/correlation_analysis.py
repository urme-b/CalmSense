from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_config import LoggerMixin


class CorrelationAnalyzer(LoggerMixin):
    def __init__(self):

        self.logger.debug("CorrelationAnalyzer initialized")

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        subset = df[features].copy()

        if method in ["pearson", "spearman", "kendall"]:
            corr_matrix = subset.corr(method=method)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'"
            )

        self.logger.info(
            f"Computed {method} correlation matrix: {len(features)} features"
        )

        return corr_matrix

    def compute_correlation_with_pvalues(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        n_features = len(features)
        corr_matrix = np.zeros((n_features, n_features))
        pval_matrix = np.zeros((n_features, n_features))

        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                elif i < j:
                    x = df[feat1].dropna()
                    y = df[feat2].dropna()

                    # Align on common indices
                    common_idx = x.index.intersection(y.index)
                    x = x.loc[common_idx]
                    y = y.loc[common_idx]

                    if len(x) < 3:
                        corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                        pval_matrix[i, j] = pval_matrix[j, i] = np.nan
                        continue

                    if method == "pearson":
                        r, p = stats.pearsonr(x, y)
                    elif method == "spearman":
                        r, p = stats.spearmanr(x, y)
                    elif method == "kendall":
                        r, p = stats.kendalltau(x, y)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    corr_matrix[i, j] = corr_matrix[j, i] = r
                    pval_matrix[i, j] = pval_matrix[j, i] = p

        corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
        pval_df = pd.DataFrame(pval_matrix, index=features, columns=features)

        return corr_df, pval_df

    def compute_vif(
        self, df: pd.DataFrame, features: Optional[List[str]] = None
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove features with zero
        valid_features = []
        for f in features:
            if df[f].std() > 1e-10:
                valid_features.append(f)
            else:
                self.logger.warning(f"Skipping {f}: zero variance")

        X = df[valid_features].dropna()

        if len(X) < len(valid_features) + 1:
            self.logger.warning("Insufficient samples for VIF calculation")
            return pd.DataFrame(
                {"feature": valid_features, "VIF": [np.nan] * len(valid_features)}
            )

        vif_values = []

        for i, feature in enumerate(valid_features):
            # Prepare y (target feature)
            y = X[feature].values
            X_others = X[[f for f in valid_features if f != feature]].values

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X_others)), X_others])

            try:
                # OLS regression: y =
                # β = (X'X)^(-1) X'y
                XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
                beta = XtX_inv @ X_with_intercept.T @ y

                # Predictions and R²
                y_pred = X_with_intercept @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_tot > 0:
                    r_squared = 1 - ss_res / ss_tot
                    r_squared = min(max(r_squared, 0), 0.9999)  # Bound to avoid inf
                    vif = 1 / (1 - r_squared)
                else:
                    vif = 1.0

            except Exception as e:
                self.logger.warning(f"VIF calculation failed for {feature}: {e}")
                vif = np.nan

            vif_values.append(vif)

        vif_df = pd.DataFrame({"feature": valid_features, "VIF": vif_values})

        # Add interpretation
        def interpret_vif(vif):
            if np.isnan(vif):
                return "unknown"
            elif vif < 5:
                return "low"
            elif vif < 10:
                return "moderate"
            else:
                return "high"

        vif_df["interpretation"] = vif_df["VIF"].apply(interpret_vif)
        vif_df = vif_df.sort_values("VIF", ascending=False)

        n_high = (vif_df["VIF"] > 10).sum()
        self.logger.info(
            f"VIF computed: {n_high}/{len(valid_features)} features have VIF > 10"
        )

        return vif_df

    def find_highly_correlated(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.8,
        method: str = "pearson",
    ) -> List[Tuple[str, str, float]]:

        corr_matrix = self.compute_correlation_matrix(df, features, method)

        high_corr_pairs = []

        features = corr_matrix.columns.tolist()
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold:
                        high_corr_pairs.append((feat1, feat2, corr))

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        self.logger.info(
            f"Found {len(high_corr_pairs)} feature pairs with |r| >= {threshold}"
        )

        return high_corr_pairs

    def remove_multicollinear(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        vif_threshold: float = 10.0,
        max_iterations: int = 100,
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        current_features = features.copy()
        removed_features = []

        for iteration in range(max_iterations):
            vif_df = self.compute_vif(df, current_features)

            max_vif = vif_df["VIF"].max()

            if np.isnan(max_vif) or max_vif <= vif_threshold:
                break

            # Remove feature with highest
            worst_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            current_features.remove(worst_feature)
            removed_features.append(worst_feature)

            self.logger.debug(
                f"Iteration {iteration + 1}: Removed {worst_feature} (VIF={max_vif:.2f})"
            )

        self.logger.info(
            f"Multicollinearity removal: {len(removed_features)} features removed, "
            f"{len(current_features)} remaining"
        )

        if removed_features:
            self.logger.info(f"Removed features: {removed_features}")

        return df[current_features]

    def compute_partial_correlation(
        self, df: pd.DataFrame, x: str, y: str, controlling: List[str]
    ) -> Dict[str, float]:

        data = df[[x, y] + controlling].dropna()

        if len(data) < len(controlling) + 3:
            return {"partial_r": np.nan, "p_value": np.nan}

        # Residualize x and y
        X_control = data[controlling].values
        X_control = np.column_stack([np.ones(len(data)), X_control])

        # Residuals of x
        x_vals = data[x].values
        beta_x = np.linalg.lstsq(X_control, x_vals, rcond=None)[0]
        residuals_x = x_vals - X_control @ beta_x

        # Residuals of y
        y_vals = data[y].values
        beta_y = np.linalg.lstsq(X_control, y_vals, rcond=None)[0]
        residuals_y = y_vals - X_control @ beta_y

        # Correlation of residuals
        partial_r, p_value = stats.pearsonr(residuals_x, residuals_y)

        return {
            "partial_r": partial_r,
            "p_value": p_value,
            "df": len(data) - len(controlling) - 2,
        }

    def generate_network_graph_data(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, List]:

        corr_matrix = self.compute_correlation_matrix(df, features, method="spearman")
        features = corr_matrix.columns.tolist()

        nodes = [{"id": f, "label": f} for f in features]

        edges = []
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j:
                    corr = corr_matrix.loc[feat1, feat2]
                    if abs(corr) >= threshold:
                        edges.append(
                            {
                                "source": feat1,
                                "target": feat2,
                                "weight": corr,
                                "abs_weight": abs(corr),
                            }
                        )

        return {"nodes": nodes, "edges": edges, "threshold": threshold}

    def cluster_features_by_correlation(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_clusters: int = 5,
        method: str = "spearman",
    ) -> Dict[int, List[str]]:

        corr_matrix = self.compute_correlation_matrix(df, features, method)
        features = corr_matrix.columns.tolist()

        # Distance matrix (1 -
        distance_matrix = 1 - np.abs(corr_matrix.values)

        try:
            from scipy.cluster.hierarchy import linkage, fcluster

            # Hierarchical clustering
            linkage_matrix = linkage(
                distance_matrix[np.triu_indices(len(features), k=1)], method="average"
            )

            # Cut tree
            clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

            # Group features by cluster
            cluster_dict = {}
            for feat, cluster in zip(features, clusters):
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = []
                cluster_dict[cluster].append(feat)

            return cluster_dict

        except ImportError:
            self.logger.warning("scipy.cluster not available for clustering")
            return {0: features}

    def select_representative_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_select: int = 20,
        correlation_threshold: float = 0.8,
    ) -> List[str]:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = self.compute_correlation_matrix(df, features)
        selected = set(features)

        # Compute variance for tie-breaking
        variances = df[features].var().to_dict()

        # Find correlated pairs and
        for feat1 in features:
            if feat1 not in selected:
                continue

            for feat2 in features:
                if feat2 not in selected or feat1 == feat2:
                    continue

                corr = abs(corr_matrix.loc[feat1, feat2])
                if corr >= correlation_threshold:
                    # Remove feature with lower
                    if variances.get(feat1, 0) >= variances.get(feat2, 0):
                        selected.discard(feat2)
                    else:
                        selected.discard(feat1)
                        break

        # If still too many,
        selected = list(selected)
        if len(selected) > n_select:
            selected.sort(key=lambda f: variances.get(f, 0), reverse=True)
            selected = selected[:n_select]

        self.logger.info(
            f"Selected {len(selected)} representative features "
            f"(correlation threshold={correlation_threshold})"
        )

        return selected
