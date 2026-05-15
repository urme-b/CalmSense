from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_config import LoggerMixin
from .hypothesis_testing import HypothesisTesting
from .correlation_analysis import CorrelationAnalyzer


class StatisticalFeatureSelector(LoggerMixin):
    def __init__(self, alpha: float = 0.05):

        self.alpha = alpha
        self.hypothesis_tester = HypothesisTesting(alpha=alpha)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.logger.debug(f"StatisticalFeatureSelector initialized with α={alpha}")

    def select_by_anova(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        alpha: Optional[float] = None,
        correction: str = "fdr_bh",
    ) -> List[str]:

        if alpha is None:
            alpha = self.alpha

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [subject_col, condition_col]]

        # Run ANOVA for all
        anova_results = self.hypothesis_tester.run_all_anova(
            df, features, subject_col, condition_col, correction
        )

        if anova_results.empty:
            return []

        # Filter by corrected p-value
        p_col = (
            "p_value_corrected"
            if "p_value_corrected" in anova_results.columns
            else "p_value"
        )
        significant = anova_results[anova_results[p_col] < alpha]

        selected_features = significant["feature"].tolist()

        self.logger.info(
            f"ANOVA selection: {len(selected_features)}/{len(features)} features "
            f"significant at α={alpha} ({correction} correction)"
        )

        return selected_features

    def select_by_effect_size(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        min_eta_squared: float = 0.06,
        effect_type: str = "partial_eta_squared",
    ) -> List[str]:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [subject_col, condition_col]]

        selected_features = []

        for feature in features:
            try:
                result = self.hypothesis_tester.repeated_measures_anova(
                    df, feature, subject_col, condition_col
                )

                effect_size = result.get(effect_type, 0)

                if not np.isnan(effect_size) and effect_size >= min_eta_squared:
                    selected_features.append(feature)

            except Exception as e:
                self.logger.warning(
                    f"Effect size computation failed for {feature}: {e}"
                )

        self.logger.info(
            f"Effect size selection: {len(selected_features)}/{len(features)} features "
            f"with {effect_type} >= {min_eta_squared}"
        )

        return selected_features

    def select_by_mutual_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        k: int = 20,
        n_neighbors: int = 3,
    ) -> List[str]:

        X = np.asarray(X)
        y = np.asarray(y)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            from sklearn.feature_selection import mutual_info_classif

            # Handle NaN
            if np.any(np.isnan(X)):
                col_means = np.nanmean(X, axis=0)
                nan_mask = np.isnan(X)
                X = X.copy()
                for j in range(X.shape[1]):
                    X[nan_mask[:, j], j] = col_means[j]

            mi_scores = mutual_info_classif(
                X, y, n_neighbors=n_neighbors, random_state=42
            )

            # Rank features
            mi_ranking = list(zip(feature_names, mi_scores))
            mi_ranking.sort(key=lambda x: x[1], reverse=True)

            selected = [name for name, score in mi_ranking[:k]]

            self.logger.info(f"Mutual information selection: top {k} features")

            return selected

        except ImportError:
            self.logger.warning("sklearn not available for mutual information")
            return self._select_by_correlation_with_target(X, y, feature_names, k)

    def _select_by_correlation_with_target(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int
    ) -> List[str]:

        correlations = []

        for i, name in enumerate(feature_names):
            # Point-biserial correlation for binary
            try:
                r, _ = stats.pointbiserialr(y, X[:, i])
                correlations.append((name, abs(r)))
            except Exception:
                correlations.append((name, 0))

        correlations.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in correlations[:k]]

    def rank_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [subject_col, condition_col]]

        if weights is None:
            weights = {"anova": 0.4, "effect_size": 0.4, "correlation": 0.2}

        results = []

        # Run ANOVA for all
        anova_results = self.hypothesis_tester.run_all_anova(
            df, features, subject_col, condition_col, "fdr_bh"
        )

        if anova_results.empty:
            return pd.DataFrame()

        # Encode target
        y = df[condition_col].astype("category").cat.codes.values

        for feature in features:
            feature_data = {
                "feature": feature,
                "anova_p": np.nan,
                "anova_p_corrected": np.nan,
                "eta_squared": np.nan,
                "partial_eta_squared": np.nan,
                "omega_squared": np.nan,
                "effect_interpretation": "unknown",
                "target_correlation": np.nan,
            }

            # ANOVA results
            anova_row = anova_results[anova_results["feature"] == feature]
            if not anova_row.empty:
                feature_data["anova_p"] = anova_row["p_value"].values[0]
                if "p_value_corrected" in anova_row.columns:
                    feature_data["anova_p_corrected"] = anova_row[
                        "p_value_corrected"
                    ].values[0]
                feature_data["eta_squared"] = anova_row["eta_squared"].values[0]
                feature_data["partial_eta_squared"] = anova_row[
                    "partial_eta_squared"
                ].values[0]
                feature_data["omega_squared"] = anova_row["omega_squared"].values[0]
                feature_data["effect_interpretation"] = anova_row[
                    "effect_interpretation"
                ].values[0]

            # Target correlation
            try:
                x = df[feature].dropna()
                y_aligned = (
                    np.asarray(y)[x.index.values]
                    if hasattr(x, "index")
                    else np.asarray(y)[: len(x)]
                )

                if len(np.unique(y_aligned)) == 2:
                    r, _ = stats.pointbiserialr(y_aligned, x.values)
                else:
                    r, _ = stats.spearmanr(x.values, y_aligned)
                feature_data["target_correlation"] = abs(r)
            except Exception:
                pass

            results.append(feature_data)

        ranking_df = pd.DataFrame(results)

        # Compute ranks for each
        ranking_df["anova_rank"] = ranking_df["anova_p"].rank(method="min")
        ranking_df["effect_rank"] = ranking_df["partial_eta_squared"].rank(
            ascending=False, method="min"
        )
        ranking_df["corr_rank"] = ranking_df["target_correlation"].rank(
            ascending=False, method="min"
        )

        # Normalize ranks to [0,
        n = len(ranking_df)
        ranking_df["anova_score"] = 1 - (ranking_df["anova_rank"] - 1) / max(n - 1, 1)
        ranking_df["effect_score"] = 1 - (ranking_df["effect_rank"] - 1) / max(n - 1, 1)
        ranking_df["corr_score"] = 1 - (ranking_df["corr_rank"] - 1) / max(n - 1, 1)

        # Combined score
        ranking_df["combined_score"] = (
            weights["anova"] * ranking_df["anova_score"].fillna(0)
            + weights["effect_size"] * ranking_df["effect_score"].fillna(0)
            + weights["correlation"] * ranking_df["corr_score"].fillna(0)
        )

        # Final ranking
        ranking_df["rank"] = (
            ranking_df["combined_score"].rank(ascending=False, method="min").astype(int)
        )
        ranking_df = ranking_df.sort_values("rank")

        self.logger.info(f"Feature ranking complete: {len(features)} features ranked")

        return ranking_df

    def select_top_features(
        self,
        df: pd.DataFrame,
        n_features: int = 20,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        method: str = "combined",
    ) -> List[str]:

        if method == "combined":
            ranking = self.rank_features(df, features, subject_col, condition_col)
            return ranking.head(n_features)["feature"].tolist()

        elif method == "anova":
            all_features = (
                features or df.select_dtypes(include=[np.number]).columns.tolist()
            )
            all_features = [
                f for f in all_features if f not in [subject_col, condition_col]
            ]

            anova_results = self.hypothesis_tester.run_all_anova(
                df, all_features, subject_col, condition_col, "none"
            )
            anova_results = anova_results.sort_values("p_value")
            return anova_results.head(n_features)["feature"].tolist()

        elif method == "effect_size":
            all_features = (
                features or df.select_dtypes(include=[np.number]).columns.tolist()
            )
            all_features = [
                f for f in all_features if f not in [subject_col, condition_col]
            ]

            anova_results = self.hypothesis_tester.run_all_anova(
                df, all_features, subject_col, condition_col, "none"
            )
            anova_results = anova_results.sort_values(
                "partial_eta_squared", ascending=False
            )
            return anova_results.head(n_features)["feature"].tolist()

        elif method == "mutual_info":
            all_features = (
                features or df.select_dtypes(include=[np.number]).columns.tolist()
            )
            all_features = [
                f for f in all_features if f not in [subject_col, condition_col]
            ]

            X = df[all_features].values
            y = df[condition_col].astype("category").cat.codes.values

            return self.select_by_mutual_info(X, y, all_features, n_features)

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_selection_overlap(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        n_features: int = 20,
    ) -> Dict[str, List[str]]:

        anova_features = self.select_top_features(
            df, n_features, features, subject_col, condition_col, "anova"
        )

        effect_features = self.select_top_features(
            df, n_features, features, subject_col, condition_col, "effect_size"
        )

        mi_features = self.select_top_features(
            df, n_features, features, subject_col, condition_col, "mutual_info"
        )

        # Set operations
        anova_set = set(anova_features)
        effect_set = set(effect_features)
        mi_set = set(mi_features)

        all_methods = list(anova_set & effect_set & mi_set)
        any_method = list(anova_set | effect_set | mi_set)

        # Count overlaps
        overlap_counts = {}
        for feature in any_method:
            count = sum(
                [feature in anova_set, feature in effect_set, feature in mi_set]
            )
            overlap_counts[feature] = count

        self.logger.info(
            f"Feature selection overlap: {len(all_methods)} features selected by all methods, "
            f"{len(any_method)} by at least one"
        )

        return {
            "anova": anova_features,
            "effect_size": effect_features,
            "mutual_info": mi_features,
            "all_methods": all_methods,
            "any_method": any_method,
            "overlap_counts": overlap_counts,
        }

    def compute_feature_importance_ci(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [subject_col, condition_col]]

        # Bootstrap sampling
        bootstrap_scores = {f: [] for f in features}

        for i in range(n_bootstrap):
            # Sample subjects with replacement
            subjects = df[subject_col].unique()
            sampled_subjects = np.random.choice(
                subjects, size=len(subjects), replace=True
            )
            sampled_df = df[df[subject_col].isin(sampled_subjects)]

            # Compute effect sizes
            for feature in features:
                try:
                    result = self.hypothesis_tester.repeated_measures_anova(
                        sampled_df, feature, subject_col, condition_col
                    )
                    eta2 = result.get("partial_eta_squared", np.nan)
                    if not np.isnan(eta2):
                        bootstrap_scores[feature].append(eta2)
                except Exception:
                    pass

        # Compute statistics
        results = []
        alpha = 1 - confidence

        for feature in features:
            scores = bootstrap_scores[feature]

            if len(scores) > 10:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                ci_lower = np.percentile(scores, 100 * alpha / 2)
                ci_upper = np.percentile(scores, 100 * (1 - alpha / 2))
            else:
                mean_score = std_score = ci_lower = ci_upper = np.nan

            results.append(
                {
                    "feature": feature,
                    "importance_mean": mean_score,
                    "importance_std": std_score,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_valid_bootstrap": len(scores),
                }
            )

        importance_df = pd.DataFrame(results)
        importance_df = importance_df.sort_values("importance_mean", ascending=False)

        return importance_df

    def generate_selection_report(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        subject_col: str = "subject_id",
        condition_col: str = "label",
        n_top: int = 20,
    ) -> str:

        ranking = self.rank_features(df, features, subject_col, condition_col)
        overlap = self.get_selection_overlap(
            df, features, subject_col, condition_col, n_top
        )

        lines = [
            "=" * 70,
            "FEATURE SELECTION REPORT",
            "=" * 70,
            "",
            f"Total features analyzed: {len(ranking)}",
            f"Features significant at α={self.alpha}: {(ranking['anova_p'] < self.alpha).sum()}",
            f"Features with medium+ effect (η² >= 0.06): {(ranking['partial_eta_squared'] >= 0.06).sum()}",
            f"Features with large effect (η² >= 0.14): {(ranking['partial_eta_squared'] >= 0.14).sum()}",
            "",
            f"TOP {n_top} FEATURES (Combined Ranking)",
            "-" * 50,
        ]

        top_features = ranking.head(n_top)
        for _, row in top_features.iterrows():
            sig = (
                "***"
                if row["anova_p"] < 0.001
                else "**"
                if row["anova_p"] < 0.01
                else "*"
                if row["anova_p"] < 0.05
                else ""
            )
            lines.append(
                f"  {int(row['rank']):2d}. {row['feature']:<30} "
                f"η²={row['partial_eta_squared']:.3f} p={row['anova_p']:.4f}{sig}"
            )

        lines.extend(
            [
                "",
                "METHOD OVERLAP",
                "-" * 50,
                f"  Selected by all methods: {len(overlap['all_methods'])}",
                f"  Selected by any method: {len(overlap['any_method'])}",
                "",
                "Features selected by all methods:",
            ]
        )

        for feature in sorted(overlap["all_methods"])[:10]:
            lines.append(f"  - {feature}")

        lines.extend(
            [
                "",
                "Significance: *** p<0.001, ** p<0.01, * p<0.05",
                "Effect size: small (0.01), medium (0.06), large (0.14)",
                "=" * 70,
            ]
        )

        return "\n".join(lines)
