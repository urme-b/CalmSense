from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_config import LoggerMixin


class DescriptiveStatistics(LoggerMixin):
    def __init__(self):

        self.logger.debug("DescriptiveStatistics initialized")

    def compute_summary(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = "label",
        numeric_only: bool = True,
    ) -> pd.DataFrame:

        if numeric_only:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c != group_col]
        else:
            feature_cols = [c for c in df.columns if c != group_col]

        results = []

        if group_col is not None and group_col in df.columns:
            groups = df[group_col].unique()
            for group in groups:
                group_data = df[df[group_col] == group][feature_cols]
                group_stats = self._compute_stats(group_data)
                group_stats["group"] = group
                results.append(group_stats)
        else:
            overall_stats = self._compute_stats(df[feature_cols])
            overall_stats["group"] = "all"
            results.append(overall_stats)

        summary_df = pd.concat(results, ignore_index=True)

        # Reorder columns
        cols = [
            "feature",
            "group",
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "range",
            "p5",
            "p95",
            "skewness",
            "kurtosis",
            "cv",
        ]
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]

        self.logger.info(
            f"Computed summary statistics for {len(feature_cols)} features "
            f"across {len(results)} groups"
        )

        return summary_df

    def _compute_stats(self, data: pd.DataFrame) -> pd.DataFrame:

        stats_list = []

        for col in data.columns:
            values = data[col].dropna()

            if len(values) == 0:
                continue

            stat_dict = {
                "feature": col,
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "range": np.max(values) - np.min(values),
                "p5": np.percentile(values, 5),
                "p95": np.percentile(values, 95),
                "skewness": stats.skew(values) if len(values) > 2 else np.nan,
                "kurtosis": stats.kurtosis(values) if len(values) > 3 else np.nan,
            }

            # Coefficient of variation
            if stat_dict["mean"] != 0:
                stat_dict["cv"] = stat_dict["std"] / abs(stat_dict["mean"])
            else:
                stat_dict["cv"] = np.nan

            stats_list.append(stat_dict)

        return pd.DataFrame(stats_list)

    def compute_normality_tests(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for feature in features:
            values = df[feature].dropna().values

            result = {
                "feature": feature,
                "n": len(values),
                "shapiro_W": np.nan,
                "shapiro_p": np.nan,
                "dagostino_K2": np.nan,
                "dagostino_p": np.nan,
                "is_normal": False,
            }

            if len(values) < 3:
                self.logger.warning(f"Insufficient data for normality test: {feature}")
                results.append(result)
                continue

            # Shapiro-Wilk test (max 5000
            try:
                sample = values[:5000] if len(values) > 5000 else values
                w_stat, w_pval = stats.shapiro(sample)
                result["shapiro_W"] = w_stat
                result["shapiro_p"] = w_pval
            except Exception as e:
                self.logger.warning(f"Shapiro-Wilk failed for {feature}: {e}")

            # D'Agostino-Pearson test (requires n
            if len(values) >= 20:
                try:
                    k2_stat, k2_pval = stats.normaltest(values)
                    result["dagostino_K2"] = k2_stat
                    result["dagostino_p"] = k2_pval
                except Exception as e:
                    self.logger.warning(f"D'Agostino test failed for {feature}: {e}")

            # Determine normality (both tests
            shapiro_normal = (
                result["shapiro_p"] > alpha
                if not np.isnan(result["shapiro_p"])
                else True
            )
            dagostino_normal = (
                result["dagostino_p"] > alpha
                if not np.isnan(result["dagostino_p"])
                else True
            )
            result["is_normal"] = shapiro_normal and dagostino_normal

            results.append(result)

        normality_df = pd.DataFrame(results)

        n_normal = normality_df["is_normal"].sum()
        self.logger.info(
            f"Normality tests complete: {n_normal}/{len(features)} features "
            f"appear normally distributed (α={alpha})"
        )

        return normality_df

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        features: Optional[List[str]] = None,
        threshold: float = 1.5,
        contamination: float = 0.1,
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for feature in features:
            values = df[feature].dropna()
            n_total = len(values)

            if n_total == 0:
                continue

            if method == "iqr":
                outlier_mask = self._iqr_outliers(values.values, threshold)
            elif method == "zscore":
                z_threshold = threshold if threshold > 1.5 else 3.0
                outlier_mask = self._zscore_outliers(values.values, z_threshold)
            elif method == "isolation_forest":
                outlier_mask = self._isolation_forest_outliers(
                    values.values, contamination
                )
            else:
                raise ValueError(
                    f"Unknown method: {method}. Use 'iqr', 'zscore', or 'isolation_forest'"
                )

            n_outliers = np.sum(outlier_mask)
            outlier_indices = values.index[outlier_mask].tolist()

            results.append(
                {
                    "feature": feature,
                    "method": method,
                    "n_total": n_total,
                    "n_outliers": n_outliers,
                    "pct_outliers": 100.0 * n_outliers / n_total,
                    "outlier_indices": outlier_indices,
                }
            )

        outlier_df = pd.DataFrame(results)

        total_outliers = outlier_df["n_outliers"].sum()
        self.logger.info(
            f"Outlier detection ({method}): {total_outliers} outliers found "
            f"across {len(features)} features"
        )

        return outlier_df

    def _iqr_outliers(self, values: np.ndarray, threshold: float = 1.5) -> np.ndarray:

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return (values < lower_bound) | (values > upper_bound)

    def _zscore_outliers(
        self, values: np.ndarray, threshold: float = 3.0
    ) -> np.ndarray:

        z_scores = np.abs(stats.zscore(values))
        return z_scores > threshold

    def _isolation_forest_outliers(
        self, values: np.ndarray, contamination: float = 0.1
    ) -> np.ndarray:

        try:
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=100
            )
            predictions = clf.fit_predict(values.reshape(-1, 1))
            return predictions == -1
        except ImportError:
            self.logger.warning("sklearn not available, falling back to IQR method")
            return self._iqr_outliers(values)

    def compute_effect_of_outliers(
        self, df: pd.DataFrame, features: Optional[List[str]] = None
    ) -> pd.DataFrame:

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for feature in features:
            values = df[feature].dropna().values

            if len(values) < 10:
                continue

            # Original statistics
            orig_mean = np.mean(values)
            orig_std = np.std(values, ddof=1)
            orig_median = np.median(values)

            # Remove IQR outliers
            outlier_mask = self._iqr_outliers(values)
            clean_values = values[~outlier_mask]

            if len(clean_values) < 3:
                continue

            clean_mean = np.mean(clean_values)
            clean_std = np.std(clean_values, ddof=1)
            clean_median = np.median(clean_values)

            results.append(
                {
                    "feature": feature,
                    "n_original": len(values),
                    "n_outliers": np.sum(outlier_mask),
                    "mean_original": orig_mean,
                    "mean_cleaned": clean_mean,
                    "mean_change_pct": 100 * (clean_mean - orig_mean) / abs(orig_mean)
                    if orig_mean != 0
                    else 0,
                    "std_original": orig_std,
                    "std_cleaned": clean_std,
                    "std_change_pct": 100 * (clean_std - orig_std) / orig_std
                    if orig_std > 0
                    else 0,
                    "median_original": orig_median,
                    "median_cleaned": clean_median,
                }
            )

        return pd.DataFrame(results)

    def generate_summary_report(
        self,
        df: pd.DataFrame,
        group_col: str = "label",
        output_format: str = "markdown",
    ) -> str:

        summary = self.compute_summary(df, group_col)
        normality = self.compute_normality_tests(df)

        if output_format == "markdown":
            report = "# Descriptive Statistics Report\n\n"
            report += "## Summary Statistics\n\n"
            report += summary.to_markdown(index=False) + "\n\n"
            report += "## Normality Tests\n\n"
            report += normality.to_markdown(index=False) + "\n"
        else:
            report = "\\section{Descriptive Statistics}\n\n"
            report += summary.to_latex(index=False) + "\n\n"
            report += "\\section{Normality Tests}\n\n"
            report += normality.to_latex(index=False) + "\n"

        return report
