from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_config import LoggerMixin


class HypothesisTesting(LoggerMixin):
    # Effect size thresholds (Cohen,
    SMALL_D = 0.2
    MEDIUM_D = 0.5
    LARGE_D = 0.8

    SMALL_ETA = 0.01
    MEDIUM_ETA = 0.06
    LARGE_ETA = 0.14

    def __init__(self, alpha: float = 0.05):

        self.alpha = alpha
        self.logger.debug(f"HypothesisTesting initialized with α={alpha}")

    def repeated_measures_anova(
        self, df: pd.DataFrame, feature: str, subject_col: str, condition_col: str
    ) -> Dict[str, Any]:

        # Prepare data
        pivot_df = df.pivot_table(
            values=feature, index=subject_col, columns=condition_col, aggfunc="mean"
        ).dropna()

        n_subjects = len(pivot_df)
        n_conditions = len(pivot_df.columns)
        conditions = pivot_df.columns.tolist()

        if n_subjects < 3:
            self.logger.warning(f"Insufficient subjects ({n_subjects}) for ANOVA")
            return self._empty_anova_result()

        if n_conditions < 2:
            self.logger.warning(f"Insufficient conditions ({n_conditions}) for ANOVA")
            return self._empty_anova_result()

        # Data matrix
        data_matrix = pivot_df.values

        # Grand mean and condition
        grand_mean = np.mean(data_matrix)
        condition_means = np.mean(data_matrix, axis=0)
        subject_means = np.mean(data_matrix, axis=1)

        # Sum of squares
        SS_total = np.sum((data_matrix - grand_mean) ** 2)
        SS_between = n_subjects * np.sum((condition_means - grand_mean) ** 2)
        SS_subjects = n_conditions * np.sum((subject_means - grand_mean) ** 2)
        SS_error = SS_total - SS_between - SS_subjects

        # Degrees of freedom
        df_between = n_conditions - 1
        df_subjects = n_subjects - 1
        df_error = df_between * df_subjects

        # Mean squares
        MS_between = SS_between / df_between
        MS_error = SS_error / df_error if df_error > 0 else np.nan

        # F-statistic
        F_stat = MS_between / MS_error if MS_error > 0 else np.nan
        p_value = (
            1 - stats.f.cdf(F_stat, df_between, df_error)
            if not np.isnan(F_stat)
            else np.nan
        )

        # Effect sizes
        eta_squared = SS_between / SS_total if SS_total > 0 else 0
        partial_eta_squared = (
            SS_between / (SS_between + SS_error) if (SS_between + SS_error) > 0 else 0
        )

        # Omega squared (less biased)
        omega_squared = (
            (SS_between - df_between * MS_error) / (SS_total + MS_error)
            if (SS_total + MS_error) > 0
            else 0
        )
        omega_squared = max(0, omega_squared)  # Can't be negative

        # Sphericity test (Mauchly)
        sphericity_W, sphericity_p, epsilon_GG, epsilon_HF = self._mauchly_sphericity(
            data_matrix
        )

        # Corrected p-values
        if not np.isnan(F_stat):
            p_value_GG = 1 - stats.f.cdf(
                F_stat, df_between * epsilon_GG, df_error * epsilon_GG
            )
            p_value_HF = 1 - stats.f.cdf(
                F_stat, df_between * epsilon_HF, df_error * epsilon_HF
            )
        else:
            p_value_GG = p_value_HF = np.nan

        # Effect size interpretation
        effect_interpretation = self._interpret_eta_squared(partial_eta_squared)

        result = {
            "feature": feature,
            "n_subjects": n_subjects,
            "n_conditions": n_conditions,
            "conditions": conditions,
            "F_statistic": F_stat,
            "p_value": p_value,
            "df_between": df_between,
            "df_error": df_error,
            "SS_between": SS_between,
            "SS_error": SS_error,
            "MS_between": MS_between,
            "MS_error": MS_error,
            "eta_squared": eta_squared,
            "partial_eta_squared": partial_eta_squared,
            "omega_squared": omega_squared,
            "sphericity_W": sphericity_W,
            "sphericity_p": sphericity_p,
            "epsilon_GG": epsilon_GG,
            "epsilon_HF": epsilon_HF,
            "p_value_GG": p_value_GG,
            "p_value_HF": p_value_HF,
            "is_significant": p_value < self.alpha if not np.isnan(p_value) else False,
            "sphericity_violated": sphericity_p < self.alpha
            if not np.isnan(sphericity_p)
            else False,
            "effect_interpretation": effect_interpretation,
        }

        self.logger.debug(
            f"ANOVA {feature}: F({df_between},{df_error})={F_stat:.2f}, "
            f"p={p_value:.4f}, η²={partial_eta_squared:.3f}"
        )

        return result

    def _mauchly_sphericity(
        self, data_matrix: np.ndarray
    ) -> Tuple[float, float, float, float]:

        n_subjects, n_conditions = data_matrix.shape

        if n_conditions < 3:
            # Sphericity is always satisfied
            return 1.0, 1.0, 1.0, 1.0

        # Compute difference scores
        diff_scores = np.zeros((n_subjects, n_conditions - 1))
        for j in range(n_conditions - 1):
            diff_scores[:, j] = data_matrix[:, j] - data_matrix[:, j + 1]

        # Covariance matrix of differences
        cov_matrix = np.cov(diff_scores.T)

        if cov_matrix.ndim == 0:  # Single difference
            return 1.0, 1.0, 1.0, 1.0

        # Mauchly's W
        try:
            det_cov = np.linalg.det(cov_matrix)
            trace_cov = np.trace(cov_matrix)
            k = n_conditions - 1

            if trace_cov > 0 and det_cov >= 0:
                W = det_cov / ((trace_cov / k) ** k)
            else:
                W = 1.0

            # Chi-squared approximation for p-value
            f = (2 * k**2 + k + 2) / (6 * k * (n_subjects - 1))
            df_chi = k * (k + 1) / 2 - 1
            chi_sq = -(n_subjects - 1 - f) * np.log(max(W, 1e-10))
            p_value = 1 - stats.chi2.cdf(chi_sq, df_chi) if df_chi > 0 else 1.0

            # Greenhouse-Geisser epsilon
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]
            sum_eigen = np.sum(eigenvalues)
            sum_eigen_sq = np.sum(eigenvalues**2)

            epsilon_GG = (
                (sum_eigen**2) / (k * sum_eigen_sq) if sum_eigen_sq > 0 else 1.0
            )
            epsilon_GG = min(1.0, max(1.0 / k, epsilon_GG))

            # Huynh-Feldt epsilon
            hf_denom = k * (n_subjects - 1 - k * epsilon_GG)
            epsilon_HF = (
                (n_subjects * k * epsilon_GG - 2) / hf_denom
                if abs(hf_denom) > 1e-10
                else 1.0
            )
            epsilon_HF = min(1.0, max(epsilon_GG, epsilon_HF))

        except Exception as e:
            self.logger.warning(f"Sphericity calculation failed: {e}")
            W, p_value, epsilon_GG, epsilon_HF = 1.0, 1.0, 1.0, 1.0

        return W, p_value, epsilon_GG, epsilon_HF

    def _empty_anova_result(self) -> Dict[str, Any]:

        return {
            "F_statistic": np.nan,
            "p_value": np.nan,
            "df_between": np.nan,
            "df_error": np.nan,
            "eta_squared": np.nan,
            "partial_eta_squared": np.nan,
            "omega_squared": np.nan,
            "sphericity_W": np.nan,
            "sphericity_p": np.nan,
            "epsilon_GG": np.nan,
            "p_value_GG": np.nan,
            "is_significant": False,
            "effect_interpretation": "insufficient_data",
        }

    def run_all_anova(
        self,
        df: pd.DataFrame,
        features: List[str],
        subject_col: str = "subject_id",
        condition_col: str = "label",
        correction: str = "bonferroni",
    ) -> pd.DataFrame:

        results = []

        for feature in features:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in DataFrame")
                continue

            result = self.repeated_measures_anova(
                df, feature, subject_col, condition_col
            )
            results.append(result)

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Apply multiple comparison correction
        if correction != "none" and "p_value" in results_df.columns:
            p_values = results_df["p_value"].values
            p_values = np.nan_to_num(p_values, nan=1.0)

            corrected_p = self._apply_correction(p_values, correction)
            results_df["p_value_corrected"] = corrected_p
            results_df["is_significant_corrected"] = corrected_p < self.alpha

        # Sort by p-value
        results_df = results_df.sort_values("p_value")

        n_sig = results_df["is_significant"].sum()
        self.logger.info(
            f"ANOVA complete: {n_sig}/{len(results)} features significant "
            f"(α={self.alpha}, correction={correction})"
        )

        return results_df

    def _apply_correction(self, p_values: np.ndarray, method: str) -> np.ndarray:

        n = len(p_values)

        if method == "bonferroni":
            return np.minimum(p_values * n, 1.0)

        elif method == "holm":
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            corrected = np.zeros(n)

            for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
                corrected[idx] = min(p * (n - i), 1.0)

            # Ensure monotonicity
            for i in range(1, n):
                if corrected[sorted_idx[i]] < corrected[sorted_idx[i - 1]]:
                    corrected[sorted_idx[i]] = corrected[sorted_idx[i - 1]]

            return corrected

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            corrected = np.zeros(n)

            for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
                corrected[idx] = min(p * n / (i + 1), 1.0)

            # Ensure monotonicity (in reverse)
            for i in range(n - 2, -1, -1):
                if corrected[sorted_idx[i]] > corrected[sorted_idx[i + 1]]:
                    corrected[sorted_idx[i]] = corrected[sorted_idx[i + 1]]

            return corrected

        return p_values

    def posthoc_tests(
        self,
        df: pd.DataFrame,
        feature: str,
        condition_col: str,
        correction: str = "bonferroni",
    ) -> pd.DataFrame:

        conditions = df[condition_col].unique()

        results = []

        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i + 1 :]:
                group1 = df[df[condition_col] == cond1][feature].dropna().values
                group2 = df[df[condition_col] == cond2][feature].dropna().values

                # Ensure equal length for
                min_len = min(len(group1), len(group2))
                if min_len < 3:
                    continue

                group1 = group1[:min_len]
                group2 = group2[:min_len]

                # Paired t-test
                t_stat, p_val = stats.ttest_rel(group1, group2)

                # Mean difference and CI
                diff = group2 - group1
                mean_diff = np.mean(diff)
                se_diff = stats.sem(diff)
                ci_margin = stats.t.ppf(0.975, len(diff) - 1) * se_diff
                ci_lower = mean_diff - ci_margin
                ci_upper = mean_diff + ci_margin

                # Effect size (Cohen's d
                d = self._paired_cohens_d(group1, group2)

                results.append(
                    {
                        "group1": cond1,
                        "group2": cond2,
                        "n": min_len,
                        "mean_group1": np.mean(group1),
                        "mean_group2": np.mean(group2),
                        "mean_diff": mean_diff,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "effect_size_d": d,
                        "effect_interpretation": self._interpret_cohens_d(d),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Apply correction
        if correction != "none":
            p_values = results_df["p_value"].values
            corrected_p = self._apply_correction(p_values, correction)
            results_df["p_value_corrected"] = corrected_p
            results_df["is_significant"] = corrected_p < self.alpha

        return results_df

    def _paired_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:

        diff = group2 - group1
        return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    def nonparametric_tests(
        self, df: pd.DataFrame, feature: str, subject_col: str, condition_col: str
    ) -> Dict[str, Any]:

        # Pivot data
        pivot_df = df.pivot_table(
            values=feature, index=subject_col, columns=condition_col, aggfunc="mean"
        ).dropna()

        n_subjects, n_conditions = pivot_df.shape
        conditions = pivot_df.columns.tolist()

        result = {
            "feature": feature,
            "n_subjects": n_subjects,
            "n_conditions": n_conditions,
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
            "kendall_w": np.nan,
            "wilcoxon_results": [],
        }

        if n_subjects < 3 or n_conditions < 2:
            return result

        # Friedman test
        try:
            data_arrays = [pivot_df[col].values for col in conditions]
            friedman_stat, friedman_p = stats.friedmanchisquare(*data_arrays)

            # Kendall's W (coefficient of
            kendall_w = friedman_stat / (n_subjects * (n_conditions - 1))

            result["friedman_stat"] = friedman_stat
            result["friedman_p"] = friedman_p
            result["kendall_w"] = kendall_w
            result["is_significant"] = friedman_p < self.alpha

        except Exception as e:
            self.logger.warning(f"Friedman test failed: {e}")

        # Pairwise Wilcoxon signed-rank tests
        wilcoxon_results = []
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i + 1 :]:
                try:
                    group1 = pivot_df[cond1].values
                    group2 = pivot_df[cond2].values

                    stat, p_val = stats.wilcoxon(group1, group2)

                    # Effect size (r =
                    # Approximate Z from p-value
                    z_val = stats.norm.ppf(1 - p_val / 2)
                    r_effect = z_val / np.sqrt(n_subjects)

                    wilcoxon_results.append(
                        {
                            "group1": cond1,
                            "group2": cond2,
                            "statistic": stat,
                            "p_value": p_val,
                            "effect_size_r": r_effect,
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Wilcoxon test failed for {cond1} vs {cond2}: {e}"
                    )

        result["wilcoxon_results"] = wilcoxon_results

        return result

    def compute_effect_size(
        self, group1: np.ndarray, group2: np.ndarray, paired: bool = False
    ) -> Dict[str, float]:

        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        std1 = np.std(group1, ddof=1)

        # Cohen's d
        if paired:
            diff = group2[: min(n1, n2)] - group1[: min(n1, n2)]
            cohens_d = (
                np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            )
        else:
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

        # Hedges' g (bias correction
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor

        # Glass's delta (uses control
        glass_delta = (mean2 - mean1) / std1 if std1 > 0 else 0

        # Eta squared (from t-test)
        if paired:
            t_stat, _ = stats.ttest_rel(group1[: min(n1, n2)], group2[: min(n1, n2)])
            df = min(n1, n2) - 1
        else:
            t_stat, _ = stats.ttest_ind(group1, group2)
            df = n1 + n2 - 2

        eta_squared = t_stat**2 / (t_stat**2 + df) if not np.isnan(t_stat) else 0

        # Correlation-based effect size (point-biserial
        r_effect = np.sqrt(eta_squared)

        return {
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
            "glass_delta": glass_delta,
            "eta_squared": eta_squared,
            "r_effect": r_effect,
            "interpretation": self._interpret_cohens_d(cohens_d),
        }

    def _interpret_cohens_d(self, d: float) -> str:

        d = abs(d)
        if d < self.SMALL_D:
            return "negligible"
        elif d < self.MEDIUM_D:
            return "small"
        elif d < self.LARGE_D:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta2: float) -> str:

        if eta2 < self.SMALL_ETA:
            return "negligible"
        elif eta2 < self.MEDIUM_ETA:
            return "small"
        elif eta2 < self.LARGE_ETA:
            return "medium"
        else:
            return "large"

    def power_analysis(
        self, effect_size: float, n_groups: int, n_subjects: int, alpha: float = 0.05
    ) -> Dict[str, float]:

        # Cohen's f from eta-squared
        # f = sqrt(eta_squared /
        # For repeated measures, effective

        df1 = n_groups - 1
        df2 = (n_subjects - 1) * (n_groups - 1)

        # Non-centrality parameter
        lambda_nc = effect_size**2 * n_subjects * n_groups

        # Critical F value
        f_crit = stats.f.ppf(1 - alpha, df1, df2)

        # Power = P(F >
        power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_nc)

        return {
            "effect_size_f": effect_size,
            "n_groups": n_groups,
            "n_subjects": n_subjects,
            "alpha": alpha,
            "df1": df1,
            "df2": df2,
            "lambda": lambda_nc,
            "f_critical": f_crit,
            "power": power,
        }
