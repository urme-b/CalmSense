from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_config import LoggerMixin


class MixedEffectsModeling(LoggerMixin):
    def __init__(self):

        self._last_model = None
        self.logger.debug("MixedEffectsModeling initialized")

    def fit_lmm(
        self,
        df: pd.DataFrame,
        formula: str,
        groups: str,
        random_effects: str = "intercept",
        reml: bool = True,
    ) -> Dict[str, Any]:

        try:
            from statsmodels.formula.api import mixedlm

            # Fit model
            model = mixedlm(formula, df, groups=df[groups])
            fitted = model.fit(reml=reml)

            # Extract results
            result = self._extract_statsmodels_results(fitted, formula, groups)
            self._last_model = fitted

            self.logger.info(
                f"LMM fit: {result['n_groups']} groups, "
                f"AIC={result['aic']:.2f}, BIC={result['bic']:.2f}"
            )

            return result

        except ImportError:
            self.logger.warning("statsmodels not available, using simplified LMM")
            return self._fit_simple_lmm(df, formula, groups)

    def _extract_statsmodels_results(
        self, fitted, formula: str, groups: str
    ) -> Dict[str, Any]:

        # Fixed effects
        fixed_effects = fitted.fe_params.to_dict()
        fixed_effects_se = fitted.bse_fe.to_dict() if hasattr(fitted, "bse_fe") else {}
        fixed_effects_pvalues = fitted.pvalues.to_dict()

        # Random effects
        random_effects = fitted.random_effects

        # Convert random effects to
        re_df = pd.DataFrame(
            {
                group: {"intercept": re.values[0] if hasattr(re, "values") else re}
                for group, re in random_effects.items()
            }
        ).T

        # Variance components
        random_var = (
            fitted.cov_re.values[0, 0]
            if hasattr(fitted.cov_re, "values")
            else fitted.cov_re
        )
        residual_var = fitted.scale

        # Model fit statistics
        ll = fitted.llf
        aic = fitted.aic
        bic = fitted.bic

        # Confidence intervals
        try:
            conf_int = fitted.conf_int()
            ci_lower = conf_int.iloc[:, 0].to_dict()
            ci_upper = conf_int.iloc[:, 1].to_dict()
        except Exception:
            ci_lower = ci_upper = {}

        return {
            "formula": formula,
            "groups": groups,
            "fixed_effects": fixed_effects,
            "fixed_effects_se": fixed_effects_se,
            "fixed_effects_pvalues": fixed_effects_pvalues,
            "fixed_effects_ci_lower": ci_lower,
            "fixed_effects_ci_upper": ci_upper,
            "random_effects_df": re_df,
            "random_effects_variance": random_var,
            "residual_variance": residual_var,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "n_observations": fitted.nobs,
            "n_groups": len(fitted.random_effects),
            "convergence": True,
        }

    def _fit_simple_lmm(
        self, df: pd.DataFrame, formula: str, groups: str
    ) -> Dict[str, Any]:

        # Parse formula
        parts = formula.split("~")
        y_var = parts[0].strip()
        x_vars_str = parts[1].strip()

        # Handle intercept-only models (formula
        if x_vars_str == "1":
            x_vars = []
        else:
            x_vars = [v.strip() for v in x_vars_str.split("+") if v.strip() != "1"]

        # Prepare data
        cols_to_select = [y_var, groups] + x_vars
        df_clean = df[cols_to_select].dropna()

        y = df_clean[y_var].values
        n = len(y)

        # Design matrix
        X_cols = [np.ones(n)]  # Intercept
        col_names = ["Intercept"]

        for var in x_vars:
            if (
                df_clean[var].dtype == "object"
                or df_clean[var].dtype.name == "category"
            ):
                # Dummy coding
                dummies = pd.get_dummies(df_clean[var], drop_first=True)
                for col in dummies.columns:
                    X_cols.append(dummies[col].values)
                    col_names.append(f"{var}_{col}")
            else:
                X_cols.append(df_clean[var].values)
                col_names.append(var)

        X = np.column_stack(X_cols)

        # OLS fit
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y

            # Predictions and residuals
            y_pred = X @ beta
            residuals = y - y_pred

            # Residual variance
            df_resid = n - X.shape[1]
            residual_var = np.sum(residuals**2) / df_resid

            # Standard errors (naive, not
            se = np.sqrt(np.diag(XtX_inv) * residual_var)

            # t-statistics and p-values
            t_stats = beta / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

            # Create results
            fixed_effects = dict(zip(col_names, beta))
            fixed_effects_se = dict(zip(col_names, se))
            fixed_effects_pvalues = dict(zip(col_names, p_values))

            # Simple random effects estimate
            group_effects = df_clean.groupby(groups)[y_var].mean() - np.mean(y)
            random_var = group_effects.var()

            # Log-likelihood (approximate)
            ll = -n / 2 * (np.log(2 * np.pi * residual_var) + 1)

            # AIC and BIC
            k = X.shape[1] + 1  # parameters + variance
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            return {
                "formula": formula,
                "groups": groups,
                "fixed_effects": fixed_effects,
                "fixed_effects_se": fixed_effects_se,
                "fixed_effects_pvalues": fixed_effects_pvalues,
                "fixed_effects_ci_lower": {
                    k: v - 1.96 * se[i]
                    for i, (k, v) in enumerate(fixed_effects.items())
                },
                "fixed_effects_ci_upper": {
                    k: v + 1.96 * se[i]
                    for i, (k, v) in enumerate(fixed_effects.items())
                },
                "random_effects_df": pd.DataFrame({"intercept": group_effects}),
                "random_effects_variance": random_var,
                "residual_variance": residual_var,
                "log_likelihood": ll,
                "aic": aic,
                "bic": bic,
                "n_observations": n,
                "n_groups": df_clean[groups].nunique(),
                "convergence": True,
                "note": "Simplified OLS-based estimation (statsmodels not available)",
            }

        except Exception as e:
            self.logger.error(f"Simple LMM failed: {e}")
            return {
                "formula": formula,
                "groups": groups,
                "fixed_effects": {},
                "convergence": False,
                "error": str(e),
            }

    def compare_models(
        self, model1: Dict[str, Any], model2: Dict[str, Any]
    ) -> Dict[str, Any]:

        ll1 = model1.get("log_likelihood", np.nan)
        ll2 = model2.get("log_likelihood", np.nan)

        aic1 = model1.get("aic", np.nan)
        aic2 = model2.get("aic", np.nan)

        bic1 = model1.get("bic", np.nan)
        bic2 = model2.get("bic", np.nan)

        # Likelihood ratio test
        # LR = -2 *
        lr_stat = -2 * (ll1 - ll2)

        # Degrees of freedom =
        # Estimate from AIC: AIC
        k1 = (aic1 + 2 * ll1) / 2 if not np.isnan(aic1) and not np.isnan(ll1) else 0
        k2 = (aic2 + 2 * ll2) / 2 if not np.isnan(aic2) and not np.isnan(ll2) else 0
        df = abs(int(k2 - k1))

        if df > 0 and not np.isnan(lr_stat):
            lr_pvalue = 1 - stats.chi2.cdf(max(0, lr_stat), df)
        else:
            lr_pvalue = np.nan

        # Model preference
        aic_diff = aic1 - aic2  # Positive means model2 is
        bic_diff = bic1 - bic2

        if not np.isnan(lr_pvalue):
            if lr_pvalue < 0.05 and aic_diff > 0:
                preferred = "model2"
            elif lr_pvalue >= 0.05:
                preferred = "model1 (simpler)"
            else:
                preferred = "model2" if aic_diff > 2 else "inconclusive"
        else:
            preferred = (
                "model2"
                if aic_diff > 2
                else "model1"
                if aic_diff < -2
                else "inconclusive"
            )

        return {
            "lr_statistic": lr_stat,
            "lr_df": df,
            "lr_pvalue": lr_pvalue,
            "model1_ll": ll1,
            "model2_ll": ll2,
            "model1_aic": aic1,
            "model2_aic": aic2,
            "aic_difference": aic_diff,
            "model1_bic": bic1,
            "model2_bic": bic2,
            "bic_difference": bic_diff,
            "preferred_model": preferred,
            "lr_significant": lr_pvalue < 0.05 if not np.isnan(lr_pvalue) else None,
        }

    def extract_random_effects(self, model_result: Dict[str, Any]) -> pd.DataFrame:

        if "random_effects_df" in model_result:
            return model_result["random_effects_df"]

        self.logger.warning("No random effects found in model result")
        return pd.DataFrame()

    def compute_icc(
        self, df: pd.DataFrame, feature: str, groups: str
    ) -> Dict[str, float]:

        df_clean = df[[feature, groups]].dropna()

        if len(df_clean) < 3:
            return {
                "icc": np.nan,
                "variance_between": np.nan,
                "variance_within": np.nan,
            }

        # Compute variance components
        grand_mean = df_clean[feature].mean()
        group_means = df_clean.groupby(groups)[feature].mean()
        group_counts = df_clean.groupby(groups).size()

        # Between-group variance
        ss_between = np.sum(group_counts * (group_means - grand_mean) ** 2)
        df_between = len(group_means) - 1

        # Within-group variance
        ss_within = (
            df_clean.groupby(groups)
            .apply(
                lambda x: np.sum((x[feature] - x[feature].mean()) ** 2),
                include_groups=False,
            )
            .sum()
        )
        df_within = len(df_clean) - len(group_means)

        if df_between == 0 or df_within == 0:
            return {
                "icc": np.nan,
                "variance_between": np.nan,
                "variance_within": np.nan,
            }

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Average group size
        n_avg = len(df_clean) / len(group_means)

        # Variance estimates
        variance_within = ms_within
        variance_between = (ms_between - ms_within) / n_avg

        # Ensure non-negative
        variance_between = max(0, variance_between)

        # ICC
        icc = (
            variance_between / (variance_between + variance_within)
            if (variance_between + variance_within) > 0
            else 0
        )

        return {
            "icc": icc,
            "variance_between": variance_between,
            "variance_within": variance_within,
            "n_groups": len(group_means),
            "n_observations": len(df_clean),
            "interpretation": self._interpret_icc(icc),
        }

    def _interpret_icc(self, icc: float) -> str:

        if np.isnan(icc):
            return "unknown"
        elif icc < 0.5:
            return "poor"
        elif icc < 0.75:
            return "moderate"
        elif icc < 0.9:
            return "good"
        else:
            return "excellent"

    def fit_random_slopes(
        self, df: pd.DataFrame, formula: str, groups: str, slope_var: str
    ) -> Dict[str, Any]:

        try:
            from statsmodels.formula.api import mixedlm

            # Fit model with random
            model = mixedlm(formula, df, groups=df[groups], re_formula=f"~{slope_var}")
            fitted = model.fit()

            result = self._extract_statsmodels_results(fitted, formula, groups)
            result["random_slopes_var"] = slope_var

            self.logger.info(f"Random slopes model fit for {slope_var}")

            return result

        except ImportError:
            self.logger.warning("statsmodels not available for random slopes")
            return self._fit_simple_lmm(df, formula, groups)

        except Exception as e:
            self.logger.error(f"Random slopes model failed: {e}")
            return {"convergence": False, "error": str(e)}

    def generate_model_summary(self, model_result: Dict[str, Any]) -> str:

        lines = [
            "=" * 60,
            "LINEAR MIXED MODEL RESULTS",
            "=" * 60,
            f"Formula: {model_result.get('formula', 'N/A')}",
            f"Groups: {model_result.get('groups', 'N/A')}",
            f"N observations: {model_result.get('n_observations', 'N/A')}",
            f"N groups: {model_result.get('n_groups', 'N/A')}",
            "",
            "Fixed Effects:",
            "-" * 40,
        ]

        fixed_effects = model_result.get("fixed_effects", {})
        se = model_result.get("fixed_effects_se", {})
        pvals = model_result.get("fixed_effects_pvalues", {})

        for name, coef in fixed_effects.items():
            se_val = se.get(name, np.nan)
            p_val = pvals.get(name, np.nan)
            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )
            lines.append(f"  {name}: {coef:.4f} (SE={se_val:.4f}, p={p_val:.4f}){sig}")

        lines.extend(
            [
                "",
                "Random Effects:",
                "-" * 40,
                f"  Variance (Intercept): {model_result.get('random_effects_variance', np.nan):.4f}",
                f"  Residual Variance: {model_result.get('residual_variance', np.nan):.4f}",
                "",
                "Model Fit:",
                "-" * 40,
                f"  Log-Likelihood: {model_result.get('log_likelihood', np.nan):.2f}",
                f"  AIC: {model_result.get('aic', np.nan):.2f}",
                f"  BIC: {model_result.get('bic', np.nan):.2f}",
                "",
                "Significance: *** p<0.001, ** p<0.01, * p<0.05",
                "=" * 60,
            ]
        )

        return "\n".join(lines)
