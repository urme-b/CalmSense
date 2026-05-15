import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_repeated_measures_data():

    np.random.seed(42)

    n_subjects = 15
    conditions = ["baseline", "stress", "recovery"]

    data = []
    for subject_id in range(1, n_subjects + 1):
        for condition in conditions:
            # Simulate condition effects
            if condition == "baseline":
                feature_mean = 100
                feature_std = 8
            elif condition == "stress":
                feature_mean = 70  # Lower during stress (increased
                feature_std = 10
            else:  # recovery
                feature_mean = 95
                feature_std = 10

            # Add subject-level random effect
            subject_effect = np.random.randn() * 5

            data.append(
                {
                    "subject_id": f"S{subject_id:02d}",
                    "condition": condition,
                    "HRV_SDNN": feature_mean
                    + subject_effect
                    + np.random.randn() * feature_std,
                    "HRV_RMSSD": feature_mean * 0.8
                    + subject_effect * 0.5
                    + np.random.randn() * feature_std * 0.8,
                    "EDA_SCL": 5
                    + (1 if condition == "stress" else 0)
                    + np.random.randn() * 0.5,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_matrix():

    np.random.seed(42)
    n_samples = 100
    n_features = 20

    # Create correlated features
    base = np.random.randn(n_samples, 5)
    noise = np.random.randn(n_samples, n_features) * 0.3

    # Features are combinations of
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:, i] = base[:, i % 5] + noise[:, i]

    return X


@pytest.fixture
def sample_correlated_features():

    np.random.seed(42)
    n_samples = 100

    # Create highly correlated pairs
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.1  # Highly correlated with x1
    x3 = np.random.randn(n_samples)  # Independent
    x4 = x3 * 0.5 + np.random.randn(n_samples) * 0.5  # Moderately correlated with x3
    x5 = np.random.randn(n_samples)  # Independent

    df = pd.DataFrame(
        {
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
            "feature_5": x5,
            "label": np.random.randint(0, 2, n_samples),
        }
    )

    return df


@pytest.fixture
def sample_labels():

    np.random.seed(42)
    return np.repeat([0, 1, 2], 50)  # 50 samples each class


# =============================================================================
# Test Descriptive Statistics
# =============================================================================


class TestDescriptiveStatistics:
    def test_compute_summary(self, sample_repeated_measures_data):

        from src.analysis import DescriptiveStatistics

        desc = DescriptiveStatistics()
        summary = desc.compute_summary(
            sample_repeated_measures_data, group_col="condition"
        )

        assert not summary.empty
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "skewness" in summary.columns
        assert "kurtosis" in summary.columns
        assert len(summary["group"].unique()) == 3  # baseline, stress, recovery

    def test_compute_normality_tests(self, sample_repeated_measures_data):

        from src.analysis import DescriptiveStatistics

        desc = DescriptiveStatistics()
        normality = desc.compute_normality_tests(
            sample_repeated_measures_data, features=["HRV_SDNN", "HRV_RMSSD"]
        )

        assert not normality.empty
        assert "shapiro_W" in normality.columns
        assert "shapiro_p" in normality.columns
        assert "is_normal" in normality.columns
        assert len(normality) == 2

    def test_detect_outliers_iqr(self, sample_repeated_measures_data):

        from src.analysis import DescriptiveStatistics

        desc = DescriptiveStatistics()
        outliers = desc.detect_outliers(
            sample_repeated_measures_data, method="iqr", features=["HRV_SDNN"]
        )

        assert not outliers.empty
        assert "n_outliers" in outliers.columns
        assert "pct_outliers" in outliers.columns
        assert outliers["pct_outliers"].values[0] >= 0
        assert outliers["pct_outliers"].values[0] <= 100

    def test_detect_outliers_zscore(self, sample_repeated_measures_data):

        from src.analysis import DescriptiveStatistics

        desc = DescriptiveStatistics()
        outliers = desc.detect_outliers(
            sample_repeated_measures_data, method="zscore", features=["HRV_SDNN"]
        )

        assert not outliers.empty
        assert outliers["method"].values[0] == "zscore"


# =============================================================================
# Test Hypothesis Testing
# =============================================================================


class TestHypothesisTesting:
    def test_repeated_measures_anova(self, sample_repeated_measures_data):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()
        result = ht.repeated_measures_anova(
            sample_repeated_measures_data,
            feature="HRV_SDNN",
            subject_col="subject_id",
            condition_col="condition",
        )

        assert "F_statistic" in result
        assert "p_value" in result
        assert "eta_squared" in result
        assert "partial_eta_squared" in result
        assert "sphericity_W" in result
        assert "epsilon_GG" in result

        # F should be positive
        assert result["F_statistic"] > 0

        # Effect size should be
        assert 0 <= result["partial_eta_squared"] <= 1

    def test_anova_detects_effect(self, sample_repeated_measures_data):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()
        result = ht.repeated_measures_anova(
            sample_repeated_measures_data,
            feature="HRV_SDNN",
            subject_col="subject_id",
            condition_col="condition",
        )

        # With our simulated data,
        # (baseline mean 100, stress
        assert result["p_value"] < 0.05
        assert result["is_significant"]

    def test_compute_effect_size(self):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()

        # Create groups with known
        np.random.seed(42)
        group1 = np.random.normal(100, 10, 50)
        group2 = np.random.normal(108, 10, 50)  # 0.8 SD difference (large

        result = ht.compute_effect_size(group1, group2)

        assert "cohens_d" in result
        assert "hedges_g" in result
        assert "glass_delta" in result
        assert "eta_squared" in result

        # Cohen's d should be
        assert 0.5 < abs(result["cohens_d"]) < 1.2
        assert result["interpretation"] in ["medium", "large"]

    def test_posthoc_tests(self, sample_repeated_measures_data):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()
        posthoc = ht.posthoc_tests(
            sample_repeated_measures_data,
            feature="HRV_SDNN",
            condition_col="condition",
            correction="bonferroni",
        )

        assert not posthoc.empty
        assert "group1" in posthoc.columns
        assert "group2" in posthoc.columns
        assert "t_statistic" in posthoc.columns
        assert "p_value_corrected" in posthoc.columns
        assert "effect_size_d" in posthoc.columns

        # Should have 3 pairwise
        assert len(posthoc) == 3

    def test_nonparametric_tests(self, sample_repeated_measures_data):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()
        result = ht.nonparametric_tests(
            sample_repeated_measures_data,
            feature="HRV_SDNN",
            subject_col="subject_id",
            condition_col="condition",
        )

        assert "friedman_stat" in result
        assert "friedman_p" in result
        assert "kendall_w" in result
        assert "wilcoxon_results" in result


# =============================================================================
# Test Dimensionality Reduction
# =============================================================================


class TestDimensionalityReduction:
    def test_fit_pca(self, sample_feature_matrix):

        from src.analysis import DimensionalityReducer

        reducer = DimensionalityReducer()
        result = reducer.fit_pca(sample_feature_matrix, n_components=0.95)

        assert "n_components" in result
        assert "explained_variance_ratio" in result
        assert "loadings" in result
        assert "transformed_data" in result

        # Transformed data should have
        assert result["transformed_data"].shape[1] <= sample_feature_matrix.shape[1]

        # Explained variance should sum
        assert np.sum(result["explained_variance_ratio"]) <= 1.0

        # Total variance explained should
        assert result["total_variance_explained"] >= 0.95

    def test_pca_variance_explained(self, sample_feature_matrix):

        from src.analysis import DimensionalityReducer

        reducer = DimensionalityReducer()
        result = reducer.fit_pca(sample_feature_matrix, n_components=5)

        # First few components should
        assert result["explained_variance_ratio"][0] > 0.1
        assert np.all(result["explained_variance_ratio"] >= 0)

    def test_get_top_loadings(self, sample_feature_matrix):

        from src.analysis import DimensionalityReducer

        feature_names = [f"feature_{i}" for i in range(sample_feature_matrix.shape[1])]

        reducer = DimensionalityReducer()
        reducer.fit_pca(sample_feature_matrix, feature_names=feature_names)

        top_loadings = reducer.get_top_loadings(n_top=5, component=1)

        assert len(top_loadings) == 5
        assert "feature" in top_loadings.columns
        assert "loading" in top_loadings.columns
        assert "abs_loading" in top_loadings.columns

    def test_fit_tsne(self, sample_feature_matrix):

        from src.analysis import DimensionalityReducer

        reducer = DimensionalityReducer()
        embedded = reducer.fit_tsne(
            sample_feature_matrix[:50],  # Use fewer samples for
            perplexity=10,
            n_iter=250,
        )

        assert embedded.shape == (50, 2)


# =============================================================================
# Test Correlation Analysis
# =============================================================================


class TestCorrelationAnalysis:
    def test_compute_correlation_matrix(self, sample_correlated_features):

        from src.analysis import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.compute_correlation_matrix(
            sample_correlated_features,
            features=["feature_1", "feature_2", "feature_3"],
            method="pearson",
        )

        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix.values), 1.0)

        # feature_1 and feature_2 should
        assert corr_matrix.loc["feature_1", "feature_2"] > 0.9

    def test_compute_vif(self, sample_correlated_features):

        from src.analysis import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()
        vif_df = analyzer.compute_vif(
            sample_correlated_features,
            features=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        )

        assert not vif_df.empty
        assert "VIF" in vif_df.columns
        assert "interpretation" in vif_df.columns

        # feature_1 and feature_2 are
        vif_1 = vif_df[vif_df["feature"] == "feature_1"]["VIF"].values[0]
        vif_3 = vif_df[vif_df["feature"] == "feature_3"]["VIF"].values[0]

        # High correlation pair should
        assert vif_1 > vif_3

    def test_find_highly_correlated(self, sample_correlated_features):

        from src.analysis import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()
        high_corr = analyzer.find_highly_correlated(
            sample_correlated_features,
            features=["feature_1", "feature_2", "feature_3"],
            threshold=0.8,
        )

        # Should find feature_1 and
        assert len(high_corr) >= 1
        pair_features = [(p[0], p[1]) for p in high_corr]
        assert ("feature_1", "feature_2") in pair_features or (
            "feature_2",
            "feature_1",
        ) in pair_features

    def test_remove_multicollinear(self, sample_correlated_features):

        from src.analysis import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()
        reduced_df = analyzer.remove_multicollinear(
            sample_correlated_features,
            features=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            vif_threshold=5.0,
        )

        # Should have removed at
        assert reduced_df.shape[1] <= 5


# =============================================================================
# Test Feature Selection
# =============================================================================


class TestFeatureSelection:
    def test_rank_features(self, sample_repeated_measures_data):

        from src.analysis import StatisticalFeatureSelector

        selector = StatisticalFeatureSelector()
        ranking = selector.rank_features(
            sample_repeated_measures_data,
            features=["HRV_SDNN", "HRV_RMSSD", "EDA_SCL"],
            subject_col="subject_id",
            condition_col="condition",
        )

        assert not ranking.empty
        assert "feature" in ranking.columns
        assert "combined_score" in ranking.columns
        assert "rank" in ranking.columns
        assert len(ranking) == 3

    def test_select_by_anova(self, sample_repeated_measures_data):

        from src.analysis import StatisticalFeatureSelector

        selector = StatisticalFeatureSelector()
        selected = selector.select_by_anova(
            sample_repeated_measures_data,
            features=["HRV_SDNN", "HRV_RMSSD", "EDA_SCL"],
            subject_col="subject_id",
            condition_col="condition",
            alpha=0.05,
        )

        # Should select features with
        assert isinstance(selected, list)

    def test_select_by_effect_size(self, sample_repeated_measures_data):

        from src.analysis import StatisticalFeatureSelector

        selector = StatisticalFeatureSelector()
        selected = selector.select_by_effect_size(
            sample_repeated_measures_data,
            features=["HRV_SDNN", "HRV_RMSSD", "EDA_SCL"],
            subject_col="subject_id",
            condition_col="condition",
            min_eta_squared=0.06,
        )

        assert isinstance(selected, list)


# =============================================================================
# Test Mixed Effects
# =============================================================================


class TestMixedEffects:
    def test_fit_lmm(self, sample_repeated_measures_data):

        from src.analysis import MixedEffectsModeling

        mem = MixedEffectsModeling()
        result = mem.fit_lmm(
            sample_repeated_measures_data,
            formula="HRV_SDNN ~ condition",
            groups="subject_id",
        )

        assert "fixed_effects" in result
        assert "random_effects_variance" in result
        assert "aic" in result
        assert "bic" in result
        assert result["convergence"]

    def test_compute_icc(self, sample_repeated_measures_data):

        from src.analysis import MixedEffectsModeling

        mem = MixedEffectsModeling()
        icc_result = mem.compute_icc(
            sample_repeated_measures_data, feature="HRV_SDNN", groups="subject_id"
        )

        assert "icc" in icc_result
        assert "variance_between" in icc_result
        assert "variance_within" in icc_result

        # ICC should be between
        assert 0 <= icc_result["icc"] <= 1

    def test_compare_models(self, sample_repeated_measures_data):

        from src.analysis import MixedEffectsModeling

        mem = MixedEffectsModeling()

        # Fit two models
        model1 = mem.fit_lmm(
            sample_repeated_measures_data,
            formula="HRV_SDNN ~ 1",  # Intercept only
            groups="subject_id",
        )

        model2 = mem.fit_lmm(
            sample_repeated_measures_data,
            formula="HRV_SDNN ~ condition",  # With condition
            groups="subject_id",
        )

        comparison = mem.compare_models(model1, model2)

        assert "lr_statistic" in comparison
        assert "aic_difference" in comparison
        assert "preferred_model" in comparison


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_empty_dataframe(self):

        from src.analysis import DescriptiveStatistics

        desc = DescriptiveStatistics()
        empty_df = pd.DataFrame()

        summary = desc.compute_summary(empty_df)
        assert summary.empty

    def test_single_group(self, sample_repeated_measures_data):

        from src.analysis import HypothesisTesting

        ht = HypothesisTesting()
        single_group = sample_repeated_measures_data[
            sample_repeated_measures_data["condition"] == "baseline"
        ]

        result = ht.repeated_measures_anova(
            single_group,
            feature="HRV_SDNN",
            subject_col="subject_id",
            condition_col="condition",
        )

        # Should return NaN for
        assert np.isnan(result["F_statistic"]) or result["n_conditions"] == 1

    def test_nan_handling(self):

        from src.analysis import DimensionalityReducer

        reducer = DimensionalityReducer()

        # Data with NaN
        X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])

        result = reducer.fit_pca(X, n_components=2)

        # Should complete without error
        assert result["transformed_data"].shape[0] == 4

    def test_constant_feature(self):

        from src.analysis import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()

        df = pd.DataFrame({"constant": [1, 1, 1, 1, 1], "varying": [1, 2, 3, 4, 5]})

        vif_df = analyzer.compute_vif(df)

        # Constant feature should be
        assert "constant" not in vif_df["feature"].values or np.isnan(
            vif_df[vif_df["feature"] == "constant"]["VIF"].values[0]
        )
