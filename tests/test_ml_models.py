import numpy as np
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():

    np.random.seed(42)
    n_samples = 200
    n_features = 20
    n_classes = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


@pytest.fixture
def sample_data_with_groups():

    np.random.seed(42)
    n_subjects = 10
    samples_per_subject = 30
    n_features = 15
    n_classes = 3

    X_list, y_list, groups_list = [], [], []

    for subject_id in range(n_subjects):
        X_subject = np.random.randn(samples_per_subject, n_features)
        y_subject = np.random.randint(0, n_classes, samples_per_subject)
        groups_subject = np.full(samples_per_subject, subject_id)

        X_list.append(X_subject)
        y_list.append(y_subject)
        groups_list.append(groups_subject)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    return X, y, groups


@pytest.fixture
def imbalanced_data():

    np.random.seed(42)
    n_features = 10

    # Class 0: 150 samples
    X_0 = np.random.randn(150, n_features)
    y_0 = np.zeros(150)

    # Class 1: 30 samples
    X_1 = np.random.randn(30, n_features) + 1
    y_1 = np.ones(30)

    # Class 2: 20 samples
    X_2 = np.random.randn(20, n_features) + 2
    y_2 = np.full(20, 2)

    X = np.vstack([X_0, X_1, X_2])
    y = np.concatenate([y_0, y_1, y_2])

    return X, y.astype(int)


# ============================================================================
# BaseMLModel Tests
# ============================================================================


class TestBaseMLModel:
    def test_base_model_abstract_method(self):

        from src.models.ml.base_model import BaseMLModel

        with pytest.raises(TypeError):
            BaseMLModel("test_model")

    def test_model_not_fitted_error(self, sample_data):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        model = LogisticRegressionClassifier()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_model_fit_returns_self(self, sample_data):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        model = LogisticRegressionClassifier()

        result = model.fit(X, y)
        assert result is model

    def test_get_params(self):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        model = LogisticRegressionClassifier(C=0.5, penalty="l1")
        params = model.get_params()

        assert "random_state" in params
        assert params["C"] == 0.5


# ============================================================================
# Classifier Tests
# ============================================================================


class TestLogisticRegressionClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        clf = LogisticRegressionClassifier()
        assert clf.C == 1.0
        assert clf.penalty == "l2"
        assert clf.solver == "lbfgs"
        assert clf.max_iter == 1000
        assert clf.class_weight == "balanced"

    def test_fit_predict(self, sample_data):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == len(y)
        assert clf.is_fitted

    def test_predict_proba(self, sample_data):

        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        probas = clf.predict_proba(X)
        assert probas.shape == (len(y), len(np.unique(y)))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestSVMClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import SVMClassifier

        clf = SVMClassifier()
        assert clf.C == 10.0
        assert clf.kernel == "rbf"
        assert clf.gamma == "scale"
        assert clf.probability is True

    def test_fit_predict(self, sample_data):

        from src.models.ml.classifiers import SVMClassifier

        X, y = sample_data
        clf = SVMClassifier()
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == len(y)


class TestRandomForestClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import RandomForestClassifier

        clf = RandomForestClassifier()
        assert clf.n_estimators == 200
        assert clf.max_depth == 10
        assert clf.min_samples_leaf == 5

    def test_feature_importance(self, sample_data):

        from src.models.ml.classifiers import RandomForestClassifier

        X, y = sample_data
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X, y)

        importance_df = clf.get_feature_importance()
        assert len(importance_df) == X.shape[1]
        assert "importance" in importance_df.columns


class TestXGBoostClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import XGBoostClassifier

        clf = XGBoostClassifier()
        assert clf.n_estimators == 200
        assert clf.max_depth == 7
        assert clf.learning_rate == 0.1
        assert clf.subsample == 0.8


class TestLightGBMClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import LightGBMClassifier

        clf = LightGBMClassifier()
        assert clf.num_leaves == 50
        assert clf.n_estimators == 200
        assert clf.learning_rate == 0.1


class TestCatBoostClassifier:
    def test_initialization(self):

        from src.models.ml.classifiers import CatBoostClassifier

        clf = CatBoostClassifier()
        assert clf.iterations == 500
        assert clf.depth == 6
        assert clf.learning_rate == 0.1
        assert clf.verbose is False


class TestGetClassifier:
    def test_get_logistic_regression(self):

        from src.models.ml.classifiers import (
            get_classifier,
            LogisticRegressionClassifier,
        )

        clf = get_classifier("lr")
        assert isinstance(clf, LogisticRegressionClassifier)

        clf = get_classifier("logistic")
        assert isinstance(clf, LogisticRegressionClassifier)

    def test_get_svm(self):

        from src.models.ml.classifiers import get_classifier, SVMClassifier

        clf = get_classifier("svm")
        assert isinstance(clf, SVMClassifier)

    def test_get_random_forest(self):

        from src.models.ml.classifiers import get_classifier, RandomForestClassifier

        clf = get_classifier("rf")
        assert isinstance(clf, RandomForestClassifier)

    def test_unknown_classifier_raises(self):

        from src.models.ml.classifiers import get_classifier

        with pytest.raises(ValueError, match="Unknown classifier"):
            get_classifier("unknown_model")

    def test_custom_params(self):

        from src.models.ml.classifiers import get_classifier

        clf = get_classifier("lr", C=0.1, max_iter=500)
        assert clf.C == 0.1
        assert clf.max_iter == 500


# ============================================================================
# CrossValidator Tests
# ============================================================================


class TestCrossValidator:
    def test_initialization_loso(self):

        from src.models.ml.cross_validation import CrossValidator

        cv = CrossValidator(cv_strategy="loso")
        assert cv.cv_strategy == "loso"
        assert cv.n_splits == 5
        assert cv.random_state == 42

    def test_initialization_kfold(self):

        from src.models.ml.cross_validation import CrossValidator

        cv = CrossValidator(cv_strategy="kfold", n_splits=10)
        assert cv.cv_strategy == "kfold"
        assert cv.n_splits == 10

    def test_loso_splits(self, sample_data_with_groups):

        from src.models.ml.cross_validation import CrossValidator

        X, y, groups = sample_data_with_groups
        cv = CrossValidator(cv_strategy="loso")

        splits = list(cv.get_loso_splits(X, y, groups))

        # Should have one split
        n_subjects = len(np.unique(groups))
        assert len(splits) == n_subjects

        # Test indices are valid
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == len(y)
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_kfold_splits(self, sample_data):

        from src.models.ml.cross_validation import CrossValidator

        X, y = sample_data
        cv = CrossValidator(cv_strategy="kfold", n_splits=5)

        splits = list(cv.get_kfold_splits(X, y))
        assert len(splits) == 5

    def test_cross_validate_loso(self, sample_data_with_groups):

        from src.models.ml.cross_validation import CrossValidator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y, groups = sample_data_with_groups
        cv = CrossValidator(cv_strategy="loso")
        model = LogisticRegressionClassifier()

        results = cv.cross_validate(model, X, y, groups)

        assert "accuracy_mean" in results
        assert "accuracy_std" in results
        assert "f1_macro_mean" in results
        assert "fold_results" in results
        assert results["cv_strategy"] == "loso"

    def test_cross_validate_requires_groups_for_loso(self, sample_data):

        from src.models.ml.cross_validation import CrossValidator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        cv = CrossValidator(cv_strategy="loso")
        model = LogisticRegressionClassifier()

        with pytest.raises(ValueError, match="LOSO requires groups"):
            cv.cross_validate(model, X, y, groups=None)

    def test_stratified_splits(self, sample_data):

        from src.models.ml.cross_validation import CrossValidator

        X, y = sample_data
        cv = CrossValidator(cv_strategy="stratified", n_splits=5)

        splits = list(cv.get_stratified_splits(X, y))
        assert len(splits) == 5


# ============================================================================
# HyperparameterTuner Tests
# ============================================================================


class TestHyperparameterTuner:
    def test_initialization(self):

        from src.models.ml.hyperparameter_tuning import HyperparameterTuner

        tuner = HyperparameterTuner()
        assert tuner.cv_strategy == "loso"
        assert tuner.n_trials == 50
        assert tuner.sampler_type == "tpe"

    def test_get_search_space(self):

        from src.models.ml.hyperparameter_tuning import HyperparameterTuner

        tuner = HyperparameterTuner()

        lr_space = tuner.get_search_space("lr")
        assert "C" in lr_space
        assert "penalty" in lr_space

        rf_space = tuner.get_search_space("rf")
        assert "n_estimators" in rf_space
        assert "max_depth" in rf_space

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="optuna not installed"),
        reason="Optuna required",
    )
    def test_tune_basic(self, sample_data_with_groups):

        from src.models.ml.hyperparameter_tuning import HyperparameterTuner

        X, y, groups = sample_data_with_groups
        tuner = HyperparameterTuner(n_trials=3, cv_strategy="loso")

        best_params, best_score, study = tuner.tune(
            "lr", X, y, groups, show_progress_bar=False
        )

        assert isinstance(best_params, dict)
        assert best_score >= 0
        assert study is not None


# ============================================================================
# Ensemble Tests
# ============================================================================


class TestStackingEnsemble:
    def test_initialization(self):

        from src.models.ml.ensemble import StackingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        base_models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = StackingEnsemble(base_models, meta_learner="lr")
        assert len(ensemble.base_models) == 2
        assert ensemble.use_probas is True

    def test_fit_predict(self, sample_data):

        from src.models.ml.ensemble import StackingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        X, y = sample_data

        base_models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = StackingEnsemble(base_models, meta_learner="lr")
        ensemble.fit(X, y)

        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)

    def test_predict_proba(self, sample_data):

        from src.models.ml.ensemble import StackingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        X, y = sample_data

        base_models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = StackingEnsemble(base_models, meta_learner="lr")
        ensemble.fit(X, y)

        probas = ensemble.predict_proba(X)
        assert probas.shape == (len(y), len(np.unique(y)))


class TestVotingEnsemble:
    def test_initialization_soft(self):

        from src.models.ml.ensemble import VotingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = VotingEnsemble(models, voting="soft")
        assert ensemble.voting == "soft"
        assert len(ensemble.models) == 2

    def test_initialization_hard(self):

        from src.models.ml.ensemble import VotingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = VotingEnsemble(models, voting="hard")
        assert ensemble.voting == "hard"

    def test_custom_weights(self):

        from src.models.ml.ensemble import VotingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = VotingEnsemble(models, voting="soft", weights=[0.7, 0.3])
        assert ensemble.weights == [0.7, 0.3]

    def test_fit_predict(self, sample_data):

        from src.models.ml.ensemble import VotingEnsemble
        from src.models.ml.classifiers import (
            RandomForestClassifier,
            LogisticRegressionClassifier,
        )

        X, y = sample_data

        models = [
            ("rf", RandomForestClassifier(n_estimators=50)),
            ("lr", LogisticRegressionClassifier()),
        ]

        ensemble = VotingEnsemble(models, voting="soft")
        ensemble.fit(X, y)

        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)


# ============================================================================
# ModelEvaluator Tests
# ============================================================================


class TestModelEvaluator:
    def test_initialization(self):

        from src.models.ml.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()
        assert evaluator.threshold == 0.5
        assert 0 in evaluator.class_names

    def test_compute_metrics(self, sample_data):

        from src.models.ml.evaluation import ModelEvaluator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)

        evaluator = ModelEvaluator()
        metrics = evaluator.compute_metrics(y, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "mcc" in metrics

    def test_compute_per_class_metrics(self, sample_data):

        from src.models.ml.evaluation import ModelEvaluator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        evaluator = ModelEvaluator()
        per_class = evaluator.compute_per_class_metrics(y, y_pred)

        assert len(per_class) == len(np.unique(y))
        assert "precision" in per_class.columns
        assert "recall" in per_class.columns
        assert "f1" in per_class.columns

    def test_confusion_matrix(self, sample_data):

        from src.models.ml.evaluation import ModelEvaluator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        evaluator = ModelEvaluator()
        cm = evaluator.get_confusion_matrix(y, y_pred)

        n_classes = len(np.unique(y))
        assert cm.shape == (n_classes, n_classes)

    def test_classification_report(self, sample_data):

        from src.models.ml.evaluation import ModelEvaluator
        from src.models.ml.classifiers import LogisticRegressionClassifier

        X, y = sample_data
        clf = LogisticRegressionClassifier()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        evaluator = ModelEvaluator()
        report = evaluator.get_classification_report(y, y_pred, output_dict=True)

        assert "accuracy" in report
        assert "macro avg" in report

    def test_statistical_comparison(self):

        from src.models.ml.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()

        # Create synthetic per-fold scores
        np.random.seed(42)
        model1_scores = np.random.uniform(0.7, 0.85, 10)
        model2_scores = np.random.uniform(0.65, 0.80, 10)

        results = evaluator.statistical_comparison(model1_scores, model2_scores)

        assert "mean_difference" in results
        assert "paired_ttest" in results
        assert "wilcoxon" in results
        assert "is_significant" in results
        assert "statistic" in results["paired_ttest"]
        assert "pvalue" in results["paired_ttest"]

    def test_compute_confidence_intervals(self):

        from src.models.ml.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()

        scores = np.array([0.75, 0.78, 0.72, 0.80, 0.77, 0.73, 0.79, 0.76])

        mean, lower, upper = evaluator.compute_confidence_intervals(
            scores, confidence=0.95
        )

        assert lower < mean < upper
        assert 0 < lower < 1
        assert 0 < upper <= 1

    def test_mcnemar_test(self, sample_data):

        from src.models.ml.evaluation import ModelEvaluator
        from src.models.ml.classifiers import (
            LogisticRegressionClassifier,
            RandomForestClassifier,
        )

        X, y = sample_data
        clf1 = LogisticRegressionClassifier()
        clf2 = RandomForestClassifier(n_estimators=50)

        clf1.fit(X, y)
        clf2.fit(X, y)

        y_pred1 = clf1.predict(X)
        y_pred2 = clf2.predict(X)

        evaluator = ModelEvaluator()
        results = evaluator.mcnemar_test(y, y_pred1, y_pred2)

        assert "statistic" in results
        assert "pvalue" in results
        assert "significant" in results


# ============================================================================
# ImbalanceHandler Tests
# ============================================================================


class TestImbalanceHandler:
    def test_initialization_smote(self):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        handler = ImbalanceHandler(strategy="smote")
        assert handler.strategy == "smote"

    def test_initialization_invalid_strategy(self):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        with pytest.raises(ValueError, match="Unknown strategy"):
            ImbalanceHandler(strategy="invalid")

    def test_analyze_imbalance(self, imbalanced_data):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        X, y = imbalanced_data
        handler = ImbalanceHandler()

        analysis = handler.analyze_imbalance(y)

        assert "n_classes" in analysis
        assert "imbalance_ratio" in analysis
        assert "class_counts" in analysis
        assert analysis["imbalance_ratio"] > 1.0

    def test_get_class_weights(self, imbalanced_data):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy="class_weight")

        weights = handler.get_class_weights(y)

        assert len(weights) == len(np.unique(y))
        # Minority class should have
        assert weights[1] > weights[0]

    def test_get_sample_weights(self, imbalanced_data):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        X, y = imbalanced_data
        handler = ImbalanceHandler()

        sample_weights = handler.get_sample_weights(y)
        assert len(sample_weights) == len(y)

    @pytest.mark.skipif(
        not pytest.importorskip("imblearn", reason="imbalanced-learn not installed"),
        reason="imbalanced-learn required",
    )
    def test_resample_smote(self, imbalanced_data):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        X, y = imbalanced_data
        handler = ImbalanceHandler(strategy="smote")

        X_resampled, y_resampled = handler.resample(X, y)

        # Should have more samples
        assert len(y_resampled) >= len(y)

    def test_recommend_strategy(self, imbalanced_data):

        from src.models.ml.imbalance_handler import ImbalanceHandler

        X, y = imbalanced_data
        handler = ImbalanceHandler()

        recommendation = handler.recommend_strategy(y, len(y))
        assert recommendation in ImbalanceHandler.STRATEGIES


# ============================================================================
# MLTrainingPipeline Tests
# ============================================================================


class TestMLTrainingPipeline:
    def test_initialization(self):

        from src.models.ml.training_pipeline import MLTrainingPipeline

        pipeline = MLTrainingPipeline()
        assert pipeline.cv_strategy == "loso"
        assert pipeline.n_tuning_trials == 30
        assert pipeline.imbalance_strategy == "class_weight"

    def test_run_baseline_only(self, sample_data_with_groups):

        from src.models.ml.training_pipeline import MLTrainingPipeline

        X, y, groups = sample_data_with_groups

        pipeline = MLTrainingPipeline(cv_strategy="loso")
        results = pipeline.run_baseline_only(X, y, groups, classifiers=["lr", "rf"])

        assert "lr" in results
        assert "rf" in results
        assert "accuracy_mean" in results["lr"]

    def test_data_analysis(self, sample_data_with_groups):

        from src.models.ml.training_pipeline import MLTrainingPipeline

        X, y, groups = sample_data_with_groups

        pipeline = MLTrainingPipeline()
        analysis = pipeline._analyze_data(X, y, groups)

        assert "n_samples" in analysis
        assert "n_features" in analysis
        assert "n_classes" in analysis
        assert "n_subjects" in analysis


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    def test_full_workflow(self, sample_data_with_groups):

        from src.models.ml.classifiers import RandomForestClassifier
        from src.models.ml.cross_validation import CrossValidator
        from src.models.ml.evaluation import ModelEvaluator

        X, y, groups = sample_data_with_groups

        # Train model
        clf = RandomForestClassifier(n_estimators=50)

        # Cross-validate
        cv = CrossValidator(cv_strategy="loso")
        cv_results = cv.cross_validate(clf, X, y, groups, return_predictions=True)

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.compute_metrics(y, cv_results["predictions"])

        assert "accuracy" in metrics
        assert cv_results["cv_strategy"] == "loso"

    def test_model_save_load(self, sample_data, tmp_path):

        from src.models.ml.classifiers import RandomForestClassifier

        X, y = sample_data

        # Train and save
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X, y)

        model_path = tmp_path / "model.pkl"
        clf.save_model(model_path)

        # Load and predict
        loaded_clf = RandomForestClassifier.load_model(model_path)
        predictions = loaded_clf.predict(X)

        assert len(predictions) == len(y)
        assert loaded_clf.is_fitted


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
