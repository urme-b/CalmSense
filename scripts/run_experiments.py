#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import WESADLoader
from src.preprocessing import SignalProcessor
from src.features import FeatureExtractor
from src.models.ml import (
    RandomForestClassifier,
    XGBoostClassifier,
    LightGBMClassifier,
    CatBoostClassifier,
    SVMClassifier,
)
from src.models.ml.hyperparameter_tuning import OptunaHyperparameterTuner
from src.models.ml.evaluation import ModelEvaluator
from src.models.ml.ensemble import StackingEnsemble, VotingEnsemble


# Configure logging
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:

    log_file = output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger("CalmSense")


class ExperimentRunner:
    # Available models
    ML_MODELS = {
        "random_forest": RandomForestClassifier,
        "xgboost": XGBoostClassifier,
        "lightgbm": LightGBMClassifier,
        "catboost": CatBoostClassifier,
        "svm": SVMClassifier,
    }

    DL_MODELS = {
        # Deep learning models would
        # "cnn_lstm": CNNLSTMClassifier,
        # "transformer": TransformerClassifier,
    }

    # Subject IDs in WESAD
    SUBJECTS = [
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
        "S13",
        "S14",
        "S15",
        "S16",
        "S17",
    ]

    def __init__(
        self, config: Dict[str, Any], output_dir: Path, logger: logging.Logger
    ):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger

        # Create output directories
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        (output_dir / "results").mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.feature_extractor = None
        self.evaluator = ModelEvaluator()

    def load_config(self, config_path: str) -> Dict[str, Any]:

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_data_pipeline(self) -> None:

        self.logger.info("Setting up data pipeline...")

        data_dir = Path(self.config.get("data_dir", "data/raw/WESAD"))

        self.data_loader = WESADLoader(str(data_dir))
        self.preprocessor = SignalProcessor(
            sampling_rate=self.config.get("sampling_rate", 700),
            window_size=self.config.get("window_size", 60),
            step_size=self.config.get("step_size", 30),
        )
        self.feature_extractor = FeatureExtractor(
            feature_groups=self.config.get("feature_groups", "all")
        )

        self.logger.info("Data pipeline initialized")

    def load_features(
        self, cache_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Check for cached features
        if cache_path and cache_path.exists():
            self.logger.info(f"Loading cached features from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data["features"], data["labels"], data["subject_ids"]

        self.logger.info("Extracting features from all subjects...")

        all_features = []
        all_labels = []
        all_subject_ids = []

        for subject_id in self.SUBJECTS:
            self.logger.info(f"Processing subject {subject_id}...")

            try:
                # Load raw data
                raw_data = self.data_loader.load_subject(subject_id)

                # Preprocess signals
                processed_data = self.preprocessor.process(raw_data)

                # Extract features
                features, labels = self.feature_extractor.extract_all(processed_data)

                # Store
                all_features.append(features)
                all_labels.append(labels)
                all_subject_ids.extend([subject_id] * len(labels))

                self.logger.info(
                    f"  Subject {subject_id}: {len(labels)} samples extracted"
                )

            except Exception as e:
                self.logger.error(f"Error processing subject {subject_id}: {e}")
                continue

        # Concatenate
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        subjects = np.array(all_subject_ids)

        # Cache features
        if cache_path:
            self.logger.info(f"Caching features to {cache_path}")
            np.savez(cache_path, features=X, labels=y, subject_ids=subjects)

        self.logger.info(f"Total: {len(y)} samples, {X.shape[1]} features")
        return X, y, subjects

    def run_loso_cv(
        self,
        model_class: type,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Any]:

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Running LOSO CV for {model_name}")
        self.logger.info(f"{'=' * 60}")

        results = {
            "model": model_name,
            "folds": [],
            "metrics": {},
            "predictions": [],
            "training_time": 0,
        }

        unique_subjects = np.unique(subjects)
        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        start_time = time.time()

        for i, test_subject in enumerate(unique_subjects):
            self.logger.info(
                f"\nFold {i + 1}/{len(unique_subjects)}: Test subject = {test_subject}"
            )

            # Split data
            train_mask = subjects != test_subject
            test_mask = subjects == test_subject

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            self.logger.info(
                f"  Train: {len(y_train)} samples, Test: {len(y_test)} samples"
            )

            # Create model
            model = model_class()

            # Hyperparameter tuning (on training
            if tune_hyperparameters and hasattr(model, "get_param_space"):
                self.logger.info("  Tuning hyperparameters...")
                tuner = OptunaHyperparameterTuner(
                    model_class=model_class,
                    n_trials=self.config.get("n_trials", 50),
                    cv=5,
                    metric="f1_weighted",
                )
                best_params = tuner.tune(X_train, y_train)
                model = model_class(**best_params)
                self.logger.info(f"  Best params: {best_params}")

            # Train
            fold_start = time.time()
            model.fit(X_train, y_train)
            fold_time = time.time() - fold_start

            # Predict
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            )

            # Evaluate fold
            fold_metrics = self.evaluator.compute_metrics(y_test, y_pred, y_proba)
            fold_metrics["subject"] = test_subject
            fold_metrics["training_time"] = fold_time

            results["folds"].append(fold_metrics)

            self.logger.info(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            self.logger.info(f"  F1: {fold_metrics['f1_weighted']:.4f}")

            # Collect predictions
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)

        results["training_time"] = time.time() - start_time

        # Aggregate metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba) if all_y_proba else None

        results["metrics"] = self.evaluator.compute_metrics(
            all_y_true, all_y_pred, all_y_proba
        )
        results["predictions"] = {
            "y_true": all_y_true.tolist(),
            "y_pred": all_y_pred.tolist(),
        }

        # Log summary
        self.logger.info(f"\n{model_name} LOSO CV Results:")
        self.logger.info(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        self.logger.info(f"  F1 (weighted): {results['metrics']['f1_weighted']:.4f}")
        self.logger.info(f"  AUC-ROC: {results['metrics'].get('auc_roc', 'N/A')}")
        self.logger.info(f"  MCC: {results['metrics']['mcc']:.4f}")
        self.logger.info(f"  Total training time: {results['training_time']:.2f}s")

        return results

    def run_ml_experiments(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        all_results = {}

        model_list = models or list(self.ML_MODELS.keys())

        for model_name in model_list:
            if model_name not in self.ML_MODELS:
                self.logger.warning(f"Unknown model: {model_name}")
                continue

            model_class = self.ML_MODELS[model_name]
            results = self.run_loso_cv(
                model_class=model_class,
                model_name=model_name,
                X=X,
                y=y,
                subjects=subjects,
                tune_hyperparameters=self.config.get("tune_hyperparameters", True),
            )
            all_results[model_name] = results

            # Save individual results
            results_path = self.output_dir / "results" / f"{model_name}_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

        return all_results

    def run_ensemble_experiments(
        self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray
    ) -> Dict[str, Any]:

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Running Ensemble Experiments")
        self.logger.info("=" * 60)

        results = {}

        # Stacking ensemble
        def make_stacking():
            base_models = [
                ("rf", RandomForestClassifier()),
                ("xgb", XGBoostClassifier()),
                ("lgb", LightGBMClassifier()),
            ]
            return StackingEnsemble(
                base_models=base_models, meta_model="logistic_regression"
            )

        results["stacking"] = self.run_loso_cv(
            model_class=make_stacking,
            model_name="stacking_ensemble",
            X=X,
            y=y,
            subjects=subjects,
            tune_hyperparameters=False,
        )

        # Voting ensemble
        def make_voting():
            return VotingEnsemble(
                models=[
                    ("rf", RandomForestClassifier()),
                    ("xgb", XGBoostClassifier()),
                    ("lgb", LightGBMClassifier()),
                ],
                voting="soft",
            )

        results["voting"] = self.run_loso_cv(
            model_class=make_voting,
            model_name="voting_ensemble",
            X=X,
            y=y,
            subjects=subjects,
            tune_hyperparameters=False,
        )

        return results

    def generate_report(self, all_results: Dict[str, Any]) -> None:

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Generating Experiment Report")
        self.logger.info("=" * 60)

        # Create summary DataFrame
        summary_data = []
        for model_name, results in all_results.items():
            metrics = results["metrics"]
            summary_data.append(
                {
                    "Model": model_name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "F1 (weighted)": f"{metrics['f1_weighted']:.4f}",
                    "Precision": f"{metrics['precision_weighted']:.4f}",
                    "Recall": f"{metrics['recall_weighted']:.4f}",
                    "MCC": f"{metrics['mcc']:.4f}",
                    "AUC-ROC": f"{metrics.get('auc_roc', 0):.4f}",
                    "Time (s)": f"{results['training_time']:.2f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Accuracy", ascending=False)

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Save summary
        summary_path = self.output_dir / "results" / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Summary saved to {summary_path}")

        # Save full results
        full_results_path = self.output_dir / "results" / "all_results.json"
        with open(full_results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        self.logger.info(f"Full results saved to {full_results_path}")

    def run(self) -> None:

        self.logger.info("Starting CalmSense Experiment Pipeline")
        self.logger.info(f"Configuration: {self.config}")

        # Setup data pipeline
        self.setup_data_pipeline()

        # Load/extract features
        cache_path = Path(
            self.config.get("feature_cache", "data/processed/features.npz")
        )
        X, y, subjects = self.load_features(cache_path)

        all_results = {}

        # Run ML experiments
        if self.config.get("run_ml", True):
            ml_models = self.config.get("ml_models", None)
            ml_results = self.run_ml_experiments(X, y, subjects, ml_models)
            all_results.update(ml_results)

        # Run ensemble experiments
        if self.config.get("run_ensemble", True):
            ensemble_results = self.run_ensemble_experiments(X, y, subjects)
            all_results.update(ensemble_results)

        # Generate report
        self.generate_report(all_results)

        self.logger.info("\nExperiment pipeline complete!")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="CalmSense Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/WESAD",
        help="Path to WESAD dataset directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for experiment outputs",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["all", "ml", "dl", "ensemble"],
        help="Which models to run",
    )

    parser.add_argument(
        "--ml-models",
        type=str,
        nargs="+",
        default=None,
        help="Specific ML models to run",
    )

    parser.add_argument(
        "--cv",
        type=str,
        default="loso",
        choices=["loso", "kfold", "stratified"],
        help="Cross-validation strategy",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning",
    )

    parser.add_argument(
        "--no-tune", action="store_true", help="Disable hyperparameter tuning"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for DL models",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--profile", action="store_true", help="Enable profiling")

    return parser.parse_args()


def main():

    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, args.log_level)

    # Build configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data_dir": args.data_dir,
            "cv_strategy": args.cv,
            "n_trials": args.n_trials,
            "tune_hyperparameters": not args.no_tune,
            "device": args.device,
            "seed": args.seed,
            "run_ml": args.models in ["all", "ml"],
            "run_dl": args.models in ["all", "dl"],
            "run_ensemble": args.models in ["all", "ensemble"],
            "ml_models": args.ml_models,
        }

    # Run experiments
    runner = ExperimentRunner(config=config, output_dir=output_dir, logger=logger)

    if args.profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        runner.run()
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        stats.dump_stats(output_dir / "profile.stats")
    else:
        runner.run()


if __name__ == "__main__":
    main()
