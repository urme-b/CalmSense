from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    from torch.cuda.amp import GradScaler, autocast

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from ...logging_config import LoggerMixin
from .base_dl_model import BaseDLModel
from .data_loader import create_loso_loaders


class EarlyStopping:
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        import logging

        logger = logging.getLogger(__name__)

        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.debug(f"EarlyStopping: no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"EarlyStopping triggered after {self.patience} epochs without improvement"
                )
        else:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(self, model: nn.Module) -> None:
        if self.restore_best:
            self.best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    def restore(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            self.best_weights = None

    def cleanup(self) -> None:
        self.best_weights = None


class DLTrainer(LoggerMixin):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Auto-detect device
        if device is None:
            if hasattr(model, "_device"):
                self.device = model._device
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)

        self.config = config or {}
        self.use_amp = self.config.get(
            "use_amp", torch.cuda.is_available() and AMP_AVAILABLE
        )
        self.clip_grad_norm = self.config.get("clip_grad_norm", 1.0)
        self.scaler = GradScaler() if self.use_amp else None
        self.scheduler = None

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rates": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.current_epoch = 0

        self.logger.info(
            f"Initialized DLTrainer on {self.device} (AMP: {self.use_amp})"
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()

                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()

                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )

                self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total

        from sklearn.metrics import f1_score, balanced_accuracy_score

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": f1,
            "balanced_accuracy": balanced_acc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        T_0 = self.config.get("scheduler_T0", 10)
        T_mult = self.config.get("scheduler_T_mult", 2)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult
        )

        early_stopping = EarlyStopping(
            patience=early_stopping_patience, mode="min", restore_best=True
        )

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    if save_best:
                        self.best_model_state = {
                            k: v.cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }

                if early_stopping(val_metrics["loss"], self.model):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            self.scheduler.step()
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{num_epochs}"
                msg += f" - train_loss: {train_metrics['loss']:.4f}"
                msg += f" - train_acc: {train_metrics['accuracy']:.4f}"
                if val_loader is not None:
                    msg += f" - val_loss: {val_metrics['loss']:.4f}"
                    msg += f" - val_acc: {val_metrics['accuracy']:.4f}"
                self.logger.info(msg)

        # Restore best weights
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.best_model_state = None
            self.logger.info(
                f"Restored best model (val_loss: {self.best_val_loss:.4f})"
            )

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(checkpoint_dir / "best_model.pt")

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed:.1f}s")

        return self.history

    def predict(
        self, test_loader: DataLoader, return_proba: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        all_preds = []
        all_proba = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)

                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)

                proba = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        predictions = np.array(all_proba) if return_proba else np.array(all_preds)
        targets = np.array(all_targets)

        return predictions, targets

    def save_checkpoint(
        self, path: Union[str, Path], include_optimizer: bool = True
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self, path: Union[str, Path], load_optimizer: bool = True
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_optimizer and "scheduler_state_dict" in checkpoint:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded from {path}")

    def plot_training_history(self, figsize: Tuple[int, int] = (14, 5)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["train_accuracy"], label="Train")
        if self.history["val_accuracy"]:
            axes[1].plot(self.history["val_accuracy"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if self.history["learning_rates"]:
            axes[2].plot(self.history["learning_rates"])
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Learning Rate")
            axes[2].set_title("Learning Rate Schedule")
            axes[2].set_yscale("log")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def loso_cross_validate(
    model_class: Type[BaseDLModel],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_kwargs: Dict,
    trainer_config: Optional[Dict] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    early_stopping_patience: int = 15,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:

    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)

    fold_results = []
    all_predictions = []
    all_targets = []

    logger = LoggerMixin().logger
    logger.info(f"Starting LOSO CV with {n_subjects} subjects")

    for fold_idx, test_subject in enumerate(unique_subjects):
        if verbose:
            logger.info(
                f"Fold {fold_idx + 1}/{n_subjects}: Testing on subject {test_subject}"
            )

        train_loader, val_loader, test_loader = create_loso_loaders(
            X, y, groups, test_subject, batch_size=batch_size, val_split=0.1
        )

        model = model_class(**model_kwargs, device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Class-balanced loss
        train_labels = y[groups != test_subject]
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor(
            len(train_labels) / (len(class_counts) * class_counts), dtype=torch.float32
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(model.get_device()))

        trainer = DLTrainer(
            model=model, optimizer=optimizer, criterion=criterion, config=trainer_config
        )

        trainer.fit(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=False,
        )

        predictions, targets = trainer.predict(test_loader)
        fold_results.append(
            {
                "subject": test_subject,
                "accuracy": accuracy_score(targets, predictions),
                "f1_macro": f1_score(targets, predictions, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(targets, predictions),
                "n_samples": len(targets),
            }
        )

        all_predictions.extend(predictions)
        all_targets.extend(targets)

        if verbose:
            logger.info(
                f"  Subject {test_subject}: "
                f"Acc={fold_results[-1]['accuracy']:.4f}, "
                f"F1={fold_results[-1]['f1_macro']:.4f}"
            )

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    results = {
        "fold_results": fold_results,
        "accuracy_mean": np.mean([r["accuracy"] for r in fold_results]),
        "accuracy_std": np.std([r["accuracy"] for r in fold_results]),
        "f1_macro_mean": np.mean([r["f1_macro"] for r in fold_results]),
        "f1_macro_std": np.std([r["f1_macro"] for r in fold_results]),
        "balanced_accuracy_mean": np.mean(
            [r["balanced_accuracy"] for r in fold_results]
        ),
        "balanced_accuracy_std": np.std([r["balanced_accuracy"] for r in fold_results]),
        "predictions": all_predictions,
        "targets": all_targets,
        "cv_strategy": "loso",
        "n_subjects": n_subjects,
    }

    logger.info(
        f"LOSO CV Results: "
        f"Accuracy={results['accuracy_mean']:.4f} +/- {results['accuracy_std']:.4f}, "
        f"F1={results['f1_macro_mean']:.4f} +/- {results['f1_macro_std']:.4f}"
    )

    return results
