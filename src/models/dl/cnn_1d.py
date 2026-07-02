"""Residual 1D-CNN for raw multichannel windows."""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ...logging_config import LoggerMixin


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.pool(self.drop(out))


class _Net(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, widths=(32, 64, 128)):
        super().__init__()
        # Strided stem downsamples long windows
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        blocks = []
        prev = widths[0]
        for w in widths:
            blocks.append(_ResidualBlock(prev, w))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


class CNN1DClassifier(LoggerMixin):
    """Sklearn-style residual 1D-CNN trained on raw windows."""

    def __init__(
        self,
        in_channels: int = 5,
        max_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        patience: int = 8,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Any = None
        self.classes_: Any = None
        self._mean: Any = None
        self._std: Any = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean) / self._std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)

        # Per-channel train stats
        self._mean = X.mean(axis=(0, 2), keepdims=True)
        self._std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        X = self._standardize(X)

        x_tr, x_val, y_tr, y_val = train_test_split(
            X,
            y_idx,
            test_size=self.val_fraction,
            stratify=y_idx,
            random_state=self.random_state,
        )

        weights = compute_class_weight("balanced", classes=np.unique(y_idx), y=y_idx)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        self.model = _Net(self.in_channels, len(self.classes_)).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs)

        tr = torch.utils.data.TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
        gen = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            tr, batch_size=self.batch_size, shuffle=True, generator=gen
        )
        x_val_t = torch.from_numpy(x_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).long().to(self.device)

        best_loss = np.inf
        best_state = None
        stale = 0

        for _ in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Early stopping
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(x_val_t), y_val_t).item()
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._standardize(np.asarray(X, dtype=np.float32))
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(self.device)
                probs.append(torch.softmax(self.model(xb), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
