from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...logging_config import LoggerMixin


class BaseDLModel(nn.Module, LoggerMixin, ABC):
    def __init__(self, input_dim: int, num_classes: int, device: str = "auto"):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self._device = self._resolve_device(device)

        self.logger.debug(
            f"Initialized {self.__class__.__name__} "
            f"(input_dim={input_dim}, num_classes={num_classes}, device={self._device})"
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_device(self) -> torch.device:
        return self._device

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("=" * 60)
        lines.append(f"Input dim: {self.input_dim}")
        lines.append(f"Num classes: {self.num_classes}")
        lines.append(f"Device: {self._device}")
        lines.append("-" * 60)

        total_params = 0
        trainable_params = 0

        lines.append(f"{'Layer':<35} {'Output Shape':<20} {'Params':>10}")
        lines.append("-" * 60)

        for name, module in self.named_modules():
            if name == "":
                continue

            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            if params > 0:
                module_name = name[:32] + "..." if len(name) > 35 else name
                lines.append(f"{module_name:<35} {'--':<20} {params:>10,}")
                total_params += params
                trainable_params += trainable

        lines.append("=" * 60)
        lines.append(f"Total params: {total_params:,}")
        lines.append(f"Trainable params: {trainable_params:,}")
        lines.append(f"Non-trainable params: {total_params - trainable_params:,}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self._device)

    def initialize_weights(self, method: str = "xavier") -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif method == "orthogonal":
                    nn.init.orthogonal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def save_model(
        self,
        path: Union[str, Path],
        save_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_class": self.__class__.__name__,
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "config": self._get_config(),
        }

        if save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")

        return path

    @classmethod
    def load_model(
        cls, path: Union[str, Path], device: str = "auto", **kwargs
    ) -> "BaseDLModel":
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", {})
        config.update(kwargs)

        model = cls(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            device=device,
            **config,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.get_device())

        return model

    def _get_config(self) -> Dict[str, Any]:
        return {}

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.logger.debug(f"Froze all parameters in {self.__class__.__name__}")

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.logger.debug(f"Unfroze all parameters in {self.__class__.__name__}")

    def get_gradient_norm(self) -> float:
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
