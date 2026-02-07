from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dl_model import BaseDLModel


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        use_pool: bool = True,
        pool_size: int = 2,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):

        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.use_pool = use_pool
        self.use_residual = use_residual

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_pool:
            self.pool = nn.MaxPool1d(pool_size)

        # Residual projection if dimensions
        if use_residual:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv(x)
        out = self.bn(out)

        if self.use_residual:
            identity = self.residual(identity)
            # Handle pooling for residual
            if self.use_pool:
                ks = (
                    self.pool.kernel_size
                    if isinstance(self.pool.kernel_size, int)
                    else self.pool.kernel_size[0]
                )
                identity = F.max_pool1d(identity, kernel_size=ks)

        out = self.relu(out)
        out = self.dropout(out)

        if self.use_pool:
            out = self.pool(out)

        if self.use_residual:
            out = out + identity

        return out


class GlobalAveragePooling1D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x.mean(dim=-1)


class CNN1D(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 700,
        base_filters: int = 64,
        num_conv_layers: int = 3,
        kernel_sizes: Optional[List[int]] = None,
        use_residual: bool = False,
        dropout: float = 0.5,
        fc_hidden_dim: int = 128,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.sequence_length = sequence_length
        self.base_filters = base_filters
        self.num_conv_layers = num_conv_layers
        self.use_residual = use_residual
        self.dropout_rate = dropout
        self.fc_hidden_dim = fc_hidden_dim

        # Default kernel sizes
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3] + [3] * (num_conv_layers - 3)
        self.kernel_sizes = kernel_sizes[:num_conv_layers]

        # Build convolutional layers
        self.conv_layers = self._build_conv_layers()

        # Global average pooling
        self.global_pool = GlobalAveragePooling1D()

        # Compute final channels
        final_channels = base_filters * (2 ** (num_conv_layers - 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

        # Initialize weights
        self.initialize_weights("kaiming")

        # Move to device
        self.to(self._device)

    def _build_conv_layers(self) -> nn.ModuleList:

        layers = nn.ModuleList()

        in_channels = self.input_dim
        out_channels = self.base_filters

        for i in range(self.num_conv_layers):
            kernel_size = self.kernel_sizes[i] if i < len(self.kernel_sizes) else 3

            # Don't pool on last
            use_pool = i < self.num_conv_layers - 1

            block = ConvBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_pool=use_pool,
                use_residual=self.use_residual,
                dropout=self.dropout_rate * 0.5 if i < self.num_conv_layers - 1 else 0,
            )
            layers.append(block)

            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Cap at 512 channels

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global average pooling
        x = self.global_pool(x)

        # Classification
        logits = self.classifier(x)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.global_pool(x)

        # Get features before final
        for layer in self.classifier[:-1]:
            x = layer(x)

        return x

    def _get_config(self) -> Dict[str, Any]:

        return {
            "sequence_length": self.sequence_length,
            "base_filters": self.base_filters,
            "num_conv_layers": self.num_conv_layers,
            "kernel_sizes": self.kernel_sizes,
            "use_residual": self.use_residual,
            "dropout": self.dropout_rate,
            "fc_hidden_dim": self.fc_hidden_dim,
        }


class CNN1DResidual(CNN1D):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 700,
        base_filters: int = 64,
        num_conv_layers: int = 4,
        dropout: float = 0.5,
        device: str = "auto",
    ):

        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            sequence_length=sequence_length,
            base_filters=base_filters,
            num_conv_layers=num_conv_layers,
            use_residual=True,
            dropout=dropout,
            device=device,
        )


class MultiScaleCNN1D(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 700,
        branch_filters: int = 32,
        branch_kernels: List[int] = None,
        shared_filters: int = 128,
        dropout: float = 0.5,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        if branch_kernels is None:
            branch_kernels = [3, 7, 15, 31]

        self.branch_kernels = branch_kernels
        self.num_branches = len(branch_kernels)

        # Parallel branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                )
                for k in branch_kernels
            ]
        )

        # Shared layers after concatenation
        concat_channels = branch_filters * self.num_branches
        self.shared = nn.Sequential(
            ConvBlock1D(concat_channels, shared_filters, kernel_size=5, use_pool=True),
            ConvBlock1D(
                shared_filters, shared_filters * 2, kernel_size=3, use_pool=False
            ),
        )

        self.global_pool = GlobalAveragePooling1D()

        self.classifier = nn.Sequential(
            nn.Linear(shared_filters * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.initialize_weights("kaiming")
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Process each branch
        branch_outputs = [branch(x) for branch in self.branches]

        # Align sequence lengths (take
        min_len = min(out.size(-1) for out in branch_outputs)
        branch_outputs = [out[..., :min_len] for out in branch_outputs]

        # Concatenate along channel dimension
        x = torch.cat(branch_outputs, dim=1)

        # Shared layers
        x = self.shared(x)

        # Pool and classify
        x = self.global_pool(x)
        logits = self.classifier(x)

        return logits

    def _get_config(self) -> Dict[str, Any]:

        return {
            "branch_kernels": self.branch_kernels,
        }
