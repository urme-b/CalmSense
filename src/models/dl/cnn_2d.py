from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    from torchvision.models import (  # noqa: F401
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        ResNet18_Weights,
        ResNet34_Weights,
        MobileNet_V3_Small_Weights,
        MobileNet_V3_Large_Weights,
    )

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from .base_dl_model import BaseDLModel


# Backbone configurations
BACKBONE_CONFIGS = {
    "efficientnet_b0": {
        "model_fn": "efficientnet_b0",
        "weights_class": "EfficientNet_B0_Weights",
        "feature_dim": 1280,
        "classifier_attr": "classifier",
    },
    "efficientnet_b1": {
        "model_fn": "efficientnet_b1",
        "weights_class": "EfficientNet_B1_Weights",
        "feature_dim": 1280,
        "classifier_attr": "classifier",
    },
    "resnet18": {
        "model_fn": "resnet18",
        "weights_class": "ResNet18_Weights",
        "feature_dim": 512,
        "classifier_attr": "fc",
    },
    "resnet34": {
        "model_fn": "resnet34",
        "weights_class": "ResNet34_Weights",
        "feature_dim": 512,
        "classifier_attr": "fc",
    },
    "mobilenet_v3_small": {
        "model_fn": "mobilenet_v3_small",
        "weights_class": "MobileNet_V3_Small_Weights",
        "feature_dim": 576,
        "classifier_attr": "classifier",
    },
    "mobilenet_v3_large": {
        "model_fn": "mobilenet_v3_large",
        "weights_class": "MobileNet_V3_Large_Weights",
        "feature_dim": 960,
        "classifier_attr": "classifier",
    },
}


class CNN2DTransferLearning(BaseDLModel):
    SUPPORTED_BACKBONES = list(BACKBONE_CONFIGS.keys())

    def __init__(
        self,
        input_dim: int = 3,
        num_classes: int = 3,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        hidden_dim: int = 256,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for transfer learning models")

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim

        # Get backbone configuration
        config = BACKBONE_CONFIGS[backbone]
        self.feature_dim = config["feature_dim"]

        # Load backbone
        self.backbone, self._feature_extractor = self._load_backbone(
            backbone, pretrained, config
        )

        # Handle channel mismatch (if
        if input_dim != 3:
            self._adapt_input_channels(input_dim)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(self._device)

    def _load_backbone(
        self, backbone: str, pretrained: bool, config: Dict
    ) -> Tuple[nn.Module, nn.Module]:

        model_fn = getattr(models, config["model_fn"])

        if pretrained:
            weights_class = getattr(models, config["weights_class"], None)
            if weights_class:
                weights = weights_class.DEFAULT
            else:
                weights = "DEFAULT"
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        # Create feature extractor by
        if backbone.startswith("efficientnet"):
            # EfficientNet: features → avgpool
            feature_extractor = nn.Sequential(
                model.features, model.avgpool, nn.Flatten()
            )
        elif backbone.startswith("resnet"):
            # ResNet: remove final fc
            feature_extractor = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
                nn.Flatten(),
            )
        elif backbone.startswith("mobilenet"):
            # MobileNet: features → avgpool
            feature_extractor = nn.Sequential(
                model.features, model.avgpool, nn.Flatten()
            )
        else:
            raise ValueError(f"Unknown backbone architecture: {backbone}")

        return model, feature_extractor

    def _adapt_input_channels(self, input_channels: int) -> None:

        if self.backbone_name.startswith("efficientnet"):
            old_conv = self._feature_extractor[0][0][0]  # features[0][0]
        elif self.backbone_name.startswith("resnet"):
            old_conv = self._feature_extractor[0]  # conv1
        elif self.backbone_name.startswith("mobilenet"):
            old_conv = self._feature_extractor[0][0][0]  # features[0][0]
        else:
            return

        # Create new conv with
        new_conv = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new channels by
        if self.pretrained:
            with torch.no_grad():
                if input_channels == 1:
                    # Average across RGB channels
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                elif input_channels > 3:
                    # Repeat RGB weights
                    repeats = (input_channels + 2) // 3
                    new_conv.weight[:] = old_conv.weight.repeat(1, repeats, 1, 1)[
                        :, :input_channels
                    ]
                else:
                    new_conv.weight[:] = old_conv.weight[:, :input_channels]

                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias

        # Replace first conv
        if self.backbone_name.startswith("efficientnet"):
            self._feature_extractor[0][0][0] = new_conv
        elif self.backbone_name.startswith("resnet"):
            self._feature_extractor[0] = new_conv
        elif self.backbone_name.startswith("mobilenet"):
            self._feature_extractor[0][0][0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self._feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            features = self._feature_extractor(x)
        return features

    def freeze_backbone(self) -> None:

        for param in self._feature_extractor.parameters():
            param.requires_grad = False
        self.logger.info(f"Froze backbone: {self.backbone_name}")

    def unfreeze_backbone(self, layers: Optional[int] = None) -> None:

        if layers is None:
            for param in self._feature_extractor.parameters():
                param.requires_grad = True
            self.logger.info("Unfroze all backbone layers")
        else:
            # Unfreeze last N layers
            all_layers = list(self._feature_extractor.children())
            for layer in all_layers[-layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            self.logger.info(f"Unfroze last {layers} backbone layers")

    def gradual_unfreeze(self, epoch: int, total_epochs: int) -> None:

        progress = epoch / total_epochs

        if progress < 0.3:
            # Keep backbone frozen
            pass
        elif progress < 0.6:
            # Unfreeze last 2 layers
            self.unfreeze_backbone(layers=2)
        elif progress < 0.8:
            # Unfreeze last 4 layers
            self.unfreeze_backbone(layers=4)
        else:
            # Unfreeze all
            self.unfreeze_backbone()

    def get_layer_groups(self) -> List[List[nn.Parameter]]:

        backbone_params = list(self._feature_extractor.parameters())
        classifier_params = list(self.classifier.parameters())

        # Split backbone into early
        mid = len(backbone_params) // 2

        return [
            backbone_params[:mid],  # Early backbone
            backbone_params[mid:],  # Late backbone
            classifier_params,  # Classifier
        ]

    def _get_config(self) -> Dict[str, Any]:

        return {
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "dropout": self.dropout_rate,
            "hidden_dim": self.hidden_dim,
        }


class MultiChannelCNN2D(BaseDLModel):
    def __init__(
        self,
        num_classes: int = 3,
        num_channels: int = 3,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        fusion: str = "concat",
        dropout: float = 0.3,
        device: str = "auto",
    ):

        super().__init__(num_channels, num_classes, device)

        self.num_channels = num_channels
        self.fusion_method = fusion

        # Create separate branches
        self.branches = nn.ModuleList(
            [
                CNN2DTransferLearning(
                    input_dim=1,  # Each branch processes single
                    num_classes=num_classes,
                    backbone=backbone,
                    pretrained=pretrained,
                    freeze_backbone=False,
                    dropout=dropout,
                    device="cpu",  # Will move to device
                )
                for _ in range(num_channels)
            ]
        )

        # Get feature dimension from
        self.feature_dim = self.branches[0].feature_dim

        # Fusion layer
        if fusion == "concat":
            fused_dim = self.feature_dim * num_channels
        else:
            fused_dim = self.feature_dim

        if fusion == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 4, 1),
            )

        # Final classifier (replace branch
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Process each channel through
        features = []
        for i in range(self.num_channels):
            channel_input = x[:, i : i + 1]  # (batch, 1, H, W)
            branch_features = self.branches[i].get_features(channel_input)
            features.append(branch_features)

        # Stack features: (batch, num_channels,
        features = torch.stack(features, dim=1)

        # Fuse features
        if self.fusion_method == "concat":
            fused = features.view(features.size(0), -1)
        elif self.fusion_method == "mean":
            fused = features.mean(dim=1)
        elif self.fusion_method == "attention":
            # Attention-weighted fusion
            weights = self.attention(features)  # (batch, num_channels, 1)
            weights = F.softmax(weights, dim=1)
            fused = (features * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Classify
        logits = self.classifier(fused)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:

        features = []
        for i in range(self.num_channels):
            channel_input = x[:, i : i + 1]
            branch_features = self.branches[i].get_features(channel_input)
            features.append(branch_features)

        features = torch.stack(features, dim=1)

        if self.fusion_method == "concat":
            return features.view(features.size(0), -1)
        else:
            return features.mean(dim=1)
