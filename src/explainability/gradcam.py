import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Stub types for environments
    import types

    nn = types.ModuleType("nn")
    nn.Module = object
    nn.Conv1d = type(None)
    nn.Conv2d = type(None)

from ..logging_config import LoggerMixin


class GradCAMExplainer(LoggerMixin):
    def __init__(
        self,
        model: Any,
        target_layer: Optional[str] = None,
        device: Optional[str] = None,
    ):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Grad-CAM")

        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Storage for activations and
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Find target layer
        if target_layer is None:
            target_layer = self._find_last_conv_layer()
            if target_layer:
                self.logger.info(f"Auto-detected target layer: {target_layer}")
            else:
                raise ValueError(
                    "Could not auto-detect convolutional layer. "
                    "Please specify target_layer explicitly."
                )

        self.target_layer = target_layer
        self._register_hooks()

    def _find_last_conv_layer(self) -> Optional[str]:

        last_conv = None

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                last_conv = name

        return last_conv

    def _get_layer(self, layer_name: str) -> nn.Module:

        parts = layer_name.split(".")
        module = self.model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def _register_hooks(self) -> None:

        self._remove_hooks()

        target_module = self._get_layer(self.target_layer)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_module.register_forward_hook(forward_hook))
        self._hooks.append(target_module.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate_heatmap(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:

        self.model.eval()

        # Prepare input
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        # Add batch dimension if
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, channels, seq_len]

        x.requires_grad_(True)

        # Forward pass
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        # Get weights: global average
        # Shape: [batch, channels, spatial...]
        if self.gradients.dim() == 3:
            # 1D: [batch, channels, length]
            weights = self.gradients.mean(dim=2, keepdim=True)
        else:
            # 2D: [batch, channels, height,
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activation
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU to keep only
        heatmap = F.relu(heatmap)

        # Convert to numpy
        heatmap = heatmap.squeeze().cpu().numpy()

        # Normalize
        if normalize and heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def generate_gradcam_plus_plus(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:

        self.model.eval()

        # Prepare input
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        x.requires_grad_(True)

        # Forward pass
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Get target score
        score = output[0, target_class]

        # First derivative
        self.model.zero_grad()
        score.backward(retain_graph=True)
        first_grad = self.gradients.clone()

        # Second derivative (approximate via
        self.model.zero_grad()
        grad_score = (first_grad**2).sum()
        grad_score.backward(retain_graph=True)
        second_grad = self.gradients.clone()

        # Third derivative
        self.model.zero_grad()
        grad_score2 = (second_grad**2).sum()
        grad_score2.backward(retain_graph=True)
        third_grad = self.gradients.clone()

        # Compute alpha weights
        global_sum = self.activations.sum(
            dim=tuple(range(2, self.activations.dim())), keepdim=True
        )
        alpha_num = second_grad
        alpha_denom = 2 * second_grad + global_sum * third_grad + 1e-10
        alpha = alpha_num / alpha_denom

        # Compute weights
        weights = (alpha * F.relu(first_grad)).sum(
            dim=tuple(range(2, first_grad.dim())), keepdim=True
        )

        # Weighted combination
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Convert to numpy
        heatmap = heatmap.squeeze().cpu().numpy()

        if normalize and heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def generate_guided_gradcam(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        target_class: Optional[int] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Get standard Grad-CAM
        gradcam = self.generate_heatmap(x, target_class, normalize=False)

        # Compute guided backpropagation
        guided_bp = self._guided_backprop(x, target_class)

        # Upsample Grad-CAM to input
        if isinstance(x, torch.Tensor):
            input_size = x.shape[-1] if x.dim() <= 3 else x.shape[-2:]
        else:
            input_size = x.shape[-1] if x.ndim <= 2 else x.shape[-2:]

        if gradcam.ndim == 1:
            # 1D case
            gradcam_upsampled = np.interp(
                np.linspace(0, 1, input_size), np.linspace(0, 1, len(gradcam)), gradcam
            )
        else:
            # 2D case
            from scipy.ndimage import zoom

            scale = (input_size[0] / gradcam.shape[0], input_size[1] / gradcam.shape[1])
            gradcam_upsampled = zoom(gradcam, scale, order=1)

        # Element-wise multiplication
        guided_gradcam = guided_bp * gradcam_upsampled

        if normalize:
            if gradcam.max() > 0:
                gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
            if np.abs(guided_gradcam).max() > 0:
                guided_gradcam = guided_gradcam / np.abs(guided_gradcam).max()

        return guided_gradcam, gradcam

    def _guided_backprop(
        self, x: Union[np.ndarray, "torch.Tensor"], target_class: Optional[int] = None
    ) -> np.ndarray:

        def guided_relu_hook(module, grad_input, grad_output):
            # Only backprop positive gradients
            return (F.relu(grad_output[0]),)

        # Replace ReLU gradients
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_backward_hook(guided_relu_hook))

        # Prepare input
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.clone()
        x.requires_grad_(True)

        # Forward and backward
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Get input gradients
        guided_bp = x.grad.squeeze().cpu().numpy()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return guided_bp

    def plot_heatmap(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        heatmap: np.ndarray,
        original_signal: bool = True,
        figsize: Tuple[int, int] = (14, 6),
        cmap: str = "jet",
        alpha: float = 0.4,
        title: Optional[str] = None,
    ) -> Figure:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Flatten if needed
        if x.ndim > 2:
            x = x.squeeze()

        # Check if 1D or
        is_1d = x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1)

        if is_1d:
            return self._plot_1d_heatmap(
                x.flatten(), heatmap, original_signal, figsize, cmap, alpha, title
            )
        else:
            return self._plot_2d_heatmap(
                x, heatmap, original_signal, figsize, cmap, alpha, title
            )

    def _plot_1d_heatmap(
        self,
        x: np.ndarray,
        heatmap: np.ndarray,
        original_signal: bool,
        figsize: Tuple[int, int],
        cmap: str,
        alpha: float,
        title: Optional[str],
    ) -> Figure:

        # Upsample heatmap to signal
        if len(heatmap) != len(x):
            heatmap = np.interp(
                np.linspace(0, 1, len(x)), np.linspace(0, 1, len(heatmap)), heatmap
            )

        fig, axes = plt.subplots(
            2 if original_signal else 1, 1, figsize=figsize, sharex=True
        )
        if not original_signal:
            axes = [axes]

        time = np.arange(len(x))

        if original_signal:
            # Plot original signal
            axes[0].plot(time, x, "b-", linewidth=1, label="Signal")
            axes[0].set_ylabel("Amplitude", fontsize=11)
            axes[0].set_title("Original Signal", fontsize=11)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot heatmap
        ax = axes[-1]

        # Create colored background based
        cmap_obj = plt.cm.get_cmap(cmap)
        colors = cmap_obj(heatmap)

        # Plot signal with heatmap
        for i in range(len(x) - 1):
            ax.plot(
                [time[i], time[i + 1]], [x[i], x[i + 1]], color=colors[i], linewidth=2
            )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax, label="Importance")

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title(
            "Signal with Grad-CAM Importance" if not title else title, fontsize=11
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_2d_heatmap(
        self,
        x: np.ndarray,
        heatmap: np.ndarray,
        original_signal: bool,
        figsize: Tuple[int, int],
        cmap: str,
        alpha: float,
        title: Optional[str],
    ) -> Figure:

        from scipy.ndimage import zoom

        # Upsample heatmap to input
        if heatmap.shape != x.shape[-2:]:
            target_shape = x.shape[-2:] if x.ndim == 2 else x.shape[1:]
            scale = (
                target_shape[0] / heatmap.shape[0],
                target_shape[1] / heatmap.shape[1],
            )
            heatmap = zoom(heatmap, scale, order=1)

        n_cols = 3 if original_signal else 2
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)

        # Original image
        if original_signal:
            if x.ndim == 3 and x.shape[0] == 3:
                # RGB image
                axes[0].imshow(np.transpose(x, (1, 2, 0)))
            else:
                axes[0].imshow(x.squeeze(), cmap="gray")
            axes[0].set_title("Original Input", fontsize=11)
            axes[0].axis("off")
            heatmap_ax = axes[1]
            overlay_ax = axes[2]
        else:
            heatmap_ax = axes[0]
            overlay_ax = axes[1]

        # Heatmap
        im = heatmap_ax.imshow(heatmap, cmap=cmap)
        heatmap_ax.set_title("Grad-CAM Heatmap", fontsize=11)
        heatmap_ax.axis("off")
        plt.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.04)

        # Overlay
        if x.ndim == 3 and x.shape[0] == 3:
            base_img = np.transpose(x, (1, 2, 0))
        else:
            base_img = np.stack([x.squeeze()] * 3, axis=-1)
            base_img = (base_img - base_img.min()) / (
                base_img.max() - base_img.min() + 1e-9
            )

        heatmap_colored = plt.cm.get_cmap(cmap)(heatmap)[:, :, :3]
        overlay = (1 - alpha) * base_img + alpha * heatmap_colored

        overlay_ax.imshow(overlay)
        overlay_ax.set_title("Overlay", fontsize=11)
        overlay_ax.axis("off")

        if title:
            plt.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig

    def plot_class_comparison(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        class_names: List[str],
        figsize: Tuple[int, int] = (16, 8),
        cmap: str = "jet",
    ) -> Figure:
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=figsize)

        if n_classes == 1:
            axes = [axes]

        for idx, (ax, class_name) in enumerate(zip(axes, class_names)):
            heatmap = self.generate_heatmap(x, target_class=idx)

            if heatmap.ndim == 1:
                cmap_obj = plt.cm.get_cmap(cmap)
                colors = cmap_obj(heatmap / (heatmap.max() + 1e-10))
                ax.fill_between(
                    range(len(heatmap)),
                    0,
                    heatmap,
                    alpha=0.7,
                    color=colors.mean(axis=0),
                )
                ax.plot(range(len(heatmap)), heatmap, "k-", linewidth=1)
            else:
                im = ax.imshow(heatmap, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"Class: {class_name}", fontsize=11)
            ax.set_xlabel("Time Step" if heatmap.ndim == 1 else "")

        plt.suptitle("Grad-CAM Comparison Across Classes", fontsize=14)
        plt.tight_layout()
        return fig

    def get_important_regions(
        self, heatmap: np.ndarray, threshold: float = 0.5, min_region_size: int = 5
    ) -> List[Dict]:

        # Threshold heatmap
        important = heatmap >= threshold

        regions = []
        in_region = False
        start = 0

        for i, val in enumerate(important):
            if val and not in_region:
                # Start of region
                start = i
                in_region = True
            elif not val and in_region:
                # End of region
                if i - start >= min_region_size:
                    regions.append(
                        {
                            "start": start,
                            "end": i - 1,
                            "length": i - start,
                            "mean_importance": float(heatmap[start:i].mean()),
                            "max_importance": float(heatmap[start:i].max()),
                            "peak_position": int(start + np.argmax(heatmap[start:i])),
                        }
                    )
                in_region = False

        # Handle region at end
        if in_region and len(heatmap) - start >= min_region_size:
            regions.append(
                {
                    "start": start,
                    "end": len(heatmap) - 1,
                    "length": len(heatmap) - start,
                    "mean_importance": float(heatmap[start:].mean()),
                    "max_importance": float(heatmap[start:].max()),
                    "peak_position": int(start + np.argmax(heatmap[start:])),
                }
            )

        # Sort by importance
        regions.sort(key=lambda r: r["max_importance"], reverse=True)

        return regions

    def set_target_layer(self, layer_name: str) -> None:

        self.target_layer = layer_name
        self._register_hooks()
        self.logger.info(f"Target layer changed to: {layer_name}")

    def __del__(self):

        self._remove_hooks()
