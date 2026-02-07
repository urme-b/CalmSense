import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import torch
    import torch.nn as nn  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..logging_config import LoggerMixin


class AttentionVisualizer(LoggerMixin):
    def __init__(
        self, model: Optional[Any] = None, layer_names: Optional[List[str]] = None
    ):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for attention visualization")

        self.model = model
        self.layer_names = layer_names or []
        self.attention_weights = {}
        self._hooks = []

        if model is not None:
            self._register_hooks()

    def _register_hooks(self) -> None:

        self._remove_hooks()  # Clear existing hooks

        def make_hook(name):
            def hook(module, input, output):
                # Handle different attention output
                if isinstance(output, tuple):
                    # Many attention layers return
                    if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        self.attention_weights[name] = output[1].detach().cpu()
                    else:
                        # Try to get attention
                        self.attention_weights[name] = output[0].detach().cpu()
                elif isinstance(output, torch.Tensor):
                    self.attention_weights[name] = output.detach().cpu()

            return hook

        for name, module in self.model.named_modules():
            # Auto-detect attention layers
            module_type = type(module).__name__.lower()
            is_attention = any(
                x in module_type for x in ["attention", "mha", "multihead"]
            )

            if is_attention or name in self.layer_names:
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)
                self.logger.debug(f"Registered hook for: {name}")

    def _remove_hooks(self) -> None:

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_attention_weights(
        self, x: Union[np.ndarray, "torch.Tensor"], layer_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:

        if self.model is None:
            raise ValueError(
                "No model set. Initialize with a model or call set_model()."
            )

        # Clear previous weights
        self.attention_weights = {}

        # Ensure model is in
        self.model.eval()

        # Convert to tensor if
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        # Add batch dimension if
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2 and hasattr(self.model, "input_dim"):
            x = x.unsqueeze(0)

        # Move to model device
        device = next(self.model.parameters()).device
        x = x.to(device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(x)

        # Convert to numpy
        result = {
            name: weights.numpy() for name, weights in self.attention_weights.items()
        }

        if layer_name is not None:
            if layer_name in result:
                return {layer_name: result[layer_name]}
            else:
                self.logger.warning(
                    f"Layer '{layer_name}' not found. Available: {list(result.keys())}"
                )

        return result

    def compute_attention_rollout(
        self,
        attention_weights: List[np.ndarray],
        discard_ratio: float = 0.0,
        head_fusion: str = "mean",
    ) -> np.ndarray:

        result = None

        for attn in attention_weights:
            # Fuse heads
            if attn.ndim == 4:  # [batch, heads, seq, seq]
                if head_fusion == "mean":
                    attn = attn.mean(axis=1)
                elif head_fusion == "max":
                    attn = attn.max(axis=1)
                elif head_fusion == "min":
                    attn = attn.min(axis=1)
                else:
                    raise ValueError(f"Unknown head_fusion: {head_fusion}")

            # Add residual connection (identity
            batch_size, seq_len, _ = attn.shape
            eye = np.eye(seq_len)
            attn = 0.5 * attn + 0.5 * eye

            # Discard low attention
            if discard_ratio > 0:
                flat_attn = attn.flatten()
                threshold = np.quantile(flat_attn, discard_ratio)
                attn[attn < threshold] = 0

            # Re-normalize rows
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

            # Accumulate
            if result is None:
                result = attn
            else:
                result = np.matmul(attn, result)

        return result

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        head_idx: Optional[int] = None,
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Figure:
        # Handle different shapes
        if attention_weights.ndim == 4:
            # [batch, heads, seq, seq]
            attn = attention_weights[layer_idx]
            if head_idx is not None:
                attn = attn[head_idx]
                title_suffix = f" (Head {head_idx})"
            else:
                attn = attn.mean(axis=0)
                title_suffix = " (All Heads Average)"
        elif attention_weights.ndim == 3:
            attn = attention_weights[layer_idx]
            title_suffix = ""
        else:
            attn = attention_weights
            title_suffix = ""

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(attn, cmap=cmap, aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", fontsize=11)

        # Add labels
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel("Key Position", fontsize=11)
        ax.set_ylabel("Query Position", fontsize=11)

        if title:
            ax.set_title(title + title_suffix, fontsize=12)
        else:
            ax.set_title(f"Attention Weights{title_suffix}", fontsize=12)

        plt.tight_layout()
        return fig

    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        batch_idx: int = 0,
        figsize: Tuple[int, int] = (16, 12),
        cmap: str = "viridis",
    ) -> Figure:
        if attention_weights.ndim != 4:
            raise ValueError("Expected 4D tensor [batch, heads, seq, seq]")

        attn = attention_weights[batch_idx]
        n_heads = attn.shape[0]

        # Compute grid layout
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_heads == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for head_idx in range(n_heads):
            row = head_idx // n_cols
            col = head_idx % n_cols
            ax = axes[row, col]

            im = ax.imshow(attn[head_idx], cmap=cmap, aspect="auto")
            ax.set_title(f"Head {head_idx}", fontsize=10)
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_heads, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle(f"Multi-Head Attention Weights (Sample {batch_idx})", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_temporal_attention(
        self,
        attention_weights: np.ndarray,
        time_axis: Optional[np.ndarray] = None,
        target_positions: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "RdYlBu_r",
    ) -> Figure:
        if attention_weights.ndim == 3:
            attn = attention_weights[0]
        else:
            attn = attention_weights

        seq_len = attn.shape[0]

        if time_axis is None:
            time_axis = np.arange(seq_len)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot attention as line
        if target_positions is None:
            # Use first, middle, and
            target_positions = [0, seq_len // 2, seq_len - 1]

        colors = plt.cm.tab10(np.linspace(0, 1, len(target_positions)))

        for pos, color in zip(target_positions, colors):
            ax.plot(
                time_axis,
                attn[pos],
                color=color,
                linewidth=2,
                label=f"Position {pos}",
                alpha=0.8,
            )
            ax.axvline(
                x=time_axis[pos], color=color, linestyle="--", alpha=0.3, linewidth=1
            )

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Attention Weight", fontsize=11)
        ax.set_title("Temporal Attention Distribution", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_attention_flow(
        self, attention_weights: List[np.ndarray], figsize: Tuple[int, int] = (14, 8)
    ) -> Figure:

        n_layers = len(attention_weights)

        fig, axes = plt.subplots(1, n_layers + 1, figsize=figsize)

        # Plot individual layer attentions
        for i, attn in enumerate(attention_weights):
            if attn.ndim == 4:
                attn = attn[0].mean(axis=0)  # Average over heads
            elif attn.ndim == 3:
                attn = attn[0]

            im = axes[i].imshow(attn, cmap="viridis", aspect="auto")
            axes[i].set_title(f"Layer {i}", fontsize=10)
            axes[i].set_xlabel("Key")
            axes[i].set_ylabel("Query")

        # Plot attention rollout
        rollout = self.compute_attention_rollout(attention_weights)
        if rollout.ndim == 3:
            rollout = rollout[0]

        im = axes[-1].imshow(rollout, cmap="viridis", aspect="auto")
        axes[-1].set_title("Attention Rollout", fontsize=10)
        axes[-1].set_xlabel("Key")
        axes[-1].set_ylabel("Query")

        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        plt.suptitle("Attention Flow Across Layers", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_cls_attention(
        self,
        attention_weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        top_k: int = 20,
    ) -> Figure:
        if attention_weights.ndim == 4:
            # Average over heads, take
            attn = attention_weights[0].mean(axis=0)
        elif attention_weights.ndim == 3:
            attn = attention_weights[0]
        else:
            attn = attention_weights

        # Get [CLS] token attention
        cls_attention = attn[0, 1:]  # Exclude self-attention

        seq_len = len(cls_attention)

        if feature_names is None:
            feature_names = [f"Pos {i}" for i in range(1, seq_len + 1)]
        else:
            feature_names = feature_names[1:]  # Exclude CLS

        # Sort by attention
        sorted_indices = np.argsort(cls_attention)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=figsize)

        # Bar plot
        positions = np.arange(len(sorted_indices))
        values = cls_attention[sorted_indices]
        labels = [feature_names[i] for i in sorted_indices]

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(positions)))
        ax.barh(positions, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Attention Weight", fontsize=11)
        ax.set_title("[CLS] Token Attention Distribution", fontsize=12)
        ax.invert_yaxis()

        plt.tight_layout()
        return fig

    def get_attention_statistics(self, attention_weights: np.ndarray) -> Dict[str, Any]:

        if attention_weights.ndim == 4:
            attn = attention_weights[0]  # First batch sample
        else:
            attn = attention_weights

        # If multi-head, analyze per
        if attn.ndim == 3:
            n_heads = attn.shape[0]
            head_stats = []

            for h in range(n_heads):
                head_attn = attn[h]
                head_stats.append(
                    {
                        "head": h,
                        "entropy": self._compute_entropy(head_attn),
                        "sparsity": self._compute_sparsity(head_attn),
                        "max_attention": float(head_attn.max()),
                        "mean_attention": float(head_attn.mean()),
                        "diagonal_ratio": self._compute_diagonal_ratio(head_attn),
                    }
                )

            avg_attn = attn.mean(axis=0)
        else:
            head_stats = None
            avg_attn = attn

        stats = {
            "entropy": self._compute_entropy(avg_attn),
            "sparsity": self._compute_sparsity(avg_attn),
            "max_attention": float(avg_attn.max()),
            "mean_attention": float(avg_attn.mean()),
            "std_attention": float(avg_attn.std()),
            "diagonal_ratio": self._compute_diagonal_ratio(avg_attn),
            "head_statistics": head_stats,
        }

        return stats

    def _compute_entropy(self, attention: np.ndarray) -> float:

        # Clip to avoid log(0)
        attention = np.clip(attention, 1e-9, 1.0)
        entropy = -np.sum(attention * np.log(attention), axis=-1)
        return float(np.mean(entropy))

    def _compute_sparsity(self, attention: np.ndarray, threshold: float = 0.1) -> float:

        return float(np.mean(attention < threshold))

    def _compute_diagonal_ratio(self, attention: np.ndarray) -> float:

        diagonal = np.diag(attention)
        return float(np.mean(diagonal))

    def extract_important_positions(
        self,
        attention_weights: np.ndarray,
        threshold: float = 0.1,
        aggregation: str = "mean",
    ) -> List[Dict]:

        if attention_weights.ndim == 4:
            attn = attention_weights[0]
            if attn.ndim == 3:
                if aggregation == "mean":
                    attn = attn.mean(axis=0)
                elif aggregation == "max":
                    attn = attn.max(axis=0)
                elif aggregation == "sum":
                    attn = attn.sum(axis=0)
        elif attention_weights.ndim == 3:
            attn = attention_weights[0]
        else:
            attn = attention_weights

        # Sum attention received by
        attention_received = attn.sum(axis=0)

        important_positions = []
        for pos, weight in enumerate(attention_received):
            if weight > threshold * attention_received.sum():
                important_positions.append(
                    {
                        "position": pos,
                        "attention_weight": float(weight),
                        "relative_importance": float(weight / attention_received.sum()),
                    }
                )

        # Sort by importance
        important_positions.sort(key=lambda x: x["attention_weight"], reverse=True)

        return important_positions

    def set_model(self, model: Any) -> None:

        self._remove_hooks()
        self.model = model
        self._register_hooks()

    def __del__(self):

        self._remove_hooks()
