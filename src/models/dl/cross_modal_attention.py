from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dl_model import BaseDLModel


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
        use_temporal: bool = True,
    ):

        super().__init__()

        self.use_temporal = use_temporal

        if use_temporal:
            # Temporal convolution for local
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
            )

        self.proj = nn.Sequential(
            nn.Linear(input_dim if not use_temporal else d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        is_sequential = x.dim() == 3

        if self.use_temporal and is_sequential:
            # Apply temporal convolution
            x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
            x = self.temporal_conv(x)
            x = x.transpose(1, 2)  # (batch, seq_len, d_model)

        x = self.proj(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):

        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self._attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Cross-attention with residual
        residual = query
        query = self.norm1(query)
        attn_out, attn_weights = self.cross_attn(
            query, key_value, key_value, need_weights=True
        )
        self._attention_weights = attn_weights.detach()
        out = attn_out + residual

        # FFN with residual
        residual = out
        out = self.norm2(out)
        out = self.ffn(out) + residual

        if return_attention:
            return out, attn_weights
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:

        return self._attention_weights


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int):

        super().__init__()

        self.num_modalities = num_modalities

        # Gate networks for each
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model * num_modalities, d_model), nn.Sigmoid()
                )
                for _ in range(num_modalities)
            ]
        )

        # Final projection
        self.proj = nn.Linear(d_model * num_modalities, d_model)

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:

        # Concatenate all features
        concat = torch.cat(modality_features, dim=-1)

        # Compute gates
        gated_features = []
        for i, gate in enumerate(self.gates):
            g = gate(concat)
            gated_features.append(g * modality_features[i])

        # Concatenate gated features and
        gated_concat = torch.cat(gated_features, dim=-1)
        fused = self.proj(gated_concat)

        return fused


class CrossModalAttention(BaseDLModel):
    DEFAULT_MODALITIES = ["ECG", "EDA", "TEMP", "ACC"]

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
        fusion_method: str = "attention",
        device: str = "auto",
    ):

        # Use sum of modality
        total_dim = sum(modality_dims.values())
        super().__init__(total_dim, num_classes, device)

        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(modality_dims)
        self.d_model = d_model
        self.nhead = nhead
        self.fusion_method = fusion_method

        # Modality-specific encoders
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(dim, d_model, dropout, use_temporal=True)
                for name, dim in modality_dims.items()
            }
        )

        # Cross-attention layers between modality
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            layer_dict = nn.ModuleDict()
            for i, mod_i in enumerate(self.modality_names):
                for j, mod_j in enumerate(self.modality_names):
                    if i != j:
                        key = f"{mod_i}_to_{mod_j}"
                        layer_dict[key] = CrossAttentionBlock(d_model, nhead, dropout)
            self.cross_attention_layers.append(layer_dict)

        # Fusion mechanism
        if fusion_method == "gated":
            self.fusion = GatedFusion(d_model, self.num_modalities)
            fused_dim = d_model
        elif fusion_method == "attention":
            self.modality_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 4), nn.Tanh(), nn.Linear(d_model // 4, 1)
            )
            fused_dim = d_model
        elif fusion_method == "concat":
            fused_dim = d_model * self.num_modalities
        else:  # mean
            fused_dim = d_model

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Store modality importance
        self._modality_importance: Dict[str, float] = {}
        self._cross_attention_weights: Dict[str, torch.Tensor] = {}

        self.initialize_weights("xavier")
        self.to(self._device)

    def forward(
        self, x: Dict[str, torch.Tensor], return_importance: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:

        # Handle missing modalities
        available_modalities = [m for m in self.modality_names if m in x]

        if len(available_modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # Encode each available modality
        encoded = {}
        for mod in available_modalities:
            encoded[mod] = self.encoders[mod](x[mod])

        # Apply cross-attention layers
        for cross_layer_dict in self.cross_attention_layers:
            new_encoded = {}
            for mod in available_modalities:
                # Start with self representation
                attended = encoded[mod]

                # Add cross-attended features from
                cross_features = []
                for other_mod in available_modalities:
                    if other_mod != mod:
                        key = f"{mod}_to_{other_mod}"
                        if key in cross_layer_dict:
                            cross_out = cross_layer_dict[key](
                                encoded[mod], encoded[other_mod]
                            )
                            cross_features.append(cross_out)

                # Average cross-attended features
                if cross_features:
                    cross_mean = torch.stack(cross_features, dim=0).mean(dim=0)
                    attended = attended + cross_mean

                new_encoded[mod] = attended

            encoded = new_encoded

        # Pool temporal dimension if
        pooled = {}
        for mod in available_modalities:
            if encoded[mod].dim() == 3:
                pooled[mod] = encoded[mod].mean(dim=1)  # Global average pooling
            else:
                pooled[mod] = encoded[mod]

        # Fuse modalities
        modality_features = [pooled[mod] for mod in available_modalities]

        if (
            self.fusion_method == "gated"
            and len(available_modalities) == self.num_modalities
        ):
            fused = self.fusion(modality_features)
        elif self.fusion_method == "attention":
            # Stack features: (batch, num_modalities,
            stacked = torch.stack(modality_features, dim=1)

            # Compute attention weights
            attn_scores = self.modality_attention(stacked).squeeze(
                -1
            )  # (batch, num_mod)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Store importance (average across
            self._modality_importance = {
                mod: attn_weights[:, i].mean().item()
                for i, mod in enumerate(available_modalities)
            }

            # Weighted sum
            fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_method == "concat":
            # Pad missing modalities with
            all_features = []
            for mod in self.modality_names:
                if mod in pooled:
                    all_features.append(pooled[mod])
                else:
                    all_features.append(
                        torch.zeros_like(pooled[available_modalities[0]])
                    )
            fused = torch.cat(all_features, dim=-1)
        else:  # mean
            fused = torch.stack(modality_features, dim=0).mean(dim=0)

        # Classification
        logits = self.classifier(fused)

        if return_importance:
            return logits, self._modality_importance
        return logits

    def get_modality_importance(self) -> Dict[str, float]:

        return self._modality_importance.copy()

    def get_cross_attention_weights(self) -> Dict[str, torch.Tensor]:

        weights = {}
        for layer_dict in self.cross_attention_layers:
            for key, block in layer_dict.items():
                attn = block.get_attention_weights()
                if attn is not None:
                    weights[key] = attn
        return weights

    def _get_config(self) -> Dict[str, Any]:

        return {
            "modality_dims": self.modality_dims,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "fusion_method": self.fusion_method,
        }


class MultimodalTransformer(BaseDLModel):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        device: str = "auto",
    ):

        total_dim = sum(modality_dims.values())
        super().__init__(total_dim, num_classes, device)

        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(modality_dims)
        self.d_model = d_model

        # Input projections
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in modality_dims.items()}
        )

        # Modality tokens (like [CLS]
        self.modality_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(1, 1, d_model))
                for name in modality_dims.keys()
            }
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Shared Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * self.num_modalities),
            nn.Linear(d_model * self.num_modalities, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.initialize_weights("xavier")
        self.to(self._device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = list(x.values())[0].size(0)

        # Process each modality
        modality_outputs = []
        total_seq_len = 0

        for name in self.modality_names:
            if name not in x:
                continue

            # Project input
            proj = self.projections[name](x[name])

            # Add modality token
            mod_token = self.modality_tokens[name].expand(batch_size, -1, -1)
            proj = torch.cat([mod_token, proj], dim=1)

            # Add positional encoding
            seq_len = proj.size(1)
            proj = proj + self.pos_encoding[:, total_seq_len : total_seq_len + seq_len]
            total_seq_len += seq_len

            modality_outputs.append(proj)

        # Concatenate all modalities
        combined = torch.cat(modality_outputs, dim=1)

        # Apply Transformer
        encoded = self.transformer(combined)

        # Extract modality token outputs
        token_outputs = []
        idx = 0
        for name in self.modality_names:
            if name in x:
                token_outputs.append(encoded[:, idx])
                idx += x[name].size(1) + 1  # +1 for modality token

        # Concatenate token outputs
        fused = torch.cat(token_outputs, dim=-1)

        # Classification
        logits = self.classifier(fused)

        return logits
