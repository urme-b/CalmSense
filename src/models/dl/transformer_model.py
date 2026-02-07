import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dl_model import BaseDLModel


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.1, bias: bool = True
    ):

        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights for
        self._attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask (e.g.,
        if attn_mask is not None:
            attn = attn + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Store attention weights
        self._attention_weights = attn.detach()

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:

        return self._attention_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):

        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.dropout(x) + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x


class StressTransformer(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        positional_encoding: str = "learnable",
        use_cls_token: bool = True,
        causal_mask: bool = False,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.use_cls_token = use_cls_token
        self.causal_mask = causal_mask
        self.positional_encoding_type = positional_encoding

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # [CLS] token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        if positional_encoding == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )

        # Store attention weights
        self._attention_weights: List[torch.Tensor] = []

        self.initialize_weights("xavier")
        self.to(self._device)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add [CLS] token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Update padding mask for
            if key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

            seq_len = seq_len + 1

        # Add positional encoding
        x = self.pos_encoder(x)

        # Generate causal mask if
        attn_mask = None
        if self.causal_mask:
            attn_mask = self._generate_causal_mask(seq_len, x.device)

        # Clear stored attention weights
        self._attention_weights = []

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

            # Store attention weights
            if return_attention:
                attn_weights = layer.self_attn.get_attention_weights()
                if attn_weights is not None:
                    self._attention_weights.append(attn_weights)

        # Final normalization
        x = self.norm(x)

        # Get representation for classification
        if self.use_cls_token:
            representation = x[:, 0]  # [CLS] token
        else:
            representation = x.mean(dim=1)  # Mean pooling

        # Classification
        logits = self.classifier(representation)

        if return_attention:
            return logits, self._attention_weights
        return logits

    def get_attention_weights(
        self, x: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:

        if x is not None:
            self.eval()
            with torch.no_grad():
                _, attn_weights = self.forward(x, return_attention=True)
            return attn_weights

        return self._attention_weights

    def get_attention_rollout(self, x: torch.Tensor) -> torch.Tensor:

        self.eval()
        with torch.no_grad():
            _, attn_weights_list = self.forward(x, return_attention=True)

        if not attn_weights_list:
            return None

        # Average attention across heads
        attn_weights = [w.mean(dim=1) for w in attn_weights_list]  # (batch, seq, seq)

        # Compute rollout (matrix multiplication
        rollout = torch.eye(attn_weights[0].size(-1), device=x.device)
        rollout = rollout.unsqueeze(0).expand(x.size(0), -1, -1)

        for attn in attn_weights:
            # Add residual connection effect
            attn_with_residual = 0.5 * attn + 0.5 * torch.eye(
                attn.size(-1), device=x.device
            )
            rollout = torch.bmm(rollout, attn_with_residual)

        # Get attention to [CLS]
        if self.use_cls_token:
            cls_attention = rollout[:, 0, 1:]  # Exclude [CLS] attending to
        else:
            cls_attention = rollout[:, 0]

        return cls_attention

    def _get_config(self) -> Dict[str, Any]:

        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate,
            "positional_encoding": self.positional_encoding_type,
            "use_cls_token": self.use_cls_token,
            "causal_mask": self.causal_mask,
        }


class TemporalConvTransformer(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.d_model = d_model

        # Convolutional stem for sequence
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(
            d_model, max_len=2000, dropout=dropout
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )

        self.initialize_weights("xavier")
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Handle both input formats:
        if x.size(1) != self.input_dim:
            x = x.transpose(1, 2)  # (batch, seq, features) ->

        # Convolutional stem
        x = self.conv_stem(x)  # (batch, d_model, reduced_seq)

        # Prepare for transformer
        x = x.transpose(1, 2)  # (batch, reduced_seq, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling and
        x = x.mean(dim=1)
        logits = self.classifier(x)

        return logits
