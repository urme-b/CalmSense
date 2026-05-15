from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dl_model import BaseDLModel


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int = 1, dropout: float = 0.1):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads

        assert hidden_dim % attention_heads == 0, (
            "hidden_dim must be divisible by attention_heads"
        )

        # Xavier-initialized query
        self.query = nn.Parameter(torch.empty(attention_heads, self.head_dim))
        nn.init.xavier_uniform_(self.query.unsqueeze(0)).squeeze_(0)

        # Key and value projections
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        # Store attention weights for
        self._attention_weights = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, seq_len, _ = hidden_states.shape

        # Project keys and values
        keys = self.key_proj(hidden_states)  # (batch, seq_len, hidden_dim)
        values = self.value_proj(hidden_states)

        # Reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.attention_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.attention_heads, self.head_dim)

        # (batch, heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Expand query for batch:
        query = self.query.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)

        # Compute attention scores
        # (batch, heads, 1, head_dim)
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(2)  # (batch, heads, seq_len)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.attention_heads, -1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store for visualization (average
        self._attention_weights = attention_weights.mean(dim=1).detach()

        # Apply attention to values
        # (batch, heads, seq_len) @
        context = torch.matmul(attention_weights.unsqueeze(2), values).squeeze(2)
        # (batch, heads, head_dim)

        # Reshape and project
        context = context.view(batch_size, self.hidden_dim)
        context = self.out_proj(context)

        if return_attention:
            return context, self._attention_weights
        return context

    def get_attention_weights(self) -> Optional[torch.Tensor]:

        return self._attention_weights


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 2
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNLSTMHybrid(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 700,
        cnn_filters: Tuple[int, int] = (64, 128),
        cnn_kernels: Tuple[int, int] = (7, 5),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.5,
        device: str = "auto",
    ):
        super().__init__(input_dim, num_classes, device)

        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.cnn_kernels = cnn_kernels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout

        # CNN feature extractor
        self.cnn = nn.Sequential(
            CNNBlock(input_dim, cnn_filters[0], cnn_kernels[0]),
            CNNBlock(cnn_filters[0], cnn_filters[1], cnn_kernels[1]),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # LSTM output dimension
        lstm_output_dim = lstm_hidden * 2  # bidirectional

        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=lstm_output_dim, attention_heads=attention_heads, dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.initialize_weights("kaiming")
        self.to(self._device)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, cnn_filters[-1], seq_len //

        # Permute for LSTM: (batch,
        lstm_in = cnn_out.permute(0, 2, 1)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(lstm_in)

        # Attention pooling
        if return_attention:
            context, attn_weights = self.attention(lstm_out, return_attention=True)
        else:
            context = self.attention(lstm_out)
            attn_weights = None

        # Classification
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:

        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            return self.cnn(x)

    def get_lstm_outputs(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            cnn_out = self.cnn(x)
            lstm_in = cnn_out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(lstm_in)
        return lstm_out

    def _get_config(self) -> Dict[str, Any]:

        return {
            "sequence_length": self.sequence_length,
            "cnn_filters": self.cnn_filters,
            "cnn_kernels": self.cnn_kernels,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout_rate,
        }


class DeepCNNLSTM(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 700,
        lstm_hidden: int = 128,
        dropout: float = 0.5,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.sequence_length = sequence_length

        # Multi-scale CNN blocks
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Skip connection projections
        self.skip1 = nn.Conv1d(64, 256, kernel_size=1)
        self.skip2 = nn.Conv1d(128, 256, kernel_size=1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Attention
        self.attention = TemporalAttention(lstm_hidden * 2, attention_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.initialize_weights("kaiming")
        self.to(self._device)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # CNN blocks
        out1 = self.cnn_block1(x)
        out2 = self.cnn_block2(out1)
        out3 = self.cnn_block3(out2)

        # Align sizes and add
        min_len = out3.size(-1)

        skip1 = F.adaptive_avg_pool1d(self.skip1(out1), min_len)
        skip2 = F.adaptive_avg_pool1d(self.skip2(out2), min_len)

        # Fuse with skip connections
        fused = out3 + skip1 + skip2

        # LSTM
        lstm_in = fused.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)

        # Attention
        if return_attention:
            context, attn_weights = self.attention(lstm_out, return_attention=True)
        else:
            context = self.attention(lstm_out)
            attn_weights = None

        # Classify
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:

        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights
