from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dl_model import BaseDLModel


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int = 64):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Compute context vector
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(
            1
        )  # (batch, hidden_dim)

        return context, weights


class BiLSTMClassifier(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = False,
        attention_dim: int = 64,
        use_layer_norm: bool = True,
        bidirectional: bool = True,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output dimension from LSTM
        lstm_output_dim = hidden_dim * self.num_directions

        # Attention
        if use_attention:
            self.attention = Attention(lstm_output_dim, attention_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Initialize
        self.initialize_weights()
        self.to(self._device)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:

        batch_size = x.size(0)

        # Layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Pack sequences if lengths
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Get representation
        if self.use_attention:
            # Create mask for padded
            mask = None
            if lengths is not None:
                max_len = lstm_out.size(1)
                mask = torch.arange(max_len, device=lstm_out.device).expand(
                    batch_size, -1
                )
                mask = mask < lengths.unsqueeze(1)

            representation, attn_weights = self.attention(lstm_out, mask)
        else:
            # Use final hidden states
            if self.bidirectional:
                # Concatenate final hidden states
                h_forward = h_n[-2]  # Final layer, forward
                h_backward = h_n[-1]  # Final layer, backward
                representation = torch.cat([h_forward, h_backward], dim=-1)
            else:
                representation = h_n[-1]
            attn_weights = None

        # Classification
        logits = self.classifier(representation)

        if return_attention and self.use_attention:
            return logits, attn_weights
        return logits

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_layer_norm:
            x = self.layer_norm(x)

        lstm_out, _ = self.lstm(x)
        return lstm_out

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:

        if not self.use_attention:
            return None

        _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights

    def _get_config(self) -> Dict[str, Any]:

        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "use_attention": self.use_attention,
            "bidirectional": self.bidirectional,
            "use_layer_norm": self.use_layer_norm,
        }


class GRUClassifier(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = False,
        attention_dim: int = 64,
        bidirectional: bool = True,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output dimension
        gru_output_dim = hidden_dim * self.num_directions

        # Attention
        if use_attention:
            self.attention = Attention(gru_output_dim, attention_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.initialize_weights()
        self.to(self._device)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:

        batch_size = x.size(0)

        # Pack sequences if lengths
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # GRU forward pass
        gru_out, h_n = self.gru(x)

        # Unpack if packed
        if lengths is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Get representation
        if self.use_attention:
            mask = None
            if lengths is not None:
                max_len = gru_out.size(1)
                mask = torch.arange(max_len, device=gru_out.device).expand(
                    batch_size, -1
                )
                mask = mask < lengths.unsqueeze(1)

            representation, attn_weights = self.attention(gru_out, mask)
        else:
            if self.bidirectional:
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                representation = torch.cat([h_forward, h_backward], dim=-1)
            else:
                representation = h_n[-1]
            attn_weights = None

        logits = self.classifier(representation)

        if return_attention and self.use_attention:
            return logits, attn_weights
        return logits

    def _get_config(self) -> Dict[str, Any]:

        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "use_attention": self.use_attention,
            "bidirectional": self.bidirectional,
        }


class StackedLSTMClassifier(BaseDLModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        device: str = "auto",
    ):

        super().__init__(input_dim, num_classes, device)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)

        # Stacked LSTM layers with
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=hidden_dim * 2,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)]
        )

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.initialize_weights()
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Project input
        x = self.input_proj(x)

        # Stacked LSTM with residual
        for lstm, ln, dropout in zip(self.lstm_layers, self.layer_norms, self.dropouts):
            residual = x
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = dropout(lstm_out)
            x = lstm_out + residual

        # Use final hidden state
        representation = x.mean(dim=1)

        logits = self.classifier(representation)
        return logits

    def _get_config(self) -> Dict[str, Any]:

        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
