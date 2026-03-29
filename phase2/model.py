"""GRU decoder for Phase 2 kinematic decoding."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GRUDecoder(nn.Module):
    """Bidirectional GRU decoder for kinematic prediction.

    Different inductive bias from transformers: recurrent state propagation
    captures temporal dynamics naturally, while bidirectionality allows
    attending to both past and future within the window.
    """

    def __init__(
        self,
        n_channels: int = 96,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, sbp: Tensor, **kwargs) -> Tensor:
        """
        Args:
            sbp: (B, T, 96) z-scored SBP

        Returns:
            (B, T, 2) predicted positions
        """
        x = self.input_proj(sbp)
        x, _ = self.gru(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config) -> nn.Module:
    """Build GRU model from config."""
    return GRUDecoder(
        n_channels=config.n_channels,
        d_model=config.gru_d_model,
        n_layers=config.gru_n_layers,
        dropout=config.gru_dropout,
        n_outputs=config.n_position_outputs,
    )
